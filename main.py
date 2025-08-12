import os, json, time, traceback
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, date, timezone
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from dateutil import parser as dtparser
import httpx, feedparser
from openai import OpenAI

# ---------- Config ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # set on Render; safe choices: gpt-5, gpt-5-mini, gpt-4o-mini
ADVISOR_TOKEN = os.getenv("ADVISOR_TOKEN", "")  # shared secret; keep server-side only
TE_API_KEY = os.getenv("TE_API_KEY", "guest:guest")  # TradingEconomics key (guest:guest works but limited)
ECB_RSS = os.getenv("ECB_RSS", "https://www.ecb.europa.eu/press/rss/press.html")
FED_RSS = os.getenv("FED_RSS", "https://www.federalreserve.gov/feeds/press_all.xml")
TIMEZONE = os.getenv("OVERSEER_TZ") or os.getenv("OVERSER_TZ") or "Europe/London"  # support both names

# OpenAI client — put timeout on the client (SDKs sometimes reject per-call timeout kwarg)
client = OpenAI(timeout=8.0)  # reads OPENAI_API_KEY from env
app = FastAPI(title="AI Overseer (Realtime FX)", version="2.1.0")


# ---------- Models ----------
class AdvisorSource(BaseModel):
    title: str
    url: str
    published: Optional[str] = None

class AdvisorOut(BaseModel):
    action: str = Field("HOLD", description="BUY | SELL | HOLD | FLATTEN")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = Field("", description="Short explanation of the decision")
    sl_pips: Optional[float] = None
    tp_pips: Optional[float] = None
    as_of: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources: List[AdvisorSource] = Field(default_factory=list)

JSON_SCHEMA = {
    "name": "AdvisorOut",
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD", "FLATTEN"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "sl_pips": {"type": "number"},
            "tp_pips": {"type": "number"},
            "as_of": {"type": "string"},
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "published": {"type": "string"}
                    },
                    "required": ["title", "url"]
                }
            }
        },
        "required": ["action", "confidence", "reason", "as_of", "sources"],
        "additionalProperties": False
    },
    "strict": True  # enforce exact JSON shape
}

SYSTEM = (
    "You are the AI Overseer for an FX day-trading bot (pairs may include GBPUSD, EURUSD). "
    "You will receive (1) a 'snapshot' with current market/indicator values from the broker and (2) a 'context' "
    "containing reputable sources (central bank RSS and TradingEconomics calendar). Use ONLY the given context plus snapshot; "
    "do not invent sources. Return ONLY JSON matching the schema. When news is market-moving (e.g., rate decisions, CPI, NFP), "
    "or when spreads are elevated, prefer HOLD or FLATTEN. If evidence is mixed or stale, lower confidence."
)

# ---------- Helpers ----------
def _auth_or_403(header_value: Optional[str]):
    if not ADVISOR_TOKEN:
        raise HTTPException(500, "Server missing ADVISOR_TOKEN")
    if header_value != ADVISOR_TOKEN:
        raise HTTPException(403, "Forbidden")

def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return dtparser.parse(s)
    except Exception:
        return None

def fetch_te_calendar() -> List[Dict[str, Any]]:
    # High-importance Euro Area & US events for today
    d1 = date.today().isoformat(); d2 = d1
    url = (
        f"https://api.tradingeconomics.com/calendar?"
        f"country=United%20States,Euro%20Area&importance=3&d1={d1}&d2={d2}&format=json&c={TE_API_KEY}"
    )
    try:
        with httpx.Client(timeout=5.5) as http:
            r = http.get(url)
            if r.status_code != 200:
                return []
            data = r.json()
            items = []
            for it in data[:30]:  # cap
                title = it.get("Event") or it.get("Title") or "Event"
                when = it.get("DateTime") or it.get("DateUtc") or it.get("Date")
                country = it.get("Country") or ""
                actual = it.get("Actual")
                forecast = it.get("Forecast")
                previous = it.get("Previous")
                link = it.get("SourceUrl") or "https://docs.tradingeconomics.com/economic_calendar/snapshot/"
                t = f"{country}: {title} (act={actual}, fcst={forecast}, prev={previous})"
                items.append({"title": t, "url": link, "published": when})
            return items
    except Exception:
        return []

def fetch_rss(url: str, limit: int = 5) -> List[Dict[str, Any]]:
    out = []
    try:
        feed = feedparser.parse(url)
        for e in (feed.entries or [])[:limit]:
            title = getattr(e, "title", "item")
            link = getattr(e, "link", url)
            published = getattr(e, "published", None) or getattr(e, "updated", None)
            out.append({"title": title, "url": link, "published": published})
    except Exception:
        pass
    return out

def build_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    context = {
        "feeds": {
            "ecb": fetch_rss(ECB_RSS, limit=5),
            "fed": fetch_rss(FED_RSS, limit=5),
            "calendar": fetch_te_calendar(),
        },
        "as_of": _iso(datetime.now(timezone.utc))
    }
    bullets: List[str] = []
    for label in ("ecb", "fed", "calendar"):
        for item in context["feeds"][label]:
            ttl = item["title"]
            pub = item.get("published")
            bullets.append(f"- {label.upper()}: {ttl} [{pub}] -> {item['url']}")
    ctx_text = "LATEST CONTEXT:\n" + "\n".join(bullets[:20])
    return {"context_text": ctx_text, "sources": sum(context["feeds"].values(), []), "as_of": context["as_of"]}

# Core decision helper (used by both routes)
def get_decision(snapshot: Dict[str, Any]) -> AdvisorOut:
    ctx = build_context(snapshot)
    user_input = json.dumps({
        "as_of": ctx["as_of"],
        "snapshot": snapshot,
        "context": ctx["context_text"]
    }, separators=(",", ":"))

    # Structured Outputs with strict JSON schema
    resp = client.responses.create(
        model=MODEL,
        instructions=SYSTEM,
        input=user_input,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        temperature=0.2
    )

    text = getattr(resp, "output_text", None)
    if not text:
        # very rare, but keep a safe fallback
        text = json.dumps(resp.model_dump(), default=str)

    data = json.loads(text)
    out = AdvisorOut(**data)
    # attach sources we fetched (for traceability)
    out.sources = [AdvisorSource(**s) for s in ctx["sources"]]
    return out

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "ecb_rss": ECB_RSS, "fed_rss": FED_RSS}

@app.post("/advice")
async def advice(request: Request, x_advisor_token: Optional[str] = Header(None)):
    _auth_or_403(x_advisor_token)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    snapshot = payload.get("snapshot") if isinstance(payload, dict) and "snapshot" in payload else payload
    if not isinstance(snapshot, dict):
        raise HTTPException(400, "Body must be a JSON object or {'snapshot': {...}}")

    try:
        out = get_decision(snapshot)
        return out.model_dump()
    except Exception as e:
        print("ADVICE ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(500, f"advisor_error:{e.__class__.__name__}")

# Compatibility route for older bots expecting long/short/skip
@app.post("/gbpusd-advice")
async def gbpusd_advice(request: Request, x_advisor_token: Optional[str] = Header(None)):
    _auth_or_403(x_advisor_token)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    snapshot = payload.get("snapshot") if isinstance(payload, dict) and "snapshot" in payload else payload
    if not isinstance(snapshot, dict):
        raise HTTPException(400, "Body must be a JSON object or {'snapshot': {...}}")

    try:
        out = get_decision(snapshot)
        act = (out.action or "HOLD").upper()
        if act == "BUY": mapped = "long"
        elif act == "SELL": mapped = "short"
        else: mapped = "skip"  # HOLD/FLATTEN -> skip
        return {
            "action": mapped,
            "confidence": out.confidence,
            "sl_pips": out.sl_pips or 0,
            "tp_pips": out.tp_pips or 0,
            "reason": out.reason,
        }
    except Exception as e:
        print("GBPUSD-ADVICE ERROR:", repr(e))
        traceback.print_exc()
        # Fail safe: signal skip
        return {
            "action": "skip",
            "confidence": 0.0,
            "sl_pips": 0,
            "tp_pips": 0,
            "reason": "advisor_error"
        }
