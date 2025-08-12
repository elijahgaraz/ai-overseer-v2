import os, json, time
from typing import Optional, Dict, Any, List
from datetime import datetime, date, timedelta, timezone
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from dateutil import parser as dtparser
import httpx, feedparser
from openai import OpenAI

# ---------- Config ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
ADVISOR_TOKEN = os.getenv("ADVISOR_TOKEN", "")  # shared secret; keep server-side only
TE_API_KEY = os.getenv("TE_API_KEY", "guest:guest")  # TradingEconomics key (supports guest:guest for limited use)
ECB_RSS = os.getenv("ECB_RSS", "https://www.ecb.europa.eu/press/rss/press.html")
FED_RSS = os.getenv("FED_RSS", "https://www.federalreserve.gov/feeds/press_all.xml")
TIMEZONE = os.getenv("OVERSER_TZ", "Europe/London")

client = OpenAI()  # reads OPENAI_API_KEY from env
app = FastAPI(title="AI Overseer (Realtime EURUSD)", version="2.0.0")


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
    }
}

SYSTEM = (
    "You are the AI Overseer for a EURUSD day-trading bot. "
    "You will receive (1) a 'snapshot' with current market/indicator values from the broker "
    "and (2) a 'context' containing reputable sources (central bank RSS and TradingEconomics calendar). "
    "Use ONLY the given context plus snapshot; do not invent sources. "
    "Return ONLY JSON matching the schema. "
    "When news is market-moving (e.g., rate decisions, CPI, NFP), or when spreads are elevated, prefer HOLD or FLATTEN. "
    "If evidence is mixed or stale, lower confidence. "
)

# ---------- Helpers ----------
def _auth_or_403(header_value: Optional[str]):
    if not ADVISOR_TOKEN:
        raise HTTPException(500, "Server missing ADVISOR_TOKEN")
    if header_value != ADVISOR_TOKEN:
        raise HTTPException(403, "Forbidden")

def _tz_now():
    try:
        # Python 3.9+ ZoneInfo
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(TIMEZONE))
    except Exception:
        return datetime.now()

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
    d1 = date.today().isoformat()
    d2 = d1
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
                # Build a compact title
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
    # Collect reputable context
    context = {
        "feeds": {
            "ecb": fetch_rss(ECB_RSS, limit=5),
            "fed": fetch_rss(FED_RSS, limit=5),
            "calendar": fetch_te_calendar(),
        },
        "as_of": _iso(datetime.now(timezone.utc))
    }
    # Prepare a short, model-friendly string
    bullets = []
    for label in ("ecb", "fed", "calendar"):
        for item in context["feeds"][label]:
            ttl = item["title"]
            pub = item.get("published")
            bullets.append(f"- {label.upper()}: {ttl} [{pub}] -> {item['url']}")
    # The overseer will pass both human-readable bullets and the structured list
    ctx_text = "LATEST CONTEXT:\n" + "\n".join(bullets[:20])
    return {"context_text": ctx_text, "sources": sum(context["feeds"].values(), []), "as_of": context["as_of"]}

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "ebc_rss": ECB_RSS, "fed_rss": FED_RSS}

@app.post("/advice")
async def advice(request: Request, x_advisor_token: Optional[str] = Header(None)):
    _auth_or_403(x_advisor_token)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    # The GUI sends raw snapshot as body, but support {"snapshot": {...}} too
    snapshot = payload.get("snapshot") if isinstance(payload, dict) and "snapshot" in payload else payload
    if not isinstance(snapshot, dict):
        raise HTTPException(400, "Body must be a JSON object or {'snapshot': {...}}")

    # Build reputable context (ECB/Fed RSS + TE calendar)
    ctx = build_context(snapshot)

    # Compose the model input
    user_input = json.dumps({
        "as_of": ctx["as_of"],
        "snapshot": snapshot,
        "context": ctx["context_text"]
    }, separators=(",", ":"))

    try:
        resp = client.responses.create(
            model=MODEL,
            instructions=SYSTEM,
            input=user_input,
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
            temperature=0.2,
            store=False
        )
        text = getattr(resp, "output_text", None)
        if not text:
            text = json.dumps(resp.model_dump(), default=str)

        # Validate and attach sources we fetched (for verifiable attribution)
        data = json.loads(text)
        out = AdvisorOut(**data)
        out.sources = [AdvisorSource(**s) for s in ctx["sources"]]
        return out.model_dump()
    except Exception as e:
        raise HTTPException(500, f"advisor_error:{e.__class__.__name__}")
