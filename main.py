import os, json, traceback
from typing import Optional, Dict, Any, List
from datetime import datetime, date, timezone
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
import httpx, feedparser
from openai import OpenAI

# ---------- Config ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # gpt-5, gpt-5-mini, or gpt-4o-mini are fine
ADVISOR_TOKEN = os.getenv("ADVISOR_TOKEN", "")
TE_API_KEY = os.getenv("TE_API_KEY", "guest:guest")
ECB_RSS = os.getenv("ECB_RSS", "https://www.ecb.europa.eu/press/rss/press.html")
FED_RSS = os.getenv("FED_RSS", "https://www.federalreserve.gov/feeds/press_all.xml")
TIMEZONE = os.getenv("OVERSEER_TZ") or os.getenv("OVERSER_TZ") or "Europe/London"

client = OpenAI(timeout=8.0)  # reads OPENAI_API_KEY
app = FastAPI(title="AI Overseer (Direction + Confidence)", version="3.0.0")

# ---------- Models ----------
class SimpleAdviceOut(BaseModel):
    direction: str = Field("HOLD", description="BUY | SELL | HOLD")
    confidence_pct: float = Field(0.0, ge=0.0, le=100.0)
    reason: Optional[str] = ""
    as_of: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

JSON_SCHEMA = {
  "name": "EntryAdvice",
  "schema": {
    "type": "object",
    "properties": {
      "direction": {"type":"string","enum":["BUY","SELL","HOLD"]},
      "confidence_pct": {"type":"number","minimum":0,"maximum":100},
      "reason": {"type":"string"},
      "as_of": {"type":"string"}
    },
    "required": ["direction","confidence_pct","reason","as_of"],  # <— add "reason"
    "additionalProperties": False
  },
  "strict": True
}


SYSTEM = (
    "You are the AI Overseer for an FX bot. You receive: "
    "(1) a 'snapshot' with current market/indicator values and "
    "(2) a 'context' summarizing reputable sources (ECB/Fed RSS + TradingEconomics calendar). "
    "Using ONLY this information, output direction and confidence percentage: "
    "- direction ∈ {BUY, SELL, HOLD}\n"
    "- confidence_pct: 0–100 (as a percentage, not probability)\n"
    "Prefer HOLD near market-moving news (e.g., CPI/NFP/rate decisions) or elevated spreads. "
    "If signals are mixed or context is stale, LOWER confidence. "
    "Return JSON ONLY conforming to the provided schema."
)

# ---------- Helpers ----------
def _auth_or_403(header_value: Optional[str]):
    if not ADVISOR_TOKEN:
        raise HTTPException(500, "Server missing ADVISOR_TOKEN")
    if header_value != ADVISOR_TOKEN:
        raise HTTPException(403, "Forbidden")

def _iso(dt: datetime) -> str:
    return dt.isoformat()

def fetch_te_calendar() -> List[Dict[str, Any]]:
    d1 = date.today().isoformat()
    url = (
        "https://api.tradingeconomics.com/calendar?"
        f"country=United%20States,Euro%20Area&importance=3&d1={d1}&d2={d1}&format=json&c={TE_API_KEY}"
    )
    try:
        with httpx.Client(timeout=5.5) as http:
            r = http.get(url)
            if r.status_code != 200:
                return []
            data = r.json()
            out = []
            for it in data[:30]:
                title = it.get("Event") or it.get("Title") or "Event"
                when = it.get("DateTime") or it.get("DateUtc") or it.get("Date")
                country = it.get("Country") or ""
                link = it.get("SourceUrl") or "https://docs.tradingeconomics.com/economic_calendar/snapshot/"
                actual = it.get("Actual"); forecast = it.get("Forecast"); previous = it.get("Previous")
                t = f"{country}: {title} (act={actual}, fcst={forecast}, prev={previous})"
                out.append({"title": t, "url": link, "published": when})
            return out
    except Exception:
        return []

def fetch_rss(url: str, limit: int = 5) -> List[Dict[str, Any]]:
    out = []
    try:
        feed = feedparser.parse(url)
        for e in (feed.entries or [])[:limit]:
            out.append({
                "title": getattr(e, "title", "item"),
                "url": getattr(e, "link", url),
                "published": getattr(e, "published", None) or getattr(e, "updated", None)
            })
    except Exception:
        pass
    return out

def build_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    ctx = {
        "feeds": {
            "ecb": fetch_rss(ECB_RSS, 5),
            "fed": fetch_rss(FED_RSS, 5),
            "calendar": fetch_te_calendar(),
        },
        "as_of": _iso(datetime.now(timezone.utc))
    }
    bullets: List[str] = []
    for label in ("ecb", "fed", "calendar"):
        for item in ctx["feeds"][label]:
            bullets.append(f"- {label.upper()}: {item['title']} [{item.get('published')}] -> {item['url']}")
    ctx_text = "LATEST CONTEXT:\n" + "\n".join(bullets[:20])
    return {"context_text": ctx_text, "as_of": ctx["as_of"]}

# --- Model call with robust fallbacks (no temperature) ---
def model_decision_json(user_input: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_input},
    ]
    # Try JSON Schema first (if supported by your SDK/model)
    try:
        ch = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        )
        return json.loads(ch.choices[0].message.content)
    except Exception:
        # Fallback: JSON object mode
        try:
            ch = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return json.loads(ch.choices[0].message.content)
        except Exception:
            # Final fallback: instruct JSON-only
            ch = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM + "\nReturn JSON ONLY with keys: direction, confidence_pct, reason, as_of."},
                    {"role": "user", "content": user_input},
                ],
            )
            return json.loads(ch.choices[0].message.content)

# Core decision helper
def get_decision(snapshot: Dict[str, Any]) -> SimpleAdviceOut:
    ctx = build_context(snapshot)
    user_input = json.dumps({
        "as_of": ctx["as_of"],
        "snapshot": snapshot,
        "context": ctx["context_text"]
    }, separators=(",", ":"))
    data = model_decision_json(user_input)
    return SimpleAdviceOut(**data)

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "ecb_rss": ECB_RSS, "fed_rss": FED_RSS}

# Primary endpoint (new minimal schema)
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

# Compatibility endpoint for older bots (maps to long/short/skip + 0..1 confidence)
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
        d = (out.direction or "HOLD").upper()
        mapped = "long" if d == "BUY" else ("short" if d == "SELL" else "skip")
        return {
            "action": mapped,
            "confidence": round((out.confidence_pct or 0.0) / 100.0, 4),
            "sl_pips": 0,
            "tp_pips": 0,
            "reason": out.reason or ""
        }
    except Exception as e:
        print("GBPUSD-ADVICE ERROR:", repr(e))
        traceback.print_exc()
        return {
            "action": "skip",
            "confidence": 0.0,
            "sl_pips": 0,
            "tp_pips": 0,
            "reason": "advisor_error"
        }
