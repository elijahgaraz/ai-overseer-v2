import os, json, traceback
from typing import Optional, Dict, Any, List
from datetime import datetime, date, timezone, timedelta
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
import httpx, feedparser
from openai import OpenAI
from email.utils import parsedate_to_datetime

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
    "required": ["direction","confidence_pct","reason","as_of"],
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
    "If the calendar shows upcoming high-impact events within 72h, mention that explicitly. "
    "If RSS feeds have no posts within the last 7 days, state 'no recent official posts' rather than implying staleness. "
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

def _safe_parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # try RFC-2822 first (common in RSS), then ISO-8601
    try:
        dt = parsedate_to_datetime(s)
        # ensure timezone-aware in UTC
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc) if dt else None
    except Exception:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

def fetch_te_calendar(countries: List[str]) -> List[Dict[str, Any]]:
    """
    High-impact economic calendar for today + next 3 days.
    """
    d1 = date.today().isoformat()
    d2 = (date.today() + timedelta(days=3)).isoformat()
    country_param = ",".join(countries)
    url = (
        "https://api.tradingeconomics.com/calendar?"
        f"country={country_param}&importance=3&d1={d1}&d2={d2}&format=json&c={TE_API_KEY}"
    )
    try:
        with httpx.Client(timeout=5.5) as http:
            r = http.get(url)
            if r.status_code != 200:
                return []
            data = r.json()
            out = []
            for it in data[:50]:
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

def fetch_rss(url: str, limit: int = 5, max_age_days: int = 7) -> List[Dict[str, Any]]:
    """
    RSS fetch with recency filter (≤ max_age_days). Skips unknown/naive timestamps.
    """
    out: List[Dict[str, Any]] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    try:
        feed = feedparser.parse(url)
        for e in (feed.entries or [])[:limit]:
            raw_pub = getattr(e, "published", None) or getattr(e, "updated", None)
            pub_dt = _safe_parse_dt(raw_pub)
            if not pub_dt or pub_dt < cutoff:
                continue
            out.append({
                "title": getattr(e, "title", "item"),
                "url": getattr(e, "link", url),
                "published": pub_dt.isoformat()
            })
    except Exception:
        pass
    return out

def _countries_for_symbol(sym: str) -> List[str]:
    """
    Choose relevant countries for TE calendar based on the pair symbol.
    Falls back to a broad set if symbol is missing.
    """
    s = (sym or "").upper()
    # Default coverage
    countries = ["United Kingdom", "United States", "Euro Area"]
    if "GBP" in s and "USD" in s:
        return ["United Kingdom", "United States"]
    if "EUR" in s and "USD" in s:
        return ["Euro Area", "United States"]
    if "GBP" in s and "EUR" in s:
        return ["United Kingdom", "Euro Area"]
    # Add others as needed
    return countries

def build_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    # Determine relevant countries
    sym = (snapshot.get("symbol") or snapshot.get("Symbol") or "").upper()
    countries = _countries_for_symbol(sym)

    # Fresh RSS (≤7 days) and upcoming calendar (today + 3 days)
    ecb_items = fetch_rss(ECB_RSS, 5, max_age_days=7)
    fed_items = fetch_rss(FED_RSS, 5, max_age_days=7)
    cal_items = fetch_te_calendar(countries)

    now_utc = datetime.now(timezone.utc)
    now_iso = _iso(now_utc)

    def bullets(label: str, items: List[Dict[str, Any]]) -> List[str]:
        if not items:
            if label in ("ecb", "fed"):
                return [f"- {label.upper()}: No recent official posts (≤7d)."]
            else:
                return [f"- {label.upper()}: No high-impact items in the window."]
        return [f"- {label.upper()}: {it['title']} [{it.get('published')}] -> {it['url']}" for it in items]

    header = (
        f"LATEST CONTEXT as of {now_iso} (tz=UTC; local_tz={TIMEZONE}):\n"
        "• RSS recency: ≤7 days; Economic calendar window: today + next 3 days; importance=high\n"
        f"• Countries for calendar: {', '.join(countries)}\n"
    )
    lines = []
    lines += bullets("ecb", ecb_items)
    lines += bullets("fed", fed_items)
    lines += bullets("calendar", cal_items)

    ctx_text = header + "\n".join(lines[:30])
    return {"context_text": ctx_text, "as_of": now_iso}

# --- Model call with robust fallbacks (no temperature) ---
def model_decision_json(user_input: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_input},
    ]
    # Try JSON Schema first
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
