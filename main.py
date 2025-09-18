import os, json, traceback, re
from typing import Optional, Dict, Any, List
from datetime import datetime, date, timezone, timedelta
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
import httpx, feedparser
from openai import OpenAI
from email.utils import parsedate_to_datetime

# ---------- Config ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ADVISOR_TOKEN = os.getenv("ADVISOR_TOKEN", "")
TE_API_KEY = os.getenv("TE_API_KEY", "guest:guest")
ECB_RSS = os.getenv("ECB_RSS", "https://www.ecb.europa.eu/press/rss/press.html")
FED_RSS = os.getenv("FED_RSS", "https://www.federalreserve.gov/feeds/press_all.xml")
EUROSTAT_RSS = os.getenv("EUROSTAT_RSS", "https://ec.europa.eu/eurostat/news/rss")

# Optional real-time headline/sentiment enrichers
BING_NEWS_KEY = os.getenv("BING_NEWS_KEY", "")
BING_NEWS_ENDPOINT = os.getenv("BING_NEWS_ENDPOINT", "https://api.bing.microsoft.com/v7.0/news/search")
MARKETAUX_KEY = os.getenv("MARKETAUX_KEY", "")
MARKETAUX_ENDPOINT = os.getenv("MARKETAUX_ENDPOINT", "https://api.marketaux.com/v1/news/all")

TIMEZONE = os.getenv("OVERSEER_TZ", "Europe/London")

# --- Tunables for caution window & relaxations (MOVED ABOVE PROMPT) ---
SOON_HOURS = float(os.getenv("SOON_HOURS", "3"))
MIN_BLOCK_HOURS = float(os.getenv("MIN_BLOCK_HOURS", "1"))
MIN_CONF_FOR_TRADE = float(os.getenv("MIN_CONF_FOR_TRADE", "65"))

REQUIRE_EVENT_TIMING_MENTION = os.getenv("REQUIRE_EVENT_TIMING_MENTION", "0") in ("1", "true", "True")
SUPPRESS_EVENT_TIMING_IN_REASON = os.getenv("SUPPRESS_EVENT_TIMING_IN_REASON", "1") in ("1", "true", "True")

SHOW_NEXT_EVENT_IN_CONTEXT = os.getenv("SHOW_NEXT_EVENT_IN_CONTEXT", "0") in ("1", "true", "True")
GRACE_MINUTES = float(os.getenv("GRACE_MINUTES", "10"))
ALLOW_TRADE_DURING_EVENT = os.getenv("ALLOW_TRADE_DURING_EVENT", "0") in ("1", "true", "True")

client = OpenAI(timeout=8.0)
app = FastAPI(title="AI Overseer (Direction + Confidence)", version="3.3.1")

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
            "direction": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "confidence_pct": {"type": "number", "minimum": 0, "maximum": 100},
            "reason": {"type": "string"},
            "as_of": {"type": "string"}
        },
        "required": ["direction", "confidence_pct", "reason", "as_of"],
        "additionalProperties": False
    },
    "strict": True
}

# ---------- System prompt (Scalper-aware) ----------
_prompt_bits = [
    # Role & mission
    "You are the AI Overseer for a high-frequency FX scalping bot (EUR/USD focus). "
    "You receive: (1) a 'snapshot' with current market/indicator values and "
    "(2) a 'context' summarizing reputable sources (ECB/Fed/Eurostat RSS + TradingEconomics calendar "
    "+ optional headlines/sentiment). Using ONLY this information, output a trading call.\n\n"

    # Output contract
    "OUTPUT (JSON ONLY):\n"
    "- direction ∈ {BUY, SELL, HOLD}\n"
    "- confidence_pct: 0–100 (as a percentage, not probability)\n"
    "- reason: 1–2 short sentences citing key factors used\n"
    "- as_of: ISO8601 timestamp (echo the input timestamp)\n\n"

    # Operating style (scalper)
    "OPERATING STYLE (Scalping):\n"
    "- Target micro-moves on M1/M5; typical hold time 1–5 minutes (extend to 5–15 only if momentum is strong).\n"
    "- Aim for tight protection ~0.8–1.2× ATR (M1/M5) and RR ≈ 1:1 to 1:1.5.\n"
    f"- Prefer HOLD only when market-moving news is imminent (within ~{SOON_HOURS}h), spreads are elevated vs typical, "
    "or signals conflict.\n"
    f"- If high-impact events are > {SOON_HOURS}h away and micro signals align, do NOT be overly cautious: choose BUY/SELL with justified confidence.\n"
    "If RSS has no posts ≤7d, you may note that briefly in reasoning as context.\n\n"

    # What we expect in the snapshot (so the model knows what it's looking at)
    "SNAPSHOT MAY INCLUDE (examples; use whatever is provided):\n"
    "- price, spread_pips, symbol, timestamp\n"
    "- ATR_M1/ATR_M5, EMA fast/slow on M1 & M5, price_vs_ema flags\n"
    "- RSI_M1/RSI_M5, ADX_M5\n"
    "- nearest support/resistance, recent swing high/low, price action notes\n"
    "- session/overlap flags, news blackout flags, max allowed spread\n\n"

    # Hard trade filters
    "HARD FILTERS (if any fail → direction=HOLD):\n"
    "- Spread filter: if spread_pips exceeds bot’s limit or is clearly elevated vs typical conditions, HOLD.\n"
    "- News blackout: if high-impact events are imminent (within the blackout window) and conditions are not exceptionally aligned, HOLD.\n"
    "- Volatility floor: if ATR_M1 and ATR_M5 are both very low (illiquid/chop), HOLD.\n\n"

    # Direction logic
    "DIRECTION BIAS (use confluence):\n"
    "- Bullish: price above slow EMA on both M1 & M5 AND fast>slow on both → BUY bias.\n"
    "- Bearish: price below slow EMA on both M1 & M5 AND fast<slow on both → SELL bias.\n"
    "- If M1 and M5 disagree OR ADX_M5 is weak, prefer HOLD unless there is a very clean pullback-to-EMA or S/R rejection with momentum confirmation.\n\n"

    # Entry patterns / what to reward or avoid in reasoning
    "ENTRY PATTERNS WE LIKE:\n"
    "- Pullback to EMA20/50 or prior S/R with clear rejection (engulfing/pin, strong close) in the bias direction.\n"
    "- Break → retest → continuation with momentum.\n"
    "AVOID:\n"
    "- Choppy ranges (weak ADX), conflicting M1/M5, RSI extremes counter to intended direction without fresh confirmation.\n\n"

    # Confidence guidance
    "CONFIDENCE GUIDANCE:\n"
    "- Increase confidence when M1 & M5 align, spread/ATR are OK, and price action confirms (clean rejection or break+retest).\n"
    "- Reduce confidence for mixed signals, nearby opposing S/R shrinking RR, or approaching events.\n\n"

    # Policy reminders (tie into your env-driven guardrails)
    f"When events are between {SOON_HOURS} and 12 hours away, you may still trade if technicals align. "
    "If calendar/RSS has nothing material, do not invent signals.\n",
]
if REQUIRE_EVENT_TIMING_MENTION:
    _prompt_bits.append(
        "If the calendar shows upcoming high-impact events, briefly mention their timing in the reason.\n"
    )
SYSTEM = "".join(_prompt_bits) + "Return JSON ONLY conforming to the provided schema."

# ---------- Helpers ----------
def _auth_or_403(header_value: Optional[str]):
    if not ADVISOR_TOKEN:
        raise HTTPException(500, "Server missing ADVISOR_TOKEN")
    if header_value != ADVISOR_TOKEN:
        raise HTTPException(403, "Forbidden")

def _safe_parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
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
    s = (sym or "").upper()
    if "EUR" in s and "USD" in s:
        return ["Euro Area", "United States"]
    return ["Euro Area", "United States"]

def fetch_bing_headlines(symbol: str, count: int = 8, freshness_hours: int = 24) -> List[Dict[str, Any]]:
    if not BING_NEWS_KEY:
        return []
    queries = [ "EURUSD OR \"EUR/USD\"", "ECB OR \"European Central Bank\"", "\"Federal Reserve\" OR Fed", "Euro area inflation OR Eurozone CPI", "US CPI OR Nonfarm payrolls OR FOMC" ]
    headers = {"Ocp-Apim-Subscription-Key": BING_NEWS_KEY}
    freshness = "Day" if freshness_hours <= 24 else "Week"
    results: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=6.5, headers=headers) as http:
            for q in queries:
                params = { "q": q, "count": max(2, count // len(queries)), "freshness": freshness, "sortBy": "Date", "safeSearch": "Off", "textDecorations": "false" }
                r = http.get(BING_NEWS_ENDPOINT, params=params)
                if r.status_code != 200: continue
                data = r.json()
                for art in (data.get("value") or []):
                    pub_dt = _safe_parse_dt(art.get("datePublished") or "")
                    if not pub_dt: continue
                    results.append({ "title": art.get("name") or "headline", "url": art.get("url") or "", "published": pub_dt.isoformat() })
        dedup = {}
        for it in sorted(results, key=lambda x: x["published"], reverse=True):
            key = (it["title"], it["url"])
            if key not in dedup:
                dedup[key] = it
        return list(dedup.values())[:count]
    except Exception:
        return []

def fetch_marketaux_sentiment(symbol: str, limit: int = 6, max_age_hours: int = 48) -> List[Dict[str, Any]]:
    if not MARKETAUX_KEY:
        return []
    q = "EURUSD OR \"EUR/USD\" OR ECB OR \"Euro area\" OR Eurozone OR FOMC OR \"Federal Reserve\""
    params = { "api_token": MARKETAUX_KEY, "search": q, "language": "en", "sort": "published_at:desc", "limit": str(limit) }
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    try:
        with httpx.Client(timeout=6.5) as http:
            r = http.get(MARKETAUX_ENDPOINT, params=params)
            if r.status_code != 200: return []
            data = r.json()
            out: List[Dict[str, Any]] = []
            for art in (data.get("data") or [])[:limit]:
                pub_dt = _safe_parse_dt(art.get("published_at"))
                if not pub_dt or pub_dt < cutoff: continue
                sentiment = None
                if isinstance(art.get("entities"), list):
                    for ent in art["entities"]:
                        if isinstance(ent, dict) and "sentiment_score" in ent:
                            sentiment = ent.get("sentiment_score")
                            break
                if sentiment is None:
                    sentiment = art.get("sentiment") or art.get("overall_sentiment_score")
                t = f"{art.get('title') or 'news'}"
                if sentiment is not None:
                    t += f" (sentiment={sentiment})"
                out.append({"title": t, "url": art.get("url") or "", "published": pub_dt.isoformat()})
            return out
    except Exception:
        return []

KEYWORDS_EURUSD = ("ecb", "deposit facility", "refi", "rate decision", "fomc", "fed", "cpi", "pce", "core pce", "nfp", "nonfarm", "payroll", "unemployment", "gdp", "pmi", "ism", "ppi", "retail sales")

def _hours_until(dt_iso: str) -> float:
    try:
        dt = _safe_parse_dt(dt_iso)
        if not dt: return float("inf")
        now = datetime.now(timezone.utc)
        delta_hours = (dt - now).total_seconds() / 3600.0
        grace_h = GRACE_MINUTES / 60.0
        if abs(delta_hours) <= grace_h or delta_hours < 0:
            return float("inf")
        return delta_hours
    except Exception:
        return float("inf")

def next_relevant_event_hours(cal_items: List[Dict[str, Any]]) -> float:
    best = float("inf")
    for it in cal_items:
        title = (it.get("title") or "").lower()
        when = it.get("published") or it.get("date") or it.get("when")
        if any(k in title for k in KEYWORDS_EURUSD):
            h = _hours_until(when)
            if h < best:
                best = h
    return best

def filter_calendar_for_eurusd(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        if any(k in (it.get("title") or "").lower() for k in KEYWORDS_EURUSD):
            out.append(it)
    return out

def build_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    sym = (snapshot.get("symbol") or snapshot.get("Symbol") or "").upper()
    countries = _countries_for_symbol(sym)
    ecb_items = fetch_rss(ECB_RSS, 5, 7)
    fed_items = fetch_rss(FED_RSS, 5, 7)
    eurostat_items = fetch_rss(EUROSTAT_RSS, 5, 7)
    cal_items_all = fetch_te_calendar(countries)
    cal_items = filter_calendar_for_eurusd(cal_items_all)
    hrs_next = next_relevant_event_hours(cal_items)
    headlines = fetch_bing_headlines(sym or "EURUSD", 8, 24)
    marketaux_items = fetch_marketaux_sentiment(sym or "EURUSD", 6, 48)
    now_iso = datetime.now(timezone.utc).isoformat()

    def bullets(label: str, items: List[Dict[str, Any]]) -> List[str]:
        lab = str(label).upper()
        if not items:
            if label in ("ecb", "fed", "eurostat"): return [f"- {lab}: No recent official posts (≤7d)."]
            elif label in ("headlines", "marketaux"): return [f"- {lab}: No recent items (provider disabled or no results)."]
            else: return [f"- {lab}: No high-impact items in the window."]
        return [f"- {lab}: {it.get('title','item')} [{it.get('published')}] -> {it.get('url','')}" for it in items]

    enabled_bits = []
    if BING_NEWS_KEY: enabled_bits.append("BingNews")
    if MARKETAUX_KEY: enabled_bits.append("Marketaux")
    enabled_str = ", ".join(enabled_bits) if enabled_bits else "none"
    next_evt_str = "∞" if hrs_next == float("inf") else f"{round(hrs_next,1)}"

    header = (
        f"LATEST CONTEXT as of {now_iso} (tz=UTC; local_tz={TIMEZONE}):\n"
        "• RSS recency: ≤7 days; Economic calendar window: today + next 3 days; importance=high\n"
        f"• Countries for calendar: {', '.join(countries)}\n"
    )
    if SHOW_NEXT_EVENT_IN_CONTEXT:
        header += f"• Next relevant high-impact event in ≈ {next_evt_str}h (imminent<{SOON_HOURS}h) (filtered {len(cal_items)} of {len(cal_items_all)} events for relevance)\n"
    header += f"• Optional enrichers enabled: {enabled_str}\n"

    lines = bullets("ecb", ecb_items) + bullets("fed", fed_items) + bullets("eurostat", eurostat_items) + bullets("calendar", cal_items) + bullets("headlines", headlines) + bullets("marketaux", marketaux_items)
    return {"context_text": header + "\n".join(lines[:40]), "as_of": now_iso, "hrs_next": hrs_next}

def model_decision_json(user_input: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_input}]
    try:
        ch = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA}
        )
        return json.loads(ch.choices[0].message.content)
    except Exception:
        try:
            ch = client.chat.completions.create(model=MODEL, messages=messages, response_format={"type": "json_object"})
            return json.loads(ch.choices[0].message.content)
        except Exception:
            ch = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM + "\nReturn JSON ONLY with keys: direction, confidence_pct, reason, as_of."},
                    {"role": "user", "content": user_input}
                ]
            )
            return json.loads(ch.choices[0].message.content)

def _infer_bias_from_snapshot(snap: Dict[str, Any]) -> Optional[str]:
    s = {k.lower(): v for k, v in snap.items()}
    ef = s.get("ema_fast"); es = s.get("ema_slow")
    if isinstance(ef, (int, float)) and isinstance(es, (int, float)):
        if ef > es: return "BUY"
        elif ef < es: return "SELL"
    rsi = s.get("rsi")
    if isinstance(rsi, (int, float)):
        if rsi >= 55: return "BUY"
        if rsi <= 45: return "SELL"
    return None

def _strong_alignment(snap: Dict[str, Any]) -> bool:
    s = {k.lower(): v for k, v in snap.items()}
    bias = _infer_bias_from_snapshot(snap)
    rsi = s.get("rsi"); adx = s.get("adx")
    score = 0
    if bias in ("BUY", "SELL"): score += 1
    if isinstance(rsi, (int, float)) and (rsi >= 58 or rsi <= 42): score += 1
    if isinstance(adx, (int, float)) and adx >= 18: score += 1
    return score >= 2

_REASON_TIME_RE = re.compile(r"(?i)\b(within|in)\s+(the\s+)?next?\s+\d+(\.\d+)?\s*hours?\b[^.]*\.?")
def _sanitize_reason(text: str) -> str:
    if not text or not SUPPRESS_EVENT_TIMING_IN_REASON: return text or ""
    text = _REASON_TIME_RE.sub("", text).strip()
    text = re.sub(r"\s{2,}", " ", text); text = re.sub(r"\s+,", ",", text)
    return text

def get_decision(snapshot: Dict[str, Any]) -> SimpleAdviceOut:
    ctx = build_context(snapshot)
    user_input = json.dumps({"as_of": ctx["as_of"], "snapshot": snapshot, "context": ctx["context_text"]}, separators=(",", ":"))
    data = model_decision_json(user_input)
    try:
        dir0 = (data.get("direction") or "HOLD").upper()
        conf0 = float(data.get("confidence_pct") or 0.0)
        hrs_next = ctx.get("hrs_next", float("inf"))
        can_flip = False
        if hrs_next >= SOON_HOURS:
            can_flip = True
        elif MIN_BLOCK_HOURS < hrs_next < SOON_HOURS and _strong_alignment(snapshot):
            can_flip = True
        elif hrs_next <= MIN_BLOCK_HOURS and ALLOW_TRADE_DURING_EVENT and _strong_alignment(snapshot):
            can_flip = True
        if dir0 == "HOLD" and conf0 >= MIN_CONF_FOR_TRADE and can_flip:
            bias = _infer_bias_from_snapshot(snapshot)
            if bias in ("BUY", "SELL"):
                data["direction"] = bias
                note = f"Relaxed rule applied (hrs_next≈{ '∞' if hrs_next==float('inf') else round(hrs_next,1) }, conf {conf0:.0f}%)."
                reason = (data.get("reason") or "").strip()
                data["reason"] = (reason + (" " if reason else "") + note).strip()
        data["reason"] = _sanitize_reason(data.get("reason") or "")
    except Exception:
        pass
    return SimpleAdviceOut(**data)

@app.get("/health")
def health():
    return { "ok": True, "model": MODEL, "bing_news_enabled": bool(BING_NEWS_KEY), "marketaux_enabled": bool(MARKETAUX_KEY), "soon_hours": SOON_HOURS }

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
        return { "action": mapped, "confidence": round((out.confidence_pct or 0.0) / 100.0, 4), "reason": out.reason or "" }
    except Exception:
        return { "action": "skip", "confidence": 0.0, "reason": "advisor_error" }
