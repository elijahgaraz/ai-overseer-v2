import os, json, traceback, re
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
EUROSTAT_RSS = os.getenv("EUROSTAT_RSS", "https://ec.europa.eu/eurostat/news/rss")

# Optional real-time headline/sentiment enrichers (auto-skip if keys missing)
BING_NEWS_KEY = os.getenv("BING_NEWS_KEY", "")
BING_NEWS_ENDPOINT = os.getenv("BING_NEWS_ENDPOINT", "https://api.bing.microsoft.com/v7.0/news/search")
MARKETAUX_KEY = os.getenv("MARKETAUX_KEY", "")  # https://www.marketaux.com/
MARKETAUX_ENDPOINT = os.getenv("MARKETAUX_ENDPOINT", "https://api.marketaux.com/v1/news/all")

TIMEZONE = os.getenv("OVERSEER_TZ") or os.getenv("OVERSER_TZ") or "Europe/London"

# --- Tunables for caution window & relaxations ---
SOON_HOURS = float(os.getenv("SOON_HOURS", "3"))                 # “imminent” window
MIN_BLOCK_HOURS = float(os.getenv("MIN_BLOCK_HOURS", "1"))       # hard block inside this window
MIN_CONF_FOR_TRADE = float(os.getenv("MIN_CONF_FOR_TRADE", "65"))# min conf to flip HOLD
REQUIRE_EVENT_TIMING_MENTION = os.getenv("REQUIRE_EVENT_TIMING_MENTION", "0") in ("1","true","True")
SUPPRESS_EVENT_TIMING_IN_REASON = os.getenv("SUPPRESS_EVENT_TIMING_IN_REASON", "1") in ("1","true","True")

client = OpenAI(timeout=8.0)  # reads OPENAI_API_KEY
app = FastAPI(title="AI Overseer (Direction + Confidence)", version="3.2.0")

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

# ---------- System prompt (conditional mention of event timing) ----------
_prompt_bits = [
    "You are the AI Overseer for an FX bot. You receive: "
    "(1) a 'snapshot' with current market/indicator values and "
    "(2) a 'context' summarizing reputable sources (ECB/Fed/Eurostat RSS + TradingEconomics calendar + optional headlines/sentiment). ",
    "Using ONLY this information, output direction and confidence percentage:\n"
    "- direction ∈ {BUY, SELL, HOLD}\n"
    "- confidence_pct: 0–100 (as a percentage, not probability)\n",
    f"Prefer HOLD only when market-moving news is imminent (within ~{SOON_HOURS} hours), spreads are elevated vs typical levels, OR signals conflict.\n",
    f"If high-impact events are > {SOON_HOURS} hours away and micro signals align, do NOT be overly cautious: choose BUY/SELL with justified confidence.\n",
    "If RSS feeds have no posts within the last 7 days, state 'no recent official posts'.\n",
    f"When events are between {SOON_HOURS} and 12 hours away, you may still trade if technicals are aligned.\n",
]
if REQUIRE_EVENT_TIMING_MENTION:
    _prompt_bits.append("If the calendar shows upcoming high-impact events, mention their timing explicitly.\n")
SYSTEM = "".join(_prompt_bits) + "Return JSON ONLY conforming to the provided schema."

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
    Defaults for EURUSD.
    """
    s = (sym or "").upper()
    if "EUR" in s and "USD" in s:
        return ["Euro Area", "United States"]
    # sensible default
    return ["Euro Area", "United States"]

# ---------- Enrichers ----------
def fetch_eurostat_recent(limit: int = 5, max_age_days: int = 7) -> List[Dict[str, Any]]:
    return fetch_rss(EUROSTAT_RSS, limit=limit, max_age_days=max_age_days)

def fetch_bing_headlines(symbol: str, count: int = 8, freshness_hours: int = 24) -> List[Dict[str, Any]]:
    """
    Bing News Search (optional). Requires BING_NEWS_KEY.
    Pulls last ~24h headlines for EURUSD/ECB/Fed, sorted by recency.
    """
    if not BING_NEWS_KEY:
        return []
    queries = [
        "EURUSD OR \"EUR/USD\"",
        "ECB OR \"European Central Bank\"",
        "\"Federal Reserve\" OR Fed",
        "Euro area inflation OR Eurozone CPI",
        "US CPI OR Nonfarm payrolls OR FOMC",
    ]
    headers = {"Ocp-Apim-Subscription-Key": BING_NEWS_KEY}
    freshness = "Day" if freshness_hours <= 24 else "Week"
    results: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=6.5, headers=headers) as http:
            for q in queries:
                params = {
                    "q": q,
                    "count": max(2, count // len(queries)),
                    "freshness": freshness,
                    "sortBy": "Date",
                    "safeSearch": "Off",
                    "textDecorations": "false",
                }
                r = http.get(BING_NEWS_ENDPOINT, params=params)
                if r.status_code != 200:
                    continue
                data = r.json()
                for art in (data.get("value") or []):
                    title = art.get("name") or "headline"
                    url = art.get("url") or ""
                    published = art.get("datePublished") or ""
                    pub_dt = _safe_parse_dt(published)
                    if not pub_dt:
                        continue
                    results.append({
                        "title": title,
                        "url": url,
                        "published": pub_dt.isoformat()
                    })
        # De-dup by title/url and keep most recent
        dedup = {}
        for it in sorted(results, key=lambda x: x["published"], reverse=True):
            key = (it["title"], it["url"])
            if key not in dedup:
                dedup[key] = it
        return list(dedup.values())[:count]
    except Exception:
        return []

def fetch_marketaux_sentiment(symbol: str, limit: int = 6, max_age_hours: int = 48) -> List[Dict[str, Any]]:
    """
    Marketaux (optional). Requires MARKETAUX_KEY.
    We query broad FX terms; provider may not tag FX perfectly, so treat as 'bonus' context.
    """
    if not MARKETAUX_KEY:
        return []
    q = "EURUSD OR \"EUR/USD\" OR ECB OR \"Euro area\" OR Eurozone OR FOMC OR \"Federal Reserve\""
    params = {
        "api_token": MARKETAUX_KEY,
        "search": q,
        "language": "en",
        "sort": "published_at:desc",
        "limit": str(limit),
    }
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    try:
        with httpx.Client(timeout=6.5) as http:
            r = http.get(MARKETAUX_ENDPOINT, params=params)
            if r.status_code != 200:
                return []
            data = r.json()
            out: List[Dict[str, Any]] = []
            for art in (data.get("data") or [])[:limit]:
                title = art.get("title") or "news"
                url = art.get("url") or ""
                published = art.get("published_at")
                pub_dt = _safe_parse_dt(published)
                if not pub_dt or pub_dt < cutoff:
                    continue
                sentiment = None
                if isinstance(art.get("entities"), list):
                    for ent in art["entities"]:
                        if isinstance(ent, dict) and "sentiment_score" in ent:
                            sentiment = ent.get("sentiment_score")
                            break
                if sentiment is None:
                    sentiment = art.get("sentiment") or art.get("overall_sentiment_score")
                t = f"{title}"
                if sentiment is not None:
                    t += f" (sentiment={sentiment})"
                out.append({"title": t, "url": url, "published": pub_dt.isoformat()})
            return out
    except Exception:
        return []

# ---------- EURUSD relevance helpers ----------
KEYWORDS_EURUSD = (
    "ecb", "deposit facility", "refi", "rate decision",
    "fomc", "fed",
    "cpi", "pce", "core pce", "nfp", "nonfarm", "payroll", "unemployment",
    "gdp", "pmi", "ism", "ppi", "retail sales"
)

def _hours_until(dt_iso: str) -> float:
    try:
        dt = _safe_parse_dt(dt_iso)
        if not dt:
            return float("inf")
        now = datetime.now(timezone.utc)
        return max(0.0, (dt - now).total_seconds() / 3600.0)
    except Exception:
        return float("inf")

def next_relevant_event_hours(cal_items: List[Dict[str, Any]]) -> float:
    """
    Return hours until the nearest likely EURUSD-relevant event by simple keyword match.
    """
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
        t = (it.get("title") or "").lower()
        if any(k in t for k in KEYWORDS_EURUSD):
            out.append(it)
    return out

# ---------- Context builder ----------
def build_context(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    # Determine relevant countries
    sym = (snapshot.get("symbol") or snapshot.get("Symbol") or "").upper()
    countries = _countries_for_symbol(sym)

    # Official & baseline
    ecb_items = fetch_rss(ECB_RSS, 5, max_age_days=7)
    fed_items = fetch_rss(FED_RSS, 5, max_age_days=7)
    eurostat_items = fetch_eurostat_recent(5, max_age_days=7)

    cal_items_all = fetch_te_calendar(countries)
    cal_items = filter_calendar_for_eurusd(cal_items_all)
    hrs_next = next_relevant_event_hours(cal_items)
    soon_threshold = SOON_HOURS  # configurable

    # Enrichers (optional)
    headlines = fetch_bing_headlines(sym or "EURUSD", count=8, freshness_hours=24)
    marketaux_items = fetch_marketaux_sentiment(sym or "EURUSD", limit=6, max_age_hours=48)

    now_utc = datetime.now(timezone.utc)
    now_iso = _iso(now_utc)

    def bullets(label: str, items: List[Dict[str, Any]]) -> List[str]:
        if not items:
            if label in ("ecb", "fed", "eurostat"):
                return [f"- {label.upper()}: No recent official posts (≤7d)."]
            elif label in ("headlines", "marketaux"):
                return [f"- {label.upper()}: No recent items (provider disabled or no results)."]
            else:
                return [f"- {label.upper()}: No high-impact items in the window."]
        return [f"- {label.upper()}: {it['title']} [{it.get('published')}] -> {it['url']}" for it in items]

    enabled_bits = []
    if BING_NEWS_KEY: enabled_bits.append("BingNews")
    if MARKETAUX_KEY: enabled_bits.append("Marketaux")
    enabled_str = ", ".join(enabled_bits) if enabled_bits else "none"

    next_evt_str = "∞" if hrs_next == float("inf") else f"{round(hrs_next,1)}"
    filtered_note = f"(filtered {len(cal_items)} of {len(cal_items_all)} events for EURUSD relevance)"

    header = (
        f"LATEST CONTEXT as of {now_iso} (tz=UTC; local_tz={TIMEZONE}):\n"
        "• RSS recency: ≤7 days; Economic calendar window: today + next 3 days; importance=high\n"
        f"• Countries for calendar: {', '.join(countries)}\n"
        f"• Next relevant high-impact event in ≈ {next_evt_str}h (imminent<{soon_threshold}h) {filtered_note}\n"
        f"• Optional enrichers enabled: {enabled_str}\n"
    )

    lines: List[str] = []
    lines += bullets("ecb", ecb_items)
    lines += bullets("fed", fed_items)
    lines += bullets("eurostat", eurostat_items)
    lines += bullets("calendar", cal_items)
    lines += bullets("headlines", headlines)
    lines += bullets("marketaux", marketaux_items)

    ctx_text = header + "\n".join(lines[:40])  # keep prompt tidy
    return {"context_text": ctx_text, "as_of": now_iso, "hrs_next": hrs_next}

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
            ch = client.chat_completions.create(  # backward-compat if needed
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM + "\nReturn JSON ONLY with keys: direction, confidence_pct, reason, as_of."},
                    {"role": "user", "content": user_input},
                ],
            )
            return json.loads(ch.choices[0].message.content)

# --- Micro-bias & strength inference ---
def _infer_bias_from_snapshot(snap: Dict[str, Any]) -> Optional[str]:
    """
    Heuristic: use EMA fast vs slow; fallback to RSI bands.
    Returns 'BUY' | 'SELL' | None
    """
    s = {k.lower(): v for k, v in snap.items()}

    # EMA bias
    ef = s.get("ema_fast") or s.get("ema20") or s.get("ema_20") or s.get("ema_short") or s.get("ema10") or s.get("ema_10")
    es = s.get("ema_slow") or s.get("ema50") or s.get("ema_50") or s.get("ema_long") or s.get("ema200") or s.get("ema_200")
    if isinstance(ef, (int, float)) and isinstance(es, (int, float)):
        if ef > es:
            return "BUY"
        elif ef < es:
            return "SELL"

    # RSI fallback
    rsi = s.get("rsi") or s.get("rsi14") or s.get("rsi_14")
    if isinstance(rsi, (int, float)):
        if rsi >= 55:
            return "BUY"
        if rsi <= 45:
            return "SELL"

    return None

def _strong_alignment(snap: Dict[str, Any]) -> bool:
    """
    Optional strength check used to allow trades even when an event is soon (but not imminent).
    """
    s = {k.lower(): v for k, v in snap.items()}
    # EMA + RSI + ADX combo
    bias = _infer_bias_from_snapshot(snap)
    rsi = s.get("rsi") or s.get("rsi14") or s.get("rsi_14")
    adx = s.get("adx") or s.get("adx14") or s.get("adx_14")
    score = 0
    if bias in ("BUY","SELL"): score += 1
    if isinstance(rsi, (int,float)) and (rsi >= 58 or rsi <= 42): score += 1
    if isinstance(adx, (int,float)) and adx >= 18: score += 1
    return score >= 2

# --- Reason sanitizer (optional) ---
_REASON_TIME_RE = re.compile(r"(?i)\b(within|in)\s+(the\s+)?next?\s+\d+(\.\d+)?\s*hours?\b[^.]*\.?")
def _sanitize_reason(text: str) -> str:
    if not text:
        return text
    if SUPPRESS_EVENT_TIMING_IN_REASON:
        text = __REASON_TIME_RE.sub("", text).strip()
        # Clean up double spaces and trailing commas
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+,", ",", text)
    return text

# Core decision helper
def get_decision(snapshot: Dict[str, Any]) -> SimpleAdviceOut:
    ctx = build_context(snapshot)
    user_input = json.dumps({
        "as_of": ctx["as_of"],
        "snapshot": snapshot,
        "context": ctx["context_text"]
    }, separators=(",", ":"))
    data = model_decision_json(user_input)

    # --- Post-LLM relaxation shim ---
    try:
        dir0 = (data.get("direction") or "HOLD").upper()
        conf0 = float(data.get("confidence_pct") or 0.0)
        hrs_next = ctx.get("hrs_next", float("inf"))

        # Hard block only if event is very close
        if hrs_next <= MIN_BLOCK_HOURS:
            data["direction"] = "HOLD"
        else:
            # If model said HOLD but confidence decent and event not imminent, or technicals are strong, flip
            can_flip_window = (hrs_next >= SOON_HOURS) or ((MIN_BLOCK_HOURS < hrs_next < SOON_HOURS) and _strong_alignment(snapshot))
            if dir0 == "HOLD" and conf0 >= MIN_CONF_FOR_TRADE and can_flip_window:
                bias = _infer_bias_from_snapshot(snapshot)
                if bias in ("BUY", "SELL"):
                    data["direction"] = bias
                    note = f"Relaxed rule: next event ~{round(hrs_next,1)}h; conf {conf0:.0f}%."
                    reason = (data.get("reason") or "").strip()
                    data["reason"] = (reason + (" " if reason else "") + note).strip()

        # Optionally remove “in the next X hours” phrasing from the reason
        data["reason"] = _sanitize_reason(data.get("reason") or "")

    except Exception:
        # Don’t let the shim break main flow
        pass

    return SimpleAdviceOut(**data)

# ---------- Routes ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL,
        "ecb_rss": ECB_RSS,
        "fed_rss": FED_RSS,
        "eurostat_rss": EUROSTAT_RSS,
        "bing_news_enabled": bool(BING_NEWS_KEY),
        "marketaux_enabled": bool(MARKETAUX_KEY),
        "soon_hours": SOON_HOURS,
        "min_block_hours": MIN_BLOCK_HOURS,
        "min_conf_for_trade": MIN_CONF_FOR_TRADE,
        "require_event_timing_mention": REQUIRE_EVENT_TIMING_MENTION,
        "suppress_event_timing_in_reason": SUPPRESS_EVENT_TIMING_IN_REASON,
    }

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
