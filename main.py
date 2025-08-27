import os, json, traceback, re, time, random
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, date, timezone, timedelta

from fastapi import FastAPI, Header, HTTPException, Request, Query
from pydantic import BaseModel, Field
import httpx, feedparser
from openai import OpenAI
from email.utils import parsedate_to_datetime

# ---------- Config ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # e.g. gpt-5, gpt-5-mini, gpt-4o-mini
ADVISOR_TOKEN = os.getenv("ADVISOR_TOKEN", "")
TE_API_KEY = os.getenv("TE_API_KEY", "guest:guest")
ECB_RSS = os.getenv("ECB_RSS", "https://www.ecb.europa.eu/press/rss/press.html")
FED_RSS = os.getenv("FED_RSS", "https://www.federalreserve.gov/feeds/press_all.xml")
EUROSTAT_RSS = os.getenv("EUROSTAT_RSS", "https://ec.europa.eu/eurostat/news/rss")

# Optional enrichers (auto-skip if keys missing)
BING_NEWS_KEY = os.getenv("BING_NEWS_KEY", "")
BING_NEWS_ENDPOINT = os.getenv("BING_NEWS_ENDPOINT", "https://api.bing.microsoft.com/v7.0/news/search")
MARKETAUX_KEY = os.getenv("MARKETAUX_KEY", "")  # https://www.marketaux.com/
MARKETAUX_ENDPOINT = os.getenv("MARKETAUX_ENDPOINT", "https://api.marketaux.com/v1/news/all")

TIMEZONE = os.getenv("OVERSEER_TZ") or os.getenv("OVERSER_TZ") or "Europe/London"

# --- Timeboxing & safety switches ---
REQUEST_DEADLINE_SEC = float(os.getenv("REQUEST_DEADLINE_SEC", "8.0"))  # hard budget for /advice
MIN_BUDGET_WARN_SEC = 2.0   # if below this, skip optional work
SAFE_MODE = os.getenv("SAFE_MODE", "0") in ("1", "true", "True")  # disable enrichers

# --- Tunables for caution window & relaxations ---
SOON_HOURS = float(os.getenv("SOON_HOURS", "3"))                  # “imminent” window for caution
MIN_BLOCK_HOURS = float(os.getenv("MIN_BLOCK_HOURS", "1"))        # hard block inside this window
MIN_CONF_FOR_TRADE = float(os.getenv("MIN_CONF_FOR_TRADE", "65")) # min confidence to flip HOLD

REQUIRE_EVENT_TIMING_MENTION = os.getenv("REQUIRE_EVENT_TIMING_MENTION", "0") in ("1", "true", "True")
SUPPRESS_EVENT_TIMING_IN_REASON = os.getenv("SUPPRESS_EVENT_TIMING_IN_REASON", "1") in ("1", "true", "True")

SHOW_NEXT_EVENT_IN_CONTEXT = os.getenv("SHOW_NEXT_EVENT_IN_CONTEXT", "0") in ("1", "true", "True")
GRACE_MINUTES = float(os.getenv("GRACE_MINUTES", "10"))           # ±minutes treated as non-blocking
ALLOW_TRADE_DURING_EVENT = os.getenv("ALLOW_TRADE_DURING_EVENT", "0") in ("1", "true", "True")

CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "5000"))   # bound prompt context size

client = OpenAI(timeout=8.0)  # OpenAI call timeout per request
app = FastAPI(title="AI Overseer (Direction + Confidence)", version="3.4.0")


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


# ---------- Lifecycle hooks ----------
@app.on_event("startup")
async def on_startup():
    pass

@app.on_event("shutdown")
async def on_shutdown():
    pass


# ---------- System prompt ----------
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

def _deadline_started() -> float:
    return time.monotonic()

def _time_left(start: float) -> float:
    return REQUEST_DEADLINE_SEC - (time.monotonic() - start)


# ---------- External fetchers (timeboxed) ----------
def fetch_te_calendar(countries: List[str], start: float) -> List[Dict[str, Any]]:
    """
    High-impact economic calendar for today + next 3 days.
    """
    if _time_left(start) < MIN_BUDGET_WARN_SEC:
        return []
    d1 = date.today().isoformat()
    d2 = (date.today() + timedelta(days=3)).isoformat()
    country_param = ",".join(countries)
    url = (
        "https://api.tradingeconomics.com/calendar?"
        f"country={country_param}&importance=3&d1={d1}&d2={d2}&format=json&c={TE_API_KEY}"
    )
    try:
        with httpx.Client(timeout=5.0) as http:
            r = http.get(url)
            if r.status_code != 200:
                return []
            data: Union[List[Dict[str, Any]], Dict[str, Any]] = r.json()
            if not isinstance(data, list):
                return []
            out: List[Dict[str, Any]] = []
            for it in data[:50]:
                if not isinstance(it, dict):
                    continue
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

def fetch_rss(url: str, start: float, limit: int = 5, max_age_days: int = 7) -> List[Dict[str, Any]]:
    """
    RSS fetch with recency filter (≤ max_age_days). Skips unknown/naive timestamps.
    """
    if _time_left(start) < MIN_BUDGET_WARN_SEC:
        return []
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

def fetch_eurostat_recent(start: float, limit: int = 5, max_age_days: int = 7) -> List[Dict[str, Any]]:
    return fetch_rss(EUROSTAT_RSS, start, limit=limit, max_age_days=max_age_days)

def fetch_bing_headlines(symbol: str, start: float, count: int = 8, freshness_hours: int = 24) -> List[Dict[str, Any]]:
    if SAFE_MODE or not BING_NEWS_KEY or _time_left(start) < MIN_BUDGET_WARN_SEC:
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
        with httpx.Client(timeout=4.5, headers=headers) as http:
            for q in queries:
                if _time_left(start) < MIN_BUDGET_WARN_SEC:
                    break
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
        dedup: Dict[tuple, Dict[str, Any]] = {}
        for it in sorted(results, key=lambda x: x["published"], reverse=True):
            key = (it["title"], it["url"])
            if key not in dedup:
                dedup[key] = it
        return list(dedup.values())[:count]
    except Exception:
        return []

def fetch_marketaux_sentiment(symbol: str, start: float, limit: int = 6, max_age_hours: int = 48) -> List[Dict[str, Any]]:
    if SAFE_MODE or not MARKETAUX_KEY or _time_left(start) < MIN_BUDGET_WARN_SEC:
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
        with httpx.Client(timeout=4.5) as http:
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
    out: List[Dict[str, Any]] = []
    for it in items:
        t = (it.get("title") or "").lower()
        if any(k in t for k in KEYWORDS_EURUSD):
            out.append(it)
    return out


# ---------- Context builder (timeboxed) ----------
def build_context(snapshot: Dict[str, Any], start: float) -> Dict[str, Any]:
    sym = (snapshot.get("symbol") or snapshot.get("Symbol") or "").upper()
    countries = _countries_for_symbol(sym)

    # Official & baseline (prioritize)
    ecb_items = fetch_rss(ECB_RSS, start, 5, max_age_days=7)
    fed_items = fetch_rss(FED_RSS, start, 5, max_age_days=7)
    eurostat_items = fetch_eurostat_recent(start, 5, max_age_days=7)

    # Calendar
    cal_items_all = fetch_te_calendar(countries, start)
    cal_items = filter_calendar_for_eurusd(cal_items_all)
    hrs_next = next_relevant_event_hours(cal_items)
    soon_threshold = SOON_HOURS

    # Optional enrichers (skip if low budget or SAFE_MODE)
    headlines = fetch_bing_headlines(sym or "EURUSD", start, count=8, freshness_hours=24)
    marketaux_items = fetch_marketaux_sentiment(sym or "EURUSD", start, limit=6, max_age_hours=48)

    now_utc = datetime.now(timezone.utc)
    now_iso = _iso(now_utc)

    def bullets(label: str, items: List[Dict[str, Any]]) -> List[str]:
        lab = str(label).upper()
        if not items:
            if label in ("ecb", "fed", "eurostat"):
                return [f"- {lab}: No recent official posts (≤7d)."]
            elif label in ("headlines", "marketaux"):
                return [f"- {lab}: No recent items (provider disabled, skipped, or none)."]
            else:
                return [f"- {lab}: No high-impact items in the window."]
        out: List[str] = []
        for it in items:
            title = it.get("title", "item")
            pub = it.get("published")
            url = it.get("url", "")
            out.append(f"- {lab}: {title} [{pub}] -> {url}")
        return out

    enabled_bits: List[str] = []
    if BING_NEWS_KEY and not SAFE_MODE: enabled_bits.append("BingNews")
    if MARKETAUX_KEY and not SAFE_MODE: enabled_bits.append("Marketaux")
    enabled_str = ", ".join(enabled_bits) if enabled_bits else "none"

    next_evt_str = "∞" if hrs_next == float("inf") else f"{round(hrs_next,1)}"
    filtered_note = f"(filtered {len(cal_items)} of {len(cal_items_all)} events for EURUSD relevance)"

    header = (
        f"LATEST CONTEXT as of {now_iso} (tz=UTC; local_tz={TIMEZONE}):\n"
        "• RSS recency: ≤7 days; Economic calendar window: today + next 3 days; importance=high\n"
        f"• Countries for calendar: {', '.join(countries)}\n"
    )
    if SHOW_NEXT_EVENT_IN_CONTEXT:
        header += (
            f"• Next relevant high-impact event in ≈ {next_evt_str}h (imminent<{soon_threshold}h) {filtered_note}\n"
        )
    header += f"• Optional enrichers enabled: {enabled_str}\n"

    lines: List[str] = []
    lines += bullets("ecb", ecb_items)
    lines += bullets("fed", fed_items)
    lines += bullets("eurostat", eurostat_items)
    lines += bullets("calendar", cal_items)
    lines += bullets("headlines", headlines)
    lines += bullets("marketaux", marketaux_items)

    ctx_text = header + "\n".join(lines[:40])
    if len(ctx_text) > CONTEXT_MAX_CHARS:
        ctx_text = ctx_text[:CONTEXT_MAX_CHARS] + "\n…(truncated)"

    return {"context_text": ctx_text, "as_of": now_iso, "hrs_next": hrs_next}


# --- OpenAI call with retry/backoff & graceful fallback ---
def _chat_complete_with_retry(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """
    Retries on transient 429/5xx up to 3 times with exponential backoff + jitter.
    Returns a plain dict.
    """
    max_attempts = 3
    base_delay = 0.7
    for attempt in range(1, max_attempts + 1):
        try:
            ch = client.chat.completions.create(model=model, messages=messages, **kwargs)
            if hasattr(ch, "model_dump"):
                return ch.model_dump()
            if hasattr(ch, "to_dict"):
                return ch.to_dict()
            return json.loads(ch.json()) if hasattr(ch, "json") else ch  # type: ignore
        except Exception as e:
            msg = str(e).lower()
            quota = ("insufficient_quota" in msg) or ("you exceeded your current quota" in msg)
            transient = any(k in msg for k in ("rate limit", "429", "timeout", "temporarily unavailable", "gateway", "5xx", "service unavailable"))
            if quota:
                raise
            if attempt == max_attempts or not transient:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(delay)


def model_decision_json(user_input: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_input},
    ]

    # 1) Strict JSON Schema
    try:
        resp = _chat_complete_with_retry(
            MODEL,
            messages,
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        )
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        pass

    # 2) JSON object
    try:
        resp = _chat_complete_with_retry(
            MODEL,
            messages,
            response_format={"type": "json_object"},
        )
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        pass

    # 3) Plain JSON-only instruction
    try:
        resp = _chat_complete_with_retry(
            MODEL,
            [
                {"role": "system", "content": SYSTEM + "\nReturn JSON ONLY with keys: direction, confidence_pct, reason, as_of."},
                {"role": "user", "content": user_input},
            ],
        )
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        now_iso = datetime.now(timezone.utc).isoformat()
        reason = "LLM unavailable"
        emsg = str(e)
        if "insufficient_quota" in emsg or "exceeded your current quota" in emsg:
            reason = "LLM unavailable (insufficient quota)"
        elif "rate limit" in emsg.lower():
            reason = "LLM unavailable (rate limited)"
        return {
            "direction": "HOLD",
            "confidence_pct": 0.0,
            "reason": reason,
            "as_of": now_iso,
        }


# --- Micro-bias & strength inference ---
def _infer_bias_from_snapshot(snap: Dict[str, Any]) -> Optional[str]:
    s = {k.lower(): v for k, v in snap.items()}

    ef = s.get("ema_fast") or s.get("ema20") or s.get("ema_20") or s.get("ema_short") or s.get("ema10") or s.get("ema_10")
    es = s.get("ema_slow") or s.get("ema50") or s.get("ema_50") or s.get("ema_long") or s.get("ema200") or s.get("ema_200")
    if isinstance(ef, (int, float)) and isinstance(es, (int, float)):
        if ef > es: return "BUY"
        if ef < es: return "SELL"

    rsi = s.get("rsi") or s.get("rsi14") or s.get("rsi_14")
    if isinstance(rsi, (int, float)):
        if rsi >= 55: return "BUY"
        if rsi <= 45: return "SELL"

    return None

def _strong_alignment(snap: Dict[str, Any]) -> bool:
    s = {k.lower(): v for k, v in snap.items()}
    bias = _infer_bias_from_snapshot(snap)
    rsi = s.get("rsi") or s.get("rsi14") or s.get("rsi_14")
    adx = s.get("adx") or s.get("adx14") or s.get("adx_14")
    score = 0
    if bias in ("BUY", "SELL"): score += 1
    if isinstance(rsi, (int, float)) and (rsi >= 58 or rsi <= 42): score += 1
    if isinstance(adx, (int, float)) and adx >= 18: score += 1
    return score >= 2


# --- Reason sanitizer ---
_REASON_TIME_RE = re.compile(r"(?i)\b(within|in)\s+(the\s+)?next?\s+\d+(\.\d+)?\s*hours?\b[^.]*\.?")
def _sanitize_reason(text: str) -> str:
    if not text:
        return text
    if SUPPRESS_EVENT_TIMING_IN_REASON:
        text = _REASON_TIME_RE.sub("", text).strip()
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+,", ",", text)
    return text


# ---------- Core decision (timeboxed) ----------
def get_decision(snapshot: Dict[str, Any], start: float) -> SimpleAdviceOut:
    # If we are already out of time, return HOLD immediately
    if _time_left(start) <= 0:
        return SimpleAdviceOut(direction="HOLD", confidence_pct=0.0, reason="Timeout before context build")

    ctx = build_context(snapshot, start)

    # If time remaining is tiny, short-circuit LLM and return a heuristic HOLD
    if _time_left(start) < MIN_BUDGET_WARN_SEC:
        return SimpleAdviceOut(
            direction="HOLD",
            confidence_pct=0.0,
            reason="Time budget too low; returned safe HOLD",
            as_of=ctx["as_of"],
        )

    user_input = json.dumps({
        "as_of": ctx["as_of"],
        "snapshot": snapshot,
        "context": ctx["context_text"]
    }, separators=(",", ":"))

    # If we are still OK on time, call the LLM (which itself has retry & small timeout)
    data = model_decision_json(user_input)

    # --- Post-LLM relaxation shim ---
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
                data["reason"] = (reason + (' ' if reason else '') + note).strip()

        data["reason"] = _sanitize_reason(data.get("reason") or "")

    except Exception:
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
        "bing_news_enabled": bool(BING_NEWS_KEY) and not SAFE_MODE,
        "marketaux_enabled": bool(MARKETAUX_KEY) and not SAFE_MODE,
        "soon_hours": SOON_HOURS,
        "min_block_hours": MIN_BLOCK_HOURS,
        "min_conf_for_trade": MIN_CONF_FOR_TRADE,
        "require_event_timing_mention": REQUIRE_EVENT_TIMING_MENTION,
        "suppress_event_timing_in_reason": SUPPRESS_EVENT_TIMING_IN_REASON,
        "show_next_event_in_context": SHOW_NEXT_EVENT_IN_CONTEXT,
        "grace_minutes": GRACE_MINUTES,
        "allow_trade_during_event": ALLOW_TRADE_DURING_EVENT,
        "context_max_chars": CONTEXT_MAX_CHARS,
        "request_deadline_sec": REQUEST_DEADLINE_SEC,
        "safe_mode": SAFE_MODE,
        "version": "3.4.0",
    }

@app.get("/health/llm")
def health_llm(probe: int = Query(0, description="Set to 1 to run a tiny LLM probe")):
    status = "idle"
    detail = ""
    if probe == 1:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Return JSON with keys ok, as_of. Nothing else."},
                    {"role": "user", "content": "Respond with {\"ok\": true, \"as_of\": ISO8601 now} only."},
                ],
                response_format={"type": "json_object"},
            )
            _ = json.loads(resp.choices[0].message.content)
            status = "ok"
        except Exception as e:
            msg = str(e).lower()
            if "insufficient_quota" in msg or "exceeded your current quota" in msg:
                status = "quota_exhausted"
            elif "rate limit" in msg or "429" in msg:
                status = "rate_limited"
            else:
                status = "error"
            detail = str(e)
    return {"llm_status": status, "detail": detail, "model": MODEL}

# Primary endpoint (new minimal schema)
@app.post("/advice")
async def advice(request: Request, x_advisor_token: Optional[str] = Header(None)):
    _auth_or_403(x_advisor_token)
    start = _deadline_started()
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    snapshot = payload.get("snapshot") if isinstance(payload, dict) and "snapshot" in payload else payload
    if not isinstance(snapshot, dict):
        raise HTTPException(400, "Body must be a JSON object or {'snapshot': {...}}")

    # Hard stop if budget already blown
    if _time_left(start) <= 0:
        return SimpleAdviceOut(direction="HOLD", confidence_pct=0.0, reason="Timeout at request entry").model_dump()

    try:
        out = get_decision(snapshot, start)
        return out.model_dump()
    except Exception as e:
        print("ADVICE ERROR:", repr(e))
        traceback.print_exc()
        # Return HOLD instead of 500 to avoid worker timeouts/kill loops
        now_iso = datetime.now(timezone.utc).isoformat()
        return SimpleAdviceOut(direction="HOLD", confidence_pct=0.0, reason=f"advisor_error:{e.__class__.__name__}", as_of=now_iso).model_dump()

# Compatibility endpoint for older bots (maps to long/short/skip + 0..1 confidence)
@app.post("/gbpusd-advice")
async def gbpusd_advice(request: Request, x_advisor_token: Optional[str] = Header(None)):
    _auth_or_403(x_advisor_token)
    start = _deadline_started()
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    snapshot = payload.get("snapshot") if isinstance(payload, dict) and "snapshot" in payload else payload
    if not isinstance(snapshot, dict):
        raise HTTPException(400, "Body must be a JSON object or {'snapshot': {...}}")

    if _time_left(start) <= 0:
        return {"action": "skip", "confidence": 0.0, "sl_pips": 0, "tp_pips": 0, "reason": "Timeout at request entry"}

    try:
        out = get_decision(snapshot, start)
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
