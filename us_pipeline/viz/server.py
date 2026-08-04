"""FastAPI backend for the stock chart app.

Endpoints:
  GET /api/search?q=appl            -> symbol suggestions
  GET /api/bars?ticker=AAPL&range=1D -> bars + header stats for one chart

Bars come from Polygon on demand and are cached to
us_pipeline/data/intraday_cache/. Windows that ended in the past never change,
so they are cached forever; any window touching today expires after 5 minutes
(the tier serves 15-minute-delayed data anyway).

Run:  .venv/bin/uvicorn server:app --reload --port 8001   (from us_pipeline/viz)
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from polygon.exceptions import BadResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
from fetch_sectors import sic_to_display_sector  # noqa: E402
from polygon_client import TIER_EARLIEST, TickerNotFound, fetch_aggs, get_client  # noqa: E402

ET = ZoneInfo("America/New_York")
CACHE_DIR = Path(__file__).parent.parent / "data" / "intraday_cache"
SECTORS_FILE = Path(__file__).parent.parent / "data" / "instruments" / "sectors.csv"
WIDE_FILE = Path(__file__).parent.parent / "data" / "instruments" / "sectors_all.csv"
SP500_FILE = Path(__file__).parent.parent / "data" / "instruments" / "sp500.txt"
DIST = Path(__file__).parent / "frontend" / "dist"

LIVE_TTL = 300      # seconds, for windows that include today
DETAILS_TTL = 86400  # ticker name/exchange changes rarely

# Regular US session, ET.
OPEN_MIN = 9 * 60 + 30
CLOSE_MIN = 16 * 60

# range -> how to fetch and trim it.
#   mult/span: Polygon aggregate size      days: calendar days to request
#   sessions:  keep only the last N trading sessions (None = keep all)
#   extended:  include pre/post-market bars
RANGES = {
    "1D": dict(mult=1, span="minute", days=7, sessions=1, extended=True),
    "1W": dict(mult=5, span="minute", days=12, sessions=5, extended=False),
    "1M": dict(mult=30, span="minute", days=32, sessions=None, extended=False),
    "3M": dict(mult=1, span="day", days=93, sessions=None, extended=False),
    "1Y": dict(mult=1, span="day", days=366, sessions=None, extended=False),
    "MAX": dict(mult=1, span="day", days=5 * 366, sessions=None, extended=False),
}

# Sector leaderboard lookbacks, in calendar days back from the latest session.
MOVER_RANGES = {"1D": 1, "1W": 7, "1M": 30, "3M": 91, "1Y": 365, "MAX": 5 * 365}

# Without a floor, daily % leaderboards over the wide universe are entirely
# sub-dollar stocks moving on a few thousand shares.
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 10_000_000

app = FastAPI(title="Stock chart API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
client = get_client()


# --------------------------------------------------------------------------- cache

def _cached(key: str, ttl: int | None, produce):
    """Return produce() memoized to CACHE_DIR/key. ttl=None means never expire."""
    f = CACHE_DIR / key
    if f.exists():
        try:
            payload = json.loads(f.read_text())
            if payload["expires"] is None or time.time() < payload["expires"]:
                return payload["data"]
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt entry, refetch
    data = produce()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"expires": None if ttl is None else time.time() + ttl, "data": data}))
    return data


# --------------------------------------------------------------------------- helpers

def today_et() -> datetime:
    return datetime.now(ET)


def et_parts(ts_ms: int) -> tuple[str, int]:
    """(ET date as YYYY-MM-DD, minutes since ET midnight) for an epoch-ms bar."""
    dt = datetime.fromtimestamp(ts_ms / 1000, ET)
    return dt.strftime("%Y-%m-%d"), dt.hour * 60 + dt.minute


def pick_agg(span_days: int) -> tuple[int, str]:
    """Aggregation for a custom range of the given length."""
    if span_days <= 1:
        return 1, "minute"
    if span_days <= 7:
        return 5, "minute"
    if span_days <= 35:
        return 30, "minute"
    return 1, "day"


def get_bars(ticker: str, mult: int, span: str, start: str, end: str) -> list[dict]:
    """Raw bars from Polygon, cached. Always the full extended-hours set."""
    ttl = LIVE_TTL if end >= today_et().strftime("%Y-%m-%d") else None

    def produce():
        aggs = fetch_aggs(client, ticker, mult, span, start, end)
        return [
            {"t": a.timestamp, "o": a.open, "h": a.high, "l": a.low, "c": a.close, "v": a.volume}
            for a in aggs
        ]

    return _cached(f"{ticker.upper()}/{span}_{mult}_{start}_{end}.json", ttl, produce)


def get_details(ticker: str) -> dict:
    def produce():
        try:
            r = client.get_ticker_details(ticker)
            return {"name": r.name, "exchange": r.primary_exchange}
        except Exception:
            return {"name": ticker.upper(), "exchange": None}

    return _cached(f"{ticker.upper()}/details.json", DETAILS_TTL, produce)


class DataTooOld(Exception):
    """Date falls outside the grouped feed's entitlement window."""


def load_universe() -> dict[str, dict]:
    """Ticker -> sector metadata for every symbol the app can rank.

    Prefers sectors_all.csv (every active US stock, ADR and ETF, built by
    `fetch_sectors.py --universe all`). Falls back to the S&P-only sectors.csv
    for anything missing, so the app still works before that backfill is run.
    """
    out = {}
    if WIDE_FILE.exists():
        with WIDE_FILE.open() as f:
            for r in csv.DictReader(f):
                out[r["symbol"]] = {
                    "name": r.get("name") or r["symbol"],
                    "type": r.get("type") or "CS",
                    "sector": r["sector"] or "Unknown",
                    "industry": (r.get("sic_description") or "").title(),
                    "took_ticker_on": r.get("took_ticker_on") or None,
                }
    with SECTORS_FILE.open() as f:
        for r in csv.DictReader(f):
            if r["symbol"] not in out:
                sic = int(r["sic_code"]) if r["sic_code"] else None
                out[r["symbol"]] = {
                    "name": r["symbol"],
                    "type": "CS",
                    "sector": sic_to_display_sector(sic, r["symbol"]),
                    "industry": (r["sic_description"] or "").title(),
                    "took_ticker_on": None,
                }
    return out


UNIVERSE = load_universe()
SP500 = {t.strip() for t in SP500_FILE.read_text().splitlines() if t.strip()}


def grouped(date: str) -> dict[str, dict]:
    """Every US ticker's daily bar for one date, in a single API call (~12k rows).

    Returns {} on weekends and holidays. Raises DataTooOld outside the window —
    the grouped feed's floor is later than per-ticker aggregates' (verified: OK at
    2021-11-01, NOT_AUTHORIZED at 2021-08-02).
    """
    def produce():
        try:
            rows = client.get_grouped_daily_aggs(date, adjusted=True)
        except BadResponse as e:
            if "NOT_AUTHORIZED" in str(e):
                return {"_too_old": True}
            raise
        return {a.ticker: {"c": a.close, "v": a.volume} for a in rows}

    ttl = LIVE_TTL if date >= today_et().strftime("%Y-%m-%d") else None
    data = _cached(f"_grouped/{date}.json", ttl, produce)
    if data.get("_too_old"):
        raise DataTooOld(date)
    return data


def grouped_near(target: str, step: int) -> tuple[str, dict]:
    """Nearest usable trading session to `target`, walking `step` days at a time.

    Weekends and holidays come back empty, so we keep stepping. A date below the
    entitlement floor jumps forward a month and switches to searching forward,
    which is what clamps the MAX leaderboard to the earliest date actually served.
    """
    d = datetime.strptime(target, "%Y-%m-%d")
    for _ in range(60):
        try:
            bars = grouped(d.strftime("%Y-%m-%d"))
            if bars:
                return d.strftime("%Y-%m-%d"), bars
            d += timedelta(days=step)
        except DataTooOld:
            d += timedelta(days=30)
            step = 1
    raise HTTPException(502, f"no usable trading session near {target}")


def get_quote(ticker: str) -> dict | None:
    """Official close, prior close, and any after-hours print for the latest session.

    Deliberately independent of the chart's range: the last bar of a range means
    something different in each aggregation (the 19:59 post-market print on 1D,
    the 15:59 continuous trade on 1W, the 16:00 auction on daily), so tying the
    header to it would show three prices for one stock. This always reports the
    official close, and surfaces the after-hours move as its own figure.
    """
    now = today_et()
    daily = get_bars(ticker, 1, "day", (now - timedelta(days=10)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
    if not daily:
        return None

    session = et_parts(daily[-1]["t"])[0]
    close = daily[-1]["c"]
    prev = daily[-2]["c"] if len(daily) > 1 else close

    after = None
    post = [b for b in get_bars(ticker, 1, "minute", session, session) if et_parts(b["t"])[1] >= CLOSE_MIN]
    if post:
        after = {
            "price": post[-1]["c"],
            "t": post[-1]["t"],
            "change": post[-1]["c"] - close,
            "change_pct": (post[-1]["c"] - close) / close * 100 if close else 0.0,
        }

    return {
        "session": session,
        "close": close,
        "prev_close": prev,
        "change": close - prev,
        "change_pct": (close - prev) / prev * 100 if prev else 0.0,
        "after": after,
    }


# --------------------------------------------------------------------------- API

@app.get("/api/search")
def search(q: str = Query(min_length=1), limit: int = 8):
    results = client.list_tickers(search=q, market="stocks", active=True, limit=limit)
    return [
        {"ticker": r.ticker, "name": r.name, "exchange": r.primary_exchange}
        for r in itertools.islice(results, limit)
    ]


@app.get("/api/bars")
def bars(
    ticker: str,
    range_: str = Query("1D", alias="range"),
    start: str | None = Query(None, alias="from"),
    end: str | None = Query(None, alias="to"),
):
    ticker = ticker.upper()
    now = today_et()

    if start and end:
        # Custom range: aggregation follows the span, regular hours only.
        span_days = max((datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days, 1)
        mult, span = pick_agg(span_days)
        spec = dict(mult=mult, span=span, sessions=None, extended=span_days <= 1)
        label = "custom"
    else:
        if range_ not in RANGES:
            raise HTTPException(400, f"unknown range {range_!r}; use {list(RANGES)} or from/to")
        spec = RANGES[range_]
        label = range_
        end = now.strftime("%Y-%m-%d")
        start = max((now - timedelta(days=spec["days"])).strftime("%Y-%m-%d"), TIER_EARLIEST)

    try:
        raw = get_bars(ticker, spec["mult"], spec["span"], start, end)
    except TickerNotFound:
        raise HTTPException(404, f"no data for {ticker}")
    except Exception as e:
        raise HTTPException(502, f"polygon error: {e}")

    intraday = spec["span"] == "minute"
    if intraday and not spec["extended"]:
        raw = [b for b in raw if OPEN_MIN <= et_parts(b["t"])[1] < CLOSE_MIN]
    if spec["sessions"]:
        keep = sorted({et_parts(b["t"])[0] for b in raw})[-spec["sessions"]:]
        raw = [b for b in raw if et_parts(b["t"])[0] in keep]

    if not raw:
        raise HTTPException(404, f"no bars for {ticker} in {start}..{end}")

    quote = get_quote(ticker)

    # 1D compares against the prior session's close; longer ranges against their
    # own first bar, so the % change is "change over this period".
    session = et_parts(raw[-1]["t"])[0]
    baseline = quote["prev_close"] if (label == "1D" and quote) else raw[0]["c"]

    last = raw[-1]["c"]
    regular = None
    if spec["extended"]:
        day = datetime.strptime(session, "%Y-%m-%d")
        regular = {
            "start": int(day.replace(hour=9, minute=30, tzinfo=ET).timestamp() * 1000),
            "end": int(day.replace(hour=16, minute=0, tzinfo=ET).timestamp() * 1000),
        }

    return {
        "ticker": ticker,
        "name": get_details(ticker)["name"],
        "range": label,
        "timespan": spec["span"],
        "multiplier": spec["mult"],
        "extended": spec["extended"],
        "session": session,
        "regular": regular,
        "baseline": baseline,
        "quote": quote,
        "last": last,  # last point of THIS series; the header uses quote.close instead
        "bars": raw,
    }


@app.get("/api/movers")
def movers(
    range_: str = Query("1D", alias="range"),
    universe: str = Query("sp500"),
    liquid: bool = Query(True),
):
    """Return leaderboard over `range`, labelled with GICS-like sectors.

    Two grouped-daily calls cover every US ticker, so neither the size of the
    universe nor the number of sectors costs extra requests — switching sector
    tabs and re-sorting are free on the client.
    """
    if range_ not in MOVER_RANGES:
        raise HTTPException(400, f"unknown range {range_!r}; use {list(MOVER_RANGES)}")
    if universe not in ("sp500", "all"):
        raise HTTPException(400, f"unknown universe {universe!r}; use 'sp500' or 'all'")

    end_date, end_bars = grouped_near(today_et().strftime("%Y-%m-%d"), -1)
    wanted = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=MOVER_RANGES[range_])).strftime("%Y-%m-%d")
    start_date, start_bars = grouped_near(wanted, -1)

    symbols = SP500 if universe == "sp500" else UNIVERSE.keys()
    rows, missing, renamed, illiquid = [], 0, 0, 0
    for sym in symbols:
        meta = UNIVERSE.get(sym)
        a, b = start_bars.get(sym), end_bars.get(sym)
        if not meta or not a or not b or not a["c"]:
            missing += 1  # unclassified, listed after the window opened, or halted
            continue
        # Symbols that took their ticker over from a different company mid-window
        # would compare two unrelated businesses. See fetch_sectors.took_ticker_on.
        if meta["took_ticker_on"] and meta["took_ticker_on"] > start_date:
            renamed += 1
            continue
        if liquid and (b["c"] < MIN_PRICE or b["c"] * b["v"] < MIN_DOLLAR_VOLUME):
            illiquid += 1
            continue
        rows.append({
            "ticker": sym,
            "name": meta["name"],
            "sector": meta["sector"],
            "industry": meta["industry"],
            "price": b["c"],
            "change_pct": (b["c"] / a["c"] - 1) * 100,
            "volume": b["v"],
        })
    rows.sort(key=lambda r: r["change_pct"], reverse=True)

    return {
        "range": range_,
        "universe": universe,
        "liquid": liquid,
        "from": start_date,
        "to": end_date,
        "clamped": start_date > wanted,  # window cut short by the entitlement floor
        "missing": missing,
        "renamed": renamed,
        "illiquid": illiquid,
        "rows": rows,
    }


# Serve the built frontend when it exists; mounted last so /api wins.
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="static")
