"""Download Benzinga news articles from Polygon for the full S&P 500 universe.

Strategy:
  - Query the news endpoint *market-wide* (no ticker filter) per day,
    paginated via next_url. ~150-300 articles/day.
  - Save raw articles per day in us_pipeline/data/news/<YYYY-MM-DD>.json
  - Resumable: skip dates whose JSON already exists.

Why market-wide and not per-ticker:
  Most articles tag 2-5 tickers. A market-wide query gives us all the news
  for all 503 stocks in one pagination loop, ~10x more efficient than 503
  per-ticker requests.

Output schema (one JSON per day):
  {
    "date": "2026-04-22",
    "fetched_utc": "2026-04-26T...Z",
    "articles": [ <raw Polygon article object>, ... ]
  }
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
NEWS_DIR = ROOT / "data" / "news"
DEFAULT_START = "2021-06-01"   # match price data start


def fetch_day(api_key: str, day: date, max_pages: int = 20) -> list[dict]:
    """Fetch all news articles for a single trading session day.

    A "trading session" for day D = articles published from D-1 16:00 ET
    through D 16:00 ET. This way features built from this set are PIT-safe
    for predictions made at D's close.
    """
    # ET = UTC-4 (DST) or UTC-5. We approximate with UTC-4. Off-by-1-hour
    # at DST edges is acceptable since news is sticky over a few minutes.
    # session start = D-1 16:00 ET = D-1 20:00 UTC
    # session end   = D   16:00 ET = D   20:00 UTC
    start_utc = (datetime.combine(day - timedelta(days=1),
                                  datetime.min.time()).replace(hour=20)).isoformat() + "Z"
    end_utc = (datetime.combine(day,
                                datetime.min.time()).replace(hour=20)).isoformat() + "Z"

    base = (f"https://api.polygon.io/v2/reference/news"
            f"?published_utc.gte={start_utc}"
            f"&published_utc.lte={end_utc}"
            f"&order=desc&sort=published_utc"
            f"&limit=1000")
    articles: list[dict] = []
    url = base + f"&apiKey={api_key}"
    pages = 0
    while url and pages < max_pages:
        for attempt in range(4):
            try:
                r = urllib.request.urlopen(url, timeout=30)
                d = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Failed to fetch {url}")

        results = d.get("results", []) or []
        articles.extend(results)
        next_url = d.get("next_url")
        if not next_url:
            break
        # next_url already encodes the cursor; need to add apiKey
        url = next_url + ("&" if "?" in next_url else "?") + f"apiKey={api_key}"
        pages += 1
    return articles


def trading_dates_in_range(start: date, end: date) -> list[date]:
    """Return weekdays (Mon-Fri) — close enough to trading days for our purpose.
    Overlap with non-trading days like holidays is harmless: just an empty fetch.
    """
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # 0..4 = Mon..Fri
            out.append(d)
        d += timedelta(days=1)
    return out


def main():
    load_dotenv(ROOT.parent / ".env")
    api_key = os.environ["POLYGON_API_KEY"]
    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.fromisoformat(DEFAULT_START)
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today()
    dates = trading_dates_in_range(start, end)
    print(f"Will fetch news for {len(dates)} days from {start} to {end}")
    print(f"Output dir: {NEWS_DIR}")

    stats = {"new": 0, "cached": 0, "empty": 0, "error": 0}
    total_articles = 0
    t0 = time.time()
    for i, d in enumerate(dates, 1):
        out_path = NEWS_DIR / f"{d.isoformat()}.json"
        if out_path.exists():
            stats["cached"] += 1
            try:
                total_articles += len(json.loads(out_path.read_text()).get("articles", []))
            except Exception:
                pass
            continue
        try:
            articles = fetch_day(api_key, d)
            payload = {
                "date": d.isoformat(),
                "fetched_utc": datetime.utcnow().isoformat() + "Z",
                "articles": articles,
            }
            out_path.write_text(json.dumps(payload))
            if articles:
                stats["new"] += 1
            else:
                stats["empty"] += 1
            total_articles += len(articles)
            if i % 25 == 0 or i == len(dates):
                print(f"  [{i:>4}/{len(dates)}] {d}: {len(articles):>4} articles "
                      f"| total={total_articles} | elapsed={time.time()-t0:.0f}s")
        except Exception as e:
            stats["error"] += 1
            print(f"  [{i:>4}/{len(dates)}] {d}: ERROR {e}")

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  status: {stats}")
    print(f"  total articles cached on disk: {total_articles}")


if __name__ == "__main__":
    main()
