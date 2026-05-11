"""Download quarterly + TTM financials from Polygon for all S&P 500 tickers.

Output: us_pipeline/data/fundamentals/<TICKER>.json — one file per ticker, raw JSON.

We store raw because:
  - Polygon returns rich nested structure with units, source XBRL tag, derivation flag
  - Re-deriving factors later may need fields we didn't initially extract
  - Disk is cheap; ~1MB per ticker × 500 = 500MB
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
INSTR_FILE = ROOT / "data" / "instruments" / "sp500.txt"
OUT_DIR = ROOT / "data" / "fundamentals"


def fetch(ticker: str, api_key: str, retries: int = 3) -> tuple[str, int, str]:
    out_path = OUT_DIR / f"{ticker}.json"
    if out_path.exists():
        try:
            n = len(json.loads(out_path.read_text()).get("results", []))
            return ticker, n, "cached"
        except Exception:
            pass

    # Pull all available reports (default ordering: most recent first)
    url = (
        f"https://api.polygon.io/vX/reference/financials"
        f"?ticker={ticker}&limit=100&apiKey={api_key}"
    )
    last_err = None
    for attempt in range(retries):
        try:
            r = urllib.request.urlopen(url, timeout=20)
            d = json.loads(r.read())
            results = d.get("results", [])
            out_path.write_text(json.dumps({"ticker": ticker, "results": results}))
            return ticker, len(results), "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                out_path.write_text(json.dumps({"ticker": ticker, "results": []}))
                return ticker, 0, "not-found"
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)
    return ticker, 0, f"error: {last_err}"


def main() -> None:
    load_dotenv(ROOT.parent / ".env")
    api_key = os.environ["POLYGON_API_KEY"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = [t.strip() for t in INSTR_FILE.read_text().splitlines() if t.strip()]
    print(f"Fetching financials for {len(tickers)} tickers -> {OUT_DIR}")

    stats = {"ok": 0, "cached": 0, "not-found": 0, "error": 0}
    total_reports = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch, t, api_key): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, n, status = fut.result()
            key = "error" if status.startswith("error") else status
            stats[key] = stats.get(key, 0) + 1
            total_reports += n
            if status.startswith("error") or status == "not-found":
                print(f"  [{i:>4}/{len(tickers)}] {ticker:<6} {status}")
            elif i % 50 == 0 or i == len(tickers):
                print(f"  [{i:>4}/{len(tickers)}] reports so far: {total_reports} | "
                      f"elapsed: {time.time()-t0:.1f}s")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"  status counts: {stats}")
    print(f"  total reports: {total_reports}")
    print(f"  avg reports/ticker: {total_reports / max(1, stats['ok'] + stats['cached']):.1f}")


if __name__ == "__main__":
    main()
