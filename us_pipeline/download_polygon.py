"""Download daily OHLCV bars from Polygon for a list of tickers.

Output: one CSV per ticker under us_pipeline/data/raw/<TICKER>.csv
Columns: date, open, high, low, close, volume, vwap, transactions
Prices are *split-adjusted* (Polygon's `adjusted=true`, the default).

Behavior:
- Resumable: if a CSV already exists, only fetches bars after the last stored date.
- Concurrent: uses a thread pool (Polygon Starter has unlimited rate, so we just
  pick a sane fan-out to avoid IO contention).
- Robust: retries with exponential backoff on transient failures.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from polygon import RESTClient

from polygon_client import TickerNotFound, fetch_aggs, get_client

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
INSTR_DIR = ROOT / "data" / "instruments"

# Polygon Stocks Starter tier: ~5-year rolling window of daily bars.
# Today (2026-04) earliest available is ~2021-06-01.
DEFAULT_START = "2021-06-01"


def load_universe(name: str) -> list[str]:
    f = INSTR_DIR / f"{name}.txt"
    return [line.strip() for line in f.read_text().splitlines() if line.strip()]


def existing_last_date(csv_path: Path) -> date | None:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["date"])
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max().date()
    except Exception:
        return None


def fetch_one(
    client: RESTClient,
    ticker: str,
    start: str,
    end: str,
    out_dir: Path,
    retries: int = 4,
) -> tuple[str, int, str]:
    """Fetch bars for one ticker. Returns (ticker, n_new_rows, status)."""
    out_path = out_dir / f"{ticker}.csv"
    last = existing_last_date(out_path)
    fetch_start = (
        (last + timedelta(days=1)).isoformat() if last else start
    )
    if fetch_start > end:
        return ticker, 0, "up-to-date"

    try:
        bars = fetch_aggs(client, ticker, 1, "day", fetch_start, end, retries=retries)
    except TickerNotFound:
        return ticker, 0, "not-found"
    except Exception as e:
        return ticker, 0, f"error: {e}"

    if not bars:
        return ticker, 0, "no-new-bars"

    df = pd.DataFrame(
        [
            {
                "date": datetime.utcfromtimestamp(b.timestamp / 1000).date().isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "vwap": b.vwap,
                "transactions": b.transactions,
            }
            for b in bars
        ]
    )

    if last is not None and out_path.exists():
        # Append new rows.
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)

    return ticker, len(df), "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="sp500", help="instrument list name (default: sp500)")
    parser.add_argument("--start", default=DEFAULT_START, help=f"start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end", default=date.today().isoformat(), help="end date YYYY-MM-DD (default: today)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent download workers (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="only download first N tickers (debug)")
    args = parser.parse_args()

    client = get_client()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_universe(args.universe)
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"Downloading {len(tickers)} tickers from {args.start} to {args.end} -> {RAW_DIR}")

    stats = {"ok": 0, "up-to-date": 0, "no-new-bars": 0, "not-found": 0, "error": 0}
    total_rows = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(fetch_one, client, t, args.start, args.end, RAW_DIR): t
            for t in tickers
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, n, status = fut.result()
            key = "error" if status.startswith("error") else status
            stats[key] = stats.get(key, 0) + 1
            total_rows += n
            if status.startswith("error") or status == "not-found":
                print(f"  [{i:>4}/{len(tickers)}] {ticker:<6} {status}")
            elif i % 25 == 0 or i == len(tickers):
                print(
                    f"  [{i:>4}/{len(tickers)}] processed | new rows so far: {total_rows} | "
                    f"elapsed: {time.time()-t0:.1f}s"
                )

    print("\nDone.")
    print(f"  status counts: {stats}")
    print(f"  total new rows: {total_rows}")
    print(f"  elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
