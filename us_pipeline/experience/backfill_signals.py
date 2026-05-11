"""Generate signals for a range of historical dates that don't already have signal CSVs.

This drives the bulk of the experience-library backfill. Each date trains a fresh
LambdaRank model on history up to (date - 31 days), validates on the prior 30 days,
predicts for that single date.

Usage:
  python us_pipeline/experience/backfill_signals.py 2025-10-01 2026-04-07
  python us_pipeline/experience/backfill_signals.py --weeks 26  # last 26 weeks back from latest data
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import qlib

ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = ROOT / "signals"


def trading_days_in_range(start: date, end: date) -> list[date]:
    """Use Qlib's calendar so we never request a non-trading day."""
    from qlib.data import D
    cal = D.calendar(start_time=start.isoformat(), end_time=end.isoformat(), freq="day")
    return [pd.Timestamp(d).date() for d in cal]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", nargs="?", help="YYYY-MM-DD")
    p.add_argument("end", nargs="?", help="YYYY-MM-DD")
    p.add_argument("--weeks", type=int, default=None, help="last N weeks back from latest data")
    p.add_argument("--max", type=int, default=None, help="cap the number of dates processed")
    args = p.parse_args()

    qlib.init(provider_uri=str(ROOT / "data" / "qlib_bin"), region="us")

    if args.weeks:
        from qlib.data import D
        latest = pd.Timestamp(D.calendar(freq="day")[-1]).date()
        end_d = latest
        start_d = latest - timedelta(weeks=args.weeks)
    else:
        start_d = date.fromisoformat(args.start)
        end_d = date.fromisoformat(args.end)

    dates = trading_days_in_range(start_d, end_d)
    if args.max:
        dates = dates[: args.max]
    print(f"Backfilling signals for {len(dates)} trading days from {start_d} to {end_d}")

    skipped = 0
    done = 0
    failed = 0
    t0 = time.time()
    for i, d in enumerate(dates, 1):
        target = d.isoformat()
        out_csv = SIGNALS_DIR / f"{target}.csv"
        if out_csv.exists():
            skipped += 1
            continue
        # Subprocess call is intentional — fresh Python process per training avoids
        # MLflow recorder leakage and memory creep.
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "signals" / "generate_signals.py"), target],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            done += 1
            if done % 5 == 0 or i == len(dates):
                elapsed = time.time() - t0
                eta = (elapsed / done) * (len(dates) - skipped - done - failed)
                print(f"  [{i:>3}/{len(dates)}] {target} ok | done={done} skipped={skipped} "
                      f"failed={failed} | elapsed={elapsed:.0f}s eta={eta:.0f}s")
        except subprocess.CalledProcessError:
            failed += 1
            print(f"  [{i:>3}/{len(dates)}] {target} FAILED")

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  generated: {done}")
    print(f"  skipped (already existed): {skipped}")
    print(f"  failed: {failed}")


if __name__ == "__main__":
    main()
