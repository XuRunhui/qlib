"""SPY trend filter — risk gate for the model.

Rule: only trade when SPY's close is above its 200-day simple moving average.
This is a basic regime filter that historically prevents trading through bear
markets where ML factor strategies tend to fail.

Output: GO / NO-GO signal + relevant numbers.

Usage:
  python us_pipeline/signals/spy_filter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import qlib
from qlib.data import D

PROVIDER = "us_pipeline/data/qlib_bin"


def main():
    qlib.init(provider_uri=PROVIDER, region="us")
    df = D.features(["SPY"],
                    ["$close",
                     "Mean($close, 200)",
                     "Mean($close, 50)",
                     "Mean($close, 20)",
                     "Std($close/Ref($close,1) - 1, 20) * 16"],
                    start_time="2024-01-01", end_time=None, freq="day")
    df.columns = ["close", "ma200", "ma50", "ma20", "ann_vol_20d"]
    df = df.dropna()
    if df.empty:
        print("ERROR: SPY data not available. Update data first.")
        sys.exit(1)

    # Most recent row
    last = df.iloc[-1]
    last_date = df.index.get_level_values("datetime")[-1]
    close = last["close"]
    ma200 = last["ma200"]
    ma50 = last["ma50"]
    ma20 = last["ma20"]
    vol = last["ann_vol_20d"]

    above_200 = close > ma200
    above_50 = close > ma50
    above_20 = close > ma20

    print(f"\n=== SPY Trend Filter — as of {last_date.date()} ===\n")
    print(f"  SPY close:      ${close:.2f}")
    print(f"  20-day MA:      ${ma20:.2f}    {'ABOVE ✓' if above_20 else 'BELOW ✗'}  "
          f"({(close/ma20-1)*100:+.1f}%)")
    print(f"  50-day MA:      ${ma50:.2f}    {'ABOVE ✓' if above_50 else 'BELOW ✗'}  "
          f"({(close/ma50-1)*100:+.1f}%)")
    print(f"  200-day MA:     ${ma200:.2f}   {'ABOVE ✓' if above_200 else 'BELOW ✗'}  "
          f"({(close/ma200-1)*100:+.1f}%)")
    print(f"  Realized vol:   {vol*100:.0f}% annualized (20d)")
    print()

    # Decision logic
    if above_200 and vol < 0.30:
        decision = "GO"
        reason = "SPY > 200MA and vol moderate"
    elif above_200 and vol >= 0.30:
        decision = "REDUCED"
        reason = f"SPY > 200MA but vol elevated ({vol*100:.0f}%) — trade at half size"
    elif not above_200 and above_50:
        decision = "REDUCED"
        reason = "SPY below 200MA but above 50MA — possible correction; trade at half size"
    else:
        decision = "NO-GO"
        reason = "SPY below both 50 and 200 MAs — likely bear market; pause trading"

    color = {"GO": "🟢", "REDUCED": "🟡", "NO-GO": "🔴"}[decision]
    print(f"  {color} DECISION: {decision}")
    print(f"     Reason: {reason}")
    print()

    sys.exit(0 if decision == "GO" else 1 if decision == "REDUCED" else 2)


if __name__ == "__main__":
    main()
