"""Backtest the Codex-proposed 4-condition circuit breaker on the 125 historical signal days.

For each consecutive day pair, evaluate:
  - prior TOP 30 1d return < -1%
  - current TOP 30 overlap >= 70%
  - current TOP 30 top-sector >= 60%
  - repeated TOP 10's prior 1d return < 0

When all 4 fire: would skipping/halving the next day's long book have helped?
Compare the realized 5d return on circuit-breaker-fired days vs. non-fired days.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import qlib

ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = ROOT / "signals"
PROVIDER = str(ROOT / "data" / "qlib_bin")


def main():
    qlib.init(provider_uri=PROVIDER, region="us")
    from qlib.data import D

    csvs = sorted(SIGNALS_DIR.glob("*.csv"))
    pairs = []
    prev = None
    for fp in csvs:
        try:
            d = pd.Timestamp(fp.stem)
        except Exception:
            continue
        if prev is not None:
            pairs.append((prev, fp))
        prev = fp

    print(f"Evaluating circuit breaker on {len(pairs)} consecutive day pairs...")

    rows = []
    for prior_fp, curr_fp in pairs:
        prior = pd.read_csv(prior_fp).sort_values("rank")
        curr = pd.read_csv(curr_fp).sort_values("rank")
        prior_top30 = set(prior.head(30)["instrument"].astype(str).str.upper())
        prior_top10 = set(prior.head(10)["instrument"].astype(str).str.upper())
        curr_top30 = set(curr.head(30)["instrument"].astype(str).str.upper())
        curr_top10 = set(curr.head(10)["instrument"].astype(str).str.upper())

        repeated_top10 = curr_top10 & prior_top10
        top30_overlap = len(curr_top30 & prior_top30) / 30
        # top sector concentration
        if "sector" in curr.columns:
            top30_df = curr.head(30)
            sector_counts = top30_df["sector"].value_counts()
            top_sector_pct = sector_counts.iloc[0] / 30 if len(sector_counts) else 0
        else:
            top_sector_pct = 0

        # 1d return realized AT curr_date for the prior TOP 30 names
        curr_date = pd.Timestamp(curr_fp.stem)
        try:
            feats = D.features(list(prior_top30), ["$close / Ref($close, 1) - 1"],
                               start_time=curr_date, end_time=curr_date + pd.Timedelta(days=2),
                               freq="day")
            feats.columns = ["ret_1d"]
            feats = feats.reset_index()
            feats["instrument"] = feats["instrument"].str.upper()
            day = feats[feats["datetime"] == curr_date].set_index("instrument")
            prior_top30_1d = day["ret_1d"].mean() * 100
            repeated_in_day = [t for t in repeated_top10 if t in day.index]
            if repeated_in_day:
                repeated_cohort_1d = day.loc[repeated_in_day, "ret_1d"].mean() * 100
            else:
                repeated_cohort_1d = None
        except Exception:
            prior_top30_1d = None
            repeated_cohort_1d = None

        # Curr TOP 30's realized 5d return (the OUTCOME we're protecting against)
        try:
            feats5 = D.features(list(curr_top30), ["Ref($close, -6)/Ref($close, -1) - 1"],
                                start_time=curr_date, end_time=curr_date + pd.Timedelta(days=15),
                                freq="day")
            feats5.columns = ["ret_5d"]
            feats5 = feats5.reset_index()
            feats5["instrument"] = feats5["instrument"].str.upper()
            curr_5d_returns = feats5[feats5["datetime"] == curr_date].set_index("instrument")
            curr_top30_5d = curr_5d_returns["ret_5d"].mean() * 100
        except Exception:
            curr_top30_5d = None

        cb_fired = (
            prior_top30_1d is not None and prior_top30_1d < -1.0
            and top30_overlap >= 0.70
            and top_sector_pct >= 0.60
            and repeated_cohort_1d is not None and repeated_cohort_1d < 0
        )
        rows.append({
            "date": curr_fp.stem,
            "prior_top30_1d_pct": prior_top30_1d,
            "top30_overlap": top30_overlap,
            "top_sector_pct": top_sector_pct,
            "repeated_cohort_1d_pct": repeated_cohort_1d,
            "cb_fired": cb_fired,
            "curr_top30_5d_pct": curr_top30_5d,
        })

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "circuit_breaker_backtest.csv", index=False)
    print(f"\nWrote backtest to data/circuit_breaker_backtest.csv\n")

    fired = df[df["cb_fired"] == True]
    not_fired = df[(df["cb_fired"] == False) & df["curr_top30_5d_pct"].notna()]
    fired_scored = fired[fired["curr_top30_5d_pct"].notna()]

    print("=" * 80)
    print("CIRCUIT BREAKER BACKTEST")
    print("=" * 80)
    print(f"Total day-pairs evaluated: {len(df)}")
    print(f"Days where circuit breaker fired: {len(fired)} ({len(fired)/len(df)*100:.1f}%)")
    print(f"Days fired AND scored: {len(fired_scored)}")
    print(f"Days NOT fired AND scored: {len(not_fired)}")

    if len(fired_scored) > 0:
        m_fired = fired_scored["curr_top30_5d_pct"].mean()
        m_baseline = not_fired["curr_top30_5d_pct"].mean()
        print(f"\nMean TOP 30 5d return:")
        print(f"  Circuit breaker FIRED: {m_fired:+.2f}% (N={len(fired_scored)})")
        print(f"  Circuit breaker not fired: {m_baseline:+.2f}% (N={len(not_fired)})")
        print(f"  Difference (fired - baseline): {m_fired - m_baseline:+.2f}pp")
        print()
        print("If we SKIPPED long trades on circuit-breaker-fired days, we'd avoid an")
        print(f"average {m_fired - m_baseline:+.2f}pp underperformance per day vs baseline.")
        print()
        print("Days where circuit breaker fired:")
        for _, r in fired_scored.iterrows():
            print(f"  {r['date']}: prior_1d={r['prior_top30_1d_pct']:+.2f}%, "
                  f"overlap={r['top30_overlap']:.0%}, sector={r['top_sector_pct']:.0%}, "
                  f"cohort_1d={r['repeated_cohort_1d_pct']:+.2f}%, "
                  f"-> realized 5d={r['curr_top30_5d_pct']:+.2f}%")
    else:
        print("\nCircuit breaker has not yet fired on any historical day with scored data.")


if __name__ == "__main__":
    main()
