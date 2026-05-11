"""Paper trading log: track how each day's signals actually performed.

Workflow:
  - After each `generate_signals.py` run, this picks up the new signals/<date>.csv
    and appends them to the running paper-trade log.
  - When `--score` is run, for each historical signals row, compute the actual
    1-day, 3-day, 5-day forward returns of those picks, and update the log.

Output:
  us_pipeline/data/paper_trade_log/trades.csv  -- one row per (date, rank, symbol)
  us_pipeline/data/paper_trade_log/summary.md  -- running performance summary

Usage:
  python us_pipeline/signals/paper_trade_log.py --ingest    # ingest any new signals/*.csv
  python us_pipeline/signals/paper_trade_log.py --score     # compute realized returns for past signals
  python us_pipeline/signals/paper_trade_log.py --report    # print running performance
  python us_pipeline/signals/paper_trade_log.py --all       # do all three
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = "us_pipeline/data/qlib_bin"
SIGNALS_DIR = Path(__file__).parent
LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "paper_trade_log"
TRADES_CSV = LOG_DIR / "trades.csv"
SUMMARY_MD = LOG_DIR / "summary.md"


def _load_existing_trades() -> pd.DataFrame:
    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV)
        df["signal_date"] = pd.to_datetime(df["signal_date"], format="mixed")
        return df
    return pd.DataFrame(columns=[
        "signal_date", "rank", "instrument", "score", "pct_rank",
        "side",  # "long" or "short"
        "ret_1d", "ret_3d", "ret_5d",
    ])


def _save_trades(df: pd.DataFrame):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRADES_CSV, index=False)


def ingest_new_signals(topk: int = 30, botk: int = 30):
    """Pick up any signals CSVs not yet in the log and append top/bottom picks."""
    existing = _load_existing_trades()
    seen_dates = set(pd.to_datetime(existing["signal_date"]).dt.date.tolist())

    new_rows = []
    for csv_file in sorted(SIGNALS_DIR.glob("*.csv")):
        # filename format YYYY-MM-DD.csv
        try:
            d = pd.Timestamp(csv_file.stem).date()
        except Exception:
            continue
        if d in seen_dates:
            continue
        df = pd.read_csv(csv_file)
        df = df.sort_values("rank")
        top = df.head(topk).copy()
        bot = df.tail(botk).copy()
        top["side"] = "long"
        bot["side"] = "short"
        for sub in (top, bot):
            for _, r in sub.iterrows():
                new_rows.append({
                    "signal_date": d,
                    "rank": int(r["rank"]),
                    "instrument": r["instrument"],
                    "score": r["score"],
                    "pct_rank": r["pct_rank"],
                    "side": r["side"],
                    "ret_1d": np.nan, "ret_3d": np.nan, "ret_5d": np.nan,
                })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        _save_trades(combined)
        print(f"Ingested {len(new_df)} new picks from "
              f"{new_df['signal_date'].nunique()} signal day(s)")
    else:
        print("No new signal files to ingest.")


def score_returns():
    """Fill ret_1d, ret_3d, ret_5d for any rows still NaN, where data is available."""
    qlib.init(provider_uri=PROVIDER, region="us")
    from qlib.data import D

    df = _load_existing_trades()
    if df.empty:
        print("No trades to score (run --ingest first).")
        return

    df["signal_date"] = pd.to_datetime(df["signal_date"], format="mixed")
    needs_scoring = df[df["ret_5d"].isna()].copy()
    if needs_scoring.empty:
        print("All trades already scored.")
        return

    # Determine date range to fetch
    min_d = needs_scoring["signal_date"].min().date()
    max_d = needs_scoring["signal_date"].max().date() + timedelta(days=15)
    instruments = sorted(needs_scoring["instrument"].unique().tolist())

    print(f"Scoring {len(needs_scoring)} trades from {min_d} to {needs_scoring['signal_date'].max().date()}...")

    feats = D.features(instruments,
                       ["Ref($close, -2)/Ref($close, -1) - 1",   # 1d (T+1 close to T+2 close)
                        "Ref($close, -4)/Ref($close, -1) - 1",   # 3d
                        "Ref($close, -6)/Ref($close, -1) - 1"],  # 5d
                       start_time=min_d.isoformat(),
                       end_time=max_d.isoformat(), freq="day")
    feats.columns = ["ret_1d", "ret_3d", "ret_5d"]
    feats = feats.reset_index()
    feats["instrument"] = feats["instrument"].str.upper()

    # Merge: each trade row's (signal_date, instrument) -> the feats row at that date
    df["instrument_u"] = df["instrument"].str.upper()
    feats_for_join = feats.rename(columns={
        "datetime": "signal_date",
        "ret_1d": "ret_1d_new",
        "ret_3d": "ret_3d_new",
        "ret_5d": "ret_5d_new",
    })
    merged = df.merge(feats_for_join,
                      left_on=["signal_date", "instrument_u"],
                      right_on=["signal_date", "instrument"],
                      how="left", suffixes=("", "_drop"))
    # Backfill NaNs from new values
    for col in ("ret_1d", "ret_3d", "ret_5d"):
        merged[col] = merged[col].fillna(merged[f"{col}_new"])

    keep_cols = ["signal_date", "rank", "instrument", "score", "pct_rank", "side",
                 "ret_1d", "ret_3d", "ret_5d"]
    merged = merged[keep_cols].sort_values(["signal_date", "side", "rank"])
    _save_trades(merged)

    n_now_scored = merged["ret_5d"].notna().sum() - df["ret_5d"].notna().sum()
    print(f"Newly scored: {n_now_scored} trades")


def report():
    df = _load_existing_trades()
    if df.empty:
        print("No trades logged yet.")
        return
    df["signal_date"] = pd.to_datetime(df["signal_date"], format="mixed")

    print(f"\n=== Paper Trade Log Summary (as of {date.today()}) ===\n")
    print(f"  Total signal days logged: {df['signal_date'].nunique()}")
    print(f"  Total picks (long+short): {len(df)}")
    print(f"    - Long picks:  {(df['side']=='long').sum()}")
    print(f"    - Short picks: {(df['side']=='short').sum()}")
    print(f"  Date range: {df['signal_date'].min().date()} to {df['signal_date'].max().date()}")

    scored = df.dropna(subset=["ret_5d"])
    print(f"\n  Scored (have realized returns): {len(scored)} / {len(df)}")
    if scored.empty:
        print("  (Run --score after waiting at least 6 trading days for returns to materialize.)")
        return

    # Aggregate by side, then by date
    print("\n=== Realized Return per Pick (with cost ~10 bps single-trip) ===")
    cost = 0.001
    for horizon in (1, 3, 5):
        col = f"ret_{horizon}d"
        long_rets = scored[scored["side"] == "long"][col].dropna()
        short_rets = scored[scored["side"] == "short"][col].dropna()
        if long_rets.empty:
            continue
        # For long: net = ret - 2*cost (entry + exit)
        long_net = long_rets - 2*cost
        short_net = -short_rets - 2*cost  # short profits = -ret
        ls_net = long_net.mean() - (-short_rets.mean()) - 4*cost  # long-short pair: 4 trades
        print(f"\n  Horizon {horizon}d:")
        print(f"    Long  picks:   mean ret = {long_rets.mean()*100:+.2f}%   "
              f"win rate = {(long_rets > 0).mean()*100:.0f}%   "
              f"net of cost = {long_net.mean()*100:+.2f}%")
        print(f"    Short picks:   mean ret = {short_rets.mean()*100:+.2f}%   "
              f"loss rate = {(short_rets < 0).mean()*100:.0f}%   "
              f"net (as short) = {short_net.mean()*100:+.2f}%")
        print(f"    Long-Short:    spread   = {(long_rets.mean() - short_rets.mean())*100:+.2f}%")

    # Per-day stats — daily IC over time
    daily = []
    for d, g in scored.groupby("signal_date"):
        long_g = g[g["side"]=="long"]
        short_g = g[g["side"]=="short"]
        daily.append({
            "date": d.date(),
            "long_avg_5d": long_g["ret_5d"].mean(),
            "short_avg_5d": short_g["ret_5d"].mean(),
            "ls_5d": long_g["ret_5d"].mean() - short_g["ret_5d"].mean(),
            "n_picks": len(g),
        })
    if daily:
        dd = pd.DataFrame(daily)
        print("\n=== Per-day Long-Short Spread (5d) ===")
        for _, r in dd.iterrows():
            mark = "✓" if r["ls_5d"] > 0 else "✗"
            print(f"  {r['date']} | long={r['long_avg_5d']*100:+5.2f}%  "
                  f"short={r['short_avg_5d']*100:+5.2f}%  "
                  f"L-S={r['ls_5d']*100:+5.2f}% {mark}  ({int(r['n_picks'])} picks)")
        print(f"\n  Days with positive L-S spread: {(dd['ls_5d']>0).sum()}/{len(dd)} "
              f"({(dd['ls_5d']>0).mean()*100:.0f}%)")
        print(f"  Mean L-S spread: {dd['ls_5d'].mean()*100:+.2f}%")

    # Save markdown summary
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(_render_summary_md(df, scored))
    print(f"\n  Saved summary: {SUMMARY_MD}")


def _render_summary_md(all_df: pd.DataFrame, scored: pd.DataFrame) -> str:
    lines = ["# Paper Trade Performance Log", ""]
    lines.append(f"**As of:** {date.today()}")
    lines.append(f"**Signal days:** {all_df['signal_date'].nunique()}")
    lines.append(f"**Picks logged:** {len(all_df)}")
    lines.append(f"**Picks scored:** {len(scored)}")
    if scored.empty:
        lines.append("\n*(Awaiting 5 trading days for returns to materialize.)*")
        return "\n".join(lines)
    cost = 0.001
    lines.append("")
    lines.append("## Realized 5d Performance")
    lines.append("")
    lines.append("| Side | N | Mean Ret | Win/Loss Rate | Net of Cost |")
    lines.append("|---|---|---|---|---|")
    for side in ("long", "short"):
        sg = scored[scored["side"]==side]
        if sg.empty:
            continue
        rets = sg["ret_5d"].dropna()
        win = (rets > 0).mean() if side == "long" else (rets < 0).mean()
        net = (rets.mean() - 2*cost) if side == "long" else (-rets.mean() - 2*cost)
        lines.append(f"| {side} | {len(sg)} | {rets.mean()*100:+.2f}% | "
                     f"{win*100:.0f}% | {net*100:+.2f}% |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", action="store_true", help="Pick up new signal CSVs and append picks to log")
    p.add_argument("--score", action="store_true", help="Score realized returns for past picks")
    p.add_argument("--report", action="store_true", help="Print running performance summary")
    p.add_argument("--all", action="store_true", help="Run ingest + score + report in sequence")
    args = p.parse_args()

    if args.all:
        ingest_new_signals()
        score_returns()
        report()
    else:
        if args.ingest:
            ingest_new_signals()
        if args.score:
            score_returns()
        if args.report:
            report()
        if not (args.ingest or args.score or args.report):
            p.print_help()


if __name__ == "__main__":
    main()
