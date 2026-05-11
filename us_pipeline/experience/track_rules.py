"""Forward-validation tracker for Codex's two high-precision rules.

Reads all signal CSVs (which now contain rule_buy_high_precision and
rule_avoid_high_precision flags), pulls realized 5d returns from the case
library, and reports rolling precision/recall/PnL split into:
  - In-sample period (Oct 2025 → Apr 2024 — the period rules were discovered on)
  - Forward period (anything after rules were derived, 2026-04-28+)

Run this every few days as new signals come in. The rule's forward precision is
the only number that matters for production — in-sample precision (91% / 90%) is
known to be biased upward.

Usage:
  python us_pipeline/experience/track_rules.py
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = ROOT / "signals"
CASES_DIR = ROOT / "experience" / "cases"

RULE_DERIVATION_DATE = pd.Timestamp("2026-04-28")  # everything after this is OOS for the rules


def parse_yaml_frontmatter(fp: Path) -> dict | None:
    text = fp.read_text()
    m = re.match(r"---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    out = {}
    for line in m.group(1).split("\n"):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v == "TBD":
            v = None
        else:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k.strip()] = v
    return out


def load_cases() -> pd.DataFrame:
    cases = []
    for fp in sorted(CASES_DIR.glob("*.md")):
        c = parse_yaml_frontmatter(fp)
        if c:
            cases.append(c)
    df = pd.DataFrame(cases)
    df["date"] = pd.to_datetime(df["date"])
    df["ret_5d_pct"] = pd.to_numeric(df["ret_5d_pct"], errors="coerce")
    return df


def load_signals_with_rules() -> pd.DataFrame:
    """Read all signal CSVs and concatenate. Only those with rule columns are useful here."""
    rows = []
    for fp in sorted(SIGNALS_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if "rule_buy_high_precision" not in df.columns:
            continue
        df["signal_date"] = pd.to_datetime(fp.stem)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def merge_signals_with_realized(signals: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Join signal-day rule fires with realized 5d returns from price data.

    We use Qlib's `Ref($close, -6)/Ref($close, -1) - 1` as the truth — this matches
    how cases are scored. For signals already covered by cases (intersect on date+ticker),
    we get the realized return for free. For others (most), we compute it on demand.
    """
    cases_idx = cases.set_index(["date", "ticker"])[["ret_5d_pct"]]
    out = signals.merge(
        cases_idx, left_on=["signal_date", "instrument"], right_index=True, how="left"
    )

    # For rows where ret_5d_pct is null but the signal date is old enough, compute via Qlib
    needs_realized = out[out["ret_5d_pct"].isna() & (out["signal_date"] < pd.Timestamp(date.today()) - pd.Timedelta(days=8))]
    if len(needs_realized) > 0:
        import qlib
        from qlib.data import D
        qlib.init(provider_uri=str(ROOT / "data" / "qlib_bin"), region="us")
        # Pull all needed (date, ticker) pairs
        dates = needs_realized["signal_date"].unique()
        tickers = needs_realized["instrument"].unique().tolist()
        feats = D.features(
            tickers, ["Ref($close, -6)/Ref($close, -1) - 1"],
            start_time=min(dates), end_time=max(dates) + pd.Timedelta(days=15), freq="day",
        )
        feats.columns = ["ret_5d"]
        feats = feats.reset_index()
        feats["instrument"] = feats["instrument"].str.upper()
        # Match: feats["datetime"] == signal_date
        match = feats.rename(columns={"datetime": "signal_date"})
        out = out.merge(
            match, on=["signal_date", "instrument"], how="left", suffixes=("", "_pulled")
        )
        out["ret_5d_pct"] = out["ret_5d_pct"].fillna(out["ret_5d"] * 100 if "ret_5d" in out.columns else np.nan)
        out = out.drop(columns=["ret_5d"], errors="ignore")
    return out


def report(df: pd.DataFrame):
    df_scored = df[df["ret_5d_pct"].notna()].copy()
    print(f"Total signal-day picks (ranked top/bot 30): {len(df)}")
    print(f"Scored: {len(df_scored)}\n")

    for rule_col, label, target_sign in [
        ("rule_buy_high_precision", "BUY rule (consensus_long_worked predictor)", +1),
        ("rule_avoid_high_precision", "AVOID rule (model_correct_short predictor)", -1),
    ]:
        fires = df_scored[df_scored[rule_col] == True]
        if len(fires) == 0:
            print(f"\n=== {label} ===\n  No fires yet.\n")
            continue
        in_sample = fires[fires["signal_date"] < RULE_DERIVATION_DATE]
        oos = fires[fires["signal_date"] >= RULE_DERIVATION_DATE]
        print(f"\n=== {label} ===")
        print(f"Total fires: {len(fires)}  ({len(in_sample)} in-sample, {len(oos)} OOS)\n")

        for tag, sub in [("In-sample (rule discovery period)", in_sample),
                          ("Out-of-sample (forward validation)", oos)]:
            if len(sub) == 0:
                print(f"  {tag}: 0 fires (yet)")
                continue
            mean = sub["ret_5d_pct"].mean()
            std = sub["ret_5d_pct"].std()
            # "Win" = move in direction the rule predicts
            if target_sign > 0:
                wins = (sub["ret_5d_pct"] > 0).sum()
                big_wins = (sub["ret_5d_pct"] > 5).sum()
                big_losses = (sub["ret_5d_pct"] < -5).sum()
            else:
                wins = (sub["ret_5d_pct"] < 0).sum()
                big_wins = (sub["ret_5d_pct"] < -5).sum()
                big_losses = (sub["ret_5d_pct"] > 5).sum()
            win_pct = wins / len(sub) * 100
            print(f"  {tag}: N={len(sub)}, mean_5d={mean:+.2f}%, std={std:.2f}, win%={win_pct:.0f}%, "
                  f"big_wins={big_wins}, big_losses={big_losses}")

        # Compare to baseline (TOP 30 / BOT 30 mean of same days)
        if rule_col == "rule_buy_high_precision":
            baseline = df_scored[df_scored["rank"] <= 30]
        else:
            baseline = df_scored[df_scored["rank"] >= 474]
        baseline_mean = baseline["ret_5d_pct"].mean()
        print(f"  baseline (all picks of same side): N={len(baseline)}, mean_5d={baseline_mean:+.2f}%")
        print(f"  rule lift over baseline: {fires['ret_5d_pct'].mean() - baseline_mean:+.2f}pp")

    # OOS verdict — the punchline
    print("\n" + "=" * 80)
    print("OOS VERDICT (the only number that matters for production)")
    print("=" * 80)
    today = pd.Timestamp(date.today())
    days_oos = (today - RULE_DERIVATION_DATE).days
    print(f"Days since rule derivation: {days_oos}")
    if days_oos < 30:
        print(f"Need at least 30 days OOS before drawing conclusions. Currently {days_oos}.")
    elif days_oos < 60:
        print(f"At {days_oos} days OOS — preliminary read, but not yet enough for production.")
    else:
        print(f"At {days_oos} days OOS — sufficient for a production go/no-go decision.")


def report_in_sample_from_cases(cases: pd.DataFrame):
    """In-sample stats from the case library — we have rule flags + realized 5d here already."""
    df = cases[cases["ret_5d_pct"].notna()].copy()
    if "rule_buy_high_precision" not in df.columns:
        print("Cases do not yet have rule columns. Run build_cases.py with the updated schema.")
        return None
    df["rule_buy_high_precision"] = df["rule_buy_high_precision"].astype(str) == "True"
    df["rule_avoid_high_precision"] = df["rule_avoid_high_precision"].astype(str) == "True"
    return df


def main():
    cases = load_cases()
    print(f"Loaded {len(cases)} cases (in-sample data)\n")

    cases_clean = report_in_sample_from_cases(cases)
    if cases_clean is None:
        return

    in_sample = cases_clean[cases_clean["date"] < RULE_DERIVATION_DATE].copy()
    in_sample["signal_date"] = in_sample["date"]
    in_sample["instrument"] = in_sample["ticker"]
    in_sample["rank"] = in_sample.get("model_rank", 0).astype(float)

    signals = load_signals_with_rules()
    if not signals.empty:
        n_inst = signals.groupby("signal_date")["rank"].max().median()
        signals = signals[(signals["rank"] <= 30) | (signals["rank"] >= n_inst - 29)].copy()
        oos_signals = signals[signals["signal_date"] >= RULE_DERIVATION_DATE].copy()
        if not oos_signals.empty:
            oos_signals = merge_signals_with_realized(oos_signals, cases)
        all_data = pd.concat([in_sample, oos_signals], ignore_index=True)
    else:
        all_data = in_sample

    print(f"In-sample case rows (Oct 2025–Apr 2026 pre-rule): {len(in_sample)}")
    print(f"OOS signal rows (Apr 28+): {len(all_data) - len(in_sample)}\n")
    report(all_data)


if __name__ == "__main__":
    main()
