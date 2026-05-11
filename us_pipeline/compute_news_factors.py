"""Compute daily news factors per (date, ticker) from cached news JSONs.

Pipeline:
  1. Walk us_pipeline/data/news/<YYYY-MM-DD>.json
  2. Each article has {tickers, insights[], published_utc, ...}
     - "insights" is an AI-generated list of {ticker, sentiment, sentiment_reasoning}
     - We trust insights for sentiment; we use the broader `tickers` list for attention.
  3. Build long format: (signal_date, ticker, sentiment_int)  where signal_date is
     the trading day this article was visible by close.
  4. Aggregate per (signal_date, ticker) into the 9 features defined in the README.

Features computed per (date, ticker):
  - news_count_1d       : # articles for ticker on this trading day
  - news_count_5d       : # articles in past 5 trading days
  - news_sent_1d        : mean sentiment on this day (pos=+1, neg=-1, neutral=0)
  - news_sent_5d        : mean sentiment over past 5 days
  - news_sent_change    : sent_1d - sent_5d
  - news_pos_ratio_5d   : positive count / total count over past 5 days
  - news_neg_ratio_5d   : negative count / total count over past 5 days
  - news_attention_z    : (count_1d - mean_30d) / std_30d
  - news_silence_dummy  : 1 if no news in past 5 days, else 0

Output: us_pipeline/data/news_factors.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

ROOT = Path(__file__).parent
NEWS_DIR = ROOT / "data" / "news"
INSTR_FILE = ROOT / "data" / "instruments" / "sp500.txt"
OUT_PARQUET = ROOT / "data" / "news_factors.parquet"

PROVIDER = str(ROOT / "data" / "qlib_bin")

SENT_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def parse_all_news(universe: set[str]) -> pd.DataFrame:
    """Build long-format (signal_date, ticker, sentiment) dataframe.

    The signal_date is the date in the JSON filename (already aligned to a
    trading session window by download_news.py — articles published from
    D-1 16:00 ET through D 16:00 ET).
    """
    rows = []
    files = sorted(NEWS_DIR.glob("*.json"))
    print(f"Parsing {len(files)} news JSONs...")
    for fp in files:
        try:
            payload = json.loads(fp.read_text())
        except Exception:
            continue
        signal_date = pd.Timestamp(payload["date"])
        for art in payload.get("articles", []):
            insights = art.get("insights") or []
            tickers_set = set(art.get("tickers", []) or [])
            # Use insights for per-ticker sentiment when available
            covered = set()
            for ins in insights:
                t = (ins.get("ticker") or "").upper()
                if not t or t not in universe:
                    continue
                rows.append({
                    "signal_date": signal_date,
                    "ticker": t,
                    "sentiment": SENT_MAP.get(ins.get("sentiment", "neutral"), 0.0),
                    "has_insight": 1,
                })
                covered.add(t)
            # Tickers mentioned in `tickers` but not in insights: count as attention
            # but no sentiment signal (record neutral).
            for t in tickers_set - covered:
                t = t.upper()
                if t in universe:
                    rows.append({
                        "signal_date": signal_date,
                        "ticker": t,
                        "sentiment": 0.0,
                        "has_insight": 0,
                    })
    df = pd.DataFrame(rows)
    print(f"  parsed {len(df)} (article, ticker) rows from {len(files)} files")
    return df


def aggregate_per_day(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per (date, ticker) raw daily aggregations: count, mean sentiment, pos/neg counts."""
    g = long_df.groupby(["signal_date", "ticker"])
    daily = pd.DataFrame({
        "count": g.size(),
        "sent_mean": g["sentiment"].mean(),
        "n_pos": g.apply(lambda x: (x["sentiment"] > 0).sum()),
        "n_neg": g.apply(lambda x: (x["sentiment"] < 0).sum()),
    }).reset_index()
    return daily


def build_full_panel(daily: pd.DataFrame, calendar_dates: list, universe: list) -> pd.DataFrame:
    """Reindex per-day aggregations to the FULL trading calendar × universe.

    Most (date, ticker) pairs have no news → fill with zero/NaN. We need
    the full grid so rolling computations work.
    """
    cal = pd.DatetimeIndex(sorted(set(calendar_dates)))
    universe = sorted(set(t.upper() for t in universe))
    print(f"  full panel: {len(cal)} dates x {len(universe)} tickers = {len(cal)*len(universe)}")

    # Index daily by (date, ticker) for fast reindex
    daily_idx = daily.set_index(["signal_date", "ticker"]).sort_index()

    # MultiIndex of all (date, ticker)
    full_idx = pd.MultiIndex.from_product([cal, universe], names=["signal_date", "ticker"])
    panel = daily_idx.reindex(full_idx)
    panel["count"] = panel["count"].fillna(0).astype(int)
    panel["n_pos"] = panel["n_pos"].fillna(0).astype(int)
    panel["n_neg"] = panel["n_neg"].fillna(0).astype(int)
    # sent_mean stays NaN when no news (0 articles)
    return panel.reset_index()


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute the rolling features per ticker."""
    panel = panel.sort_values(["ticker", "signal_date"]).reset_index(drop=True)

    # Per-ticker rolling
    out = []
    for ticker, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("signal_date").copy()
        cnt = g["count"]
        sent_sum = g["sent_mean"] * g["count"].clip(lower=1)  # weighted by article count
        # 5d rolling sum of articles
        cnt_5d = cnt.rolling(5, min_periods=1).sum()
        sent_5d_sum = sent_sum.rolling(5, min_periods=1).sum()
        # mean sentiment over 5d (weighted by article count); 0 if no articles
        with np.errstate(divide="ignore", invalid="ignore"):
            sent_5d_mean = (sent_5d_sum / cnt_5d.replace(0, np.nan)).fillna(0.0)
        n_pos_5d = g["n_pos"].rolling(5, min_periods=1).sum()
        n_neg_5d = g["n_neg"].rolling(5, min_periods=1).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pos_ratio_5d = (n_pos_5d / cnt_5d.replace(0, np.nan)).fillna(0.0)
            neg_ratio_5d = (n_neg_5d / cnt_5d.replace(0, np.nan)).fillna(0.0)

        # Attention z-score over 30d
        cnt_30d_mean = cnt.rolling(30, min_periods=10).mean()
        cnt_30d_std = cnt.rolling(30, min_periods=10).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            attention_z = ((cnt - cnt_30d_mean) / cnt_30d_std.replace(0, np.nan)).fillna(0.0)

        silence = (cnt_5d == 0).astype(int)

        g["news_count_1d"] = cnt
        g["news_count_5d"] = cnt_5d
        g["news_sent_1d"] = g["sent_mean"].fillna(0.0)
        g["news_sent_5d"] = sent_5d_mean
        g["news_sent_change"] = g["news_sent_1d"] - g["news_sent_5d"]
        g["news_pos_ratio_5d"] = pos_ratio_5d
        g["news_neg_ratio_5d"] = neg_ratio_5d
        g["news_attention_z"] = attention_z
        g["news_silence_dummy"] = silence
        out.append(g)
    full = pd.concat(out, ignore_index=True)
    feat_cols = [c for c in full.columns if c.startswith("news_")]
    return full[["signal_date", "ticker"] + feat_cols].rename(
        columns={"signal_date": "datetime", "ticker": "instrument"}
    )


def main():
    qlib.init(provider_uri=PROVIDER, region="us")
    universe = set(t.strip().upper() for t in INSTR_FILE.read_text().splitlines() if t.strip())
    print(f"Universe: {len(universe)} tickers")

    long_df = parse_all_news(universe)
    if long_df.empty:
        print("No news data found. Run download_news.py first.")
        return

    print("\nAggregating per-day per-ticker...")
    daily = aggregate_per_day(long_df)
    print(f"  daily aggregations: {len(daily)} (date, ticker) rows with news")

    print("\nReindexing to full trading calendar × universe...")
    cal = D.calendar(start_time="2021-06-01", end_time=None, freq="day")
    panel = build_full_panel(daily, cal, list(universe))

    print("\nComputing rolling features...")
    feats = compute_features(panel)
    feats.to_parquet(OUT_PARQUET, index=False)

    print(f"\nWrote {len(feats)} rows -> {OUT_PARQUET}")
    print(f"  unique tickers: {feats['instrument'].nunique()}")
    print(f"  date range: {feats['datetime'].min()} -> {feats['datetime'].max()}")
    print()
    factor_cols = [c for c in feats.columns if c.startswith("news_")]
    print("Coverage (% non-zero) per factor:")
    for c in factor_cols:
        nonzero = (feats[c] != 0).mean() * 100
        print(f"  {c:<25} {nonzero:>6.1f}%")
    print()
    print("Mean / Std after winsorization at 1%/99%:")
    for c in factor_cols:
        s = feats[c].dropna()
        if len(s) == 0:
            continue
        lo, hi = s.quantile([0.01, 0.99])
        sw = s.clip(lo, hi)
        print(f"  {c:<25}  mean={sw.mean():>+8.3f}  std={sw.std():>8.3f}")


if __name__ == "__main__":
    main()
