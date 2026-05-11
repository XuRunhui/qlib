"""Compute daily-frequency fundamental factors with point-in-time alignment.

Vectorized implementation using pandas merge_asof (the right tool for PIT joins).

Algorithm:
  1. Parse all reports for all tickers into one big DataFrame indexed by
     (ticker, pit_date). pit_date = filing_date when present, else end_date+60d.
  2. For each ticker, build a TTM panel (rolling 4-quarter sums of flow items,
     latest stock items as-of each report date).
  3. Use merge_asof to attach the most-recent-as-of-trading-day TTM snapshot to
     every (ticker, trading_date).
  4. Combine with daily close to compute price-based ratios (PE, PB, PS, etc).

Output: us_pipeline/data/fundamental_factors.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

ROOT = Path(__file__).parent
FUND_DIR = ROOT / "data" / "fundamentals"
INSTR_FILE = ROOT / "data" / "instruments" / "sp500.txt"
OUT_PARQUET = ROOT / "data" / "fundamental_factors.parquet"

PROVIDER = str(ROOT / "data" / "qlib_bin")
START = "2021-06-01"
END = "2026-04-24"


def _val(d: dict, key: str):
    if not d:
        return np.nan
    item = d.get(key)
    if item is None:
        return np.nan
    v = item.get("value") if isinstance(item, dict) else item
    if v is None:
        return np.nan
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_all_reports() -> pd.DataFrame:
    """Read every JSON in fundamentals/ and produce a flat DataFrame."""
    rows = []
    files = sorted(FUND_DIR.glob("*.json"))
    for fp in files:
        ticker = fp.stem.upper()
        try:
            raw = json.loads(fp.read_text())
        except Exception:
            continue
        for rep in raw.get("results", []):
            fin = rep.get("financials", {})
            inc = fin.get("income_statement", {})
            bs = fin.get("balance_sheet", {})
            cf = fin.get("cash_flow_statement", {})
            filing = rep.get("filing_date")
            end_date = rep.get("end_date")
            if not end_date:
                continue
            end_ts = pd.Timestamp(end_date)
            pit = pd.Timestamp(filing) if filing else end_ts + pd.Timedelta(days=60)
            rows.append({
                "ticker": ticker,
                "pit_date": pit,
                "end_date": end_ts,
                "period": rep.get("fiscal_period"),
                "timeframe": rep.get("timeframe"),
                "revenues": _val(inc, "revenues"),
                "gross_profit": _val(inc, "gross_profit"),
                "operating_income": _val(inc, "operating_income_loss"),
                "net_income": _val(inc, "net_income_loss"),
                "diluted_eps": _val(inc, "diluted_earnings_per_share"),
                "diluted_shares": _val(inc, "diluted_average_shares"),
                "assets": _val(bs, "assets"),
                "equity_p": _val(bs, "equity_attributable_to_parent"),
                "equity_t": _val(bs, "equity"),
                "long_term_debt": _val(bs, "long_term_debt"),
                "current_liab": _val(bs, "current_liabilities"),
                "noncurrent_liab": _val(bs, "noncurrent_liabilities"),
                "ocf": _val(cf, "net_cash_flow_from_operating_activities"),
                "icf": _val(cf, "net_cash_flow_from_investing_activities"),
            })
    df = pd.DataFrame(rows)
    df["equity"] = df["equity_p"].fillna(df["equity_t"])
    df = df.drop(columns=["equity_p", "equity_t"])
    return df


def build_ttm_panel(reports: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker TTM panel: rolling sum of last 4 quarterlies + latest balance items.

    Output is one row per (ticker, pit_date) with TTM flows + latest stock items.
    """
    out = []
    for ticker, g in reports.groupby("ticker"):
        # Use only quarterly + ttm rows (skip annuals as they overlap quarterlies)
        g = g.sort_values("pit_date").reset_index(drop=True)

        # Use TTM rows where Polygon provides them
        ttm_rows = g[g["timeframe"] == "ttm"].copy()
        # For quarterlies, build our own TTM
        q = g[g["timeframe"] == "quarterly"].copy()

        # Rolling 4-quarter sum on flow items (in chronological order)
        flow_cols = ["revenues", "gross_profit", "operating_income", "net_income",
                     "diluted_eps", "ocf", "icf"]
        if not q.empty:
            q = q.sort_values("pit_date").reset_index(drop=True)
            for c in flow_cols:
                q[f"ttm_{c}"] = q[c].rolling(4, min_periods=4).sum()
            # Stock items (balance sheet, shares): take as-of value, no sum
            stock_cols = ["assets", "equity", "long_term_debt",
                          "current_liab", "noncurrent_liab", "diluted_shares"]
            for c in stock_cols:
                q[f"ttm_{c}"] = q[c]  # use latest snapshot at this pit_date
            q["ticker"] = ticker
            keep = ["ticker", "pit_date"] + [f"ttm_{c}" for c in flow_cols + stock_cols]
            out.append(q[keep])

        # Add Polygon TTM rows as fallback (won't be used if quarterlies cover same date)
        if not ttm_rows.empty:
            ttm_rows = ttm_rows.sort_values("pit_date").reset_index(drop=True)
            for c in flow_cols:
                ttm_rows[f"ttm_{c}"] = ttm_rows[c]
            stock_cols = ["assets", "equity", "long_term_debt",
                          "current_liab", "noncurrent_liab", "diluted_shares"]
            for c in stock_cols:
                ttm_rows[f"ttm_{c}"] = ttm_rows[c]
            ttm_rows["ticker"] = ticker
            keep = ["ticker", "pit_date"] + [f"ttm_{c}" for c in flow_cols + stock_cols]
            out.append(ttm_rows[keep])

    panel = pd.concat(out, ignore_index=True)
    # Resolve duplicates: if both quarterly-derived and ttm-row exist for same pit_date,
    # prefer quarterly-derived (already the case via concat order if we drop dupes keep='first')
    panel = panel.sort_values(["ticker", "pit_date"]).drop_duplicates(["ticker", "pit_date"], keep="first")
    panel = panel.dropna(subset=["pit_date"]).reset_index(drop=True)
    return panel


def attach_yoy(panel: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, pit_date), compute YoY of TTM revenue and EPS."""
    panel = panel.sort_values(["ticker", "pit_date"]).reset_index(drop=True)
    # Use merge_asof per ticker: prior_panel has pit_date shifted by ~365d
    prior = panel.copy()
    prior["pit_date_lookup"] = prior["pit_date"] + pd.Timedelta(days=365)
    prior = prior[["ticker", "pit_date_lookup", "ttm_revenues", "ttm_diluted_eps"]].rename(
        columns={"ttm_revenues": "prior_ttm_revenues",
                 "ttm_diluted_eps": "prior_ttm_eps"}
    )
    # For each row in panel, find the prior TTM (must look at pit_date <= panel.pit_date - 300d, say)
    panel = panel.copy()
    panel["lookup_for_yoy"] = panel["pit_date"] - pd.Timedelta(days=300)
    # merge_asof needs both sides sorted on the merge key
    prior_sorted = prior.sort_values(["pit_date_lookup"])
    panel_sorted = panel.sort_values(["lookup_for_yoy"])
    merged = pd.merge_asof(
        panel_sorted,
        prior_sorted.rename(columns={"pit_date_lookup": "lookup_for_yoy"}),
        on="lookup_for_yoy",
        by="ticker",
        direction="backward",
    )
    merged["fund_revenue_yoy"] = (merged["ttm_revenues"] - merged["prior_ttm_revenues"]) / merged["prior_ttm_revenues"].abs()
    merged["fund_eps_yoy"] = (merged["ttm_diluted_eps"] - merged["prior_ttm_eps"]) / merged["prior_ttm_eps"].abs()
    return merged.drop(columns=["lookup_for_yoy", "prior_ttm_revenues", "prior_ttm_eps"])


def join_to_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Join PIT panel to daily trading calendar via merge_asof."""
    qlib.init(provider_uri=PROVIDER, region="us")
    tickers = sorted(panel["ticker"].unique().tolist())
    closes = D.features(tickers, ["$close"], start_time=START, end_time=END, freq="day")
    closes.columns = ["close"]
    closes = closes.reset_index()
    closes["instrument"] = closes["instrument"].str.upper()
    closes = closes.rename(columns={"instrument": "ticker", "datetime": "trade_date"})
    closes = closes.dropna(subset=["close"])
    closes = closes[closes["close"] > 0]

    # merge_asof per ticker: each trade_date gets the most recent panel row with pit_date <= trade_date
    panel = panel.sort_values(["pit_date"]).reset_index(drop=True)
    closes = closes.sort_values(["trade_date"]).reset_index(drop=True)
    out = pd.merge_asof(
        closes,
        panel,
        left_on="trade_date",
        right_on="pit_date",
        by="ticker",
        direction="backward",
    )
    return out


def compute_factors(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()
    close = df["close"]
    shares = df["ttm_diluted_shares"]
    mcap = close * shares

    def safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / b
        return r.replace([np.inf, -np.inf], np.nan)

    # Use abs-value EPS denom only when EPS positive (negative EPS -> NaN PE)
    pos_eps = df["ttm_diluted_eps"].where(df["ttm_diluted_eps"] > 0, np.nan)
    pos_equity = df["ttm_equity"].where(df["ttm_equity"] > 0, np.nan)
    pos_rev = df["ttm_revenues"].where(df["ttm_revenues"] > 0, np.nan)

    total_debt = df["ttm_long_term_debt"].fillna(0).clip(lower=0) + df["ttm_current_liab"].fillna(0).clip(lower=0)
    fcf = df["ttm_ocf"].fillna(0) + df["ttm_icf"].fillna(0)  # OCF + ICF; capex is negative ICF

    df["fund_pe_ttm"] = safe_div(close, pos_eps)
    df["fund_pb"] = safe_div(mcap, pos_equity)
    df["fund_ps_ttm"] = safe_div(mcap, pos_rev)
    df["fund_roe"] = safe_div(df["ttm_net_income"], pos_equity)
    df["fund_roa"] = safe_div(df["ttm_net_income"], df["ttm_assets"])
    df["fund_gross_margin"] = safe_div(df["ttm_gross_profit"], pos_rev)
    df["fund_op_margin"] = safe_div(df["ttm_operating_income"], pos_rev)
    df["fund_net_margin"] = safe_div(df["ttm_net_income"], pos_rev)
    df["fund_de_ratio"] = safe_div(total_debt, pos_equity)
    df["fund_log_mcap"] = np.log(mcap.where(mcap > 0, np.nan))
    df["fund_fcf_yield"] = safe_div(fcf, mcap.where(mcap > 0, np.nan))

    factor_cols = [c for c in df.columns if c.startswith("fund_")]
    out = df[["trade_date", "ticker"] + factor_cols].rename(
        columns={"trade_date": "datetime", "ticker": "instrument"}
    )
    return out


def main():
    print("Step 1: parse all report JSONs...")
    reports = parse_all_reports()
    print(f"  parsed {len(reports)} report rows from {reports['ticker'].nunique()} tickers")

    print("\nStep 2: build TTM panel...")
    panel = build_ttm_panel(reports)
    print(f"  TTM panel: {len(panel)} rows ({panel['ticker'].nunique()} tickers)")

    print("\nStep 3: compute YoY...")
    panel = attach_yoy(panel)
    print(f"  panel with YoY: {len(panel)} rows")

    print("\nStep 4: join to daily trading calendar...")
    joined = join_to_calendar(panel)
    print(f"  joined panel: {len(joined)} rows")

    print("\nStep 5: compute factors...")
    out = compute_factors(joined)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_PARQUET}")
    print(f"  unique tickers: {out['instrument'].nunique()}")
    print(f"  date range: {out['datetime'].min()} -> {out['datetime'].max()}")
    print()
    factor_cols = [c for c in out.columns if c.startswith("fund_")]
    print("Coverage per factor (% non-NaN):")
    cov = out[factor_cols].notna().mean() * 100
    for c in factor_cols:
        print(f"  {c:<25} {cov[c]:>6.1f}%")
    print()
    print("Stats (median, std after winsorization at 1%/99%):")
    for c in factor_cols:
        s = out[c].dropna()
        if len(s) == 0:
            continue
        lo, hi = s.quantile([0.01, 0.99])
        sw = s.clip(lo, hi)
        print(f"  {c:<25}  median={sw.median():>10.4f}  std={sw.std():>10.4f}")


if __name__ == "__main__":
    main()
