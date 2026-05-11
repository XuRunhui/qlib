"""Generate today's trading signals using production model (C: LR + Alpha158, sector-neutral).

Workflow:
  1. Initialize Qlib with current bin data (must be updated via download_polygon.py first)
  2. Train LambdaRank model on ALL available history up to (today - 30 days)
  3. Validate on the last 30 days
  4. Predict for the most recent trading day (today's close)
  5. Output ranked CSV: top 30 (BUY) and bottom 30 (AVOID)

Output:
  us_pipeline/signals/<YYYY-MM-DD>.csv  -- rank, symbol, score, sector, signal_strength
  us_pipeline/signals/<YYYY-MM-DD>_summary.md  -- human-readable summary

Usage:
  python us_pipeline/signals/generate_signals.py            # uses today's date
  python us_pipeline/signals/generate_signals.py 2026-04-24 # specific date
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROVIDER = "us_pipeline/data/qlib_bin"
EXPNAME = "live_signals"
SIGNALS_DIR = Path(__file__).parent
SECTORS_CSV = Path(__file__).resolve().parent.parent / "data" / "instruments" / "sectors.csv"

LABEL_5D = "Ref($close, -6) / Ref($close, -1) - 1"


def get_latest_trading_date():
    """Find the most recent date in our data."""
    qlib.init(provider_uri=PROVIDER, region="us")
    from qlib.data import D
    cal = D.calendar(freq="day")
    if len(cal) == 0:
        raise RuntimeError("Empty Qlib calendar — run download_polygon.py first")
    return pd.Timestamp(cal[-1]).date()


def make_handler_cfg(train_start: str, train_end: str,
                     valid_start: str, valid_end: str,
                     test_start: str, test_end: str):
    """Sector-neutral, no-fundamentals (variant C from Exp 5)."""
    return {
        "start_time": train_start, "end_time": test_end,
        "fit_start_time": train_start, "fit_end_time": train_end,
        "instruments": "sp500",
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "SectorNeutralRank",
             "module_path": "us_pipeline.sector_processor",
             "kwargs": {"fields_group": "label",
                        "sector_csv": "us_pipeline/data/instruments/sectors.csv"}},
        ],
        "label": [LABEL_5D],
    }


def train_and_predict(target_date: date) -> pd.DataFrame:
    """Train LambdaRank on history up to target_date - 30d; predict target_date."""
    train_start = "2021-06-01"
    target_str = target_date.isoformat()
    valid_end = (target_date - timedelta(days=1)).isoformat()
    valid_start = (target_date - timedelta(days=30)).isoformat()
    train_end = (target_date - timedelta(days=31)).isoformat()

    print(f"  Train: {train_start} -> {train_end}")
    print(f"  Valid: {valid_start} -> {valid_end}")
    print(f"  Predict: {target_str} (single day)")

    handler_cfg = make_handler_cfg(train_start, train_end,
                                   valid_start, valid_end,
                                   target_str, target_str)
    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_cfg,
            },
            "segments": {
                "train": [train_start, train_end],
                "valid": [valid_start, valid_end],
                "test":  [target_str, target_str],
            }
        }
    })
    model = init_instance_by_config({
        "class": "LGBRankModel", "module_path": "us_pipeline.lgb_rank_model",
        "kwargs": {
            "n_bins": 16, "learning_rate": 0.02, "num_leaves": 64, "max_depth": 6,
            "lambda_l1": 50.0, "lambda_l2": 100.0,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "num_threads": 8, "early_stopping_rounds": 50, "num_boost_round": 1000,
        }
    })

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with R.start(experiment_name=EXPNAME, recorder_name=f"signals_{target_str}", resume=False):
            model.fit(dataset)
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
            pred = recorder.load_object("pred.pkl")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    return pred


def _aggregate_news_for_signal(target_date: date, instruments: list[str]) -> pd.DataFrame:
    """Per-ticker news aggregates for the 5 trading sessions ending at target_date.
    Same logic as build_cases.py's aggregate_news_for_ticker but vectorized."""
    import json as _json
    from datetime import timedelta as _td
    NEWS_DIR = Path(__file__).resolve().parent.parent / "data" / "news"
    SENT_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    universe = set(s.upper() for s in instruments)
    rows = {t: {"n_articles": 0, "n_pos": 0, "n_neg": 0, "n_neutral": 0,
                "title": "", "reasoning": "", "published_utc": "", "keywords": []}
            for t in universe}
    sig_ts = pd.Timestamp(target_date)
    for offset in range(5):
        d = (sig_ts - pd.tseries.offsets.BDay(offset)).date()
        f = NEWS_DIR / f"{d.isoformat()}.json"
        if not f.exists():
            continue
        try:
            payload = _json.loads(f.read_text())
        except Exception:
            continue
        for art in payload.get("articles", []) or []:
            insights = art.get("insights") or []
            ticker_insight_map = {}
            for ins in insights:
                t = (ins.get("ticker") or "").upper()
                if t in universe:
                    ticker_insight_map[t] = ins
            for t in (set(art.get("tickers", []) or []) | set(ticker_insight_map.keys())):
                t = t.upper()
                if t not in universe:
                    continue
                rows[t]["n_articles"] += 1
                ins = ticker_insight_map.get(t)
                sent = (ins.get("sentiment") if ins else "neutral") or "neutral"
                if sent == "positive": rows[t]["n_pos"] += 1
                elif sent == "negative": rows[t]["n_neg"] += 1
                else: rows[t]["n_neutral"] += 1
                # Capture first (most recent — we iterate offset=0 forward)
                if not rows[t]["published_utc"]:
                    rows[t]["title"] = art.get("title", "")
                    rows[t]["reasoning"] = (ins.get("sentiment_reasoning", "") if ins else "")
                    rows[t]["published_utc"] = art.get("published_utc", "")
                    rows[t]["keywords"] = (art.get("keywords") or [])[:5]
    def _clean_text(value, max_len):
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:max_len]

    out = []
    for t, r in rows.items():
        n = r["n_articles"]
        if n == 0:
            label = "NO-NEWS"
        else:
            net = r["n_pos"] - r["n_neg"]
            if r["n_pos"] >= 3 and r["n_neg"] == 0:
                label = "STRONG-POS"
            elif r["n_neg"] >= 2 and r["n_pos"] == 0:
                label = "STRONG-NEG"
            elif net > 0:
                label = "POS"
            elif net < 0:
                label = "NEG"
            else:
                label = "NEUTRAL"
        # pub_lag_days, pub_hour_utc
        pub_lag = None
        pub_hour = None
        if r["published_utc"]:
            try:
                pub_ts = pd.to_datetime(r["published_utc"].replace("Z", "")).tz_localize(None)
                pub_lag = (sig_ts - pub_ts).total_seconds() / 86400
                pub_hour = pub_ts.hour
            except Exception:
                pass
        out.append({
            "instrument": t,
            "news_count_5d": n,
            "news_pos_count_5d": r["n_pos"],
            "news_neg_count_5d": r["n_neg"],
            "news_sentiment": label,
            "most_recent_published_utc": r["published_utc"],
            "pub_lag_days": pub_lag,
            "pub_hour_utc": pub_hour,
            "most_recent_title": _clean_text(r["title"], 160),
            "most_recent_reasoning": _clean_text(r["reasoning"], 280),
            "most_recent_keywords": _clean_text(";".join(r["keywords"]), 160),
        })
    return pd.DataFrame(out)


def attach_metadata(pred: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Add sector + close + recent return + ranking + news aggregates + rule flags."""
    from qlib.data import D

    score = pred.iloc[:, 0].rename("score")
    instruments = score.index.get_level_values("instrument").unique().tolist()
    end = target_date.isoformat()
    start = (target_date - timedelta(days=30)).isoformat()
    feats = D.features(instruments,
                       ["$close",
                        "$close / Ref($close, 5) - 1",   # 5d return
                        "$close / Ref($close, 20) - 1",  # 20d return
                        "Std($close/Ref($close,1) - 1, 20) * 16"],  # 20d ann vol
                       start_time=start, end_time=end, freq="day")
    feats.columns = ["close", "ret_5d", "ret_20d", "ann_vol_20d"]

    # Most recent row per instrument
    feats = feats.dropna().groupby(level="instrument").tail(1).reset_index()
    feats["instrument"] = feats["instrument"].str.upper()

    sectors = pd.read_csv(SECTORS_CSV)
    sectors["instrument"] = sectors["symbol"].str.upper()

    df = score.reset_index()
    df["instrument"] = df["instrument"].str.upper()
    df = df.merge(feats[["instrument", "close", "ret_5d", "ret_20d", "ann_vol_20d"]],
                  on="instrument", how="left")
    df = df.merge(sectors[["instrument", "sector", "sic_description"]],
                  on="instrument", how="left")

    # Attach news aggregates
    news_df = _aggregate_news_for_signal(target_date, df["instrument"].tolist())
    df = df.merge(news_df, on="instrument", how="left")

    # Ranking
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["pct_rank"] = (df["rank"] - 0.5) / len(df)

    # Codex rules (paper-track only — see L53/L54)
    n_total = len(df)
    df["rule_avoid_high_precision"] = (
        (df["rank"] >= max(1, n_total - 29))   # bottom 30 (rank ≥ 474 in 503-stock universe)
        & (df["news_sentiment"] == "STRONG-POS")
        & (df["pub_lag_days"] < 1)
        & (df["pub_hour_utc"] >= 16)
        & (df["ret_5d"] > 0)
    ).fillna(False)
    df["rule_buy_high_precision"] = (
        (df["rank"] <= 30)
        & (df["news_sentiment"].isin(["STRONG-POS", "POS"]))
        & (df["ret_5d"] * 100 <= -1.79)
        & (df["pub_lag_days"] <= 1)
    ).fillna(False)

    # Legal veto rule (Codex L64+L68, paper-track only) — narrower than L44's broad STRONG-NEG.
    # Catches BSX-Apr-22 type cases: model BUY + STRONG-NEG news + legal/securities-fraud catalyst.
    # Codex N=20 spot-check: proxy (news_neg_count >= 2) was 10/20 legal; real keyword match on
    # catalyst text is structurally cleaner. L68: legal-keyword fires actually averaged +3.13% / 5d
    # — so this is a NARROWER classifier, not a confirmed loss-predictor. Keep paper-track only.
    LEGAL_RE = re.compile(
        r"\b(?:"
        r"class[- ]action|securities[- ](?:fraud|litigation)|"
        r"shareholders?(?: rights| litigation)?|"
        r"lawsuit|sued|complaint filed|lead plaintiff|"
        r"investigation|investigated|SEC|DOJ|"
        r"false statements|misleading statements|materially false|"
        r"fraud|alleged|alleges|violations?"
        r")\b",
        re.IGNORECASE,
    )

    def _has_legal_keyword(row):
        text = " ".join([
            str(row.get("most_recent_title", "") or ""),
            str(row.get("most_recent_reasoning", "") or ""),
            str(row.get("most_recent_keywords", "") or ""),
        ])
        return bool(LEGAL_RE.search(text))

    df["rule_buy_legal_veto"] = (
        (df["rank"] <= 30)
        & (df["news_sentiment"] == "STRONG-NEG")
        & df.apply(_has_legal_keyword, axis=1)
    ).fillna(False)
    return df


def compute_persistence_diagnostics(df: pd.DataFrame, target_date: date) -> dict:
    """Codex-proposed diagnostics: overlap with prior signal day + repeated cohort returns.

    Returns dict of scalar metrics for daily summary. None if prior day's signal CSV missing.
    """
    target_str = target_date.isoformat()
    prior_csv = None
    for f in sorted(SIGNALS_DIR.glob("*.csv"), reverse=True):
        if f.stem < target_str:
            prior_csv = f
            break
    if prior_csv is None:
        return {}
    try:
        prior = pd.read_csv(prior_csv).sort_values("rank")
    except Exception:
        return {}
    prior_top30 = set(prior.head(30)["instrument"].astype(str).str.upper())
    prior_top10 = set(prior.head(10)["instrument"].astype(str).str.upper())
    prior_bot30 = set(prior.tail(30)["instrument"].astype(str).str.upper())

    today_top30 = set(df.head(30)["instrument"].astype(str).str.upper())
    today_top10 = set(df.head(10)["instrument"].astype(str).str.upper())
    today_bot30 = set(df.tail(30)["instrument"].astype(str).str.upper())

    # Repeated TOP 10 cohort: names in BOTH today's and yesterday's TOP 10.
    # Prior 1d return = (close on prior_signal_date) / (close on day-before-prior_signal_date) - 1
    # Approximate via prior CSV's ret_1d if column exists, else compute from Qlib.
    repeated_top10 = today_top10 & prior_top10

    # Pull realized 1d returns for prior signal date's TOP 30.
    # Logic: prior_signal_date = T-1, current target_date = T.
    # If you traded prior's TOP 30 at close of T-1, you'd be holding into T's close.
    # So "prior TOP 30 1d return" = (T close / T-1 close - 1) for those names.
    prior_signal_date = pd.Timestamp(prior_csv.stem)
    try:
        from qlib.data import D
        prior_top30_list = list(prior.head(30)["instrument"].astype(str).str.upper())
        feats = D.features(prior_top30_list, ["$close / Ref($close, 1) - 1"],
                           start_time=prior_signal_date, end_time=pd.Timestamp(target_date) + pd.Timedelta(days=2),
                           freq="day")
        feats.columns = ["ret_1d_realized"]
        feats = feats.reset_index()
        feats["instrument"] = feats["instrument"].str.upper()
        target_ts = pd.Timestamp(target_date)
        # The 1d return measured AT the current target_date (T close vs T-1 close)
        target_returns = feats[feats["datetime"] == target_ts].set_index("instrument")
        prior_top30_1d_mean = target_returns["ret_1d_realized"].mean()
        if pd.isna(prior_top30_1d_mean):
            prior_top30_1d_mean = None
        # Repeated TOP 10's prior 1d return
        repeated_in_prior = [t for t in repeated_top10 if t in target_returns.index]
        if repeated_in_prior:
            repeated_top10_prior_1d_ret = target_returns.loc[repeated_in_prior, "ret_1d_realized"].mean()
            if pd.isna(repeated_top10_prior_1d_ret):
                repeated_top10_prior_1d_ret = None
        else:
            repeated_top10_prior_1d_ret = None
    except Exception:
        prior_top30_1d_mean = None
        repeated_top10_prior_1d_ret = None

    # Sector concentration of today's TOP 30
    top30_df = df.head(30)
    sector_counts = top30_df["sector"].value_counts() if "sector" in top30_df.columns else pd.Series()
    top_sector = sector_counts.index[0] if len(sector_counts) else None
    top_sector_count = int(sector_counts.iloc[0]) if len(sector_counts) else 0
    top_sector_pct = top_sector_count / 30 if len(sector_counts) else 0.0
    hhi = float(((sector_counts / 30) ** 2).sum()) if len(sector_counts) else 0.0

    # Cohort-overheat pre-loss diagnostic (Codex L65) — pre-loss signal vs CB's post-loss signal
    top10_df = df.head(10)
    top10_sector_counts = top10_df["sector"].value_counts() if "sector" in top10_df.columns else pd.Series()
    top10_sector_pct = top10_sector_counts.iloc[0] / 10 if len(top10_sector_counts) else 0.0
    top10_pos_count = (top10_df.get("news_sentiment", pd.Series()).isin(["POS", "STRONG-POS"])).sum()
    top10_strong_neg_count = (top10_df.get("news_sentiment", pd.Series()) == "STRONG-NEG").sum()
    top10_pre5d_mean_pct = (top10_df["ret_5d"].mean() * 100) if "ret_5d" in top10_df.columns else None

    # Repeated cohort 3d prior return (for CB subrule, Codex L66)
    repeated_cohort_3d_prior_pct = None
    if repeated_top10:
        try:
            from qlib.data import D
            feats3 = D.features(list(repeated_top10), ["$close / Ref($close, 3) - 1"],
                                start_time=prior_signal_date, end_time=pd.Timestamp(target_date) + pd.Timedelta(days=2),
                                freq="day")
            feats3.columns = ["ret_3d_prior"]
            feats3 = feats3.reset_index()
            feats3["instrument"] = feats3["instrument"].str.upper()
            day_data = feats3[feats3["datetime"] == prior_signal_date].set_index("instrument")
            mean_3d = day_data["ret_3d_prior"].mean()
            if not pd.isna(mean_3d):
                repeated_cohort_3d_prior_pct = mean_3d * 100
        except Exception:
            pass

    # News negative pct of repeated cohort
    news_neg_pct = None
    if "news_neg_count_5d" in df.columns and "news_count_5d" in df.columns and repeated_top10:
        cohort_df = df[df["instrument"].isin(repeated_top10)]
        total_count = cohort_df["news_count_5d"].sum()
        neg_count = cohort_df["news_neg_count_5d"].sum()
        if total_count > 0:
            news_neg_pct = float(neg_count) / float(total_count)

    return {
        "top30_overlap_prev": len(today_top30 & prior_top30) / 30,
        "top10_overlap_prev": len(today_top10 & prior_top10) / 10,
        "bot30_overlap_prev": len(today_bot30 & prior_bot30) / 30,
        "top10_same_names": ";".join(sorted(repeated_top10)) if repeated_top10 else "",
        "top10_same_count": len(repeated_top10),
        "prior_top30_1d_mean_pct": (prior_top30_1d_mean * 100) if prior_top30_1d_mean is not None else None,
        "repeated_top10_prior_1d_ret_pct": (repeated_top10_prior_1d_ret * 100) if repeated_top10_prior_1d_ret is not None else None,
        "repeated_cohort_3d_prior_pct": repeated_cohort_3d_prior_pct,
        "top_sector": top_sector,
        "top_sector_pct": top_sector_pct,
        "top10_sector_pct": top10_sector_pct,
        "top10_pos_count": int(top10_pos_count),
        "top10_strong_neg_count": int(top10_strong_neg_count),
        "top10_pre5d_mean_pct": top10_pre5d_mean_pct,
        "news_neg_pct": news_neg_pct,
        "sector_hhi": hhi,
        "prior_signal_date": prior_csv.stem,
    }


def evaluate_circuit_breaker(diag: dict) -> dict:
    """Codex's 4-condition circuit breaker — paper-track only, NOT for live trading.

    Triggers when:
      - prior TOP 30 1d return < -1%
      - current TOP 30 overlap >= 70%
      - current TOP 30 top-sector >= 60%
      - repeated TOP 10's prior 1d return < 0
    Action candidate: cut long book to 0.5x or require manual review.
    """
    if not diag:
        return {"circuit_breaker_fired": False, "reason": "no_prior_data"}
    conds = {
        "prior_loss": diag.get("prior_top30_1d_mean_pct") is not None and diag["prior_top30_1d_mean_pct"] < -1.0,
        "high_overlap": diag.get("top30_overlap_prev", 0) >= 0.70,
        "concentrated": diag.get("top_sector_pct", 0) >= 0.60,
        "cohort_lost": diag.get("repeated_top10_prior_1d_ret_pct") is not None and diag["repeated_top10_prior_1d_ret_pct"] < 0,
    }
    fired = all(conds.values())

    # Codex L66 subrule: when CB fires, predict if it's a "bad fire" (loss continues)
    # vs a "recovery fire" (cohort bounces back). 3/3 in-sample with 1 FP across 14 fires.
    # Conditions: 4-cond CB fires + cohort still hot + shallow first crack + meaningful neg news.
    subrule_bad = (
        fired
        and diag.get("repeated_cohort_3d_prior_pct") is not None and diag["repeated_cohort_3d_prior_pct"] > 0
        and diag.get("top10_pre5d_mean_pct") is not None and diag["top10_pre5d_mean_pct"] > 3
        and diag.get("repeated_top10_prior_1d_ret_pct") is not None and diag["repeated_top10_prior_1d_ret_pct"] > -4
        and diag.get("news_neg_pct") is not None and diag["news_neg_pct"] >= 0.12
    )
    cb_classification = "no_fire"
    if fired:
        cb_classification = "BAD_likely_continues" if subrule_bad else "RECOVERY_likely_bounces"

    # Cohort-overheat tail-risk diagnostic (L65 → renamed L69 after backtest).
    # Codex N=125 backtest: when fired (40% of days), TOP 30 mean 5d -0.28% vs +2.29% baseline,
    # big-loss rate 18.8% vs 2.7%, t≈-3.69. NOT a deterministic loss predictor — TOP 10 still
    # averages +1.18% on fire days. Reframe: this is a fat-tail / lower-expectancy WARNING.
    overheat = (
        diag.get("top10_sector_pct", 0) >= 0.70
        and diag.get("top10_pos_count", 0) >= 5
        and diag.get("top10_pre5d_mean_pct") is not None and diag["top10_pre5d_mean_pct"] > 0
        and diag.get("top10_strong_neg_count", 0) == 0
    )

    return {
        "circuit_breaker_fired": fired,
        "cb_prior_loss": conds["prior_loss"],
        "cb_high_overlap": conds["high_overlap"],
        "cb_concentrated": conds["concentrated"],
        "cb_cohort_lost": conds["cohort_lost"],
        "cb_classification": cb_classification,
        "cohort_overheat_tail_risk": overheat,
    }


def write_outputs(df: pd.DataFrame, target_date: date,
                  topk: int = 30, botk: int = 30,
                  diag: dict | None = None):
    diag = diag or {}
    target_str = target_date.isoformat()
    csv_path = SIGNALS_DIR / f"{target_str}.csv"
    md_path = SIGNALS_DIR / f"{target_str}_summary.md"

    # Save full ranked CSV
    import csv as _csv
    out_cols = ["rank", "instrument", "score", "pct_rank",
                "close", "ret_5d", "ret_20d", "ann_vol_20d",
                "sector", "sic_description",
                "news_count_5d", "news_pos_count_5d", "news_neg_count_5d",
                "news_sentiment", "most_recent_published_utc",
                "pub_lag_days", "pub_hour_utc",
                "most_recent_title", "most_recent_reasoning", "most_recent_keywords",
                "rule_avoid_high_precision", "rule_buy_high_precision",
                "rule_buy_legal_veto"]
    df[out_cols].to_csv(csv_path, index=False, quoting=_csv.QUOTE_ALL, lineterminator="\n")
    print(f"  Wrote full rankings: {csv_path}")

    # Build human-readable summary
    top = df.head(topk).copy()
    bot = df.tail(botk).copy().sort_values("rank", ascending=False)

    lines = []
    lines.append(f"# Trading Signals — {target_str}")
    lines.append("")
    lines.append(f"**Model**: LambdaRank on Alpha158 (Production config from Exp 5)")
    lines.append(f"**Universe**: {len(df)} S&P 500 stocks")
    lines.append(f"**Holding horizon**: 5-day swing trade")
    lines.append(f"**Trade execution**: Buy at next-day close (T+1)")
    lines.append("")

    # Portfolio diagnostics (Codex-proposed, L59)
    if diag:
        lines.append("## 📊 Portfolio diagnostics (vs. yesterday)")
        lines.append("")
        cb_fired = diag.get("circuit_breaker_fired", False)
        overheat = diag.get("cohort_overheat_tail_risk", False)
        cb_class = diag.get("cb_classification", "no_fire")

        if cb_fired:
            if cb_class == "BAD_likely_continues":
                cb_label = "🔴 **CIRCUIT BREAKER FIRED — subrule says BAD (loss likely continues, in-sample 3/3 + 1FP)**"
            else:
                cb_label = "🟡 CIRCUIT BREAKER FIRED — subrule says RECOVERY (cohort likely bounces, in-sample pattern)"
        else:
            cb_label = "🟢 No circuit breaker"
        if overheat:
            cb_label += " · ⚠ **Cohort-overheat tail-risk** (L69: when fired, mean 5d -0.28% vs +2.29% baseline, big-loss rate 18.8% vs 2.7%; not a hard veto, sizing alert)"
        lines.append(f"**Status**: {cb_label}")
        lines.append("")
        lines.append("| Metric | Value | Threshold |")
        lines.append("|---|---|---|")
        if diag.get("prior_signal_date"):
            lines.append(f"| Prior signal date | {diag['prior_signal_date']} | — |")
        if diag.get("prior_top30_1d_mean_pct") is not None:
            lines.append(f"| Prior TOP 30 1d return | **{diag['prior_top30_1d_mean_pct']:+.2f}%** | < -1% triggers |")
        if "top30_overlap_prev" in diag:
            lines.append(f"| TOP 30 overlap with prior | **{diag['top30_overlap_prev']*100:.0f}%** | ≥ 70% triggers |")
        if "top10_overlap_prev" in diag:
            lines.append(f"| TOP 10 overlap with prior | {diag['top10_overlap_prev']*100:.0f}% | (informational) |")
        if "bot30_overlap_prev" in diag:
            lines.append(f"| BOT 30 overlap with prior | {diag['bot30_overlap_prev']*100:.0f}% | (typically ~30%) |")
        if "top_sector_pct" in diag:
            lines.append(f"| Top sector concentration | **{diag['top_sector_pct']*100:.0f}% {diag.get('top_sector','')}** | ≥ 60% triggers |")
        if "sector_hhi" in diag:
            lines.append(f"| Sector HHI | {diag['sector_hhi']:.3f} | (informational) |")
        if diag.get("repeated_top10_prior_1d_ret_pct") is not None:
            lines.append(f"| Repeated TOP 10 cohort prior 1d ret | **{diag['repeated_top10_prior_1d_ret_pct']:+.2f}%** | < 0 triggers |")
        if diag.get("top10_same_names"):
            lines.append(f"| Repeated TOP 10 names | `{diag['top10_same_names']}` | (cohort identification) |")
        lines.append("")
        if cb_fired:
            lines.append("> 🚨 **Circuit breaker logic** (Codex-proposed L59, paper-track only): all 4 conditions met — prior TOP 30 lost money, today's TOP 30 is ≥70% same names, top sector ≥60% concentrated, AND the repeated cohort lost money yesterday. **Action candidate**: cut long book to 0.5×, or require manual override before trading. **NOT a production rule yet** — paper-track for 30+ days first.")
            lines.append("")



    # Codex rule fires (paper-track only, in-sample-discovered, NOT for live trading)
    buy_rule_fires = df[df["rule_buy_high_precision"] == True]
    avoid_rule_fires = df[df["rule_avoid_high_precision"] == True]
    if len(buy_rule_fires) > 0 or len(avoid_rule_fires) > 0:
        lines.append("---")
        lines.append("")
        lines.append("## 🔬 Codex high-precision rules — paper-track only")
        lines.append("")
        lines.append("> **Caveat:** these rules were discovered in-sample on 1132 cases. They are NOT validated forward yet. Track them for 30-60 days before any live use. See README L53/L54.")
        lines.append("")
        if len(buy_rule_fires) > 0:
            lines.append(f"### `rule_buy_high_precision` fires ({len(buy_rule_fires)} picks)")
            lines.append(f"Definition: TOP 30 BUY + POS/STRONG-POS news + recent 5d pullback (≤-1.79%) + fresh catalyst (lag ≤ 1 day)")
            lines.append(f"In-sample stats: 91% precision, mean +6.34% / 5d. Watch-list candidates for 1.5× sizing IF forward validation holds.")
            lines.append("")
            lines.append("| # | Symbol | Score | 5d_pre | News | Pub_lag | Sector |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in buy_rule_fires.iterrows():
                lines.append(f"| {int(r['rank'])} | **{r['instrument']}** | {r['score']:+.3f} | "
                             f"{r['ret_5d']*100:+.1f}% | {r['news_sentiment']} | "
                             f"{r['pub_lag_days']:.1f}d | {r['sector']} |")
            lines.append("")
        if len(avoid_rule_fires) > 0:
            lines.append(f"### `rule_avoid_high_precision` fires ({len(avoid_rule_fires)} picks)")
            lines.append(f"Definition: BOTTOM 30 AVOID + STRONG-POS news + same-day catalyst after 16:00 UTC + ret_5d_pre > 0")
            lines.append(f"In-sample stats: 90% precision, mean stock return -3.52%. Watch-list candidates for short-side IF forward validation holds.")
            lines.append("")
            lines.append("| # | Symbol | Score | 5d_pre | News | Pub_hour | Sector |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in avoid_rule_fires.iterrows():
                lines.append(f"| {int(r['rank'])} | **{r['instrument']}** | {r['score']:+.3f} | "
                             f"{r['ret_5d']*100:+.1f}% | {r['news_sentiment']} | "
                             f"{int(r['pub_hour_utc']) if pd.notna(r['pub_hour_utc']) else '-'} UTC | {r['sector']} |")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"## 🟢 TOP {topk} — BUY signals (highest expected 5d return)")
    lines.append("")
    lines.append("| # | Symbol | Score | Close | 5d ret | 20d ret | Ann Vol | Sector |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        lines.append(
            f"| {int(r['rank'])} | **{r['instrument']}** | {r['score']:+.3f} | "
            f"${r['close']:.2f} | {r['ret_5d']*100:+.1f}% | {r['ret_20d']*100:+.1f}% | "
            f"{r['ann_vol_20d']*100:.0f}% | {r['sector']} |"
        )
    lines.append("")
    lines.append(f"## 🔴 BOTTOM {botk} — AVOID / SHORT candidates (lowest expected 5d return)")
    lines.append("")
    lines.append("| # | Symbol | Score | Close | 5d ret | 20d ret | Ann Vol | Sector |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in bot.iterrows():
        lines.append(
            f"| {int(r['rank'])} | **{r['instrument']}** | {r['score']:+.3f} | "
            f"${r['close']:.2f} | {r['ret_5d']*100:+.1f}% | {r['ret_20d']*100:+.1f}% | "
            f"{r['ann_vol_20d']*100:.0f}% | {r['sector']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Sector exposure of TOP 30 picks")
    lines.append("")
    sector_counts = top["sector"].value_counts()
    lines.append("| Sector | Count | % of TOP30 |")
    lines.append("|---|---|---|")
    for s, c in sector_counts.items():
        lines.append(f"| {s} | {c} | {c/topk*100:.0f}% |")
    lines.append("")
    lines.append("## Risk notes")
    lines.append("")
    lines.append(f"- Average annualized vol of TOP 30: **{top['ann_vol_20d'].mean()*100:.0f}%**")
    lines.append(f"- Average annualized vol of BOTTOM 30: **{bot['ann_vol_20d'].mean()*100:.0f}%**")
    lines.append(f"- Highest-vol pick: **{top.loc[top['ann_vol_20d'].idxmax(), 'instrument']}** "
                 f"({top['ann_vol_20d'].max()*100:.0f}%)")
    lines.append("")
    lines.append("⚠️ **Reminder:** Run `spy_filter.py` before trading. If SPY < 200-day MA, skip the day.")

    md_path.write_text("\n".join(lines))
    print(f"  Wrote summary: {md_path}")
    return csv_path, md_path


def main():
    if len(sys.argv) > 1:
        target = pd.Timestamp(sys.argv[1]).date()
        qlib.init(provider_uri=PROVIDER, region="us")
    else:
        target = get_latest_trading_date()
        print(f"Using latest data date: {target}")

    print(f"\nGenerating signals for {target}...")
    pred = train_and_predict(target)
    print(f"  Got predictions for {len(pred)} instruments")

    df = attach_metadata(pred, target)
    print(f"  Enriched with sector / close / momentum / vol")

    # Persistence diagnostics + circuit breaker (Codex-proposed)
    diag = compute_persistence_diagnostics(df, target)
    cb = evaluate_circuit_breaker(diag)
    diag.update(cb)

    csv_path, md_path = write_outputs(df, target, diag=diag)

    print(f"\n=== Today's TOP 10 picks ===")
    for _, r in df.head(10).iterrows():
        print(f"  #{int(r['rank']):2d} {r['instrument']:<6} score={r['score']:+.3f}  "
              f"${r['close']:>7.2f}  5d_ret={r['ret_5d']*100:+5.1f}%  vol={r['ann_vol_20d']*100:.0f}%  "
              f"[{r['sector']}]")
    print()
    print(f"View full signal report: {md_path}")


if __name__ == "__main__":
    main()
