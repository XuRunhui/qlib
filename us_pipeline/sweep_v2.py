"""Stacked sweep B->C->A->D, each experiment building on the prior best.

Baseline:  5d horizon, top30/drop5, MSE+CSRankNorm, long-only
B+:        + sector-neutral rank label
C+:        B + long-short evaluation (top30 - bottom30 portfolio)
A+:        C + LambdaRank loss (replace MSE)
D+:        A + multi-horizon ensemble (1d/5d/10d) of B-style sector-neutral models

Outputs IC + portfolio metrics for each.
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

PROVIDER = "us_pipeline/data/qlib_bin"
EXPNAME = "sweep_v2"

TRAIN = ["2021-06-01", "2024-06-30"]
VALID = ["2024-07-01", "2025-03-31"]
TEST = ["2025-04-01", "2026-04-22"]

LABEL_5D = "Ref($close, -6) / Ref($close, -1) - 1"
LABEL_1D = "Ref($close, -2) / Ref($close, -1) - 1"
LABEL_10D = "Ref($close, -11) / Ref($close, -1) - 1"


def base_handler_cfg(label: str, sector_neutral: bool = False) -> dict:
    cfg = {
        "start_time": "2021-06-01",
        "end_time": "2026-04-22",
        "fit_start_time": "2021-06-01",
        "fit_end_time": "2024-06-30",
        "instruments": "sp500",
        "infer_processors": [
            {"class": "RobustZScoreNorm",
             "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "label": [label],
    }
    if sector_neutral:
        cfg["learn_processors"] = [
            {"class": "DropnaLabel"},
            {
                "class": "SectorNeutralRank",
                "module_path": "us_pipeline.sector_processor",
                "kwargs": {
                    "fields_group": "label",
                    "sector_csv": "us_pipeline/data/instruments/sectors.csv",
                },
            },
        ]
    else:
        cfg["learn_processors"] = [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]
    return cfg


def lgb_mse_kwargs() -> dict:
    return {
        "loss": "mse",
        "colsample_bytree": 0.8,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "lambda_l1": 50.0,
        "lambda_l2": 100.0,
        "max_depth": 6,
        "num_leaves": 64,
        "num_threads": 8,
        "early_stopping_rounds": 50,
        "num_boost_round": 1000,
    }


def make_dataset(handler_cfg: dict) -> dict:
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_cfg,
            },
            "segments": {"train": TRAIN, "valid": VALID, "test": TEST},
        },
    }


def make_model() -> dict:
    return {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": lgb_mse_kwargs(),
    }


def port_long_only(topk: int = 30, n_drop: int = 5, benchmark: str = "RSP") -> dict:
    return {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {"signal": "<PRED>", "topk": topk, "n_drop": n_drop},
        },
        "backtest": {
            "start_time": TEST[0],
            "end_time": TEST[1],
            "account": 100000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "limit_threshold": None,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 1,
            },
        },
    }


def train_and_get_pred(name: str, handler_cfg: dict, model_cfg: dict | None = None):
    """Train one model, return (recorder_id, pred_dataframe)."""
    if model_cfg is None:
        model_cfg = make_model()
    dataset = init_instance_by_config(make_dataset(handler_cfg))
    model = init_instance_by_config(model_cfg)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with R.start(experiment_name=EXPNAME, recorder_name=name, resume=False):
            model.fit(dataset)
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
            sa = SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252)
            sa.generate()
            pred = recorder.load_object("pred.pkl")
            rid = recorder.id
    return rid, pred


def run_backtest_from_pred(name: str, pred: pd.DataFrame, port_cfg: dict, parent_rid: str | None = None) -> dict:
    """Re-create a recorder, attach pred, and run backtest."""
    # We need to wrap the pred as a "model" stub and a "dataset" stub for PortAnaRecord
    # Easier path: just compute portfolio metrics manually from pred.
    return compute_long_short_metrics(pred, name)


def compute_long_short_metrics(pred: pd.DataFrame, name: str, topk: int = 30) -> dict:
    """Compute IC + long, short, and long-short portfolio returns from prediction."""
    # pred has index (datetime, instrument), single column 'score'
    if isinstance(pred, pd.DataFrame):
        score = pred.iloc[:, 0]
    else:
        score = pred
    score.name = "score"

    # Need next-day actual returns to compute IC and portfolio P&L
    from qlib.data import D
    instruments = score.index.get_level_values("instrument").unique().tolist()
    dates = score.index.get_level_values("datetime").unique()
    start, end = dates.min(), dates.max()
    # Fetch close, compute t+1 -> t+6 (5d) actual return as the "ground truth" for portfolio
    # but for portfolio we just need t -> t+1 daily return
    feats = D.features(instruments,
                       ["Ref($close, -2)/Ref($close, -1) - 1"],
                       start_time=start, end_time=end, freq="day")
    feats.columns = ["next_ret"]

    df = score.to_frame().join(feats, how="left").dropna()

    # IC per day
    ics = df.groupby(level="datetime").apply(
        lambda x: x["score"].corr(x["next_ret"])
    )
    rank_ics = df.groupby(level="datetime").apply(
        lambda x: x["score"].rank().corr(x["next_ret"].rank())
    )

    # Long top-k and Short bottom-k portfolios per day
    def select(group, k=topk):
        s = group["score"]
        long = group.loc[s.nlargest(k).index, "next_ret"].mean()
        short = group.loc[s.nsmallest(k).index, "next_ret"].mean()
        return pd.Series({"long_ret": long, "short_ret": short})

    portfolio = df.groupby(level="datetime").apply(select)
    portfolio["long_short"] = portfolio["long_ret"] - portfolio["short_ret"]
    portfolio["market"] = df.groupby(level="datetime")["next_ret"].mean()
    portfolio["long_excess"] = portfolio["long_ret"] - portfolio["market"]

    # Apply trading cost: assume top30 with ~1/6 turnover/day for long-only,
    # double for long-short. One-way cost = 0.001 (10 bps).
    cost_long = 0.001 * (5 / topk)  # long-only with drop=5 turnover
    cost_long_short = cost_long * 2

    n_days = len(portfolio)
    ann = 252
    long_ann_no_cost = portfolio["long_excess"].mean() * ann
    long_ann_w_cost = (portfolio["long_excess"].mean() - cost_long) * ann
    long_short_ann_no_cost = portfolio["long_short"].mean() * ann
    long_short_ann_w_cost = (portfolio["long_short"].mean() - cost_long_short) * ann

    long_ir_w_cost = (portfolio["long_excess"].mean() - cost_long) / portfolio["long_excess"].std() * np.sqrt(ann)
    long_short_ir_w_cost = (portfolio["long_short"].mean() - cost_long_short) / portfolio["long_short"].std() * np.sqrt(ann)

    # Max drawdown of long_short cumulative
    cum_ls = (1 + portfolio["long_short"] - cost_long_short).cumprod()
    dd_ls = (cum_ls / cum_ls.cummax() - 1).min()
    cum_long = (1 + portfolio["long_excess"] - cost_long).cumprod()
    dd_long = (cum_long / cum_long.cummax() - 1).min()

    return {
        "name": name,
        "IC": ics.mean(),
        "ICIR": ics.mean() / ics.std(),
        "RankIC": rank_ics.mean(),
        "RankICIR": rank_ics.mean() / rank_ics.std(),
        "long_ann_no_cost": long_ann_no_cost,
        "long_ann_w_cost": long_ann_w_cost,
        "long_ir_w_cost": long_ir_w_cost,
        "long_max_dd_w_cost": dd_long,
        "ls_ann_no_cost": long_short_ann_no_cost,
        "ls_ann_w_cost": long_short_ann_w_cost,
        "ls_ir_w_cost": long_short_ir_w_cost,
        "ls_max_dd_w_cost": dd_ls,
        "n_test_days": n_days,
    }


def main():
    qlib.init(provider_uri=PROVIDER, region="us")

    results = []

    # ============================================================
    # Baseline: 5d horizon, MSE + CSRankNorm, long-only top30/drop5 vs RSP
    # ============================================================
    print("\n" + "="*80)
    print("BASELINE: 5d, MSE+CSRankNorm")
    print("="*80)
    rid, pred = train_and_get_pred("baseline", base_handler_cfg(LABEL_5D, sector_neutral=False))
    r = compute_long_short_metrics(pred, "baseline")
    results.append(r)
    print(f"  IC={r['IC']:+.4f}  long_ann_w_cost={r['long_ann_w_cost']:+.4f}  ls_ann_w_cost={r['ls_ann_w_cost']:+.4f}")

    # ============================================================
    # B: + Sector-neutral rank label
    # ============================================================
    print("\n" + "="*80)
    print("B: 5d, MSE + Sector-Neutral Rank")
    print("="*80)
    rid_B, pred_B = train_and_get_pred("B_sector_neutral", base_handler_cfg(LABEL_5D, sector_neutral=True))
    r = compute_long_short_metrics(pred_B, "B_sector_neutral")
    results.append(r)
    print(f"  IC={r['IC']:+.4f}  long_ann_w_cost={r['long_ann_w_cost']:+.4f}  ls_ann_w_cost={r['ls_ann_w_cost']:+.4f}")

    # ============================================================
    # C: B + Long-Short eval (already computed above; duplicated for table clarity)
    # ============================================================
    # The long-short metric is already computed in compute_long_short_metrics for every model.
    # So C "stack" = same model B but emphasize long-short result. We just relabel.
    print("\n" + "="*80)
    print("C: Same as B, but report long-short portfolio")
    print("="*80)
    r_C = dict(r)
    r_C["name"] = "C_long_short_of_B"
    results.append(r_C)
    print(f"  long_short_ann_w_cost = {r_C['ls_ann_w_cost']:+.4f}, IR = {r_C['ls_ir_w_cost']:+.3f}")

    # ============================================================
    # A: B + LambdaRank loss
    # ============================================================
    print("\n" + "="*80)
    print("A: 5d, LambdaRank + Sector-Neutral Rank")
    print("="*80)
    model_lr_cfg = {
        "class": "LGBRankModel",
        "module_path": "us_pipeline.lgb_rank_model",
        "kwargs": {
            "n_bins": 16,
            "learning_rate": 0.02,
            "num_leaves": 64,
            "max_depth": 6,
            "lambda_l1": 50.0,
            "lambda_l2": 100.0,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "num_threads": 8,
            "early_stopping_rounds": 50,
            "num_boost_round": 1000,
        },
    }
    try:
        rid_A, pred_A = train_and_get_pred(
            "A_lambdarank", base_handler_cfg(LABEL_5D, sector_neutral=True),
            model_cfg=model_lr_cfg,
        )
        # Convert pred_A from Series to DataFrame for compute_long_short_metrics
        if isinstance(pred_A, pd.Series):
            pred_A = pred_A.to_frame("score")
        r = compute_long_short_metrics(pred_A, "A_lambdarank")
        results.append(r)
        print(f"  IC={r['IC']:+.4f}  long_ann_w_cost={r['long_ann_w_cost']:+.4f}  ls_ann_w_cost={r['ls_ann_w_cost']:+.4f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  A FAILED: {e}")

    # ============================================================
    # D: Multi-horizon ensemble of sector-neutral models
    # ============================================================
    print("\n" + "="*80)
    print("D: Ensemble of {1d, 5d, 10d} Sector-Neutral models")
    print("="*80)

    print("  Training 1d model...")
    _, pred_1d = train_and_get_pred("D_1d", base_handler_cfg(LABEL_1D, sector_neutral=True))
    print("  Training 10d model...")
    _, pred_10d = train_and_get_pred("D_10d", base_handler_cfg(LABEL_10D, sector_neutral=True))

    # Use B (5d) prediction we already have
    p1 = pred_1d.iloc[:, 0].rename("p1")
    p5 = pred_B.iloc[:, 0].rename("p5")
    p10 = pred_10d.iloc[:, 0].rename("p10")

    # Rank within each day before averaging (so models are on same scale)
    def rank_per_day(s):
        return s.groupby(level="datetime").rank(pct=True)

    p1_r = rank_per_day(p1)
    p5_r = rank_per_day(p5)
    p10_r = rank_per_day(p10)

    ensemble = (0.2 * p1_r + 0.5 * p5_r + 0.3 * p10_r).to_frame("score")
    r = compute_long_short_metrics(ensemble, "D_ensemble_1d_5d_10d")
    results.append(r)
    print(f"  IC={r['IC']:+.4f}  long_ann_w_cost={r['long_ann_w_cost']:+.4f}  ls_ann_w_cost={r['ls_ann_w_cost']:+.4f}")

    # ============================================================
    # FINAL TABLE
    # ============================================================
    print("\n\n" + "="*140)
    print("STACKED IMPROVEMENTS — All metrics on test set 2025-04-01 to 2026-04-22")
    print("="*140)
    header = (f"{'name':<28} {'IC':>8} {'RankIC':>8} {'ICIR':>7} "
              f"| {'LongExNoCost':>12} {'LongExCost':>11} {'LongIR':>7} {'LongDD':>8} "
              f"| {'LSnoCost':>10} {'LSwCost':>9} {'LS_IR':>7} {'LS_DD':>8}")
    print(header)
    print("-"*140)
    for r in results:
        print(f"{r['name']:<28} {r['IC']:>+8.4f} {r['RankIC']:>+8.4f} {r['ICIR']:>+7.3f} "
              f"| {r['long_ann_no_cost']:>+12.4f} {r['long_ann_w_cost']:>+11.4f} "
              f"{r['long_ir_w_cost']:>+7.3f} {r['long_max_dd_w_cost']:>+8.4f} "
              f"| {r['ls_ann_no_cost']:>+10.4f} {r['ls_ann_w_cost']:>+9.4f} "
              f"{r['ls_ir_w_cost']:>+7.3f} {r['ls_max_dd_w_cost']:>+8.4f}")
    print("="*140)


if __name__ == "__main__":
    main()
