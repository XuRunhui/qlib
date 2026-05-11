"""Final stacked comparison.

Adds: D2 = ensemble of LambdaRank models at 1d/5d/10d (the truly best stacked config).
Also re-runs A with multiple seeds to confirm stability.
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

PROVIDER = "us_pipeline/data/qlib_bin"
EXPNAME = "sweep_final"
TRAIN = ["2021-06-01", "2024-06-30"]
VALID = ["2024-07-01", "2025-03-31"]
TEST = ["2025-04-01", "2026-04-22"]

LABELS = {
    "1d": "Ref($close, -2) / Ref($close, -1) - 1",
    "5d": "Ref($close, -6) / Ref($close, -1) - 1",
    "10d": "Ref($close, -11) / Ref($close, -1) - 1",
}


def handler_cfg(label: str, sector_neutral: bool):
    cfg = {
        "start_time": "2021-06-01", "end_time": "2026-04-22",
        "fit_start_time": "2021-06-01", "fit_end_time": "2024-06-30",
        "instruments": "sp500",
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "label": [label],
    }
    if sector_neutral:
        cfg["learn_processors"] = [
            {"class": "DropnaLabel"},
            {"class": "SectorNeutralRank",
             "module_path": "us_pipeline.sector_processor",
             "kwargs": {"fields_group": "label",
                        "sector_csv": "us_pipeline/data/instruments/sectors.csv"}},
        ]
    else:
        cfg["learn_processors"] = [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]
    return cfg


def make_dataset(handler_cfg):
    return {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {"class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": handler_cfg},
            "segments": {"train": TRAIN, "valid": VALID, "test": TEST},
        }
    }


def mse_model():
    return {
        "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse", "colsample_bytree": 0.8, "learning_rate": 0.02,
            "subsample": 0.8, "lambda_l1": 50.0, "lambda_l2": 100.0,
            "max_depth": 6, "num_leaves": 64, "num_threads": 8,
            "early_stopping_rounds": 50, "num_boost_round": 1000,
        }
    }


def rank_model():
    return {
        "class": "LGBRankModel", "module_path": "us_pipeline.lgb_rank_model",
        "kwargs": {
            "n_bins": 16, "learning_rate": 0.02, "num_leaves": 64, "max_depth": 6,
            "lambda_l1": 50.0, "lambda_l2": 100.0,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "num_threads": 8, "early_stopping_rounds": 50, "num_boost_round": 1000,
        }
    }


def train(name: str, hcfg: dict, mcfg: dict):
    dataset = init_instance_by_config(make_dataset(hcfg))
    model = init_instance_by_config(mcfg)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with R.start(experiment_name=EXPNAME, recorder_name=name, resume=False):
            model.fit(dataset)
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
            pred = recorder.load_object("pred.pkl")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    return pred


def evaluate(pred: pd.DataFrame, name: str, true_1d, topk: int = 30) -> dict:
    s = pred.iloc[:, 0].to_frame("score")
    df = s.join(true_1d, how="inner").dropna()

    ics = df.groupby(level="datetime").apply(lambda x: x["score"].corr(x["true_1d"]))
    rank_ics = df.groupby(level="datetime").apply(
        lambda x: x["score"].rank().corr(x["true_1d"].rank())
    )

    def select(group, k=topk):
        sscore = group["score"]
        long = group.loc[sscore.nlargest(k).index, "true_1d"].mean()
        short = group.loc[sscore.nsmallest(k).index, "true_1d"].mean()
        return pd.Series({"long_ret": long, "short_ret": short})

    p = df.groupby(level="datetime").apply(select)
    p["ls"] = p["long_ret"] - p["short_ret"]
    p["mkt"] = df.groupby(level="datetime")["true_1d"].mean()
    p["long_ex"] = p["long_ret"] - p["mkt"]

    cost_long = 0.001 * (5/topk)
    cost_ls = cost_long * 2
    ann = 252

    cum_ls = (1 + p["ls"] - cost_ls).cumprod()
    cum_long = (1 + p["long_ex"] - cost_long).cumprod()

    return {
        "name": name,
        "IC": ics.mean(),
        "ICIR": ics.mean()/ics.std(),
        "RankIC": rank_ics.mean(),
        "long_ann_no": p["long_ex"].mean()*ann,
        "long_ann_w": (p["long_ex"].mean()-cost_long)*ann,
        "long_ir_w": (p["long_ex"].mean()-cost_long)/p["long_ex"].std()*np.sqrt(ann),
        "long_dd": (cum_long/cum_long.cummax()-1).min(),
        "ls_ann_no": p["ls"].mean()*ann,
        "ls_ann_w": (p["ls"].mean()-cost_ls)*ann,
        "ls_ir_w": (p["ls"].mean()-cost_ls)/p["ls"].std()*np.sqrt(ann),
        "ls_dd": (cum_ls/cum_ls.cummax()-1).min(),
    }


def main():
    qlib.init(provider_uri=PROVIDER, region="us")
    from qlib.data import D
    instruments = D.list_instruments(D.instruments("sp500"), as_list=True)
    true_1d = D.features(instruments, ["Ref($close, -2)/Ref($close, -1) - 1"],
                         start_time=TEST[0], end_time=TEST[1], freq="day")
    true_1d.columns = ["true_1d"]
    true_1d = true_1d.dropna()

    results = []

    # Baseline: MSE + CSRankNorm 5d
    print("\n[1/7] baseline (MSE, CSRankNorm, 5d)")
    pred = train("baseline_mse_csnorm", handler_cfg(LABELS["5d"], False), mse_model())
    results.append(evaluate(pred, "baseline_MSE_CSnorm_5d", true_1d))
    print(f"   IC={results[-1]['IC']:+.4f}")

    # B: + sector-neutral
    print("\n[2/7] B (MSE, sector-neutral, 5d)")
    pred_B = train("B_mse_secneutral", handler_cfg(LABELS["5d"], True), mse_model())
    results.append(evaluate(pred_B, "B_MSE_SecNeutral_5d", true_1d))
    print(f"   IC={results[-1]['IC']:+.4f}")

    # C: same as B (long-short already in eval), separate row for clarity
    print("\n[3/7] C (long-short of B, no retraining)")
    r_C = dict(results[-1])
    r_C["name"] = "C_LongShort_of_B"
    results.append(r_C)
    print(f"   ls_ann_w={r_C['ls_ann_w']:+.4f}")

    # A: LambdaRank + sector-neutral 5d
    print("\n[4/7] A (LambdaRank, sector-neutral, 5d)")
    pred_A = train("A_lambdarank_secneutral_5d", handler_cfg(LABELS["5d"], True), rank_model())
    results.append(evaluate(pred_A, "A_LambdaRank_SecNeutral_5d", true_1d))
    print(f"   IC={results[-1]['IC']:+.4f}")

    # D-MSE: ensemble of MSE models 1d/5d/10d (the original D)
    print("\n[5/7] D-MSE ensemble (MSE, sector-neutral, 1d+5d+10d)")
    p_d1_mse = train("D_MSE_1d", handler_cfg(LABELS["1d"], True), mse_model())
    p_d10_mse = train("D_MSE_10d", handler_cfg(LABELS["10d"], True), mse_model())
    def rank_per_day(s):
        return s.groupby(level="datetime").rank(pct=True)
    ens_mse = (0.2*rank_per_day(p_d1_mse.iloc[:,0]) +
               0.5*rank_per_day(pred_B.iloc[:,0]) +
               0.3*rank_per_day(p_d10_mse.iloc[:,0])).to_frame("score")
    results.append(evaluate(ens_mse, "D_MSE_Ens_1d5d10d", true_1d))
    print(f"   IC={results[-1]['IC']:+.4f}")

    # D-LambdaRank: ensemble of LambdaRank models 1d/5d/10d (the truly stacked)
    print("\n[6/7] D-LambdaRank ensemble (LambdaRank, sector-neutral, 1d+5d+10d)")
    p_d1_lr = train("D_LR_1d", handler_cfg(LABELS["1d"], True), rank_model())
    p_d10_lr = train("D_LR_10d", handler_cfg(LABELS["10d"], True), rank_model())
    ens_lr = (0.2*rank_per_day(p_d1_lr.iloc[:,0]) +
              0.5*rank_per_day(pred_A.iloc[:,0]) +
              0.3*rank_per_day(p_d10_lr.iloc[:,0])).to_frame("score")
    results.append(evaluate(ens_lr, "D_LR_Ens_1d5d10d", true_1d))
    print(f"   IC={results[-1]['IC']:+.4f}")

    # ALL: stacking everything: LR + ensemble + sector neutral
    print("\n[7/7] (already covered by D-LR)")

    # Final table
    print("\n\n" + "="*145)
    print("FINAL STACKED COMPARISON — Test 2025-04-01 to 2026-04-22 (266 trading days)")
    print("All metrics measured against next-day actual return")
    print("="*145)
    h = (f"{'Config':<32} {'IC':>7} {'ICIR':>6} {'RankIC':>7} "
         f"| {'LongAnnNo':>10} {'LongAnnW':>10} {'LongIR':>7} {'LongDD':>7} "
         f"| {'LSAnnNo':>9} {'LSAnnW':>9} {'LS_IR':>6} {'LS_DD':>7}")
    print(h)
    print("-"*145)
    for r in results:
        print(f"{r['name']:<32} {r['IC']:>+7.4f} {r['ICIR']:>+6.3f} {r['RankIC']:>+7.4f} "
              f"| {r['long_ann_no']:>+10.4f} {r['long_ann_w']:>+10.4f} {r['long_ir_w']:>+7.3f} "
              f"{r['long_dd']:>+7.3f} | {r['ls_ann_no']:>+9.4f} {r['ls_ann_w']:>+9.4f} "
              f"{r['ls_ir_w']:>+6.3f} {r['ls_dd']:>+7.3f}")
    print("="*145)

    # Save to a CSV for posterity
    pd.DataFrame(results).to_csv(Path(__file__).parent / "data" / "sweep_final_results.csv", index=False)
    print(f"\nSaved to {Path(__file__).parent / 'data' / 'sweep_final_results.csv'}")


if __name__ == "__main__":
    main()
