"""Experiment 4: LambdaRank + (Alpha158 + 13 fundamental factors).

Compares against the previous best (Experiment 3-A: LambdaRank + Alpha158 alone).
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
EXPNAME = "exp4_fundamentals"

TRAIN = ["2021-06-01", "2024-06-30"]
VALID = ["2024-07-01", "2025-03-31"]
TEST = ["2025-04-01", "2026-04-22"]

LABEL_5D = "Ref($close, -6) / Ref($close, -1) - 1"


def handler_cfg_with_fund(label: str, sector_neutral: bool):
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


def make_dataset(handler_cfg, with_fund: bool):
    if with_fund:
        handler = {
            "class": "Alpha158WithFundamentals",
            "module_path": "us_pipeline.handler_alpha158_fund",
            "kwargs": handler_cfg,
        }
    else:
        handler = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": handler_cfg,
        }
    return {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler,
            "segments": {"train": TRAIN, "valid": VALID, "test": TEST},
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


def train(name: str, hcfg: dict, with_fund: bool):
    dataset = init_instance_by_config(make_dataset(hcfg, with_fund))
    model = init_instance_by_config(rank_model())
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

    # Capture feature importance
    importance = None
    try:
        # LGBRankModel stores LightGBM Booster as .model
        if hasattr(model, "model") and model.model is not None:
            try:
                names = model.model.feature_name()
            except Exception:
                names = [f"f{i}" for i in range(len(model.model.feature_importance()))]
            imp = model.model.feature_importance(importance_type="gain")
            importance = pd.Series(imp, index=names).sort_values(ascending=False)
    except Exception:
        pass

    return pred, importance


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
    importances = {}

    # Reference: Experiment 3-A baseline (LambdaRank + Alpha158, sector-neutral)
    print("\n[1/2] reference (LambdaRank + Alpha158, sector-neutral)")
    pred_ref, imp_ref = train("ref_lr_alpha158_secneutral",
                              handler_cfg_with_fund(LABEL_5D, sector_neutral=True),
                              with_fund=False)
    results.append(evaluate(pred_ref, "Ref_LR_Alpha158_SecN", true_1d))
    importances["Ref_LR_Alpha158_SecN"] = imp_ref
    print(f"   IC={results[-1]['IC']:+.4f}")

    # Experiment 4: LambdaRank + Alpha158 + Fundamentals, sector-neutral
    print("\n[2/2] EXP4 (LambdaRank + Alpha158 + Fundamentals, sector-neutral)")
    pred_e4, imp_e4 = train("exp4_lr_alpha158_fund_secneutral",
                             handler_cfg_with_fund(LABEL_5D, sector_neutral=True),
                             with_fund=True)
    results.append(evaluate(pred_e4, "EXP4_LR_Alpha158+Fund_SecN", true_1d))
    importances["EXP4_LR_Alpha158+Fund_SecN"] = imp_e4
    print(f"   IC={results[-1]['IC']:+.4f}")

    # Print results table
    print("\n\n" + "="*145)
    print("EXPERIMENT 4 — LambdaRank + Fundamentals  (test 2025-04-01 to 2026-04-22)")
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

    # Print fundamental factor importance ranking from EXP4
    if importances.get("EXP4_LR_Alpha158+Fund_SecN") is not None:
        imp = importances["EXP4_LR_Alpha158+Fund_SecN"]
        # Show ALL features ranked by importance, then highlight fundamentals
        fund_imp = imp[imp.index.str.startswith("fund_")] if hasattr(imp.index, 'str') else None
        if fund_imp is None:
            # The model uses positional indices; we lost feature names. Skip.
            print("\n(feature importance not available — model used positional indices)")
        else:
            print("\nTop 20 most important features (EXP4):")
            print(imp.head(20).to_string())
            print("\nFundamental factors ranked by importance (EXP4):")
            print(fund_imp.sort_values(ascending=False).to_string())
            print(f"\nTotal importance share captured by fundamentals: "
                  f"{fund_imp.sum() / imp.sum() * 100:.1f}%")

    # Persist
    pd.DataFrame(results).to_csv(Path(__file__).parent / "data" / "exp4_results.csv", index=False)
    print(f"\nSaved results to {Path(__file__).parent / 'data' / 'exp4_results.csv'}")


if __name__ == "__main__":
    main()
