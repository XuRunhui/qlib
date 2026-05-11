"""Experiment 5: Walk-forward sweep across 4 model variants × 12 monthly OOS windows.

Variants (2x2):
  - features:  Alpha158  vs  Alpha158 + Fundamentals
  - loss:      MSE       vs  LambdaRank
All variants use sector-neutral rank label, 5-day forward return target.

For each test month from 2025-05 to 2026-04:
  - Train on [2021-06-01, month_start - 30 days]
  - Validate on [month_start - 30 days, month_start - 1 day]
  - Predict (out-of-sample) on [month_start, month_end]

Metrics per (variant, month):
  - IC mean, IC std (intra-month)
  - Long top-30 PnL (excess vs market)
  - Long-short top30 - bot30 PnL
  - Drawdown within month
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
EXPNAME = "exp5_walkforward"

LABEL_5D = "Ref($close, -6) / Ref($close, -1) - 1"

# OOS test months: predict each month using prior data only
TEST_MONTHS = [
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-10-31"),
    ("2025-11-01", "2025-11-30"),
    ("2025-12-01", "2025-12-31"),
    ("2026-01-01", "2026-01-31"),
    ("2026-02-01", "2026-02-28"),
    ("2026-03-01", "2026-03-31"),
    ("2026-04-01", "2026-04-22"),  # truncated due to data end
]


def handler_cfg(label: str, train_start: str, train_end: str,
                valid_start: str, valid_end: str,
                test_start: str, test_end: str):
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
        "label": [label],
    }


def make_dataset(handler_cfg_dict: dict, with_fund: bool, segments: dict):
    if with_fund:
        handler = {
            "class": "Alpha158WithFundamentals",
            "module_path": "us_pipeline.handler_alpha158_fund",
            "kwargs": handler_cfg_dict,
        }
    else:
        handler = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": handler_cfg_dict,
        }
    return {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {"handler": handler, "segments": segments},
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


VARIANTS = {
    "A_Alpha158_MSE":           {"with_fund": False, "model_fn": mse_model},
    "B_Alpha158+Fund_MSE":      {"with_fund": True,  "model_fn": mse_model},
    "C_Alpha158_LR":            {"with_fund": False, "model_fn": rank_model},
    "D_Alpha158+Fund_LR":       {"with_fund": True,  "model_fn": rank_model},
}


def train_and_predict(variant_name: str, with_fund: bool, model_fn,
                      train_end: str, valid_start: str, valid_end: str,
                      test_start: str, test_end: str, run_name: str):
    train_start = "2021-06-01"
    hcfg = handler_cfg(LABEL_5D, train_start, train_end,
                       valid_start, valid_end, test_start, test_end)
    segments = {
        "train": [train_start, train_end],
        "valid": [valid_start, valid_end],
        "test":  [test_start, test_end],
    }
    dataset = init_instance_by_config(make_dataset(hcfg, with_fund, segments))
    model = init_instance_by_config(model_fn())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with R.start(experiment_name=EXPNAME, recorder_name=run_name, resume=False):
            model.fit(dataset)
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
            pred = recorder.load_object("pred.pkl")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    return pred


def evaluate_month(pred: pd.DataFrame, true_1d: pd.DataFrame, topk: int = 30) -> dict:
    s = pred.iloc[:, 0].to_frame("score")
    df = s.join(true_1d, how="inner").dropna()
    if len(df) == 0:
        return {"IC": np.nan, "IC_std": np.nan, "n_days": 0,
                "long_ann": np.nan, "ls_ann": np.nan, "long_dd": np.nan, "ls_dd": np.nan}

    ics = df.groupby(level="datetime").apply(lambda x: x["score"].corr(x["true_1d"]))

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

    cum_long = (1 + p["long_ex"] - cost_long).cumprod()
    cum_ls = (1 + p["ls"] - cost_ls).cumprod()

    return {
        "IC": ics.mean(),
        "IC_std": ics.std(),
        "n_days": len(p),
        "long_ann": (p["long_ex"].mean() - cost_long) * ann,
        "ls_ann": (p["ls"].mean() - cost_ls) * ann,
        "long_dd": (cum_long / cum_long.cummax() - 1).min(),
        "ls_dd": (cum_ls / cum_ls.cummax() - 1).min(),
        "long_total_ret": cum_long.iloc[-1] - 1,
        "ls_total_ret": cum_ls.iloc[-1] - 1,
    }


def main():
    qlib.init(provider_uri=PROVIDER, region="us")
    from qlib.data import D
    instruments = D.list_instruments(D.instruments("sp500"), as_list=True)
    true_1d = D.features(instruments, ["Ref($close, -2)/Ref($close, -1) - 1"],
                         start_time=TEST_MONTHS[0][0], end_time=TEST_MONTHS[-1][1], freq="day")
    true_1d.columns = ["true_1d"]
    true_1d = true_1d.dropna()

    rows = []
    total_runs = len(VARIANTS) * len(TEST_MONTHS)
    run_idx = 0
    for test_start, test_end in TEST_MONTHS:
        # 30-day validation window before test
        ts = pd.Timestamp(test_start)
        valid_end = (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        valid_start = (ts - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        train_end = (ts - pd.Timedelta(days=31)).strftime("%Y-%m-%d")

        # Filter true_1d to this test month for evaluation
        mask = (true_1d.index.get_level_values("datetime") >= ts) & \
               (true_1d.index.get_level_values("datetime") <= pd.Timestamp(test_end))
        true_1d_month = true_1d[mask]

        for variant_name, vcfg in VARIANTS.items():
            run_idx += 1
            run_name = f"{variant_name}_{test_start}"
            print(f"[{run_idx}/{total_runs}] {run_name} | "
                  f"train_end={train_end} valid={valid_start}..{valid_end}")
            try:
                pred = train_and_predict(
                    variant_name, vcfg["with_fund"], vcfg["model_fn"],
                    train_end, valid_start, valid_end,
                    test_start, test_end, run_name,
                )
                metrics = evaluate_month(pred, true_1d_month)
                rows.append({"variant": variant_name, "month": test_start[:7], **metrics})
                print(f"     IC={metrics['IC']:+.4f} std={metrics['IC_std']:.3f} "
                      f"long_total={metrics['long_total_ret']*100:+.1f}% "
                      f"ls_total={metrics['ls_total_ret']*100:+.1f}%")
            except Exception as e:
                print(f"     FAILED: {e}")
                rows.append({"variant": variant_name, "month": test_start[:7],
                             "IC": np.nan, "IC_std": np.nan, "n_days": 0,
                             "long_ann": np.nan, "ls_ann": np.nan,
                             "long_dd": np.nan, "ls_dd": np.nan,
                             "long_total_ret": np.nan, "ls_total_ret": np.nan,
                             "error": str(e)[:100]})

    df = pd.DataFrame(rows)
    out_csv = Path(__file__).parent / "data" / "exp5_walkforward_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n\nSaved to {out_csv}")

    # Aggregate by variant
    print("\n\n" + "=" * 130)
    print("EXPERIMENT 5 — Walk-Forward Aggregate Stats (12 monthly OOS windows)")
    print("=" * 130)
    summary = df.groupby("variant").agg(
        ic_mean=("IC", "mean"),
        ic_consistency=("IC", lambda s: (s > 0).mean()),
        ic_std_across_months=("IC", "std"),
        avg_intra_month_ic_std=("IC_std", "mean"),
        long_total_avg=("long_total_ret", "mean"),
        long_total_sum=("long_total_ret", lambda s: ((1+s).prod() - 1)),
        ls_total_avg=("ls_total_ret", "mean"),
        ls_total_sum=("ls_total_ret", lambda s: ((1+s).prod() - 1)),
        long_dd_avg=("long_dd", "mean"),
        ls_dd_avg=("ls_dd", "mean"),
    )
    print(summary.round(4).to_string())
    print()

    # Per-month breakdown
    print("=" * 130)
    print("Monthly IC by variant")
    print("=" * 130)
    pivot = df.pivot(index="month", columns="variant", values="IC")
    print(pivot.round(4).to_string())
    print()

    print("=" * 130)
    print("Monthly Long-only excess return (compounded over month, with cost) by variant")
    print("=" * 130)
    pivot2 = df.pivot(index="month", columns="variant", values="long_total_ret") * 100
    print(pivot2.round(2).to_string())
    print()

    print("=" * 130)
    print("Monthly Long-Short total return by variant")
    print("=" * 130)
    pivot3 = df.pivot(index="month", columns="variant", values="ls_total_ret") * 100
    print(pivot3.round(2).to_string())


if __name__ == "__main__":
    main()
