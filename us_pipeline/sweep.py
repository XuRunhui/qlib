"""Sweep label horizons / strategies / benchmarks and report a comparison table.

Trains a fresh LightGBM model per config, evaluates IC and runs backtest.
All using the data we already have (no API calls needed).
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from copy import deepcopy
from datetime import datetime

import qlib
import yaml
from qlib.config import C
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

warnings.filterwarnings("ignore")

PROVIDER = "us_pipeline/data/qlib_bin"


def base_config():
    """Common config; per-run overrides applied later."""
    return yaml.safe_load(
        """
data_handler_config:
    start_time: 2021-06-01
    end_time: 2026-04-22
    fit_start_time: 2021-06-01
    fit_end_time: 2024-06-30
    instruments: sp500
    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSRankNorm
          kwargs:
              fields_group: label
    label: ["Ref($close, -2) / Ref($close, -1) - 1"]
model:
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
        loss: mse
        colsample_bytree: 0.8
        learning_rate: 0.02
        subsample: 0.8
        lambda_l1: 50.0
        lambda_l2: 100.0
        max_depth: 6
        num_leaves: 64
        num_threads: 8
        early_stopping_rounds: 50
        num_boost_round: 1000
strategy:
    topk: 30
    n_drop: 5
    benchmark: SPY
"""
    )


def make_dataset(cfg):
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": cfg["data_handler_config"],
            },
            "segments": {
                "train": ["2021-06-01", "2024-06-30"],
                "valid": ["2024-07-01", "2025-03-31"],
                "test": ["2025-04-01", "2026-04-22"],
            },
        },
    }


def make_port_cfg(cfg):
    return {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": cfg["strategy"]["topk"],
                "n_drop": cfg["strategy"]["n_drop"],
            },
        },
        "backtest": {
            "start_time": "2025-04-01",
            "end_time": "2026-04-22",
            "account": 100000,
            "benchmark": cfg["strategy"]["benchmark"],
            "exchange_kwargs": {
                "limit_threshold": None,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 1,
            },
        },
    }


def run_one(name, overrides):
    cfg = base_config()
    # Apply overrides via dotted keys
    for k, v in overrides.items():
        path = k.split(".")
        d = cfg
        for p in path[:-1]:
            d = d[p]
        d[path[-1]] = v

    print(f"\n{'='*78}\n[{name}] starting | overrides={overrides}\n{'='*78}")

    # Fresh model + dataset each run
    dataset = init_instance_by_config(make_dataset(cfg))
    model = init_instance_by_config(cfg["model"])

    # Capture stdout from training
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with R.start(experiment_name="sweep", recorder_name=name, resume=False):
            model.fit(dataset)
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
            sa = SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252)
            sa.generate()
            pa = PortAnaRecord(recorder, make_port_cfg(cfg), risk_analysis_freq="day")
            pa.generate()

            # Read back metrics
            ic_metrics = recorder.list_metrics()
            artifacts = recorder.list_artifacts()
    train_log_tail = "\n".join(buf.getvalue().splitlines()[-3:])

    bench = cfg["strategy"]["benchmark"]
    label = cfg["data_handler_config"]["label"][0]
    topk = cfg["strategy"]["topk"]
    ndrop = cfg["strategy"]["n_drop"]

    return {
        "name": name,
        "label": label,
        "topk_drop": f"{topk}/{ndrop}",
        "bench": bench,
        "IC": ic_metrics.get("IC", float("nan")),
        "RankIC": ic_metrics.get("Rank IC", float("nan")),
        "ICIR": ic_metrics.get("ICIR", float("nan")),
        "RankICIR": ic_metrics.get("Rank ICIR", float("nan")),
        "ann_excess_no_cost": ic_metrics.get("excess_return_without_cost.annualized_return", float("nan")),
        "ann_excess_w_cost": ic_metrics.get("excess_return_with_cost.annualized_return", float("nan")),
        "ir_w_cost": ic_metrics.get("excess_return_with_cost.information_ratio", float("nan")),
        "max_dd_w_cost": ic_metrics.get("excess_return_with_cost.max_drawdown", float("nan")),
        "bench_ann": ic_metrics.get("benchmark_return.annualized_return", float("nan")),
        "train_tail": train_log_tail,
    }


def main():
    qlib.init(provider_uri=PROVIDER, region="us")

    # Define sweep configs
    runs = [
        # Baseline (what we already ran)
        ("baseline_1d_top30drop5_SPY", {}),

        # Horizon experiments
        ("h2d_top30drop5_SPY", {
            "data_handler_config.label": ["Ref($close, -3) / Ref($close, -1) - 1"],
        }),
        ("h5d_top30drop5_SPY", {
            "data_handler_config.label": ["Ref($close, -6) / Ref($close, -1) - 1"],
        }),
        ("h10d_top30drop5_SPY", {
            "data_handler_config.label": ["Ref($close, -11) / Ref($close, -1) - 1"],
        }),
        ("h20d_top30drop5_SPY", {
            "data_handler_config.label": ["Ref($close, -21) / Ref($close, -1) - 1"],
        }),

        # Lower turnover at best horizon (will pick after seeing results, but pre-stage some)
        ("h5d_top50drop2_SPY", {
            "data_handler_config.label": ["Ref($close, -6) / Ref($close, -1) - 1"],
            "strategy.topk": 50,
            "strategy.n_drop": 2,
        }),
        ("h5d_top50drop2_RSP", {
            "data_handler_config.label": ["Ref($close, -6) / Ref($close, -1) - 1"],
            "strategy.topk": 50,
            "strategy.n_drop": 2,
            "strategy.benchmark": "RSP",
        }),

        # Compare baseline against equal-weight RSP benchmark
        ("baseline_1d_top30drop5_RSP", {
            "strategy.benchmark": "RSP",
        }),
        ("h5d_top30drop5_RSP", {
            "data_handler_config.label": ["Ref($close, -6) / Ref($close, -1) - 1"],
            "strategy.benchmark": "RSP",
        }),

        # Aggressive: top 10 stocks
        ("h5d_top10drop2_SPY", {
            "data_handler_config.label": ["Ref($close, -6) / Ref($close, -1) - 1"],
            "strategy.topk": 10,
            "strategy.n_drop": 2,
        }),
    ]

    results = []
    for name, overrides in runs:
        try:
            r = run_one(name, overrides)
            results.append(r)
            print(f"[{name}] IC={r['IC']:+.4f}  RankIC={r['RankIC']:+.4f}  "
                  f"AnnExc(cost)={r['ann_excess_w_cost']:+.4f}  IR={r['ir_w_cost']:+.3f}  "
                  f"vs {r['bench']} ({r['bench_ann']:+.3f})")
        except Exception as e:
            print(f"[{name}] FAILED: {e}")

    # Print summary table
    print("\n\n" + "=" * 130)
    print("SWEEP SUMMARY")
    print("=" * 130)
    header = (f"{'name':<32} {'label':<40} {'k/d':<7} {'bench':<5} "
              f"{'IC':>8} {'RankIC':>8} {'ICIR':>7} {'AnnExC':>8} {'IR':>7} {'MaxDD':>7} {'Bench':>8}")
    print(header)
    print("-" * 130)
    for r in results:
        # Truncate label
        lab = r["label"]
        if len(lab) > 38:
            lab = lab[:36] + ".."
        print(f"{r['name']:<32} {lab:<40} {r['topk_drop']:<7} {r['bench']:<5} "
              f"{r['IC']:>+8.4f} {r['RankIC']:>+8.4f} {r['ICIR']:>+7.3f} "
              f"{r['ann_excess_w_cost']:>+8.4f} {r['ir_w_cost']:>+7.3f} {r['max_dd_w_cost']:>+7.3f} "
              f"{r['bench_ann']:>+8.4f}")
    print("=" * 130)


if __name__ == "__main__":
    main()
