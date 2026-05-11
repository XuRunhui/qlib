"""Experiment 7: LambdaRank + Alpha158 + 9 News factors.

Tests:
  Single-window: same train/valid/test as Exp 3-A and Exp 4 for direct comparison.
  Walk-forward: 12 monthly OOS windows like Exp 5 to test stability.
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
EXPNAME_SW = "exp7_news_singlewindow"
EXPNAME_WF = "exp7_news_walkforward"

LABEL_5D = "Ref($close, -6) / Ref($close, -1) - 1"

TRAIN = ["2021-06-01", "2024-06-30"]
VALID = ["2024-07-01", "2025-03-31"]
TEST = ["2025-04-01", "2026-04-22"]

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
    ("2026-04-01", "2026-04-22"),
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


def make_dataset(handler_cfg_dict: dict, with_news: bool, segments: dict):
    if with_news:
        handler = {
            "class": "Alpha158WithNews",
            "module_path": "us_pipeline.handler_alpha158_news",
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


def train_and_predict(experiment_name: str, run_name: str, with_news: bool,
                      train_start: str, train_end: str,
                      valid_start: str, valid_end: str,
                      test_start: str, test_end: str):
    hcfg = handler_cfg(LABEL_5D, train_start, train_end,
                       valid_start, valid_end, test_start, test_end)
    segments = {
        "train": [train_start, train_end],
        "valid": [valid_start, valid_end],
        "test":  [test_start, test_end],
    }
    dataset = init_instance_by_config(make_dataset(hcfg, with_news, segments))
    model = init_instance_by_config(rank_model())
    buf = io.StringIO()
    importance = None
    with contextlib.redirect_stdout(buf):
        with R.start(experiment_name=experiment_name, recorder_name=run_name, resume=False):
            model.fit(dataset)
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
            pred = recorder.load_object("pred.pkl")
            try:
                if hasattr(model, "model") and model.model is not None:
                    imp = model.model.feature_importance(importance_type="gain")
                    importance = pd.Series(imp).sort_values(ascending=False)
            except Exception:
                pass
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    return pred, importance


def evaluate(pred: pd.DataFrame, name: str, true_1d, topk: int = 30) -> dict:
    s = pred.iloc[:, 0].to_frame("score")
    df = s.join(true_1d, how="inner").dropna()
    if len(df) == 0:
        return {"name": name, "IC": np.nan, "ICIR": np.nan, "RankIC": np.nan,
                "long_ann_no": np.nan, "long_ann_w": np.nan, "long_ir_w": np.nan, "long_dd": np.nan,
                "ls_ann_no": np.nan, "ls_ann_w": np.nan, "ls_ir_w": np.nan, "ls_dd": np.nan,
                "long_total_ret": np.nan, "ls_total_ret": np.nan, "n_days": 0}

    ics = df.groupby(level="datetime").apply(lambda x: x["score"].corr(x["true_1d"]))
    rank_ics = df.groupby(level="datetime").apply(lambda x: x["score"].rank().corr(x["true_1d"].rank()))

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
        "long_total_ret": cum_long.iloc[-1] - 1,
        "ls_total_ret": cum_ls.iloc[-1] - 1,
        "n_days": len(p),
    }


def main():
    qlib.init(provider_uri=PROVIDER, region="us")
    from qlib.data import D
    instruments = D.list_instruments(D.instruments("sp500"), as_list=True)
    true_1d = D.features(instruments, ["Ref($close, -2)/Ref($close, -1) - 1"],
                         start_time=TEST_MONTHS[0][0], end_time=TEST_MONTHS[-1][1], freq="day")
    true_1d.columns = ["true_1d"]
    true_1d = true_1d.dropna()

    # ============================================================
    # PART 1: Single-window (compare to Exp 3-A and Exp 4)
    # ============================================================
    print("\n" + "="*80)
    print("EXP 7 — PART 1: Single-window comparison")
    print("="*80)

    print("\n[1/2] Reference: LambdaRank + Alpha158 (Exp 3-A baseline)")
    pred_ref, imp_ref = train_and_predict(EXPNAME_SW, "ref_alpha158", False,
                                           TRAIN[0], TRAIN[1], VALID[0], VALID[1],
                                           TEST[0], TEST[1])
    sw_results = []
    sw_results.append(evaluate(pred_ref, "Ref_LR_Alpha158", true_1d))
    print(f"   IC={sw_results[-1]['IC']:+.4f}  long_ann_w={sw_results[-1]['long_ann_w']:+.4f}")

    print("\n[2/2] EXP7: LambdaRank + Alpha158 + 9 News")
    pred_n, imp_n = train_and_predict(EXPNAME_SW, "exp7_alpha158_news", True,
                                       TRAIN[0], TRAIN[1], VALID[0], VALID[1],
                                       TEST[0], TEST[1])
    sw_results.append(evaluate(pred_n, "EXP7_LR_Alpha158+News", true_1d))
    print(f"   IC={sw_results[-1]['IC']:+.4f}  long_ann_w={sw_results[-1]['long_ann_w']:+.4f}")

    # Print single-window table
    print("\n" + "="*145)
    print("EXP 7 SINGLE-WINDOW RESULTS (test 2025-04-01 to 2026-04-22)")
    print("="*145)
    h = (f"{'Config':<28} {'IC':>7} {'ICIR':>6} {'RankIC':>7} "
         f"| {'LongAnnNo':>10} {'LongAnnW':>10} {'LongIR':>7} {'LongDD':>7} "
         f"| {'LSAnnNo':>9} {'LSAnnW':>9} {'LS_IR':>6} {'LS_DD':>7}")
    print(h)
    print("-"*145)
    for r in sw_results:
        print(f"{r['name']:<28} {r['IC']:>+7.4f} {r['ICIR']:>+6.3f} {r['RankIC']:>+7.4f} "
              f"| {r['long_ann_no']:>+10.4f} {r['long_ann_w']:>+10.4f} {r['long_ir_w']:>+7.3f} "
              f"{r['long_dd']:>+7.3f} | {r['ls_ann_no']:>+9.4f} {r['ls_ann_w']:>+9.4f} "
              f"{r['ls_ir_w']:>+6.3f} {r['ls_dd']:>+7.3f}")
    print("="*145)

    # Map important features in EXP7 — feature index 158-166 are the 9 news features
    if imp_n is not None:
        imp_n.index.name = "feature_id"
        # Top features
        print("\nTop 30 most important features (EXP7 — News added):")
        FEATURE_NAMES = ["news_count_1d", "news_count_5d", "news_sent_1d", "news_sent_5d",
                         "news_sent_change", "news_pos_ratio_5d", "news_neg_ratio_5d",
                         "news_attention_z", "news_silence_dummy"]
        def label_feat(idx):
            i = int(str(idx).replace("Column_", "").replace("Feat_", "").replace("f", ""))
            if 158 <= i <= 166:
                return f"news_feat_{i-158}: {FEATURE_NAMES[i-158]}"
            return f"alpha158_feat_{i}"

        # imp_n has index like 'Column_158' or just integer; handle both
        imp_n_clean = imp_n.copy()
        imp_n_clean.index = [label_feat(i) for i in imp_n_clean.index]
        print(imp_n_clean.head(30).to_string())

        # Just the news features
        news_imp = imp_n_clean[imp_n_clean.index.str.startswith("news_feat_")]
        if len(news_imp):
            print(f"\nTotal importance share captured by 9 news features: "
                  f"{news_imp.sum() / imp_n_clean.sum() * 100:.1f}%")
            print("\nNews features ranked by gain importance:")
            print(news_imp.to_string())

    # ============================================================
    # PART 2: Walk-forward (12 monthly OOS, like Exp 5)
    # ============================================================
    print("\n\n" + "="*80)
    print("EXP 7 — PART 2: Walk-forward validation")
    print("="*80)

    wf_rows = []
    for test_start, test_end in TEST_MONTHS:
        ts = pd.Timestamp(test_start)
        valid_end = (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        valid_start = (ts - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        train_end = (ts - pd.Timedelta(days=31)).strftime("%Y-%m-%d")
        mask = (true_1d.index.get_level_values("datetime") >= ts) & \
               (true_1d.index.get_level_values("datetime") <= pd.Timestamp(test_end))
        true_1d_month = true_1d[mask]

        for variant_name, with_news in [("Ref_LR_Alpha158", False),
                                        ("EXP7_LR_Alpha158+News", True)]:
            run_name = f"{variant_name}_{test_start}"
            print(f"  {run_name} | train_end={train_end}")
            try:
                pred, _ = train_and_predict(EXPNAME_WF, run_name, with_news,
                                            "2021-06-01", train_end,
                                            valid_start, valid_end,
                                            test_start, test_end)
                m = evaluate(pred, run_name, true_1d_month)
                wf_rows.append({"variant": variant_name, "month": test_start[:7], **m})
                print(f"    IC={m['IC']:+.4f} long_total={m['long_total_ret']*100:+.1f}% "
                      f"ls_total={m['ls_total_ret']*100:+.1f}%")
            except Exception as e:
                print(f"    FAILED: {e}")

    wf_df = pd.DataFrame(wf_rows)
    out_csv = Path(__file__).parent / "data" / "exp7_walkforward_results.csv"
    wf_df.to_csv(out_csv, index=False)

    # Summary
    print("\n\n" + "="*130)
    print("EXP 7 WALK-FORWARD AGGREGATE")
    print("="*130)
    summary = wf_df.groupby("variant").agg(
        ic_mean=("IC", "mean"),
        ic_consistency=("IC", lambda s: (s > 0).mean()),
        long_total_avg=("long_total_ret", "mean"),
        long_total_compound=("long_total_ret", lambda s: ((1+s).prod() - 1)),
        ls_total_compound=("ls_total_ret", lambda s: ((1+s).prod() - 1)),
        long_dd_avg=("long_dd", "mean"),
        ls_dd_avg=("ls_dd", "mean"),
    )
    print(summary.round(4).to_string())

    print("\n\nMonthly IC by variant:")
    pivot = wf_df.pivot(index="month", columns="variant", values="IC")
    print(pivot.round(4).to_string())

    print("\n\nMonthly Long-only excess (compounded over month, with cost):")
    pivot2 = wf_df.pivot(index="month", columns="variant", values="long_total_ret") * 100
    print(pivot2.round(2).to_string())


if __name__ == "__main__":
    main()
