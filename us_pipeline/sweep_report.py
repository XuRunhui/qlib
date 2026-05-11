"""Re-read all sweep recorders and print a clean comparison table."""
from __future__ import annotations

import qlib
from qlib.workflow import R

qlib.init(provider_uri="us_pipeline/data/qlib_bin", region="us")

# Map recorder_name -> short config description (kept in sync with sweep.py order)
RUN_ORDER = [
    "baseline_1d_top30drop5_SPY",
    "h2d_top30drop5_SPY",
    "h5d_top30drop5_SPY",
    "h10d_top30drop5_SPY",
    "h20d_top30drop5_SPY",
    "h5d_top50drop2_SPY",
    "h5d_top50drop2_RSP",
    "baseline_1d_top30drop5_RSP",
    "h5d_top30drop5_RSP",
    "h5d_top10drop2_SPY",
]


def get_metric(m, key, default=float("nan")):
    return m.get(key, default)


def main() -> None:
    recs = R.list_recorders(experiment_name="sweep")
    by_name = {}
    for rid, rec in recs.items():
        by_name[rec.name] = rec

    rows = []
    for name in RUN_ORDER:
        rec = by_name.get(name)
        if rec is None:
            print(f"SKIP {name}: no recorder found")
            continue
        m = rec.list_metrics()
        rows.append({
            "name": name,
            "IC": get_metric(m, "IC"),
            "RankIC": get_metric(m, "Rank IC"),
            "ICIR": get_metric(m, "ICIR"),
            "RankICIR": get_metric(m, "Rank ICIR"),
            "ann_no_cost": get_metric(m, "1day.excess_return_without_cost.annualized_return"),
            "ann_w_cost": get_metric(m, "1day.excess_return_with_cost.annualized_return"),
            "ir_w_cost": get_metric(m, "1day.excess_return_with_cost.information_ratio"),
            "max_dd_w_cost": get_metric(m, "1day.excess_return_with_cost.max_drawdown"),
            "train_l2": get_metric(m, "l2.train"),
            "valid_l2": get_metric(m, "l2.valid"),
        })

    print()
    print("=" * 130)
    print(f"{'CONFIG':<32} {'IC':>8} {'RankIC':>8} {'ICIR':>7} {'AnnExNoCost':>12} {'AnnExCost':>10} {'IR':>7} {'MaxDD':>8} {'L2train':>8} {'L2valid':>8}")
    print("-" * 130)
    for r in rows:
        print(f"{r['name']:<32} {r['IC']:>+8.4f} {r['RankIC']:>+8.4f} {r['ICIR']:>+7.3f} "
              f"{r['ann_no_cost']:>+12.4f} {r['ann_w_cost']:>+10.4f} {r['ir_w_cost']:>+7.3f} "
              f"{r['max_dd_w_cost']:>+8.4f} {r['train_l2']:>8.4f} {r['valid_l2']:>8.4f}")
    print("=" * 130)


if __name__ == "__main__":
    main()
