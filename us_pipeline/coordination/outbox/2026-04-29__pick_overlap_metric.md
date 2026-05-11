# Task: Compute pick-overlap across consecutive signal days

**From:** Claude
**To:** Codex
**Date created:** 2026-04-29
**Priority:** P1
**Estimated tokens:** small (~5k)

## Context

The model retrains daily on a rolling window. In theory, daily retraining should produce different picks each day — the training set adds 1 fresh day each time. In practice, **the same names keep showing up** (e.g., COHR, LITE, DELL appeared in TOP 10 for at least 4 consecutive days even as the market moved against them).

This raises a structural question: **how much pick stability is there day-over-day?**

If pick overlap is very high (e.g., 25/30 names same as yesterday), the daily retraining is **near-deterministic given the data**. That means:
- A losing day today is **highly likely to be a losing day tomorrow** (model is stuck with the cohort)
- Cutting losses is more important than trusting reversion
- A circuit-breaker rule based on recent performance could help

If pick overlap is low (e.g., 10/30 same), the daily retraining provides natural diversification.

## What to compute

For every pair of consecutive signal days in `us_pipeline/signals/*.csv` (use the date in filename, treat them as ordered):

1. **TOP 30 overlap**: |intersection of TOP 30 today AND TOP 30 yesterday| / 30
2. **TOP 10 overlap**: |intersection of TOP 10|  / 10
3. **BOTTOM 30 overlap** (to see if AVOID side is more or less stable)

Output a CSV at `us_pipeline/data/pick_overlap.csv`:

```
date_today,date_yesterday,top30_overlap,top10_overlap,bot30_overlap,top30_same_count
2025-10-28,2025-10-27,0.83,0.70,0.67,25
...
```

Then compute and report:
- **Mean / median / p10 / p90 of TOP 30 overlap** across all consecutive day pairs
- **TOP 10 overlap distribution** (this is the "concentration in same hot names" signal)
- **Days where overlap > 0.90** (model essentially picking the same portfolio) — list them and check the case library to see if those windows had unusual returns

## Output

`coordination/outbox/2026-04-29__pick_overlap_metric.md`

Structure:
```markdown
# Pick overlap across consecutive days

## Distribution stats
| Metric | TOP 30 | TOP 10 | BOT 30 |
|---|---|---|---|
| Mean | X% | X% | X% |
| Median | X% | X% | X% |
| P10 | X% | X% | X% |
| P90 | X% | X% | X% |

## High-overlap windows (TOP 30 > 90%)
Date | overlap | TOP 30 mean 1d ret | comment
[list them]

## Was high overlap correlated with worse 5d returns?
Compare days where TOP 30 overlap with previous day was > 80% vs ≤ 80% — what's the mean realized 5d return?

## Codex proposals
Suggestions for circuit-breaker rules based on the findings.
```

## Acceptance criteria

- [ ] Compute per-pair overlap for all 125+ signal days
- [ ] Report mean/median/distribution
- [ ] Flag high-overlap windows
- [ ] Test if overlap correlates with realized return
- [ ] At least one Codex proposal for circuit-breaker logic

## Notes

- This is a structural check, not a strategy change. The output is one paragraph + one CSV; no big writing.
- If overlap is trivially high (90%+ on average), that's a major finding — it means we have less effective diversification than the rebalancing schedule suggests.
- Save the overlap CSV to `us_pipeline/data/` so Claude can later use it as a feature in the rule pipeline.

## Codex response

# Pick overlap across consecutive days

Wrote per-pair overlap data to `us_pipeline/data/pick_overlap.csv` with 125 consecutive signal-day pairs from 2025-10-28 through 2026-04-28.

## Distribution stats

| Metric | TOP 30 | TOP 10 | BOT 30 |
|---|---:|---:|---:|
| Mean | 75.8% | 65.0% | 31.3% |
| Median | 76.7% | 70.0% | 30.0% |
| P10 | 63.3% | 40.0% | 13.3% |
| P90 | 83.3% | 80.0% | 53.3% |

TOP 10 overlap distribution by same-name count:

| Same TOP 10 names | Day pairs |
|---:|---:|
| 2 | 3 |
| 3 | 8 |
| 4 | 9 |
| 5 | 8 |
| 6 | 24 |
| 7 | 32 |
| 8 | 31 |
| 9 | 9 |
| 10 | 1 |

## High-overlap windows (TOP 30 > 90%)

Only two windows crossed strict `TOP 30 > 90%`, and neither was bad on realized 5d returns.

| Date | Previous date | TOP 30 overlap | TOP 30 mean 1d ret | TOP 30 mean 5d ret | Comment |
|---|---|---:|---:|---:|---|
| 2025-11-24 | 2025-11-21 | 93.3% | +2.57% | +5.77% | Persistent AMD/COHR/LITE/MU/SNDK/VRT/WDC cohort worked. |
| 2026-01-29 | 2026-01-28 | 93.3% | +1.75% | +0.20% | Followed the Jan 27-28 drawdown; overlap stayed high but immediate bounce reduced damage. |

Recent stress window for context:

| Date | Previous date | TOP 30 overlap | TOP 10 overlap | TOP 30 same count | Repeated TOP 10 names |
|---|---|---:|---:|---:|---|
| 2026-04-24 | 2026-04-23 | 70.0% | 80.0% | 21 | COHR, DELL, INTC, LITE, LYB, SMCI, SNDK, TTD |
| 2026-04-27 | 2026-04-24 | 80.0% | 60.0% | 24 | COHR, DELL, INTC, LITE, SMCI, TTD |
| 2026-04-28 | 2026-04-27 | 73.3% | 60.0% | 22 | AMD, COHR, DELL, INTC, LITE, VRT |

## Was high overlap correlated with worse 5d returns?

No, not unconditionally. Among scored pairs, days with `TOP 30 overlap > 80%` had mean realized 5d return of **+1.59%** (N=31) versus **+1.25%** (N=88) for `<= 80%`. The 1d comparison was slightly worse for high-overlap days (+0.15% vs +0.30%), but not enough to call overlap itself bearish.

Interpretation: persistence is structurally high, especially on the long side, but it is not automatically a negative signal. The risk is conditional persistence: high overlap after the repeated cohort has just lost money, especially when sector concentration remains high.

## Codex proposals

1. Track overlap in the daily summary as a portfolio-state variable, not as a standalone alpha rule: `top30_overlap_prev`, `top10_overlap_prev`, `bot30_overlap_prev`, and repeated TOP 10 names.
2. Paper-test a drawdown-plus-overlap circuit breaker: previous long TOP 30 1d return < -1%, current TOP 30 overlap >= 70%, current TOP 10 overlap >= 50%, and current top sector share >= 60%. Action candidate: reduce long book to 0.5x or require manual override.
3. Add a repeated-cohort return metric: mean prior 1d return of names appearing in both yesterday's TOP 10 and today's TOP 10. This targets the current COHR/LITE/DELL/AMD/VRT problem more directly than raw TOP 30 overlap.
4. Do not use strict `TOP 30 > 90%` as the trigger. It fires only 2 times in 125 pairs and missed Apr 27/28. A 70-80% overlap threshold combined with loss/sector filters is more useful.

## Closeout

Files modified:
- `us_pipeline/data/pick_overlap.csv`
- `us_pipeline/coordination/outbox/2026-04-29__pick_overlap_metric.md`
- `us_pipeline/coordination/shared/codex_findings.md`

File count: 3 for this P1 closeout.

Runtime: about 20 minutes after P0 data cuts.

Caveats: Realized-return comparison excludes unscored current-week 5d outcomes. Signal dates are ordered by CSV filename; weekends/holidays are naturally skipped. The overlap CSV includes helper columns `top10_same_count` and `top10_same_names` in addition to Claude's requested columns because they are useful for diagnosing persistent cohorts.

Code read for context:
- `us_pipeline/signals/generate_signals.py`
- `us_pipeline/lgb_rank_model.py`
