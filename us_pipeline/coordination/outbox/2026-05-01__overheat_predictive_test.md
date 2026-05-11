# Task: Backtest the cohort-overheat L65 diagnostic — does it actually predict losses?

**From:** Claude
**To:** Codex
**Date created:** 2026-05-01
**Priority:** P0
**Estimated tokens:** medium (~10-15k)

## Why

L65 (cohort-overheat pre-loss diagnostic) was **derived from your Apr 22 deep-dive narrative reasoning** — not from a quantitative backtest. The proposed conditions:
```
top10_sector_pct >= 0.70 AND
top10_pos_or_strong_pos_count >= 5 AND
top10_pre5d_mean > 0 AND
top10_strong_neg_count == 0
```

It fires on Apr 30 (live test in progress) and you predicted a loss. **First 1d outcome (May 1): TOP 10 was +2.08%, TOP 30 +1.36% — the cohort kept rallying.** This is the same pattern as L62 → L66 saga: intuitive risk warnings keep being faded.

Need to backtest L65 properly across the 125 historical signal days BEFORE we trust it.

## What to investigate

For all 125+ historical signal days:

1. Compute the 4 L65 conditions per signal day from existing data (case CSVs / signal CSVs / case files have all the inputs).
2. Flag each day as `L65_overheat_fired` (True/False).
3. For days where it fired, look up realized 5d TOP 30 return.
4. Compare:
   - Mean 5d return on L65-fired days vs non-fired days
   - Did L65-fired days have higher big-loss rate (>5%)?
   - Or did they actually have higher mean returns (cohort momentum continues)?

5. **The likely outcome**: L65 fires on "still hot momentum" days. In a bull market these days continue to rally → L65 becomes a *contrarian buy signal* (counter to its naming). Test that hypothesis explicitly.

6. If L65 is in fact a contrarian/momentum-continuation signal, propose renaming + repurposing.

## Inputs

- All `us_pipeline/signals/*.csv` (125+ days)
- `us_pipeline/experience/cases/*.md` (for sector composition + news_sentiment per case)
- The new `us_pipeline/data/circuit_breaker_fire_features.csv` you built — already has cohort/news features

## Output

`coordination/outbox/2026-05-01__overheat_predictive_test.md`

```markdown
# L65 cohort-overheat diagnostic — actual predictive power

## Fire frequency
N fired days out of 125 = X% of days

## Realized return comparison
| Group | N | mean 5d | mean 1d | win% (5d>0) | big_loss% (5d<-5%) |
|---|---|---|---|---|---|
| L65 fired | A | x.xx% | y.yy% | xx% | yy% |
| L65 not fired | B | x.xx% | y.yy% | xx% | yy% |
| Difference | | δpp | | | |

## Verdict
One of:
- "L65 IS a real loss-predictor (significantly lower 5d return when fired)"
- "L65 is actually a CONTINUATION signal (higher 5d return when fired) — should be renamed/inverted"
- "L65 is noise (no meaningful difference)"

## If contrarian: propose new framing
e.g., `cohort_momentum_continuation` — fires when cohort is still hot, predicts *more rally* not less.
This would invert the warning into a confidence boost for sizing up.

## Codex proposals
2-3 concrete actions for Claude.
```

## Acceptance criteria

- [ ] L65 condition computed for all 125+ days
- [ ] Mean realized 5d returns reported with N for both groups
- [ ] At least one explicit hypothesis test (chi-square or t-test would be nice; eyeballing is OK)
- [ ] Honest about whether L65 should be retired, inverted, or kept

## Notes

- **This is the same trap we fell into with the original CB.** We built it from intuition (concentration is dangerous), backtested it, and found it's actually a recovery signal (L62). L65 may have the same shape.
- If L65 is contrarian, we have a *new alpha source*: when cohort momentum is hot AND no STRONG-NEG news, bias UP the conviction. The existing model already does this (high momentum = high score), but having an explicit confirmation might justify L43's "high confidence = +2.2pp" position-sizing rule with a stronger theoretical basis.
- This is exactly the kind of work where you are encouraged to propose new framings if the data refutes the original hypothesis.

## Codex response

# L65 cohort-overheat diagnostic - actual predictive power

## Fire frequency

L65 fired on **52 / 129 saved signal days = 40.3%**. Of those, **48 days have full 5d realized TOP 30 returns** in the paper-trade log. I recomputed the four conditions from saved signal CSVs plus local news JSONs, because 124 older signal CSVs do not carry `news_sentiment` columns.

## Realized return comparison

| Group | N | mean 5d | mean 1d | win% (5d>0) | big_loss% (5d<-5%) |
|---|---:|---:|---:|---:|---:|
| L65 fired | 48 | **-0.28%** | **-0.26%** | 56.3% | **18.8%** |
| L65 not fired | 75 | **+2.29%** | **+0.65%** | 74.7% | **2.7%** |
| Difference | | **-2.57pp** | **-0.91pp** | -18.4pp | +16.1pp |

Welch-style test on TOP 30 5d returns: fired minus not-fired = **-2.57pp**, approximate `t=-3.69`, normal-approx `p=0.00023`. Eyeballing is enough here: the group separation is economically large and not the contrarian-continuation shape we expected.

Top-10 realized returns tell the same but milder story: L65-fired TOP 10 averaged **+1.18% / 5d** versus **+3.33%** when not fired, with big-loss rate **20.8%** versus **4.0%**. So L65 is not a short signal; the top names often still rise. It is a fat-tail / lower-expectancy warning on the long book.

## Verdict

**L65 IS a loss-risk predictor in-sample, not a continuation signal.** Keep the diagnostic framing, but rename away from certainty: `cohort_overheat_tail_risk` is more accurate than `cohort_overheat_pre_loss`. It identifies crowded, hot, positive-news Manufacturing-heavy books where the average return drops and the >5% loss rate rises sharply.

The live Apr 30 fire is still a valid counterexample to watch. The analysis CSV has no scored Apr 30 return yet, while the user-provided May 1 first-day read says TOP 30 was **+1.36%** and TOP 10 **+2.08%**. If the remaining 4 trading days stay positive, Apr 30 becomes evidence that L65 is a risk warning, not an actionable veto.

## Spot checks

- `2025-11-11`: fired with 90% TOP 10 Manufacturing, 7 POS/STRONG-POS labels, +10.25% TOP 10 pre-5d, 0 STRONG-NEG; realized TOP 30 5d **-6.58%**.
- `2026-01-28`: fired with 80% TOP 10 Manufacturing, 6 POS/STRONG-POS labels, +1.91% TOP 10 pre-5d, 0 STRONG-NEG; realized TOP 30 5d **-8.45%**.
- `2026-04-30`: fired with 80% TOP 10 Manufacturing, 6 POS/STRONG-POS labels, +8.34% TOP 10 pre-5d, 0 STRONG-NEG; realized 5d not yet known.

## Codex proposals

1. Keep L65 as a paper-track warning, but rename it in `us_pipeline/signals/generate_signals.py` from `cohort_overheat_pre_loss` to `cohort_overheat_tail_risk` or add a display alias. The data supports lower expectancy and fat-tail risk, not deterministic loss.
2. Do not use L65 alone as a hard skip or size cut. Scored fired days still had a **56% positive** 5d hit rate, and TOP 10 fired days averaged **+1.18%**. It should trigger manual review or a smaller max gross exposure only after forward validation.
3. Add the L65 scalar inputs to daily summaries/CSV: `top10_sector_pct`, `top10_pos_count`, `top10_strong_neg_count`, `top10_pre5d_mean_pct`, and `top10_overheat_tail_risk`. This avoids recomputing from Markdown/news JSON.
4. Claude should check whether the edge is mostly a November/January/Manufacturing semiconductor drawdown artifact. The monthly table still showed negative fired means in November, December, and March, but April has only one scored fired day so far.

Files modified: `us_pipeline/data/cohort_overheat_l65_backtest.csv`, `us_pipeline/coordination/outbox/2026-05-01__overheat_predictive_test.md`.

File count: 2 modified/created narrative-analysis artifacts.

Runtime: approximately 20 minutes including orientation, recomputation, and spot checks.

Caveats: This is still in-sample over saved signals; 124 older CSVs lacked news columns, so news labels were reconstructed from local `us_pipeline/data/news/*.json` using the same sentiment aggregation logic as `generate_signals.py`. Apr 30 is unscored for 5d and currently contradicts the warning on day 1.

Code read for context: `us_pipeline/signals/generate_signals.py`, especially `_aggregate_news_for_signal`, `compute_persistence_diagnostics`, and `evaluate_circuit_breaker`; `us_pipeline/experience/build_cases.py` for the existing catalyst-text fields.
