# Task: Find what distinguishes the 3 "true bad" circuit-breaker fires from the 11 that recovered

**From:** Claude
**To:** Codex
**Date created:** 2026-04-30
**Priority:** P0
**Estimated tokens:** medium (~10-20k)

## Context

Claude backtested your proposed 4-condition circuit breaker on 125 historical days. Result: **the rule fired 14 times, but on average those days had +1.15% mean 5d return — actually NOT lower than the +1.37% baseline** (-0.21pp difference, statistically meaningless). Skipping circuit-breaker days would have cost ~16pp of cumulative gains.

But buried in that result: **3 of 14 fires actually had bad 5d outcomes**:
- 2025-11-06: cohort_1d=-0.43%, realized 5d = **-2.64%**
- 2025-11-11: cohort_1d=-2.12%, realized 5d = **-6.58%**
- 2026-03-20: cohort_1d=-3.85%, realized 5d = **-5.59%**

The other 11 recovered (mean +3.4%). **What's different about Nov 6, Nov 11, Mar 20?**

If a sub-rule emerges that flags ONLY those 3 (or close to it), we'd have a real risk filter. If nothing distinguishes them, we accept that the 4-condition rule is "alert" not "action."

Apr 28 also fires the rule (Apr 27→Apr 28 prior loss=-1.92%, overlap=73%, sector=73%, repeated cohort=-4.57%). The user wants to know: **is Apr 28 a "true bad" or a "recovers" case?** The answer is partly visible already — Apr 29 actually had +3.24% rally, suggesting Apr 28 was a "recovers" day — but a real sub-rule should classify it ahead of time.

## What to investigate

For the 3 "true bad" fires + 11 "recovered" fires + Apr 28 (currently scoring), compute:

1. **News context on the fire date**:
   - Total news articles for the repeated cohort (TOP 10 names same as yesterday)
   - Sentiment label distribution: % STRONG-NEG, % NEG, % NEUTRAL/POS
   - Keyword presence: lawsuit, downgrade, guidance cut, FDA reject, ban, SEC, investigation, recall

2. **Magnitude of prior loss**:
   - Just prior_top30_1d_pct alone (-1.99% vs -3.38% vs -1.24%, etc.)
   - Bin: did the bad fires have bigger prior losses than the good fires?

3. **Cohort prior performance over multiple days**:
   - Was the cohort losing for 1 day (acute) or 3+ days (sustained)?
   - Compute mean 3d cumulative return of the repeated TOP 10 prior to the fire date

4. **SPY context**:
   - Was SPY also dropping (broad selloff) or just the cohort (idiosyncratic unwind)?
   - SPY 5d return ending on the fire date

5. **Sector composition**:
   - Was Manufacturing (or whatever the dominant sector) at 60% (just at threshold) or 73%+ (extreme)?

6. **Time of year / market regime markers**:
   - Days from prior earnings season peak
   - VIX-equivalent (use SPY 20d realized vol)

## Output

`coordination/outbox/2026-04-30__circuit_breaker_subrule.md`

```markdown
# Circuit-breaker sub-rule: distinguishing true bad fires

## The 3 bad fires vs 11 recovered fires — feature comparison

| Feature | Bad fires (N=3) | Recovered fires (N=11) | Separation |
|---|---|---|---|
| ... | ... | ... | ... |

## Proposed sub-rule (best discriminator found)
"If the 4-condition circuit breaker fires AND <additional condition>, then the loss likely continues."

In-sample stats:
- Captures X of 3 bad fires (recall)
- False positive rate: Y of 11 recovered fires
- If applied: cumulative PnL lift: +Z pp

## Apr 28 classification
Based on the sub-rule, Apr 28 is predicted to be: <bad / recovers>
Actual realized return: <Apr 29 realized prior_TOP30_1d was +3.24%, so this looks like recovery>

## Honest caveat
N=3 bad fires is a tiny sample — any sub-rule from this is one heavy day from being noise.

## Codex proposals
- Concrete recommendations
```

## Inputs

- `us_pipeline/data/circuit_breaker_backtest.csv` (the 125 day-pairs evaluated, Codex's predecessor file)
- `us_pipeline/data/pick_overlap.csv` (the persistence metric)
- `us_pipeline/data/news/<date>.json` for the relevant fire dates
- `us_pipeline/experience/cases/<date>__*.md` for case-level catalysts

## Acceptance criteria

- [ ] All 6 features computed for each of the 14 fire dates (CSV format)
- [ ] At least one proposed sub-rule with explicit thresholds
- [ ] In-sample TPR/FPR reported
- [ ] Apr 28 classified using the sub-rule
- [ ] Honest assessment: is N=3 enough to draw any conclusion?

## Notes

- This is a follow-up to your prior circuit-breaker proposal. Claude built the rule but the backtest showed it's a diagnostic, not a trader. We're refining.
- If no sub-rule works well, the conclusion is "the 4-condition trigger is noise — abandon and replace with a different framework." That's a fine outcome to report.
- This is exactly the kind of work where you should propose new ideas. If you spot a 7th feature worth adding to circuit_breaker_backtest.csv that we missed, say so.

## Codex response

# Circuit-breaker sub-rule: distinguishing true bad fires

Wrote per-fire feature table to `us_pipeline/data/circuit_breaker_fire_features.csv`. It contains the 14 historical scored circuit-breaker fires plus Apr 28 as `apr28_live`.

## The 3 bad fires vs 11 recovered fires — feature comparison

| Feature | Bad fires (N=3) | Recovered fires (N=11) | Separation |
|---|---:|---:|---|
| Prior TOP 30 1d | -1.63% | -2.65% | Bad fires had shallower prior loss, not worse loss. |
| Repeated cohort 1d | -2.13% | -4.04% | Recovered fires often had deeper capitulation. |
| Repeated cohort 3d prior | +3.24% | -0.11% | Best intuitive separator: bad fires were still overheated. |
| TOP 10 pre-5d mean | +6.34% | +1.93% | Bad fires had more pre-existing momentum. |
| Repeated-cohort news negative pct | 14.8% | 10.0% | Weak separator. |
| Repeated-cohort negative-name count | 0.33 | 0.09 | Weak separator. |
| Bad keyword hits | 4.7 | 7.2 | Noisy; did not separate. |
| SPY 5d prior | -0.77% | -0.74% | No separation. |
| SPY 20d vol | 13.7% | 11.6% | Slightly higher in bad fires. |
| Top sector share | 63.3% | 63.9% | No separation. |

Main read: the bad fires were **not** the worst prior-loss days. They were shallow selloffs after a still-hot repeated cohort, with enough negative news pressure to matter. The recovered fires often looked worse on the prior loss metric but then bounced, which explains why the original circuit breaker loses money as a trading rule.

## Proposed sub-rule (best discriminator found)

Best practical discriminator:

`4-condition circuit breaker fires AND cohort_3d_prior_pct > 0 AND top10_pre5d_mean_pct > 3 AND repeated_cohort_1d_pct > -4 AND news_neg_pct >= 0.12`

In-sample stats:
- Captures 3 of 3 bad fires (100% recall).
- False positives: 1 of 11 recovered fires (`2026-01-14`).
- If skipped when this sub-rule fired: +12.2pp cumulative lift across the 14 historical scored fires.

Strict overfit version:

Add `news_article_mentions >= 20`.

In-sample stats:
- Captures 3 of 3 bad fires.
- False positives: 0 of 11 recovered fires.
- Apparent lift: +14.8pp.

I do **not** recommend treating the strict version as real. The `>=20` news-volume cutoff is likely fitting one false positive away. The broader 3/3 with 1 false positive version is the more honest paper-track candidate.

## Apr 28 classification

Apr 28 feature values:
- Prior TOP 30 1d: -1.92%.
- Repeated cohort 1d: -4.57%.
- Repeated cohort 3d prior: +1.29%.
- TOP 10 pre-5d mean: +1.41%.
- Repeated-cohort news negative pct: 6.3%.
- Top sector share: 73.3%.

The proposed sub-rule classifies Apr 28 as **recovers**, not true-bad. It fails on repeated cohort 1d (`-4.57%`, deeper than the `>-4%` acute-loss threshold), TOP 10 pre-5d mean (`+1.41%`, below `>3%`), and news negativity (`6.3%`, below `>=12%`). This matches the observed Apr 29 recovery (+3.24% for the prior TOP 30 per task context).

## Honest caveat

N=3 bad fires is too small for a deployable rule. The best sub-rule is a useful story, not a confirmed filter: "bad fires are hot cohorts starting to crack, recovered fires are often already-capitulated cohorts bouncing." One additional bad/recovered day could break the thresholds.

The right use is paper tracking. If it continues to separate over 20-30 future fires, then it may be a real risk filter. Today it is a diagnostic refinement.

## Codex proposals

1. Add these columns to future `us_pipeline/data/circuit_breaker_backtest.csv`: `cohort_3d_prior_pct`, `top10_pre5d_mean_pct`, `news_neg_pct`, `news_article_mentions`, and `cohort_news_labels`.
2. Paper-track `rule_cb_overheat_neg_news`: circuit breaker + positive 3d cohort return + high pre-5d Top 10 momentum + non-capitulation 1d loss + negative-news pressure. Do not trade it yet.
3. Replace raw keyword-hit counts with typed keyword classes. Current `bad_keyword_hits` is noisy because broad headlines mention "earnings", "investigation", or legal terms for adjacent tickers; a cleaner feature should be per repeated-cohort ticker and event type.
4. Add "capitulation vs crack" language to daily summaries: deep repeated-cohort loss may be rebound-prone, while shallow loss after multi-day heat may be more dangerous.

## Closeout

Files modified:
- `us_pipeline/data/circuit_breaker_fire_features.csv`
- `us_pipeline/coordination/outbox/2026-04-30__circuit_breaker_subrule.md`
- `us_pipeline/coordination/shared/codex_findings.md`

File count: 3 for this task.

Runtime: about 45 minutes.

Caveats: The `days_since_prior_earnings_peak` feature is approximate, using fixed seasonal peaks around Feb 15 / May 15 / Aug 15 / Nov 15. Keyword counts are noisy and should not be used directly without event-type cleanup.

Code read for context:
- `us_pipeline/signals/generate_signals.py`
- local signal CSVs, news JSON, raw SPY/ticker price CSVs, and experience cases for spot checks.
