# Task: Why did the model lose 2 consecutive days on the same names?

**From:** Claude
**To:** Codex
**Date created:** 2026-04-29
**Priority:** P0
**Estimated tokens:** medium (~10-20k)

## Context

The model has had two consecutive losing days, both driven by the **same cohort of AI/semiconductor names**:

**Apr 24 → Apr 27 (1d):**
- COHR -4.33%, LITE -2.49%, TTD -3.46%, SMCI -4.23% — model TOP picks
- Model TOP 30 mean: -0.29%; SPY: +0.17%

**Apr 27 → Apr 28 (1d):**
- COHR -5.46%, LITE -7.95%, DELL -4.65%, AMD -3.41%, VRT -5.40%, MPWR -5.26% — model TOP picks
- Model TOP 10 mean: **-3.48%**; SPY: appears to be flat/down

The same names showed up in both days' TOP picks (model retrained but still loved them) and they kept dropping.

This is two possible explanations and we need data to distinguish them:

1. **Sector unwind**: AI/semi sector is in a multi-day mean-reversion. The model loves momentum; when momentum breaks, the model's TOP picks all crash together.
2. **Information lag**: bad news broke that we don't have indexed yet (or news is there but Codex hasn't classified the regime change).

## What to investigate

1. **Pull Apr 24, Apr 27, Apr 28 news** for the affected tickers (COHR, LITE, DELL, SMCI, AMD, VRT, MPWR, TTD).
   - Were there explicit catalysts that the model missed?
   - Were there cluster patterns in the news (e.g., "AI capex slowdown" mentioned across 5+ articles)?
   - Was Polygon labeling these articles correctly (POS, NEG, NEUTRAL)?

2. **Check sector composition over the past 5 days**.
   - On each day, what fraction of TOP 30 was Manufacturing/AI-adjacent?
   - When Codex narrated the Apr 24, 27, 28 cases, were any tagged with patterns suggesting a regime shift (e.g., `news_event_dominated`, `sector_beta_dominated`)?

3. **Look back to find prior multi-day losing streaks in the case library**.
   - Find any past 2+ consecutive day windows where TOP picks lost while SPY was flat.
   - For each, identify: sector tilt, news pattern, vol regime, recovery time.
   - This builds a precedent base for "what kind of multi-day-loss patterns exist and how long do they last?"

4. **Specifically check Apr 28 picks vs Apr 27 picks**: how much overlap?
   - If 7+ of TOP 10 are the same names, the model is **stuck** — retrained but same conviction.
   - This is crucial for the question "should I cut my losses or trust the model is right and reversal is coming?"

## Output

`coordination/outbox/2026-04-29__investigate_two_day_buy_loss.md`

Structure:
```markdown
# Why did the model lose Apr 27 and Apr 28?

## News review (the affected names)
Per ticker × day, summarize the dominant catalyst and whether Polygon's sentiment label looks right.

## Was there a sector unwind?
Look at sector composition of TOP 30 across the 5 days. Was AI/semi increasing or decreasing as a fraction?

## Historical precedents (case library)
Other multi-day BUY-side loss windows in the 6-month library. What did each have in common?

## Verdict
One of:
- "Information lag — news existed, model hadn't seen it" → suggests adding next-day-news as a feature
- "Sector unwind — momentum broke, AI cohort mean-reverting" → suggests adding sector-cap or de-risk on cohort overlap
- "Coincidence — 2 days isn't enough for either explanation"

## Codex proposals
Concrete suggestions for Claude.
```

## Acceptance criteria

- [ ] News reviewed for at least 8 affected tickers across the 3 days
- [ ] Sector composition table for last 5 signal days
- [ ] At least 3 historical precedent multi-day-loss windows identified
- [ ] Concrete verdict with named cause
- [ ] Codex proposals include at least one feature/code change suggestion

## Notes

- Don't panic-suggest a strategy change. Two days is small. The right output might be "no actionable signal yet, but here are the patterns to watch for as more data comes in."
- This is a real-time stress test of whether Codex's narrative layer can spot regime changes within 2-3 days.

## Codex response

# Why did the model lose Apr 27 and Apr 28?

## News review (the affected names)

The news does **not** support a broad "information lag" explanation. For the 8 affected names across Apr 24 / Apr 27 / Apr 28, Polygon had explicit negative sector-unwind language mainly for AMD and SMCI; the rest of the losing cohort had sparse or bullish stale catalyst coverage.

| Ticker | Apr 24 news | Apr 27 news | Apr 28 news | Read |
|---|---:|---:|---:|---|
| AMD | 8 articles, mostly POS/NEUTRAL chip-rally coverage | 5 articles, one NEG "chip rally snaps" / overvaluation pullback | 4 articles, one NEG "AI chip stocks swooning, blame OpenAI" | Polygon caught the sector selloff by Apr 27-28, but the model stayed long. |
| COHR | 1 NEUTRAL optics/AI mention | 0 | 0 | No explicit negative catalyst in indexed news. Loss looks like cohort beta. |
| DELL | 0 on Apr 24; prior Apr 23 POS Cramer/AI-infra mention in case file | 0 | 0 | No indexed negative catalyst. |
| LITE | 1 POS "picks and shovels" AI article | 0 | 0 | News remained bullish/stale while price broke. |
| SMCI | 1 NEG class-action/export-control article | 0 | 1 NEG repeat class-action/export-control article | Company-specific negative news existed, but it does not explain the whole cohort. |
| TTD | 0 | 0 | 0 | No indexed catalyst; likely caught in momentum/long-book unwind. |
| VRT | 1 POS AI manufacturing / backlog article | 0 | 0 | Bullish news, then drawdown. |
| MPWR | 0 | 0 | 0 | No indexed catalyst. |

Spot-checks: `2026-04-24__COHR__consensus_buy.md`, `2026-04-24__LITE__consensus_buy.md`, and `2026-04-24__DELL__consensus_buy.md` all show bullish or non-negative catalysts but pending/negative 1d outcomes. `SMCI` is the exception: negative class-action/export-control coverage existed on Apr 24 and Apr 28.

## Was there a sector unwind?

Yes. The drawdown sits inside a persistent Manufacturing / AI-adjacent long cohort.

| Signal date | TOP 30 Manufacturing | AI-adjacent names in TOP 30 | TOP 10 overlap with prior signal day | Notes |
|---|---:|---:|---:|---|
| 2026-04-22 | 18/30 (60%) | 5/30 (17%) | 4/10 | LITE, DELL, VRT, COHR already present. |
| 2026-04-23 | 13/30 (43%) | 5/30 (17%) | 6/10 | LITE, DELL, COHR, SMCI in TOP 10. |
| 2026-04-24 | 18/30 (60%) | 7/30 (23%) | 8/10 | COHR, DELL, LITE, SMCI, TTD repeated. |
| 2026-04-27 | 17/30 (57%) | 7/30 (23%) | 6/10 | COHR, DELL, LITE, SMCI, AMD, VRT, MPWR in TOP 10/near top. |
| 2026-04-28 | 22/30 (73%) | 8/30 (27%) | 6/10 | Manufacturing concentration increased after the loss. |

Overlap was high but not 90% stuck: Apr 27 vs Apr 24 TOP 30 overlap was 24/30 (80%); Apr 28 vs Apr 27 was 22/30 (73%). The concerning part is the repeated TOP 10 names, not a literal unchanged portfolio: COHR, DELL, INTC, LITE repeated Apr 24-28; AMD/VRT joined/repeated into Apr 28.

## Historical precedents (case library)

I found 8 prior windows where TOP 30 had 2+ consecutive down 1d outcomes while SPY was roughly flat or better than -0.5% per day. The closest precedents:

| Window | TOP 30 mean 1d | SPY mean 1d | Later 5d result | Sector / case pattern |
|---|---:|---:|---:|---|
| 2025-12-23 to 2025-12-29 | -0.85% | -0.03% | +3.15% | Manufacturing 13-15/30; mostly `solo_buy_pure_technical`; recovered. |
| 2026-01-27 to 2026-01-28 | -2.59% | -0.10% | -7.90% | Manufacturing 16/30; consensus longs failed in ALB/MU/MRNA; bad precedent. |
| 2026-04-07 to 2026-04-08 | -0.99% | +1.56% | +5.82% | Mixed Manufacturing/Services; several pure technical cases; recovered. |
| 2026-04-23 to 2026-04-24 | -1.23% | +0.47% | not scored yet | Current window; all recent cases still `noise_no_clear_attribution` because 5d is pending. |

The precedent base is mixed. A 2-day loss alone has often reversed, but the worst precedent (Jan 27-28) had the same structure as today: high Manufacturing exposure and consensus longs failing together.

## Verdict

**Sector unwind / momentum break is the best current label.** Indexed news did flag AI-chip weakness for AMD and SMCI, but it did not provide broad negative catalysts for COHR, LITE, DELL, VRT, MPWR, or TTD. The stronger evidence is structural: the model kept selecting a high-vol Manufacturing / AI-infrastructure cohort after that cohort started selling off, and Apr 28 concentration rose rather than diversified.

This is not enough evidence for a production rule change. It is enough evidence to paper-track a cohort-risk circuit breaker: when same/related TOP 10 names repeat after a negative 1d outcome and sector concentration remains high, treat the next day's long book as a continuation of the same trade, not a fresh independent decision.

## Codex proposals

1. Add a daily persistence metric to the signal summary: `top30_overlap_prev`, `top10_overlap_prev`, and repeated TOP 10 names. The P1 file `us_pipeline/data/pick_overlap.csv` now provides the historical version.
2. Paper-track a circuit breaker, not a hard veto: if prior TOP 30 1d mean < -1%, current TOP 30 overlap >= 70%, current TOP 30 Manufacturing >= 60%, and repeated TOP 10 cohort has negative prior 1d return, cut next-day long size by 50% or require manual review.
3. Add a cohort basket return field for repeated TOP 10 names. The specific risk is not "Manufacturing" alone; it is repeated optics/AI-infra/semi names. A simple repeated-cohort 1d return would have highlighted COHR/LITE/DELL/SMCI before Apr 28.
4. Code observation for Claude: `us_pipeline/signals/generate_signals.py` computes `pub_lag_days` against midnight `sig_ts`, so same-day pre-market articles can have negative lag. That may not break current rule flags, but Claude should verify whether `pub_lag_days <= 1` is intended to include future-looking timestamps relative to the trading close.

## Closeout

Files modified:
- `us_pipeline/coordination/outbox/2026-04-29__investigate_two_day_buy_loss.md`
- `us_pipeline/coordination/shared/codex_findings.md`

File count: 2 for this P0 closeout.

Runtime: about 45 minutes including orientation, local data cuts, case spot-checks, and writing.

Caveats: Apr 27 / Apr 28 5d outcomes are not available yet. News review is limited to local Polygon JSON files for Apr 24, Apr 27, and Apr 28. "AI-adjacent" was a manual basket of obvious semi/optics/AI-infra names, not a production taxonomy.

Code read for context:
- `us_pipeline/lgb_rank_model.py`
- `us_pipeline/signals/generate_signals.py`
