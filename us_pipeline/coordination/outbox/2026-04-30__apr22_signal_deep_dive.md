# Task: Apr 22 deep-dive — what the news was saying before this losing signal

**From:** Claude
**To:** Codex
**Date created:** 2026-04-30
**Priority:** P0
**Estimated tokens:** medium (~10-15k)

## Context — why Apr 22 specifically

The user asked to retrospectively understand the Apr 22 signal: model output, news context BEFORE Apr 22, and how the picks ultimately performed. This is a teaching moment — Apr 22 was the **first signal** of the losing-cohort window we're now studying.

Realized aggregate (computed by Claude):

| Hold | End | TOP 30 | BOT 30 | L-S |
|---|---|---|---|---|
| 1d | Apr 23 | -2.16% | +1.08% | **-3.24%** |
| 5d | Apr 29 | -2.27% | -0.59% | -1.68% |
| 6d | Apr 30 | +0.41% | +0.97% | -0.56% |

SPY 5d (Apr 22→Apr 28): +0.07%. **Apr 22 was a clear loser** — TOP 30 lost almost 3% over 5 days while market was flat.

The TOP 10 was 60% Manufacturing (LITE, INTC, DELL, CCL, SNDK, TPL, VRT, COHR, BSX, WDC) — heavy AI/semi/optics cohort. Looking at news from Apr 17-22:

- **LITE (#1)**: only 2 articles, both NEUTRAL/POS
- **INTC (#2)**: 6 articles, all POSITIVE (+29% by Apr 29 — actually worked!)
- **DELL (#3)**: 1 POSITIVE article
- **VRT (#7)**: AI Demand Drives 30% Revenue Growth — POS
- **COHR (#8)**: bullish silicon-carbide catalysts (POS) — actually fell -13% by Apr 29
- **BSX (#9)**: ⚠ **2 NEGATIVE articles about class-action litigation** — fell -9.9% by Apr 29
- **TPL (#6)**: NO-NEWS, fell -1.96% by Apr 29

**Two interesting observations:**
1. The model picked names where news was *bullish or absent* — yet most fell anyway. This is consistent with the "cohort momentum unwind" pattern we identified for Apr 27/28.
2. BSX is the exception — news flagged a real negative catalyst (litigation), and the stock did fall. Suggests our existing rules might have caught this.

## What to investigate

### Part 1 — Per-pick narrative for Apr 22 TOP 10
For each of the 10 names listed above, write a 2-sentence note covering:
- What the dominant news theme was Apr 17-22 (use the catalyst text in the case files: `2026-04-22__<TICKER>__*.md`)
- Whether the existing case_type and lesson_tag (already filled by your prior pass) holds up given the realized 5d loss
- Whether any of the 4 Codex rules / circuit-breaker conditions would have flagged this pick **before** the loss (could we have known?)

### Part 2 — News pattern that might predict cohort unwind
Cohort unwind hypothesis: when 5+ TOP 10 names are in the SAME sector AND all have non-negative news AND collectively just had a +X% week, the cohort is at risk of reversion.

For each of the 7 historical "consensus_long_failed" big-loss days (Codex's prior list):
- Was the TOP 10 sector concentration ≥ 50%?
- Were there 5+ POSITIVE-news names with no STRONG-NEG warnings?
- Was the prior-5d cohort return > 0?

If yes/yes/yes pattern repeats, that's a cohort-overheat predictor distinct from the cohort-loss predictor (the existing circuit breaker).

### Part 3 — BSX special case
The BSX class-action news on Apr 22 was correctly tagged STRONG-NEG by Polygon. The model still ranked BSX #9 BUY. **Why didn't the existing rule_buy_high_precision flag this as "sentiment STRONG-NEG → don't buy"?** Look at the case file for `2026-04-22__BSX__consensus_buy.md` and tell me:
- Did the rule_buy_high_precision flag fire on this pick?
- Should it have?
- Is there a missing veto rule: "if model BUY but news has lawsuit/class-action keywords, skip"?

## Inputs

- `us_pipeline/signals/2026-04-22.csv` (note: this file was generated before the news columns were added, so it lacks news fields)
- `us_pipeline/experience/cases/2026-04-22__*.md` (these have full structured data + your prior narratives)
- `us_pipeline/data/news/2026-04-{17,21,22}.json` (raw news for context)
- For Part 2: pull the historical big-loss `consensus_long_failed` cases from `coordination/shared/codex_findings.md` (your prior meta-finding listed them: 2025-11-12 SNDK, 2025-11-14 SNDK, 2026-01-28 ALB, 2026-02-12 ORCL, etc.)
- For Part 3: read `us_pipeline/signals/generate_signals.py` lines defining `rule_buy_high_precision`

## Output

`coordination/outbox/2026-04-30__apr22_signal_deep_dive.md`

Sections:
1. **Apr 22 TOP 10 per-pick narrative** (1 paragraph each)
2. **Cohort-overheat pre-loss pattern** (table + verdict)
3. **BSX gap analysis** (specific veto rule recommendation)
4. **Codex proposals** (concrete suggestions for Claude — at least 2)

## Acceptance criteria

- [ ] All 10 Apr 22 TOP 10 names covered
- [ ] Cohort-overheat hypothesis tested on 5+ historical days
- [ ] BSX-specific veto rule proposed
- [ ] Honest about whether Apr 22 was predictable in advance vs only-in-hindsight

## Notes

- Apr 22 is now known to be the START of a losing window — frame your analysis as "what could we have seen on Apr 22 that would have warned us?"
- Don't be afraid to conclude "nothing visible at the time would have helped" if that's what the data says. The point is to learn what's predictable, not to invent predictability.
- For the BSX case specifically, this could become a high-precision veto: if news label is STRONG-NEG AND keywords contain ('lawsuit','class-action','SEC','investigation'), drop the pick from the long book.

## Codex response

# Apr 22 deep-dive — what was visible before the losing signal

Important caveat: the task text gives Claude's aggregate Apr 22 result as TOP 30 -2.27% over 5d and L-S -1.68%. The current local `paper_trade_log/trades.csv` appears refreshed with different per-pick realized values, so I anchor the aggregate verdict to Claude's task numbers and use local case/news artifacts for ex-ante context.

## Apr 22 TOP 10 per-pick narrative

**LITE (#1, Manufacturing, POS news).** News was constructive: the case file cites a Nvidia investment / purchase commitment and sold-out AI optics demand through 2027. The `consensus_buy` label holds ex ante; none of the buy high-precision flags fired because pre-5d return was already +6.0%, not a pullback.

**INTC (#2, Manufacturing, POS/STRONG-POS news).** The news theme was an AI/foundry turnaround: BNP upgrade, margin expansion, agentic-AI data-center demand, and foundry optionality. This was not a warning case; it worked in the local case file and would not have been vetoed by any current rule.

**DELL (#3, Manufacturing, POS news).** News linked DELL to AI/GPU infrastructure demand, but the catalyst was broad AI supply-chain exposure rather than a direct earnings reset. `consensus_buy` holds as an ex-ante tag; no rule fired because the stock had already run +21.1% over 5d.

**CCL (#4, Transportation/Utilities, STRONG-POS news).** The dominant story was a cruise/oil relief setup: lower crude and ceasefire macro improving the cost picture. This was the only Top 10 name that would have fired `rule_buy_high_precision` from the reconstructed fields: TOP 30 + POS/STRONG-POS + pre-5d pullback -6.5% + fresh news.

**SNDK (#5, Manufacturing, POS news).** News was positive but hot: NAND flash supply crunch, products fully allocated, and a large recent price move. This fits the cohort-overheat shape better than a company-specific warning; no current rule fired because pre-5d return was +9.8% and the news was stale-ish by the rule definition.

**TPL (#6, Finance/Real Estate, NO-NEWS).** No matched ticker insight in the Apr 22 brief; this was a pure technical pick outside the semi/AI cluster. No case file was generated, no news rule could fire, and it would not have warned us about the Top-30 cohort problem.

**VRT (#7, Manufacturing, POS news).** News said Vertiv beat Q1 EPS and guided to 29-31% organic growth, even though the headline itself said the stock was sinking. This is a subtle overheat marker: fundamentals looked good but price was already reacting poorly; no current rule fired because the signal was still POS, not negative.

**COHR (#8, Manufacturing, STRONG-POS news).** The brief highlighted FY2026 Q3 earnings timing plus silicon-carbide and InP capability improvements. Ex ante this looked like confirmation, but the later drawdown makes it a `consensus_long_failed`-style candidate; no current rule fired because pre-5d return was +13.7%, not a fresh pullback.

**BSX (#9, Manufacturing, STRONG-NEG lawsuit news).** This was the clean exception. The case file says class-action allegations around EP segment growth sustainability, adverse procedure-volume trends, competitive pressure, and questionable guidance revisions; the stock then fell sharply. The prior `noise_no_clear_attribution` lesson tag no longer holds: this should be retagged as `legal_veto_buy_worked` or `news_overrides_buy_on_securities_litigation`.

**WDC (#10, Manufacturing, STRONG-POS news).** News was bullish but also momentum-heavy: large move after ceasefire claims and NAND/HDD exposure. Like SNDK, it flags cohort heat, not direct bad news; no current rule fired because the news was positive and the stock had already risen +6.6% over 5d.

## Cohort-overheat pre-loss pattern

Test: for 7 historical big `consensus_long_failed` dates, did the TOP 10 have sector concentration >=50%, 5+ POS/STRONG-POS names with no STRONG-NEG warnings, and positive pre-5d TOP 10 return?

| Date | Top sector in TOP 10 | Sector share | POS/STRONG-POS names | STRONG-NEG names | TOP 10 pre-5d mean | Pattern hit? |
|---|---|---:|---:|---:|---:|---|
| 2025-11-12 | Manufacturing | 90% | 8 | 0 | +8.1% | Yes |
| 2026-02-26 | Manufacturing | 80% | 8 | 0 | +9.2% | Yes |
| 2026-03-03 | Services | 40% | 5 | 0 | +6.6% | No: sector not concentrated |
| 2025-12-04 | Manufacturing | 90% | 4 | 0 | +5.5% | No: only 4 positive-news names |
| 2025-11-14 | Manufacturing | 90% | 5 | 0 | -3.9% | No: not pre-overheated |
| 2026-01-28 | Manufacturing | 80% | 6 | 0 | +1.9% | Yes |
| 2026-01-23 | Manufacturing | 80% | 7 | 0 | +5.8% | Yes |
| 2026-04-22 | Manufacturing | 80% | 8 | 1 | +5.8% | Near miss: BSX STRONG-NEG |

Verdict: cohort-overheat is plausible but not clean. It hit 4 of 7 historical failed dates, but Apr 22 itself technically fails the "no STRONG-NEG warnings" condition because BSX was clearly negative. If BSX is handled separately as a legal veto, the remaining Apr 22 cohort looks exactly like overheat: 7+ Manufacturing/AI-adjacent names, mostly positive news, and positive pre-5d momentum.

## BSX gap analysis

`rule_buy_high_precision` did **not** fire on BSX. It should not have fired under its intended design: that rule is a positive-news/pullback boost rule, not a veto. The code requires `news_sentiment in ['STRONG-POS', 'POS']`, `ret_5d <= -1.79%`, and fresh news; BSX had `STRONG-NEG` lawsuit news and only +0.4% pre-5d.

What was missing is a long-side veto:

`model_action == BUY and news_sentiment == STRONG-NEG and event_type/legal keywords in ('lawsuit', 'class action', 'securities fraud', 'SEC', 'investigation', 'false statements')`

This would have caught BSX without touching normal STRONG-NEG conflict-buy cases like macro or generic valuation articles. That distinction matters because README L44 says STRONG-NEG BUYs are profitable on average; the proposed veto is narrower: legal/securities-fraud STRONG-NEG, not all negative news.

## Could Apr 22 have been predicted?

Partly, but not with the current rules. The broad cohort loss was mostly not visible from individual news: 8 of 10 Top 10 names had non-negative or no news, and several were genuine winners. The visible warnings were portfolio-shaped: 60% TOP 30 Manufacturing, 80% TOP 10 Manufacturing, many names with positive AI/semi news after a strong pre-5d run, plus one obvious BSX legal contradiction.

Honest read: a BSX veto was predictable. The full Top-30 loss was only weakly predictable as a cohort-overheat risk, not as a high-confidence skip signal.

## Codex proposals

1. Add a paper-track long veto in `us_pipeline/signals/generate_signals.py`: `rule_buy_legal_veto = BUY and STRONG-NEG and legal/securities-fraud keywords`. Track hit count and 5d outcomes separately from L44's broad STRONG-NEG conflicts.
2. Add a cohort-overheat diagnostic alongside the existing circuit breaker: `top10_sector_pct >= 0.70`, `top10_pos_or_strong_pos_count >= 5`, `top10_pre5d_mean > 0`, and `top10_strong_neg_count == 0`. This is a pre-loss diagnostic, while the current circuit breaker is a post-loss diagnostic.
3. Retag `2026-04-22__BSX__conflict_buy_news_negative.md` from `noise_no_clear_attribution` to a legal-specific tag in the next narrative pass; current tag underuses a clean ex-ante warning.
4. For future deep-dives, store representative catalyst text in signal CSVs. Apr 22's CSV lacks news fields, forcing reconstruction from briefs/cases and making rule audit harder.

## Closeout

Files modified:
- `us_pipeline/coordination/outbox/2026-04-30__apr22_signal_deep_dive.md`
- `us_pipeline/coordination/shared/codex_findings.md`

File count: 2 for this task.

Runtime: about 45 minutes.

Caveats: Several Apr 22 Top 10 names (`TPL`, `VRT`, `COHR`, `WDC`) had no case files, so I used `news_briefs/2026-04-22.md` and the signal CSV for those. Current local `paper_trade_log` differs from the aggregate return values supplied in the task; I used Claude's aggregate values for the main verdict.

Code read for context:
- `us_pipeline/signals/generate_signals.py`
- `us_pipeline/lgb_rank_model.py` was previously read for model context.
