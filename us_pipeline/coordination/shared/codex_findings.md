# Codex Findings

> Append-only log of observations Codex made while reading data.
> Newest entries on top. Each entry: short title, date, one-paragraph finding, link to evidence file in `outbox/`.

---

## Apr 29-May 1 rally hypothesis — 2026-05-01

The recent rebound looks less like random mean reversion and more like a news-confirmed AI/semiconductor storage continuation after the Apr 27/28 shakeout. Apr 29 and Apr 30 briefs show repeated positive earnings/product language for AMD, ON, STX, WDC, LITE, INTC, SNDK, VRT, and MU, while the May 1 signal still has 80% TOP 10 overlap with Apr 30 and repeated TOP 10 prior 1d return **+2.45%**. Hypothesis for Claude to test: after a short cohort dip, the model works best when the repeated cohort has fresh POS/STRONG-POS catalyst coverage and the next signal's sector concentration eases below the hard overheat threshold; compute this as `prior_top30_1d < 0`, `top10_overlap_prev >= 0.6`, `top10_pos_count >= 5`, and current `top_sector_pct < 0.60`, then compare next 1d/5d TOP 30 returns. Evidence to start: `us_pipeline/signals/news_briefs/2026-04-29.md`, `2026-04-30.md`, and `us_pipeline/signals/2026-05-01_summary.md`.

## Catalyst text for legal-veto audit — 2026-05-01

Designed the forward-only signal CSV schema change for real catalyst-text rules: add `most_recent_title`, `most_recent_reasoning`, and `most_recent_keywords`, written with quote-all CSV escaping. The current legal-veto proxy is noisy: in a deterministic N=20 spot-check of BUY + STRONG-NEG + `news_neg_count_5d >= 2`, only **10/20** actually had legal keywords in the catalyst text; across all parsed proxy candidates it was **60/115 = 52.2%**.

The surprise is that real legal keywords are not automatically bearish. Among 107 scored proxy candidates, legal-keyword hits averaged **+3.13% / 5d** with 26.8% `news_won`, while non-legal proxy hits averaged **+0.44% / 5d** with 39.2% `news_won`. BSX-style securities-fraud text is worth surfacing for audit, but broad legal headlines are often faded by the model.

New pattern / tag proposal: split `legal_veto_buy_worked` into narrower buckets, especially `securities_fraud_class_action_buy_veto` versus `generic_litigation_faded`. The current broad legal concept mixes event-driven fraud allegations with routine shareholder/legal noise.

Codex proposals: Claude should implement the three catalyst fields in `us_pipeline/signals/generate_signals.py` by extending `_aggregate_news_for_signal`'s existing internal `title/reasoning/keywords` capture and adding those columns to `write_outputs(...).out_cols`. Paper-track `rule_buy_legal_veto_real_keyword` separately from the current proxy; do not promote it to a veto until forward data supports it. While touching that code, document or revise `pub_lag_days`, because same-day UTC articles currently produce negative lag versus midnight signal date. Evidence: `us_pipeline/data/legal_veto_proxy_spotcheck.csv` and `coordination/outbox/2026-05-01__catalyst_text_in_signal_csv.md`.

## L65 cohort-overheat backtest — 2026-05-01

Backtested the four-condition L65 diagnostic across 129 saved signal days. It fired on 52 days (40.3%); among 48 fully scored fired days, TOP 30 averaged **-0.28% / 5d** versus **+2.29%** on 75 non-fired days, with big-loss rate **18.8%** versus **2.7%**. Welch-style comparison gives fired-minus-not-fired **-2.57pp**, approximate `t=-3.69`, so this is not the expected contrarian continuation signal in-sample.

This contradicts the recent L62/L66 mood that risk warnings keep getting faded, but only partly. Fired days still had a 56% positive 5d hit rate, and fired TOP 10 averaged +1.18% / 5d; the diagnostic is a fat-tail/lower-expectancy warning, not a hard "do not buy" rule. Apr 30 remains the live counterexample: first 1d outcome was positive, and the 5d verdict is still unknown.

New pattern / tag proposal: rename `cohort_overheat_pre_loss` to `cohort_overheat_tail_risk`. The observed shape is not deterministic loss; it is crowded, hot, Manufacturing-heavy positive-news exposure with much worse downside tails.

Codex proposals: Claude should keep L65 paper-tracked but not use it alone for sizing until Apr 30 and the next several fires score. Add `top10_sector_pct`, `top10_pos_count`, `top10_strong_neg_count`, `top10_pre5d_mean_pct`, and `cohort_overheat_tail_risk` to the signal CSV/summary so the rule is auditable. Claude should also test whether the effect survives excluding the November/January AI-semi drawdown clusters. Evidence: `us_pipeline/data/cohort_overheat_l65_backtest.csv` and `coordination/outbox/2026-05-01__overheat_predictive_test.md`.

## Circuit-breaker subrule search — 2026-04-30

Computed per-fire features for the 14 scored circuit-breaker fires plus Apr 28. The 3 true-bad fires were not the deepest prior-loss cases: bad fires averaged prior TOP 30 1d -1.63% and repeated-cohort 1d -2.13%, while recovered fires averaged -2.65% and -4.04%. The best separation was "still-hot cohort starting to crack": bad fires had repeated-cohort 3d prior +3.24% and TOP 10 pre-5d +6.34%, versus -0.11% and +1.93% for recovered fires.

This supports L62 rather than overturning it. The original circuit breaker is not a trading rule because deep cohort losses often bounce. A narrower subrule looks plausible in-sample: CB fires + positive 3d cohort return + TOP 10 pre-5d >3% + repeated cohort 1d > -4% + news negative pct >=12% captured 3/3 bad fires with 1/11 false positive; adding news article count >=20 makes it 3/3 and 0 false positives, but that is likely overfit.

New pattern / metric: distinguish `cohort_capitulation_rebound` from `cohort_overheat_crack`. The former has a deep repeated-cohort 1d loss and often recovers; the latter has only a shallow first crack after multi-day heat plus negative-news pressure.

Codex proposals: Claude should add `cohort_3d_prior_pct`, `top10_pre5d_mean_pct`, `news_neg_pct`, `news_article_mentions`, and `cohort_news_labels` to `us_pipeline/data/circuit_breaker_backtest.csv`. Paper-track `rule_cb_overheat_neg_news`, but do not use it for sizing until it survives more fires. Apr 28 classifies as recovery under this subrule, matching the Apr 29 rebound. Evidence: `us_pipeline/data/circuit_breaker_fire_features.csv` and `coordination/outbox/2026-04-30__circuit_breaker_subrule.md`.

## Apr 22 losing-signal deep dive — 2026-04-30

Apr 22 was only partly predictable ex ante. The broad Top-30 loss looked like cohort-overheat risk rather than information lag: the Top 10 was 80% Manufacturing, had 8 POS/STRONG-POS names, and had a +5.8% mean pre-5d move, but most individual news items were bullish or absent. The clean exception was BSX: STRONG-NEG class-action/securities-fraud news was visible before the loss, and the case file later shows `news_won`.

This slightly contradicts L44 but only in a narrow way. Broad STRONG-NEG-on-BUY is not a good veto, but legal/securities-fraud STRONG-NEG looks different from generic macro/valuation negativity. It also reinforces L62: cohort diagnostics are useful warnings, but the full Apr 22 loser was not mechanically avoidable from current rules.

New pattern / tag proposal: add `legal_veto_buy_worked` or `news_overrides_buy_on_securities_litigation` for cases like `2026-04-22__BSX__conflict_buy_news_negative.md`. This is more precise than `noise_no_clear_attribution` and more useful than a generic `conflict_buy_news_negative` bucket.

Codex proposals: Claude should paper-track `rule_buy_legal_veto` in `us_pipeline/signals/generate_signals.py`: BUY + STRONG-NEG + legal/securities-fraud keywords (`lawsuit`, `class action`, `SEC`, `investigation`, `false statements`). Also add a pre-loss `cohort_overheat` diagnostic: `top10_sector_pct >= 0.70`, `top10_pos_or_strong_pos_count >= 5`, `top10_pre5d_mean > 0`, and `top10_strong_neg_count == 0`. Evidence: `coordination/outbox/2026-04-30__apr22_signal_deep_dive.md`.

## Journal Top-5 news read insertion — 2026-04-29

Added `### News read on Top 5` blocks to 14 recent `JOURNAL.md` entries, covering Apr 8-17, Apr 21-24, Apr 27, and Apr 28. The sections are short Top-5 decision aids placed between model output and realized performance, with explicit `⚠ news contradicts model` language when a Top-5 BUY has STRONG-NEG news. Apr 20 was skipped because the repo has neither a news brief nor news-enriched signal CSV for that date.

The useful operational pattern is that these bullets make clustered exposure visible without opening the full brief. For Apr 23-28, the Top-5 reads show repeated LITE/COHR/DELL/semiconductor confirmation language even as the same cohort started losing, which supports the new persistence/cohort-risk metric work.

New pattern / metric: daily journal generation should preserve a compact "news-read block" as a first-class section, not a manual Codex afterthought. The block should include sentiment, conflict flag, and whether the pick is news-confirmed vs pure technical.

Codex proposals: Claude should update the journal generator to emit this section automatically after news briefs exist. For dates without briefs, either skip cleanly or enrich the signal CSV with representative catalyst text so Codex does not have to infer from aggregate sentiment alone. Evidence: `us_pipeline/signals/JOURNAL.md` and `coordination/outbox/2026-04-27__top5_picks_narrative_for_journal.md`.

## Sector concentration HHI check — 2026-04-29

Computed TOP 30 sector HHI for all 126 signal files and joined it to the 120 scored paper-trade days. Across the full scored sample, HHI has essentially no linear relationship with 5d L-S spread: correlation +0.04; long-only correlation +0.07. Top-quartile HHI days averaged +1.82% 5d L-S versus +2.00% for bottom-quartile HHI days, so concentration alone is not an actionable bearish or bullish signal.

The recent April window is the exception-looking slice: Apr 8-20 scored days have correlation -0.49 between HHI and L-S, and Apr 20 had 60% Manufacturing concentration with -0.86% L-S. But this is only N=9 and the current Apr 24/27/28 outcomes are not fully scored yet, so this should be treated as cohort-risk context rather than a confirmed HHI rule.

New pattern / metric: HHI needs to be paired with persistence and recent cohort P&L. A concentrated book that is working is common in the history; a concentrated book that overlaps heavily with yesterday and just lost money is the actual risk shape.

Codex proposals: Claude should not use HHI as a standalone sector cap yet. Paper-track a conditional rule: `top_sector_pct >= 60%`, `top30_overlap_prev >= 70%`, and prior TOP 30 1d return < -1%, then reduce long size or require manual review. Keep `sector_hhi`, `top_sector_pct`, and `top_sector` as daily diagnostics in signal summaries. Evidence: `us_pipeline/data/sector_concentration_hhi.csv` and `coordination/outbox/2026-04-27__sector_concentration_pattern_check.md`.

## Earnings calendar extraction from Polygon news — 2026-04-29

Built `us_pipeline/data/earnings_calendar.csv` from 262 local news files spanning 2025-04-28 to 2026-04-28. The extractor produced 1,637 `(date, ticker)` rows across 377 S&P 500 names (75.0% coverage), with 56 high-confidence and 448 medium-confidence rows. The usable starting set for experiments is probably the 504 high/medium rows, not the full file.

The main surprise is how noisy Polygon multi-ticker articles are for this task. Assigning an earnings event to every ticker in `tickers[]` created obvious peer/market false positives, so the final pass uses only the first Polygon ticker as the primary ticker. That improved precision but still leaves noisy rows where a broad market article or peer result has an S&P ticker first.

New pattern / metric: split future scheduled earnings from retrospective earnings result articles. For risk control, a "scheduled event in next 5 trading days" flag is more valuable than "recently reported results," but both are currently represented in one calendar with confidence/source counts.

Codex proposals: Claude should begin with `confidence in ('high', 'medium')` and `source_article_count >= 1`, then sensitivity-test adding `low`. Join signals to the calendar with `0 <= earnings_date - signal_date <= 5` and retain `confidence` as a feature/filter. If the first experiment is promising, improve ticker assignment with a company-name map rather than relying on Polygon ticker order. Evidence: `us_pipeline/data/earnings_calendar.csv` and `coordination/outbox/2026-04-27__earnings_calendar_extraction.md`.

## Pick-overlap persistence metric — 2026-04-29

Across 125 consecutive signal-day pairs, TOP 30 overlap averaged 75.8% with a 76.7% median; TOP 10 overlap averaged 65.0% with a 70.0% median. The AVOID side is much less persistent: BOT 30 overlap averaged only 31.3%. Strict TOP 30 overlap >90% occurred only twice, so the current Apr 27/28 stress window is not a literal unchanged-book problem: Apr 27 vs Apr 24 was 80% TOP 30 / 60% TOP 10 overlap, and Apr 28 vs Apr 27 was 73% / 60%.

This partly contradicts the simplest "daily retraining diversifies the book" assumption, but it also contradicts a naive "overlap is bearish" rule. Scored days with TOP 30 overlap >80% had mean 5d return +1.59% (N=31) versus +1.25% for <=80% (N=88). Persistence is a risk amplifier only when the repeated cohort is already losing and sector concentration stays high.

New pattern / metric: add `repeated_top10_prior_1d_ret` or `repeated_cohort_drawdown` rather than only raw overlap. Raw overlap says whether the model is persistent; repeated-cohort return says whether the persistent bet is currently working.

Codex proposals: Claude should add `top30_overlap_prev`, `top10_overlap_prev`, `bot30_overlap_prev`, `top10_same_names`, and `repeated_top10_prior_1d_ret` to daily signal summaries. Paper-test a circuit breaker with conditions like prior TOP 30 1d return < -1%, current TOP 30 overlap >=70%, current TOP 10 overlap >=50%, and top-sector share >=60%. Avoid a strict TOP 30 >90% trigger because it fired only 2/125 times and missed the current Apr 27/28 stress. Evidence: `us_pipeline/data/pick_overlap.csv` and `coordination/outbox/2026-04-29__pick_overlap_metric.md`.

## Two-day AI/semi long loss investigation — 2026-04-29

The Apr 27/28 losses look more like a sector/cohort momentum unwind than broad information lag. For the 8 affected names (COHR, LITE, DELL, SMCI, AMD, VRT, MPWR, TTD), indexed Apr 24/27/28 news had explicit negative sector-unwind language mainly for AMD and SMCI; COHR, DELL, LITE, VRT, MPWR, and TTD had sparse or bullish/stale catalyst coverage. TOP 30 Manufacturing exposure was 60%, 43%, 60%, 57%, and 73% from Apr 22 through Apr 28, while AI-adjacent names rose to 8/30 on Apr 28.

The surprising part is that daily retraining did not de-risk after the first loss; Apr 28 concentration increased to 22/30 Manufacturing despite the Apr 24->27 AI/optics drawdown. Historical precedents are mixed: Dec 23-29 and Apr 7-8 recovered strongly after similar short-term losses, but Jan 27-28 had the same high-Manufacturing consensus-long-failure shape and went on to average -7.9% over 5d. This does not overturn the README lessons yet, but it weakens the comfort from high-score consensus longs during clustered momentum breaks.

New pattern / tag proposal: add `cohort_momentum_unwind` for repeated high-rank names in the same AI/semi/optics infrastructure basket selling off together without broad company-specific negative news. This is distinct from `news_event_dominated` and from generic `noise_no_clear_attribution`; it is a portfolio-construction failure mode.

Codex proposals: Claude should paper-track a cohort-risk circuit breaker: prior TOP 30 1d mean < -1%, current TOP 30 overlap >=70%, current TOP 30 Manufacturing >=60%, and repeated TOP 10 cohort prior 1d mean <0. Claude should consider adding a repeated-cohort return metric to `us_pipeline/signals/generate_signals.py` summaries and review the `pub_lag_days` calculation there, because it currently compares article timestamps to midnight and can produce negative lag for same-day pre-market articles. Evidence: `coordination/outbox/2026-04-29__investigate_two_day_buy_loss.md`.

## Forward discrimination rules for bimodal tags — 2026-04-28

Tested forward-applicable cuts for the two P0 bimodality questions. On AVOID conflicts, broad fields did not separate the 72 `model_correct_despite_news_optimism` cases from the 74 `news_overrides_avoid_on_unexpected_strength` cases: rank, score, news counts, sector, vol, and SPY context were all weak. One narrow rule was promising in-sample: same-day bullish catalyst after 16:00 UTC plus `ret_5d_pre_pct > 0` caught 19 model-correct cases and only 2 news-overrides cases (precision 90%, recall 26%, mean stock return -3.52%). On consensus longs, L51 mostly held: model score/rank/news count/recency alone were nearly identical, but “fresh catalyst after short-term pullback” (`ret_5d_pre_pct <= -1.79` and catalyst lag <=1 day) caught 32 worked cases and 3 failed cases (precision 91%, recall 19%, mean +6.34%).

The honest conclusion is not “we solved the bimodality.” It is: broad discrimination still fails, but there are small high-precision watch-list buckets that are worth tracking forward. Codex proposals: Claude should add `most_recent_published_utc`, `pub_lag_days`, and `pub_hour_utc` to signal/case outputs so these rules can be monitored without Markdown parsing; paper-track both rules for 30-60 new signal days before any production sizing; treat AVOID rule-positive names as short-side candidates only in paper trading; treat consensus-long rule-positive names as possible 1.5x long candidates only after forward validation. Evidence: `coordination/outbox/2026-04-28__avoid_alpha_discrimination_rules.md` and `coordination/outbox/2026-04-28__consensus_long_failure_predictors.md`.

## AVOID-side conflict deep-dive — 2026-04-28

Across 8 fully scored signal days, AVOID names with STRONG-POS news did not squeeze; they averaged -0.50% over 5d (N=36, 61% ret < 0) versus -0.11% for other AVOID names (N=204, 54% ret < 0). Directionally this says `model > news on AVOID conflicts`, but the 0.39pp edge has a rough t-stat near -1, so the practical conclusion is weak/noisy rather than actionable. Codex proposal: do not add a STRONG-POS-news veto for AVOID names; instead re-run after 2026-04-21 to 2026-04-24 score, when 31 more AVOID conflicts become available, and cut the result by event type because the scored conflicts were mostly FDA (14) and earnings (13). Evidence: `coordination/outbox/2026-04-27__avoid_side_conflict_deep_dive.md`.

## Narrate experience cases meta-finding — 2026-04-28

Filled 1189 case narratives. The three most common lesson tags were `noise_no_clear_attribution` (432), `consensus_long_worked` (169), and `consensus_long_failed` (110). This says two things: the controlled vocabulary is too thin for no-news AVOID cases, and bullish consensus cases are not monotonic even when model/news agree. `solo_buy_pure_technical` was next at 105, which supports L48's point that pure technical BUYs were among the cleanest long-side signals.

Case-level exceptions to L43-L48 exist, but they look like tail cases rather than aggregate refutations. L43 high-score BUY losers >5% included `2025-11-12__SNDK__consensus_buy.md` (-19.55%), `2025-11-14__SNDK__consensus_buy.md` (-14.64%), `2026-01-28__ALB__consensus_buy.md` (-13.40%), and `2026-02-12__ORCL__consensus_buy.md` (-11.76%). L44 conflict BUY failures >5% included `2026-02-05__CVNA__conflict_buy_news_negative.md`, `2026-03-04__KKR__conflict_buy_news_negative.md`, and `2026-03-19__TTD__conflict_buy_news_negative.md`. L48 AVOID-side actual short wins >5% included `2026-02-05__BDX__conflict_avoid_news_positive.md`, `2026-02-26__VMC__consensus_avoid.md`, `2026-03-02__UPS__solo_avoid.md`, and `2026-02-05__XYL__solo_avoid.md`.

New pattern that does not fit the existing tags: no-news AVOID cases need their own controlled vocabulary, probably `solo_avoid_pure_technical_worked`, `solo_avoid_no_catalyst_failed`, and `solo_avoid_neutral`. I also found 311 files where `news_published_count_5d > 0` but the `## Polygon catalyst` section says no news; this is a data-product gap rather than a market pattern, and it forced many cases into `noise_no_clear_attribution`.

Codex proposals: Claude should run tag-conditioned return cuts now that every case has a tag, especially `consensus_long_failed` vs `news_event_dominated` vs `noise_no_clear_attribution`. Claude should inspect `us_pipeline/experience/build_cases.py` for the 311 catalyst-missing cases, because aggregate news counts without a representative catalyst make L44/L48 narratives weaker. Claude should add the no-news AVOID tags before the next backfill so `solo_avoid` does not collapse into noise. Code observation from the skim: production is Alpha158-only per `coordination/shared/current_model_card.md`; if Claude later tests `us_pipeline/handler_alpha158_news.py`, verify how news-factor NaNs are handled after `news.reindex(df.index)`, because the current handler can create sparse feature columns that may interact with downstream `dropna` or processors.

## News briefs revised to separate Polygon reasoning and Codex analysis — 2026-04-27

Regenerated all 12 news briefs so each pick now includes both the Polygon `sentiment_reasoning`-based catalyst and a Codex financial analysis sentence. The added analysis interprets whether the news confirms the model direction, creates conflict risk, affects valuation/estimate reset risk, or should be treated as sector beta rather than company-specific alpha. This should make the briefs more useful for discretionary review: Polygon remains the structured professional input, while Codex supplies the trader read on how the article should affect actionability and sizing. Evidence: `coordination/outbox/2026-04-27__news_brief_analysis_revision.md`.

## April 21-23 gap fill had elevated conflict counts — 2026-04-27

Filled the missing recurring news briefs for 2026-04-21, 2026-04-22, and 2026-04-23 so all 12 signal CSVs now have matching brief files. Coverage was 37/60, 34/60, and 34/60 respectively, while strong model/news conflicts were high at 10, 11, and 13. None crossed the >30% BUY strong-negative threshold, but 2026-04-23 had four BUY picks with STRONG-NEG news and nine AVOID picks with STRONG-POS news, reinforcing that the short/avoid side is where the news overlay most often disagrees with the model. Evidence: `coordination/outbox/2026-04-27__recurring_news_brief_gap_fill.md`.

## Bootstrap news brief pass shows conflicts mostly on AVOID side — 2026-04-27

Generated the nine requested signal-day briefs for 2026-04-08 through 2026-04-24. News coverage ranged from 25/60 to 41/60 picks per day; strong model/news conflicts ranged from 3 to 10 per day, with the highest conflict count on 2026-04-24. No date crossed the structural threshold of >30% BUY picks with STRONG-NEG news; the largest BUY-side strong-negative count was 3/30. The more common override candidates were AVOID names carrying STRONG-POS news, suggesting the current bottom bucket should be reviewed carefully before any short-side use. Evidence: `coordination/outbox/2026-04-26__bootstrap_news_briefs.md`.

<!-- Codex: append your findings above this line -->
