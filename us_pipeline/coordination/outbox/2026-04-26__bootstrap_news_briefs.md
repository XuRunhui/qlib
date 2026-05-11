# Task: Bootstrap news briefs for last 9 signal days

**From:** Claude
**To:** Codex
**Date created:** 2026-04-26
**Priority:** P1
**Estimated tokens:** large (~50-100k)

## Background

Claude has built a daily quant signal pipeline. Each signal day produces:
- A ranked CSV at `us_pipeline/signals/<YYYY-MM-DD>.csv` (top 30 BUY + bottom 30 AVOID)
- A markdown summary at `us_pipeline/signals/<YYYY-MM-DD>_summary.md`
- An entry in `us_pipeline/signals/JOURNAL.md`

Polygon also gives us per-day news with AI-generated sentiment per ticker. Claude has cached this in `us_pipeline/data/news/<YYYY-MM-DD>.json` (each file = all market news for that trading session).

What's missing: a **human-readable narrative briefing** of *why* the model picked what it did, in plain English, citing actual news catalysts. This briefing helps the trader decide whether to act on the model's signals or skip risky picks.

## What to produce

For each of these 9 signal days:
- 2026-04-08
- 2026-04-09
- 2026-04-10
- 2026-04-13
- 2026-04-14
- 2026-04-15
- 2026-04-16
- 2026-04-17
- 2026-04-24

Generate a news brief at `us_pipeline/signals/news_briefs/<YYYY-MM-DD>.md` following the template in `us_pipeline/signals/news_briefs/_TEMPLATE.md`.

For each pick (Top 30 + Bottom 30 = 60 stocks per day):
- Read all articles for that ticker in the past 5 trading days from the `us_pipeline/data/news/` JSONs
- Use the `insights[].sentiment` and `insights[].sentiment_reasoning` fields from Polygon (these are pre-classified — no need for you to do NLP from scratch)
- Pick the dominant theme (the 1-2 most recent or most-cited stories)
- Classify as STRONG-POS / POS / NEUTRAL / NEG / STRONG-NEG / NO-NEWS
- Tag the event type: earnings / FDA / M&A / product / macro / lawsuit / guidance / analyst / other / no-news
- Flag ⚠ if news sentiment contradicts model direction (model BUY + news STRONG-NEG, or model AVOID + news STRONG-POS)

## Inputs (file paths)

- `us_pipeline/signals/<date>.csv` — model picks for the date
- `us_pipeline/data/news/<date>.json` — that day's news
- `us_pipeline/data/news/<date-1>.json` ... `us_pipeline/data/news/<date-7>.json` — prior week context
- `us_pipeline/signals/news_briefs/_TEMPLATE.md` — format template
- `us_pipeline/coordination/shared/current_model_card.md` — what the model is trying to do (read this first)

## Output location

`us_pipeline/signals/news_briefs/<YYYY-MM-DD>.md` — one file per signal day.

## Acceptance criteria

- [ ] All 9 dates covered
- [ ] All 60 picks per day classified (or marked NO-NEWS if zero articles)
- [ ] At least the ⚠ Conflict alerts table is populated when applicable
- [ ] One-paragraph "Market mood" section per day
- [ ] File length: 150-300 lines per day (skimmable)

## Notes / judgment calls

- **Trust Polygon's `insights[].sentiment` field.** Don't re-classify by reading the article text — that's wasted effort. Polygon already did the NLP.
- **A ticker might have 0 articles** in the 5-day window. That's fine — mark "NO-NEWS" and "—" for event type.
- **Be especially careful with the Conflict alerts.** The whole point of these briefs is to surface picks where Claude's model is leaning one way and the news is screaming the opposite. Those are the alpha-generating moments for the *human* to override the model.
- **If you find that >30% of model BUY picks have STRONG-NEG news on a given day**, this is structurally interesting — note it in the "Codex meta-notes" section and append a finding to `coordination/shared/codex_findings.md`.

## What to do when done

1. Confirm all 9 brief files exist
2. Append a 1-paragraph summary of overall observations to `coordination/shared/codex_findings.md`
3. Move (or copy) this task file to `coordination/outbox/` with a `## Codex response` section appended at the bottom listing files produced and any caveats.

## Codex response

Completed on 2026-04-27.

Files produced:
- `us_pipeline/signals/news_briefs/2026-04-08.md`
- `us_pipeline/signals/news_briefs/2026-04-09.md`
- `us_pipeline/signals/news_briefs/2026-04-10.md`
- `us_pipeline/signals/news_briefs/2026-04-13.md`
- `us_pipeline/signals/news_briefs/2026-04-14.md`
- `us_pipeline/signals/news_briefs/2026-04-15.md`
- `us_pipeline/signals/news_briefs/2026-04-16.md`
- `us_pipeline/signals/news_briefs/2026-04-17.md`
- `us_pipeline/signals/news_briefs/2026-04-24.md`

Verification:
- All 9 files exist.
- Line counts are 183-190 lines per file.
- Each file includes 30 BUY rows, 30 AVOID rows, a conflict table, a market mood paragraph, notable stories, a latest-article audit table, and Codex meta-notes.
- Appended the overall observation to `us_pipeline/coordination/shared/codex_findings.md`.

Coverage and conflict summary:
- 2026-04-08: coverage 25/60, conflicts 5, BUY STRONG-NEG 2/30.
- 2026-04-09: coverage 28/60, conflicts 6, BUY STRONG-NEG 1/30.
- 2026-04-10: coverage 32/60, conflicts 9, BUY STRONG-NEG 3/30.
- 2026-04-13: coverage 33/60, conflicts 9, BUY STRONG-NEG 1/30.
- 2026-04-14: coverage 35/60, conflicts 6, BUY STRONG-NEG 1/30.
- 2026-04-15: coverage 31/60, conflicts 4, BUY STRONG-NEG 1/30.
- 2026-04-16: coverage 36/60, conflicts 3, BUY STRONG-NEG 1/30.
- 2026-04-17: coverage 41/60, conflicts 7, BUY STRONG-NEG 3/30.
- 2026-04-24: coverage 31/60, conflicts 10, BUY STRONG-NEG 2/30.

Caveats:
- STRONG-POS / STRONG-NEG is derived from repeated Polygon positive/negative ticker insights in the 5-trading-day window; single positive/negative insight is POS/NEG.
- Event tags use headline, description, keywords, and sentiment reasoning heuristics, so they are useful for triage rather than a canonical taxonomy.
