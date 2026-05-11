# Task: Auto-generate Top 5 narrative for daily journal

**From:** Claude
**To:** Codex
**Date created:** 2026-04-27
**Priority:** P2
**Estimated tokens:** small per day (~2k); medium total (~25k for 12 days)

## Background

The user explicitly mentioned this in the project conversation as a useful addition: "在每天的 journal 里自动列出'今天 Top 5 picks 的新闻摘要'" (in each day's journal, auto-list a news summary for today's Top 5 picks).

Currently:
- `JOURNAL.md` has structured per-day entries with model output (top 5 names, scores)
- `news_briefs/<date>.md` has full per-pick news context

The trader looking at the journal in the morning needs a **2-minute decision aid** — not the full brief, just a short narrative telling them what the model is leaning into today and whether the news supports or contradicts.

## What to produce

For each of the 12 historical signal days, append a new section to that day's `JOURNAL.md` entry called **`### News read on Top 5`** that contains 5 short bullets:

```
### News read on Top 5
- **COHR** (#1, score +0.39): Polygon: optical/AI infrastructure tailwinds confirmed by 3 articles. Codex: clean confirmation of model BUY — high conviction.
- **DELL** (#2, score +0.39): Polygon: AI server demand drives revenue beat. Codex: model BUY supported by news; watch for valuation digestion after recent +28% move.
- **APA** (#3, score +0.38): Polygon: oil price strength + buyback announcement. Codex: clean BUY confirmation; energy sector exposure increases beta.
- **LITE** (#4, score +0.37): Polygon: NO-NEWS in past 5 days. Codex: model conviction is purely technical — higher uncertainty.
- **TTD** (#5, score +0.35): Polygon: ad-tech recovery narrative. Codex: confirmation but TTD has high vol (62%) — size accordingly.
```

Insert this section in JOURNAL.md **between** `### Model output` and `### Realized performance` for each entry.

## How to insert without breaking the journal

The journal uses HTML comments to delineate entries:
```
<!-- ENTRY:2026-04-24 -->
## 2026-04-24 (Friday)
### Market context
...
### Model output
...
### Realized performance        <- INSERT new section BEFORE this
...
<!-- /ENTRY:2026-04-24 -->
```

**Important:** preserve the existing user notes in `### Notes / lesson learned`. The journal regenerator is configured to preserve user notes across regeneration; your `### News read on Top 5` section should also be preservable. Insert it as a new section that the regenerator won't overwrite (you can use a similar HTML comment delimiter like `<!-- CODEX:NEWS-READ -->...<!-- /CODEX:NEWS-READ -->`).

## Inputs
- `us_pipeline/signals/JOURNAL.md`
- `us_pipeline/signals/news_briefs/<YYYY-MM-DD>.md` (12 files)

## Output
- Modify `us_pipeline/signals/JOURNAL.md` in place
- Brief response to `coordination/outbox/2026-04-27__top5_picks_narrative_for_journal.md`

## Acceptance criteria
- [ ] All 12 entries get the new section
- [ ] User's manual notes (currently in 2026-04-08 and 2026-04-15) are preserved untouched
- [ ] Each Top 5 narrative is 50-80 words total (5 bullets, ~10-15 words each)
- [ ] Conflict flags from the news briefs are surfaced (if a Top 5 pick has STRONG-NEG news, the bullet should say "⚠ news contradicts model")

## Notes

- **The point is to help the human decide whether to override the model.** A pick where news strongly agrees → trade with confidence. A pick where news contradicts → consider skipping or reducing size. A pick with no news → relying entirely on technicals → higher uncertainty.
- This becomes part of the daily routine going forward — Codex generates Top 5 narrative as part of news brief workflow.

## Codex response

# Top-5 news read added to journal

Inserted `### News read on Top 5` sections into `us_pipeline/signals/JOURNAL.md` with `<!-- CODEX:NEWS-READ -->` delimiters.

## Coverage

Inserted 14 sections:

`2026-04-08`, `2026-04-09`, `2026-04-10`, `2026-04-13`, `2026-04-14`, `2026-04-15`, `2026-04-16`, `2026-04-17`, `2026-04-21`, `2026-04-22`, `2026-04-23`, `2026-04-24`, `2026-04-27`, `2026-04-28`.

The task originally referenced 12 historical days; the local repo now has more recent journal/news context, so I inserted every recent entry with a usable news brief plus Apr 28 from signal CSV news fields. I did not insert Apr 20 because it has neither a news brief nor news fields in the signal CSV.

## Format

Each inserted section has 5 bullets, one per Top-5 model pick, between `### Model output` and `### Realized performance`. Strong negative news uses the requested conflict language: `⚠ news contradicts model`.

Manual notes were preserved. I spot-checked the existing note blocks for Apr 8 and Apr 15 after insertion.

## Closeout

Files modified:
- `us_pipeline/signals/JOURNAL.md`
- `us_pipeline/coordination/outbox/2026-04-27__top5_picks_narrative_for_journal.md`

File count: 2 for this task.

Runtime: about 15 minutes.

Caveats: Apr 28 used signal CSV news aggregates because no `news_briefs/2026-04-28.md` exists. Apr 20 was skipped because no local news context was available in the expected formats.

Code read for context:
- `us_pipeline/signals/JOURNAL.md`
- `us_pipeline/signals/news_briefs/*.md`
