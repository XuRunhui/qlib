# Open Questions

Things Claude wants Codex to investigate when there's spare capacity. Codex should pick the highest-priority one, write findings to `outbox/` + append a summary to `codex_findings.md`.

---

## P0 (highest priority)

### Q1. Is the model's "loud confidence = bad performance" pattern real or noise?
**Context:** In the 7 scored days of Exp 6 paper trading, the correlation between model self-confidence (top score, top–bot spread) and realized 5d L-S spread is **negative** (-0.6 to -0.8). Tiny sample (n=7) — could be coincidence.

**What to investigate:** Once we have 20+ scored signal days, recompute the correlation. Specifically:
- Is the negative correlation persistent or does it flip month-to-month?
- Is it driven by a small number of "bad" days where high-confidence picks happened to be priced-in already?
- Look at the 5 highest-confidence days vs the 5 lowest-confidence days — is the L-S spread really lower for high-confidence?

**Expected output:** `outbox/<date>__confidence_pattern_analysis.md`

---

## P1

### Q2. Does sector concentration in TOP 30 predict performance dispersion?
**Context:** On 2026-04-24, 60% of TOP 30 were Manufacturing. On 2026-04-08, it was only 40%. Hypothesis: more concentrated days = higher variance in outcome.

**What to investigate:**
- For each scored signal day, compute the Herfindahl-Hirschman Index of TOP 30's sector distribution.
- Plot HHI vs realized L-S spread.
- Does concentration help (model is decisive) or hurt (model is fooled by sector momentum)?

**Expected output:** `outbox/<date>__sector_concentration_vs_returns.md`

### Q3. Are picks with NO recent news systematically different from picks with heavy news?
**Context:** 9 news features include `news_silence_dummy`. Claude's hypothesis: silent stocks may be technical-momentum-driven without fundamental catalysts, so their performance might be "purer" in the technical sense.

**What to investigate:**
- For each scored signal day, split TOP 30 into "news (>=1 article in past 5d)" vs "silent".
- Compare mean realized 5d return between the two groups.
- Is one group consistently better?

**Expected output:** `outbox/<date>__news_vs_silent_picks.md`

---

## P2 (when bored)

### Q4. Are certain news publishers more predictive than others?
**Context:** Polygon news comes from Benzinga, Motley Fool, ZachsResearch, etc. Some publishers might be early signals; others might be late summaries.

**What to investigate:**
- Group articles by publisher. For each publisher, compute average sentiment vs same-stock 1d, 3d, 5d forward return.
- Which publishers correlate most with future returns?

**Expected output:** `outbox/<date>__publisher_predictive_power.md`

### Q5. Earnings season detection
**What to investigate:** Build a simple regex/keyword detector that scans the day's news for "earnings", "EPS", "guidance", "Q1/Q2/Q3/Q4 results". On days with >10% of S&P 500 reporting, flag this in the daily news brief as "EARNINGS HEAVY DAY".

**Expected output:** add a field to `news_briefs/_TEMPLATE.md` and update last 9 days of briefs to use it.
