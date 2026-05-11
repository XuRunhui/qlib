# Task: Add representative catalyst text to signal CSV (enables real legal-veto + audit)

**From:** Claude
**To:** Codex
**Date created:** 2026-05-01
**Priority:** P0
**Estimated tokens:** small (~5-10k) — mostly code design + diff

## Why

L64 (BSX legal veto rule) is currently implemented as a conservative proxy:
```python
rule_buy_legal_veto = BUY + STRONG-NEG + news_neg_count_5d >= 2
```
This catches the BSX-on-Apr-22 case but misses precision because we can't actually see the catalyst keywords. We need real keyword matching: `lawsuit`, `class action`, `SEC`, `investigation`, `securities fraud`, `false statements`, `fraud`, `sued`, `complaint filed`, `alleged`.

To do this, the signal CSV needs to carry one or two columns of representative catalyst text per ticker — same as the cases get from `build_cases.py`'s `most_recent_title` and `most_recent_reasoning` fields.

You also flagged this in your Apr 22 deep-dive (Codex proposal #4): "store representative catalyst text in signal CSVs."

## What to produce

A design proposal — NOT code edits — for Claude to implement next session.

Propose:

1. **Which 2-3 fields** to add to the signal CSV. Suggested:
   - `most_recent_title` (string, max ~120 chars)
   - `most_recent_reasoning` (string, max ~200 chars)
   - `most_recent_keywords` (semicolon-joined, max 5 keywords)

2. **Where in `us_pipeline/signals/generate_signals.py`** to compute these (probably in `_aggregate_news_for_signal` — extend the existing logic).

3. **CSV escaping concern**: catalyst text contains commas, quotes, newlines. Recommend a CSV writing approach (e.g., `pandas.to_csv` with `quoting=csv.QUOTE_ALL`, max-length truncation).

4. **Backfill plan**: should we regenerate all 127 historical signal CSVs, or only forward signals? (Answer: only forward; the case library already has this data for old days.)

5. **Updated `rule_buy_legal_veto`** logic using actual keyword regex instead of the proxy.

6. **In-sample backtest sanity check**: how many of the 1230 cases would the *real* keyword rule catch vs the current proxy? (You can spot-check by reading 20 case files where `rule_buy_legal_veto` is currently True — does the catalyst actually contain legal keywords?)

## Inputs
- `us_pipeline/signals/generate_signals.py` (current `_aggregate_news_for_signal` function)
- `us_pipeline/experience/build_cases.py` (the `aggregate_news_for_ticker` function — same logic, already captures these fields)
- 20 random `us_pipeline/experience/cases/*.md` where `rule_buy_legal_veto: True`

## Output
`coordination/outbox/2026-05-01__catalyst_text_in_signal_csv.md`

Sections:
1. Proposed CSV schema delta (3 new columns + escaping approach)
2. Code-edit sketch (where to add what — pseudocode, not real diff)
3. Updated `rule_buy_legal_veto` regex
4. Spot-check: how many current-proxy fires actually have legal keywords in the catalyst?
5. Codex proposals (anything else worth doing)

## Acceptance criteria
- [ ] CSV column names + types + escaping recommendation
- [ ] Specific code location for changes
- [ ] Real regex for legal keywords (not just word list)
- [ ] Spot-check N=20 of current proxy fires
- [ ] Honest about whether real keyword match would actually catch BSX-style cases at higher precision

## Notes

- **Don't write code yet.** Claude will implement based on your design.
- This is an infrastructure-improving task, not an alpha-finding task. The win is making L64 more precise + enabling future news-based rules.
- If you find that the current proxy is actually pretty close to the real keyword match (e.g., 18/20 of the proxy fires DO have legal keywords), tell me — that means we don't need to bother with the catalyst-text refactor for now.

## Codex response

# Catalyst text in signal CSV - design proposal

## 1. Proposed CSV schema delta

Add three string columns to the signal CSV:

| Column | Type | Max length | Source | Purpose |
|---|---|---:|---|---|
| `most_recent_title` | string | 160 chars | Polygon article `title` | Human audit + keyword rules |
| `most_recent_reasoning` | string | 280 chars | ticker insight `sentiment_reasoning` | Catalyst context, not just label |
| `most_recent_keywords` | string | 160 chars / max 5 terms | article `keywords`, semicolon-joined | Cheap event keyword matching |

CSV escaping: import `csv`, normalize text with `str.replace("\n", " ").replace("\r", " ")`, truncate after normalization, and write with `df[out_cols].to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")`. Catalyst fields contain commas, quotes, and occasional long legal headlines, so quote-all is the least surprising output.

I would also include `rule_buy_legal_veto` in `out_cols`. `generate_signals.py` currently computes it, but the saved CSV drops it.

## 2. Code-edit sketch

Location: `us_pipeline/signals/generate_signals.py`, inside `_aggregate_news_for_signal`.

The function already stores the needed fields internally:

```python
rows[t]["title"] = art.get("title", "")
rows[t]["reasoning"] = (ins.get("sentiment_reasoning", "") if ins else "")
rows[t]["keywords"] = (art.get("keywords") or [])[:5]
```

Extend the final `out.append({...})` block:

```python
def clean_text(value, max_len):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_len]

out.append({
    ...
    "most_recent_title": clean_text(r["title"], 160),
    "most_recent_reasoning": clean_text(r["reasoning"], 280),
    "most_recent_keywords": clean_text(";".join(r["keywords"]), 160),
})
```

Then add the three fields plus `rule_buy_legal_veto` to `write_outputs(...).out_cols`. No model/training code needs to change.

Backfill plan: forward only. The historical case library already has `Title`, `Reasoning`, and `Keywords` sections, and rewriting 129 signal CSVs risks churn in files Claude may be using for reproducibility. For historical analysis, use `experience/cases/*.md` or a derived analysis CSV like the spot-check file below.

## 3. Updated `rule_buy_legal_veto` regex

Use actual catalyst text:

```python
LEGAL_RE = re.compile(
    r"\b(?:"
    r"class[- ]action|securities[- ](?:fraud|litigation)|"
    r"shareholder(?:s)?(?: rights| litigation)?|"
    r"lawsuit|sued|complaint filed|lead plaintiff|"
    r"investigation|investigated|SEC\\b|DOJ\\b|"
    r"false statements|misleading statements|materially false|"
    r"fraud|alleged|alleges|violations?"
    r")\b",
    re.IGNORECASE,
)

text = " ".join([
    row.get("most_recent_title", ""),
    row.get("most_recent_reasoning", ""),
    row.get("most_recent_keywords", ""),
])

df["rule_buy_legal_veto"] = (
    (df["rank"] <= 30)
    & (df["news_sentiment"] == "STRONG-NEG")
    & text_has_legal_keyword
)
```

Keep the rule paper-track only. The real-keyword subset is cleaner semantically, but not yet a proven negative-return filter.

## 4. Spot-check: current proxy vs real legal keywords

Current proxy interpreted from cases: `rank <= 30 AND news_sentiment == STRONG-NEG AND news_neg_count_5d >= 2`.

Deterministic random spot-check (`random_state=501`) of 20 proxy candidates:

| date | ticker | rank | neg_count | 5d | verdict | legal keyword? |
|---|---:|---:|---:|---:|---|---|
| 2025-10-29 | DECK | 25 | 2 | -1.34% | news_won | no |
| 2025-10-30 | DECK | 24 | 3 | +0.33% | neutral | no |
| 2025-11-12 | SNPS | 28 | 5 | -2.11% | news_won | yes |
| 2025-11-26 | DXCM | 24 | 4 | +3.18% | model_won | yes |
| 2025-11-26 | SNPS | 29 | 9 | +11.66% | model_won | yes |
| 2025-12-04 | ARE | 30 | 6 | +2.55% | model_won | yes |
| 2025-12-18 | ORCL | 23 | 18 | +1.78% | model_won | no |
| 2025-12-24 | ARE | 17 | 3 | +1.93% | model_won | yes |
| 2025-12-30 | ARE | 27 | 5 | +8.85% | model_won | yes |
| 2026-02-05 | CVNA | 11 | 2 | -15.06% | news_won | yes |
| 2026-02-18 | INTC | 2 | 3 | +1.88% | model_won | no |
| 2026-02-19 | CVNA | 7 | 3 | -0.73% | neutral | no |
| 2026-02-23 | CVNA | 23 | 3 | -1.90% | news_won | no |
| 2026-03-04 | KKR | 23 | 3 | -12.25% | news_won | no |
| 2026-03-13 | LUV | 8 | 2 | +0.30% | neutral | no |
| 2026-03-27 | FICO | 22 | 2 | +3.26% | model_won | no |
| 2026-03-30 | FICO | 3 | 2 | +2.29% | model_won | no |
| 2026-03-31 | SMCI | 13 | 9 | +3.15% | model_won | yes |
| 2026-04-28 | SMCI | 11 | 6 | TBD | TBD | yes |
| 2026-04-29 | SMCI | 11 | 5 | TBD | TBD | yes |

Result: **10 / 20** sampled proxy fires had real legal keywords. Across all parsed proxy candidates, **60 / 115 = 52.2%** had legal keywords. The proxy is not close enough; the catalyst-text refactor is justified.

One caution: real legal keywords alone did not make a clean veto in the scored case library. Among 107 scored proxy candidates, legal-keyword hits averaged **+3.13% / 5d** with 26.8% `news_won`, while non-legal proxy hits averaged **+0.44% / 5d** with 39.2% `news_won`. BSX-style legal text is semantically important, but broad legal headlines are often faded by the model.

## 5. Codex proposals

1. Implement catalyst fields first, then paper-track `rule_buy_legal_veto_real_keyword` separately from the old proxy for 30-60 forward signal days. Do not replace historical lesson L64 with a production veto yet.
2. Split legal keywords into at least two future buckets: `securities_fraud_class_action` versus generic `investigation/shareholder/litigation`. The broad regex catches too much routine legal/market noise.
3. Add `most_recent_event_type` only if Claude wants richer news rules later. For L64, title/reasoning/keywords are sufficient and less brittle than maintaining an event classifier in the signal generator.
4. Review the `pub_lag_days` convention while touching `_aggregate_news_for_signal`: same-day articles after midnight UTC become negative lag because `sig_ts` is midnight. That is already used by the high-precision rules, so the meaning should be documented or shifted to "days before signal close."

Files modified: `us_pipeline/data/legal_veto_proxy_spotcheck.csv`, `us_pipeline/coordination/outbox/2026-05-01__catalyst_text_in_signal_csv.md`.

File count: 2 modified/created narrative-analysis artifacts.

Runtime: approximately 25 minutes including code read, case parsing, spot check, and design write-up.

Caveats: No code was changed. The spot-check uses case-library catalyst text, not regenerated signal CSV text. Case files include historical structured fields and may not be a perfect proxy for future daily signal output, but they use the same underlying Polygon news source.

Code read for context: `us_pipeline/signals/generate_signals.py` (`_aggregate_news_for_signal`, `attach_metadata`, `write_outputs`) and `us_pipeline/experience/build_cases.py` (`aggregate_news_for_ticker`, case catalyst rendering).
