# Experience Library

> A growing collection of **specific cases** — model picks paired with their actual outcomes. Each case is a structured fact + a narrative interpretation. Patterns emerge as cases accumulate.

## Why this exists

The walk-forward backtest tells us *aggregate* statistics: "the model has IC of 0.045 with 100% positive months." But aggregates hide the **conditional** structure: when does the model win? When does it lose? Does the news disagree usefully? Does sector concentration predict performance?

**The experience library is a detailed answer.** Each case is one (signal day, ticker) where something interesting happened. Once we have hundreds of cases, statistics over the cases reveal the conditional patterns the aggregate hides.

## Structure

```
experience/
├── README.md         <- this file (the contract + current findings)
├── cases/            <- one .md per case, named <date>__<ticker>__<tag>.md
│   ├── 2026-04-24__TTD__news-warned-correctly.md
│   └── ...
└── patterns/         <- when 5+ cases share a pattern, promoted here
    ├── pattern_news_overrides_model_when_negative.md
    └── ...
```

## Case schema

Every case file has the same machine-parseable header (YAML frontmatter) followed by free-form sections:

```markdown
---
date: 2026-04-24
ticker: TTD
case_type: conflict_buy_news_negative
model_rank: 5
model_score: 0.353
model_action: BUY
news_sentiment: NEG
news_strong: false
news_event_type: earnings
news_published_count_5d: 8
news_pos_ratio_5d: 0.25
news_neg_ratio_5d: 0.50
sector: Services
ret_20d_pct: 6.7
ann_vol_20d_pct: 62
sector_concentration_top30_pct: 60
top_score_of_day: 0.39
spy_above_200ma_pct: 6.8

# realized (filled in 5+ trading days later)
ret_1d_pct: -3.46
ret_3d_pct: TBD
ret_5d_pct: TBD
verdict: news_won  # one of: model_won | news_won | both_won | both_lost | neutral
---

## Polygon catalyst
"Revenue growth slowed to 14% in Q4 from 22% prior year, facing increased competition from Amazon and big tech."

## Codex narrative
The article cited specific deceleration metrics + competitive threat. Model's BUY came from technical momentum (+10% in past 20 days), but news suggests the momentum was on borrowed time — a classic "news warned of fundamental risk that technical signals miss" pattern.

## Lesson tag
news_overrides_momentum_buy_on_revenue_deceleration
```

## Case types (the 4 quadrants × outcome)

| Symbol | Meaning |
|---|---|
| 🟢 `consensus_buy` | Model BUY (rank ≤ 5) + News STRONG-POS — both agree |
| ⚪ `solo_buy` | Model BUY (rank ≤ 5) + NO-NEWS — model alone |
| 🟡 `conflict_buy_news_negative` | Model BUY + News STRONG-NEG — disagreement on long side |
| ⚪ `solo_avoid` | Model AVOID (rank ≥ 499) + NO-NEWS |
| 🟡 `conflict_avoid_news_positive` | Model AVOID + News STRONG-POS |
| 🔴 `consensus_avoid` | Model AVOID + News STRONG-NEG |
| 💥 `big_realized_move` | |ret_5d| > 8% — ex-post tag, applied additively |

## Verdict labels

After 5 trading days when realized return is in:

| verdict | meaning |
|---|---|
| `model_won` | Model direction was right; News was wrong (or confirmed) |
| `news_won` | News direction was right; Model was wrong |
| `both_won` | Same direction, both right (boring but useful) |
| `both_lost` | Same direction, both wrong (rare but informative) |
| `neutral` | |ret_5d| < 1% — no clear signal |

## Current findings (updated as cases accumulate)

### Quadrant statistics (auto-generated from cases)

Run `python us_pipeline/experience/summarize_experience.py` to refresh.

```
Total cases: 1189
Date range: 2025-10-27 to 2026-04-27
Scored (have ret_5d): 1132

==========================================================================================
PER CASE TYPE
==========================================================================================
case_type                               N  scored   mean_5d     std   win% verdict_distribution
----------------------------------------------------------------------------------------------------------------------------------
conflict_avoid_news_positive          208     195     -0.24   +3.65    50% model_won:75 neutral:46 news_won:74
conflict_buy_news_negative            107      99     +1.95   +7.29    61% model_won:55 neutral:11 news_won:33
consensus_avoid                        22      22     -0.73   +4.26    55% both_lost:7 both_won:9 neutral:6
consensus_buy                         349     331     +3.46  +11.18    59% both_lost:118 both_won:186 neutral:27
solo_avoid                            333     318     +0.26   +3.40    44% model_lost_alone:137 model_won:98 neutral:83
solo_buy                              170     167     +4.41   +9.99    67% model_lost_alone:50 model_won:105 neutral:12

==========================================================================================
PER VERDICT (only scored cases)
==========================================================================================
  both_lost                 N= 125  mean_5d=   -6.31%
  both_won                  N= 195  mean_5d=   +9.86%
  model_lost_alone          N= 187  mean_5d=   +0.73%
  model_won                 N= 333  mean_5d=   +2.35%
  neutral                   N= 185  mean_5d=   -0.04%
  news_won                  N= 107  mean_5d=   +0.47%

==========================================================================================
QUADRANT MATRIX — model action × news label, mean 5d return
==========================================================================================

Model action = BUY:
  news_bin           N   mean_5d   win%
  NO-NEWS          167     +4.41%    67%
  POS              262     +3.27%    58%
  STRONG-NEG        99     +1.95%    61%
  STRONG-POS        69     +4.17%    65%

Model action = AVOID:
  news_bin           N   mean_5d   win%
  NEG               22     -0.73%    55%
  NO-NEWS          318     +0.26%    44%
  STRONG-POS       195     -0.24%    50%

==========================================================================================
CONDITIONAL: high vs low sector concentration days (BUY side only)
==========================================================================================
  Median sector concentration: 57%
  HIGH concentration (>57%): N=292 mean_5d=   +3.76%
  LOW concentration (<=57%): N=305 mean_5d=   +3.20%

==========================================================================================
CONDITIONAL: high vs low top_score days (BUY side only)
==========================================================================================
  Median top score: 0.383
  HIGH confidence (>0.383): N=290 mean_5d=   +4.60%
  LOW confidence (<=0.383): N=307 mean_5d=   +2.41%

```

### Confirmed patterns (5+ cases)

(none yet)

### Tentative patterns (2-4 cases)

- **News warns of fundamental risk that technical momentum misses** — observed in TTD/SMCI/COHR/LITE on 2026-04-24. Pattern: high-momentum stocks (+10-30% recent) get a NEG news article citing a specific business deceleration; model still ranks them BUY based on price action; they then drop -2% to -5% the next day. Hypothesis: model is catching the top of a momentum spike that fundamentals warned about.

## Backfill protocol

The bulk of the case library is built by **looking backward** at historical signals:

1. Generate signals for each historical date via `generate_signals_batch.py`
2. For each (date, pick), compute the structured features
3. Filter to "interesting" cases per the case_type rules above
4. Write case files with realized returns already filled in
5. Codex pass: read each case, write the narrative + lesson tag

After backfill, the daily routine adds 5-15 new cases per day automatically.

## How patterns get promoted

When 5+ cases share a similar narrative, write a `pattern_*.md` file in `patterns/`:

```markdown
# Pattern: News overrides momentum BUY on revenue deceleration

## Cases supporting (N=8)
- 2026-04-24__TTD: -3.46% on day (news flagged 14% rev growth)
- 2026-04-24__SMCI: -4.23% on day (news flagged competitive pressure)
- ...

## Conditions
- Model rank ≤ 10 (high-confidence BUY)
- 20d momentum > +5%
- News in past 5d cites: revenue/earnings deceleration, competitive threat, margin pressure
- Sentiment label: NEG or STRONG-NEG

## Empirical result
- N=8 cases
- Mean 5d realized return: -3.2%
- Std: 1.8%
- Win rate (any positive day): 25%

## Recommendation
Reduce position size to 0.3× standard or skip when this pattern triggers. Rule:
```python
if pick.rank <= 10 and pick.ret_20d > 0.05 and codex_news_label in ['NEG', 'STRONG-NEG'] \
   and any(kw in news_text for kw in ['deceleration', 'competitive', 'margin pressure']):
    return SKIP
```

## Status
- Discovered: 2026-04-27
- Confirmed: 2026-05-XX (when N reaches 10+)
```

Patterns that hold up over additional data → promoted to `README.md` Lessons section.
Patterns that decay → moved to `patterns/retired/` with a note explaining what changed.
