# Current Production Model Card

**Last updated:** 2026-04-26 (Claude, after Experiment 5 walk-forward results)

## Model identifier
**Variant C: LambdaRank + Alpha158 + Sector-Neutral**

## Specification

| Component | Value |
|---|---|
| Algorithm | LightGBM with `lambdarank` objective |
| Loss | LambdaRank (NDCG-optimizing pairwise) |
| Features | 158 (Qlib's Alpha158 — purely technical OHLCV-derived) |
| Label | 5-day forward return: `Ref($close, -6) / Ref($close, -1) - 1` |
| Label transform | Sector-neutral cross-sectional rank → quantized to 16 bins |
| Universe | S&P 500 (currently 503 tickers) |
| Train window | 2021-06-01 → (signal_date − 31 days) |
| Validation window | (signal_date − 30) → (signal_date − 1) |
| Cost model | 5 bps open + 15 bps close = 20 bps round-trip |

## Validated performance (Exp 5 walk-forward, 12 monthly OOS windows)

| Metric | Value |
|---|---|
| Mean monthly IC | +0.045 |
| Months with positive IC | 12 / 12 (100%) |
| Long compound return (12mo) | +74% |
| Long-Short compound return (12mo) | +110% |
| Avg long monthly max DD | -3.9% |

## Live paper-trade window (Exp 6, 8 days)
- Mean 5d L-S spread: +5.30%
- 7/7 scored days positive
- Top picks heavily concentrated in Manufacturing (60% of TOP 30 on 2026-04-24)

## Known weaknesses (for Codex to keep in mind when reading briefs)
- Tested only on a single market regime (post-COVID bull market, no major correction)
- 5-year training data limit (no 2008/2020 stress tests)
- Survivorship bias: universe = current S&P 500, not point-in-time constituents
- Sector concentration not capped at deployment time
- Long alpha is real (+5%); short alpha basically zero (~0%) — long-only is the safer deployment

## What Codex should flag in news briefs
- Any pick that has STRONG-NEG news in past 5 days while model says BUY
- Any pick that has STRONG-POS news in past 5 days while model says AVOID
- Days where >5 picks have earnings reports scheduled (event risk)
- Macro events (Fed, CPI, NFP) that affect the whole portfolio
