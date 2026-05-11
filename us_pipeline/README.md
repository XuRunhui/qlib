# US Equity Quant Research Log

> A living research journal for an end-to-end US equity ML pipeline built on top of Microsoft Qlib.
> Each entry records the **hypothesis**, **setup**, **result**, and **interpretation**, with the
> goal of compounding intuition about US market microstructure over time.

**Researcher:** Runhui Xu
**Started:** 2026-04-25
**Investing accounts:** Robinhood (taxable, swing-trade book), Chase (taxable, long-term ETF book)
**Tax status:** Non-Resident Alien (NRA) — capital gains tax-exempt federally; dividends withheld 30% (10% via W-8BEN treaty)
**Strategy thesis:** Daily-frequency cross-sectional ranking on S&P 500, holding 1–10 days, sized for a small retail book. NRA tax status makes capital-gains-driven swing trading particularly favorable.

---

## Table of Contents

1. [Infrastructure](#infrastructure)
2. [Data](#data)
3. [Universe](#universe)
4. [Modeling Framework](#modeling-framework)
5. [Experiment Log](#experiment-log)
6. [Open Questions & Backlog](#open-questions--backlog)
7. [Lessons Learned (Running List)](#lessons-learned-running-list)

---

## Infrastructure

| Component | Choice | Rationale |
|---|---|---|
| **Quant framework** | Microsoft Qlib (editable install) | Mature US/CN equity ML platform; built-in Alpha158/360 factor sets, MLflow tracking, backtest engine |
| **Python env** | uv-managed venv at `.venv/`, Python 3.10 | Reproducible, isolated from system; ~10s to recreate |
| **Storage** | Polygon CSVs → Qlib binary format | Qlib binary is column-oriented + memory-mapped; 503 stocks × 5y loads in ~2s |
| **Experiment tracking** | MLflow (file backend at `mlruns/`) | Default Qlib backend; sufficient for local research |

**Key directories:**
```
us_pipeline/
├── universe.py                  # S&P 500 ticker scraper
├── download_polygon.py          # Polygon EOD downloader (resumable)
├── fetch_sectors.py             # SIC-code-based sector classifier
├── to_qlib_bin.py               # CSV → Qlib bin conversion
├── sector_processor.py          # Custom Qlib processor: sector-neutral rank
├── lgb_rank_model.py            # Custom Qlib model: LightGBM LambdaRank
├── workflow_lightgbm_us.yaml    # Reference workflow config
├── sweep.py                     # Horizon/topk/benchmark sweep (v1)
├── sweep_v2.py                  # Stacked B→C→A→D sweep (v2)
├── sweep_final.py               # Final stacked comparison
├── sweep_report.py              # Re-read results from MLflow
├── signals/                     # Daily live-signal pipeline (Experiment 6)
│   ├── generate_signals.py      # Train C-model + predict for given date
│   ├── spy_filter.py            # GO/REDUCED/NO-GO risk gate based on SPY MAs
│   ├── paper_trade_log.py       # Ingest signals + score realized returns
│   ├── daily_journal.py         # Auto-update JOURNAL.md with each day's full context
│   ├── JOURNAL.md               # ⭐ DAILY DECISION JOURNAL — running record + manual notes
│   ├── news_briefs/             # ⭐ Codex-generated plain-English context per signal day
│   │   ├── _TEMPLATE.md
│   │   └── <YYYY-MM-DD>.md
│   ├── <YYYY-MM-DD>.csv         # Full ranked picks for date
│   └── <YYYY-MM-DD>_summary.md  # Human-readable Top30 / Bottom30 report
├── coordination/                # ⭐ Claude ↔ Codex handoff protocol
│   ├── README.md                # The contract: who reads/writes what
│   ├── inbox/                   # Claude → Codex tasks
│   ├── outbox/                  # Codex → Claude responses
│   └── shared/                  # State both agents read
│       ├── current_model_card.md
│       ├── open_questions.md
│       └── codex_findings.md
├── experience/                  # ⭐ Experience library — case-based intuition (Exp 8)
│   ├── README.md                # Schema + current findings (auto-updated)
│   ├── backfill_signals.py      # Generate signals for historical date ranges
│   ├── build_cases.py           # Filter to interesting cases, build structured records
│   ├── summarize_experience.py  # Quadrant statistics + conditional cuts
│   ├── cases/                   # One .md per (date, ticker) case
│   │   └── <YYYY-MM-DD>__<TICKER>__<case_type>.md
│   └── patterns/                # Promoted findings (5+ supporting cases)
└── data/
    ├── instruments/
    │   ├── sp500.txt            # 503 current S&P 500 tickers
    │   └── sectors.csv          # SIC code + 9-bucket sector mapping
    ├── raw/                     # 504 CSVs (S&P 500 + SPY/RSP/QQQ benchmarks)
    ├── qlib_bin/                # Qlib binary data (~150MB)
    ├── fundamentals/            # 503 JSON files of raw quarterly financials (Exp 4)
    ├── fundamental_factors.parquet  # 13 PIT-aligned daily fund factors (Exp 4)
    ├── news/                    # 1279 JSON files of daily Polygon Benzinga news (Exp 7)
    ├── news_factors.parquet     # 9 PIT-aligned daily news factors (Exp 7)
    ├── paper_trade_log/         # Live signal scoring (Exp 6)
    │   ├── trades.csv           # Per-pick log: signal_date, rank, ret_1d/3d/5d
    │   └── summary.md           # Aggregate performance summary
    ├── exp4_results.csv         # Single-window fundamentals comparison
    ├── exp5_walkforward_results.csv  # 4 variants × 12 months
    └── sweep_final_results.csv  # Earlier sweeps
```

**Daily operational workflow** (after US close, ~5pm ET):
```bash
python us_pipeline/download_polygon.py        # ~12s, fetch new bars
python us_pipeline/to_qlib_bin.py             # ~3s, rebuild Qlib bin
python us_pipeline/download_news.py           # ~5s for one new day
python us_pipeline/signals/spy_filter.py      # ~2s, GO/NO-GO check
python us_pipeline/signals/generate_signals.py  # ~40s, generate today's picks
python us_pipeline/signals/paper_trade_log.py --all  # ~5s, ingest + score
python us_pipeline/signals/daily_journal.py   # ~5s, update JOURNAL.md
python us_pipeline/experience/build_cases.py  # ~30s, refresh case library
python us_pipeline/experience/track_rules.py  # ~5s, forward-validation status of L53/L54 rules
```
Total: ~2 minutes, can be cron-scheduled.

**Two-agent workflow: Claude + Codex**
This pipeline uses a hybrid agent setup where two LLM-powered agents share work via files:
- **Claude** owns deep strategy, model code, experiment design, README/lessons synthesis
- **Codex** (separate session) owns high-volume narrative work — daily news brief generation, LLM-based sentiment summaries, ad-hoc investigations

The handoff happens through `us_pipeline/coordination/`:
- `inbox/` — Claude leaves tasks for Codex
- `outbox/` — Codex returns responses
- `shared/` — continuously-updated state both agents read (model card, open questions, findings)
- `signals/news_briefs/<date>.md` — Codex auto-generates one per signal day, gives plain-English context for why each pick was selected

Read the full protocol at [`coordination/README.md`](coordination/README.md). When starting a Codex session, point it at that file first — it'll know what to do.

**The journal (`signals/JOURNAL.md`)** is the single most important artifact for compounding intuition over time:
- One entry per signal day, chronologically ordered
- Auto-fills market context (SPY level, MAs, vol, gate), model output (top/bot picks, sector tilt, score distribution, confidence), and realized returns (filled in 5 trading days later)
- A `### Notes / lesson learned` section per entry — *manual fill* — for capturing observations only a human notices ("this picked tech the day before earnings", "this was a Fed week", "all top picks have already run +20%")
- Notes are **preserved across regeneration** — re-running `daily_journal.py` only updates auto-fields, never overwrites your manual annotations
- After 30+ entries, run a meta-analysis script to find correlations between signal characteristics and realized performance (this already revealed a tentative negative correlation between model confidence and realized L-S spread in the first 7 days — pattern to watch).

---

## Data

### Polygon API capability inventory (Stocks Starter $29/mo, audited 2026-04-26)

The actual entitlements on this tier substantially exceed what the published feature matrix suggests. Empirical test results below.

#### ✅ Available (19 endpoints)

| Category | Endpoint | Coverage | Quant value |
|---|---|---|---|
| **Prices** | | | |
| Daily aggregates | `/v2/aggs/.../day/...` | 5y rolling | 🔥🔥🔥 currently in use |
| Hour aggregates | `/v2/aggs/.../hour/...` | 5y rolling | 🔥🔥 intraday signals |
| **Minute aggregates** | `/v2/aggs/.../minute/...` | **5y+ rolling (verified)** | 🔥🔥🔥 **major hidden value** — VWAP execution, opening drift |
| Snapshot | `/v2/snapshot/...` | EOD + 15m delayed intraday | 🔥🔥 daily pre-market state |
| **Fundamentals** | | | |
| **Stock Financials** | `/vX/reference/financials` | Quarterly + TTM, ~5y | 🔥🔥🔥 **49 line items** (23 income, 18 balance sheet, 8 cash flow, 5 comprehensive) |
| Splits | `/v3/reference/splits` | Full history | 🔥 adjustment validation |
| Dividends | `/v3/reference/dividends` | Full history | 🔥🔥 dividend yield factor, ex-div day handling |
| **Reference** | | | |
| Ticker details | `/v3/reference/tickers/{ticker}` | SIC code, market cap, listing | 🔥🔥 currently used for sectors |
| All tickers list | `/v3/reference/tickers` | 30K+ active US tickers | 🔥🔥 universe management |
| Ticker Events | `/vX/reference/tickers/.../events` | Renames, restructurings | 🔥 data hygiene |
| Related Companies | `/v1/related-companies/...` | Algorithmic peers | 🔥 pair trading |
| **Sentiment / Flow** | | | |
| **News (Benzinga)** | `/v2/reference/news` | Articles + sentiment labels | 🔥🔥 NLP sentiment factor |
| **Short Interest** | `/stocks/v1/short-interest` | Bi-monthly SEC reports | 🔥🔥 short squeeze indicator |
| **Short Volume** | `/stocks/v1/short-volume` | **Daily FINRA data** | 🔥🔥🔥 retail vs institutional sell pressure |
| **Market structure** | | | |
| Market Status | `/v1/marketstatus/now` | Real-time | 🔥 live trading orchestration |
| Conditions | `/v3/reference/conditions` | Trade/quote codes | 🔧 data cleaning |
| IPOs | `/vX/reference/ipos` | Upcoming + recent | 🔥 new-issue strategies |
| **Options metadata** | | | |
| Options contracts | `/v3/reference/options/contracts` | Contract list (no prices) | 🔥 options universe (no pricing) |

#### ❌ Requires upgrade (6 endpoints)

| Endpoint | Required tier | Add-on cost | Use case |
|---|---|---|---|
| Tick trades / quotes | Stocks Advanced | $79/mo | Microstructure, market making |
| Real-time last trade/quote | Stocks Advanced | $79/mo | Live pricing (vs current 15m delay) |
| Options price aggregates | Options Starter | $29/mo extra | Options strategies |
| Indices aggregates (SPX/NDX) | Indices subscription | $29/mo extra | Direct index data (workaround: SPY/QQQ ETFs) |
| Forex | Forex Starter | $29/mo extra | FX pairs |
| Crypto | Crypto Starter | $29/mo extra | BTC/ETH/etc |

#### Highest-ROI underutilized data (ranked)

1. **Stock Financials** — 49 line items × 503 tickers × 20 quarterly reports ≈ 500K data points immediately accessible. Enables full set of value (PE, PB, EV/EBITDA, FCF Yield), quality (ROE, ROA, margins, leverage), and growth (revenue/earnings YoY) factors. Expected IC lift: +0.005 to +0.015 on top of current LambdaRank model.
2. **Daily Short Volume** — FINRA-reported daily short selling activity. Serves as a contemporaneous sentiment signal (uncommon at this price tier). Expected IC lift: +0.005-0.010, especially on small/mid caps and meme stocks.
3. **Minute aggregates** — 5y of minute-level OHLCV opens up: opening 30-minute return as a feature, intraday VWAP-relative position, time-of-day momentum patterns. Expected IC lift: +0.005.
4. **Benzinga news with sentiment labels** — pre-tagged sentiment removes the need for an LLM/transformer pipeline for basic sentiment factors.

### Provider: Polygon.io / Massive.com (Stocks Starter, $29/mo)

**Why Polygon over alternatives:**
- ✅ Direct SIP feed (consolidated tape from all US exchanges) → highest quality EOD data
- ✅ Unlimited API rate limit on Starter tier → 503 tickers full history downloads in **12.5 seconds**
- ✅ Includes splits, dividends, fundamentals (financials API) for future feature engineering
- ✅ Same provider as Robinhood and Webull's data backends → as close to broker reality as we can get
- ⚠️ History capped at ~5 years rolling window on Starter tier (vs. Tiingo's 30y at $10/mo)

### Current data inventory (as of 2026-04-25)

| Item | Coverage |
|---|---|
| **Tickers** | 503 current S&P 500 + SPY + RSP + QQQ benchmarks |
| **Date range** | 2021-06-01 → 2026-04-24 (~5y, 1231 trading days) |
| **Frequency** | Daily OHLCV + VWAP + transaction count |
| **Adjustments** | Split-adjusted (Polygon's `adjusted=true`) |
| **Total rows** | ~610K bars (~150MB on disk after Qlib bin conversion) |
| **Sector metadata** | 4-digit SIC codes, mapped to 9 coarse sector buckets |

### Known data limitations

| Issue | Impact | Mitigation |
|---|---|---|
| **Survivorship bias** | High. Universe = *current* S&P 500, not historical constituents. Backtests overestimate alpha because we miss companies that were dropped (often after underperformance). | Accept for now; upgrade to Polygon's historical-constituents API or Tiingo if results merit. |
| **Short history (5y)** | Models trained only on post-COVID + AI rally regime. Cannot validate against 2008 GFC, 2020 COVID crash, or pre-ZIRP rate environments. | Plan: subscribe to Tiingo ($10/mo) once a working strategy is identified, retrain on 10–30y. |
| **No intraday data** | Cannot compute open-to-open returns or VWAP execution. All trades modeled at close. | Acceptable for swing strategy. Polygon Advanced ($79/mo) adds real-time if needed. |
| **Adjustment timing** | Polygon back-adjusts on split/dividend events. Means historical CSVs may shift slightly when re-downloaded. | Re-download incrementally; accept minor lookback inconsistencies. |

---

## Universe

**Current:** S&P 500 constituents (503 tickers; 3 are dual-listed pairs like GOOGL/GOOG).

**Sector breakdown (SIC-based):**
| Sector | Count |
|---|---|
| Manufacturing | 201 |
| Finance/Insurance/RealEstate | 99 |
| Services | 76 |
| Transportation/Utilities | 64 |
| Retail | 30 |
| Mining | 14 |
| Wholesale | 10 |
| Construction | 8 |
| Agriculture | 1 |

**Why S&P 500 (and not Russell 2000 or Dow 30):**
- 503 names ≈ enough cross-sectional samples per day for ML
- Liquidity floor (avg $50M+ daily volume) → tradeable in retail size
- Quality floor (S&P committee curates) → less noise than Russell 2000
- Matches the universe Qlib's Alpha158 factor set was designed for (large/mid-cap)

---

## Modeling Framework

### Features
**Alpha158** (Qlib built-in): 158 cross-sectional factors derived from OHLCV — momentum, reversal, volatility, volume, candlestick shapes, price ratios. Originally designed for Chinese A-share market by Microsoft researchers.

### Labels (forward returns, 1-day-ahead trade execution model)

| Notation | Formula | Meaning |
|---|---|---|
| 1d | `Ref($close, -2) / Ref($close, -1) - 1` | Hold from T+1 close to T+2 close |
| 5d | `Ref($close, -6) / Ref($close, -1) - 1` | Hold from T+1 close to T+6 close |
| 10d | `Ref($close, -11) / Ref($close, -1) - 1` | Hold from T+1 close to T+11 close |

**Convention:** signal generated at close of T → trade at close of T+1 → hold N days. This avoids the unrealistic "predict T+1 close from features known at T+1" trap.

### Train / Validation / Test split

| Segment | Range | Days | Purpose |
|---|---|---|---|
| **Train** | 2021-06-01 → 2024-06-30 | ~770 | Model fitting |
| **Validation** | 2024-07-01 → 2025-03-31 | ~190 | Early stopping, hyperparameter tuning |
| **Test (out-of-sample)** | 2025-04-01 → 2026-04-22 | 266 | Held out, never seen during training |

### Trading cost model

| Component | Value | Notes |
|---|---|---|
| Open cost | 5 bps (0.05%) | Buy-side: spread + slippage |
| Close cost | 15 bps (0.15%) | Sell-side: includes SEC fee |
| Min cost per trade | $1 | Robinhood/Chase are $0; safety margin |
| Total round-trip | ~20 bps | Reasonable for liquid S&P 500 names |

### Evaluation metrics

| Metric | Formula | What it measures |
|---|---|---|
| **IC** | mean of daily Pearson(score, next_day_return) | Linear predictive power |
| **Rank IC** | mean of daily Spearman(score, next_day_return) | Ranking power, robust to outliers |
| **ICIR** | IC.mean() / IC.std() | Stability of signal across days |
| **Long Excess** | mean(top30 returns) − mean(market returns) | Long-only alpha |
| **Long-Short Return** | mean(top30) − mean(bot30) | Pure alpha (market-neutral) |
| **IR** | mean / std × sqrt(252) | Risk-adjusted return (annualized) |
| **MaxDD** | min(cum_return / cummax − 1) | Worst peak-to-trough loss |

---

## Experiment Log

### Experiment 1 — Baseline: Qlib defaults on US equities
**Date:** 2026-04-25
**Hypothesis:** Qlib's stock-picking pipeline (Alpha158 + LightGBM + TopkDropoutStrategy), known to deliver ~IR 1.0 on China CSI 300, will produce a useful (positive IR) signal on US S&P 500 with no modifications other than data swap.
**Config:** `workflow_lightgbm_us.yaml` — Alpha158, MSE loss, CSRankNorm label transform, 1-day forward return label, top30/drop5 long-only, benchmark SPY.

**Result:**
| Metric | Value |
|---|---|
| IC | -0.004 |
| Rank IC | -0.005 |
| Annualized excess (with cost) vs SPY | **-4.1%** |
| SPY benchmark return | +23.1% |

**Interpretation:** The naive port to US data **fails to generate alpha**. Three structural problems identified:
1. **Daily horizon noise** — 1-day predictions are dominated by noise in efficient large-cap US market.
2. **Cap-weighted SPY benchmark unfair** — top30 equal-weight strategies cannot beat cap-weighted SPY in a Mag-7 dominated rally without explicitly tilting to those names.
3. **Alpha158 is CN-tuned** — many features capture A-share-specific microstructure (price limits, T+1 settlement effects, retail-driven momentum patterns).

---

### Experiment 2 — Horizon × Turnover × Benchmark sweep
**Date:** 2026-04-25
**Hypothesis:** The baseline failure is **not** a model capacity problem but a **task formulation** problem. Specifically: (a) longer horizons (5d, 10d) carry more signal-to-noise; (b) RSP (equal-weighted) is the fair benchmark for an equal-weighted strategy; (c) lower turnover reduces cost drag.

**Config:** 10 runs spanning horizon ∈ {1d, 2d, 5d, 10d, 20d}, topk/drop ∈ {30/5, 50/2, 10/2}, benchmark ∈ {SPY, RSP}.

**Selected results (cross-sectional summary):**

| Config | IC | Ann. Excess (cost) vs benchmark | IR | Comment |
|---|---|---|---|---|
| baseline 1d top30/5 vs SPY | -0.004 | -4.1% | -0.37 | Original baseline |
| **5d top30/5 vs RSP** | +0.004 | **+10.8%** | **+0.88** | ⭐ Sweet spot |
| 10d top30/5 vs SPY | +0.013 | -4.1% | -0.31 | Higher IC but kills IR via turnover |
| 5d top10/2 vs SPY | +0.004 | -3.5% | -0.20 | Concentration trap |
| baseline 1d top30/5 vs RSP | -0.004 | +3.8% | +0.43 | Same model, fair benchmark → +800 bps |

**Interpretation:**
1. **Benchmark choice was responsible for ~10% annualized "missing" alpha.** SPY's 23.1% return in 2025-04→2026-04 was driven by the 30%-weighted Mag 7 (NVDA +90%, META +35%, MSFT +20%, etc.). RSP returned ~14% in the same period. Comparing an equal-weight 30-stock strategy to cap-weight SPY is structurally biased.
2. **5-day horizon is the empirical sweet spot.** Confirms academic literature on US momentum (Jegadeesh & Titman: 6–12 month works best, but reversal at 1–5 day for short-term).
3. **Concentration (top10) destroys returns via cost.** Daily turnover ratio = n_drop/topk; top10/2 = 20% per day → even with 10 bps round-trip, this compounds into ~5% annualized cost drag.
4. **Real (cost-adjusted) alpha exists at IC=0.004 level**, but it's small and easily wiped by execution friction.

---

### Experiment 3 — Stacked improvements: B (sector-neutral) → C (long-short) → A (LambdaRank) → D (ensemble)
**Date:** 2026-04-26
**Hypothesis:** Each addition compounds the previous: sector-neutral labels remove sector beta noise, long-short evaluation reveals true alpha, LambdaRank loss optimizes the ranking objective directly, and a multi-horizon ensemble averages out per-horizon noise.

**Config:** Each experiment trains a fresh LightGBM model with the marginal change applied. All use 5-day horizon, top30/drop5 sizing.

**Result (test set 2025-04-01 → 2026-04-22, 266 days):**

| Config | IC | Long Ann (cost) | Long IR | LongMax DD | LS Ann (cost) | LS IR | LS Max DD |
|---|---|---|---|---|---|---|---|
| baseline (MSE + CSRank) | +0.0022 | +2.3% | +0.19 | -13.6% | -0.7% | -0.05 | -19.2% |
| B: + Sector-Neutral Rank | +0.0026 | -4.5% | -0.37 | -17.9% | -3.0% | -0.22 | -20.0% |
| C: Long-Short of B | +0.0026 | (same as B) | | | -3.0% | -0.22 | -20.0% |
| **A: + LambdaRank loss** | **+0.0451** | **+74.7%** | **+3.04** | **-9.0%** | **+102.4%** | **+2.96** | **-12.9%** |
| D-MSE: 1d/5d/10d ensemble (MSE) | +0.0071 | +2.2% | +0.22 | -11.6% | +3.0% | +0.22 | -15.6% |
| D-LR: 1d/5d/10d ensemble (LambdaRank) | +0.0412 | +72.0% | +2.85 | -9.4% | +99.3% | +2.85 | -12.1% |

**Interpretation:**

1. ⚡ **LambdaRank is the breakthrough — 22× IC improvement (0.002 → 0.045) from just changing the loss function.**
   - With Alpha158's noisy features, MSE wastes optimization capacity learning the *absolute level* of rank values (which we don't care about).
   - LambdaRank focuses gradient on **mis-ordered pairs at the top of the list** (NDCG-relevant pairs), which is exactly what a long-only top-K strategy needs.
   - Required custom Qlib model wrapper (`lgb_rank_model.py`) to pass `group` parameter to LightGBM.

2. ❌ **Sector-neutral rank (B) hurt performance in this test period.**
   - Rationale was: remove sector beta from the label so model learns pure stock-specific alpha.
   - In practice 2025-04→2026-04 had *huge* sector dispersion (tech vs. utilities/consumer staples), and CSRankNorm naturally exploited this. Removing it destroyed the sector-tilt signal.
   - **Lesson:** Sector neutrality is a tool for risk management, not necessarily alpha. It helps when sector beta is noise; it hurts when sector beta is signal.

3. ❌ **Multi-horizon ensemble (D) made things worse.**
   - 1d models have IC ≈ 0 — adding them as 20% weight to the ensemble simply diluted the strong 5d signal.
   - **Lesson:** Ensemble quality depends on each component being independently useful. Garbage components add noise, not robustness.

4. ⚠️ **The 102% annualized long-short return is suspiciously high. Need walk-forward validation before any reliance.**
   - Test period is only 1 year of unique market regime
   - Daily IC has high std (0.21) — most days are losing trades, but tail days are big winners
   - Real frictions (slippage, partial fills, borrow costs for shorts) may halve realized returns
   - **Realistic expectation:** 20–40% annualized alpha if signal generalizes; lower if it doesn't

5. ✅ **Cleanest practical config: A_LambdaRank_SecNeutral_5d (long-only)**
   - +75% annualized excess vs RSP (equal-weight S&P 500)
   - -9% max drawdown
   - IR 3.0
   - Implementable in Robinhood (long-only, no shorts needed, no margin)

---

### Experiment 4 — Adding fundamental factors (Alpha158 + 13 fundamentals)
**Date:** 2026-04-26
**Hypothesis:** Fundamental factors (PE, PB, ROE, revenue growth, margins, etc.) carry information uncorrelated with the technical Alpha158 features. Adding them on top of the LambdaRank model from Experiment 3-A should lift IC by 0.005–0.015 and improve risk-adjusted returns.

**Setup:**
- Downloaded all available quarterly financial reports from Polygon `/vX/reference/financials` for 503 S&P 500 tickers (35,668 reports total, ~71/ticker, going back ~9 years).
- Computed 13 daily fundamental factors with **strict point-in-time alignment** (using `filing_date` as the as-of date, NOT `end_date` — critical to avoid lookahead bias):
  - **Value:** `fund_pe_ttm`, `fund_pb`, `fund_ps_ttm`, `fund_fcf_yield`
  - **Quality:** `fund_roe`, `fund_roa`, `fund_gross_margin`, `fund_op_margin`, `fund_net_margin`, `fund_de_ratio`
  - **Growth:** `fund_revenue_yoy`, `fund_eps_yoy`
  - **Size:** `fund_log_mcap`
- Coverage: 85–97% non-NaN per factor (gross_margin lower at 58% — financials/REITs don't report it).
- Built `Alpha158WithFundamentals` Qlib handler that merges fundamentals into the feature group of the standard Alpha158 pipeline.
- Trained the LambdaRank-Sector-Neutral-5d configuration with the augmented feature set.

**Result:**

| Config | IC | ICIR | Long Ann (cost) | Long IR | Long MaxDD | LS Ann (cost) | LS IR | LS MaxDD |
|---|---|---|---|---|---|---|---|---|
| Reference: LR + Alpha158 (Exp 3-A) | +0.0451 | +0.21 | +74.7% | +3.04 | -9.0% | +102.4% | +2.96 | -12.9% |
| **Exp 4: LR + Alpha158 + 13 Fundamentals** | **+0.0289** | **+0.21** | **+60.8%** | **+2.92** | **-7.3%** | **+74.2%** | **+2.79** | **-8.7%** |

**Feature importance (Top 16 features by LightGBM gain in Exp 4):**

| Rank | Feature | Source |
|---|---|---|
| 1 | **fund_revenue_yoy** | Fundamental ⭐ |
| 2 | KLEN (candle length) | Alpha158 |
| 3 | (Alpha158 momentum) | Alpha158 |
| 4 | **fund_op_margin** | Fundamental |
| 5 | **fund_pb** | Fundamental |
| 6 | **fund_net_margin** | Fundamental |
| 7 | **fund_eps_yoy** | Fundamental |
| 8 | **fund_log_mcap** | Fundamental |
| 9–10 | (Alpha158) | Alpha158 |
| 11 | **fund_ps_ttm** | Fundamental |
| 12 | (Alpha158) | Alpha158 |
| 13 | **fund_pe_ttm** | Fundamental |
| 14 | **fund_roe** | Fundamental |
| 15 | **fund_gross_margin** | Fundamental |
| 16 | **fund_roa** | Fundamental |

→ 11 of the top 16 features used by the model are fundamentals. The model strongly *wants* them.

**Daily IC stability comparison:**

| Metric | Reference (Alpha158 only) | EXP4 (+ Fundamentals) |
|---|---|---|
| Avg monthly IC mean | +0.045 | +0.029 |
| **Avg monthly IC std** | **~0.21** | **~0.13** (-40%) |
| Worst-month IC | +0.016 (Jul) | +0.009 (Feb) |
| Best-month IC | +0.113 (Apr 26) | +0.065 (Sep) |

**Interpretation:**

1. ⚠️ **Counterintuitive headline result: adding fundamentals lowered raw IC (-0.016) and lowered absolute returns (-14% annualized).** This contradicts the Tier-1 hypothesis.

2. 🔥 **However, fundamentals dramatically improved risk-adjusted properties:**
   - Daily IC volatility cut by ~40% (0.21 → 0.13)
   - Long-only max drawdown improved -9.0% → -7.3% (19% better)
   - Long-short max drawdown improved -12.9% → -8.7% (33% better)
   - IR remained roughly stable (3.04 → 2.92)

3. **Why this happens — a useful market lesson:** Technical features (momentum, volatility, candle shapes) are *fast and noisy* signals — they generate occasional huge IC days but with high error. Fundamentals are *slow and stable* signals — companies don't change quality from quarter to quarter. The model that uses both averages out the technical noise, sacrificing peak signal for consistency.

4. **The catch — and likely real explanation for IC loss:** LambdaRank is a *ranking-only* loss; it doesn't care about the magnitude of mistakes. Adding 13 highly-utilized fundamental features to 158 technical features means the model's gradient updates are split — for most pair comparisons, the fundamentals "agree with" or "disagree with" the technicals, and LambdaRank is forced to choose. In a 1-year test window where the technical signals happened to be **right** about ranking direction, the fundamentals' tempering effect lowered IC.

5. **Question for next experiment:** Is the IC drop a regime artifact or a real cost? Walk-forward validation across multiple periods would tell us. If fundamentals consistently lower IC by ~30% but consistently improve drawdown by ~30%, that's a real **risk/return tradeoff** decision — and lower-DD is preferable for real-money trading.

6. **Selection conclusion:** For real deployment, the EXP4 model is arguably **the better choice** despite lower headline returns: the -8.7% drawdown vs -12.9% is a meaningful difference for a small retail account that can't tolerate large losses (7-10% drawdowns trigger emotional sell-decisions for most retail traders). The IR of 2.9 is functionally equivalent — both models are statistically "the same quality" risk-adjusted, but EXP4 will *feel* much smoother in live trading.

**Decision:** Adopt EXP4 (LR + Alpha158 + Fundamentals + sector-neutral) as the working production config. Move forward to walk-forward validation to confirm both models' performance is real and not regime luck.

> **Update from Experiment 5:** This decision was REVERSED after walk-forward results showed the EXP4 drawdown improvement was a single-window artifact. Production config reverted to **C: LambdaRank + Alpha158 (no fundamentals)** with 100% positive-IC months and 2× the returns of EXP4.

---

### Experiment 5 — Walk-forward validation (2x2: features × loss × 12 months)
**Date:** 2026-04-26
**Hypothesis:** The Experiment 3-A and 4 results were measured on a single fixed test window (2025-04 → 2026-04). To distinguish "true alpha" from "lucky regime", retrain monthly across the full test period and measure IC + PnL per month. Specifically test whether (a) LambdaRank's 5× IC lift over MSE generalizes across regimes, and (b) fundamentals' supposed drawdown improvement holds outside the single-window observation.

**Setup:**
- 2 × 2 grid: features ∈ {Alpha158, Alpha158 + Fundamentals}, loss ∈ {MSE, LambdaRank}.
- 12 monthly OOS windows from 2025-05 to 2026-04. For each: retrain on `[2021-06-01, month_start − 31d]`, validate on prior 30 days, predict the month.
- All variants use sector-neutral rank label, 5-day forward return target, top30/drop5 portfolio.
- Total 48 model trainings × ~25s ≈ 25 minutes wall time.

**Aggregate Result (12 months walk-forward, IR, drawdown, IC consistency are the headline metrics):**

| Variant | Mean IC | IC consistency<br>(% months > 0) | Long compound return<br>(12mo) | LS compound return<br>(12mo) | Avg long DD/month |
|---|---|---|---|---|---|
| A: Alpha158 + MSE | +0.0094 | 67% (8/12) | +23% | +6% | -3.2% |
| B: Alpha158+Fund + MSE | +0.0125 | 83% (10/12) | +16% | +1% | -2.7% |
| **C: Alpha158 + LambdaRank** | **+0.0453** | **100% (12/12)** ⭐ | **+74%** | **+110%** | -3.9% |
| D: Alpha158+Fund + LambdaRank | +0.0260 | 100% (12/12) | +36% | +50% | -4.0% |

**Per-month IC table:**

| Month | A (MSE) | B (MSE+Fund) | C (LR) | D (LR+Fund) |
|---|---|---|---|---|
| 2025-05 | -0.009 | +0.002 | +0.053 | +0.039 |
| 2025-06 | +0.017 | +0.024 | +0.070 | +0.033 |
| 2025-07 | +0.017 | +0.029 | +0.034 | +0.025 |
| 2025-08 | -0.013 | +0.010 | +0.014 | +0.033 |
| 2025-09 | +0.019 | +0.015 | +0.077 | +0.046 |
| 2025-10 | +0.022 | +0.008 | +0.026 | +0.010 |
| 2025-11 | +0.029 | +0.021 | +0.050 | +0.024 |
| 2025-12 | +0.007 | +0.023 | +0.015 | +0.015 |
| 2026-01 | -0.011 | -0.027 | +0.032 | +0.009 |
| 2026-02 | -0.012 | -0.003 | +0.021 | +0.006 |
| 2026-03 | +0.015 | +0.032 | +0.048 | +0.036 |
| 2026-04 | +0.030 | +0.017 | +0.104 | +0.036 |

**Interpretation — five major findings:**

1. ⭐ **LambdaRank's IC advantage is REAL and ROBUST across regimes.**
   - Mean IC lift over MSE: **0.009 → 0.045 (5× lift), Alpha158-only.**
   - **100% positive-IC months** vs 67% for MSE. This is the strongest possible robustness signal.
   - The single-window result from Exp 3-A was not a lucky window — it generalizes.

2. ⚠️ **Fundamentals UNDER LAMBDARANK HURT performance — Exp 4's "smoothing" claim was an artifact.**
   - IC dropped 0.045 → 0.026 (consistent with Exp 4) — but this drop is real cost, not a tradeoff for stability.
   - **Drawdown was NOT improved** in walk-forward (C: -3.9% avg DD, D: -4.0% avg DD; basically identical).
   - Long compound return: C +74% vs D +36%. Long-short: C +110% vs D +50%. **Adding fundamentals cost ~50% of total return.**
   - Lesson: the Exp 4 single-window drawdown improvement (-9.0% → -7.3%) was a REGIME-SPECIFIC fluke. Walk-forward washes it out.

3. 🧠 **The deepest cross-finding: feature × loss interaction matters more than either alone.**
   - MSE is weak → fundamentals add value (A: 67% → B: 83% consistency, +0.004 IC).
   - LambdaRank is strong → fundamentals dilute the ranking pairs (C: 0.045 → D: 0.026 IC).
   - Mechanism: LambdaRank's gradient updates focus on *misordered pairs at top of list*. With 158 technical features, almost every pair has a clear technical "winner". With 13 fundamentals added, many pairs have technicals saying X > Y but fundamentals saying X < Y — the model has to make a coin-flip judgment, dampening the gradient's signal-to-noise ratio.

4. 📉 **2026-02 was the universally hard month** — all 4 variants had IC near zero or negative.
   - A: -0.012, B: -0.003, C: +0.021 (saved by LambdaRank but barely), D: +0.006.
   - Models work in normal markets but no model is regime-proof. This is the "tail risk" all quant strategies face.
   - **Implication for live trading**: expect 1-2 months per year where the model adds zero or negative value. Position sizing must accommodate this.

5. 🚀 **In high-signal months, LambdaRank captures 2-3× more alpha.**
   - Best month (Sep 2025): A long +7.6%, C long +14.4% (+90% relative); A LS +4.4%, C LS +15.6% (+255%!).
   - Best month (Apr 2026): A long +5.5%, C long +7.9%; A LS +4.4%, C LS +11.4% (+159%).
   - This asymmetric upside capture is exactly what makes LambdaRank work — it doesn't help much in bad months, but takes huge wins in good ones.

**Decision:** Production config reverts to **C: LambdaRank + Alpha158 (no fundamentals) + sector-neutral 5d label + top30/drop5**.

---

### Experiment 6 — Live signal generation & first paper-trade scoring window
**Date:** 2026-04-26
**Hypothesis:** The walk-forward results from Experiment 5 (mean IC 0.045, 100% positive months) should translate into actionable signals when run as a daily pipeline. Test by generating signals retrospectively for 8 historical days that were never explicitly used in any train/test split, score the realized returns, and check whether the live performance matches the backtest expectation.

**Setup:**
- Built three new scripts in `us_pipeline/signals/`:
  - `generate_signals.py`: trains the C model on rolling history (anchored 2021-06-01 to D-31), validates on D-30..D-1, predicts for date D. Outputs ranked CSV + markdown summary.
  - `spy_filter.py`: risk gate based on SPY's position vs 20/50/200-day MAs. Returns GO / REDUCED / NO-GO.
  - `paper_trade_log.py`: ingests signal CSVs, scores realized 1d/3d/5d returns, prints aggregate performance.
- Backfilled signals for 8 trading days (2026-04-08 through 2026-04-17), then scored them after the data window closed.
- This is a **true out-of-sample test** — none of these dates were in the original Exp 5 train or validation segments because the production model uses a fresh rolling window per signal date.

**Result (8 signal days, 270 long picks + 270 short picks; data scored through 2026-04-24):**

| Horizon | Long mean ret | Long win rate | Short mean ret | Long-Short spread | Net of cost |
|---|---|---|---|---|---|
| 1d | +1.43% | 69% | +0.01% | +1.42% | +1.23% |
| 3d | +3.78% | 76% | -0.06% | +3.84% | +3.58% |
| **5d** | **+5.32%** | **76%** | **+0.02%** | **+5.30%** | **+5.12%** |

**Per-day Long-Short 5d spread (the "did the model earn money today" view):**

| Signal Date | Long avg | Short avg | L-S Spread |
|---|---|---|---|
| 2026-04-08 | +8.29% | -0.27% | +8.55% |
| 2026-04-09 | +7.41% | +1.58% | +5.83% |
| 2026-04-10 | +6.78% | +0.89% | +5.90% |
| 2026-04-13 | +4.90% | -0.53% | +5.42% |
| 2026-04-14 | +4.94% | -0.10% | +5.04% |
| 2026-04-15 | +1.39% | +0.18% | +1.21% |
| 2026-04-16 | +3.54% | -1.64% | +5.18% |
| **Days positive** | | | **7/7 (100%)** |

**SPY trend filter status (as of 2026-04-24):** 🟢 **GO**
- SPY $713.94, +6.8% above 200-day MA, vol 17% — model approved to trade.

**Today's TOP 5 picks (signal date 2026-04-24):**

| # | Symbol | Score | Sector | Notes |
|---|---|---|---|---|
| 1 | COHR | +0.390 | Manufacturing | Optical/AI infrastructure |
| 2 | DELL | +0.388 | Manufacturing | AI server demand |
| 3 | APA | +0.375 | Mining | Energy/oil |
| 4 | LITE | +0.367 | Manufacturing | Optical components |
| 5 | TTD | +0.353 | Services | Ad tech |

Sector concentration: 18/30 (60%) Manufacturing, 6/30 (20%) Services. The model is heavily long on the AI/semiconductor/hardware theme.

**Interpretation:**

1. ✅ **Live performance MATCHES Exp 5 walk-forward expectations.** Both showed mean monthly long-short returns in the +5–10% range; the live signals delivered +5.30% mean 5d L-S spread.

2. ✅ **100% positive days in the 7-day live window** is consistent with the 100% positive months from Exp 5. Statistical luck or real signal? Both windows are too small to be conclusive on their own, but consistency between the two orthogonal validations strengthens confidence.

3. ⚠️ **Long picks heavily outperform short picks.** Long avg +5.32%, short avg only +0.02% (basically market neutral). This means the alpha is mostly in the "find the winners" half of the rank distribution, not the "find the losers" half. Implication: long-only is the right deployment for retail accounts (avoiding short borrow costs and Robinhood's PDT rules); the ~110% LS return from Exp 5 was inflated by a market that happened to fall — the *consistent* alpha source is long-side picking.

4. ⚠️ **Sector concentration is real and worth flagging.** 60% Manufacturing in TOP 30 means a sector-wide downturn (e.g., semiconductor cycle, supply chain shock) would hit the portfolio hard. The sector-neutral label transform helps the *training* signal but doesn't constrain the final portfolio composition. Consider adding a sector cap (max 30% per sector) at the strategy level.

5. ⚠️ **The 1.21% L-S day on 2026-04-15** is a hint of what "average" days look like. Recent days have been unusually high-signal — the long-run average should land in the +1–3% per 5d range, not +5%. **Don't extrapolate the 8-day window to "+250% annualized."**

6. **Realistic annualized estimate from this 8-day window**: a long-only strategy holding the daily Top 30 with average 5-day net return of +5.12% per pick implies, with overlapping daily entries (5-day avg holding period) and ~6% turnover/day, somewhere around **40-80% annualized excess return** before accounting for execution slippage beyond the 20bps modeled. Subtract another ~10-20% for slippage in real Robinhood execution, and a realistic forward expectation is **20-50% annualized**.

**Decision:** The live pipeline is operational and performance matches the backtest. **Begin formal paper trading**: continue running the daily pipeline, log signals, score after 5 trading days, and accumulate at least **30 signal days** (≈6 weeks) before considering any real-money deployment.

**Pipeline runtime cost:**
- Daily data update: ~12 seconds (`download_polygon.py`)
- Signal generation: ~40 seconds (`generate_signals.py`) — full LambdaRank training + prediction
- Total daily ops: < 1 minute, can be cron-scheduled after US close

---

### Experiment 7 — News sentiment factors (Polygon Benzinga)
**Date:** 2026-04-27
**Hypothesis:** News carries fast event-driven information that complements slow technical features. Polygon provides AI-classified sentiment per article per ticker, ~150-700 articles/day with ~20% S&P 500 coverage. Adding 9 news factors (count, sentiment, attention z-score, etc.) on top of Alpha158 should lift IC by 0.005-0.01 by capturing earnings beats, FDA decisions, M&A announcements, etc., that price-only features miss until after the move.

**Setup:**
- Downloaded all news for 1279 trading days (2021-06-01 → 2026-04-24): **711,445 articles, 851 MB on disk**.
- 9 daily features per ticker: `news_count_1d`, `news_count_5d`, `news_sent_1d`, `news_sent_5d`, `news_sent_change`, `news_pos_ratio_5d`, `news_neg_ratio_5d`, `news_attention_z`, `news_silence_dummy`.
- PIT-aligned: each article belongs to the trading session whose close is the next 4pm ET after publish time.
- Built `Alpha158WithNews` Qlib handler (167 features = 158 Alpha158 + 9 news).
- Trained LambdaRank with sector-neutral 5d label, both single-window AND 12-month walk-forward.

**Result (single-window, 2025-04 → 2026-04):**

| Config | IC | Long Ann (cost) | Long IR | Long DD | LS Ann | LS IR |
|---|---|---|---|---|---|---|
| Ref: LR + Alpha158 | +0.0446 | +75.6% | +3.39 | -9.0% | +98.1% | +3.38 |
| Exp 7: + 9 News features | +0.0442 | +74.2% | +3.25 | -9.0% | +93.7% | +3.16 |

**Result (walk-forward, 12 monthly OOS):**

| Config | Mean IC | Consistency | Long compound | LS compound | Avg DD |
|---|---|---|---|---|---|
| Ref: LR + Alpha158 | +0.0453 | 100% (12/12) | +73.7% | +109.5% | -3.86% |
| **Exp 7: + 9 News features** | +0.0449 | 100% (12/12) | **+78.8%** | +109.5% | **-3.56%** |

**Feature importance — model's actual usage of news features:**

| Feature | LightGBM gain | Used by model? |
|---|---|---|
| `news_count_5d` (attention) | 13.8 | ✅ yes, slightly |
| `news_attention_z` (anomalous attention) | 6.7 | ✅ yes |
| `news_count_1d` (attention) | 6.3 | ✅ yes |
| `news_sent_1d` (sentiment direction) | **0.0** | ❌ |
| `news_sent_5d` | **0.0** | ❌ |
| `news_sent_change` | **0.0** | ❌ |
| `news_pos_ratio_5d` | **0.0** | ❌ |
| `news_neg_ratio_5d` | **0.0** | ❌ |
| `news_silence_dummy` | **0.0** | ❌ |

→ Total importance share captured by all 9 news features: **0.2%** of model gain. **Model used attention features but completely ignored sentiment direction.**

**Interpretation:**

1. ⚠ **News sentiment did NOT add detectable IC.** Single-window: -0.0004 IC. Walk-forward: -0.0004 IC. Within noise floor; both numbers indistinguishable.

2. 🟡 **However, news features delivered modest risk improvement** — long compound return +5.1pp better, drawdown 8% smaller. These are within 1-year noise but consistent across single-window and walk-forward, suggesting a small real effect.

3. ❌ **Model completely ignores sentiment polarity** (positive/negative/neutral). Six of nine features got 0.0 gain. Three attention features got non-trivial usage. The implication: in a liquid US large-cap universe, *the existence of news* matters slightly (probably as a regime / quality flag), but *the directional content* of that news adds nothing the price hasn't already captured.

4. 📊 **Why sentiment failed — three possible reasons:**
   - **EMH at large-cap level**: prices respond to news within seconds. By the close of the trading session in which an article is published, Alpha158 momentum/reversal features have already absorbed the move.
   - **Polygon sentiment is right-skewed** (9% positive vs 1.4% negative). The signal has low variance, hard for ML to discriminate.
   - **3-bucket sentiment is too coarse** for cross-sectional ranking. Whether "positive" news is bullish or bearish for a stock depends on its current price level, expectations, etc. — context the labels don't carry.

5. 💡 **The interesting tail signal: news_count as a regime modifier.** The 3 attention features being used (and only those) suggests the right next experiment isn't "more news features" but **interaction features**: does the model's other signal change reliability when news count is high? E.g., does momentum work less well on high-news days?

6. **Verdict:** **News factors do NOT promote to production C model.** Production stays at LambdaRank + Alpha158 only. News data remains valuable for the Codex narrative briefing pipeline (where it informs the human about *why* picks were chosen) but adds no quantitative alpha at the model level.

---

## Open Questions & Backlog

The previous decision to use Exp 4 was based on a single-window observation that didn't survive walk-forward. The 12-month evidence overwhelmingly favors C: 100% positive-IC months, 2× the returns, equal drawdown.

**Updated portfolio expectations (based on 12-month walk-forward of variant C, with 20bps round-trip cost):**

| Metric | Value (extrapolated annualized) | Notes |
|---|---|---|
| Long-only excess return vs market | **+74% / year** | Compounded across 12 monthly OOS slices |
| Long-short return | **+110% / year** | Requires margin account; not for Robinhood |
| Mean monthly IC | +0.045 | Consistent across all 12 months |
| Worst month IC | +0.014 (Aug 2025) | Even worst month was positive |
| Worst month long return | -0.5% (Feb 2026) | Single-month max DD ≈ -3-5% |
| Months profitable (long) | 11/12 | Only Feb 2026 was slightly negative |

These numbers are still optimistic (no slippage modeling, no survivorship adjustment, single market regime). Realistic expectation: **30-50% annualized excess return** with **15-25% max drawdown over a multi-year period**.

---

## Open Questions & Backlog

### High priority — must resolve before any real capital
1. ~~Walk-forward validation~~ ✅ DONE in Experiment 5.
2. ~~Live signal generation~~ ✅ DONE in Experiment 6. Pipeline operational; Top/Bottom 30 picks generated daily; SPY filter in place.
3. **Continue paper trading for 30+ signal days.** Currently 8 days, all positive. Need 30+ days for statistical confidence before real capital.
3a. **Survivorship bias quantification.** Compare backtested return on (current S&P 500) vs (S&P 500 as of test start date). Likely overstates by 2–5%.
3b. **Stress test on adverse market period.** Find a 1-2 month historical window with sharp market correction. Subscribe to Tiingo for longer history to enable this.
3c. **❌ [REJECTED 2026-04-28 by N=1132 evidence] Sector cap.** L47 showed concentration 40-50% is the **best**-performing tier, beating <30%. A 30% per-sector cap would push us out of the sweet spot. **NEW hypothesis to test**: cap kicks in only above 60% concentration (the truly extreme cases). Dropped from production roadmap pending more data.

3d. **🚨 [PROMOTED 2026-04-28 from L43] Top-score-based position sizing.** High-confidence days (top_score > 0.383) had +4.60% mean 5d vs low-confidence +2.41% — a **2.2pp lift**. Build a sizer: trade base size when score ≤ 0.383, 1.5× when > 0.42, 0.5× when < 0.30. This is an inversion of L26/L37 and the highest-ROI portfolio-level change after the Exp 5 walk-forward.

3e. **🆕 [DEFERRED to forward validation] Rule-1 BUY (consensus_long_worked predictor).** L53. Already implemented in `signals/<date>.csv` as `rule_buy_high_precision` flag. Need 30-60 days OOS before triggering 1.5× sizing on these picks. Track via `experience/track_rules.py`.

3f. **🆕 [DEFERRED to forward validation] Rule-2 AVOID (model_correct short predictor).** L52/L53. First case where the model + Codex's narrative layer found real shorting alpha. Already implemented as `rule_avoid_high_precision` flag. Need 30-60 days OOS validation before any short-side deployment. **First plausible candidate for shorts in any version of this strategy.**

### Medium priority — feature engineering
4. ~~Fundamentals~~ ✅ DONE in Exp 4 + Exp 5. **Result: under LambdaRank, fundamentals HURT** (-50% returns). REJECTED for production. Lesson: feature value is loss-function-dependent.
4a. ~~News sentiment factors~~ ✅ DONE in Exp 7. **Result: did NOT improve IC** (-0.0004 in walk-forward). Modest DD improvement but within noise. Model ignored sentiment direction entirely; only attention/count features got non-zero gain. Production stays at Alpha158 only. News data still useful for Codex briefings.
4b. **Interaction features instead of additive features.** Exp 7 hinted that `news_count_5d` is used as a regime modifier (signal "this is a noisy day"), not as a directional signal. Build interaction terms like `momentum × (1 + news_attention_z)` so the model's confidence is *modulated* by news intensity. This is a different mechanism than just adding raw news features.
4c. **Daily Short Volume factor.** Polygon endpoint `/stocks/v1/short-volume` is included on Starter — daily FINRA-reported short selling. Build `short_ratio_5d_ma` and `short_ratio_change` features. **Different mechanism from fundamentals/news** — short volume is a quantitative behavioral signal, not a narrative signal. Worth testing despite Exp 4/7 failures.
4c. **Minute-level intraday features.** 5y of minute aggregates available. Compute opening-30-min return, intraday VWAP deviation, time-of-day momentum decay as additional daily features.
4d. **News sentiment factor.** Benzinga sentiment labels available pre-tagged via `/v2/reference/news`. Build aggregate sentiment score (positive_news_5d - negative_news_5d) per ticker.
4e. **Try fundamentals with different loss functions (MSE+Fund variant B was actually decent).** Per Exp 5, fundamentals lifted MSE consistency from 67% → 83%. Combine with regime-aware ensembling: use C in good signal months, B in bad signal months.
5. **Sector dummies as features (vs. as label transform).** Let model learn sector × momentum interactions instead of forcing sector neutrality.
6. **Macro state features.** VIX level, 10y-2y yield spread, dollar index. Helps model adapt to regime.
7. **Earnings calendar.** Avoid holding through earnings (event risk), or specifically trade pre/post-earnings drift.

### Low priority — exploration
8. **Try LSTM/Transformer on Alpha360.** Only after fundamentals are in. Adds ~2-3 days of training time but might capture sequential patterns LightGBM misses.
9. **Russell 1000 universe.** Adds mid-caps where alpha is theoretically larger but data quality drops.
10. **Higher frequency (4h, 1h) models.** Requires Polygon Advanced ($79/mo).

### Infrastructure debt
11. **Daily incremental update job.** Cron scheduled `download_polygon.py` to run after US close (~5pm ET).
12. **Backtest reproducibility.** Set random seeds across LightGBM + numpy + pandas; results currently vary ±0.005 IC between runs.
13. **MLflow → SQLite backend.** File backend will get slow with hundreds of experiments.

---

## Lessons Learned (Running List)

### About US markets vs. CN markets
- **L1.** US large-cap returns at 1-day horizon are dominated by noise. Profitable signals exist at 5–20 day horizons; daily prediction is closer to a coin flip.
- **L2.** Cap-weighted indices (SPY, QQQ) have built-in momentum bias from Mag-7 concentration. Equal-weight indices (RSP) are the fair benchmark for ML strategies that select 30+ stocks.
- **L3.** Sector dispersion in US markets is large and persistent. Removing sector beta via neutralization can destroy alpha if you don't replace it with something better (e.g., sector-relative momentum).

### About modeling
- **L4.** **Loss function choice matters enormously when features are noisy.** LambdaRank → MSE is a 22× IC lift on the same features. This was the single biggest finding of all experiments.
- **L5.** Validation loss flat ≠ no signal exists. The CSRankNorm + MSE setup makes validation L2 ≈ 1.0 by construction (label is N(0,1)); meaningful improvement is in the 4th decimal place. Don't over-rely on training curves; use IC on a true held-out set as the source of truth.
- **L6.** Ensemble of mixed-quality models is worse than the best single model. Average only useful components.

### About execution & infrastructure
- **L7.** Survivorship bias is the #1 risk of casual quant work. Always note when a backtest universe is "current constituents" vs "point-in-time."
- **L8.** Test periods of <2 years cannot distinguish model alpha from regime luck. Walk-forward validation across multiple market regimes is mandatory before committing capital.
- **L9.** `min_cost: $1` for cost model is fine for $100K accounts but underestimates impact for <$10K accounts where bid-ask spread (often $0.05–$0.10) becomes significant fraction of position size. For real Robinhood deployment with small accounts, halve all expected returns.

### About data providers
- **L10.** Polygon's $29 Starter is the right entry point: unlimited rate + clean data outweighs the 5y history limit. Tiingo $10 is a worthy add-on for backfilling history once a strategy is proven.
- **L11.** Polygon's published feature matrix understates what Starter actually delivers. Empirical audit (2026-04-26) revealed access to: minute aggregates (5y), full quarterly financials (49 line items), daily short volume (FINRA), short interest, news sentiment, and related-companies graph. These together are worth $50-100/mo at competing providers.
- **L12.** **Point-in-time alignment is non-negotiable for fundamentals.** Use SEC `filing_date` (when the data became public), not `period_end_date`. The gap is typically 30–60 days and using `end_date` produces silent lookahead bias that inflates backtest IC by 0.005–0.02.

### About what makes a good factor (Experiment 4 insights)
- **L13.** **Adding features can lower IC even when the features are heavily used by the model.** Exp 4 had 11 of top 16 features as fundamentals, yet IC dropped 0.045 → 0.029. The model's *willingness* to use a feature ≠ that feature *improving* the prediction objective.
- **L14.** **Fundamentals act as a regularizer in noisy ML pipelines.** Adding slow-moving features dampens the technical signals' explosiveness — both the upside (peak IC) and the downside (max drawdown). Net effect: same IR but smoother equity curve. For real-money trading where psychology matters, smoother is usually better.
- **L15.** **In a noisy 1-year test window, IC differences of ±0.02 are not statistically significant.** The IC standard error with ~250 daily observations and IC volatility of 0.15-0.20 is roughly 0.01-0.013. So 0.029 vs 0.045 is "different" but well within the noise floor — a different test year could easily reverse the ranking.
- **L16.** **Risk metrics (drawdown, IC volatility) are more reliable than mean IC at small sample sizes.** Mean IC has ~30% measurement error at 1 year; drawdown improvement of 19-33% is much more likely to be a real effect.

### About walk-forward validation (Experiment 5 insights)
- **L17.** **Walk-forward validation can completely overturn single-window conclusions.** Exp 4 single-window: "fundamentals reduce DD from -9% to -7.3%". Exp 5 walk-forward: "fundamentals don't reduce DD at all (-3.9% vs -4.0%) but cost 50% of returns". The takeaway: **never adopt a strategy from a single backtest window, period.** Walk-forward is the minimum bar.
- **L18.** **Feature value is loss-function-dependent.** Fundamentals helped under MSE (consistency 67% → 83%) but hurt under LambdaRank (IC 0.045 → 0.026). The two loss functions process feature information differently, and a feature set can be a regularizer for one and a noise source for another. Always validate features in combination with the chosen loss.
- **L19.** **IC consistency (% of months with positive IC) is a better stability metric than IC standard deviation.** Variant C: IC std 0.27 across months but 100% positive consistency. Variant A: IC std 0.16 but only 67% consistency. C is better for live trading despite higher std — it never has a "lose money" month, while A loses 4 months per year.
- **L20.** **Even the best models have 1-2 "dead months" per year where alpha disappears.** Feb 2026 was hard for all 4 variants. This is structural, not a model defect — it implies position sizing must allow for 1-2 months of zero or negative excess return without forced liquidation.
- **L21.** **LambdaRank's edge is asymmetric — much larger in high-signal months than low-signal months.** Sept 2025: C captured 2× MSE's alpha. Feb 2026: both ~0. This suggests LambdaRank has higher *capacity* to exploit available alpha rather than higher *creation* of alpha. Conclusion: LambdaRank is the right loss when there *is* signal in the features; not magic that creates signal where none exists.

### About live signal generation (Experiment 6 insights)
- **L22.** **Long-side picking is where the alpha is, not short-side.** Across 8 live signal days, long picks averaged +5.32% (5d) while short picks averaged +0.02%. The Exp 5 long-short backtest's high returns were partly driven by a market that fell on certain days, allowing shorts to profit. The **persistent** alpha source is "find tomorrow's winners," not "find tomorrow's losers." Implication: long-only deployment captures most of the alpha at half the operational complexity (no margin, no borrow, no PDT risk).
- **L23.** **The model has a strong AI/semiconductor sector bias in 2026.** 60% of TOP 30 = Manufacturing (most of these are semiconductors and AI hardware). Sector-neutral rank in *training* doesn't constrain *deployment* portfolio composition — the trained model still picks names it has learned to associate with momentum, and in 2026 that means tech hardware. A simple max-30%-per-sector cap at portfolio construction would diversify without changing the model.
- **L24.** **Live performance ≈ backtest performance is the strongest validation signal.** Exp 5 walk-forward predicted ~5–10% per-month L-S spreads; Exp 6 live signals delivered +5.30% mean 5d L-S over 8 days (annualized rate matches expectation). When backtest and live numbers agree across orthogonal validation methodologies, the underlying signal is much more likely real than from either alone.
- **L25.** **8 days of live signals is statistically meaningless on its own** — it's only useful as a *consistency check* with the prior backtest. With 8 observations, the standard error of mean L-S spread is ±2-3 percentage points; a "true" mean of +5% could easily produce 8 days ranging from +1% to +9%. Need 30+ signal days minimum before drawing performance conclusions independently.

### About the daily decision journal (early observations)
- **L26.** **Tentative pattern (n=7, NOT statistically significant): model self-confidence correlates NEGATIVELY with realized 5d L-S spread.**
  - Top score correlation: −0.63
  - Score spread (top−bot): −0.61
  - Top-30 avg vol: −0.81
  - Top-30 avg 20d momentum: −0.73
  - Hypothesis: when the model is loudly confident (top score >0.5, picking high-vol momentum names), it may be over-fitting to a momentum spike that has already crested. When it picks "boring" names quietly, it does better. If this pattern holds at n=30+, **a confidence-based position sizer that *reduces* size on extreme-confidence days would be counterintuitive but profitable**. Watch this one.
- **L27.** **The act of writing a daily journal entry is itself a discipline.** Even when the auto-fields are mechanical, forcing yourself to type a `### Notes / lesson learned` paragraph creates a feedback loop: "what was unusual today?" Long term this builds the same kind of pattern recognition a human discretionary trader develops — but anchored to data, not to memory.

### About news sentiment as a quant feature (Experiment 7 insights)
- **L28.** **News sentiment direction adds zero IC on liquid US large caps with technical features present.** Polygon's per-ticker AI-classified sentiment, fed as 6 directional features, got 0.0 LightGBM gain in Exp 7. Both single-window and walk-forward agree: prices already reflect news within seconds for stocks of this size. The "news alpha" academic literature works on small caps; on S&P 500 it's table stakes.
- **L29.** **News *attention* (count, z-score) — not sentiment polarity — is the only news signal the model finds useful.** Three count-based features got 0.2% of total feature gain. Hypothesis: news intensity acts as a "regime flag" telling the model how much to trust other signals. This suggests the right way to use news is **as an interaction term** (modulating other signals' confidence), not as additive features.
- **L30.** **Right-skewed sentiment data limits its signal value.** Polygon's labels are 6× more often "positive" than "negative". Low variance in the signal makes cross-sectional ranking uninformative. Future: try filtering for absolute-conviction sentiment ("STRONG-POS / STRONG-NEG only"), or using sentiment_reasoning text to build richer features via LLM.
- **L31.** **A failed quant feature can still be a successful infrastructure asset.** News data didn't move the model, but it powers the human-readable daily briefings (Codex pipeline) — those briefings are a critical risk-management tool even if the underlying data doesn't quantitatively predict returns. Don't conflate "useless for ML" with "useless full stop."

### About multi-agent workflow (Codex co-work setup)
- **L32.** **Two-agent setup with file-based handoff scales effort, not capability.** Claude (deep strategy + code) + Codex (high-volume narrative) lets each agent operate within their strength. The contract is: Codex never modifies models; Claude never does bulk LLM summarization. Communication via `coordination/inbox/`, `coordination/outbox/`, `coordination/shared/`. All markdown — auditable, replayable, version-controllable.
- **L33.** **First Codex insight (2026-04-27): the AVOID side has more news/model conflicts than the BUY side.** Across 12 signal days, BUY picks with STRONG-NEG news max out at 3-4/day; AVOID picks with STRONG-POS news regularly hit 9-13/day. Three possible explanations: (a) Polygon's right-skewed sentiment (6× more "positive" labels) inflates the count artificially; (b) the model finds names to short that look bullish in the media (genuine alpha); (c) the model is wrong on its AVOID side and shouldn't be shorted. Pending Codex's deep-dive task to disambiguate. This finding alone validates the multi-agent architecture: Claude wouldn't have noticed without reading 720 individual news entries.

### About the first model loss day (Apr 27 reality check)
- **L34.** **The model can be wrong, and Apr 27 was the first proof.** TOP 30 picks (anchored on Apr 24 signals) returned -0.29% vs SPY +0.17%. After 11 days of positive (or near-zero) L-S spreads and the +5%/day average from paper trading, this was the first day the model meaningfully underperformed. **Apr 21 also went negative on 3d (-2.37%).** With 11 scored days now, we have 2 negative days (18%) — already higher than the 0% from the original 7-day window. This is normal — Exp 5 walk-forward predicted ~1 dead month per year, and consistent positive performance in Exp 6 was always going to revert.
- **L35.** **Codex's news-flag warnings on 2026-04-24 were predictive.** Of 10 conflicts Codex flagged (BUY+STRONG-NEG, AVOID+STRONG-POS), the realized 1-day returns matched Codex's call 6/10 times (60% — meaningfully above the 50% null). Most striking: TTD (model #5 BUY) carried Codex-tagged NEG news ("revenue slowed, Amazon competition") and dropped -3.46%; COHR (model #1) and LITE (model #4) both flagged with "crowded growth momentum" language and dropped -4.3% and -2.5%. **The Codex narrative layer caught risks the quantitative model missed.** This is the strongest single piece of evidence so far that the multi-agent setup adds real value beyond cost savings.
- **L36.** **Apr 24 had the highest conflict count (10) of all 12 backfilled days — and was the first signal day to lose money.** This is correlation, not causation, with N=1, but it's exactly the kind of pattern worth tracking. Hypothesis to test once we have more data: **on days where Codex flags >8 conflicts, the model's L-S spread is meaningfully lower than on calm-news days.** If true, conflict count becomes a position-sizing signal — reduce exposure on high-conflict days.
- **L37.** **The "loud confidence is bad" pattern (L26) found a confirming data point.** Apr 24 had the highest top-score (+0.39) and highest top-bot spread (0.84) of any signal day. It also lost money. With the prior 7-day correlation of -0.6 to -0.8 between confidence and realized return, today's outcome strengthens that hypothesis. We now have 8 scored days with confidence data. Sample is still small but pattern is consistent.
- **L38.** **The biggest losers today were exactly the model's most confident picks.** COHR (#1, score +0.39) -4.3%, LITE (#4, score +0.37) -2.5%, TTD (#5, score +0.35) -3.5%. These are the names the model "loved most." DELL (#2) and APA (#3) — also high confidence — held up better. The pattern: when the model loves a high-vol momentum name (66% avg vol on Apr 24's TOP 30), it's often catching the top of a momentum spike that's about to mean-revert. This converges with L26 and is now becoming a real strategy concern.
- **L39.** **Sector concentration risk materialized.** Apr 24 had 60% Manufacturing in TOP 30 (highest of any day). Today's 3 worst performers (COHR, LITE, TTD — though TTD is Services) are all part of the AI/tech-hardware narrative the model crowded into. If we'd had a 30% sector cap, the realized return would likely have been less negative. The Q3 backlog item to add a sector cap just got more urgent.

### About the experience library (Experiment 8 onwards)
- **L40.** **Single-day evidence overfit our intuition — early experience-library data refutes our Apr 24 reading.** After Apr 24's loss with 4 BUY+STRONG-NEG picks, the obvious lesson seemed "trust news warnings on momentum BUY". With N=6 conflict_buy_news_negative cases across the broader sample, **the mean 5d return is +9.1% with 83% win rate** — the OPPOSITE of what Apr 24 suggested. The model is *correct* to BUY despite negative news on average; Apr 24 was the rare exception. **This is exactly what the experience library is built to prevent — N=1 inference disasters.**
- **L41.** **The backfill design choice (filter to ~9 cases per signal day) gives ~1500 cases in 6 months.** This is enough for meaningful conditional statistics (each quadrant cell will have N=50-300). Smaller filters (3 cases/day) under-sample the tails; larger (60 cases/day, all picks) drown the signal in noise. The "interesting picks only" filter is the right balance.
- **L42.** **Codex's narrative layer + Claude's structured layer = the experience library's two halves.** Claude builds the YAML frontmatter (rank, score, sector, vol, returns, verdict) and the Polygon catalyst section automatically. Codex writes the prose narrative + lesson_tag. **Stats run on the structured layer; pattern discovery runs on the narrative layer.** Neither alone is sufficient.

### About the experience library — first findings from N=1132 scored cases (6 months, Oct 2025 – Apr 2026)

The 6-month backfill produced 1189 cases (1132 with realized 5d returns). For the first time we have statistically meaningful conditional cuts. Key findings:

- **L43.** **The Apr 24 "model loved high-confidence picks then lost" interpretation was wrong.** Across 1132 cases, **high-confidence days (top_score > 0.383) for BUY picks averaged +4.60%** vs low-confidence days +2.41% — the opposite of what L26/L37 hypothesized. **Confidence IS a real signal, just not on Apr 24.** L26 was N=7-day inference; with N=597 BUY cases the pattern reverses cleanly. Promote: **higher top_score correlates with higher realized 5d return** (lift of ~2.2pp).
- **L44.** **News STRONG-NEG on a model BUY does NOT predict failure.** The clean stats: STRONG-NEG news on BUY picks had mean +1.95% (N=99). Only 18% had losses >5%. The model is still right on average even when news disagrees. **Apr 24 was a 1-in-10 day where the conflict actually mattered**, but the typical conflict-buy-news-negative pick is mildly profitable. The Codex narrative layer is good at *flagging* conflicts but bad at *predicting which conflicts matter*.
- **L45.** **Big winners and big losers are nearly indistinguishable on the structured features.** The 221 picks with >+5% gains and the 114 picks with >-5% losses had the SAME median: top_score 0.39 vs 0.38, model_rank 3, sector concentration 57%, news count 2. Only one feature meaningfully differentiated: **annual vol — winners had 81%, losers had 73.5%.** Counter-intuitive: high-vol picks won more in absolute terms. This is the "fat tail" finance signature — vol is a predictor of *magnitude*, not *direction*.
- **L46.** **Recent 20-day momentum is NOT a pick-quality predictor (within the model's selections).** Across momentum quartiles, mean 5d return was Q1 +3.3%, Q2 +4.6%, Q3 +2.7%, Q4 +3.3% — **no monotonic pattern**. The model already factors in momentum; conditioning on its quartile of momentum doesn't add information. The "buying the top of momentum" hypothesis (L37) is **falsified at the cross-section** by 1132 cases. Apr 24's TTD/COHR/LITE losses were idiosyncratic, not systematic.
- **L47.** **Sector concentration sweet spot is 40-50%, not <30%.** Surprising: when TOP 30 has 40-50% in one sector, mean 5d = **+4.96%** (best of all tiers). Below 40% concentration: only +0.16% (N=27, very low). Above 50%: +3.3%. Hypothesis: **moderate sector tilt is a confidence signal** ("the model has a clear theme"). Extreme diversification (no sector dominant) means the model is unsure → underperforms. **This rejects L39's "add a 30% sector cap" recommendation as currently scoped.** A sector cap would force us OUT of the best-performing concentration tier (40-50%). New hypothesis: cap might be useful only above 60% concentration.
- **L48.** **The 6 case-types have meaningfully different profiles, but most are profitable.**

| case_type | N | mean 5d | win% | what it means |
|---|---|---|---|---|
| `solo_buy` (top-5 + no news) | 167 | **+4.41%** | 67% | Model's purest signal — performed best |
| `consensus_buy` (top-5 + STRONG-POS) | 331 | +3.46% | 59% | Both agree → moderate +ve, but news adds little |
| `conflict_buy_news_negative` | 99 | +1.95% | 61% | Conflict shaves ~2.5pp off but stays positive |
| `solo_avoid` (bot-5 + no news) | 318 | +0.26% | 44% | Model **wrong on AVOID** — these names actually went UP |
| `conflict_avoid_news_positive` | 195 | -0.24% | 50% | News right, model wrong (news predicted up, prices were flat) |
| `consensus_avoid` (bot-5 + STRONG-NEG) | 22 | -0.73% | 55% | Both agree shorts → still mostly went up; small N |

**The biggest insight here is the AVOID side.** Across all 535 AVOID cases, mean 5d = **+0.10%** — the bottom-30 picks **don't go down on average**. They drift up slightly. **The Robinhood long-only deployment is the right architecture** because the model has no real shorting alpha in this universe/period. The +110% L-S backtests in Exp 5 were partly a function of the universe drifting up, not the model finding actual shorts.

### Codex contributions (2026-04-28) — case narratives + tag-conditioned cuts

Codex completed the 1189-case narration task. Each case now has a 2-3 sentence narrative + a controlled-vocabulary lesson_tag. Running `summarize_experience.py` with tag conditioning revealed extreme bimodality in BUY-side outcomes that pre-tag cuts couldn't see:

**BUY side, tag-conditioned (cleanest cuts in the project so far):**

| Tag | N | mean 5d | win% |
|---|---|---|---|
| `consensus_long_worked` | 169 | **+10.14%** | 100% ⭐ |
| `solo_buy_pure_technical` | 105 | **+9.92%** | 100% ⭐ |
| `model_correct_despite_news_warning` | 49 | +5.95% | 100% |
| `outlier_one_off_event` | 7 | +22.03% | 100% |
| `news_event_dominated` | 30 | +2.22% | 57% |
| `noise_no_clear_attribution` | 71 | -1.82% | 30% |
| `solo_buy_no_catalyst_failed` | 50 | -6.13% | 0% |
| `consensus_long_failed` | 110 | **-6.71%** | 0% 💀 |

→ The same model picks split into two **mirror-image populations** (+10% / -7%). Tags identify which is which after the fact.

**AVOID side, tag-conditioned (refutes L48):**

| Tag | N | mean 5d | short win% |
|---|---|---|---|
| `model_correct_despite_news_optimism` | 72 | **-3.59%** | 100% ⭐ |
| `news_overrides_avoid_on_unexpected_strength` | 74 | **+3.18%** | 0% |
| `consensus_short_worked` | 9 | -4.52% | 100% |
| `noise_no_clear_attribution` | 304 | +0.14% | 46% |
| `sector_beta_dominated` | 66 | +0.61% | 41% |

→ The AVOID "no alpha" finding (L48) was wrong **at the aggregate level only**. With tags, AVOID splits cleanly: ~33% really do fall, ~33% really do rise, ~33% drift in noise. **L48 underestimated the model.**

### About tag-conditioned analysis (lessons L49-L52)

- **L49.** **Codex's qualitative tags create cleaner partitions than any quantitative cut.** Quantitative conditional cuts (top_score, momentum, sector concentration) yielded ±2pp differences. Tag-conditioned cuts yield **±17pp** differences (consensus_long_worked +10% vs consensus_long_failed -7%). The narrative layer captures something the structured features don't — likely "is this name riding a real catalyst or fighting it." The combination of structured + narrative is more powerful than either alone (validates L42 with concrete numbers).
- **L50.** **`solo_buy_pure_technical` is the model's purest alpha source.** When the model picks a top-5 BUY and there's no news to confound, the realized 5d return is +9.92% with 100% win rate (N=105). This is the **cleanest signal in the project** — pure technical conviction with no news noise. **Production implication: prioritize solo_buy positions with full sizing; downsize news-conflicted picks.**
- **L51.** **`consensus_long_failed` and `consensus_long_worked` look identical ex-ante but flip on outcome.** Both have the same news+model agreement structure; one mints +10%, the other loses -7%. The only way to distinguish them is **after the fact**, by reading the actual catalyst text + market response. This is the irreducible noise in the system. Production implication: when both model and news agree, you're betting on whether the agreement is "the market is about to wake up" or "the agreement is already priced in." **No cheap way to call this in advance** — it's the thing forward-only experimentation will eventually map out.
- **L52.** **L48 was wrong about AVOID.** The aggregate "+0.10%" hid two clean populations: 72 cases where model was right + news was wrong (5d -3.59%) and 74 cases where news was right + model was wrong (5d +3.18%). **The AVOID side has real shorting alpha — we just couldn't see it without Codex's tag classification.** Production implication (paper-trading-only for now): on shorts, only execute when the model's AVOID *agrees with* the recent news pattern (`consensus_short_worked` + `model_correct_despite_news_optimism` = 81 cases, mean -3.7%, 100% win rate). Skip when news contradicts the AVOID — those names rise on average.

### Bug fixes from Codex code review

- **B1 (2026-04-28, Codex flag):** `build_cases.py` was attaching the Polygon catalyst text only when an article appeared on offset=0 (the signal date itself), missing 311 cases where news was 1-5 days old. Fixed to capture the most recent article across the full 5-day window. **Codex caught this from reading the data, not the code** — the report cited 311 cases with `news_count > 0` but empty catalyst sections. Bug confirmed and fixed in 5 minutes.

### Forward-validation rules from Codex (Experiment 9)

Codex completed two P0 discrimination tasks: can we predict ex-ante which `consensus_long` cases will be `worked` vs `failed`, and which `conflict_avoid` cases will follow the model vs the news? Honest answer: **broad features fail to discriminate (L51 confirmed), but two narrow high-precision rules survived in-sample**:

#### Rule 1 — BUY: Fresh catalyst after pullback

```
IF rank ≤ 30
AND news_sentiment IN [STRONG-POS, POS]
AND ret_5d_pre ≤ -1.79%       (recent pullback)
AND pub_lag_days ≤ 1          (catalyst is fresh, ≤ 1 day old)
THEN classify as "high-quality consensus long; candidate for 1.5× sizing"
```

In-sample: 44 fires, mean **+6.27% / 5d**, win rate 89%, 20 big wins (>5%) vs 3 big losses. **Lift over BUY-side baseline: +2.80pp.** Recall 19% (the rule fires rarely; that's the price of high precision).

#### Rule 2 — AVOID: Same-day bullish catalyst on a name that already ran

```
IF rank ≥ 474                 (bottom 30)
AND news_sentiment = STRONG-POS
AND pub_lag_days < 1          (catalyst hit today)
AND pub_hour_utc ≥ 16          (catalyst hit after Asia-Europe close, near US close)
AND ret_5d_pre > 0            (already ran on the news)
THEN classify as "model is correctly fading the optimism; short-side candidate"
```

In-sample: 35 fires, mean **-3.03% / 5d** (i.e. the stock falls), short win rate 80%, 8 big wins (drops >5%) vs 0 big losses. **Lift over AVOID-side baseline: -3.07pp** (this is alpha, not noise).

#### Status — paper-track only, NOT live-trading rules yet

These rules were **discovered in-sample** by trying many cuts and selecting the best. **In-sample precision (91% / 90%) is biased upward by selection.** Per Codex's recommendation, forward-validate for 30-60 days before any production sizing.

#### Implementation in pipeline (2026-04-28)

- `build_cases.py` schema extended: cases now carry `pub_lag_days`, `pub_hour_utc`, `most_recent_published_utc`, `rule_avoid_high_precision`, `rule_buy_high_precision`.
- `signals/generate_signals.py` extended: live signal CSVs now carry the same 5 fields. Daily summary report adds a "🔬 Codex high-precision rules" section that lists the picks where rules fire.
- `experience/track_rules.py` (new): forward-validation tracker. Reports in-sample vs OOS precision/PnL. Run after the daily routine to monitor whether rules survive forward.

#### Lessons (L53-L55)

- **L53.** **Both rules require fresh news (`pub_lag_days ≤ 1`).** The BUY rule needs the news to be < 1 day old; the AVOID rule needs same-day intraday news. **Stale news is irrelevant on both sides.** Production implication: the news *recency* dimension was missing from our prior analysis (L29 only had attention z-score and counts) — recency is a stronger signal than volume.
- **L54.** **Honest discrimination has low recall.** Rule 1 fires on 44 of 597 BUY-side cases (7%). Rule 2 fires on 35 of 535 AVOID-side cases (6.5%). High precision comes with low recall — we can't replace the model's full pick set with rules. The rules are *additive* (mark some picks as higher-conviction), not *substitutive*.
- **L55.** **The "Codex finds something quantitative analysis can't" pattern repeats.** L43-L48 came from quantitative cuts and produced ±2pp differentials. L49-L52 came from Codex tags and produced ±17pp differentials. L53-L54 came from a *combined* read (Codex's narratives + structured features). Each layer of human/AI judgment compounds: pure quant < quant + tags < quant + tags + recency. **There is no obvious end to this layering** — every additional layer of context adds a few pp of edge.

### About reality after the 6-month backfill (Apr 28-29 reality checks)

- **L56.** **Two consecutive losing days on the same cohort exposed real model concentration risk.** Apr 27 → Apr 28: TOP 10 mean -3.48%, dominated by the same AI/semiconductor names that hurt on Apr 24. The model retrained but kept loving the same names (COHR, LITE, DELL, SMCI, AMD, VRT, MPWR all in TOP 10 both days). **Daily retraining did NOT diversify away the cohort risk.** This is a structural finding: when the model is "right" about a sector being attractive, it stays right (in its training data) even as the live market mean-reverts. **The diversification we assumed daily retraining provides may not exist.** Codex queued to investigate via pick-overlap metric.
- **L57.** **Cumulative paper-trade L-S spread is +1.23% per 5d window across 120 days, with 65% positive days.** This is **dramatically lower than the +5.30% we saw in the 7-day Apr 8-16 window** that defined our early expectations. Walk-forward (Exp 5) showed +110% LS compounded over 12 months because it weighted differently; the per-pick paper-trade metric is the more honest number for forward live deployment. **Realistic expectation: +1-2% per 5d window L-S, not +5%.** Annualized this is still meaningful (~50%+ if held sustained, less with friction) but the exuberance has been corrected.
- **L58.** **First OOS data point on Codex's BUY rule was negative (1d).** Apr 27 BUY rule fired on COIN (-1.31%), HOOD (-2.24%), BX (+0.96%) — 1-day mean -0.86%. This is 1 day of a 5-day rule, so wait for Apr 27 to score on May 2. Don't conclude. But: the first OOS data point not validating in-sample expectation is the standard shape of overfit rule discovery. **In-sample 91% precision doesn't survive OOS in finance** — the question is whether it survives at 60-70% (still useful) or collapses to baseline (overfit).

### About persistence, cohort risk, and the circuit breaker (Codex's 4-task batch, 2026-04-29)

Codex completed 4 follow-up tasks: cohort investigation, pick-overlap metric, sector concentration check, earnings calendar. Three findings significantly changed how we think about portfolio construction:

- **L59.** **Daily retraining is NOT diversifying — TOP 30 overlap averages 76%.** Across 125 consecutive day-pairs: TOP 30 overlap mean=75.8% (median 76.7%), TOP 10 mean=65%, BOT 30 mean=31.3%. **The "fresh decision each day" assumption is false on the long side; the model is closer to a 5-day-held position with daily ~25% rebalancing.** AVOID side IS diversifying (BOT 30 only 31% overlap), which makes intuitive sense — the model has weak short conviction. Implication: when the long cohort goes against us, daily retraining will NOT save us; the same names persist. **The cohort risk is structural, not coincidental.**

- **L60.** **Sector concentration HHI alone is not predictive — but PERSISTENCE × HHI × LOSING is.** Across 120 scored days, HHI correlation with 5d L-S return is +0.04 (essentially zero). Top-quartile HHI averaged +1.82% L-S vs +2.00% for bottom-quartile — no signal. So a naive "if concentration > 60%, cut size" rule would have hurt. The correct reading: high concentration that is currently working is fine; high concentration that just lost money is the actual risk shape. **L47's "40-50% sector concentration is optimal" finding survives** because that statistic is unconditional; the new finding is that concentration BECOMES bearish only when combined with overlap and loss.

- **L61.** **Cohort investigation: 2-day Apr 27/28 losses were SECTOR UNWIND, not info lag.** For the 8 affected names, Polygon news had explicit AI-chip-pullback language only for AMD and SMCI. COHR/LITE/DELL/VRT/MPWR/TTD lost without any company-specific catalyst — they fell because the AI/optics cohort all moved together. **Apr 28 actually INCREASED Manufacturing concentration to 73% (22/30) despite the Apr 27 loss** — the model retrained but stayed in the trade. Historical precedent is mixed: 2 prior cohort-loss windows (Dec 23-29, Apr 7-8) recovered strongly within 5d; 1 (Jan 27-28) continued losing -7.9% over 5d. **Without more evidence, "cohort momentum unwind" is a real failure mode but recovery is the modal outcome.**

- **L62.** ⚠️ **The Codex circuit breaker, when backtested on 125 historical days, would NOT have helped.** Backtest of Codex's proposed 4-condition rule (prior 1d <-1%, overlap ≥70%, sector ≥60%, repeated cohort 1d <0):
  - Fired 14 times historically (11.2% of days)
  - Mean realized 5d return on fire days: **+1.15%**
  - Mean realized 5d return on non-fire days: **+1.37%**
  - **Difference: -0.21pp, statistically meaningless.**
  - Of 14 fires, only 3 had truly bad 5d outcomes (Nov 6 -2.6%, Nov 11 -6.6%, Mar 20 -5.6%). The other 11 recovered.
  - **Skipping all circuit-breaker days would have cost cumulative ~16pp of gains, not saved them.**

**The circuit breaker is implemented and visible in daily summaries (now lights up 🚨 when triggered) — but it is a DIAGNOSTIC TOOL, not a trading rule.** When it fires, it tells the trader "you're in a high-cohort-risk situation" so they can override discretionarily, but the historical evidence says following the model on those days has been profitable on average. This is the **clearest example yet of "intuitively right ≠ statistically right"** in this project.

- **L63.** **Earnings calendar built from news (Codex P1).** 1638 (date, ticker) earnings events extracted from 262 news files; 504 high/medium-confidence rows. Coverage: 75% of S&P 500 has at least one earnings event in the past year. **Now a feature/filter we can use.** Codex flagged: separate scheduled-future earnings from retrospective earnings result articles, and improve ticker-from-headline mapping for cleaner data.

### About the BSX legal case + circuit-breaker subrule (Codex's 2026-04-30 batch)

Apr 28 fired the circuit breaker; Apr 29's actual +3.24% rebound proved L62 right (don't trust the CB as a trading rule). But Codex's two follow-ups produced concrete, narrower rules with much better in-sample shape than the original CB.

- **L64.** **BSX missed-veto: legal/securities-fraud STRONG-NEG news IS a real veto signal, even though L44 said "STRONG-NEG on BUY barely matters."** Apr 22's BSX (rank #9 BUY) had clearly visible STRONG-NEG news about a class-action lawsuit. The model bought it; it fell -9.9% over 5d. Codex's narrower veto: `BUY + STRONG-NEG + legal keywords (lawsuit, class action, SEC, investigation)` — distinct from generic STRONG-NEG (macro, valuation worries). **Implementation note**: signal CSV doesn't carry catalyst text yet, so the implemented `rule_buy_legal_veto` uses a conservative proxy `STRONG-NEG + news_neg_count_5d ≥ 2` (repeated negative coverage usually = real legal/lawsuit story, not single-article noise). Real keyword filter requires adding catalyst text to the CSV — queued for next session.

- **L65.** **Cohort-overheat is a PRE-LOSS diagnostic, structurally different from CB (POST-loss).** Codex's proposed pre-loss rule: `top10_sector_pct ≥ 70% + top10_pos_count ≥ 5 + top10_pre5d_mean > 0 + top10_strong_neg_count = 0`. This fires BEFORE the cohort breaks, when the model is over-concentrated in a hot sector with no warning news. **Apr 30's signal has this flag lit right now** (TOP 10 is 70%+ Manufacturing, hot 5d, all POS news). Whether it predicts loss or just continued rally is the live test.

- **L66.** **CB sub-rule (Codex finding) — captures 3/3 historical bad fires with 1 false positive across 14 fires.** When the 4-condition CB fires AND `cohort_3d_prior > 0 AND top10_pre5d > 3% AND repeated_cohort_1d > -4% AND news_neg_pct ≥ 12%`, the loss likely continues. When CB fires but these subrule conditions don't hold, the cohort typically recovers. **Apr 28 classifies as RECOVERY** under the subrule — matching the actual Apr 29 +3.24% rebound. **This is the first sub-rule that turns the CB from a useless trading signal into a discriminating one.** In-sample: skipping CB-fires that match the subrule would have given +12.2pp cumulative lift across 14 fires (vs the original CB which lost 16pp).
  - Honest caveat: 3/3 with 1 FP across N=14 is fragile. Forward validation needed before any sizing change. The bad fire signature was *shallow first crack on a still-hot cohort with ≥12% negative news*, not the deepest losses (which often capitulate and bounce).

- **L67.** **L62's "stubborn cohort recovers" pattern survives — but L66 finds the exception class.** Most CB fires recover (11/14 historically). But L66 carves out the 3 that don't. The mental model is now: **CB fire alone = not actionable. CB fire + subrule = candidate veto.** Implementation: `cb_classification` field in signal summary now reads BAD/RECOVERY/no_fire automatically.

### Now-live signal CSV / summary fields (per-day machine-readable diagnostics)

Each `signals/<date>.csv` now carries: `news_count_5d, news_pos_count_5d, news_neg_count_5d, news_sentiment, most_recent_published_utc, pub_lag_days, pub_hour_utc, most_recent_title, most_recent_reasoning, most_recent_keywords, rule_avoid_high_precision, rule_buy_high_precision, rule_buy_legal_veto`. Each daily summary `.md` carries CB fire status, CB subrule classification (BAD/RECOVERY), cohort-overheat tail-risk flag, and the picks the rule_buy_high_precision flagged. **Three rules + two diagnostics now run live each day** — all paper-track only until 30+ days OOS.

### About L65→L69 reframing + L68 legal-keyword reality (Codex's 2026-05-01 batch)

- **L68.** **The "BUY+STRONG-NEG+legal keywords" set is structurally cleaner than the proxy, but is NOT a confirmed loss-predictor.** Codex did the keyword backtest: proxy fires (BUY+STRONG-NEG+news_neg_count≥2) match real legal keywords only **52% of the time** (60/115). Surprise: among 107 scored proxy candidates, those WITH legal keywords averaged **+3.13% / 5d** with 26.8% news_won, while those WITHOUT averaged **+0.44% / 5d** with 39.2% news_won. **Real legal keywords correlate with HIGHER returns, not lower.** Hypothesis: legal news (lawsuits, class actions, SEC investigations) is mostly priced in by the time the article publishes, and the model's BUY rank captures the post-news rebound. **The veto rule still fires today (May 1: SMCI flagged for class action lawsuit news), but Claude has implemented it as a paper-track audit flag, not a sizing reduction.** Real-keyword regex now uses catalyst text fields (added to signal CSV in this session); proxy retired in code.

- **L69.** **L65 (cohort-overheat) backtest verdict — IS a real tail-risk warning, NOT a contrarian continuation signal.** Codex's quantitative test on 125 days, 48 fired & scored: mean 5d **-0.28% vs +2.29% baseline**, big-loss rate **18.8% vs 2.7%**, t≈-3.69, p≈0.0002. **This is the FIRST risk-warning rule that survived a backtest** — L62 and the original CB both got refuted. But the framing matters: TOP 10 still averages +1.18% on fire days; this is a **fat-tail / lower-expectancy alert**, not a hard veto. Codex proposed renaming `cohort_overheat_pre_loss` → `cohort_overheat_tail_risk`; Claude implemented. Daily summary now reads: "when fired, mean 5d -0.28% vs +2.29% baseline, big-loss rate 18.8% vs 2.7%; not a hard veto, sizing alert." **Apr 30 is the live counterexample to watch** (1d so far +2.08%, full 5d verdict on May 7).

- **L70.** **The pattern flipped: not all risk warnings get faded.** L59-L67 showed that intuitive "concentration is dangerous" rules kept being refuted by data (CB original lost money, sector caps would have hurt, persistence wasn't bearish). L69 is the **first concentration-related rule that holds up statistically** — the difference is the *combination* of conditions (sector >70% + 5+ POS news + hot pre-5d + no STRONG-NEG warning), not any single one. Lesson: it took 8 lessons of false starts to find the right multi-condition framing. **Quantitative validation is non-negotiable.** Without Codex's backtest, we'd have either retired L65 prematurely (matching the L62 vibe) or kept it on intuition (matching the original CB mistake). Both would have been wrong.

### About first-real-OOS rule verdicts (May 7 update — both flagship rules failed)

After the 5-day forward window closed for both rules that survived in-sample backtests, we now have actual out-of-sample data points. **Both failed.**

- **L71.** ❌ **`rule_buy_high_precision` (Codex's flagship) — first OOS test FAILED.** Apr 27 fired on COIN, HOOD, BX. Full 5d realized: COIN +3.21%, HOOD **-8.81%**, BX +2.26%, **mean -1.12%**. In-sample expectation was +6.34% / 5d. **OOS miss: -7.46pp.** The TOP 30 broad portfolio for Apr 27 returned +7.46% over the same period — so the supposed "high-precision" subset *underperformed the broad pool by ~9pp*. This is the canonical signature of in-sample search overfit: with N=35 historical fires and many candidate cuts tested, picking the best-looking rule produces a 91% in-sample precision that doesn't generalize. **Do NOT size up on this rule.** Need 5-10 more OOS fires to determine if Apr 27 is an outlier or if the rule is truly overfit.

- **L72.** ❌ **`cohort_overheat_tail_risk` (L69) — first OOS test FAILED.** Apr 30 fired the L69 diagnostic. In-sample expectation: TOP 30 fired-day mean -0.28% / 5d vs +2.29% baseline (-2.57pp lift, t≈-3.69, p≈0.0002). **OOS realized: TOP 30 +4.93%, TOP 10 +5.46%.** TOP 30 BEAT baseline by +2.64pp. Cohort kept rallying — INTC +16%, AMD +15%, STX +14%, DELL +10% over 5 days. **L69 also fails on first OOS data point.** Honest interpretation: even rules that look statistically significant in-sample (p<0.001!) can fail on the first OOS check in this regime. This is a strong reminder of **regime sensitivity** — the in-sample period had cohort capitulations that didn't bounce; the May 1-7 week had cohort rallies that didn't crack.

- **L73.** 🤔 **The model's BROAD ranking keeps winning even when special-precision rules fail.** Cumulative paper trade is now **+1.38% / 5d, 66% positive days** across 127 scored days — **better than the +1.20% / 65% from a week ago.** The May 1-7 rally was very profitable for the broad TOP 30 strategy. **Pattern**: every rule we've tried to overlay on top of the model's ranking has either failed OOS or been statistically indistinguishable from baseline. The brutal lesson: **rule mining on top of a working model is a false-discovery factory.** The simple "trade TOP 30, equal weight, hold 5d" strategy keeps beating every clever overlay we add. **Production recommendation reverts to: just trade the model's broad ranking; treat all rules as research curiosities, not actionable.**

- **L74 (meta).** Three takeaways from the L71-L72 OOS failures:
  1. **In-sample multi-condition rule search has a high false-discovery rate**, even with backtests showing p < 0.001. The selection bias from trying many cuts dwarfs the apparent significance.
  2. **The 30-60 day OOS validation Codex repeatedly suggested** is not a formality — it's the only thing standing between us and over-confident rule deployment. Today vindicates Codex's caution.
  3. **The model itself has real, persistent alpha** (+1.38% / 5d, 66% positive). The challenge isn't finding the model; it's resisting the urge to over-engineer on top of it.

---

**Last updated:** 2026-05-07 (evening) — **The week of OOS truth.** Both flagship rules (rule_buy_high_precision, L69 cohort-overheat) FAILED their first OOS tests. L71-L74 added. Apr 27 BUY rule mean OOS -1.12% vs in-sample +6.34%. Apr 30 L69 OOS TOP 30 +4.93% vs in-sample -0.28%. **The model's broad ranking is now demonstrably +1.38%/5d, 66% positive over 127 days — better than it was a week ago. All rule overlays have failed.** Production stays at "just trade TOP 30, equal-weight, 5d hold" — no rules.
**Next session:** (1) Continue forward-tracking — need 5-10 more fires of each rule before drawing strong conclusions on overfit vs noise; (2) Codex task: what features actually correlate with realized 5d return ACROSS regimes? Maybe the in-sample-best rules are regime-conditional, not universal. (3) Today (May 7) fired CB+RECOVERY subrule — score on May 14 to add another OOS data point.
