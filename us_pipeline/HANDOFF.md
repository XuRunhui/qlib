# Project Handoff — US Equity Quant Pipeline

> **Read this first if you're a new Claude session picking up this project on a new machine.**
> This is a structured pointer to all the work done so far, what state things are in, and how to resume.

**Last handoff written**: 2026-05-10 (Sunday evening)
**Original developer**: Runhui Xu (NRA grad student, runhuixu@umich.edu)
**Original assistant**: Claude (Opus 4.7, 1M context)
**Companion agent**: Codex (via CLI, file-based handoff under `coordination/`)

---

## 🎯 What this project is

A daily-frequency US equity quant pipeline built on top of Microsoft Qlib. It:
1. Downloads S&P 500 daily OHLCV + news + fundamentals from Polygon.io
2. Trains a LambdaRank model on Alpha158 features with sector-neutral 5d labels
3. Generates daily TOP 30 / BOTTOM 30 picks
4. Tracks realized 5d returns and accumulates them in an "experience library" of 1230+ case files
5. Uses a second agent (Codex) for high-volume narrative summarization + rule discovery
6. Records every signal-day decision in a journal so we can review predictions vs reality

**Total accumulated knowledge**: ~74 numbered lessons (L1–L74), 9 named experiments (Exp 1–9), 1230 case files spanning 6 months, 134 signal days with 128 fully scored.

---

## ⚡ Quick start on the new machine (15 minutes from clone to running)

### 1. Clone the repo (your fork)

```bash
git clone git@github.com:XuRunhui/qlib.git
cd qlib
git checkout us-pipeline   # or whatever branch this was pushed to
```

### 2. Set up the environment (uv, Python 3.10)

```bash
# Install uv if not present: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python python3.10 --seed --python-preference only-managed .venv
source .venv/bin/activate
uv pip install -e . polygon-api-client python-dotenv fake-useragent lxml html5lib beautifulsoup4
```

### 3. Restore data from the zip

The git repo does NOT contain the large data files (news JSONs, fundamentals, raw price CSVs, qlib bin, parquets, mlruns). They are in a separate archive `us_pipeline_data.tar.gz` (~1.5 GB).

```bash
# Once you've copied the tar.gz to the new machine (e.g., from cloud / USB / scp)
tar -xzf us_pipeline_data.tar.gz -C .
# This restores:
#   us_pipeline/data/news/        (851 MB — 1289 JSON files, 2021-06 → today)
#   us_pipeline/data/fundamentals/ (257 MB — 503 ticker JSONs)
#   us_pipeline/data/raw/          (38 MB — 506 OHLCV CSVs)
#   us_pipeline/data/qlib_bin/     (30 MB — Qlib binary data)
#   us_pipeline/data/*.parquet     (news_factors, fundamental_factors)
#   mlruns/                        (290 MB — MLflow tracking)
```

### 4. Restore the `.env` file

The `.env` is NOT in git for security. Create it manually:

```bash
echo "POLYGON_API_KEY=<your-polygon-key>" > .env
```

Polygon plan: Stocks Starter ($29/month). Subscription is at [massive.com](https://massive.com).

### 5. Smoke test

```bash
source .venv/bin/activate
python -c "import qlib; qlib.init(provider_uri='us_pipeline/data/qlib_bin', region='us'); from qlib.data import D; print(D.features(['AAPL'], ['\$close'], start_time='2026-05-01').tail())"
# Should print last few AAPL closes; if it works, you're set
```

### 6. Catch the data up to today

```bash
# This pulls anything missing
python us_pipeline/download_polygon.py
python us_pipeline/download_news.py
python us_pipeline/to_qlib_bin.py
# Then generate any missing signal days
python us_pipeline/signals/generate_signals.py   # generates for the latest data day
```

---

## 📂 What's in this repo (after clone + tar restore)

```
qlib/                                  # microsoft/qlib + our editable install
├── us_pipeline/                       # OUR WORK — start here
│   ├── HANDOFF.md                     # ← YOU ARE READING THIS
│   ├── README.md                      # ⭐ Full research log: 9 experiments + 74 lessons
│   ├── download_polygon.py            # Polygon daily OHLCV downloader
│   ├── download_news.py               # Polygon Benzinga news downloader
│   ├── download_fundamentals.py       # Polygon quarterly financials
│   ├── compute_fundamental_factors.py # Build 13 PIT-aligned fund factors
│   ├── compute_news_factors.py        # Build 9 PIT-aligned news factors
│   ├── fetch_sectors.py               # SIC code → 9-bucket sector map
│   ├── to_qlib_bin.py                 # Convert CSVs → Qlib binary
│   ├── universe.py                    # Scrape S&P 500 ticker list
│   ├── sector_processor.py            # Custom Qlib processor: sector-neutral rank
│   ├── lgb_rank_model.py              # Custom LightGBM LambdaRank wrapper
│   ├── handler_alpha158_fund.py       # Alpha158 + fundamentals handler
│   ├── handler_alpha158_news.py       # Alpha158 + news handler
│   ├── workflow_lightgbm_us.yaml      # Reference Qlib workflow config
│   ├── sweep*.py                      # Experiment harnesses (Exp 1-3)
│   ├── exp4_run.py, exp5_walkforward.py, exp7_news.py  # Big experiments
│   ├── data/                          # ⚠ NOT IN GIT — restored from tar.gz
│   │   ├── instruments/sp500.txt      # 503 current tickers (IS in git)
│   │   ├── instruments/sectors.csv    # SIC mapping (IS in git)
│   │   ├── raw/                       # 506 OHLCV CSVs (NOT in git)
│   │   ├── qlib_bin/                  # Qlib binary data (NOT in git)
│   │   ├── news/                      # 1289 news JSONs (NOT in git)
│   │   ├── fundamentals/              # 503 fund JSONs (NOT in git)
│   │   ├── news_factors.parquet       # (NOT in git)
│   │   ├── fundamental_factors.parquet # (NOT in git)
│   │   ├── pick_overlap.csv           # IS in git (analysis output)
│   │   ├── circuit_breaker_*.csv      # IS in git
│   │   ├── earnings_calendar.csv      # IS in git
│   │   └── sector_concentration_hhi.csv # IS in git
│   ├── signals/                       # Daily signal output (all in git)
│   │   ├── generate_signals.py        # Main signal-generation script
│   │   ├── spy_filter.py              # GO/NO-GO risk gate
│   │   ├── paper_trade_log.py         # Scoring + cumulative metrics
│   │   ├── daily_journal.py           # Auto-update JOURNAL.md
│   │   ├── JOURNAL.md                 # ⭐ Daily decision log + manual notes
│   │   ├── <YYYY-MM-DD>.csv           # 134 ranked CSVs (Oct 2025 → May 2026)
│   │   ├── <YYYY-MM-DD>_summary.md    # Human-readable summaries
│   │   └── news_briefs/<date>.md      # Codex-generated per-day briefs
│   ├── experience/                    # Case-based experience library
│   │   ├── README.md                  # Schema + current stats
│   │   ├── build_cases.py             # Build cases from signals + news
│   │   ├── backfill_signals.py        # Generate historical signals
│   │   ├── summarize_experience.py    # Quadrant statistics
│   │   ├── track_rules.py             # Forward-validate rules
│   │   ├── backtest_circuit_breaker.py # Backtest CB rule
│   │   ├── cases/                     # 1230 case files (md, in git)
│   │   └── patterns/                  # Promoted patterns (empty currently)
│   └── coordination/                  # Claude ↔ Codex handoff
│       ├── README.md                  # Protocol contract
│       ├── inbox/                     # Tasks Claude → Codex
│       ├── outbox/                    # Responses Codex → Claude
│       └── shared/
│           ├── current_model_card.md
│           ├── open_questions.md
│           └── codex_findings.md      # ⭐ Codex's running insights
├── qlib/                              # Microsoft's qlib code (upstream)
├── mlruns/                            # MLflow tracking (NOT in git, in tar)
├── .env                               # NOT in git, manually recreated
└── pyproject.toml
```

---

## 🧠 What the project knows (cheat sheet)

If you only read 5 things, read these:

1. **`us_pipeline/README.md`** — the full research log. 74 numbered lessons, 9 experiments documented. The "Last updated" line at the bottom tells you the latest project state.

2. **`us_pipeline/coordination/shared/codex_findings.md`** — what Codex (the other agent) has discovered. Append-only, newest at top.

3. **`us_pipeline/signals/JOURNAL.md`** — daily decision log. Has per-day model output, news context, manual notes from the human user.

4. **`us_pipeline/experience/README.md`** — the case library's quadrant statistics. Tells you which patterns work and which don't.

5. **`us_pipeline/coordination/inbox/`** + **`outbox/`** — open tasks queued for Codex, plus the responses already received.

### The 10 most load-bearing lessons (out of 74)

- **L4** — LambdaRank beats MSE by 5× IC (single biggest finding)
- **L17** — Walk-forward validation is non-negotiable; single-window can overturn
- **L33** — Codex's AVOID-side news conflict observation validated the multi-agent setup
- **L43** — High model confidence → higher returns (+2.2pp lift)
- **L48** → **L52** — AVOID alpha exists but is bimodal; tag-conditioning reveals it
- **L62** — Stubborn cohort persistence is real alpha; "concentration is dangerous" intuitions get faded
- **L66** — CB subrule: when CB fires AND cohort still hot AND news_neg ≥12% → predicts bad outcome (3/3 in-sample, currently 1/2 OOS — RECOVERY classifications working, BAD untested)
- **L71** — `rule_buy_high_precision` FAILED first OOS test (-7.46pp miss)
- **L72** — L69 cohort-overheat FAILED first OOS test (+5.21pp wrong direction)
- **L73** — Simple TOP 30 broad strategy keeps winning; rule overlays keep failing OOS

### Current production stance

**Trade TOP 30 from the model, equal-weight, 5-day hold, ignore all rule overlays.**

Cumulative: 134 signal days logged, 128 scored, **66% positive days, +1.43% mean L-S spread / 5d**.

---

## 🛠 The daily routine (run after US close, ~5pm ET)

```bash
cd /path/to/qlib && source .venv/bin/activate

python us_pipeline/download_polygon.py             # ~12s, fetch new prices
python us_pipeline/to_qlib_bin.py                  # ~3s, rebuild Qlib bin
python us_pipeline/download_news.py                # ~5s for one new day
python us_pipeline/signals/spy_filter.py           # ~2s, GO/NO-GO
python us_pipeline/signals/generate_signals.py     # ~40s, today's picks
python us_pipeline/signals/paper_trade_log.py --all # ~5s, ingest + score
python us_pipeline/signals/daily_journal.py        # ~5s, update JOURNAL.md
python us_pipeline/experience/build_cases.py       # ~30s, refresh cases
python us_pipeline/experience/track_rules.py       # ~5s, rule OOS status
```

~2 minutes total. Cron-friendly.

Also weekly: patch SPY/RSP/QQQ benchmarks (they're not in the SP500 universe so download_polygon.py doesn't fetch them):

```python
# Run any time benchmarks are stale
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from polygon import RESTClient
c = RESTClient(os.environ['POLYGON_API_KEY'])
import pandas as pd
from datetime import datetime
for sym in ['SPY', 'RSP', 'QQQ']:
    bars = list(c.list_aggs(ticker=sym, multiplier=1, timespan='day',
                from_='2026-05-01', to='2026-12-31', adjusted=True, sort='asc'))
    with open(f'us_pipeline/data/raw/{sym}.csv') as f:
        existing = [line.split(',')[0] for line in f.read().splitlines()]
    for b in bars:
        d = datetime.utcfromtimestamp(b.timestamp/1000).date().isoformat()
        if d in existing: continue
        with open(f'us_pipeline/data/raw/{sym}.csv', 'a') as f:
            f.write(f'{d},{b.open},{b.high},{b.low},{b.close},{b.volume},{b.vwap},{b.transactions}\n')
"
# Then rebuild bin: python us_pipeline/to_qlib_bin.py
```

---

## 🤖 The Codex co-work protocol

This is a two-agent setup. **Claude** (you, the strategist) and **Codex** (a separate CLI agent that handles bulk narrative work) communicate exclusively via files in `us_pipeline/coordination/`.

- **Claude owns**: model code, experiments, decisions, README lessons.
- **Codex owns**: narrative writing, case classification, summaries, proposals.
- **Hard rule**: Codex never edits model code or runs experiments.

### How to start a Codex session on the new machine

Copy this prompt verbatim into a fresh Codex CLI session pointed at the repo:

```
You are operating as a co-worker in a quant research repo at /path/to/qlib. There is a separate Claude session that handles model code, experiment design, strategy synthesis, and final decisions. You handle high-volume narrative work — reading data, summarizing, classifying, and contributing your own observations.

Communication is file-based. No direct messaging.

ROLES:
  - Claude owns: model code changes, experiment design, production config decisions, README lessons promotion, what-to-build-next decisions.
  - You own: bulk narrative writing, classification, summarization, basic statistical sanity checks, and PROPOSING ideas Claude should consider.
  - You are encouraged to read code (us_pipeline/*.py) when it helps you understand the system.
  - You are encouraged to propose hypotheses, flag suspicions, suggest experiments. Append to coordination/shared/codex_findings.md.
  - Hard rule: never edit model code, never run experiments, never modify production config.

STEP 1 — Read in this order:
  1. us_pipeline/HANDOFF.md (this file)
  2. us_pipeline/coordination/README.md
  3. us_pipeline/coordination/shared/codex_findings.md (your prior findings)
  4. us_pipeline/README.md (skim Lessons L43-L74)
  5. us_pipeline/experience/README.md

STEP 2 — Pick the highest-priority task in inbox/ that doesn't already have an outbox/ entry.

STEP 3 — Execute end-to-end. Read code where helpful. Be quantitative. Be brief.

STEP 4 — When done:
  - Spot-check 2-3 case files you cited
  - Move the task to outbox/ with "## Codex response" appended
  - Append meta-finding to coordination/shared/codex_findings.md (newest first, above the marker line)
  - Include a "Codex proposals" paragraph with concrete file-path-cited suggestions

Begin.
```

---

## 🚦 Where we left off (2026-05-10)

### Last signal generated: 2026-05-08
TOP 10 for May 11 (Mon) trade: COHR, STX, CIEN, CNC, SMCI, INTC, AMD, DELL, MU, MCHP. 9 of 10 are Manufacturing. ⚠ Cohort-overheat flag lit (but L72 says fade — don't act on it).

### Live OOS rule status
| Rule | In-sample claim | OOS data points | Status |
|---|---|---|---|
| `rule_buy_high_precision` | +6.34% / 5d, 91% precision | Apr 27: -1.12% / 5d (FAIL by -7.46pp) | **OVERFIT, paper-track only** |
| `cohort_overheat_tail_risk` (L69) | -0.28% vs +2.29%, p<0.001 | Apr 30: +4.93% / 5d (FAIL by +5.21pp wrong direction) | **OVERFIT, paper-track only** |
| CB subrule RECOVERY (L66) | classify CB fires as recovery | Apr 28: +10.12% ✅, May 7: +4.04% 1d ✅ | **2/2 OOS — promising, keep tracking** |
| `rule_buy_legal_veto` (real keywords) | narrower than proxy | May 1: SMCI flagged, no scored data yet | **Awaiting OOS evidence** |

### Open Codex tasks (in `coordination/inbox/`)
- (none currently — last set was completed May 1)

### Open questions Claude wants Codex to investigate
- Does the L66 RECOVERY classification continue to hold OOS? (2/2 so far)
- Has the model's edge degraded in May vs April? (current week was +6.56% TOP 30 5d — outlier or sustained?)
- The "cohort momentum continues" thesis — is there a regime indicator that predicts when it will fail?

### Suggested next experiment
**Walk-forward retraining**: the current production model is from Apr 27 (the most recent training). It hasn't been retrained in 11 trading days. Test: retrain weekly vs daily, see if performance changes. If daily retraining doesn't help (we already showed top-30 overlap is 76%), maybe weekly is fine and saves compute.

---

## 💼 The 90-second strategy recommendation (for the human user)

If/when the user is ready to deploy real money:

1. **70% of total capital → VOO + QQQ + BRK.B** (dollar-cost average, never touch)
2. **30% → active sleeve in Robinhood**: TOP 30 from this model, equal-weight, 5-day hold, rebalance Mondays
3. **Trade everything the model says; ignore every rule overlay** (L71, L72 confirmed they fail OOS)
4. **Skip days when SPY filter says NO-GO**
5. **Per-stock stop-loss -10%; account drawdown -20% → halt for 1 month**
6. **Start with $500 → ramp to $3K over 3 months only if real-money returns match paper**

NRA tax status means federal capital gains = 0% (huge advantage). W-8BEN form for dividends → 10% withholding instead of 30%.

---

## 🔬 Infrastructure quick reference

### Polygon API ($29/mo Stocks Starter — verified entitlements)
- ✅ 5y daily/hour/minute OHLCV
- ✅ Full quarterly financials (49 line items)
- ✅ Daily short volume (FINRA)
- ✅ Benzinga news with per-ticker sentiment + reasoning
- ❌ Tick trades / real-time quotes (need Stocks Advanced $79)
- ❌ Options price aggregates (need Options Starter +$29)

### Model architecture
- LightGBM with `lambdarank` objective (NDCG-optimizing pairwise)
- 158 Alpha158 features (technical, OHLCV-derived)
- Sector-neutral cross-sectional rank label, quantized to 16 bins
- 5-day forward return: `Ref($close, -6) / Ref($close, -1) - 1`
- Train: rolling [2021-06-01, signal_date − 31 days]
- Validation: 30 days prior to signal date
- Universe: S&P 500 (~503 tickers, current constituents)

### Trading cost model
- Open 5 bps, close 15 bps = 20 bps round-trip
- Min cost $1 (Robinhood/Chase are $0)

---

## 🧾 Final checklist for new machine

- [ ] Clone fork: `git clone git@github.com:XuRunhui/qlib.git && git checkout us-pipeline`
- [ ] Install uv + Python 3.10 venv
- [ ] `uv pip install -e . polygon-api-client python-dotenv fake-useragent lxml html5lib beautifulsoup4`
- [ ] Copy `us_pipeline_data.tar.gz` to repo root, extract: `tar -xzf us_pipeline_data.tar.gz`
- [ ] Create `.env` with `POLYGON_API_KEY=...`
- [ ] Smoke test: `python -c "import qlib; qlib.init(...); print(D.features(['AAPL'], ['\$close'])...)"`
- [ ] Run daily routine to catch up to today
- [ ] Read `us_pipeline/README.md` Lessons section (especially L62, L71-L74)
- [ ] If using Codex too: copy prompt above, point Codex at the repo

After that, you're caught up and can resume from where Claude/Codex/user left off.
