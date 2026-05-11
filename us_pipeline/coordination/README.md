# Coordination Protocol — Claude ↔ Codex

> Two-agent co-work setup. **Claude does deep strategy, Codex does high-volume narrative work.** They never talk directly — they share state through files in this directory.

---

## Who does what

| Domain | Owner | Why |
|---|---|---|
| Strategy & model design | **Claude** | Needs full context of all experiments + code; expensive but rare calls |
| Pipeline code, debugging | **Claude** | Tight feedback loops, needs running shell |
| Experiment interpretation, README updates | **Claude** | Synthesis across many runs |
| Daily news summaries (per stock) | **Codex** | Reads ~30 stocks × 5 days of articles per signal day; high volume, low judgment |
| Routine status reports | **Codex** | Mechanical, cheap |
| LLM-driven analyses (sentiment reasoning, event classification) | **Codex** | Pure NLP work |
| Ad-hoc investigations Claude requests | **Codex** | Async, parallelizable |

**Hard rule:** Codex never modifies model code or experiment configs. Codex only reads data + writes narrative reports. Claude never tries to do bulk LLM summarization.

**Soft encouragement (added 2026-04-28):** Codex is a knowledgeable engineer in its own right. **It should read the model code, the experiment scripts, and the README to ground its narratives** in what the system actually does — not just guess from data fields. Codex is also encouraged to **propose its own ideas, hypotheses, and recommendations** — append them to `shared/codex_findings.md` or end-of-task outbox responses. **The decision authority remains with Claude** (which experiments to run, which features to add, what production config to ship), but Codex's suggestions are valuable inputs. Think of it as: Codex is a quant analyst, Claude is the PM/lead engineer. Both contribute to the conversation; the lead engineer decides.

---

## File contract

```
us_pipeline/coordination/
├── README.md         <- this file (the contract)
├── inbox/            <- tasks Claude → Codex (one md per task)
├── outbox/           <- responses Codex → Claude
└── shared/           <- continuously updated state both read
```

```
us_pipeline/signals/
└── news_briefs/      <- Codex auto-generates daily, Claude reads when relevant
    └── <YYYY-MM-DD>.md
```

### Inbox / outbox naming convention

`<YYYY-MM-DD>__<task-name>.md`  (double underscore separator)

When Codex finishes a task:
1. Move the inbox file → outbox with the same filename, **append** the response below the original task.
2. Or write a new file in outbox with the same name and prefix `--RESPONSE--` if you want to keep history.

### Shared state files

**`shared/current_model_card.md`** — Claude updates after each successful experiment. Codex reads this to know what the live model is doing.

**`shared/open_questions.md`** — Claude lists "things I'd like investigated when there's compute time." Codex picks one, investigates, drops a finding in `codex_findings.md`.

**`shared/codex_findings.md`** — Codex's running notebook of recurring observations. When Claude reads this and decides something is worth promoting, it goes into the main `README.md`'s Lessons section.

---

## Standard inbox task template

```markdown
# Task: <one-line title>

**From:** Claude
**To:** Codex
**Date created:** YYYY-MM-DD
**Priority:** P0 (blocking) / P1 (this session) / P2 (when free)
**Estimated tokens:** small (<5k) / medium (5-20k) / large (>20k)

## Inputs (file paths Codex should read)
- path/to/file1
- path/to/file2

## What to produce
<concrete description of expected output>

## Output location
<exact file path Codex should write>

## Format / template
<inline markdown template or pointer to one>

## Acceptance criteria
- [ ] criterion 1
- [ ] criterion 2

## Notes
<context that helps Codex make judgment calls>
```

---

## Standard recurring tasks for Codex

Codex should run these without prompting:

### 1. Daily news brief (after signals are generated)

**Trigger:** existence of a new `us_pipeline/signals/<date>.csv`
**Output:** `us_pipeline/signals/news_briefs/<date>.md`
**Inputs:**
- `us_pipeline/signals/<date>.csv` — top/bottom 30 picks
- `us_pipeline/data/news/<date>.json` — that day's market news
- Last 5 trading days of news files for context

**What to produce:** see `us_pipeline/signals/news_briefs/_TEMPLATE.md`

### 2. Weekly journal review (every Friday)

**Output:** `coordination/outbox/<YYYY-MM-DD>__weekly_review.md`
**Action:**
- Read the past week's `JOURNAL.md` entries
- Identify the day with biggest miss (model vs realized)
- Look up that day's news brief
- Hypothesize *why* the model was wrong (or right when news contradicted it)
- Append findings to `shared/codex_findings.md`

### 3. Pattern hunt (when Claude posts in `shared/open_questions.md`)

**Trigger:** new entry in `shared/open_questions.md`
**Action:** investigate, write finding to `outbox/`, append to `codex_findings.md`

---

## Conflict resolution

If Codex finds something that contradicts a Claude lesson in the main README:
1. Don't edit README directly
2. Write a clear note in `codex_findings.md` under "## CONFLICTS WITH README"
3. Claude will adjudicate next session

---

## Operational notes

- **Codex doesn't run heavy compute.** No retraining. No backtests. If a task requires that, escalate (write to outbox/ with "ESCALATE: needs Claude").
- **All paths are relative to repo root** (`/home/ruxu/qlib`).
- **Markdown is the lingua franca.** No YAML, no JSON in coordination files (they're for humans to skim).
- **Brevity wins.** Codex outputs should be skimmable. <300 words per finding unless asked otherwise.
