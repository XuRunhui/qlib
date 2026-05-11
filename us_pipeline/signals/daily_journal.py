"""Daily decision journal — single growing markdown file.

For every signal date, append a structured entry capturing:
  1. Market context (SPY level, MA position, vol regime, risk gate)
  2. Model output (top/bot 5, sector tilt, score distribution, confidence)
  3. Realized performance (filled in once 5 trading days have passed)
  4. (Manual) lesson / observation field — left blank for human to fill

Ouput: us_pipeline/signals/JOURNAL.md (one file, monotonic-append)

The script is idempotent: re-running for the same date overwrites that date's
entry only if the realized performance is now available (so daily entries get
"upgraded" once their forward window matures).

Usage:
  python us_pipeline/signals/daily_journal.py                # all dates that have signal CSVs
  python us_pipeline/signals/daily_journal.py 2026-04-24     # one specific date
  python us_pipeline/signals/daily_journal.py --rebuild      # nuke and rebuild from scratch
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = "us_pipeline/data/qlib_bin"
SIGNALS_DIR = Path(__file__).parent
JOURNAL_PATH = SIGNALS_DIR / "JOURNAL.md"
SECTORS_CSV = Path(__file__).resolve().parent.parent / "data" / "instruments" / "sectors.csv"

ENTRY_START = "<!-- ENTRY:{date} -->"
ENTRY_END = "<!-- /ENTRY:{date} -->"


def _spy_context(target_date: date) -> dict:
    """Compute SPY's position and vol on the target date."""
    from qlib.data import D
    end = target_date.isoformat()
    start = (target_date - timedelta(days=300)).isoformat()
    df = D.features(["SPY"],
                    ["$close",
                     "$close / Ref($close, 1) - 1",
                     "Mean($close, 20)",
                     "Mean($close, 50)",
                     "Mean($close, 200)",
                     "Std($close/Ref($close,1) - 1, 20) * 16"],
                    start_time=start, end_time=end, freq="day")
    df.columns = ["close", "daily_ret", "ma20", "ma50", "ma200", "ann_vol_20d"]
    df = df.dropna()
    if df.empty:
        return {}
    last = df.iloc[-1]
    above_200 = last["close"] > last["ma200"]
    above_50 = last["close"] > last["ma50"]
    if above_200 and last["ann_vol_20d"] < 0.30:
        gate = "🟢 GO"
    elif above_200:
        gate = "🟡 REDUCED (vol elevated)"
    elif above_50:
        gate = "🟡 REDUCED (below 200-MA)"
    else:
        gate = "🔴 NO-GO"
    return {
        "spy_close": float(last["close"]),
        "spy_daily_ret": float(last["daily_ret"]) * 100,
        "spy_above_20ma_pct": (float(last["close"]) / float(last["ma20"]) - 1) * 100,
        "spy_above_50ma_pct": (float(last["close"]) / float(last["ma50"]) - 1) * 100,
        "spy_above_200ma_pct": (float(last["close"]) / float(last["ma200"]) - 1) * 100,
        "ann_vol_20d_pct": float(last["ann_vol_20d"]) * 100,
        "gate": gate,
    }


def _signal_summary(target_date: date, topk: int = 30) -> dict:
    """Read the signal CSV for the date and summarize."""
    csv = SIGNALS_DIR / f"{target_date.isoformat()}.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv).sort_values("rank")
    top = df.head(topk)
    bot = df.tail(topk).sort_values("rank", ascending=False)

    sector_counts = top["sector"].value_counts()
    sector_str = ", ".join(f"{s} {c} ({c/topk*100:.0f}%)"
                           for s, c in sector_counts.items())

    return {
        "n_universe": len(df),
        "top5_syms": top["instrument"].head(5).tolist(),
        "top5_scores": top["score"].head(5).round(3).tolist(),
        "bot5_syms": bot["instrument"].head(5).tolist(),
        "bot5_scores": bot["score"].head(5).round(3).tolist(),
        "top1_score": float(top["score"].iloc[0]),
        "score_spread": float(top["score"].iloc[0] - bot["score"].iloc[0]),
        "top_avg_vol_pct": float(top["ann_vol_20d"].mean()) * 100,
        "bot_avg_vol_pct": float(bot["ann_vol_20d"].mean()) * 100,
        "top_avg_ret_20d_pct": float(top["ret_20d"].mean()) * 100,
        "bot_avg_ret_20d_pct": float(bot["ret_20d"].mean()) * 100,
        "sector_tilt": sector_str,
    }


def _realized_perf(target_date: date, topk: int = 30) -> dict:
    """Look up realized 1d/3d/5d returns of the top/bot picks if data is available."""
    from qlib.data import D
    csv = SIGNALS_DIR / f"{target_date.isoformat()}.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv).sort_values("rank")
    top = df.head(topk)
    bot = df.tail(topk)
    instruments = sorted(set(top["instrument"].tolist() + bot["instrument"].tolist()))

    # Pull forward returns for the day after target_date
    end_d = target_date + timedelta(days=20)
    feats = D.features(instruments,
                       ["Ref($close, -2)/Ref($close, -1) - 1",   # 1d
                        "Ref($close, -4)/Ref($close, -1) - 1",   # 3d
                        "Ref($close, -6)/Ref($close, -1) - 1"],  # 5d
                       start_time=target_date.isoformat(),
                       end_time=end_d.isoformat(), freq="day")
    feats.columns = ["ret_1d", "ret_3d", "ret_5d"]
    feats = feats.reset_index()
    feats["instrument"] = feats["instrument"].str.upper()
    feats_target = feats[feats["datetime"] == pd.Timestamp(target_date)]
    if feats_target["ret_5d"].isna().all():
        return {"scored": False}

    top_u = top["instrument"].str.upper()
    bot_u = bot["instrument"].str.upper()
    top_join = feats_target[feats_target["instrument"].isin(top_u)]
    bot_join = feats_target[feats_target["instrument"].isin(bot_u)]

    out = {"scored": True}
    for col in ("ret_1d", "ret_3d", "ret_5d"):
        out[f"top_{col}"] = float(top_join[col].mean()) * 100
        out[f"bot_{col}"] = float(bot_join[col].mean()) * 100
        out[f"ls_{col}"] = out[f"top_{col}"] - out[f"bot_{col}"]
    out["top_5d_winrate"] = float((top_join["ret_5d"] > 0).mean()) * 100
    out["bot_5d_lossrate"] = float((bot_join["ret_5d"] < 0).mean()) * 100
    return out


def _verdict(realized: dict) -> str:
    if not realized.get("scored"):
        return "(awaiting 5-day forward window)"
    ls = realized.get("ls_ret_5d", 0)
    if ls > 4:
        return "🚀 Strong (L-S > 4%)"
    elif ls > 1:
        return "✅ Positive"
    elif ls > -1:
        return "🟡 Neutral"
    elif ls > -3:
        return "🔴 Negative"
    else:
        return "💀 Very negative (L-S < -3%)"


def _format_entry(target_date: date, ctx: dict, sig: dict, realized: dict) -> str:
    """Render one journal entry as markdown."""
    weekday = target_date.strftime("%A")
    lines = []
    lines.append(ENTRY_START.format(date=target_date))
    lines.append(f"## {target_date} ({weekday})")
    lines.append("")
    if not sig:
        lines.append("*No signal generated for this date.*")
        lines.append("")
        lines.append(ENTRY_END.format(date=target_date))
        return "\n".join(lines)

    # Market context
    lines.append("### Market context")
    if ctx:
        lines.append(f"- **SPY**: ${ctx['spy_close']:.2f} ({ctx['spy_daily_ret']:+.2f}% on day)")
        lines.append(f"- **vs MAs**: 20d {ctx['spy_above_20ma_pct']:+.1f}% · "
                     f"50d {ctx['spy_above_50ma_pct']:+.1f}% · "
                     f"200d {ctx['spy_above_200ma_pct']:+.1f}%")
        lines.append(f"- **Vol regime**: {ctx['ann_vol_20d_pct']:.0f}% annualized (20d realized)")
        lines.append(f"- **Risk gate**: {ctx['gate']}")
    else:
        lines.append("- *(SPY data not available)*")
    lines.append("")

    # Model output
    lines.append("### Model output")
    top_str = ", ".join(f"**{s}** ({sc:+.2f})"
                        for s, sc in zip(sig["top5_syms"], sig["top5_scores"]))
    bot_str = ", ".join(f"{s} ({sc:+.2f})"
                        for s, sc in zip(sig["bot5_syms"], sig["bot5_scores"]))
    lines.append(f"- **Top 5**: {top_str}")
    lines.append(f"- **Bot 5**: {bot_str}")
    lines.append(f"- **Sector tilt (top 30)**: {sig['sector_tilt']}")
    lines.append(f"- **Top-30 avg 20d-vol**: {sig['top_avg_vol_pct']:.0f}% "
                 f"(vs bot-30: {sig['bot_avg_vol_pct']:.0f}%)")
    lines.append(f"- **Top-30 avg 20d-momentum**: {sig['top_avg_ret_20d_pct']:+.1f}% "
                 f"(vs bot-30: {sig['bot_avg_ret_20d_pct']:+.1f}%)")
    lines.append(f"- **Confidence**: top score = {sig['top1_score']:+.3f}, "
                 f"top–bot spread = {sig['score_spread']:.3f}")
    lines.append("")

    # Realized
    lines.append("### Realized performance")
    if realized.get("scored"):
        lines.append(f"| Horizon | Top-30 | Bot-30 | L-S Spread |")
        lines.append(f"|---|---|---|---|")
        for h in ("1d", "3d", "5d"):
            lines.append(f"| {h} | {realized[f'top_ret_{h}']:+.2f}% | "
                         f"{realized[f'bot_ret_{h}']:+.2f}% | "
                         f"{realized[f'ls_ret_{h}']:+.2f}% |")
        lines.append("")
        lines.append(f"- **5d top-30 win rate**: {realized['top_5d_winrate']:.0f}%")
        lines.append(f"- **5d bot-30 loss rate**: {realized['bot_5d_lossrate']:.0f}%")
        lines.append(f"- **Verdict**: {_verdict(realized)}")
    else:
        lines.append(f"*Awaiting 5 trading days for forward returns to materialize.*")
    lines.append("")

    # Manual notes
    lines.append("### Notes / lesson learned")
    lines.append("*(fill in after observing the day)*")
    lines.append("")
    lines.append(ENTRY_END.format(date=target_date))
    return "\n".join(lines)


def _existing_entries() -> dict:
    """Return {date: (start_idx, end_idx)} for existing entries in JOURNAL.md."""
    if not JOURNAL_PATH.exists():
        return {}
    text = JOURNAL_PATH.read_text()
    entries = {}
    lines = text.split("\n")
    cur_date = None
    cur_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("<!-- ENTRY:") and s.endswith(" -->"):
            cur_date = s[len("<!-- ENTRY:"):-len(" -->")]
            cur_start = i
        elif s.startswith("<!-- /ENTRY:") and s.endswith(" -->"):
            d = s[len("<!-- /ENTRY:"):-len(" -->")]
            if cur_date == d and cur_start is not None:
                entries[cur_date] = (cur_start, i)
            cur_date = None
            cur_start = None
    return entries


def _read_journal_lines() -> list[str]:
    if not JOURNAL_PATH.exists():
        return []
    return JOURNAL_PATH.read_text().split("\n")


def _has_user_notes(lines: list[str], start: int, end: int) -> bool:
    """Check if user added text under '### Notes / lesson learned' beyond the placeholder."""
    in_notes = False
    for line in lines[start:end+1]:
        s = line.strip()
        if s == "### Notes / lesson learned":
            in_notes = True
            continue
        if in_notes:
            if s.startswith("<!--") or s.startswith("##") or s.startswith("###"):
                break
            if s and s != "*(fill in after observing the day)*":
                return True
    return False


def _extract_notes(lines: list[str], start: int, end: int) -> list[str]:
    out = []
    in_notes = False
    for line in lines[start:end+1]:
        s = line.strip()
        if s == "### Notes / lesson learned":
            in_notes = True
            continue
        if in_notes:
            if s.startswith("<!--") or s.startswith("##") or s.startswith("###"):
                break
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return out


def _generate_or_update(target_date: date, force_rebuild: bool = False):
    qlib.init(provider_uri=PROVIDER, region="us") if "qlib" not in sys.modules or not _qlib_inited() else None
    ctx = _spy_context(target_date)
    sig = _signal_summary(target_date)
    realized = _realized_perf(target_date) if sig else {}

    new_entry = _format_entry(target_date, ctx, sig, realized)

    if not JOURNAL_PATH.exists() or force_rebuild:
        if force_rebuild and JOURNAL_PATH.exists():
            JOURNAL_PATH.unlink()
        header = (
            "# Daily Decision Journal\n\n"
            "_Auto-generated by `us_pipeline/signals/daily_journal.py`. "
            "Re-running updates entries whose forward returns are now available. "
            "User notes under '### Notes / lesson learned' are preserved across regenerations._\n\n"
            "---\n\n"
        )
        JOURNAL_PATH.write_text(header + new_entry + "\n\n---\n\n")
        return "created"

    existing = _existing_entries()
    lines = _read_journal_lines()

    if target_date.isoformat() in existing:
        start, end = existing[target_date.isoformat()]
        # Preserve user notes if present
        old_notes = _extract_notes(lines, start, end)
        keep_notes = old_notes and any(
            l.strip() and l.strip() != "*(fill in after observing the day)*"
            for l in old_notes
        )
        if keep_notes:
            # Replace the placeholder with the saved notes inside the new entry
            new_entry = new_entry.replace(
                "### Notes / lesson learned\n*(fill in after observing the day)*",
                "### Notes / lesson learned\n" + "\n".join(old_notes),
            )
        # Splice the new entry into the file
        new_lines = lines[:start] + new_entry.split("\n") + lines[end+1:]
        JOURNAL_PATH.write_text("\n".join(new_lines))
        return "updated"
    else:
        # Append new entry — keep entries in chronological order
        # Find the right insert spot (before any existing entry with later date)
        insert_at = len(lines)
        existing_sorted = sorted(existing.items(), key=lambda kv: kv[0])
        for ed, (s, _e) in existing_sorted:
            if ed > target_date.isoformat():
                insert_at = s
                break
        block = "\n" + new_entry + "\n\n---\n"
        new_lines = lines[:insert_at] + block.split("\n") + lines[insert_at:]
        JOURNAL_PATH.write_text("\n".join(new_lines))
        return "appended"


def _qlib_inited() -> bool:
    try:
        from qlib.config import C
        return C is not None and getattr(C, "_provider_uri", None) is not None
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("date", nargs="?", help="Date YYYY-MM-DD; default = all dates with signal CSVs")
    p.add_argument("--rebuild", action="store_true", help="Wipe JOURNAL.md and regenerate from scratch")
    args = p.parse_args()

    qlib.init(provider_uri=PROVIDER, region="us")

    if args.date:
        d = pd.Timestamp(args.date).date()
        action = _generate_or_update(d, force_rebuild=args.rebuild)
        print(f"{action}: {d}")
    else:
        # Find all signal CSVs
        dates = []
        for csv_file in sorted(SIGNALS_DIR.glob("*.csv")):
            try:
                d = pd.Timestamp(csv_file.stem).date()
                dates.append(d)
            except Exception:
                continue
        if not dates:
            print("No signal CSVs found.")
            return
        if args.rebuild:
            print(f"Rebuilding journal for {len(dates)} dates...")
            for i, d in enumerate(dates):
                action = _generate_or_update(d, force_rebuild=(i == 0))
                print(f"  {action}: {d}")
        else:
            for d in dates:
                action = _generate_or_update(d, force_rebuild=False)
                print(f"  {action}: {d}")
        print(f"\nJournal: {JOURNAL_PATH}")


if __name__ == "__main__":
    main()
