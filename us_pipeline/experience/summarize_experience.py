"""Aggregate the case library into quadrant statistics.

Reads all case .md files, parses YAML frontmatter, computes mean realized returns
+ verdict distributions per case_type / per verdict / per various conditional cuts.

Output: prints a structured report to stdout AND updates the README's
"Current findings" section.

Usage:
  python us_pipeline/experience/summarize_experience.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "cases"
README = ROOT / "README.md"


def parse_case(fp: Path) -> dict | None:
    text = fp.read_text()
    m = re.match(r"---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    front = m.group(1)
    out = {}
    for line in front.split("\n"):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v == "TBD":
            v = None
        else:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k.strip()] = v
    out["filename"] = fp.name
    return out


def load_all_cases() -> pd.DataFrame:
    cases = []
    for fp in sorted(CASES_DIR.glob("*.md")):
        c = parse_case(fp)
        if c:
            cases.append(c)
    return pd.DataFrame(cases)


def fmt(x, places=2):
    if x is None or pd.isna(x):
        return "  -- "
    return f"{x:+.{places}f}"


def report(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Total cases: {len(df)}")
    lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    lines.append(f"Scored (have ret_5d): {df['ret_5d_pct'].notna().sum()}")
    lines.append("")

    # Per case-type
    lines.append("=" * 90)
    lines.append("PER CASE TYPE")
    lines.append("=" * 90)
    lines.append(f"{'case_type':<35} {'N':>5} {'scored':>7} {'mean_5d':>9} {'std':>7} {'win%':>6} {'verdict_distribution'}")
    lines.append("-" * 130)
    for ct, g in df.groupby("case_type"):
        scored = g[g["ret_5d_pct"].notna()]
        n = len(g)
        ns = len(scored)
        if ns == 0:
            mean = std = winrate = float("nan")
        else:
            mean = scored["ret_5d_pct"].mean()
            std = scored["ret_5d_pct"].std()
            # Win = positive for BUY-side, negative for AVOID-side
            wins = 0
            for _, row in scored.iterrows():
                if row.get("model_action") == "BUY":
                    wins += int(row["ret_5d_pct"] > 0)
                elif row.get("model_action") == "AVOID":
                    wins += int(row["ret_5d_pct"] < 0)
            winrate = wins / ns * 100
        verdict_counts = scored["verdict"].value_counts().to_dict() if ns else {}
        verdicts_str = " ".join(f"{k}:{v}" for k, v in sorted(verdict_counts.items()))
        lines.append(f"{ct:<35} {n:>5} {ns:>7} {fmt(mean):>9} {fmt(std,2):>7} {winrate:>5.0f}% {verdicts_str}")
    lines.append("")

    # Per verdict
    lines.append("=" * 90)
    lines.append("PER VERDICT (only scored cases)")
    lines.append("=" * 90)
    scored = df[df["ret_5d_pct"].notna()]
    if len(scored):
        for v, g in scored.groupby("verdict"):
            mean5 = g["ret_5d_pct"].mean()
            n = len(g)
            lines.append(f"  {v:<25} N={n:>4}  mean_5d={fmt(mean5):>8}%")
    lines.append("")

    # The big quadrant matrix
    lines.append("=" * 90)
    lines.append("QUADRANT MATRIX — model action × news label, mean 5d return")
    lines.append("=" * 90)
    if len(scored):
        scored = scored.copy()
        # Bin news
        def bin_news(s):
            if s == "STRONG-POS":
                return "STRONG-POS"
            if s == "STRONG-NEG":
                return "STRONG-NEG"
            if s == "POS":
                return "POS"
            if s == "NEG":
                return "NEG"
            if s == "NO-NEWS":
                return "NO-NEWS"
            return "NEUTRAL"
        scored["news_bin"] = scored["news_sentiment"].apply(bin_news)

        for action in ["BUY", "AVOID"]:
            sub = scored[scored["model_action"] == action]
            if sub.empty:
                continue
            lines.append(f"\nModel action = {action}:")
            lines.append(f"  {'news_bin':<14} {'N':>5} {'mean_5d':>9} {'win%':>6}")
            for nb, g in sub.groupby("news_bin"):
                m = g["ret_5d_pct"].mean()
                if action == "BUY":
                    win = (g["ret_5d_pct"] > 0).mean() * 100
                else:
                    win = (g["ret_5d_pct"] < 0).mean() * 100
                lines.append(f"  {nb:<14} {len(g):>5} {fmt(m):>9}% {win:>5.0f}%")
    lines.append("")

    # Sector concentration analysis
    lines.append("=" * 90)
    lines.append("CONDITIONAL: high vs low sector concentration days (BUY side only)")
    lines.append("=" * 90)
    buy_scored = scored[scored["model_action"] == "BUY"] if "model_action" in scored.columns else pd.DataFrame()
    if not buy_scored.empty:
        thresh = buy_scored["sector_concentration_top30_pct"].median()
        hi = buy_scored[buy_scored["sector_concentration_top30_pct"] > thresh]
        lo = buy_scored[buy_scored["sector_concentration_top30_pct"] <= thresh]
        lines.append(f"  Median sector concentration: {thresh:.0f}%")
        lines.append(f"  HIGH concentration (>{thresh:.0f}%): N={len(hi)} mean_5d={fmt(hi['ret_5d_pct'].mean()):>8}%")
        lines.append(f"  LOW concentration (<={thresh:.0f}%): N={len(lo)} mean_5d={fmt(lo['ret_5d_pct'].mean()):>8}%")
    lines.append("")

    # High-confidence vs low-confidence buy
    lines.append("=" * 90)
    lines.append("CONDITIONAL: high vs low top_score days (BUY side only)")
    lines.append("=" * 90)
    if not buy_scored.empty and "top_score_of_day" in buy_scored.columns:
        thresh = buy_scored["top_score_of_day"].median()
        hi = buy_scored[buy_scored["top_score_of_day"] > thresh]
        lo = buy_scored[buy_scored["top_score_of_day"] <= thresh]
        lines.append(f"  Median top score: {thresh:.3f}")
        lines.append(f"  HIGH confidence (>{thresh:.3f}): N={len(hi)} mean_5d={fmt(hi['ret_5d_pct'].mean()):>8}%")
        lines.append(f"  LOW confidence (<={thresh:.3f}): N={len(lo)} mean_5d={fmt(lo['ret_5d_pct'].mean()):>8}%")
    lines.append("")

    return "\n".join(lines)


def update_readme(stats_text: str):
    if not README.exists():
        return
    content = README.read_text()
    marker_start = "### Quadrant statistics (auto-generated from cases)"
    marker_end = "### Confirmed patterns"
    if marker_start not in content or marker_end not in content:
        print("(README markers not found, skipping update)")
        return
    s = content.index(marker_start)
    e = content.index(marker_end)
    new_section = (
        f"{marker_start}\n\n"
        f"Run `python us_pipeline/experience/summarize_experience.py` to refresh.\n\n"
        f"```\n{stats_text}\n```\n\n"
    )
    new_content = content[:s] + new_section + content[e:]
    README.write_text(new_content)
    print(f"Updated README with latest stats.")


def main():
    df = load_all_cases()
    if df.empty:
        print("No cases yet. Run build_cases.py first.")
        return
    text = report(df)
    print(text)
    update_readme(text)


if __name__ == "__main__":
    main()
