"""Build structured case files from existing signal CSVs + news + realized returns.

For each signal date, identifies "interesting" picks via filter rules and writes
one Markdown case per pick. Realized returns are auto-filled when the data exists.

Idempotent: rerunning updates verdicts/realized fields without overwriting any
manual narrative additions (Codex's later passes).

Usage:
  python us_pipeline/experience/build_cases.py                  # all signal dates
  python us_pipeline/experience/build_cases.py 2026-04-08 2026-04-24  # date range
  python us_pipeline/experience/build_cases.py --rebuild        # wipe and regenerate from scratch
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import qlib

ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = ROOT / "signals"
NEWS_DIR = ROOT / "data" / "news"
CASES_DIR = ROOT / "experience" / "cases"
SECTORS_CSV = ROOT / "data" / "instruments" / "sectors.csv"

PROVIDER = str(ROOT / "data" / "qlib_bin")

# Polygon sentiment → numeric
SENT_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


@dataclass
class Pick:
    date: str
    ticker: str
    rank: int
    score: float
    pct_rank: float
    close: float
    ret_5d: float       # 20d momentum, pre-signal
    ret_20d: float
    ann_vol_20d: float
    sector: str


@dataclass
class NewsAgg:
    n_articles: int = 0
    n_pos: int = 0
    n_neg: int = 0
    n_neutral: int = 0
    sentiment_label: str = "NO-NEWS"  # STRONG-POS / POS / NEUTRAL / NEG / STRONG-NEG / NO-NEWS
    most_recent_title: str = ""
    most_recent_reasoning: str = ""
    most_recent_published_utc: str = ""
    keywords: list = field(default_factory=list)


def load_sector_map():
    df = pd.read_csv(SECTORS_CSV)
    return dict(zip(df["symbol"].str.upper(), df["sector"]))


def load_picks(signal_csv: Path, sector_map: dict) -> list[Pick]:
    df = pd.read_csv(signal_csv).sort_values("rank")
    out = []
    for _, r in df.iterrows():
        out.append(Pick(
            date=signal_csv.stem,
            ticker=str(r["instrument"]).upper(),
            rank=int(r["rank"]),
            score=float(r["score"]),
            pct_rank=float(r["pct_rank"]),
            close=float(r["close"]),
            ret_5d=float(r["ret_5d"]) if pd.notna(r["ret_5d"]) else 0.0,
            ret_20d=float(r["ret_20d"]) if pd.notna(r["ret_20d"]) else 0.0,
            ann_vol_20d=float(r["ann_vol_20d"]) if pd.notna(r["ann_vol_20d"]) else 0.0,
            sector=sector_map.get(str(r["instrument"]).upper(), "Unknown"),
        ))
    return out


def aggregate_news_for_ticker(ticker: str, signal_date: pd.Timestamp,
                              days_back: int = 5) -> NewsAgg:
    """Read news JSONs for the past `days_back` trading sessions ending at signal_date."""
    agg = NewsAgg()
    for offset in range(days_back):
        d = (signal_date - pd.tseries.offsets.BDay(offset)).date()
        f = NEWS_DIR / f"{d.isoformat()}.json"
        if not f.exists():
            continue
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        for art in payload.get("articles", []) or []:
            insights = art.get("insights") or []
            ticker_insight = None
            for ins in insights:
                if (ins.get("ticker") or "").upper() == ticker:
                    ticker_insight = ins
                    break
            if not ticker_insight and ticker not in (art.get("tickers") or []):
                continue
            agg.n_articles += 1
            sent = (ticker_insight or {}).get("sentiment", "neutral") if ticker_insight else "neutral"
            if sent == "positive":
                agg.n_pos += 1
            elif sent == "negative":
                agg.n_neg += 1
            else:
                agg.n_neutral += 1
            # Keep the most recent (we iterate from offset=0 = today, then back).
            # Bug fix 2026-04-28 (Codex flagged): previously only captured catalyst when offset==0,
            # leaving 311 cases with article counts but no catalyst text. Now we capture the first
            # article we see across all offsets — since we iterate offset=0,1,2,...
            # the first one captured is the most recent.
            if not agg.most_recent_published_utc:
                agg.most_recent_title = art.get("title", "")
                agg.most_recent_reasoning = (ticker_insight or {}).get("sentiment_reasoning", "") if ticker_insight else ""
                agg.most_recent_published_utc = art.get("published_utc", "")
                agg.keywords = (art.get("keywords") or [])[:5]

    # Aggregate sentiment label
    if agg.n_articles == 0:
        agg.sentiment_label = "NO-NEWS"
    else:
        net = agg.n_pos - agg.n_neg
        if agg.n_pos >= 3 and agg.n_neg == 0:
            agg.sentiment_label = "STRONG-POS"
        elif agg.n_neg >= 2 and agg.n_pos == 0:
            agg.sentiment_label = "STRONG-NEG"
        elif net > 0:
            agg.sentiment_label = "POS"
        elif net < 0:
            agg.sentiment_label = "NEG"
        else:
            agg.sentiment_label = "NEUTRAL"
    return agg


def get_realized_returns(tickers: list[str], signal_date: pd.Timestamp) -> pd.DataFrame:
    """Pull next-day forward returns for tickers."""
    from qlib.data import D
    end = signal_date + pd.Timedelta(days=20)
    feats = D.features(tickers,
                       ["Ref($close, -2)/Ref($close, -1) - 1",
                        "Ref($close, -4)/Ref($close, -1) - 1",
                        "Ref($close, -6)/Ref($close, -1) - 1"],
                       start_time=signal_date, end_time=end, freq="day")
    feats.columns = ["ret_1d", "ret_3d", "ret_5d"]
    feats = feats.reset_index()
    feats["instrument"] = feats["instrument"].str.upper()
    return feats[feats["datetime"] == signal_date].set_index("instrument")


def get_market_context(signal_date: pd.Timestamp) -> dict:
    from qlib.data import D
    feats = D.features(["SPY"],
                       ["$close",
                        "Mean($close, 200)",
                        "Std($close/Ref($close,1) - 1, 20) * 16"],
                       start_time=signal_date - pd.Timedelta(days=400),
                       end_time=signal_date, freq="day")
    feats.columns = ["close", "ma200", "vol_20d"]
    feats = feats.dropna()
    if feats.empty:
        return {}
    last = feats.iloc[-1]
    return {
        "spy_close": float(last["close"]),
        "spy_above_200ma_pct": (float(last["close"]) / float(last["ma200"]) - 1) * 100,
        "spy_vol_20d_pct": float(last["vol_20d"]) * 100,
    }


def get_sector_concentration(picks: list[Pick]) -> float:
    """Top sector's share of TOP 30."""
    top30 = [p for p in picks if p.rank <= 30]
    counts = {}
    for p in top30:
        counts[p.sector] = counts.get(p.sector, 0) + 1
    if not counts:
        return 0.0
    return max(counts.values()) / 30 * 100


def classify_case(pick: Pick, news: NewsAgg) -> str | None:
    """Decide whether this (pick, news) deserves a case file."""
    is_top = pick.rank <= 5
    is_avoid = pick.rank >= 499
    is_top30 = pick.rank <= 30
    is_bot30 = pick.rank >= 474

    # 🟢 Strong consensus
    if is_top and news.sentiment_label in ("STRONG-POS", "POS"):
        return "consensus_buy"
    if is_avoid and news.sentiment_label in ("STRONG-NEG", "NEG"):
        return "consensus_avoid"

    # 🟡 Conflicts
    if is_top30 and news.sentiment_label in ("STRONG-NEG",):
        return "conflict_buy_news_negative"
    if is_bot30 and news.sentiment_label in ("STRONG-POS",):
        return "conflict_avoid_news_positive"

    # ⚪ Solo
    if is_top and news.sentiment_label == "NO-NEWS":
        return "solo_buy"
    if is_avoid and news.sentiment_label == "NO-NEWS":
        return "solo_avoid"

    return None  # not interesting


def determine_verdict(pick: Pick, news: NewsAgg, ret_5d_pct: float | None) -> str:
    if ret_5d_pct is None:
        return "TBD"
    is_buy_side = pick.rank <= 30
    abs_move = abs(ret_5d_pct)
    if abs_move < 1.0:
        return "neutral"

    moved_up = ret_5d_pct > 0
    model_predicted_up = is_buy_side
    news_predicted_up = news.sentiment_label in ("STRONG-POS", "POS")
    news_predicted_dn = news.sentiment_label in ("STRONG-NEG", "NEG")

    model_right = (model_predicted_up == moved_up)
    news_right = (
        (news_predicted_up and moved_up)
        or (news_predicted_dn and not moved_up)
    )

    if news.sentiment_label == "NO-NEWS":
        return "model_won" if model_right else "model_lost_alone"
    if model_right and news_right:
        return "both_won"
    if model_right and not news_right:
        return "model_won"
    if not model_right and news_right:
        return "news_won"
    return "both_lost"


def _parse_pub_time(pub_str: str, signal_date: str):
    """Returns (pub_lag_days, pub_hour_utc) or (None, None)."""
    if not pub_str or "T" not in pub_str:
        return None, None
    try:
        ts = pd.to_datetime(pub_str.replace("Z", "")).tz_localize(None)
        sig = pd.to_datetime(signal_date)
        lag = (sig - ts).total_seconds() / 86400
        return lag, ts.hour
    except Exception:
        return None, None


def render_case_md(pick: Pick, news: NewsAgg, case_type: str,
                   realized: dict, market_ctx: dict, sector_concentration_pct: float,
                   top_score_of_day: float) -> str:
    ret_1d = realized.get("ret_1d")
    ret_3d = realized.get("ret_3d")
    ret_5d = realized.get("ret_5d")
    verdict = determine_verdict(pick, news, ret_5d * 100 if ret_5d is not None else None)
    pub_lag_days, pub_hour_utc = _parse_pub_time(news.most_recent_published_utc, pick.date)

    # Codex rules (in-sample-only, paper-track only — see L53/L54 in main README)
    rule_avoid_high_precision = (
        pick.rank >= 474
        and news.sentiment_label == "STRONG-POS"
        and pub_lag_days is not None and pub_lag_days < 1
        and pub_hour_utc is not None and pub_hour_utc >= 16
        and pick.ret_5d > 0
    )
    rule_buy_high_precision = (
        pick.rank <= 30
        and news.sentiment_label in ("STRONG-POS", "POS")
        and pick.ret_5d * 100 <= -1.79
        and pub_lag_days is not None and pub_lag_days <= 1
    )

    lines = []
    lines.append("---")
    lines.append(f"date: {pick.date}")
    lines.append(f"ticker: {pick.ticker}")
    lines.append(f"case_type: {case_type}")
    lines.append(f"model_rank: {pick.rank}")
    lines.append(f"model_score: {pick.score:.4f}")
    lines.append(f"model_action: {'BUY' if pick.rank <= 30 else ('AVOID' if pick.rank >= 474 else 'NEUTRAL')}")
    lines.append(f"news_sentiment: {news.sentiment_label}")
    lines.append(f"news_published_count_5d: {news.n_articles}")
    lines.append(f"news_pos_count_5d: {news.n_pos}")
    lines.append(f"news_neg_count_5d: {news.n_neg}")
    lines.append(f"most_recent_published_utc: {news.most_recent_published_utc or ''}")
    lines.append(f"pub_lag_days: {f'{pub_lag_days:.2f}' if pub_lag_days is not None else 'TBD'}")
    lines.append(f"pub_hour_utc: {pub_hour_utc if pub_hour_utc is not None else 'TBD'}")
    lines.append(f"rule_avoid_high_precision: {rule_avoid_high_precision}")
    lines.append(f"rule_buy_high_precision: {rule_buy_high_precision}")
    lines.append(f"sector: {pick.sector}")
    lines.append(f"ret_5d_pre_pct: {pick.ret_5d * 100:.2f}")
    lines.append(f"ret_20d_pre_pct: {pick.ret_20d * 100:.2f}")
    lines.append(f"ann_vol_20d_pct: {pick.ann_vol_20d * 100:.0f}")
    lines.append(f"close: {pick.close:.2f}")
    lines.append(f"sector_concentration_top30_pct: {sector_concentration_pct:.0f}")
    lines.append(f"top_score_of_day: {top_score_of_day:.4f}")
    lines.append(f"spy_close: {market_ctx.get('spy_close', 0):.2f}")
    lines.append(f"spy_above_200ma_pct: {market_ctx.get('spy_above_200ma_pct', 0):.2f}")
    lines.append(f"spy_vol_20d_pct: {market_ctx.get('spy_vol_20d_pct', 0):.1f}")
    lines.append(f"ret_1d_pct: {ret_1d*100:.2f}" if ret_1d is not None else "ret_1d_pct: TBD")
    lines.append(f"ret_3d_pct: {ret_3d*100:.2f}" if ret_3d is not None else "ret_3d_pct: TBD")
    lines.append(f"ret_5d_pct: {ret_5d*100:.2f}" if ret_5d is not None else "ret_5d_pct: TBD")
    lines.append(f"verdict: {verdict}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Case: {pick.ticker} on {pick.date} — {case_type}")
    lines.append("")
    lines.append("## Polygon catalyst")
    if news.most_recent_title:
        lines.append(f"**Title:** {news.most_recent_title}")
        lines.append("")
        lines.append(f"**Reasoning:** {news.most_recent_reasoning or '(no per-ticker reasoning)'}")
        lines.append("")
        lines.append(f"**Published:** {news.most_recent_published_utc}")
        if news.keywords:
            lines.append(f"**Keywords:** {', '.join(news.keywords)}")
    else:
        lines.append("*No news in past 5 trading days for this ticker.*")
    lines.append("")
    lines.append("## Codex narrative")
    lines.append("*(awaiting Codex pass — leave blank, Codex will fill)*")
    lines.append("")
    lines.append("## Lesson tag")
    lines.append("*(awaiting Codex pass)*")
    lines.append("")
    return "\n".join(lines)


def case_filename(pick: Pick, case_type: str) -> str:
    return f"{pick.date}__{pick.ticker.replace('.', '-')}__{case_type}.md"


def preserve_codex_sections(existing_md: str, new_md: str) -> str:
    """If existing case has Codex narrative/lesson_tag filled in, preserve them."""
    if not existing_md:
        return new_md
    # Extract Codex narrative + lesson tag from existing
    def extract(content, header):
        m = re.search(rf"## {header}\n(.+?)(?=\n##|\Z)", content, re.DOTALL)
        if not m:
            return None
        body = m.group(1).strip()
        if "*(awaiting Codex pass" in body or not body:
            return None
        return body

    old_narrative = extract(existing_md, "Codex narrative")
    old_tag = extract(existing_md, "Lesson tag")
    if not old_narrative and not old_tag:
        return new_md
    out = new_md
    if old_narrative:
        out = out.replace(
            "## Codex narrative\n*(awaiting Codex pass — leave blank, Codex will fill)*",
            f"## Codex narrative\n{old_narrative}",
        )
    if old_tag:
        out = out.replace(
            "## Lesson tag\n*(awaiting Codex pass)*",
            f"## Lesson tag\n{old_tag}",
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", nargs="?", help="YYYY-MM-DD")
    p.add_argument("end", nargs="?", help="YYYY-MM-DD")
    p.add_argument("--rebuild", action="store_true", help="wipe cases/ and regenerate")
    args = p.parse_args()

    qlib.init(provider_uri=PROVIDER, region="us")
    sector_map = load_sector_map()

    if args.rebuild and CASES_DIR.exists():
        shutil.rmtree(CASES_DIR)
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    # Find signal CSVs in range
    all_csvs = sorted(SIGNALS_DIR.glob("*.csv"))
    csvs = []
    for csv in all_csvs:
        try:
            d = pd.Timestamp(csv.stem).date()
        except Exception:
            continue
        if args.start and d < date.fromisoformat(args.start):
            continue
        if args.end and d > date.fromisoformat(args.end):
            continue
        csvs.append(csv)

    print(f"Building cases from {len(csvs)} signal days...")

    total_cases = 0
    case_type_counts = {}
    for i, csv in enumerate(csvs, 1):
        signal_date = pd.Timestamp(csv.stem)
        picks = load_picks(csv, sector_map)
        if not picks:
            continue
        sector_concentration_pct = get_sector_concentration(picks)
        top_score_of_day = max(p.score for p in picks)
        market_ctx = get_market_context(signal_date)

        # Filter to interesting picks
        interesting = []
        for pick in picks:
            news = aggregate_news_for_ticker(pick.ticker, signal_date)
            case_type = classify_case(pick, news)
            if case_type:
                interesting.append((pick, news, case_type))

        if not interesting:
            continue

        # Pull realized returns once for all interesting tickers
        tickers = [p.ticker for p, _, _ in interesting]
        try:
            realized_df = get_realized_returns(tickers, signal_date)
        except Exception:
            realized_df = pd.DataFrame()

        for pick, news, case_type in interesting:
            realized = {}
            if not realized_df.empty and pick.ticker in realized_df.index:
                row = realized_df.loc[pick.ticker]
                realized = {
                    "ret_1d": float(row["ret_1d"]) if pd.notna(row["ret_1d"]) else None,
                    "ret_3d": float(row["ret_3d"]) if pd.notna(row["ret_3d"]) else None,
                    "ret_5d": float(row["ret_5d"]) if pd.notna(row["ret_5d"]) else None,
                }

            new_md = render_case_md(pick, news, case_type, realized, market_ctx,
                                    sector_concentration_pct, top_score_of_day)
            fp = CASES_DIR / case_filename(pick, case_type)
            existing = fp.read_text() if fp.exists() else ""
            final_md = preserve_codex_sections(existing, new_md)
            fp.write_text(final_md)
            total_cases += 1
            case_type_counts[case_type] = case_type_counts.get(case_type, 0) + 1

        if i % 5 == 0 or i == len(csvs):
            print(f"  [{i:>3}/{len(csvs)}] {signal_date.date()}: {len(interesting)} new cases | total={total_cases}")

    print(f"\nDone. Total cases written: {total_cases}")
    print(f"Case-type counts:")
    for ct, n in sorted(case_type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {ct:<35} {n}")


if __name__ == "__main__":
    main()
