"""Fetch the current S&P 500 ticker list from Wikipedia and write it to disk.

Output: us_pipeline/data/instruments/sp500.txt — one ticker per line, normalized
        to Polygon's convention (BRK.B -> BRK.B; some Wikipedia tickers use ".")
"""
from __future__ import annotations

import sys
from pathlib import Path

import io
import urllib.request

import pandas as pd

OUT_FILE = Path(__file__).parent / "data" / "instruments" / "sp500.txt"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 qlib-pipeline/0.1"


def fetch_sp500() -> list[str]:
    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    tickers = df[col].astype(str).str.strip().str.upper().tolist()
    # Polygon uses "." for class shares (BRK.B, BF.B); Wikipedia already uses "."
    return sorted(set(tickers))


def main() -> None:
    tickers = fetch_sp500()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(tickers) + "\n")
    print(f"Wrote {len(tickers)} tickers -> {OUT_FILE}")


if __name__ == "__main__":
    sys.exit(main())
