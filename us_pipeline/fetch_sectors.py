"""Fetch SIC sector codes for each ticker from Polygon and save them to CSV.

Polygon's `/v3/reference/tickers/{ticker}` endpoint returns a `sic_code` (4-digit)
and `sic_description`. We map SIC codes to a coarser ~12-bucket sector group.

Two universes, two outputs:
  --universe sp500  (default) -> data/instruments/sectors.csv      503 index names
  --universe all              -> data/instruments/sectors_all.csv  every active US
                                 stock, ADR and ETF, for the chart app's sector view
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

ROOT = Path(__file__).parent
INSTR_FILE = ROOT / "data" / "instruments" / "sp500.txt"
OUT_FILE = ROOT / "data" / "instruments" / "sectors.csv"
WIDE_FILE = ROOT / "data" / "instruments" / "sectors_all.csv"


def sic_to_sector(sic: int | None) -> str:
    """Map 4-digit SIC code to coarse sector bucket (12 sectors)."""
    if sic is None:
        return "Unknown"
    s = int(sic)
    # Reference: https://www.sec.gov/info/edgar/siccodes.htm
    if 100 <= s <= 999:
        return "Agriculture"
    if 1000 <= s <= 1499:
        return "Mining"
    if 1500 <= s <= 1799:
        return "Construction"
    if 2000 <= s <= 3999:
        return "Manufacturing"
    if 4000 <= s <= 4999:
        return "Transportation_Utilities"
    if 5000 <= s <= 5199:
        return "Wholesale"
    if 5200 <= s <= 5999:
        return "Retail"
    if 6000 <= s <= 6799:
        return "Finance_Insurance_RealEstate"
    if 7000 <= s <= 8999:
        return "Services"
    if 9000 <= s <= 9999:
        return "PublicAdmin"
    return "Unknown"


# SIC predates software, so its divisions are useless for browsing by sector:
# AAPL and NVDA are "Manufacturing" alongside XOM and JNJ, MSFT and GOOGL are
# "Services", AMZN is "Retail". The mapping below regroups the same SIC codes
# into GICS-like buckets for the chart app's sector tabs.
#
# `sic_to_sector` above is deliberately left alone — the sector-neutral processor
# and every logged experiment use its buckets, so changing it would silently
# invalidate past results. The two mappings coexist on purpose.
SECTOR_OVERRIDES = {
    # SIC 7389 "Services-Business Services, NEC" is a catch-all holding at least
    # four GICS sectors' worth of companies. No code range can separate them.
    "V": "Financials", "MA": "Financials", "PYPL": "Financials",
    "FIS": "Financials", "FISV": "Financials", "GPN": "Financials",
    "CPAY": "Financials", "MSCI": "Financials", "FICO": "Financials",
    "AKAM": "Technology", "EBAY": "Consumer Discretionary",
    "DASH": "Consumer Discretionary", "ACN": "Industrials",
    "BR": "Industrials", "CSGP": "Real Estate",
    # Other well-known SIC misfits.
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "NFLX": "Communication Services", "GOOGL": "Communication Services",
    "GOOG": "Communication Services", "META": "Communication Services",
}


def sic_to_display_sector(sic: int | None, ticker: str | None = None) -> str:
    """Map SIC code to a GICS-like bucket for the chart app's sector tabs.

    Separate from `sic_to_sector` — see the note above. Ticker overrides win.
    """
    if ticker and ticker.upper() in SECTOR_OVERRIDES:
        return SECTOR_OVERRIDES[ticker.upper()]
    if sic is None:
        return "Unknown"
    s = int(sic)

    if 2833 <= s <= 2836 or 3826 <= s <= 3851 or 8000 <= s <= 8099 or s in (5047, 5122, 6324):
        return "Health Care"
    if 3570 <= s <= 3579 or 3670 <= s <= 3699 or 7370 <= s <= 7379 or s in (3559, 3661, 3663, 3669, 3827):
        return "Technology"
    if 4810 <= s <= 4899 or 2711 <= s <= 2796 or 7812 <= s <= 7841 or s in (7900, 7990, 7993, 7996, 7997):
        return "Communication Services"
    if 6000 <= s <= 6299 or 6700 <= s <= 6779:
        return "Financials"
    if 6500 <= s <= 6599 or s == 6798:
        return "Real Estate"
    if 1200 <= s <= 1399 or 2900 <= s <= 2999 or s in (4610, 4922, 4923, 4924):
        return "Energy"
    if 4900 <= s <= 4991:
        return "Utilities"
    if 800 <= s <= 1099 or 1400 <= s <= 1499 or 2600 <= s <= 2699 or 2800 <= s <= 2824 or 2840 <= s <= 2899 or 3200 <= s <= 3399:
        return "Materials"
    # GICS puts mass merchants and grocers in Staples, not Discretionary.
    if 2000 <= s <= 2199 or 5140 <= s <= 5149 or 5300 <= s <= 5499 or s in (2080, 2086, 5912):
        return "Consumer Staples"
    if 2300 <= s <= 2399 or 3000 <= s <= 3199 or 3630 <= s <= 3652 or 3710 <= s <= 3716 or 5000 <= s <= 5999 or 7000 <= s <= 7011:
        return "Consumer Discretionary"
    return "Industrials"


def fetch_one(client: RESTClient, ticker: str) -> dict:
    try:
        d = client.get_ticker_details(ticker).__dict__
        sic = d.get("sic_code")
        return {
            "symbol": ticker,
            "sic_code": int(sic) if sic else None,
            "sic_description": d.get("sic_description"),
            "industry": d.get("market"),
            "sector": sic_to_sector(int(sic) if sic else None),
        }
    except Exception as e:
        return {
            "symbol": ticker,
            "sic_code": None,
            "sic_description": None,
            "industry": None,
            "sector": "Unknown",
            "error": str(e)[:100],
        }


def list_universe(client: RESTClient, types: tuple[str, ...]) -> list[dict]:
    """Every active US ticker of the given types, from the paginated list endpoint.

    Cheap (a handful of pages) and carries `type` and `name` — but *not* sic_code,
    which is why the sector backfill still needs one detail call per stock.
    """
    out = []
    for t in types:
        got = list(client.list_tickers(market="stocks", type=t, active=True, limit=1000))
        print(f"  type={t:5s} {len(got):5d} tickers")
        out += [{"symbol": r.ticker, "name": r.name, "type": t} for r in got]
    return out


def took_ticker_on(client: RESTClient, ticker: str) -> str | None:
    """Date this symbol was taken over from a *different* ticker, if ever.

    Precomputed here rather than at request time: the chart app needs it for
    every row it ranks, and one events call per ticker is far too slow to do
    inside a web request once the universe is thousands of names wide.
    """
    try:
        events = getattr(client.get_ticker_events(ticker), "events", None) or []
        changes = sorted(
            (e["date"], e["ticker_change"]["ticker"])
            for e in events
            if e.get("type") == "ticker_change" and e.get("ticker_change")
        )
        for i, (date, tick) in enumerate(changes):
            if tick == ticker and i > 0 and changes[i - 1][1] != ticker:
                return date
    except Exception:
        pass
    return None


def fetch_wide(client: RESTClient, row: dict) -> dict:
    """One row of the wide universe: SIC sector for stocks, `ETFs` for funds."""
    if row["type"] == "ETF":
        # Funds have no meaningful SIC code, and `type` already classifies them.
        return {**row, "sic_code": None, "sic_description": None,
                "sector": "ETFs", "took_ticker_on": None}
    try:
        d = client.get_ticker_details(row["symbol"]).__dict__
        sic = int(d["sic_code"]) if d.get("sic_code") else None
        desc = d.get("sic_description")
    except Exception:
        sic, desc = None, None
    return {
        **row,
        "sic_code": sic,
        "sic_description": desc,
        "sector": sic_to_display_sector(sic, row["symbol"]),
        "took_ticker_on": took_ticker_on(client, row["symbol"]),
    }


def build_wide_universe(client: RESTClient, workers: int) -> None:
    """Write sectors_all.csv: every active US stock, ADR and ETF with a sector."""
    print("Listing universe...")
    rows = list_universe(client, ("CS", "ADRC", "ETF"))
    stocks = sum(1 for r in rows if r["type"] != "ETF")
    print(f"\n{len(rows)} tickers ({stocks} need a SIC + rename lookup, ETFs do not)")

    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_wide, client, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            out.append(fut.result())
            if i % 250 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(rows)} | {rate:.0f}/s | eta {(len(rows)-i)/rate/60:.1f}min")

    df = pd.DataFrame(out).sort_values("symbol").reset_index(drop=True)
    df.to_csv(WIDE_FILE, index=False)
    print(f"\nWrote {len(df)} rows -> {WIDE_FILE}  ({time.time()-t0:.0f}s)")
    print("\nSector distribution:")
    print(df["sector"].value_counts())
    print(f"\nRenamed into their current ticker: {df['took_ticker_on'].notna().sum()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=("sp500", "all"), default="sp500",
                        help="sp500 -> sectors.csv (modeling); all -> sectors_all.csv (chart app)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    load_dotenv(ROOT.parent / ".env")
    api_key = os.environ["POLYGON_API_KEY"]
    client = RESTClient(api_key)

    if args.universe == "all":
        build_wide_universe(client, args.workers)
        return

    tickers = [t.strip() for t in INSTR_FILE.read_text().splitlines() if t.strip()]
    print(f"Fetching sector for {len(tickers)} tickers...")

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, client, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 100 == 0:
                print(f"  {i}/{len(tickers)} | elapsed: {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"\nWrote {len(df)} rows -> {OUT_FILE}")
    print("\nSector distribution:")
    print(df["sector"].value_counts())
    n_unknown = (df["sector"] == "Unknown").sum()
    print(f"\nUnknown sector: {n_unknown}")


if __name__ == "__main__":
    main()
