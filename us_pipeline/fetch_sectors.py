"""Fetch SIC sector codes for each ticker from Polygon and save to a parquet file.

Polygon's `/v3/reference/tickers/{ticker}` endpoint returns a `sic_code` (4-digit)
and `sic_description`. We map SIC codes to a coarser ~12-bucket sector group.
"""
from __future__ import annotations

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


def main() -> None:
    load_dotenv(ROOT.parent / ".env")
    api_key = os.environ["POLYGON_API_KEY"]
    client = RESTClient(api_key)

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
