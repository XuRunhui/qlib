#!/usr/bin/env python
"""
Download latest US stock market data from Yahoo Finance.
Uses existing symbol list and updates from 2020-11-10 to present.
"""

import pandas as pd
import qlib
from qlib.data import D
from qlib.constant import REG_US
from yahooquery import Ticker
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm

def main():
    print("=" * 80)
    print("DOWNLOADING LATEST US STOCK DATA FROM YAHOO FINANCE")
    print("=" * 80)

    # Initialize Qlib to get existing symbol list
    qlib_data_dir = "~/.qlib/qlib_data/us_data"
    qlib.init(provider_uri=qlib_data_dir, region=REG_US)

    # Get all instruments
    print("\nLoading existing instrument list...")
    instruments = D.instruments('sp500')  # Use SP500 as primary list
    symbol_list = D.list_instruments(instruments=instruments,
                                     start_time='2020-01-01',
                                     end_time='2020-11-10',
                                     as_list=True)

    # Add major indices
    symbol_list.extend(['^GSPC', '^NDX', '^DJI'])

    print(f"Found {len(symbol_list)} symbols to update")
    print(f"Sample symbols: {symbol_list[:10]}")

    # Date range for update
    start_date = "2020-11-09"  # Overlap by 1 day to ensure continuity
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\nDownloading data from {start_date} to {end_date}")
    print("This will download data for all symbols...")
    print("\nNote: This is a fresh download approach, not an update.")
    print("For full updates, we recommend using the official collector script")
    print("once the symbol source APIs are accessible.\n")

    # Create output directory
    output_dir = Path("~/.qlib/yahoo_data_raw").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving raw CSV files to: {output_dir}\n")

    successful = 0
    failed = 0
    failed_symbols = []

    # Download data for each symbol
    for i, symbol in enumerate(tqdm(symbol_list, desc="Downloading")):
        try:
            ticker = Ticker(symbol, asynchronous=False)
            df = ticker.history(interval="1d", start=start_date, end=end_date)

            if isinstance(df, pd.DataFrame) and not df.empty:
                # Save to CSV
                df = df.reset_index()
                csv_path = output_dir / f"{symbol.replace('^', '_').replace('/', '_')}.csv"
                df.to_csv(csv_path, index=False)
                successful += 1
            else:
                failed += 1
                failed_symbols.append(symbol)

            # Rate limiting
            if i % 10 == 0:
                time.sleep(0.5)

        except Exception as e:
            failed += 1
            failed_symbols.append(symbol)
            if failed < 10:  # Only show first 10 errors
                print(f"\nError downloading {symbol}: {e}")

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Total symbols processed: {len(symbol_list)}")
    print(f"Success rate: {successful/len(symbol_list)*100:.1f}%")

    if failed_symbols:
        print(f"\nFirst 20 failed symbols: {failed_symbols[:20]}")

    print(f"\nRaw data saved to: {output_dir}")
    print("\nTo use this data with Qlib, you need to:")
    print("1. Normalize the data")
    print("2. Convert to Qlib binary format")
    print("3. Update the calendar and instruments")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
