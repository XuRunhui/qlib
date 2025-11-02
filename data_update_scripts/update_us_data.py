#!/usr/bin/env python
"""
Download and update US stock market data from Yahoo Finance.
This script fetches the latest data and updates the Qlib data directory.
"""

import sys
import pandas as pd
from pathlib import Path

# Add scripts path for imports
scripts_path = Path(__file__).parent / "scripts" / "data_collector" / "yahoo"
sys.path.insert(0, str(scripts_path.parent.parent))

from yahoo.collector import Run

def main():
    """Update US stock data to the latest available date."""

    qlib_data_dir = Path("~/.qlib/qlib_data/us_data").expanduser().resolve()

    print("=" * 80)
    print("US STOCK DATA UPDATE UTILITY")
    print("=" * 80)
    print(f"\nQlib data directory: {qlib_data_dir}")

    # Check current data end date
    calendar_file = qlib_data_dir / "calendars" / "day.txt"
    if calendar_file.exists():
        with open(calendar_file) as f:
            lines = f.readlines()
            current_end_date = lines[-1].strip()
            print(f"Current data end date: {current_end_date}")
    else:
        print("No existing calendar found")
        current_end_date = "2020-11-10"

    # Calculate update date range
    start_date = pd.Timestamp(current_end_date) - pd.Timedelta(days=1)
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    print(f"\nUpdate period:")
    print(f"  Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"  End:   {end_date}")
    print("\n" + "=" * 80)
    print("Starting data update (this may take a while)...")
    print("=" * 80 + "\n")

    # Run the update
    try:
        runner = Run(
            source_dir="~/.qlib/stock_data/source_us",
            normalize_dir="~/.qlib/stock_data/normalize_us",
            max_workers=4,
            interval="1d",
            region="US"
        )

        runner.update_data_to_bin(
            qlib_data_1d_dir=str(qlib_data_dir),
            delay=0.1,
            check_data_length=None,
            exists_skip=False
        )

        print("\n" + "=" * 80)
        print("✓ DATA UPDATE COMPLETED SUCCESSFULLY")
        print("=" * 80)

        # Verify the update
        if calendar_file.exists():
            with open(calendar_file) as f:
                lines = f.readlines()
                new_end_date = lines[-1].strip()
                print(f"\nNew data end date: {new_end_date}")
                print(f"Total trading days: {len(lines)}")

    except Exception as e:
        print(f"\n✗ ERROR during data update: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
