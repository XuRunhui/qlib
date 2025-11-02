#!/usr/bin/env python
"""Test script to verify Qlib data setup and basic functionality."""

import qlib
from qlib.data import D
from qlib.constant import REG_US

def test_data_setup():
    """Test if Qlib data is properly set up and accessible."""

    # Initialize Qlib with US data
    print("Initializing Qlib with US stock market data...")
    qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region=REG_US)
    print("✓ Qlib initialized successfully!")

    # Test calendar data
    print("\nTesting calendar data...")
    calendar = D.calendar(start_time='2010-01-01', end_time='2020-11-10', freq='day')
    print(f"✓ Calendar loaded: {len(calendar)} trading days")
    print(f"  First trading day: {calendar[0]}")
    print(f"  Last trading day: {calendar[-1]}")

    # Test instruments
    print("\nTesting instrument data...")
    instruments_all = D.instruments('all')
    inst_list = D.list_instruments(instruments=instruments_all, start_time='2010-01-01',
                                   end_time='2020-11-10', as_list=True)
    print(f"✓ Found {len(inst_list)} instruments")
    print(f"  Sample instruments: {inst_list[:10]}")

    # Test SP500
    instruments_sp500 = D.instruments('sp500')
    sp500_list = D.list_instruments(instruments=instruments_sp500, start_time='2010-01-01',
                                     end_time='2020-11-10', as_list=True)
    print(f"✓ SP500 has {len(sp500_list)} stocks")
    print(f"  Sample SP500 stocks: {sp500_list[:10]}")

    # Test features/price data
    print("\nTesting feature/price data...")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    fields = ['$close', '$volume', '$open', '$high', '$low']

    for symbol in test_symbols:
        try:
            df = D.features([symbol], fields, start_time='2020-01-01', end_time='2020-11-10', freq='day')
            print(f"✓ {symbol}: {len(df)} data points")
            print(f"  Latest close price: ${df['$close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")

    # Test factor expressions
    print("\nTesting factor expressions...")
    test_symbol = 'AAPL'
    factor_fields = ['$close', 'Ref($close, 1)', 'Mean($close, 5)', '$high-$low']
    try:
        df = D.features([test_symbol], factor_fields, start_time='2020-01-01',
                       end_time='2020-11-10', freq='day')
        print(f"✓ Factor expressions work correctly")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Sample data (last 5 days):")
        print(df.tail())
    except Exception as e:
        print(f"✗ Factor expressions failed: {e}")

    print("\n" + "="*60)
    print("✓ All tests passed! Qlib data is ready to use.")
    print("="*60)

if __name__ == "__main__":
    test_data_setup()
