# US Stock Market Data Update Guide

## Current Situation

The existing Qlib US stock market data ends on **2020-11-10** (almost 5 years old).

## Data Collection Approach

### Method 1: Official Qlib Collector (Recommended but Currently Failing)

The official way to update data is using the Yahoo Finance collector:

```bash
cd scripts/data_collector/yahoo
python collector.py update_data_to_bin \
    --qlib_data_1d_dir ~/.qlib/qlib_data/us_data \
    --region US \
    --delay 0.1
```

**Current Issue**: The collector relies on multiple data sources for stock symbols:
- **EastMoney API** (Chinese financial data provider) - Currently DOWN
- **NASDAQ FTP** (ftp://ftp.nasdaqtrader.com)
- **NYSE API** (https://www.nyse.com/api/quotes/filter)

The EastMoney API is failing, which blocks the entire update process.

### Method 2: Direct Yahoo Finance Download (Current Workaround)

We've created `download_latest_us_data.py` that:

1. **Uses existing symbol list** from current Qlib data (523 SP500 stocks)
2. **Downloads directly from Yahoo Finance** using `yahooquery` library
3. **Date range**: 2020-11-09 to present (2025-11-02)
4. **Output**: Raw CSV files in `~/.qlib/yahoo_data_raw/`

#### How It Works

```python
# The script:
1. Initializes Qlib to read existing instruments
2. Gets SP500 stock list + major indices (^GSPC, ^NDX, ^DJI)
3. Downloads historical data for each symbol
4. Saves to individual CSV files
```

#### Data Schema

Each CSV contains:
- `symbol`: Stock ticker
- `date`: Trading date
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume
- `adjclose`: Adjusted close price (accounts for splits/dividends)

## Progress

**Currently downloading**: ~523 symbols from Yahoo Finance
- **Period**: 2020-11-09 to 2025-11-02 (~5 years, ~1,260 trading days)
- **Estimated time**: ~10-15 minutes (depends on API rate limits)

## Next Steps After Download

### 1. Normalize the Data

The raw Yahoo data needs to be normalized to Qlib format:

```bash
cd scripts/data_collector/yahoo
python collector.py normalize_data \
    --source_dir ~/.qlib/yahoo_data_raw \
    --normalize_dir ~/.qlib/yahoo_data_normalized \
    --region US \
    --interval 1d
```

This step:
- Standardizes date format
- Calculates adjustment factors
- Handles missing data
- Aligns with trading calendar

### 2. Convert to Qlib Binary Format

```bash
cd scripts
python dump_bin.py dump_update \
    --csv_path ~/.qlib/yahoo_data_normalized \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --freq day \
    --date_field_name date \
    --symbol_field_name symbol \
    --exclude_fields symbol,date
```

This creates the efficient binary format Qlib uses for fast data access.

### 3. Update Instruments

Update the SP500 component list:

```bash
cd scripts/data_collector/us_index
python collector.py --index_name SP500 \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --method parse_instruments
```

### 4. Verify the Update

```bash
python test_data_setup.py
```

Check that:
- Calendar extends to 2025
- All symbols have recent data
- Features can be calculated

## Alternative: Download Completely Fresh Data

If you prefer a clean slate, download all data from scratch:

```bash
# Download data from 2000-01-01 to present
cd scripts/data_collector/yahoo

# Step 1: Download raw data
python collector.py download_data \
    --source_dir ~/.qlib/stock_data/source_us_new \
    --region US \
    --start 2000-01-01 \
    --end 2025-11-02 \
    --delay 0.1 \
    --interval 1d

# Step 2: Normalize
python collector.py normalize_data \
    --source_dir ~/.qlib/stock_data/source_us_new \
    --normalize_dir ~/.qlib/stock_data/normalize_us_new \
    --region US \
    --interval 1d

# Step 3: Convert to binary
cd ../../
python dump_bin.py dump \
    --csv_path ~/.qlib/stock_data/normalize_us_new \
    --qlib_dir ~/.qlib/qlib_data/us_data_new \
    --include_fields open,close,high,low,volume,factor,change \
    --freq day

# Step 4: Update instruments
cd data_collector/us_index
python collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data_new
python collector.py --index_name NASDAQ100 --qlib_dir ~/.qlib/qlib_data/us_data_new
python collector.py --index_name DJIA --qlib_dir ~/.qlib/qlib_data/us_data_new
```

**Time estimate**: 4-8 hours for 8,000+ stocks over 25 years

## Code Explanation: How the Collector Works

### Symbol Collection (utils.py:293-377)

```python
def get_us_stock_symbols():
    """Fetches US stock symbols from 3 sources"""

    # Source 1: EastMoney API (Chinese provider covering US stocks)
    # Returns ~8,000 symbols from NYSE, NASDAQ, AMEX
    def _get_eastmoney():
        url = "http://4.push2.eastmoney.com/api/qt/clist/get?..."
        # m:105 = NASDAQ, m:106 = NYSE, m:107 = AMEX

    # Source 2: NASDAQ FTP (official NASDAQ data)
    # Gets both NASDAQ-listed and other exchanges
    def _get_nasdaq():
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"
        # Parses pipe-delimited file

    # Source 3: NYSE API (official NYSE data)
    def _get_nyse():
        url = "https://www.nyse.com/api/quotes/filter"
        # JSON API with equity filter

    # Combine and deduplicate
    all_symbols = _get_eastmoney() + _get_nasdaq() + _get_nyse()
    return sorted(set(all_symbols))
```

### Data Collection (collector.py:126-193)

```python
class YahooCollector:
    def get_data(self, symbol, interval, start, end):
        """Downloads data from Yahoo Finance"""

        # Uses yahooquery library (wrapper around Yahoo Finance API)
        resp = Ticker(symbol).history(
            interval="1d",  # or "1m" for minute data
            start=start,
            end=end
        )

        # Returns DataFrame with:
        # - date, open, high, low, close, volume, adjclose
```

### Data Normalization (collector.py:382-507)

```python
class YahooNormalize:
    def normalize(self, df):
        """Standardizes raw Yahoo data"""

        # 1. Calculate price changes
        change = close / previous_close - 1

        # 2. Calculate adjustment factor
        factor = adjclose / close

        # 3. Apply factor to all prices
        adjusted_prices = raw_prices * factor

        # 4. Adjust volume (inverse of price adjustment)
        adjusted_volume = raw_volume / factor

        # 5. Normalize to first-day close (Qlib convention)
        # All prices relative to first trading day
        normalized = prices / first_close
```

## Why Yahoo Finance?

1. **Free and Accessible**: No API key required
2. **Comprehensive Coverage**: US, international, indices, ETFs
3. **Adjusted Prices**: Handles splits and dividends automatically
4. **Historical Depth**: Data back to 1960s for many stocks
5. **Real-time Updates**: Same-day data available

## Data Quality Notes

### Yahoo Finance Limitations

- **Survivorship Bias**: Delisted stocks disappear from results
- **Occasional Gaps**: Some stocks have missing days
- **Corporate Actions**: Splits/dividends may have slight delays
- **Penny Stocks**: Very low-volume stocks may have inconsistent data

### Qlib's Handling

Qlib normalizes data to handle:
- Missing values (filled with NaN)
- Stock splits (via adjustment factor)
- Dividends (via adjustment factor)
- Calendar alignment (reindexes to trading days)

## Monitoring the Download

Check progress:
```bash
# Watch the log file
tail -f ~/qlib/download_log.txt

# Check number of downloaded files
ls -1 ~/.qlib/yahoo_data_raw/*.csv | wc -l

# Check file sizes
du -sh ~/.qlib/yahoo_data_raw/
```

## Troubleshooting

### Download Fails Midway

The script saves each symbol individually, so you can resume:
- Already downloaded symbols are in CSV files
- Remove failed downloads and retry those symbols

### API Rate Limiting

Yahoo Finance may throttle requests:
- Script includes `time.sleep(0.5)` every 10 requests
- Increase delay if you get connection errors
- Consider using `delay=1.0` for slower but more reliable downloads

### Symbol Not Found

Some symbols may not exist or be renamed:
- The script logs failed symbols
- You can manually exclude problematic symbols
- Check Yahoo Finance website to verify symbol

## Expected Results

After successful download and conversion:

```
~/.qlib/qlib_data/us_data/
├── calendars/
│   └── day.txt           # Trading days: 1999-12-31 to 2025-11-01
├── instruments/
│   ├── all.txt           # All stocks with date ranges
│   └── sp500.txt         # SP500 components
└── features/
    ├── close.bin         # Binary feature files
    ├── open.bin
    ├── high.bin
    ├── low.bin
    ├── volume.bin
    └── factor.bin
```

Total size: ~1-2 GB for 5 years of daily data

## Using Updated Data

Once data is updated, your existing workflow continues normally:

```bash
# Run pipeline with latest data
qrun workflow_config_us_lightgbm.yaml

# Extract signals
python extract_signals.py
```

The model will now train on:
- **Train**: 2010-2021 (instead of 2010-2016)
- **Valid**: 2022-2023 (instead of 2017-2018)
- **Test**: 2024-2025 (instead of 2019-2020)

You'll get **current investment signals** for stocks as of 2025!

---

**Status**: Currently downloading data in background process.
**Check progress**: `tail -f download_log.txt`
