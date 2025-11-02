# Qlib Project Status

**Last Updated**: 2025-11-02 12:57 UTC

---

## 📂 Project Organization

All files have been organized into clean folders:

```
qlib/
├── 📁 data_update_scripts/      # Data collection tools
│   ├── download_latest_us_data.py
│   └── update_us_data.py
│
├── 📁 signal_extraction/        # Trading signal generation
│   ├── extract_signals.py
│   └── test_data_setup.py
│
├── 📁 documentation/            # Comprehensive guides
│   ├── DATA_UPDATE_GUIDE.md
│   └── INVESTMENT_SIGNALS_GUIDE.md
│
├── 📄 workflow_config_us_lightgbm.yaml
├── 📄 README_ORGANIZATION.md    # Folder structure guide
└── 📄 PROJECT_STATUS.md         # This file
```

---

## 🔄 Current Activity

### Data Download (IN PROGRESS)

**Status**: ⏳ **41% Complete** (212 / 523 symbols)

**Script**: `data_update_scripts/download_latest_us_data.py`

**Progress**:
```
Downloading:  41%|████      | 212/523 [03:37<06:02,  1.17s/it]
```

**Monitor**:
```bash
# Watch live progress
tail -f download_log.txt

# Check downloaded files
ls -1 ~/.qlib/yahoo_data_raw/*.csv | wc -l

# Check disk usage
du -sh ~/.qlib/yahoo_data_raw/
```

**Estimated Time Remaining**: ~6 minutes (at current rate of 1.17s per symbol)

**Data Range Being Downloaded**:
- **Start**: 2020-11-09 (overlap with existing data)
- **End**: 2025-11-02 (today)
- **Period**: ~5 years (~1,260 trading days)

**Symbols**:
- 520 SP500 stocks
- 3 major indices: ^GSPC (S&P 500), ^NDX (NASDAQ 100), ^DJI (Dow Jones)

---

## 📊 Data Status

### Current Data (Existing)
- **Provider**: Yahoo Finance (via Qlib official download)
- **Date Range**: 1999-12-31 to 2020-11-10
- **Coverage**: 8,994 US stocks
- **Size**: 813 MB (binary format)
- **Status**: ✅ Working, but **outdated** (5 years old)

### New Data (Downloading)
- **Provider**: Yahoo Finance (direct via `yahooquery`)
- **Date Range**: 2020-11-09 to 2025-11-02
- **Coverage**: 523 SP500 stocks + 3 indices
- **Format**: Raw CSV files
- **Status**: ⏳ 41% complete

### Next Steps After Download
1. ⏳ **Normalize** - Convert Yahoo format to Qlib standard
2. ⏳ **Convert to Binary** - Create efficient binary files
3. ⏳ **Update Calendar** - Add trading days 2020-11-11 to 2025-11-02
4. ⏳ **Verify** - Test data integrity

---

## 🤖 Model Status

### Current Model
- **Algorithm**: LightGBM (Gradient Boosting)
- **Features**: Alpha158 (158 technical indicators)
- **Training Period**: 2010-01-01 to 2016-12-31
- **Validation**: 2017-01-01 to 2018-12-31
- **Test**: 2019-01-01 to 2020-11-01

### Performance (on old data)
- **IC** (Information Coefficient): 0.0076 ❌ Very weak
- **ICIR**: 0.067 ❌ Low consistency
- **Annual Return**: -4.6% (without costs) ❌ Negative
- **Annual Return**: -7.7% (with costs) ❌ Worse with trading costs
- **Max Drawdown**: -24.1% ❌ High risk

**Conclusion**: ⚠️ **Model needs improvement before real trading**

### After Data Update
With fresh data through 2025, you can retrain with:
- **Training**: 2010-2021 (11 years instead of 6)
- **Validation**: 2022-2023
- **Test**: 2024-2025

This should provide:
- More recent market patterns
- Better generalization
- Current trading signals

---

## 📈 Investment Signals Status

### Current Signals (Old Data)
- **Date**: 2020-10-30 (5 years ago!)
- **Top BUY**: FTI, ILMN, ETSY, DVN, NCLH
- **Top AVOID**: ABMD, BWA, APA, BKR, UA
- **Status**: ⚠️ **Outdated** - Not suitable for current trading

### After Data Update + Retrain
You'll get:
- **Date**: 2025-11-02 (today)
- Fresh signals based on current market conditions
- Better model performance (hopefully)

**How to Generate**:
```bash
# 1. Run pipeline with updated data
qrun workflow_config_us_lightgbm.yaml

# 2. Extract signals
cd signal_extraction
python extract_signals.py

# 3. View signals
cat investment_signals_latest.csv
```

---

## 🛠️ Technical Details

### Data Collection Architecture

**Problem Identified**: Official Qlib collector relies on 3 symbol sources:
1. ❌ **EastMoney API** - Currently DOWN/inaccessible
2. ✅ **NASDAQ FTP** - Working
3. ✅ **NYSE API** - Working

Since EastMoney API is critical and currently failing, the official `update_data_to_bin` command doesn't work.

**Solution Implemented**: Direct Yahoo Finance download
- Uses existing symbol list from current Qlib data
- Bypasses failed API dependency
- Downloads directly via `yahooquery` library
- Saves to CSV for manual processing

### Data Format

**Yahoo Finance Raw Format**:
```csv
symbol,date,open,high,low,close,volume,adjclose
AAPL,2020-11-09,116.32,119.62,116.05,116.32,122138400,115.12
```

**Qlib Normalized Format**:
- All prices relative to first day's close
- Adjustment factor for splits/dividends
- Aligned to trading calendar
- Missing values handled
- Binary format for fast access

**Conversion Pipeline**:
```
Raw CSV → Normalize → Binary → Qlib Data
```

---

## 📝 Documentation

All guides available in `documentation/` folder:

### 1. DATA_UPDATE_GUIDE.md
Comprehensive guide covering:
- Current data status
- How Qlib collects data
- The API failure issue
- Download solution (current approach)
- Next steps after download
- Code explanation
- Troubleshooting

**Use when**: Updating data, understanding data pipeline

### 2. INVESTMENT_SIGNALS_GUIDE.md
Complete trading guide covering:
- How to run Qlib pipeline
- Understanding model output
- Signal interpretation
- Investment strategies
- Portfolio construction
- Risk management
- Model improvement tips

**Use when**: Running pipeline, interpreting signals, building strategies

### 3. README_ORGANIZATION.md
Project structure guide covering:
- Folder organization
- File purposes
- Quick start workflow
- Data flow diagram
- Current status
- Troubleshooting

**Use when**: Understanding project structure, getting started

---

## 🚀 Quick Commands

### Monitor Download
```bash
# Live progress
tail -f download_log.txt

# Count downloaded files
ls -1 ~/.qlib/yahoo_data_raw/*.csv | wc -l

# Check size
du -sh ~/.qlib/yahoo_data_raw/
```

### After Download Completes
```bash
# Verify downloaded data
ls -lh ~/.qlib/yahoo_data_raw/ | head -20

# Count total files
find ~/.qlib/yahoo_data_raw/ -name "*.csv" | wc -l

# Check for errors in log
grep -i "error" download_log.txt
```

### Test Existing Data
```bash
cd signal_extraction
python test_data_setup.py
```

### Run Pipeline (When Ready)
```bash
qrun workflow_config_us_lightgbm.yaml
```

### Extract Signals (After Pipeline)
```bash
cd signal_extraction
python extract_signals.py
```

---

## ⚠️ Important Notes

### Data Quality
- **Survivorship Bias**: Yahoo Finance doesn't include delisted stocks
- **Missing Data**: Some stocks may have gaps
- **Corporate Actions**: Splits/dividends handled via adjustment factor

### Model Limitations
- Current model shows **very weak** predictive power (IC = 0.0076)
- Negative returns in backtest
- **DO NOT use for real trading** without significant improvements

### Recommended Improvements
1. **More data**: Extend training period to 2010-2021
2. **Better features**: Add fundamental data, sentiment analysis
3. **Ensemble models**: Combine multiple algorithms
4. **Hyperparameter tuning**: Optimize model parameters
5. **Market regime detection**: Adapt strategy to market conditions

---

## 🎯 Next Steps

### Immediate (After Download Completes)
1. ✅ **Verify download** - Check for errors in log
2. ⏳ **Normalize data** - Convert to Qlib format
3. ⏳ **Convert to binary** - Create Qlib data files
4. ⏳ **Test integration** - Verify data loads correctly

### Short-term
1. ⏳ **Retrain model** - Use extended data (2010-2025)
2. ⏳ **Generate new signals** - Get current investment recommendations
3. ⏳ **Evaluate performance** - Check if IC improves
4. ⏳ **Backtest strategy** - Test on recent data (2024-2025)

### Long-term
1. ⏳ **Add fundamental data** - P/E ratios, earnings, revenue
2. ⏳ **Sentiment analysis** - News, social media, analyst reports
3. ⏳ **Multi-model ensemble** - Combine LightGBM, neural nets, linear models
4. ⏳ **Risk management** - Position sizing, stop-loss, portfolio optimization
5. ⏳ **Live trading** - Paper trading → Small positions → Scale up

---

## 🔗 Resources

### Qlib Documentation
- Main Docs: https://qlib.readthedocs.io/
- GitHub: https://github.com/microsoft/qlib
- Examples: https://github.com/microsoft/qlib/tree/main/examples

### Data Sources
- Yahoo Finance: https://finance.yahoo.com/
- yahooquery: https://yahooquery.dpguthrie.com/
- NASDAQ FTP: ftp://ftp.nasdaqtrader.com/

### Trading & Quant
- Quantopian Archive: https://www.quantopian.com/
- QuantConnect: https://www.quantconnect.com/
- Alphalens: https://github.com/quantopian/alphalens

---

## 📞 Support

For issues:
1. **Data collection**: See `documentation/DATA_UPDATE_GUIDE.md`
2. **Signal extraction**: See `documentation/INVESTMENT_SIGNALS_GUIDE.md`
3. **Qlib framework**: https://github.com/microsoft/qlib/issues
4. **Yahoo Finance API**: https://github.com/dpguthrie/yahooquery/issues

---

**Status Summary**:
- 📁 Project organized into clean folders ✅
- 📥 Data download in progress (41% complete) ⏳
- 📚 Documentation complete ✅
- 🎯 Ready for next steps after download ⏳

**ETA to completion**: ~6 minutes for download, then manual normalization/conversion needed
