# Qlib Project Organization

This document explains the organized structure of the Qlib quantitative investment project.

## 📁 Folder Structure

```
qlib/
├── data_update_scripts/       # Scripts for downloading and updating market data
├── signal_extraction/         # Scripts for extracting investment signals
├── documentation/             # Comprehensive guides and documentation
├── workflow_config_us_lightgbm.yaml   # Pipeline configuration
└── README_ORGANIZATION.md     # This file
```

---

## 📂 data_update_scripts/

**Purpose**: Tools for downloading and updating US stock market data from Yahoo Finance

### Files:

#### `download_latest_us_data.py`
- **What it does**: Downloads latest US stock data from Yahoo Finance
- **Date range**: 2020-11-09 to present (updates old data to current)
- **Coverage**: 523 SP500 stocks + major indices (^GSPC, ^NDX, ^DJI)
- **Output**: Raw CSV files in `~/.qlib/yahoo_data_raw/`
- **Usage**:
  ```bash
  cd data_update_scripts
  python download_latest_us_data.py
  ```
- **Status**: Currently running in background (see `download_log.txt`)

#### `update_us_data.py`
- **What it does**: Wrapper around official Qlib collector with update functionality
- **Usage**: (Currently disabled - requires working symbol source APIs)
  ```bash
  cd data_update_scripts
  python update_us_data.py
  ```

### Dependencies:
```bash
pip install yahooquery tqdm pandas qlib
```

---

## 📂 signal_extraction/

**Purpose**: Extract actionable BUY/AVOID signals from model predictions

### Files:

#### `extract_signals.py`
- **What it does**: Loads predictions from Qlib experiments and generates investment signals
- **Input**: `pred.pkl` from MLflow experiment tracker
- **Output**:
  - Console: Top 30 BUY signals, Bottom 10 AVOID signals
  - CSV: `investment_signals_latest.csv`
- **Usage**:
  ```bash
  cd signal_extraction
  python extract_signals.py
  ```
- **Features**:
  - Signal strength interpretation (Strong Buy, Buy, Avoid, Strong Avoid)
  - Historical trend analysis (last 5-10 days)
  - Statistical summary of prediction quality
  - Comprehensive explanation of how to use signals

#### `test_data_setup.py`
- **What it does**: Verifies Qlib data integrity and accessibility
- **Checks**:
  - Calendar data (trading days)
  - Instrument lists (stocks available)
  - Price data (OHLCV for sample stocks)
  - Factor expressions (technical indicators)
- **Usage**:
  ```bash
  cd signal_extraction
  python test_data_setup.py
  ```
- **When to run**: After data updates, before training models

---

## 📂 documentation/

**Purpose**: Comprehensive guides for understanding and using the system

### Files:

#### `DATA_UPDATE_GUIDE.md`
- **Topics covered**:
  - Current data status (ends 2020-11-10)
  - How Qlib collects data (3-source approach)
  - The API failure problem (EastMoney down)
  - Solution: Direct Yahoo Finance download
  - Next steps after download (normalize, convert, verify)
  - Alternative: Fresh data from scratch
  - Code explanation: How collectors work
  - Troubleshooting common issues
- **Audience**: Users updating data, developers understanding the system

#### `INVESTMENT_SIGNALS_GUIDE.md`
- **Topics covered**:
  - Complete guide to running Qlib pipeline
  - Understanding model output (IC, ICIR, backtest metrics)
  - Signal format and interpretation
  - Investment strategies (long-only, long-short, risk-adjusted)
  - Portfolio construction guidelines
  - Output files explained
  - Customization options
  - Important warnings (model limitations, risk management)
  - Workflow architecture diagram
  - Advanced usage examples
- **Audience**: Traders, portfolio managers, quant researchers

---

## 🔧 Configuration Files

### `workflow_config_us_lightgbm.yaml`
**Location**: Root directory

**Purpose**: Pipeline configuration for LightGBM model on US stocks

**Key settings**:
```yaml
qlib_init:
    provider_uri: "~/.qlib/qlib_data/us_data"
    region: us

market: sp500
benchmark: ^GSPC

data_handler_config:
    start_time: 2010-01-01
    end_time: 2020-11-01

task:
    model:
        class: LGBModel
        kwargs:
            learning_rate: 0.2
            max_depth: 8
            num_leaves: 210

    dataset:
        handler:
            class: Alpha158  # 158 technical features
        segments:
            train: [2010-01-01, 2016-12-31]
            valid: [2017-01-01, 2018-12-31]
            test: [2019-01-01, 2020-11-01]

port_analysis_config:
    strategy:
        class: TopkDropoutStrategy
        kwargs:
            topk: 30      # Hold top 30 stocks
            n_drop: 3     # Drop 3 stocks each rebalance
```

**Usage**:
```bash
qrun workflow_config_us_lightgbm.yaml
```

---

## 🚀 Quick Start Workflow

### 1. Update Data (In Progress)
```bash
# Currently running in background
tail -f download_log.txt  # Monitor progress
```

### 2. Verify Data
```bash
cd signal_extraction
python test_data_setup.py
```

### 3. Run Pipeline
```bash
cd ..
qrun workflow_config_us_lightgbm.yaml
```

### 4. Extract Signals
```bash
cd signal_extraction
python extract_signals.py
```

### 5. Review Signals
```bash
# Signals saved to:
cat investment_signals_latest.csv

# Or view in documentation:
less documentation/INVESTMENT_SIGNALS_GUIDE.md
```

---

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                   DATA COLLECTION                       │
├─────────────────────────────────────────────────────────┤
│  data_update_scripts/download_latest_us_data.py         │
│    ↓                                                     │
│  Raw CSV files: ~/.qlib/yahoo_data_raw/                 │
│    ↓                                                     │
│  [Manual] Normalize & Convert to Binary                 │
│    ↓                                                     │
│  Qlib Data: ~/.qlib/qlib_data/us_data/                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   MODEL TRAINING                        │
├─────────────────────────────────────────────────────────┤
│  qrun workflow_config_us_lightgbm.yaml                  │
│    ↓                                                     │
│  Features: Alpha158 (158 technical indicators)          │
│    ↓                                                     │
│  Model: LightGBM (gradient boosting)                    │
│    ↓                                                     │
│  Predictions: MLflow/pred.pkl                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                 SIGNAL EXTRACTION                       │
├─────────────────────────────────────────────────────────┤
│  signal_extraction/extract_signals.py                   │
│    ↓                                                     │
│  Investment Signals:                                    │
│    - Top 30 BUY (highest predicted returns)             │
│    - Bottom 10 AVOID (lowest predicted returns)         │
│    ↓                                                     │
│  Output: investment_signals_latest.csv                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 Current Status

### Data Collection
- ✅ Download script created
- 🔄 **Currently downloading** (see `download_log.txt`)
- ⏳ Pending: Normalization and conversion
- ⏳ Pending: Integration with Qlib binary format

### Model Training
- ✅ Configuration ready (`workflow_config_us_lightgbm.yaml`)
- ⏸️ Waiting for updated data

### Signal Extraction
- ✅ Scripts ready (`extract_signals.py`)
- ⏸️ Currently using old signals (2020-10-30)

---

## 📝 Important Notes

### Data Dates
- **Old data**: 1999-12-31 to 2020-11-10
- **Downloading**: 2020-11-09 to 2025-11-02 (in progress)
- **Target**: Full coverage through November 2025

### Model Performance (Current)
Based on old data (2019-2020 test period):
- **IC**: 0.0076 (very weak correlation)
- **Annual Return**: -7.7% (with costs)
- **Conclusion**: Model needs improvement before real trading

### Next Steps After Download Completes
1. Normalize downloaded CSV files
2. Convert to Qlib binary format
3. Update trading calendar
4. Verify data integrity
5. Retrain model with extended data (2010-2025)
6. Generate fresh investment signals
7. Evaluate new model performance

---

## 🆘 Troubleshooting

### Issue: Download fails
**Solution**: Check `download_log.txt` for errors. Script saves progress, can resume.

### Issue: Qlib can't find data
**Solution**: Run `signal_extraction/test_data_setup.py` to diagnose.

### Issue: Extract signals fails
**Solution**: Ensure you've run `qrun workflow_config_us_lightgbm.yaml` first.

### Issue: Model performance poor
**Solution**: See `documentation/INVESTMENT_SIGNALS_GUIDE.md` section on "Improving the Model"

---

## 📚 Additional Resources

- [Qlib Official Docs](https://qlib.readthedocs.io/)
- [Alpha158 Features](https://qlib.readthedocs.io/en/latest/advanced/alpha.html)
- [Yahoo Finance API](https://pypi.org/project/yahooquery/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

## 👥 Contact & Support

For issues related to:
- **Data collection**: See `documentation/DATA_UPDATE_GUIDE.md`
- **Signal extraction**: See `documentation/INVESTMENT_SIGNALS_GUIDE.md`
- **Qlib framework**: https://github.com/microsoft/qlib/issues

---

**Last Updated**: 2025-11-02
**Status**: Data download in progress (14% complete)
