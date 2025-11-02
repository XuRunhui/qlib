# 🚀 START HERE - Qlib Quantitative Investment Project

Welcome! This is a clean, organized Qlib project for quantitative investment analysis.

---

## 📂 What's Where?

### 🎯 Main Folders

| Folder | Purpose | Start With |
|--------|---------|------------|
| **data_update_scripts/** | Download & update market data | `download_latest_us_data.py` |
| **signal_extraction/** | Generate trading signals | `extract_signals.py` |
| **documentation/** | Complete guides | `DATA_UPDATE_GUIDE.md` |

### 📄 Key Files

- **PROJECT_STATUS.md** - Current status & progress
- **README_ORGANIZATION.md** - Detailed folder structure
- **workflow_config_us_lightgbm.yaml** - Model configuration

---

## ⚡ Quick Start (3 Steps)

### 1️⃣ Check What's Happening Now

```bash
cat PROJECT_STATUS.md
```

**Current**: Downloading latest US stock data (41% complete)

### 2️⃣ Monitor the Download

```bash
# Watch live progress
tail -f download_log.txt

# Or check file count
ls -1 ~/.qlib/yahoo_data_raw/*.csv | wc -l
```

### 3️⃣ Read the Guides

```bash
# For data collection & updates
less documentation/DATA_UPDATE_GUIDE.md

# For trading signals & strategies
less documentation/INVESTMENT_SIGNALS_GUIDE.md

# For project organization
less README_ORGANIZATION.md
```

---

## 🎓 What This Project Does

### The Pipeline

```
📥 COLLECT DATA          →   🤖 TRAIN MODEL        →   💹 GENERATE SIGNALS
(Yahoo Finance)              (LightGBM + Alpha158)      (BUY/AVOID recommendations)
523 US stocks                158 technical features    Top 30 stocks to buy
2020-2025 daily data         Predict future returns    Bottom 10 to avoid
```

### Output Example

**Investment Signals** (from `extract_signals.py`):

```
Date: 2020-10-30

📈 TOP 30 STOCKS TO BUY
   Stock    Score      Action
   FTI      0.0488    Strong Buy
   ILMN     0.0488    Strong Buy
   ETSY     0.0488    Strong Buy
   ...

📉 BOTTOM 10 STOCKS TO AVOID
   Stock    Score      Action
   ABMD    -0.0253    Strong Avoid
   BWA     -0.0253    Strong Avoid
   ...
```

---

## 🔧 Common Tasks

### Test Data Setup
```bash
cd signal_extraction
python test_data_setup.py
```

### Run Full Pipeline
```bash
qrun workflow_config_us_lightgbm.yaml
```

### Extract Trading Signals
```bash
cd signal_extraction
python extract_signals.py
```

### View Latest Signals
```bash
cat investment_signals_latest.csv
```

---

## 📊 Current Data Status

| Metric | Value |
|--------|-------|
| **Existing Data** | 1999-12-31 to 2020-11-10 (outdated) |
| **Downloading** | 2020-11-09 to 2025-11-02 (in progress) |
| **Coverage** | 523 SP500 stocks + 3 indices |
| **Progress** | 41% complete (212/523 symbols) |
| **ETA** | ~6 minutes |

---

## ⚠️ Important Notes

### Model Performance (Current)
- **IC**: 0.0076 (very weak predictive power)
- **Returns**: -7.7% annually (negative!)
- **Status**: ⚠️ **DO NOT use for real trading without improvements**

### What Needs Improvement
1. More training data (2010-2025 instead of 2010-2020)
2. Better features (add fundamentals, sentiment)
3. Model tuning (hyperparameters, ensemble)
4. Risk management (position sizing, stop-loss)

**See**: `documentation/INVESTMENT_SIGNALS_GUIDE.md` for detailed improvement strategies

---

## 🆘 Troubleshooting

### Download Failing?
```bash
# Check errors
grep -i "error" download_log.txt

# Script saves progress, can resume
cd data_update_scripts
python download_latest_us_data.py
```

### Data Not Loading?
```bash
cd signal_extraction
python test_data_setup.py
```

### Signals Not Generating?
Make sure you ran the pipeline first:
```bash
qrun workflow_config_us_lightgbm.yaml
```

---

## 📚 Learn More

### Documentation
- **DATA_UPDATE_GUIDE.md** - How data collection works
- **INVESTMENT_SIGNALS_GUIDE.md** - Complete trading guide
- **README_ORGANIZATION.md** - Project structure
- **PROJECT_STATUS.md** - Current status

### External Resources
- [Qlib Docs](https://qlib.readthedocs.io/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Alpha158 Features](https://qlib.readthedocs.io/en/latest/advanced/alpha.html)

---

## 🎯 Next Steps After Download

1. ✅ **Wait for download** - Currently 41% done
2. ⏳ **Normalize data** - Convert to Qlib format
3. ⏳ **Update binary files** - Integrate with existing data
4. ⏳ **Retrain model** - Use extended 2010-2025 data
5. ⏳ **Generate new signals** - Get current recommendations
6. ⏳ **Evaluate & improve** - Analyze performance, iterate

**Detailed instructions**: `documentation/DATA_UPDATE_GUIDE.md`

---

## 💡 Pro Tips

1. **Monitor download**: Use `tail -f download_log.txt` in a separate terminal
2. **Check disk space**: Data will need ~500MB-1GB
3. **Read guides first**: Save time by understanding the workflow
4. **Test incrementally**: Run `test_data_setup.py` after each major change
5. **Start small**: Test strategies on paper before real money

---

## 📞 Need Help?

1. Check `PROJECT_STATUS.md` for current state
2. Read relevant guide in `documentation/`
3. Look for errors in `download_log.txt`
4. Check Qlib GitHub issues: https://github.com/microsoft/qlib/issues

---

**Last Updated**: 2025-11-02  
**Status**: Data download in progress (41%)  
**Ready for**: Monitoring, reading documentation, planning next steps
