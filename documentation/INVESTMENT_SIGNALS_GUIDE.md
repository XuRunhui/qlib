# Qlib Investment Signals Guide

## 📊 Complete Guide to Getting Investment Signals from Qlib

This guide explains how to run the Qlib quantitative investment pipeline and extract actionable trading signals from the model predictions.

---

## 🚀 Quick Start

### Step 1: Run the Pipeline

```bash
# Run the LightGBM model with Alpha158 features on US market data
qrun workflow_config_us_lightgbm.yaml
```

### Step 2: Extract Investment Signals

```bash
# Extract and interpret the predictions
python extract_signals.py
```

This will:
- Load prediction results from the experiment
- Generate BUY and AVOID signals
- Save signals to `investment_signals_latest.csv`
- Display comprehensive interpretation

---

## 📈 Understanding the Output

### Pipeline Output Summary

When you run `qrun`, you'll see:

1. **Model Training Metrics**
   ```
   Training until validation scores don't improve for 50 rounds
   [1] train's l2: 0.997171  valid's l2: 0.997952
   ```
   - Lower L2 loss is better
   - Early stopping prevents overfitting

2. **Signal Quality Metrics**
   ```
   IC (Information Coefficient): 0.0076
   ICIR (IC Information Ratio): 0.067
   Rank IC: 0.0058
   ```
   - **IC**: Correlation between predictions and actual returns
     - Positive = predictions have some predictive power
     - IC > 0.05 is considered good
     - Our IC (0.0076) is weak but positive

   - **ICIR**: Consistency of predictions
     - IC divided by standard deviation
     - Higher is better

   - **Rank IC**: Correlation between predicted ranking and actual ranking

3. **Backtest Performance**
   ```
   Excess Return (without cost):
     mean: -0.000195
     annualized_return: -0.046344
     information_ratio: -0.295548
     max_drawdown: -0.241273
   ```
   - Shows how the strategy performs vs benchmark
   - Negative returns indicate underperformance
   - This suggests model needs improvement

---

## 💡 Investment Signals Explained

### Signal Format

Each stock gets a **signal score** representing predicted future return:

| Score Range | Interpretation | Action |
|------------|----------------|---------|
| > 0.01 | Strong Buy | High confidence long position |
| 0 to 0.01 | Buy | Moderate long position |
| ≈ 0 | Neutral | Market-neutral, likely to follow market |
| 0 to -0.01 | Avoid | Underperform, avoid or light short |
| < -0.01 | Strong Avoid | High confidence short or avoid |

### Example Signal Output (2020-10-30)

**Top Stocks to BUY:**
```
Rank  Stock   Score      Interpretation
1     FTI     0.048791   Strong Buy
2     ILMN    0.048791   Strong Buy
3     ETSY    0.048791   Strong Buy
4     DVN     0.034867   Strong Buy
5     NCLH    0.034867   Strong Buy
```

**Stocks to AVOID/SHORT:**
```
Rank  Stock   Score       Interpretation
1     SCHW    -0.013253   Strong Avoid
2     PWR     -0.013253   Strong Avoid
3     AIG     -0.013253   Strong Avoid
```

---

## 🎯 How to Use the Signals

### Strategy 1: Long-Only Portfolio

1. **Select Top 20-30 Stocks**
   - Choose stocks with highest positive scores
   - Diversify to reduce single-stock risk

2. **Weight Allocation**
   - **Equal-weighted**: Each stock gets 1/N of capital
   - **Score-weighted**: Higher scores get more capital
   ```python
   weight[stock] = score[stock] / sum(scores)
   ```

3. **Rebalancing**
   - Daily: For active trading (higher costs)
   - Weekly: Good balance of responsiveness and costs
   - Monthly: Lower costs, less responsive

### Strategy 2: Long-Short Portfolio

1. **Long Top 30 Stocks** (positive scores)
   - Allocate 100% of capital

2. **Short Bottom 10-20 Stocks** (negative scores)
   - If shorting allowed
   - Creates market-neutral position

3. **Dollar-Neutral**
   - Long positions = Short positions
   - Reduces market exposure

### Strategy 3: Risk-Adjusted Portfolio

1. **Filter by Signal Strength**
   - Only trade signals with |score| > 0.01
   - Reduces noise from weak signals

2. **Position Sizing**
   - Limit single position to 5-10% of portfolio
   - Use Kelly criterion for optimal sizing

3. **Stop-Loss Orders**
   - Exit if stock drops 5-10% from entry
   - Protects against adverse moves

---

## 📁 Output Files

### 1. Prediction Results (`pred.pkl`)
- Stored in MLflow experiment tracker
- Contains predictions for all stocks and dates
- Access via: `recorder.load_object("pred.pkl")`

### 2. Investment Signals CSV (`investment_signals_latest.csv`)
```csv
Date,Stock,Signal,Score
2020-10-30,FTI,BUY,0.048791
2020-10-30,ILMN,BUY,0.048791
2020-10-30,ETSY,BUY,0.048791
...
2020-10-30,SCHW,AVOID,-0.013253
```

### 3. Backtest Results (`port_analysis_1day.pkl`)
- Portfolio returns and metrics
- Trade history
- Performance attribution

---

## 🔧 Customization Options

### Adjust Signal Parameters

Edit `extract_signals.py`:

```python
# Change number of buy/avoid signals
signals = generate_investment_signals(
    pred_df,
    top_n=50,      # More stocks in long portfolio
    bottom_n=20    # More stocks to avoid/short
)
```

### Modify Strategy in YAML

Edit `workflow_config_us_lightgbm.yaml`:

```yaml
port_analysis_config:
    strategy:
        class: TopkDropoutStrategy
        kwargs:
            signal: <PRED>
            topk: 30          # Top K stocks to hold
            n_drop: 3         # Drop N stocks each rebalance
```

### Change Trading Parameters

```yaml
backtest:
    account: 100000000        # Starting capital
    exchange_kwargs:
        deal_price: close     # Entry price (open/close/vwap)
        open_cost: 0.0005     # Buy commission (0.05%)
        close_cost: 0.0015    # Sell commission (0.15%)
        min_cost: 5           # Minimum commission per trade
```

---

## ⚠️ Important Warnings

### 1. Model Limitations

The current model shows **weak predictive power**:
- IC = 0.0076 (very low correlation)
- Negative backtest returns

**This model should NOT be used for real trading without improvements!**

### 2. Improving the Model

Consider:
- **More recent data**: Current data ends 2020, use fresher data
- **Better features**: Add fundamental data, sentiment, macroeconomic factors
- **Ensemble models**: Combine LightGBM with neural networks
- **Hyperparameter tuning**: Optimize model parameters
- **Market regime detection**: Different strategies for bull/bear markets

### 3. Risk Management

- **Never risk more than you can afford to lose**
- **Start with paper trading** (simulated)
- **Gradually increase position sizes**
- **Monitor performance continuously**
- **Have exit criteria before entering trades**

### 4. Transaction Costs

Model shows:
```
Without cost: -0.046 annual return
With cost:    -0.077 annual return
```

Transaction costs significantly impact performance. Consider:
- Reducing trading frequency
- Using limit orders instead of market orders
- Batching small trades

---

## 📊 Workflow Architecture

```
┌─────────────────┐
│   Raw Data      │  Stock prices, volumes
│  (Yahoo Finance)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engine  │  Alpha158: 158 technical factors
│  (Alpha158)     │  - Price ratios, momentum, volatility
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │  LightGBM: Gradient boosting
│  (LightGBM)     │  - Train: 2010-2016
└────────┬────────┘  - Valid: 2017-2018
         │           - Test:  2019-2020
         ▼
┌─────────────────┐
│  Predictions    │  Score for each stock/date
│  (pred.pkl)     │  Higher = Better expected return
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Trading Strategy │  TopK: Buy top 30 stocks
│ (TopkDropout)   │  Rebalance daily
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Backtest      │  Simulate trading
│   (Simulator)   │  Calculate returns
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Investment      │  BUY/AVOID signals
│   Signals       │  Ready for execution
└─────────────────┘
```

---

## 🛠️ Advanced Usage

### Access Predictions Programmatically

```python
import qlib
from qlib.workflow import R

# Initialize
qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")

# Load experiment
exp = R.get_exp(experiment_name="workflow")
recorder = list(exp.list_recorders().values())[0]

# Get predictions
pred_df = recorder.load_object("pred.pkl")

# Get latest signals
latest_date = pred_df.index.get_level_values('datetime').max()
signals = pred_df.xs(latest_date, level='datetime')['score']
top_10 = signals.nlargest(10)

print("Top 10 stocks to buy today:")
print(top_10)
```

### Run Backtest with Custom Strategy

```python
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest

# Define your strategy
strategy = TopkDropoutStrategy(
    signal=pred_df,
    topk=50,
    n_drop=5
)

# Run backtest
portfolio_metrics = backtest(strategy, ...)
```

### Export to Trading System

```python
# Generate orders for your broker API
def generate_orders(signals, capital=100000):
    top_stocks = signals.nlargest(30)
    orders = []

    for stock, score in top_stocks.items():
        weight = score / top_stocks.sum()
        quantity = int(capital * weight / get_price(stock))
        orders.append({
            'symbol': stock,
            'side': 'BUY',
            'quantity': quantity,
            'type': 'MARKET'
        })

    return orders
```

---

## 📚 Further Reading

- [Qlib Documentation](https://qlib.readthedocs.io/)
- [Alpha158 Factor Details](https://qlib.readthedocs.io/en/latest/advanced/alpha.html)
- [Strategy Implementation](https://qlib.readthedocs.io/en/latest/component/strategy.html)
- [Backtest Analysis](https://qlib.readthedocs.io/en/latest/component/backtest.html)

---

## ✅ Summary Checklist

- [x] Data downloaded (US stocks, 8,994 symbols)
- [x] Model trained (LightGBM + Alpha158)
- [x] Predictions generated (2019-2020 test period)
- [x] Signals extracted (BUY/AVOID lists)
- [x] Results saved to CSV
- [x] Interpretation guide provided

**Next Steps:**
1. Improve model (IC too low for real trading)
2. Add more features (fundamentals, sentiment)
3. Test different strategies
4. Paper trade before live trading
5. Implement proper risk management

---

**Generated by Qlib Pipeline**
*For educational purposes only. Not financial advice.*
