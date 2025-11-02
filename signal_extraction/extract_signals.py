#!/usr/bin/env python
"""
Extract and interpret investment signals from Qlib predictions.
This script shows how to get actionable trading signals from model outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import qlib
from qlib.data import D
from qlib.constant import REG_US
from qlib.workflow import R

def load_predictions():
    """Load the prediction results from the latest experiment."""

    # Initialize Qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region=REG_US)

    # Get the latest experiment
    experiment = R.get_exp(experiment_name="workflow")

    # List all recorders in this experiment
    recorders = experiment.list_recorders()
    print(f"Found {len(recorders)} recorders in the experiment")
    print(f"Recorders type: {type(recorders)}")

    # Get the latest recorder - handle both dict and DataFrame
    if isinstance(recorders, dict):
        recorder_id = list(recorders.keys())[0]
        print(f"\nUsing recorder: {recorder_id}")
    else:
        latest_recorder = recorders.iloc[0]
        recorder_id = latest_recorder['id']
        print(f"\nUsing recorder: {recorder_id}")
        print(f"Experiment ID: {latest_recorder['experiment_id']}")

    # Load the prediction results
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_name="workflow")
    pred_df = recorder.load_object("pred.pkl")

    return pred_df, recorder

def generate_investment_signals(pred_df, top_n=30, bottom_n=10):
    """
    Generate investment signals from predictions.

    Args:
        pred_df: DataFrame with prediction scores
        top_n: Number of top stocks to buy (long position)
        bottom_n: Number of bottom stocks to short (or avoid)

    Returns:
        Dictionary with buy/sell/hold signals
    """

    # Get the latest trading date
    latest_date = pred_df.index.get_level_values('datetime').max()
    print(f"\n{'='*80}")
    print(f"INVESTMENT SIGNALS FOR: {latest_date.date()}")
    print(f"{'='*80}")

    # Get predictions for the latest date
    latest_predictions = pred_df.xs(latest_date, level='datetime')['score']

    # Sort by prediction score (higher = better expected return)
    sorted_predictions = latest_predictions.sort_values(ascending=False)

    # Top stocks to BUY (Long positions)
    buy_signals = sorted_predictions.head(top_n)

    # Bottom stocks to AVOID/SHORT
    avoid_signals = sorted_predictions.tail(bottom_n)

    print(f"\n📈 TOP {top_n} STOCKS TO BUY (LONG POSITIONS)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Stock':<10} {'Signal Score':<15} {'Interpretation':<30}")
    print("-" * 80)

    for rank, (instrument, score) in enumerate(buy_signals.items(), 1):
        stock = instrument[1] if isinstance(instrument, tuple) else instrument
        interpretation = "Strong Buy" if score > 0.01 else "Buy" if score > 0 else "Weak Buy"
        print(f"{rank:<6} {stock:<10} {score:>14.6f} {interpretation:<30}")

    print(f"\n📉 BOTTOM {bottom_n} STOCKS TO AVOID/SHORT")
    print("-" * 80)
    print(f"{'Rank':<6} {'Stock':<10} {'Signal Score':<15} {'Interpretation':<30}")
    print("-" * 80)

    for rank, (instrument, score) in enumerate(avoid_signals.items(), 1):
        stock = instrument[1] if isinstance(instrument, tuple) else instrument
        interpretation = "Strong Avoid" if score < -0.01 else "Avoid" if score < 0 else "Weak Avoid"
        print(f"{rank:<6} {stock:<10} {score:>14.6f} {interpretation:<30}")

    # Calculate statistics
    print(f"\n{'='*80}")
    print("SIGNAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total stocks with predictions: {len(latest_predictions)}")
    print(f"Average signal score: {latest_predictions.mean():.6f}")
    print(f"Signal score std dev: {latest_predictions.std():.6f}")
    print(f"Max signal score: {latest_predictions.max():.6f}")
    print(f"Min signal score: {latest_predictions.min():.6f}")

    # Return structured signals
    return {
        'date': latest_date,
        'buy_signals': buy_signals.to_dict(),
        'avoid_signals': avoid_signals.to_dict(),
        'all_predictions': latest_predictions
    }

def analyze_historical_signals(pred_df, days_back=10):
    """Analyze signal evolution over recent days."""

    # Get last N trading dates
    all_dates = pred_df.index.get_level_values('datetime').unique().sort_values()
    recent_dates = all_dates[-days_back:]

    print(f"\n{'='*80}")
    print(f"SIGNAL TREND ANALYSIS (Last {days_back} Trading Days)")
    print(f"{'='*80}\n")

    for date in recent_dates:
        try:
            date_predictions = pred_df.xs(date, level='datetime')['score']
            top_3 = date_predictions.nlargest(3)

            print(f"Date: {date.date()}")
            print(f"  Top 3 stocks:")
            for stock, score in top_3.items():
                stock_name = stock if isinstance(stock, str) else stock[0] if isinstance(stock, tuple) else str(stock)
                print(f"    {stock_name}: {score:.6f}")
            print()
        except Exception as e:
            print(f"Date: {date.date()} - No data available")
            continue

def save_signals_to_csv(signals, filename="investment_signals.csv"):
    """Save signals to a CSV file for easy reference."""

    # Prepare buy signals
    buy_df = pd.DataFrame([
        {'Stock': k[1] if isinstance(k, tuple) else k,
         'Signal': 'BUY',
         'Score': v}
        for k, v in signals['buy_signals'].items()
    ])

    # Prepare avoid signals
    avoid_df = pd.DataFrame([
        {'Stock': k[1] if isinstance(k, tuple) else k,
         'Signal': 'AVOID',
         'Score': v}
        for k, v in signals['avoid_signals'].items()
    ])

    # Combine and save
    combined_df = pd.concat([buy_df, avoid_df], ignore_index=True)
    combined_df['Date'] = signals['date'].date()
    combined_df = combined_df[['Date', 'Stock', 'Signal', 'Score']]

    combined_df.to_csv(filename, index=False)
    print(f"\n✅ Signals saved to: {filename}")

    return combined_df

def explain_signal_interpretation():
    """Explain how to interpret the signals."""

    print(f"\n{'='*80}")
    print("HOW TO INTERPRET THE SIGNALS")
    print(f"{'='*80}\n")

    print("""
📊 SIGNAL SCORE INTERPRETATION:

The signal score represents the model's prediction of the stock's future return.

• POSITIVE SCORES (> 0):
  - The model predicts the stock will OUTPERFORM the market average
  - Higher positive scores = stronger buy signal
  - Scores > 0.01 are considered "Strong Buy"

• NEGATIVE SCORES (< 0):
  - The model predicts the stock will UNDERPERFORM the market average
  - Lower negative scores = stronger avoid/short signal
  - Scores < -0.01 are considered "Strong Avoid"

• NEAR-ZERO SCORES (≈ 0):
  - The model predicts market-neutral performance
  - These stocks are likely to move in line with the overall market

🎯 RECOMMENDED STRATEGY:

1. LONG PORTFOLIO (Buy signals):
   - Allocate capital to top 20-30 stocks with highest scores
   - Weight by signal strength (higher score = larger position)
   - Consider equal-weighted or score-weighted allocation

2. SHORT PORTFOLIO (Avoid/Short signals):
   - If short selling is allowed, consider shorting bottom 10-20 stocks
   - If short selling not allowed, simply avoid these stocks
   - Use these as a "do not buy" list

3. RISK MANAGEMENT:
   - Diversify across multiple top signals (don't put all in #1)
   - Set stop-loss orders to limit downside
   - Rebalance periodically (daily/weekly) as new signals are generated
   - Consider transaction costs when trading

4. PORTFOLIO CONSTRUCTION:
   - TopK Strategy: Select top K stocks with highest scores
   - Weight proportional to signal strength
   - Typical: 20-50 stocks for diversification

⚠️  IMPORTANT NOTES:

• These are PREDICTIONS, not guarantees
• Past performance doesn't guarantee future results
• Always consider your risk tolerance
• Factor in transaction costs and taxes
• The model was trained on historical data (2010-2017)
  and validated on 2017-2018 data
• Current predictions are for the test period (2019-2020)

📈 SIGNAL QUALITY METRICS (from pipeline output):

• IC (Information Coefficient): 0.0076
  - Measures correlation between prediction and actual returns
  - Positive IC means predictions have some predictive power
  - IC > 0.05 is generally considered good

• ICIR (IC Information Ratio): 0.067
  - IC divided by its standard deviation
  - Measures consistency of predictions
  - Higher is better, but this value is quite low

• Rank IC: 0.0058
  - Correlation between predicted rank and actual return rank
  - Positive but low value

⚠️  NOTE: The IC values suggest weak predictive power. Consider:
  - Using ensemble of multiple models
  - Improving feature engineering
  - Using more recent training data
  - Combining with fundamental analysis
""")

if __name__ == "__main__":
    print("Loading prediction results from Qlib experiment...")

    # Load predictions
    pred_df, recorder = load_predictions()

    print(f"\nPrediction data shape: {pred_df.shape}")
    print(f"Date range: {pred_df.index.get_level_values('datetime').min()} to {pred_df.index.get_level_values('datetime').max()}")

    # Generate investment signals
    signals = generate_investment_signals(pred_df, top_n=30, bottom_n=10)

    # Analyze historical trend
    analyze_historical_signals(pred_df, days_back=5)

    # Save to CSV
    signals_df = save_signals_to_csv(signals, filename="investment_signals_latest.csv")

    # Display interpretation guide
    explain_signal_interpretation()

    print(f"\n{'='*80}")
    print("✅ SIGNAL EXTRACTION COMPLETE")
    print(f"{'='*80}")
