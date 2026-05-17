# AAPL Stock Direction Predictor

## Overview
Artificial Neural Network (ANN) that predicts 
daily AAPL stock price direction (UP or DOWN)
using technical indicators.

## Features Used
- Daily Returns + Log Returns
- RSI (14 day)
- Bollinger Band Position
- MACD + Signal Line
- Volume Z-Score
- Moving Average Crossovers (20/50/200)
- Rolling Volatility

## Results
- Accuracy: ~52%
- Strategy Sharpe: 0.44
- Buy & Hold Sharpe: 2.03
- Max Drawdown: -17.90%

## Key Concepts Applied
- Walk forward validation (no data leakage)
- Feature scaling with StandardScaler
- Dropout regularization (overfitting prevention)
- Sharpe ratio evaluation over accuracy

## Tech Stack
Python, PyTorch, yfinance, pandas, 
numpy, scikit-learn, matplotlib

## Note
AAPL 2019-2024 was an exceptional bull market
making directional prediction extremely difficult.
Model demonstrates proper quant evaluation methodology.
