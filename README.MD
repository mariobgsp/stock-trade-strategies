# Google Gemini --- Simple IHSG Stock Validator (Advanced CLI)

A sophisticated, modular **Command Line Interface (CLI)** tool designed
for the **Indonesian Stock Exchange (IDX)**. It combines **algorithmic
backtesting**, **pattern recognition (VCP, Triangles)**, **sentiment
analysis**, and **dynamic risk management** to validate trading ideas.

## 🚀 Key Features

### 1. Deep Grid Search Optimization

-   Multi-Strategy Testing: RSI Reversal, Moving Average Momentum,
    Stochastic Oscillator.
-   Hold Period Optimization: 1--20 days.
-   Distinguishes historical vs actionable strategies.

### 2. Smart Trade Plan & Risk Management

-   Dynamic SL: Entry - (2 × ATR)
-   Dynamic TP: Entry + (3 × ATR)
-   "Wait" logic using Support/Resistance/Fibonacci.

### 3. Advanced Pattern Recognition

-   VCP detection.
-   Triangle & Pennant identification.

### 4. Fibonacci & Context Analysis

-   120‑day swing high/low Fibonacci.
-   OBV divergence.
-   Adaptive Trend Filter (EMA 200/50/20).

### 5. Hybrid News Sentiment

-   Yahoo Finance → Google News RSS fallback.
-   Sentiment scoring via TextBlob.

## 🛠️ Installation

    pip install -r requirements.txt

## 🖥️ Usage

    python main.py

## 📖 Understanding the Report

-   BUY ON DIP, BUY MOMENTUM, WAIT signals.
-   Smart Trade Plan with EXECUTE NOW or PENDING.

## ⚙️ Configuration

    BACKTEST_PERIOD = "2y"
    MAX_HOLD_DAYS = 20
    ATR_PERIOD = 14
    SL_MULTIPLIER = 2.0
    TP_MULTIPLIER = 3.0
    FIB_LOOKBACK_DAYS = 120

## ⚠️ Disclaimer

Educational tool only. No profit guarantees. DYOR.
