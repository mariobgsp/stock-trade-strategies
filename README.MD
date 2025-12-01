# 🇮🇩 IHSG Ultimate Stock Scanner (V3.5 - Sniper Pro)

**"Precision Over Frequency. One Shot, One Kill."**

A professional-grade CLI trading tool for the Indonesian Stock Exchange
(IDX), optimized for high‑probability *Sniper* setups and disciplined
Swing Trading risk management.

------------------------------------------------------------------------

## 🌟 New in V3.5 (Sniper Pro Update)

-   🦎 **Dynamic Chameleon MA** --- Automatically finds which MA
    (10--200) the stock respects the most.\
-   🕯️ **Inside Bar Detection** --- Detects volatility coils ideal for
    sniper entries.\
-   📊 **RS Score (0--99)** --- Ranks each stock versus IHSG to ensure
    you trade leaders.\
-   ⚖️ **Swing--Sniper Calibration** --- 3.0 ATR stops, 6.0 ATR
    targets.\
-   🧠 **Multi‑Timeframe Logic** --- Weekly trend + daily confirmation.

------------------------------------------------------------------------

## 🚀 Key Features

### 1. 🏦 Smart Money Forensics (Bandar Detector)

-   VSA anomalies (Churning, Stopping Volume, No Demand)\
-   Stealth accumulation\
-   Net volume pressure\
-   NVI + VWAP institutional behavior

### 2. 🛡️ Professional Risk Management (3R System)

-   Automatic lot sizing based on balance & risk %\
-   ATR‑based dynamic stop loss\
-   Trailing stop & breakeven signals

### 3. 🧠 Hybrid Strategy Engine (Priority Order)

1.  🚀 Sniper RSI + Stoch\
2.  📦 Fresh Box Breakout\
3.  💰 Support Bounce\
4.  ⚡ Low Cheat (VCP)\
5.  📈 Minervini Trend Following

### 4. 🔎 Algorithmic Backtesting

-   Breakout reliability scoring\
-   VSA accuracy measurement\
-   Dynamic MA Fit ("magic line")

### 5. 📉 Pattern Recognition

-   VCP\
-   Rectangles & Inside Bars\
-   Triangles & Channels

------------------------------------------------------------------------

## 🛠️ Installation

    pip install yfinance pandas numpy pandas_ta textblob beautifulsoup4 requests

------------------------------------------------------------------------

## 🖥️ Usage

### Quick Start

    python main.py BBCA

### Power User Mode

    python main.py BMRI --balance 500000000 --risk 2.0

### Full Argument List

  Flag        Description                     Default
  ----------- ------------------------------- -------------
  ticker      Stock Code (e.g., BBRI, GOTO)   Required
  --balance   Account Balance (IDR)           100,000,000
  --risk      Risk per trade (%)              1.0%
  --period    Data lookback                   2y
  --sl        Stop Loss ATR Multiplier        3.0
  --tp        Take Profit ATR Multiplier      6.0
  --min_vol   Min daily transaction (IDR)     1 Billion

------------------------------------------------------------------------

## 📊 Sample Output Preview

    STOCK: Bank Central Asia Tbk (BBCA)
    PRICE: Rp 10,200 (+1.50%)

    LIQUIDITY: Healthy (525.50B)

    TREND HEALTH:
      Daily:  PERFECT UPTREND (6/6)
      Weekly: UPTREND
      RS:     85/99

    SMART MONEY:
      Bullish Accumulation
      Buy Pressure: 60.5%
      VSA: Stopping Volume

    TRADE PLAN:
      ACTION: BUY
      ENTRY:  Rp 10,200
      SL:     Rp 9,900
      TP:     Rp 12,000
      MGMT:   Move SL to BE at Rp 10,500 (1R)

    FINAL VERDICT:
      Strength: ELITE SWING SETUP (Score 5)
      Probability: ~82.5%
      Recommendation: WATCHLIST / BUY

------------------------------------------------------------------------

## ⚠️ Disclaimer

This tool is for **educational purposes only**.\
It uses mathematical and algorithmic proxies and does not represent real
brokerage data.\
Always perform your own research.\
Built with Python & ❤️ for IHSG.