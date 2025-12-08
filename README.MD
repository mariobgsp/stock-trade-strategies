# IHSG Ultimate Scanner (Sniper Edition)

A high-precision stock analysis tool designed for the Indonesian Stock
Exchange (IDX/IHSG). This program combines classical Technical Analysis
(Minervini Trend Template), Smart Money/Bandarmology logic, and advanced
Machine Learning (XGBoost) to identify high-probability swing trading
setups.

## 🚀 Key Features

### 1. 🤖 AI & Machine Learning (XGBoost)

The system uses Extreme Gradient Boosting (XGBoost) trained on
historical data to provide three distinct predictions:

-   **Price Forecast:** Predicts Bullish/Bearish directional bias with
    > confidence levels (Regime Aware).

-   **Bandar Accumulation AI:** Detects subtle volume and price
    > footprints indicating Smart Money accumulation.

-   **AI Support Projection:** Regresses a projected \"floor\" price for
    > the next 5 days.

### 2. 💰 Smart Money Analysis (Bandarmology Proxy)

Uses public data to approximate institutional activity:

-   **CMF & MFI Analysis:** Tracks money flow divergence.

-   **NVI (Net Volume Index):** Detects accumulation on low volume days.

-   **Stealth Accumulation:** Identifies high volume/low spread
    > anomalies.

-   **Distribution Detection:** Warns when selling pressure dominates.

### 3. 💎 Setup & Structure

Automatically detects specific price structures:

-   **VCP (Volatility Contraction Pattern):** Detects tightening price
    > action.

-   **Low Cheat:** Identifies early entry points within a base.

-   **Inside Bar:** Scans for coiling price action and breakout
    > triggers.

-   **TTM Squeeze:** Detects explosive energy building up (Bollinger
    > Bands inside Keltner Channels).

### 4. 📈 Trend & Fundamentals

-   **Minervini Trend Template:** Checks specific moving average
    > alignments (EMA 50/150/200) to ensure Stage 2 Uptrend.

-   **Market Regime:** Filters signals based on the broader market
    > (\^JKSE) health.

-   **Fundamentals:** Checks Altman Z-Score, Market Cap, and
    > Profitability.

### 5. 🛡️ Risk Management (Sniper Logic)

-   **Position Sizing:** Calculates exact lot size based on account
    > balance and risk %.

-   **R:R Calculation:** Projects 3R targets and Stop Losses based on
    > ATR.

## 🛠️ Installation

### Prerequisites

-   Python 3.8 or higher.

### Install Dependencies

Run the following command to install all required libraries, including
the Machine Learning packages:

pip install yfinance pandas numpy requests beautifulsoup4 textblob
xgboost scikit-learn

## 💻 Usage

Run the scanner from your terminal using main.py.

### Basic Scan

Analyze a specific stock ticker (e.g., BBCA, ANTM, TLKM):

python main.py BBCA

*Note: You do not need to add .JK, the system adds it automatically.*

### Advanced Configuration

You can customize the simulation parameters using flags:

python main.py BMRI \--period 5y \--risk 2.0 \--balance 50000000

### Available Arguments

  -----------------------------------------------------------------------
  **Flag**                **Default**             **Description**
  ----------------------- ----------------------- -----------------------
  ticker                  (Required)              The stock symbol (e.g.,
                                                  BBCA, GOTO).

  \--balance              100,000,000             Your trading account
                                                  balance in IDR.

  \--risk                 1.0                     Risk per trade in
                                                  percent (e.g., 1.0 =
                                                  1%).

  \--period               2y                      Historical data
                                                  lookback period (e.g.,
                                                  1y, 2y, 5y).

  \--sl                   3.0                     Stop Loss multiplier (x
                                                  ATR).

  \--tp                   6.0                     Take Profit multiplier
                                                  (x ATR).

  \--min_vol              1,000,000,000           Minimum daily volume
                                                  (IDR) to filter
                                                  illiquid stocks.
  -----------------------------------------------------------------------

## 📊 Understanding the Report

1.  **Liquidity Check:** Ensures the stock is tradable and not a
    > \"fried\" stock with zero volume.

2.  **Trend Health:** A score out of 6 based on Mark Minervini\'s trend
    > template. Score \>= 4 is ideal.

3.  **Smart Money:** Buying Pressure vs. Selling Pressure and
    > Accumulation signals.

4.  **AI/Machine Learning:**

    -   *Forecast:* Directional prediction (Bullish/Bearish).

    -   *Confidence:* Only trust signals \> 70%.

    -   *Accumulation Score:* Probability that a \"Bandar\" is
        > collecting shares.

    -   *AI Support:* A predicted safe price level to place bids.

5.  **Trade Plan:** The exact Entry, Stop Loss, and Target prices with
    > calculated Lot sizes.

## ⚠️ Disclaimer

This software is for educational and research purposes only. The
predictions generated by XGBoost and the technical indicators are based
on historical probabilities and do not guarantee future results. Always
do your own research (DYOR) before trading.