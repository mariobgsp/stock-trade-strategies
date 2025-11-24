# **Simple IHSG Stock Validator (Advanced CLI)**

A sophisticated, modular Command Line Interface (CLI) tool designed for the Indonesian Stock Exchange (IDX). It combines algorithmic backtesting, advanced pattern recognition (VCP, Triangles), institutional money flow analysis ("Bandarmology"), and dynamic risk management to validate trading ideas.

## **🚀 Key Features**

### **1\. 🧠 Deep Grid Search Optimization**

Instead of guessing parameters, the engine runs a simulation on historical data (default: last 2 years) to find what *actually* works for a specific stock.

* **Multi-Strategy Testing:** Automatically tests RSI Reversal, Moving Average Momentum, and Stochastic Oscillator strategies.  
* **Hold Period Optimization:** Iterates through holding periods from **1 to 20 days** to find the statistical "sweet spot" for exits.  
* **Active vs. Historical:** Distinguishes between the historically best strategy and strategies that are *actionable today*.

### **2\. 🛡️ Smart Trade Plan & Risk Management**

Generates a concrete, actionable trading plan based on **ATR (Average True Range)** volatility, avoiding arbitrary percentage stops.

* **Dynamic Stop Loss:** Automatically set at Entry \- (Multiplier x ATR) to avoid market noise.  
* **Dynamic Take Profit:** Set at Entry \+ (Multiplier x ATR) for a healthy Risk-to-Reward ratio.  
* **Smart "Wait" Logic:** If the signal is "WAIT", the tool calculates a **Recommended Limit Order** price based on Fibonacci levels or Support/Resistance.

### **3\. 📉 Advanced Pattern Recognition**

Uses custom algorithms to detect complex price structures:

* **VCP (Volatility Contraction Pattern):** Detects if price volatility is tightening (a la Mark Minervini), signaling a potential explosive move.  
* **Geometric Patterns:** Identifies Symmetrical, Ascending, and Descending Triangles, as well as Bullish Pennants using slope analysis of recent pivots.

### **4\. 🏦 Smart Money Analysis (Bandarmology)**

Detects institutional activity using a combination of volume indicators:

* **Chaikin Money Flow (CMF):** Measures buying/selling pressure based on close price relative to high/low.  
* **Money Flow Index (MFI):** Volume-weighted RSI to detect overheated buying.  
* **Volume Spikes:** Alerts on abnormal volume (\> 2x average).  
* **OBV Divergence:** Detects hidden Accumulation (Price down, Volume up).

### **5\. 🔢 Context & Sentiment**

* **Fibonacci Retracement:** Auto-calculates the 120-day Swing High/Low to find the **0.618 Golden Ratio** support level.  
* **Adaptive Trend Filter:** Automatically switches between EMA 200, EMA 50, or EMA 20 depending on how long the stock has been listed (IPO safe).  
* **Hybrid News Sentiment:** Fetches news via Yahoo Finance (primary) or Google RSS (fallback) and analyzes sentiment using NLP.

## **🛠️ Installation**

1. **Clone the repository**:  
   git clone \[https://github.com/yourusername/ihsg-stock-validator.git\](https://github.com/yourusername/ihsg-stock-validator.git)  
   cd ihsg-stock-validator

2. **Install dependencies**:  
   pip install \-r requirements.txt

**Requirements:** Python 3.10+ is recommended.

## **🖥️ Usage**

### **1\. Standard Mode (Interactive or Direct)**

Run the script without arguments to be prompted, or pass a ticker directly.

python main.py BBCA

### **2\. Custom Mode (Modular Configuration)**

You can override almost any parameter using command-line flags.

**Example: Scalper Mode (Short term, tight stop)**

python main.py GOTO \--period 6mo \--sl 1.5 \--tp 2.0 \--hold 5

**Example: Swing Trader Mode (Long term, loose stop)**

python main.py TLKM \--period 5y \--sl 3.0 \--tp 6.0 \--fib 200

**Example: Smart Money Focus (Sensitive Flow)**

python main.py BREN \--cmf 10 \--mfi 10

### **🔧 Available Arguments**

| Flag | Description | Default |
| :---- | :---- | :---- |
| ticker | Stock Ticker (e.g., BBCA, ANTM) | None |
| \--period | Backtest timeframe (e.g., 1y, 2y, 5y) | 2y |
| \--hold | Max holding days for Grid Search | 20 |
| \--sl | Stop Loss ATR Multiplier | 2.0 |
| \--tp | Take Profit ATR Multiplier | 3.0 |
| \--rsi | RSI Period length | 14 |
| \--fib | Fibonacci lookback days | 120 |
| \--cmf | Chaikin Money Flow period | 20 |
| \--mfi | Money Flow Index period | 14 |

## **📖 Understanding the Report**

* **ACTION:** The primary recommendation (BUY ON DIP, BUY MOMENTUM, or WAIT).  
* **SMART TRADE PLAN:**  
  * **EXECUTE NOW:** Conditions are met, enter at market.  
  * **PENDING:** Conditions not met yet. Set a Limit Order at the "WAIT FOR" price (usually a Fibonacci or Support level).  
* **CHART PATTERNS:**  
  * **\[+\] VCP DETECTED:** Volatility has contracted (e.g., from 10% drop to 4% drop). Watch for a breakout.  
  * **\[+\] GEOMETRY:** A Triangle or Pennant is forming.  
* **SMART MONEY:**  
  * **INSTITUTIONAL BUYING:** Strong accumulation detected.  
  * **INSTITUTIONAL SELLING:** Distribution detected.  
  * **RETAIL NOISE:** No clear institutional footprint.

## **⚠️ Disclaimer**

For Educational Purposes Only.  
This software is an algorithmic analysis tool. It calculates probabilities based on historical data. It does not guarantee future profits.

* **Always Do Your Own Research (DYOR).**  
* **Market Conditions Change:** A strategy with a 90% historical win rate can fail today if news/macroeconomics shifts.  
* The creator is not responsible for any financial losses incurred from using this tool.
