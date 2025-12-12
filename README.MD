# 📈 IHSG Stock Validator & Bandarmology Analyzer

A powerful CLI (Command Line Interface) tool designed for the Indonesia Stock Exchange (IHSG). It automates technical analysis, optimizes indicators based on backtesting, and provides "Bandarmology" insights using Volume Weighted Average Price (VWAP) and Accumulation/Distribution logic.

## 🚀 Features

* **Multi-Timeframe Analysis:** Automatically scans Daily, Weekly, and Monthly trends.
* **Auto-Optimization:**
    * **RSI:** Backtests periods (7-25) to find the most profitable setting for the specific stock over the last 2 years.
    * **Stochastic:** Backtests various %K and %D combinations to find the best sensitivity.
* **Bandarmology Proxy:**
    * Uses **VWAP (Volume Weighted Average Price)** to estimate the "Smart Money" average holding price.
    * Detects Phases: **Accumulation** (Green), **Absorption** (Yellow), and **Distribution** (Red).
* **Flow Analysis:**
    * **A/D Line & OBV:** Detects hidden Divergences (e.g., Price falling but Money Flow rising = Bullish Divergence).
    * **Volume Spike Detection:** Highlights days with significant volume activity.
* **Dynamic Support:** Scans 10 different Moving Averages (EMA/SMA) to find which line the stock historically "bounces" off most reliably.
* **Historical Log:** View the last 60 days of Bandar flow day-by-day or grouped by phase.

## 🛠️ Installation

1.  **Install Python:** Ensure you have Python installed (version 3.8 or higher recommended).
2.  **Install Dependencies:**
    Run the following command in your terminal/command prompt:
    pip install yfinance pandas numpy

## 💻 Usage

Run the script from your terminal. You can use either the local ticker (e.g., `BBRI`) or the Yahoo Finance format (`BBRI.JK`).

### 1. Default Mode (Detailed History)
Shows a day-by-day log of the last 60 trading days.
python main.py BBRI

### 2. Compressed Mode (Grouped History)
Groups consecutive days of the same phase (e.g., "Accumulating") into single rows for a cleaner summary.
python main.py BBRI -c
# OR
python main.py BBRI --compressed

## 📊 Understanding the Output

### Trend & Indicators
| Column | Description |
| :--- | :--- |
| **Trend** | Trend status based on SMA structures (Uptrend, Downtrend, Correction, etc). |
| **Opt. RSI** | The RSI period (e.g., 8) that historically yielded the highest profit for this stock. |
| **Opt. Stoch** | The Best Stochastic settings found via brute-force backtesting. |

### Bandarmology (Smart Money)
| Term | Meaning |
| :--- | :--- |
| **VWAP** | Volume Weighted Average Price. Approximates the "Bandar's" average cost. |
| **Accumulation** | Price is **< 2% below** VWAP. Smart money is likely buying cheaper. |
| **Absorbing** | Price is **near** VWAP (-2% to +5%). Smart money is maintaining price. |
| **Distribution** | Price is **> 5% above** VWAP. Smart money is in profit and likely selling. |

### Flow Log
* **DIVERGENCE:**
    * 💎 **BULL DIV:** Price is dropping, but Accumulation/Distribution (A/D) Line is rising. Strong Buy signal.
    * 💣 **BEAR DIV:** Price is rising, but A/D Line is dropping. Strong Sell signal.
* **OBV TREND:** Shows if the 5-day volume trend is rising (supporting the move) or falling.
* **DAY %:** Price change on that specific date.
* **VOLUME:** Daily volume. `(G)` = Green Candle, `(R)` = Red Candle.

## Disclaimer

This tool is for **educational and research purposes only**.
* **Bandarmology Proxy:** This tool uses Volume and Price action to *simulate* Bandarmology. It does not have access to real Broker Summary (Broxsum) data (e.g., Code YP, PD, CC).
* **Optimization:** "Optimal" indicators are based on past performance, which does not guarantee future results.
* **No Financial Advice:** Do not trade based solely on this tool. Always do your own research.

---
*Created for IHSG Analysis Automation*
