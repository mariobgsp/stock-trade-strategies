Simple IHSG Stock Validator (Advanced CLI)

A sophisticated, modular Command Line Interface (CLI) tool designed for the Indonesian Stock Exchange (IDX). It combines algorithmic backtesting, pattern recognition (VCP, Triangles), sentiment analysis, and dynamic risk management to validate trading ideas.

🚀 Key Features

1. 🧠 Deep Grid Search Optimization

Instead of using fixed indicators, the engine runs a simulation on historical data (last 2 years) to find what actually works for a specific stock.

Multi-Strategy Testing: Tests RSI Reversal, Moving Average Momentum, and Stochastic Oscillator simultaneously.

Hold Period Optimization: Iterates through holding periods from 1 to 20 days to find the statistical "sweet spot" for exits.

Active vs. Historical: Distinguishes between the historically best strategy and strategies that are actionable today.

2. 🛡️ Smart Trade Plan & Risk Management

Generates a concrete trading plan based on ATR (Average True Range) volatility, not arbitrary percentages.

Dynamic Stop Loss: Automatically set at Entry - (2.0 x ATR) to avoid market noise.

Dynamic Take Profit: Set at Entry + (3.0 x ATR) for a healthy Risk-to-Reward ratio.

Smart "Wait" Logic: If the signal is "WAIT", the tool calculates a Recommended Limit Order price based on Support, Resistance, or Fibonacci levels.

3. 📉 Advanced Pattern Recognition

Uses custom algorithms to detect complex price structures:

VCP (Volatility Contraction Pattern): Detects if price volatility is tightening (a la Mark Minervini), signaling a potential explosive move.

Geometric Patterns: Identifies Symmetrical, Ascending, and Descending Triangles, as well as Bullish Pennants using slope analysis of recent pivots.

4. 🔢 Fibonacci & Context Analysis

Fibonacci Retracement: Auto-calculates the 120-day Swing High/Low to find the 0.618 Golden Ratio support level.

OBV Divergence: Detects Accumulation (Price down, Volume up) or Distribution.

Adaptive Trend Filter: Automatically switches between EMA 200, EMA 50, or EMA 20 depending on how long the stock has been listed (IPO safe).

5. 📰 Hybrid News Sentiment

Dual-Engine Fetcher: First tries Yahoo Finance native news; fails over to a robust Google News RSS scraper if data is missing.

Sentiment Scoring: Uses NLP (TextBlob) to categorize headlines as Bullish, Bearish, or Neutral.

🛠️ Installation

Clone the repository or download the files.

Install dependencies using the provided requirements file:

pip install -r requirements.txt


Required Libraries:

yfinance: Market data.

pandas & numpy: Data manipulation.

pandas_ta: Technical analysis indicators.

textblob: Sentiment analysis.

beautifulsoup4 & lxml: RSS XML parsing.

🖥️ Usage

Run the main script in your terminal:

python main.py


Enter a ticker symbol (e.g., BBCA, GOTO, ANTM, BREN).

Note: You do not need to type .JK; the tool adds it automatically.

Wait for the analysis (Data Fetching -> Backtesting -> Pattern Recognition).

Read the report.

📖 Understanding the Report

The "Action" Section

BUY ON DIP: Price is Oversold (RSI/Stoch) or near Support/Fibonacci.

BUY MOMENTUM: Moving Averages have crossed up (Golden Cross).

WAIT: Trend is down, Sentiment is bad, or Price is in "No Man's Land".

The "Smart Trade Plan"

This section tells you exactly what to do.

Status: EXECUTE NOW: Market order recommended.

Status: PENDING: Do not buy yet. Look at the "WAIT FOR" price.

Example: "WAIT FOR: Rp 850 (Waiting for Golden Ratio 0.618 Fib Retracement)"

Chart Patterns

[+] VCP DETECTED: Volatility has contracted (e.g., from 10% drop to 4% drop). Watch for a breakout.

[+] GEOMETRY: A Triangle or Pennant is forming.

⚙️ Configuration (Bongkar Pasang)

You can tweak the core logic constants at the top of engine.py:

# engine.py

BACKTEST_PERIOD = "2y"       # Lookback range
MAX_HOLD_DAYS = 20           # Max hold days for optimization
ATR_PERIOD = 14              # Volatility period
SL_MULTIPLIER = 2.0          # Multiplier for Stop Loss
TP_MULTIPLIER = 3.0          # Multiplier for Take Profit
FIB_LOOKBACK_DAYS = 120      # Swing High/Low lookback


⚠️ Disclaimer

For Educational Purposes Only.
This software is an algorithmic analysis tool. It calculates probabilities based on historical data. It does not guarantee future profits.

Always Do Your Own Research (DYOR).

Market Conditions Change: A strategy with a 90% historical win rate can fail today if news/macroeconomics shifts.

The creator is not responsible for any financial losses incurred from using this tool.