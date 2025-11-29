# 🇮🇩 IHSG Stock Validator 

**Don't just guess. Validate with Math & Smart Money Flow.**

A professional-grade Command Line Interface (CLI) trading tool specifically engineered for the Indonesian Stock Exchange (IDX).  
This tool combines Quantitative Backtesting, Smart Money (Bandarmology) Forensics, and Algorithmic Pattern Recognition to generate institutional-quality trade plans.

---

## 🌟 What's New in V3.1
- **Advanced Bandarmology VSA**: Detects *Stopping Volume* (Absorption) and *No Demand* signals using Volume Spread Analysis.
- **Buying Pressure Meter**: Visualizes the dominance of Buyers vs Sellers over the last 20 days.
- **Big Money Spikes**: Tracks days where volume exceeded 1.25× average (Accumulation vs Distribution count).
- **A/D Line Divergence**: Detects hidden accumulation when price drops but Money Flow rises.
- **Adaptive IPO Mode**: Adjusts trend filters for stocks listed < 260 days (e.g., "IPO Power Trend").
- **Final Verdict Engine**: Produces a clear recommendation: `WATCHLIST`, `WAIT`, or `AVOID`.

---

## 🚀 Key Features

### 1. 🏦 Smart Money Forensics (The "Bandar" Detector)
Composite index of 5 institutional footprints:
- VSA Anomalies: Churning, Stopping Volume, No Demand.
- Net Volume Pressure: Ratio of Up-Volume vs Down-Volume.
- VWAP Anchoring: Detects institutional defence of price.
- NVI (Negative Volume Index): Tracks "Smart Money" on low-volume days.
- Accumulation/Distribution (A/D) Line: Spots flow divergences.

### 2. 🛡️ Professional Risk Management (3R)
Generates a full trade plan:
- Position sizing calculated from your IDR balance and risk %.
- Dynamic Stop Loss using ATR to avoid market noise.
- Reward-to-Risk targeting minimum 1:3.

### 3. 🧠 Hybrid Strategy Engine
Prioritized setups:
1. Rectangle/Box Breakout (momentum)
2. Box Support Bounce (value play)
3. Low Cheat (VCP) (early entry in trend)
4. Minervini Trend Follow (Stage 2 trend)

### 4. 🔎 Algorithmic Backtesting (Self-Optimizing)
Learns from ticker history:
- Breakout reliability
- Smart Money signal accuracy on that ticker
- Which moving averages the stock respects (20/50/200)

### 5. 📉 Pattern Recognition
- VCP (Volatility Contraction Pattern)
- TTM Squeeze (Bollinger inside Keltner)
- Geometric shapes: Triangles (ascending, descending, symmetrical)

---

## 🛠️ Installation

```bash
# Clone or download the repository
git clone <repo-url>
cd ihsg-ultimate-scanner

# Install Python dependencies
pip install yfinance pandas numpy pandas_ta textblob beautifulsoup4 requests
```

---

## 🖥️ Usage

### Quick Start
Analyze a stock using default settings (IDR 100,000,000 balance, 1% risk):

```bash
python main.py BBCA
```

### Power User Mode
Customize account size and risk:

```bash
# Example: 500 Juta Balance, Risking 2% per trade
python main.py BMRI --balance 500000000 --risk 2.0
```

### Full Argument List

| Flag        | Description                                   | Default           |
|-------------|-----------------------------------------------|-------------------|
| ticker      | Stock Code (e.g., BBRI, GOTO)                 | **Required**      |
| --balance   | Your Account Balance (IDR)                    | 100,000,000       |
| --risk      | Risk per trade (%)                            | 1.0               |
| --period    | Data lookback period                          | 2y                |
| --sl        | Stop Loss ATR Multiplier                      | 2.5               |
| --tp        | Take Profit ATR Multiplier                    | 5.0               |
| --min_vol   | Minimum daily transaction (IDR)               | 1,000,000,000     |

---

## 📊 Sample Output Preview

```
STOCK: Bank Central Asia Tbk (BBCA)
PRICE: Rp 10,200 (🟢 +1.50%)

✅ LIQUIDITY CHECK
   Healthy Liquidity (525.50B IDR)

✅ TREND HEALTH
   Status: PERFECT UPTREND (Stage 2) (Score: 6/6)
   [EMA50: 9,850] | [EMA200: 9,100]

🔹 SMART MONEY (Bandarmology Proxy)
   Status: BULLISH (Accumulation)
   Pressure:  [████████████░░░░░░░░] 60.5% Buy Vol
   Big Moves: 3 Accumulation Days vs 1 Distribution Days
   - VSA: Stopping Volume (Potential Reversal)
   - Price > VWAP (Inst. Support)

🚀 TRADE PLAN (3R)
   ACTION:      BUY (EXECUTE (Box Breakout))
   ENTRY:       Rp 10,200
   STOP LOSS:   Rp 9,900
   TARGET (3R): Rp 11,100

   --- POSITION SIZING (Bal: 100 Jt) ---
   🛒 BUY:      33 LOTS
   🔥 Risk:     Rp 1,000,000 (1.0%)

🏁 FINAL VERDICT
   Signal Strength: ELITE SWING SETUP (Score: 5)
   👉 RECOMMENDATION: WATCHLIST / BUY
```

---

## ⚠️ Disclaimer
For **educational purposes only**. This tool uses mathematical probabilities and algorithmic proxies for "Bandarmology" (volume/price heuristics) and **does not** represent actual brokerage data. Always do your own research and consult licensed financial professionals before trading.

---

## ❤️ Built With
- Python
- yfinance, pandas, numpy, pandas_ta, textblob
- Requests & BeautifulSoup for auxiliary data

---
