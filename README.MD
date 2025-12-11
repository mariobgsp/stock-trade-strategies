# IHSG Swing Trade Analysis Engine

## 📊 Professional Quantitative Trading Analysis System

A comprehensive CLI application for analyzing Indonesian stocks (IHSG) with advanced technical analysis, institutional flow detection, and swing trade signal generation using multiple indicators and Minervini strategies.

---

## 🎯 Features

### 1. **Support & Resistance Analysis**
- **Pivot Point Analysis**: Classic pivot calculations (P, R1, R2, S1, S2)
- **Fibonacci Retracement**: From highest high to identify key support levels
- **Historical Bounce Detection**: Identifies previous support levels that held
- **Nearest Level Identification**: Dynamically finds closest support/resistance
- **Multi-method Confluence**: Combines all methods for accurate level identification

### 2. **Bandar Flow Analysis (Institutional Activity)**
- **Accumulation/Distribution Detection**: Identifies when institutions are entering/exiting
- **On-Balance Volume (OBV)**: Tracks volume flow trends
- **Accumulation/Distribution Line**: Measures buying/selling pressure
- **Money Flow Index (MFI)**: 0-100 oscillator for flow detection
- **VWAP (Volume Weighted Average Price)**: Current pricing relative to volume
- **Candle Absorption Pattern**: Detects consolidation and absorption
- **Phase Classification**: 4 phases (Accumulating, Distributing, Neutral, Transition)
- **Confidence Scoring**: Probability of current phase accuracy

### 3. **Dynamic Technical Indicators (6+ Variations)**
- **Moving Averages**: Tests 6 periods (5, 10, 20, 30, 50, 100)
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Identifies optimal periods for current stock
  - Shows trend direction and support levels
  
- **Relative Strength Index (RSI)**: Tests 5 periods (9, 14, 21, 25, 28)
  - Oversold/Overbought detection
  - Momentum confirmation
  - Divergence identification
  
- **Stochastic Oscillator**: Tests 4 periods (9, 14, 21, 28)
  - %K and %D lines
  - Bullish/Bearish crossovers
  - Extreme condition detection

### 4. **Pattern Recognition (Minervini Strategies)**

#### VCP (Volatility Contraction Pattern)
- Detects consolidation periods with decreasing volatility
- Identifies breakout-ready setups
- Scores based on: volatility trend, price range, volume decline
- Backtested for high probability breakouts

#### Low Cheat Pattern
- Identifies institutional accumulation through fake sells
- Low goes below support but closes above it
- Shows smart money entering positions
- Multiple occurrences increase confidence

#### Superclose MA
- All moving averages converge within 5% threshold
- Indicates volatility squeeze building
- Signals imminent breakout potential
- High probability for directional move

#### Consolidation Pattern
- Identifies price compression zones
- Low volatility periods < 5% range
- Can breakout either direction
- Entry point for multiple strategies

### 5. **Trade Signal Generation**

Generates signals with 6 levels:
- 🟢🟢 **STRONG_BUY**: Multiple confluences, high probability
- 🟢 **BUY**: Strong setup, good entry
- 🟡 **WEAK_BUY**: Moderate setup, requires confirmation
- 🟠 **HOLD**: Setup forming, incomplete
- 🔴 **WAIT**: Insufficient setup, wait for better confluence
- ❌ **SELL**: Downtrend signals

### 6. **Risk/Reward Management (3:1 Setup)**
- Calculates 1R and 3R targets automatically
- Stop loss placement at validated support
- Risk/Reward ratio of 1:3
- Position sizing guidelines
- Scale-out strategy at 1R

### 7. **Backtesting & Validation**
- Historical testing on 3-year data
- Win rate calculation
- Average return analysis
- Expectancy measurement
- Performance metrics per indicator

### 8. **Probability Analysis**
- Overall trade success probability (25-95%)
- Probability of reaching 1R target
- Probability of reaching 3R target
- Confluence-based scoring
- Technical strength assessment (0-100%)

### 9. **Entry Conditions**
Pattern-specific entry rules:
- Breakout entries
- Pullback entries
- Low cheat confirmation
- Support bounce entries
- Candle absorption signals

---

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Required Libraries
```bash
pip install yfinance pandas numpy
```

### System Requirements
- Internet connection (for data download)
- 100MB free disk space
- Terminal/Command line access

---

## 🚀 Installation

1. **Clone or download the repository**
```bash
# Create project directory
mkdir ihsg-swing-trader
cd ihsg-swing-trader
```

2. **Install dependencies**
```bash
pip install yfinance pandas numpy
```

3. **Verify files**
```
Project structure:
├── engine.py          # Core analysis engine
├── main.py           # CLI interface
├── README.md         # Documentation
```

---

## 💻 Usage

### Basic Command
```bash
python main.py {STOCK_TICKER}
```

### Examples
```bash
# Analyze Telkom (TLKM)
python main.py TLKM.JK

# Analyze BRI (BBRI)
python main.py BBRI.JK

# Analyze Bank Mandiri (BMRI)
python main.py BMRI.JK

# Analyze Astra (ASII)
python main.py ASII.JK
```

### Output Sections
The analysis generates 9 detailed sections:

1. **Price & Support/Resistance** - Current levels and distances
2. **Bandar Flow Analysis** - Institutional activity detection
3. **Dynamic Indicators** - Optimized MA, RSI, Stochastic
4. **Pattern Recognition** - VCP, Low Cheat, Superclose MA, Consolidation
5. **Trade Recommendation** - Signal, entry, stops, targets
6. **Success Probability** - Win rates and target probabilities
7. **Supporting Indicators** - Confluence count and list
8. **Trading Action** - Specific entry conditions and actions
9. **Analysis Summary** - Complete recommendation and risk management

---

## 📊 Analysis Methodology

### Data Source
- **Provider**: Yahoo Finance (yfinance)
- **Timeframe**: 3 years of daily OHLCV data
- **Validation**: OJK ticker price guidelines compliance
- **Volume**: Filters out zero-volume days

### Indicator Calculation

#### Support & Resistance
```
Pivot = (High + Low + Close) / 3
R1 = (2 × Pivot) - Low
S1 = (2 × Pivot) - High
Fibonacci: 23.6%, 38.2%, 50%, 61.8%, 78.6% retracements
```

#### VWAP
```
VWAP = Σ(Typical Price × Volume) / Σ(Volume)
Where: Typical Price = (High + Low + Close) / 3
```

#### On-Balance Volume
```
OBV(t) = OBV(t-1) + Volume(t)  if Close(t) > Close(t-1)
OBV(t) = OBV(t-1) - Volume(t)  if Close(t) < Close(t-1)
OBV(t) = OBV(t-1)              if Close(t) = Close(t-1)
```

#### Money Flow Index
```
MFI = 100 - (100 / (1 + Positive MF / Negative MF))
Over 60 = Distribution, Under 40 = Accumulation
```

#### RSI
```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
```

#### Stochastic
```
%K = 100 × (Close - Low(14)) / (High(14) - Low(14))
%D = 3-period SMA of %K
```

### Scoring System

#### Bandar Phase (0-6 points)
- OBV trend: 2 points
- A/D trend: 2 points
- MFI reading: 2 points
- Volume absorption: 2 points
- Result: 4+ = Accumulating, -2 or less = Distributing

#### Trade Signal (0-10 points)
- Bandar accumulation: 2 points
- Price > VWAP: 1 point
- RSI favorable: 2 points
- Stochastic oversold: 2 points
- VCP detected: 2 points
- Low cheat: 2 points
- Superclose MA: 3 points
- Consolidation: 1 point
- Near support: 2 points

#### Success Probability (25-95%)
- Base: 50%
- Signal strength: +5 to +15%
- Bandar phase: +12% or -10%
- RSI conditions: +8-10%
- Stochastic: +8%
- Pattern bonuses: +5 to +12% each

---

## 🎓 Trading Guide

### Best Practices

1. **Risk Management**
   - Risk only 1-2% of portfolio per trade
   - Use calculated stop loss STRICTLY
   - Never move stop loss against position
   - Target 3:1 risk/reward minimum

2. **Entry Strategy**
   - Wait for indicator confluence (3+)
   - Confirm with volume
   - Use support levels for entry
   - Consider price action patterns

3. **Bandar Analysis**
   - ACCUMULATING = More bullish setups
   - DISTRIBUTING = Avoid long positions
   - Look for accumulation on high volume + low price movement
   - Multiple tests of support confirm accumulation

4. **Indicator Optimization**
   - Use recommended MA periods for trend
   - RSI below 40 = better oversold entry
   - Stochastic < 30 = strong oversold
   - Combination improves accuracy

5. **Pattern Trading**
   - VCP: Buy breakout with volume
   - Low Cheat: Enter on reversal confirmation
   - Superclose MA: Buy at support on breakout
   - Consolidation: Buy dips to support

### Entry Checklist
- [ ] Confidence probability > 60%
- [ ] Multiple indicator confluence (3+)
- [ ] Bandar in accumulation phase (optional but preferable)
- [ ] Price at or near support
- [ ] Volume trend favorable
- [ ] Stop loss below support calculated
- [ ] 3:1 risk/reward available
- [ ] Position size calculated (1-2% risk)

### Stop Loss Placement
```
Stop Loss = Support Level - 2% buffer
Example: If Support = 10,000
         Stop Loss = 10,000 × 0.98 = 9,800
```

### Target Calculation
```
Risk = Entry - Stop Loss
Target 1R = Entry + Risk
Target 3R = Entry + (Risk × 3)

Example: 
Entry = 10,500, Support = 10,000
Risk = 500
Target 1R = 11,000 (1:1)
Target 3R = 12,000 (1:3)
```

---

## 📈 Example Analysis Output

```
================================================================================
IHSG SWING TRADE ANALYSIS ENGINE
Analyzing: TLKM.JK
Analysis Date: 2024-01-15 10:30:45
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣  PRICE & SUPPORT/RESISTANCE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Price:        $3,250.00
Nearest Support:      $3,180.00 (2.15% away)
Nearest Resistance:   $3,420.00 (5.23% away)
VWAP (20-period):     $3,240.00

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2️⃣  BANDAR FLOW ANALYSIS (Institutional Activity)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bandar Phase:         🟢 ACCUMULATING
Status:               📈 ACCUMULATING - Institutional buyers entering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3️⃣  DYNAMIC TECHNICAL INDICATORS (Optimized)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recommended MA Periods:     [5, 10, 20, 30, 50]
  └─ MA(5): $3,240.00 (0.31% away)
  └─ MA(10): $3,235.00 (0.46% away)
  └─ MA(20): $3,220.00 (1.23% away)
  └─ MA(30): $3,200.00 (1.54% away)
  └─ MA(50): $3,180.00 (2.15% away)

Optimal RSI Period:         14
Current RSI Value:          35.42
RSI Signal:                 🔵 Oversold - Strong buying signal

Optimal Stochastic Period:  14
Stochastic %K:              28.65
Stochastic %D:              32.10
Stochastic Signal:          🔵 Oversold - Strong buy signal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4️⃣  PATTERN RECOGNITION (Mark Minervini Strategies)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VCP (Volatility Contraction):      ✅ DETECTED - VCP forming with 3.2% range and declining volume
Low Cheat Pattern:                 ✅ DETECTED - Low Cheat detected: 3 instances
Superclose MA (Squeeze Ready):     ❌ Not detected - MAs not converged
Consolidation Pattern:             ✅ DETECTED - Consolidation: 2.8% range

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5️⃣  TRADE RECOMMENDATION & ENTRY STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trade Signal:               🟢 BUY
Signal Strength:            ████████░░ 82.3%
Technical Setup Strength:   ████████░░ 81.5%

Entry Price:                $3,250.00
Entry Reason:               Strong setup with key indicators aligned (Score: 7/10)

Stop Loss:                  $3,176.00 (-2.27%)
Target 1R (1:1 Risk/Reward): $3,324.00 (+2.27%)
Target 3R (1:3 Risk/Reward): $3,472.00 (+6.82%)
Risk/Reward Ratio:          1:3.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6️⃣  SUCCESS PROBABILITY & TRADE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Success Probability: 78.2%
Probability Assessment:     🟢 High - Strong setup

Price Reaching 1R Target:   73.2%
Price Reaching 3R Target:   63.2%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7️⃣  SUPPORTING INDICATORS CONFLUENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Confluences:          8 indicators aligned

  1. ✓ Bandar Accumulating
  2. ✓ Price > VWAP
  3. ✓ RSI Oversold (35.42)
  4. ✓ Stochastic Oversold (28.65)
  5. ✓ VCP Detected
  6. ✓ Low Cheat Pattern
  7. ✓ Deep Support (2.15% away)
  8. ✓ Consolidation Pattern

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8️⃣  TRADING ACTION & NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action:                     📈 BUY on pullback or breakout confirmation

Entry Conditions:
  • Wait for breakout above upper consolidation range
  • Entry on breakout with volume confirmation
  • Stop loss just below consolidation low

Risk Management:
  • Position Size:         Risk max 2.27% of portfolio
  • Reward Target:         Aim for 3R or more (3.0R setup)
  • Stop Placement:        STRICTLY at $3,176.00
  • Scale Out:             Consider taking profit at 1R ($3,324.00)
```

---

## 🔍 Technical Details

### Algorithm Complexity
- Data loading: O(n) where n = 1095 days
- Support/Resistance: O(n) for all methods
- Bandar analysis: O(n) for OBV, A/D, MFI calculations
- Indicator optimization: O(n × m) where m = number of periods tested
- Pattern recognition: O(n) for each pattern
- Overall: O(n) linear complexity

### Accuracy Considerations
- Moving averages require 100+ data points for reliability
- RSI and Stochastic need 30+ days minimum
- Fibonacci levels tested on 1-year ranges
- Bounce levels from 60-day lookback
- VCP detection uses 60-day window
- Low cheat detection uses 20-day lookback

### Limitations
- Backtesting on 3 years past; results may vary
- No real-time data updates (end of day only)
- Assumes sufficient volume for entries
- Does not account for gaps or corporate actions
- Technical analysis not 100% predictive

---

## 🛡️ Risk Disclaimer

**IMPORTANT**: This tool is for **educational and analysis purposes only**.

- Past performance does not guarantee future results
- All trading involves risk of loss
- Follow OJK (Otoritas Jasa Keuangan) guidelines
- Only risk capital you can afford to lose
- Consult financial advisor before trading
- Test strategies in paper trading first
- This analysis does not constitute investment advice
- Stock prices can move against analysis
- No support/resistance level is 100% guaranteed

---

## 📞 Support & Documentation

### File Structure
```
├── engine.py
│   ├── SupportResistanceAnalyzer
│   ├── BandarFlowAnalyzer
│   ├── DynamicIndicatorsOptimizer
│   ├── PatternRecognition
│   ├── BacktestAnalyzer
│   └── SwingTradeEngine
│
├── main.py
│   ├── TradingCLI
│   └── Display methods
│
└── README.md (this file)
```

### Key Classes

**engine.py:**
- `SwingTradeEngine`: Main orchestrator
- `SupportResistanceAnalyzer`: S/R calculations
- `BandarFlowAnalyzer`: Institutional flow detection
- `DynamicIndicatorsOptimizer`: MA, RSI, Stochastic testing
- `PatternRecognition`: VCP, Low Cheat, patterns
- `BacktestAnalyzer`: Historical validation

**main.py:**
- `TradingCLI`: Command-line interface
- Display functions for formatted output
- Helper methods for interpretation

### Troubleshooting

**"Insufficient data for ticker"**
- Stock has less than 100 days of history
- Check ticker symbol format (e.g., TLKM.JK)
- Ensure stock is listed on Indonesian exchange

**"Error fetching data"**
- Internet connection issue
- Invalid ticker symbol
- yfinance service temporarily unavailable
- Check ticker is valid on Yahoo Finance

**"ModuleNotFoundError"**
- Required package not installed
- Run: `pip install yfinance pandas numpy`

---

## 🎯 Version Information

**Version**: 1.0.0
**Release Date**: 2024
**Python**: 3.8+
**Dependencies**: 
- yfinance >= 0.2.0
- pandas >= 1.3.0
- numpy >= 1.21.0

---

## 📝 Changelog

### v1.0.0 - Initial Release
- ✅ Support & Resistance Analysis
- ✅ Bandar Flow Detection
- ✅ Dynamic Indicator Optimization (6+ variations)
- ✅ Pattern Recognition (VCP, Low Cheat, Superclose MA, Consolidation)
- ✅ Trade Signal Generation (6 levels)
- ✅ Risk/Reward Management (3:1)
- ✅ Probability Analysis
- ✅ Backtesting Framework
- ✅ CLI Interface with 9-section output

---

## 📚 References & Theory

### Books & Papers
- "A Beginner's Guide to Day Trading Online" - Toni Turner
- "Market Wizards" - Jack D. Schwager
- Minervini, Mark. "Trade Like a Stock Market Wizard"

### Indicators
- Pivot Points: Widely used in professional trading
- Fibonacci: Natural support/resistance levels
- VWAP: Institutional benchmark pricing
- OBV: Volume trend analysis
- Money Flow Index: Accumulation/Distribution measurement
- RSI: Momentum oscillator
- Stochastic: Overbought/Oversold detection

### Patterns
- VCP (Volatility Contraction): Mark Minervini strategy
- Low Cheat: Institutional accumulation indicator
- Superclose MA: Volatility squeeze signal
- Consolidation: Breakout preparation

---

## 🎓 Learning Path

1. **Start**: Run analysis on 5-10 stocks
2. **Observe**: Note which patterns most frequent
3. **Paper Trade**: Test signals in simulated account
4. **Backtest**: Review historical accuracy
5. **Live Trade**: Start with small position sizes
6. **Optimize**: Adjust parameters based on experience
7. **Scale**: Increase position size with confidence

---

## 💡 Tips for Success

1. **Wait for Confluence**: Don't trade single indicators
2. **Respect Support**: Most reliable entry area
3. **Follow Rules**: Never skip stop loss placement
4. **Risk Management**: 1-2% per trade maximum
5. **Trade with Trend**: Bandar accumulation = bias bullish
6. **Patience**: Miss some trades to avoid losses
7. **Journal**: Track all trades and outcomes
8. **Learn**: Analyze both wins and losses

---

## 📧 Contact & Feedback

For issues, suggestions, or improvements, please refer to documentation or create detailed error logs with:
- Ticker symbol analyzed
- Date of analysis
- Error message
- Python version

---

**Happy Trading! 📈**

Remember: The goal is consistent, sustainable profits through proper risk management and technical analysis. Use this tool to improve your decision-making, not to replace proper due diligence and financial advice.