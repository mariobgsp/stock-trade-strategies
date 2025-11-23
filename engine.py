import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import random
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from datetime import datetime, timedelta

# ==========================================
# 1. MODULAR CONFIGURATION (BONGKAR PASANG)
# ==========================================

# Timeframe & Data
BACKTEST_PERIOD = "2y"       # Data range to fetch
MAX_HOLD_DAYS = 20           # Max days to test in Grid Search (1 to 20)
FIB_LOOKBACK_DAYS = 120      # Lookback for Major Swing High/Low (Fibonacci)

# Strategy A: RSI Parameters
RSI_PERIOD = 14
RSI_TEST_LEVELS = [30, 40]   # Oversold levels to test

# Strategy B: Moving Average Pairs (Fast, Slow)
MA_TEST_PAIRS = [(5, 20), (10, 50), (20, 200)]

# Strategy C: Stochastic Parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20

# Risk Management Defaults
ATR_PERIOD = 14
SL_MULTIPLIER = 2.0          # Stop Loss = 2x ATR (Safe from noise)
TP_MULTIPLIER = 3.0          # Take Profit = 3x ATR (1:1.5 Risk Reward)

# Trend & OBV
OBV_LOOKBACK_DAYS = 5        # For divergence check
TREND_EMA_DEFAULT = 200      # Default Major trend filter

# ==========================================
# 2. CLASS DEFINITION
# ==========================================

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = self._format_ticker(ticker)
        self.df = None
        self.info = {}
        self.news_analysis = {"sentiment": "Neutral", "score": 0, "headlines": []}
        self.active_trend_col = f"EMA_{TREND_EMA_DEFAULT}" # Will adapt if IPO
        self.data_len = 0

    def _format_ticker(self, ticker):
        ticker = ticker.upper().strip()
        if not ticker.endswith(".JK"):
            ticker += ".JK"
        return ticker

    def fetch_data(self):
        """Fetches 2 years of daily data using yfinance."""
        try:
            # FIX: Added auto_adjust=True to silence FutureWarning
            self.df = yf.download(self.ticker, period=BACKTEST_PERIOD, progress=False, auto_adjust=True)
            
            if self.df.empty:
                return False
            
            # Flatten MultiIndex columns if present (common in new yfinance versions)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)
            
            # Store data length for IPO logic
            self.data_len = len(self.df)

            # Get basic info
            ticker_obj = yf.Ticker(self.ticker)
            try:
                self.info = ticker_obj.info
                # Fallback for name if info fails
                if 'longName' not in self.info:
                    self.info['longName'] = self.ticker
            except:
                self.info['longName'] = self.ticker
                
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def analyze_news_sentiment(self):
        """
        Robust Hybrid News Fetcher:
        1. Tries Yahoo Finance Native News first.
        2. Fallback to Google News RSS (Very reliable, no rate limits).
        """
        headlines = []
        
        try:
            # --- METHOD 1: YAHOO FINANCE NATIVE ---
            try:
                ticker_obj = yf.Ticker(self.ticker)
                yf_news = ticker_obj.news
                
                if yf_news:
                    for item in yf_news[:5]:
                        title = item.get('title')
                        if title:
                            headlines.append(title)
            except Exception:
                pass # Fail silently and move to fallback

            # --- METHOD 2: GOOGLE NEWS RSS FALLBACK ---
            # If Yahoo gave no results (common for smaller ID stocks), use Google RSS
            if not headlines:
                try:
                    # Clean name for search
                    query = self.ticker.replace(".JK", "")
                    long_name = self.info.get('longName', '')
                    if long_name and long_name != self.ticker:
                        query = long_name.replace("PT ", "").replace(" Tbk", "").strip()
                    
                    # Robust RSS URL (Global English)
                    rss_url = f"https://news.google.com/rss/search?q={query}+Indonesia+stock&hl=en-US&gl=US&ceid=US:en"
                    
                    response = requests.get(rss_url, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, features="xml")
                        items = soup.findAll('item')
                        for item in items[:5]:
                            title = item.find('title').text
                            headlines.append(title)
                except Exception as e:
                    print(f"RSS Fallback Error: {e}")

            # --- SENTIMENT ANALYSIS ---
            if not headlines:
                self.news_analysis = {
                    "sentiment": "Neutral (No News Found)", 
                    "score": 0, 
                    "headlines": ["No recent news available via Yahoo or Google RSS."]
                }
                return

            scores = []
            final_headlines = []

            for title in headlines:
                blob = TextBlob(title)
                scores.append(blob.sentiment.polarity)
                final_headlines.append(title)

            if scores:
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = 0

            # Categorize
            if avg_score > 0.1:
                sentiment = "Positive (Bullish)"
            elif avg_score < -0.1:
                sentiment = "Negative (Bearish)"
            else:
                sentiment = "Neutral"

            self.news_analysis = {
                "sentiment": sentiment,
                "score": round(avg_score, 3),
                "headlines": final_headlines[:3] # Show top 3
            }
            
        except Exception as e:
            # Absolute Fail-safe
            self.news_analysis = {
                "sentiment": "Neutral (Error)", 
                "score": 0, 
                "headlines": [f"Analysis Error: {str(e)}"]
            }

    def prepare_indicators(self):
        """
        Calculates technical indicators using pandas_ta.
        ADAPTIVE: Adjusts for IPOs/Short history.
        """
        if self.df is None or self.df.empty:
            return

        # 1. Adaptive Trend Filter (For IPOs)
        # If history < 200 days, we can't use EMA 200.
        if self.data_len >= 200:
            self.df['EMA_200'] = ta.ema(self.df['Close'], length=200)
            self.active_trend_col = 'EMA_200'
        elif self.data_len >= 50:
            self.df['EMA_50'] = ta.ema(self.df['Close'], length=50)
            self.active_trend_col = 'EMA_50'
        else:
            # Very new IPO, just use a short MA as reference
            self.df['EMA_20'] = ta.ema(self.df['Close'], length=20)
            self.active_trend_col = 'EMA_20'

        # 2. Standard Oscillators (Require min ~14 days)
        if self.data_len > RSI_PERIOD:
            self.df['RSI'] = ta.rsi(self.df['Close'], length=RSI_PERIOD)
            
            stoch = ta.stoch(self.df['High'], self.df['Low'], self.df['Close'], k=STOCH_K_PERIOD, d=STOCH_D_PERIOD)
            self.df = pd.concat([self.df, stoch], axis=1)

        # 3. EMAs for Crossover (Safe Calculation)
        # Only calculate if data length allows the slow MA
        for fast, slow in MA_TEST_PAIRS:
            if self.data_len > slow:
                self.df[f'EMA_{fast}'] = ta.ema(self.df['Close'], length=fast)
                self.df[f'EMA_{slow}'] = ta.ema(self.df['Close'], length=slow)

        # 4. OBV (On-Balance Volume)
        self.df['OBV'] = ta.obv(self.df['Close'], self.df['Volume'])

        # 5. ATR (Average True Range) for Stop Loss
        self.df['ATR'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=ATR_PERIOD)

    def run_backtest_simulation(self, condition_series, hold_days):
        """
        Simulates trades.
        Input: A pandas Series of Booleans (True where signal triggers).
        Output: Win Rate (%)
        """
        if condition_series is None:
            return 0.0 
        
        signals = self.df[condition_series].copy()
        
        if signals.empty:
            return 0.0

        # Vectorized Forward Return Calculation
        # We look at the price 'hold_days' into the future
        future_close = self.df['Close'].shift(-hold_days)
        current_close = self.df['Close']
        
        # Calculate percent change if bought today and sold in X days
        returns = (future_close - current_close) / current_close
        
        # Filter only rows where signal was True
        trade_returns = returns[condition_series]
        
        # Drop NaNs (last few days where future data doesn't exist)
        trade_returns = trade_returns.dropna()
        
        if trade_returns.empty:
            return 0.0
            
        wins = (trade_returns > 0).sum()
        total_trades = len(trade_returns)
        
        return (wins / total_trades) * 100

    def optimize_stock(self):
        """
        Deep Grid Search:
        Returns a LIST of the best strategy for EACH category (RSI, MA, Stoch).
        """
        # Trackers for best in each category
        best_rsi = {"strategy": "RSI Reversal", "win_rate": 0, "details": "N/A", "hold_days": 0, "is_triggered_today": False}
        best_ma = {"strategy": "MA Momentum", "win_rate": 0, "details": "N/A", "hold_days": 0, "is_triggered_today": False}
        best_stoch = {"strategy": "Stoch Reversal", "win_rate": 0, "details": "N/A", "hold_days": 0, "is_triggered_today": False}

        # 1. Test RSI
        if 'RSI' in self.df.columns:
            for level in RSI_TEST_LEVELS:
                condition = self.df['RSI'] < level
                for days in range(1, MAX_HOLD_DAYS + 1):
                    wr = self.run_backtest_simulation(condition, days)
                    if wr > best_rsi['win_rate']:
                        best_rsi = {
                            "strategy": "RSI Reversal",
                            "details": f"RSI < {level}",
                            "win_rate": wr,
                            "hold_days": days,
                            "is_triggered_today": self.df['RSI'].iloc[-1] < level
                        }

        # 2. Test MA
        for fast, slow in MA_TEST_PAIRS:
            fast_col, slow_col = f'EMA_{fast}', f'EMA_{slow}'
            if fast_col in self.df.columns and slow_col in self.df.columns:
                condition = self.df[fast_col] > self.df[slow_col]
                for days in range(1, MAX_HOLD_DAYS + 1):
                    wr = self.run_backtest_simulation(condition, days)
                    if wr > best_ma['win_rate']:
                        best_ma = {
                            "strategy": "MA Momentum",
                            "details": f"EMA {fast} > EMA {slow}",
                            "win_rate": wr,
                            "hold_days": days,
                            "is_triggered_today": self.df[fast_col].iloc[-1] > self.df[slow_col].iloc[-1]
                        }

        # 3. Test Stochastic
        k_col = f"STOCHk_{STOCH_K_PERIOD}_{STOCH_D_PERIOD}_{STOCH_D_PERIOD}"
        d_col = f"STOCHd_{STOCH_K_PERIOD}_{STOCH_D_PERIOD}_{STOCH_D_PERIOD}"
        if k_col in self.df.columns:
            condition = (self.df[k_col] < STOCH_OVERSOLD) & (self.df[k_col] > self.df[d_col])
            for days in range(1, MAX_HOLD_DAYS + 1):
                wr = self.run_backtest_simulation(condition, days)
                if wr > best_stoch['win_rate']:
                    best_stoch = {
                        "strategy": "Stoch Reversal",
                        "details": f"Stoch K < {STOCH_OVERSOLD}",
                        "win_rate": wr,
                        "hold_days": days,
                        "is_triggered_today": (self.df[k_col].iloc[-1] < STOCH_OVERSOLD) & (self.df[k_col].iloc[-1] > self.df[d_col].iloc[-1])
                    }

        # Collect valid strategies (win_rate > 0)
        all_strats = [s for s in [best_rsi, best_ma, best_stoch] if s['win_rate'] > 0]
        
        # Sort by Win Rate Descending
        all_strats.sort(key=lambda x: x['win_rate'], reverse=True)
        
        if not all_strats:
             return [{"strategy": "None", "win_rate": 0, "details": "No profitable strategy found", "hold_days": 0, "is_triggered_today": False}]

        return all_strats

    def detect_vcp_pattern(self):
        """
        Volatility Contraction Pattern (VCP) Detection.
        Logic: 
        1. Find recent Pivot Highs and Lows.
        2. Measure the 'depth' of each pullback (High to Low).
        3. Check if pullbacks are getting smaller (contracting).
        4. Returns: { "detected": Bool, "msg": Str }
        """
        try:
            # We need at least ~60 days to detect VCP structure
            if self.data_len < 60:
                return {"detected": False, "msg": "Insufficient Data"}

            # Work on last 60 days of Close prices
            # Using a simple rolling max/min to find local pivots
            window_size = 5
            recent_df = self.df[-60:].copy()
            
            # Find Local Highs (Peaks)
            recent_df['is_peak'] = recent_df['High'] == recent_df['High'].rolling(window=5, center=True).max()
            peaks = recent_df[recent_df['is_peak']]
            
            # Find Local Lows (Troughs)
            recent_df['is_trough'] = recent_df['Low'] == recent_df['Low'].rolling(window=5, center=True).min()
            troughs = recent_df[recent_df['is_trough']]
            
            # Need at least 2 peaks and 2 troughs to compare volatility
            if len(peaks) < 2 or len(troughs) < 2:
                return {"detected": False, "msg": "No clear Swing Pattern"}

            # Get last 2 significant pullbacks
            last_peak_price = peaks['High'].iloc[-1]
            prev_peak_price = peaks['High'].iloc[-2]
            
            last_trough_price = troughs['Low'].iloc[-1]
            prev_trough_price = troughs['Low'].iloc[-2]
            
            # Calculate Depth of Corrections (Drawdown %)
            # Correction 1 (Older)
            depth_1 = (prev_peak_price - prev_trough_price) / prev_peak_price
            
            # Correction 2 (Newer)
            depth_2 = (last_peak_price - last_trough_price) / last_peak_price
            
            # VCP Condition:
            # 1. Newer depth must be smaller than older depth (Contraction)
            # 2. Depths must be reasonable (e.g. not 0.1% noise)
            # 3. Price is currently near the recent High (Setup for breakout)
            
            is_contracting = depth_2 < (depth_1 * 0.9) # At least 10% tighter
            current_price = self.df['Close'].iloc[-1]
            near_breakout = current_price >= (last_peak_price * 0.95)
            
            if is_contracting and near_breakout:
                d1_pct = depth_1 * 100
                d2_pct = depth_2 * 100
                return {
                    "detected": True, 
                    "msg": f"Contraction detected! Volatility dropped from {d1_pct:.1f}% to {d2_pct:.1f}%."
                }
            else:
                return {"detected": False, "msg": "Volatility is not contracting clearly."}
                
        except Exception as e:
            return {"detected": False, "msg": f"Error calculating VCP: {str(e)}"}

    def detect_geometric_patterns(self):
        """
        Detects Triangles (Symmetrical, Ascending, Descending) and Pennants.
        Logic: Analyzes slopes of recent peaks and troughs.
        """
        result = {"pattern": "None", "msg": ""}
        try:
            if self.data_len < 60: return result
            
            # Identify Pivots (reuse simple logic)
            df = self.df[-60:].copy()
            df['is_peak'] = df['High'] == df['High'].rolling(window=5, center=True).max()
            df['is_trough'] = df['Low'] == df['Low'].rolling(window=5, center=True).min()
            
            peaks = df[df['is_peak']]
            troughs = df[df['is_trough']]
            
            if len(peaks) < 2 or len(troughs) < 2: return result
            
            # Latest 2 peaks and 2 troughs
            p2, p1 = peaks['High'].iloc[-1], peaks['High'].iloc[-2] # p2 is latest
            t2, t1 = troughs['Low'].iloc[-1], troughs['Low'].iloc[-2] # t2 is latest
            
            # Slopes (Simplified: positive = up, negative = down)
            # Tolerance 1% for flat lines
            is_lower_highs = p2 < (p1 * 0.99)
            is_higher_lows = t2 > (t1 * 1.01)
            is_flat_highs  = abs(p2 - p1) / p1 < 0.01
            is_flat_lows   = abs(t2 - t1) / t1 < 0.01
            
            # 1. Symmetrical Triangle (Coil)
            if is_lower_highs and is_higher_lows:
                result["pattern"] = "Symmetrical Triangle"
                result["msg"] = "Price coiling. Lower Highs & Higher Lows. Breakout imminent."
                
            # 2. Ascending Triangle (Bullish)
            elif is_flat_highs and is_higher_lows:
                result["pattern"] = "Ascending Triangle"
                result["msg"] = "Bullish Setup. Flat Resistance, Higher Lows."
                
            # 3. Descending Triangle (Bearish)
            elif is_lower_highs and is_flat_lows:
                result["pattern"] = "Descending Triangle"
                result["msg"] = "Bearish Setup. Lower Highs, Flat Support."
            
            # 4. Pennant Check (Requires a 'Pole')
            # Pole: Price surge > 10% in short time before the current consolidation
            if result["pattern"] != "None":
                # Look back before the consolidation (approx 20-40 days ago)
                pre_consolid_df = self.df[-50:-20] 
                if not pre_consolid_df.empty:
                    low_start = pre_consolid_df['Low'].min()
                    high_end = pre_consolid_df['High'].max()
                    move_pct = (high_end - low_start) / low_start
                    
                    # If there was a >15% move before this triangle, it's a Pennant
                    if move_pct > 0.15:
                        result["pattern"] = f"Bullish Pennant ({result['pattern']})"
                        result["msg"] += " + Strong Pole Detected."

        except Exception: pass
        return result

    def get_market_context(self):
        """
        Calculates Price Action levels and Context (IPO Safe).
        Added Fibonacci Logic and VCP/Geo Patterns.
        """
        last_price = self.df['Close'].iloc[-1]
        
        # 1. Basic 20-Day Support/Resistance
        lookback = min(20, self.data_len)
        recent_window = self.df[-lookback:]
        support = recent_window['Low'].min()
        resistance = recent_window['High'].max()
        dist_to_supp = ((last_price - support) / support) * 100
        
        # 2. FIBONACCI CALCULATION (Major Trend)
        fib_lookback = min(FIB_LOOKBACK_DAYS, self.data_len)
        fib_window = self.df[-fib_lookback:]
        swing_high = fib_window['High'].max()
        swing_low = fib_window['Low'].min()
        trend_range = swing_high - swing_low
        
        fib_levels = {}
        if trend_range > 0:
            fib_levels = {
                "1.0 (Low)": swing_low,
                "0.786": swing_high - (0.786 * trend_range),
                "0.618 (Golden)": swing_high - (0.618 * trend_range),
                "0.5 (Half)": swing_high - (0.5 * trend_range),
                "0.382": swing_high - (0.382 * trend_range),
                "0.236": swing_high - (0.236 * trend_range),
                "0.0 (High)": swing_high
            }
        
        # 3. Trend
        if self.active_trend_col in self.df.columns:
            ema_val = self.df[self.active_trend_col].iloc[-1]
            trend = "UPTREND" if last_price > ema_val else "DOWNTREND"
        else:
            trend = "NEUTRAL"
            
        # 4. ATR Value
        if 'ATR' in self.df.columns:
            atr = self.df['ATR'].iloc[-1]
        else:
            atr = 0
        
        # 5. OBV Divergence (Crucial Feature)
        obv_status = "Neutral"
        if self.data_len > OBV_LOOKBACK_DAYS:
            curr_obv = self.df['OBV'].iloc[-1]
            prev_obv = self.df['OBV'].iloc[-OBV_LOOKBACK_DAYS]
            last_p = self.df['Close'].iloc[-1]
            prev_p = self.df['Close'].iloc[-OBV_LOOKBACK_DAYS]
            
            if last_p < prev_p and curr_obv > prev_obv:
                obv_status = "Bullish Divergence"
            elif last_p > prev_p and curr_obv < prev_obv:
                obv_status = "Bearish Divergence"

        # 6. Patterns
        vcp_status = self.detect_vcp_pattern()
        geo_status = self.detect_geometric_patterns()

        return {
            "price": last_price,
            "support": support,
            "resistance": resistance,
            "dist_support": dist_to_supp,
            "fib_levels": fib_levels,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "trend": trend,
            "atr": atr,
            "obv_status": obv_status,
            "vcp": vcp_status,
            "geo": geo_status
        }

    def calculate_trade_plan(self, action, current_price, atr, support, resistance, best_strategy, fib_levels):
        """
        Generates specific Entry, SL, and TP targets.
        SMART WAIT + FIBONACCI INJECTION.
        """
        plan = {
            "entry": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward": "N/A",
            "status": "ACTIVE"
        }
        
        # Logic 1: ACTIVE BUY SIGNAL
        if "BUY" in action:
            plan['entry'] = current_price
            plan['status'] = "EXECUTE NOW (Market Order)"
            
            # SL: 2x ATR below entry
            sl_dist = (atr * SL_MULTIPLIER) if atr > 0 else (current_price * 0.05)
            plan['stop_loss'] = current_price - sl_dist
            
            # TP: 3x ATR above entry
            tp_dist = (atr * TP_MULTIPLIER) if atr > 0 else (current_price * 0.10)
            plan['take_profit'] = current_price + tp_dist
            
            plan['risk_reward'] = "1:1.5 (Dynamic ATR)"
            
        # Logic 2: WAIT SIGNAL (Smart Suggestion)
        elif "WAIT" in action:
            plan['status'] = "PENDING (Wait for Limit)"
            
            strategy_type = best_strategy.get('strategy', 'None')
            
            if "RSI" in strategy_type or "Stoch" in strategy_type:
                # INJECT FIBONACCI PREDICTION
                fib_618 = fib_levels.get('0.618 (Golden)', 0)
                
                if fib_618 > 0 and fib_618 < current_price and fib_618 > (current_price * 0.85):
                    plan['entry'] = fib_618
                    plan['note'] = f"Waiting for Golden Ratio (0.618 Fib) Retracement."
                else:
                    plan['entry'] = support
                    plan['note'] = f"Strategy uses {best_strategy['details']}. Wait for Support."
            elif "MA" in strategy_type:
                plan['entry'] = resistance
                plan['note'] = f"Momentum Strategy. Buy Breakout > {resistance:,.0f}."
            else:
                plan['entry'] = support
                plan['note'] = "Wait for Support."

            # Calculate SL/TP based on this theoretical Entry
            if plan['entry'] > 0:
                sl_dist = (atr * SL_MULTIPLIER) if atr > 0 else (plan['entry'] * 0.05)
                plan['stop_loss'] = plan['entry'] - sl_dist
                
                tp_dist = (atr * TP_MULTIPLIER) if atr > 0 else (plan['entry'] * 0.10)
                plan['take_profit'] = plan['entry'] + tp_dist
                plan['risk_reward'] = "Projection (If Filled)"
            
        return plan

    def generate_final_report(self):
        """Orchestrates the analysis."""
        if not self.fetch_data():
            return None
            
        self.prepare_indicators()
        self.analyze_news_sentiment()
        
        top_strategies = self.optimize_stock()
        best = top_strategies[0]
        ctx = self.get_market_context()
        
        action = "WAIT"
        trigger_msg = "No clear signal matched best strategy."
        
        if best['is_triggered_today']:
            if "RSI" in str(best['strategy']) or "Stoch" in str(best['strategy']):
                action = "ACTION: BUY ON DIP (Statistical Winner)"
            elif "MA" in str(best['strategy']):
                action = "ACTION: BUY MOMENTUM (Statistical Winner)"
            trigger_msg = f"Triggered by {best['details']}"
        elif ctx['dist_support'] < 3.0: 
             if action == "WAIT":
                action = "ACTION: BUY ON DIP (Support Proximity)"
                trigger_msg = f"Price is {ctx['dist_support']:.1f}% from Support."

        if ctx['trend'] == "DOWNTREND" and "MOMENTUM" in action:
            action = "WAIT (Downtrend Limit)"
            trigger_msg += f" [Blocked by Trend]"
            
        if "Negative" in self.news_analysis['sentiment'] and "BUY" in action:
             action = "CAUTION / WAIT"
             trigger_msg += " [Blocked by Negative Sentiment]"

        # Chart Pattern Override (VCP or Triangle)
        if (ctx['vcp']['detected'] or ctx['geo']['pattern'] != "None") and action == "WAIT":
             trigger_msg += f" [PATTERN WATCH: {ctx['geo']['pattern'] if ctx['geo']['pattern'] != 'None' else 'VCP'}]"

        trade_plan = self.calculate_trade_plan(
            action, 
            ctx['price'], 
            ctx['atr'], 
            ctx['support'], 
            ctx['resistance'],
            best,
            ctx['fib_levels']
        )

        return {
            "ticker": self.ticker,
            "name": self.info.get('longName', self.ticker),
            "price": ctx['price'],
            "action": action,
            "trigger": trigger_msg,
            "sentiment": self.news_analysis,
            "best_strategy": best,
            "all_strategies": top_strategies, 
            "context": ctx,
            "trade_plan": trade_plan,
            "is_ipo": self.data_len < 200,
            "days_listed": self.data_len
        }