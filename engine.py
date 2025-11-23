import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import random
from GoogleNews import GoogleNews
from textblob import TextBlob
from datetime import datetime, timedelta

# ==========================================
# 1. MODULAR CONFIGURATION (BONGKAR PASANG)
# ==========================================

# Timeframe & Data
BACKTEST_PERIOD = "2y"       # Data range to fetch
MAX_HOLD_DAYS = 20           # Max days to test in Grid Search (1 to 20)

# Strategy A: RSI Parameters
RSI_PERIOD = 14
RSI_TEST_LEVELS = [30, 40]   # Oversold levels to test

# Strategy B: Moving Average Pairs (Fast, Slow)
MA_TEST_PAIRS = [(5, 20), (10, 50), (20, 200)]

# Strategy C: Stochastic Parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20

# Trend & OBV
OBV_LOOKBACK_DAYS = 5        # For divergence check
TREND_EMA = 200              # Major trend filter

# ==========================================
# 2. CLASS DEFINITION
# ==========================================

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = self._format_ticker(ticker)
        self.df = None
        self.info = {}
        self.news_analysis = {"sentiment": "Neutral", "score": 0, "headlines": []}

    def _format_ticker(self, ticker):
        ticker = ticker.upper().strip()
        if not ticker.endswith(".JK"):
            ticker += ".JK"
        return ticker

    def fetch_data(self):
        """Fetches 2 years of daily data using yfinance."""
        try:
            # FIX 1: Added auto_adjust=True to silence FutureWarning
            self.df = yf.download(self.ticker, period=BACKTEST_PERIOD, progress=False, auto_adjust=True)
            if self.df.empty:
                return False
            
            # Flatten MultiIndex columns if present (common in new yfinance versions)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)
            
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
        Fetches top 5 English news and calculates sentiment polarity.
        FIX 2: Added robust error handling and retries for HTTP 429.
        """
        try:
            # Force English language for TextBlob compatibility
            googlenews = GoogleNews(lang='en', region='US') 
            
            # 1. Determine Search Query (Prioritize Clean Company Name)
            query_term = self.ticker.replace(".JK", "") # Default fallback
            
            long_name = self.info.get('longName', '')
            
            # If we have a real name (not just the ticker placeholder from fetch_data fallback)
            if long_name and long_name != self.ticker:
                # Remove "PT", "Tbk", and dots to get the clean common name
                clean_name = long_name.replace("PT ", "").replace("PT.", "").replace(" Tbk", "").replace(" tbk", "").strip()
                if clean_name:
                    query_term = clean_name
            
            # Final Query: Name + Context (e.g. "Bank Central Asia Indonesia stock")
            query = f"{query_term} Indonesia stock"

            results = []
            
            # Retry Mechanism for 429 Errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    googlenews.search(query)
                    results = googlenews.result()
                    if results:
                        break # Success
                    time.sleep(random.uniform(1, 2)) # Random delay
                except Exception:
                    # If error, wait longer and retry
                    time.sleep(2)
                    continue

            scores = []
            headlines = []
            
            # Process top 5
            for item in results[:5]:
                title = item.get('title', '')
                if not title:
                    continue
                blob = TextBlob(title)
                polarity = blob.sentiment.polarity
                scores.append(polarity)
                headlines.append(title)

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
                "headlines": headlines[:3] # Keep top 3 for display
            }
            
        except Exception as e:
            # Fail gracefully so the rest of the app still runs
            # print(f"News Warning: Could not fetch news ({e})") 
            self.news_analysis = {
                "sentiment": "Neutral (News Unavailable)", 
                "score": 0, 
                "headlines": ["News data temporarily unavailable (Rate Limit)."]
            }

    def prepare_indicators(self):
        """Calculates technical indicators using pandas_ta."""
        if self.df is None or self.df.empty:
            return

        # Trend
        self.df['EMA_200'] = ta.ema(self.df['Close'], length=TREND_EMA)
        
        # RSI
        self.df['RSI'] = ta.rsi(self.df['Close'], length=RSI_PERIOD)
        
        # Stochastic
        stoch = ta.stoch(self.df['High'], self.df['Low'], self.df['Close'], k=STOCH_K_PERIOD, d=STOCH_D_PERIOD)
        self.df = pd.concat([self.df, stoch], axis=1) # Append STOCHk and STOCHd
        
        # EMAs for Crossover
        for fast, slow in MA_TEST_PAIRS:
            self.df[f'EMA_{fast}'] = ta.ema(self.df['Close'], length=fast)
            self.df[f'EMA_{slow}'] = ta.ema(self.df['Close'], length=slow)

        # OBV
        self.df['OBV'] = ta.obv(self.df['Close'], self.df['Volume'])

    def run_backtest_simulation(self, condition_series, hold_days):
        """
        Simulates trades.
        Input: A pandas Series of Booleans (True where signal triggers).
        Output: Win Rate (%)
        """
        signals = self.df[condition_series].copy()
        
        if signals.empty:
            return 0.0

        wins = 0
        total_trades = 0

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
        Iterates Strategies -> Parameters -> Hold Days (1-20).
        Finds the absolute best combo.
        """
        best_result = {
            "strategy": None,
            "details": "",
            "win_rate": -1,
            "hold_days": 0,
            "is_triggered_today": False
        }

        # 1. Test RSI Strategies
        for level in RSI_TEST_LEVELS:
            # Condition: RSI crosses below level (Dip opportunity)
            condition = self.df['RSI'] < level
            
            # Loop days 1 to MAX_HOLD_DAYS
            for days in range(1, MAX_HOLD_DAYS + 1):
                wr = self.run_backtest_simulation(condition, days)
                if wr > best_result['win_rate']:
                    current_val = self.df['RSI'].iloc[-1]
                    best_result = {
                        "strategy": "RSI Reversal",
                        "details": f"RSI < {level}",
                        "win_rate": wr,
                        "hold_days": days,
                        "is_triggered_today": current_val < level
                    }

        # 2. Test MA Crossover Strategies
        for fast, slow in MA_TEST_PAIRS:
            # Condition: Fast > Slow (Momentum)
            # We check if Fast was < Slow yesterday, and > Slow today (Crossover)
            fast_col = f'EMA_{fast}'
            slow_col = f'EMA_{slow}'
            
            # Logic: Buy when Fast is above Slow. 
            # Simple logic: Condition is True when trend is valid
            condition = self.df[fast_col] > self.df[slow_col]
            
            for days in range(1, MAX_HOLD_DAYS + 1):
                wr = self.run_backtest_simulation(condition, days)
                if wr > best_result['win_rate']:
                    curr_fast = self.df[fast_col].iloc[-1]
                    curr_slow = self.df[slow_col].iloc[-1]
                    best_result = {
                        "strategy": "MA Momentum",
                        "details": f"EMA {fast} > EMA {slow}",
                        "win_rate": wr,
                        "hold_days": days,
                        "is_triggered_today": curr_fast > curr_slow
                    }

        # 3. Test Stochastic
        stoch_k_col = f"STOCHk_{STOCH_K_PERIOD}_{STOCH_D_PERIOD}_{STOCH_D_PERIOD}"
        stoch_d_col = f"STOCHd_{STOCH_K_PERIOD}_{STOCH_D_PERIOD}_{STOCH_D_PERIOD}"
        
        # Condition: K < Oversold AND K > D (Turning up)
        condition = (self.df[stoch_k_col] < STOCH_OVERSOLD) & (self.df[stoch_k_col] > self.df[stoch_d_col])
        
        for days in range(1, MAX_HOLD_DAYS + 1):
            wr = self.run_backtest_simulation(condition, days)
            if wr > best_result['win_rate']:
                k_val = self.df[stoch_k_col].iloc[-1]
                d_val = self.df[stoch_d_col].iloc[-1]
                triggered = (k_val < STOCH_OVERSOLD) and (k_val > d_val)
                
                best_result = {
                    "strategy": "Stoch Reversal",
                    "details": f"Stoch K < {STOCH_OVERSOLD} & K crosses D",
                    "win_rate": wr,
                    "hold_days": days,
                    "is_triggered_today": triggered
                }

        return best_result

    def get_market_context(self):
        """Calculates Price Action levels and Context."""
        last_price = self.df['Close'].iloc[-1]
        
        # Support/Resis (Last 20 days)
        recent_window = self.df[-20:]
        support = recent_window['Low'].min()
        resistance = recent_window['High'].max()
        
        dist_to_supp = ((last_price - support) / support) * 100
        dist_to_res = ((resistance - last_price) / last_price) * 100
        
        # Trend
        ema_200 = self.df['EMA_200'].iloc[-1]
        trend = "UPTREND" if last_price > ema_200 else "DOWNTREND"
        
        # OBV Divergence (Simple check)
        curr_obv = self.df['OBV'].iloc[-1]
        prev_obv = self.df['OBV'].iloc[-OBV_LOOKBACK_DAYS]
        
        obv_status = "Neutral"
        if last_price < self.df['Close'].iloc[-OBV_LOOKBACK_DAYS] and curr_obv > prev_obv:
            obv_status = "Bullish Divergence (Accumulation)"
        elif last_price > self.df['Close'].iloc[-OBV_LOOKBACK_DAYS] and curr_obv < prev_obv:
            obv_status = "Bearish Divergence (Distribution)"

        return {
            "price": last_price,
            "support": support,
            "resistance": resistance,
            "dist_support": dist_to_supp,
            "trend": trend,
            "obv_status": obv_status
        }

    def generate_final_report(self):
        """Orchestrates the analysis."""
        if not self.fetch_data():
            return None
            
        self.prepare_indicators()
        self.analyze_news_sentiment()
        
        best_strat = self.optimize_stock()
        context = self.get_market_context()
        
        # DECISION LOGIC
        action = "WAIT"
        trigger_msg = "No clear signal matched the best historical strategy."
        
        # Logic 1: Check if Best Strategy is Active
        if best_strat['is_triggered_today']:
            if "RSI" in best_strat['strategy'] or "Stoch" in best_strat['strategy']:
                action = "ACTION: BUY ON DIP (Statistical Winner)"
            elif "MA" in best_strat['strategy']:
                action = "ACTION: BUY MOMENTUM (Statistical Winner)"
            trigger_msg = f"Triggered by Best Strategy: {best_strat['details']}"
            
        # Logic 2: Price Action Override (Support bounce)
        elif context['dist_support'] < 3.0: # Within 3% of support
             if action == "WAIT": # Only override if no other signal
                action = "ACTION: BUY ON DIP (Support Proximity)"
                trigger_msg = f"Price is {context['dist_support']:.1f}% from 20-day Support."

        # Logic 3: Trend Filter overrides
        if context['trend'] == "DOWNTREND" and "MOMENTUM" in action:
            action = "WAIT (Downtrend Limit)"
            trigger_msg += " [Blocked by 200 EMA Downtrend]"
            
        # Logic 4: Sentiment Override
        if "Negative" in self.news_analysis['sentiment'] and "BUY" in action:
             action = "CAUTION / WAIT"
             trigger_msg += " [Blocked by Negative Sentiment]"

        return {
            "ticker": self.ticker,
            "name": self.info.get('longName', self.ticker),
            "price": context['price'],
            "action": action,
            "trigger": trigger_msg,
            "sentiment": self.news_analysis,
            "best_strategy": best_strat,
            "context": context
        }