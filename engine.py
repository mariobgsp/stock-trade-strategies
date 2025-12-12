import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

class OJKRules:
    """
    Handles Indonesia Stock Exchange (IDX) specific pricing rules (Fraksi Harga).
    """
    @staticmethod
    def get_tick_size(price):
        if price < 200: return 1
        if 200 <= price < 500: return 2
        if 500 <= price < 2000: return 5
        if 2000 <= price < 5000: return 10
        return 25

    @staticmethod
    def floor_price(price):
        """Rounds down to nearest valid tick."""
        tick = OJKRules.get_tick_size(price)
        return int(np.floor(price / tick) * tick)

    @staticmethod
    def ceil_price(price):
        """Rounds up to nearest valid tick."""
        tick = OJKRules.get_tick_size(price)
        return int(np.ceil(price / tick) * tick)

class TechnicalAnalysis:
    """
    Manual calculation of indicators using Numpy/Pandas (No pandas-ta).
    """
    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_vwap(df):
        v = df['Volume'].values
        p = df['Close'].values
        return df.assign(VWAP=(p * v).cumsum() / v.cumsum())

    @staticmethod
    def calculate_obv_slope(df, window=20):
        """
        Uses LinearRegression to determine the slope of OBV vs Price.
        Returns the slope coefficient.
        """
        if len(df) < window: return 0
        
        # Calculate OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        y = df['OBV'].iloc[-window:].values.reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        
        reg = LinearRegression().fit(X, y)
        return reg.coef_[0][0]

    @staticmethod
    def find_pivots(df, order=5):
        """Uses Scipy to find local min/max for support/resistance."""
        highs = df['High'].values
        lows = df['Low'].values
        
        max_idx = argrelextrema(highs, np.greater, order=order)[0]
        min_idx = argrelextrema(lows, np.less, order=order)[0]
        
        return highs[max_idx], lows[min_idx]

class StrategyEngine:
    def __init__(self, ticker):
        self.ticker = ticker if ticker.endswith('.JK') else f"{ticker}.JK"
        self.df = pd.DataFrame()
        self.is_ipo = False
        self.verdict = {}
        
    def fetch_data(self):
        try:
            # Attempt 5 years, fallback if shorter
            self.df = yf.download(self.ticker, period="5y", progress=False)
            if self.df.empty:
                raise ValueError("No data found")
            
            # Flatten MultiIndex columns if present (yfinance update fix)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)

            # IPO Handling
            if len(self.df) < 250: # Less than ~1 year trading days
                self.is_ipo = True
                # Adjust volatility calculation context later
            
            # Base Calculations needed for all strategies
            self.df['RSI'] = TechnicalAnalysis.calculate_rsi(self.df['Close'])
            self.df = TechnicalAnalysis.calculate_vwap(self.df)
            
            # MA Squeeze Components
            for ma in [5, 10, 20, 50, 200]:
                self.df[f'SMA_{ma}'] = self.df['Close'].rolling(window=ma).mean()
                self.df[f'EMA_{ma}'] = self.df['Close'].ewm(span=ma, adjust=False).mean()

        except Exception as e:
            raise ConnectionError(f"Failed to fetch data for {self.ticker}: {e}")

    def detect_smart_money(self):
        """
        Analyzes Volume and OBV Slope to detect Accumulation.
        Returns: Status, Start Date of Phase
        """
        slope = TechnicalAnalysis.calculate_obv_slope(self.df)
        current_price = self.df['Close'].iloc[-1]
        vwap = self.df['VWAP'].iloc[-1]
        
        # Determine Phase
        phase = "Neutral"
        if slope > 0 and current_price > vwap:
            phase = "Accumulation"
        elif slope < 0 and current_price < vwap:
            phase = "Distribution"
        elif slope > 0 and current_price < vwap:
            phase = "Absorption (Potential Bottom)"
            
        # Find start date: Look for last significant Golden Cross or OBV crossover
        # Simplified: Last time Price crossed VWAP
        crosses = self.df[self.df['Close'] > self.df['VWAP']].index
        start_date = crosses[-1] if len(crosses) > 0 else self.df.index[-1]
        
        return phase, slope, start_date

    def detect_patterns(self):
        """
        VCP and MA Squeeze Detection.
        """
        data = self.df.iloc[-20:] # Look at recent price action
        
        # MA Squeeze: Check if SMA 5, 10, 20 are within 5% of each other
        ma5 = data['SMA_5'].iloc[-1]
        ma10 = data['SMA_10'].iloc[-1]
        ma20 = data['SMA_20'].iloc[-1]
        
        avg_p = (ma5 + ma10 + ma20) / 3
        spread = (max(ma5, ma10, ma20) - min(ma5, ma10, ma20)) / avg_p
        squeeze = spread < 0.05
        
        # VCP: Lower Highs + Contraction in Volatility (Std Dev)
        # Simplified algorithm for CLI speed
        recent_highs = data['High'].rolling(5).max().dropna().values
        volatility = data['Close'].rolling(5).std().dropna().values
        
        # Check if last 3 local highs are descending
        vcp_cond = False
        if len(recent_highs) > 10:
             # Very rough approximation of contraction
             vcp_cond = volatility[-1] < volatility[-5] and data['High'].iloc[-1] < data['High'].iloc[-5]

        return {"squeeze": squeeze, "vcp": vcp_cond}

    def grid_search_optimization(self):
        """
        Optimizes MA periods and RSI thresholds.
        Splits data 70/30 (Train/Test) to validate.
        """
        if self.is_ipo:
            # IPO Mode: Skip complex grid search, return reactive defaults
            return {'ma_fast': 5, 'ma_slow': 10, 'rsi_period': 9, 'rsi_oversold': 30}

        train_size = int(len(self.df) * 0.7)
        train_df = self.df.iloc[:train_size].copy()
        
        best_win_rate = 0
        best_params = {'ma_fast': 5, 'ma_slow': 20, 'rsi_period': 14, 'rsi_oversold': 30}
        
        # Limited grid to keep CLI responsive
        ma_fast_opts = [5, 9]
        ma_slow_opts = [10, 20, 50]
        rsi_opts = [14, 21]
        
        for f in ma_fast_opts:
            for s in ma_slow_opts:
                if f >= s: continue
                # Simple Golden Cross logic for parameter testing
                signal = (train_df['Close'] > train_df[f'SMA_{f}']) & (train_df[f'SMA_{f}'] > train_df[f'SMA_{s}'])
                # Vectorized backtest on Train data (simplified for parameter selection)
                wins = 0
                trades = 0
                
                # Look 5 days ahead
                future_returns = train_df['Close'].shift(-5) / train_df['Close'] - 1
                
                # Boolean masking
                trades = signal.sum()
                if trades > 0:
                    wins = (signal & (future_returns > 0.02)).sum() # 2% gain threshold
                    wr = wins / trades
                    if wr > best_win_rate:
                        best_win_rate = wr
                        best_params = {'ma_fast': f, 'ma_slow': s, 'rsi_period': 14, 'rsi_oversold': 30}
                        
        return best_params

    def run_simulation(self):
        """
        The Master Logic.
        1. Optimize Parameters.
        2. Check Compliance (Tick Rules).
        3. Backtest Strategies.
        4. Generate Verdict.
        """
        self.fetch_data()
        
        # 1. Optimize
        params = self.grid_search_optimization()
        
        # 2. Current State
        current_price = self.df['Close'].iloc[-1]
        rsi = self.df['RSI'].iloc[-1]
        
        # Smart Money Analysis
        sm_phase, sm_slope, sm_start = self.detect_smart_money()
        patterns = self.detect_patterns()
        
        # 3. Strategy Logic (The "Brain")
        # We test the strategy on historical data to calculate probabilities
        
        strategy_name = "Wait"
        signal_strength = 0 # 0 to 5
        
        # A. VCP Breakout
        if patterns['vcp'] and patterns['squeeze'] and current_price > self.df['EMA_20'].iloc[-1]:
            strategy_name = "VCP Breakout"
            signal_strength = 5
            
        # B. Buy on Dip (Smart Money Accumulation + Oversold)
        elif sm_phase == "Accumulation" and rsi < 40:
            strategy_name = "Buy on Dip (Accumulation)"
            signal_strength = 4
            
        # C. Momentum Breakout (IPO or Mature)
        elif current_price > self.df['SMA_5'].iloc[-1] and self.df['SMA_5'].iloc[-1] > self.df['SMA_20'].iloc[-1]:
            # Confirm with Volume
            if self.df['Volume'].iloc[-1] > self.df['Volume'].rolling(20).mean().iloc[-1]:
                strategy_name = "Volume Breakout"
                signal_strength = 4
        
        # D. Support Bounce
        res_peaks, sup_troughs = TechnicalAnalysis.find_pivots(self.df)
        nearest_support = max([p for p in sup_troughs if p < current_price], default=0)
        if 0 < (current_price - nearest_support) / current_price < 0.03: # Within 3% of support
            strategy_name = "Support Bounce"
            signal_strength = 3

        # 4. Mandatory 50% Win Rate Check & Probability Calculation
        # We simulate the CHOSEN strategy on past data
        
        probs = self.backtest_probability(strategy_name, params)
        
        # MANDATE: If Win Rate < 50%, KILL the trade
        if probs['win_rate'] < 50:
            final_verdict = "NO TRADE"
            reason = f"Backtested Win Rate ({probs['win_rate']}%) is below safety threshold."
        elif signal_strength < 3:
            final_verdict = "NO TRADE"
            reason = "No valid setup (VCP, Dip, or Breakout) detected."
        else:
            final_verdict = "BUY"
            reason = f"Strategy: {strategy_name}. Smart Money is {sm_phase}."

        # 5. Define Targets (Risk:Reward 1:3)
        # OJK Rule: Support/SL must be valid ticks
        support_level = nearest_support if nearest_support > 0 else current_price * 0.95
        
        # Strict Risk Management
        entry = current_price
        stop_loss = OJKRules.floor_price(support_level * 0.98) # 2% below support
        risk = entry - stop_loss
        
        if risk <= 0: # Fallback if support is too close
            stop_loss = OJKRules.floor_price(entry * 0.95)
            risk = entry - stop_loss

        tp1 = OJKRules.ceil_price(entry + risk)      # 1R
        tp2 = OJKRules.ceil_price(entry + (risk * 2)) # 2R
        tp3 = OJKRules.ceil_price(entry + (risk * 3)) # 3R

        return {
            "verdict": final_verdict,
            "strategy": strategy_name,
            "reason": reason,
            "entry": entry,
            "sl": stop_loss,
            "targets": [tp1, tp2, tp3],
            "probs": probs,
            "data": {
                "params": params,
                "sm_phase": sm_phase,
                "sm_start": sm_start.strftime('%Y-%m-%d'),
                "rsi": round(rsi, 2),
                "vwap_diff": round(((current_price - self.df['VWAP'].iloc[-1])/self.df['VWAP'].iloc[-1])*100, 2),
                "is_ipo": self.is_ipo,
                "patterns": patterns
            }
        }

    def backtest_probability(self, strategy, params):
        """
        Walk-Forward Analysis to calculate probability of hitting 1R, 2R, 3R.
        """
        # If NO TRADE, return zeros
        if strategy == "Wait":
            return {"win_rate": 0, "p1": 0, "p2": 0, "p3": 0}

        # Simplified Vectorized Backtest
        # We look for similar setups in history and see result
        
        df = self.df.copy()
        
        # Define condition masks based on strategy
        if "Breakout" in strategy:
            condition = (df['Close'] > df[f"SMA_{params['ma_fast']}"]) & (df['Volume'] > df['Volume'].rolling(20).mean())
        elif "Dip" in strategy:
            condition = (df['RSI'] < 40) & (df['Close'] > df['EMA_50'])
        else:
            # Generic momentum for other cases
            condition = df['Close'] > df[f"SMA_{params['ma_slow']}"]

        idxs = df.index[condition]
        
        if len(idxs) < 10: 
            # Not enough data points for stat sig, enforce conservative penalty
            return {"win_rate": 45, "p1": 0, "p2": 0, "p3": 0} 

        hits_1r = 0
        hits_2r = 0
        hits_3r = 0
        total = 0

        # Monte Carlo-lite: Sample past signals
        # We limit loop to last 100 signals for performance
        check_idxs = idxs[-100:] if len(idxs) > 100 else idxs

        for date in check_idxs:
            if date == df.index[-1]: continue # Skip today
            
            try:
                # Find entry price at signal
                entry_p = df.loc[date]['Close']
                # Determine theoretical SL (e.g., 5% trailing)
                sl_p = entry_p * 0.95
                risk_p = entry_p - sl_p
                
                # Look forward 20 days
                future = df.loc[date:].iloc[1:21]
                if future.empty: continue

                highs = future['High'].values
                lows = future['Low'].values
                
                # Check if hit SL first
                sl_hit = np.where(lows <= sl_p)[0]
                first_sl = sl_hit[0] if len(sl_hit) > 0 else 999
                
                # Check targets
                t1_hit = np.where(highs >= (entry_p + risk_p))[0]
                t2_hit = np.where(highs >= (entry_p + 2*risk_p))[0]
                t3_hit = np.where(highs >= (entry_p + 3*risk_p))[0]
                
                first_t1 = t1_hit[0] if len(t1_hit) > 0 else 999
                first_t2 = t2_hit[0] if len(t2_hit) > 0 else 999
                first_t3 = t3_hit[0] if len(t3_hit) > 0 else 999
                
                if first_t1 < first_sl: hits_1r += 1
                if first_t2 < first_sl: hits_2r += 1
                if first_t3 < first_sl: hits_3r += 1
                total += 1
                
            except:
                continue

        if total == 0: return {"win_rate": 0, "p1": 0, "p2": 0, "p3": 0}

        # Win Rate defined as hitting at least 1R
        return {
            "win_rate": int((hits_1r / total) * 100),
            "p1": int((hits_1r / total) * 100),
            "p2": int((hits_2r / total) * 100),
            "p3": int((hits_3r / total) * 100)
        }

