import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# ==========================================
# 1. OJK / IDX COMPLIANCE UTILITIES
# ==========================================
class IDXRules:
    """
    Handles Indonesia Stock Exchange specific fraction (tick) rules.
    """
    @staticmethod
    def get_tick_size(price):
        if price < 200: return 1
        if 200 <= price < 500: return 2
        if 500 <= price < 2000: return 5
        if 2000 <= price < 5000: return 10
        return 25

    @staticmethod
    def round_to_tick(price):
        """Rounds a raw price to the nearest valid IDX tick."""
        tick = IDXRules.get_tick_size(price)
        return round(price / tick) * tick

# ==========================================
# 2. CORE CALCULATION ENGINE (Manual Indicators)
# ==========================================
class TechnicalAnalysis:
    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_obv(df):
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_stochastic(df, k_window=14, d_window=3):
        low_min = df['Low'].rolling(window=k_window).min()
        high_max = df['High'].rolling(window=k_window).max()
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_window).mean()
        return k, d

    @staticmethod
    def calculate_vwap(df):
        q = df['Volume'] * ((df['High'] + df['Low'] + df['Close']) / 3)
        return q.cumsum() / df['Volume'].cumsum()

    @staticmethod
    def detect_squeeze(df, periods=[5, 10, 20]):
        # "Superclose" logic: Check if MAs are within 5% of each other
        ma_values = []
        for p in periods:
            ma_values.append(df['Close'].rolling(window=p).mean().iloc[-1])
        
        min_ma = min(ma_values)
        max_ma = max(ma_values)
        
        # Avoid division by zero
        if min_ma == 0: return False
        
        diff_pct = (max_ma - min_ma) / min_ma
        return diff_pct < 0.05

    @staticmethod
    def detect_vcp(df, window=20):
        # Volatility Contraction Pattern logic: 
        # Look for decreasing standard deviation over split windows
        recent_vol = df['Close'].rolling(window=window).std().iloc[-1]
        past_vol = df['Close'].shift(window).rolling(window=window).std().iloc[-1]
        
        # Simple VCP check: Volatility is currently tighter than it was before
        if pd.isna(recent_vol) or pd.isna(past_vol) or past_vol == 0:
            return False
        return recent_vol < (past_vol * 0.7)

# ==========================================
# 3. SMART MONEY & SCIPY ANALYSIS
# ==========================================
class SmartMoneyAnalyzer:
    @staticmethod
    def analyze_obv_slope(obv_series, window=20):
        """
        Uses sklearn LinearRegression to determine the mathematical slope of OBV.
        Positive slope + Flat Price = Accumulation.
        """
        y = obv_series.iloc[-window:].values.reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        
        if len(y) < window: return 0, "Insufficient Data"
        
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0][0]
        
        start_date = obv_series.index[-window]
        return slope, start_date

    @staticmethod
    def find_bounce_zones(df, order=10):
        """
        Uses scipy.signal.argrelextrema to find historical rejection levels.
        """
        # Find local minima (Support)
        min_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
        # Find local maxima (Resistance)
        max_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
        
        supports = df.iloc[min_idx]['Low'].values
        resistances = df.iloc[max_idx]['High'].values
        
        return supports, resistances

# ==========================================
# 4. STRATEGY & BACKTESTING KERNEL
# ==========================================
class QuantEngine:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        if not self.ticker.endswith(".JK"):
            self.ticker += ".JK"
        self.df = None
        self.analysis_results = {}

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period="3y", interval="1d", progress=False)
            if self.df.empty:
                raise ValueError("No data found")
            
            # Basic cleaning
            self.df = self.df.dropna()
            
            # Calculate Indicators
            self.df['RSI'] = TechnicalAnalysis.calculate_rsi(self.df['Close'])
            self.df['OBV'] = TechnicalAnalysis.calculate_obv(self.df)
            self.df['VWAP'] = TechnicalAnalysis.calculate_vwap(self.df)
            self.df['Stoch_K'], self.df['Stoch_D'] = TechnicalAnalysis.calculate_stochastic(self.df)
            
        except Exception as e:
            return False, str(e)
        return True, "Data Loaded"

    def backtest_strategy(self, fast_ma, slow_ma, rsi_threshold):
        """
        Backtests a Golden Cross + RSI filter strategy over the 3-year dataset.
        Returns: Win Rate, Trade Count, Last Signal
        """
        data = self.df.copy()
        data['FastMA'] = data['Close'].rolling(window=fast_ma).mean()
        data['SlowMA'] = data['Close'].rolling(window=slow_ma).mean()
        
        # Vectorized Signal Logic
        # Buy: Fast > Slow AND RSI < rsi_threshold (Buy on dip/momentum start)
        data['Signal'] = np.where(
            (data['FastMA'] > data['SlowMA']) & 
            (data['FastMA'].shift(1) <= data['SlowMA'].shift(1)) & 
            (data['RSI'] < rsi_threshold), 1, 0
        )
        
        # Simulate Trades (Simplified for speed)
        trades = data[data['Signal'] == 1]
        if trades.empty:
            return 0.0, 0, None

        wins = 0
        total_trades = 0
        
        # Iterate to check outcomes (1:3 RR)
        for date, row in trades.iterrows():
            entry = row['Close']
            stop_loss = entry * 0.95 # 5% risk
            target = entry * 1.15    # 15% reward (1:3)
            
            # Look forward 20 days
            future = data.loc[date:].iloc[1:21]
            if future.empty: continue
            
            hit_tp = future[future['High'] >= target].shape[0] > 0
            hit_sl = future[future['Low'] <= stop_loss].shape[0] > 0
            
            if hit_tp and not hit_sl:
                wins += 1
            elif hit_tp and hit_sl:
                # If both hit, check which happened first (approximate)
                tp_idx = future[future['High'] >= target].index[0]
                sl_idx = future[future['Low'] <= stop_loss].index[0]
                if tp_idx < sl_idx: wins += 1
            
            total_trades += 1

        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        return win_rate, total_trades, data.iloc[-1]

    def optimize_and_analyze(self):
        # 1. Grid Search for Best Parameters
        best_wr = 0
        best_params = (0, 0, 0)
        
        ma_combinations = [(5, 10), (10, 20), (20, 50), (50, 100)]
        rsi_options = [50, 60, 70] # Filter out overbought entries
        
        for f, s in ma_combinations:
            for r in rsi_options:
                wr, count, _ = self.backtest_strategy(f, s, r)
                if wr > best_wr and count > 3: # Minimum sample size constraint
                    best_wr = wr
                    best_params = (f, s, r)

        # 2. Get Current Technical State
        current_price = self.df['Close'].iloc[-1]
        
        # Scipy Supports
        supports, resistances = SmartMoneyAnalyzer.find_bounce_zones(self.df)
        nearest_support = supports[supports < current_price].max() if any(supports < current_price) else current_price * 0.9
        
        # Sklearn Smart Money
        obv_slope, sm_date = SmartMoneyAnalyzer.analyze_obv_slope(self.df['OBV'])
        
        # Patterns
        is_squeeze = TechnicalAnalysis.detect_squeeze(self.df)
        is_vcp = TechnicalAnalysis.detect_vcp(self.df)
        
        # 3. Formulate Verdict
        # MANDATE: If Win Rate < 65%, force NO TRADE unless Smart Money overrides
        verdict = "WAIT"
        why = "Market conditions are ambiguous."
        
        strong_accumulation = obv_slope > 0 and current_price > self.df['VWAP'].iloc[-1]
        
        if best_wr >= 65:
            verdict = "BUY"
            why = f"Backtested Strategy ({best_params[0]}/{best_params[1]} MA) shows {best_wr:.1f}% Win Rate."
        elif best_wr >= 50 and (strong_accumulation or is_squeeze):
            verdict = "BUY (SPECULATIVE)"
            why = "Win rate moderate, but Smart Money Accumulation or Squeeze detected."
            best_wr += 15 # Boost confidence due to confluence
        else:
            verdict = "NO TRADE"
            why = f"Best strategy only yielded {best_wr:.1f}% WR. Too risky."

        # 4. IDX Compliant Trade Plan
        risk_per_share = (current_price - nearest_support)
        # Enforce minimum risk distance (don't set SL too tight)
        if risk_per_share / current_price < 0.02:
            risk_per_share = current_price * 0.03 # Min 3% risk

        stop_loss = IDXRules.round_to_tick(current_price - risk_per_share)
        tp1 = IDXRules.round_to_tick(current_price + (risk_per_share * 1.5))
        tp2 = IDXRules.round_to_tick(current_price + (risk_per_share * 3.0)) # 1:3 RR
        tp3 = IDXRules.round_to_tick(current_price + (risk_per_share * 4.5))

        # Pack Results
        return {
            "ticker": self.ticker,
            "current_price": current_price,
            "verdict": verdict,
            "why": why,
            "win_rate": best_wr,
            "best_params": best_params,
            "smart_money_slope": obv_slope,
            "smart_money_date": sm_date,
            "patterns": {
                "VCP": is_vcp,
                "Squeeze": is_squeeze,
                "Accumulation": strong_accumulation
            },
            "trade_plan": {
                "sl": stop_loss,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3
            },
            "technicals": {
                "rsi": self.df['RSI'].iloc[-1],
                "stoch_k": self.df['Stoch_K'].iloc[-1],
                "vwap": self.df['VWAP'].iloc[-1]
            }
        }

