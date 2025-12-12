import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# --- 1. OJK/IDX Microstructure Compliance ---
class IDXRules:
    """
    Handles Indonesia Stock Exchange specific microstructure rules.
    Ref: IDX Regulation No. II-A (Fraksi Harga)
    """
    @staticmethod
    def get_tick_size(price: float) -> int:
        if price < 200:
            return 1
        elif 200 <= price < 500:
            return 2
        elif 500 <= price < 2000:
            return 5
        elif 2000 <= price < 5000:
            return 10
        else:
            return 25

    @staticmethod
    def round_to_tick(price: float) -> int:
        """Rounds a price to the nearest valid IDX tick."""
        tick = IDXRules.get_tick_size(price)
        return int(round(price / tick) * tick)

    @staticmethod
    def calculate_targets(entry_price: float, risk_reward_ratio: float = 3.0) -> Tuple[int, int]:
        """
        Calculates Stop Loss (1 tick below support roughly) and Target
        based on rigid Risk:Reward.
        Here we define risk as ~3-5% for swing or based on recent low.
        For this engine, we use a dynamic risk based on ATR or Pivot, 
        but enforced to integer ticks.
        """
        # Default fallback risk if no technical stop provided: 4%
        risk_per_share = entry_price * 0.04 
        stop_loss = entry_price - risk_per_share
        
        # Enforce Tick Rules
        stop_loss = IDXRules.round_to_tick(stop_loss)
        risk_actual = entry_price - stop_loss
        
        if risk_actual <= 0: # Sanity check for low volatility
            tick = IDXRules.get_tick_size(entry_price)
            stop_loss = entry_price - (2 * tick) # Min 2 ticks risk
            risk_actual = entry_price - stop_loss

        target_price = entry_price + (risk_actual * risk_reward_ratio)
        target_price = IDXRules.round_to_tick(target_price)
        
        return stop_loss, target_price

# --- 2. Technical Analysis Engine (Manual Calculations) ---
class TAEngine:
    """
    Manual implementation of technical indicators using numpy/pandas.
    No external lib usage (pandas-ta forbidden).
    """
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def stoch_osc(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return pd.DataFrame({'k': k, 'd': d})

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = pd.Series(index=close.index, dtype='float64')
        obv.iloc[0] = volume.iloc[0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        v = df['Volume'].values
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return df.assign(vwap=(tp * v).cumsum() / v.cumsum())['vwap']

    @staticmethod
    def pivot_points(df: pd.DataFrame) -> Dict[str, float]:
        """Standard Pivot Points based on previous day."""
        last = df.iloc[-2] # Previous completed day
        P = (last['High'] + last['Low'] + last['Close']) / 3
        S1 = (P * 2) - last['High']
        R1 = (P * 2) - last['Low']
        return {'P': P, 'S1': S1, 'R1': R1}

    @staticmethod
    def get_fib_levels(df: pd.DataFrame) -> Dict[str, float]:
        """Calculates Fib Retracement from last significant swing high to recent low."""
        # Simple swing detection: Max of last 60 days to Min of last 10 days
        lookback_high = 60
        lookback_low = 20
        
        recent_high = df['High'].tail(lookback_high).max()
        recent_low = df['Low'].tail(lookback_low).min()
        
        diff = recent_high - recent_low
        return {
            '0.0': recent_low,
            '0.382': recent_low + (diff * 0.382),
            '0.5': recent_low + (diff * 0.5),
            '0.618': recent_low + (diff * 0.618),
            '1.0': recent_high
        }

    @staticmethod
    def check_ma_squeeze(df: pd.DataFrame) -> bool:
        """Detects if SMA 3, 5, 10, 20 are within 5% range."""
        last = df.iloc[-1]
        mas = [last['SMA_3'], last['SMA_5'], last['SMA_10'], last['SMA_20']]
        min_ma = min(mas)
        max_ma = max(mas)
        
        if min_ma == 0: return False
        spread = (max_ma - min_ma) / min_ma
        return spread < 0.05

    @staticmethod
    def check_vcp(df: pd.DataFrame) -> bool:
        """
        Simplified VCP detection: 
        Detects decreasing volatility (Standard Deviation of High-Low) over 3 windows.
        """
        w1 = df['High'].sub(df['Low']).rolling(20).std().iloc[-1]
        w2 = df['High'].sub(df['Low']).rolling(20).std().iloc[-20]
        w3 = df['High'].sub(df['Low']).rolling(20).std().iloc[-40]
        
        # Volatility should be decreasing
        return w1 < w2 < w3

# --- 3. Smart Money / Bandarmology Module ---
class Bandarmology:
    @staticmethod
    def analyze_flow(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects Stealth Accumulation.
        Logic: Price is consolidating (slope ~ 0) BUT OBV is rising (slope > 0).
        """
        window = 14
        if len(df) < window:
            return {"status": "INSUFFICIENT_DATA", "phase": "N/A", "strength": "N/A"}

        recent = df.tail(window)
        
        # Calculate Linear Regression Slopes
        y_price = recent['Close'].values
        y_obv = recent['OBV'].values
        x = np.arange(len(y_price))
        
        slope_price = np.polyfit(x, y_price, 1)[0]
        slope_obv = np.polyfit(x, y_obv, 1)[0]
        
        # Normalize slope to percentage of mean to make it comparable
        price_slope_norm = slope_price / recent['Close'].mean()
        
        # Thresholds
        is_price_flat = abs(price_slope_norm) < 0.002 # < 0.2% daily trend change
        is_obv_rising = slope_obv > 0
        
        status = "NEUTRAL"
        strength = "Weak"
        
        if is_price_flat and is_obv_rising:
            status = "STEALTH ACCUMULATION"
            strength = "Strong" if slope_obv > (recent['OBV'].std() * 0.5) else "Moderate"
        elif slope_price < -0.005 and slope_obv > 0:
            status = "DIVERGENCE (BULLISH)"
            strength = "Moderate"
        elif slope_price > 0.005 and slope_obv < 0:
            status = "DISTRIBUTION"
            strength = "Strong"
        elif slope_price > 0.005 and slope_obv > 0:
            status = "MARKUP"
            strength = "Strong"
            
        return {
            "status": status,
            "phase_start": df.index[-window].strftime('%Y-%m-%d'),
            "strength": strength,
            "vwap_gap": (df.iloc[-1]['Close'] - df.iloc[-1]['vwap']) / df.iloc[-1]['vwap']
        }

# --- 4. Backtesting & Optimization Engine ---
@dataclass
class BacktestResult:
    win_rate: float
    total_trades: int
    avg_return: float
    params: Dict[str, int]
    is_valid: bool

class StrategyOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def backtest_strategy(self, rsi_period: int, ma_period: int) -> BacktestResult:
        """
        Simulates a simple Buy on Dip/Breakout strategy using dynamic params.
        Entry: Close > MA AND RSI < 40 (Dip) OR Close > MA * 1.02 (Breakout)
        Exit: Fixed 1:3 RR based on ATR risk.
        """
        balance = 1000000
        wins = 0
        losses = 0
        trades = 0
        
        # Pre-calc indicators for speed
        temp_df = self.df.copy()
        temp_df['TEST_MA'] = TAEngine.sma(temp_df['Close'], ma_period)
        temp_df['TEST_RSI'] = TAEngine.rsi(temp_df['Close'], rsi_period)
        
        i = ma_period + 5
        while i < len(temp_df) - 5: # -5 to allow trade to play out slightly
            row = temp_df.iloc[i]
            
            # Entry Condition (Simplified Hybrid)
            # 1. Trend is Up (Close > MA)
            # 2. RSI is not overbought (< 70)
            entry_signal = (row['Close'] > row['TEST_MA']) and (row['TEST_RSI'] < 70) and (row['TEST_RSI'] > 40)
            
            if entry_signal:
                entry_price = row['Close']
                # Determine Stop/Target using IDX Rules logic
                sl, tp = IDXRules.calculate_targets(entry_price, 3.0)
                
                # Check future candles for outcome
                outcome = 0 # 0 pending, 1 win, -1 loss
                for future_idx in range(i+1, min(i+30, len(temp_df))):
                    f_row = temp_df.iloc[future_idx]
                    if f_row['Low'] <= sl:
                        outcome = -1
                        break
                    if f_row['High'] >= tp:
                        outcome = 1
                        break
                
                if outcome == 1:
                    wins += 1
                elif outcome == -1:
                    losses += 1
                
                # Skip forward to avoid overlapping trades
                if outcome != 0:
                    trades += 1
                    i = future_idx 
            
            i += 1
            
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        return BacktestResult(
            win_rate=win_rate,
            total_trades=total,
            avg_return=0.0, # Simplified
            params={'rsi': rsi_period, 'ma': ma_period},
            is_valid=win_rate > 65 and total > 5 # Min 5 trades to be stat significant
        )

    def optimize(self) -> BacktestResult:
        """Grid Search for best params."""
        rsi_params = [9, 14, 21, 25]
        ma_params = [20, 50, 100, 200]
        
        best_result = BacktestResult(0, 0, 0, {}, False)
        
        for r in rsi_params:
            for m in ma_params:
                res = self.backtest_strategy(r, m)
                if res.is_valid and res.win_rate > best_result.win_rate:
                    best_result = res
                    
        return best_result

# --- 5. Main Data Fetcher ---
def fetch_data(ticker: str) -> pd.DataFrame:
    # Append .JK if missing
    if not ticker.endswith('.JK'):
        ticker = f"{ticker}.JK"
    
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period="3y", interval="1d", progress=False, auto_adjust=False)
    
    if df.empty:
        raise ValueError("No data found. Check ticker symbol.")
        
    # Flat column index if multi-index (common in new yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate Base Indicators needed for all modules
    df['SMA_3'] = TAEngine.sma(df['Close'], 3)
    df['SMA_5'] = TAEngine.sma(df['Close'], 5)
    df['SMA_10'] = TAEngine.sma(df['Close'], 10)
    df['SMA_20'] = TAEngine.sma(df['Close'], 20)
    df['OBV'] = TAEngine.obv(df['Close'], df['Volume'])
    df['vwap'] = TAEngine.vwap(df)
    
    # Clean NaN
    df.dropna(inplace=True)
    return df