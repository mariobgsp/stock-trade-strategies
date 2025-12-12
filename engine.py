import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
import warnings
from datetime import timedelta, datetime

# Suppress warnings for clean CLI output
warnings.filterwarnings('ignore')

class OJKCompliance:
    """Handles Indonesia Stock Exchange (IDX) specific rules."""
    
    @staticmethod
    def get_tick_size(price):
        if price < 200: return 1
        elif price < 500: return 2
        elif price < 2000: return 5
        elif price < 5000: return 10
        else: return 25

    @staticmethod
    def round_to_tick(price):
        tick = OJKCompliance.get_tick_size(price)
        return round(price / tick) * tick

class TechnicalAnalysis:
    """Manual implementation of indicators without pandas-ta."""
    
    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def sma(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def obv(close, volume):
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)

    @staticmethod
    def vwap(df):
        v = df['Volume'].values
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return df.assign(VWAP=(tp * v).cumsum() / v.cumsum())['VWAP']

class PatternRecognition:
    @staticmethod
    def identify_bounce_zones(df, order=5):
        """Uses scipy to find local minima/maxima."""
        # Find local minima
        ilocs_min = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
        ilocs_max = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
        
        supports = df['Low'].iloc[ilocs_min].tail(5).values
        resistances = df['High'].iloc[ilocs_max].tail(5).values
        return supports, resistances

    @staticmethod
    def detect_vcp(df, lookback=20):
        """Detects Volatility Contraction Pattern logic."""
        # Check if volatility (Standard Deviation) is decreasing over windows
        recent = df['Close'].tail(lookback)
        half_lookback = int(lookback/2)
        vol_1 = recent.iloc[-half_lookback:].std()
        vol_2 = recent.iloc[-lookback:-half_lookback].std()
        
        # Check for price tightening (High-Low range decreasing)
        range_recent = (df['High'] - df['Low']).tail(5).mean()
        range_prev = (df['High'] - df['Low']).iloc[-10:-5].mean()
        
        is_vcp = vol_1 < vol_2 and range_recent < range_prev
        return is_vcp

    @staticmethod
    def detect_ma_squeeze(df):
        """Checks if SMA 3, 5, 10, 20 are within 5% range."""
        last = df.iloc[-1]
        mas = [last['SMA_3'], last['SMA_5'], last['SMA_10'], last['SMA_20']]
        min_ma = min(mas)
        max_ma = max(mas)
        
        # Avoid division by zero
        if min_ma == 0: return False
        
        spread = (max_ma - min_ma) / min_ma
        return spread < 0.05

class SmartMoneyAnalyzer:
    @staticmethod
    def analyze_accumulation(df):
        """Uses sklearn LinearRegression to determine OBV slope."""
        # Look at last 20 days
        window = 20
        if len(df) < window: return "Neutral", 0, None
        
        subset = df.tail(window)
        X = np.arange(len(subset)).reshape(-1, 1)
        y = subset['OBV'].values.reshape(-1, 1)
        
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0][0]
        
        # Determine start date of trend
        # Simplistic approach: Finding when slope last changed sign or kept consistency
        start_date = subset.index[0]
        
        if slope > 0:
            status = "Accumulation"
        elif slope < 0:
            status = "Distribution"
        else:
            status = "Neutral"
            
        return status, slope, start_date

class TradingEngine:
    def __init__(self, ticker):
        self.ticker = ticker + ".JK" # Append Jakarta code
        self.df = None
        self.is_ipo = False
        self.risk_reward_ratio = 3.0
        self.min_win_rate = 0.60
        self.optimized_params = {}
        self.strategies = {}

    def fetch_data(self):
        # Fetch max to determine IPO status first
        try:
            full_df = yf.download(self.ticker, period="5y", progress=False)
            if full_df.empty:
                raise ValueError("No data found.")
                
            # IPO Logic
            history_days = (full_df.index[-1] - full_df.index[0]).days
            if history_days < 180: # Less than 6 months
                self.is_ipo = True
                self.df = full_df # Use all data
            else:
                self.is_ipo = False
                self.df = full_df # Use 5y data (handled as mature)
            
            # Basic cleanup
            self.df.dropna(subset=['Close', 'Volume'], inplace=True)
            if len(self.df) < 10: raise ValueError("Insufficient data points.")
            
        except Exception as e:
            return False, str(e)
        return True, "Data fetched successfully."

    def apply_indicators(self, rsi_p=14, stoch_k=14, ma_windows=[3, 5, 10, 20, 50, 200]):
        # Clean copy for iteration
        df = self.df.copy()
        
        # MAs
        for w in ma_windows:
            df[f'SMA_{w}'] = TechnicalAnalysis.sma(df['Close'], w)
            df[f'EMA_{w}'] = TechnicalAnalysis.ema(df['Close'], w)
            
        # Oscillators
        df['RSI'] = TechnicalAnalysis.rsi(df['Close'], rsi_p)
        df['Stoch_K'], df['Stoch_D'] = TechnicalAnalysis.stochastic(df['High'], df['Low'], df['Close'], stoch_k)
        
        # Volume
        df['OBV'] = TechnicalAnalysis.obv(df['Close'], df['Volume'])
        df['VWAP'] = TechnicalAnalysis.vwap(df)
        
        return df

    def backtest_strategy_logic(self, df, strategy_type, params):
        """
        Vectorized-style backtest for speed.
        Returns: Win Rate, Trade Log
        """
        df['Signal'] = 0
        close = df['Close']
        
        # Logic Implementations
        if strategy_type == "Breakout":
            # Price > SMA_20 AND Price > Resistance (Simulated by previous high)
            condition = (close > df['SMA_20']) & (close > close.shift(1).rolling(10).max())
            df.loc[condition, 'Signal'] = 1
            
        elif strategy_type == "Pullback":
            # Price > SMA_50 (Trend) but RSI < 40 (Pullback)
            condition = (close > df['SMA_50']) & (df['RSI'] < 40)
            df.loc[condition, 'Signal'] = 1
            
        elif strategy_type == "VCP_Breakout":
            # Volatility < threshold (simplified here as we pre-calced VCP flag) but vol > avg vol
            # Requires loop for proper VCP context usually, simplistic proxy here:
            # Consolidation (low std dev) followed by price jump
            std_dev = close.rolling(5).std()
            condition = (std_dev < std_dev.shift(5)) & (close > close.shift(1) * 1.02)
            df.loc[condition, 'Signal'] = 1

        elif strategy_type == "IPO_Momentum":
            # EMA 5 > EMA 10 > EMA 20
            condition = (df['EMA_5'] > df['EMA_10']) & (df['EMA_10'] > df['EMA_20'])
            df.loc[condition, 'Signal'] = 1

        # Simulate Trades
        trades = []
        in_trade = False
        entry_price = 0
        
        # Walk forward simulation
        signals = df[df['Signal'] == 1].index
        
        # Limit backtest to last 1 year (or max available) to be relevant
        test_start_idx = max(0, len(df) - 252)
        
        wins = 0
        total_trades = 0
        hit_counts = {'1R': 0, '2R': 0, '3R': 0}
        
        # Iterate only through signal days for speed
        relevant_indices = [i for i in range(test_start_idx, len(df)-1) if df['Signal'].iloc[i] == 1]
        
        for i in relevant_indices:
            entry_price = df['Close'].iloc[i]
            stop_loss = entry_price * 0.95 # Generic 5% SL for testing
            target_1r = entry_price + (entry_price - stop_loss)
            target_2r = entry_price + (entry_price - stop_loss) * 2
            target_3r = entry_price + (entry_price - stop_loss) * 3
            
            # Forward look 10 days
            future = df.iloc[i+1 : i+11]
            if future.empty: continue
            
            hit_tp = False
            hit_sl = False
            max_reached = 0
            
            for _, row in future.iterrows():
                if row['Low'] <= stop_loss:
                    hit_sl = True
                    break
                if row['High'] >= target_1r:
                    hit_counts['1R'] += 1
                    max_reached = 1
                if row['High'] >= target_2r:
                    hit_counts['2R'] += 1
                    max_reached = 2
                if row['High'] >= target_3r:
                    hit_counts['3R'] += 1
                    max_reached = 3
                    hit_tp = True
                    break # Max win
            
            if not hit_sl and max_reached >= 1:
                wins += 1
            elif hit_tp:
                wins += 1
            
            total_trades += 1
            
        win_rate = (wins / total_trades) if total_trades > 0 else 0
        
        probs = {
            '1R': (hit_counts['1R']/total_trades) if total_trades > 0 else 0,
            '2R': (hit_counts['2R']/total_trades) if total_trades > 0 else 0,
            '3R': (hit_counts['3R']/total_trades) if total_trades > 0 else 0,
        }
        
        return win_rate, probs

    def optimize(self):
        """
        Grid Search for best parameters. 
        If IPO, defaults to momentum settings.
        """
        if self.is_ipo:
            # IPO Mode: Skip grid search, force Momentum strategy
            df = self.apply_indicators()
            wr, probs = self.backtest_strategy_logic(df, "IPO_Momentum", {})
            return "IPO_Momentum", df, wr, probs
        
        # Mature Mode: Grid Search
        best_wr = 0
        best_strat = None
        best_df = None
        best_probs = {'1R': 0, '2R': 0, '3R': 0}
        
        # Test vars
        strategies = ["Breakout", "Pullback", "VCP_Breakout"]
        rsi_settings = [14, 9, 21]
        
        for strat in strategies:
            for r in rsi_settings:
                df = self.apply_indicators(rsi_p=r)
                wr, probs = self.backtest_strategy_logic(df, strat, {})
                
                # Layering Logic: If WR < 60%, try filtering (Simulated by checking trend)
                if wr < 0.60:
                    # Filter: Only trade if Price > SMA 200 (Mature stock logic)
                    if 'SMA_200' in df.columns:
                        # Re-run backtest on subset
                        trend_df = df[df['Close'] > df['SMA_200']].copy()
                        if len(trend_df) > 50:
                            wr_filtered, probs_filtered = self.backtest_strategy_logic(trend_df, strat, {})
                            if wr_filtered > wr:
                                wr = wr_filtered
                                probs = probs_filtered
                
                if wr > best_wr:
                    best_wr = wr
                    best_strat = strat
                    best_df = df
                    best_probs = probs
        
        return best_strat, best_df, best_wr, best_probs

    def analyze(self):
        """Main execution method called by CLI."""
        success, msg = self.fetch_data()
        if not success: return {"error": msg}
        
        strat_name, df, win_rate, probs = self.optimize()
        
        # Current Signal Generation
        last_row = df.iloc[-1]
        
        # Current Price vs VWAP
        vwap_diff = ((last_row['Close'] - last_row['VWAP']) / last_row['VWAP']) * 100
        
        # Determine Current Action
        # Re-eval current condition based on best strategy
        action = "NO TRADE"
        
        # Simple signal check based on best strategy logic for *today*
        # (Re-implementing logic specifically for the last row check)
        trigger = False
        if strat_name == "Breakout":
            trigger = (last_row['Close'] > last_row['SMA_20'])
        elif strat_name == "Pullback":
            trigger = (last_row['Close'] > last_row['SMA_50']) and (last_row['RSI'] < 45) # Relaxed slightly for current signal
        elif strat_name == "VCP_Breakout":
             trigger = PatternRecognition.detect_vcp(df)
        elif strat_name == "IPO_Momentum":
            trigger = (last_row['EMA_5'] > last_row['EMA_10'])
            
        if trigger and win_rate >= self.min_win_rate:
            action = "BUY"
        elif trigger and win_rate < self.min_win_rate:
            action = "WAIT (High Risk)"
        
        # Calculate Targets
        current_price = last_row['Close']
        supports, resistances = PatternRecognition.identify_bounce_zones(df)
        
        # Pivot Calculation
        pp = (last_row['High'] + last_row['Low'] + last_row['Close']) / 3
        
        # Stop Loss: Recent Swing Low or ATR based
        swing_low = supports[-1] if len(supports) > 0 else current_price * 0.95
        stop_loss_price = OJKCompliance.round_to_tick(swing_low)
        
        # Ensure SL is below current price, else default to 5%
        if stop_loss_price >= current_price:
            stop_loss_price = OJKCompliance.round_to_tick(current_price * 0.95)
            
        risk = current_price - stop_loss_price
        tp1 = OJKCompliance.round_to_tick(current_price + risk)
        tp2 = OJKCompliance.round_to_tick(current_price + (risk * 2))
        tp3 = OJKCompliance.round_to_tick(current_price + (risk * 3))
        
        # Smart Money
        accum_status, obv_slope, trend_start = SmartMoneyAnalyzer.analyze_accumulation(df)
        
        # Format Start Date
        start_date_str = trend_start.strftime('%Y-%m-%d') if trend_start else "N/A"

        return {
            "verdict": action,
            "strategy": strat_name,
            "win_rate": win_rate,
            "probs": probs,
            "current_price": current_price,
            "entry": current_price,
            "stop_loss": stop_loss_price,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "vwap_diff": vwap_diff,
            "is_ipo": self.is_ipo,
            "accum_status": accum_status,
            "obv_slope": obv_slope,
            "accum_start": start_date_str,
            "rsi": last_row['RSI'],
            "ma_squeeze": PatternRecognition.detect_ma_squeeze(df),
            "vcp_detected": PatternRecognition.detect_vcp(df),
            "indicators": {
                "SMA_20": last_row['SMA_20'] if 'SMA_20' in last_row else 0,
                "VWAP": last_row['VWAP']
            }
        }

