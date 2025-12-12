import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime
import warnings

# Suppress minor pandas warnings for cleaner CLI output
warnings.filterwarnings('ignore')

class OJKCompliance:
    """
    Handles Indonesia Stock Exchange (IDX) specific rules, 
    specifically the Fraction (Tick Size) rules.
    """
    @staticmethod
    def get_tick_size(price):
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
    def round_to_tick(price):
        """Rounds a raw price to the nearest valid IDX price step."""
        if price <= 0: return 0
        tick = OJKCompliance.get_tick_size(price)
        return round(price / tick) * tick

class TechnicalAnalysis:
    """
    Manual implementation of technical indicators without pandas-ta.
    """
    @staticmethod
    def calculate_indicators(df, params):
        # 1. Moving Averages
        df['SMA_3'] = df['Close'].rolling(window=3).mean()
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # 2. RSI (Manual)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. Stochastic Oscillator (Manual)
        low_min = df['Low'].rolling(window=params['stoch_k']).min()
        high_max = df['High'].rolling(window=params['stoch_k']).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=params['stoch_d']).mean()
        
        # 4. VWAP
        cum_vol = df['Volume'].cumsum()
        cum_vol_price = (df['Close'] * df['Volume']).cumsum()
        df['VWAP'] = cum_vol_price / cum_vol
        
        # 5. On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df

    @staticmethod
    def detect_ma_squeeze(df):
        """
        Detects 'Superclose' or Squeeze: SMA 3, 5, 10, 20 compression < 5% range.
        """
        if len(df) < 20: return False
        
        last_row = df.iloc[-1]
        mas = [last_row['SMA_3'], last_row['SMA_5'], last_row['SMA_10'], last_row['SMA_20']]
        
        # Calculate range of MAs
        min_ma = min(mas)
        max_ma = max(mas)
        
        # Calculate percentage spread relative to close
        spread = (max_ma - min_ma) / last_row['Close']
        
        return spread < 0.05 # True if MAs are within 5% of each other

    @staticmethod
    def identify_smart_money(df):
        """
        Uses Sklearn LinearRegression to find OBV slope and Start Date.
        """
        if len(df) < 30: return 0, "Insufficient Data"
        
        # Look back 20 days for accumulation
        lookback = 20
        subset = df.iloc[-lookback:].copy()
        
        # Prepare data for sklearn
        X = np.array(range(len(subset))).reshape(-1, 1)
        y_obv = subset['OBV'].values.reshape(-1, 1)
        y_price = subset['Close'].values.reshape(-1, 1)
        
        # Calculate Slopes
        reg_obv = LinearRegression().fit(X, y_obv)
        reg_price = LinearRegression().fit(X, y_price)
        
        obv_slope = reg_obv.coef_[0][0]
        price_slope = reg_price.coef_[0][0]
        
        # Logic: Accumulation if Price is flat/down but OBV is UP
        is_accumulating = obv_slope > 0 and price_slope <= 0
        
        # Find start date: The first day in the window where OBV exceeded its MA
        # (Simplified for CLI: Return start of the lookback window)
        start_date = subset.index[0].strftime('%Y-%m-%d')
        
        return obv_slope, start_date, is_accumulating

    @staticmethod
    def find_scipy_bounce_levels(df):
        """
        Uses Scipy ArgRelExtrema to find historical bounce zones.
        """
        # Find local minima (support candidates)
        # order=10 means it must be the lowest point in a 20-day window
        local_min_indices = argrelextrema(df['Low'].values, np.less, order=10)[0]
        if len(local_min_indices) == 0:
            return df['Low'].min()
        
        # Return the most recent significant low
        return df.iloc[local_min_indices[-1]]['Low']

class StrategyEngine:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = None
        
    def fetch_data(self):
        """Fetches 3 years of data using yfinance."""
        ticker_obj = yf.Ticker(self.ticker)
        self.df = ticker_obj.history(period="3y")
        if self.df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        
    def backtest_strategy(self, df, logic_func, params):
        """
        Core Backtester.
        Logic: Fixed 1:3 RR based on ATR or Swing Low.
        """
        balance = 10000000 # Dummy balance
        trades = []
        in_trade = False
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        # Iterate through history (skipping first 200 for MA calc)
        # Note: Optimization loop calls this often, so we keep logic tight.
        for i in range(200, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            
            if not in_trade:
                signal, sl_price = logic_func(df.iloc[:i+1], params)
                if signal:
                    entry_price = current_bar['Close']
                    stop_loss = OJKCompliance.round_to_tick(sl_price)
                    risk = entry_price - stop_loss
                    
                    if risk <= 0: continue # Invalid trade math
                    
                    # Target 3R
                    take_profit = OJKCompliance.round_to_tick(entry_price + (risk * 3))
                    
                    in_trade = True
            else:
                # Check exit
                low = current_bar['Low']
                high = current_bar['High']
                
                if low <= stop_loss:
                    trades.append(0) # Loss
                    in_trade = False
                elif high >= take_profit:
                    trades.append(1) # Win
                    in_trade = False
                    
        win_rate = (sum(trades) / len(trades)) * 100 if len(trades) > 0 else 0
        return win_rate, len(trades)

    # --- STRATEGIES ---
    
    def strat_buy_on_dip(self, subset, params):
        """Logic: RSI oversold + Bounce from Scipy Support"""
        curr = subset.iloc[-1]
        
        # Condition 1: RSI Oversold
        if curr['RSI'] > params['rsi_lower']: return False, 0
        
        # Condition 2: Price near Scipy Support
        scipy_support = TechnicalAnalysis.find_scipy_bounce_levels(subset)
        if curr['Close'] < scipy_support * 1.05: # Within 5% of support
             return True, scipy_support * 0.98 # SL below support
        
        return False, 0

    def strat_breakout(self, subset, params):
        """Logic: Price breaks SMA20 + Volume Surge + OBV Slope Positive"""
        curr = subset.iloc[-1]
        prev = subset.iloc[-2]
        
        # Condition 1: Cross over SMA 20
        if not (prev['Close'] < prev['SMA_20'] and curr['Close'] > curr['SMA_20']):
            return False, 0
            
        # Condition 2: Volume > SMA20 Volume (approximated)
        avg_vol = subset['Volume'].tail(20).mean()
        if curr['Volume'] < avg_vol * 1.2:
            return False, 0
            
        return True, curr['Low'] # SL at candle low

    def strat_super_squeeze(self, subset, params):
        """Logic: MA Squeeze (3,5,10,20) detected"""
        is_squeezing = TechnicalAnalysis.detect_ma_squeeze(subset)
        if is_squeezing:
            # If squeezing, we buy anticipating expansion. 
            # SL is Lowest Low of last 5 days
            sl = subset['Low'].tail(5).min()
            return True, sl
        return False, 0

    def run_optimization(self):
        """
        Grid Search to find > 65% Win Rate.
        """
        self.fetch_data()
        
        # Define Grid
        rsi_params = [{'rsi_period': 14, 'rsi_lower': x, 'stoch_k': 14, 'stoch_d': 3} for x in [25, 30, 35, 40]]
        
        best_win_rate = 0
        best_setup = None
        
        # Add indicators initially with default params to dataframe to save time, 
        # but re-calc inside loop if params change significantly (simplified here)
        self.df = TechnicalAnalysis.calculate_indicators(self.df, {'rsi_period': 14, 'stoch_k': 14, 'stoch_d': 3})

        strategies = [
            ('Buy On Dip', self.strat_buy_on_dip),
            ('Breakout', self.strat_breakout),
            ('MA Squeeze', self.strat_super_squeeze)
        ]

        # Optimization Loop
        for strat_name, strat_func in strategies:
            for p in rsi_params:
                # Run Backtest
                wr, trade_count = self.backtest_strategy(self.df, strat_func, p)
                
                # Strict Filtering: Require > 65% WR and at least 5 trades
                if wr > 65 and trade_count > 5:
                    if wr > best_win_rate:
                        best_win_rate = wr
                        best_setup = {
                            'strategy': strat_name,
                            'win_rate': wr,
                            'params': p,
                            'trade_count': trade_count
                        }

        # Analyze Current Market State for Final Output
        current_data = self.df
        
        obv_slope, smart_money_date, is_accum = TechnicalAnalysis.identify_smart_money(current_data)
        scipy_support = TechnicalAnalysis.find_scipy_bounce_levels(current_data)
        
        # Check current signal based on BEST strategy
        final_verdict = "WAIT / NO TRADE"
        trade_plan = {}
        logic_text = "No strategy met the strict 65% Win Rate criteria on historical data."
        
        if best_setup:
            signal, sl_raw = strategies[[s[0] for s in strategies].index(best_setup['strategy'])][1](current_data, best_setup['params'])
            
            if signal:
                final_verdict = "BUY"
                curr_price = current_data.iloc[-1]['Close']
                sl = OJKCompliance.round_to_tick(sl_raw)
                risk = curr_price - sl
                tp1 = OJKCompliance.round_to_tick(curr_price + risk)
                tp2 = OJKCompliance.round_to_tick(curr_price + (risk * 2))
                tp3 = OJKCompliance.round_to_tick(curr_price + (risk * 3))
                
                trade_plan = {
                    'entry': curr_price,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp3': tp3
                }
                
                logic_text = f"Triggered {best_setup['strategy']} setup. Historical WR: {best_setup['win_rate']:.1f}%."
                if is_accum:
                    logic_text += " CONFIRMED by Smart Money Accumulation (OBV Divergence)."
            else:
                logic_text = f"Best strategy is {best_setup['strategy']} ({best_setup['win_rate']:.1f}% WR), but no entry signal today."

        # Technical Deep Dive Data
        last_row = self.df.iloc[-1]
        
        return {
            'verdict': final_verdict,
            'plan': trade_plan,
            'logic': logic_text,
            'stats': best_setup,
            'deep_dive': {
                'rsi': last_row['RSI'],
                'stoch_k': last_row['Stoch_K'],
                'ma_squeeze': TechnicalAnalysis.detect_ma_squeeze(self.df),
                'obv_slope': obv_slope,
                'smart_money_start': smart_money_date,
                'support_scipy': scipy_support,
                'price': last_row['Close']
            }
        }

