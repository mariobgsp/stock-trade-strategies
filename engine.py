import yfinance as yf
import pandas as pd
import numpy as np
import scipy.signal
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class IDXConfig:
    """Configuration strictly for Indonesia Stock Exchange rules."""
    @staticmethod
    def get_tick_size(price):
        if price < 200: return 1
        elif 200 <= price < 500: return 2
        elif 500 <= price < 2000: return 5
        elif 2000 <= price < 5000: return 10
        else: return 25

    @staticmethod
    def adjust_price(price, is_stop_loss=False):
        """Rounds price to nearest valid IDX tick."""
        if pd.isna(price): return price
        price = float(price)
        tick = IDXConfig.get_tick_size(price)
        
        # Logic: If Stop Loss, floor to lower tick to ensure trigger. 
        # If Target, ceil or floor depending on strategy, but standard rounding applied here.
        adjusted = round(price / tick) * tick
        return int(adjusted)

class TechnicalAnalysis:
    """Manual implementation of indicators without pandas-ta."""
    
    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_sma(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_stoch(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def calculate_obv(close, volume):
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_vwap(df, window=20):
        """Rolling VWAP as a proxy for short-term institutional average."""
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return (tp * v).rolling(window=window).sum() / v.rolling(window=window).sum()

class QuantEngine:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        if not self.ticker.endswith('.JK'):
            self.ticker += '.JK'
        self.df = None
        self.is_ipo = False
        self.analysis_results = {}

    def fetch_data(self):
        # Attempt 5 years first
        try:
            self.df = yf.download(self.ticker, period="5y", progress=False, auto_adjust=True)
            if self.df.empty:
                raise ValueError("No data found")
        except:
            # Fallback for IPOs or connection issues
            self.df = yf.download(self.ticker, period="max", progress=False, auto_adjust=True)
        
        if len(self.df) < 5:
            raise ValueError(f"Insufficient data for {self.ticker}")

        # Flatten multi-index columns if present (yfinance update fix)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)

        # Clean data
        self.df = self.df.dropna()
        
        # IPO Detection (Less than 1 year of trading days approx 250)
        if len(self.df) < 250:
            self.is_ipo = True
        
        return self.df

    def analyze_smart_money(self):
        """Analyze accumulation/distribution using Linear Regression on OBV."""
        # Calculate OBV
        self.df['OBV'] = TechnicalAnalysis.calculate_obv(self.df['Close'], self.df['Volume'])
        self.df['VWAP'] = TechnicalAnalysis.calculate_vwap(self.df)
        
        # Analyze last 20 days slope
        lookback = min(20, len(self.df))
        recent_data = self.df.iloc[-lookback:].copy()
        
        X = np.arange(len(recent_data)).reshape(-1, 1)
        
        # OBV Slope
        y_obv = recent_data['OBV'].values.reshape(-1, 1)
        reg_obv = LinearRegression().fit(X, y_obv)
        obv_slope = reg_obv.coef_[0][0]
        
        # Price Slope
        y_price = recent_data['Close'].values.reshape(-1, 1)
        reg_price = LinearRegression().fit(X, y_price)
        price_slope = reg_price.coef_[0][0]

        status = "Neutral"
        start_date = recent_data.index[0].strftime('%Y-%m-%d')
        
        # Logic: Price Flat/Down but OBV Up = Accumulation
        if obv_slope > 0 and price_slope <= 0:
            status = "Accumulation (Strong)"
        elif obv_slope > 0 and price_slope > 0:
            status = "Markup Phase"
        elif obv_slope < 0 and price_slope >= 0:
            status = "Distribution (Warning)"
        elif obv_slope < 0 and price_slope < 0:
            status = "Markdown Phase"

        current_price = self.df['Close'].iloc[-1]
        vwap_val = self.df['VWAP'].iloc[-1]
        vwap_diff = ((current_price - vwap_val) / vwap_val) * 100
        
        return {
            'status': status,
            'strength': obv_slope,
            'start_date': start_date,
            'vwap_diff': vwap_diff,
            'vwap_price': vwap_val
        }

    def detect_patterns(self):
        """Detect VCP, Squeeze, and Bounce Zones using Scipy."""
        close = self.df['Close']
        
        # 1. MA Squeeze (Superclose)
        ma3 = TechnicalAnalysis.calculate_sma(close, 3).iloc[-1]
        ma5 = TechnicalAnalysis.calculate_sma(close, 5).iloc[-1]
        ma10 = TechnicalAnalysis.calculate_sma(close, 10).iloc[-1]
        ma20 = TechnicalAnalysis.calculate_sma(close, 20).iloc[-1]
        
        mas = [ma3, ma5, ma10, ma20]
        squeeze_range = (max(mas) - min(mas)) / min(mas)
        is_squeeze = squeeze_range < 0.05 # 5% compression
        
        # 2. Bounce Zones (Scipy argrelextrema)
        # Find local minima over last 50 periods
        n = 5 # Order of extrema
        local_min_idxs = scipy.signal.argrelextrema(close.values, np.less_equal, order=n)[0]
        recent_bounces = close.iloc[local_min_idxs].tail(3).values.tolist()
        
        # 3. VCP (Volatility Contraction)
        # Simplified VCP: Std Dev of price reduces over 3 consecutive windows
        vol_win_1 = close.iloc[-10:].std()
        vol_win_2 = close.iloc[-20:-10].std()
        vol_win_3 = close.iloc[-30:-20].std()
        
        is_vcp = (vol_win_1 < vol_win_2) and (vol_win_2 < vol_win_3)
        
        return {
            'is_squeeze': is_squeeze,
            'squeeze_pct': squeeze_range * 100,
            'bounce_levels': recent_bounces,
            'is_vcp': is_vcp
        }

    def optimizer_loop(self):
        """Grid Search for indicators. Simplified for performance."""
        # If IPO, skip optimization, return fast defaults
        if self.is_ipo:
            return {'ma_fast': 5, 'ma_slow': 10, 'rsi_period': 9, 'rsi_thresh': 30}

        best_score = -np.inf
        best_params = {'ma_fast': 10, 'ma_slow': 20, 'rsi_period': 14, 'rsi_thresh': 30}
        
        # Grid Search space
        ma_fasts = [5, 10, 20]
        ma_slows = [20, 50]
        rsis = [9, 14, 21]
        
        # Simple backtest on last year of data to find best fit
        lookback_slice = self.df.iloc[-250:] if len(self.df) > 250 else self.df
        
        for fast in ma_fasts:
            for slow in ma_slows:
                if fast >= slow: continue
                for rsi_p in rsis:
                    # vectorized approximate backtest
                    c = lookback_slice['Close']
                    ma_f_s = TechnicalAnalysis.calculate_sma(c, fast)
                    ma_s_s = TechnicalAnalysis.calculate_sma(c, slow)
                    rsi_s = TechnicalAnalysis.calculate_rsi(c, rsi_p)
                    
                    # Logic: MA Crossover + RSI < 50 (Trend Following Dip)
                    # Shift signals to avoid lookahead bias
                    signal = (ma_f_s > ma_s_s) & (rsi_s < 50)
                    ret = c.pct_change().shift(-1) * signal
                    
                    score = ret.sum() # Total return as score
                    if score > best_score:
                        best_score = score
                        best_params = {'ma_fast': fast, 'ma_slow': slow, 'rsi_period': rsi_p, 'rsi_thresh': 30}
                        
        return best_params

    def run_strategy(self):
        """Core Strategy Execution & Backtesting."""
        if self.df is None: self.fetch_data()
        
        # 1. Get Optimized Params
        params = self.optimizer_loop()
        
        # 2. Apply Indicators
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        
        self.df['MA_Fast'] = TechnicalAnalysis.calculate_sma(close, params['ma_fast'])
        self.df['MA_Slow'] = TechnicalAnalysis.calculate_sma(close, params['ma_slow'])
        self.df['RSI'] = TechnicalAnalysis.calculate_rsi(close, params['rsi_period'])
        k, d = TechnicalAnalysis.calculate_stoch(high, low, close)
        self.df['Stoch_K'] = k
        
        # 3. Analyze Current State
        patterns = self.detect_patterns()
        smart_money = self.analyze_smart_money()
        
        current_price = close.iloc[-1]
        
        # 4. Generate Signal
        signal_score = 0
        reasons = []
        
        # Logic A: MA Trend
        if current_price > self.df['MA_Fast'].iloc[-1] > self.df['MA_Slow'].iloc[-1]:
            signal_score += 1
            reasons.append("Uptrend (Price > MA_Fast > MA_Slow)")
        
        # Logic B: Pattern
        if patterns['is_vcp']:
            signal_score += 2
            reasons.append("VCP Pattern Detected (Volatility Contraction)")
        if patterns['is_squeeze']:
            signal_score += 2
            reasons.append("Superclose Squeeze (Breakout Imminent)")
            
        # Logic C: Smart Money
        if "Accumulation" in smart_money['status'] or "Markup" in smart_money['status']:
            signal_score += 1
            reasons.append(f"Smart Money: {smart_money['status']}")
            
        # Logic D: Indicators (Buy on Dip)
        if self.df['RSI'].iloc[-1] < 45 and current_price > self.df['MA_Slow'].iloc[-1]:
             signal_score += 1
             reasons.append(f"RSI Oversold in Uptrend ({self.df['RSI'].iloc[-1]:.1f})")

        # 5. Formulate Verdict
        verdict = "NO TRADE"
        
        # IPO Special Case
        if self.is_ipo:
            # Simple momentum for IPO
            if current_price > TechnicalAnalysis.calculate_ema(close, 5).iloc[-1]:
                verdict = "BUY"
                reasons.append("IPO Momentum Validated (Price > EMA5)")
        else:
            # Mature Stock Logic (Needs score >= 3 for strong buy)
            if signal_score >= 3:
                verdict = "BUY"
            elif signal_score == 2:
                verdict = "WATCHLIST"

        # 6. Risk Calculation (OJK Compliant)
        # Stop Loss: Recent Low or 5% below entry
        support_level = min(patterns['bounce_levels']) if patterns['bounce_levels'] else current_price * 0.95
        # Ensure SL is not too far (max 8% risk)
        if (current_price - support_level)/current_price > 0.08:
            support_level = current_price * 0.92
            
        sl_price = IDXConfig.adjust_price(support_level, is_stop_loss=True)
        risk = current_price - sl_price
        
        # Risk:Reward 1:3
        tp1 = IDXConfig.adjust_price(current_price + risk)
        tp2 = IDXConfig.adjust_price(current_price + (risk * 2))
        tp3 = IDXConfig.adjust_price(current_price + (risk * 3))
        
        # 7. Probability (Monte Carlo Simulation Lite)
        # Calculate volatility-based probability of hitting targets
        daily_vol = close.pct_change().std()
        days_to_expiry = 20 # Swing horizon
        drift = close.pct_change().mean()
        
        # Probability formula (simplified Brownian motion probability)
        # Z-score distance to target
        import scipy.stats as stats
        
        def calc_prob(target):
            if target <= current_price: return 0
            log_ret = np.log(target / current_price)
            # Adjust drift for conservative estimate (assume 0 drift for safety)
            prob = 1 - stats.norm.cdf(log_ret / (daily_vol * np.sqrt(days_to_expiry)))
            return prob * 100

        prob_1r = calc_prob(tp1)
        prob_2r = calc_prob(tp2)
        prob_3r = calc_prob(tp3)
        
        # Win Rate Mandate Check
        # If Prob 1R < 50%, force NO TRADE (Safety Override)
        if prob_1r < 50 and verdict == "BUY":
            verdict = "NO TRADE"
            reasons.append("Safety Override: Statistical Win Rate < 50%")
            
        return {
            'ticker': self.ticker,
            'current_price': current_price,
            'verdict': verdict,
            'plan': {
                'entry': current_price,
                'sl': sl_price,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3
            },
            'probabilities': {
                '1R': prob_1r,
                '2R': prob_2r,
                '3R': prob_3r
            },
            'reasons': reasons,
            'smart_money': smart_money,
            'params': params,
            'patterns': patterns,
            'is_ipo': self.is_ipo
        }

