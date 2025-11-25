import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from datetime import datetime, timedelta

# ==========================================
# 1. DEFAULT CONFIGURATION
# ==========================================
DEFAULT_CONFIG = {
    "BACKTEST_PERIOD": "2y",
    "MAX_HOLD_DAYS": 20,
    "FIB_LOOKBACK_DAYS": 120,
    "RSI_PERIOD": 14,
    "RSI_LOWER": 30,
    "ATR_PERIOD": 14,
    # Base Multipliers (Will be adjusted for Short/Swing)
    "SL_MULTIPLIER": 2.0,
    "TP_MULTIPLIER": 3.0,
    "CMF_PERIOD": 20,
    "MFI_PERIOD": 14,
    "VOL_MA_PERIOD": 20
}

MA_TEST_PAIRS = [(5, 20), (20, 50), (50, 200)] 
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20
OBV_LOOKBACK_DAYS = 5
TREND_EMA_DEFAULT = 200

class StockAnalyzer:
    def __init__(self, ticker, user_config=None):
        self.ticker = self._format_ticker(ticker)
        self.market_ticker = "^JKSE" # IHSG Index
        self.df = None
        self.market_df = None
        self.info = {}
        self.news_analysis = {"sentiment": "Neutral", "score": 0, "headlines": []}
        self.active_trend_col = f"EMA_{TREND_EMA_DEFAULT}"
        self.data_len = 0
        self.config = DEFAULT_CONFIG.copy()
        if user_config:
            self.config.update(user_config)

    def _format_ticker(self, ticker):
        ticker = ticker.upper().strip()
        if not ticker.endswith(".JK") and not ticker.startswith("^"):
            ticker += ".JK"
        return ticker

    def fetch_data(self):
        try:
            period = self.config["BACKTEST_PERIOD"]
            # 1. Fetch Stock Data
            self.df = yf.download(self.ticker, period=period, progress=False, auto_adjust=True)
            
            # 2. Fetch Market Data (IHSG) for Relative Strength
            try:
                self.market_df = yf.download(self.market_ticker, period=period, progress=False, auto_adjust=True)
                if isinstance(self.market_df.columns, pd.MultiIndex):
                    self.market_df.columns = self.market_df.columns.get_level_values(0)
            except:
                self.market_df = None

            if self.df.empty: return False
            
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)
            
            self.data_len = len(self.df)

            ticker_obj = yf.Ticker(self.ticker)
            try:
                self.info = ticker_obj.info
                if 'longName' not in self.info: self.info['longName'] = self.ticker
            except: self.info['longName'] = self.ticker
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def analyze_news_sentiment(self):
        headlines = []
        try:
            try:
                ticker_obj = yf.Ticker(self.ticker)
                yf_news = ticker_obj.news
                if yf_news:
                    for item in yf_news[:5]:
                        title = item.get('title')
                        if title: headlines.append(title)
            except Exception: pass

            if not headlines:
                try:
                    query = self.ticker.replace(".JK", "")
                    long_name = self.info.get('longName', '')
                    if long_name and long_name != self.ticker:
                        query = long_name.replace("PT ", "").replace(" Tbk", "").strip()
                    rss_url = f"https://news.google.com/rss/search?q={query}+Indonesia+stock&hl=en-US&gl=US&ceid=US:en"
                    response = requests.get(rss_url, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, features="xml")
                        items = soup.findAll('item')
                        for item in items[:5]:
                            title = item.find('title').text
                            headlines.append(title)
                except Exception: pass

            if not headlines:
                self.news_analysis = {"sentiment": "Neutral", "score": 0, "headlines": []}
                return

            scores = []
            final_headlines = []
            for title in headlines:
                blob = TextBlob(title)
                scores.append(blob.sentiment.polarity)
                final_headlines.append(title)

            avg_score = sum(scores) / len(scores) if scores else 0
            if avg_score > 0.1: sentiment = "Positive (Bullish)"
            elif avg_score < -0.1: sentiment = "Negative (Bearish)"
            else: sentiment = "Neutral"

            self.news_analysis = {"sentiment": sentiment, "score": round(avg_score, 3), "headlines": final_headlines[:3]}
        except Exception as e:
            self.news_analysis = {"sentiment": "Error", "score": 0, "headlines": [str(e)]}

    # --- MANUAL INDICATOR CALCULATIONS ---
    
    def calc_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calc_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def calc_stoch(self, high, low, close, k_period, d_period):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d

    def calc_atr(self, high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calc_obv(self, close, volume):
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    def calc_mfi(self, high, low, close, volume, period):
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        mf_ratio = positive_flow.rolling(window=period).sum() / negative_flow.rolling(window=period).sum()
        return 100 - (100 / (1 + mf_ratio))

    def calc_cmf(self, high, low, close, volume, period):
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_volume = mf_multiplier * volume
        return mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()

    def prepare_indicators(self):
        if self.df is None or self.df.empty: return

        # 1. Adaptive Trend (EMA)
        if self.data_len >= 200:
            self.df['EMA_200'] = self.calc_ema(self.df['Close'], 200)
            self.active_trend_col = 'EMA_200'
        elif self.data_len >= 50:
            self.df['EMA_50'] = self.calc_ema(self.df['Close'], 50)
            self.active_trend_col = 'EMA_50'
        else:
            self.df['EMA_20'] = self.calc_ema(self.df['Close'], 20)
            self.active_trend_col = 'EMA_20'

        # 2. Oscillators
        rsi_p = self.config["RSI_PERIOD"]
        if self.data_len > rsi_p:
            self.df['RSI'] = self.calc_rsi(self.df['Close'], rsi_p)
            k, d = self.calc_stoch(self.df['High'], self.df['Low'], self.df['Close'], STOCH_K_PERIOD, STOCH_D_PERIOD)
            self.df[f"STOCHk"] = k
            self.df[f"STOCHd"] = d

        # 3. MA Cross
        for fast, slow in MA_TEST_PAIRS:
            if self.data_len > slow:
                self.df[f'EMA_{fast}'] = self.calc_ema(self.df['Close'], fast)
                self.df[f'EMA_{slow}'] = self.calc_ema(self.df['Close'], slow)

        # 4. Volume & Smart Money
        cmf_p = self.config["CMF_PERIOD"]
        mfi_p = self.config["MFI_PERIOD"]
        vol_p = self.config["VOL_MA_PERIOD"]
        
        self.df['OBV'] = self.calc_obv(self.df['Close'], self.df['Volume'])
        self.df['CMF'] = self.calc_cmf(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], cmf_p)
        self.df['MFI'] = self.calc_mfi(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], mfi_p)
        self.df['VOL_MA'] = self.df['Volume'].rolling(window=vol_p).mean()
        
        # 5. ATR & Relative Volume (RVOL)
        atr_p = self.config["ATR_PERIOD"]
        self.df['ATR'] = self.calc_atr(self.df['High'], self.df['Low'], self.df['Close'], atr_p)
        self.df['RVOL'] = self.df['Volume'] / self.df['VOL_MA']

    def run_backtest_simulation(self, condition_series, hold_days):
        if condition_series is None: return 0.0 
        signals = self.df[condition_series].copy()
        if signals.empty: return 0.0
        future_close = self.df['Close'].shift(-hold_days)
        current_close = self.df['Close']
        returns = (future_close - current_close) / current_close
        trade_returns = returns[condition_series].dropna()
        if trade_returns.empty: return 0.0
        return ((trade_returns > 0).sum() / len(trade_returns)) * 100

    def optimize_stock(self, days_min, days_max):
        """Optimizes strategies for a specific timeframe (Short vs Swing)."""
        best_res = {"strategy": None, "win_rate": -1, "details": "N/A", "hold_days": 0, "is_triggered_today": False}
        
        rsi_levels = [self.config["RSI_LOWER"], self.config["RSI_LOWER"] + 10]

        if 'RSI' in self.df.columns:
            for level in rsi_levels:
                condition = self.df['RSI'] < level
                for days in range(days_min, days_max + 1):
                    wr = self.run_backtest_simulation(condition, days)
                    if wr > best_res['win_rate']:
                        best_res = {"strategy": "RSI Reversal", "details": f"RSI < {level}", "win_rate": wr, "hold_days": days, "is_triggered_today": self.df['RSI'].iloc[-1] < level}

        for fast, slow in MA_TEST_PAIRS:
            fast_col, slow_col = f'EMA_{fast}', f'EMA_{slow}'
            if fast_col in self.df.columns and slow_col in self.df.columns:
                condition = self.df[fast_col] > self.df[slow_col]
                for days in range(days_min, days_max + 1):
                    wr = self.run_backtest_simulation(condition, days)
                    if wr > best_res['win_rate']:
                        best_res = {"strategy": "MA Momentum", "details": f"Uptrend (EMA {fast} > EMA {slow})", "win_rate": wr, "hold_days": days, "is_triggered_today": self.df[fast_col].iloc[-1] > self.df[slow_col].iloc[-1]}

        if 'STOCHk' in self.df.columns:
            condition = (self.df['STOCHk'] < STOCH_OVERSOLD) & (self.df['STOCHk'] > self.df['STOCHd'])
            for days in range(days_min, days_max + 1):
                wr = self.run_backtest_simulation(condition, days)
                if wr > best_res['win_rate']:
                    best_res = {"strategy": "Stoch Reversal", "details": f"Stoch K < {STOCH_OVERSOLD}", "win_rate": wr, "hold_days": days, "is_triggered_today": (self.df['STOCHk'].iloc[-1] < STOCH_OVERSOLD) & (self.df['STOCHk'].iloc[-1] > self.df['STOCHd'].iloc[-1])}

        return best_res

    def detect_vcp_pattern(self):
        try:
            if self.data_len < 60: return {"detected": False, "msg": "Insufficient Data"}
            recent_df = self.df[-60:].copy()
            recent_df['is_peak'] = recent_df['High'] == recent_df['High'].rolling(window=5, center=True).max()
            peaks = recent_df[recent_df['is_peak']]
            recent_df['is_trough'] = recent_df['Low'] == recent_df['Low'].rolling(window=5, center=True).min()
            troughs = recent_df[recent_df['is_trough']]
            if len(peaks) < 2 or len(troughs) < 2: return {"detected": False, "msg": "No clear Swing Pattern"}
            
            last_peak_price = peaks['High'].iloc[-1]
            prev_peak_price = peaks['High'].iloc[-2]
            last_trough_price = troughs['Low'].iloc[-1]
            prev_trough_price = troughs['Low'].iloc[-2]
            
            depth_1 = (prev_peak_price - prev_trough_price) / prev_peak_price
            depth_2 = (last_peak_price - last_trough_price) / last_peak_price
            is_contracting = depth_2 < (depth_1 * 0.9)
            current_price = self.df['Close'].iloc[-1]
            near_breakout = current_price >= (last_peak_price * 0.95)
            
            if is_contracting and near_breakout:
                return {"detected": True, "msg": f"Contraction from {depth_1*100:.1f}% to {depth_2*100:.1f}%."}
            else:
                return {"detected": False, "msg": "Volatility not contracting."}
        except Exception as e: return {"detected": False, "msg": f"Error: {str(e)}"}

    def detect_geometric_patterns(self):
        result = {"pattern": "None", "msg": ""}
        try:
            if self.data_len < 60: return result
            df = self.df[-60:].copy()
            df['is_peak'] = df['High'] == df['High'].rolling(window=5, center=True).max()
            df['is_trough'] = df['Low'] == df['Low'].rolling(window=5, center=True).min()
            peaks = df[df['is_peak']]
            troughs = df[df['is_trough']]
            if len(peaks) < 2 or len(troughs) < 2: return result
            
            p2, p1 = peaks['High'].iloc[-1], peaks['High'].iloc[-2] 
            t2, t1 = troughs['Low'].iloc[-1], troughs['Low'].iloc[-2] 
            is_lower_highs = p2 < (p1 * 0.99)
            is_higher_lows = t2 > (t1 * 1.01)
            is_flat_highs  = abs(p2 - p1) / p1 < 0.01
            is_flat_lows   = abs(t2 - t1) / t1 < 0.01
            
            if is_lower_highs and is_higher_lows:
                result["pattern"] = "Symmetrical Triangle"
                result["msg"] = "Price coiling. Breakout imminent."
            elif is_flat_highs and is_higher_lows:
                result["pattern"] = "Ascending Triangle"
                result["msg"] = "Bullish Setup. Flat Resistance."
            elif is_lower_highs and is_flat_lows:
                result["pattern"] = "Descending Triangle"
                result["msg"] = "Bearish Setup. Flat Support."
            
            if result["pattern"] != "None":
                pre_consolid_df = self.df[-50:-20] 
                if not pre_consolid_df.empty:
                    low_start = pre_consolid_df['Low'].min()
                    high_end = pre_consolid_df['High'].max()
                    move_pct = (high_end - low_start) / low_start
                    if move_pct > 0.15:
                        result["pattern"] = f"Bullish Pennant ({result['pattern']})"
                        result["msg"] += " + Strong Pole Detected."
        except Exception: pass
        return result

    def detect_candle_patterns(self):
        res = {"pattern": "None", "sentiment": "Neutral"}
        try:
            if self.data_len < 4: return res
            df = self.df.iloc[-4:].copy()
            df['Body'] = abs(df['Close'] - df['Open'])
            df['BodyAvg'] = df['Body'].rolling(window=10).mean()
            df['TotalRange'] = df['High'] - df['Low']
            df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            df['IsGreen'] = df['Close'] > df['Open']
            df['IsRed'] = df['Close'] < df['Open']
            
            c0, c1, c2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
            avg_body = c0['BodyAvg'] if not np.isnan(c0['BodyAvg']) else c0['Body']
            
            is_doji = c0['Body'] <= (0.1 * c0['TotalRange'])
            is_marubozu = (c0['Body'] > 2 * avg_body) and (c0['UpperShadow'] < 0.1 * c0['Body'])
            is_hammer_shape = (c0['LowerShadow'] > 2 * c0['Body']) and (c0['UpperShadow'] < 0.5 * c0['Body'])
            is_inverted_shape = (c0['UpperShadow'] > 2 * c0['Body']) and (c0['LowerShadow'] < 0.5 * c0['Body'])
            recent_move = self.df['Close'].iloc[-1] - self.df['Close'].iloc[-5]
            is_uptrend, is_downtrend = recent_move > 0, recent_move < 0

            if c1['IsRed'] and c0['IsGreen'] and (c0['Close'] > c1['Open']): res = {"pattern": "Bullish Engulfing", "sentiment": "Strong Reversal Up"}
            elif c1['IsGreen'] and c0['IsRed'] and (c0['Close'] < c1['Open']): res = {"pattern": "Bearish Engulfing", "sentiment": "Strong Reversal Down"}
            elif is_downtrend and c2['IsRed'] and (c1['Body'] < 0.5 * c2['Body']) and c0['IsGreen'] and (c0['Close'] > (c2['Open'] + c2['Close'])/2): res = {"pattern": "Morning Star", "sentiment": "Major Bullish Reversal"}
            elif is_uptrend and c2['IsGreen'] and (c1['Body'] < 0.5 * c2['Body']) and c0['IsRed'] and (c0['Close'] < (c2['Open'] + c2['Close'])/2): res = {"pattern": "Evening Star", "sentiment": "Major Bearish Reversal"}
            elif c0['IsGreen'] and c1['IsGreen'] and c2['IsGreen'] and (c0['Close'] > c1['Close'] > c2['Close']): res = {"pattern": "Three White Soldiers", "sentiment": "Strong Bullish Momentum"}
            elif res["pattern"] == "None":
                if is_hammer_shape:
                    if is_downtrend: res = {"pattern": "Hammer", "sentiment": "Bullish Reversal"}
                    elif is_uptrend: res = {"pattern": "Hanging Man", "sentiment": "Bearish Warning"}
                elif is_inverted_shape:
                    if is_downtrend: res = {"pattern": "Inverted Hammer", "sentiment": "Potential Bottom"}
                    elif is_uptrend: res = {"pattern": "Shooting Star", "sentiment": "Bearish Reversal"}
                elif is_marubozu: res = {"pattern": "Marubozu", "sentiment": "Strong Trend"}
                elif is_doji: res = {"pattern": "Doji", "sentiment": "Indecision"}
        except Exception: pass
        return res

    def validate_signal(self, action, context):
        score = 0
        reasons = []
        
        rvol = self.df['RVOL'].iloc[-1] if 'RVOL' in self.df.columns else 1.0
        if rvol > 1.2:
            score += 1
            reasons.append("High Volume")
        
        if "BUYING" in context['smart_money']:
            score += 1
            reasons.append("Smart Money Accumulation")
            
        if self.market_df is not None and len(self.market_df) > 5:
            stock_ret = (self.df['Close'].iloc[-1] - self.df['Close'].iloc[-5]) / self.df['Close'].iloc[-5]
            market_ret = (self.market_df['Close'].iloc[-1] - self.market_df['Close'].iloc[-5]) / self.market_df['Close'].iloc[-5]
            if stock_ret > market_ret:
                score += 1
                reasons.append("Outperforming Index (Leader)")
                
        if "Bullish" in context['candle']['sentiment'] or "Strong" in context['candle']['sentiment']:
            score += 1
            reasons.append("Bullish Candle Pattern")
            
        if 'EMA_50' in self.df.columns and self.df['Close'].iloc[-1] > self.df['EMA_50'].iloc[-1]:
            score += 1
            reasons.append("Aligned with Med-Term Trend")
            
        verdict = "WEAK"
        if score >= 4: verdict = "STRONG CONVICTION"
        elif score >= 2: verdict = "MODERATE"
        
        return score, verdict, reasons

    def get_market_context(self):
        last_price = self.df['Close'].iloc[-1]
        lookback = min(20, self.data_len)
        recent = self.df[-lookback:]
        support, resistance = recent['Low'].min(), recent['High'].max()
        dist_supp = ((last_price - support) / support) * 100
        
        fib_len = self.config["FIB_LOOKBACK_DAYS"]
        fib_lookback = min(fib_len, self.data_len)
        fib_win = self.df[-fib_lookback:]
        sh, sl = fib_win['High'].max(), fib_win['Low'].min()
        rng = sh - sl
        fibs = { "1.0 (Low)": sl, "0.618 (Golden)": sh-(0.618*rng), "0.5 (Half)": sh-(0.5*rng), "0.382": sh-(0.382*rng), "0.0 (High)": sh } if rng > 0 else {}
        
        trend = "NEUTRAL"
        if self.active_trend_col in self.df.columns:
            trend = "UPTREND" if last_price > self.df[self.active_trend_col].iloc[-1] else "DOWNTREND"
        
        atr = self.df['ATR'].iloc[-1] if 'ATR' in self.df.columns else 0
        
        obv_status = "Neutral"
        money_flow = "Neutral"
        if self.data_len > OBV_LOOKBACK_DAYS:
            curr_obv, prev_obv = self.df['OBV'].iloc[-1], self.df['OBV'].iloc[-OBV_LOOKBACK_DAYS]
            last_p, prev_p = self.df['Close'].iloc[-1], self.df['Close'].iloc[-OBV_LOOKBACK_DAYS]
            if last_p < prev_p and curr_obv > prev_obv: obv_status = "Bullish Divergence"
            elif last_p > prev_p and curr_obv < prev_obv: obv_status = "Bearish Divergence"
            elif curr_obv > prev_obv: obv_status = "Rising"
            else: obv_status = "Falling"

            cmf = self.df['CMF'].iloc[-1] if 'CMF' in self.df.columns else 0
            mfi = self.df['MFI'].iloc[-1] if 'MFI' in self.df.columns else 50
            vol = self.df['Volume'].iloc[-1]
            vol_ma = self.df['VOL_MA'].iloc[-1] if 'VOL_MA' in self.df.columns else 1
            
            if cmf > 0.05: money_flow = "INSTITUTIONAL BUYING" if mfi < 80 else "BUYING FRENZY"
            elif cmf < -0.05: money_flow = "INSTITUTIONAL SELLING"
            else: money_flow = "RETAIL NOISE"
            if vol > (2.0 * vol_ma): money_flow += " [HIGH VOLUME]"

        return {
            "price": last_price, "support": support, "resistance": resistance,
            "dist_support": dist_supp, "fib_levels": fibs, "trend": trend,
            "atr": atr, "obv_status": obv_status, "smart_money": money_flow,
            "vcp": self.detect_vcp_pattern(), "geo": self.detect_geometric_patterns(),
            "candle": self.detect_candle_patterns()
        }

    def calculate_trade_plan(self, plan_type, action, current_price, atr, support, resistance, best_strategy, fib_levels):
        """Generates separate plans for Short Term and Swing."""
        plan = {"type": plan_type, "entry": 0, "stop_loss": 0, "take_profit": 0, "risk_reward": "N/A", "status": "ACTIVE"}
        
        if plan_type == "SHORT_TERM":
            sl_mult = 1.5
            tp_mult = 2.0
        else: # SWING
            sl_mult = self.config["SL_MULTIPLIER"]
            tp_mult = 4.0

        if "BUY" in action:
            plan['entry'] = current_price
            plan['status'] = "EXECUTE NOW (Market)"
            plan['stop_loss'] = current_price - (atr * sl_mult)
            plan['take_profit'] = current_price + (atr * tp_mult)
            plan['risk_reward'] = f"1:{tp_mult/sl_mult:.1f}"
            
        elif "WAIT" in action:
            plan['status'] = "PENDING (Limit)"
            strategy_type = best_strategy.get('strategy', 'None')
            
            if "RSI" in strategy_type or "Stoch" in strategy_type:
                target_fib = 0
                for _, price in sorted(fib_levels.items(), key=lambda x: x[1], reverse=True):
                    if price < current_price:
                        target_fib = price
                        break
                
                if target_fib > (current_price * 0.85):
                    plan['entry'] = target_fib
                    plan['note'] = "Wait for Fib Support"
                else:
                    plan['entry'] = support
                    plan['note'] = "Wait for Support"
                    
            elif "MA" in strategy_type:
                plan['entry'] = resistance
                plan['note'] = "Buy Breakout"
            else:
                plan['entry'] = support
                plan['note'] = "Wait for Support"

            if plan['entry'] > 0:
                plan['stop_loss'] = plan['entry'] - (atr * sl_mult)
                plan['take_profit'] = plan['entry'] + (atr * tp_mult)
                plan['risk_reward'] = "Projection"
            
        return plan

    def generate_final_report(self):
        if not self.fetch_data(): return None
        self.prepare_indicators()
        self.analyze_news_sentiment()
        
        # Dual Optimization
        # UPDATED: Swing optimization looks up to 60 days (approx 3 months)
        best_short = self.optimize_stock(1, 5)
        best_swing = self.optimize_stock(6, 60)
        
        ctx = self.get_market_context()
        
        # Decision Logic (Short Term)
        action_short = "WAIT"
        if best_short['is_triggered_today']: action_short = "ACTION: BUY (Signal)"
        elif ctx['dist_support'] < 2.0: action_short = "ACTION: BUY (Support Scalp)"
        
        # Decision Logic (Swing)
        action_swing = "WAIT"
        if best_swing['is_triggered_today'] and ctx['trend'] == "UPTREND": action_swing = "ACTION: BUY (Trend)"
        elif ctx['dist_support'] < 3.0 and ctx['smart_money'] == "INSTITUTIONAL BUYING": action_swing = "ACTION: BUY (Accumulation)"

        val_score, val_verdict, val_reasons = self.validate_signal(action_short if action_short == "BUY" else action_swing, ctx)

        plan_short = self.calculate_trade_plan("SHORT_TERM", action_short, ctx['price'], ctx['atr'], ctx['support'], ctx['resistance'], best_short, ctx['fib_levels'])
        plan_swing = self.calculate_trade_plan("SWING", action_swing, ctx['price'], ctx['atr'], ctx['support'], ctx['resistance'], best_swing, ctx['fib_levels'])

        return {
            "ticker": self.ticker, "name": self.info.get('longName', self.ticker),
            "price": ctx['price'], 
            "sentiment": self.news_analysis, 
            "context": ctx,
            "plans": [plan_short, plan_swing],
            "validation": {"score": val_score, "verdict": val_verdict, "reasons": val_reasons},
            "is_ipo": self.data_len < 200, "days_listed": self.data_len
        }


