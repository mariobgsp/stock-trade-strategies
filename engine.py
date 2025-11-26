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
    "MAX_HOLD_DAYS": 60, # 3 Months (Trading Days)
    "FIB_LOOKBACK_DAYS": 120,
    "RSI_PERIOD": 14,
    "RSI_LOWER": 30,
    "ATR_PERIOD": 14,
    "SL_MULTIPLIER": 2.5, # Default Wider for Swing Safety
    "TP_MULTIPLIER": 5.0, 
    "CMF_PERIOD": 20,
    "MFI_PERIOD": 14,
    "VOL_MA_PERIOD": 20,
    "MIN_MARKET_CAP": 500_000_000_000, 
    "MIN_DAILY_VOL": 1_000_000_000
}

MA_TEST_PAIRS = [(5, 20), (20, 50), (50, 200)] 
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20
OBV_LOOKBACK_DAYS = 10
TREND_EMA_DEFAULT = 200

class StockAnalyzer:
    def __init__(self, ticker, user_config=None):
        self.ticker = self._format_ticker(ticker)
        self.market_ticker = "^JKSE"
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
            self.df = yf.download(self.ticker, period=period, progress=False, auto_adjust=True)
            
            try:
                self.market_df = yf.download(self.market_ticker, period=period, progress=False, auto_adjust=True)
                if isinstance(self.market_df.columns, pd.MultiIndex):
                    self.market_df.columns = self.market_df.columns.get_level_values(0)
            except: self.market_df = None

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

    # --- MATH HELPERS ---
    def calc_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calc_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def calc_sma(self, series, period):
        return series.rolling(window=period).mean()

    def calc_std(self, series, period):
        return series.rolling(window=period).std()

    def calc_slope(self, series, period=20):
        if len(series) < period: return 0
        y = series.iloc[-period:].values
        x = np.arange(len(y))
        try:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except: return 0

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

    def calc_stoch(self, high, low, close, k_period, d_period):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d

    def prepare_indicators(self):
        if self.df is None or self.df.empty: return

        self.df['EMA_50'] = self.calc_ema(self.df['Close'], 50)
        self.df['EMA_150'] = self.calc_ema(self.df['Close'], 150)
        self.df['EMA_200'] = self.calc_ema(self.df['Close'], 200)
        self.active_trend_col = 'EMA_200'

        rsi_p = self.config["RSI_PERIOD"]
        self.df['RSI'] = self.calc_rsi(self.df['Close'], rsi_p)
        k, d = self.calc_stoch(self.df['High'], self.df['Low'], self.df['Close'], STOCH_K_PERIOD, STOCH_D_PERIOD)
        self.df[f"STOCHk"] = k
        self.df[f"STOCHd"] = d

        cmf_p, mfi_p, vol_p = self.config["CMF_PERIOD"], self.config["MFI_PERIOD"], self.config["VOL_MA_PERIOD"]
        self.df['OBV'] = self.calc_obv(self.df['Close'], self.df['Volume'])
        self.df['CMF'] = self.calc_cmf(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], cmf_p)
        self.df['MFI'] = self.calc_mfi(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], mfi_p)
        self.df['VOL_MA'] = self.df['Volume'].rolling(window=vol_p).mean()
        self.df['RVOL'] = self.df['Volume'] / self.df['VOL_MA']

        atr_p = self.config["ATR_PERIOD"]
        self.df['ATR'] = self.calc_atr(self.df['High'], self.df['Low'], self.df['Close'], atr_p)

    def check_trend_template(self):
        res = {"status": "FAIL", "score": 0, "details": []}
        try:
            if self.data_len < 260:
                res["details"].append("Insufficient data for full trend check")
                return res

            curr = self.df['Close'].iloc[-1]
            ema_50 = self.df['EMA_50'].iloc[-1]
            ema_150 = self.df['EMA_150'].iloc[-1]
            ema_200 = self.df['EMA_200'].iloc[-1]
            
            year_high = self.df['High'].iloc[-260:].max()
            year_low = self.df['Low'].iloc[-260:].min()
            
            c1 = curr > ema_150 and curr > ema_200
            c2 = ema_150 > ema_200
            slope_200 = self.calc_slope(self.df['EMA_200'], 20)
            c3 = slope_200 > 0
            c4 = curr > ema_50
            c5 = curr >= (1.25 * year_low)
            c6 = curr >= (0.75 * year_high)
            
            score = sum([c1, c2, c3, c4, c5, c6])
            res["score"] = score
            
            if score == 6: res["status"] = "PERFECT UPTREND (Stage 2)"
            elif score >= 4: res["status"] = "STRONG UPTREND"
            elif score <= 2: res["status"] = "DOWNTREND / BASE"
                
            if c1 and c2: res["details"].append("MA Alignment (Price > 150 > 200)")
            if c3: res["details"].append("200-Day MA Rising")
            if c5: res["details"].append("> 25% Off Lows (Momentum)")
            if c6: res["details"].append("Near 52-Week Highs (Leader)")
            if not c4: res["details"].append("WARNING: Price below 50 EMA")

        except Exception as e: res["details"].append(f"Error: {str(e)}")
        return res

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
        best_res = {"strategy": None, "win_rate": -1, "details": "N/A", "hold_days": 0, "is_triggered_today": False}
        rsi_levels = [self.config["RSI_LOWER"], self.config["RSI_LOWER"] + 10]

        if 'RSI' in self.df.columns:
            for level in rsi_levels:
                condition = self.df['RSI'] < level
                for days in range(days_min, days_max + 1):
                    wr = self.run_backtest_simulation(condition, days)
                    if wr > best_res['win_rate']:
                        best_res = {"strategy": "RSI Reversal", "details": f"RSI < {level}", "win_rate": wr, "hold_days": days, "is_triggered_today": self.df['RSI'].iloc[-1] < level}

        if 'EMA_50' in self.df.columns and self.df['EMA_50'].iloc[-1] > self.df['EMA_200'].iloc[-1]:
             condition = self.df['EMA_50'] > self.df['EMA_200'] 
             for days in range(days_min, days_max + 1):
                wr = self.run_backtest_simulation(condition, days)
                if wr > best_res['win_rate']:
                    best_res = {"strategy": "MA Trend", "details": "Trend Following (50 > 200)", "win_rate": wr, "hold_days": days, "is_triggered_today": True}

        if 'STOCHk' in self.df.columns:
            condition = (self.df['STOCHk'] < STOCH_OVERSOLD) & (self.df['STOCHk'] > self.df['STOCHd'])
            for days in range(days_min, days_max + 1):
                wr = self.run_backtest_simulation(condition, days)
                if wr > best_res['win_rate']:
                    best_res = {"strategy": "Stoch Reversal", "details": f"Stoch K < {STOCH_OVERSOLD}", "win_rate": wr, "hold_days": days, "is_triggered_today": (self.df['STOCHk'].iloc[-1] < STOCH_OVERSOLD) & (self.df['STOCHk'].iloc[-1] > self.df['STOCHd'].iloc[-1])}

        return best_res

    def check_fundamentals(self):
        res = {"market_cap": 0, "eps": 0, "status": "Unknown", "warning": ""}
        try:
            mcap = self.info.get('marketCap', 0)
            eps = self.info.get('trailingEps', 0)
            res['market_cap'] = mcap
            res['eps'] = eps
            min_cap = self.config["MIN_MARKET_CAP"]
            if mcap == 0: res['status'] = "Unknown (Data Missing)"
            elif mcap < min_cap:
                res['status'] = "SMALL CAP (High Risk)"
                res['warning'] = "Market Cap < 500B IDR. Prone to manipulation."
            elif eps < 0:
                res['status'] = "UNPROFITABLE"
                res['warning'] = "Company has negative Earnings Per Share."
            else: res['status'] = "GOOD"
        except Exception: pass
        return res

    def detect_ttm_squeeze(self):
        res = {"detected": False, "msg": ""}
        try:
            if self.data_len < 20: return res
            sma20 = self.calc_sma(self.df['Close'], 20)
            std20 = self.calc_std(self.df['Close'], 20)
            upper_bb = sma20 + (2.0 * std20)
            lower_bb = sma20 - (2.0 * std20)
            atr20 = self.calc_atr(self.df['High'], self.df['Low'], self.df['Close'], 20)
            upper_kc = sma20 + (1.5 * atr20)
            lower_kc = sma20 - (1.5 * atr20)
            is_squeeze = (upper_bb.iloc[-1] < upper_kc.iloc[-1]) and (lower_bb.iloc[-1] > lower_kc.iloc[-1])
            if is_squeeze: res = {"detected": True, "msg": "TTM Squeeze ON! Massive breakout imminent."}
        except Exception: pass
        return res

    def calculate_pivot_points(self):
        pivots = {"P": 0, "R1": 0, "S1": 0}
        try:
            if self.data_len < 2: return pivots
            prev = self.df.iloc[-2] 
            high, low, close = prev['High'], prev['Low'], prev['Close']
            p = (high + low + close) / 3
            r1 = (2 * p) - low
            s1 = (2 * p) - high
            pivots = {"P": p, "R1": r1, "S1": s1}
        except Exception: pass
        return pivots

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

    def _detect_geometry_on_slice(self, df_slice):
        result = {"pattern": "None", "msg": ""}
        if len(df_slice) < 60: return result
        
        df = df_slice[-60:].copy()
        df['is_peak'] = df['High'] == df['High'].rolling(window=5, center=True).max()
        df['is_trough'] = df['Low'] == df['Low'].rolling(window=5, center=True).min()
        peaks = df[df['is_peak']]
        troughs = df[df['is_trough']]
        
        if len(peaks) < 2 or len(troughs) < 2: return result
        
        p2, p1 = peaks['High'].iloc[-1], peaks['High'].iloc[-2]
        t2, t1 = troughs['Low'].iloc[-1], troughs['Low'].iloc[-2]
        
        p2_idx, p1_idx = df.index.get_loc(peaks.index[-1]), df.index.get_loc(peaks.index[-2])
        t2_idx, t1_idx = df.index.get_loc(troughs.index[-1]), df.index.get_loc(troughs.index[-2])

        m_res = (p2 - p1) / (p2_idx - p1_idx) if (p2_idx - p1_idx) != 0 else 0
        m_sup = (t2 - t1) / (t2_idx - t1_idx) if (t2_idx - t1_idx) != 0 else 0
        
        if m_res < -0.01 and m_sup > 0.01: result["pattern"] = "Symmetrical Triangle"
        elif abs(m_res) < 0.01 and m_sup > 0.01: result["pattern"] = "Ascending Triangle"
        elif m_res < -0.01 and abs(m_sup) < 0.01: result["pattern"] = "Descending Triangle"
        
        c_res = p1 - (m_res * p1_idx)
        c_sup = t1 - (m_sup * t1_idx)
        apex_x = 0
        if (m_res - m_sup) != 0: apex_x = (c_sup - c_res) / (m_res - m_sup)
        result["apex_dist"] = apex_x - (len(df) - 1)
        
        return result

    def detect_geometric_patterns(self):
        res = self._detect_geometry_on_slice(self.df)
        if res["pattern"] != "None":
            if "apex_dist" in res and 0 < res["apex_dist"] < 30:
                res["msg"] += f" Apex in ~{int(res['apex_dist'])} days."
        return res

    def backtest_pattern_reliability(self):
        if self.data_len < 200: return {"accuracy": "N/A", "count": 0}
        wins = 0
        total_patterns = 0
        
        for i in range(100, self.data_len - 20, 5):
            slice_df = self.df.iloc[:i]
            res = self._detect_geometry_on_slice(slice_df)
            if res["pattern"] != "None":
                total_patterns += 1
                future_window = self.df.iloc[i : i+20]
                entry_price = slice_df['Close'].iloc[-1]
                max_price = future_window['High'].max()
                if max_price > (entry_price * 1.03): wins += 1
        
        if total_patterns == 0: return {"accuracy": "N/A", "count": 0}
        win_rate = (wins / total_patterns) * 100
        verdict = "Likely Success" if win_rate > 60 else "Likely Fail" if win_rate < 40 else "Coin Flip"
        return { "accuracy": f"{win_rate:.1f}%", "count": total_patterns, "verdict": verdict, "wins": wins }

    def detect_candle_patterns(self):
        res = {"pattern": "None", "sentiment": "Neutral"}
        try:
            if self.data_len < 4: return res
            df = self.df.iloc[-4:].copy()
            df['Body'] = abs(df['Close'] - df['Open'])
            c0, c1 = df.iloc[-1], df.iloc[-2]
            is_green = c0['Close'] > c0['Open']
            if not is_green and c0['Open'] > c1['Close'] and c0['Close'] < c1['Open']: 
                res = {"pattern": "Bearish Engulfing", "sentiment": "Strong Reversal Down"}
            elif is_green and c0['Close'] > c1['Open'] and c0['Open'] < c1['Close']: 
                res = {"pattern": "Bullish Engulfing", "sentiment": "Strong Reversal Up"}
        except Exception: pass
        return res

    def validate_signal(self, action, context, trend_template):
        score = 0
        reasons = []
        
        if trend_template["status"] in ["PERFECT UPTREND (Stage 2)", "STRONG UPTREND"]:
            score += 2; reasons.append("Stage 2 Uptrend (Minervini)")
        
        rvol = self.df['RVOL'].iloc[-1] if 'RVOL' in self.df.columns else 1.0
        if rvol > 1.2: score += 1; reasons.append("High Volume")
        
        if "BUYING" in context['smart_money']: score += 1; reasons.append("Smart Money Accumulation")
            
        if self.market_df is not None and len(self.market_df) > 5:
            s_ret = (self.df['Close'].iloc[-1] - self.df['Close'].iloc[-5]) / self.df['Close'].iloc[-5]
            m_ret = (self.market_df['Close'].iloc[-1] - self.market_df['Close'].iloc[-5]) / self.market_df['Close'].iloc[-5]
            if s_ret > m_ret: score += 1; reasons.append("Leader vs IHSG")
                
        if context['squeeze']['detected']: score += 2; reasons.append("TTM Squeeze Firing")

        pat_stats = context.get('pattern_stats', {})
        if "Success" in pat_stats.get('verdict', ''):
            score += 1; reasons.append("Historical Pattern Success")

        verdict = "WEAK"
        if score >= 5: verdict = "ELITE SWING SETUP"
        elif score >= 3: verdict = "MODERATE"
        return score, verdict, reasons

    def get_market_context(self):
        last_price = self.df['Close'].iloc[-1]
        lookback = min(20, self.data_len)
        recent = self.df[-lookback:]
        support, resistance = recent['Low'].min(), recent['High'].max()
        dist_supp = ((last_price - support) / support) * 100
        
        fib_len = self.config["FIB_LOOKBACK_DAYS"]
        fib_win = self.df[-min(fib_len, self.data_len):]
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
            cmf = self.df['CMF'].iloc[-1] if 'CMF' in self.df.columns else 0
            mfi = self.df['MFI'].iloc[-1] if 'MFI' in self.df.columns else 50
            vol = self.df['Volume'].iloc[-1]
            vol_ma = self.df['VOL_MA'].iloc[-1] if 'VOL_MA' in self.df.columns else 1
            
            if cmf > 0.05: money_flow = "INSTITUTIONAL BUYING" if mfi < 80 else "BUYING FRENZY"
            elif cmf < -0.05: money_flow = "INSTITUTIONAL SELLING"
            else: money_flow = "RETAIL NOISE"
            if vol > (2.0 * vol_ma): money_flow += " [HIGH VOLUME]"
        
        geo_status = self.detect_geometric_patterns()
        pattern_stats = {}
        if geo_status["pattern"] != "None":
            pattern_stats = self.backtest_pattern_reliability()

        return {
            "price": last_price, "support": support, "resistance": resistance,
            "dist_support": dist_supp, "fib_levels": fibs, "trend": trend,
            "atr": atr, "obv_status": obv_status, "smart_money": money_flow,
            "vcp": self.detect_vcp_pattern(), "geo": geo_status,
            "candle": self.detect_candle_patterns(),
            "fundamental": self.check_fundamentals(),
            "squeeze": self.detect_ttm_squeeze(),
            "pivots": self.calculate_pivot_points(),
            "pattern_stats": pattern_stats
        }

    def adjust_to_tick_size(self, price):
        if price < 200: tick = 1
        elif price < 500: tick = 2
        elif price < 2000: tick = 5
        elif price < 5000: tick = 10
        else: tick = 25
        return round(price / tick) * tick

    def calculate_trade_plan(self, plan_type, action, current_price, atr, support, resistance, best_strategy, fib_levels, pivots, trend_status):
        plan = {"type": plan_type, "entry": 0, "stop_loss": 0, "take_profit": 0, "risk_reward": "N/A", "status": "ACTIVE"}
        sl_mult = self.config["SL_MULTIPLIER"]
        tp_mult = self.config["TP_MULTIPLIER"]

        if "UPTREND" not in trend_status:
             action = "WAIT"

        if "BUY" in action:
            plan['entry'] = self.adjust_to_tick_size(current_price)
            plan['status'] = "EXECUTE NOW (Market)"
            
            sl_price = self.adjust_to_tick_size(current_price - (atr * sl_mult))
            tp_price = self.adjust_to_tick_size(current_price + (atr * tp_mult))
            
            plan['stop_loss'] = sl_price
            plan['take_profit'] = tp_price
            plan['risk_reward'] = f"1:{tp_mult/sl_mult:.1f}"

            risk = current_price - sl_price
            if risk > 0:
                tp_3r = current_price + (risk * 3.0)
                plan['take_profit_3r'] = self.adjust_to_tick_size(tp_3r)
            
        elif "WAIT" in action:
            plan['status'] = "PENDING (Limit)"
            strategy_type = best_strategy.get('strategy', 'None')
            target_price = support 
            
            if "RSI" in strategy_type or "Stoch" in strategy_type:
                target_fib = 0
                for _, price in sorted(fib_levels.items(), key=lambda x: x[1], reverse=True):
                    if price < current_price: target_fib = price; break
                
                if target_fib > (current_price * 0.85): target_price = target_fib; plan['note'] = "Wait for Fib Support"
                else: target_price = support; plan['note'] = "Wait for Major Support"
            elif "MA" in strategy_type:
                target_price = resistance; plan['note'] = "Buy Breakout"

            plan['entry'] = self.adjust_to_tick_size(target_price)
            if plan['entry'] > 0:
                sl_price = self.adjust_to_tick_size(plan['entry'] - (atr * sl_mult))
                plan['stop_loss'] = sl_price
                plan['take_profit'] = self.adjust_to_tick_size(plan['entry'] + (atr * tp_mult))
                
                risk = plan['entry'] - sl_price
                if risk > 0:
                    tp_3r = plan['entry'] + (risk * 3.0)
                    plan['take_profit_3r'] = self.adjust_to_tick_size(tp_3r)
                
                plan['risk_reward'] = "Projection"
            
        return plan

    def generate_final_report(self):
        if not self.fetch_data(): return None
        self.prepare_indicators()
        self.analyze_news_sentiment()
        
        best_strategy = self.optimize_stock(1, 60) 
        
        ctx = self.get_market_context()
        trend_template = self.check_trend_template()
        
        action = "WAIT"
        if best_strategy['is_triggered_today'] and trend_template['score'] >= 4: 
            action = "ACTION: BUY (Trend)"
        elif ctx['dist_support'] < 3.0 and ctx['smart_money'] == "INSTITUTIONAL BUYING": 
            action = "ACTION: BUY (Accumulation)"

        val_score, val_verdict, val_reasons = self.validate_signal(action, ctx, trend_template)

        plan = self.calculate_trade_plan("OPTIMIZED_SWING", action, ctx['price'], ctx['atr'], ctx['support'], ctx['resistance'], best_strategy, ctx['fib_levels'], ctx['pivots'], trend_template['status'])

        return {
            "ticker": self.ticker, "name": self.info.get('longName', self.ticker),
            "price": ctx['price'], "sentiment": self.news_analysis, "context": ctx,
            "plans": [plan], 
            "validation": {"score": val_score, "verdict": val_verdict, "reasons": val_reasons},
            "trend_template": trend_template, 
            "is_ipo": self.data_len < 200, "days_listed": self.data_len
        }


