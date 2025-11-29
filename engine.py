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
# 1. CONFIGURATION (MERGED)
# ==========================================
DEFAULT_CONFIG = {
    "BACKTEST_PERIOD": "2y",
    "MAX_HOLD_DAYS": 60,
    "FIB_LOOKBACK_DAYS": 120,
    "RSI_PERIOD": 14,
    "RSI_LOWER": 30,
    "ATR_PERIOD": 14,
    "SL_MULTIPLIER": 2.5,
    "TP_MULTIPLIER": 3.0, # Adjusted for 3R
    "CMF_PERIOD": 20,
    "MFI_PERIOD": 14,
    "VOL_MA_PERIOD": 20,
    "MIN_MARKET_CAP": 500_000_000_000, 
    "MIN_DAILY_VOL": 1_000_000_000,
    "MIN_ADTV_IDR": 5_000_000_000, # NEW: Anti-Gorengan Filter
    "ACCOUNT_BALANCE": 100_000_000, # NEW: For Sizing
    "RISK_PER_TRADE_PCT": 1.0       # NEW: For Sizing
}

# Financial Dictionary for Better Sentiment (New)
FIN_BULLISH = {'surge', 'jump', 'soar', 'record', 'profit', 'dividend', 'buyback', 'growth', 'beat', 'bull', 'merger', 'acquisition', 'climb', 'gain', 'net income'}
FIN_BEARISH = {'plunge', 'drop', 'slump', 'loss', 'miss', 'debt', 'bankrupt', 'sue', 'lawsuit', 'investigation', 'bear', 'cut', 'down', 'fail', 'weak', 'inflation'}

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
            self.df = yf.download(self.ticker, period=period, progress=False, auto_adjust=True, multi_level_index=False)
            try:
                self.market_df = yf.download(self.market_ticker, period=period, progress=False, auto_adjust=True, multi_level_index=False)
            except: self.market_df = None

            if self.df.empty: return False
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
            # 1. Google News RSS (Indonesia Focus)
            query = self.ticker.replace(".JK", "")
            long_name = self.info.get('longName', '')
            if long_name and long_name != self.ticker:
                query = long_name.replace("PT ", "").replace(" Tbk", "").strip()
            
            rss_url = f"https://news.google.com/rss/search?q={query}+Indonesia+saham&hl=id-ID&gl=ID&ceid=ID:id"
            response = requests.get(rss_url, timeout=5)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, features="xml")
                items = soup.findAll('item')
                for item in items[:5]: headlines.append(item.find('title').text)

            if not headlines:
                self.news_analysis = {"sentiment": "Neutral", "score": 0, "headlines": []}
                return

            # 2. Dictionary Analysis (Enhanced)
            score = 0
            for title in headlines:
                t_lower = title.lower()
                bull = sum(1 for w in FIN_BULLISH if w in t_lower)
                bear = sum(1 for w in FIN_BEARISH if w in t_lower)
                score += (bull * 1.5) - (bear * 1.5)

            final_score = score / len(headlines) if headlines else 0
            sentiment = "Positive (Bullish)" if final_score > 0.2 else "Negative (Bearish)" if final_score < -0.2 else "Neutral"
            self.news_analysis = {"sentiment": sentiment, "score": round(final_score, 3), "headlines": headlines[:3]}
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

    def calc_force_index(self, close, volume, period):
        fi = close.diff(1) * volume
        return self.calc_ema(fi, period)

    def calc_stoch(self, high, low, close, k_period, d_period):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
        
    def calc_amihud(self, close, volume, period):
        ret = close.pct_change().abs()
        dol_vol = close * volume
        amihud = (ret / dol_vol) * 1000000000
        return amihud.rolling(window=period).mean()

    def prepare_indicators(self):
        if self.df is None or self.df.empty: return

        # Calculate all key MAs for backtesting
        self.df['EMA_20'] = self.calc_ema(self.df['Close'], 20)
        self.df['EMA_50'] = self.calc_ema(self.df['Close'], 50)
        self.df['EMA_100'] = self.calc_ema(self.df['Close'], 100) # Added
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
        self.df['TxValue'] = self.df['Close'] * self.df['Volume'] # Value Traded
        
        self.df['EFI'] = self.calc_force_index(self.df['Close'], self.df['Volume'], 13)
        
        # --- NEW: VWAP & NVI (Enhanced Smart Money) ---
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['VWAP'] = (tp * self.df['Volume']).rolling(20).sum() / self.df['Volume'].rolling(20).sum()
        
        # NVI
        roc = self.df['Close'].pct_change()
        vol_down = self.df['Volume'] < self.df['Volume'].shift(1)
        nvi = [1000]
        for i in range(1, len(self.df)):
            if vol_down.iloc[i]:
                prev = nvi[-1]
                change = prev * roc.iloc[i]
                nvi.append(prev + change)
            else:
                nvi.append(nvi[-1])
        self.df['NVI'] = pd.Series(nvi, index=self.df.index)
        self.df['NVI_EMA'] = self.df['NVI'].ewm(span=255).mean()
        
        self.df['AMIHUD'] = self.calc_amihud(self.df['Close'], self.df['Volume'], 20)
        atr_p = self.config["ATR_PERIOD"]
        self.df['ATR'] = self.calc_atr(self.df['High'], self.df['Low'], self.df['Close'], atr_p)

    def check_liquidity_quality(self):
        # NEW: Anti-Gorengan Filter
        try:
            adtv = self.df['TxValue'].rolling(20).mean().iloc[-1]
            min_adtv = self.config["MIN_ADTV_IDR"]
            if adtv < min_adtv:
                return {"status": "FAIL", "msg": f"Low Liquidity ({adtv/1e9:.2f}B IDR)", "adtv": adtv}
            return {"status": "PASS", "msg": f"Healthy Liquidity ({adtv/1e9:.2f}B IDR)", "adtv": adtv}
        except: return {"status": "UNKNOWN", "msg": "Calc Error", "adtv": 0}

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

    # --- NEW: RECTANGLE / BOX PATTERN ---
    def detect_rectangle_pattern(self):
        res = {"detected": False, "top": 0, "bottom": 0, "msg": "No Pattern"}
        try:
            if self.data_len < 60: return res
            
            window = self.df.iloc[-60:].copy()
            window['is_high'] = window['High'] == window['High'].rolling(5, center=True).max()
            window['is_low'] = window['Low'] == window['Low'].rolling(5, center=True).min()
            
            highs = window[window['is_high']]['High'].values
            lows = window[window['is_low']]['Low'].values
            
            if len(highs) < 2 or len(lows) < 2: return res
            
            highs.sort(); lows.sort()
            
            best_top, max_h = 0, 0
            for h in highs:
                c = [x for x in highs if abs(x - h)/h < 0.02]
                if len(c) > max_h: max_h, best_top = len(c), sum(c)/len(c)
            
            best_bot, max_l = 0, 0
            for l in lows:
                c = [x for x in lows if abs(x - l)/l < 0.02]
                if len(c) > max_l: max_l, best_bot = len(c), sum(c)/len(c)
            
            valid = max_h >= 2 and max_l >= 2
            if not valid or best_bot == 0: return res
            
            height_pct = (best_top - best_bot) / best_bot
            curr = self.df['Close'].iloc[-1]
            valid_loc = curr > (best_bot * 0.98)
            
            if valid and 0.03 < height_pct < 0.25 and valid_loc:
                status = "INSIDE BOX"
                if curr > best_top: status = "BREAKOUT"
                
                res = {
                    "detected": True, "top": best_top, "bottom": best_bot, 
                    "height_pct": height_pct * 100, "status": status, 
                    "msg": f"Box ({best_bot:,.0f}-{best_top:,.0f})"
                }
        except: pass
        return res

    # --- NEW: SMART MONEY ANALYSIS (NVI + VWAP) ---
    def analyze_smart_money_enhanced(self):
        res = {"status": "NEUTRAL", "signals": []}
        try:
            c0 = self.df.iloc[-1]
            score = 0
            
            if c0['Close'] > c0['VWAP']:
                score += 1
                res['signals'].append("Price > VWAP (Inst. Support)")
            else:
                score -= 1
                res['signals'].append("Price < VWAP (Weakness)")

            if c0['NVI'] > c0['NVI_EMA']:
                score += 1
                res['signals'].append("NVI > EMA (Smart Accumulation)")
            
            if c0['RVOL'] > 2.0 and (c0['High']-c0['Low']) < self.df['ATR'].iloc[-1] * 0.5:
                 score -= 2
                 res['signals'].append("VSA: Churning Detected (Distr.)")

            if score >= 2: res['status'] = "BULLISH (Accumulation)"
            elif score <= -1: res['status'] = "BEARISH (Distribution)"
            
        except: pass
        return res

    # --- ORIGINAL FEATURES (RETAINED) ---
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
            elif mcap < min_cap: res['status'] = "SMALL CAP"; res['warning'] = "Market Cap < 500B IDR."
            elif eps < 0: res['status'] = "UNPROFITABLE"; res['warning'] = "Company has negative EPS."
            else: res['status'] = "GOOD"
        except: pass
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
            if is_squeeze: res = {"detected": True, "msg": "TTM Squeeze ON! Volatility compression."}
        except: pass
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
        except: pass
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
            if is_contracting and near_breakout: return {"detected": True, "msg": f"Contraction from {depth_1*100:.1f}% to {depth_2*100:.1f}%."}
            else: return {"detected": False, "msg": "Volatility not contracting."}
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

    def detect_low_cheat(self):
        res = {"detected": False, "msg": ""}
        try:
            if self.data_len < 20: return res
            # Simple check for low cheat (Tight consolidation below pivot)
            c0 = self.df.iloc[-1]
            vol_ma = self.df['Volume'].rolling(20).mean().iloc[-1]
            atr = self.df['ATR'].iloc[-1]
            vol_dry = c0['Volume'] < vol_ma * 0.8
            spread_tight = (c0['High'] - c0['Low']) < (atr / 1.5)
            recent_high = self.df['High'].iloc[-20:].max()
            below_pivot = c0['Close'] < recent_high
            if vol_dry and spread_tight and below_pivot:
                 res = {"detected": True, "msg": "Valid Low Cheat Setup (Tight + Dry Vol)"}
        except: pass
        return res

    def backtest_fib_bounce(self):
        res = {"accuracy": "N/A", "count": 0, "verdict": "Unproven"}
        try:
            if self.data_len < 200: return res
            wins = 0
            count = 0
            lookback = self.config["FIB_LOOKBACK_DAYS"]
            for i in range(lookback + 20, self.data_len - 20, 10):
                past_df = self.df.iloc[i-lookback:i]
                future_df = self.df.iloc[i:i+15]
                sh = past_df['High'].max()
                sl = past_df['Low'].min()
                rng = sh - sl
                fib_618 = sh - (0.618 * rng)
                current_low = self.df['Low'].iloc[i]
                if abs(current_low - fib_618) / fib_618 < 0.02:
                    count += 1
                    if future_df['High'].max() > (current_low * 1.05):
                        wins += 1
            if count == 0: return res
            win_rate = (wins / count) * 100
            verdict = "GOLDEN ZONE" if win_rate > 65 else "WEAK SUPPORT" if win_rate < 40 else "NEUTRAL"
            res = {"accuracy": f"{win_rate:.1f}%", "count": count, "verdict": verdict}
        except: pass
        return res

    # --- NEW: POSITION SIZING ---
    def calculate_position_size(self, entry, stop_loss):
        balance = self.config["ACCOUNT_BALANCE"]
        risk_pct = self.config["RISK_PER_TRADE_PCT"]
        if entry <= stop_loss: return 0, 0
        risk_amt = balance * (risk_pct / 100)
        risk_per_share = entry - stop_loss
        shares = risk_amt / risk_per_share
        lots = int(shares / 100)
        max_cap = balance * 0.25 # Max 25% portfolio
        if (lots * 100 * entry) > max_cap: lots = int((max_cap / entry) / 100)
        return lots, risk_amt

    def adjust_to_tick_size(self, price):
        if price < 200: tick = 1
        elif price < 500: tick = 2
        elif price < 2000: tick = 5
        elif price < 5000: tick = 10
        else: tick = 25
        return round(price / tick) * tick

    # --- ENHANCED: TRADE PLAN WITH REASONING ---
    def calculate_trade_plan(self, ctx, trend_status, best_strategy, rect, sm_status):
        plan = {"type": "OPTIMIZED", "entry": 0, "stop_loss": 0, "take_profit": 0, "status": "WAIT", "reason": "No Signal"}
        
        atr = ctx['atr']
        current_price = ctx['price']
        
        trigger = False
        raw_sl = 0
        
        # Priority 1: Rectangle Breakout (New Feature)
        if rect['detected'] and "BREAKOUT" in rect['status']:
             plan["status"] = "EXECUTE (Box Breakout)"
             plan["reason"] = f"MOMENTUM: Price broke Box Resistance (Rp {rect['top']:,.0f}). Buyers in control."
             raw_sl = rect['top'] - atr
             trigger = True

        # Priority 2: Rectangle Support Bounce
        elif rect['detected'] and abs(current_price - rect['bottom'])/rect['bottom'] < 0.02:
             plan["status"] = "EXECUTE (Support Bounce)"
             plan["reason"] = f"VALUE: Price at Box Support (Rp {rect['bottom']:,.0f}). Low risk floor."
             raw_sl = rect['bottom'] - (atr * 0.5)
             trigger = True

        # Priority 3: Low Cheat
        elif ctx['low_cheat']['detected']:
             plan["status"] = "EARLY ENTRY (Low Cheat)"
             plan["reason"] = "VCP: Valid Low Cheat Setup (Tight Spreads + Dry Volume)."
             raw_sl = current_price - (atr * 1.5)
             trigger = True
             
        # Priority 4: Standard Trend Strategy (Original)
        elif best_strategy['is_triggered_today'] and "UPTREND" in trend_status:
             plan["status"] = "EXECUTE (Trend Follow)"
             plan["reason"] = f"STRATEGY: {best_strategy['strategy']} Triggered in Uptrend."
             raw_sl = current_price - (atr * 2.5)
             trigger = True

        # Smart Money Filter (Safety)
        if "BEARISH" in sm_status['status'] and trigger:
             plan["reason"] += " [WARNING: Smart Money is Distributing]"

        if trigger:
             plan['entry'] = self.adjust_to_tick_size(current_price)
             plan['stop_loss'] = self.adjust_to_tick_size(raw_sl)
             risk = plan['entry'] - plan['stop_loss']
             
             if risk > 0:
                 tp_3r = plan['entry'] + (risk * 3.0)
                 plan['take_profit'] = self.adjust_to_tick_size(tp_3r)
                 lots, risk_amt = self.calculate_position_size(plan['entry'], plan['stop_loss'])
                 plan['lots'] = lots
                 plan['risk_amt'] = risk_amt
             else:
                 plan['status'] = "WAIT"
                 plan['reason'] = "Risk Invalid (Stop > Entry)"
                 
        return plan

    def get_market_context(self):
        last_price = self.df['Close'].iloc[-1]
        fib_len = self.config["FIB_LOOKBACK_DAYS"]
        fib_win = self.df[-min(fib_len, self.data_len):]
        sh, sl = fib_win['High'].max(), fib_win['Low'].min()
        rng = sh - sl
        fibs = { "1.0 (Low)": sl, "0.618 (Golden)": sh-(0.618*rng), "0.5 (Half)": sh-(0.5*rng), "0.382": sh-(0.382*rng), "0.0 (High)": sh } if rng > 0 else {}
        
        atr = self.df['ATR'].iloc[-1] if 'ATR' in self.df.columns else 0
        
        # New Smart Money Analysis
        sm = self.analyze_smart_money_enhanced()
        
        return {
            "price": last_price, "fib_levels": fibs, "atr": atr,
            "smart_money": sm, # New Enhanced
            "vcp": self.detect_vcp_pattern(), 
            "geo": self.detect_geometric_patterns(),
            "low_cheat": self.detect_low_cheat(),
            "squeeze": self.detect_ttm_squeeze(),
            "fundamental": self.check_fundamentals(),
            "pivots": self.calculate_pivot_points(),
            "pattern_stats": self.backtest_pattern_reliability(),
            "fib_stats": self.backtest_fib_bounce(),
        }

    def generate_final_report(self):
        if not self.fetch_data(): return None
        self.prepare_indicators()
        self.analyze_news_sentiment()
        
        # New Feature Calls
        rect = self.detect_rectangle_pattern()
        liq = self.check_liquidity_quality()
        best_strategy = self.optimize_stock(1, 60)
        
        trend_template = self.check_trend_template()
        ctx = self.get_market_context()
        
        # Pass everything to Trade Plan
        plan = self.calculate_trade_plan(ctx, trend_template['status'], best_strategy, rect, ctx['smart_money'])

        return {
            "ticker": self.ticker, "name": self.info.get('longName', self.ticker),
            "price": ctx['price'], "sentiment": self.news_analysis, "context": ctx,
            "plan": plan, "trend_template": trend_template, "liquidity": liq,
            "rectangle": rect, "best_strategy": best_strategy
        }
