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
    "MAX_HOLD_DAYS": 60,
    "FIB_LOOKBACK_DAYS": 120,
    "RSI_PERIOD": 14,
    "RSI_LOWER": 30,
    "ATR_PERIOD": 14,
    "SL_MULTIPLIER": 2.5,
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

        self.df['EMA_20'] = self.calc_ema(self.df['Close'], 20)
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
        
        self.df['EFI'] = self.calc_force_index(self.df['Close'], self.df['Volume'], 13)
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['VWAP'] = (tp * self.df['Volume']).rolling(20).sum() / self.df['Volume'].rolling(20).sum()
        self.df['AMIHUD'] = self.calc_amihud(self.df['Close'], self.df['Volume'], 20)

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

    # --- BACKTESTERS ---
    def backtest_smart_money_predictivity(self):
        res = {"accuracy": "N/A", "avg_return": 0, "count": 0, "verdict": "Unproven", "best_horizon": 0}
        try:
            if self.data_len < 100: return res
            cond = (self.df['CMF'] > 0.05) & (self.df['MFI'] < 80) & (self.df['Close'] > self.df['VWAP'])
            signals = self.df[cond]
            if len(signals) < 5: return res 
            best_win_rate = -1
            best_stats = None
            for h in range(1, 41):
                wins, total_return, valid_signals = 0, 0, 0
                indices = signals.index
                i = 0
                while i < len(indices):
                    idx = indices[i]
                    loc = self.df.index.get_loc(idx)
                    if loc > (self.data_len - (h + 1)): 
                        i += 1; continue
                    entry = self.df['Close'].iloc[loc]
                    future_high = self.df['High'].iloc[loc+1 : loc+h+1].max()
                    if future_high > (entry * 1.02): wins += 1
                    exit_price = self.df['Close'].iloc[loc + h]
                    total_return += (exit_price - entry) / entry
                    valid_signals += 1
                    i += 5
                if valid_signals > 0:
                    wr = (wins / valid_signals) * 100
                    if wr > best_win_rate:
                        best_win_rate = wr
                        best_stats = {"accuracy": f"{wr:.1f}%", "avg_return": f"{(total_return / valid_signals) * 100:.1f}%", "count": valid_signals, "best_horizon": h}
            if best_stats:
                verdict = "POOR"
                if best_win_rate > 70: verdict = "HIGHLY PREDICTIVE"
                elif best_win_rate > 50: verdict = "MODERATE"
                best_stats["verdict"] = verdict
                res = best_stats
        except Exception: pass
        return res

    def backtest_volume_breakout_behavior(self):
        res = {"accuracy": "N/A", "avg_return_5d": 0, "count": 0, "behavior": "Unknown", "best_horizon": 0}
        try:
            if self.data_len < 100: return res
            min_liq = self.config["MIN_DAILY_VOL"]
            tx_value = self.df['Close'] * self.df['Volume']
            signals = (
                (self.df['Close'] > self.df['Open']) & 
                (self.df['Close'] > self.df['Close'].shift(1)) & 
                (self.df['Volume'] > self.df['VOL_MA']) & 
                (tx_value > min_liq)
            )
            breakout_indices = self.df.index[signals]
            if len(breakout_indices) < 5: return res
            best_win_rate = -1
            best_stats = None
            numeric_indices = [self.df.index.get_loc(i) for i in breakout_indices]
            for h in range(1, 41): 
                wins, total_return, valid_count = 0, 0, 0
                for idx in numeric_indices:
                    if idx > (self.data_len - (h + 1)): continue 
                    entry_price = self.df['Close'].iloc[idx]
                    future_price = self.df['Close'].iloc[idx + h]
                    ret = (future_price - entry_price) / entry_price
                    if ret > 0.02: wins += 1
                    total_return += ret
                    valid_count += 1
                if valid_count > 0:
                    wr = (wins / valid_count) * 100
                    if wr > best_win_rate:
                        best_win_rate = wr
                        best_stats = {"accuracy": f"{wr:.1f}%", "avg_return_5d": f"{(total_return / valid_count) * 100:.1f}%", "count": valid_count, "best_horizon": h}
            if best_stats:
                behavior = "HONEST (Trend Follower)" if best_win_rate > 60 else "FAKEOUT (Fade the Pop)" if best_win_rate < 40 else "MIXED / CHOPPY"
                best_stats["behavior"] = behavior
                res = best_stats
        except Exception: pass
        return res

    def backtest_ma_support(self, period=20):
        res = {"accuracy": "N/A", "count": 0, "verdict": "Unknown"}
        try:
            if self.data_len < 100: return res
            ma_col = f'EMA_{period}'
            if ma_col not in self.df.columns: return res
            touched_ma = (self.df['Low'] <= self.df[ma_col]) & (self.df['High'] >= self.df[ma_col])
            uptrend = self.df['Close'] > self.df['EMA_50']
            signals = touched_ma & uptrend
            if signals.sum() < 5: return res
            wins, valid_count = 0, 0
            indices = self.df.index[signals]
            numeric_indices = [self.df.index.get_loc(i) for i in indices]
            for idx in numeric_indices:
                if idx > (self.data_len - 6): continue
                entry = self.df[ma_col].iloc[idx]
                future_price = self.df['Close'].iloc[idx + 5]
                if future_price > (entry * 1.02): wins += 1
                valid_count += 1
            if valid_count == 0: return res
            win_rate = (wins / valid_count) * 100
            verdict = "STRONG SUPPORT" if win_rate > 65 else "WEAK SUPPORT"
            res = {"accuracy": f"{win_rate:.1f}%", "count": valid_count, "verdict": verdict}
        except Exception: pass
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
        except Exception: pass
        return res

    # --- HELPER FOR PATTERN DETECTION ---
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
        """
        Backtests pattern reliability. 
        """
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

    def _detect_low_cheat_on_slice(self, df_slice):
        res = {"detected": False}
        try:
            if len(df_slice) < 20: return res
            c0 = df_slice.iloc[-1]
            vol_ma = df_slice['Volume'].rolling(20).mean().iloc[-1]
            atr = df_slice['High'].rolling(14).max() - df_slice['Low'].rolling(14).min()
            vol_dry = c0['Volume'] < vol_ma * 0.8
            spread_tight = (c0['High'] - c0['Low']) < (atr / 14)
            recent_high = df_slice['High'].iloc[-20:].max()
            below_pivot = c0['Close'] < recent_high
            if vol_dry and spread_tight and below_pivot:
                res = {"detected": True}
        except: pass
        return res

    def detect_low_cheat(self):
        res = {"detected": False, "msg": ""}
        try:
            if self.data_len < 20: return res
            if self._detect_low_cheat_on_slice(self.df)["detected"]:
                 res = {"detected": True, "msg": "Valid Low Cheat Setup (Tight + Dry Vol)"}
        except Exception: pass
        return res

    def backtest_low_cheat_performance(self):
        res = {"accuracy": "N/A", "count": 0, "verdict": "Unproven"}
        try:
            if self.data_len < 100: return res
            wins = 0
            valid_count = 0
            for i in range(100, self.data_len - 20, 5):
                slice_df = self.df.iloc[:i]
                if self._detect_low_cheat_on_slice(slice_df)["detected"]:
                    valid_count += 1
                    entry = slice_df['Close'].iloc[-1]
                    future = self.df.iloc[i : i+10]
                    max_price = future['High'].max()
                    min_price = future['Low'].min()
                    if max_price > (entry * 1.03) and min_price > (entry * 0.98):
                        wins += 1
            if valid_count == 0: return res
            win_rate = (wins / valid_count) * 100
            verdict = "HIGH PROBABILITY" if win_rate > 65 else "RISKY" if win_rate < 40 else "MODERATE"
            res = {"accuracy": f"{win_rate:.1f}%", "count": valid_count, "verdict": verdict}
        except Exception: pass
        return res

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
            elif mcap < min_cap: res['status'] = "SMALL CAP (High Risk)"; res['warning'] = "Market Cap < 500B IDR. Prone to manipulation."
            elif eps < 0: res['status'] = "UNPROFITABLE"; res['warning'] = "Company has negative Earnings Per Share."
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

    def detect_volume_breakout(self):
        res = {"detected": False, "msg": ""}
        try:
            if self.data_len < 2: return res
            c0 = self.df.iloc[-1] 
            c1 = self.df.iloc[-2] 
            is_green = c0['Close'] > c0['Open']
            is_up = c0['Close'] > c1['Close']
            vol_ok = c0['Volume'] > c0['VOL_MA']
            tx_value = c0['Close'] * c0['Volume']
            min_liq = self.config["MIN_DAILY_VOL"]
            liq_ok = tx_value > min_liq
            if is_green and is_up and vol_ok and liq_ok:
                res = {"detected": True, "msg": "High Volume Accumulation Day"}
        except Exception: pass
        return res
    
    def detect_vsa_anomalies(self):
        res = {"detected": False, "msg": ""}
        try:
            if self.data_len < 2: return res
            c0 = self.df.iloc[-1]
            spread = c0['High'] - c0['Low']
            avg_spread = (self.df['High'] - self.df['Low']).rolling(20).mean().iloc[-1]
            vol_ratio = c0['RVOL'] 
            is_down_or_flat = c0['Close'] <= c0['Open']
            if is_down_or_flat and vol_ratio > 1.5 and spread < avg_spread:
                res = {"detected": True, "msg": "Stopping Volume (Absorption)"}
            elif vol_ratio > 2.0 and spread < (0.5 * avg_spread):
                res = {"detected": True, "msg": "Churning (High Effort, Low Result)"}
        except Exception: pass
        return res

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

    def calculate_probability(self, best_strategy, context, trend_template):
        base_prob = best_strategy.get('win_rate', 50)
        if base_prob < 0: base_prob = 50
        prob = base_prob
        
        if trend_template["status"] in ["PERFECT UPTREND (Stage 2)", "STRONG UPTREND"]: prob += 15
        elif trend_template["status"] == "DOWNTREND / BASE": prob -= 20
            
        if "BUYING" in context['smart_money']: prob += 10
        elif "SELLING" in context['smart_money']: prob -= 10
        
        if context['vol_breakout']['detected']: prob += 5
        if context['vsa']['detected'] and "Stopping" in context['vsa']['msg']: prob += 5
        if context['low_cheat']['detected']: prob += 10
        
        if context['vcp']['detected'] or context['geo']['pattern'] != "None": prob += 5
        if "Success" in context['pattern_stats'].get('verdict', ''): prob += 5
        if "Bullish" in context['candle']['sentiment']: prob += 5
        
        prob = max(1, min(99, prob))
        verdict = "LOW PROBABILITY"
        if prob >= 75: verdict = "HIGH PROBABILITY"
        elif prob >= 60: verdict = "MODERATE PROBABILITY"
        return {"value": prob, "verdict": verdict}

    def validate_signal(self, action, context, trend_template):
        score = 0
        reasons = []
        
        if trend_template["status"] in ["PERFECT UPTREND (Stage 2)", "STRONG UPTREND"]:
            score += 2; reasons.append("Stage 2 Uptrend (Minervini)")
        
        rvol = self.df['RVOL'].iloc[-1] if 'RVOL' in self.df.columns else 1.0
        if rvol > 1.2: score += 1; reasons.append("High Volume")
        
        if context['vol_breakout']['detected']:
             score += 2; reasons.append("Abnormal Accumulation Day")
        
        if context['low_cheat']['detected']:
             score += 2; reasons.append("Low Cheat Entry")
        
        if "BUYING" in context['smart_money']: score += 1; reasons.append("Smart Money Accumulation")
        
        if context['vsa']['detected']:
             reasons.append(f"VSA: {context['vsa']['msg']}")
             if "Absorption" in context['vsa']['msg']: score += 1

        if self.market_df is not None and len(self.market_df) > 5:
            s_ret = (self.df['Close'].iloc[-1] - self.df['Close'].iloc[-5]) / self.df['Close'].iloc[-5]
            m_ret = (self.market_df['Close'].iloc[-1] - self.market_df['Close'].iloc[-5]) / self.market_df['Close'].iloc[-5]
            if s_ret > m_ret: score += 1; reasons.append("Leader vs IHSG")
                
        if context['squeeze']['detected']: score += 2; reasons.append("TTM Squeeze Firing")

        pat_stats = context.get('pattern_stats', {})
        if "Success" in pat_stats.get('verdict', ''):
             score += 1; reasons.append(f"Historical {context['geo']['pattern']} Success")

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
            vwap = self.df['VWAP'].iloc[-1]
            
            amihud = self.df['AMIHUD'].iloc[-1] if 'AMIHUD' in self.df.columns else 0
            
            if cmf > 0.05 and last_price > vwap: money_flow = "INSTITUTIONAL BUYING"
            elif cmf < -0.05 or last_price < vwap: money_flow = "INSTITUTIONAL SELLING"
            else: money_flow = "RETAIL NOISE / Indecision"
            if vol > (2.0 * vol_ma): money_flow += " [HIGH VOLUME]"
            
            if amihud < 0.0000001 and money_flow == "RETAIL NOISE / Indecision":
                 money_flow += " (Liquid / Stealth)"

        return {
            "price": last_price, "support": support, "resistance": resistance,
            "dist_support": dist_supp, "fib_levels": fibs, "trend": trend,
            "atr": atr, "obv_status": obv_status, "smart_money": money_flow,
            "vcp": self.detect_vcp_pattern(), "geo": self.detect_geometric_patterns(),
            "candle": self.detect_candle_patterns(), "vsa": self.detect_vsa_anomalies(),
            "efi": self.df['EFI'].iloc[-1] if 'EFI' in self.df.columns else 0,
            "fundamental": self.check_fundamentals(),
            "squeeze": self.detect_ttm_squeeze(),
            "pivots": self.calculate_pivot_points(),
            "pattern_stats": self.backtest_pattern_reliability(),
            "vol_breakout": self.detect_volume_breakout(),
            "sm_predict": self.backtest_smart_money_predictivity(),
            "breakout_behavior": self.backtest_volume_breakout_behavior(),
            "lc_stats": self.backtest_low_cheat_performance(), 
            "low_cheat": self.detect_low_cheat(),
            "fib_stats": self.backtest_fib_bounce(),
            "ma_stats": self.backtest_ma_support(20)
        }

    def adjust_to_tick_size(self, price):
        if price < 200: tick = 1
        elif price < 500: tick = 2
        elif price < 2000: tick = 5
        elif price < 5000: tick = 10
        else: tick = 25
        return round(price / tick) * tick

    def calculate_trade_plan(self, plan_type, action, current_price, atr, support, resistance, best_strategy, fib_levels, pivots, trend_status, low_cheat, vol_breakout, ma_stats):
        plan = {"type": plan_type, "entry": 0, "stop_loss": 0, "take_profit": 0, "risk_reward": "N/A", "status": "ACTIVE"}
        sl_mult = self.config["SL_MULTIPLIER"]
        tp_mult = self.config["TP_MULTIPLIER"]

        if "UPTREND" not in trend_status and not low_cheat['detected']:
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
            
            if vol_breakout['detected']:
                plan['entry'] = self.adjust_to_tick_size(current_price)
                plan['status'] = "EXECUTE NOW (Momentum)"
                plan['note'] = "High Volume Accumulation"
                sl_price = self.adjust_to_tick_size(current_price - (atr * 1.5))
                plan['stop_loss'] = sl_price
                plan['take_profit'] = self.adjust_to_tick_size(current_price + (atr * tp_mult))
                risk = current_price - sl_price
                if risk > 0:
                     tp_3r = current_price + (risk * 3.0)
                     plan['take_profit_3r'] = self.adjust_to_tick_size(tp_3r)
                return plan

            if low_cheat['detected']:
                plan['entry'] = self.adjust_to_tick_size(current_price)
                plan['status'] = "EARLY ENTRY (Low Cheat)"
                plan['note'] = "Valid Low Cheat (Tight Stop)"
                sl_price = self.adjust_to_tick_size(current_price - (atr * 1.5))
                plan['stop_loss'] = sl_price
                plan['take_profit'] = self.adjust_to_tick_size(current_price + (atr * tp_mult))
                risk = current_price - sl_price
                if risk > 0:
                     tp_3r = current_price + (risk * 3.0)
                     plan['take_profit_3r'] = self.adjust_to_tick_size(tp_3r)
                return plan
            
            if "PERFECT" in trend_status and "STRONG" in ma_stats['verdict']:
                ema20 = self.df['EMA_20'].iloc[-1]
                if abs(current_price - ema20) / current_price < 0.02:
                    plan['entry'] = self.adjust_to_tick_size(current_price)
                    plan['status'] = "EXECUTE NOW (Power Trend Dip)"
                    sl_price = self.adjust_to_tick_size(ema20 - (atr * 2.0))
                    plan['stop_loss'] = sl_price
                    plan['take_profit'] = self.adjust_to_tick_size(ema20 + (atr * tp_mult))
                    risk = current_price - sl_price
                    if risk > 0: tp_3r = current_price + (risk * 3.0); plan['take_profit_3r'] = self.adjust_to_tick_size(tp_3r)
                    return plan
                elif ema20 > (current_price * 0.9):
                    plan['entry'] = self.adjust_to_tick_size(ema20)
                    plan['note'] = "Wait for EMA 20 Bounce (Power Trend)"
                    sl_price = self.adjust_to_tick_size(ema20 - (atr * 2.0))
                    plan['stop_loss'] = sl_price
                    plan['take_profit'] = self.adjust_to_tick_size(ema20 + (atr * tp_mult))
                    risk = ema20 - sl_price
                    if risk > 0: tp_3r = ema20 + (risk * 3.0); plan['take_profit_3r'] = self.adjust_to_tick_size(tp_3r)
                    plan['risk_reward'] = "Projection"
                    return plan

            strategy_type = best_strategy.get('strategy', 'None')
            target_price = support 
            
            if "RSI" in strategy_type or "Stoch" in strategy_type:
                target_fib = 0
                for _, price in sorted(fib_levels.items(), key=lambda x: x[1], reverse=True):
                    if price < current_price: target_fib = price; break
                
                if target_fib > 0 and (current_price - target_fib) / current_price < 0.02:
                    plan['entry'] = self.adjust_to_tick_size(current_price)
                    plan['status'] = "EXECUTE NOW (Near Support)"
                    sl_price = self.adjust_to_tick_size(current_price - (atr * sl_mult))
                    plan['stop_loss'] = sl_price
                    plan['take_profit'] = self.adjust_to_tick_size(current_price + (atr * tp_mult))
                    risk = current_price - sl_price
                    if risk > 0:
                         tp_3r = current_price + (risk * 3.0)
                         plan['take_profit_3r'] = self.adjust_to_tick_size(tp_3r)
                    return plan

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
        
        if ctx['vol_breakout']['detected']:
            action = "ACTION: BUY (Vol Breakout)"

        val_score, val_verdict, val_reasons = self.validate_signal(action, ctx, trend_template)
        
        prob_data = self.calculate_probability(best_strategy, ctx, trend_template)

        plan = self.calculate_trade_plan("OPTIMIZED_SWING", action, ctx['price'], ctx['atr'], ctx['support'], ctx['resistance'], best_strategy, ctx['fib_levels'], ctx['pivots'], trend_template['status'], ctx['low_cheat'], ctx['vol_breakout'], ctx['ma_stats'])

        return {
            "ticker": self.ticker, "name": self.info.get('longName', self.ticker),
            "price": ctx['price'], "sentiment": self.news_analysis, "context": ctx,
            "plans": [plan], 
            "validation": {"score": val_score, "verdict": val_verdict, "reasons": val_reasons},
            "probability": prob_data,
            "trend_template": trend_template, 
            "is_ipo": self.data_len < 200, "days_listed": self.data_len
        }


