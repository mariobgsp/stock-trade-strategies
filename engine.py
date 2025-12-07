import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from datetime import datetime, timedelta
# --- NEW: Sklearn Imports (Gradient Boosting) ---
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

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
    "SL_MULTIPLIER": 3.0, 
    "TP_MULTIPLIER": 6.0,
    "CMF_PERIOD": 20,
    "MFI_PERIOD": 14,
    "VOL_MA_PERIOD": 20,
    "MIN_MARKET_CAP": 500_000_000_000, 
    "MIN_DAILY_VOL": 1_000_000_000,
    "MIN_ADTV_IDR": 5_000_000_000,
    "ACCOUNT_BALANCE": 100_000_000,
    "RISK_PER_TRADE_PCT": 1.0
}

MA_TEST_PAIRS = [(5, 20), (20, 50), (50, 200)] 
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20
OBV_LOOKBACK_DAYS = 10
TREND_EMA_DEFAULT = 200

# Financial Dictionary for Better Sentiment
FIN_BULLISH = {
    'naik', 'lonjakan', 'rekor', 'laba', 'untung', 'dividen', 'buyback', 
    'akuisisi', 'merger', 'tumbuh', 'menguat', 'bullish', 'hijau', 
    'positif', 'tertinggi', 'cuan', 'proyeksi', 'ekspansi', 'divestasi',
    'surplus', 'pendapatan naik', 'laba bersih'
}

FIN_BEARISH = {
    'turun', 'anjlok', 'rugi', 'merugi', 'bangkrut', 'pailit', 'utang', 
    'gagal', 'batal', 'melemah', 'bearish', 'merah', 'negatif', 
    'terendah', 'boncos', 'suspen', 'gugat', 'sanksi', 'denda',
    'penggelapan', 'korupsi', 'phk', 'defisit'
}

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

            score = 0
            for title in headlines:
                t_lower = title.lower()
                bull = sum(1 for w in FIN_BULLISH if w in t_lower)
                bear = sum(1 for w in FIN_BEARISH if w in t_lower)
                score += (bull * 1.5) - (bear * 1.5)

            final_score = score / len(headlines) if headlines else 0
            sentiment = "Positive" if final_score > 0.2 else "Negative" if final_score < -0.2 else "Neutral"
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

    def calc_ad_line(self, high, low, close, volume):
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0) 
        mfv = mfm * volume
        ad_line = mfv.cumsum()
        return ad_line

    def calc_pvt(self, close, volume):
        pct_change = close.pct_change().fillna(0)
        pvt = (pct_change * volume).cumsum()
        return pvt

    def prepare_indicators(self):
        if self.df is None or self.df.empty: return

        self.df['EMA_20'] = self.calc_ema(self.df['Close'], 20)
        self.df['EMA_50'] = self.calc_ema(self.df['Close'], 50)
        self.df['EMA_100'] = self.calc_ema(self.df['Close'], 100)
        self.df['EMA_150'] = self.calc_ema(self.df['Close'], 150)
        self.df['EMA_200'] = self.calc_ema(self.df['Close'], 200)
        self.active_trend_col = 'EMA_200'

        self.df['RSI'] = self.calc_rsi(self.df['Close'], self.config["RSI_PERIOD"])
        k, d = self.calc_stoch(self.df['High'], self.df['Low'], self.df['Close'], STOCH_K_PERIOD, STOCH_D_PERIOD)
        self.df[f"STOCHk"] = k
        self.df[f"STOCHd"] = d

        self.df['OBV'] = self.calc_obv(self.df['Close'], self.df['Volume'])
        self.df['AD_Line'] = self.calc_ad_line(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'])
        self.df['PVT'] = self.calc_pvt(self.df['Close'], self.df['Volume'])
        
        self.df['CMF'] = self.calc_cmf(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], self.config["CMF_PERIOD"])
        self.df['MFI'] = self.calc_mfi(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], self.config["MFI_PERIOD"])
        self.df['VOL_MA'] = self.df['Volume'].rolling(window=self.config["VOL_MA_PERIOD"]).mean()
        self.df['RVOL'] = self.df['Volume'] / self.df['VOL_MA']
        self.df['TxValue'] = self.df['Close'] * self.df['Volume']
        self.df['EFI'] = self.calc_force_index(self.df['Close'], self.df['Volume'], 13)
        self.df['AMIHUD'] = self.calc_amihud(self.df['Close'], self.df['Volume'], 20)

        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['VWAP'] = (tp * self.df['Volume']).rolling(20).sum() / self.df['Volume'].rolling(20).sum()
        
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

        self.df['ATR'] = self.calc_atr(self.df['High'], self.df['Low'], self.df['Close'], self.config["ATR_PERIOD"])

    def check_liquidity_quality(self):
        try:
            adtv = self.df['TxValue'].rolling(20).mean().iloc[-1]
            min_adtv = self.config["MIN_ADTV_IDR"]
            if adtv < min_adtv:
                return {"status": "FAIL", "msg": f"Low Liquidity ({adtv/1e9:.2f}B IDR). High Risk.", "adtv": adtv}
            return {"status": "PASS", "msg": f"Healthy Liquidity ({adtv/1e9:.2f}B IDR)", "adtv": adtv}
        except: return {"status": "UNKNOWN", "msg": "Calc Error", "adtv": 0}

    def get_market_regime(self):
        regime = "NEUTRAL"
        if self.market_df is not None and not self.market_df.empty:
            m_close = self.market_df['Close']
            m_ma200 = m_close.rolling(200).mean().iloc[-1]
            if m_close.iloc[-1] > m_ma200:
                regime = "BULLISH"
            else:
                regime = "BEARISH"
        return regime

    def check_weekly_trend(self):
        try:
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            weekly_df = self.df.resample('W').apply(logic).dropna()
            if len(weekly_df) < 30: return "UNKNOWN"
            weekly_df['EMA_30'] = weekly_df['Close'].ewm(span=30, adjust=False).mean()
            curr = weekly_df['Close'].iloc[-1]
            ema = weekly_df['EMA_30'].iloc[-1]
            if curr > ema: return "UPTREND"
            return "DOWNTREND"
        except: return "UNKNOWN"

    def check_trend_template(self):
        res = {"status": "FAIL", "score": 0, "max_score": 6, "details": []}
        try:
            has_200 = self.data_len >= 200
            if self.data_len < 50:
                 res["details"].append("Insufficient data (Need > 50 days)")
                 return res

            curr = self.df['Close'].iloc[-1]
            ema_50 = self.df['EMA_50'].iloc[-1]
            
            regime = self.get_market_regime()
            if regime == "BEARISH":
                res["details"].append("NOTE: Market (IHSG) is Bearish. Be cautious.")

            if not has_200:
                res["status"] = "IPO / NEW LISTING"
                res["max_score"] = 3
                ath = self.df['High'].max()
                atl = self.df['Low'].min()
                c1 = curr > ema_50
                c2 = curr >= (0.75 * ath)
                c3 = curr >= (1.25 * atl)
                score = sum([c1, c2, c3])
                res["score"] = score
                if score == 3: res["status"] = "IPO POWER TREND"
                elif score >= 1: res["status"] = "IPO UPTREND"
                if c1: res["details"].append("Price > EMA 50 (Short Term Trend)")
                if c2: res["details"].append("Near All-Time Highs")
                res["details"].append(f"Note: No EMA 200 yet ({self.data_len} days listed)")
                return res

            lookback = min(self.data_len, 260)
            ema_150 = self.df['EMA_150'].iloc[-1]
            ema_200 = self.df['EMA_200'].iloc[-1]
            year_high = self.df['High'].iloc[-lookback:].max()
            year_low = self.df['Low'].iloc[-lookback:].min()
            
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
            elif score == 3: res["status"] = "WEAK UPTREND / RECOVERY"
            elif score <= 2: res["status"] = "DOWNTREND / BASE"
            
            if c1 and c2: res["details"].append("MA Alignment (Price > 150 > 200)")
            if c3: res["details"].append("200-Day MA Rising")
            if c5: res["details"].append(f"> 25% Off {lookback}-Day Lows")
            if c6: res["details"].append(f"Near {lookback}-Day Highs")
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

    # --- UPDATED: STRICT BACKTEST (SNIPER RULE > 1.5x) ---
    def backtest_volume_breakout_behavior(self):
        res = {"accuracy": "N/A", "avg_return_5d": 0, "count": 0, "behavior": "Unknown", "best_horizon": 0}
        try:
            if self.data_len < 100: return res
            min_liq = self.config["MIN_DAILY_VOL"]
            tx_value = self.df['Close'] * self.df['Volume']
            
            # --- SNIPER UPGRADE: Use 1.5x Volume Rule for Backtest too ---
            signals = (
                (self.df['Close'] > self.df['Open']) & 
                (self.df['Close'] > self.df['Close'].shift(1)) & 
                (self.df['Volume'] > 1.5 * self.df['VOL_MA']) & # Strict Rule
                (tx_value > min_liq)
            )
            
            breakout_indices = self.df.index[signals]
            if len(breakout_indices) < 3: # Relaxed min count slightly as strict rule yields fewer trades
                 return res
                 
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

    def backtest_ma_support_all(self):
        res = {"best_ma": "None", "accuracy": "N/A", "count": 0, "verdict": "Unknown", "details": {}}
        try:
            if self.data_len < 200: return res
            best_win_rate = -1
            periods = [20, 50, 100, 200]
            for p in periods:
                ma_col = f'EMA_{p}'
                if ma_col not in self.df.columns: continue
                slope = self.df[ma_col].diff(5) > 0
                touched_ma = (self.df['Low'] <= self.df[ma_col]) & (self.df['High'] >= self.df[ma_col])
                signals = touched_ma & slope
                indices = self.df.index[signals]
                numeric_indices = [self.df.index.get_loc(i) for i in indices]
                if len(numeric_indices) < 3: continue
                wins = 0
                valid_count = 0
                for idx in numeric_indices:
                    if idx > (self.data_len - 10): continue 
                    if valid_count > 0 and idx - numeric_indices[valid_count-1] < 5: continue
                    entry = self.df[ma_col].iloc[idx]
                    future_high = self.df['High'].iloc[idx+1 : idx+10].max() 
                    if future_high > (entry * 1.03): wins += 1
                    valid_count += 1
                if valid_count > 0:
                    wr = (wins / valid_count) * 100
                    res["details"][f"EMA{p}"] = f"{wr:.0f}% ({wins}/{valid_count})"
                    if wr > best_win_rate:
                        best_win_rate = wr
                        verdict = "STRONG SUPPORT" if wr > 65 else "WEAK SUPPORT" if wr < 40 else "MODERATE"
                        res["best_ma"] = f"EMA {p}"
                        res["accuracy"] = f"{wr:.1f}%"
                        res["count"] = valid_count
                        res["verdict"] = verdict
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
        
        if 'EMA_50' in self.df.columns:
             condition = self.df['EMA_50'] > self.df['EMA_200'] 
             for days in range(days_min, days_max + 1):
                wr = self.run_backtest_simulation(condition, days)
                if wr > best_res['win_rate']:
                    best_res = {"strategy": "MA Trend", "details": "Trend Following (50 > 200)", "win_rate": wr, "hold_days": days, "is_triggered_today": True}
        return best_res

    # --- NEW: DYNAMIC BEST FIT MA ---
    def find_best_dynamic_ma(self):
        best_ma = {"period": 0, "bounces": 0, "price": 0}
        try:
            close = self.df['Close']
            low = self.df['Low']
            
            for p in range(20, 201, 5):
                ema = close.ewm(span=p, adjust=False).mean()
                touches = (low <= ema * 1.01) & (low >= ema * 0.99) & (close > ema)
                bounces = touches.sum()
                
                if bounces > best_ma["bounces"]:
                    best_ma["period"] = p
                    best_ma["bounces"] = bounces
                    best_ma["price"] = ema.iloc[-1]
            
            if best_ma["bounces"] < 3: 
                best_ma = {"period": 50, "bounces": 0, "price": self.df['EMA_50'].iloc[-1]} 
                
        except: pass
        return best_ma

    # --- UPDATED: Fundamentals with Altman Z-Score ---
    def check_fundamentals(self):
        res = {"market_cap": 0, "eps": 0, "pe": 0, "roe": 0, "pbv": 0, "status": "Unknown", "warning": "", "z_score": "N/A"}
        try:
            mcap = self.info.get('marketCap', 0)
            eps = self.info.get('trailingEps', 0)
            
            res['market_cap'] = mcap
            res['eps'] = eps
            res['pe'] = self.info.get('trailingPE', 0)
            res['roe'] = self.info.get('returnOnEquity', 0)
            res['pbv'] = self.info.get('priceToBook', 0)

            # PBV Fix: Calculate manually if data is suspicious
            if (res['pbv'] > 50 or res['pbv'] == 0) and res['pe'] > 0 and res['roe'] > 0:
                res['pbv'] = res['pe'] * res['roe']

            total_debt = self.info.get('totalDebt', 0)
            total_cash = self.info.get('totalCash', 0)
            if total_debt > 0 and total_cash > 0:
                curr_ratio = self.info.get('currentRatio', 0)
                if curr_ratio < 1.0:
                    res['z_score'] = "DISTRESS ZONE (High Risk)"
                elif total_debt > (3 * total_cash):
                    res['z_score'] = "GREY ZONE (High Debt)"
                else:
                    res['z_score'] = "SAFE ZONE"

            min_cap = self.config["MIN_MARKET_CAP"]
            if mcap == 0: res['status'] = "Unknown (Data Missing)"
            elif mcap < min_cap: res['status'] = "SMALL CAP (High Risk)"; res['warning'] = "Market Cap < 500B IDR."
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

    # --- NEW: Detect Inside Bar ---
    def detect_inside_bar(self):
        res = {"detected": False, "high": 0, "low": 0, "msg": ""}
        try:
            if self.data_len < 2: return res
            
            curr = self.df.iloc[-1]
            prev = self.df.iloc[-2]
            
            # Inside Bar Logic: High < Prev High AND Low > Prev Low
            is_inside = (curr['High'] < prev['High']) and (curr['Low'] > prev['Low'])
            
            if is_inside:
                # Check volume for "Coil" effect (low volume is better)
                is_low_vol = curr['Volume'] < prev['Volume']
                msg = "Inside Bar (Coil)" if is_low_vol else "Inside Bar (Volatility Contraction)"
                
                res = {
                    "detected": True,
                    "high": prev['High'], # Breakout trigger
                    "low": prev['Low'],   # Breakdown trigger
                    "msg": msg
                }
        except: pass
        return res
    
    # --- NEW: Detect Stealth Accumulation ---
    def detect_stealth_accumulation(self):
        # Check for high volume with low price movement near recent highs
        try:
            if self.data_len < 2: return False
            c0 = self.df.iloc[-1]
            spread = c0['High'] - c0['Low']
            avg_spread = self.df['ATR'].iloc[-1]
            
            # High Volume (>1.5x) + Small Spread (<0.8x ATR) + Close in upper half
            if c0['RVOL'] > 1.5 and spread < (0.8 * avg_spread) and c0['Close'] > ((c0['High'] + c0['Low']) / 2):
                return True
        except: pass
        return False

    def analyze_smart_money_enhanced(self):
        res = {"status": "NEUTRAL", "signals": [], "metrics": {}}
        try:
            c0 = self.df.iloc[-1]
            score = 0
            
            recent = self.df.iloc[-20:]
            green_vol = recent[recent['Close'] > recent['Open']]['Volume'].sum()
            red_vol = recent[recent['Close'] < recent['Open']]['Volume'].sum()
            total_vol = green_vol + red_vol
            buy_pressure = (green_vol / total_vol) * 100 if total_vol > 0 else 50
            
            avg_vol = recent['Volume'].mean()
            if avg_vol > 0:
                spikes = recent[recent['Volume'] > 1.25 * avg_vol]
                green_spikes = len(spikes[spikes['Close'] > spikes['Open']])
                red_spikes = len(spikes[spikes['Close'] < spikes['Open']])
            else:
                green_spikes, red_spikes = 0, 0

            res['metrics'] = {'buy_pressure': buy_pressure, 'green_spikes': green_spikes, 'red_spikes': red_spikes}

            if c0['Close'] > c0['VWAP']: score += 1; res['signals'].append("Price > VWAP (Inst. Support)")
            else: score -= 1; res['signals'].append("Price < VWAP (Weakness)")

            if c0['NVI'] > c0['NVI_EMA']: score += 1; res['signals'].append("NVI > EMA (Accumulation)")
            
            if 'AD_Line' in self.df.columns and len(self.df) > 5:
                ad_slope = self.calc_slope(self.df['AD_Line'], 5)
                price_slope = self.calc_slope(self.df['Close'], 5)
                if ad_slope > 0 and price_slope < 0:
                     score += 2; res['signals'].append("BULLISH DIV: Price down, A/D Line up!")
                elif ad_slope > 0:
                     score += 1; res['signals'].append("A/D Line Rising (Buying Pressure)")

            if 'PVT' in self.df.columns and len(self.df) > 5:
                pvt_slope = self.calc_slope(self.df['PVT'], 5)
                if pvt_slope > 0: score += 1; res['signals'].append("PVT Rising (Positive Volume Trend)")
            
            # New: Stealth Accumulation Check
            if self.detect_stealth_accumulation():
                score += 2; res['signals'].append("Stealth Accumulation (High Vol, Low Spread)")

            if buy_pressure > 60:
                 score += 1; res['signals'].append(f"Buying Pressure Dominant ({buy_pressure:.0f}%)")
            elif buy_pressure < 40:
                 score -= 1; res['signals'].append(f"Selling Pressure Dominant ({100-buy_pressure:.0f}%)")

            spread = c0['High'] - c0['Low']
            avg_spread = self.df['ATR'].iloc[-1]
            is_down = c0['Close'] < c0['Open']
            is_up = c0['Close'] > c0['Open']
            
            lower_wick = min(c0['Open'], c0['Close']) - c0['Low']
            body = abs(c0['Close'] - c0['Open'])
            
            if is_down and c0['RVOL'] > 1.5 and lower_wick > (0.4 * spread):
                 score += 2; res['signals'].append("VSA: Stopping Volume (Potential Reversal)")
            
            if is_down and c0['RVOL'] > 1.5 and body < (0.3 * spread):
                 score += 2; res['signals'].append("Wyckoff: Effort vs Result (Bullish)")

            if is_up and c0['RVOL'] < 0.7:
                 score -= 1; res['signals'].append("VSA: No Demand (Weak Rally)")

            if score >= 2: res['status'] = "BULLISH (Accumulation)"
            elif score <= -2: res['status'] = "BEARISH (Distribution)"
        except: pass
        return res

    def calculate_position_size(self, entry, stop_loss):
        balance = self.config["ACCOUNT_BALANCE"]
        risk_pct = self.config["RISK_PER_TRADE_PCT"]
        if entry <= stop_loss: return 0, 0
        risk_amt = balance * (risk_pct / 100)
        risk_per_share = entry - stop_loss
        shares = risk_amt / risk_per_share
        lots = int(shares / 100)
        max_cap = balance * 0.25
        if (lots * 100 * entry) > max_cap: lots = int((max_cap / entry) / 100)
        return lots, risk_amt

    def adjust_to_tick_size(self, price):
        if price < 200: tick = 1
        elif price < 500: tick = 2
        elif price < 2000: tick = 5
        elif price < 5000: tick = 10
        else: tick = 25
        return round(price / tick) * tick

    # --- UPDATED: Sniper Edition Trade Plan with Swing Adjustments & RSI/Stoch ---
    def calculate_trade_plan_hybrid(self, ctx, trend_status, best_strategy):
        # Default Plan
        plan = {
            "type": "HYBRID", 
            "entry": 0, 
            "stop_loss": 0, 
            "take_profit": 0, 
            "status": "WAIT", 
            "reason": "No Signal"
        }
        
        atr = ctx['atr']
        current_price = ctx['price']
        
        # Container for valid setups
        valid_setups = []
        
        # 0. NEW PRIORITY: SNIPER RSI+STOCH
        try:
            curr_rsi = self.df['RSI'].iloc[-1]
            prev_rsi = self.df['RSI'].iloc[-2]
            curr_k = self.df['STOCHk'].iloc[-1]
            prev_k = self.df['STOCHk'].iloc[-2]
            curr_d = self.df['STOCHd'].iloc[-1]
            
            # Logic: RSI healthy (>50 or rising from >40) AND Stoch crossing up from oversold
            rsi_ok = (curr_rsi > 50) or (curr_rsi > 40 and curr_rsi > prev_rsi)
            stoch_buy = (prev_k < 20) and (curr_k > prev_k) and (curr_k > curr_d)
            
            if rsi_ok and stoch_buy:
                valid_setups.append({
                    "reason": "SNIPER: RSI Trend + Stoch Reversal",
                    "entry": current_price,
                    "sl": current_price - (atr * 2.0),
                    "type": "SNIPER_MOMENTUM"
                })
        except: pass

        # 0.5. NEW: Inside Bar Breakout
        ib = ctx.get('inside_bar', {})
        if ib.get('detected'):
             valid_setups.append({
                "reason": f"PATTERN: Inside Bar Coil (Breakout > {ib['high']})",
                "entry": ib['high'],
                "sl": ib['low'],
                "type": "INSIDE_BAR"
            })

        # 3. Check Low Cheat (VCP)
        if ctx['low_cheat']['detected']:
            valid_setups.append({
                "reason": "VCP: Valid Low Cheat Setup",
                "entry": current_price,
                "sl": current_price - (atr * 2.0), # Swing: Wider SL
                "type": "EARLY_ENTRY"
            })
             
        # 4. Check Standard Trend Strategy (Golden Confluence)
        if best_strategy['is_triggered_today'] and "STRONG" in trend_status and "BULLISH" in ctx['smart_money']['status']:
            valid_setups.append({
                "reason": f"CONFLUENCE: {best_strategy['strategy']} + Smart Money + Strong Trend",
                "entry": current_price,
                "sl": current_price - (atr * 3.0), # Swing: Wide SL for Trend Following
                "type": "TREND"
            })
        
        # 5. Dynamic MA Support
        best_ma = ctx.get('best_ma', {})
        if best_ma.get('bounces', 0) >= 3:
             ma_price = best_ma['price']
             if abs(current_price - ma_price) / ma_price < 0.015:
                 valid_setups.append({
                    "reason": f"SNIPER: Bounce off Dynamic EMA {best_ma['period']}",
                    "entry": ma_price,
                    "sl": ma_price - (atr * 1.5),
                    "type": "DYNAMIC_MA"
                })

        # --- DECISION LOGIC ---
        if not valid_setups:
            plan["reason"] = "No valid entry. Wait for support or breakout."
            return plan

        primary_setup = valid_setups[0]
        combined_reasons = [s['reason'] for s in valid_setups]
        final_reason = " + ".join(combined_reasons)
        
        # Calculate Entry
        entry_price = self.adjust_to_tick_size(primary_setup['entry'])
        
        # Pullback Logic: If entry is breakout, suggest Limit slightly below
        if primary_setup['type'] == "BREAKOUT":
             entry_price = self.adjust_to_tick_size(primary_setup['entry'] * 0.995) # 0.5% discount
             final_reason += " (Limit Order)"

        if current_price > entry_price and primary_setup['type'] != "BREAKOUT" and primary_setup['type'] != "INSIDE_BAR":
            entry_price = current_price

        stop_loss = self.adjust_to_tick_size(primary_setup['sl'])
        if stop_loss >= entry_price:
            stop_loss = self.adjust_to_tick_size(entry_price - atr)

        risk = entry_price - stop_loss
        
        if risk > 0:
            plan['status'] = "EXECUTE"
            plan['reason'] = final_reason
            plan['entry'] = entry_price
            plan['stop_loss'] = stop_loss
            plan['take_profit'] = self.adjust_to_tick_size(entry_price + (risk * self.config["TP_MULTIPLIER"])) # Use Swing Target
            
            lots, risk_amt = self.calculate_position_size(plan['entry'], plan['stop_loss'])
            plan['lots'] = lots
            plan['risk_amt'] = risk_amt
        else:
             plan['status'] = "WAIT"
             plan['reason'] = "Risk Invalid (Stop > Entry)"

        return plan

    def calculate_probability(self, best_strategy, context, trend_template):
        base_prob = best_strategy.get('win_rate', 50)
        if base_prob < 0: base_prob = 50
        
        # Use sigmoid-like dampening to prevent overconfidence
        # Base: 50% -> Max movement from signals capped at +/- 40%
        signal_score = 0
        
        if trend_template["status"] in ["PERFECT UPTREND (Stage 2)", "STRONG UPTREND"]: signal_score += 15
        elif trend_template["status"] == "DOWNTREND / BASE": signal_score -= 20
        
        if "BUYING" in context['smart_money']: signal_score += 10
        elif "SELLING" in context['smart_money']: signal_score -= 10
        
        if context['vol_breakout']['detected']: signal_score += 5
        if context['vsa']['detected'] and "Stopping" in context['vsa']['msg']: signal_score += 5
        if context['low_cheat']['detected']: signal_score += 10
        if context['vcp']['detected']: signal_score += 5
        if "Bullish" in context['candle']['sentiment']: signal_score += 5

        # Blend historical win rate with current signal strength
        # Weight: 40% History, 60% Current Signal
        final_prob = (base_prob * 0.4) + (50 + signal_score) * 0.6
        
        # Hard caps
        final_prob = max(10, min(90, final_prob))
        
        verdict = "LOW PROBABILITY"
        if final_prob >= 75: verdict = "HIGH PROBABILITY"
        elif final_prob >= 60: verdict = "MODERATE PROBABILITY"
        
        return {"value": round(final_prob, 1), "verdict": verdict}

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
        
        verdict = "WEAK"
        if score >= 5: verdict = "ELITE SWING SETUP"
        elif score >= 3: verdict = "MODERATE"
        return score, verdict, reasons
    
    # --- NEW: MACHINE LEARNING MODULE 1 (PRICE PREDICTION) ---
    def predict_machine_learning(self):
        try:
            if self.data_len < 100: return {"confidence": 0.0, "prediction": "N/A", "msg": "Insufficient Data for AI"}
            
            # 1. Feature Engineering
            ml_df = self.df.copy()
            ml_df['Target'] = (ml_df['Close'].shift(-5) > ml_df['Close']).astype(int) # Predict if price is higher in 5 days
            
            # Use safe features (Oscillators, Ratios) - NOT raw prices
            features = ['RSI', 'CMF', 'MFI', 'ATR', 'RVOL', 'Close', 'EMA_50']
            
            # Normalize Price vs EMA
            ml_df['Dist_EMA50'] = (ml_df['Close'] - ml_df['EMA_50']) / ml_df['EMA_50']
            ml_df['Price_Change'] = ml_df['Close'].pct_change()
            
            model_features = ['RSI', 'CMF', 'MFI', 'RVOL', 'Dist_EMA50', 'Price_Change']
            ml_df = ml_df.dropna()
            
            if len(ml_df) < 50: return {"confidence": 0.0, "prediction": "N/A", "msg": "Data clean failed"}

            X = ml_df[model_features].iloc[:-5] # All data except last 5 (no target)
            y = ml_df['Target'].iloc[:-5]
            
            # 2. Train Model (Gradient Boosting Classifier)
            # Replaced RandomForestClassifier with GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            clf.fit(X, y)
            
            # 3. Predict on Current Data
            last_row = ml_df[model_features].iloc[-1:].values
            prediction = clf.predict(last_row)[0]
            probability = clf.predict_proba(last_row)[0][1] # Probability of Class 1 (Up)
            
            conf_pct = probability * 100
            sentiment = "BULLISH" if conf_pct > 50 else "BEARISH"
            
            return {
                "confidence": conf_pct,
                "prediction": sentiment,
                "msg": f"AI forecasts {sentiment} movement (5 Days)"
            }
        except Exception as e:
            return {"confidence": 0.0, "prediction": "Error", "msg": str(e)}

    # --- NEW: MACHINE LEARNING MODULE 2 (BANDAR/ACCUMULATION DETECTOR) ---
    def predict_bandar_accumulation(self):
        try:
            if self.data_len < 150: return {"probability": 0, "verdict": "Insufficient Data"}
            
            # 1. Define Accumulation (Target)
            # Accumulation is successful if Price rises > 5% in next 10 days WITHOUT dropping > 2% first
            ml_df = self.df.copy()
            
            future_high = ml_df['High'].shift(-10).rolling(10).max()
            future_low = ml_df['Low'].shift(-10).rolling(10).min()
            
            # Target: 1 if Pumped, 0 if not
            ml_df['Target'] = ((future_high > ml_df['Close'] * 1.05) & (future_low > ml_df['Close'] * 0.98)).astype(int)
            
            # 2. Features for Smart Money (Volume & Money Flow focused)
            ml_df['CMF'] = self.calc_cmf(ml_df['High'], ml_df['Low'], ml_df['Close'], ml_df['Volume'], 20)
            ml_df['RVOL'] = ml_df['Volume'] / ml_df['Volume'].rolling(20).mean()
            ml_df['NVI_Change'] = ml_df['NVI'].pct_change(5)
            ml_df['VWAP_Ratio'] = ml_df['Close'] / ml_df['VWAP']
            ml_df['Body_Size'] = (abs(ml_df['Close'] - ml_df['Open']) / (ml_df['High'] - ml_df['Low'])).fillna(0)
            
            features = ['CMF', 'RVOL', 'NVI_Change', 'VWAP_Ratio', 'Body_Size', 'MFI']
            ml_df = ml_df.dropna()
            
            if len(ml_df) < 100: return {"probability": 0, "verdict": "Data Error"}

            X = ml_df[features].iloc[:-10]
            y = ml_df['Target'].iloc[:-10]
            
            # 3. Train Gradient Boosting
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
            clf.fit(X, y)
            
            # 4. Predict
            last_row = ml_df[features].iloc[-1:].values
            prob = clf.predict_proba(last_row)[0][1] * 100
            
            verdict = "DETECTED" if prob > 65 else "POSSIBLE" if prob > 40 else "NONE"
            
            return {
                "probability": prob,
                "verdict": verdict,
                "msg": f"Accumulation Probability: {prob:.1f}%"
            }
            
        except: return {"probability": 0, "verdict": "Error"}

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
        
        atr = self.df['ATR'].iloc[-1] if 'ATR' in self.df.columns else 0

        change_pct = 0
        if len(self.df) >= 2:
            c0 = self.df['Close'].iloc[-1]
            c1 = self.df['Close'].iloc[-2]
            change_pct = ((c0 - c1) / c1) * 100
        
        ma_values = {
            "EMA_20": self.df['EMA_20'].iloc[-1] if 'EMA_20' in self.df.columns else 0,
            "EMA_50": self.df['EMA_50'].iloc[-1] if 'EMA_50' in self.df.columns else 0,
            "EMA_150": self.df['EMA_150'].iloc[-1] if 'EMA_150' in self.df.columns else 0,
            "EMA_200": self.df['EMA_200'].iloc[-1] if 'EMA_200' in self.df.columns else 0,
        }
        
        sm = self.analyze_smart_money_enhanced()
        weekly_trend = self.check_weekly_trend() # Sniper: Weekly Trend Check
        
        # New: Dynamic MA
        best_ma = self.find_best_dynamic_ma()

        # New: Detect Inside Bar
        inside_bar = self.detect_inside_bar()

        return {
            "price": last_price, "change_pct": change_pct, 
            "support": support, "resistance": resistance,
            "dist_support": dist_supp, "fib_levels": fibs, 
            "atr": atr, "smart_money": sm, "weekly_trend": weekly_trend,
            "best_ma": best_ma, # Added dynamic MA
            "inside_bar": inside_bar, # Added inside bar
            "vcp": self.detect_vcp_pattern(), 
            "candle": self.detect_candle_patterns(), 
            "vsa": self.detect_vsa_anomalies(),
            "low_cheat": self.detect_low_cheat(),
            "squeeze": self.detect_ttm_squeeze(),
            "fundamental": self.check_fundamentals(),
            "pivots": self.calculate_pivot_points(),
            "vol_breakout": self.detect_volume_breakout(),
            "sm_predict": self.backtest_smart_money_predictivity(),
            "breakout_behavior": self.backtest_volume_breakout_behavior(),
            "lc_stats": self.backtest_low_cheat_performance(), 
            "fib_stats": self.backtest_fib_bounce(),
            "ma_stats": self.backtest_ma_support_all(),
            "ml_prediction": self.predict_machine_learning(), # General Price Prediction
            "bandar_ml": self.predict_bandar_accumulation() # --- NEW: Bandar Accumulation Prediction
        }

    def generate_final_report(self):
        if not self.fetch_data(): return None
        self.prepare_indicators()
        self.analyze_news_sentiment()
        
        liq = self.check_liquidity_quality()
        best_strategy = self.optimize_stock(1, 60)
        
        trend_template = self.check_trend_template()
        ctx = self.get_market_context()
        
        action = "WAIT"
        if best_strategy['is_triggered_today'] and trend_template['score'] >= 4: action = "ACTION: BUY"
        val_score, val_verdict, val_reasons = self.validate_signal(action, ctx, trend_template)
        prob_data = self.calculate_probability(best_strategy, ctx, trend_template)

        plan = self.calculate_trade_plan_hybrid(ctx, trend_template['status'], best_strategy)

        return {
            "ticker": self.ticker, "name": self.info.get('longName', self.ticker),
            "price": ctx['price'], "change_pct": ctx['change_pct'], 
            "sentiment": self.news_analysis, "context": ctx,
            "plan": plan, 
            "validation": {"score": val_score, "verdict": val_verdict, "reasons": val_reasons},
            "probability": prob_data,
            "trend_template": trend_template, "liquidity": liq,
            "best_strategy": best_strategy,
            "is_ipo": self.data_len < 200, "days_listed": self.data_len
        }