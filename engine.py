import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ==========================================
# 1. DEFAULT CONFIGURATION
# ==========================================
DEFAULT_CONFIG = {
    "BACKTEST_PERIOD": "2y",
    "RSI_PERIOD": 14,
    "ATR_PERIOD": 14,
    "SL_MULTIPLIER": 2.5,
    "TP_MULTIPLIER": 3.0,
    "MIN_ADTV_IDR": 5_000_000_000, 
    "ACCOUNT_BALANCE": 100_000_000,
    "RISK_PER_TRADE_PCT": 1.0
}

FIN_BULLISH = {'surge', 'jump', 'soar', 'record', 'profit', 'dividend', 'buyback', 'growth', 'beat', 'bull', 'merger', 'acquisition', 'climb', 'gain', 'net income'}
FIN_BEARISH = {'plunge', 'drop', 'slump', 'loss', 'miss', 'debt', 'bankrupt', 'sue', 'lawsuit', 'investigation', 'bear', 'cut', 'down', 'fail', 'weak', 'inflation'}

class StockAnalyzer:
    def __init__(self, ticker, user_config=None):
        self.ticker = self._format_ticker(ticker)
        self.df = None
        self.info = {}
        self.news_analysis = {"sentiment": "Neutral", "score": 0, "headlines": []}
        self.config = DEFAULT_CONFIG.copy()
        if user_config:
            self.config.update(user_config)
        self.data_len = 0

    def _format_ticker(self, ticker):
        ticker = ticker.upper().strip()
        if not ticker.endswith(".JK") and not ticker.startswith("^"):
            ticker += ".JK"
        return ticker

    def fetch_data(self):
        try:
            self.df = yf.download(self.ticker, period=self.config["BACKTEST_PERIOD"], progress=False, auto_adjust=True, multi_level_index=False)
            if self.df.empty: return False
            self.data_len = len(self.df)
            ticker_obj = yf.Ticker(self.ticker)
            try:
                self.info = ticker_obj.info
                if 'longName' not in self.info: self.info['longName'] = self.ticker
            except: self.info['longName'] = self.ticker
            return True
        except Exception as e:
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
        except:
            self.news_analysis = {"sentiment": "Error", "score": 0, "headlines": []}

    def prepare_indicators(self):
        if self.df is None or self.df.empty: return
        self.df['EMA_50'] = self.df['Close'].ewm(span=50, adjust=False).mean()
        self.df['EMA_150'] = self.df['Close'].ewm(span=150, adjust=False).mean()
        self.df['EMA_200'] = self.df['Close'].ewm(span=200, adjust=False).mean()
        self.df['TxValue'] = self.df['Close'] * self.df['Volume']
        
        # VWAP
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['VWAP_20'] = (tp * self.df['Volume']).rolling(20).sum() / self.df['Volume'].rolling(20).sum()
        
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

        # ATR
        high_low = self.df['High'] - self.df['Low']
        high_close = (self.df['High'] - self.df['Close'].shift(1)).abs()
        low_close = (self.df['Low'] - self.df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=self.config["ATR_PERIOD"]).mean()

    def check_liquidity_quality(self):
        try:
            adtv = self.df['TxValue'].rolling(20).mean().iloc[-1]
            min_adtv = self.config["MIN_ADTV_IDR"]
            if adtv < min_adtv:
                return {"status": "FAIL", "msg": f"Low Liquidity ({adtv/1e9:.2f}B IDR)", "adtv": adtv}
            return {"status": "PASS", "msg": f"Healthy Liquidity ({adtv/1e9:.2f}B IDR)", "adtv": adtv}
        except: return {"status": "UNKNOWN", "msg": "Calc Error", "adtv": 0}

    def analyze_smart_money_slice(self, df_slice):
        # Helper for backtest to analyze a slice of data
        res = "NEUTRAL"
        try:
            c0 = df_slice.iloc[-1]
            score = 0
            if c0['Close'] > c0['VWAP_20']: score += 1
            else: score -= 1
            if c0['NVI'] > c0['NVI_EMA']: score += 1
            
            if score >= 1: res = "BULLISH"
            elif score <= -1: res = "BEARISH"
        except: pass
        return res

    def analyze_smart_money(self):
        # Live analysis wrapper
        res = {"status": "NEUTRAL", "signals": []}
        try:
            status = self.analyze_smart_money_slice(self.df)
            res['status'] = status
            
            c0 = self.df.iloc[-1]
            if c0['Close'] > c0['VWAP_20']: res['signals'].append("Price > VWAP (Inst. Support)")
            if c0['NVI'] > c0['NVI_EMA']: res['signals'].append("NVI Rising (Accumulation)")
            if status == "BEARISH": res['signals'].append("Weak Institutional Flows")
        except: pass
        return res

    def check_trend_template(self):
        res = {"status": "FAIL", "score": 0, "details": []}
        try:
            if self.data_len < 200: return res
            curr = self.df['Close'].iloc[-1]
            ema_50 = self.df['EMA_50'].iloc[-1]
            ema_150 = self.df['EMA_150'].iloc[-1]
            ema_200 = self.df['EMA_200'].iloc[-1]
            year_high = self.df['High'].iloc[-250:].max()
            year_low = self.df['Low'].iloc[-250:].min()
            
            c1 = curr > ema_150 and curr > ema_200
            c2 = ema_150 > ema_200
            c3 = self.df['EMA_200'].iloc[-1] > self.df['EMA_200'].iloc[-20]
            c4 = curr > ema_50
            c5 = curr >= (1.25 * year_low)
            c6 = curr >= (0.75 * year_high)
            
            score = sum([c1, c2, c3, c4, c5, c6])
            res["score"] = score
            if score == 6: res["status"] = "PERFECT UPTREND"
            elif score >= 4: res["status"] = "STRONG UPTREND"
            else: res["status"] = "WEAK / DOWNTREND"
        except Exception as e: res["details"].append(str(e))
        return res

    def detect_rectangle_pattern_slice(self, df_slice):
        # Helper for backtest to detect pattern on a slice
        res = {"detected": False, "top": 0, "bottom": 0, "status": "None"}
        try:
            if len(df_slice) < 60: return res
            window = df_slice.iloc[-60:].copy()
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
            
            curr = df_slice['Close'].iloc[-1]
            height_pct = (best_top - best_bot) / best_bot
            
            if 0.03 < height_pct < 0.25 and curr > (best_bot * 0.98):
                status = "INSIDE"
                if curr > best_top: status = "BREAKOUT"
                res = {"detected": True, "top": best_top, "bottom": best_bot, "status": status}
        except: pass
        return res

    def detect_rectangle_pattern(self):
        # Wrapper for live usage
        raw = self.detect_rectangle_pattern_slice(self.df)
        msg = "No Pattern"
        if raw['detected']:
            height = (raw['top'] - raw['bottom']) / raw['bottom']
            msg = f"Box ({raw['bottom']:,.0f}-{raw['top']:,.0f})"
            return {**raw, "height_pct": height * 100, "msg": msg}
        return {**raw, "height_pct": 0, "msg": msg}

    def run_backtest(self):
        # HISTORICAL SIMULATION
        trades = []
        open_trade = None
        
        try:
            # Start after 200 days to allow EMAs to stabilize
            start_idx = 200 
            if self.data_len < start_idx + 10: 
                return {"count": 0, "win_rate": 0, "profit_factor": 0, "equity_curve": []}

            # Loop through history
            for i in range(start_idx, self.data_len):
                current_slice = self.df.iloc[:i+1]
                curr_bar = self.df.iloc[i]
                curr_date = self.df.index[i]
                
                # Check Open Trade
                if open_trade:
                    # Check Exit
                    if curr_bar['Low'] <= open_trade['sl']:
                        # Stop Loss Hit
                        open_trade['exit_price'] = open_trade['sl']
                        open_trade['result'] = 'LOSS'
                        open_trade['pnl_pct'] = (open_trade['sl'] - open_trade['entry']) / open_trade['entry']
                        trades.append(open_trade)
                        open_trade = None
                    elif curr_bar['High'] >= open_trade['tp']:
                        # Target Hit
                        open_trade['exit_price'] = open_trade['tp']
                        open_trade['result'] = 'WIN'
                        open_trade['pnl_pct'] = (open_trade['tp'] - open_trade['entry']) / open_trade['entry']
                        trades.append(open_trade)
                        open_trade = None
                    elif (i - open_trade['idx']) > 60:
                        # Time Stop (Held too long)
                        open_trade['exit_price'] = curr_bar['Close']
                        open_trade['result'] = 'TIMEOUT'
                        open_trade['pnl_pct'] = (curr_bar['Close'] - open_trade['entry']) / open_trade['entry']
                        trades.append(open_trade)
                        open_trade = None
                    continue # Skip entry logic if in trade

                # Entry Logic (Optimized)
                # 1. Quick Trend Check
                if curr_bar['Close'] < curr_bar['EMA_200']: continue
                if curr_bar['EMA_50'] < curr_bar['EMA_200']: continue
                
                # 2. Pattern Check (Heavy)
                rect = self.detect_rectangle_pattern_slice(current_slice)
                if not rect['detected'] or rect['status'] != "BREAKOUT": continue
                
                # 3. Smart Money Check
                sm_status = self.analyze_smart_money_slice(current_slice)
                if sm_status == "BEARISH": continue # Filter bad trades
                
                # 4. Liquidity Check (Historical)
                adtv = (current_slice['TxValue'].tail(20).mean())
                if adtv < self.config["MIN_ADTV_IDR"]: continue

                # EXECUTE ENTRY
                atr = curr_bar['ATR']
                entry_price = curr_bar['Close']
                sl_price = rect['top'] - atr # Stop below breakout level
                tp_price = entry_price + ((entry_price - sl_price) * 3.0) # 3R Target
                
                open_trade = {
                    "date": curr_date,
                    "idx": i,
                    "entry": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "result": "OPEN"
                }

        except Exception as e: print(f"BT Error: {e}")
        
        # Calculate Metrics
        wins = len([t for t in trades if t['result'] == 'WIN'])
        total = len(trades)
        win_rate = (wins/total * 100) if total > 0 else 0
        total_pnl = sum([t['pnl_pct'] for t in trades])
        
        return {
            "count": total,
            "wins": wins,
            "win_rate": win_rate,
            "total_return": total_pnl * 100
        }

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

    def adjust_tick(self, price):
        if price < 200: tick = 1
        elif price < 500: tick = 2
        elif price < 2000: tick = 5
        elif price < 5000: tick = 10
        else: tick = 25
        return round(price / tick) * tick

    def calculate_trade_plan(self, current_price, atr, trend_status, rect, sm_status):
        plan = {"status": "WAIT", "entry": 0, "stop_loss": 0, "take_profit": 0, "lots": 0, "reason": "No Valid Setup"}
        
        is_uptrend = "UPTREND" in trend_status
        is_breakout = rect['detected'] and "BREAKOUT" in rect['status']
        is_support = rect['detected'] and abs(current_price - rect['bottom'])/rect['bottom'] < 0.02
        
        if "BEARISH" in sm_status['status']:
             plan["reason"] = "Smart Money is Distributing (Selling). Risky."
             return plan

        trigger = False
        raw_sl = 0
        
        if is_breakout:
            plan["status"] = "EXECUTE (Breakout)"
            plan["reason"] = f"MOMENTUM: Box Breakout + {sm_status['status']} Flow."
            trigger = True
            raw_sl = rect['top'] - atr 
            
        elif is_support:
            plan["status"] = "EXECUTE (Support Bounce)"
            plan["reason"] = f"VALUE: Box Support Floor + {sm_status['status']} Flow."
            trigger = True
            raw_sl = rect['bottom'] - (atr * 0.5)
            
        elif is_uptrend:
            plan["status"] = "EXECUTE (Trend Follow)"
            plan["reason"] = "TREND: Minervini Stage 2 Dip."
            trigger = True
            raw_sl = current_price - (atr * self.config["SL_MULTIPLIER"])
            
        if trigger:
            plan["entry"] = self.adjust_tick(current_price)
            plan["stop_loss"] = self.adjust_tick(raw_sl)
            risk = plan["entry"] - plan["stop_loss"]
            
            if risk > 0:
                plan["take_profit"] = self.adjust_tick(plan["entry"] + (risk * 3.0))
                lots, risk_amt = self.calculate_position_size(plan["entry"], plan["stop_loss"])
                plan["lots"] = lots
                plan["risk_amt"] = risk_amt
            else:
                plan["status"] = "WAIT"
                plan["reason"] = "Risk Invalid (Stop Loss > Entry)"

        return plan

    def generate_report(self):
        if not self.fetch_data(): return None
        self.prepare_indicators()
        self.analyze_news_sentiment()
        
        trend = self.check_trend_template()
        liq = self.check_liquidity_quality()
        rect = self.detect_rectangle_pattern()
        sm = self.analyze_smart_money()
        bt_results = self.run_backtest() # <--- Run Backtest
        
        atr = self.df['ATR'].iloc[-1] if 'ATR' in self.df.columns else 0
        curr = self.df['Close'].iloc[-1]
        
        plan = self.calculate_trade_plan(curr, atr, trend['status'], rect, sm)
        
        return {
            "ticker": self.ticker, "name": self.info.get('longName', self.ticker),
            "price": curr, "trend": trend, "liquidity": liq, "rectangle": rect,
            "smart_money": sm, "plan": plan, "sentiment": self.news_analysis,
            "backtest": bt_results # <--- Return Results
        }
