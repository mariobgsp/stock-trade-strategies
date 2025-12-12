import sys
import time
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class StockAnalyzer:
    def __init__(self, ticker):
        self.clean_ticker = ticker.upper()
        if not self.clean_ticker.endswith(".JK") and len(self.clean_ticker) == 4:
            self.ticker = f"{self.clean_ticker}.JK"
        else:
            self.ticker = self.clean_ticker

    # --- MATH HELPERS ---
    def calculate_sma(self, series, length):
        return series.rolling(window=length).mean()

    def calculate_ema(self, series, length):
        return series.ewm(span=length, adjust=False).mean()

    def calculate_rsi(self, series, length=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
        avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_stoch(self, high, low, close, k=14, d=3):
        lowest_low = low.rolling(window=k).min()
        highest_high = high.rolling(window=k).max()
        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_line = k_line.rolling(window=d).mean()
        return k_line, d_line

    def calculate_ad(self, high, low, close, volume, open_):
        high_low_diff = high - low
        high_low_diff = high_low_diff.replace(0, np.nan)
        mfm = ((close - low) - (high - close)) / high_low_diff
        mfm = mfm.fillna(0)
        return (mfm * volume).cumsum()

    def calculate_obv(self, close, volume):
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        obv = (volume * direction).cumsum()
        return obv.fillna(0)

    # --- DATA FETCHING ---
    def fetch_data(self, interval="1d", period="2y"):
        time.sleep(2) # Anti-ban delay
        try:
            stock = yf.Ticker(self.ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                time.sleep(1)
                stock = yf.Ticker(self.clean_ticker)
                df = stock.history(period=period, interval=interval)
            
            if df.empty: return None

            df.reset_index(inplace=True)
            df.columns = [c.lower() for c in df.columns]
            df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
            
            df['ad'] = self.calculate_ad(df['high'], df['low'], df['close'], df['volume'], df['open'])
            df['obv'] = self.calculate_obv(df['close'], df['volume'])
            return df
        except: return None

    # --- ANALYSIS ---
    def detect_trend(self, df):
        if len(df) < 50: return "No Data"
        if len(df) > 200: fast, slow = 50, 200
        else: fast, slow = 10, 20 

        sma_fast = self.calculate_sma(df['close'], fast).iloc[-1]
        sma_slow = self.calculate_sma(df['close'], slow).iloc[-1]
        price = df['close'].iloc[-1]

        if price > sma_fast > sma_slow: return "UPTREND 🚀"
        elif price < sma_fast < sma_slow: return "DOWNTREND 🔻"
        elif sma_fast > sma_slow and price < sma_fast: return "CORRECTION ⚠️"
        elif sma_fast < sma_slow and price > sma_fast: return "REVERSAL? 🔄"
        else: return "SIDEWAYS 〰️"

    def optimize_rsi(self, df):
        best_rsi = (14, -999) 
        for p in range(7, 25):
            d = df.copy()
            d['rsi'] = self.calculate_rsi(d['close'], length=p)
            d['sig'] = np.where(d['rsi'] < 30, 1, np.where(d['rsi'] > 70, -1, 0))
            d['ret'] = d['close'].pct_change().shift(-1) * d['sig']
            prof = d['ret'].sum()
            if prof > best_rsi[1]: best_rsi = (p, prof)
        return best_rsi[0]

    def optimize_stoch(self, df):
        best_stoch = (14, 3, -999) 
        for k in range(5, 22, 3): 
            for d in [3, 5]:
                try:
                    df_c = df.copy()
                    k_line, _ = self.calculate_stoch(df_c['high'], df_c['low'], df_c['close'], k, d)
                    df_c['k'] = k_line
                    df_c['sig'] = np.where(df_c['k'] < 20, 1, np.where(df_c['k'] > 80, -1, 0))
                    df_c['ret'] = df_c['close'].pct_change().shift(-1) * df_c['sig']
                    prof = df_c['ret'].sum()
                    if prof > best_stoch[2]: best_stoch = (k, d, prof)
                except: continue
        return best_stoch[0], best_stoch[1] 

    def format_volume(self, num):
        if num >= 1_000_000_000: return f"{num/1_000_000_000:.1f}B"
        if num >= 1_000_000: return f"{num/1_000_000:.1f}M"
        if num >= 1_000: return f"{num/1_000:.1f}K"
        return str(int(num))

    # --- MODE 1: DETAILED HISTORY ---
    def get_detailed_bandar_flow(self, df):
        if len(df) < 80: return [] 
        subset_indices = df.index[-60:] 
        history = []
        for idx in subset_indices:
            loc = df.index.get_loc(idx)
            
            # Logic
            window = df.iloc[loc-19 : loc+1]
            roll_vol = window['volume'].sum()
            roll_pv = (window['close'] * window['volume']).sum()
            vwap = roll_pv / roll_vol if roll_vol > 0 else 0
            curr_close = df['close'].iloc[loc]
            diff = ((curr_close - vwap) / vwap) * 100
            
            if diff < -2: status = "Accumulating"
            elif -2 <= diff <= 5: status = "Absorbing"
            else: status = "Distributing"
            
            reg_window = df.iloc[loc-14 : loc+1]
            x = np.arange(len(reg_window))
            slope_p = np.polyfit(x, reg_window['close'].values, 1)[0]
            slope_ad = np.polyfit(x, df['ad'].iloc[loc-14 : loc+1].values, 1)[0]
            div_sig = "-"
            if slope_p < 0 and slope_ad > 0: div_sig = "BULL DIV 💎"
            elif slope_p > 0 and slope_ad < 0: div_sig = "BEAR DIV 💣"
            
            # Change % (Day %)
            day_change_str = "-"
            if loc > 0:
                prev_close = df['close'].iloc[loc-1]
                day_pct = ((curr_close - prev_close) / prev_close) * 100
                day_change_str = f"{day_pct:+.1f}%"

            # Vol & OBV
            vol = df['volume'].iloc[loc]
            open_p = df['open'].iloc[loc]
            vol_color = "(G)" if curr_close >= open_p else "(R)"
            vol_str = f"{self.format_volume(vol)} {vol_color}"
            
            obv_window = df['obv'].iloc[loc-4 : loc+1]
            if len(obv_window) == 5:
                slope_obv = np.polyfit(np.arange(5), obv_window.values, 1)[0]
                obv_trend = "Rising ↗️" if slope_obv > 0 else "Falling ↘️"
            else: obv_trend = "-"

            date_str = str(df['date'].iloc[loc]).split(" ")[0]
            history.append([date_str, vwap, status, div_sig, curr_close, day_change_str, vol_str, obv_trend])
            
        return history

    # --- MODE 2: COMPRESSED HISTORY ---
    def get_compressed_bandar_flow(self, df):
        if len(df) < 80: return [] 
        subset_indices = df.index[-60:] 
        raw_history = []
        
        for idx in subset_indices:
            loc = df.index.get_loc(idx)
            window = df.iloc[loc-19 : loc+1]
            roll_vol = window['volume'].sum()
            roll_pv = (window['close'] * window['volume']).sum()
            vwap = roll_pv / roll_vol if roll_vol > 0 else 0
            curr_close = df['close'].iloc[loc]
            open_p = df['open'].iloc[loc]
            diff = ((curr_close - vwap) / vwap) * 100
            
            if diff < -2: status = "Accumulating"
            elif -2 <= diff <= 5: status = "Absorbing"
            else: status = "Distributing"
            
            reg_window = df.iloc[loc-14 : loc+1]
            x = np.arange(len(reg_window))
            slope_p = np.polyfit(x, reg_window['close'].values, 1)[0]
            slope_ad = np.polyfit(x, df['ad'].iloc[loc-14 : loc+1].values, 1)[0]
            div_sig = None
            if slope_p < 0 and slope_ad > 0: div_sig = "BULL DIV 💎"
            elif slope_p > 0 and slope_ad < 0: div_sig = "BEAR DIV 💣"
            
            date_str = str(df['date'].iloc[loc]).split(" ")[0]
            date_short = f"{date_str.split('-')[1]}-{date_str.split('-')[2]}"
            
            raw_history.append({'date': date_short, 'close': curr_close, 'open': open_p, 'vwap': vwap, 'status': status, 'div': div_sig})

        compressed = []
        if not raw_history: return []

        curr = raw_history[0]
        block = {
            'start_date': curr['date'], 'end_date': curr['date'], 'status': curr['status'],
            'start_price': curr['open'], 'end_price': curr['close'],
            'divs': {curr['div']} if curr['div'] else set(), 'count': 1, 'vwap_sum': curr['vwap']
        }

        for i in range(1, len(raw_history)):
            row = raw_history[i]
            if row['status'] == block['status']:
                block['end_date'] = row['date']
                block['end_price'] = row['close']
                block['count'] += 1
                block['vwap_sum'] += row['vwap']
                if row['div']: block['divs'].add(row['div'])
            else:
                compressed.append(block)
                block = {
                    'start_date': row['date'], 'end_date': row['date'], 'status': row['status'],
                    'start_price': row['open'], 'end_price': row['close'],
                    'divs': {row['div']} if row['div'] else set(), 'count': 1, 'vwap_sum': row['vwap']
                }
        compressed.append(block)
        return compressed

    def analyze_bandar_vwap(self, df, lookback):
        if len(df) < lookback: return None
        rolling_vol = df['volume'].rolling(lookback).sum()
        rolling_pv = (df['close'] * df['volume']).rolling(lookback).sum()
        vwap = rolling_pv / rolling_vol
        avg = vwap.iloc[-1]
        diff = ((df['close'].iloc[-1] - avg) / avg) * 100
        
        if diff < -2: s = "ACCUMULATION 🟢"; d = "Price < Avg (Cheap)"
        elif -2 <= diff <= 5: s = "HOLDING 🟡"; d = "Price ~ Avg (Base)"
        else: s = "DISTRIBUTION 🔴"; d = "Price > Avg (Profit)"
        return avg, diff, s, d

    def analyze_ad_line(self, df):
        if len(df) < 21: return "Insufficient Data"
        lookback = 20
        y_p = df['close'].iloc[-lookback:].values
        y_a = df['ad'].iloc[-lookback:].values
        x = np.arange(lookback)
        sp = np.polyfit(x, y_p, 1)[0]
        sa = np.polyfit(x, y_a, 1)[0]
        if sp > 0 and sa > 0: return "Inflow 🟢"
        elif sp < 0 and sa < 0: return "Outflow 🔴"
        elif sp < 0 and sa > 0: return "BULL DIV 💎" 
        elif sp > 0 and sa < 0: return "BEAR DIV 💣"
        else: return "Neutral"

    def analyze_ma_bounces(self, df):
        if len(df) < 200: return []
        c = df['close']
        ma_candidates = {
            'EMA 8': self.calculate_ema(c, 8), 'EMA 21': self.calculate_ema(c, 21),
            'EMA 34': self.calculate_ema(c, 34), 'SMA 50': self.calculate_sma(c, 50),
            'EMA 90': self.calculate_ema(c, 90), 'SMA 200': self.calculate_sma(c, 200),
        }
        results = []
        for name, ma in ma_candidates.items():
            if ma.isnull().all(): continue
            touches, bounces = 0, 0
            subset = df.iloc[-250:].copy()
            subset['ma'] = ma.iloc[-250:]
            for i in range(1, len(subset) - 5):
                if subset['low'].iloc[i] <= subset['ma'].iloc[i]*1.015 and subset['high'].iloc[i] >= subset['ma'].iloc[i]*0.985:
                    touches += 1
                    if subset['close'].iloc[i+5] > subset['close'].iloc[i]: bounces += 1
            if touches > 2: results.append((name, (bounces/touches)*100, touches, ma.iloc[-1]))
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return results[:5]

    def print_row(self, c1, c2, c3, c4, c5, c6):
        print(f"{c1:<7} | {c2:<16} | {c3:<10} | {c4:<10} | {c5:<20} | {c6}")

    def run(self, compressed_mode=False):
        print(f"\n[+] Fetching data for {self.ticker}...")
        df_daily = self.fetch_data("1d", "2y")
        df_weekly = self.fetch_data("1wk", "5y")
        df_monthly = self.fetch_data("1mo", "10y")
        
        if df_daily is None:
            print("[-] Error: Data not found.")
            return

        print("\n" + "="*96)
        print(f"  🚀 STOCK MASTER: {self.ticker}")
        print("="*96)
        
        # 1. MULTI-TIMEFRAME
        print(f"{'TF':<7} | {'TREND':<16} | {'OPT. RSI':<10} | {'OPT. STOCH':<10} | {'BANDAR (VWAP)':<20} | {'A/D FLOW'}")
        print("-" * 96)
        timeframes = [("DAILY", df_daily, 20), ("WEEKLY", df_weekly, 10), ("MONTHLY", df_monthly, 6)]
        for tf_name, df, bd_lb in timeframes:
            if df is None or len(df) < 25:
                self.print_row(tf_name, "No Data", "-", "-", "-", "-")
                continue
            trend = self.detect_trend(df)
            rsi_p = self.optimize_rsi(df)
            k_opt, d_opt = self.optimize_stoch(df)
            ad_status = self.analyze_ad_line(df)
            res = self.analyze_bandar_vwap(df, bd_lb)
            bd_str = f"{res[2].split(' ')[0]} @ {res[0]:,.0f}" if res else "-"
            self.print_row(tf_name, trend, f"RSI: {rsi_p}", f"{k_opt},{d_opt},3", bd_str, ad_status)

        # 2. BANDAR DEEP DIVE
        print("\n" + "="*40)
        print(f" 🐋 BANDAR DEEP DIVE (Daily Details)")
        print("="*40)
        bd_data = self.analyze_bandar_vwap(df_daily, 20)
        if bd_data:
            avg, diff, status, desc = bd_data
            curr_p = df_daily['close'].iloc[-1]
            print(f" Current Price    : {curr_p:,.0f}")
            print(f" Avg Price (20d)  : {avg:,.0f} (Smart Money Cost)")
            print(f" Gap to Avg       : {diff:+.2f}%")
            print(f" Phase            : {status}")
            print(f" Insight          : {desc}")
        else:
            print(" [-] Insufficient data.")

        # 3. DYNAMIC SUPPORT
        print("\n" + "="*40)
        print(f" 🛡️  DYNAMIC SUPPORT LEVELS")
        print("="*40)
        ma_bounces = self.analyze_ma_bounces(df_daily)
        if ma_bounces:
            print(f"{'MA TYPE':<10} {'WIN RATE':<10} {'TOUCHES':<8} {'PRICE':<10} {'GAP'}")
            print("-" * 50)
            curr_p = df_daily['close'].iloc[-1]
            for name, prob, touches, val in ma_bounces:
                dist = ((curr_p - val) / curr_p) * 100
                near_str = " 🎯" if 0 <= dist <= 3 else ""
                print(f"{name:<10} {prob:<9.0f}% {touches:<8} {val:<10,.0f} {dist:+.1f}%{near_str}")
        else:
            print(" [-] No reliable MA bounces found.")

        # 4. HISTORICAL FLOW (CONDITIONAL)
        if compressed_mode:
            print("\n" + "="*110)
            print(f" 📜 COMPRESSED BANDAR FLOW (60 Days Log - Ascending)")
            print("="*110)
            print(f" {'PERIOD':<14} {'DUR':<5} {'PHASE':<14} {'FLOW':<18} {'CHG %':<8} {'AVG VWAP':<10} {'SIG'}")
            print("-" * 110)
            history = self.get_compressed_bandar_flow(df_daily)
            for b in history:
                # FIX: Use pre-formatted short dates directly
                period_str = f"{b['start_date']} > {b['end_date']}"
                dur_str = f"{b['count']}d"
                flow_str = f"{b['start_price']:.0f} -> {b['end_price']:.0f}"
                pct_chg = ((b['end_price'] - b['start_price']) / b['start_price']) * 100
                pct_str = f"{pct_chg:+.1f}%"
                avg_vwap = b['vwap_sum'] / b['count']
                sig_str = ", ".join(list(b['divs'])) if b['divs'] else "-"
                print(f" {period_str:<14} {dur_str:<5} {b['status']:<14} {flow_str:<18} {pct_str:<8} {avg_vwap:<10.0f} {sig_str}")
        else:
            print("\n" + "="*118)
            print(f" 📜 DETAILED BANDAR FLOW (Last 60 Days - Ascending)")
            print("="*118)
            print(f" {'DATE':<12} {'PRICE':<8} {'VWAP':<10} {'PHASE':<14} {'DIVERGENCE':<12} {'DAY %':<8} {'VOLUME':<12} {'OBV TREND'}")
            print("-" * 118)
            history = self.get_detailed_bandar_flow(df_daily)
            for date, vwap, status, div, close, day_chg, vol, obv_t in history:
                print(f" {date:<12} {close:<8.0f} {vwap:<10.0f} {status:<14} {div:<12} {day_chg:<8} {vol:<12} {obv_t}")
        
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Analysis Tool")
    parser.add_argument("ticker", help="Stock Ticker Symbol (e.g., BBRI)")
    parser.add_argument("-c", "--compressed", action="store_true", help="View compressed history log")
    
    args = parser.parse_args()
    
    app = StockAnalyzer(args.ticker)
    app.run(compressed_mode=args.compressed)