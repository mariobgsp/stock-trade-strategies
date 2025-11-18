import pandas as pd
import numpy as np

# --- FUNGSI BANTUAN (HELPER) ---
def _get_scalar(row, col_name, default=0):
    """
    Fungsi pengaman: Mengambil nilai tunggal dari baris data.
    """
    try:
        val = row.get(col_name, default)
        if isinstance(val, pd.Series):
            val = val.iloc[0] 
        if pd.isna(val): 
            return default
        return val
    except:
        return default

def scan_for_signals(df, clustered_supports, clustered_resistances):
    """
    Memindai sinyal historis (FIX: Menampilkan daftar sinyal + % RETURN).
    """
    print("\n--- (DETECTOR) Memulai Pemindaian Sinyal Historis ---")
    
    # 1. Bersihkan Data
    df_clean = df.dropna(subset=['close']).copy()
    total_bars = len(df_clean)
    look_ahead_days = 10 # Periode lookahead untuk perhitungan return
    
    # 2. CEK DATA
    if total_bars < 200:
        print(f"  [INFO] Data history terlalu pendek ({total_bars} bar).")
        return pd.DataFrame() 

    print(f"  [INFO] Memindai {total_bars} bar data historis...")

    signals_found = []
    support_levels = clustered_supports['Level'].values
    proximity = 0.02

    # Loop scan (Mulai dari bar ke-200)
    for i in range(200, total_bars):
        bar = df_clean.iloc[i]
        prev_bar = df_clean.iloc[i-1]
        
        # Ambil Data
        ma50 = _get_scalar(bar, 'sma_50')
        ma200 = _get_scalar(bar, 'sma_200')
        ma50_prev = _get_scalar(prev_bar, 'sma_50')
        ma200_prev = _get_scalar(prev_bar, 'sma_200')
        close_val = _get_scalar(bar, 'close')
        open_val = _get_scalar(bar, 'open')
        low_val = _get_scalar(bar, 'low')
        
        # --- PERHITUNGAN RETURN (Dilakukan per bar, diakses jika sinyal ditemukan) ---
        entry_price = close_val
        future_idx = min(i + look_ahead_days, total_bars - 1)
        future_price = _get_scalar(df_clean.iloc[future_idx], 'close')
        pct_return = (future_price - entry_price) / entry_price if entry_price > 0 else 0
        
        # 1. Sinyal Golden Cross
        if (ma50_prev < ma200_prev) and (ma50 > ma200) and (close_val > ma50):
            signals_found.append({
                "Tanggal": bar.name, 
                "Sinyal (Label)": "TREND - GOLDEN CROSS", 
                "Harga": close_val, 
                "Return 10D (%)": f"{pct_return:+.2%}", # ADDED RETURN
                "Detail": "MA 50 Cross MA 200"
            })

        # 2. Sinyal Buy The Dip
        is_uptrend = close_val > ma200
        if is_uptrend:
            if (low_val < ma50) and (close_val > ma50) and (close_val > open_val):
                 signals_found.append({
                    "Tanggal": bar.name, 
                    "Sinyal (Label)": "BUY THE DIP (MA 50)", 
                    "Harga": close_val, 
                    "Return 10D (%)": f"{pct_return:+.2%}", # ADDED RETURN
                    "Detail": "Pantul MA 50"
                })
                 
        # 3. Sinyal Reversal Engulfing
        is_bull_engulf = (close_val > open_val) and \
                         (_get_scalar(prev_bar, 'close') < _get_scalar(prev_bar, 'open')) and \
                         (close_val > _get_scalar(prev_bar, 'open')) and \
                         (open_val < _get_scalar(prev_bar, 'close'))
        
        if is_bull_engulf:
             for s in clustered_supports['Level'].values:
                 if abs(low_val - s) / s < proximity:
                     signals_found.append({
                        "Tanggal": bar.name, 
                        "Sinyal (Label)": "REVERSAL - BULLISH ENGULFING", 
                        "Harga": close_val, 
                        "Return 10D (%)": f"{pct_return:+.2%}", # ADDED RETURN
                        "Detail": f"Support {s:.0f}"
                    })
                     break

    if not signals_found:
        print("  [INFO] Scan selesai. Tidak ditemukan pola Golden Cross / BTD yang valid.")
        return pd.DataFrame()
    else:
        df_signals = pd.DataFrame(signals_found).set_index('Tanggal')
        print(f"  [SUKSES] Ditemukan {len(signals_found)} sinyal historis.")
        # --- FIX: CETAK DAFTAR SINYAL HISTORIS DENGAN FORMAT Penuh ---
        print("\n--- DAFTAR SINYAL HISTORIS ---")
        print(df_signals.to_string())
        print("------------------------------")
        # ------------------------------------------------------------
        return df_signals


def analyze_behavior(df, 
                     clustered_s_base, clustered_r_base, raw_s_base, raw_r_base,
                     clustered_s_detail, clustered_r_detail, raw_s_detail, raw_r_detail,
                     market_structure, short_term_trend, vol_today, vol_short_term,
                     daily_momentum, rsi_behavior, fib_levels, pivot_levels, num_days):
    """
    Laporan Analisis Harian.
    """
    print("\n\n--- (ANALISIS HARI INI) Laporan Komprehensif ---")
    
    summary_findings = {}
    
    # Data Cleaning
    df_hist = df.dropna(subset=['close']).copy()
    if df_hist.empty: return {}
    
    last_bar = df_hist.iloc[-1]
    last_bar_future = df.iloc[-1]
    
    current_price = _get_scalar(last_bar, 'close')
    
    # Pivot & RSI Manual Calculation (Anti-NaN)
    if len(df_hist) > 1:
        prev = df_hist.iloc[-2]
        pp = (prev['high'] + prev['low'] + prev['close']) / 3
        pivot_display = {"PP": f"{pp:.0f}"}
    else:
        pivot_display = "Data Kurang"

    rsi_val = _get_scalar(last_bar, 'rsi_14', 50)
    rsi_display = f"{rsi_val:.2f} ({'Bullish' if rsi_val > 50 else 'Bearish'})"

    # --- Isi Laporan ---
    summary_findings["Market Structure"] = market_structure
    summary_findings["Tren Jangka Pendek"] = short_term_trend
    summary_findings["Momentum Harian"] = daily_momentum
    summary_findings["Perilaku RSI"] = rsi_display
    summary_findings["Volume Status"] = vol_today
    summary_findings["Harga Saat Ini"] = f"{current_price:.0f}"
    
    summary_findings["Proyeksi Fibonacci"] = fib_levels
    summary_findings["Proyeksi Pivot Points"] = pivot_display
    
    sr_base_list = ["--- S/R JANGKA PANJANG (BASIS) ---"]
    for idx, row in clustered_r_base.head(3).iterrows(): sr_base_list.append(f"  R: {row['Level']:.0f}")
    for idx, row in clustered_s_base.head(3).iterrows(): sr_base_list.append(f"  S: {row['Level']:.0f}")
    summary_findings[f"S/R Jangka Panjang"] = sr_base_list

    sr_detail_list = [f"--- S/R JANGKA PENDEK (DETAIL {num_days} HARI) ---"]
    for idx, row in clustered_r_detail.head(3).iterrows(): sr_detail_list.append(f"  R: {row['Level']:.0f}")
    for idx, row in clustered_s_detail.head(3).iterrows(): sr_detail_list.append(f"  S: {row['Level']:.0f}")
    summary_findings[f"S/R Jangka Pendek"] = sr_detail_list

    # --- Indikator Tambahan ---
    adx = _get_scalar(last_bar, 'adx_14', 0)
    rvol = _get_scalar(last_bar, 'rvol', 0)
    atr = _get_scalar(last_bar, 'atrr_14', 0)
    
    span_a = _get_scalar(last_bar, 'isa_9', 0)
    span_b = _get_scalar(last_bar, 'isb_26', 0)
    c_max = max(float(span_a), float(span_b))
    c_min = min(float(span_a), float(span_b))
    
    ichi_stat = "Konsolidasi (Dalam Cloud)"
    if current_price > c_max: ichi_stat = "Bullish (Di Atas Cloud)"
    elif current_price < c_min: ichi_stat = "Bearish (Di Bawah Cloud)"

    f_span_a = _get_scalar(last_bar_future, 'isa_9', 0)
    f_span_b = _get_scalar(last_bar_future, 'isb_26', 0)
    future_cloud = "Bullish (Hijau)" if f_span_a > f_span_b else "Bearish (Merah)"

    summary_findings["Indikator Tambahan"] = {
        "ADX": f"{adx:.2f}", "RVOL": f"{rvol:.2f}x", "ATR": f"{atr:.0f}",
        "Ichimoku (Now)": ichi_stat, "Ichimoku (Future)": future_cloud
    }
    
    patterns = []
    if _get_scalar(last_bar, 'cdl_hammer', 0) == 100: patterns.append("Hammer")
    if _get_scalar(last_bar, 'cdl_morningstar', 0) == 100: patterns.append("Morning Star")
    if _get_scalar(last_bar, 'cdl_doji', 0) == 100: patterns.append("Doji")
    if _get_scalar(last_bar, 'cdl_engulfing', 0) == 100: patterns.append("Bullish Engulfing")
    summary_findings["Pola Candle"] = ", ".join(patterns) if patterns else "Normal"

    return summary_findings


def recommend_trade(df, clustered_supports, clustered_resistances, 
                     raw_supports, raw_resistances, 
                     raw_s_detail, raw_r_detail,
                     market_structure, min_rr_ratio=1.5):
    """
    Rekomendasi Trade ULTIMATE.
    """
    print("\n--- (REKOMENDASI TRADE - ULTIMATE SETUP) ---")
    
    df_clean = df.dropna(subset=['close']).copy()
    if df_clean.empty: return {}
    last_bar = df_clean.iloc[-1]
    
    current_price = _get_scalar(last_bar, 'close')
    
    recommendation = {
        "Rekomendasi": "WAIT / NO TRADE", "Alasan": "Syarat belum terpenuhi.",
        "Skor Kualitas": "0 Poin", "Tipe Trade": "N/A",
        "Setup Entry": "N/A", "Stop Loss (SL)": "N/A",
        "Take Profit (TP)": "N/A", "Risk/Reward": "N/A"
    }

    # Data Indikator
    adx_val = _get_scalar(last_bar, 'adx_14', 0)
    atr_val = _get_scalar(last_bar, 'atrr_14', current_price * 0.02)
    if pd.isna(atr_val) or atr_val == 0: atr_val = current_price * 0.02
    
    rvol_val = _get_scalar(last_bar, 'rvol', 0)
    is_ma50_up = _get_scalar(last_bar, 'slope_ma50_up', False)
    
    span_a = _get_scalar(last_bar, 'isa_9', 0)
    span_b = _get_scalar(last_bar, 'isb_26', 0)
    is_above_cloud = current_price > max(float(span_a), float(span_b))
    
    is_morningstar = _get_scalar(last_bar, 'cdl_morningstar', 0) == 100
    is_engulfing = _get_scalar(last_bar, 'cdl_engulfing', 0) == 100
    is_hammer = _get_scalar(last_bar, 'cdl_hammer', 0) == 100

    # Target & SL
    valid_s = clustered_supports[clustered_supports['Level'] < current_price]
    if not valid_s.empty:
        nearest_s = valid_s['Level'].max()
        s_type = "Cluster"
    else:
        s_recent = raw_s_detail[raw_s_detail < current_price]
        nearest_s = s_recent.max() if not s_recent.empty else current_price * 0.95
        s_type = "Pivot"

    sl_atr = current_price - (2 * float(atr_val))
    sl_structure = nearest_s * 0.97
    
    dist_struct = (current_price - sl_structure) / current_price
    if dist_struct > 0.08: 
        sl_price = sl_atr; sl_basis = "2x ATR (Tight)"
    elif sl_atr > sl_structure: 
        sl_price = sl_structure; sl_basis = f"Structure ({s_type})"
    else:
        sl_price = sl_atr; sl_basis = "2x ATR"

    tp_price = 0
    tp_basis = "N/A"
    
    try:
        if len(raw_s_detail) >= 2 and len(raw_r_detail) >= 1:
            A = raw_s_detail.iloc[-2]; B = raw_r_detail.iloc[-1]; C = raw_s_detail.iloc[-1]
            date_A = raw_s_detail.index[-2]; date_B = raw_r_detail.index[-1]; date_C = raw_s_detail.index[-1]
            if (date_A < date_B < date_C) and (C < B):
                 fib_ext = C + ((B - A) * 1.618)
                 if fib_ext > current_price: tp_price = fib_ext; tp_basis = "Fib Ext 1.618"
    except: pass
    
    if tp_price == 0:
        valid_r = clustered_resistances[clustered_resistances['Level'] > current_price]
        if not valid_r.empty: tp_price = valid_r['Level'].min() * 0.99; tp_basis = "Res Cluster"
        else: tp_price = current_price * 1.1; tp_basis = "ATH / 10%"

    risk = current_price - sl_price
    reward = tp_price - current_price
    rr = reward / risk if risk > 0 else 0

    # Scoring
    score = 0
    if "UPTREND" in market_structure and is_ma50_up: score += 2
    elif "UPTREND" in market_structure: score += 1
    if is_above_cloud: score += 1
    else: score -= 1
    if is_morningstar: score += 2
    elif is_engulfing or is_hammer: score += 1

    rsi_val = _get_scalar(last_bar, 'rsi_14', 50)
    if adx_val > 20: score += 1
    if rvol_val > 1.2: score += 1
    if rsi_val < 65: score += 1

    # Keputusan
    trade_type = "SWING" if adx_val > 20 else "QUICK"
    min_score = 4 

    recommendation["Setup Entry"] = f"{nearest_s:.0f} - {current_price:.0f}"
    recommendation["Stop Loss (SL)"] = f"{sl_price:.0f} ({sl_basis})"
    recommendation["Take Profit (TP)"] = f"{tp_price:.0f} ({tp_basis})"
    recommendation["Risk/Reward"] = f"1 : {rr:.2f}"
    recommendation["Skor Kualitas"] = f"{score} Poin"
    recommendation["Tipe Trade"] = trade_type
    
    dist_to_support = (current_price - nearest_s) / nearest_s
    entry_valid = dist_to_support < 0.06
    
    if rr >= 1.5 and score >= min_score and entry_valid:
        recommendation["Rekomendasi"] = "STRONG BUY"
    elif rr >= 2.0 and score >= (min_score - 1) and entry_valid:
        recommendation["Rekomendasi"] = "SPECULATIVE BUY"
    else:
        recommendation["Rekomendasi"] = "WAIT / MONITOR"
        if not entry_valid: recommendation["Alasan"] = "Harga jauh dari support."
        else: recommendation["Alasan"] = f"Skor ({score}) / RR ({rr:.2f}) kurang."

    return recommendation


def analyze_historical_performance(df, df_signals, look_ahead_days=10):
    """ Analisis performa. """
    df_clean = df.dropna(subset=['close'])
    target = ["BUY THE DIP (MA 50)", "TREND - GOLDEN CROSS"]
    filtered = df_signals[df_signals['Sinyal (Label)'].isin(target)]
    if filtered.empty: return pd.DataFrame()

    results = []
    for date, row in filtered.iterrows():
        try:
            idx = df_clean.index.get_loc(date)
            future_idx = min(idx + look_ahead_days, len(df_clean) - 1)
            entry = row['Harga']
            future = df_clean.iloc[future_idx]['close']
            if isinstance(future, pd.Series): future = future.iloc[0]
            ret = (future - entry) / entry
            results.append({"Sinyal": row['Sinyal (Label)'], "Return": ret})
        except: continue

    if not results: return pd.DataFrame()
    return pd.DataFrame(results).groupby('Sinyal')['Return'].agg(
        Avg_Return=lambda x: f"{x.mean():.2%}",
        Win_Rate=lambda x: f"{(x > 0).sum() / len(x):.0%}"
    )