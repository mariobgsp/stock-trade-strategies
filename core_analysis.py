import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def cluster_levels(levels, cluster_percent=0.02):
    """
    Mengelompokkan level harga yang berdekatan.
    """
    if levels.empty:
        return pd.DataFrame(columns=['Level', 'Hits', 'Kekuatan'])
    
    sorted_levels = levels.sort_values().values
    clusters = []
    
    i = 0
    while i < len(sorted_levels):
        current_cluster_levels = [sorted_levels[i]]
        cluster_base_price = sorted_levels[i]
        
        j = i + 1
        while j < len(sorted_levels):
            level = sorted_levels[j]
            if (level - cluster_base_price) / cluster_base_price <= cluster_percent:
                current_cluster_levels.append(level)
                j += 1
            else:
                break 
        
        cluster_mean = np.mean(current_cluster_levels)
        cluster_hits = len(current_cluster_levels)
        clusters.append({'Level': cluster_mean, 'Hits': cluster_hits})
        i = j 
        
    df_clusters = pd.DataFrame(clusters).sort_values(by='Hits', ascending=False)
    df_clusters['Level'] = df_clusters['Level'].round(0)
    
    def tentukan_kekuatan(hits):
        if hits >= 5: return "Kuat"
        elif hits >= 3: return "Sedang"
        else: return "Lemah"
    
    df_clusters['Kekuatan'] = df_clusters['Hits'].apply(tentukan_kekuatan)
    
    return df_clusters

def find_support_resistance(df):
    """
    Mendeteksi level S/R mentah dan yang sudah di-cluster.
    """
    df_clean = df.dropna(subset=['high', 'low'])
    
    try:
        peaks_indices, _ = find_peaks(df_clean['high'], distance=10, prominence=df_clean['high'].std() * 0.8)
        valleys_indices, _ = find_peaks(-df_clean['low'], distance=10, prominence=df_clean['low'].std() * 0.8)

        raw_supports = df_clean['low'].iloc[valleys_indices]
        raw_resistances = df_clean['high'].iloc[peaks_indices]
        
        clustered_supports = cluster_levels(raw_supports, cluster_percent=0.02)
        clustered_resistances = cluster_levels(raw_resistances, cluster_percent=0.02)
        
        return clustered_supports, clustered_resistances, raw_supports, raw_resistances
    except:
        return pd.DataFrame(columns=['Level']), pd.DataFrame(columns=['Level']), pd.Series(), pd.Series()

def detect_market_structure(df):
    """
    Menganalisis TREN PASAR (Market Structure).
    """
    df_clean = df.dropna(subset=['close'])
    if len(df_clean) < 200: return "DATA KURANG"
        
    last_bar = df_clean.iloc[-1]
    
    try:
        price = last_bar['close']
        ma50 = last_bar['sma_50']
        ma200 = last_bar['sma_200']
        
        if pd.isna(ma50) or pd.isna(ma200): return "MA Belum Terhitung"

        if price > ma50 and ma50 > ma200: return "UPTREND (Sehat)"
        if price < ma50 and ma50 < ma200: return "DOWNTREND (Parah)"
        if price > ma200 and price < ma50: return "KONSOLIDASI (Koreksi)"
        if price < ma200 and price > ma50: return "KONSOLIDASI (Rebound)"
        if price > ma200: return "UPTREND (Early)"
        if price < ma200: return "DOWNTREND"
        return "SIDEWAYS"
    except: return "ERROR (Indikator MA)"

def analyze_short_term_trend(detail_df, num_days):
    """
    Menganalisis tren jangka pendek.
    """
    clean_df = detail_df.dropna(subset=['close'])
    
    if clean_df.empty or len(clean_df) < 2:
        return f"DATA KURANG ({num_days} Hari)"
    
    start_price = clean_df.iloc[0]['close']
    end_price = clean_df.iloc[-1]['close']
    
    if start_price == 0: return "Error (Harga Awal 0)"
        
    change_pct = (end_price - start_price) / start_price
    threshold = (len(clean_df) / 21) * 0.02 
    
    if change_pct > threshold:
        return f"Uptrend ({change_pct:+.2%}) dalam {len(clean_df)} hari"
    elif change_pct < -threshold:
        return f"Downtrend ({change_pct:+.2%}) dalam {len(clean_df)} hari"
    else:
        return f"Sideways ({change_pct:+.2%}) dalam {len(clean_df)} hari"

def analyze_volume_profile(detail_df):
    """
    Analisis Volume Akumulasi/Distribusi.
    """
    clean_df = detail_df.dropna(subset=['close', 'volume'])
    if len(clean_df) < 20: return "Data Kurang", "Data Kurang"
         
    last_bar = clean_df.iloc[-1]
    vol_ma = last_bar.get('volume_ma20', last_bar['volume'])
    if pd.isna(vol_ma): vol_ma = 0
    
    is_high_vol = last_bar['volume'] > (vol_ma * 1.5)
    is_green = last_bar['close'] > last_bar['open']
    
    today_status = "Volume Normal"
    if is_high_vol and is_green: today_status = "Akumulasi Kuat (Vol Tinggi)"
    elif is_high_vol and not is_green: today_status = "Distribusi Kuat (Vol Tinggi)"

    akumulasi_days = 0
    distribusi_days = 0
    
    for index, bar in clean_df.iterrows():
        v_ma = bar.get('volume_ma20', 0)
        if pd.isna(v_ma) or v_ma == 0: continue
        
        if bar['volume'] > (v_ma * 1.5):
            if bar['close'] > bar['open']: akumulasi_days += 1
            else: distribusi_days += 1
    
    short_term_status = "Seimbang"
    if akumulasi_days > distribusi_days: short_term_status = f"Dominasi Akumulasi ({akumulasi_days} hari)"
    elif distribusi_days > akumulasi_days: short_term_status = f"Dominasi Distribusi ({distribusi_days} hari)"
    
    return today_status, short_term_status

def analyze_daily_momentum(df):
    """ Analisis candle terakhir. """
    clean_df = df.dropna(subset=['close'])
    if len(clean_df) < 1: return "Data Kurang"
    
    last_bar = clean_df.iloc[-1]
    o, h, l, c = last_bar['open'], last_bar['high'], last_bar['low'], last_bar['close']
    rng = h - l
    
    if rng == 0: return "Doji (Flat)"
    
    is_green = c > o
    body_pct = abs(c - o) / rng
    
    if is_green and body_pct > 0.8: return "Bullish Marubozu (Sangat Kuat)"
    if not is_green and body_pct > 0.8: return "Bearish Marubozu (Sangat Lemah)"
    if (h - max(o,c)) > rng * 0.6: return "Shooting Star / Rejection Atas"
    if (min(o,c) - l) > rng * 0.6: return "Hammer / Rejection Bawah"
    
    return "Bullish Normal" if is_green else "Bearish Normal"

def analyze_rsi_behavior(detail_df):
    """ Analisis RSI (Trend & Divergence). """
    clean_df = detail_df.dropna(subset=['rsi_14'])
    if len(clean_df) < 2: return "Data RSI Kurang"
    
    last = clean_df.iloc[-1]['rsi_14']
    prev = clean_df.iloc[-2]['rsi_14']
    
    trend = "Naik" if last > prev else "Turun"
    state = "Netral"
    if last > 70: state = "Overbought (>70)"
    elif last < 30: state = "Oversold (<30)"
    elif last > 50: state = "Bullish Zone"
    else: state = "Bearish Zone"
    
    return f"{state}, Arah {trend} ({last:.1f})"

def backtest_oscillator_levels(df):
    """
    Menganalisis data historis untuk level pembalikan, 
    menambahkan Rata-rata Perubahan Harga 5 Hari.
    (FIXED: Mengisi kolom count dan membersihkan NaN)
    """
    
    peak_stats = pd.DataFrame()
    valley_stats = pd.DataFrame()
    look_ahead_days = 5
    
    df_clean = df.dropna(subset=['close', 'rsi_14', 'stochk_10_3_3'])
    total_bars = len(df_clean)

    try:
        # 1. PEAKS (Turn Bearish)
        peaks_indices, _ = find_peaks(df_clean['high'], distance=10, prominence=df_clean['high'].std() * 0.8)
        if peaks_indices.size > 0:
            df_peaks = df_clean.iloc[peaks_indices]
            
            peak_changes = []
            for idx in df_peaks.index: 
                entry_idx = df_clean.index.get_loc(idx)
                entry_price = df_clean.iloc[entry_idx]['high']
                
                future_idx = min(entry_idx + look_ahead_days, total_bars - 1)
                future_price = df_clean.iloc[future_idx]['close'] 
                
                change = (future_price - entry_price) / entry_price
                peak_changes.append(change)
            
            peak_stats = df_peaks[['rsi_14', 'stochk_10_3_3']].describe(percentiles=[.25, .5, .75])
            
            # Tambahkan baris Avg Change 5D dan isi kolom 'count'
            peak_stats.loc['Avg Change 5D'] = np.nan
            peak_stats.loc['Avg Change 5D', 'rsi_14'] = np.mean(peak_changes)
            peak_stats.loc['Avg Change 5D', 'count'] = len(peak_changes) # FIX: Isi kolom count
            peak_stats.loc['Avg Change 5D', 'stochk_10_3_3'] = np.nan 
            
        # 2. VALLEYS (Turn Bullish)
        valleys_indices, _ = find_peaks(-df_clean['low'], distance=10, prominence=df_clean['low'].std() * 0.8)
        if valleys_indices.size > 0:
            df_valleys = df_clean.iloc[valleys_indices]

            valley_changes = []
            for idx in df_valleys.index:
                entry_idx = df_clean.index.get_loc(idx)
                entry_price = df_clean.iloc[entry_idx]['low']
                
                future_idx = min(entry_idx + look_ahead_days, total_bars - 1)
                future_price = df_clean.iloc[future_idx]['close']
                
                change = (future_price - entry_price) / entry_price
                valley_changes.append(change)

            valley_stats = df_valleys[['rsi_14', 'stochk_10_3_3']].describe(percentiles=[.25, .5, .75])
            
            # Tambahkan baris Avg Change 5D dan isi kolom 'count'
            valley_stats.loc['Avg Change 5D'] = np.nan
            valley_stats.loc['Avg Change 5D', 'rsi_14'] = np.mean(valley_changes)
            valley_stats.loc['Avg Change 5D', 'count'] = len(valley_changes) # FIX: Isi kolom count
            valley_stats.loc['Avg Change 5D', 'stochk_10_3_3'] = np.nan
            
    except Exception as e:
        print(f"Debug Backtest Error: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Format Rata-rata Perubahan Harga
    if 'Avg Change 5D' in peak_stats.index:
         peak_stats.loc['Avg Change 5D', 'rsi_14'] = f"{peak_stats.loc['Avg Change 5D', 'rsi_14']:+.2%}"
    if 'Avg Change 5D' in valley_stats.index:
         valley_stats.loc['Avg Change 5D', 'rsi_14'] = f"{valley_stats.loc['Avg Change 5D', 'rsi_14']:+.2%}"
        
    return peak_stats, valley_stats

def calculate_fibonacci_levels(detail_df):
    """ Fibonacci Retracement. """
    clean_df = detail_df.dropna(subset=['close'])
    if clean_df.empty: return {}

    swing_low = clean_df['low'].min()
    swing_high = clean_df['high'].max()
    
    levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    res = {"Swing": f"L:{swing_low:.0f} - H:{swing_high:.0f}"}
    
    for lvl in levels:
        price = swing_high - (swing_high - swing_low) * lvl
        res[f"Ret {lvl:.3f}"] = f"{price:.0f}"
    return res

def calculate_pivot_points(df):
    """ Pivot Points Harian. """
    clean_df = df.dropna(subset=['close'])
    if len(clean_df) < 2: return {}
    
    prev = clean_df.iloc[-2] # Kemarin
    pp = (prev['high'] + prev['low'] + prev['close']) / 3
    r1 = (2 * pp) - prev['low']
    s1 = (2 * pp) - prev['high']
    
    return {"PP": f"{pp:.0f}", "R1": f"{r1:.0f}", "S1": f"{s1:.0f}"}