import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def cluster_levels(levels, cluster_percent=0.02):
    """
    Mengelompokkan level harga yang berdekatan.
    (VERSI DIPERBARUI: Menambahkan label 'Kekuatan')
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
        if hits >= 5:
            return "Kuat"
        elif hits >= 3: # (3 atau 4 hits)
            return "Sedang"
        else: # (1 atau 2 hits)
            return "Lemah"
    
    df_clusters['Kekuatan'] = df_clusters['Hits'].apply(tentukan_kekuatan)
    
    return df_clusters

def find_support_resistance(df):
    """
    Mendeteksi level S/R mentah dan yang sudah di-cluster.
    (Menggunakan kolom lowercase 'high' dan 'low')
    """
    peaks_indices, _ = find_peaks(df['high'], distance=10, prominence=df['high'].std() * 0.8)
    valleys_indices, _ = find_peaks(-df['low'], distance=10, prominence=df['low'].std() * 0.8)

    raw_supports = df['low'].iloc[valleys_indices]
    raw_resistances = df['high'].iloc[peaks_indices]
    
    clustered_supports = cluster_levels(raw_supports, cluster_percent=0.02)
    clustered_resistances = cluster_levels(raw_resistances, cluster_percent=0.02)
    
    return clustered_supports, clustered_resistances, raw_supports, raw_resistances

def detect_market_structure(df):
    """
    Menganalisis TREN PASAR (Market Structure) berdasarkan posisi harga
    terhadap MA 50 dan MA 200.
    """
    if len(df) < 200:
        return "DATA KURANG (MA 200)"
        
    last_bar = df.iloc[-1]
    
    try:
        price = last_bar['close']
        ma50 = last_bar['sma_50']
        ma200 = last_bar['sma_200']
    except KeyError as e:
        print(f"Error di detect_market_structure: Kolom {e} tidak ada.")
        return "ERROR (Kolom MA)"

    if price > ma50 and ma50 > ma200:
        return "UPTREND (Sehat)"
    if price < ma50 and ma50 < ma200:
        return "DOWNTREND (Parah)"
    if price > ma200 and price < ma50:
        return "KONSOLIDASI (Koreksi dalam Uptrend)"
    if price < ma200 and price > ma50:
        return "KONSOLIDASI (Rebound dalam Downtrend)"
    if price > ma200 and ma50 < ma200:
        return "TRANSISI (Potensi Bullish)"
    if price < ma200 and ma50 > ma200:
        return "TRANSISI (Potensi Bearish)"
    return "SIDEWAYS (Tidak Terdefinisi)"

def analyze_short_term_trend(detail_df, num_days):
    """
    Menganalisis tren jangka pendek (berdasarkan data 'X' hari terakhir).
    """
    if detail_df.empty or len(detail_df) < 2:
        return f"DATA KURANG ({num_days} Hari)"
    
    start_price = detail_df.iloc[0]['close']
    end_price = detail_df.iloc[-1]['close']
    change_pct = (end_price - start_price) / start_price
    threshold = (num_days / 21) * 0.02
    
    if change_pct > threshold:
        return f"Uptrend ({change_pct:+.2%}) dalam {num_days} hari"
    elif change_pct < -threshold:
        return f"Downtrend ({change_pct:+.2%}) dalam {num_days} hari"
    else:
        return f"Sideways ({change_pct:+.2%}) dalam {num_days} hari"

def analyze_volume_profile(detail_df):
    """
    Menganalisis volume untuk tanda-tanda Akumulasi atau Distribusi.
    Melihat hari ini dan data 'detail_df' (X bulan terakhir).
    """
    if len(detail_df) < 20: 
         return "DATA KURANG (Volume)", "DATA KURANG (Volume)"
         
    last_bar = detail_df.iloc[-1]
    try:
        is_high_volume = last_bar['volume'] > (last_bar['volume_ma20'] * 1.5)
        is_green_candle = last_bar['close'] > last_bar['open']
        is_red_candle = last_bar['close'] < last_bar['open']
    except KeyError:
        return "ERROR (Kolom Volume)", "ERROR (Kolom Volume)"

    today_status = "Volume Normal"
    if is_high_volume and is_green_candle:
        today_status = "Akumulasi Kuat (Vol Tinggi, Hijau)"
    elif is_high_volume and is_red_candle:
        today_status = "Distribusi Kuat (Vol Tinggi, Merah)"
    elif is_high_volume:
        today_status = "Volume Tinggi (Netral/Doji)"

    if len(detail_df) < 25: 
        return today_status, "DATA KURANG (Jangka Pendek)"

    akumulasi_days = 0
    distribusi_days = 0
    
    for index, bar in detail_df.iterrows():
        if pd.isna(bar['volume_ma20']) or bar['volume_ma20'] == 0:
            continue 
        is_high_vol_week = bar['volume'] > (bar['volume_ma20'] * 1.5)
        if is_high_vol_week and (bar['close'] > bar['open']):
            akumulasi_days += 1
        elif is_high_vol_week and (bar['close'] < bar['open']):
            distribusi_days += 1
    
    short_term_status = "Seimbang (Tidak ada Vol dominan)"
    dominance_threshold = max(3, len(detail_df) * 0.1) 
    
    if akumulasi_days > distribusi_days and akumulasi_days >= dominance_threshold:
        short_term_status = f"Cenderung Akumulasi ({akumulasi_days} hari Vol Kuat)"
    elif distribusi_days > akumulasi_days and distribusi_days >= dominance_threshold:
        short_term_status = f"Cenderung Distribusi ({distribusi_days} hari Vol Kuat)"
    
    return today_status, short_term_status

def analyze_daily_momentum(df):
    """
    Menganalisis candle (OHLC) HARI TERAKHIR untuk momentum intraday.
    """
    if len(df) < 1:
        return "DATA KURANG"
        
    last_bar = df.iloc[-1]
    o, h, l, c = last_bar['open'], last_bar['high'], last_bar['low'], last_bar['close']
    
    total_range = h - l
    if total_range == 0:
        return "Tidak ada Pergerakan (Doji Sempurna)"
        
    is_green_candle = c > o
    is_red_candle = c < o
    body_size = abs(c - o)
    close_percentile = (c - l) / total_range
    body_vs_range_ratio = body_size / total_range
    
    if is_green_candle and close_percentile > 0.9 and body_vs_range_ratio > 0.7:
        return "Uptrend (Bullish Marubozu / Sangat Kuat)"
    if is_red_candle and close_percentile < 0.1 and body_vs_range_ratio > 0.7:
        return "Downtrend (Bearish Marubozu / Sangat Lemah)"
        
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    if lower_wick > total_range * 0.5 and close_percentile > 0.5:
        return "Netral (Tekanan Beli Kuat / Hammer)"
    if upper_wick > total_range * 0.5 and close_percentile < 0.5:
        return "Netral (Tekanan Jual Kuat / Shooting Star)"
    if body_vs_range_ratio < 0.2:
        return "Sideways (Ragu-ragu / Spinning Top)"
        
    if is_green_candle and close_percentile > 0.5:
        return "Uptrend (Bullish Normal)"
    if is_red_candle and close_percentile < 0.5:
        return "Downtrend (Bearish Normal)"
        
    if is_green_candle:
        return "Uptrend (Bullish Normal)"
    elif is_red_candle:
        return "Downtrend (Bearish Normal)"

    return "Sideways (Tidak Terdefinisi)"

# --- [PERUBAHAN] Fungsi ini sekarang menyertakan ARAH pergerakan ---
def analyze_rsi_behavior(detail_df, overbought=70, oversold=30, mid=50):
    """
    Menganalisis perilaku RSI (14) pada data detail.
    Fokus pada sinyal crossing DAN ARAH PERGERAKAN.
    """
    if len(detail_df) < 2:
        return "DATA KURANG (RSI)"
    
    last_bar = detail_df.iloc[-1]
    prev_bar = detail_df.iloc[-2]
    
    try:
        rsi_val = last_bar['rsi_14']
        rsi_prev = prev_bar['rsi_14']
    except KeyError:
        return "ERROR (Kolom rsi_14)"
    
    # Tentukan Arah Pergerakan RSI
    direction_text = "Datar"
    if rsi_val > rsi_prev:
        direction_text = "Naik (Menuju Overbought)"
    elif rsi_val < rsi_prev:
        direction_text = "Turun (Menuju Oversold)"
    
    # 1. Cek Sinyal Crossing (Prioritas)
    is_bullish_cross = rsi_prev < oversold and rsi_val > oversold
    if is_bullish_cross:
        return f"SINYAL UPTREND (Cross ke atas {oversold}) - Nilai: {rsi_val:.2f} (Arah: {direction_text})"
        
    is_bearish_cross = rsi_prev > overbought and rsi_val < overbought
    if is_bearish_cross:
        return f"SINYAL DOWNTREND (Cross ke bawah {overbought}) - Nilai: {rsi_val:.2f} (Arah: {direction_text})"

    # 2. Cek Status Zona (Jika tidak ada crossing)
    if rsi_val > overbought:
        return f"Overbought (> {overbought}) - Nilai: {rsi_val:.2f} (Arah: {direction_text})"
    if rsi_val < oversold:
        return f"Oversold (< {oversold}) - Nilai: {rsi_val:.2f} (Arah: {direction_text})"
        
    # 3. Cek Status Tren (di atas/bawah 50)
    if rsi_val > mid:
        return f"Bullish (di atas {mid}) - Nilai: {rsi_val:.2f} (Arah: {direction_text})"
    else:
        return f"Bearish (di bawah {mid}) - Nilai: {rsi_val:.2f} (Arah: {direction_text})"

def backtest_oscillator_levels(df):
    """
    Menganalisis data historis untuk menemukan di level osilator (RSI/Stoch)
    mana harga cenderung berbalik (puncak/lembah).
    """
    print("\n--- (Backtest Osilator) Menganalisis Puncak & Lembah Historis ---")
    
    peak_stats = pd.DataFrame()
    valley_stats = pd.DataFrame()
    
    try:
        peaks_indices, _ = find_peaks(df['high'], distance=10, prominence=df['high'].std() * 0.8)
        if peaks_indices.size > 0:
            df_peaks = df.iloc[peaks_indices]
            peak_stats = df_peaks[['rsi_14', 'stochk_10_3_3']].describe(percentiles=[.25, .5, .75])
        else:
            print("Tidak ditemukan puncak historis yang signifikan untuk dianalisis.")
            
        valleys_indices, _ = find_peaks(-df['low'], distance=10, prominence=df['low'].std() * 0.8)
        if valleys_indices.size > 0:
            df_valleys = df.iloc[valleys_indices]
            valley_stats = df_valleys[['rsi_14', 'stochk_10_3_3']].describe(percentiles=[.25, .5, .75])
        else:
            print("Tidak ditemukan lembah historis yang signifikan untuk dianalisis.")
            
    except Exception as e:
        print(f"Error saat backtest osilator: {e}")
        return pd.DataFrame(), pd.DataFrame()
        
    return peak_stats, valley_stats

# --- [PERUBAHAN] Fungsi baru untuk Fibonacci Retracement ---
def calculate_fibonacci_levels(detail_df):
    """
    Menghitung level Fibonacci Retracement berdasarkan
    titik tertinggi dan terendah dari data 'detail'.
    """
    if detail_df.empty:
        return {}

    swing_low = detail_df['low'].min()
    swing_high = detail_df['high'].max()
    is_uptrend = detail_df.iloc[-1]['close'] > detail_df.iloc[0]['close']
    
    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    results = {}

    if is_uptrend:
        results['Status'] = f"Uptrend (Swing Low: {swing_low:.0f}, Swing High: {swing_high:.0f})"
        for level in fib_levels:
            price = swing_high - (swing_high - swing_low) * level
            results[f"{level*100:.1f}%"] = f"{price:.0f}"
    else:
        results['Status'] = f"Downtrend (Swing Low: {swing_low:.0f}, Swing High: {swing_high:.0f})"
        for level in fib_levels:
            price = swing_low + (swing_high - swing_low) * level
            results[f"{level*100:.1f}%"] = f"{price:.0f}"
            
    return results

# --- [PERUBAHAN] Fungsi baru untuk Pivot Points Harian ---
def calculate_pivot_points(df):
    """
    Menghitung level Pivot Points (PP) harian, R1, S1, R2, S2
    berdasarkan data HARI SEBELUMNYA.
    """
    if len(df) < 2:
        return {}
        
    prev_bar = df.iloc[-2]
    h = prev_bar['high']
    l = prev_bar['low']
    c = prev_bar['close']
    
    pp = (h + l + c) / 3
    r1 = (2 * pp) - l
    s1 = (2 * pp) - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    
    results = {
        "Pivot Point (PP)": f"{pp:.0f}",
        "Resistance 1 (R1)": f"{r1:.0f}",
        "Support 1 (S1)": f"{s1:.0f}",
        "Resistance 2 (R2)": f"{r2:.0f}",
        "Support 2 (S2)": f"{s2:.0f}"
    }
    return results