import pandas as pd

def scan_for_signals(df, clustered_supports, clustered_resistances):
    """
    Memindai seluruh riwayat data (timeframe) untuk menemukan 
    dan melabeli sinyal trading spesifik (TERMASUK Reversal & Buy The Dip).
    
    [VERSI DIPERBARUI]: BTD dan STR sekarang membutuhkan konfirmasi osilator.
    """
    print("\n--- (DETECTOR) Memulai Pemindaian Sinyal Historis ---")
    
    signals_found = []
    support_levels = clustered_supports['Level'].values
    resistance_levels = clustered_resistances['Level'].values
    proximity = 0.02 # Toleransi 2%
    
    if len(df) < 201:
        print("Data tidak cukup untuk pemindaian MA 200.")
        return pd.DataFrame() 

    for i in range(200, len(df)):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        
        # --- KUMPULKAN STATUS INDIKATOR (BULLISH) ---
        rsi_val = bar['rsi_14']
        rsi_bullish_cross = prev_bar['rsi_14'] < 30 and rsi_val > 30
        
        macd_val = bar['macd_12_26_9']
        macds_val = bar['macds_12_26_9']
        macd_bullish_cross = prev_bar['macd_12_26_9'] < prev_bar['macds_12_26_9'] and macd_val > macds_val

        stochk_val = bar['stochk_10_3_3']
        stochd_val = bar['stochd_10_3_3']
        stoch_bullish_cross = prev_bar['stochk_10_3_3'] < prev_bar['stochd_10_3_3'] and stochk_val > stochd_val
        stoch_leaving_oversold = stoch_bullish_cross and prev_bar['stochk_10_3_3'] < 30 
        
        is_oscillator_buy_confirm = stoch_leaving_oversold or rsi_bullish_cross
        
        # --- KUMPULKAN STATUS INDIKATOR (BEARISH) ---
        rsi_bearish_cross = prev_bar['rsi_14'] > 70 and rsi_val < 70
        stoch_bearish_cross = prev_bar['stochk_10_3_3'] > prev_bar['stochd_10_3_3'] and stochk_val < stochd_val
        stoch_leaving_overbought = stoch_bearish_cross and prev_bar['stochk_10_3_3'] > 70
        
        is_oscillator_sell_confirm = stoch_leaving_overbought or rsi_bearish_cross
        
        # --- SISA INDIKATOR ---
        volume_breakout = bar['volume'] > (bar['volume_ma20'] * 1.5)
        ma50_val = bar['sma_50']
        ma200_val = bar['sma_200']
        ma50_prev = prev_bar['sma_50']
        ma200_prev = prev_bar['sma_200']
        
        # Filter Whipsaw (Fake Cross)
        is_golden_cross = (ma50_prev < ma200_prev and ma50_val > ma200_val) and (bar['close'] > ma50_val) 
        is_death_cross = (ma50_prev > ma200_prev and ma50_val < ma200_val) and (bar['close'] < ma50_val)
        
        # FILTER TREN UTAMA
        is_main_uptrend = bar['close'] > ma200_val
        
        # Pola Reversal Candlestick
        is_bearish_engulfing = (prev_bar['close'] > prev_bar['open']) and \
                               (bar['close'] < bar['open']) and \
                               (bar['close'] < prev_bar['open']) and \
                               (bar['open'] > prev_bar['close'])
        is_bullish_engulfing = (prev_bar['close'] < prev_bar['open']) and \
                               (bar['close'] > bar['open']) and \
                               (bar['close'] > prev_bar['open']) and \
                               (bar['open'] < prev_bar['close'])
        
        # --- LOGIKA PENCARIAN & PELABELAN "SPOT" ---
        
        # 1. SINYAL TREN (Golden/Death Cross)
        if is_golden_cross:
            signals_found.append({
                "Tanggal": bar.name, "Sinyal (Label)": "TREND - GOLDEN CROSS",
                "Harga": bar['close'], 
                "Detail": "MA 50 cross MA 200 (Dikonfirmasi harga > MA 50)"
            })
            continue 
        
        if is_death_cross:
            signals_found.append({
                "Tanggal": bar.name, "Sinyal (Label)": "TREND - DEATH CROSS",
                "Harga": bar['close'], 
                "Detail": "MA 50 cross MA 200 (Dikonfirmasi harga < MA 50)"
            })
            continue 
            
        # 2. SINYAL REVERSAL (Engulfing)
        if is_bullish_engulfing:
            for sup_level in support_levels:
                is_near_support = abs(bar['low'] - sup_level) / sup_level < proximity
                if is_near_support:
                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": "REVERSAL - BULLISH ENGULFING",
                        "Harga": bar['close'], "Detail": f"Pola Engulfing Bullish di S {sup_level:.0f}"
                    })
                    break 
            if signals_found and signals_found[-1]["Tanggal"] == bar.name: continue 
            
        if is_bearish_engulfing:
            for res_level in resistance_levels:
                is_near_resistance = abs(bar['high'] - res_level) / res_level < proximity
                if is_near_resistance:
                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": "REVERSAL - BEARISH ENGULFING",
                        "Harga": bar['close'], "Detail": f"Pola Engulfing Bearish di R {res_level:.0f}"
                    })
                    break
            if signals_found and signals_found[-1]["Tanggal"] == bar.name: continue

        
        # 3. SINYAL FILTER TREN
        if is_main_uptrend:
            
            # SINYAL "BUY THE DIP" (Pantulan MA 50)
            is_dip_buy = (bar['low'] < ma50_val) and (bar['close'] > ma50_val) and (bar['close'] > bar['open'])
            if is_dip_buy and is_oscillator_buy_confirm: # <-- Ditambah konfirmasi
                signals_found.append({
                    "Tanggal": bar.name, "Sinyal (Label)": "BUY THE DIP (MA 50)",
                    "Harga": bar['close'], "Detail": f"Pantul MA 50 (Dikonfirmasi Stoch/RSI Cross)"
                })
                continue
                
            # SINYAL "ALL VALID CROSSOVER" (BUY)
            if rsi_bullish_cross and macd_bullish_cross and stoch_leaving_oversold and volume_breakout:
                signals_found.append({
                    "Tanggal": bar.name, "Sinyal (Label)": "BUY - ALL VALID CROSSOVER",
                    "Harga": bar['close'], "Detail": "RSI keluar OS, MACD Cross, Stoch Cross, Vol Spike"
                })
                continue 
            
            # SINYAL "BREAKOUT RESISTANCE" (BUY)
            for res_level in resistance_levels:
                is_breakout = prev_bar['close'] < res_level and bar['close'] > res_level
                
                is_not_rsi_overbought = rsi_val < 70 
                is_not_stoch_overbought = stochk_val < 80

                if is_breakout and volume_breakout and is_not_rsi_overbought and is_not_stoch_overbought:
                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": "BREAKOUT - RESISTANCE",
                        "Harga": bar['close'], 
                        "Detail": f"Tembus Res {res_level:.0f} (Vol Kuat, RSI < 70, Stoch < 80)"
                    })
                    break
            
            # SINYAL "BOUNCE / RbS FLIP" (BUY)
            for sup_level in support_levels:
                is_near_support = abs(bar['low'] - sup_level) / sup_level < proximity
                is_bounce = bar['close'] > bar['open'] 
                is_confirmed_buy = rsi_bullish_cross or stoch_leaving_oversold
                
                if is_near_support and is_bounce and is_confirmed_buy:
                    is_rbs_flip = (abs(clustered_resistances['Level'] - sup_level) / sup_level < proximity).any()
                    signal_label = "BOUNCE - RbS FLIP" if is_rbs_flip else "BOUNCE - SUPPORT"
                    detail = f"Pantulan dari Zona Flip {sup_level:.0f}" if is_rbs_flip else f"Pantulan dari Sup {sup_level:.0f}"
                    
                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": signal_label, "Harga": bar['close'], "Detail": detail
                    })
                    break 

        else: # (if not is_main_uptrend) -> Tren Turun
            
            # SINYAL "SELL THE RALLY" (BARU)
            is_rally_sell = (bar['high'] > ma50_val) and (bar['close'] < ma50_val) and (bar['close'] < bar['open'])
            if is_rally_sell and is_oscillator_sell_confirm: # <-- Ditambah konfirmasi
                signals_found.append({
                    "Tanggal": bar.name, "Sinyal (Label)": "SELL THE RALLY (MA 50)",
                    "Harga": bar['close'], "Detail": f"Tolak MA 50 (Dikonfirmasi Stoch/RSI Cross)"
                })
                continue

            # SINYAL "REJECTION / SbR FLIP" (SELL)
            for res_level in resistance_levels:
                is_near_resistance = abs(bar['high'] - res_level) / res_level < proximity
                is_rejection = bar['close'] < bar['open'] 
                is_confirmed_sell = rsi_bearish_cross or stoch_leaving_overbought
                
                if is_near_resistance and is_rejection and is_confirmed_sell:
                    is_sbr_flip = (abs(clustered_supports['Level'] - res_level) / res_level < proximity).any()
                    signal_label = "REJECTION - SbR FLIP" if is_sbr_flip else "REJECTION - RESISTANCE"
                    detail = f"Ditolak dari Zona Flip {res_level:.0f}" if is_sbr_flip else f"Ditolak dari Res {res_level:.0f}"

                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": signal_label, "Harga": bar['close'], "Detail": detail
                    })
                    break

    # --- Cetak Hasil Pemindaian ---
    if not signals_found:
        print("Tidak ditemukan sinyal historis signifikan yang sesuai kriteria.")
        return pd.DataFrame()
    else:
        df_signals = pd.DataFrame(signals_found).set_index('Tanggal')
        print(f"Ditemukan total {len(df_signals)} 'spot' historis berlabel (sudah difilter tren):")
        print(df_signals.to_string())
        
        return df_signals


def analyze_behavior(df, 
                     clustered_s_base, clustered_r_base, raw_s_base, raw_r_base,
                     clustered_s_detail, clustered_r_detail, raw_s_detail, raw_r_detail,
                     market_structure, 
                     short_term_trend,
                     vol_today,
                     vol_short_term,
                     daily_momentum,
                     rsi_behavior,
                     fib_levels,   # <-- Argumen baru
                     pivot_levels, # <-- Argumen baru
                     num_days):
    """
    Menganalisis perilaku HARI TERAKHIR dan MENGEMBALIKAN rangkuman (dictionary).
    """
    print("\n\n--- (ANALISIS HARI INI) Menganalisis Perilaku Saham ---")
    
    summary_findings = {}
    proximity = 0.02 
    last_bar = df.iloc[-1]
    
    summary_findings["Market Structure (MA)"] = market_structure
    summary_findings["Tren Jangka Pendek (Detail)"] = short_term_trend
    summary_findings["Momentum Harian (OHLC)"] = daily_momentum
    summary_findings["Perilaku RSI (Detail)"] = rsi_behavior
    summary_findings["Analisis Volume (Hari Ini)"] = vol_today
    summary_findings["Analisis Volume (Detail)"] = vol_short_term
    summary_findings["Harga Saat Ini"] = f"{last_bar['close']:.0f}"
    
    # Tambahkan hasil Fib & Pivot ke laporan
    summary_findings[f"Proyeksi Fibonacci (Detail {num_days} Hari)"] = fib_levels
    summary_findings["Proyeksi Pivot Points (Harian)"] = pivot_levels
    
    # Laporan S/R Basis (Jangka Panjang)
    sr_base_list = ["--- S/R JANGKA PANJANG (BASIS) ---"]
    if clustered_r_base.empty:
        sr_base_list.append("  R (Basis): Tidak ada")
    else:
        for idx, row in clustered_r_base.head(3).iterrows():
            sr_base_list.append(f"  R (Basis): {row['Level']:.0f} (Hits: {row['Hits']}, {row['Kekuatan']})")
    
    if clustered_s_base.empty:
        sr_base_list.append("  S (Basis): Tidak ada")
    else:
        for idx, row in clustered_s_base.head(3).iterrows():
            sr_base_list.append(f"  S (Basis): {row['Level']:.0f} (Hits: {row['Hits']}, {row['Kekuatan']})")
    summary_findings[f"S/R Jangka Panjang (Basis)"] = sr_base_list

    # Laporan S/R Detail (Jangka Pendek)
    sr_detail_list = [f"--- S/R JANGKA PENDEK (DETAIL {num_days} HARI) ---"]
    if clustered_r_detail.empty:
        sr_detail_list.append("  R (Detail): Tidak ada")
    else:
        for idx, row in clustered_r_detail.head(3).iterrows():
            sr_detail_list.append(f"  R (Detail): {row['Level']:.0f} (Hits: {row['Hits']}, {row['Kekuatan']})")

    if clustered_s_detail.empty:
        sr_detail_list.append("  S (Detail): Tidak ada")
    else:
        for idx, row in clustered_s_detail.head(3).iterrows():
            sr_detail_list.append(f"  S (Detail): {row['Level']:.0f} (Hits: {row['Hits']}, {row['Kekuatan']})")
    summary_findings[f"S/R Jangka Pendek (Detail {num_days} Hari)"] = sr_detail_list

    # --- Logika Kesimpulan ---
    recent_resistances = raw_r_base[raw_r_base.index < last_bar.name]
    if recent_resistances.empty:
        print("\nKesimpulan: Tidak ada data resistance historis yang cukup.")
        return None 

    last_resistance_level = recent_resistances.iloc[-1]
    
    is_sideways = ("SIDEWAYS" in market_structure) 
    is_uptrend = ("UPTREND" in market_structure) 
    is_downtrend = ("DOWNTREND" in market_structure)
    
    is_breakout_resistance = last_bar['close'] > last_resistance_level
    is_volume_strong = "Akumulasi Kuat" in vol_today
    is_breakout_bb_upper = last_bar['close'] > last_bar['bbu_20_2.0_2.0']
    
    kesimpulan = "Perilaku tidak terdeteksi."
    
    if is_uptrend and is_breakout_resistance and is_volume_strong:
        kesimpulan = "**Pola Terdeteksi: Melanjutkan Uptrend, Breakout Resistance (Tervalidasi Volume)**"
    elif is_downtrend and is_breakout_resistance:
        kesimpulan = f"**Peringatan: Tren {market_structure}. Breakout saat ini berisiko 'Bull Trap'.**"
    elif is_breakout_resistance and not is_volume_strong:
        kesimpulan = "**Peringatan: Potensi False Breakout (Volume Lemah)**"
    elif is_uptrend and not is_breakout_resistance:
         kesimpulan = f"**Pola Terdetaksi: {market_structure}, bergerak menuju Res {last_resistance_level:.0f}**"
    elif is_downtrend:
        kesimpulan = f"**Pola Terdeteksi: Masih dalam Fase {market_structure}.**"
    
    summary_findings["Kesimpulan Utama"] = kesimpulan
    volatility_check = last_bar['bbb_20_2.0_2.0'] < (df['bbb_20_2.0_2.0'].mean() * 0.7)
    summary_findings["Volatilitas (BB Width)"] = 'Rendah (Squeeze)' if volatility_check else 'Normal/Tinggi'

    ma_list = [10, 20, 50, 100, 200]
    ma_detail_list = []
    all_ma_bullish = True
    ma_summary_text = ""
    
    for ma_len in ma_list:
        ma_col = f'sma_{ma_len}'
        if ma_col in last_bar:
            ma_val = last_bar[ma_col]
            status = "di ATAS" if last_bar['close'] > ma_val else "di BAWAH"
            if last_bar['close'] < ma_val:
                all_ma_bullish = False
            ma_detail_list.append(f"MA {ma_len}: {ma_val:.0f} (Harga {status} MA)")
        else:
            ma_detail_list.append(f"MA {ma_len}: Tidak terhitung (data kurang)")
            all_ma_bullish = False
            
    if all_ma_bullish:
        ma_summary_text = "TREN SANGAT KUAT (Harga di atas semua MA)"
    elif last_bar['close'] > last_bar['sma_50']:
        ma_summary_text = "TREN JANGKA MENENGAH NAIK (Harga di atas MA 50)"
    else:
        ma_summary_text = "TREN JANGKA MENENGAH TURUN (Harga di bawah MA 50)"

    summary_findings["Status MA (Ringkasan)"] = ma_summary_text
    summary_findings["Status MA (Detail)"] = ma_detail_list 

    return summary_findings


def recommend_trade(df, clustered_supports, clustered_resistances, 
                     raw_supports, raw_resistances, 
                     raw_s_detail, raw_r_detail, # <-- Argumen baru
                     market_structure, min_rr_ratio=1.5):
    """
    Menganalisis HARI TERAKHIR untuk rekomendasi trade.
    (VERSI UPDATE: TP menggunakan Fib Extension, SL berdasarkan support TERDEKAT)
    """
    print("\n--- (REKOMENDASI TRADE) Menganalisis Sinyal Beli Hari Ini ---")
    
    last_bar = df.iloc[-1]
    current_price = last_bar['close']
    
    recommendation = {
        "Rekomendasi": "JANGAN BELI (Tahan/Tunggu)",
        "Alasan": "Tidak ada sinyal beli yang jelas atau R/R tidak memenuhi.",
        "Harga Saat Ini": f"{current_price:.0f}",
        "Area Beli (Entry)": "N/A",
        "Stop Loss (SL)": "N/A",
        "Take Profit (TP)": "N/A",
        "Risk/Reward (R/R)": "N/A"
    }

    # --- 2. Filter Kondisi WAJIB ---
    if "DOWNTREND" in market_structure:
        recommendation["Alasan"] = f"Kondisi pasar {market_structure}, risiko terlalu tinggi."
        return recommendation

    if last_bar['rsi_14'] > 70 or last_bar['stochk_10_3_3'] > 80:
        recommendation["Alasan"] = "Indikator Overbought (RSI > 70 atau Stoch > 80)."
        return recommendation
        
    # --- 3. Tentukan Level SL (Support TERDEKAT dari data BASIS) ---
    valid_strong_supports_df = clustered_supports[clustered_supports['Level'] < current_price]
    s_raw_recent = raw_supports.iloc[-3:]
    valid_soft_supports_series = s_raw_recent[s_raw_recent < current_price]
    
    nearest_support_level = 0
    nearest_support_type = "N/A"

    nearest_strong_support = 0
    if not valid_strong_supports_df.empty:
        nearest_strong_support = valid_strong_supports_df['Level'].max()

    nearest_soft_support = 0
    if not valid_soft_supports_series.empty:
        nearest_soft_support = valid_soft_supports_series.max()

    if nearest_strong_support == 0 and nearest_soft_support == 0:
        recommendation["Alasan"] = "Tidak ada support (strong/soft) terdeteksi di bawah harga saat ini untuk basis SL."
        return recommendation
    elif nearest_soft_support > nearest_strong_support:
        nearest_support_level = nearest_soft_support
        nearest_support_type = "Soft (Raw Pivot)"
    else:
        nearest_support_level = nearest_strong_support
        nearest_support_type = "Strong (Cluster)"

    entry_support_level = nearest_support_level
    
    # --- 4. [LOGIKA TP BARU] Tentukan Level TP (Fib Extension atau Fallback) ---
    tp_price = 0
    tp_basis = "N/A"

    # Coba hitung Fib Extension 1.618 (membutuhkan 3 pivot: Low-High-Low)
    try:
        if len(raw_s_detail) < 2 or len(raw_r_detail) < 1:
            raise ValueError("Tidak cukup data pivot detail (L-H-L)")

        # Tentukan titik A, B, C untuk pola L-H-L (Uptrend)
        # Ambil dari data S/R mentah JANGKA PENDEK (detail)
        A_low = raw_s_detail.iloc[-2]
        B_high = raw_r_detail.iloc[-1]
        C_low = raw_s_detail.iloc[-1]

        # Validasi pola (tanggal harus berurutan L-H-L dan C adalah koreksi)
        if (A_low.name < B_high.name) and (B_high.name < C_low.name) and (C_low < B_high):
            swing_range = B_high - A_low
            fib_target_1618 = C_low + (swing_range * 1.618)
            
            # Target harus di atas harga saat ini
            if fib_target_1618 > current_price:
                tp_price = fib_target_1618
                tp_basis = f"Fib Ext 1.618 ({tp_price:.0f})"
        
        if tp_price == 0: # Jika pola L-H-L tidak valid
             raise ValueError("Pola L-H-L tidak valid atau target di bawah harga")

    except Exception as e:
        # Fallback: Jika Fib Ext gagal, gunakan Resistance Kuat (Basis) TERDEKAT
        # print(f"Info: Gagal hitung Fib Ext ({e}), fallback ke R terdekat.")
        
        # Gunakan clustered_resistances (dari data BASIS)
        valid_strong_resistances_df = clustered_resistances[clustered_resistances['Level'] > current_price]
        if valid_strong_resistances_df.empty:
            recommendation["Alasan"] = "Fib Ext gagal & Tidak ada R terdekat untuk basis TP."
            return recommendation
            
        # Ambil R terdekat (harga terendah yang masih di atas harga saat ini)
        nearest_strong_resistance = valid_strong_resistances_df['Level'].min()
        tp_price = nearest_strong_resistance * 0.99 # Ambil 1% di bawah R
        tp_basis = f"Nearest Strong R ({nearest_strong_resistance:.0f})"

    # --- 5. Hitung Risk/Reward (R/R) ---
    sl_price = nearest_support_level * 0.98 
    
    risk_per_share = current_price - sl_price
    reward_per_share = tp_price - current_price
    
    if risk_per_share <= 0 or reward_per_share <= 0:
        recommendation["Alasan"] = f"Logika S/R tidak valid (Harga: {current_price:.0f}, SL: {sl_price:.0f}, TP: {tp_price:.0f})."
        return recommendation

    rr_ratio = reward_per_share / risk_per_share

    # --- 6. Buat Keputusan Akhir ---
    is_near_entry_support = (current_price - entry_support_level) / entry_support_level < 0.03
    sma_50_val = last_bar['sma_50']
    is_ma50_bounce = (last_bar['low'] < sma_50_val) and (current_price > sma_50_val) and (last_bar['close'] > last_bar['open'])

    is_rr_good = rr_ratio >= min_rr_ratio
    is_confirmed = (last_bar['close'] > last_bar['open']) or (last_bar['macd_12_26_9'] > last_bar['macds_12_26_9'])

    recommendation["Area Beli (Entry)"] = f"~{entry_support_level:.0f} (S/R {nearest_support_type}) ATAU ~{sma_50_val:.0f} (MA 50)"
    recommendation["Stop Loss (SL)"] = f"~{sl_price:.0f} (Basis: S {nearest_support_type} {entry_support_level:.0f})"
    recommendation["Take Profit (TP)"] = f"~{tp_price:.0f} (Basis: {tp_basis})"
    recommendation["Risk/Reward (R/R)"] = f"1 : {rr_ratio:.2f}"
    
    if not is_rr_good:
        recommendation["Alasan"] = f"Risk/Reward tidak menarik (Hanya 1:{rr_ratio:.2f})."
        return recommendation 

    if not is_confirmed:
        recommendation["Alasan"] = "Tidak ada konfirmasi sinyal (candle merah/MACD bearish)."
        return recommendation 

    if is_near_entry_support:
        recommendation["Rekomendasi"] = "REKOMENDASI BELI"
        recommendation["Alasan"] = f"Harga dekat Area Beli S/R ({entry_support_level:.0f}), R/R bagus, Sinyal terkonfirmasi."
    elif is_ma50_bounce:
        recommendation["Rekomendasi"] = "REKOMENDASI BELI"
        recommendation["Alasan"] = f"Harga memantul (Bounce) dari SMA 50, R/R bagus, Sinyal terkonfirmasi."
    else:
        recommendation["Alasan"] = f"Harga 'nanggung', jauh dari Area Beli S/R ({entry_support_level:.0f}) dan tidak memantul dari MA50."
         
    return recommendation


def analyze_historical_performance(df, df_signals, look_ahead_days=10):
    """
    Menganalisis sinyal BTD dan STR dari df_signals untuk melihat 
    perilaku harga 'look_ahead_days' hari ke depan.
    """
    
    target_signals = ["BUY THE DIP (MA 50)", "SELL THE RALLY (MA 50)"]
    df_signals_filtered = df_signals[df_signals['Sinyal (Label)'].isin(target_signals)]

    if df_signals_filtered.empty:
        return pd.DataFrame()

    results = []

    for signal_date, signal_row in df_signals_filtered.iterrows():
        try:
            signal_idx = df.index.get_loc(signal_date)
            entry_price = signal_row['Harga'] # Harga saat sinyal
            
            future_idx = min(signal_idx + look_ahead_days, len(df) - 1)
            future_bar = df.iloc[future_idx]
            
            future_price = future_bar['close']
            pct_change = (future_price - entry_price) / entry_price
            
            results.append({
                "Sinyal": signal_row['Sinyal (Label)'],
                "Tanggal": signal_date.date(),
                f"Perubahan {look_ahead_days} Hari": pct_change
            })

        except KeyError:
            continue 
        except Exception as e:
            print(f"Error saat analisis historis: {e}")
            continue

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    
    df_summary = df_results.groupby('Sinyal')[f"Perubahan {look_ahead_days} Hari"].agg(
        Rata_Rata_Perubahan=lambda x: f"{x.mean():.2%}",
        Jumlah_Sinyal='count',
        Sinyal_Positif=lambda x: (x > 0).sum(),
        Sinyal_Negatif=lambda x: (x < 0).sum()
    )
    
    return df_summary