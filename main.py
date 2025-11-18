import argparse
import pandas as pd 

# --- Impor Modul Custom ---
from data_loader import get_stock_data_from_csv
from core_analysis import (
    find_support_resistance, 
    detect_market_structure,
    analyze_short_term_trend, 
    analyze_volume_profile,
    analyze_daily_momentum,
    analyze_rsi_behavior,
    backtest_oscillator_levels,
    calculate_fibonacci_levels,
    calculate_pivot_points
)
from reporting import (
    scan_for_signals, 
    analyze_behavior,
    recommend_trade,
    analyze_historical_performance
)

def main():
    # 1. Setup Argumen
    parser = argparse.ArgumentParser(description="Backtesting Behavior Saham (Ultimate Setup).")
    parser.add_argument('--file', type=str, required=True, 
                        help="Path ke file CSV (Contoh: BBCA.JK_history.csv)")
    parser.add_argument('--days', type=int, default=60,
                        help="Jumlah HARI trading terakhir untuk analisis detail (default: 60)")
                        
    args = parser.parse_args()

    # 2. Load Data
    data = get_stock_data_from_csv(args.file)
    
    if data is not None:
        # --- FIX PENTING DISINI ---
        # Kita pisahkan data: 
        # 'data' = Full (termasuk 26 hari masa depan Ichimoku) untuk reporting.py
        # 'data_hist' = Hanya history (s/d hari ini) untuk slicing detail
        data_hist = data.dropna(subset=['close'])
        
        # Setup DataFrame 'detail' mengambil dari data_hist (bukan data raw)
        days_to_analyze = args.days
        if days_to_analyze > len(data_hist):
            print(f"Peringatan: Data hanya tersedia {len(data_hist)} hari.")
            detail_data = data_hist
            days_to_analyze = len(data_hist)
        else:
            detail_data = data_hist.iloc[-days_to_analyze:]
        
        print(f"\n--- Menggunakan {len(data_hist)} bar (Basis) dan {len(detail_data)} bar (Detail) ---")

        # 3. Analisis Inti (Support/Resistance & Market Structure)
        # Gunakan data_hist agar find_peaks tidak bingung dengan NaN
        print("\n--- (Analisis Basis) Menghitung S/R & Struktur ---")
        clustered_s_base, clustered_r_base, raw_s_base, raw_r_base = find_support_resistance(data_hist)
        market_structure = detect_market_structure(data_hist)
        
        # Backtest Osilator (Gunakan data historis)
        peak_stats, valley_stats = backtest_oscillator_levels(data_hist)
        
        print("\n--- HASIL BACKTEST OSILATOR (LEVEL PEMBALIKAN HISTORIS) ---")
        if not peak_stats.empty:
            print("\n== Statistik Indikator saat Puncak (Turn Bearish) ==")
            print(peak_stats.to_string())
        else:
            print("\n== Statistik Puncak: Tidak ada data ==")
            
        if not valley_stats.empty:
            print("\n== Statistik Indikator saat Lembah (Turn Bullish) ==")
            print(valley_stats.to_string())
        else:
            print("\n== Statistik Lembah: Tidak ada data ==")

        # 4. Analisis Detail (Mikro)
        print("\n--- (Analisis Detail) Menghitung Tren Jangka Pendek ---")
        clustered_s_detail, clustered_r_detail, raw_s_detail, raw_r_detail = find_support_resistance(detail_data)
        
        # Analisis Tambahan
        short_term_trend = analyze_short_term_trend(detail_data, days_to_analyze) 
        vol_today, vol_short_term = analyze_volume_profile(detail_data) 
        daily_momentum = analyze_daily_momentum(data_hist)
        rsi_behavior = analyze_rsi_behavior(detail_data)
        
        fib_levels = calculate_fibonacci_levels(detail_data)
        pivot_levels = calculate_pivot_points(data_hist)

        # 5. Scan Sinyal Historis (Kirim full 'data' karena reporting.py sudah handle NaN)
        df_signals = scan_for_signals(data, clustered_s_base, clustered_r_base)
        
        historical_analysis = analyze_historical_performance(data, df_signals)
        print("\n--- ANALISIS PERFORMA SINYAL HISTORIS (10 Hari ke Depan) ---")
        if historical_analysis.empty:
            print("Belum ada data sinyal historis yang cukup.")
        else:
            print(historical_analysis.to_string())

        # 6. Laporan Harian (Summary)
        # Kirim full 'data' agar bisa melihat Future Cloud Ichimoku
        summary_report = analyze_behavior(
            data, 
            clustered_s_base, clustered_r_base, raw_s_base, raw_r_base,
            clustered_s_detail, clustered_r_detail, raw_s_detail, raw_r_detail,
            market_structure, short_term_trend, vol_today, vol_short_term,
            daily_momentum, rsi_behavior, fib_levels, pivot_levels, days_to_analyze 
        )
        
        if summary_report:
            for key, value in summary_report.items():
                if isinstance(value, list) or isinstance(value, dict): 
                    print(f"  - {key}:")
                    if isinstance(value, dict):
                        for k, v in value.items(): print(f"    -> {k}: {v}")
                    else:
                        for item in value: print(f"    -> {item}")
                else:
                    print(f"  - {key}: {value}")
        
        # 7. Rekomendasi Trade
        trade_recommendation = recommend_trade(
            data, 
            clustered_s_base, clustered_r_base, 
            raw_s_base, raw_r_base, 
            raw_s_detail, raw_r_detail,
            market_structure, 
            min_rr_ratio=1.5
        )
        
        if trade_recommendation:
            for key, value in trade_recommendation.items():
                print(f"  - {key}: {value}")

if __name__ == "__main__":
    main()