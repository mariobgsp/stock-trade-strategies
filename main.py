import argparse
import pandas as pd 

# --- Impor fungsi dari file .py Anda yang lain ---
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
# from charting import create_chart

def main():
    # 1. Setup Argumen Command Line
    parser = argparse.ArgumentParser(description="Backtesting Behavior Saham dari CSV.")
    parser.add_argument('--file', type=str, required=True, 
                        help="Path ke file CSV (Contoh: BBCA.JK_history.csv)")
    
    parser.add_argument('--days', type=int, default=60,
                        help="Jumlah HARI trading terakhir untuk analisis detail (default: 60, sekitar 3 bln)")
                        
    args = parser.parse_args()

    # 2. Ambil dan Hitung Data dari CSV
    data = get_stock_data_from_csv(args.file)
    
    if data is not None:
        
        # Buat DataFrame 'detail'
        days_to_analyze = args.days
        if days_to_analyze > len(data):
            print(f"Peringatan: --days={args.days} terlalu besar, menggunakan seluruh data ({len(data)} hari) sebagai detail.")
            detail_data = data
            days_to_analyze = len(data)
        else:
            detail_data = data.iloc[-days_to_analyze:]
        
        print(f"\n--- Menggunakan {len(data)} bar (Basis) dan {len(detail_data)} bar (Detail {days_to_analyze} hari) ---")

        # 3. Analisis BASIS (Full Data 1-3 Tahun)
        print("\n--- (Analisis Basis) Menghitung S/R Makro & Struktur MA ---")
        clustered_s_base, clustered_r_base, raw_s_base, raw_r_base = find_support_resistance(data)
        market_structure = detect_market_structure(data)
        
        # Panggil Backtest Osilator
        peak_stats, valley_stats = backtest_oscillator_levels(data) 
        
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

        # 4. Analisis DETAIL (Data 'X' Hari Terakhir)
        print("\n--- (Analisis Detail) Menghitung S/R Mikro & Tren Jangka Pendek ---")
        clustered_s_detail, clustered_r_detail, raw_s_detail, raw_r_detail = find_support_resistance(detail_data)
        
        # 5. Analisis Gabungan (Momentum & Volume)
        short_term_trend = analyze_short_term_trend(detail_data, days_to_analyze) 
        vol_today, vol_short_term = analyze_volume_profile(detail_data) 
        daily_momentum = analyze_daily_momentum(data)
        rsi_behavior = analyze_rsi_behavior(detail_data)
        
        fib_levels = calculate_fibonacci_levels(detail_data)
        pivot_levels = calculate_pivot_points(data)

        # 6. Jalankan Pemindai Sinyal Historis
        df_signals = scan_for_signals(data, clustered_s_base, clustered_r_base)
        
        # 7. Panggil fungsi analisis perilaku historis
        historical_analysis = analyze_historical_performance(data, df_signals)
        print("\n--- ANALISIS PERFORMA SINYAL HISTORIS (10 Hari ke Depan) ---")
        if historical_analysis.empty:
            print("Tidak ada sinyal BTD/STR historis yang ditemukan untuk dianalisis.")
        else:
            print(historical_analysis.to_string())

        # 8. Ambil Rangkuman Laporan Hari Terakhir
        summary_report = analyze_behavior(
            data, 
            clustered_s_base, clustered_r_base, raw_s_base, raw_r_base,
            clustered_s_detail, clustered_r_detail, raw_s_detail, raw_r_detail,
            market_structure, 
            short_term_trend,
            vol_today,
            vol_short_term,
            daily_momentum,
            rsi_behavior,
            fib_levels,
            pivot_levels,
            days_to_analyze 
        )
        
        # 9. Cetak Rangkuman Laporan
        print("\n--- RANGKUMAN ANALISIS HARI TERAKHIR ---")
        if summary_report:
            for key, value in summary_report.items():
                if isinstance(value, list) or isinstance(value, dict): 
                    print(f"  - {key}:")
                    
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"    -> {k}: {v}")
                    else:
                        for item in value:
                            print(f"    -> {item}")
                else:
                    print(f"  - {key}: {value}")
        
        
        # 10. Dapatkan Rekomendasi Trade
        print("\n--- (REKOMENDASI) Menggunakan S/R Jangka Panjang (Basis) ---")
        # --- [PERUBAHAN] Kirim raw_s_detail dan raw_r_detail ke 'recommend_trade' ---
        trade_recommendation = recommend_trade(
            data, 
            clustered_s_base, clustered_r_base, 
            raw_s_base, raw_r_base, 
            raw_s_detail, raw_r_detail, # <-- Argumen baru untuk Fib Ext
            market_structure, 
            min_rr_ratio=1.5
        )
        # --- [AKHIR PERUBAHAN] ---
        
        # 11. Cetak Rekomendasi Trade
        print("\n--- REKOMENDASI TRADE HARI INI ---")
        if trade_recommendation:
            for key, value in trade_recommendation.items():
                print(f"  - {key}: {value}")

        
        # 12. Buat dan simpan visualisasi (Masih dikomentari)
        # output_filename = args.file.replace('.csv', '_analysis_chart.png').replace('.CSV', '_analysis_chart.png')
        # create_chart(data, clustered_s_base, clustered_r_base, raw_s_base, raw_r_base, df_signals, market_structure, filename=output_filename)

# Ini adalah entry point program
if __name__ == "__main__":
    main()