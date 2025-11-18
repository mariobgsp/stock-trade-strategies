import yfinance as yf
import os

# --- Ubah Parameter Di Sini ---

# Ticker saham. 
# PENTING: Untuk saham Indonesia, tambahkan suffix ".JK"
# Contoh: "BBCA.JK", "TLKM.JK", "GOTO.JK"
TICKER = "ADRO.JK"

# Tanggal mulai dan akhir (YYYY-MM-DD)
START_DATE = "2022-11-15"
END_DATE = "2025-11-19" # Biasanya yfinance mengambil data HINGGA (tapi tidak termasuk) tanggal ini.

# Nama file untuk output CSV
NAMA_FILE_OUTPUT = f"stockdata/{TICKER}_history.csv"

# --- Logika Program ---

print(f"Mulai men-download data untuk {TICKER}...")

try:
    # 1. Download data menggunakan yfinance
    data = yf.download(TICKER, start=START_DATE, end=END_DATE)

    # 2. Cek apakah data kosong
    if data.empty:
        print(f"Tidak ada data yang ditemukan untuk {TICKER} pada rentang tanggal tersebut.")
    else:
        # 3. Simpan data (yang berupa DataFrame pandas) ke file CSV
        data.to_csv(NAMA_FILE_OUTPUT)
        
        # Dapatkan path lengkap file untuk konfirmasi
        lokasi_file = os.path.abspath(NAMA_FILE_OUTPUT)
        print(f"\nBerhasil! Data telah disimpan di:")
        print(f"{lokasi_file}")

except Exception as e:
    print(f"\nTerjadi kesalahan saat mengunduh data: {e}")