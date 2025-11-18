import pandas as pd
import pandas_ta as ta

def get_stock_data_from_csv(filepath):
    """
    Membaca data saham dan menghitung indikator teknikal LENGKAP (All-in-One).
    Termasuk: 5-Dimensi (ATR, ADX, RVOL) + Ichimoku + Candle Patterns.
    """
    print(f"Membaca data dari {filepath}...")
    try:
        # Sesuaikan skiprows jika header CSV Anda bergeser
        df = pd.read_csv(filepath, header=0, skiprows=[1, 2])
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {filepath}")
        return None
    except Exception as e:
        print(f"Error saat membaca CSV: {e}")
        return None

    if df.empty:
        print("Error: File CSV kosong.")
        return None

    # --- Standarisasi Kolom ---
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'Date'}, inplace=True)

    date_col = None
    if 'Date' in df.columns: date_col = 'Date'
    elif 'Tanggal' in df.columns: date_col = 'Tanggal'
    
    if not date_col:
        print(f"Error: CSV harus punya kolom Date/Tanggal.")
        return None
        
    try:
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
        df.set_index(date_col, inplace=True)
    except:
        return None

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Kolom {required_cols} wajib ada.")
        return None
    
    for col in required_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_cols, inplace=True)
    
    # --- PERHITUNGAN INDIKATOR (ULTIMATE VERSION) ---
    try:
        # 1. Dasar (RSI, MACD, BB, Stoch, MA)
        df.ta.rsi(close=df['Close'], length=14, append=True)
        df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(close=df['Close'], length=20, std=2, append=True)
        df.ta.stoch(high=df['High'], low=df['Low'], close=df['Close'], k=10, d=3, smooth_k=3, append=True)
        
        df.ta.sma(close=df['Close'], length=10, append=True)
        df.ta.sma(close=df['Close'], length=20, append=True)
        df.ta.sma(close=df['Close'], length=50, append=True)
        df.ta.sma(close=df['Close'], length=200, append=True)
        df['volume_ma20'] = df['Volume'].rolling(window=20).mean()

        # 2. Setup 5-Dimensi (ATR, ADX, RVOL, Slope)
        # Gunakan assign langsung untuk memastikan nama kolom benar
        atr_series = df.ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14)
        df['atrr_14'] = atr_series 

        adx_df = df.ta.adx(high=df['High'], low=df['Low'], close=df['Close'], length=14)
        if not adx_df.empty:
             df['adx_14'] = adx_df.iloc[:, 0] # Ambil kolom pertama (ADX main)

        df['rvol'] = df['Volume'] / df['volume_ma20']

        sma50 = df.ta.sma(close=df['Close'], length=50)
        df['slope_ma50_up'] = sma50 > sma50.shift(5)

        # 3. Ichimoku Cloud (Span A & Span B)
        # Ichimoku return 2 dataframe (conversion/base & spans)
        ichimoku_df, span_df = df.ta.ichimoku(high=df['High'], low=df['Low'], close=df['Close'])
        df = pd.concat([df, ichimoku_df, span_df], axis=1)

        # 4. Candlestick Patterns
        # 100 = Bullish, -100 = Bearish
        df.ta.cdl_pattern(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                          name=["hammer", "doji", "morningstar", "engulfing"], append=True)

        # 5. Finalisasi Nama Kolom (Lowercase)
        df.columns = df.columns.str.lower()
        
        print("Indikator berhasil: Dasar + 5-Dimensi + Ichimoku + Candle Pattern.")
        
    except Exception as e:
        print(f"Error indikator: {e}")
        return None

    return df