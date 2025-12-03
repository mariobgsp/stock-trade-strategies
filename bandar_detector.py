import argparse
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bandar Detector - Smart Money Analysis Tool')
    parser.add_argument('ticker', type=str, help='Stock Ticker Symbol (e.g., AAPL, BBCA.JK)')
    return parser.parse_args()

def fetch_data(ticker):
    print(f"[*] Fetching data for {ticker}...")
    try:
        # Fetch 2 years to ensure enough data for 255-day EMA if possible
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=False)
        
        # Handle yfinance MultiIndex columns (Price, Ticker) -> (Price)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            print(f"[!] Error: No data found for ticker '{ticker}'.")
            sys.exit(1)
            
        return df
    except Exception as e:
        print(f"[!] Error fetching data: {e}")
        sys.exit(1)

def calculate_indicators(df):
    # --- Data Prep & IPO Handling ---
    data_len = len(df)
    is_ipo = data_len < 500 # Approx 2 years
    short_window = 20
    long_window = 255
    
    # Adjust windows for very new listings to avoid NaN everywhere
    if data_len < short_window:
        short_window = max(5, data_len) # Fallback for very short data
    
    # --- 1. VWAP (Rolling 20) ---
    # Typical Price
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    # Rolling Cumulative Price * Vol / Rolling Cumulative Vol
    v_mw = df['Volume'].rolling(window=short_window, min_periods=1)
    df['VWAP'] = (df['TP'] * df['Volume']).rolling(window=short_window, min_periods=1).sum() / v_mw.sum()

    # --- 2. CMF (Chaikin Money Flow - 20 day) ---
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Handle division by zero (High == Low)
    high_low = df['High'] - df['Low']
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low.replace(0, np.nan)
    mf_multiplier = mf_multiplier.fillna(0)
    mf_volume = mf_multiplier * df['Volume']
    
    df['CMF'] = mf_volume.rolling(window=short_window, min_periods=1).sum() / df['Volume'].rolling(window=short_window, min_periods=1).sum()

    # --- 3. NVI (Negative Volume Index) ---
    # Start at 1000. Only change if Vol < Prev Vol
    nvi = [1000.0]
    df_vol = df['Volume'].values
    df_close = df['Close'].values
    
    for i in range(1, len(df)):
        if df_vol[i] < df_vol[i-1]:
            # Pct change
            pct_change = (df_close[i] - df_close[i-1]) / df_close[i-1]
            nvi_val = nvi[-1] * (1.0 + pct_change)
            nvi.append(nvi_val)
        else:
            nvi.append(nvi[-1])
            
    df['NVI'] = nvi
    
    # NVI EMA (255)
    # If IPO < 255 days, use a smaller span or cumulative mean logic via pandas ewm
    ema_span = long_window if data_len >= long_window else data_len
    df['NVI_EMA'] = df['NVI'].ewm(span=ema_span, adjust=False, min_periods=1).mean()

    # --- 4. OBV (On-Balance Volume) ---
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_EMA'] = df['OBV'].ewm(span=short_window, adjust=False, min_periods=1).mean()

    # --- 5. Amihud Absorption (Custom Formula) ---
    # Volume / (Absolute Price Change + 0.00001)
    price_change_abs = df['Close'].diff().abs()
    df['Absorption_Raw'] = df['Volume'] / (price_change_abs + 0.00001)
    
    # Z-Score (20 day window)
    abs_mean = df['Absorption_Raw'].rolling(window=short_window, min_periods=5).mean()
    abs_std = df['Absorption_Raw'].rolling(window=short_window, min_periods=5).std()
    
    # Avoid division by zero in Z-score
    df['Absorption_Z'] = ((df['Absorption_Raw'] - abs_mean) / abs_std).fillna(0)

    # --- 6. RSI (14) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan) # Handle division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) # Neutral fill if no data

    # --- 7. Stochastic (14, 3, 3) ---
    low_14 = df['Low'].rolling(window=14, min_periods=1).min()
    high_14 = df['High'].rolling(window=14, min_periods=1).max()
    df['K_Percent'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14).replace(0, np.nan))
    df['D_Percent'] = df['K_Percent'].rolling(window=3, min_periods=1).mean()

    return df, is_ipo

def scan_signals(df):
    signals = []
    
    # Need at least 2 rows to compare "Yesterday" vs "Today"
    if len(df) < 2:
        return []

    # Iterate through the DataFrame (starting from index 1)
    # We use iterrows for simplicity in a CLI script, though vectorized is faster for massive datasets
    # Using index access for speed
    
    dates = df.index
    close = df['Close'].values
    vwap = df['VWAP'].values
    cmf = df['CMF'].values
    nvi = df['NVI'].values
    nvi_ema = df['NVI_EMA'].values
    obv = df['OBV'].values
    obv_ema = df['OBV_EMA'].values
    abs_z = df['Absorption_Z'].values
    rsi = df['RSI'].values
    
    for i in range(1, len(df)):
        date_str = dates[i].strftime('%Y-%m-%d')
        price = close[i]
        curr_rsi = rsi[i]
        
        # 1. START BUY
        # (Price > VWAP) AND (CMF > 0.05) AND (Yesterday this condition was False)
        bull_cond = (close[i] > vwap[i]) and (cmf[i] > 0.05)
        prev_bull_cond = (close[i-1] > vwap[i-1]) and (cmf[i-1] > 0.05)
        
        if bull_cond and not prev_bull_cond:
            signals.append([date_str, "START BUY", f"{price:.2f}", f"{curr_rsi:.1f}", "Price > VWAP & CMF > 0.05"])

        # 2. START SELL
        # (Price < VWAP) AND (CMF < -0.05) AND (Yesterday this condition was False)
        bear_cond = (close[i] < vwap[i]) and (cmf[i] < -0.05)
        prev_bear_cond = (close[i-1] < vwap[i-1]) and (cmf[i-1] < -0.05)
        
        if bear_cond and not prev_bear_cond:
            signals.append([date_str, "START SELL", f"{price:.2f}", f"{curr_rsi:.1f}", "Price < VWAP & CMF < -0.05"])

        # 3. NVI ENTRY
        # Triggered when NVI crosses above NVI_EMA
        if (nvi[i] > nvi_ema[i]) and (nvi[i-1] <= nvi_ema[i-1]):
            signals.append([date_str, "NVI ENTRY", f"{price:.2f}", f"{curr_rsi:.1f}", "Smart Money Accumulation"])

        # 4. OBV BREAKOUT
        # Triggered when OBV crosses above OBV_EMA
        if (obv[i] > obv_ema[i]) and (obv[i-1] <= obv_ema[i-1]):
            signals.append([date_str, "OBV BREAKOUT", f"{price:.2f}", f"{curr_rsi:.1f}", "Volume Trend Change"])

        # 5. ABSORPTION
        # Triggered when Amihud Z-Score > 2.5
        if abs_z[i] > 2.5:
             signals.append([date_str, "ABSORPTION", f"{price:.2f}", f"{curr_rsi:.1f}", f"Vol Spike/Price Stable (Z={abs_z[i]:.1f})"])

    return signals

def analyze_timeframes(df):
    matrix = []
    periods = {
        '1 Week': 5,
        '1 Month': 20,
        '3 Month': 60,
        '6 Month': 120,
        'YTD': 'ytd'
    }
    
    current_price = df['Close'].iloc[-1]
    
    for name, lookback in periods.items():
        if lookback == 'ytd':
            start_of_year = datetime(df.index[-1].year, 1, 1)
            # Find closest date
            period_df = df[df.index >= start_of_year]
            if period_df.empty:
                row = [name, "N/A", "N/A", "N/A"]
                matrix.append(row)
                continue
            past_price = period_df['Close'].iloc[0]
            avg_cmf = period_df['CMF'].mean()
        else:
            if len(df) <= lookback:
                row = [name, "N/A", "N/A", "N/A"]
                matrix.append(row)
                continue
            
            past_price = df['Close'].iloc[-(lookback+1)] # +1 to get start of period
            avg_cmf = df['CMF'].tail(lookback).mean()

        price_change_pct = ((current_price - past_price) / past_price) * 100
        
        # Verdict Logic
        verdict = "NEUTRAL"
        price_up = price_change_pct > 0
        flow_pos = avg_cmf > 0
        
        if price_up and flow_pos:
            verdict = "STRONG BULL"
        elif not price_up and flow_pos:
            verdict = "DIVERGENCE (BUY)"
        elif price_up and not flow_pos:
            verdict = "WEAK BULL"
        elif not price_up and not flow_pos:
            verdict = "BEARISH"
            
        matrix.append([name, f"{price_change_pct:+.2f}%", f"{avg_cmf:+.3f}", verdict])
        
    return matrix

def print_table(headers, data, title=None):
    if title:
        print(f"\n=== {title} ===")
    
    if not data:
        print("No signals detected in the analysis period.")
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
            
    # Create format string
    fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    separator = "-+-".join(["-" * w for w in col_widths])
    
    print(fmt.format(*headers))
    print(separator)
    for row in data:
        print(fmt.format(*row))

def main():
    args = parse_arguments()
    df = fetch_data(args.ticker)
    
    # Calculate Indicators
    df, is_ipo = calculate_indicators(df)
    
    # Header Info
    last_date = df.index[-1].strftime('%Y-%m-%d')
    print(f"\nAnalysis for: {args.ticker.upper()}")
    print(f"Data Range: {df.index[0].strftime('%Y-%m-%d')} to {last_date}")
    print(f"Total Trading Days: {len(df)}")
    
    if is_ipo:
        print("\n[WARNING] IPO/New Listing detected (< 500 days). Some indicators (255 EMA) may be incomplete or approximate.")

    # --- Output 1: Signal Log ---
    signals = scan_signals(df)
    # Filter to show last 15 signals if list is huge, otherwise show all
    # showing all as requested "Chronological History" usually implies full log, 
    # but for CLI usability, let's show last 25 to keep terminal clean, or all if piped.
    # The prompt asked for "The Signal Log", implying comprehensive. Let's do all.
    
    print_table(
        ["Date", "Action", "Price", "RSI", "Details"], 
        signals, 
        title="SIGNAL LOG (Chronological)"
    )

    # --- Output 2: Timeframe Matrix ---
    tf_matrix = analyze_timeframes(df)
    print_table(
        ["Period", "Price %", "Avg CMF", "Verdict"],
        tf_matrix,
        title="TIMEFRAME MATRIX"
    )

    # --- Output 3: Today's Dashboard ---
    last = df.iloc[-1]
    
    # Bandarmology Logic
    vwap_status = "BULLISH" if last['Close'] > last['VWAP'] else "BEARISH"
    
    # OBV Trend
    # 10 day divergence: Price change vs OBV change
    p_10d_chg = df['Close'].diff(10).iloc[-1]
    obv_10d_chg = df['OBV'].diff(10).iloc[-1]
    
    obv_status = "NEUTRAL"
    if last['OBV'] > last['OBV_EMA']:
        obv_status = "UPTREND"
    else:
        obv_status = "DOWNTREND"
        
    if p_10d_chg < 0 and obv_10d_chg > 0:
        obv_status += " (BULL DIV)"
    elif p_10d_chg > 0 and obv_10d_chg < 0:
        obv_status += " (BEAR DIV)"

    # Stoch Cross
    k = last['K_Percent']
    d = last['D_Percent']
    stoch_status = f"K:{k:.1f} D:{d:.1f}"
    if k > d and df['K_Percent'].iloc[-2] <= df['D_Percent'].iloc[-2]:
        stoch_status += " (GOLDEN X)"
    
    dashboard = [
        ["Bandar Position (Price vs VWAP)", f"{vwap_status} (Price: {last['Close']:.2f} / VWAP: {last['VWAP']:.2f})"],
        ["Money Flow (Current CMF)", f"{last['CMF']:.3f}"],
        ["OBV Status", obv_status],
        ["RSI (14)", f"{last['RSI']:.1f}"],
        ["Stochastic (14,3,3)", stoch_status],
        ["Absorption Z-Score", f"{last['Absorption_Z']:.2f}"]
    ]
    
    print_table(
        ["Indicator", "Value/Status"],
        dashboard,
        title="TODAY'S DASHBOARD"
    )

if __name__ == "__main__":
    main()