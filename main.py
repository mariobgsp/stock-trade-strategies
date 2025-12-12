import sys
import argparse
from engine import QuantEngine

def print_separator(char='-'):
    print(char * 60)

def normalize_ticker(ticker):
    ticker = ticker.strip().upper()
    if not ticker.endswith('.JK'):
        ticker += '.JK'
    return ticker

def format_probability(prob):
    # Safety rating visualization
    if prob >= 80: return f"{prob:.1f}% (EXCELLENT)"
    if prob >= 60: return f"{prob:.1f}% (GOOD)"
    if prob >= 40: return f"{prob:.1f}% (MODERATE)"
    return f"{prob:.1f}% (RISKY)"

def main():
    parser = argparse.ArgumentParser(description="IHSG Swing Trading Quant CLI")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., BBCA)")
    args = parser.parse_args()

    ticker = normalize_ticker(args.ticker)
    
    print("\n")
    print_separator('=')
    print(f" INITIALIZING QUANT ENGINE FOR: {ticker}")
    print_separator('=')
    print(" [1/4] Fetching OHLCV Data (yfinance)...")
    
    try:
        engine = QuantEngine(ticker)
        engine.fetch_data()
        
        print(f" [2/4] Detecting Status: {engine.data_status}")
        print(" [3/4] Running Grid Search & Backtesting...")
        print(" [4/4] Analyzing Smart Money Flow (sklearn)...")
        
        result = engine.run_optimization_and_generate_signal()
        
        # --- UI RENDERING ---
        
        print("\n")
        print_separator('=')
        print(f" VERDICT: {result['verdict']}")
        print_separator('=')
        
        if "NO TRADE" not in result['verdict'] and "WAIT" not in result['verdict']:
            print(f" STRATEGY: {result['strategy_name']}")
            print(f" LOGIC   : Triggered by {result['strategy_name']} logic.")
            print(f"           Price is {result['vwap_text']}.")
            if result['patterns']:
                print(f"           Patterns Detected: {', '.join(result['patterns'])}")
            print("\n TRADE PLAN (OJK Tick Compliant):")
            print(f" > ENTRY : {int(result['entry'])}")
            print(f" > STOP  : {int(result['sl'])} (RISK: {int(result['entry'] - result['sl'])})")
            print(f" > TP 1  : {int(result['tp1'])} (1R)")
            print(f" > TP 2  : {int(result['tp2'])} (2R)")
            print(f" > TP 3  : {int(result['tp3'])} (3R)")
            
            print("\n SAFETY SCORE (Historical Probability):")
            print(f" Win Rate (>1R)      : {result['win_rate']:.1f}%")
            print(f" Probability hit 1R  : {format_probability(result['probs'][0])}")
            print(f" Probability hit 2R  : {format_probability(result['probs'][1])}")
            print(f" Probability hit 3R  : {format_probability(result['probs'][2])}")
        
        else:
            print(" System could not find a setup with >60% Probability.")
            print(" Recommendation: DO NOT ENTER.")
        
        print("\n")
        print_separator()
        print(" DATA INTELLIGENCE")
        print_separator()
        print(f" Asset Class : {result['data_status']}")
        print(f" Smart Money : {result['sm_status']}")
        print(f" Flow Start  : {result['sm_date']}")
        print_separator('-')
        print(" KEY LEVELS")
        print(f" Nearest Sup : {int(result['supports'][0])}")
        print(f" Nearest Res : {int(result['supports'][1])}")
        print(f" Golden Fib  : {int(result['supports'][2])}")
        print_separator('-')
        print(" INDICATORS (Optimized)")
        print(f" RSI         : {result['indicators']['RSI']:.2f}")
        print(f" Stochastic  : {result['indicators']['Stoch_K']:.2f}")
        print(f" Fast MA     : {result['indicators']['SMA_Fast']:.2f}")
        print_separator('=')
        print("\n")

    except Exception as e:
        print(f"\n [ERROR CRITICAL]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

