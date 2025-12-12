import sys
import argparse
from colorama import init, Fore, Style, Back
from engine import QuantEngine

# Initialize Colorama
init(autoreset=True)

def print_header():
    print(f"\n{Back.BLUE}{Fore.WHITE} ========================================================== {Style.RESET_ALL}")
    print(f"{Back.BLUE}{Fore.WHITE}      IHSG SWING TRADER PRO - QUANT ENGINE (v1.0)           {Style.RESET_ALL}")
    print(f"{Back.BLUE}{Fore.WHITE} ========================================================== {Style.RESET_ALL}\n")

def print_verdict_card(data):
    verdict = data['verdict']
    color = Fore.WHITE
    
    if "BUY" in verdict:
        color = Fore.GREEN
    elif "NO TRADE" in verdict:
        color = Fore.RED
    else:
        color = Fore.YELLOW

    print(f"{Style.BRIGHT}TARGET: {Fore.CYAN}{data['ticker']}{Style.RESET_ALL} | PRICE: {Fore.CYAN}Rp {int(data['current_price'])}{Style.RESET_ALL}")
    print(f"\n{Style.BRIGHT}------------------ AI VERDICT ------------------")
    print(f"{Style.BRIGHT}SIGNAL: {color}{verdict}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}WHY?:   {Fore.WHITE}{data['why']}{Style.RESET_ALL}")
    
    if data['patterns']['Accumulation']:
        print(f"        {Fore.GREEN}✔ Smart Money Accumulation Detected since {data['smart_money_date'].strftime('%Y-%m-%d')}{Style.RESET_ALL}")
    if data['patterns']['VCP']:
        print(f"        {Fore.GREEN}✔ VCP Pattern (Volatility Contraction) Confirmed{Style.RESET_ALL}")
    if data['patterns']['Squeeze']:
        print(f"        {Fore.MAGENTA}✔ MA Squeeze / Superclose Detected{Style.RESET_ALL}")

def print_trade_plan(data):
    if "NO TRADE" in data['verdict']:
        return

    plan = data['trade_plan']
    print(f"\n{Style.BRIGHT}------------------ TRADE PLAN (OJK RULES) ------------------")
    print(f"ENTRY: {Fore.CYAN}Rp {int(data['current_price'])}{Style.RESET_ALL} (Market)")
    print(f"STOP LOSS: {Fore.RED}Rp {int(plan['sl'])}{Style.RESET_ALL} (risk 1R)")
    print(f"TARGET 1:  {Fore.GREEN}Rp {int(plan['tp1'])}{Style.RESET_ALL} (1.5R - Secure Profits)")
    print(f"TARGET 2:  {Fore.GREEN}Rp {int(plan['tp2'])}{Style.RESET_ALL} (3.0R - Golden Ratio)")
    print(f"TARGET 3:  {Fore.GREEN}Rp {int(plan['tp3'])}{Style.RESET_ALL} (4.5R - Runner)")

def print_stats(data):
    wr = data['win_rate']
    wr_color = Fore.GREEN if wr > 65 else Fore.RED
    
    print(f"\n{Style.BRIGHT}------------------ RISK & LOGIC ------------------")
    print(f"Hist. Win Rate: {wr_color}{wr:.1f}%{Style.RESET_ALL} (Based on 3Y Backtest)")
    print(f"Best Strategy:  MA Crossover {data['best_params'][0]}/{data['best_params'][1]} + RSI < {data['best_params'][2]}")
    
    print(f"\n{Style.BRIGHT}------------------ TECH DEEP DIVE ------------------")
    print(f"RSI (14):     {data['technicals']['rsi']:.2f}")
    print(f"Stoch K:      {data['technicals']['stoch_k']:.2f}")
    print(f"VWAP:         {data['technicals']['vwap']:.2f}")
    print(f"OBV Slope:    {data['smart_money_slope']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='IHSG Swing Trading Quant CLI')
    parser.add_argument('ticker', type=str, help='Stock Ticker (e.g., BBRI, TLKM)')
    args = parser.parse_args()

    print_header()
    print(f"Initializing Quantum Engine for {args.ticker}...")
    
    engine = QuantEngine(args.ticker)
    
    # 1. Fetch
    print("Fetching 3 years of OHLCV data from Yahoo Finance...")
    success, msg = engine.fetch_data()
    if not success:
        print(f"{Fore.RED}Error: {msg}{Style.RESET_ALL}")
        sys.exit(1)

    # 2. Analyze
    print("Running Sklearn Linear Regression on OBV...")
    print("Calculating Scipy Local Extrema for Support/Resistance...")
    print("Grid Searching Strategy Parameters (Optimization Loop)...")
    
    try:
        results = engine.optimize_and_analyze()
        
        # 3. Output
        print_verdict_card(results)
        print_trade_plan(results)
        print_stats(results)
        
    except Exception as e:
        print(f"{Fore.RED}Analysis Failed: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

