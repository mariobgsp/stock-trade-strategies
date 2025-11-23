import os
import sys
from engine import StockAnalyzer

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*60)
    print("      SIMPLE IHSG STOCK VALIDATOR (CLI)      ")
    print("      Features: Sentiment + Deep Grid Search Backtest")
    print("="*60)

def print_report(data):
    if not data:
        print("\n[!] Error: Could not fetch data or invalid ticker.")
        return

    # 1. Basic Info
    print(f"\nSTOCK: {data['name']} ({data['ticker']})")
    print(f"PRICE: Rp {data['price']:,.0f}")
    
    # 2. THE BIG ACTION
    print("\n" + "*"*40)
    print(f"  {data['action']}")
    print("*"*40)
    print(f"Trigger Logic: {data['trigger']}")

    # 3. Sentiment Analysis
    print(f"\n--- NEWS SENTIMENT (English Source) ---")
    s_data = data['sentiment']
    print(f"Rating: {s_data['sentiment']} (Score: {s_data['score']})")
    print("Recent Headlines:")
    if not s_data['headlines']:
        print("  - No recent English news found.")
    for hl in s_data['headlines']:
        print(f"  - {hl}")

    # 4. Market Context (Price Action)
    ctx = data['context']
    print(f"\n--- PRICE ACTION & CONTEXT ---")
    print(f"Trend (200 EMA): {ctx['trend']}")
    print(f"OBV Status:      {ctx['obv_status']}")
    print(f"Support (20d):   Rp {ctx['support']:,.0f} (Dist: {ctx['dist_support']:.1f}%)")
    print(f"Resistance (20d):Rp {ctx['resistance']:,.0f}")

    # 5. Optimization Results (The Smart Feature)
    best = data['best_strategy']
    print(f"\n--- HISTORICAL OPTIMIZATION (Last 2 Years) ---")
    print(f"Best Strategy:   {best['strategy']}")
    print(f"Configuration:   {best['details']}")
    print(f"Optimum Hold:    EXACTLY {best['hold_days']} DAYS")
    print(f"Hist. Win Rate:  {best['win_rate']:.1f}%")
    
    print("\n" + "="*60)

def main():
    while True:
        clear_screen()
        print_header()
        
        ticker = input("\nEnter Ticker (e.g. BBCA, ANTM) or 'Q' to quit: ").strip()
        
        if ticker.lower() == 'q':
            print("Goodbye! Cuan always.")
            sys.exit()
            
        if not ticker:
            continue
            
        print(f"\nAnalyzing {ticker.upper()}... (Fetching Data & Crunching Numbers)")
        print("Please wait, running Grid Search on 1-20 day holding periods...")
        
        analyzer = StockAnalyzer(ticker)
        report_data = analyzer.generate_final_report()
        
        print_report(report_data)
        
        input("\nPress Enter to search another stock...")

if __name__ == "__main__":
    main()