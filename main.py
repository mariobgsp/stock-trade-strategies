import os
import sys
from engine import StockAnalyzer

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*60)
    print("      SIMPLE IHSG STOCK VALIDATOR (CLI)      ")
    print("      Features: Sentiment + Grid Search + Chart Patterns")
    print("="*60)

def print_report(data):
    if not data:
        print("\n[!] Error: Could not fetch data or invalid ticker.")
        return

    # 1. Basic Info
    print(f"\nSTOCK: {data['name']} ({data['ticker']})")
    print(f"PRICE: Rp {data['price']:,.0f}")
    
    if data['is_ipo']:
        print(f"[!] WARNING: IPO DETECTED ({data['days_listed']} days listed)")
    
    # 2. THE BIG ACTION
    print("\n" + "*"*40)
    print(f"  {data['action']}")
    print("*"*40)
    print(f"Logic: {data['trigger']}")

    # 3. CHART PATTERNS
    print(f"\n--- CHART PATTERNS (Pattern Recognition) ---")
    
    # VCP
    vcp = data['context'].get('vcp', {})
    if vcp.get('detected'):
        print(f"[+] VCP DETECTED: {vcp['msg']}")
    else:
        print(f"[-] VCP: No clear contraction detected.")
        
    # Geometry (Triangles/Pennants)
    geo = data['context'].get('geo', {})
    if geo.get('pattern') != "None":
        print(f"[+] GEOMETRY: {geo['pattern']}")
        print(f"    {geo['msg']}")
    else:
        print(f"[-] GEOMETRY: No Triangle or Pennant formed yet.")

    # 4. SMART TRADE PLAN
    plan = data['trade_plan']
    print(f"\n--- SMART TRADE PLAN (Status: {plan['status']}) ---")
    
    if "PENDING" in plan['status']:
         print(f"ADVICE:       {plan.get('note', 'Wait for setup.')}")
         print(f"WAIT FOR:     Rp {plan['entry']:,.0f} (Ideal Entry)")
    else:
         print(f"ENTRY PRICE:  Rp {plan['entry']:,.0f}")
    
    if plan['entry'] > 0:
        entry = plan['entry']
        sl_pct = ((plan['stop_loss'] - entry) / entry) * 100
        tp_pct = ((plan['take_profit'] - entry) / entry) * 100
        print(f"STOP LOSS:    Rp {plan['stop_loss']:,.0f} ({sl_pct:.1f}%)")
        print(f"TAKE PROFIT:  Rp {plan['take_profit']:,.0f} (+{tp_pct:.1f}%)")
        print(f"Risk/Reward:  {plan['risk_reward']}")

    # 5. NEWS SENTIMENT (RESTORED)
    print(f"\n--- NEWS SENTIMENT ---")
    s_data = data['sentiment']
    print(f"Rating: {s_data['sentiment']} (Score: {s_data['score']})")
    if not s_data['headlines']:
        print("  - No recent news found.")
    for hl in s_data['headlines']:
        print(f"  - {hl}")

    # 6. CONTEXT (OBV & Dist Support)
    ctx = data['context']
    print(f"\n--- CONTEXT & INDICATORS ---")
    print(f"Trend:           {ctx['trend']}")
    print(f"OBV Status:      {ctx['obv_status']}")
    print(f"Volatility(ATR): Rp {ctx['atr']:,.0f} (Daily Range)")
    print(f"Support (20d):   Rp {ctx['support']:,.0f} (Dist: {ctx['dist_support']:.1f}%)")
    print(f"Resistance (20d):Rp {ctx['resistance']:,.0f}")

    # 7. FIBONACCI
    print(f"\n--- FIBONACCI KEY LEVELS (Last 120 Days) ---")
    fibs = ctx.get('fib_levels', {})
    if fibs:
        print(f"High (0.0):      Rp {fibs.get('0.0 (High)', 0):,.0f}")
        print(f"0.5 Halfway:     Rp {fibs.get('0.5 (Half)', 0):,.0f}")
        print(f"0.618 GOLDEN:    Rp {fibs.get('0.618 (Golden)', 0):,.0f}  <-- Strong Support")
        print(f"Low (1.0):       Rp {fibs.get('1.0 (Low)', 0):,.0f}")

    # 8. Active Strategies
    print(f"\n--- ✅ STRATEGIES ACTIVE TODAY ---")
    active_strats = [s for s in data['all_strategies'] if s['is_triggered_today']]
    if not active_strats:
        print("[-] No strategies are currently triggered.")
    else:
        for strat in active_strats:
            print(f"[+] {strat['strategy']}")
            print(f"    Criteria:    {strat['details']}")
            print(f"    Hist. Stats: Win Rate {strat['win_rate']:.1f}% | Hold {strat['hold_days']} Days")
            print("-" * 30)

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
        
        analyzer = StockAnalyzer(ticker)
        report_data = analyzer.generate_final_report()
        
        print_report(report_data)
        
        input("\nPress Enter to search another stock...")

if __name__ == "__main__":
    main()