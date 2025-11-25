import os
import sys
import argparse
from engine import StockAnalyzer, DEFAULT_CONFIG

def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*60)
    print("      SIMPLE IHSG STOCK VALIDATOR (Advanced)      ")
    print("      Features: Config + Smart Money + Pattern + Candles")
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

    # 2. VALIDATION SCORE (Moved Up)
    val = data['validation']
    stars = "⭐" * val['score']
    print(f"\n--- CONFLUENCE SCORE: {val['score']}/5 {stars} ---")
    print(f"Verdict: {val['verdict']}")
    if val['reasons']:
        print(f"Factors: {', '.join(val['reasons'])}")
    else:
        print("Factors: None (Weak Setup)")

    # 3. DUAL TRADE PLANS (Replaces the old "Big Action")
    for plan in data['plans']:
        p_name = "⚡ SHORT TERM (1-5 Days)" if plan['type'] == "SHORT_TERM" else "🌊 SWING TERM (>5 Days)"
        print(f"\n{p_name}")
        
        # Determine color/emphasis based on status
        status_display = plan['status']
        if "EXECUTE" in status_display:
            status_display = f"!!! {status_display} !!!"
            
        print(f"Status:      {status_display}")
        
        if "PENDING" in plan['status']:
             print(f"Note:        {plan.get('note', '')}")
             print(f"WAIT FOR:    Rp {plan['entry']:,.0f}")
        else:
             print(f"ENTRY:       Rp {plan['entry']:,.0f}")
        
        if plan['entry'] > 0:
            sl_pct = ((plan['stop_loss'] - plan['entry']) / plan['entry']) * 100
            tp_pct = ((plan['take_profit'] - plan['entry']) / plan['entry']) * 100
            print(f"STOP LOSS:   Rp {plan['stop_loss']:,.0f} ({sl_pct:.1f}%)")
            print(f"TAKE PROFIT: Rp {plan['take_profit']:,.0f} (+{tp_pct:.1f}%)")

    # 4. CHART PATTERNS
    print(f"\n--- CHART & CANDLE PATTERNS ---")
    
    candle = data['context'].get('candle', {})
    if candle.get('pattern') != "None":
        print(f"[+] CANDLESTICK: {candle['pattern']}")
        print(f"    Signal: {candle['sentiment']}")
    else:
        print(f"[-] Candle: No major reversal pattern detected.")

    vcp = data['context'].get('vcp', {})
    if vcp.get('detected'):
        print(f"[+] VCP DETECTED: {vcp['msg']}")
        
    geo = data['context'].get('geo', {})
    if geo.get('pattern') != "None":
        print(f"[+] GEOMETRY: {geo['pattern']}")
        print(f"    {geo['msg']}")

    # 5. NEWS SENTIMENT
    print(f"\n--- NEWS SENTIMENT ---")
    s_data = data['sentiment']
    print(f"Rating: {s_data['sentiment']} (Score: {s_data['score']})")
    if not s_data['headlines']:
        print("  - No recent news found.")
    for hl in s_data['headlines']:
        print(f"  - {hl}")

    # 6. CONTEXT
    ctx = data['context']
    print(f"\n--- CONTEXT & SMART MONEY ---")
    print(f"Trend:           {ctx['trend']}")
    print(f"OBV Status:      {ctx['obv_status']}")
    print(f"Money Flow:      {ctx['smart_money']}") 
    print(f"Volatility(ATR): Rp {ctx['atr']:,.0f} (Daily Range)")
    print(f"Support (20d):   Rp {ctx['support']:,.0f} (Dist: {ctx['dist_support']:.1f}%)")
    print(f"Resistance (20d):Rp {ctx['resistance']:,.0f}")

    # 7. FIBONACCI
    print(f"\n--- FIBONACCI KEY LEVELS ---")
    fibs = ctx.get('fib_levels', {})
    curr_p = data['price']
    
    if fibs:
        def get_fib_label(price):
            if curr_p > price: return "[SUPPORT]"
            elif curr_p < price: return "[RESISTANCE]"
            else: return "[AT LEVEL]"

        print(f"High (0.0):      Rp {fibs.get('0.0 (High)', 0):,.0f} {get_fib_label(fibs.get('0.0 (High)', 0))}")
        print(f"0.382 Level:     Rp {fibs.get('0.382', 0):,.0f} {get_fib_label(fibs.get('0.382', 0))}")
        print(f"0.5 Halfway:     Rp {fibs.get('0.5 (Half)', 0):,.0f} {get_fib_label(fibs.get('0.5 (Half)', 0))}")
        print(f"0.618 GOLDEN:    Rp {fibs.get('0.618 (Golden)', 0):,.0f} {get_fib_label(fibs.get('0.618 (Golden)', 0))}")
        print(f"Low (1.0):       Rp {fibs.get('1.0 (Low)', 0):,.0f} {get_fib_label(fibs.get('1.0 (Low)', 0))}")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="IHSG Stock Validator with Custom Config")
    
    parser.add_argument('ticker', nargs='?', help='Stock Ticker (e.g. BBCA)')
    parser.add_argument('--period', type=str, default=DEFAULT_CONFIG['BACKTEST_PERIOD'], help='Backtest Period (e.g. 1y, 2y)')
    parser.add_argument('--hold', type=int, default=DEFAULT_CONFIG['MAX_HOLD_DAYS'], help='Max Hold Days')
    parser.add_argument('--rsi', type=int, default=DEFAULT_CONFIG['RSI_PERIOD'], help='RSI Period')
    parser.add_argument('--sl', type=float, default=DEFAULT_CONFIG['SL_MULTIPLIER'], help='Stop Loss ATR Multiplier')
    parser.add_argument('--tp', type=float, default=DEFAULT_CONFIG['TP_MULTIPLIER'], help='Take Profit ATR Multiplier')
    
    # Argument Flags
    parser.add_argument('--fib', type=int, default=DEFAULT_CONFIG['FIB_LOOKBACK_DAYS'], help='Fibonacci Lookback Days')
    parser.add_argument('--cmf', type=int, default=DEFAULT_CONFIG['CMF_PERIOD'], help='CMF Period')
    parser.add_argument('--mfi', type=int, default=DEFAULT_CONFIG['MFI_PERIOD'], help='MFI Period')

    args = parser.parse_args()

    if not args.ticker:
        clear_screen()
        print_header()
        ticker_input = input("\nEnter Ticker (e.g. BBCA, ANTM): ").strip()
        if not ticker_input:
            print("No ticker provided. Exiting.")
            return
        args.ticker = ticker_input
    else:
        clear_screen()
        print_header()

    user_config = {
        "BACKTEST_PERIOD": args.period,
        "MAX_HOLD_DAYS": args.hold,
        "RSI_PERIOD": args.rsi,
        "SL_MULTIPLIER": args.sl,
        "TP_MULTIPLIER": args.tp,
        "FIB_LOOKBACK_DAYS": args.fib,
        "CMF_PERIOD": args.cmf,
        "MFI_PERIOD": args.mfi
    }

    print(f"\nAnalyzing {args.ticker.upper()}... (Config: {user_config})")
    
    analyzer = StockAnalyzer(args.ticker, user_config)
    report_data = analyzer.generate_final_report()
    
    print_report(report_data)

if __name__ == "__main__":
    main()


