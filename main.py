import os
import sys
import argparse
from engine import StockAnalyzer, DEFAULT_CONFIG

def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*60)
    print("      SIMPLE IHSG STOCK VALIDATOR (Dual-Engine)      ")
    print("      Features: Scalp vs Swing + Confluence Check")
    print("="*60)

def print_report(data):
    if not data:
        print("\n[!] Error: Could not fetch data or invalid ticker.")
        return

    # 1. Basic Info
    print(f"\nSTOCK: {data['name']} ({data['ticker']})")
    print(f"PRICE: Rp {data['price']:,.0f}")
    if data['is_ipo']: print(f"[!] WARNING: IPO DETECTED ({data['days_listed']} days listed)")

    # 2. VALIDATION SCORE (New)
    val = data['validation']
    stars = "⭐" * val['score']
    print(f"\n--- CONFLUENCE SCORE: {val['score']}/5 {stars} ---")
    print(f"Verdict: {val['verdict']}")
    if val['reasons']:
        print(f"Factors: {', '.join(val['reasons'])}")
    else:
        print("Factors: None (Weak Setup)")

    # 3. DUAL TRADE PLANS (New)
    for plan in data['plans']:
        p_name = "⚡ SHORT TERM (1-5 Days)" if plan['type'] == "SHORT_TERM" else "🌊 SWING TERM (>5 Days)"
        print(f"\n{p_name}")
        print(f"Status:      {plan['status']}")
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
    print(f"\n--- CHART & PATTERNS ---")
    candle = data['context'].get('candle', {})
    if candle.get('pattern') != "None": print(f"[+] CANDLE: {candle['pattern']} ({candle['sentiment']})")
    
    vcp = data['context'].get('vcp', {})
    if vcp.get('detected'): print(f"[+] VCP: {vcp['msg']}")
    
    geo = data['context'].get('geo', {})
    if geo.get('pattern') != "None": print(f"[+] GEO: {geo['pattern']}")

    # 5. CONTEXT
    ctx = data['context']
    print(f"\n--- MARKET CONTEXT ---")
    print(f"Trend:       {ctx['trend']}")
    print(f"Smart Money: {ctx['smart_money']}")
    print(f"Support:     Rp {ctx['support']:,.0f} (Dist: {ctx['dist_support']:.1f}%)")
    
    # 6. NEWS
    print(f"\n--- NEWS SENTIMENT ---")
    s_data = data['sentiment']
    print(f"Score: {s_data['score']} ({s_data['sentiment']})")
    for hl in s_data['headlines']: print(f"- {hl}")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', nargs='?')
    parser.add_argument('--period', default=DEFAULT_CONFIG['BACKTEST_PERIOD'])
    parser.add_argument('--sl', type=float, default=DEFAULT_CONFIG['SL_MULTIPLIER'])
    parser.add_argument('--tp', type=float, default=DEFAULT_CONFIG['TP_MULTIPLIER'])
    args = parser.parse_args()

    if not args.ticker:
        clear_screen()
        print_header()
        args.ticker = input("\nEnter Ticker (e.g. BBCA): ").strip()

    user_config = {
        "BACKTEST_PERIOD": args.period,
        "SL_MULTIPLIER": args.sl,
        "TP_MULTIPLIER": args.tp
    }

    print(f"\nAnalyzing {args.ticker.upper()}...")
    analyzer = StockAnalyzer(args.ticker, user_config)
    print_report(analyzer.generate_final_report())

if __name__ == "__main__":
    main()


