import os
import sys
import argparse
from engine import StockAnalyzer, DEFAULT_CONFIG

def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*60)
    print("      SIMPLE IHSG STOCK VALIDATOR (Professional Swing)      ")
    print("      Features: Minervini Trend + Smart Money + Optimized")
    print("="*60)

def print_report(data):
    if not data:
        print("\n[!] Error: Could not fetch data or invalid ticker.")
        return

    print(f"\nSTOCK: {data['name']} ({data['ticker']})")
    print(f"PRICE: Rp {data['price']:,.0f}")
    if data['is_ipo']: print(f"[!] WARNING: IPO DETECTED ({data['days_listed']} days listed)")

    # 1. SWING QUALITY CHECK
    tt = data['trend_template']
    print(f"\n--- SWING QUALITY CHECK (Minervini Trend) ---")
    print(f"Status: {tt['status']} ({tt['score']}/6)")
    if tt['details']:
        for det in tt['details']:
            prefix = "✅" if "WARNING" not in det and "Error" not in det else "⚠️"
            print(f" {prefix} {det}")

    # 2. FUNDAMENTALS
    fund = data['context'].get('fundamental', {})
    print(f"\n--- FUNDAMENTALS ---")
    print(f"Market Cap: Rp {fund.get('market_cap', 0):,.0f}")
    if fund.get('warning'): print(f"⚠️ {fund['warning']}")

    # 3. OPTIMIZED TRADE PLAN (Single Plan)
    val = data['validation']
    plan = data['plans'][0] # Get the single optimized plan
    
    print(f"\n--- 🌊 OPTIMIZED SWING PLAN (1-60 Days) ---")
    
    thesis = "Wait for setup."
    if "STRONG" in val['verdict']: thesis = "STRONG BUY. Inst. Accumulation + Trend."
    elif "MODERATE" in val['verdict']: thesis = "Speculative Buy. Watch stops."
    print(f"Thesis:      {thesis}")

    status_display = plan['status']
    if "EXECUTE" in status_display: status_display = f"!!! {status_display} !!!"
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

    # 4. CONTEXT & PATTERNS
    print(f"\n--- MARKET CONTEXT ---")
    ctx = data['context']
    print(f"Smart Money: {ctx['smart_money']}")
    
    sqz = ctx.get('squeeze', {})
    if sqz.get('detected'): print(f"[!!!] {sqz['msg']}")

    vcp = ctx.get('vcp', {})
    if vcp.get('detected'): print(f"[+] {vcp['msg']}")

    # 5. CONTEXT & PIVOTS
    ctx = data['context']
    pivots = ctx.get('pivots', {})
    print(f"\n--- CONTEXT & SMART MONEY ---")
    print(f"Trend:       {ctx['trend']}")
    print(f"Smart Money: {ctx['smart_money']}")
    print(f"Pivot (P):   Rp {pivots.get('P', 0):,.0f}")
    print(f"Supp (S1):   Rp {pivots.get('S1', 0):,.0f}")
    print(f"Resis (R1):  Rp {pivots.get('R1', 0):,.0f}")

    # 6. FIBONACCI LEVELS (RESTORED)
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

    # 7. NEWS
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
    
    parser.add_argument('--fib', type=int, default=DEFAULT_CONFIG['FIB_LOOKBACK_DAYS'])
    parser.add_argument('--cmf', type=int, default=DEFAULT_CONFIG['CMF_PERIOD'])
    parser.add_argument('--mfi', type=int, default=DEFAULT_CONFIG['MFI_PERIOD'])

    args = parser.parse_args()

    if not args.ticker:
        clear_screen()
        print_header()
        args.ticker = input("\nEnter Ticker (e.g. BBCA, ANTM): ").strip()

    user_config = {
        "BACKTEST_PERIOD": args.period,
        "SL_MULTIPLIER": args.sl,
        "TP_MULTIPLIER": args.tp,
        "FIB_LOOKBACK_DAYS": args.fib,
        "CMF_PERIOD": args.cmf,
        "MFI_PERIOD": args.mfi
    }

    print(f"\nAnalyzing {args.ticker.upper()}... (Config: {user_config})")
    analyzer = StockAnalyzer(args.ticker, user_config)
    print_report(analyzer.generate_final_report())

if __name__ == "__main__":
    main()
