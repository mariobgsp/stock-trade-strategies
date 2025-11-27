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
    
    # 3. PROBABILITY & VALIDATION
    prob = data['probability']
    val = data['validation']
    
    print(f"\n--- PREDICTIVE ANALYTICS ---")
    print(f"Success Prob: {prob['value']:.1f}% ({prob['verdict']})")
    print(f"Confluence:   {val['score']}/5 ({val['verdict']})")
    if val['reasons']:
        print(f"Factors:      {', '.join(val['reasons'])}")

    # 4. OPTIMIZED TRADE PLAN (Single Plan)
    plan = data['plans'][0] 
    
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
        tp_pct_dyn = ((plan['take_profit'] - plan['entry']) / plan['entry']) * 100
        
        print(f"STOP LOSS:   Rp {plan['stop_loss']:,.0f} ({sl_pct:.1f}%)")
        print("-" * 30)
        print(f"TARGET A:    Rp {plan['take_profit']:,.0f} (+{tp_pct_dyn:.1f}%) [Conservative]")
        
        if 'take_profit_3r' in plan:
            tp_pct_3r = ((plan['take_profit_3r'] - plan['entry']) / plan['entry']) * 100
            print(f"TARGET B:    Rp {plan['take_profit_3r']:,.0f} (+{tp_pct_3r:.1f}%) [Max Profit 1:3]")

    # 5. CONTEXT & PATTERNS
    print(f"\n--- MARKET CONTEXT ---")
    ctx = data['context']
    
    # Smart Money Section
    print(f"Smart Money: {ctx['smart_money']}")
    
    # NEW: BANDAR BEHAVIOR (Dynamic Horizon)
    beh = ctx.get('breakout_behavior', {})
    if beh.get('count', 0) > 0:
        horizon = beh.get('best_horizon', 5)
        print(f"    ↳ Breakout Behavior: {beh['behavior']}")
        print(f"    ↳ Win Rate ({horizon} Day Hold): {beh['accuracy']} (Avg Return {beh['avg_return_5d']})")
    else:
        print(f"    ↳ Breakout Behavior: No recent samples.")
    
    # Pattern Backtest Stats
    sm_stats = ctx.get('sm_predict', {})
    if sm_stats.get('count', 0) > 0:
        print(f"    ↳ Smart Money Accuracy: {sm_stats['accuracy']} (Avg Return {sm_stats['avg_return']})")
    
    vsa = ctx.get('vsa', {})
    if vsa.get('detected'):
        print(f"VSA Signal:  {vsa['msg']}")

    efi = ctx.get('efi', 0)
    efi_power = "Bulls Power" if efi > 0 else "Bears Power"
    print(f"Force Index: {efi_power} ({efi:.0f})")
    
    sqz = ctx.get('squeeze', {})
    if sqz.get('detected'): print(f"[!!!] {sqz['msg']}")
    
    vol_brk = ctx.get('vol_breakout', {})
    if vol_brk.get('detected'): print(f"[!!!] {vol_brk['msg']}")

    vcp = ctx.get('vcp', {})
    if vcp.get('detected'): print(f"[+] {vcp['msg']}")
    
    geo = ctx.get('geo', {})
    if geo.get('pattern') != "None": 
        print(f"[+] GEO: {geo['pattern']}")
        print(f"    {geo['msg']}")
        
        p_stats = ctx.get('pattern_stats', {})
        if p_stats.get('count', 0) > 0:
            print(f"    ↳ Historical Reliability: {p_stats['accuracy']} ({p_stats['wins']}/{p_stats['count']} Wins)")
            print(f"    ↳ Projection: {p_stats['verdict']}")

    # 6. CONTEXT & PIVOTS
    pivots = ctx.get('pivots', {})
    print(f"\n--- PIVOT POINTS (Daily) ---")
    print(f"Pivot (P):   Rp {pivots.get('P', 0):,.0f}")
    print(f"Supp (S1):   Rp {pivots.get('S1', 0):,.0f}")
    print(f"Resis (R1):  Rp {pivots.get('R1', 0):,.0f}")

    # 7. FIBONACCI LEVELS (RESTORED)
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

    # 8. NEWS
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
    
    parser.add_argument('--min_vol', type=int, default=DEFAULT_CONFIG['MIN_DAILY_VOL'])

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
        "MFI_PERIOD": args.mfi,
        "MIN_DAILY_VOL": args.min_vol
    }

    print(f"\nAnalyzing {args.ticker.upper()}... (Config: {user_config})")
    analyzer = StockAnalyzer(args.ticker, user_config)
    print_report(analyzer.generate_final_report())

if __name__ == "__main__":
    main()


