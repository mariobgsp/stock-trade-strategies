import os
import sys
import argparse
from engine import StockAnalyzer, DEFAULT_CONFIG

def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*65)
    print("      IHSG ULTIMATE SCANNER (V3.2 - Geometry Edition)      ")
    print("      Trend + VSA + Patterns + Smart Money + Prediction")
    print("="*65)

def print_report(data, balance):
    if not data:
        print("\n[!] Error: Could not fetch data or invalid ticker.")
        return

    # --- ENHANCED HEADER ---
    chg = data.get('change_pct', 0)
    chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
    chg_color = "🟢" if chg >= 0 else "🔴"
    
    print(f"\nSTOCK: {data['name']} ({data['ticker']})")
    print(f"PRICE: Rp {data['price']:,.0f} ({chg_color} {chg_str})")
    if data['is_ipo']: print(f"[!] WARNING: IPO DETECTED ({data['days_listed']} days listed)")

    # 1. LIQUIDITY (New)
    liq = data['liquidity']
    symbol = "✅" if liq['status'] == "PASS" else "⚠️ "
    print(f"\n{symbol} LIQUIDITY CHECK")
    print(f"   {liq['msg']}")

    # 2. FUNDAMENTALS (Enhanced)
    fund = data['context'].get('fundamental', {})
    print(f"\n📊 FUNDAMENTALS")
    print(f"   Market Cap: Rp {fund.get('market_cap', 0):,.0f}")
    if fund.get('pe'): print(f"   P/E Ratio:  {fund.get('pe', 0):.2f}x")
    if fund.get('pbv'): print(f"   PBV Ratio:  {fund.get('pbv', 0):.2f}x")
    if fund.get('roe'): print(f"   ROE:        {fund.get('roe', 0)*100:.2f}%")
    if fund.get('warning'): print(f"   ⚠️ {fund['warning']}")

    # 3. TREND HEALTH (Minervini)
    tt = data['trend_template']
    ma = data['context'].get('ma_values', {})
    symbol = "✅" if "UPTREND" in tt['status'] else "⚠️ "
    print(f"\n{symbol} TREND HEALTH")
    print(f"   Status: {tt['status']} (Score: {tt['score']}/{tt.get('max_score', 6)})")
    print(f"   [EMA50: {ma.get('EMA_50', 0):,.0f}] | [EMA200: {ma.get('EMA_200', 0):,.0f}]")
    for det in tt['details']: print(f"   - {det}")

    # 4. SMART MONEY (Detailed)
    sm = data['context']['smart_money']
    symbol = "✅" if "BULLISH" in sm['status'] else "⚠️ " if "BEARISH" in sm['status'] else "🔹"
    print(f"\n{symbol} SMART MONEY (Bandarmology Proxy)")
    print(f"   Status: {sm['status']}")
    
    metrics = sm.get('metrics', {})
    if metrics:
        bp = metrics.get('buy_pressure', 50)
        bar_len = 20
        fill = int(bp / 100 * bar_len)
        bar = "█" * fill + "░" * (bar_len - fill)
        print(f"   Pressure:  [{bar}] {bp:.1f}% Buy Vol")
        
        g_spikes = metrics.get('green_spikes', 0)
        r_spikes = metrics.get('red_spikes', 0)
        print(f"   Big Moves: {g_spikes} Accumulation Days vs {r_spikes} Distribution Days")

    for s in sm['signals']: print(f"   - {s}")

    # 5. PATTERNS & PREDICTION (New & Improved)
    print(f"\n💎 PATTERN ANALYSIS & PREDICTION")
    
    # Historical Counts
    counts = data['context'].get('pattern_counts', {})
    print(f"   History:   {counts.get('Total', 0)} Patterns detected ({counts.get('Triangle', 0)} Triangles, {counts.get('Channel', 0)} Channels)")
    
    # Current Active Pattern
    rect = data['rectangle']
    geo = data['context']['geo']
    
    found_pattern = False
    
    if rect['detected']:
        found_pattern = True
        print(f"   [RECTANGLE] Status: {rect['status']}")
        print(f"               Range: {rect['bottom']:,.0f} - {rect['top']:,.0f}")
        
    if geo['pattern'] != "None":
        found_pattern = True
        print(f"   [{geo['pattern'].upper()}]")
        print(f"               Prediction: {geo.get('prediction', 'N/A')}")
        print(f"               Action:     {geo.get('action', 'N/A')}")
        if "Apex" in geo['msg']: print(f"               Note:       {geo['msg']}")

    if not found_pattern:
        print(f"   No distinct chart patterns currently forming.")

    # 6. TRADE PLAN (Enhanced with Sizing & Reasoning)
    plan = data['plan']
    print(f"\n🚀 TRADE PLAN (3R)")
    
    if "WAIT" in plan['status']:
        print(f"   Action: {plan['status']}")
        print(f"   Reason: {plan['reason']}")
    else:
        print(f"   ACTION:      BUY ({plan['status']})")
        print(f"   WHY?:        {plan['reason']}")
        print(f"   ENTRY:       Rp {plan['entry']:,.0f}")
        print(f"   STOP LOSS:   Rp {plan['stop_loss']:,.0f}")
        print(f"   TARGET (3R): Rp {plan['take_profit']:,.0f}")
        
        risk = plan['entry'] - plan['stop_loss']
        reward = plan['take_profit'] - plan['entry']
        rrr = reward / risk if risk > 0 else 0
        print(f"   Ratio:       1:{rrr:.1f}")

        print(f"\n   --- POSITION SIZING (Bal: {balance/1e6:.0f} Jt) ---")
        if plan.get('lots', 0) > 0:
            print(f"   🛒 BUY:      {plan['lots']} LOTS")
            print(f"   💰 Capital:  Rp {plan['lots'] * 100 * plan['entry']:,.0f}")
            print(f"   🔥 Risk:     Rp {plan['risk_amt']:,.0f} ({DEFAULT_CONFIG['RISK_PER_TRADE_PCT']}%)")
        else:
            print("   [!] Stop Loss too tight or risk too high.")

    # 7. CONCLUSION (New)
    print(f"\n🏁 FINAL VERDICT")
    val = data['validation']
    prob = data['probability']
    
    print(f"   Signal Strength: {val['verdict']} (Score: {val['score']})")
    print(f"   Win Probability: {prob['verdict']} (~{prob['value']}%)")
    
    status = data['plan']['status']
    if "EXECUTE" in status or "EARLY ENTRY" in status:
        print(f"\n   👉 RECOMMENDATION: WATCHLIST / BUY")
        print(f"      Setup confirmed. Ensure Risk Management is applied.")
    elif "WAIT" in status:
        print(f"\n   👉 RECOMMENDATION: WAIT")
        print(f"      No valid entry yet. {data['plan']['reason']}")
    else:
        print(f"\n   👉 RECOMMENDATION: AVOID")
        print(f"      Trend or Fundamentals broken.")

    print("\n" + "="*65)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', nargs='?')
    
    parser.add_argument('--balance', type=int, default=100_000_000, help="Account Balance (IDR)")
    parser.add_argument('--risk', type=float, default=1.0, help="Risk per trade (%)")
    
    parser.add_argument('--period', default=DEFAULT_CONFIG['BACKTEST_PERIOD'], help="Data period (e.g. 1y, 2y)")
    parser.add_argument('--sl', type=float, default=DEFAULT_CONFIG['SL_MULTIPLIER'], help="Stop Loss ATR Multiplier")
    parser.add_argument('--tp', type=float, default=DEFAULT_CONFIG['TP_MULTIPLIER'], help="Take Profit ATR Multiplier")
    parser.add_argument('--fib', type=int, default=DEFAULT_CONFIG['FIB_LOOKBACK_DAYS'], help="Fibonacci Lookback Days")
    parser.add_argument('--cmf', type=int, default=DEFAULT_CONFIG['CMF_PERIOD'], help="CMF Period")
    parser.add_argument('--mfi', type=int, default=DEFAULT_CONFIG['MFI_PERIOD'], help="MFI Period")
    parser.add_argument('--min_vol', type=int, default=DEFAULT_CONFIG['MIN_DAILY_VOL'], help="Min Volume Filter")

    args = parser.parse_args()

    if not args.ticker:
        clear_screen()
        print_header()
        args.ticker = input("\nEnter Ticker (e.g. BBCA): ").strip()

    user_config = {
        "ACCOUNT_BALANCE": args.balance,
        "RISK_PER_TRADE_PCT": args.risk,
        "BACKTEST_PERIOD": args.period,
        "SL_MULTIPLIER": args.sl,
        "TP_MULTIPLIER": args.tp,
        "FIB_LOOKBACK_DAYS": args.fib,
        "CMF_PERIOD": args.cmf,
        "MFI_PERIOD": args.mfi,
        "MIN_DAILY_VOL": args.min_vol
    }

    print(f"\nRunning Mega Analysis on {args.ticker.upper()}... (Config: {user_config})")
    analyzer = StockAnalyzer(args.ticker, user_config)
    print_report(analyzer.generate_final_report(), args.balance)

if __name__ == "__main__":
    main()


