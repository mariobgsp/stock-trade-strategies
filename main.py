import os
import sys
import argparse
from engine import StockAnalyzer, DEFAULT_CONFIG

def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*65)
    print("      IHSG ULTIMATE SCANNER (V3.4 - Sniper Edition)      ")
    print("      Precision Entry + Multi-Timeframe + Smart Money + AI")
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
    if fund.get('z_score'): print(f"   🛡️ Safety:   {fund['z_score']} (Altman Z)")

    # 3. TREND HEALTH (Minervini + Weekly)
    tt = data['trend_template']
    ma = data['context'].get('ma_values', {})
    symbol = "✅" if "UPTREND" in tt['status'] else "⚠️ "
    print(f"\n{symbol} TREND HEALTH")
    print(f"   Daily Status:  {tt['status']} (Score: {tt['score']}/{tt.get('max_score', 6)})")
    
    wt = data['context'].get('weekly_trend', 'UNKNOWN')
    wt_sym = "✅" if wt == "UPTREND" else "🔻"
    print(f"   Weekly Status: {wt_sym} {wt}")
    
    print(f"   [EMA50: {ma.get('EMA_50', 0):,.0f}] | [EMA200: {ma.get('EMA_200', 0):,.0f}]")
    for det in tt['details']: print(f"   - {det}")

    # 4. SMART MONEY (Detailed with ML)
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
    
    # --- NEW: BANDAR ML SECTION ---
    b_ml = data['context'].get('bandar_ml', {})
    prob = b_ml.get('probability', 0)
    print(f"   AI Accumulation Score: {prob:.1f}% ({b_ml.get('verdict', 'Unknown')})")

    for s in sm['signals']: print(f"   - {s}")
    
    # --- NEW: AI PREDICTION (GRADIENT BOOSTING) ---
    ml = data['context'].get('ml_prediction', {})
    print(f"\n🤖 AI / MACHINE LEARNING (Gradient Boosting)")
    if ml.get('prediction') == "N/A":
         print(f"   Status: Insufficient Data")
    else:
        conf = ml.get('confidence', 0)
        emoji = "🚀" if conf > 60 else "🐻" if conf < 40 else "⚖️"
        print(f"   Forecast:   {emoji} {ml.get('prediction')} ({conf:.1f}% Confidence)")
        print(f"   Logic:      {ml.get('msg')}")

    # 5. SETUP & STRUCTURE (Replaced Patterns)
    print(f"\n💎 SETUP & STRUCTURE")
    
    # VCP Check
    vcp = data['context'].get('vcp', {})
    if vcp.get('detected'):
         print(f"   [VCP DETECTED] {vcp['msg']}")
         
    # Low Cheat Check
    lc = data['context'].get('low_cheat', {})
    if lc.get('detected'):
         print(f"   [LOW CHEAT]    {lc['msg']}")
         
    # Inside Bar Check
    ib = data['context'].get('inside_bar', {})
    if ib.get('detected'):
         print(f"   [INSIDE BAR]   {ib['msg']} (High: {ib['high']:.0f}, Low: {ib['low']:.0f})")
    
    # Support & Resistance Levels
    ctx = data['context']
    print(f"\n   --- KEY LEVELS ---")
    print(f"   Resistance: Rp {ctx.get('resistance', 0):,.0f}")
    print(f"   Support:    Rp {ctx.get('support', 0):,.0f}")
    
    # Dynamic MA
    best_ma = ctx.get('best_ma', {})
    if best_ma.get('period', 0) > 0:
        print(f"   Dynamic MA: EMA {best_ma['period']} @ Rp {best_ma['price']:,.0f} ({best_ma['bounces']} bounces)")
        
    # Fib Levels
    fibs = ctx.get('fib_levels', {})
    if fibs:
        print(f"   Golden Fib: Rp {fibs.get('0.618 (Golden)', 0):,.0f}")

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
        
        # New: Trailing Stop Logic
        risk_unit = plan['entry'] - plan['stop_loss']
        if risk_unit > 0:
            t1 = plan['entry'] + risk_unit
            print(f"   MANAGEMENT:  At Rp {t1:,.0f} (1R), move SL to Breakeven.")
        
        risk = plan['entry'] - plan['stop_loss']
        reward = plan['take_profit'] - plan['entry']
        rrr = reward / risk if risk > 0 else 0
        print(f"   Ratio:       1:{rrr:.1f}")

        print(f"\n   --- POSITION SIZING (Bal: {balance/1e6:.0f} Jt) ---")
        if plan.get('lots', 0) > 0:
            print(f"   🛒 BUY:      {plan['lots']} LOTS")
            print(f"   � Capital:  Rp {plan['lots'] * 100 * plan['entry']:,.0f}")
            print(f"   🔥 Risk:     Rp {plan['risk_amt']:,.0f} ({DEFAULT_CONFIG['RISK_PER_TRADE_PCT']}%)")
        else:
            print("   [!] Stop Loss too tight or risk too high.")

    # 7. CONCLUSION (Probability-Aware)
    print(f"\n🏁 FINAL VERDICT")
    val = data['validation']
    prob = data['probability']
    prob_val = prob['value']
    
    print(f"   Signal Strength: {val['verdict']} (Score: {val['score']})")
    print(f"   Win Probability: {prob['verdict']} (~{prob_val}%)")
    
    status = data['plan']['status']
    
    # --- SMART LOGIC UPDATE ---
    if "EXECUTE" in status or "EARLY ENTRY" in status:
        if prob_val >= 60:
            print(f"\n   👉 RECOMMENDATION: WATCHLIST / BUY")
            print(f"      Setup confirmed and stats look good.")
        else:
            print(f"\n   👉 RECOMMENDATION: RISKY / SPECULATIVE BUY")
            print(f"      Setup is valid, but statistical win rate is low ({prob_val}%).")
            print(f"      Reduce position size if you enter.")
            
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
    
    parser.add_argument('--balance', type=int, default=1000_000, help="Account Balance (IDR)")
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