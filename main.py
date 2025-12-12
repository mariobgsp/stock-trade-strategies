"""
main.py - CLI Interface for IHSG Swing Trading System
Clean, minimalist, beginner-friendly UX
"""

import sys
from engine import TradingEngine


def print_header():
    """Print clean header"""
    print("\n" + "=" * 70)
    print("      IHSG SWING TRADING SYSTEM - Professional Analysis")
    print("=" * 70 + "\n")


def print_section(title):
    """Print section separator"""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def format_currency(value):
    """Format as Indonesian Rupiah"""
    return f"Rp {value:,.0f}"


def display_signal(signal):
    """Display trading signal in clean format"""
    
    if signal is None:
        print("\n[ERROR] Unable to generate signal. Please check the ticker symbol.\n")
        return
    
    # VERDICT CARD
    print_section("VERDICT")
    verdict = signal['verdict']
    if verdict == 'BUY':
        print(f"\n   >>> {verdict} <<<")
        print(f"   Strategy: {signal['strategy']}")
    else:
        print(f"\n   >>> {verdict} <<<")
        print(f"\n   {signal['reason']}")
        print("\n   Recommendation: Wait for better market conditions.")
        print("\n" + "=" * 70 + "\n")
        return
    
    # TRADE PLAN
    print_section("TRADE PLAN")
    print(f"\n   Entry Price:        {format_currency(signal['entry'])}")
    print(f"   Stop Loss:          {format_currency(signal['stop_loss'])}")
    print(f"   Take Profit 1R:     {format_currency(signal['tp_1r'])}")
    print(f"   Take Profit 2R:     {format_currency(signal['tp_2r'])}")
    print(f"   Take Profit 3R:     {format_currency(signal['tp_3r'])}")
    
    risk = signal['entry'] - signal['stop_loss']
    reward = signal['tp_3r'] - signal['entry']
    print(f"\n   Risk per share:     {format_currency(risk)}")
    print(f"   Reward per share:   {format_currency(reward)}")
    print(f"   Risk:Reward Ratio:  1:{reward/risk:.1f}")
    
    # THE LOGIC
    print_section("THE LOGIC (Why This Trade?)")
    print(f"\n   {signal['explanation']}")
    print(f"\n   This setup has historically worked {signal['win_rate']:.1f}% of the time")
    print(f"   based on the last 3 years of data.")
    
    # SAFETY SCORE
    print_section("SAFETY SCORE")
    win_rate = signal['win_rate']
    
    if win_rate >= 65:
        safety = "HIGH"
        stars = "★★★★★"
    elif win_rate >= 60:
        safety = "GOOD"
        stars = "★★★★☆"
    elif win_rate >= 55:
        safety = "MODERATE"
        stars = "★★★☆☆"
    else:
        safety = "LOW"
        stars = "★★☆☆☆"
    
    print(f"\n   Safety Rating:  {safety} {stars}")
    print(f"   Win Rate:       {win_rate:.1f}%")
    print(f"   Confidence:     Historical backtest shows {win_rate:.0f} wins")
    print(f"                   out of every 100 similar setups.")
    
    # TECHNICAL DEEP DIVE
    print_section("TECHNICAL DEEP DIVE")
    print(f"\n   Current Price:      {format_currency(signal['current_price'])}")
    print(f"   RSI (14):           {signal['rsi']:.2f}")
    print(f"   Stochastic %K:      {signal['stoch_k']:.2f}")
    print(f"   Stochastic %D:      {signal['stoch_d']:.2f}")
    print(f"   VWAP:               {format_currency(signal['vwap'])}")
    print(f"   OBV Slope:          {signal['obv_slope']:.0f}")
    
    print(f"\n   Pattern Recognition:")
    print(f"      VCP Detected:         {'YES' if signal['is_vcp'] else 'NO'}")
    if signal['is_vcp']:
        print(f"      VCP Contractions:     {signal['vcp_contractions']}")
    print(f"      MA Squeeze:           {'YES' if signal['is_squeeze'] else 'NO'}")
    if signal['is_squeeze']:
        print(f"      Squeeze Range:        {signal['squeeze_pct']:.2f}%")
    
    print(f"\n   Smart Money Analysis:")
    print(f"      Current Phase:        {signal['smart_money_phase']}")
    print(f"      Phase Started:        {signal['smart_money_start']}")
    
    print(f"\n   Support & Resistance:")
    print(f"      Pivot Point:          {format_currency(signal['pivots']['PP'])}")
    print(f"      Resistance 1:         {format_currency(signal['pivots']['R1'])}")
    print(f"      Resistance 2:         {format_currency(signal['pivots']['R2'])}")
    print(f"      Support 1:            {format_currency(signal['pivots']['S1'])}")
    print(f"      Support 2:            {format_currency(signal['pivots']['S2'])}")
    
    print(f"\n   Fibonacci Levels:")
    for level, price in signal['fibonacci'].items():
        print(f"      {level:12s}:        {format_currency(price)}")
    
    # FOOTER
    print("\n" + "=" * 70)
    print("  Disclaimer: This is for educational purposes only.")
    print("  Always do your own research and manage your risk.")
    print("=" * 70 + "\n")


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("\n[ERROR] Please provide a stock ticker symbol.")
        print("Usage: python main.py <TICKER>")
        print("Example: python main.py BBRI\n")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    print_header()
    print(f"Analyzing {ticker} on Indonesia Stock Exchange...")
    print("Fetching 3 years of data and running advanced analysis...")
    print("This may take 30-60 seconds...\n")
    
    # Initialize engine
    engine = TradingEngine(ticker, years=3)
    
    # Fetch data
    if not engine.fetch_data():
        print(f"\n[ERROR] Could not fetch data for {ticker}.")
        print("Please check if the ticker is valid on IDX.\n")
        sys.exit(1)
    
    print("Data loaded successfully. Running backtests and optimization...")
    
    # Generate signal
    signal = engine.generate_signal()
    
    # Display results
    display_signal(signal)


if __name__ == "__main__":
    main()
