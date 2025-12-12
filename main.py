import argparse
import sys
from engine import TradingEngine

def print_header(title):
    print(f"\n{'='*50}")
    print(f" {title.upper()}")
    print(f"{'='*50}")

def print_separator():
    print(f"{'-'*50}")

def format_percentage(val):
    return f"{val * 100:.1f}%"

def main():
    parser = argparse.ArgumentParser(description='Senior Quant Swing Trading CLI for IDX')
    parser.add_argument('ticker', type=str, help='Stock Ticker (e.g., BBRI, GOTO)')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"Initializing Quant Engine for {ticker}...")
    
    engine = TradingEngine(ticker)
    result = engine.analyze()

    if "error" in result:
        print(f"CRITICAL ERROR: {result['error']}")
        sys.exit(1)

    # --- VERDICT SECTION ---
    print_header("STRATEGIC VERDICT")
    
    print(f"VERDICT:  {result['verdict']}")
    print_separator()
    
    print("TRADE PLAN:")
    print(f"  Entry Price:  {result['entry']:,.0f}")
    print(f"  Stop Loss:    {result['stop_loss']:,.0f}")
    print(f"  Target 1 (1R): {result['tp1']:,.0f}")
    print(f"  Target 2 (2R): {result['tp2']:,.0f}")
    print(f"  Target 3 (3R): {result['tp3']:,.0f}")
    
    print_separator()
    print("THE LOGIC:")
    
    vwap_context = "Healthy" if result['vwap_diff'] > 0 else "Weak"
    pos_neg = "ABOVE" if result['vwap_diff'] > 0 else "BELOW"
    
    explanation = (
        f"Triggered by {result['strategy']} logic. "
        f"Price is {vwap_context} ({abs(result['vwap_diff']):.2f}% {pos_neg} VWAP). "
    )
    
    if result['accum_status'] == "Accumulation":
        explanation += "Smart Money is accumulating (OBV Slope +)."
    elif result['accum_status'] == "Distribution":
        explanation += "WARNING: Smart Money distribution detected."
        
    print(explanation)
    
    print_separator()
    print("SAFETY SCORE (Backtest Validation):")
    print(f"  Historical Win Rate: {format_percentage(result['win_rate'])}")
    print(f"  Prob to hit 1R:      {format_percentage(result['probs']['1R'])}")
    print(f"  Prob to hit 2R:      {format_percentage(result['probs']['2R'])}")
    print(f"  Prob to hit 3R:      {format_percentage(result['probs']['3R'])}")

    # --- DATA SECTION ---
    print_header("QUANTITATIVE DATA")
    
    # Asset Status
    asset_type = "IPO / New Listing (< 6 mo data)" if result['is_ipo'] else "Mature Stock"
    print(f"Asset Class:  {asset_type}")
    
    # Smart Money
    print(f"Smart Money:  {result['accum_status']} (Slope: {result['obv_slope']:.4f})")
    print(f"Phase Start:  {result['accum_start']}")
    
    print_separator()
    print("KEY LEVELS & PATTERNS:")
    print(f"  RSI (14):     {result['rsi']:.2f}")
    print(f"  VWAP:         {result['indicators']['VWAP']:,.2f}")
    print(f"  VCP Pattern:  {'DETECTED' if result['vcp_detected'] else 'None'}")
    print(f"  MA Squeeze:   {'ACTIVE' if result['ma_squeeze'] else 'None'}")
    
    print("\nDisclaimer: For Educational Purposes Only. Not Financial Advice.")
    print("Market Data provided by yfinance. Rules based on IDX/OJK constraints.")

if __name__ == "__main__":
    main()

