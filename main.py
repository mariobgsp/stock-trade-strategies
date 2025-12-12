#!/usr/bin/env python3
"""
IHSG Swing Trading CLI - Main Entry Point
Production-Grade Interface for Indonesia Stock Exchange (IDX)

Execution: python main.py {stockticker}
Example: python main.py BBCA.JK
"""

import sys
import argparse
from datetime import datetime
from engine import (
    DataLoader, TechnicalAnalysis, PatternRecognition,
    SmartMoneyAnalysis, IndicatorOptimizer, BacktestEngine,
    SignalGenerator
)


class TradingCLI:
    """Command-line interface for trading signals."""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.metadata = None
        self.signal_data = None
    
    def fetch_and_analyze(self):
        """Fetch data and run complete analysis."""
        print(f"\n{'='*70}")
        print(f"IHSG SWING TRADING ANALYZER - {self.ticker.upper()}")
        print(f"{'='*70}\n")
        
        # Fetch data
        print(f"[*] Fetching data for {self.ticker}...")
        try:
            self.data, self.metadata = DataLoader.fetch_data(self.ticker, years=5)
        except ValueError as e:
            print(f"[ERROR] {str(e)}")
            sys.exit(1)
        
        print(f"[â] Data loaded: {len(self.data)} candles")
        print(f"    Period: {self.metadata['first_date'].date()} to {self.metadata['last_date'].date()}")
        print(f"    Years available: {self.metadata['years_available']:.2f}")
        
        if self.metadata['is_ipo']:
            print(f"    Status: IPO/New Listing (Less than 18 months)")
        else:
            print(f"    Status: Mature Stock")
        
        # Generate signal
        print(f"\n[*] Running technical analysis and backtesting...")
        self.signal_data = SignalGenerator.generate_signal(
            self.data, self.ticker, self.metadata['is_ipo']
        )
        print(f"[â] Analysis complete")
        
        return self.signal_data
    
    def display_verdict_section(self):
        """Display top-level verdict and trade plan."""
        print(f"\n{'='*70}")
        print("VERDICT & TRADE PLAN")
        print(f"{'='*70}\n")
        
        s = self.signal_data
        
        # Main verdict
        print(f"VERDICT: {s['verdict']}")
        print(f"Current Price: {s['current_price']:.2f}")
        print(f"Win Rate Confidence: {s['win_rate']:.1f}%")
        
        # Trade plan
        print(f"\nTRADE PLAN:")
        print(f"  Entry Price:       {s['entry']:.2f}")
        print(f"  Stop Loss:         {s['stop_loss']:.2f}")
        print(f"  Take Profit 1R:    {s['take_profit_1r']:.2f}")
        print(f"  Take Profit 2R:    {s['take_profit_2r']:.2f}")
        print(f"  Take Profit 3R:    {s['take_profit_3r']:.2f}")
        
        # Risk calculation
        risk = s['entry'] - s['stop_loss']
        reward_1r = s['take_profit_1r'] - s['entry']
        reward_2r = s['take_profit_2r'] - s['entry']
        reward_3r = s['take_profit_3r'] - s['entry']
        
        print(f"\nRISK/REWARD:")
        print(f"  Risk (SL):         {risk:.2f} ({(risk/s['entry']*100):.2f}%)")
        print(f"  Reward 1R:         {reward_1r:.2f} ({(reward_1r/s['entry']*100):.2f}%)")
        print(f"  Reward 2R:         {reward_2r:.2f} ({(reward_2r/s['entry']*100):.2f}%)")
        print(f"  Reward 3R:         {reward_3r:.2f} ({(reward_3r/s['entry']*100):.2f}%)")
        
        # The logic
        print(f"\nTHE LOGIC:")
        logic = self._generate_logic_text()
        for line in logic:
            print(f"  {line}")
        
        # Safety score (Win rate + target probabilities)
        print(f"\nSAFETY SCORE:")
        print(f"  Overall Win Rate:     {s['win_rate']:.1f}%")
        print(f"  Probability to 1R:    {s['target_probabilities']['1r']:.1f}%")
        print(f"  Probability to 2R:    {s['target_probabilities']['2r']:.1f}%")
        print(f"  Probability to 3R:    {s['target_probabilities']['3r']:.1f}%")
    
    def _generate_logic_text(self):
        """Generate plain English explanation of the signal."""
        s = self.signal_data
        lines = []
        
        # Strategy explanation
        if s['strategy'] == 'NO TRADE':
            lines.append("Signal: NO TRADE - Insufficient confluences at this time.")
            lines.append("Action: Wait for better setup.")
        else:
            lines.append(f"Triggered by: {s['strategy'].upper()} Strategy")
            
            # Add context
            if s['is_vcp']:
                lines.append(f"Pattern: VCP (Volatility Contraction {s['vcp_pct']:.1f}%)")
            
            if s['ma_squeeze']:
                lines.append("Pattern: MA Squeeze detected (tight 5% band)")
            
            if s['is_low_cheat']:
                lines.append("Pattern: Low Cheat setup (Minervini consolidating base)")
            
            # VWAP context
            lines.append(f"VWAP Status: Price is {s['vwap_diff_pct']:.1f}% {'ABOVE' if s['vwap_diff_pct'] > 0 else 'BELOW'} VWAP ({s['vwap']:.2f})")
            
            # Smart money
            lines.append(f"Smart Money: {s['phase']} (Strength: {s['phase_strength']}%, Started: {s['phase_start'].date()})")
            
            # RSI context
            if s['rsi'] < 30:
                lines.append(f"RSI Status: Oversold ({s['rsi']:.1f}) - potential bounce")
            elif s['rsi'] > 70:
                lines.append(f"RSI Status: Overbought ({s['rsi']:.1f}) - caution recommended")
            else:
                lines.append(f"RSI Status: Neutral ({s['rsi']:.1f}) - healthy momentum")
            
            # Final recommendation
            lines.append(f"Recommendation: {s['verdict']} with {s['win_rate']:.0f}% confidence")
        
        return lines
    
    def display_data_section(self):
        """Display detailed technical analysis data."""
        print(f"\n{'='*70}")
        print("DETAILED ANALYSIS")
        print(f"{'='*70}\n")
        
        s = self.signal_data
        
        # Asset status
        print("ASSET STATUS:")
        asset_type = "IPO/New Listing" if s['is_ipo'] else "Mature Stock"
        print(f"  Type: {asset_type}")
        print(f"  Data Points: {s['data_points']} candles")
        print(f"  History: {s['first_date'].date()} to {s['last_date'].date()}")
        print(f"  Years Available: {s['data_points']/252:.2f}")
        
        # Smart money
        print(f"\nSMART MONEY ANALYSIS:")
        print(f"  Phase: {s['phase']}")
        print(f"  Strength: {s['phase_strength']}%")
        print(f"  Phase Started: {s['phase_start'].date()}")
        print(f"  OBV Slope: {s['obv_slope']:.4f}")
        
        # Key levels
        print(f"\nKEY LEVELS:")
        print(f"  Current Price: {s['current_price']:.2f}")
        
        print(f"\n  Pivot Points:")
        print(f"    Resistance 2: {s['pivot_points']['r2']:.2f}")
        print(f"    Resistance 1: {s['pivot_points']['r1']:.2f}")
        print(f"    Pivot:        {s['pivot_points']['pivot']:.2f}")
        print(f"    Support 1:    {s['pivot_points']['s1']:.2f}")
        print(f"    Support 2:    {s['pivot_points']['s2']:.2f}")
        
        print(f"\n  Fibonacci Retracement (from swing):")
        print(f"    100% (Top):   {s['fib_levels']['level_100']:.2f}")
        print(f"    78.6%:        {s['fib_levels']['level_786']:.2f}")
        print(f"    61.8%:        {s['fib_levels']['level_618']:.2f}")
        print(f"    50.0%:        {s['fib_levels']['level_500']:.2f}")
        print(f"    38.2%:        {s['fib_levels']['level_382']:.2f}")
        print(f"    23.6%:        {s['fib_levels']['level_236']:.2f}")
        print(f"    0% (Bottom):  {s['fib_levels']['level_0']:.2f}")
        
        if s['support_levels'].size > 0:
            print(f"\n  Historical Bounce Zones (Support):")
            for i, level in enumerate(sorted(s['support_levels'])[-3:]):
                print(f"    Level {i+1}: {level:.2f}")
        
        if s['resistance_levels'].size > 0:
            print(f"\n  Historical Bounce Zones (Resistance):")
            for i, level in enumerate(s['resistance_levels'][:3]):
                print(f"    Level {i+1}: {level:.2f}")
        
        # Indicator values
        print(f"\nINDICATOR VALUES:")
        print(f"  RSI ({14}):        {s['rsi']:.2f}")
        print(f"  Stochastic %K:     {s['k_percent']:.2f}")
        print(f"  Stochastic %D:     {s['d_percent']:.2f}")
        print(f"  VWAP:              {s['vwap']:.2f}")
        print(f"  VWAP Differential: {s['vwap_diff_pct']:.2f}%")
        
        print(f"\n  Moving Averages:")
        print(f"    EMA 5:   {s['ema_5']:.2f}")
        print(f"    EMA 10:  {s['ema_10']:.2f}")
        print(f"    EMA 20:  {s['ema_20']:.2f}")
        print(f"    SMA 3:   {s['sma_3']:.2f}")
        print(f"    SMA 5:   {s['sma_5']:.2f}")
        print(f"    SMA 10:  {s['sma_10']:.2f}")
        print(f"    SMA 20:  {s['sma_20']:.2f}")
        
        # Pattern flags
        print(f"\nPATTERN FLAGS:")
        print(f"  VCP (Volatility Contraction): {'YES' if s['is_vcp'] else 'NO'}")
        if s['is_vcp']:
            print(f"    VCP Range %: {s['vcp_pct']:.2f}%")
        print(f"  MA Squeeze (5% band):        {'YES' if s['ma_squeeze'] else 'NO'}")
        print(f"  Low Cheat Pattern:           {'YES' if s['is_low_cheat'] else 'NO'}")
        
        # Backtest metrics
        if s['backtest_metrics']:
            print(f"\nBACKTEST METRICS:")
            metrics = s['backtest_metrics']
            print(f"  Strategy: {s['strategy'].upper()}")
            print(f"  Total Trades Tested: {metrics.get('total_trades', 0)}")
            print(f"  Wins: {metrics.get('wins', 0)}")
            print(f"  Losses: {metrics.get('losses', 0)}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
            if metrics.get('avg_win'):
                print(f"  Avg Win: {metrics.get('avg_win', 0):.4f}")
            if metrics.get('avg_loss'):
                print(f"  Avg Loss: {metrics.get('avg_loss', 0):.4f}")
        
        print(f"\n{'='*70}\n")
    
    def run(self):
        """Run the complete analysis pipeline."""
        try:
            self.fetch_and_analyze()
            self.display_verdict_section()
            self.display_data_section()
            
            print(f"{'='*70}")
            print(f"Analysis complete. Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Disclaimer: This is not financial advice. Always do your own research.")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n[ERROR] Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IHSG Swing Trading Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py BBCA.JK
  python main.py ASII.JK
  python main.py JSMR.JK
        """
    )
    
    parser.add_argument(
        'ticker',
        help='Stock ticker (e.g., BBCA.JK for IDX stocks)'
    )
    
    args = parser.parse_args()
    
    cli = TradingCLI(args.ticker)
    cli.run()


if __name__ == '__main__':
    main()
