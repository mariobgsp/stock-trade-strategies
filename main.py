#!/usr/bin/env python3
"""
IHSG Swing Trade Analysis CLI
Command: python main.py {stockticker}
Example: python main.py TLKM.JK
"""

import sys
import os
from datetime import datetime
from engine import (
    SwingTradeEngine, TradeSignal, BandarPhase
)
from typing import Optional

class TradingCLI:
    """Command Line Interface for Swing Trade Analysis"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.engine = None
        self.recommendation = None
        
    def run(self):
        """Main execution method"""
        print("\n" + "="*80)
        print(f"IHSG SWING TRADE ANALYSIS ENGINE")
        print(f"Analyzing: {self.ticker}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        try:
            # Initialize engine
            print(f"📊 Loading 3 years of historical data for {self.ticker}...")
            self.engine = SwingTradeEngine(self.ticker)
            print(f"✅ Data loaded: {len(self.engine.df)} candles\n")
            
            # Generate recommendation
            print("⚙️  Analyzing multiple indicators and patterns...")
            self.recommendation = self.engine.generate_trade_recommendation()
            print("✅ Analysis complete\n")
            
            # Display results
            self.display_results()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def display_results(self):
        """Display analysis results in formatted output"""
        rec = self.recommendation
        
        # SECTION 1: CURRENT PRICE & SUPPORT/RESISTANCE
        print("━"*80)
        print("1️⃣  PRICE & SUPPORT/RESISTANCE ANALYSIS")
        print("━"*80)
        print(f"Current Price:        ${rec.entry_price:.2f}")
        print(f"Nearest Support:      ${rec.nearest_support:.2f} ({self._calc_pct_distance(rec.entry_price, rec.nearest_support)}% away)")
        print(f"Nearest Resistance:   ${rec.nearest_resistance:.2f} ({self._calc_pct_distance(rec.entry_price, rec.nearest_resistance)}% away)")
        print(f"VWAP (20-period):     ${rec.vwap:.2f}")
        print()
        
        # SECTION 2: BANDAR FLOW ANALYSIS
        print("━"*80)
        print("2️⃣  BANDAR FLOW ANALYSIS (Institutional Activity)")
        print("━"*80)
        bandar_status = self._get_bandar_status(rec.bandar_phase)
        print(f"Bandar Phase:         {bandar_status}")
        print(f"Status:               {'📈 ACCUMULATING - Institutional buyers entering' if rec.bandar_phase == 'ACCUMULATING' else '📉 DISTRIBUTING - Institutional selling' if rec.bandar_phase == 'DISTRIBUTING' else '➡️  NEUTRAL - No clear direction'}")
        print()
        
        # SECTION 3: TECHNICAL INDICATORS ANALYSIS
        print("━"*80)
        print("3️⃣  DYNAMIC TECHNICAL INDICATORS (Optimized)")
        print("━"*80)
        optimal = self.engine.indicator_optimizer.find_optimal_indicators()
        
        print(f"Recommended MA Periods:     {optimal['ma_periods']}")
        for period in optimal['ma_periods']:
            if period in optimal['ma_values']:
                dist = self._calc_pct_distance(rec.entry_price, optimal['ma_values'][period])
                print(f"  └─ MA({period}): ${optimal['ma_values'][period]:.2f} ({dist}% away)")
        print()
        
        print(f"Optimal RSI Period:         {optimal['rsi_period']}")
        print(f"Current RSI Value:          {optimal['rsi_value']:.2f}")
        rsi_interpretation = self._interpret_rsi(optimal['rsi_value'])
        print(f"RSI Signal:                 {rsi_interpretation}")
        print()
        
        print(f"Optimal Stochastic Period:  {optimal['stoch_period']}")
        print(f"Stochastic %K:              {optimal['stoch_k']:.2f}")
        print(f"Stochastic %D:              {optimal['stoch_d']:.2f}")
        stoch_signal = self._interpret_stochastic(optimal['stoch_k'], optimal['stoch_d'])
        print(f"Stochastic Signal:          {stoch_signal}")
        print()
        
        # SECTION 4: PATTERN RECOGNITION
        print("━"*80)
        print("4️⃣  PATTERN RECOGNITION (Mark Minervini Strategies)")
        print("━"*80)
        
        # VCP
        vcp_detected, vcp_conf, vcp_desc = self.engine.pattern_analyzer.detect_vcp()
        vcp_marker = "✅ DETECTED" if vcp_detected else "❌ Not detected"
        print(f"VCP (Volatility Contraction): {vcp_marker} - {vcp_desc}")
        
        # Low Cheat
        low_cheat, low_cheat_conf, low_cheat_desc = self.engine.pattern_analyzer.detect_low_cheat()
        lc_marker = "✅ DETECTED" if low_cheat else "❌ Not detected"
        print(f"Low Cheat Pattern:            {lc_marker} - {low_cheat_desc}")
        
        # Superclose MA
        superclose_ma, superclose_conf, superclose_desc = self.engine.pattern_analyzer.detect_superclose_ma()
        sc_marker = "✅ DETECTED" if superclose_ma else "❌ Not detected"
        print(f"Superclose MA (Squeeze Ready): {sc_marker} - {superclose_desc}")
        
        # Consolidation
        consolidation, consol_conf, consol_desc = self.engine.pattern_analyzer.detect_consolidation()
        cons_marker = "✅ DETECTED" if consolidation else "❌ Not detected"
        print(f"Consolidation Pattern:        {cons_marker} - {consol_desc}")
        print()
        
        # SECTION 5: TRADE RECOMMENDATION
        print("━"*80)
        print("5️⃣  TRADE RECOMMENDATION & ENTRY STRATEGY")
        print("━"*80)
        
        signal_emoji = self._get_signal_emoji(rec.signal)
        print(f"Trade Signal:               {signal_emoji} {rec.signal.value}")
        print(f"Signal Strength:            {self._get_signal_strength_bar(rec.success_probability)}")
        print(f"Technical Setup Strength:   {self._get_strength_bar(rec.technical_strength)}")
        print()
        
        print(f"Entry Price:                ${rec.entry_price:.2f}")
        print(f"Entry Reason:               {rec.entry_reason}")
        print()
        
        print(f"Stop Loss:                  ${rec.stop_loss:.2f} (-{self._calc_pct_distance(rec.entry_price, rec.stop_loss):.2f}%)")
        print(f"Target 1R (1:1 Risk/Reward): ${rec.target_1r:.2f} (+{self._calc_pct_distance(rec.entry_price, rec.target_1r):.2f}%)")
        print(f"Target 3R (1:3 Risk/Reward): ${rec.target_3r:.2f} (+{self._calc_pct_distance(rec.entry_price, rec.target_3r):.2f}%)")
        print(f"Risk/Reward Ratio:          1:{rec.risk_reward_ratio:.1f}")
        print()
        
        # SECTION 6: PROBABILITY ANALYSIS
        print("━"*80)
        print("6️⃣  SUCCESS PROBABILITY & TRADE ANALYSIS")
        print("━"*80)
        
        prob_percentage = rec.success_probability
        print(f"Overall Success Probability: {prob_percentage:.1f}%")
        print(f"Probability Assessment:     {self._get_probability_assessment(prob_percentage)}")
        print()
        
        print(f"Price Reaching 1R Target:   {self._estimate_target_probability(prob_percentage, 1):.1f}%")
        print(f"Price Reaching 3R Target:   {self._estimate_target_probability(prob_percentage, 3):.1f}%")
        print()
        
        # SECTION 7: SUPPORTING INDICATORS
        print("━"*80)
        print("7️⃣  SUPPORTING INDICATORS CONFLUENCE")
        print("━"*80)
        
        confluence_count = len(rec.supporting_indicators)
        print(f"Total Confluences:          {confluence_count} indicators aligned\n")
        
        for i, indicator in enumerate(rec.supporting_indicators, 1):
            print(f"  {i}. ✓ {indicator}")
        print()
        
        # SECTION 8: TRADING ACTION
        print("━"*80)
        print("8️⃣  TRADING ACTION & NEXT STEPS")
        print("━"*80)
        
        action = self._get_trading_action(rec.signal, prob_percentage)
        print(f"Action:                     {action}")
        print()
        
        # Entry conditions
        print("Entry Conditions:")
        entry_conditions = self._get_entry_conditions(rec.pattern_detected)
        for condition in entry_conditions:
            print(f"  • {condition}")
        print()
        
        # Risk management
        print("Risk Management:")
        print(f"  • Position Size:         Risk max {self._calc_pct_distance(rec.entry_price, rec.stop_loss):.2f}% of portfolio")
        print(f"  • Reward Target:         Aim for 3R or more ({rec.risk_reward_ratio:.1f}R setup)")
        print(f"  • Stop Placement:        STRICTLY at ${rec.stop_loss:.2f}")
        print(f"  • Scale Out:             Consider taking profit at 1R (${rec.target_1r:.2f})")
        print()
        
        # SECTION 9: SUMMARY
        print("━"*80)
        print("9️⃣  ANALYSIS SUMMARY & RECOMMENDATION")
        print("━"*80)
        
        summary = self._generate_summary(rec)
        print(summary)
        print()
        
        print("="*80)
        print("Analysis Complete")
        print("="*80)
        print("\n📌 DISCLAIMER:")
        print("   This analysis is for educational purposes only.")
        print("   Past performance does not guarantee future results.")
        print("   Always conduct your own due diligence before trading.")
        print("   Follow OJK guidelines and risk management rules.\n")
    
    # ==================== HELPER METHODS ====================
    
    def _calc_pct_distance(self, price1: float, price2: float) -> float:
        """Calculate percentage distance between two prices"""
        return abs((price2 - price1) / price1 * 100)
    
    def _get_bandar_status(self, phase: str) -> str:
        """Get bandar status emoji and text"""
        phases = {
            'ACCUMULATING': '🟢 ACCUMULATING',
            'DISTRIBUTING': '🔴 DISTRIBUTING',
            'NEUTRAL': '🟡 NEUTRAL',
            'TRANSITION': '🟠 TRANSITION'
        }
        return phases.get(phase, phase)
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi < 30:
            return f"🔵 Oversold - Strong buying signal"
        elif rsi < 40:
            return f"🟢 Approaching oversold - Favorable entry"
        elif rsi < 50:
            return f"🟡 Neutral - Building upside momentum"
        elif rsi < 60:
            return f"🟠 Neutral - Moderate momentum"
        elif rsi < 70:
            return f"🟠 Approaching overbought - Caution"
        else:
            return f"🔴 Overbought - Consider waiting for pullback"
    
    def _interpret_stochastic(self, k: float, d: float) -> str:
        """Interpret Stochastic values"""
        if pd.isna(k) or pd.isna(d):
            return "N/A"
        
        if k < 30 and d < 30:
            return f"🔵 Oversold - Strong buy signal"
        elif k > d and k < 50:
            return f"🟢 Bullish cross - Entry signal"
        elif k > 70:
            return f"🔴 Overbought - Caution"
        else:
            return f"🟡 Neutral"
    
    def _get_signal_emoji(self, signal: TradeSignal) -> str:
        """Get emoji for trade signal"""
        emojis = {
            TradeSignal.STRONG_BUY: '🟢🟢 STRONG BUY',
            TradeSignal.BUY: '🟢 BUY',
            TradeSignal.WEAK_BUY: '🟡 WEAK BUY',
            TradeSignal.HOLD: '🟠 HOLD',
            TradeSignal.WAIT: '🔴 WAIT',
            TradeSignal.SELL: '❌ SELL'
        }
        return emojis.get(signal, str(signal))
    
    def _get_signal_strength_bar(self, probability: float) -> str:
        """Get visual bar for signal strength"""
        filled = int(probability / 10)
        empty = 10 - filled
        bar = '█' * filled + '░' * empty
        return f"{bar} {probability:.1f}%"
    
    def _get_strength_bar(self, strength: float) -> str:
        """Get visual bar for technical strength"""
        filled = int(strength / 10)
        empty = 10 - filled
        bar = '█' * filled + '░' * empty
        return f"{bar} {strength:.1f}%"
    
    def _get_probability_assessment(self, prob: float) -> str:
        """Get assessment based on probability"""
        if prob >= 80:
            return "🟢 Very High - Excellent setup"
        elif prob >= 70:
            return "🟢 High - Strong setup"
        elif prob >= 60:
            return "🟡 Moderate - Decent setup"
        elif prob >= 50:
            return "🟠 Fair - Some risk"
        else:
            return "🔴 Low - Wait for better setup"
    
    def _estimate_target_probability(self, base_prob: float, target_r: int) -> float:
        """Estimate probability of reaching target based on RR"""
        # Higher targets have slightly lower probability
        reduction = (target_r - 1) * 5
        return max(base_prob - reduction, 20.0)
    
    def _get_trading_action(self, signal: TradeSignal, prob: float) -> str:
        """Get recommended action"""
        if signal == TradeSignal.STRONG_BUY and prob >= 75:
            return "🚀 BUY NOW - High confidence setup"
        elif signal == TradeSignal.BUY and prob >= 65:
            return "📈 BUY on pullback or breakout confirmation"
        elif signal == TradeSignal.WEAK_BUY:
            return "⏳ WAIT for better confirmation or entry at support"
        elif signal == TradeSignal.HOLD:
            return "🔄 MONITOR - Setup forming, wait for confirmation"
        else:
            return "⏸️  WAIT - Not ready, insufficient setup strength"
    
    def _get_entry_conditions(self, pattern: str) -> list:
        """Get specific entry conditions based on pattern"""
        conditions = []
        
        if pattern == "VCP":
            conditions = [
                "Wait for breakout above upper consolidation range",
                "Entry on breakout with volume confirmation",
                "Stop loss just below consolidation low"
            ]
        elif pattern == "Low Cheat":
            conditions = [
                "Enter on confirmation candle closing above low",
                "Look for reversal pattern confirmation",
                "Stop loss just below the created low"
            ]
        elif pattern == "Superclose MA":
            conditions = [
                "Enter on first breakout candle",
                "Wait for close above resistance",
                "Volatility squeeze breakout imminent"
            ]
        elif pattern == "Consolidation":
            conditions = [
                "Buy on breakout above consolidation range",
                "Alternatively, buy on deep retracement to support",
                "Stop loss at consolidation low"
            ]
        else:
            conditions = [
                "Enter on confluence of multiple indicators",
                "Confirm with volume and price action",
                "Use support level as stop loss"
            ]
        
        return conditions
    
    def _generate_summary(self, rec) -> str:
        """Generate summary recommendation"""
        summary = f"""
The {self.ticker} setup shows the following characteristics:

PRIMARY SETUP: {rec.pattern_detected}
- The stock is forming a {rec.pattern_detected} pattern with {rec.technical_strength:.0f}% technical strength
- Bandar phase indicates {rec.bandar_phase} behavior
- Multiple indicators are aligned with {len(rec.supporting_indicators)} confluence points

KEY LEVELS:
- Support: ${rec.nearest_support:.2f} (Entry area / Stop Loss area)
- Current Price: ${rec.entry_price:.2f}
- Resistance: ${rec.nearest_resistance:.2f} (Profit target area)

PROBABILITY ASSESSMENT:
- {rec.success_probability:.0f}% chance of reaching 1R target
- {self._estimate_target_probability(rec.success_probability, 3):.0f}% chance of reaching 3R target
- Risk/Reward ratio of 1:{rec.risk_reward_ratio:.1f} provides excellent risk management

RECOMMENDATION:
{self._get_trading_action(rec.signal, rec.success_probability)}

Remember to:
✓ Only risk 1-2% of your portfolio per trade
✓ Use proper position sizing
✓ Follow the 3:1 risk/reward setup
✓ Wait for confirmation before entering
✓ Respect your stop loss at all times
        """
        return summary

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py {STOCK_TICKER}")
        print("Example: python main.py TLKM.JK")
        print("\nSupported: Indonesian stocks with .JK extension")
        sys.exit(1)
    
    ticker = sys.argv[1]
    
    # Validate ticker format
    if not ticker.endswith('.JK'):
        ticker = ticker + '.JK'
    
    cli = TradingCLI(ticker)
    cli.run()

if __name__ == "__main__":
    import pandas as pd  # Add missing import
    main()