"""
IHSG Swing Trade Analysis Engine
Professional Quantitative Trading Analysis System
Author: Quant Analyst
Purpose: Multi-indicator swing trade analysis for Indonesian stocks
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ==================== ENUMS & DATA CLASSES ====================

class TradeSignal(Enum):
    """Trade signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WAIT = "WAIT"
    SELL = "SELL"

class BandarPhase(Enum):
    """Bandar accumulation/distribution phase"""
    ACCUMULATING = "ACCUMULATING"
    DISTRIBUTING = "DISTRIBUTING"
    NEUTRAL = "NEUTRAL"
    TRANSITION = "TRANSITION"

@dataclass
class SupportResistanceLevel:
    """Support/Resistance level data"""
    price: float
    type: str  # "PIVOT", "FIBONACCI", "SUPPORT", "RESISTANCE"
    strength: float  # 0-1 confidence level
    source: str  # Where it came from
    bounces: int = 0  # Number of bounces from this level

@dataclass
class TradeRecommendation:
    """Trade recommendation output"""
    signal: TradeSignal
    entry_price: float
    entry_reason: str
    stop_loss: float
    target_1r: float  # 1R target
    target_3r: float  # 3R target
    success_probability: float  # 0-100%
    risk_reward_ratio: float
    supporting_indicators: List[str]
    bandar_phase: str
    vwap: float
    nearest_support: float
    nearest_resistance: float
    pattern_detected: str
    technical_strength: float  # 0-100% how strong technical setup is

# ==================== SUPPORT & RESISTANCE ANALYSIS ====================

class SupportResistanceAnalyzer:
    """Analyze support and resistance levels using multiple methods"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.high = df['High'].values
        self.low = df['Low'].values
        self.close = df['Close'].values
        self.volume = df['Volume'].values
        
    def calculate_pivot_points(self) -> Dict:
        """Calculate classic pivot points"""
        high = self.high[-1]
        low = self.low[-1]
        close = self.close[-1]
        
        pivot = (high + low + close) / 3
        resistance1 = (2 * pivot) - low
        support1 = (2 * pivot) - high
        resistance2 = pivot + (high - low)
        support2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'r1': resistance1,
            'r2': resistance2,
            's1': support1,
            's2': support2,
            'method': 'pivot_points'
        }
    
    def calculate_fibonacci_support(self, lookback: int = 252) -> Dict:
        """Calculate Fibonacci retracement from highest high"""
        if len(self.df) < lookback:
            lookback = len(self.df) - 1
        
        recent = self.df.iloc[-lookback:]
        highest_high = recent['High'].max()
        lowest_low = recent['Low'].min()
        
        diff = highest_high - lowest_low
        
        # Fibonacci levels
        levels = {
            'level_0': highest_high,
            'level_236': highest_high - (diff * 0.236),
            'level_382': highest_high - (diff * 0.382),
            'level_500': highest_high - (diff * 0.500),
            'level_618': highest_high - (diff * 0.618),
            'level_786': highest_high - (diff * 0.786),
            'level_1': lowest_low,
        }
        
        return {
            'highest_high': highest_high,
            'lowest_low': lowest_low,
            'levels': levels,
            'method': 'fibonacci'
        }
    
    def find_bounce_prices(self, lookback: int = 60) -> List[float]:
        """Find historical bounce prices (local support levels)"""
        recent = self.df.iloc[-lookback:]
        bounces = []
        
        # Find local minima (bounce points)
        for i in range(1, len(recent) - 1):
            if (recent['Low'].iloc[i] < recent['Low'].iloc[i-1] and 
                recent['Low'].iloc[i] < recent['Low'].iloc[i+1]):
                bounces.append(recent['Low'].iloc[i])
        
        # Sort by recency and proximity
        bounces = sorted(set(bounces), reverse=True)[:5]
        return bounces
    
    def get_nearest_support_resistance(self) -> Tuple[float, float]:
        """Get nearest support and resistance from multiple methods"""
        current_price = self.close[-1]
        
        pivots = self.calculate_pivot_points()
        fibs = self.calculate_fibonacci_support()
        bounces = self.find_bounce_prices()
        
        # Collect all levels
        all_supports = []
        all_resistances = []
        
        # Add pivot points
        if pivots['s1'] < current_price:
            all_supports.append(pivots['s1'])
        if pivots['s2'] < current_price:
            all_supports.append(pivots['s2'])
        if pivots['r1'] > current_price:
            all_resistances.append(pivots['r1'])
        if pivots['r2'] > current_price:
            all_resistances.append(pivots['r2'])
        
        # Add Fibonacci levels
        for level_name, level_price in fibs['levels'].items():
            if level_price < current_price and level_price > 0:
                all_supports.append(level_price)
            elif level_price > current_price:
                all_resistances.append(level_price)
        
        # Add bounce prices
        all_supports.extend([b for b in bounces if b < current_price])
        
        # Get nearest
        nearest_support = max(all_supports) if all_supports else current_price * 0.95
        nearest_resistance = min(all_resistances) if all_resistances else current_price * 1.05
        
        return nearest_support, nearest_resistance

# ==================== BANDAR FLOW ANALYSIS ====================

class BandarFlowAnalyzer:
    """Analyze institutional accumulation/distribution patterns"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.close = df['Close'].values
        self.volume = df['Volume'].values
        self.high = df['High'].values
        self.low = df['Low'].values
        self.open = df['Open'].values
        
    def calculate_vwap(self, period: int = 20) -> float:
        """Calculate Volume Weighted Average Price"""
        recent = self.df.iloc[-period:]
        typical_price = (recent['High'] + recent['Low'] + recent['Close']) / 3
        vwap = (typical_price * recent['Volume']).sum() / recent['Volume'].sum()
        return float(vwap)
    
    def calculate_on_balance_volume(self) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(0.0, index=self.df.index)
        obv.iloc[0] = self.volume[0]
        
        for i in range(1, len(self.df)):
            if self.close[i] > self.close[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.volume[i]
            elif self.close[i] < self.close[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.volume[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_accumulation_distribution(self) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        clv = np.where(self.high == self.low, 0, clv)
        ad = pd.Series(clv * self.volume).cumsum()
        return ad
    
    def calculate_money_flow_index(self, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        tp = (self.high + self.low + self.close) / 3
        mf = tp * self.volume
        
        positive_mf = pd.Series(0.0, index=self.df.index)
        negative_mf = pd.Series(0.0, index=self.df.index)
        
        for i in range(1, len(tp)):
            if tp[i] > tp[i-1]:
                positive_mf.iloc[i] = mf[i]
            else:
                negative_mf.iloc[i] = mf[i]
        
        pos_mf_sum = positive_mf.rolling(period).sum()
        neg_mf_sum = negative_mf.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + pos_mf_sum / neg_mf_sum))
        return mfi
    
    def detect_accumulation_phase(self, lookback: int = 20) -> Tuple[BandarPhase, float]:
        """Detect if bandar is accumulating or distributing"""
        recent = self.df.iloc[-lookback:]
        
        # Get indicators
        obv = self.calculate_on_balance_volume()
        ad = self.calculate_accumulation_distribution()
        mfi = self.calculate_money_flow_index()
        
        obv_trend = obv.iloc[-1] - obv.iloc[-lookback]
        ad_trend = ad.iloc[-1] - ad.iloc[-lookback]
        mfi_recent = mfi.iloc[-1]
        
        # Get volume profile
        avg_volume = recent['Volume'].mean()
        current_volume = recent['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Calculate price movement
        price_change = ((recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / 
                       recent['Close'].iloc[0]) * 100
        
        # Scoring system
        accumulation_score = 0
        
        # OBV accumulation (rising OBV = accumulation)
        if obv_trend > 0:
            accumulation_score += 2
        
        # A/D accumulation
        if ad_trend > 0:
            accumulation_score += 2
        
        # MFI (< 40 = accumulation, > 60 = distribution)
        if mfi_recent < 40:
            accumulation_score += 2
        elif mfi_recent > 60:
            accumulation_score -= 2
        
        # Volume absorption (high volume with small price move = accumulation)
        if volume_ratio > 1.3:
            if abs(price_change) < 1:
                accumulation_score += 2
            elif price_change > 0:
                accumulation_score += 1
        
        # Price action on low volume (distribution)
        if volume_ratio < 0.8 and price_change > 2:
            accumulation_score -= 1
        
        confidence = min(abs(accumulation_score) / 6.0, 1.0)
        
        if accumulation_score >= 4:
            return BandarPhase.ACCUMULATING, confidence
        elif accumulation_score <= -2:
            return BandarPhase.DISTRIBUTING, confidence
        else:
            return BandarPhase.NEUTRAL, confidence
    
    def detect_candle_absorption(self, lookback: int = 5) -> Tuple[bool, float]:
        """Detect candle absorption pattern (accumulation indicator)"""
        recent = self.df.iloc[-lookback:].reset_index(drop=True)
        
        absorption_count = 0
        total_possible = len(recent) - 1
        
        for i in range(1, len(recent)):
            curr_high = recent['High'].iloc[i]
            curr_low = recent['Low'].iloc[i]
            curr_open = recent['Open'].iloc[i]
            curr_close = recent['Close'].iloc[i]
            
            prev_high = recent['High'].iloc[i-1]
            prev_low = recent['Low'].iloc[i-1]
            
            # Inside bar (absorption indicator)
            if curr_high <= prev_high and curr_low >= prev_low:
                absorption_count += 1
        
        absorption_ratio = absorption_count / total_possible if total_possible > 0 else 0
        return absorption_count > 0, absorption_ratio

# ==================== DYNAMIC INDICATORS OPTIMIZER ====================

class DynamicIndicatorsOptimizer:
    """Optimize and calculate dynamic moving averages, RSI, and Stochastic"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.close = df['Close'].values
        self.high = df['High'].values
        self.low = df['Low'].values
        
    def calculate_ma(self, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return pd.Series(self.close).rolling(period).mean()
    
    def calculate_ema(self, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return pd.Series(self.close).ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        close_series = pd.Series(self.close)
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator (K and D)"""
        high_series = pd.Series(self.high)
        low_series = pd.Series(self.low)
        close_series = pd.Series(self.close)
        
        lowest_low = low_series.rolling(period).min()
        highest_high = high_series.rolling(period).max()
        
        k = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
        k_smooth = k.rolling(smooth_k).mean()
        d = k_smooth.rolling(smooth_d).mean()
        
        return k_smooth, d
    
    def optimize_indicators(self) -> Dict:
        """Test multiple indicator combinations"""
        ma_periods = [5, 10, 20, 30, 50, 100]
        rsi_periods = [9, 14, 21, 25, 28]
        stoch_periods = [9, 14, 21, 28]
        
        results = {}
        
        # Test different MA combinations
        for period in ma_periods:
            ma = self.calculate_ma(period)
            results[f'MA_{period}'] = ma
        
        # Test different RSI periods
        for period in rsi_periods:
            rsi = self.calculate_rsi(period)
            results[f'RSI_{period}'] = rsi
        
        # Test different Stochastic periods
        for period in stoch_periods:
            k, d = self.calculate_stochastic(period)
            results[f'Stoch_{period}_K'] = k
            results[f'Stoch_{period}_D'] = d
        
        return results
    
    def find_optimal_indicators(self) -> Dict:
        """Find optimal indicators based on current price action"""
        current_price = self.close[-1]
        recent_data = self.df.iloc[-60:]
        
        optimal = {
            'ma_periods': [],
            'rsi_period': None,
            'stoch_period': None
        }
        
        # Find MAs closest to current price (for trend confirmation)
        ma_scores = []
        for period in [5, 10, 20, 30, 50]:
            ma = self.calculate_ma(period).iloc[-1]
            if not pd.isna(ma):
                distance = abs(current_price - ma) / current_price
                trend = "uptrend" if ma < current_price else "downtrend"
                ma_scores.append((period, ma, distance, trend))
        
        # Top 5 MAs by relevance
        ma_scores.sort(key=lambda x: x[2])
        optimal['ma_periods'] = [m[0] for m in ma_scores[:5]]
        optimal['ma_values'] = {m[0]: m[1] for m in ma_scores[:5]}
        
        # Optimize RSI
        best_rsi_period = 14
        best_rsi_score = 0
        for period in [9, 14, 21, 25, 28]:
            rsi = self.calculate_rsi(period).iloc[-1]
            if 30 < rsi < 70:  # Good RSI range for swing trading
                best_rsi_score = abs(50 - rsi)  # Closer to 50 = optimal
                best_rsi_period = period
        optimal['rsi_period'] = best_rsi_period
        optimal['rsi_value'] = self.calculate_rsi(best_rsi_period).iloc[-1]
        
        # Optimize Stochastic
        best_stoch_period = 14
        best_stoch_score = 0
        for period in [9, 14, 21, 28]:
            k, d = self.calculate_stochastic(period)
            k_val = k.iloc[-1]
            if not pd.isna(k_val) and (k_val < 30 or k_val > 70):
                best_stoch_period = period
        optimal['stoch_period'] = best_stoch_period
        k, d = self.calculate_stochastic(best_stoch_period)
        optimal['stoch_k'] = k.iloc[-1]
        optimal['stoch_d'] = d.iloc[-1]
        
        return optimal

# ==================== PATTERN RECOGNITION ====================

class PatternRecognition:
    """Detect VCP, Low Cheat, and Superclose MA patterns"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.close = df['Close'].values
        self.high = df['High'].values
        self.low = df['Low'].values
        self.volume = df['Volume'].values
        self.open = df['Open'].values
        
    def detect_vcp(self, lookback: int = 60) -> Tuple[bool, float, str]:
        """
        Detect Volatility Contraction Pattern (Mark Minervini)
        Typical consolidation with decreasing volatility before breakout
        """
        recent = self.df.iloc[-lookback:].reset_index(drop=True)
        
        # Calculate volatility
        returns = np.log(recent['Close'] / recent['Close'].shift(1))
        volatility = returns.rolling(10).std()
        
        # Check for contracting volatility
        vol_trend = volatility.iloc[-1] - volatility.iloc[-20]
        vol_contracting = vol_trend < 0
        
        # Check for consolidation (price range)
        price_high = recent['High'].iloc[-30:].max()
        price_low = recent['Low'].iloc[-30:].min()
        price_range = (price_high - price_low) / price_low
        
        consolidation = price_range < 0.10  # Less than 10% range
        
        # Check volume trend (should be declining)
        avg_vol_early = recent['Volume'].iloc[-30:-20].mean()
        avg_vol_late = recent['Volume'].iloc[-10:].mean()
        declining_volume = avg_vol_late < avg_vol_early
        
        vcp_detected = (vol_contracting and consolidation and declining_volume)
        confidence = 0.8 if vcp_detected else 0.0
        
        pattern_desc = ""
        if vcp_detected:
            pattern_desc = f"VCP forming with {price_range*100:.1f}% range and declining volume"
        
        return vcp_detected, confidence, pattern_desc
    
    def detect_low_cheat(self, lookback: int = 20) -> Tuple[bool, float, str]:
        """
        Detect Low Cheat Pattern
        Price makes low below support but closes above support (institutional accumulation)
        """
        recent = self.df.iloc[-lookback:].reset_index(drop=True)
        
        low_cheat_signals = 0
        total_bars = 0
        
        for i in range(2, len(recent)):
            prev_low = recent['Low'].iloc[i-2:i].min()
            curr_low = recent['Low'].iloc[i]
            curr_close = recent['Close'].iloc[i]
            
            # Low goes below previous support but closes above it
            if curr_low < prev_low and curr_close > prev_low:
                low_cheat_signals += 1
            
            total_bars += 1
        
        low_cheat_detected = low_cheat_signals >= 2
        confidence = min(low_cheat_signals / total_bars, 1.0) if total_bars > 0 else 0
        
        pattern_desc = f"Low Cheat detected: {low_cheat_signals} instances"
        
        return low_cheat_detected, confidence, pattern_desc
    
    def detect_superclose_ma(self, threshold: float = 0.05) -> Tuple[bool, float, str]:
        """
        Detect Superclose MA Pattern
        All moving averages (3, 5, 10, 20) are close together (within 5% of price)
        Ready for volatility squeeze and breakout
        """
        ma_periods = [3, 5, 10, 20]
        mas = {}
        
        for period in ma_periods:
            ma = pd.Series(self.close).rolling(period).mean()
            mas[period] = ma.iloc[-1]
        
        current_price = self.close[-1]
        
        # Check if all MAs are within threshold
        all_close = True
        for period, ma_value in mas.items():
            if pd.isna(ma_value):
                all_close = False
                break
            distance = abs(ma_value - current_price) / current_price
            if distance > threshold:
                all_close = False
                break
        
        if all_close:
            # Calculate average distance
            distances = [abs(ma - current_price) / current_price for ma in mas.values()]
            avg_distance = np.mean(distances)
            confidence = 1.0 - avg_distance
            
            pattern_desc = f"SUPERCLOSE MA - All MAs within {avg_distance*100:.2f}% - Breakout imminent!"
            return True, confidence, pattern_desc
        
        return False, 0.0, "MAs not converged"
    
    def detect_consolidation(self, lookback: int = 30) -> Tuple[bool, float, str]:
        """Detect consolidation pattern for potential breakout"""
        recent = self.df.iloc[-lookback:]
        
        high = recent['High'].max()
        low = recent['Low'].min()
        range_pct = (high - low) / low * 100
        
        # Consolidation: low volatility (< 5% range in period)
        consolidating = range_pct < 5
        
        current = self.close[-1]
        middle = (high + low) / 2
        
        consolidation_score = 0
        if consolidating:
            consolidation_score += 2
        
        # Check if price is in upper or lower part of range
        if current > middle:
            consolidation_score += 1  # Bullish consolidation
        
        confidence = min(consolidation_score / 3.0, 1.0)
        
        pattern_desc = f"Consolidation: {range_pct:.1f}% range"
        return consolidating, confidence, pattern_desc

# ==================== BACKTEST & PERFORMANCE ANALYZER ====================

class BacktestAnalyzer:
    """Backtest trading signals and calculate performance metrics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.trades = []
        
    def backtest_strategy(self, signal_function) -> Dict:
        """Backtest a trading strategy"""
        returns = []
        entry_points = []
        
        for i in range(20, len(self.df) - 1):
            test_df = self.df.iloc[:i+1]
            signal = signal_function(test_df)
            
            if signal:
                entry_price = test_df['Close'].iloc[-1]
                entry_points.append((i, entry_price))
                
                # Calculate 20-period return
                future_price = self.df['Close'].iloc[i+20]
                ret = (future_price - entry_price) / entry_price * 100
                returns.append(ret)
        
        if not returns:
            return {'win_rate': 0, 'avg_return': 0, 'signals': 0}
        
        returns = np.array(returns)
        win_rate = (returns > 0).sum() / len(returns) * 100
        avg_return = returns.mean()
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'signals': len(returns),
            'max_return': returns.max(),
            'min_return': returns.min(),
            'std_dev': returns.std()
        }
    
    def calculate_expectancy(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate mathematical expectancy of a strategy"""
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
        return expectancy

# ==================== MAIN TRADING ENGINE ====================

class SwingTradeEngine:
    """Main engine combining all analyses"""
    
    def __init__(self, ticker: str, days: int = 1095):  # 3 years
        self.ticker = ticker
        self.days = days
        self.df = self._fetch_data()
        
        if self.df is None or len(self.df) < 100:
            raise ValueError(f"Insufficient data for {ticker}")
        
        # Initialize analyzers
        self.sr_analyzer = SupportResistanceAnalyzer(self.df)
        self.bandar_analyzer = BandarFlowAnalyzer(self.df)
        self.indicator_optimizer = DynamicIndicatorsOptimizer(self.df)
        self.pattern_analyzer = PatternRecognition(self.df)
        self.backtest = BacktestAnalyzer(self.df)
        
    def _fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance with OJK compliance"""
        try:
            start_date = datetime.now() - timedelta(days=self.days)
            df = yf.download(self.ticker, start=start_date, progress=False)
            
            if df is None or len(df) == 0:
                return None
            
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df[df['Volume'] > 0]  # Remove zero volume days
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def generate_trade_recommendation(self) -> TradeRecommendation:
        """Generate comprehensive trade recommendation"""
        
        current_price = self.df['Close'].iloc[-1]
        
        # 1. Support & Resistance
        support, resistance = self.sr_analyzer.get_nearest_support_resistance()
        
        # 2. Bandar Analysis
        bandar_phase, bandar_confidence = self.bandar_analyzer.detect_accumulation_phase()
        vwap = self.bandar_analyzer.calculate_vwap()
        absorption, absorption_ratio = self.bandar_analyzer.detect_candle_absorption()
        
        # 3. Dynamic Indicators
        optimal = self.indicator_optimizer.find_optimal_indicators()
        rsi = optimal['rsi_value']
        k = optimal['stoch_k']
        
        # 4. Patterns
        vcp_detected, vcp_conf, vcp_desc = self.pattern_analyzer.detect_vcp()
        low_cheat, low_cheat_conf, low_cheat_desc = self.pattern_analyzer.detect_low_cheat()
        superclose_ma, superclose_conf, superclose_desc = self.pattern_analyzer.detect_superclose_ma()
        consolidation, consol_conf, consol_desc = self.pattern_analyzer.detect_consolidation()
        
        # 5. Determine Trade Signal
        signal, reason, supporting_indicators = self._determine_signal(
            current_price, support, resistance, vwap, bandar_phase,
            rsi, k, vcp_detected, low_cheat, superclose_ma, consolidation
        )
        
        # 6. Calculate Targets (R:R 3:1)
        risk = current_price - support
        target_1r = current_price + risk
        target_3r = current_price + (risk * 3)
        stop_loss = support * 0.98  # 2% buffer
        
        # 7. Calculate Success Probability
        probability = self._calculate_success_probability(
            signal, bandar_phase, rsi, k, vcp_detected, low_cheat,
            superclose_ma, consolidation
        )
        
        # 8. Technical Strength
        technical_strength = self._calculate_technical_strength(
            vcp_detected, low_cheat, superclose_ma, consolidation,
            bandar_phase, rsi, k
        )
        
        # Identify pattern
        pattern_name = ""
        if vcp_detected:
            pattern_name = "VCP"
        elif low_cheat:
            pattern_name = "Low Cheat"
        elif superclose_ma:
            pattern_name = "Superclose MA"
        elif consolidation:
            pattern_name = "Consolidation"
        else:
            pattern_name = "Multi-indicator Setup"
        
        return TradeRecommendation(
            signal=signal,
            entry_price=current_price,
            entry_reason=reason,
            stop_loss=stop_loss,
            target_1r=target_1r,
            target_3r=target_3r,
            success_probability=probability,
            risk_reward_ratio=3.0,
            supporting_indicators=supporting_indicators,
            bandar_phase=bandar_phase.value,
            vwap=vwap,
            nearest_support=support,
            nearest_resistance=resistance,
            pattern_detected=pattern_name,
            technical_strength=technical_strength
        )
    
    def _determine_signal(self, current_price, support, resistance, vwap, 
                         bandar_phase, rsi, k, vcp, low_cheat, superclose_ma, 
                         consolidation) -> Tuple[TradeSignal, str, List[str]]:
        """Determine trade signal based on all indicators"""
        
        indicators = []
        buy_score = 0
        
        # Bandar accumulation
        if bandar_phase == BandarPhase.ACCUMULATING:
            buy_score += 2
            indicators.append("Bandar Accumulating")
        
        # Price action
        if current_price > vwap:
            buy_score += 1
            indicators.append("Price > VWAP")
        
        # RSI signal
        if 35 < rsi < 65:  # Neutral to bullish
            if rsi < 40:
                buy_score += 2
                indicators.append(f"RSI Oversold ({rsi:.1f})")
            elif rsi > 50:
                buy_score += 1
                indicators.append(f"RSI Bullish ({rsi:.1f})")
        
        # Stochastic signal
        if k < 30:
            buy_score += 2
            indicators.append(f"Stochastic Oversold ({k:.1f})")
        elif k > 70:
            buy_score -= 1
            indicators.append(f"Stochastic Overbought ({k:.1f})")
        
        # Pattern signals
        if vcp:
            buy_score += 2
            indicators.append("VCP Detected")
        if low_cheat:
            buy_score += 2
            indicators.append("Low Cheat Pattern")
        if superclose_ma:
            buy_score += 3
            indicators.append("Superclose MA - Breakout Ready")
        if consolidation:
            buy_score += 1
            indicators.append("Consolidation Pattern")
        
        # Support proximity
        dist_to_support = (current_price - support) / support
        if dist_to_support < 0.03:  # Close to support
            buy_score += 2
            indicators.append(f"Deep Support ({dist_to_support*100:.1f}%)")
        
        # Determine signal
        if buy_score >= 8:
            return (TradeSignal.STRONG_BUY, 
                   f"Multiple confluence points detected (Score: {buy_score}/10)", 
                   indicators)
        elif buy_score >= 6:
            return (TradeSignal.BUY, 
                   f"Strong setup with key indicators aligned (Score: {buy_score}/10)", 
                   indicators)
        elif buy_score >= 4:
            return (TradeSignal.WEAK_BUY, 
                   f"Moderate setup with some indicators aligned (Score: {buy_score}/10)", 
                   indicators)
        elif buy_score >= 2:
            return (TradeSignal.HOLD, 
                   f"Setup forming but not ready (Score: {buy_score}/10)", 
                   indicators)
        else:
            return (TradeSignal.WAIT, 
                   "Insufficient confluences, wait for better setup", 
                   indicators)
    
    def _calculate_success_probability(self, signal, bandar_phase, rsi, k, 
                                       vcp, low_cheat, superclose_ma, 
                                       consolidation) -> float:
        """Calculate probability of trade success"""
        
        base_probability = 50.0  # Base case
        
        # Signal strength
        if signal == TradeSignal.STRONG_BUY:
            base_probability += 15
        elif signal == TradeSignal.BUY:
            base_probability += 10
        elif signal == TradeSignal.WEAK_BUY:
            base_probability += 5
        
        # Bandar phase
        if bandar_phase == BandarPhase.ACCUMULATING:
            base_probability += 12
        elif bandar_phase == BandarPhase.DISTRIBUTING:
            base_probability -= 10
        
        # RSI conditions
        if 35 < rsi < 45:
            base_probability += 8
        elif rsi < 30:
            base_probability += 10
        
        # Stochastic conditions
        if k < 30:
            base_probability += 8
        
        # Pattern bonuses
        if vcp:
            base_probability += 10
        if low_cheat:
            base_probability += 10
        if superclose_ma:
            base_probability += 12
        if consolidation:
            base_probability += 5
        
        return min(max(base_probability, 25.0), 95.0)
    
    def _calculate_technical_strength(self, vcp, low_cheat, superclose_ma, 
                                     consolidation, bandar_phase, rsi, k) -> float:
        """Calculate overall technical strength (0-100%)"""
        
        strength = 50.0
        
        # Pattern strength
        patterns = sum([vcp, low_cheat, superclose_ma, consolidation])
        strength += patterns * 10
        
        # Indicator strength
        indicator_strength = 0
        if 30 < rsi < 70:
            indicator_strength += 10
        if k < 30 or k > 70:
            indicator_strength += 10
        
        strength += indicator_strength
        
        # Bandar strength
        if bandar_phase == BandarPhase.ACCUMULATING:
            strength += 15
        
        return min(strength, 100.0)