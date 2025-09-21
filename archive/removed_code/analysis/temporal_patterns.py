"""
Temporal Pattern Analyzer

Advanced time-based pattern recognition system for cryptocurrency trading.
Analyzes temporal patterns including market sessions, cyclical behaviors,
time-of-day effects, and seasonal trends to enhance trading decisions.

Key Features:
- Market session analysis (Asian, European, American sessions)
- Intraday pattern recognition (hourly volatility patterns)
- Weekly cyclical behavior analysis
- Holiday and weekend effects
- Time-based volume patterns
- Seasonal trend analysis
- Market open/close behavior patterns
- News event timing correlations

Pattern Types Detected:
- Session-based volatility patterns
- Time-of-day price movements
- Weekly recurring patterns
- Monthly seasonal effects
- Holiday market behavior
- News release timing patterns
- Market microstructure patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import asyncio
from collections import defaultdict, deque
import warnings
import statistics
from enum import Enum
import calendar
import pytz

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


class MarketSession(Enum):
    """Market trading sessions."""
    ASIAN = "asian"
    EUROPEAN = "european"
    AMERICAN = "american"
    OVERLAP_ASIAN_EUROPEAN = "asian_european"
    OVERLAP_EUROPEAN_AMERICAN = "european_american"
    OFF_HOURS = "off_hours"


class PatternType(Enum):
    """Types of temporal patterns."""
    INTRADAY = "intraday"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"
    SESSION = "session"
    NEWS_EVENT = "news_event"
    HOLIDAY = "holiday"


class PatternStrength(Enum):
    """Strength of detected patterns."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class TemporalPattern:
    """Represents a detected temporal pattern."""
    pattern_id: str
    pattern_type: PatternType
    strength: PatternStrength
    confidence: float
    description: str
    start_time: datetime
    end_time: datetime
    frequency: str  # 'daily', 'weekly', 'monthly'
    expected_return: float
    volatility: float
    volume_factor: float
    statistical_significance: float
    historical_occurrences: int
    success_rate: float
    metadata: Dict[str, Any]


@dataclass
class SessionAnalysis:
    """Analysis results for a specific market session."""
    session: MarketSession
    avg_volatility: float
    avg_volume: float
    avg_return: float
    price_direction_bias: float  # -1 (bearish) to 1 (bullish)
    breakout_probability: float
    reversal_probability: float
    optimal_entry_time: time
    optimal_exit_time: time
    confidence_score: float
    sample_size: int


@dataclass
class CyclicalBehavior:
    """Cyclical behavior analysis."""
    cycle_type: str  # 'hourly', 'daily', 'weekly', 'monthly'
    period_length: int
    amplitude: float
    phase_shift: float
    trend: float
    seasonality_strength: float
    confidence: float
    next_predicted_high: datetime
    next_predicted_low: datetime
    forecasted_values: List[Tuple[datetime, float]]


class TemporalPatternAnalyzer:
    """
    Advanced temporal pattern recognition and analysis system.
    
    Analyzes time-based patterns in cryptocurrency markets to identify
    recurring behaviors, optimal trading times, and cyclical opportunities.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("TemporalPatternAnalyzer")
        
        # Configuration
        self.config = {
            'min_pattern_strength': config_manager.get('patterns.min_strength', 0.6),
            'min_statistical_significance': config_manager.get('patterns.min_significance', 0.05),
            'pattern_lookback_days': config_manager.get('patterns.lookback_days', 90),
            'session_timezone': config_manager.get('patterns.timezone', 'UTC'),
            'min_sample_size': config_manager.get('patterns.min_sample_size', 30),
            'volatility_threshold': config_manager.get('patterns.volatility_threshold', 0.02),
            'volume_threshold': config_manager.get('patterns.volume_threshold', 1.5)
        }
        
        # Market session definitions (UTC times)
        self.session_definitions = {
            MarketSession.ASIAN: (time(0, 0), time(9, 0)),       # 00:00-09:00 UTC
            MarketSession.EUROPEAN: (time(7, 0), time(16, 0)),   # 07:00-16:00 UTC
            MarketSession.AMERICAN: (time(13, 0), time(22, 0)),  # 13:00-22:00 UTC
            MarketSession.OVERLAP_ASIAN_EUROPEAN: (time(7, 0), time(9, 0)),    # 07:00-09:00 UTC
            MarketSession.OVERLAP_EUROPEAN_AMERICAN: (time(13, 0), time(16, 0)) # 13:00-16:00 UTC
        }
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.volume_data: Dict[str, pd.DataFrame] = {}
        self.news_data: Dict[str, pd.DataFrame] = {}
        
        # Pattern storage
        self.detected_patterns: Dict[str, List[TemporalPattern]] = defaultdict(list)
        self.session_analyses: Dict[str, Dict[MarketSession, SessionAnalysis]] = defaultdict(dict)
        self.cyclical_behaviors: Dict[str, List[CyclicalBehavior]] = defaultdict(list)
        
        # Pattern cache
        self.pattern_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Statistical tracking
        self.pattern_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Timezone setup
        self.timezone = pytz.timezone(self.config['session_timezone'])
    
    def update_data(self, symbol: str, price_data: pd.DataFrame, 
                   volume_data: Optional[pd.DataFrame] = None,
                   news_data: Optional[pd.DataFrame] = None):
        """Update market data for pattern analysis."""
        try:
            # Store price data
            if not price_data.empty:
                # Ensure timestamp is timezone-aware
                if 'timestamp' in price_data.columns:
                    price_data = price_data.copy()
                    if price_data['timestamp'].dt.tz is None:
                        price_data['timestamp'] = price_data['timestamp'].dt.tz_localize('UTC')
                    
                    price_data = price_data.sort_values('timestamp')
                    self.price_data[symbol] = price_data.tail(10000)  # Keep last 10k candles
                    self.logger.debug(f"Updated price data for {symbol}: {len(price_data)} candles")
            
            # Store volume data
            if volume_data is not None and not volume_data.empty:
                if 'timestamp' in volume_data.columns:
                    volume_data = volume_data.copy()
                    if volume_data['timestamp'].dt.tz is None:
                        volume_data['timestamp'] = volume_data['timestamp'].dt.tz_localize('UTC')
                    
                    self.volume_data[symbol] = volume_data.tail(10000)
            
            # Store news data
            if news_data is not None and not news_data.empty:
                if 'timestamp' in news_data.columns:
                    news_data = news_data.copy()
                    if news_data['timestamp'].dt.tz is None:
                        news_data['timestamp'] = news_data['timestamp'].dt.tz_localize('UTC')
                    
                    self.news_data[symbol] = news_data.tail(5000)
            
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {e}")
    
    def analyze_all_patterns(self, symbol: str) -> Dict[str, Any]:
        """Analyze all temporal patterns for a symbol."""
        if symbol not in self.price_data:
            self.logger.warning(f"No price data available for {symbol}")
            return {}
        
        results = {}
        
        try:
            # Analyze market sessions
            session_analysis = self.analyze_market_sessions(symbol)
            results['sessions'] = session_analysis
            
            # Analyze intraday patterns
            intraday_patterns = self.analyze_intraday_patterns(symbol)
            results['intraday'] = intraday_patterns
            
            # Analyze weekly patterns
            weekly_patterns = self.analyze_weekly_patterns(symbol)
            results['weekly'] = weekly_patterns
            
            # Analyze monthly/seasonal patterns
            seasonal_patterns = self.analyze_seasonal_patterns(symbol)
            results['seasonal'] = seasonal_patterns
            
            # Analyze cyclical behaviors
            cyclical_analysis = self.analyze_cyclical_behaviors(symbol)
            results['cyclical'] = cyclical_analysis
            
            # Analyze news event patterns
            news_patterns = self.analyze_news_event_patterns(symbol)
            results['news_events'] = news_patterns
            
            # Generate pattern summary
            pattern_summary = self.generate_pattern_summary(symbol, results)
            results['summary'] = pattern_summary
            
            self.logger.info(f"Completed temporal pattern analysis for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis for {symbol}: {e}")
        
        return results
    
    def analyze_market_sessions(self, symbol: str) -> Dict[MarketSession, SessionAnalysis]:
        """Analyze behavior during different market sessions."""
        price_data = self.price_data[symbol]
        if len(price_data) < self.config['min_sample_size']:
            return {}
        
        session_results = {}
        
        for session, (start_time, end_time) in self.session_definitions.items():
            try:
                # Filter data for this session
                session_data = self._filter_by_session(price_data, session)
                
                if len(session_data) < self.config['min_sample_size']:
                    continue
                
                # Calculate session statistics
                returns = session_data['close'].pct_change().dropna()
                volumes = session_data['volume']
                
                # Calculate volatility (using high-low range)
                if 'high' in session_data.columns and 'low' in session_data.columns:
                    volatility = ((session_data['high'] - session_data['low']) / session_data['close']).mean()
                else:
                    volatility = returns.std()
                
                # Calculate directional bias
                positive_returns = (returns > 0).sum()
                total_returns = len(returns)
                direction_bias = (positive_returns / total_returns - 0.5) * 2 if total_returns > 0 else 0
                
                # Calculate breakout/reversal probabilities
                breakout_prob = self._calculate_breakout_probability(session_data)
                reversal_prob = 1.0 - breakout_prob
                
                # Find optimal entry/exit times
                optimal_entry, optimal_exit = self._find_optimal_session_times(session_data, session)
                
                # Calculate confidence based on sample size and consistency
                confidence = min(1.0, len(session_data) / 100) * (1 - abs(direction_bias)) * 0.5 + 0.5
                
                session_analysis = SessionAnalysis(
                    session=session,
                    avg_volatility=volatility,
                    avg_volume=volumes.mean(),
                    avg_return=returns.mean(),
                    price_direction_bias=direction_bias,
                    breakout_probability=breakout_prob,
                    reversal_probability=reversal_prob,
                    optimal_entry_time=optimal_entry,
                    optimal_exit_time=optimal_exit,
                    confidence_score=confidence,
                    sample_size=len(session_data)
                )
                
                session_results[session] = session_analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing {session.value} session for {symbol}: {e}")
        
        # Store results
        self.session_analyses[symbol] = session_results
        
        return session_results
    
    def analyze_intraday_patterns(self, symbol: str) -> List[TemporalPattern]:
        """Analyze intraday (hourly) patterns."""
        price_data = self.price_data[symbol]
        if len(price_data) < self.config['min_sample_size'] * 24:  # Need data for multiple days
            return []
        
        patterns = []
        
        try:
            # Group by hour of day
            price_data_copy = price_data.copy()
            price_data_copy['hour'] = price_data_copy['timestamp'].dt.hour
            price_data_copy['returns'] = price_data_copy['close'].pct_change()
            
            hourly_stats = price_data_copy.groupby('hour').agg({
                'returns': ['mean', 'std', 'count'],
                'volume': 'mean',
                'close': lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) > 1 else 0
            }).round(6)
            
            # Detect significant hourly patterns
            for hour in range(24):
                if hour not in hourly_stats.index:
                    continue
                
                hour_data = hourly_stats.loc[hour]
                
                # Extract statistics
                avg_return = hour_data[('returns', 'mean')]
                return_std = hour_data[('returns', 'std')]
                sample_size = hour_data[('returns', 'count')]
                avg_volume = hour_data[('volume', 'mean')]
                
                if sample_size < 10:  # Minimum sample size
                    continue
                
                # Calculate statistical significance (t-test)
                if return_std > 0:
                    t_stat = abs(avg_return) / (return_std / np.sqrt(sample_size))
                    p_value = 2 * (1 - stats.t.cdf(t_stat, sample_size - 1))
                else:
                    p_value = 1.0
                
                # Determine pattern strength
                if p_value < 0.001:
                    strength = PatternStrength.VERY_STRONG
                elif p_value < 0.01:
                    strength = PatternStrength.STRONG
                elif p_value < 0.05:
                    strength = PatternStrength.MODERATE
                elif p_value < 0.1:
                    strength = PatternStrength.WEAK
                else:
                    strength = PatternStrength.VERY_WEAK
                
                # Only include statistically significant patterns
                if p_value < self.config['min_statistical_significance']:
                    pattern_id = f"intraday_hour_{hour:02d}"
                    
                    # Calculate success rate
                    positive_returns = (price_data_copy[price_data_copy['hour'] == hour]['returns'] > 0).sum()
                    success_rate = positive_returns / sample_size
                    
                    pattern = TemporalPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.INTRADAY,
                        strength=strength,
                        confidence=1 - p_value,
                        description=f"Hour {hour:02d}:00 - Avg return: {avg_return:.4f}, Volume factor: {avg_volume:.0f}",
                        start_time=datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0),
                        end_time=datetime.now().replace(hour=(hour + 1) % 24, minute=0, second=0, microsecond=0),
                        frequency='daily',
                        expected_return=avg_return,
                        volatility=return_std,
                        volume_factor=avg_volume,
                        statistical_significance=p_value,
                        historical_occurrences=int(sample_size),
                        success_rate=success_rate,
                        metadata={
                            'hour': hour,
                            't_statistic': t_stat,
                            'sample_size': int(sample_size)
                        }
                    )
                    
                    patterns.append(pattern)
            
            self.detected_patterns[symbol].extend(patterns)
            
        except Exception as e:
            self.logger.error(f"Error analyzing intraday patterns for {symbol}: {e}")
        
        return patterns
    
    def analyze_weekly_patterns(self, symbol: str) -> List[TemporalPattern]:
        """Analyze weekly (day-of-week) patterns."""
        price_data = self.price_data[symbol]
        if len(price_data) < self.config['min_sample_size'] * 7:  # Need multiple weeks
            return []
        
        patterns = []
        
        try:
            # Group by day of week
            price_data_copy = price_data.copy()
            price_data_copy['day_of_week'] = price_data_copy['timestamp'].dt.dayofweek  # 0=Monday
            price_data_copy['returns'] = price_data_copy['close'].pct_change()
            
            daily_stats = price_data_copy.groupby('day_of_week').agg({
                'returns': ['mean', 'std', 'count'],
                'volume': 'mean'
            }).round(6)
            
            # Detect significant daily patterns
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for day in range(7):
                if day not in daily_stats.index:
                    continue
                
                day_data = daily_stats.loc[day]
                
                # Extract statistics
                avg_return = day_data[('returns', 'mean')]
                return_std = day_data[('returns', 'std')]
                sample_size = day_data[('returns', 'count')]
                avg_volume = day_data[('volume', 'mean')]
                
                if sample_size < 5:  # Minimum sample size
                    continue
                
                # Calculate statistical significance
                if return_std > 0:
                    t_stat = abs(avg_return) / (return_std / np.sqrt(sample_size))
                    p_value = 2 * (1 - stats.t.cdf(t_stat, sample_size - 1))
                else:
                    p_value = 1.0
                
                # Determine pattern strength
                if p_value < 0.01:
                    strength = PatternStrength.VERY_STRONG
                elif p_value < 0.05:
                    strength = PatternStrength.STRONG
                elif p_value < 0.1:
                    strength = PatternStrength.MODERATE
                elif p_value < 0.2:
                    strength = PatternStrength.WEAK
                else:
                    strength = PatternStrength.VERY_WEAK
                
                # Only include patterns with reasonable significance
                if p_value < 0.2:  # More lenient for weekly patterns
                    pattern_id = f"weekly_day_{day}"
                    
                    # Calculate success rate
                    positive_returns = (price_data_copy[price_data_copy['day_of_week'] == day]['returns'] > 0).sum()
                    success_rate = positive_returns / sample_size
                    
                    pattern = TemporalPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.WEEKLY,
                        strength=strength,
                        confidence=1 - p_value,
                        description=f"{day_names[day]} - Avg return: {avg_return:.4f}, Volume: {avg_volume:.0f}",
                        start_time=datetime.now(),  # Will be updated with next occurrence
                        end_time=datetime.now(),
                        frequency='weekly',
                        expected_return=avg_return,
                        volatility=return_std,
                        volume_factor=avg_volume,
                        statistical_significance=p_value,
                        historical_occurrences=int(sample_size),
                        success_rate=success_rate,
                        metadata={
                            'day_of_week': day,
                            'day_name': day_names[day],
                            't_statistic': t_stat,
                            'sample_size': int(sample_size)
                        }
                    )
                    
                    patterns.append(pattern)
            
            self.detected_patterns[symbol].extend(patterns)
            
        except Exception as e:
            self.logger.error(f"Error analyzing weekly patterns for {symbol}: {e}")
        
        return patterns
    
    def analyze_seasonal_patterns(self, symbol: str) -> List[TemporalPattern]:
        """Analyze monthly and seasonal patterns."""
        price_data = self.price_data[symbol]
        if len(price_data) < self.config['min_sample_size'] * 30:  # Need multiple months
            return []
        
        patterns = []
        
        try:
            # Group by month
            price_data_copy = price_data.copy()
            price_data_copy['month'] = price_data_copy['timestamp'].dt.month
            price_data_copy['returns'] = price_data_copy['close'].pct_change()
            
            monthly_stats = price_data_copy.groupby('month').agg({
                'returns': ['mean', 'std', 'count'],
                'volume': 'mean'
            }).round(6)
            
            # Detect significant monthly patterns
            month_names = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            
            for month in range(1, 13):
                if month not in monthly_stats.index:
                    continue
                
                month_data = monthly_stats.loc[month]
                
                # Extract statistics
                avg_return = month_data[('returns', 'mean')]
                return_std = month_data[('returns', 'std')]
                sample_size = month_data[('returns', 'count')]
                avg_volume = month_data[('volume', 'mean')]
                
                if sample_size < 10:  # Minimum sample size
                    continue
                
                # Calculate statistical significance
                if return_std > 0:
                    t_stat = abs(avg_return) / (return_std / np.sqrt(sample_size))
                    p_value = 2 * (1 - stats.t.cdf(t_stat, sample_size - 1))
                else:
                    p_value = 1.0
                
                # Determine pattern strength
                if p_value < 0.05:
                    strength = PatternStrength.STRONG
                elif p_value < 0.1:
                    strength = PatternStrength.MODERATE
                elif p_value < 0.2:
                    strength = PatternStrength.WEAK
                else:
                    strength = PatternStrength.VERY_WEAK
                
                # Only include patterns with some significance
                if p_value < 0.3:  # Very lenient for seasonal patterns
                    pattern_id = f"seasonal_month_{month:02d}"
                    
                    # Calculate success rate
                    positive_returns = (price_data_copy[price_data_copy['month'] == month]['returns'] > 0).sum()
                    success_rate = positive_returns / sample_size
                    
                    pattern = TemporalPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.SEASONAL,
                        strength=strength,
                        confidence=max(0.1, 1 - p_value),
                        description=f"{month_names[month - 1]} - Avg return: {avg_return:.4f}",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        frequency='yearly',
                        expected_return=avg_return,
                        volatility=return_std,
                        volume_factor=avg_volume,
                        statistical_significance=p_value,
                        historical_occurrences=int(sample_size),
                        success_rate=success_rate,
                        metadata={
                            'month': month,
                            'month_name': month_names[month - 1],
                            't_statistic': t_stat,
                            'sample_size': int(sample_size)
                        }
                    )
                    
                    patterns.append(pattern)
            
            self.detected_patterns[symbol].extend(patterns)
            
        except Exception as e:
            self.logger.error(f"Error analyzing seasonal patterns for {symbol}: {e}")
        
        return patterns
    
    def analyze_cyclical_behaviors(self, symbol: str) -> List[CyclicalBehavior]:
        """Analyze cyclical behaviors using spectral analysis."""
        price_data = self.price_data[symbol]
        if len(price_data) < 100:  # Need sufficient data for spectral analysis
            return []
        
        cyclical_behaviors = []
        
        try:
            # Prepare price series
            prices = price_data['close'].values
            returns = np.diff(np.log(prices))
            
            # Remove trend (detrend)
            from scipy import signal
            detrended_returns = signal.detrend(returns)
            
            # Perform FFT to find dominant frequencies
            fft = np.fft.fft(detrended_returns)
            frequencies = np.fft.fftfreq(len(detrended_returns))
            
            # Find dominant frequencies (excluding DC component)
            power_spectrum = np.abs(fft[1:len(fft)//2])
            frequencies = frequencies[1:len(frequencies)//2]
            
            # Find peaks in the power spectrum
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(power_spectrum, height=np.std(power_spectrum))
            
            # Analyze top 3 dominant cycles
            if len(peaks) > 0:
                # Sort peaks by power
                peak_powers = power_spectrum[peaks]
                sorted_indices = np.argsort(peak_powers)[::-1]
                top_peaks = peaks[sorted_indices[:3]]
                
                for i, peak_idx in enumerate(top_peaks):
                    frequency = frequencies[peak_idx]
                    if frequency <= 0:  # Skip negative/zero frequencies
                        continue
                    
                    period = 1 / frequency  # Period in data points
                    amplitude = power_spectrum[peak_idx]
                    
                    # Convert period to time units (assuming hourly data)
                    if period < 48:  # Less than 2 days
                        cycle_type = 'hourly'
                        period_hours = int(period)
                    elif period < 336:  # Less than 2 weeks
                        cycle_type = 'daily'
                        period_hours = int(period)
                    elif period < 8760:  # Less than 1 year
                        cycle_type = 'weekly'
                        period_hours = int(period)
                    else:
                        cycle_type = 'monthly'
                        period_hours = int(period)
                    
                    # Calculate phase and trend
                    phase = np.angle(fft[peak_idx])
                    
                    # Estimate trend from linear regression
                    x = np.arange(len(returns))
                    trend_slope = np.polyfit(x, returns, 1)[0]
                    
                    # Calculate seasonality strength
                    seasonality_strength = amplitude / np.std(detrended_returns)
                    
                    # Calculate confidence based on peak prominence
                    confidence = min(1.0, amplitude / np.mean(power_spectrum))
                    
                    # Predict next high and low
                    current_phase = (len(returns) * frequency + phase) % (2 * np.pi)
                    cycles_to_high = (2 * np.pi - current_phase) / (2 * np.pi) if current_phase > np.pi else (np.pi - current_phase) / (2 * np.pi)
                    cycles_to_low = (2 * np.pi - current_phase + np.pi) / (2 * np.pi) % 1
                    
                    time_to_high = cycles_to_high * period
                    time_to_low = cycles_to_low * period
                    
                    next_high = datetime.now() + timedelta(hours=time_to_high)
                    next_low = datetime.now() + timedelta(hours=time_to_low)
                    
                    # Generate forecasted values
                    forecast_points = min(int(period * 2), 168)  # Max 1 week forecast
                    forecasted_values = []
                    
                    for j in range(forecast_points):
                        future_phase = (len(returns) + j) * frequency + phase
                        predicted_value = amplitude * np.cos(future_phase) + trend_slope * j
                        forecast_time = datetime.now() + timedelta(hours=j)
                        forecasted_values.append((forecast_time, predicted_value))
                    
                    behavior = CyclicalBehavior(
                        cycle_type=cycle_type,
                        period_length=int(period),
                        amplitude=amplitude,
                        phase_shift=phase,
                        trend=trend_slope,
                        seasonality_strength=seasonality_strength,
                        confidence=confidence,
                        next_predicted_high=next_high,
                        next_predicted_low=next_low,
                        forecasted_values=forecasted_values
                    )
                    
                    cyclical_behaviors.append(behavior)
            
            self.cyclical_behaviors[symbol] = cyclical_behaviors
            
        except Exception as e:
            self.logger.error(f"Error analyzing cyclical behaviors for {symbol}: {e}")
        
        return cyclical_behaviors
    
    def analyze_news_event_patterns(self, symbol: str) -> List[TemporalPattern]:
        """Analyze patterns around news events."""
        if symbol not in self.news_data or self.news_data[symbol].empty:
            return []
        
        patterns = []
        
        try:
            news_data = self.news_data[symbol]
            price_data = self.price_data[symbol]
            
            # Analyze price movements around news events
            for _, news_event in news_data.iterrows():
                event_time = news_event['timestamp']
                
                # Find price data around the event (±2 hours)
                start_time = event_time - timedelta(hours=2)
                end_time = event_time + timedelta(hours=2)
                
                event_prices = price_data[
                    (price_data['timestamp'] >= start_time) & 
                    (price_data['timestamp'] <= end_time)
                ]
                
                if len(event_prices) < 5:  # Need sufficient data points
                    continue
                
                # Calculate price movement patterns
                pre_event_prices = event_prices[event_prices['timestamp'] < event_time]
                post_event_prices = event_prices[event_prices['timestamp'] >= event_time]
                
                if len(pre_event_prices) > 0 and len(post_event_prices) > 0:
                    pre_price = pre_event_prices['close'].iloc[-1]
                    post_price = post_event_prices['close'].iloc[0]
                    
                    # Calculate immediate impact
                    immediate_return = (post_price - pre_price) / pre_price
                    
                    # Calculate volatility spike
                    pre_volatility = pre_event_prices['close'].pct_change().std()
                    post_volatility = post_event_prices['close'].pct_change().std()
                    volatility_spike = post_volatility / pre_volatility if pre_volatility > 0 else 1.0
                    
                    # Determine if this creates a significant pattern
                    if abs(immediate_return) > 0.01 or volatility_spike > 1.5:  # 1% move or 50% volatility increase
                        pattern_id = f"news_event_{event_time.strftime('%Y%m%d_%H%M')}"
                        
                        pattern = TemporalPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.NEWS_EVENT,
                            strength=PatternStrength.MODERATE,
                            confidence=0.7,
                            description=f"News event impact: {immediate_return:.2%} return, {volatility_spike:.1f}x volatility",
                            start_time=start_time,
                            end_time=end_time,
                            frequency='event_driven',
                            expected_return=immediate_return,
                            volatility=post_volatility,
                            volume_factor=event_prices['volume'].mean() if 'volume' in event_prices.columns else 1.0,
                            statistical_significance=0.05,  # Assume significant for news events
                            historical_occurrences=1,
                            success_rate=1.0 if abs(immediate_return) > 0.005 else 0.0,
                            metadata={
                                'event_time': event_time,
                                'immediate_return': immediate_return,
                                'volatility_spike': volatility_spike,
                                'sentiment_score': news_event.get('sentiment_score', 0.0)
                            }
                        )
                        
                        patterns.append(pattern)
            
            self.detected_patterns[symbol].extend(patterns)
            
        except Exception as e:
            self.logger.error(f"Error analyzing news event patterns for {symbol}: {e}")
        
        return patterns
    
    def generate_pattern_summary(self, symbol: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive pattern summary."""
        summary = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'total_patterns_detected': 0,
            'pattern_strength_distribution': defaultdict(int),
            'most_significant_patterns': [],
            'best_trading_sessions': [],
            'optimal_trading_times': {},
            'cyclical_predictions': [],
            'pattern_confidence_average': 0.0
        }
        
        try:
            all_patterns = []
            
            # Collect all patterns
            for pattern_type, patterns in analysis_results.items():
                if pattern_type == 'summary':
                    continue
                
                if isinstance(patterns, list) and patterns:
                    all_patterns.extend(patterns)
                elif isinstance(patterns, dict):
                    for session_patterns in patterns.values():
                        if isinstance(session_patterns, list):
                            all_patterns.extend(session_patterns)
            
            if not all_patterns:
                return summary
            
            # Pattern statistics
            summary['total_patterns_detected'] = len(all_patterns)
            
            # Strength distribution
            for pattern in all_patterns:
                if hasattr(pattern, 'strength'):
                    summary['pattern_strength_distribution'][pattern.strength.name] += 1
            
            # Most significant patterns (top 5 by confidence)
            if hasattr(all_patterns[0], 'confidence'):
                significant_patterns = sorted(
                    all_patterns, 
                    key=lambda x: x.confidence if hasattr(x, 'confidence') else 0, 
                    reverse=True
                )[:5]
                
                summary['most_significant_patterns'] = [
                    {
                        'pattern_id': p.pattern_id if hasattr(p, 'pattern_id') else 'unknown',
                        'description': p.description if hasattr(p, 'description') else 'No description',
                        'confidence': p.confidence if hasattr(p, 'confidence') else 0,
                        'expected_return': p.expected_return if hasattr(p, 'expected_return') else 0
                    }
                    for p in significant_patterns
                ]
            
            # Best trading sessions
            if 'sessions' in analysis_results:
                sessions = analysis_results['sessions']
                best_sessions = sorted(
                    sessions.items(),
                    key=lambda x: x[1].avg_return * x[1].confidence_score,
                    reverse=True
                )[:3]
                
                summary['best_trading_sessions'] = [
                    {
                        'session': session.name,
                        'avg_return': analysis.avg_return,
                        'confidence': analysis.confidence_score,
                        'optimal_entry': analysis.optimal_entry_time.strftime('%H:%M'),
                        'optimal_exit': analysis.optimal_exit_time.strftime('%H:%M')
                    }
                    for session, analysis in best_sessions
                ]
            
            # Optimal trading times (from intraday patterns)
            if 'intraday' in analysis_results:
                intraday_patterns = analysis_results['intraday']
                if intraday_patterns:
                    best_hours = sorted(
                        intraday_patterns,
                        key=lambda x: abs(x.expected_return) * x.confidence,
                        reverse=True
                    )[:5]
                    
                    summary['optimal_trading_times'] = {
                        'best_hours': [
                            {
                                'hour': p.metadata.get('hour', 0),
                                'expected_return': p.expected_return,
                                'confidence': p.confidence
                            }
                            for p in best_hours
                        ]
                    }
            
            # Cyclical predictions
            if 'cyclical' in analysis_results:
                cyclical_behaviors = analysis_results['cyclical']
                if cyclical_behaviors:
                    for behavior in cyclical_behaviors[:3]:  # Top 3 cycles
                        summary['cyclical_predictions'].append({
                            'cycle_type': behavior.cycle_type,
                            'period_length': behavior.period_length,
                            'next_predicted_high': behavior.next_predicted_high.isoformat(),
                            'next_predicted_low': behavior.next_predicted_low.isoformat(),
                            'confidence': behavior.confidence
                        })
            
            # Average confidence
            confidences = [p.confidence for p in all_patterns if hasattr(p, 'confidence')]
            if confidences:
                summary['pattern_confidence_average'] = statistics.mean(confidences)
            
        except Exception as e:
            self.logger.error(f"Error generating pattern summary for {symbol}: {e}")
        
        return summary
    
    # Helper methods
    def _filter_by_session(self, data: pd.DataFrame, session: MarketSession) -> pd.DataFrame:
        """Filter data by market session."""
        if session not in self.session_definitions:
            return pd.DataFrame()
        
        start_time, end_time = self.session_definitions[session]
        
        # Convert to time-only for filtering
        data_copy = data.copy()
        data_copy['time_only'] = data_copy['timestamp'].dt.time
        
        if start_time <= end_time:
            # Normal session (doesn't cross midnight)
            mask = (data_copy['time_only'] >= start_time) & (data_copy['time_only'] <= end_time)
        else:
            # Session crosses midnight
            mask = (data_copy['time_only'] >= start_time) | (data_copy['time_only'] <= end_time)
        
        return data_copy[mask]
    
    def _calculate_breakout_probability(self, session_data: pd.DataFrame) -> float:
        """Calculate probability of breakout during session."""
        if len(session_data) < 10:
            return 0.5
        
        # Calculate range expansion vs contraction
        if 'high' in session_data.columns and 'low' in session_data.columns:
            session_ranges = session_data['high'] - session_data['low']
            avg_range = session_ranges.mean()
            expanding_sessions = (session_ranges > avg_range * 1.2).sum()
            breakout_prob = expanding_sessions / len(session_data)
        else:
            # Fallback: use return volatility
            returns = session_data['close'].pct_change().dropna()
            high_vol_sessions = (returns.abs() > returns.std() * 1.5).sum()
            breakout_prob = high_vol_sessions / len(returns)
        
        return np.clip(breakout_prob, 0.0, 1.0)
    
    def _find_optimal_session_times(self, session_data: pd.DataFrame, session: MarketSession) -> Tuple[time, time]:
        """Find optimal entry and exit times within a session."""
        if len(session_data) < 10:
            # Return session boundaries as default
            start_time, end_time = self.session_definitions[session]
            return start_time, end_time
        
        try:
            # Group by hour within session
            session_data_copy = session_data.copy()
            session_data_copy['hour'] = session_data_copy['timestamp'].dt.hour
            session_data_copy['returns'] = session_data_copy['close'].pct_change()
            
            hourly_returns = session_data_copy.groupby('hour')['returns'].mean()
            
            if len(hourly_returns) < 2:
                start_time, end_time = self.session_definitions[session]
                return start_time, end_time
            
            # Find hours with best and worst average returns
            best_hour = hourly_returns.idxmax()
            worst_hour = hourly_returns.idxmin()
            
            # Optimal entry: hour before best performance or start of session
            optimal_entry = time(max(0, best_hour - 1), 0)
            
            # Optimal exit: best performing hour or end of session
            optimal_exit = time(best_hour, 59)
            
            return optimal_entry, optimal_exit
            
        except Exception as e:
            self.logger.debug(f"Error finding optimal session times: {e}")
            start_time, end_time = self.session_definitions[session]
            return start_time, end_time
    
    # Public interface methods
    def get_patterns_by_type(self, symbol: str, pattern_type: PatternType) -> List[TemporalPattern]:
        """Get patterns of a specific type."""
        if symbol not in self.detected_patterns:
            return []
        
        return [p for p in self.detected_patterns[symbol] if p.pattern_type == pattern_type]
    
    def get_current_session(self) -> MarketSession:
        """Get the current market session."""
        current_time = datetime.now(self.timezone).time()
        
        for session, (start_time, end_time) in self.session_definitions.items():
            if session in [MarketSession.OVERLAP_ASIAN_EUROPEAN, MarketSession.OVERLAP_EUROPEAN_AMERICAN]:
                continue  # Skip overlap sessions for current session detection
            
            if start_time <= end_time:
                if start_time <= current_time <= end_time:
                    return session
            else:
                if current_time >= start_time or current_time <= end_time:
                    return session
        
        return MarketSession.OFF_HOURS
    
    def get_next_significant_time(self, symbol: str) -> Optional[Tuple[datetime, str]]:
        """Get the next significant time event based on patterns."""
        if symbol not in self.detected_patterns:
            return None
        
        now = datetime.now()
        upcoming_events = []
        
        # Check intraday patterns
        intraday_patterns = self.get_patterns_by_type(symbol, PatternType.INTRADAY)
        for pattern in intraday_patterns:
            if pattern.strength.value >= 3:  # Moderate or stronger
                next_occurrence = now.replace(
                    hour=pattern.metadata['hour'],
                    minute=0,
                    second=0,
                    microsecond=0
                )
                if next_occurrence <= now:
                    next_occurrence += timedelta(days=1)
                
                upcoming_events.append((next_occurrence, f"Intraday pattern: {pattern.description}"))
        
        # Check cyclical behaviors
        if symbol in self.cyclical_behaviors:
            for behavior in self.cyclical_behaviors[symbol]:
                if behavior.confidence > 0.6:
                    upcoming_events.append((
                        behavior.next_predicted_high,
                        f"Cyclical high: {behavior.cycle_type} cycle"
                    ))
                    upcoming_events.append((
                        behavior.next_predicted_low,
                        f"Cyclical low: {behavior.cycle_type} cycle"
                    ))
        
        # Return the nearest event
        if upcoming_events:
            upcoming_events.sort(key=lambda x: x[0])
            return upcoming_events[0]
        
        return None
    
    def get_trading_recommendation(self, symbol: str) -> Dict[str, Any]:
        """Get trading recommendation based on temporal patterns."""
        current_session = self.get_current_session()
        current_time = datetime.now().time()
        
        recommendation = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_session': current_session.value,
            'recommendation': 'hold',
            'confidence': 0.5,
            'reasoning': [],
            'optimal_entry_time': None,
            'expected_duration': None,
            'risk_level': 'medium'
        }
        
        try:
            # Check session analysis
            if symbol in self.session_analyses and current_session in self.session_analyses[symbol]:
                session_analysis = self.session_analyses[symbol][current_session]
                
                if session_analysis.confidence_score > 0.7:
                    if session_analysis.avg_return > 0.001:  # 0.1% threshold
                        recommendation['recommendation'] = 'buy'
                        recommendation['confidence'] = session_analysis.confidence_score
                        recommendation['reasoning'].append(f"Positive session bias: {session_analysis.avg_return:.3%}")
                    elif session_analysis.avg_return < -0.001:
                        recommendation['recommendation'] = 'sell'
                        recommendation['confidence'] = session_analysis.confidence_score
                        recommendation['reasoning'].append(f"Negative session bias: {session_analysis.avg_return:.3%}")
                
                recommendation['optimal_entry_time'] = session_analysis.optimal_entry_time.strftime('%H:%M')
            
            # Check intraday patterns
            current_hour = current_time.hour
            intraday_patterns = self.get_patterns_by_type(symbol, PatternType.INTRADAY)
            
            for pattern in intraday_patterns:
                if pattern.metadata['hour'] == current_hour and pattern.strength.value >= 3:
                    if pattern.expected_return > 0.002:  # 0.2% threshold
                        if recommendation['recommendation'] != 'sell':
                            recommendation['recommendation'] = 'buy'
                        recommendation['confidence'] = max(recommendation['confidence'], pattern.confidence)
                        recommendation['reasoning'].append(f"Hourly pattern: {pattern.expected_return:.3%}")
                    elif pattern.expected_return < -0.002:
                        if recommendation['recommendation'] != 'buy':
                            recommendation['recommendation'] = 'sell'
                        recommendation['confidence'] = max(recommendation['confidence'], pattern.confidence)
                        recommendation['reasoning'].append(f"Hourly pattern: {pattern.expected_return:.3%}")
            
            # Check cyclical behaviors
            if symbol in self.cyclical_behaviors:
                for behavior in self.cyclical_behaviors[symbol]:
                    if behavior.confidence > 0.6:
                        time_to_high = (behavior.next_predicted_high - datetime.now()).total_seconds() / 3600
                        time_to_low = (behavior.next_predicted_low - datetime.now()).total_seconds() / 3600
                        
                        if 0 < time_to_high < 6:  # High predicted within 6 hours
                            if recommendation['recommendation'] != 'sell':
                                recommendation['recommendation'] = 'buy'
                            recommendation['reasoning'].append(f"Cyclical high in {time_to_high:.1f}h")
                        elif 0 < time_to_low < 6:  # Low predicted within 6 hours
                            if recommendation['recommendation'] != 'buy':
                                recommendation['recommendation'] = 'sell'
                            recommendation['reasoning'].append(f"Cyclical low in {time_to_low:.1f}h")
            
            # Determine risk level
            if recommendation['confidence'] > 0.8:
                recommendation['risk_level'] = 'low'
            elif recommendation['confidence'] > 0.6:
                recommendation['risk_level'] = 'medium'
            else:
                recommendation['risk_level'] = 'high'
            
            # Determine expected duration
            if any('hourly' in reason.lower() for reason in recommendation['reasoning']):
                recommendation['expected_duration'] = 'short'  # 1-4 hours
            elif any('session' in reason.lower() for reason in recommendation['reasoning']):
                recommendation['expected_duration'] = 'medium'  # 4-12 hours
            else:
                recommendation['expected_duration'] = 'long'  # 12+ hours
        
        except Exception as e:
            self.logger.error(f"Error generating trading recommendation for {symbol}: {e}")
        
        return recommendation


# Import required scipy stats for statistical operations
try:
    from scipy import stats
    from scipy.signal import find_peaks
except ImportError:
    # Fallback implementation
    class MockStats:
        class t:
            @staticmethod
            def cdf(x, df):
                # Simplified approximation
                return 0.5 + 0.5 * np.tanh(x / 2)
    
    stats = MockStats()
    
    def find_peaks(data, height=None):
        # Simple peak finding
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                if height is None or data[i] >= height:
                    peaks.append(i)
        return np.array(peaks), {}


# Example usage and testing
if __name__ == "__main__":
    import json
    from ..config.manager import ConfigurationManager
    
    def main():
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize analyzer
        config_manager = ConfigurationManager()
        analyzer = TemporalPatternAnalyzer(config_manager)
        
        # Create sample data with temporal patterns
        dates = pd.date_range(start='2024-01-01', periods=2000, freq='1H')
        
        # Add some artificial patterns
        prices = 45000 + 1000 * np.sin(np.arange(2000) * 2 * np.pi / 24)  # Daily cycle
        prices += 500 * np.sin(np.arange(2000) * 2 * np.pi / 168)  # Weekly cycle
        prices += np.random.normal(0, 200, 2000)  # Noise
        
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.exponential(1000, 2000)
        })
        
        # Update analyzer with data
        analyzer.update_data('BTCUSDT', price_data)
        
        # Analyze all patterns
        analysis_results = analyzer.analyze_all_patterns('BTCUSDT')
        
        print("Temporal Pattern Analysis Results:")
        print(f"Summary: {json.dumps(analysis_results.get('summary', {}), indent=2, default=str)}")
        
        # Get current session
        current_session = analyzer.get_current_session()
        print(f"\nCurrent market session: {current_session.value}")
        
        # Get trading recommendation
        recommendation = analyzer.get_trading_recommendation('BTCUSDT')
        print(f"\nTrading recommendation: {json.dumps(recommendation, indent=2, default=str)}")
        
        # Get next significant time
        next_event = analyzer.get_next_significant_time('BTCUSDT')
        if next_event:
            print(f"\nNext significant time: {next_event[0]} - {next_event[1]}")
    
    # Run the example
    main()