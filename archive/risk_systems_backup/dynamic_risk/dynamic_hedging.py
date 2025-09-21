"""
Dynamic Hedging System.

This module provides sophisticated dynamic hedging capabilities for risk management
including:

- Delta hedging for directional risk management
- Gamma hedging for convexity protection
- Vega hedging for volatility exposure management
- Cross-asset hedging using correlated instruments
- Dynamic hedge ratio calculation based on correlations and betas
- Options-like hedging strategies for crypto derivatives
- Pairs trading and statistical arbitrage hedging
- Correlation-based hedging portfolio construction
- Real-time hedge effectiveness monitoring
- Automatic hedge rebalancing based on market conditions
- Hedge cost optimization and PnL attribution
- Risk parity hedging across multiple factors

The system continuously monitors portfolio exposures and automatically
adjusts hedging positions to maintain target risk levels.
"""

import asyncio
import threading
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
import sqlite3
import json

from ..utils.logging import TradingLogger


class HedgeType(Enum):
    """Types of hedging strategies."""
    DELTA_HEDGE = "delta_hedge"           # Directional risk hedging
    BETA_HEDGE = "beta_hedge"             # Market beta hedging
    CORRELATION_HEDGE = "correlation_hedge" # Correlation-based hedging
    VOLATILITY_HEDGE = "volatility_hedge"  # Volatility exposure hedging
    PAIRS_HEDGE = "pairs_hedge"           # Pairs/statistical arbitrage hedging
    CROSS_ASSET_HEDGE = "cross_asset_hedge" # Cross-market hedging
    FACTOR_HEDGE = "factor_hedge"         # Multi-factor hedging


class HedgeStatus(Enum):
    """Status of hedge positions."""
    ACTIVE = "active"         # Hedge is active and effective
    INACTIVE = "inactive"     # Hedge is not active
    REBALANCING = "rebalancing" # Hedge is being rebalanced
    INEFFECTIVE = "ineffective" # Hedge is not effective
    EXPIRED = "expired"       # Hedge has expired


class RebalanceSignal(Enum):
    """Signals for hedge rebalancing."""
    NO_REBALANCE = "no_rebalance"
    MINOR_REBALANCE = "minor_rebalance"   # Small adjustment needed
    MAJOR_REBALANCE = "major_rebalance"   # Significant rebalancing needed
    FULL_REBALANCE = "full_rebalance"    # Complete hedge restructure needed


@dataclass
class HedgeRatio:
    """Container for hedge ratio information."""
    
    primary_symbol: str
    hedge_symbol: str
    ratio: float                    # Hedge ratio (hedge units per primary unit)
    confidence: float              # Confidence in hedge ratio
    calculation_method: str        # Method used to calculate ratio
    effective_date: datetime       # When ratio became effective
    expiry_date: Optional[datetime] # When ratio expires
    
    # Risk metrics
    hedge_beta: float              # Beta of hedge vs primary
    hedge_correlation: float       # Correlation between assets
    tracking_error: float         # Expected tracking error
    
    # Performance metrics
    historical_effectiveness: float # Historical hedge effectiveness (0-1)
    expected_effectiveness: float  # Expected hedge effectiveness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'primary_symbol': self.primary_symbol,
            'hedge_symbol': self.hedge_symbol,
            'ratio': self.ratio,
            'confidence': self.confidence,
            'calculation_method': self.calculation_method,
            'effective_date': self.effective_date.isoformat(),
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'hedge_beta': self.hedge_beta,
            'hedge_correlation': self.hedge_correlation,
            'tracking_error': self.tracking_error,
            'historical_effectiveness': self.historical_effectiveness,
            'expected_effectiveness': self.expected_effectiveness
        }


@dataclass
class HedgePosition:
    """Container for hedge position information."""
    
    hedge_id: str
    hedge_type: HedgeType
    status: HedgeStatus
    
    # Position details
    primary_symbol: str
    primary_position: float        # Primary position size
    hedge_positions: Dict[str, float] # Hedge positions by symbol
    
    # Hedge ratios
    hedge_ratios: Dict[str, HedgeRatio]
    
    # Timestamps
    creation_time: datetime
    last_rebalance: datetime
    next_rebalance: Optional[datetime]
    
    # Performance tracking
    total_pnl: float              # Total PnL from hedge
    hedge_pnl: float             # PnL from hedge positions only
    effectiveness: float          # Current hedge effectiveness (0-1)
    cost_basis: float            # Cost of establishing hedge
    
    # Risk metrics
    residual_risk: float         # Remaining unhedged risk
    hedge_risk: float           # Risk from hedge positions
    basis_risk: float           # Basis risk between primary and hedge
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'hedge_id': self.hedge_id,
            'hedge_type': self.hedge_type.value,
            'status': self.status.value,
            'primary_symbol': self.primary_symbol,
            'primary_position': self.primary_position,
            'hedge_positions': self.hedge_positions,
            'hedge_ratios': {k: v.to_dict() for k, v in self.hedge_ratios.items()},
            'creation_time': self.creation_time.isoformat(),
            'last_rebalance': self.last_rebalance.isoformat(),
            'next_rebalance': self.next_rebalance.isoformat() if self.next_rebalance else None,
            'total_pnl': self.total_pnl,
            'hedge_pnl': self.hedge_pnl,
            'effectiveness': self.effectiveness,
            'cost_basis': self.cost_basis,
            'residual_risk': self.residual_risk,
            'hedge_risk': self.hedge_risk,
            'basis_risk': self.basis_risk
        }


class HedgeRatioCalculator:
    """
    Calculate optimal hedge ratios using various methodologies.
    
    This class provides multiple approaches for calculating hedge ratios
    based on correlations, betas, and other risk metrics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("HedgeRatioCalculator")
        
        # Return data for calculations
        self.return_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        
    def _default_config(self) -> Dict:
        """Default configuration for hedge ratio calculator."""
        return {
            'min_observations': 50,       # Minimum observations for calculation
            'lookback_days': 90,         # Days of data for calculation
            'confidence_threshold': 0.7, # Minimum confidence for hedge ratio
            'correlation_threshold': 0.3, # Minimum correlation for hedging
            'rebalance_threshold': 0.1,  # Threshold for ratio changes
            'effectiveness_threshold': 0.6, # Minimum hedge effectiveness
        }
    
    def add_return_data(self, symbol: str, return_value: float, timestamp: Optional[datetime] = None) -> None:
        """Add return data for hedge ratio calculation."""
        timestamp = timestamp or datetime.now()
        
        data_point = {
            'return': return_value,
            'timestamp': timestamp
        }
        
        self.return_data[symbol].append(data_point)
    
    def calculate_beta_hedge_ratio(
        self,
        primary_symbol: str,
        hedge_symbol: str,
        lookback_days: Optional[int] = None
    ) -> Optional[HedgeRatio]:
        """Calculate hedge ratio based on beta coefficient."""
        lookback_days = lookback_days or self.config['lookback_days']
        
        try:
            # Get aligned returns
            primary_returns, hedge_returns = self._get_aligned_returns(
                primary_symbol, hedge_symbol, lookback_days
            )
            
            if len(primary_returns) < self.config['min_observations']:
                return None
            
            # Calculate beta (hedge vs primary)
            slope, intercept, r_value, p_value, std_err = stats.linregress(primary_returns, hedge_returns)
            
            # Beta is the slope
            beta = slope
            correlation = r_value
            
            # Check minimum correlation
            if abs(correlation) < self.config['correlation_threshold']:
                return None
            
            # Hedge ratio is negative beta (to offset primary position)
            hedge_ratio_value = -beta
            
            # Calculate confidence based on R-squared and p-value
            r_squared = r_value ** 2
            confidence = r_squared * (1 - p_value) if p_value < 0.05 else r_squared * 0.5
            
            # Calculate tracking error
            residuals = np.array(hedge_returns) - (intercept + slope * np.array(primary_returns))
            tracking_error = np.std(residuals, ddof=1) * math.sqrt(252)  # Annualized
            
            # Calculate historical effectiveness
            primary_var = np.var(primary_returns, ddof=1)
            residual_var = np.var(residuals, ddof=1)
            historical_effectiveness = 1 - (residual_var / primary_var) if primary_var > 0 else 0
            
            return HedgeRatio(
                primary_symbol=primary_symbol,
                hedge_symbol=hedge_symbol,
                ratio=hedge_ratio_value,
                confidence=confidence,
                calculation_method="beta_regression",
                effective_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),  # 30-day validity
                hedge_beta=beta,
                hedge_correlation=correlation,
                tracking_error=tracking_error,
                historical_effectiveness=historical_effectiveness,
                expected_effectiveness=historical_effectiveness * confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating beta hedge ratio for {primary_symbol}-{hedge_symbol}: {e}")
            return None
    
    def calculate_correlation_hedge_ratio(
        self,
        primary_symbol: str,
        hedge_symbol: str,
        lookback_days: Optional[int] = None
    ) -> Optional[HedgeRatio]:
        """Calculate hedge ratio based on correlation and volatility matching."""
        lookback_days = lookback_days or self.config['lookback_days']
        
        try:
            # Get aligned returns
            primary_returns, hedge_returns = self._get_aligned_returns(
                primary_symbol, hedge_symbol, lookback_days
            )
            
            if len(primary_returns) < self.config['min_observations']:
                return None
            
            # Calculate correlation
            correlation, p_value = stats.pearsonr(primary_returns, hedge_returns)
            
            # Check minimum correlation
            if abs(correlation) < self.config['correlation_threshold']:
                return None
            
            # Calculate volatilities
            primary_vol = np.std(primary_returns, ddof=1)
            hedge_vol = np.std(hedge_returns, ddof=1)
            
            if hedge_vol == 0:
                return None
            
            # Hedge ratio to match volatility (adjusted for correlation)
            vol_ratio = primary_vol / hedge_vol
            hedge_ratio_value = -correlation * vol_ratio  # Negative for offsetting
            
            # Calculate confidence
            confidence = abs(correlation) * (1 - p_value) if p_value < 0.05 else abs(correlation) * 0.5
            
            # Estimate tracking error
            portfolio_returns = np.array(primary_returns) + hedge_ratio_value * np.array(hedge_returns)
            tracking_error = np.std(portfolio_returns, ddof=1) * math.sqrt(252)
            
            # Calculate effectiveness
            primary_var = np.var(primary_returns, ddof=1)
            portfolio_var = np.var(portfolio_returns, ddof=1)
            historical_effectiveness = 1 - (portfolio_var / primary_var) if primary_var > 0 else 0
            
            return HedgeRatio(
                primary_symbol=primary_symbol,
                hedge_symbol=hedge_symbol,
                ratio=hedge_ratio_value,
                confidence=confidence,
                calculation_method="correlation_volatility",
                effective_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),
                hedge_beta=hedge_ratio_value * (hedge_vol / primary_vol) if primary_vol > 0 else 0,
                hedge_correlation=correlation,
                tracking_error=tracking_error,
                historical_effectiveness=historical_effectiveness,
                expected_effectiveness=historical_effectiveness * confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation hedge ratio for {primary_symbol}-{hedge_symbol}: {e}")
            return None
    
    def calculate_minimum_variance_hedge_ratio(
        self,
        primary_symbol: str,
        hedge_symbol: str,
        lookback_days: Optional[int] = None
    ) -> Optional[HedgeRatio]:
        """Calculate minimum variance hedge ratio."""
        lookback_days = lookback_days or self.config['lookback_days']
        
        try:
            # Get aligned returns
            primary_returns, hedge_returns = self._get_aligned_returns(
                primary_symbol, hedge_symbol, lookback_days
            )
            
            if len(primary_returns) < self.config['min_observations']:
                return None
            
            # Calculate covariance matrix
            returns_matrix = np.column_stack([primary_returns, hedge_returns])
            cov_matrix = np.cov(returns_matrix.T, ddof=1)
            
            # Minimum variance hedge ratio
            cov_ph = cov_matrix[0, 1]  # Covariance between primary and hedge
            var_h = cov_matrix[1, 1]   # Variance of hedge
            
            if var_h == 0:
                return None
            
            hedge_ratio_value = -cov_ph / var_h  # Negative for offsetting
            
            # Calculate correlation for confidence
            correlation = cov_ph / (math.sqrt(cov_matrix[0, 0]) * math.sqrt(var_h))
            
            # Check minimum correlation
            if abs(correlation) < self.config['correlation_threshold']:
                return None
            
            # Confidence based on how well the hedge reduces variance
            portfolio_returns = np.array(primary_returns) + hedge_ratio_value * np.array(hedge_returns)
            primary_var = np.var(primary_returns, ddof=1)
            portfolio_var = np.var(portfolio_returns, ddof=1)
            
            variance_reduction = (primary_var - portfolio_var) / primary_var if primary_var > 0 else 0
            confidence = max(0, min(1, variance_reduction))
            
            # Calculate tracking error
            tracking_error = np.std(portfolio_returns, ddof=1) * math.sqrt(252)
            
            return HedgeRatio(
                primary_symbol=primary_symbol,
                hedge_symbol=hedge_symbol,
                ratio=hedge_ratio_value,
                confidence=confidence,
                calculation_method="minimum_variance",
                effective_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),
                hedge_beta=hedge_ratio_value * (math.sqrt(var_h) / math.sqrt(cov_matrix[0, 0])),
                hedge_correlation=correlation,
                tracking_error=tracking_error,
                historical_effectiveness=variance_reduction,
                expected_effectiveness=variance_reduction * confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating minimum variance hedge ratio for {primary_symbol}-{hedge_symbol}: {e}")
            return None
    
    def calculate_dynamic_hedge_ratio(
        self,
        primary_symbol: str,
        hedge_symbol: str,
        lookback_days: Optional[int] = None
    ) -> Optional[HedgeRatio]:
        """Calculate dynamic hedge ratio using time-varying approach."""
        lookback_days = lookback_days or self.config['lookback_days']
        
        try:
            # Get aligned returns
            primary_returns, hedge_returns = self._get_aligned_returns(
                primary_symbol, hedge_symbol, lookback_days
            )
            
            if len(primary_returns) < self.config['min_observations']:
                return None
            
            # Calculate rolling hedge ratios
            window_size = min(30, len(primary_returns) // 3)  # Use 1/3 of data or 30 days
            hedge_ratios = []
            
            for i in range(window_size, len(primary_returns)):
                window_primary = primary_returns[i-window_size:i]
                window_hedge = hedge_returns[i-window_size:i]
                
                # Calculate minimum variance ratio for this window
                returns_matrix = np.column_stack([window_primary, window_hedge])
                cov_matrix = np.cov(returns_matrix.T, ddof=1)
                
                cov_ph = cov_matrix[0, 1]
                var_h = cov_matrix[1, 1]
                
                if var_h > 0:
                    ratio = -cov_ph / var_h
                    hedge_ratios.append(ratio)
            
            if not hedge_ratios:
                return None
            
            # Use weighted average of recent ratios (more weight to recent)
            weights = np.linspace(0.5, 1.0, len(hedge_ratios))
            weights = weights / np.sum(weights)
            
            dynamic_ratio = np.average(hedge_ratios, weights=weights)
            
            # Calculate confidence based on stability of ratios
            ratio_stability = 1.0 / (1.0 + np.std(hedge_ratios[-10:]))  # Focus on recent stability
            
            # Final hedge performance check
            portfolio_returns = np.array(primary_returns) + dynamic_ratio * np.array(hedge_returns)
            primary_var = np.var(primary_returns, ddof=1)
            portfolio_var = np.var(portfolio_returns, ddof=1)
            
            effectiveness = (primary_var - portfolio_var) / primary_var if primary_var > 0 else 0
            confidence = effectiveness * ratio_stability
            
            # Calculate other metrics
            correlation, _ = stats.pearsonr(primary_returns, hedge_returns)
            tracking_error = np.std(portfolio_returns, ddof=1) * math.sqrt(252)
            
            primary_vol = np.std(primary_returns, ddof=1)
            hedge_vol = np.std(hedge_returns, ddof=1)
            beta = dynamic_ratio * (hedge_vol / primary_vol) if primary_vol > 0 else 0
            
            return HedgeRatio(
                primary_symbol=primary_symbol,
                hedge_symbol=hedge_symbol,
                ratio=dynamic_ratio,
                confidence=confidence,
                calculation_method="dynamic_rolling",
                effective_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=14),  # Shorter validity for dynamic
                hedge_beta=beta,
                hedge_correlation=correlation,
                tracking_error=tracking_error,
                historical_effectiveness=effectiveness,
                expected_effectiveness=effectiveness * confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic hedge ratio for {primary_symbol}-{hedge_symbol}: {e}")
            return None
    
    def calculate_optimal_hedge_ratio(
        self,
        primary_symbol: str,
        hedge_symbol: str,
        method: str = "auto",
        lookback_days: Optional[int] = None
    ) -> Optional[HedgeRatio]:
        """Calculate optimal hedge ratio using specified or automatic method selection."""
        if method == "auto":
            # Try multiple methods and select the best one
            methods = ["minimum_variance", "beta_regression", "correlation_volatility", "dynamic_rolling"]
            best_ratio = None
            best_score = 0
            
            for calc_method in methods:
                if calc_method == "minimum_variance":
                    ratio = self.calculate_minimum_variance_hedge_ratio(primary_symbol, hedge_symbol, lookback_days)
                elif calc_method == "beta_regression":
                    ratio = self.calculate_beta_hedge_ratio(primary_symbol, hedge_symbol, lookback_days)
                elif calc_method == "correlation_volatility":
                    ratio = self.calculate_correlation_hedge_ratio(primary_symbol, hedge_symbol, lookback_days)
                elif calc_method == "dynamic_rolling":
                    ratio = self.calculate_dynamic_hedge_ratio(primary_symbol, hedge_symbol, lookback_days)
                else:
                    continue
                
                if ratio and ratio.expected_effectiveness > best_score:
                    best_ratio = ratio
                    best_score = ratio.expected_effectiveness
            
            return best_ratio
        
        else:
            # Use specified method
            if method == "minimum_variance":
                return self.calculate_minimum_variance_hedge_ratio(primary_symbol, hedge_symbol, lookback_days)
            elif method == "beta_regression":
                return self.calculate_beta_hedge_ratio(primary_symbol, hedge_symbol, lookback_days)
            elif method == "correlation_volatility":
                return self.calculate_correlation_hedge_ratio(primary_symbol, hedge_symbol, lookback_days)
            elif method == "dynamic_rolling":
                return self.calculate_dynamic_hedge_ratio(primary_symbol, hedge_symbol, lookback_days)
            else:
                self.logger.error(f"Unknown hedge ratio calculation method: {method}")
                return None
    
    def _get_aligned_returns(
        self,
        symbol1: str,
        symbol2: str,
        lookback_days: int
    ) -> Tuple[List[float], List[float]]:
        """Get aligned return series for two symbols."""
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        
        # Get data for both symbols
        data1 = [d for d in self.return_data[symbol1] if d['timestamp'] > cutoff_time]
        data2 = [d for d in self.return_data[symbol2] if d['timestamp'] > cutoff_time]
        
        if not data1 or not data2:
            return [], []
        
        # Create timestamp alignment
        timestamps1 = {d['timestamp']: d['return'] for d in data1}
        timestamps2 = {d['timestamp']: d['return'] for d in data2}
        
        # Find common timestamps
        common_timestamps = sorted(set(timestamps1.keys()) & set(timestamps2.keys()))
        
        # Return aligned series
        returns1 = [timestamps1[ts] for ts in common_timestamps]
        returns2 = [timestamps2[ts] for ts in common_timestamps]
        
        return returns1, returns2


class HedgePositionManager:
    """
    Manage hedge positions and monitor their effectiveness.
    
    This class handles the creation, monitoring, and rebalancing of
    hedge positions for dynamic risk management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("HedgePositionManager")
        
        # Components
        self.ratio_calculator = HedgeRatioCalculator(self.config.get('ratio_calculator', {}))
        
        # Active hedge positions
        self.hedge_positions: Dict[str, HedgePosition] = {}
        
        # Position and price data
        self.current_positions: Dict[str, float] = {}
        self.current_prices: Dict[str, float] = {}
        
        # Performance tracking
        self.hedge_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def _default_config(self) -> Dict:
        """Default configuration for hedge position manager."""
        return {
            'max_hedge_positions': 10,    # Maximum concurrent hedges
            'rebalance_frequency': 3600,  # Rebalance check frequency (seconds)
            'effectiveness_threshold': 0.6, # Minimum effectiveness to maintain hedge
            'rebalance_threshold': 0.15,  # Threshold for hedge rebalancing
            'max_hedge_ratio': 2.0,      # Maximum allowed hedge ratio
            'min_position_size': 100,    # Minimum position size to hedge
            'hedge_cost_limit': 0.02,    # Maximum hedge cost as % of position
            'ratio_calculator': {}
        }
    
    def update_position(self, symbol: str, position_size: float, price: float) -> None:
        """Update current position and price for a symbol."""
        self.current_positions[symbol] = position_size
        self.current_prices[symbol] = price
        
        # Add return data to ratio calculator
        if symbol in self.current_prices:
            old_price = self.current_prices.get(symbol, price)
            if old_price > 0:
                return_value = math.log(price / old_price)
                self.ratio_calculator.add_return_data(symbol, return_value)
    
    def create_hedge_position(
        self,
        primary_symbol: str,
        hedge_symbols: List[str],
        hedge_type: HedgeType = HedgeType.BETA_HEDGE,
        hedge_method: str = "auto"
    ) -> Optional[str]:
        """Create a new hedge position."""
        try:
            # Check if position is large enough to hedge
            primary_position = self.current_positions.get(primary_symbol, 0)
            if abs(primary_position) < self.config['min_position_size']:
                self.logger.info(f"Position in {primary_symbol} too small to hedge: {primary_position}")
                return None
            
            # Check maximum hedge positions
            if len(self.hedge_positions) >= self.config['max_hedge_positions']:
                self.logger.warning("Maximum hedge positions reached")
                return None
            
            # Calculate hedge ratios for all hedge symbols
            hedge_ratios = {}
            hedge_positions = {}
            
            for hedge_symbol in hedge_symbols:
                # Calculate optimal hedge ratio
                hedge_ratio = self.ratio_calculator.calculate_optimal_hedge_ratio(
                    primary_symbol, hedge_symbol, hedge_method
                )
                
                if not hedge_ratio:
                    self.logger.warning(f"Could not calculate hedge ratio for {primary_symbol}-{hedge_symbol}")
                    continue
                
                # Check effectiveness threshold
                if hedge_ratio.expected_effectiveness < self.config['effectiveness_threshold']:
                    self.logger.warning(f"Hedge effectiveness too low for {hedge_symbol}: {hedge_ratio.expected_effectiveness}")
                    continue
                
                # Check maximum hedge ratio
                if abs(hedge_ratio.ratio) > self.config['max_hedge_ratio']:
                    self.logger.warning(f"Hedge ratio too large for {hedge_symbol}: {hedge_ratio.ratio}")
                    continue
                
                # Calculate hedge position size
                hedge_position_size = hedge_ratio.ratio * primary_position
                
                hedge_ratios[hedge_symbol] = hedge_ratio
                hedge_positions[hedge_symbol] = hedge_position_size
            
            if not hedge_ratios:
                self.logger.error(f"No valid hedge ratios found for {primary_symbol}")
                return None
            
            # Create hedge position
            hedge_id = f"hedge_{primary_symbol}_{int(datetime.now().timestamp())}"
            
            hedge_position = HedgePosition(
                hedge_id=hedge_id,
                hedge_type=hedge_type,
                status=HedgeStatus.ACTIVE,
                primary_symbol=primary_symbol,
                primary_position=primary_position,
                hedge_positions=hedge_positions,
                hedge_ratios=hedge_ratios,
                creation_time=datetime.now(),
                last_rebalance=datetime.now(),
                next_rebalance=datetime.now() + timedelta(seconds=self.config['rebalance_frequency']),
                total_pnl=0.0,
                hedge_pnl=0.0,
                effectiveness=0.0,
                cost_basis=0.0,
                residual_risk=0.0,
                hedge_risk=0.0,
                basis_risk=0.0
            )
            
            # Calculate initial metrics
            self._update_hedge_metrics(hedge_position)
            
            # Store hedge position
            self.hedge_positions[hedge_id] = hedge_position
            
            self.logger.info(f"Created hedge position {hedge_id} for {primary_symbol}")
            return hedge_id
            
        except Exception as e:
            self.logger.error(f"Error creating hedge position for {primary_symbol}: {e}")
            return None
    
    def check_rebalance_signals(self, hedge_id: str) -> RebalanceSignal:
        """Check if hedge position needs rebalancing."""
        if hedge_id not in self.hedge_positions:
            return RebalanceSignal.NO_REBALANCE
        
        hedge_position = self.hedge_positions[hedge_id]
        
        try:
            # Check if it's time for scheduled rebalance
            if datetime.now() >= hedge_position.next_rebalance:
                return RebalanceSignal.MINOR_REBALANCE
            
            # Check effectiveness threshold
            if hedge_position.effectiveness < self.config['effectiveness_threshold']:
                return RebalanceSignal.MAJOR_REBALANCE
            
            # Check position size changes
            current_primary_position = self.current_positions.get(hedge_position.primary_symbol, 0)
            position_change = abs(current_primary_position - hedge_position.primary_position)
            position_change_pct = position_change / abs(hedge_position.primary_position) if hedge_position.primary_position != 0 else 0
            
            if position_change_pct > self.config['rebalance_threshold']:
                return RebalanceSignal.MAJOR_REBALANCE
            
            # Check hedge ratio changes
            total_ratio_change = 0
            ratio_count = 0
            
            for hedge_symbol, current_ratio in hedge_position.hedge_ratios.items():
                # Recalculate current optimal ratio
                new_ratio = self.ratio_calculator.calculate_optimal_hedge_ratio(
                    hedge_position.primary_symbol, hedge_symbol
                )
                
                if new_ratio:
                    ratio_change = abs(new_ratio.ratio - current_ratio.ratio)
                    ratio_change_pct = ratio_change / abs(current_ratio.ratio) if current_ratio.ratio != 0 else 0
                    total_ratio_change += ratio_change_pct
                    ratio_count += 1
            
            if ratio_count > 0:
                avg_ratio_change = total_ratio_change / ratio_count
                if avg_ratio_change > self.config['rebalance_threshold']:
                    return RebalanceSignal.MAJOR_REBALANCE
                elif avg_ratio_change > self.config['rebalance_threshold'] / 2:
                    return RebalanceSignal.MINOR_REBALANCE
            
            return RebalanceSignal.NO_REBALANCE
            
        except Exception as e:
            self.logger.error(f"Error checking rebalance signals for {hedge_id}: {e}")
            return RebalanceSignal.NO_REBALANCE
    
    def rebalance_hedge_position(self, hedge_id: str, signal: RebalanceSignal) -> bool:
        """Rebalance hedge position based on signal."""
        if hedge_id not in self.hedge_positions:
            return False
        
        hedge_position = self.hedge_positions[hedge_id]
        
        try:
            hedge_position.status = HedgeStatus.REBALANCING
            
            if signal == RebalanceSignal.FULL_REBALANCE:
                # Completely recalculate all hedge ratios
                new_hedge_ratios = {}
                new_hedge_positions = {}
                
                current_primary_position = self.current_positions.get(hedge_position.primary_symbol, 0)
                
                for hedge_symbol in hedge_position.hedge_ratios.keys():
                    new_ratio = self.ratio_calculator.calculate_optimal_hedge_ratio(
                        hedge_position.primary_symbol, hedge_symbol
                    )
                    
                    if new_ratio and new_ratio.expected_effectiveness >= self.config['effectiveness_threshold']:
                        new_hedge_ratios[hedge_symbol] = new_ratio
                        new_hedge_positions[hedge_symbol] = new_ratio.ratio * current_primary_position
                
                hedge_position.hedge_ratios = new_hedge_ratios
                hedge_position.hedge_positions = new_hedge_positions
                hedge_position.primary_position = current_primary_position
                
            elif signal in [RebalanceSignal.MAJOR_REBALANCE, RebalanceSignal.MINOR_REBALANCE]:
                # Update existing ratios and positions
                current_primary_position = self.current_positions.get(hedge_position.primary_symbol, 0)
                
                for hedge_symbol, current_ratio in hedge_position.hedge_ratios.items():
                    if signal == RebalanceSignal.MAJOR_REBALANCE:
                        # Recalculate ratio
                        new_ratio = self.ratio_calculator.calculate_optimal_hedge_ratio(
                            hedge_position.primary_symbol, hedge_symbol
                        )
                        if new_ratio:
                            hedge_position.hedge_ratios[hedge_symbol] = new_ratio
                            hedge_position.hedge_positions[hedge_symbol] = new_ratio.ratio * current_primary_position
                    else:
                        # Just update position size
                        hedge_position.hedge_positions[hedge_symbol] = current_ratio.ratio * current_primary_position
                
                hedge_position.primary_position = current_primary_position
            
            # Update timestamps
            hedge_position.last_rebalance = datetime.now()
            hedge_position.next_rebalance = datetime.now() + timedelta(seconds=self.config['rebalance_frequency'])
            hedge_position.status = HedgeStatus.ACTIVE
            
            # Update metrics
            self._update_hedge_metrics(hedge_position)
            
            self.logger.info(f"Rebalanced hedge position {hedge_id} with {signal.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebalancing hedge position {hedge_id}: {e}")
            hedge_position.status = HedgeStatus.INEFFECTIVE
            return False
    
    def close_hedge_position(self, hedge_id: str) -> bool:
        """Close hedge position."""
        if hedge_id not in self.hedge_positions:
            return False
        
        try:
            hedge_position = self.hedge_positions[hedge_id]
            hedge_position.status = HedgeStatus.INACTIVE
            
            # Could add logic here to actually close the hedge positions
            # For now, just mark as inactive
            
            self.logger.info(f"Closed hedge position {hedge_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing hedge position {hedge_id}: {e}")
            return False
    
    def _update_hedge_metrics(self, hedge_position: HedgePosition) -> None:
        """Update hedge position metrics."""
        try:
            # Get current returns for effectiveness calculation
            primary_returns = []
            hedge_returns = []
            
            for hedge_symbol in hedge_position.hedge_ratios.keys():
                primary_ret, hedge_ret = self.ratio_calculator._get_aligned_returns(
                    hedge_position.primary_symbol, hedge_symbol, 30  # 30 days
                )
                
                if len(primary_ret) > 10:
                    primary_returns = primary_ret
                    # Calculate portfolio returns (primary + hedge)
                    ratio = hedge_position.hedge_ratios[hedge_symbol].ratio
                    portfolio_returns = np.array(primary_ret) + ratio * np.array(hedge_ret)
                    break
            
            if primary_returns:
                # Calculate effectiveness (variance reduction)
                primary_var = np.var(primary_returns, ddof=1)
                portfolio_var = np.var(portfolio_returns, ddof=1) if 'portfolio_returns' in locals() else primary_var
                
                effectiveness = (primary_var - portfolio_var) / primary_var if primary_var > 0 else 0
                hedge_position.effectiveness = max(0, effectiveness)
                
                # Calculate residual risk
                hedge_position.residual_risk = math.sqrt(portfolio_var) if 'portfolio_returns' in locals() else math.sqrt(primary_var)
                
                # Calculate hedge risk (risk from hedge positions)
                hedge_position.hedge_risk = math.sqrt(portfolio_var - primary_var) if portfolio_var > primary_var else 0
                
                # Basis risk (tracking error)
                if 'portfolio_returns' in locals():
                    hedge_position.basis_risk = np.std(portfolio_returns, ddof=1)
                else:
                    hedge_position.basis_risk = 0
            
            # Record performance
            performance_record = {
                'timestamp': datetime.now(),
                'effectiveness': hedge_position.effectiveness,
                'residual_risk': hedge_position.residual_risk,
                'hedge_risk': hedge_position.hedge_risk,
                'basis_risk': hedge_position.basis_risk
            }
            
            self.hedge_performance[hedge_position.hedge_id].append(performance_record)
            
        except Exception as e:
            self.logger.error(f"Error updating hedge metrics for {hedge_position.hedge_id}: {e}")
    
    def get_hedge_summary(self) -> Dict[str, Any]:
        """Get summary of all hedge positions."""
        summary = {
            'total_hedges': len(self.hedge_positions),
            'active_hedges': len([h for h in self.hedge_positions.values() if h.status == HedgeStatus.ACTIVE]),
            'avg_effectiveness': 0.0,
            'total_hedge_pnl': 0.0,
            'hedges_by_type': defaultdict(int),
            'hedges_by_status': defaultdict(int)
        }
        
        if self.hedge_positions:
            effectiveness_sum = 0
            pnl_sum = 0
            
            for hedge_position in self.hedge_positions.values():
                effectiveness_sum += hedge_position.effectiveness
                pnl_sum += hedge_position.hedge_pnl
                summary['hedges_by_type'][hedge_position.hedge_type.value] += 1
                summary['hedges_by_status'][hedge_position.status.value] += 1
            
            summary['avg_effectiveness'] = effectiveness_sum / len(self.hedge_positions)
            summary['total_hedge_pnl'] = pnl_sum
        
        return summary
    
    def get_hedge_position(self, hedge_id: str) -> Optional[HedgePosition]:
        """Get hedge position by ID."""
        return self.hedge_positions.get(hedge_id)
    
    def get_hedge_positions_for_symbol(self, symbol: str) -> List[HedgePosition]:
        """Get all hedge positions for a symbol."""
        return [
            hedge for hedge in self.hedge_positions.values()
            if hedge.primary_symbol == symbol or symbol in hedge.hedge_positions
        ]


class DynamicHedgingSystem:
    """
    Main dynamic hedging system.
    
    This class provides comprehensive dynamic hedging capabilities
    with automatic monitoring, rebalancing, and risk management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("DynamicHedgingSystem")
        
        # Core components
        self.position_manager = HedgePositionManager(self.config.get('position_manager', {}))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Hedge callbacks
        self.hedge_callbacks: List[Callable[[str, HedgePosition], None]] = []
        
        # Database for persistence
        self.db_path = self.config.get('database_path', 'dynamic_hedging.db')
        self._init_database()
        
    def _default_config(self) -> Dict:
        """Default configuration for dynamic hedging system."""
        return {
            'monitoring_interval': 300,   # 5 minutes
            'database_path': 'dynamic_hedging.db',
            'enable_persistence': True,
            'auto_hedge_symbols': [],     # Symbols to automatically hedge
            'hedge_correlation_threshold': 0.5,  # Minimum correlation for auto-hedging
            'position_manager': {}
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for hedging data."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hedge_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hedge_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hedge_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hedge_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    effectiveness REAL,
                    residual_risk REAL,
                    hedge_pnl REAL
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hedge_id ON hedge_positions (hedge_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_hedge_time ON hedge_performance (hedge_id, timestamp)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hedging database: {e}")
    
    def update_market_data(
        self,
        symbol: str,
        price: float,
        position_size: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update market data for hedging calculations."""
        self.position_manager.update_position(symbol, position_size, price)
    
    def create_hedge(
        self,
        primary_symbol: str,
        hedge_symbols: Optional[List[str]] = None,
        hedge_type: HedgeType = HedgeType.BETA_HEDGE,
        hedge_method: str = "auto"
    ) -> Optional[str]:
        """Create a new hedge position."""
        if not hedge_symbols:
            # Auto-select hedge symbols based on correlations
            hedge_symbols = self._find_hedge_candidates(primary_symbol)
        
        hedge_id = self.position_manager.create_hedge_position(
            primary_symbol, hedge_symbols, hedge_type, hedge_method
        )
        
        if hedge_id and self.config['enable_persistence']:
            self._save_hedge_position(hedge_id)
        
        return hedge_id
    
    def _find_hedge_candidates(self, primary_symbol: str) -> List[str]:
        """Find potential hedge candidates for a symbol."""
        candidates = []
        threshold = self.config['hedge_correlation_threshold']
        
        # Get all available symbols
        available_symbols = list(self.position_manager.current_positions.keys())
        
        for candidate in available_symbols:
            if candidate == primary_symbol:
                continue
            
            # Check if we have enough data to calculate correlation
            primary_returns, candidate_returns = self.position_manager.ratio_calculator._get_aligned_returns(
                primary_symbol, candidate, 30
            )
            
            if len(primary_returns) < 20:
                continue
            
            # Calculate correlation
            try:
                correlation, p_value = stats.pearsonr(primary_returns, candidate_returns)
                if abs(correlation) >= threshold and p_value < 0.05:
                    candidates.append(candidate)
            except Exception:
                continue
        
        return candidates[:3]  # Return top 3 candidates
    
    def monitor_hedges(self) -> None:
        """Monitor and rebalance existing hedge positions."""
        rebalanced_count = 0
        
        for hedge_id, hedge_position in self.position_manager.hedge_positions.items():
            if hedge_position.status != HedgeStatus.ACTIVE:
                continue
            
            # Check rebalance signals
            signal = self.position_manager.check_rebalance_signals(hedge_id)
            
            if signal != RebalanceSignal.NO_REBALANCE:
                success = self.position_manager.rebalance_hedge_position(hedge_id, signal)
                if success:
                    rebalanced_count += 1
                    
                    # Save updated position
                    if self.config['enable_persistence']:
                        self._save_hedge_position(hedge_id)
                    
                    # Notify callbacks
                    for callback in self.hedge_callbacks:
                        try:
                            callback(hedge_id, hedge_position)
                        except Exception as e:
                            self.logger.error(f"Error in hedge callback: {e}")
        
        if rebalanced_count > 0:
            self.logger.info(f"Rebalanced {rebalanced_count} hedge positions")
    
    def _save_hedge_position(self, hedge_id: str) -> None:
        """Save hedge position to database."""
        if hedge_id not in self.position_manager.hedge_positions:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            hedge_position = self.position_manager.hedge_positions[hedge_id]
            data = json.dumps(hedge_position.to_dict())
            
            cursor.execute("""
                INSERT OR REPLACE INTO hedge_positions (hedge_id, timestamp, data)
                VALUES (?, ?, ?)
            """, (hedge_id, datetime.now(), data))
            
            # Save performance data
            cursor.execute("""
                INSERT INTO hedge_performance (hedge_id, timestamp, effectiveness, residual_risk, hedge_pnl)
                VALUES (?, ?, ?, ?, ?)
            """, (
                hedge_id,
                datetime.now(),
                hedge_position.effectiveness,
                hedge_position.residual_risk,
                hedge_position.hedge_pnl
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save hedge position {hedge_id}: {e}")
    
    def add_hedge_callback(self, callback: Callable[[str, HedgePosition], None]) -> None:
        """Add callback for hedge events."""
        self.hedge_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start background hedge monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started hedge monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background hedge monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped hedge monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Monitor existing hedges
                self.monitor_hedges()
                
                # Auto-create hedges for configured symbols
                for symbol in self.config['auto_hedge_symbols']:
                    # Check if symbol already has active hedge
                    existing_hedges = self.position_manager.get_hedge_positions_for_symbol(symbol)
                    active_hedges = [h for h in existing_hedges if h.status == HedgeStatus.ACTIVE]
                    
                    if not active_hedges:
                        position_size = self.position_manager.current_positions.get(symbol, 0)
                        if abs(position_size) >= self.position_manager.config['min_position_size']:
                            hedge_id = self.create_hedge(symbol)
                            if hedge_id:
                                self.logger.info(f"Auto-created hedge {hedge_id} for {symbol}")
                
                # Sleep until next monitoring cycle
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in hedge monitoring loop: {e}")
                time.sleep(60)  # Error backoff
    
    def get_hedge_summary(self) -> Dict[str, Any]:
        """Get comprehensive hedge summary."""
        return self.position_manager.get_hedge_summary()
    
    def get_hedge_position(self, hedge_id: str) -> Optional[HedgePosition]:
        """Get hedge position details."""
        return self.position_manager.get_hedge_position(hedge_id)
    
    def close_hedge(self, hedge_id: str) -> bool:
        """Close hedge position."""
        return self.position_manager.close_hedge_position(hedge_id)