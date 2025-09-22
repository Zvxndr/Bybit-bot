"""
Bybit-Specific Enhanced Backtesting Engine

This module implements a comprehensive backtesting system specifically designed for Bybit trading:

Enhanced Features:
- Historical data integration with funding rate simulation
- Bybit-specific fee structure (VIP tiers, maker/taker)
- Perpetual swap liquidation risk modeling
- Realistic order execution with market impact
- Funding cost calculation and P&L impact
- Enhanced slippage modeling based on order book depth
- Tiered margin requirements and leverage constraints

This engine extends the base BacktestEngine with Bybit-specific enhancements
while maintaining full compatibility with existing strategy pipeline.

Author: Trading Bot Team
Version: 3.0.0 - Bybit Enhanced
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import math

from .backtest_engine import BacktestEngine, BacktestResults, BacktestTrade
from ..exchange.bybit_client import BybitClient, BybitCredentials
from ..data.historical_data_manager import HistoricalDataManager, DataFetchRequest, EnhancedMarketData
from ..utils.rate_limiter import RateLimiter
from ..config_manager import ConfigurationManager
from ..utils.logging import TradingLogger
from ..core.strategy_manager import BaseStrategy, TradingSignal, SignalType


class BybitVIPTier(Enum):
    """Bybit VIP tier levels affecting fee rates."""
    NO_VIP = "No VIP"
    VIP1 = "VIP 1"
    VIP2 = "VIP 2"
    VIP3 = "VIP 3"
    PRO1 = "PRO 1"
    PRO2 = "PRO 2"
    PRO3 = "PRO 3"


class BybitContractType(Enum):
    """Bybit contract types with different fee structures."""
    LINEAR_PERPETUAL = "linear_perpetual"
    INVERSE_PERPETUAL = "inverse_perpetual"
    LINEAR_FUTURES = "linear_futures"
    INVERSE_FUTURES = "inverse_futures"
    SPOT = "spot"


@dataclass
class BybitFeeStructure:
    """Bybit fee structure for different VIP tiers and contract types."""
    
    # Linear Perpetual (USDT) - Most common
    LINEAR_PERP_FEES = {
        BybitVIPTier.NO_VIP: {"maker": Decimal("0.0001"), "taker": Decimal("0.0006")},
        BybitVIPTier.VIP1: {"maker": Decimal("0.0001"), "taker": Decimal("0.0005")},
        BybitVIPTier.VIP2: {"maker": Decimal("0.0000"), "taker": Decimal("0.0004")},
        BybitVIPTier.VIP3: {"maker": Decimal("0.0000"), "taker": Decimal("0.0003")},
        BybitVIPTier.PRO1: {"maker": Decimal("-0.0001"), "taker": Decimal("0.0002")},
        BybitVIPTier.PRO2: {"maker": Decimal("-0.0002"), "taker": Decimal("0.0001")},
        BybitVIPTier.PRO3: {"maker": Decimal("-0.0002"), "taker": Decimal("0.0001")},
    }
    
    # Spot trading fees
    SPOT_FEES = {
        BybitVIPTier.NO_VIP: {"maker": Decimal("0.001"), "taker": Decimal("0.001")},
        BybitVIPTier.VIP1: {"maker": Decimal("0.0009"), "taker": Decimal("0.0009")},
        BybitVIPTier.VIP2: {"maker": Decimal("0.0008"), "taker": Decimal("0.0008")},
        BybitVIPTier.VIP3: {"maker": Decimal("0.0007"), "taker": Decimal("0.0007")},
        BybitVIPTier.PRO1: {"maker": Decimal("0.0006"), "taker": Decimal("0.0006")},
        BybitVIPTier.PRO2: {"maker": Decimal("0.0005"), "taker": Decimal("0.0005")},
        BybitVIPTier.PRO3: {"maker": Decimal("0.0004"), "taker": Decimal("0.0004")},
    }


@dataclass
class BybitMarginRequirements:
    """Bybit margin requirements for different symbols and tiers."""
    
    # Initial margin requirements by leverage
    INITIAL_MARGIN_TIERS = {
        # BTCUSDT examples
        "BTCUSDT": [
            {"leverage_range": (1, 12.5), "initial_margin": Decimal("0.08"), "maintenance_margin": Decimal("0.05")},
            {"leverage_range": (12.5, 25), "initial_margin": Decimal("0.10"), "maintenance_margin": Decimal("0.055")},
            {"leverage_range": (25, 50), "initial_margin": Decimal("0.125"), "maintenance_margin": Decimal("0.06")},
            {"leverage_range": (50, 100), "initial_margin": Decimal("0.25"), "maintenance_margin": Decimal("0.125")},
        ],
        # Default for other symbols
        "DEFAULT": [
            {"leverage_range": (1, 10), "initial_margin": Decimal("0.10"), "maintenance_margin": Decimal("0.05")},
            {"leverage_range": (10, 20), "initial_margin": Decimal("0.125"), "maintenance_margin": Decimal("0.06")},
            {"leverage_range": (20, 50), "initial_margin": Decimal("0.20"), "maintenance_margin": Decimal("0.10")},
        ]
    }


@dataclass
class BybitTrade(BacktestTrade):
    """Enhanced trade record with Bybit-specific data."""
    
    # Bybit-specific fields
    funding_cost: Decimal = Decimal('0')
    liquidation_price: Optional[Decimal] = None
    margin_used: Decimal = Decimal('0')
    leverage: Decimal = Decimal('1')
    contract_type: BybitContractType = BybitContractType.LINEAR_PERPETUAL
    vip_tier: BybitVIPTier = BybitVIPTier.NO_VIP
    maker_order: bool = False  # True if order was maker, False if taker
    
    # Market impact and execution quality
    market_impact: Decimal = Decimal('0')
    execution_delay_ms: int = 0
    order_book_depth: Decimal = Decimal('0')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Bybit-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'funding_cost': float(self.funding_cost),
            'liquidation_price': float(self.liquidation_price) if self.liquidation_price else None,
            'margin_used': float(self.margin_used),
            'leverage': float(self.leverage),
            'contract_type': self.contract_type.value,
            'vip_tier': self.vip_tier.value,
            'maker_order': self.maker_order,
            'market_impact': float(self.market_impact),
            'execution_delay_ms': self.execution_delay_ms,
            'order_book_depth': float(self.order_book_depth)
        })
        return base_dict


@dataclass
class BybitBacktestResults(BacktestResults):
    """Enhanced backtest results with Bybit-specific metrics."""
    
    # Bybit-specific metrics
    total_funding_cost: Decimal = Decimal('0')
    funding_impact_pct: Decimal = Decimal('0')
    liquidation_events: int = 0
    average_leverage: Decimal = Decimal('1')
    max_leverage_used: Decimal = Decimal('1')
    margin_utilization_pct: Decimal = Decimal('0')
    
    # Fee analysis
    maker_fee_total: Decimal = Decimal('0')
    taker_fee_total: Decimal = Decimal('0')
    maker_rebate_total: Decimal = Decimal('0')
    fee_savings_from_vip: Decimal = Decimal('0')
    
    # Market impact metrics
    average_market_impact: Decimal = Decimal('0')
    total_market_impact: Decimal = Decimal('0')
    average_execution_delay_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Bybit-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'total_funding_cost': float(self.total_funding_cost),
            'funding_impact_pct': float(self.funding_impact_pct),
            'liquidation_events': self.liquidation_events,
            'average_leverage': float(self.average_leverage),
            'max_leverage_used': float(self.max_leverage_used),
            'margin_utilization_pct': float(self.margin_utilization_pct),
            'maker_fee_total': float(self.maker_fee_total),
            'taker_fee_total': float(self.taker_fee_total),
            'maker_rebate_total': float(self.maker_rebate_total),
            'fee_savings_from_vip': float(self.fee_savings_from_vip),
            'average_market_impact': float(self.average_market_impact),
            'total_market_impact': float(self.total_market_impact),
            'average_execution_delay_ms': self.average_execution_delay_ms
        })
        return base_dict


class BybitEnhancedBacktestEngine(BacktestEngine):
    """
    Enhanced backtesting engine specifically designed for Bybit trading.
    
    This engine provides:
    1. Bybit-specific fee calculation with VIP tier support
    2. Funding rate impact on perpetual positions
    3. Liquidation risk modeling and margin requirements
    4. Enhanced market impact and execution simulation
    5. Historical data integration with funding rates
    6. Realistic order execution with latency modeling
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        bybit_client: Optional[BybitClient] = None,
        historical_data_manager: Optional[HistoricalDataManager] = None,
        initial_balance: Decimal = Decimal('10000'),
        vip_tier: BybitVIPTier = BybitVIPTier.NO_VIP,
        default_leverage: Decimal = Decimal('10'),
        contract_type: BybitContractType = BybitContractType.LINEAR_PERPETUAL
    ):
        super().__init__(config_manager, initial_balance)
        
        # Bybit-specific configuration
        self.vip_tier = vip_tier
        self.default_leverage = default_leverage
        self.contract_type = contract_type
        
        # Enhanced components
        self.bybit_client = bybit_client
        self.historical_data_manager = historical_data_manager
        
        # Fee structures
        self.fee_structure = BybitFeeStructure()
        self.margin_requirements = BybitMarginRequirements()
        
        # Enhanced state tracking
        self.funding_rates: Dict[str, pd.DataFrame] = {}
        self.liquidation_history: List[Dict] = []
        self.margin_usage: Dict[str, Decimal] = {}
        
        # Market impact modeling
        self.order_book_depth_history: Dict[str, List[Decimal]] = {}
        self.execution_delay_model = self._initialize_delay_model()
        
        # Enhanced logging
        self.logger = TradingLogger("BybitEnhancedBacktestEngine")
        self.logger.info(f"Initialized with VIP tier: {vip_tier.value}, Leverage: {default_leverage}x")
    
    def _initialize_delay_model(self) -> Dict[str, Any]:
        """Initialize execution delay modeling parameters."""
        return {
            'base_delay_ms': 50,        # Base execution delay
            'volatility_factor': 2.0,   # Delay increases with volatility
            'size_factor': 1.5,         # Delay increases with order size
            'liquidity_factor': 1.8,    # Delay decreases with liquidity
            'max_delay_ms': 500         # Maximum execution delay
        }
    
    async def run_enhanced_backtest(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h"
    ) -> BybitBacktestResults:
        """
        Run enhanced backtest with Bybit-specific features.
        
        Args:
            strategy: Strategy to test
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Data timeframe
            
        Returns:
            BybitBacktestResults with enhanced metrics
        """
        try:
            self.logger.info(f"Starting Bybit enhanced backtest: {strategy.strategy_id}")
            self.logger.info(f"Symbol: {symbol}, Period: {start_date} to {end_date}")
            self.logger.info(f"VIP Tier: {self.vip_tier.value}, Leverage: {self.default_leverage}x")
            
            # Reset state
            self._reset_enhanced_state()
            
            # Fetch enhanced historical data
            enhanced_data = await self._fetch_enhanced_data(symbol, start_date, end_date, timeframe)
            if enhanced_data is None:
                raise ValueError("Failed to fetch enhanced historical data")
            
            # Merge OHLCV with funding data
            backtest_data = enhanced_data.merge_funding_data()
            
            self.logger.info(f"Enhanced data loaded: {len(backtest_data)} rows with funding rates")
            
            # Store funding rates for calculations
            if enhanced_data.funding_data is not None:
                self.funding_rates[symbol] = enhanced_data.funding_data
            
            # Initialize strategy
            await strategy.on_start()
            
            # Main enhanced backtest loop
            for timestamp, row in backtest_data.iterrows():
                await self._process_enhanced_timestamp(strategy, symbol, timestamp, row, backtest_data)
            
            # Close remaining positions
            await self._close_all_enhanced_positions(symbol, backtest_data.iloc[-1])
            
            # Stop strategy
            await strategy.on_stop()
            
            # Calculate enhanced results
            results = await self._calculate_enhanced_results(symbol, start_date, end_date)
            
            self.logger.info(f"Enhanced backtest completed:")
            self.logger.info(f"  Final Balance: ${results.final_balance:,.2f}")
            self.logger.info(f"  Total Return: {results.total_return_pct:.2f}%")
            self.logger.info(f"  Funding Cost: ${results.total_funding_cost:.2f}")
            self.logger.info(f"  Liquidations: {results.liquidation_events}")
            self.logger.info(f"  Avg Leverage: {results.average_leverage:.1f}x")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced backtest: {e}")
            raise
    
    async def _fetch_enhanced_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Optional[EnhancedMarketData]:
        """Fetch enhanced historical data with funding rates."""
        try:
            if self.historical_data_manager is None:
                self.logger.warning("No HistoricalDataManager available, using basic data")
                return None
            
            # Create data fetch request
            request = DataFetchRequest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                include_funding_rates=True,
                include_quality_assessment=True
            )
            
            # Fetch enhanced data
            enhanced_data = await self.historical_data_manager.fetch_historical_data(request)
            
            if enhanced_data and enhanced_data.quality_metrics:
                self.logger.info(f"Data quality score: {enhanced_data.quality_metrics.quality_score:.3f}")
                
                if enhanced_data.quality_metrics.quality_score < 0.9:
                    self.logger.warning("Data quality below threshold - results may be less reliable")
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Error fetching enhanced data: {e}")
            return None
    
    async def _process_enhanced_timestamp(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timestamp: datetime,
        current_data: pd.Series,
        full_data: pd.DataFrame
    ) -> None:
        """Process timestamp with Bybit-specific enhancements."""
        try:
            # Update positions with current prices and funding
            await self._update_enhanced_positions(symbol, current_data, timestamp)
            
            # Check liquidation risk
            await self._check_liquidation_risk(symbol, current_data, timestamp)
            
            # Record daily balance including funding costs
            if len(self.daily_balances) == 0 or timestamp.date() != self.daily_balances[-1][0].date():
                self.daily_balances.append((timestamp, self.current_balance))
            
            # Generate trading signal (same as base engine)
            current_idx = full_data.index.get_loc(timestamp) 
            lookback_periods = getattr(strategy, 'lookback_periods', 100)
            start_idx = max(0, current_idx - lookback_periods + 1)
            
            strategy_data = full_data.iloc[start_idx:current_idx + 1]
            signal = await strategy.generate_signal(symbol, strategy_data)
            
            if signal:
                await self._execute_enhanced_signal(signal, current_data, timestamp)
                
        except Exception as e:
            self.logger.error(f"Error processing enhanced timestamp {timestamp}: {e}")
    
    async def _execute_enhanced_signal(
        self,
        signal: TradingSignal,
        market_data: pd.Series,
        timestamp: datetime
    ) -> None:
        """Execute signal with Bybit-specific enhancements."""
        try:
            symbol = signal.symbol
            current_price = Decimal(str(market_data.get('close', market_data['close'])))
            
            # Enhanced execution based on signal type
            if signal.signal_type == SignalType.BUY:
                await self._execute_enhanced_buy(signal, current_price, timestamp, market_data)
            elif signal.signal_type == SignalType.SELL:
                await self._execute_enhanced_sell(signal, current_price, timestamp, market_data)
            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                await self._close_enhanced_position(symbol, current_price, timestamp, market_data)
                
        except Exception as e:
            self.logger.error(f"Error executing enhanced signal: {e}")
    
    async def _execute_enhanced_buy(
        self,
        signal: TradingSignal,
        current_price: Decimal,
        timestamp: datetime,
        market_data: pd.Series
    ) -> None:
        """Execute buy signal with Bybit-specific features."""
        try:
            symbol = signal.symbol
            
            # Calculate position size with leverage
            leverage = getattr(signal, 'leverage', self.default_leverage)
            leverage = min(leverage, self._get_max_leverage(symbol, current_price))
            
            # Risk assessment
            if self.risk_manager:
                risk_assessment = await self.risk_manager.assess_trade_risk(
                    symbol=symbol,
                    side="buy",
                    entry_price=current_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    current_balance=self.current_balance,
                    strategy_id=signal.strategy_id
                )
                
                if not risk_assessment.is_approved:
                    self.logger.debug(f"Buy signal rejected: {risk_assessment.risk_reason}")
                    return
                
                position_value = risk_assessment.position_size * current_price
            else:
                # Default: 5% of balance with leverage
                position_value = self.current_balance * Decimal('0.05') * leverage
            
            # Calculate quantity and margin
            quantity = position_value / current_price
            margin_required = position_value / leverage
            
            # Check margin availability
            if margin_required > self.current_balance:
                self.logger.debug(f"Insufficient margin for buy: {margin_required} > {self.current_balance}")
                return
            
            # Enhanced execution with market impact
            execution_details = await self._simulate_enhanced_execution(
                symbol, "buy", quantity, current_price, market_data, timestamp
            )
            
            if not execution_details['success']:
                return
            
            execution_price = execution_details['execution_price']
            commission = execution_details['commission']
            is_maker = execution_details['is_maker']
            market_impact = execution_details['market_impact']
            execution_delay = execution_details['execution_delay_ms']
            
            # Calculate liquidation price
            liquidation_price = self._calculate_liquidation_price(
                side='long',
                entry_price=execution_price,
                leverage=leverage,
                symbol=symbol
            )
            
            # Reserve margin
            self.current_balance -= margin_required
            self.margin_usage[symbol] = self.margin_usage.get(symbol, Decimal('0')) + margin_required
            
            # Update or create position
            if symbol in self.positions:
                # Average into existing position
                existing = self.positions[symbol]
                total_quantity = existing['quantity'] + quantity
                total_value = (existing['quantity'] * existing['entry_price'] + 
                              quantity * execution_price)
                avg_price = total_value / total_quantity
                
                # Update margin and liquidation price
                total_margin = existing.get('margin_used', Decimal('0')) + margin_required
                new_leverage = (total_quantity * avg_price) / total_margin
                new_liquidation = self._calculate_liquidation_price(
                    side='long', entry_price=avg_price, leverage=new_leverage, symbol=symbol
                )
                
                self.positions[symbol].update({
                    'quantity': total_quantity,
                    'entry_price': avg_price,
                    'margin_used': total_margin,
                    'leverage': new_leverage,
                    'liquidation_price': new_liquidation
                })
            else:
                # New position
                self.positions[symbol] = {
                    'side': 'long',
                    'quantity': quantity,
                    'entry_price': execution_price,
                    'entry_timestamp': timestamp,
                    'margin_used': margin_required,
                    'leverage': leverage,
                    'liquidation_price': liquidation_price,
                    'unrealized_pnl': Decimal('0'),
                    'funding_cost': Decimal('0')
                }
            
            # Create enhanced trade record
            trade = BybitTrade(
                timestamp=timestamp,
                symbol=symbol,
                side='buy',
                quantity=quantity,
                entry_price=execution_price,
                commission=commission,
                slippage=execution_price - current_price,
                strategy_id=signal.strategy_id,
                # Bybit-specific fields
                margin_used=margin_required,
                leverage=leverage,
                liquidation_price=liquidation_price,
                contract_type=self.contract_type,
                vip_tier=self.vip_tier,
                maker_order=is_maker,
                market_impact=market_impact,
                execution_delay_ms=execution_delay,
                order_book_depth=execution_details.get('order_book_depth', Decimal('0'))
            )
            self.trades.append(trade)
            
            self.logger.debug(
                f"Enhanced buy executed: {symbol} {quantity:.6f} @ {execution_price:.2f}, "
                f"Leverage: {leverage}x, Liquidation: {liquidation_price:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing enhanced buy: {e}")
    
    async def _simulate_enhanced_execution(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        market_data: pd.Series,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Simulate realistic order execution with Bybit-specific features."""
        try:
            # Base execution simulation
            base_slippage = self.slippage_rate
            
            # Enhanced slippage based on order size and market conditions
            volatility = self._calculate_volatility(market_data)
            volume = Decimal(str(market_data.get('volume', 1000000)))  # Default volume
            
            # Order size impact
            size_impact = min(quantity * price / volume * Decimal('10'), Decimal('0.005'))
            
            # Volatility impact
            volatility_impact = volatility * Decimal('0.1')
            
            # Combined slippage
            total_slippage = base_slippage + size_impact + volatility_impact
            total_slippage = min(total_slippage, self.max_slippage)
            
            # Execution price with slippage
            if side == "buy":
                execution_price = price * (Decimal('1') + total_slippage)
            else:
                execution_price = price * (Decimal('1') - total_slippage)
            
            # Determine if order is maker or taker (simplified model)
            # In reality, this depends on order type and market conditions
            is_maker = self._simulate_maker_probability(total_slippage)
            
            # Calculate commission
            commission = self._calculate_bybit_commission(
                quantity, execution_price, is_maker
            )
            
            # Simulate execution delay
            execution_delay = self._simulate_execution_delay(
                quantity, price, volatility, volume
            )
            
            # Market impact calculation
            market_impact = size_impact * price
            
            return {
                'success': True,
                'execution_price': execution_price,
                'commission': commission,
                'is_maker': is_maker,
                'market_impact': market_impact,
                'execution_delay_ms': execution_delay,
                'order_book_depth': volume * price  # Simplified depth measure
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating execution: {e}")
            return {'success': False}
    
    def _calculate_bybit_commission(
        self,
        quantity: Decimal,
        price: Decimal,
        is_maker: bool
    ) -> Decimal:
        """Calculate Bybit commission based on VIP tier and maker/taker."""
        try:
            notional = quantity * price
            
            # Get fee rate based on contract type and VIP tier
            if self.contract_type == BybitContractType.LINEAR_PERPETUAL:
                fee_rates = self.fee_structure.LINEAR_PERP_FEES[self.vip_tier]
            else:
                fee_rates = self.fee_structure.SPOT_FEES[self.vip_tier]
            
            # Apply maker or taker fee
            fee_rate = fee_rates['maker'] if is_maker else fee_rates['taker']
            commission = notional * fee_rate
            
            return commission
            
        except Exception as e:
            self.logger.error(f"Error calculating commission: {e}")
            return Decimal('0')
    
    def _calculate_liquidation_price(
        self,
        side: str,
        entry_price: Decimal,
        leverage: Decimal,
        symbol: str
    ) -> Decimal:
        """Calculate liquidation price for Bybit perpetual positions."""
        try:
            # Get maintenance margin rate for symbol
            margin_tiers = self.margin_requirements.INITIAL_MARGIN_TIERS.get(
                symbol, self.margin_requirements.INITIAL_MARGIN_TIERS["DEFAULT"]
            )
            
            # Find appropriate tier based on leverage
            maintenance_margin_rate = Decimal('0.05')  # Default
            for tier in margin_tiers:
                if tier['leverage_range'][0] <= leverage <= tier['leverage_range'][1]:
                    maintenance_margin_rate = tier['maintenance_margin']
                    break
            
            if side == 'long':
                # Long liquidation: entry_price * (1 - 1/leverage + maintenance_margin_rate)
                liquidation_price = entry_price * (
                    Decimal('1') - Decimal('1') / leverage + maintenance_margin_rate
                )
            else:
                # Short liquidation: entry_price * (1 + 1/leverage - maintenance_margin_rate)
                liquidation_price = entry_price * (
                    Decimal('1') + Decimal('1') / leverage - maintenance_margin_rate
                )
            
            return liquidation_price
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidation price: {e}")
            return Decimal('0')
    
    def _get_max_leverage(self, symbol: str, price: Decimal) -> Decimal:
        """Get maximum allowed leverage for symbol."""
        try:
            # This would normally depend on position size and Bybit's risk limits
            # For backtesting, we'll use simplified tiers
            
            if symbol == "BTCUSDT":
                return Decimal('100')  # BTCUSDT allows up to 100x
            elif symbol.endswith("USDT"):
                return Decimal('50')   # Most USDT pairs allow 50x
            else:
                return Decimal('25')   # Conservative default
                
        except Exception:
            return Decimal('10')
    
    def _calculate_volatility(self, market_data: pd.Series) -> Decimal:
        """Calculate recent volatility for market impact modeling."""
        try:
            # Simple volatility calculation using high-low range
            high = Decimal(str(market_data.get('high', market_data.get('close', 0))))
            low = Decimal(str(market_data.get('low', market_data.get('close', 0))))
            close = Decimal(str(market_data.get('close', 0)))
            
            if close > 0:
                volatility = (high - low) / close
                return volatility
            
            return Decimal('0.01')  # Default volatility
            
        except Exception:
            return Decimal('0.01')
    
    def _simulate_maker_probability(self, slippage: Decimal) -> bool:
        """Simulate probability of order being maker vs taker."""
        try:
            # Higher slippage suggests more urgent execution (taker)
            # Lower slippage suggests patient execution (maker)
            import random
            
            # Base maker probability of 30%
            base_maker_prob = 0.3
            
            # Adjust based on slippage
            slippage_factor = float(slippage) * 1000  # Convert to basis points
            adjusted_prob = max(0.1, base_maker_prob - slippage_factor * 0.1)
            
            return random.random() < adjusted_prob
            
        except Exception:
            return False  # Default to taker
    
    def _simulate_execution_delay(
        self,
        quantity: Decimal,
        price: Decimal,
        volatility: Decimal,
        volume: Decimal
    ) -> int:
        """Simulate realistic execution delay in milliseconds."""
        try:
            import random
            
            base_delay = self.execution_delay_model['base_delay_ms']
            
            # Size factor
            notional = quantity * price
            size_factor = min(float(notional) / 10000, 5.0)  # Larger orders take longer
            
            # Volatility factor
            vol_factor = min(float(volatility) * 100, 3.0)
            
            # Liquidity factor (inverse relationship)
            liquidity_factor = max(0.5, 10000 / max(float(volume), 1))
            
            # Calculate total delay
            total_delay = (
                base_delay * 
                (1 + size_factor * self.execution_delay_model['size_factor'] / 10) *
                (1 + vol_factor * self.execution_delay_model['volatility_factor'] / 10) *
                (liquidity_factor * self.execution_delay_model['liquidity_factor'] / 10)
            )
            
            # Add random component and cap
            random_factor = random.uniform(0.8, 1.2)
            final_delay = min(
                int(total_delay * random_factor),
                self.execution_delay_model['max_delay_ms']
            )
            
            return final_delay
            
        except Exception:
            return 50  # Default delay
    
    def _reset_enhanced_state(self) -> None:
        """Reset enhanced backtest state."""
        super()._reset_state()
        self.funding_rates.clear()
        self.liquidation_history.clear()
        self.margin_usage.clear()
        self.order_book_depth_history.clear()
    
    async def _update_enhanced_positions(
        self,
        symbol: str,
        market_data: pd.Series,
        timestamp: datetime
    ) -> None:
        """Update positions with funding costs and enhanced tracking."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            current_price = Decimal(str(market_data.get('close', market_data['close'])))
            
            # Update unrealized PnL
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            if position['side'] == 'long':
                unrealized_pnl = quantity * (current_price - entry_price)
            else:
                unrealized_pnl = quantity * (entry_price - current_price)
            
            position['unrealized_pnl'] = unrealized_pnl
            
            # Calculate and apply funding costs
            if self.contract_type == BybitContractType.LINEAR_PERPETUAL:
                funding_cost = await self._calculate_funding_cost(
                    symbol, position, timestamp, market_data
                )
                position['funding_cost'] = position.get('funding_cost', Decimal('0')) + funding_cost
                
                # Apply funding cost to balance (funding is paid/received every 8 hours)
                if funding_cost != 0:
                    self.current_balance -= funding_cost
                    
                    # Track funding costs in trades
                    for trade in reversed(self.trades):
                        if (trade.symbol == symbol and 
                            trade.exit_price is None and
                            isinstance(trade, BybitTrade)):
                            trade.funding_cost += funding_cost
                            break
            
        except Exception as e:
            self.logger.error(f"Error updating enhanced positions: {e}")
    
    async def _calculate_funding_cost(
        self,
        symbol: str,
        position: Dict,
        timestamp: datetime,
        market_data: pd.Series
    ) -> Decimal:
        """Calculate funding cost for perpetual positions."""
        try:
            # Check if it's a funding time (every 8 hours: 00:00, 08:00, 16:00 UTC)
            if timestamp.hour not in [0, 8, 16] or timestamp.minute != 0:
                return Decimal('0')
            
            # Get funding rate for this timestamp
            if symbol not in self.funding_rates:
                return Decimal('0')
            
            funding_data = self.funding_rates[symbol]
            
            # Find closest funding rate
            closest_rate = None
            min_time_diff = timedelta.max
            
            for _, rate_row in funding_data.iterrows():
                rate_time = rate_row.name if hasattr(rate_row, 'name') else rate_row['timestamp']
                time_diff = abs(timestamp - rate_time)
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_rate = rate_row['funding_rate']
            
            if closest_rate is None:
                return Decimal('0')
            
            # Calculate funding cost
            # Funding cost = position_size * mark_price * funding_rate
            position_size = position['quantity']
            mark_price = Decimal(str(market_data.get('close', market_data['close'])))
            funding_rate = Decimal(str(closest_rate))
            
            if position['side'] == 'long':
                funding_cost = position_size * mark_price * funding_rate
            else:  # short position receives funding when rate is positive
                funding_cost = -position_size * mark_price * funding_rate
            
            return funding_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating funding cost: {e}")
            return Decimal('0')
    
    async def _check_liquidation_risk(
        self,
        symbol: str,
        market_data: pd.Series,
        timestamp: datetime
    ) -> None:
        """Check for liquidation events."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            current_price = Decimal(str(market_data.get('close', market_data['close'])))
            liquidation_price = position.get('liquidation_price')
            
            if liquidation_price is None:
                return
            
            # Check liquidation condition
            is_liquidated = False
            
            if position['side'] == 'long' and current_price <= liquidation_price:
                is_liquidated = True
            elif position['side'] == 'short' and current_price >= liquidation_price:
                is_liquidated = True
            
            if is_liquidated:
                await self._execute_liquidation(symbol, current_price, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error checking liquidation risk: {e}")
    
    async def _execute_liquidation(
        self,
        symbol: str,
        liquidation_price: Decimal,
        timestamp: datetime
    ) -> None:
        """Execute liquidation of position."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # Liquidation penalty (typically 0.5% of position value)
            liquidation_penalty = position['quantity'] * liquidation_price * Decimal('0.005')
            
            # Close position at liquidation price
            margin_returned = position.get('margin_used', Decimal('0')) - liquidation_penalty
            margin_returned = max(margin_returned, Decimal('0'))  # Can't be negative
            
            self.current_balance += margin_returned
            self.margin_usage[symbol] = Decimal('0')
            
            # Record liquidation
            liquidation_event = {
                'symbol': symbol,
                'timestamp': timestamp,
                'liquidation_price': liquidation_price,
                'position_size': position['quantity'],
                'margin_lost': position.get('margin_used', Decimal('0')) - margin_returned,  
                'side': position['side']
            }
            self.liquidation_history.append(liquidation_event)
            
            # Update trade record
            for trade in reversed(self.trades):
                if (trade.symbol == symbol and 
                    trade.exit_price is None and
                    isinstance(trade, BybitTrade)):
                    
                    trade.exit_price = liquidation_price
                    trade.exit_timestamp = timestamp
                    trade.pnl = -trade.margin_used  # Total loss of margin
                    trade.duration_hours = (timestamp - trade.timestamp).total_seconds() / 3600
                    break
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.warning(
                f"LIQUIDATION: {symbol} at {liquidation_price:.2f}, "
                f"Margin lost: ${liquidation_penalty:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing liquidation: {e}")
    
    # Additional methods for sell signals and position closing
    async def _execute_enhanced_sell(
        self,
        signal: TradingSignal,
        current_price: Decimal,
        timestamp: datetime,
        market_data: pd.Series
    ) -> None:
        """Execute sell signal - close long position."""
        await self._close_enhanced_position(signal.symbol, current_price, timestamp, market_data)
    
    async def _close_enhanced_position(
        self,
        symbol: str,
        current_price: Decimal,
        timestamp: datetime,
        market_data: pd.Series  
    ) -> None:
        """Close position with enhanced Bybit features."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # Enhanced execution simulation
            execution_details = await self._simulate_enhanced_execution(
                symbol, "sell", position['quantity'], current_price, market_data, timestamp
            )
            
            if not execution_details['success']:
                return
            
            execution_price = execution_details['execution_price']
            commission = execution_details['commission']
            
            # Calculate PnL including funding costs
            entry_price = position['entry_price']
            quantity = position['quantity']
            margin_used = position.get('margin_used', Decimal('0'))
            funding_cost = position.get('funding_cost', Decimal('0'))
            
            if position['side'] == 'long':
                gross_pnl = quantity * (execution_price - entry_price)
            else:
                gross_pnl = quantity * (entry_price - execution_price)
            
            net_pnl = gross_pnl - commission - funding_cost
            
            # Return margin plus PnL
            self.current_balance += margin_used + net_pnl
            self.margin_usage[symbol] = Decimal('0')
            
            # Update trade record
            duration = timestamp - position['entry_timestamp']
            
            for trade in reversed(self.trades):
                if (trade.symbol == symbol and 
                    trade.exit_price is None and
                    isinstance(trade, BybitTrade)):
                    
                    trade.exit_price = execution_price
                    trade.exit_timestamp = timestamp
                    trade.pnl = net_pnl
                    trade.commission += commission
                    trade.funding_cost = funding_cost
                    trade.duration_hours = duration.total_seconds() / 3600
                    break
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.debug(
                f"Enhanced position closed: {symbol} @ {execution_price:.2f}, "
                f"Net PnL: ${net_pnl:.2f} (Funding: ${funding_cost:.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error closing enhanced position: {e}")
    
    async def _close_all_enhanced_positions(
        self,
        symbol: str,
        final_data: pd.Series
    ) -> None:
        """Close all remaining positions at backtest end."""
        try:
            symbols_to_close = list(self.positions.keys())
            final_timestamp = final_data.name if hasattr(final_data, 'name') else datetime.now()
            
            for pos_symbol in symbols_to_close:
                current_price = Decimal(str(final_data.get('close', final_data['close'])))
                await self._close_enhanced_position(pos_symbol, current_price, final_timestamp, final_data)
                
        except Exception as e:
            self.logger.error(f"Error closing all enhanced positions: {e}")
    
    async def _calculate_enhanced_results(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> BybitBacktestResults:
        """Calculate enhanced backtest results with Bybit-specific metrics."""
        try:
            # Get base results
            base_results = await self._calculate_results(start_date, end_date)
            
            # Calculate Bybit-specific metrics
            bybit_trades = [t for t in self.trades if isinstance(t, BybitTrade)]
            
            # Funding costs
            total_funding_cost = sum(t.funding_cost for t in bybit_trades)
            funding_impact_pct = (total_funding_cost / self.initial_balance) * 100
            
            # Liquidation events
            liquidation_events = len(self.liquidation_history)
            
            # Leverage metrics
            leverages = [t.leverage for t in bybit_trades if t.leverage > 0]
            average_leverage = sum(leverages) / len(leverages) if leverages else Decimal('1')
            max_leverage_used = max(leverages) if leverages else Decimal('1')
            
            # Margin utilization
            max_margin_used = max(self.margin_usage.values()) if self.margin_usage.values() else Decimal('0')
            margin_utilization_pct = (max_margin_used / self.initial_balance) * 100
            
            # Fee analysis
            maker_trades = [t for t in bybit_trades if t.maker_order]
            taker_trades = [t for t in bybit_trades if not t.maker_order]
            
            maker_fee_total = sum(t.commission for t in maker_trades)
            taker_fee_total = sum(t.commission for t in taker_trades)
            maker_rebate_total = sum(abs(t.commission) for t in maker_trades if t.commission < 0)
            
            # Calculate VIP savings (compare to no-VIP fees)
            fee_savings_from_vip = self._calculate_vip_savings(bybit_trades)
            
            # Market impact
            market_impacts = [t.market_impact for t in bybit_trades if t.market_impact > 0]
            average_market_impact = sum(market_impacts) / len(market_impacts) if market_impacts else Decimal('0')
            total_market_impact = sum(market_impacts)
            
            # Execution delays
            delays = [t.execution_delay_ms for t in bybit_trades if t.execution_delay_ms > 0]
            average_execution_delay_ms = int(sum(delays) / len(delays)) if delays else 0
            
            # Create enhanced results
            enhanced_results = BybitBacktestResults(
                # Base results
                start_date=base_results.start_date,
                end_date=base_results.end_date,
                initial_balance=base_results.initial_balance,
                final_balance=base_results.final_balance,
                total_return=base_results.total_return,
                total_return_pct=base_results.total_return_pct,
                annual_return_pct=base_results.annual_return_pct,
                max_drawdown_pct=base_results.max_drawdown_pct,
                sharpe_ratio=base_results.sharpe_ratio,
                sortino_ratio=base_results.sortino_ratio,
                calmar_ratio=base_results.calmar_ratio,
                total_trades=base_results.total_trades,
                winning_trades=base_results.winning_trades,
                losing_trades=base_results.losing_trades,
                win_rate_pct=base_results.win_rate_pct,
                profit_factor=base_results.profit_factor,
                average_win=base_results.average_win,
                average_loss=base_results.average_loss,
                largest_win=base_results.largest_win,
                largest_loss=base_results.largest_loss,
                average_trade_duration=base_results.average_trade_duration,
                total_commission=base_results.total_commission,
                total_slippage=base_results.total_slippage,
                daily_returns=base_results.daily_returns,
                equity_curve=base_results.equity_curve,
                drawdown_series=base_results.drawdown_series,
                trades=base_results.trades,
                
                # Bybit-specific results
                total_funding_cost=total_funding_cost,
                funding_impact_pct=Decimal(str(funding_impact_pct)),
                liquidation_events=liquidation_events,
                average_leverage=average_leverage,
                max_leverage_used=max_leverage_used,
                margin_utilization_pct=Decimal(str(margin_utilization_pct)),
                maker_fee_total=maker_fee_total,
                taker_fee_total=taker_fee_total,
                maker_rebate_total=maker_rebate_total,
                fee_savings_from_vip=fee_savings_from_vip,
                average_market_impact=average_market_impact,
                total_market_impact=total_market_impact,
                average_execution_delay_ms=average_execution_delay_ms
            )
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced results: {e}")
            raise
    
    def _calculate_vip_savings(self, bybit_trades: List[BybitTrade]) -> Decimal:
        """Calculate fee savings from VIP tier vs no-VIP."""
        try:
            if self.vip_tier == BybitVIPTier.NO_VIP:
                return Decimal('0')
            
            total_savings = Decimal('0')
            no_vip_fees = self.fee_structure.LINEAR_PERP_FEES[BybitVIPTier.NO_VIP]
            current_vip_fees = self.fee_structure.LINEAR_PERP_FEES[self.vip_tier]
            
            for trade in bybit_trades:
                notional = trade.quantity * trade.entry_price
                
                if trade.maker_order:
                    no_vip_fee = notional * no_vip_fees['maker']
                    current_fee = notional * current_vip_fees['maker']
                else:
                    no_vip_fee = notional * no_vip_fees['taker']
                    current_fee = notional * current_vip_fees['taker']
                
                savings = no_vip_fee - current_fee
                total_savings += savings
            
            return total_savings
            
        except Exception as e:
            self.logger.error(f"Error calculating VIP savings: {e}")
            return Decimal('0')
    
    def generate_enhanced_report(
        self,
        results: BybitBacktestResults,
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive enhanced backtest report."""
        try:
            base_report = self.generate_report(results, None)
            
            enhanced_section = f"""

## Bybit-Specific Metrics

### Funding & Financing
- **Total Funding Cost**: ${results.total_funding_cost:.2f}
- **Funding Impact on Returns**: {results.funding_impact_pct:.2f}%

### Leverage & Risk
- **Average Leverage Used**: {results.average_leverage:.1f}x
- **Maximum Leverage Used**: {results.max_leverage_used:.1f}x  
- **Margin Utilization**: {results.margin_utilization_pct:.1f}%
- **Liquidation Events**: {results.liquidation_events}

### Fee Analysis
- **Maker Fees Paid**: ${results.maker_fee_total:.2f}
- **Taker Fees Paid**: ${results.taker_fee_total:.2f}
- **Maker Rebates Received**: ${results.maker_rebate_total:.2f}
- **VIP Tier Savings**: ${results.fee_savings_from_vip:.2f}
- **Current VIP Tier**: {self.vip_tier.value}

### Execution Quality
- **Average Market Impact**: ${results.average_market_impact:.4f}
- **Total Market Impact**: ${results.total_market_impact:.2f}
- **Average Execution Delay**: {results.average_execution_delay_ms}ms

### Risk Events
- **Liquidations**: {len(self.liquidation_history)}
"""
            
            if self.liquidation_history:
                enhanced_section += "\n### Liquidation Details\n"
                for i, liq in enumerate(self.liquidation_history, 1):
                    enhanced_section += f"- **Event {i}**: {liq['symbol']} at ${liq['liquidation_price']:.2f} ({liq['timestamp']})\n"
            
            enhanced_section += f"""

### Performance vs Traditional Backtesting
- **Funding Cost Impact**: Realistic perpetual swap costs included
- **Liquidation Risk**: {results.liquidation_events} events vs 0 in simple backtesting
- **Enhanced Slippage**: Market impact and depth considered
- **Bybit Fee Structure**: VIP tier {self.vip_tier.value} applied
"""
            
            full_report = base_report + enhanced_section
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(full_report)
                self.logger.info(f"Enhanced report saved to: {save_path}")
            
            return full_report
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced report: {e}")
            return f"Error generating enhanced report: {e}"