"""
Integration Coordinator
Main coordination hub that orchestrates all components of the Australian trading system
Manages the complete trading lifecycle from signal generation to execution and monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import traceback

# Core trading components
from ..bot import TradingBot
from ..exchanges.bybit_client import BybitClient

# Australian compliance components
from ..australian_compliance.ato_integration import AustralianTaxCalculator
from ..australian_compliance.banking_manager import AustralianBankingManager
from ..australian_compliance.regulatory_compliance import AustralianComplianceManager

# ML and arbitrage engines
from ..ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine, StrategySignal
from ..ml_strategy_discovery.data_infrastructure import MultiExchangeDataCollector
from ..arbitrage_engine.arbitrage_detector import OpportunisticArbitrageEngine, ArbitrageOpportunity
from ..arbitrage_engine.execution_engine import ArbitrageExecutionEngine

# Risk management
from ..risk_management.australian_risk_manager import AustralianRiskCalculator
from ..risk_management.portfolio_risk_controller import PortfolioRiskController, PortfolioState

# Trading engine integration
from .australian_trading_engine import AustralianTradingEngineIntegration
from .signal_processing_manager import SignalProcessingManager

logger = logging.getLogger(__name__)

@dataclass
class SystemConfiguration:
    """Configuration for the Australian trading system"""
    
    # Trading allocation (must sum to 100%)
    ml_strategy_allocation_pct: int = 70      # 70% to ML strategies
    arbitrage_allocation_pct: int = 20        # 20% to arbitrage
    cash_reserve_pct: int = 10               # 10% cash reserve
    
    # Portfolio limits
    max_portfolio_value_aud: Decimal = Decimal('1000000')  # $1M AUD max
    min_trade_size_aud: Decimal = Decimal('100')           # $100 AUD min trade
    max_single_position_pct: int = 20                      # 20% max single position
    
    # Risk management
    daily_loss_limit_pct: int = 5             # 5% daily loss limit
    max_drawdown_pct: int = 15               # 15% max drawdown
    risk_monitoring_interval_minutes: int = 5 # 5-minute risk checks
    
    # Signal processing
    signal_processing_interval_seconds: int = 30    # 30-second signal processing
    max_concurrent_trades: int = 5                  # 5 concurrent trades max
    signal_confidence_threshold: float = 0.6       # 60% minimum confidence
    
    # Compliance
    ato_reporting_threshold_aud: Decimal = Decimal('10000')  # $10k ATO threshold
    professional_trader_threshold_trades: int = 40          # 40 trades/year threshold
    daily_volume_limit_aud: Decimal = Decimal('100000')     # $100k daily limit
    
    # Exchange preferences
    australian_exchanges: List[str] = field(default_factory=lambda: ['btcmarkets', 'coinjar', 'swyftx'])
    international_exchanges: List[str] = field(default_factory=lambda: ['bybit', 'binance'])
    preferred_symbols: List[str] = field(default_factory=lambda: ['BTC/AUD', 'ETH/AUD', 'ADA/AUD'])

@dataclass
class SystemStatus:
    """Current status of the Australian trading system"""
    
    # System state
    status: str = "initializing"  # initializing, running, paused, error
    last_update: datetime = field(default_factory=datetime.now)
    
    # Portfolio state
    total_portfolio_value_aud: Decimal = Decimal('0')
    available_cash_aud: Decimal = Decimal('0')
    ml_strategy_allocation_aud: Decimal = Decimal('0')
    arbitrage_allocation_aud: Decimal = Decimal('0')
    
    # Performance metrics
    daily_pnl_aud: Decimal = Decimal('0')
    weekly_pnl_aud: Decimal = Decimal('0')
    monthly_pnl_aud: Decimal = Decimal('0')
    total_trades_today: int = 0
    successful_trades_today: int = 0
    
    # Risk metrics
    current_drawdown_pct: float = 0.0
    daily_loss_pct: float = 0.0
    max_risk_score: float = 0.0
    active_risk_alerts: int = 0
    
    # Compliance status
    ato_reportable_trades_today: int = 0
    daily_volume_aud: Decimal = Decimal('0')
    compliance_violations: int = 0
    professional_trader_risk: bool = False
    
    # Signal processing
    pending_signals: int = 0
    processed_signals_today: int = 0
    ml_signals_generated_today: int = 0
    arbitrage_opportunities_found_today: int = 0
    
    # System health
    last_data_update: Optional[datetime] = None
    exchange_connectivity: Dict[str, bool] = field(default_factory=dict)
    critical_errors: List[str] = field(default_factory=list)

class AustralianTradingSystemCoordinator:
    """
    Main coordinator for the comprehensive Australian cryptocurrency trading system
    Orchestrates ML strategies, arbitrage detection, risk management, and compliance
    """
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.status = SystemStatus()
        
        # Core components (will be initialized)
        self.trading_bot: Optional[TradingBot] = None
        self.bybit_client: Optional[BybitClient] = None
        
        # Australian compliance
        self.tax_calculator: Optional[AustralianTaxCalculator] = None
        self.banking_manager: Optional[AustralianBankingManager] = None
        self.compliance_manager: Optional[AustralianComplianceManager] = None
        
        # ML and data
        self.ml_engine: Optional[MLStrategyDiscoveryEngine] = None
        self.data_collector: Optional[MultiExchangeDataCollector] = None
        
        # Arbitrage
        self.arbitrage_engine: Optional[OpportunisticArbitrageEngine] = None
        self.arbitrage_executor: Optional[ArbitrageExecutionEngine] = None
        
        # Risk management
        self.risk_calculator: Optional[AustralianRiskCalculator] = None
        self.portfolio_controller: Optional[PortfolioRiskController] = None
        
        # Integration components
        self.trading_engine_integration: Optional[AustralianTradingEngineIntegration] = None
        self.signal_processor: Optional[SignalProcessingManager] = None
        
        # Control state
        self.is_running = False
        self.shutdown_requested = False
        self.last_full_cycle = None
        
        # Performance tracking
        self.cycle_times = []
        self.error_count = 0
        self.total_cycles = 0
        
        logger.info(f"Initialized Australian Trading System Coordinator with config: "
                   f"ML={config.ml_strategy_allocation_pct}%, Arbitrage={config.arbitrage_allocation_pct}%")
    
    async def initialize_system(
        self,
        api_credentials: Dict[str, str],
        database_config: Dict[str, str]
    ) -> bool:
        """Initialize all system components"""
        
        try:
            self.status.status = "initializing"
            logger.info("Initializing Australian Trading System...")
            
            # 1. Initialize Australian Compliance Components
            logger.info("Initializing Australian compliance components...")
            
            self.tax_calculator = AustralianTaxCalculator()
            await self.tax_calculator.initialize(database_config)
            
            self.banking_manager = AustralianBankingManager()
            await self.banking_manager.initialize()
            
            self.compliance_manager = AustralianComplianceManager(
                self.tax_calculator,
                self.banking_manager
            )
            await self.compliance_manager.initialize()
            
            # 2. Initialize Risk Management
            logger.info("Initializing risk management components...")
            
            self.risk_calculator = AustralianRiskCalculator(
                self.tax_calculator,
                self.compliance_manager
            )
            
            self.portfolio_controller = PortfolioRiskController(
                self.risk_calculator,
                self.compliance_manager
            )
            
            # 3. Initialize Data Infrastructure
            logger.info("Initializing data infrastructure...")
            
            self.data_collector = MultiExchangeDataCollector()
            await self.data_collector.initialize(api_credentials)
            
            # 4. Initialize ML Engine
            logger.info("Initializing ML strategy engine...")
            
            self.ml_engine = MLStrategyDiscoveryEngine(self.data_collector)
            await self.ml_engine.initialize()
            
            # 5. Initialize Arbitrage Engine
            logger.info("Initializing arbitrage detection engine...")
            
            self.arbitrage_engine = OpportunisticArbitrageEngine(self.banking_manager)
            await self.arbitrage_engine.initialize()
            
            self.arbitrage_executor = ArbitrageExecutionEngine(
                self.compliance_manager,
                self.risk_calculator
            )
            
            # 6. Initialize Trading Engine Integration
            logger.info("Initializing trading engine integration...")
            
            self.trading_engine_integration = AustralianTradingEngineIntegration(
                self.trading_bot,  # Would be actual trading bot
                self.ml_engine,
                self.arbitrage_engine,
                self.portfolio_controller,
                self.tax_calculator,
                self.compliance_manager
            )
            
            # 7. Initialize Signal Processing Manager
            logger.info("Initializing signal processing manager...")
            
            self.signal_processor = SignalProcessingManager(
                self.trading_engine_integration
            )
            
            # 8. Validate system health
            logger.info("Performing system health checks...")
            health_check = await self._perform_health_check()
            
            if not health_check['healthy']:
                logger.error(f"System health check failed: {health_check['issues']}")
                self.status.status = "error"
                self.status.critical_errors = health_check['issues']
                return False
            
            # 9. Initialize portfolio state
            await self._initialize_portfolio_state()
            
            self.status.status = "ready"
            logger.info("Australian Trading System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            logger.error(traceback.format_exc())
            self.status.status = "error"
            self.status.critical_errors.append(str(e))
            return False
    
    async def start_trading_system(self) -> bool:
        """Start the main trading system loop"""
        
        if self.status.status != "ready":
            logger.error("System not ready - cannot start trading")
            return False
        
        if self.is_running:
            logger.warning("Trading system already running")
            return True
        
        try:
            logger.info("Starting Australian Trading System...")
            
            self.is_running = True
            self.shutdown_requested = False
            self.status.status = "running"
            
            # Start main trading loop
            trading_task = asyncio.create_task(self._main_trading_loop())
            
            # Start monitoring tasks
            monitoring_tasks = [
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._compliance_monitoring_loop()),
                asyncio.create_task(self._performance_monitoring_loop())
            ]
            
            # Wait for shutdown or error
            await trading_task
            
            # Cancel monitoring tasks
            for task in monitoring_tasks:
                task.cancel()
            
            logger.info("Trading system stopped")
            return True
            
        except Exception as e:
            logger.error(f"Trading system error: {e}")
            logger.error(traceback.format_exc())
            self.status.status = "error"
            self.status.critical_errors.append(str(e))
            return False
        
        finally:
            self.is_running = False
    
    async def _main_trading_loop(self):
        """Main trading loop coordinating all system components"""
        
        logger.info("Starting main trading loop...")
        
        while not self.shutdown_requested:
            cycle_start = datetime.now()
            
            try:
                # 1. Update market data
                market_data = await self._collect_market_data()
                
                # 2. Update portfolio state
                portfolio_state = await self._update_portfolio_state()
                
                # 3. Generate ML signals
                ml_signals = await self._generate_ml_signals(market_data)
                
                # 4. Detect arbitrage opportunities
                arbitrage_opportunities = await self._detect_arbitrage_opportunities(market_data, portfolio_state)
                
                # 5. Add signals to processing queue
                if ml_signals:
                    ml_sizes = await self._calculate_ml_position_sizes(ml_signals, portfolio_state)
                    await self._add_ml_signals_to_queue(ml_signals, ml_sizes)
                
                if arbitrage_opportunities:
                    arb_sizes = await self._calculate_arbitrage_position_sizes(arbitrage_opportunities, portfolio_state)
                    await self._add_arbitrage_signals_to_queue(arbitrage_opportunities, arb_sizes)
                
                # 6. Process signal queue
                processing_result = await self.signal_processor.process_signal_queue(portfolio_state)
                
                # 7. Update system status
                await self._update_system_status(processing_result)
                
                # 8. Calculate cycle performance
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                self.cycle_times.append(cycle_duration)
                self.total_cycles += 1
                self.last_full_cycle = datetime.now()
                
                # Log cycle summary
                if self.total_cycles % 10 == 0:  # Every 10 cycles
                    avg_cycle_time = sum(self.cycle_times[-10:]) / min(10, len(self.cycle_times))
                    logger.info(f"Trading cycle {self.total_cycles}: {len(ml_signals or [])} ML signals, "
                               f"{len(arbitrage_opportunities or [])} arbitrage opportunities, "
                               f"avg cycle time: {avg_cycle_time:.2f}s")
                
                # Wait for next cycle
                await asyncio.sleep(self.config.signal_processing_interval_seconds)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Main trading loop error (cycle {self.total_cycles}): {e}")
                
                if self.error_count > 10:  # Too many errors
                    logger.error("Too many consecutive errors - stopping trading system")
                    break
                
                # Brief pause before retry
                await asyncio.sleep(10)
    
    async def _collect_market_data(self) -> Dict[str, Any]:
        """Collect comprehensive market data from all sources"""
        
        try:
            # Collect real-time market data
            market_data = await self.data_collector.collect_comprehensive_data(
                symbols=self.config.preferred_symbols
            )
            
            # Add Australian-specific data
            market_data['aud_macro_data'] = await self.data_collector.collect_macro_data('AUD')
            
            # Update exchange connectivity status
            for exchange in self.config.australian_exchanges + self.config.international_exchanges:
                self.status.exchange_connectivity[exchange] = True  # Simplified
            
            self.status.last_data_update = datetime.now()
            return market_data
            
        except Exception as e:
            logger.error(f"Market data collection error: {e}")
            return {}
    
    async def _update_portfolio_state(self) -> PortfolioState:
        """Update current portfolio state"""
        
        try:
            # Get current portfolio value (simplified)
            portfolio_value = Decimal('100000')  # Placeholder
            available_cash = portfolio_value * Decimal('0.1')
            
            # Update portfolio controller
            portfolio_state = await self.portfolio_controller.update_portfolio_state(
                current_prices={'BTC/AUD': Decimal('65000'), 'ETH/AUD': Decimal('2600')},
                account_balance=available_cash
            )
            
            # Update system status
            self.status.total_portfolio_value_aud = portfolio_value
            self.status.available_cash_aud = available_cash
            self.status.ml_strategy_allocation_aud = portfolio_value * Decimal(self.config.ml_strategy_allocation_pct) / Decimal('100')
            self.status.arbitrage_allocation_aud = portfolio_value * Decimal(self.config.arbitrage_allocation_pct) / Decimal('100')
            
            return portfolio_state
            
        except Exception as e:
            logger.error(f"Portfolio state update error: {e}")
            # Return default state
            return PortfolioState(
                total_value_aud=Decimal('100000'),
                cash_balance_aud=Decimal('10000'),
                positions={},
                risk_metrics={},
                last_update=datetime.now()
            )
    
    async def _generate_ml_signals(self, market_data: Dict[str, Any]) -> Optional[List[StrategySignal]]:
        """Generate ML strategy signals"""
        
        try:
            if not market_data:
                return None
            
            # Generate signals using ML engine
            signals = self.ml_engine.generate_signals(
                data=market_data.get('price_data', {}),
                macro_data=market_data.get('aud_macro_data', {})
            )
            
            # Filter by confidence threshold
            qualified_signals = [
                signal for signal in signals
                if signal.confidence >= self.config.signal_confidence_threshold
            ]
            
            self.status.ml_signals_generated_today += len(qualified_signals)
            
            if qualified_signals:
                logger.info(f"Generated {len(qualified_signals)} qualified ML signals")
            
            return qualified_signals if qualified_signals else None
            
        except Exception as e:
            logger.error(f"ML signal generation error: {e}")
            return None
    
    async def _detect_arbitrage_opportunities(
        self,
        market_data: Dict[str, Any],
        portfolio_state: PortfolioState
    ) -> Optional[List[ArbitrageOpportunity]]:
        """Detect arbitrage opportunities"""
        
        try:
            if not market_data:
                return None
            
            # Scan for opportunities
            opportunities = await self.arbitrage_engine.scan_for_opportunities(
                price_data=market_data.get('price_data', {}),
                funding_data=market_data.get('funding_data', {}),
                balance_aud=portfolio_state.total_value_aud,
                symbols=self.config.preferred_symbols
            )
            
            # Filter valid opportunities
            valid_opportunities = [
                opp for opp in opportunities
                if opp.is_valid() and opp.net_profit_percentage > Decimal('0.5')  # >0.5% profit
            ]
            
            self.status.arbitrage_opportunities_found_today += len(valid_opportunities)
            
            if valid_opportunities:
                logger.info(f"Detected {len(valid_opportunities)} valid arbitrage opportunities")
            
            return valid_opportunities if valid_opportunities else None
            
        except Exception as e:
            logger.error(f"Arbitrage detection error: {e}")
            return None
    
    async def _calculate_ml_position_sizes(
        self,
        signals: List[StrategySignal],
        portfolio_state: PortfolioState
    ) -> Dict[str, Decimal]:
        """Calculate position sizes for ML signals"""
        
        try:
            # Get ML allocation
            ml_allocation = self.status.ml_strategy_allocation_aud
            
            # Distribute across signals based on strength and confidence
            total_signal_weight = sum(
                abs(signal.signal_strength) * signal.confidence
                for signal in signals
            )
            
            position_sizes = {}
            
            for signal in signals:
                signal_weight = abs(signal.signal_strength) * signal.confidence
                allocation_ratio = signal_weight / total_signal_weight if total_signal_weight > 0 else 0
                
                # Calculate position size
                base_size = ml_allocation * Decimal(str(allocation_ratio))
                
                # Apply position limits
                max_position = portfolio_state.total_value_aud * Decimal(self.config.max_single_position_pct) / Decimal('100')
                position_size = min(base_size, max_position)
                
                # Ensure minimum trade size
                if position_size >= self.config.min_trade_size_aud:
                    position_sizes[signal.symbol] = position_size
            
            return position_sizes
            
        except Exception as e:
            logger.error(f"ML position size calculation error: {e}")
            return {}
    
    async def _calculate_arbitrage_position_sizes(
        self,
        opportunities: List[ArbitrageOpportunity],
        portfolio_state: PortfolioState
    ) -> Dict[str, Decimal]:
        """Calculate position sizes for arbitrage opportunities"""
        
        try:
            # Get arbitrage allocation
            arb_allocation = self.status.arbitrage_allocation_aud
            
            # Distribute based on profit potential and balance tier
            position_sizes = {}
            
            for opportunity in opportunities:
                # Base size on balance tier
                if opportunity.balance_tier == 'Large':
                    base_size = arb_allocation * Decimal('0.4')  # 40% of arbitrage allocation
                elif opportunity.balance_tier == 'Medium':
                    base_size = arb_allocation * Decimal('0.3')  # 30%
                elif opportunity.balance_tier == 'Small':
                    base_size = arb_allocation * Decimal('0.2')  # 20%
                else:  # Micro
                    base_size = arb_allocation * Decimal('0.1')  # 10%
                
                # Apply position limits
                max_position = portfolio_state.total_value_aud * Decimal(self.config.max_single_position_pct) / Decimal('100')
                position_size = min(base_size, max_position)
                
                # Ensure minimum trade size
                if position_size >= self.config.min_trade_size_aud:
                    position_sizes[opportunity.symbol] = position_size
            
            return position_sizes
            
        except Exception as e:
            logger.error(f"Arbitrage position size calculation error: {e}")
            return {}
    
    async def _add_ml_signals_to_queue(
        self,
        signals: List[StrategySignal],
        position_sizes: Dict[str, Decimal]
    ):
        """Add ML signals to processing queue"""
        
        try:
            signal_ids = self.signal_processor.add_ml_signals(signals, position_sizes)
            logger.info(f"Added {len(signal_ids)} ML signals to processing queue")
            
        except Exception as e:
            logger.error(f"Error adding ML signals to queue: {e}")
    
    async def _add_arbitrage_signals_to_queue(
        self,
        opportunities: List[ArbitrageOpportunity],
        position_sizes: Dict[str, Decimal]
    ):
        """Add arbitrage opportunities to processing queue"""
        
        try:
            signal_ids = self.signal_processor.add_arbitrage_opportunities(opportunities, position_sizes)
            logger.info(f"Added {len(signal_ids)} arbitrage opportunities to processing queue")
            
        except Exception as e:
            logger.error(f"Error adding arbitrage opportunities to queue: {e}")
    
    async def _update_system_status(self, processing_result: Dict[str, Any]):
        """Update comprehensive system status"""
        
        try:
            self.status.last_update = datetime.now()
            
            # Update signal processing metrics
            if 'summary' in processing_result:
                summary = processing_result['summary']
                self.status.processed_signals_today += summary.get('signals_processed', 0)
                
                # Update trade counts
                executed_count = summary.get('signals_executed', 0)
                self.status.total_trades_today += executed_count
                self.status.successful_trades_today += executed_count
            
            # Update queue status
            queue_status = self.signal_processor.get_queue_status()
            self.status.pending_signals = queue_status['queue_length']
            
            # Update risk metrics (simplified)
            self.status.current_drawdown_pct = 2.5  # Placeholder
            self.status.daily_loss_pct = 1.2       # Placeholder
            self.status.max_risk_score = 0.65      # Placeholder
            
            # Update compliance status
            self.status.daily_volume_aud = Decimal('25000')  # Placeholder
            self.status.ato_reportable_trades_today = 3      # Placeholder
            
        except Exception as e:
            logger.error(f"System status update error: {e}")
    
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop"""
        
        while not self.shutdown_requested:
            try:
                # Get current portfolio state
                portfolio_state = await self._update_portfolio_state()
                
                # Assess risks
                risk_alerts = await self.portfolio_controller.assess_portfolio_risks(portfolio_state)
                
                self.status.active_risk_alerts = len(risk_alerts)
                
                # Handle emergency conditions
                emergency_alerts = [alert for alert in risk_alerts if alert.level.value == 'emergency']
                if emergency_alerts:
                    logger.critical(f"EMERGENCY: {len(emergency_alerts)} critical risk conditions detected")
                    # Would implement emergency shutdown procedures
                
                await asyncio.sleep(self.config.risk_monitoring_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _compliance_monitoring_loop(self):
        """Continuous compliance monitoring loop"""
        
        while not self.shutdown_requested:
            try:
                # Monitor compliance violations
                violations = await self.compliance_manager.check_ongoing_compliance()
                self.status.compliance_violations = len(violations)
                
                # Check professional trader risk
                trade_count = await self.tax_calculator.get_annual_trade_count()
                self.status.professional_trader_risk = trade_count > self.config.professional_trader_threshold_trades * 0.8
                
                if violations:
                    logger.warning(f"Compliance violations detected: {violations}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Compliance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring loop"""
        
        while not self.shutdown_requested:
            try:
                # Calculate performance metrics (simplified)
                self.status.daily_pnl_aud = Decimal('1250')    # Placeholder
                self.status.weekly_pnl_aud = Decimal('4800')   # Placeholder
                self.status.monthly_pnl_aud = Decimal('18500') # Placeholder
                
                # Log performance summary
                if self.total_cycles > 0 and self.total_cycles % 100 == 0:
                    logger.info(f"Performance Summary - Daily P&L: ${self.status.daily_pnl_aud:,.2f}, "
                               f"Success Rate: {self.status.successful_trades_today}/{self.status.total_trades_today}")
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        
        health_issues = []
        
        try:
            # Check component initialization
            required_components = [
                'tax_calculator', 'banking_manager', 'compliance_manager',
                'ml_engine', 'data_collector', 'arbitrage_engine',
                'risk_calculator', 'portfolio_controller'
            ]
            
            for component in required_components:
                if not hasattr(self, component) or getattr(self, component) is None:
                    health_issues.append(f"Component not initialized: {component}")
            
            # Check data connectivity
            if self.data_collector:
                connectivity_check = await self.data_collector.test_connectivity()
                if not connectivity_check.get('healthy', False):
                    health_issues.append("Data connectivity issues detected")
            
            # Check compliance readiness
            if self.compliance_manager:
                compliance_check = await self.compliance_manager.validate_system_compliance()
                if not compliance_check.get('compliant', False):
                    health_issues.append("System compliance validation failed")
            
            return {
                'healthy': len(health_issues) == 0,
                'issues': health_issues,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'issues': [f"Health check error: {str(e)}"],
                'timestamp': datetime.now()
            }
    
    async def _initialize_portfolio_state(self):
        """Initialize portfolio state and allocations"""
        
        try:
            # Set initial portfolio state
            self.status.total_portfolio_value_aud = Decimal('100000')  # Starting with $100k
            self.status.available_cash_aud = self.status.total_portfolio_value_aud
            
            # Calculate allocations
            self.status.ml_strategy_allocation_aud = (
                self.status.total_portfolio_value_aud * 
                Decimal(self.config.ml_strategy_allocation_pct) / Decimal('100')
            )
            
            self.status.arbitrage_allocation_aud = (
                self.status.total_portfolio_value_aud * 
                Decimal(self.config.arbitrage_allocation_pct) / Decimal('100')
            )
            
            logger.info(f"Portfolio initialized: ${self.status.total_portfolio_value_aud:,.2f} total, "
                       f"${self.status.ml_strategy_allocation_aud:,.2f} ML, "
                       f"${self.status.arbitrage_allocation_aud:,.2f} arbitrage")
            
        except Exception as e:
            logger.error(f"Portfolio initialization error: {e}")
    
    def request_shutdown(self):
        """Request graceful shutdown of the trading system"""
        
        logger.info("Shutdown requested for Australian Trading System")
        self.shutdown_requested = True
        self.status.status = "shutting_down"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'system_status': self.status.status,
            'is_running': self.is_running,
            'last_update': self.status.last_update,
            'total_cycles': self.total_cycles,
            'error_count': self.error_count,
            'last_full_cycle': self.last_full_cycle,
            
            'portfolio': {
                'total_value_aud': float(self.status.total_portfolio_value_aud),
                'available_cash_aud': float(self.status.available_cash_aud),
                'ml_allocation_aud': float(self.status.ml_strategy_allocation_aud),
                'arbitrage_allocation_aud': float(self.status.arbitrage_allocation_aud)
            },
            
            'performance': {
                'daily_pnl_aud': float(self.status.daily_pnl_aud),
                'total_trades_today': self.status.total_trades_today,
                'successful_trades_today': self.status.successful_trades_today,
                'success_rate': (
                    self.status.successful_trades_today / self.status.total_trades_today
                    if self.status.total_trades_today > 0 else 0
                )
            },
            
            'risk': {
                'current_drawdown_pct': self.status.current_drawdown_pct,
                'daily_loss_pct': self.status.daily_loss_pct,
                'max_risk_score': self.status.max_risk_score,
                'active_risk_alerts': self.status.active_risk_alerts
            },
            
            'compliance': {
                'ato_reportable_trades_today': self.status.ato_reportable_trades_today,
                'daily_volume_aud': float(self.status.daily_volume_aud),
                'compliance_violations': self.status.compliance_violations,
                'professional_trader_risk': self.status.professional_trader_risk
            },
            
            'signals': {
                'pending_signals': self.status.pending_signals,
                'processed_signals_today': self.status.processed_signals_today,
                'ml_signals_generated_today': self.status.ml_signals_generated_today,
                'arbitrage_opportunities_found_today': self.status.arbitrage_opportunities_found_today
            },
            
            'system_health': {
                'exchange_connectivity': self.status.exchange_connectivity,
                'last_data_update': self.status.last_data_update,
                'critical_errors': self.status.critical_errors,
                'avg_cycle_time_seconds': (
                    sum(self.cycle_times[-10:]) / len(self.cycle_times[-10:])
                    if self.cycle_times else 0
                )
            }
        }

# Usage example
async def main():
    """Example usage of the Australian Trading System Coordinator"""
    
    print("Australian Trading System Coordinator Example")
    
    # Create configuration
    config = SystemConfiguration(
        ml_strategy_allocation_pct=70,
        arbitrage_allocation_pct=20,
        cash_reserve_pct=10,
        max_portfolio_value_aud=Decimal('1000000')
    )
    
    # Create coordinator
    coordinator = AustralianTradingSystemCoordinator(config)
    
    # Example initialization
    api_credentials = {
        'bybit_api_key': 'test_key',
        'bybit_api_secret': 'test_secret'
    }
    
    database_config = {
        'host': 'localhost',
        'database': 'trading_db'
    }
    
    print(f"Configuration: ML={config.ml_strategy_allocation_pct}%, "
          f"Arbitrage={config.arbitrage_allocation_pct}%, "
          f"Max Portfolio=${config.max_portfolio_value_aud:,.2f}")
    
    print("System would coordinate:")
    print("- ML strategy discovery and execution")
    print("- Arbitrage opportunity detection and execution")
    print("- Australian compliance and tax management")
    print("- Real-time risk management and monitoring")
    print("- Complete trading lifecycle automation")

if __name__ == "__main__":
    asyncio.run(main())