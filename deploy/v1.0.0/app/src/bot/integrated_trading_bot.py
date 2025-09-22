"""
Integrated Trading Bot - Phase 10

This module provides the main TradingBot class that integrates ALL phases:
- Phase 1: Core Trading Engine (order execution, market data, position management)  
- Phase 2: Risk Management (position sizing, drawdown protection, volatility control)
- Phase 3: Backtesting Engine (strategy testing, performance analysis, optimization)
- Phase 4: System Monitoring (health checks, performance tracking, alerting)
- Phase 5: Tax and Reporting (trade logging, tax calculations, compliance reports)
- Phase 6: Advanced Features (regime detection, portfolio optimization, news analysis)

This unified system provides enterprise-grade trading capabilities with comprehensive
risk management, monitoring, and adaptive behavior.

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

# Core Phase 1 imports
from core.trading_engine import TradingEngine, OrderType, OrderSide
from core.market_data import MarketDataManager, MarketDataType
from core.position_manager import PositionManager, Position

# Risk Management Phase 2 imports
from risk_management.risk_manager import RiskManager, RiskMetrics
from risk_management.portfolio_risk import PortfolioRiskManager
from risk_management.drawdown_protection import DrawdownProtectionManager

# Backtesting Phase 3 imports
from backtesting.backtesting_engine import BacktestingEngine, BacktestConfig
from backtesting.strategy_optimizer import StrategyOptimizer
from backtesting.performance_analyzer import PerformanceAnalyzer

# Monitoring Phase 4 imports
from monitoring.system_monitor import SystemMonitor, SystemHealthStatus
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alerting_system import AlertingSystem, AlertLevel

# Tax and Reporting Phase 5 imports
from tax_reporting.trade_logger import TradeLogger
from tax_reporting.tax_calculator import TaxCalculator
from tax_reporting.compliance_reporter import ComplianceReporter

# Advanced Features Phase 6 imports
from advanced.regime_detector import RegimeDetector, MarketRegime
from advanced.portfolio_optimizer import PortfolioOptimizer, OptimizationMethod
from advanced.automated_reporter import AutomatedReporter
from advanced.news_analyzer import NewsAnalyzer, SentimentLevel
from advanced.parameter_optimizer import ParameterOptimizer


class BotStatus(Enum):
    """Bot operational status"""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class BotConfiguration:
    """Comprehensive bot configuration"""
    # Core settings
    exchange: str = "bybit"
    trading_pairs: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    base_currency: str = "USDT"
    initial_capital: float = 10000.0
    
    # Trading settings
    max_position_size: float = 0.1  # 10% of portfolio per position
    max_daily_trades: int = 50
    min_trade_size: float = 10.0
    max_spread_tolerance: float = 0.002  # 0.2%
    
    # Risk management settings
    max_portfolio_risk: float = 0.02  # 2% portfolio risk per trade
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_drawdown: float = 0.15  # 15% maximum drawdown
    volatility_lookback: int = 20
    
    # System settings
    update_frequency_seconds: int = 5
    market_data_buffer_size: int = 1000
    max_memory_usage_mb: int = 2048
    log_level: str = "INFO"
    
    # Advanced features settings
    enable_regime_detection: bool = True
    enable_portfolio_optimization: bool = True
    enable_news_analysis: bool = True
    enable_parameter_optimization: bool = True
    
    # Reporting settings
    enable_automated_reporting: bool = True
    report_frequency_hours: int = 24
    enable_email_alerts: bool = False
    email_recipients: List[str] = field(default_factory=list)
    
    # API and external services
    api_rate_limit: int = 10  # requests per second
    news_update_frequency_minutes: int = 30
    regime_update_frequency_minutes: int = 60


class IntegratedTradingBot:
    """
    Unified Trading Bot - Phase 10 Integration
    
    This class integrates all trading system phases into a cohesive,
    production-ready trading bot with enterprise-grade capabilities.
    """
    
    def __init__(self, config_manager: Optional['ConfigurationManager'] = None, config: Optional[BotConfiguration] = None):
        """Initialize the unified trading bot"""
        # Import here to avoid circular imports
        from config_manager import ConfigurationManager, ConfigurationError
        
        # Use configuration manager if provided, otherwise create one
        self.config_manager = config_manager or ConfigurationManager()
        if not self.config_manager.config:
            self.config_manager.load_config()
        
        # Legacy support for old BotConfiguration
        self.config = config or BotConfiguration()
        self.status = BotStatus.INITIALIZING
        self.start_time = None
        self.stop_event = threading.Event()
        
        # Initialize logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing IntegratedTradingBot in {self.config_manager.config.environment.value} environment")
        
        # Validate environment credentials before initializing components
        self._validate_credentials()
        
        # Initialize all phase components
        self._initialize_components()
        
        # Initialize Strategy Graduation Manager
        from .strategy_graduation import StrategyGraduationManager
        self.graduation_manager = StrategyGraduationManager(self.config_manager)
        
        # Performance metrics
        self.performance_metrics = {}
        self.system_metrics = {}
        self.trade_count = 0
        self.total_pnl = 0.0
        
        # Control flags
        self.trading_enabled = True
        self.risk_override = False
        self.maintenance_mode = False
        
        # Strategy management
        self.active_strategies = {}  # strategy_id -> strategy instance
        self.paper_strategies = {}   # strategies in paper trading
        self.live_strategies = {}    # strategies in live trading
        
        self.logger.info("IntegratedTradingBot initialization complete")
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('integrated_trading_bot.log'),
                logging.StreamHandler()
            ]
        )
    
    def _validate_credentials(self):
        """Validate environment-specific credentials"""
        try:
            credentials = self.config_manager.get_current_credentials()
            env_name = self.config_manager.config.environment.value
            
            # Validate credentials exist
            if not credentials.api_key or not credentials.api_secret:
                raise ConfigurationError(f"Missing API credentials for {env_name} environment")
            
            # Log environment and safety information
            if credentials.is_testnet:
                self.logger.info(f"✅ TESTNET MODE - Environment: {env_name}")
                self.logger.info(f"✅ Safe to trade - Using testnet credentials")
            else:
                self.logger.warning(f"⚠️  LIVE TRADING MODE - Environment: {env_name}")
                self.logger.warning(f"⚠️  REAL MONEY AT RISK - Using live credentials")
            
            self.logger.info(f"Base URL: {credentials.base_url}")
            self.logger.info(f"API Key: {credentials.api_key[:8]}...")
            
        except Exception as e:
            self.logger.error(f"Credential validation failed: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize all phase components"""
        try:
            self.logger.info("Initializing Phase 1: Core Trading Engine")
            # Phase 1: Core Trading Engine - use environment-specific credentials
            exchange_credentials = self.config_manager.get_current_credentials()
            self.trading_engine = TradingEngine(
                exchange_name=exchange_credentials.base_url,
                api_key=exchange_credentials.api_key,
                api_secret=exchange_credentials.api_secret,
                is_testnet=exchange_credentials.is_testnet,
                api_rate_limit=self.config.api_rate_limit
            )
            self.market_data_manager = MarketDataManager(
                trading_pairs=self.config.trading_pairs,
                buffer_size=self.config.market_data_buffer_size
            )
            self.position_manager = PositionManager(
                initial_capital=self.config.initial_capital,
                base_currency=self.config.base_currency
            )
            
            self.logger.info("Initializing Phase 2: Risk Management")
            # Phase 2: Risk Management
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.config.max_portfolio_risk,
                max_daily_loss=self.config.max_daily_loss,
                volatility_lookback=self.config.volatility_lookback
            )
            self.portfolio_risk_manager = PortfolioRiskManager(
                max_position_size=self.config.max_position_size,
                correlation_threshold=0.7
            )
            self.drawdown_protection = DrawdownProtectionManager(
                max_drawdown=self.config.max_drawdown,
                emergency_stop_drawdown=self.config.max_drawdown * 1.5
            )
            
            self.logger.info("Initializing Phase 3: Backtesting Engine")
            # Phase 3: Backtesting Engine
            self.backtesting_engine = BacktestingEngine()
            self.strategy_optimizer = StrategyOptimizer()
            self.performance_analyzer = PerformanceAnalyzer()
            
            self.logger.info("Initializing Phase 4: System Monitoring")
            # Phase 4: System Monitoring
            self.system_monitor = SystemMonitor(
                max_memory_mb=self.config.max_memory_usage_mb,
                check_interval_seconds=30
            )
            self.performance_tracker = PerformanceTracker()
            self.alerting_system = AlertingSystem(
                email_enabled=self.config.enable_email_alerts,
                email_recipients=self.config.email_recipients
            )
            
            self.logger.info("Initializing Phase 5: Tax and Reporting")
            # Phase 5: Tax and Reporting
            self.trade_logger = TradeLogger()
            self.tax_calculator = TaxCalculator()
            self.compliance_reporter = ComplianceReporter()
            
            # Phase 6: Advanced Features (optional based on config)
            if self.config.enable_regime_detection:
                self.logger.info("Initializing Phase 6: Advanced Features - Regime Detection")
                self.regime_detector = RegimeDetector()
            else:
                self.regime_detector = None
                
            if self.config.enable_portfolio_optimization:
                self.logger.info("Initializing Phase 6: Advanced Features - Portfolio Optimization")
                self.portfolio_optimizer = PortfolioOptimizer()
            else:
                self.portfolio_optimizer = None
                
            if self.config.enable_automated_reporting:
                self.logger.info("Initializing Phase 6: Advanced Features - Automated Reporting")
                self.automated_reporter = AutomatedReporter()
            else:
                self.automated_reporter = None
                
            if self.config.enable_news_analysis:
                self.logger.info("Initializing Phase 6: Advanced Features - News Analysis")
                self.news_analyzer = NewsAnalyzer()
            else:
                self.news_analyzer = None
                
            if self.config.enable_parameter_optimization:
                self.logger.info("Initializing Phase 6: Advanced Features - Parameter Optimization")
                self.parameter_optimizer = ParameterOptimizer()
            else:
                self.parameter_optimizer = None
            
            self.logger.info("All phase components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.status = BotStatus.ERROR
            raise
    
    async def start(self):
        """Start the trading bot with full integration"""
        try:
            self.logger.info("Starting IntegratedTradingBot with all phases")
            self.status = BotStatus.STARTING
            self.start_time = datetime.now()
            
            # Start all components
            await self._start_components()
            
            # Start main trading loop
            self.trading_task = asyncio.create_task(self._main_trading_loop())
            
            # Start monitoring tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start advanced features tasks
            if self.config.enable_regime_detection:
                self.regime_task = asyncio.create_task(self._regime_detection_loop())
            
            if self.config.enable_news_analysis:
                self.news_task = asyncio.create_task(self._news_analysis_loop())
            
            if self.config.enable_automated_reporting:
                self.reporting_task = asyncio.create_task(self._automated_reporting_loop())
            
            self.status = BotStatus.RUNNING
            self.logger.info("IntegratedTradingBot started successfully - all systems operational")
            
            # Send startup alert
            await self.alerting_system.send_alert(
                AlertLevel.INFO,
                "TradingBot Startup",
                "Integrated trading bot started successfully with all phases"
            )
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {e}")
            self.status = BotStatus.ERROR
            await self.alerting_system.send_alert(
                AlertLevel.CRITICAL,
                "TradingBot Startup Failed",
                f"Failed to start trading bot: {e}"
            )
            raise
    
    async def _start_components(self):
        """Start all component systems"""
        # Start core systems
        await self.trading_engine.start()
        await self.market_data_manager.start()
        
        # Start monitoring
        await self.system_monitor.start()
        await self.performance_tracker.start()
        
        # Initialize advanced features
        if self.regime_detector:
            await self.regime_detector.initialize()
        
        if self.news_analyzer:
            await self.news_analyzer.initialize()
    
    async def _main_trading_loop(self):
        """Main trading loop integrating all phases"""
        self.logger.info("Starting main integrated trading loop")
        
        while not self.stop_event.is_set() and self.status == BotStatus.RUNNING:
            try:
                loop_start_time = time.time()
                
                # 1. Update market data (Phase 1)
                market_data = await self.market_data_manager.get_latest_data()
                
                if market_data.empty:
                    await asyncio.sleep(self.config.update_frequency_seconds)
                    continue
                
                # 2. System health checks (Phase 4)
                system_health = await self.system_monitor.get_system_health()
                if system_health.status != SystemHealthStatus.HEALTHY:
                    self.logger.warning(f"System health degraded: {system_health.status}")
                    if system_health.status == SystemHealthStatus.CRITICAL:
                        await self._emergency_shutdown("Critical system health")
                        break
                
                # 3. Risk assessment (Phase 2)
                portfolio_risk = await self._assess_portfolio_risk()
                if portfolio_risk.requires_immediate_action:
                    await self._handle_risk_breach(portfolio_risk)
                    continue
                
                # 4. Advanced market analysis (Phase 6)
                market_context = await self._analyze_market_context(market_data)
                
                # 5. Trading decision making
                trading_signals = await self._generate_trading_signals(market_data, market_context)
                
                # 6. Execute trades (Phase 1)
                if trading_signals and self.trading_enabled and not market_context.get('trading_halted', False):
                    executed_trades = await self._execute_trading_signals(trading_signals)
                    
                    # 7. Log trades (Phase 5)
                    for trade in executed_trades:
                        await self.trade_logger.log_trade(trade)
                        self.trade_count += 1
                
                # 8. Update performance metrics
                await self._update_performance_metrics()
                
                # 9. Portfolio optimization (Phase 6)
                if self.portfolio_optimizer and self.trade_count % 10 == 0:
                    await self._optimize_portfolio()
                
                # 10. Strategy graduation evaluation (every 100 iterations or based on time)
                current_loop_duration = time.time() - loop_start_time
                if self.graduation_manager and (self.trade_count % 100 == 0 or current_loop_duration > 1800):
                    await self.run_strategy_graduation_cycle()
                
                # Control loop timing
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, self.config.update_frequency_seconds - loop_duration)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                await self.alerting_system.send_alert(
                    AlertLevel.ERROR,
                    "Trading Loop Error",
                    f"Error in main trading loop: {e}"
                )
                await asyncio.sleep(self.config.update_frequency_seconds)
    
    async def _assess_portfolio_risk(self) -> Any:
        """Comprehensive portfolio risk assessment"""
        # Get current positions
        positions = await self.position_manager.get_all_positions()
        
        # Calculate risk metrics
        portfolio_value = sum(pos.market_value for pos in positions.values())
        total_exposure = sum(abs(pos.market_value) for pos in positions.values())
        
        # Risk calculations
        current_drawdown = await self.drawdown_protection.calculate_current_drawdown()
        daily_pnl = await self.performance_tracker.get_daily_pnl()
        
        # Check risk thresholds
        risk_metrics = RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            current_drawdown=current_drawdown,
            daily_pnl=daily_pnl,
            requires_immediate_action=(
                current_drawdown > self.config.max_drawdown or
                daily_pnl < -self.config.max_daily_loss * self.config.initial_capital
            )
        )
        
        return risk_metrics
    
    async def _analyze_market_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market context using advanced features"""
        context = {
            'timestamp': datetime.now(),
            'trading_halted': False,
            'halt_reason': '',
            'regime': None,
            'sentiment': None,
            'volatility': None
        }
        
        try:
            # Regime detection
            if self.regime_detector:
                regime_result = self.regime_detector.detect_regime(market_data)
                context['regime'] = regime_result.regime
                context['regime_confidence'] = regime_result.confidence
                
                # Check if we should trade in this regime
                should_trade = self.regime_detector.should_trade_in_regime(
                    regime_result.regime, 'trend_following'
                )
                if not should_trade:
                    context['trading_halted'] = True
                    context['halt_reason'] = f"Regime filter: {regime_result.regime.value}"
            
            # News sentiment analysis
            if self.news_analyzer and not context['trading_halted']:
                sentiment_result = await self.news_analyzer.get_current_sentiment()
                context['sentiment'] = sentiment_result
                
                # Check sentiment-based trading halt
                should_halt, halt_reason = self.news_analyzer.should_halt_trading('conservative')
                if should_halt:
                    context['trading_halted'] = True
                    context['halt_reason'] = halt_reason
            
            # Volatility analysis
            if len(market_data) >= self.config.volatility_lookback:
                returns = market_data['close'].pct_change().dropna()
                current_volatility = returns.rolling(self.config.volatility_lookback).std().iloc[-1]
                context['volatility'] = current_volatility
                
                # High volatility check
                if current_volatility > 0.05:  # 5% daily volatility threshold
                    context['high_volatility'] = True
            
        except Exception as e:
            self.logger.error(f"Error analyzing market context: {e}")
        
        return context
    
    async def _generate_trading_signals(self, market_data: pd.DataFrame, context: Dict) -> List[Dict]:
        """Generate trading signals based on integrated analysis"""
        signals = []
        
        try:
            # Simple momentum strategy example
            if len(market_data) >= 20:
                # Calculate technical indicators
                sma_short = market_data['close'].rolling(5).mean()
                sma_long = market_data['close'].rolling(20).mean()
                rsi = self._calculate_rsi(market_data['close'], 14)
                
                current_price = market_data['close'].iloc[-1]
                
                # Generate signals based on context
                for pair in self.config.trading_pairs:
                    signal = None
                    
                    # Bullish conditions
                    if (sma_short.iloc[-1] > sma_long.iloc[-1] and 
                        rsi.iloc[-1] < 70 and 
                        context.get('regime') not in [MarketRegime.CRASH, MarketRegime.BEAR_MARKET]):
                        
                        signal = {
                            'pair': pair,
                            'side': 'buy',
                            'price': current_price,
                            'confidence': 0.7,
                            'reason': 'Momentum bullish + favorable regime'
                        }
                    
                    # Bearish conditions  
                    elif (sma_short.iloc[-1] < sma_long.iloc[-1] and 
                          rsi.iloc[-1] > 30 and
                          context.get('regime') not in [MarketRegime.BULL_MARKET, MarketRegime.BUBBLE]):
                        
                        signal = {
                            'pair': pair,
                            'side': 'sell',
                            'price': current_price,
                            'confidence': 0.6,
                            'reason': 'Momentum bearish + regime confirmation'
                        }
                    
                    if signal:
                        signals.append(signal)
        
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _execute_trading_signals(self, signals: List[Dict]) -> List[Dict]:
        """Execute trading signals with integrated risk management"""
        executed_trades = []
        
        for signal in signals:
            try:
                # Risk validation
                position_size = await self._calculate_position_size(signal)
                if position_size <= 0:
                    continue
                
                # Execute trade
                order_result = await self.trading_engine.place_order(
                    symbol=signal['pair'],
                    side=OrderSide.BUY if signal['side'] == 'buy' else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position_size
                )
                
                if order_result.success:
                    trade = {
                        'timestamp': datetime.now(),
                        'pair': signal['pair'],
                        'side': signal['side'],
                        'quantity': position_size,
                        'price': order_result.fill_price,
                        'order_id': order_result.order_id,
                        'reason': signal['reason']
                    }
                    executed_trades.append(trade)
                    
                    self.logger.info(f"Trade executed: {trade}")
                
            except Exception as e:
                self.logger.error(f"Error executing signal {signal}: {e}")
        
        return executed_trades
    
    async def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size with risk management"""
        try:
            # Get portfolio value
            portfolio_value = await self.position_manager.get_portfolio_value()
            
            # Risk-based position sizing
            risk_amount = portfolio_value * self.config.max_portfolio_risk
            
            # Calculate position size based on signal confidence
            confidence_multiplier = signal.get('confidence', 0.5)
            base_position_size = risk_amount * confidence_multiplier
            
            # Maximum position size constraint
            max_position_value = portfolio_value * self.config.max_position_size
            position_value = min(base_position_size, max_position_value)
            
            # Convert to quantity
            current_price = signal['price']
            position_size = position_value / current_price
            
            # Minimum trade size check
            if position_value < self.config.min_trade_size:
                return 0
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            # Get current positions and portfolio value
            positions = await self.position_manager.get_all_positions()
            portfolio_value = await self.position_manager.get_portfolio_value()
            
            # Update performance tracker
            await self.performance_tracker.update_metrics({
                'portfolio_value': portfolio_value,
                'trade_count': self.trade_count,
                'total_positions': len(positions),
                'timestamp': datetime.now()
            })
            
            # Store in system metrics
            self.performance_metrics = {
                'portfolio_value': portfolio_value,
                'trade_count': self.trade_count,
                'total_pnl': portfolio_value - self.config.initial_capital,
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _optimize_portfolio(self):
        """Optimize portfolio using advanced optimization"""
        if not self.portfolio_optimizer:
            return
        
        try:
            # Get historical data for optimization
            market_data = await self.market_data_manager.get_historical_data(days=30)
            
            if len(market_data) < 20:
                return
            
            # Calculate returns
            returns_data = {}
            for pair in self.config.trading_pairs:
                if pair in market_data.columns:
                    returns_data[pair] = market_data[pair].pct_change().dropna()
            
            if not returns_data:
                return
            
            returns_df = pd.DataFrame(returns_data)
            self.portfolio_optimizer.load_data(returns_df)
            
            # Optimize portfolio
            current_positions = await self.position_manager.get_all_positions()
            current_weights = {pair: 0.0 for pair in self.config.trading_pairs}
            
            portfolio_value = await self.position_manager.get_portfolio_value()
            for pair, position in current_positions.items():
                if pair in current_weights:
                    current_weights[pair] = position.market_value / portfolio_value
            
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                OptimizationMethod.MAX_SHARPE,
                current_portfolio=current_weights
            )
            
            if optimization_result and optimization_result.sharpe_ratio > 1.0:
                self.logger.info(f"Portfolio optimization suggests rebalancing: {optimization_result.weights}")
                # Note: In production, this would trigger rebalancing logic
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
    
    async def _monitoring_loop(self):
        """System monitoring loop"""
        while not self.stop_event.is_set() and self.status == BotStatus.RUNNING:
            try:
                # System health check
                health_status = await self.system_monitor.get_system_health()
                
                # Performance monitoring
                performance_metrics = await self.performance_tracker.get_current_metrics()
                
                # Check for alerts
                if health_status.status == SystemHealthStatus.WARNING:
                    await self.alerting_system.send_alert(
                        AlertLevel.WARNING,
                        "System Health Warning",
                        f"System health degraded: {health_status.message}"
                    )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _regime_detection_loop(self):
        """Regime detection loop"""
        if not self.regime_detector:
            return
        
        while not self.stop_event.is_set() and self.status == BotStatus.RUNNING:
            try:
                # Update regime detection every hour
                market_data = await self.market_data_manager.get_historical_data(days=1)
                
                if len(market_data) >= 20:
                    regime_result = self.regime_detector.detect_regime(market_data)
                    self.logger.info(f"Current market regime: {regime_result.regime.value} (confidence: {regime_result.confidence:.3f})")
                
                await asyncio.sleep(self.config.regime_update_frequency_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in regime detection loop: {e}")
                await asyncio.sleep(self.config.regime_update_frequency_minutes * 60)
    
    async def _news_analysis_loop(self):
        """News analysis loop"""
        if not self.news_analyzer:
            return
        
        while not self.stop_event.is_set() and self.status == BotStatus.RUNNING:
            try:
                # Analyze news sentiment
                articles = await self.news_analyzer.fetch_news_articles(hours_back=1)
                if articles:
                    sentiment_result = self.news_analyzer.analyze_sentiment(articles)
                    self.logger.info(f"News sentiment: {sentiment_result.overall_sentiment.value} (score: {sentiment_result.sentiment_score:+.3f})")
                
                await asyncio.sleep(self.config.news_update_frequency_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in news analysis loop: {e}")
                await asyncio.sleep(self.config.news_update_frequency_minutes * 60)
    
    async def _automated_reporting_loop(self):
        """Automated reporting loop"""
        if not self.automated_reporter:
            return
        
        while not self.stop_event.is_set() and self.status == BotStatus.RUNNING:
            try:
                # Generate reports periodically
                await asyncio.sleep(self.config.report_frequency_hours * 3600)
                
                # Generate daily report
                await self._generate_daily_report()
                
            except Exception as e:
                self.logger.error(f"Error in automated reporting loop: {e}")
                await asyncio.sleep(self.config.report_frequency_hours * 3600)
    
    async def _generate_daily_report(self):
        """Generate daily performance report"""
        try:
            # Collect performance data
            performance_data = await self.performance_tracker.get_daily_report_data()
            
            # Generate report
            report_data = self.automated_reporter.generate_daily_report(performance_data)
            
            # Save and email report
            output_path = self.automated_reporter.save_report(report_data)
            
            if self.config.enable_email_alerts and self.config.email_recipients:
                email_sent = await self.automated_reporter.email_report(
                    report_data,
                    self.config.email_recipients,
                    "Daily Trading Report"
                )
                if email_sent:
                    self.logger.info("Daily report emailed successfully")
            
            self.logger.info(f"Daily report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    async def _handle_risk_breach(self, risk_metrics: Any):
        """Handle risk threshold breaches"""
        self.logger.warning("Risk threshold breach detected - taking protective action")
        
        # Temporary trading halt
        self.trading_enabled = False
        
        # Close high-risk positions
        positions = await self.position_manager.get_all_positions()
        for symbol, position in positions.items():
            if abs(position.unrealized_pnl / position.market_value) > 0.1:  # 10% loss
                await self.trading_engine.close_position(symbol)
                self.logger.info(f"Closed high-risk position: {symbol}")
        
        # Send alert
        await self.alerting_system.send_alert(
            AlertLevel.CRITICAL,
            "Risk Threshold Breach",
            f"Risk management triggered protective actions. Current drawdown: {risk_metrics.current_drawdown:.2%}"
        )
        
        # Re-enable trading after cooldown
        await asyncio.sleep(300)  # 5-minute cooldown
        self.trading_enabled = True
        self.logger.info("Trading re-enabled after risk cooldown")
    
    async def _emergency_shutdown(self, reason: str):
        """Emergency shutdown procedure"""
        self.logger.critical(f"Emergency shutdown initiated: {reason}")
        self.status = BotStatus.ERROR
        
        # Stop all trading
        self.trading_enabled = False
        
        # Close all positions
        positions = await self.position_manager.get_all_positions()
        for symbol in positions.keys():
            try:
                await self.trading_engine.close_position(symbol)
                self.logger.info(f"Emergency close position: {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to close position {symbol}: {e}")
        
        # Send critical alert
        await self.alerting_system.send_alert(
            AlertLevel.CRITICAL,
            "Emergency Shutdown",
            f"Trading bot emergency shutdown: {reason}"
        )
    
    async def stop(self):
        """Gracefully stop the trading bot"""
        self.logger.info("Stopping IntegratedTradingBot...")
        self.status = BotStatus.STOPPING
        
        # Signal all loops to stop
        self.stop_event.set()
        
        # Cancel all tasks
        if hasattr(self, 'trading_task'):
            self.trading_task.cancel()
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
        if hasattr(self, 'regime_task'):
            self.regime_task.cancel()
        if hasattr(self, 'news_task'):
            self.news_task.cancel()
        if hasattr(self, 'reporting_task'):
            self.reporting_task.cancel()
        
        # Stop all components
        await self.trading_engine.stop()
        await self.market_data_manager.stop()
        await self.system_monitor.stop()
        
        # Final report
        if self.automated_reporter:
            await self._generate_daily_report()
        
        self.status = BotStatus.STOPPED
        self.logger.info("IntegratedTradingBot stopped successfully")
    
    # ===== STRATEGY GRADUATION METHODS =====
    
    async def register_strategy_for_graduation(
        self,
        strategy_id: str,
        strategy_name: str,
        strategy_instance: Any,
        config: Dict[str, Any],
        start_in_paper: bool = True
    ):
        """Register a new strategy for automatic graduation tracking"""
        
        from .strategy_graduation import StrategyStage
        
        # Register with graduation manager
        initial_stage = StrategyStage.PAPER_VALIDATION if start_in_paper else StrategyStage.LIVE_TRADING
        record = self.graduation_manager.register_strategy(
            strategy_id=strategy_id,
            name=strategy_name,
            config=config,
            initial_stage=initial_stage
        )
        
        # Store strategy instance
        self.active_strategies[strategy_id] = strategy_instance
        
        if start_in_paper:
            self.paper_strategies[strategy_id] = strategy_instance
            self.logger.info(f"Strategy {strategy_name} registered for paper trading validation")
        else:
            self.live_strategies[strategy_id] = strategy_instance
            self.logger.info(f"Strategy {strategy_name} registered for live trading")
        
        return record
    
    async def run_strategy_graduation_cycle(self):
        """Main strategy graduation evaluation cycle"""
        
        try:
            self.logger.info("Starting strategy graduation evaluation cycle")
            
            # Update performance metrics for all active strategies
            await self._update_strategy_performance_metrics()
            
            # Run graduation evaluation
            decisions = await self.graduation_manager.evaluate_all_strategies()
            
            # Execute graduation decisions
            for strategy_id, decision in decisions.items():
                await self._handle_graduation_decision(strategy_id, decision)
            
            # Generate graduation report
            graduation_report = self.graduation_manager.get_graduation_report()
            self.logger.info(f"Graduation cycle complete. Active strategies: {graduation_report['summary']['active_strategies']}")
            
            return graduation_report
            
        except Exception as e:
            self.logger.error(f"Error in strategy graduation cycle: {e}")
            return None
    
    async def _update_strategy_performance_metrics(self):
        """Update performance metrics for all active strategies"""
        
        from .strategy_graduation import PerformanceMetrics
        
        for strategy_id, strategy_instance in self.active_strategies.items():
            try:
                # Get strategy record
                if strategy_id not in self.graduation_manager.strategies:
                    continue
                
                record = self.graduation_manager.strategies[strategy_id]
                
                # Calculate current performance metrics
                # This would integrate with your actual strategy performance tracking
                metrics = await self._calculate_strategy_metrics(strategy_instance, record)
                
                # Update graduation manager
                record.add_performance_snapshot(metrics)
                
            except Exception as e:
                self.logger.error(f"Error updating metrics for strategy {strategy_id}: {e}")
    
    async def _calculate_strategy_metrics(self, strategy_instance: Any, record):
        """Calculate performance metrics for a strategy"""
        
        from .strategy_graduation import PerformanceMetrics
        
        # This is a placeholder implementation
        # In practice, you'd integrate with your actual strategy performance tracking
        
        # Get strategy performance data
        # performance_data = await strategy_instance.get_performance_summary()
        
        # For now, create sample metrics
        # Replace this with actual performance calculation
        metrics = PerformanceMetrics(
            total_return=0.05,  # 5% return
            annualized_return=0.15,
            sharpe_ratio=1.1,
            sortino_ratio=1.3,
            max_drawdown=0.08,
            volatility=0.12,
            win_rate=0.52,
            profit_factor=1.25,
            trades_count=45,
            execution_success_rate=0.98,
            validation_score=0.75,
            confidence_level="MEDIUM"
        )
        
        return metrics
    
    async def _handle_graduation_decision(self, strategy_id: str, decision):
        """Handle graduation decision for a strategy"""
        
        from .strategy_graduation import GraduationDecision, StrategyStage
        
        if strategy_id not in self.active_strategies:
            return
        
        strategy_instance = self.active_strategies[strategy_id]
        record = self.graduation_manager.strategies[strategy_id]
        
        old_stage = record.current_stage
        
        # Execute the decision (already done in graduation manager)
        # Now update our local strategy tracking
        
        new_stage = record.current_stage
        
        if old_stage != new_stage:
            await self._move_strategy_between_stages(strategy_id, old_stage, new_stage)
    
    async def _move_strategy_between_stages(self, strategy_id: str, old_stage, new_stage):
        """Move strategy between paper trading and live trading"""
        
        from .strategy_graduation import StrategyStage
        
        strategy_instance = self.active_strategies[strategy_id]
        record = self.graduation_manager.strategies[strategy_id]
        
        # Remove from old stage tracking
        if old_stage in [StrategyStage.PAPER_VALIDATION, StrategyStage.LIVE_CANDIDATE]:
            if strategy_id in self.paper_strategies:
                del self.paper_strategies[strategy_id]
        elif old_stage in [StrategyStage.LIVE_TRADING, StrategyStage.UNDER_REVIEW]:
            if strategy_id in self.live_strategies:
                del self.live_strategies[strategy_id]
        
        # Add to new stage tracking
        if new_stage in [StrategyStage.PAPER_VALIDATION, StrategyStage.LIVE_CANDIDATE]:
            self.paper_strategies[strategy_id] = strategy_instance
            await self._configure_strategy_for_paper_trading(strategy_instance)
            
        elif new_stage in [StrategyStage.LIVE_TRADING]:
            self.live_strategies[strategy_id] = strategy_instance
            await self._configure_strategy_for_live_trading(strategy_instance, record.allocated_capital)
            
        elif new_stage == StrategyStage.RETIRED:
            # Remove from active strategies
            if strategy_id in self.active_strategies:
                await self._retire_strategy(strategy_instance)
                del self.active_strategies[strategy_id]
        
        self.logger.info(f"Strategy {record.name} moved from {old_stage.value} to {new_stage.value}")
    
    async def _configure_strategy_for_paper_trading(self, strategy_instance):
        """Configure strategy for paper trading mode"""
        
        # Set paper trading parameters
        if hasattr(strategy_instance, 'set_trading_mode'):
            await strategy_instance.set_trading_mode('paper')
        
        if hasattr(strategy_instance, 'set_api_credentials'):
            # Use testnet credentials
            if self.config_manager.config and hasattr(self.config_manager.config, 'exchange'):
                testnet_creds = self.config_manager.config.exchange.get_credentials('development')
                await strategy_instance.set_api_credentials(testnet_creds)
        
        self.logger.info(f"Strategy {strategy_instance} configured for paper trading")
    
    async def _configure_strategy_for_live_trading(self, strategy_instance, allocated_capital: float):
        """Configure strategy for live trading mode"""
        
        # Set live trading parameters
        if hasattr(strategy_instance, 'set_trading_mode'):
            await strategy_instance.set_trading_mode('live')
        
        if hasattr(strategy_instance, 'set_api_credentials'):
            # Use live credentials
            if self.config_manager.config and hasattr(self.config_manager.config, 'exchange'):
                live_creds = self.config_manager.config.exchange.get_credentials('production')
                await strategy_instance.set_api_credentials(live_creds)
        
        if hasattr(strategy_instance, 'set_capital_allocation'):
            await strategy_instance.set_capital_allocation(allocated_capital)
        
        self.logger.info(f"Strategy {strategy_instance} configured for live trading with ${allocated_capital:,.2f}")
    
    async def _retire_strategy(self, strategy_instance):
        """Retire a strategy permanently"""
        
        if hasattr(strategy_instance, 'shutdown'):
            await strategy_instance.shutdown()
        
        self.logger.info(f"Strategy {strategy_instance} retired")
    
    def get_strategy_graduation_status(self) -> Dict[str, Any]:
        """Get current status of strategy graduation system"""
        
        return {
            'graduation_report': self.graduation_manager.get_graduation_report(),
            'paper_strategies_count': len(self.paper_strategies),
            'live_strategies_count': len(self.live_strategies),
            'total_strategies': len(self.active_strategies),
            'paper_strategies': list(self.paper_strategies.keys()),
            'live_strategies': list(self.live_strategies.keys())
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'status': self.status.value,
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'trading_enabled': self.trading_enabled,
            'maintenance_mode': self.maintenance_mode,
            'trade_count': self.trade_count,
            'performance_metrics': self.performance_metrics,
            'system_health': await self.system_monitor.get_system_health() if hasattr(self, 'system_monitor') else None,
            'configuration': {
                'trading_pairs': self.config.trading_pairs,
                'max_position_size': self.config.max_position_size,
                'max_daily_loss': self.config.max_daily_loss,
                'advanced_features_enabled': {
                    'regime_detection': self.config.enable_regime_detection,
                    'portfolio_optimization': self.config.enable_portfolio_optimization,
                    'news_analysis': self.config.enable_news_analysis,
                    'parameter_optimization': self.config.enable_parameter_optimization
                }
            }
        }
    
    async def pause(self):
        """Pause trading operations"""
        self.status = BotStatus.PAUSED
        self.trading_enabled = False
        self.logger.info("Trading bot paused")
    
    async def resume(self):
        """Resume trading operations"""
        self.status = BotStatus.RUNNING
        self.trading_enabled = True
        self.logger.info("Trading bot resumed")
    
    async def update_configuration(self, new_config: Dict[str, Any]):
        """Update bot configuration dynamically"""
        try:
            # Update configuration
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info(f"Updated config: {key} = {value}")
            
            # Reinitialize components if needed
            if any(key.startswith('enable_') for key in new_config.keys()):
                self.logger.info("Reinitializing components after configuration change")
                # Note: In production, this would carefully reinitialize changed components
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise