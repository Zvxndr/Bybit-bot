"""
Core trading bot module.

This module contains the main TradingBot class that orchestrates
all components of the trading system.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

# Import unified configuration system
try:
    from .core.config.manager import UnifiedConfigurationManager
    from .core.config.schema import UnifiedConfigurationSchema
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False

from .config import Config
from .database import DatabaseManager
from .data import DataProvider, DataCollector, DataSanitizer
from .utils.logging import TradingLogger


class TradingBot:
    """
    Main trading bot class that orchestrates all components.
    
    This class manages the complete lifecycle of the trading bot including:
    - Component initialization
    - Data collection and management
    - Strategy execution
    - Risk management
    - Performance monitoring
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        unified_config: Optional[UnifiedConfigurationSchema] = None,
        db_manager: DatabaseManager = None,
        paper_trade: bool = True,
        backtest_only: bool = False,
        dashboard_only: bool = False
    ):
        # Configuration handling (unified config takes precedence)
        if unified_config and UNIFIED_CONFIG_AVAILABLE:
            self.unified_config = unified_config
            self.config = None
            self.logger = TradingLogger("TradingBot")
            self.logger.info("TradingBot initialized with unified configuration")
        elif config:
            self.config = config
            self.unified_config = None
            self.logger = TradingLogger("TradingBot")
            self.logger.info("TradingBot initialized with legacy configuration")
        else:
            raise ValueError("Either config or unified_config must be provided")
        
        self.db_manager = db_manager
        self.paper_trade = paper_trade
        self.backtest_only = backtest_only
        self.dashboard_only = dashboard_only
        
        # Component initialization flags
        self._initialized = False
        self._running = False
        
        # Core components (will be initialized in initialize())
        self.data_sanitizer: Optional[DataSanitizer] = None
        self.data_collector: Optional[DataCollector] = None
        self.data_provider: Optional[DataProvider] = None
        
        # Trading components - now implemented
        self.strategy_manager = None
        self.risk_manager = None
        self.trading_engine = None
        self.portfolio_manager = None
        
        # Statistics
        self.start_time: Optional[datetime] = None
        self.stats = {
            'uptime_seconds': 0,
            'strategies_active': 0,
            'trades_executed': 0,
            'current_mode': config.trading.mode
        }
    
    async def initialize(self) -> None:
        """Initialize all bot components."""
        if self._initialized:
            self.logger.warning("Bot already initialized")
            return
        
        self.logger.info("Initializing trading bot components...")
        
        try:
            # Initialize data components
            self.data_sanitizer = DataSanitizer()
            
            if not self.dashboard_only:
                self.data_collector = DataCollector(
                    self.config.exchange,
                    self.db_manager,
                    self.data_sanitizer
                )
            
            self.data_provider = DataProvider(
                self.db_manager,
                self.data_collector
            )
            
            # Initialize core trading components if available
            if not self.dashboard_only:
                try:
                    from .config_manager import ConfigurationManager
                    from .risk_management.risk_manager import RiskManager
                    from decimal import Decimal
                    
                    # Initialize configuration manager
                    config_manager = ConfigurationManager("config/config.yaml")
                    
                    # Initialize risk manager
                    self.risk_manager = RiskManager(config_manager)
                    
                    # Initialize portfolio manager
                    from .risk_management.portfolio_manager import PortfolioManager
                    initial_balance = Decimal('10000')  # Default balance
                    self.portfolio_manager = PortfolioManager(
                        config_manager=config_manager,
                        risk_manager=self.risk_manager,
                        initial_balance=initial_balance
                    )
                    
                    self.logger.info("Core trading components initialized successfully")
                    
                except ImportError as e:
                    self.logger.warning(f"Some trading components not available: {e}")
                    self.logger.info("Running in limited mode - data collection only")
            
            self._initialized = True
            self.logger.info("Bot initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Bot initialization failed: {e}")
            raise
    
    async def run(self) -> None:
        """Main bot execution loop."""
        if not self._initialized:
            raise RuntimeError("Bot not initialized. Call initialize() first.")
        
        self.start_time = datetime.utcnow()
        self._running = True
        
        self.logger.info("Starting trading bot main loop...")
        self.logger.info(f"Mode: {self.config.trading.mode}")
        self.logger.info(f"Paper trading: {self.paper_trade}")
        self.logger.info(f"Backtest only: {self.backtest_only}")
        self.logger.info(f"Dashboard only: {self.dashboard_only}")
        
        try:
            if self.dashboard_only:
                await self._run_dashboard_mode()
            elif self.backtest_only:
                await self._run_backtest_mode()
            else:
                await self._run_trading_mode()
        
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            raise
        finally:
            self._running = False
    
    async def _run_dashboard_mode(self) -> None:
        """Run in dashboard-only mode."""
        self.logger.info("Running in dashboard-only mode")
        
        # TODO: Start Streamlit dashboard
        # For now, just keep the bot alive
        while self._running:
            await asyncio.sleep(10)
            self._update_stats()
    
    async def _run_backtest_mode(self) -> None:
        """Run in backtest-only mode."""
        self.logger.info("Running in backtest-only mode")
        
        # TODO: Implement backtesting logic
        # For now, placeholder
        self.logger.info("Backtest mode not yet implemented")
        await asyncio.sleep(1)
    
    async def _run_trading_mode(self) -> None:
        """Run in full trading mode."""
        self.logger.info(f"Running in trading mode (paper: {self.paper_trade})")
        
        # Start trading components if available
        if self.risk_manager and self.portfolio_manager:
            self.logger.info("Starting trading components...")
            try:
                # Update daily performance tracking
                await self.portfolio_manager.update_daily_performance()
                self.logger.info("Trading components started successfully")
            except Exception as e:
                self.logger.error(f"Error starting trading components: {e}")
        
        # Main trading loop
        while self._running:
            try:
                # Update statistics
                self._update_stats()
                
                if self.risk_manager and self.portfolio_manager:
                    # Update portfolio performance
                    await self.portfolio_manager.update_daily_performance()
                    
                    # Get portfolio performance metrics
                    performance = await self.portfolio_manager.calculate_performance_metrics()
                    
                    # Calculate current risk metrics
                    portfolio_value = self.portfolio_manager.get_total_value()
                    risk_metrics = await self.risk_manager.calculate_risk_metrics(portfolio_value)
                    
                    # Log current status
                    self.logger.info(
                        f"Portfolio Value: {performance.total_value:.2f} USDT, "
                        f"Total PnL: {performance.total_pnl:.2f} ({performance.total_pnl_percentage:.2f}%), "
                        f"Risk Level: {risk_metrics.risk_level.value}"
                    )
                    
                    # Update internal stats
                    self.stats.update({
                        'portfolio_value': float(performance.total_value),
                        'total_pnl': float(performance.total_pnl),
                        'risk_level': risk_metrics.risk_level.value,
                        'open_positions': performance.total_trades  # Using total trades as proxy for now
                    })
                    
                    # Check for rebalancing needs
                    rebalance_needed = await self.portfolio_manager.check_rebalancing_needed()
                    if rebalance_needed:
                        self.logger.info(f"Rebalancing needed for: {', '.join(rebalance_needed)}")
                
                else:
                    # Limited mode - just log status
                    self.logger.debug("Trading loop iteration completed (limited mode)")
                
                # Sleep between iterations
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def _update_stats(self) -> None:
        """Update bot statistics."""
        if self.start_time:
            self.stats['uptime_seconds'] = (datetime.utcnow() - self.start_time).total_seconds()
        
        # TODO: Update other statistics
        # - Active strategies count
        # - Trades executed
        # - Current portfolio value
        # - Risk metrics
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the bot."""
        self.logger.info("Initiating bot shutdown...")
        
        self._running = False
        
        try:
            # Stop trading components
            if hasattr(self, 'trading_engine') and self.trading_engine:
                await self.trading_engine.stop()
                self.logger.info("Trading engine stopped")
            
            if hasattr(self, 'strategy_manager') and self.strategy_manager:
                # Stop all active strategies
                for strategy_id in list(self.strategy_manager.strategies.keys()):
                    await self.strategy_manager.stop_strategy(strategy_id)
                self.logger.info("All strategies stopped")
            
            # Final portfolio update
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                await self.portfolio_manager.update_daily_performance()
                final_performance = await self.portfolio_manager.calculate_performance_metrics()
                self.logger.info(
                    f"Final Portfolio: {final_performance.total_value:.2f} USDT, "
                    f"Total PnL: {final_performance.total_pnl:.2f} ({final_performance.total_pnl_percentage:.2f}%)"
                )
            
        except Exception as e:
            self.logger.error(f"Error during component shutdown: {e}")
        
        # Close database connections
        if self.db_manager:
            self.db_manager.close()
        
        self.logger.info("Bot shutdown completed")
    
    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            'initialized': self._initialized,
            'running': self._running,
            'mode': self.config.trading.mode,
            'paper_trade': self.paper_trade,
            'backtest_only': self.backtest_only,
            'dashboard_only': self.dashboard_only,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stats': self.stats.copy()
        }
    
    def get_health(self) -> Dict:
        """Get bot health status."""
        health = {
            'status': 'healthy' if self._running else 'stopped',
            'components': {}
        }
        
        # Check database health
        if self.db_manager:
            health['components']['database'] = self.db_manager.health_check()
        
        # Check data collector health
        if self.data_collector:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                health['components']['data_collector'] = loop.run_until_complete(
                    self.data_collector.health_check()
                )
            except Exception as e:
                health['components']['data_collector'] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        # TODO: Check other components
        
        return health