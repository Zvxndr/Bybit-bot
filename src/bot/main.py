"""
Main entry point for the Bybit Trading Bot.

This module initializes and runs the complete trading system with proper
error handling, logging, and graceful shutdown capabilities.
"""

import asyncio
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

import click
from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import unified configuration system
try:
    from bot.core.config.manager import UnifiedConfigurationManager
    from bot.core.config.schema import UnifiedConfigurationSchema
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("Unified configuration system not available, falling back to legacy config")
    from bot.config import Config
    UNIFIED_CONFIG_AVAILABLE = False

from bot.core import TradingBot
from bot.database import DatabaseManager
from bot.utils.logging import setup_logging


class GracefulShutdown:
    """Handle graceful shutdown of the trading bot."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.bot: Optional[TradingBot] = None
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        if self.bot:
            asyncio.create_task(self.bot.shutdown())
    
    def register_bot(self, bot: TradingBot):
        """Register the trading bot for shutdown handling."""
        self.bot = bot


# Global shutdown handler
shutdown_handler = GracefulShutdown()


@click.command()
@click.option(
    "--config-path",
    default="config/config.yaml",
    help="Path to configuration file",
    type=click.Path(exists=True)
)
@click.option(
    "--mode",
    type=click.Choice(["conservative", "aggressive"]),
    help="Trading mode (overrides config file)"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
@click.option(
    "--paper-trade",
    is_flag=True,
    help="Run in paper trading mode"
)
@click.option(
    "--backtest-only",
    is_flag=True,
    help="Run backtesting only, no live trading"
)
@click.option(
    "--dashboard-only",
    is_flag=True,
    help="Run dashboard only, no trading"
)
@click.option(
    "--unified-config",
    help="Path to unified configuration file (.yaml or .json)",
    type=click.Path(exists=True)
)
@click.option(
    "--config-env",
    default="default",
    help="Configuration environment to use (default, production, staging, etc.)"
)
def main(
    config_path: str,
    mode: Optional[str],
    debug: bool,
    paper_trade: bool,
    backtest_only: bool,
    dashboard_only: bool,
    unified_config: Optional[str],
    config_env: str
):
    """
    Start the Bybit Trading Bot.
    
    The bot supports multiple operational modes:
    - Full trading with risk management
    - Paper trading for testing
    - Backtesting for strategy validation
    - Dashboard-only for monitoring
    """
    try:
        # Load configuration (prefer unified config)
        unified_config_schema = None
        config = None
        
        if unified_config and UNIFIED_CONFIG_AVAILABLE:
            logger.info(f"Loading unified configuration from {unified_config}...")
            config_manager = UnifiedConfigurationManager()
            unified_config_schema = config_manager.load_configuration(
                config_path=unified_config,
                environment=config_env
            )
            logger.info(f"Loaded unified configuration for environment: {config_env}")
            
        elif UNIFIED_CONFIG_AVAILABLE:
            # Try to load default unified config
            logger.info("Loading default unified configuration...")
            config_manager = UnifiedConfigurationManager()
            try:
                unified_config_schema = config_manager.load_configuration(environment=config_env)
                logger.info(f"Loaded default unified configuration for environment: {config_env}")
            except Exception as e:
                logger.warning(f"Failed to load unified configuration: {e}")
                logger.info("Falling back to legacy configuration...")
                from bot.config import Config
                config = Config.from_file(config_path)
        else:
            # Fall back to legacy configuration
            logger.info(f"Loading legacy configuration from {config_path}...")
            config = Config.from_file(config_path)
        
        # Override mode if specified (for legacy config)
        if config and mode and mode in ["conservative", "aggressive"]:
            config.trading.mode = mode  # type: ignore
            logger.info(f"Trading mode overridden to: {mode}")
        elif config and mode:
            logger.warning(f"Invalid trading mode '{mode}'. Using default: {config.trading.mode}")
        
        # Override debug setting (for legacy config)
        if config and debug:
            config.logging.level = "DEBUG"
        
        # Setup logging
        if config:
            setup_logging(config.logging)
        else:
            # Use basic logging for unified config
            import logging
            logging.basicConfig(level=logging.INFO if not debug else logging.DEBUG)
        
        logger.info("=" * 60)
        logger.info("BYBIT TRADING BOT STARTING")
        logger.info("=" * 60)
        
        if unified_config_schema:
            logger.info(f"Configuration: Unified Config (Environment: {config_env})")
            logger.info(f"Paper Trading: {paper_trade}")
            logger.info(f"Debug: {debug}")
        else:
            logger.info(f"Configuration: Legacy Config ({config_path})")
            logger.info(f"Mode: {config.trading.mode.upper()}")
            logger.info(f"Paper Trading: {paper_trade}")
            logger.info(f"Debug: {debug}")
        
        # Initialize database (handle both config types)
        logger.info("Initializing database...")
        if config:
            db_manager = DatabaseManager(config.database)
        else:
            # Use unified config database settings or defaults
            db_manager = DatabaseManager(None)  # This will need to be updated when DatabaseManager supports unified config
        
        db_manager.initialize()
        
        # Create and run the trading bot
        asyncio.run(run_bot(
            config=config,
            unified_config=unified_config_schema,
            db_manager=db_manager,
            paper_trade=paper_trade,
            backtest_only=backtest_only,
            dashboard_only=dashboard_only
        ))
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("Trading bot shutdown complete")


async def run_bot(
    config: Optional['Config'],
    unified_config: Optional[UnifiedConfigurationSchema],
    db_manager: DatabaseManager,
    paper_trade: bool,
    backtest_only: bool,
    dashboard_only: bool
):
    """
    Main async function to run the trading bot.
    """
    bot = None
    try:
        # Initialize the trading bot (prefer unified config)
        if unified_config:
            bot = TradingBot(
                unified_config=unified_config,
                db_manager=db_manager,
                paper_trade=paper_trade,
                backtest_only=backtest_only,
                dashboard_only=dashboard_only
            )
        else:
            bot = TradingBot(
                config=config,
                db_manager=db_manager,
                paper_trade=paper_trade,
                backtest_only=backtest_only,
                dashboard_only=dashboard_only
            )
        
        # Register for graceful shutdown
        shutdown_handler.register_bot(bot)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
        signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)
        
        # Initialize all components
        await bot.initialize()
        
        # Start the bot
        logger.info("Starting trading bot main loop...")
        await bot.run()
        
    except Exception as e:
        logger.error(f"Error in bot execution: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Ensure cleanup
        if bot is not None:
            try:
                await bot.shutdown()
            except Exception as e:
                logger.error(f"Error during bot shutdown: {e}")


if __name__ == "__main__":
    main()