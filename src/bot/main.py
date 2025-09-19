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

from bot.config import Config
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
def main(
    config_path: str,
    mode: Optional[str],
    debug: bool,
    paper_trade: bool,
    backtest_only: bool,
    dashboard_only: bool
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
        # Load configuration
        logger.info("Loading configuration...")
        config = Config.from_file(config_path)
        
        # Override mode if specified
        if mode:
            config.trading.mode = mode
            logger.info(f"Trading mode overridden to: {mode}")
        
        # Override debug setting
        if debug:
            config.logging.level = "DEBUG"
        
        # Setup logging
        setup_logging(config.logging)
        
        logger.info("=" * 60)
        logger.info("BYBIT TRADING BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Mode: {config.trading.mode.upper()}")
        logger.info(f"Paper Trading: {paper_trade}")
        logger.info(f"Debug: {debug}")
        logger.info(f"Config: {config_path}")
        
        # Initialize database
        logger.info("Initializing database...")
        db_manager = DatabaseManager(config.database)
        db_manager.initialize()
        
        # Create and run the trading bot
        asyncio.run(run_bot(
            config=config,
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
    config: Config,
    db_manager: DatabaseManager,
    paper_trade: bool,
    backtest_only: bool,
    dashboard_only: bool
):
    """
    Main async function to run the trading bot.
    """
    try:
        # Initialize the trading bot
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
        if 'bot' in locals():
            await bot.shutdown()


if __name__ == "__main__":
    main()