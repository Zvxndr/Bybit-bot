#!/usr/bin/env python3
"""
Phase 5 Live Trading System - Quick Start Guide

This script demonstrates how to start and use the Phase 5 live trading system.
It provides examples for different deployment scenarios and trading modes.

Usage:
    python phase5_quickstart.py [mode] [environment]
    
    mode: paper (default), live, hybrid
    environment: development (default), staging, production

Examples:
    python phase5_quickstart.py paper development    # Safe paper trading
    python phase5_quickstart.py hybrid staging       # Hybrid mode in staging
    python phase5_quickstart.py live production      # Live trading in production

Author: Trading Bot Team
Version: 1.0.0 - Phase 5 Implementation
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot.config_manager import ConfigurationManager
from src.bot.live_trading.phase5_orchestrator import Phase5LiveTradingOrchestrator
from src.bot.live_trading import TradingMode, DeploymentEnvironment


def print_banner():
    """Print Phase 5 startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🚀 PHASE 5 LIVE TRADING SYSTEM - QUICK START                            ║
║                                                                              ║
║     Production-Ready Live Trading with Real-Time Monitoring                 ║
║     ✅ WebSocket Data Feeds  ✅ Multi-Mode Execution                        ║
║     ✅ Live Monitoring       ✅ Alert System                                ║
║     ✅ Risk Management       ✅ Emergency Controls                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
🔧 PHASE 5 QUICK START GUIDE

📋 Available Trading Modes:
   • paper      - Paper trading (simulation only, no real money)
   • hybrid     - Limited live trading with paper fallback
   • live       - Full live trading (USE WITH CAUTION)

🏗️ Available Environments:
   • development - Local development with mock data
   • staging     - Pre-production testing environment
   • production  - Live production environment

⚠️  SAFETY RECOMMENDATIONS:
   • ALWAYS start with paper mode for testing
   • Test thoroughly in staging before production
   • Use live mode only after extensive paper trading
   • Keep emergency stop procedures ready

📊 Monitoring:
   • Dashboard: http://localhost:8080 (after startup)
   • Logs: Check console output for system status
   • Alerts: Monitor for risk and system notifications

🛑 Emergency Procedures:
   • Ctrl+C for graceful shutdown
   • Emergency stop via dashboard
   • Mode switching via orchestrator API

💡 Example Usage:
   python phase5_quickstart.py paper development    # Safe testing
   python phase5_quickstart.py hybrid staging       # Limited live testing
   python phase5_quickstart.py live production      # Production trading
    """
    print(help_text)


def validate_configuration() -> bool:
    """Validate that required configuration is present."""
    print("🔍 Validating configuration...")
    
    # Check for required configuration files
    config_files = [
        project_root / "config" / "config.yaml",
        project_root / "config" / "secrets.yaml"
    ]
    
    missing_files = []
    for config_file in config_files:
        if not config_file.exists():
            missing_files.append(str(config_file))
    
    if missing_files:
        print(f"❌ Missing configuration files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Please create the required configuration files before starting.")
        return False
    
    # Validate environment variables
    required_env_vars = ["BYBIT_API_KEY", "BYBIT_API_SECRET"]
    missing_env_vars = []
    
    for env_var in required_env_vars:
        if not os.getenv(env_var):
            missing_env_vars.append(env_var)
    
    if missing_env_vars:
        print(f"❌ Missing environment variables:")
        for var in missing_env_vars:
            print(f"   • {var}")
        print("\n💡 Please set the required environment variables.")
        return False
    
    print("✅ Configuration validation passed")
    return True


def parse_arguments() -> tuple:
    """Parse command line arguments."""
    # Default values
    trading_mode = TradingMode.PAPER
    environment = DeploymentEnvironment.DEVELOPMENT
    
    # Parse trading mode
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        if mode_arg == "live":
            trading_mode = TradingMode.LIVE
        elif mode_arg == "hybrid":
            trading_mode = TradingMode.HYBRID
        elif mode_arg == "paper":
            trading_mode = TradingMode.PAPER
        elif mode_arg in ["help", "-h", "--help"]:
            print_help()
            sys.exit(0)
        else:
            print(f"❌ Unknown trading mode: {mode_arg}")
            print("   Available modes: paper, hybrid, live")
            sys.exit(1)
    
    # Parse environment
    if len(sys.argv) > 2:
        env_arg = sys.argv[2].lower()
        if env_arg == "production":
            environment = DeploymentEnvironment.PRODUCTION
        elif env_arg == "staging":
            environment = DeploymentEnvironment.STAGING
        elif env_arg == "development":
            environment = DeploymentEnvironment.DEVELOPMENT
        else:
            print(f"❌ Unknown environment: {env_arg}")
            print("   Available environments: development, staging, production")
            sys.exit(1)
    
    return trading_mode, environment


def confirm_live_trading(trading_mode: TradingMode, environment: DeploymentEnvironment) -> bool:
    """Confirm live trading with user."""
    if trading_mode == TradingMode.LIVE and environment == DeploymentEnvironment.PRODUCTION:
        print("\n⚠️  LIVE TRADING CONFIRMATION REQUIRED")
        print("   You are about to start LIVE TRADING in PRODUCTION")
        print("   This will place REAL ORDERS with REAL MONEY")
        print("\n❓ Are you absolutely sure you want to proceed?")
        
        confirmation = input("   Type 'CONFIRM LIVE TRADING' to proceed: ")
        
        if confirmation != "CONFIRM LIVE TRADING":
            print("❌ Live trading cancelled - confirmation not provided")
            return False
    
    elif trading_mode in [TradingMode.LIVE, TradingMode.HYBRID]:
        print(f"\n⚠️  {trading_mode.value.upper()} TRADING WARNING")
        print(f"   Starting {trading_mode.value} trading in {environment.value}")
        
        if trading_mode == TradingMode.LIVE:
            print("   This will place REAL ORDERS with REAL MONEY")
        else:
            print("   This may place limited real orders")
        
        confirmation = input("   Type 'YES' to proceed: ")
        
        if confirmation.upper() != "YES":
            print("❌ Trading cancelled")
            return False
    
    return True


def print_startup_info(trading_mode: TradingMode, environment: DeploymentEnvironment):
    """Print startup information."""
    mode_emoji = {
        TradingMode.PAPER: "📝",
        TradingMode.HYBRID: "⚡", 
        TradingMode.LIVE: "🔴"
    }
    
    env_emoji = {
        DeploymentEnvironment.DEVELOPMENT: "🛠️",
        DeploymentEnvironment.STAGING: "🧪",
        DeploymentEnvironment.PRODUCTION: "🏭"
    }
    
    print(f"\n🚀 STARTING PHASE 5 SYSTEM")
    print(f"   {mode_emoji[trading_mode]} Trading Mode: {trading_mode.value.upper()}")
    print(f"   {env_emoji[environment]} Environment: {environment.value.upper()}")
    print(f"\n📊 Dashboard will be available at: http://localhost:8080")
    print(f"🔄 Starting system components...")


def print_running_info():
    """Print information while system is running."""
    info = """
✅ PHASE 5 SYSTEM IS RUNNING

📊 Monitoring:
   • Dashboard: http://localhost:8080
   • Logs: Monitor console output
   • Health: Check component status

🎛️  Controls:
   • Ctrl+C: Graceful shutdown
   • Dashboard: Emergency stop, mode switching
   • API: Programmatic control

⚠️  Safety:
   • Emergency stop available at any time
   • Real-time risk monitoring active
   • Alert system monitoring for issues

💡 Tips:
   • Monitor the dashboard for performance metrics
   • Watch for alerts and system notifications
   • Use emergency stop if needed

Press Ctrl+C to stop the system gracefully...
    """
    print(info)


async def main():
    """Main entry point for Phase 5 quick start."""
    try:
        # Print banner
        print_banner()
        
        # Parse arguments
        trading_mode, environment = parse_arguments()
        
        # Validate configuration
        if not validate_configuration():
            sys.exit(1)
        
        # Confirm live trading if needed
        if not confirm_live_trading(trading_mode, environment):
            sys.exit(0)
        
        # Print startup info
        print_startup_info(trading_mode, environment)
        
        # Load configuration
        config = ConfigurationManager()
        
        # Apply environment-specific overrides
        if environment == DeploymentEnvironment.STAGING:
            config.update({
                'trading.max_position_size': 0.01,  # Smaller positions in staging
                'risk_management.max_drawdown': 0.05,  # Lower risk limits
            })
        elif environment == DeploymentEnvironment.PRODUCTION:
            config.update({
                'alerts.email.enabled': True,  # Enable email alerts in production
                'monitoring.dashboard.public': False,  # Secure dashboard
            })
        
        # Create orchestrator
        orchestrator = Phase5LiveTradingOrchestrator(config, trading_mode)
        
        # Start system
        if await orchestrator.start():
            print_running_info()
            
            # Keep running until interrupted
            try:
                while orchestrator.running and not orchestrator.shutdown_requested:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested by user...")
        else:
            print("❌ Failed to start Phase 5 system")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error starting Phase 5 system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the Phase 5 system
    asyncio.run(main())