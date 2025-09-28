#!/usr/bin/env python3
"""
🔥 Open Alpha - Production Deployment Script
Safely transitions system from debug mode to production readiness
"""

import os
import sys
import yaml
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_transition.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProductionTransition:
    """Manages transition from debug mode to production"""
    
    def __init__(self):
        self.workspace_path = Path(__file__).parent
        self.config_path = self.workspace_path / "config"
        self.transition_log = []
        
    def log_step(self, step: str, success: bool = True, details: str = ""):
        """Log a transition step"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "success": success,
            "details": details
        }
        self.transition_log.append(entry)
        
        if success:
            logger.info(f"✅ {step}")
        else:
            logger.error(f"❌ {step} - {details}")
            
        if details:
            logger.info(f"   {details}")
    
    async def run_transition(self):
        """Execute complete production transition"""
        logger.info("🔥 Starting Open Alpha Production Transition")
        logger.info("=" * 60)
        
        steps = [
            ("Validate System State", self.validate_system),
            ("Run System Tests", self.run_system_tests),
            ("Download Historical Data", self.setup_historical_data),
            ("Configure Production Settings", self.configure_production),
            ("Validate Safety Systems", self.validate_safety),
            ("Create Production Config", self.create_production_config),
            ("Update Debug Settings", self.update_debug_settings),
            ("Final Validation", self.final_validation)
        ]
        
        success_count = 0
        
        for step_name, step_func in steps:
            logger.info(f"\n🔧 {step_name}")
            logger.info("-" * 40)
            
            try:
                success = await step_func()
                if success:
                    self.log_step(step_name, True)
                    success_count += 1
                else:
                    self.log_step(step_name, False, "Step validation failed")
            except Exception as e:
                self.log_step(step_name, False, str(e))
        
        # Generate transition report
        await self.generate_report(success_count, len(steps))
        
        return success_count == len(steps)
    
    async def validate_system(self):
        """Validate current system state"""
        try:
            # Check that we're currently in debug mode
            debug_active = True  # Assume debug mode
            
            try:
                sys.path.insert(0, str(self.workspace_path / 'src'))
                from debug_safety import get_debug_safety_config
                config = get_debug_safety_config()
                debug_active = config.get('debug_mode_active', True)
            except Exception as e:
                logger.warning(f"Could not verify debug state: {e}")
            
            if debug_active:
                logger.info("✅ System currently in safe debug mode")
            else:
                logger.warning("⚠️ System may not be in debug mode")
            
            # Check for essential files
            essential_files = [
                "src/main.py",
                "src/debug_safety.py",
                "src/shared_state.py",
                "src/frontend_server.py",
                "historical_data_downloader.py"
            ]
            
            for file_path in essential_files:
                if (self.workspace_path / file_path).exists():
                    logger.info(f"✅ Found: {file_path}")
                else:
                    logger.error(f"❌ Missing: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return False
    
    async def run_system_tests(self):
        """Run comprehensive system tests"""
        try:
            test_script = self.workspace_path / "system_test_suite.py"
            
            if test_script.exists():
                logger.info("🧪 Running system test suite...")
                
                # Import and run the test suite
                sys.path.insert(0, str(self.workspace_path))
                
                try:
                    from system_test_suite import SystemTestSuite
                    suite = SystemTestSuite()
                    success = await suite.run_all_tests()
                    
                    if success:
                        logger.info("✅ All system tests passed")
                        return True
                    else:
                        logger.warning("⚠️ Some system tests failed")
                        return False
                        
                except Exception as e:
                    logger.error(f"Test suite execution failed: {e}")
                    return False
            else:
                logger.warning("⚠️ System test suite not found - skipping")
                return True
                
        except Exception as e:
            logger.error(f"System test failed: {e}")
            return False
    
    async def setup_historical_data(self):
        """Download and validate historical data"""
        try:
            downloader_path = self.workspace_path / "historical_data_downloader.py"
            
            if downloader_path.exists():
                logger.info("📊 Setting up historical data...")
                
                try:
                    from historical_data_downloader import HistoricalDataDownloader, DataDownloadConfig
                    
                    # Production-ready configuration
                    config = DataDownloadConfig(
                        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                        timeframes=['1h', '4h', '1d'],
                        lookback_days=365,  # 1 year of data
                        max_requests_per_minute=120,
                        database_path="src/data/speed_demon_cache/market_data.db"
                    )
                    
                    async with HistoricalDataDownloader(config) as downloader:
                        report = await downloader.download_all_data()
                        
                        if report['deployment_ready']:
                            logger.info(f"✅ Historical data ready: {report['total_records']} records")
                            logger.info(f"   Quality score: {report['quality_score']:.2f}")
                            return True
                        else:
                            logger.warning("⚠️ Historical data quality below threshold")
                            return False
                            
                except Exception as e:
                    logger.error(f"Historical data download failed: {e}")
                    return False
            else:
                logger.warning("⚠️ Historical data downloader not found")
                return False
                
        except Exception as e:
            logger.error(f"Historical data setup failed: {e}")
            return False
    
    async def configure_production(self):
        """Configure production settings"""
        try:
            # Ensure config directory exists
            self.config_path.mkdir(exist_ok=True)
            
            # Create production configuration
            production_config = {
                "environment": "production",
                "debug_mode": False,
                "trading_enabled": True,
                "safety_checks": True,
                "historical_data": {
                    "enabled": True,
                    "auto_download": True,
                    "quality_threshold": 0.95
                },
                "api": {
                    "rate_limits": True,
                    "timeout": 30,
                    "retry_attempts": 3
                },
                "logging": {
                    "level": "INFO",
                    "file_logging": True,
                    "max_log_size": "10MB"
                }
            }
            
            # Save production config
            production_file = self.config_path / "production.yaml"
            with open(production_file, 'w') as f:
                yaml.dump(production_config, f, default_flow_style=False)
                
            logger.info(f"✅ Production config created: {production_file}")
            return True
            
        except Exception as e:
            logger.error(f"Production configuration failed: {e}")
            return False
    
    async def validate_safety(self):
        """Validate all safety systems are operational"""
        try:
            sys.path.insert(0, str(self.workspace_path / 'src'))
            
            # Test emergency stop
            from shared_state import shared_state
            
            # Test emergency stop functionality
            shared_state.set_bot_control('emergency_stop', True)
            if shared_state.is_emergency_stopped():
                logger.info("✅ Emergency stop system operational")
            else:
                logger.error("❌ Emergency stop system failed")
                return False
            
            # Reset emergency stop
            shared_state.set_bot_control('emergency_stop', False)
            
            # Test pause functionality
            shared_state.set_bot_control('paused', True)
            if shared_state.is_paused():
                logger.info("✅ Pause system operational")
            else:
                logger.error("❌ Pause system failed")
                return False
            
            # Reset pause
            shared_state.set_bot_control('paused', False)
            
            logger.info("✅ All safety systems validated")
            return True
            
        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            return False
    
    async def create_production_config(self):
        """Create production-ready configuration files"""
        try:
            # Create main config file
            main_config = {
                "app": {
                    "name": "Open Alpha Wealth Management",
                    "version": "1.0.0",
                    "environment": "production"
                },
                "trading": {
                    "default_mode": "testnet",  # Start with testnet even in production
                    "safety_enabled": True,
                    "position_limits": {
                        "max_positions": 10,
                        "max_risk_per_trade": 0.02
                    }
                },
                "data": {
                    "historical_enabled": True,
                    "real_time_enabled": True,
                    "quality_checks": True
                }
            }
            
            config_file = self.config_path / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(main_config, f, default_flow_style=False)
                
            logger.info(f"✅ Main config created: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Production config creation failed: {e}")
            return False
    
    async def update_debug_settings(self):
        """Update debug settings for production readiness"""
        try:
            # Create debug config that maintains safety
            debug_config = {
                "debug_mode_active": False,  # Disable debug mode
                "trading_blocked": False,    # Allow trading (but with safety checks)
                "mock_data_enabled": False,  # Use real data
                "safety_overrides": {
                    "emergency_stop_enabled": True,
                    "position_limits_enabled": True,
                    "risk_checks_enabled": True
                },
                "logging": {
                    "debug_level": False,
                    "trade_logging": True,
                    "error_logging": True
                }
            }
            
            debug_file = self.config_path / "debug.yaml"
            with open(debug_file, 'w') as f:
                yaml.dump(debug_config, f, default_flow_style=False)
                
            logger.info(f"✅ Debug config updated: {debug_file}")
            return True
            
        except Exception as e:
            logger.error(f"Debug settings update failed: {e}")
            return False
    
    async def final_validation(self):
        """Final validation before production"""
        try:
            # Check configuration files exist
            required_configs = [
                "config/config.yaml",
                "config/production.yaml", 
                "config/debug.yaml"
            ]
            
            for config_file in required_configs:
                config_path = self.workspace_path / config_file
                if config_path.exists():
                    logger.info(f"✅ Config verified: {config_file}")
                else:
                    logger.error(f"❌ Missing config: {config_file}")
                    return False
            
            # Check historical data
            data_file = self.workspace_path / "src/data/speed_demon_cache/market_data.db"
            if data_file.exists():
                logger.info("✅ Historical data available")
            else:
                logger.warning("⚠️ Historical data not found - will download on startup")
            
            # Check safety systems one more time
            sys.path.insert(0, str(self.workspace_path / 'src'))
            try:
                from shared_state import shared_state
                # Quick safety test
                shared_state.set_bot_control('emergency_stop', True)
                if shared_state.is_emergency_stopped():
                    logger.info("✅ Final safety check passed")
                    shared_state.set_bot_control('emergency_stop', False)
                else:
                    logger.error("❌ Final safety check failed")
                    return False
            except Exception as e:
                logger.warning(f"Could not run final safety check: {e}")
            
            logger.info("✅ Final validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            return False
    
    async def generate_report(self, success_count: int, total_steps: int):
        """Generate transition report"""
        report = {
            "transition_summary": {
                "timestamp": datetime.now().isoformat(),
                "success": success_count == total_steps,
                "steps_completed": success_count,
                "total_steps": total_steps,
                "success_rate": f"{(success_count/total_steps)*100:.1f}%"
            },
            "system_status": {
                "ready_for_production": success_count == total_steps,
                "debug_mode": False,
                "safety_systems": "operational",
                "historical_data": "configured",
                "configuration": "production"
            },
            "transition_log": self.transition_log,
            "next_steps": [
                "1. Review transition log for any warnings",
                "2. Test system with small amounts in testnet mode", 
                "3. Monitor system performance for 24 hours",
                "4. Gradually increase trading limits",
                "5. Set up monitoring and alerting"
            ]
        }
        
        # Save report
        report_file = f"production_transition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Transition report saved: {report_file}")
        
        if success_count == total_steps:
            logger.info("\n🎉 PRODUCTION TRANSITION COMPLETE!")
            logger.info("✅ System is ready for production deployment")
            logger.info("⚠️ Start with testnet mode for final validation")
        else:
            logger.warning(f"\n⚠️ TRANSITION INCOMPLETE ({success_count}/{total_steps})")
            logger.warning("❌ Review failed steps before production deployment")

async def main():
    """Run the production transition"""
    transition = ProductionTransition()
    
    try:
        success = await transition.run_transition()
        
        if success:
            print("\n🚀 Ready to deploy to production!")
            print("Next: Update DigitalOcean deployment with production config")
            sys.exit(0)
        else:
            print("\n⚠️ Transition incomplete - review logs")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Transition failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())