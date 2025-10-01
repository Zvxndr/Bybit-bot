"""
Private Use Final Setup Script

This script completes the final configuration for private use of the Bybit Trading Bot
with full ML integration. It addresses all remaining setup requirements and ensures
the system is ready for personal trading operations.

Features:
- Private API configuration setup
- Personal risk parameters
- Private key management
- Local development optimization
- Final system validation

Author: Trading Bot Team
Version: 1.0.0 - Private Use Edition
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivateUseSetup:
    """Handles final setup for private use deployment."""
    
    def __init__(self):
        self.setup_dir = Path(__file__).parent
        self.config_dir = self.setup_dir / 'config'
        self.secrets_template = self.config_dir / 'secrets.yaml.template'
        self.secrets_file = self.config_dir / 'secrets.yaml'
        
        self.logger = logger
        self.setup_complete = False
        
    def run_final_setup(self) -> bool:
        """Execute complete final setup for private use."""
        
        print("\n" + "="*60)
        print("🔥 BYBIT BOT - FINAL PRIVATE USE SETUP 🔥")
        print("="*60)
        print("Configuring system for personal trading operations...")
        print()
        
        setup_steps = [
            ("🔐 Configure Private API Settings", self._setup_private_api),
            ("⚙️ Set Personal Risk Parameters", self._configure_risk_params),
            ("🛡️ Setup Security Configuration", self._setup_security),
            ("📊 Configure Private Dashboard", self._setup_private_dashboard),
            ("🚀 Optimize for Local Development", self._optimize_local_dev),
            ("✅ Validate Final Configuration", self._validate_final_config),
            ("📝 Generate User Guide", self._generate_user_guide)
        ]
        
        completed = 0
        for step_name, step_func in setup_steps:
            print(f"🔄 {step_name}...")
            try:
                if step_func():
                    print(f"✅ {step_name} - COMPLETED")
                    completed += 1
                else:
                    print(f"⚠️ {step_name} - NEEDS ATTENTION")
            except Exception as e:
                print(f"❌ {step_name} - ERROR: {e}")
        
        success_rate = completed / len(setup_steps)
        
        print("\n" + "="*60)
        print("📊 FINAL SETUP SUMMARY")
        print("="*60)
        print(f"✅ Completed: {completed}/{len(setup_steps)} steps")
        print(f"📈 Success Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print("🚀 PRIVATE USE SETUP SUCCESSFUL!")
            self.setup_complete = True
            self._display_next_actions()
            return True
        else:
            print("⚠️ SETUP NEEDS ATTENTION")
            self._display_troubleshooting()
            return False
    
    def _setup_private_api(self) -> bool:
        """Setup private API configuration."""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create private API configuration
            private_config = {
                'api': {
                    'testnet': {
                        'enabled': True,
                        'base_url': 'https://api-testnet.bybit.com',
                        'note': 'Safe for testing - uses testnet funds'
                    },
                    'mainnet': {
                        'enabled': False,
                        'base_url': 'https://api.bybit.com',
                        'note': 'REAL TRADING - Enable only when ready for live trading'
                    }
                },
                'private_use': {
                    'owner': 'Private User',
                    'setup_date': datetime.now().isoformat(),
                    'mode': 'testnet_first',
                    'risk_level': 'conservative'
                }
            }
            
            config_file = self.config_dir / 'private_api_config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(private_config, f, indent=2)
            
            # Create secrets template if it doesn't exist
            if not self.secrets_file.exists():
                secrets_template_content = """# Private Use API Credentials
# IMPORTANT: Never commit this file to version control!

# Testnet Credentials (Safe for Testing)
BYBIT_TESTNET_API_KEY: "your_testnet_api_key_here"
BYBIT_TESTNET_API_SECRET: "your_testnet_secret_here"

# Mainnet Credentials (REAL TRADING - Keep Secure!)
BYBIT_MAINNET_API_KEY: "your_mainnet_api_key_here" 
BYBIT_MAINNET_API_SECRET: "your_mainnet_secret_here"

# Private Configuration
PRIVATE_USE_MODE: true
OWNER_EMAIL: "your_email@example.com"

# Optional: Advanced Settings
ENABLE_NOTIFICATIONS: true
MAX_DAILY_TRADES: 50
EMERGENCY_STOP_LOSS: 0.05  # 5% portfolio loss limit
"""
                
                with open(self.secrets_file, 'w') as f:
                    f.write(secrets_template_content)
            
            print("  → API configuration files created")
            print("  → Edit config/secrets.yaml with your API credentials")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Private API setup failed: {e}")
            return False
    
    def _configure_risk_params(self) -> bool:
        """Configure personal risk management parameters."""
        try:
            risk_config = {
                'position_sizing': {
                    'max_position_size': 0.02,  # 2% of portfolio per trade
                    'max_portfolio_risk': 0.10,  # 10% total portfolio risk
                    'risk_per_trade': 0.01,  # 1% risk per trade
                },
                'stop_loss': {
                    'default_stop_loss': 0.03,  # 3% stop loss
                    'trailing_stop': True,
                    'emergency_stop': 0.05  # 5% portfolio emergency stop
                },
                'take_profit': {
                    'default_take_profit': 0.06,  # 6% take profit (2:1 R/R)
                    'partial_profit_levels': [0.03, 0.05, 0.08],
                    'profit_scaling': True
                },
                'private_limits': {
                    'max_daily_trades': 20,
                    'max_concurrent_positions': 5,
                    'minimum_account_balance': 1000,  # USD
                    'trading_hours': {
                        'enabled': False,  # Trade 24/7 by default
                        'start_hour': 9,
                        'end_hour': 17
                    }
                }
            }
            
            risk_file = self.config_dir / 'private_risk_config.yaml'
            with open(risk_file, 'w') as f:
                yaml.dump(risk_config, f, indent=2)
            
            print("  → Conservative risk parameters configured")
            print("  → 2% position sizing, 1% risk per trade")
            print("  → Emergency stops and profit taking enabled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Risk configuration failed: {e}")
            return False
    
    def _setup_security(self) -> bool:
        """Setup security configuration for private use."""
        try:
            # Create .gitignore entries for sensitive files
            gitignore_file = self.setup_dir / '.gitignore'
            gitignore_entries = [
                "# Private Use - Security",
                "config/secrets.yaml",
                "config/private_*.yaml",
                "*.env",
                "logs/private_*.log",
                "data/private/",
                "backups/",
                ".vscode/settings.json",
                ""
            ]
            
            # Add entries to .gitignore if not already present
            if gitignore_file.exists():
                existing_content = gitignore_file.read_text()
                new_entries = [entry for entry in gitignore_entries if entry not in existing_content]
                if new_entries:
                    with open(gitignore_file, 'a') as f:
                        f.write('\n' + '\n'.join(new_entries))
            else:
                with open(gitignore_file, 'w') as f:
                    f.write('\n'.join(gitignore_entries))
            
            # Create environment file template
            env_file = self.setup_dir / '.env.template'
            env_content = """# Private Use Environment Variables
# Copy to .env and fill in your values

# Trading Mode
TRADING_MODE=testnet
PRIVATE_USE=true

# API Configuration
BYBIT_TESTNET_API_KEY=your_testnet_key
BYBIT_TESTNET_API_SECRET=your_testnet_secret

# Risk Management
MAX_POSITION_SIZE=0.02
EMERGENCY_STOP_LOSS=0.05

# Notifications
ENABLE_EMAIL_NOTIFICATIONS=true
OWNER_EMAIL=your_email@example.com

# Dashboard
DASHBOARD_PORT=8501
ENABLE_FIRE_THEME=true
"""
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            print("  → Security configurations updated")
            print("  → .gitignore updated to protect sensitive files")
            print("  → Environment template created")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security setup failed: {e}")
            return False
    
    def _setup_private_dashboard(self) -> bool:
        """Configure dashboard for private use."""
        try:
            dashboard_config = {
                'private_dashboard': {
                    'title': 'Private Bybit Trading Bot',
                    'theme': 'fire_cybersigilism',
                    'port': 8501,
                    'features': {
                        'ml_predictions': True,
                        'strategy_graduation': True,
                        'performance_analytics': True,
                        'live_trading_controls': True,
                        'risk_monitoring': True
                    }
                },
                'display_options': {
                    'hide_api_keys': True,
                    'show_personal_metrics': True,
                    'enable_mobile_view': True,
                    'auto_refresh_seconds': 5
                }
            }
            
            dashboard_file = self.config_dir / 'private_dashboard_config.yaml'
            with open(dashboard_file, 'w') as f:
                yaml.dump(dashboard_config, f, indent=2)
            
            print("  → Private dashboard configuration created")
            print("  → Fire Cybersigilism theme enabled")
            print("  → Personal metrics and ML features activated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Dashboard setup failed: {e}")
            return False
    
    def _optimize_local_dev(self) -> bool:
        """Optimize configuration for local development."""
        try:
            # Create local development configuration
            dev_config = {
                'development': {
                    'mode': 'local',
                    'debug': True,
                    'fast_testing': True,
                    'mock_trading': True  # Safe testing without real orders
                },
                'performance': {
                    'cache_data': True,
                    'parallel_processing': True,
                    'memory_optimization': True
                },
                'logging': {
                    'level': 'INFO',
                    'file_rotation': True,
                    'max_file_size': '10MB',
                    'backup_count': 5
                }
            }
            
            dev_file = self.config_dir / 'development.yaml'
            with open(dev_file, 'w') as f:
                yaml.dump(dev_config, f, indent=2)
            
            # Create VS Code settings for better development experience
            vscode_dir = self.setup_dir / '.vscode'
            vscode_dir.mkdir(exist_ok=True)
            
            vscode_settings = {
                "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
                "python.linting.enabled": True,
                "python.linting.pylintEnabled": True,
                "python.formatting.provider": "black",
                "files.exclude": {
                    "**/__pycache__": True,
                    "**/*.pyc": True,
                    "config/secrets.yaml": True
                },
                "python.analysis.extraPaths": ["./src"],
                "python.terminal.activateEnvironment": True
            }
            
            vscode_settings_file = vscode_dir / 'settings.json'
            with open(vscode_settings_file, 'w') as f:
                json.dump(vscode_settings, f, indent=2)
            
            print("  → Local development optimizations applied")
            print("  → VS Code settings configured")
            print("  → Debug mode and safe testing enabled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Local development optimization failed: {e}")
            return False
    
    def _validate_final_config(self) -> bool:
        """Validate that all configurations are properly set up."""
        try:
            required_files = [
                'config/private_api_config.yaml',
                'config/private_risk_config.yaml',
                'config/private_dashboard_config.yaml',
                'config/development.yaml',
                'config/secrets.yaml',
                '.env.template'
            ]
            
            missing_files = []
            for file_path in required_files:
                if not (self.setup_dir / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                print(f"  ⚠️ Missing files: {missing_files}")
                return False
            
            # Validate key directories exist
            key_dirs = ['src', 'config', 'scripts']
            for dir_name in key_dirs:
                if not (self.setup_dir / dir_name).exists():
                    print(f"  ⚠️ Missing directory: {dir_name}")
                    return False
            
            print("  → All configuration files present")
            print("  → Directory structure validated")
            print("  → Ready for private use deployment")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Final validation failed: {e}")
            return False
    
    def _generate_user_guide(self) -> bool:
        """Generate comprehensive user guide for private use."""
        try:
            user_guide = """# 🔥 Private Bybit Trading Bot - User Guide

## Quick Start

### 1. Initial Setup
```bash
# 1. Configure your API credentials
cp .env.template .env
# Edit .env with your Bybit API keys

# 2. Edit secrets configuration
# Edit config/secrets.yaml with your API credentials

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the system
python activate_ml_engine.py
```

### 2. Access Dashboard
- Open browser: http://localhost:8501
- Fire Cybersigilism theme with real-time ML predictions
- Monitor strategy graduation and live trading

### 3. Safety First
- **Always start with TESTNET** 
- Test strategies thoroughly before live trading
- Monitor risk parameters closely
- Use emergency stop-loss settings

## Configuration Files

### API Configuration (`config/secrets.yaml`)
- **Testnet**: Safe for learning and testing
- **Mainnet**: Real trading - use with caution
- Keep this file secure and never share

### Risk Management (`config/private_risk_config.yaml`)
- Position sizing: 2% max per trade
- Stop loss: 3% default
- Emergency portfolio stop: 5%
- Customize based on your risk tolerance

### Dashboard (`config/private_dashboard_config.yaml`)
- Fire Cybersigilism theme
- Real-time ML predictions
- Strategy graduation monitoring
- Performance analytics

## Features

### 🤖 ML Integration
- Real-time AI predictions
- Ensemble model insights  
- Strategy auto-graduation
- Performance attribution

### 🛡️ Risk Management
- Conservative position sizing
- Automatic stop-losses
- Emergency portfolio protection
- Daily trading limits

### 📊 Analytics
- Real-time performance tracking
- Strategy graduation progress
- ML model confidence levels
- Profit/loss attribution

### 🔥 Fire Dashboard
- Cyberpunk-themed interface
- Real-time updates
- Mobile-friendly design
- Live trading controls

## Safety Guidelines

### For Beginners
1. **Start with testnet only**
2. **Use small position sizes**
3. **Monitor closely for first week**
4. **Understand each strategy before enabling**

### Risk Management
1. **Never risk more than you can afford to lose**
2. **Set emergency stop-loss limits**
3. **Monitor daily trading performance**
4. **Keep API keys secure**

### Advanced Usage
1. **Customize ML model weights**
2. **Adjust strategy graduation thresholds**
3. **Fine-tune risk parameters**
4. **Enable additional trading pairs**

## Support

### Self-Help
- Check logs in `logs/` directory
- Review configuration files
- Test with mock trading first

### Troubleshooting
- Ensure API credentials are correct
- Verify internet connection for data feeds
- Check Bybit API status
- Review error logs for specific issues

---

🔥 **Happy Trading with your Private AI-Powered Bot!** 🔥

Remember: This is for personal use only. Trade responsibly!
"""
            
            user_guide_file = self.setup_dir / 'PRIVATE_USE_GUIDE.md'
            with open(user_guide_file, 'w') as f:
                f.write(user_guide)
            
            print("  → Comprehensive user guide created")
            print("  → See PRIVATE_USE_GUIDE.md for detailed instructions")
            
            return True
            
        except Exception as e:
            self.logger.error(f"User guide generation failed: {e}")
            return False
    
    def _display_next_actions(self):
        """Display next actions for the user."""
        print("\n" + "🎯" + " "*54 + "🎯")
        print("🔥 PRIVATE USE SETUP COMPLETE - NEXT ACTIONS 🔥")
        print("🎯" + " "*54 + "🎯")
        print()
        print("1. 🔐 CONFIGURE YOUR API CREDENTIALS:")
        print("   → Edit config/secrets.yaml with your Bybit API keys")
        print("   → Start with testnet credentials for safety")
        print("   → Never share or commit these files to version control")
        print()
        print("2. 🚀 ACTIVATE THE SYSTEM:")
        print("   → Run: python activate_ml_engine.py")
        print("   → This will initialize all ML components")
        print("   → Start with testnet mode for safe testing")
        print()
        print("3. 📊 ACCESS YOUR PRIVATE DASHBOARD:")
        print("   → Open: http://localhost:8501")
        print("   → Fire Cybersigilism theme with AI predictions")
        print("   → Monitor strategy performance and graduation")
        print()
        print("4. 📖 READ THE USER GUIDE:")
        print("   → See: PRIVATE_USE_GUIDE.md")
        print("   → Complete setup and safety instructions")
        print("   → Troubleshooting and advanced configuration")
        print()
        print("5. 🛡️ SAFETY REMINDERS:")
        print("   → Always test with testnet first")
        print("   → Start with small position sizes")
        print("   → Monitor risk parameters closely")
        print("   → Set emergency stop-loss limits")
        print()
        print("🔥 YOUR PRIVATE AI TRADING BOT IS READY! 🔥")
        print("="*60)
    
    def _display_troubleshooting(self):
        """Display troubleshooting steps."""
        print("\n" + "⚠️" + " "*52 + "⚠️")
        print("SETUP NEEDS ATTENTION - TROUBLESHOOTING")
        print("⚠️" + " "*52 + "⚠️")
        print()
        print("1. Check missing dependencies:")
        print("   → pip install -r requirements.txt")
        print()
        print("2. Verify directory structure:")
        print("   → Ensure src/, config/, scripts/ directories exist")
        print()
        print("3. Review error messages above")
        print("   → Address any specific configuration issues")
        print()
        print("4. Try running setup again:")
        print("   → python private_use_setup.py")


def main():
    """Main setup execution."""
    setup = PrivateUseSetup()
    
    try:
        success = setup.run_final_setup()
        
        if success:
            print("\n✅ PRIVATE USE SETUP COMPLETED SUCCESSFULLY!")
            return 0
        else:
            print("\n⚠️ PRIVATE USE SETUP NEEDS ATTENTION")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        return 2
    except Exception as e:
        print(f"\n💥 Setup failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())