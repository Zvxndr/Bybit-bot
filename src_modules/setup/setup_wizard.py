#!/usr/bin/env python3
"""
Enhanced Setup Wizard System
Complete configuration wizard with real-time validation
Addresses audit finding: Incomplete setup wizard
"""

import asyncio
import os
import sys
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import getpass
import re
from datetime import datetime

# Import our security components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.secure_storage import SecureConfigManager
from security.api_validator import APIKeyValidator, APIKeyInfo, PermissionManager, ExchangeType, PermissionLevel

class SetupStep(Enum):
    """Setup wizard steps"""
    WELCOME = "welcome"
    SECURITY_SETUP = "security_setup"
    EXCHANGE_SELECTION = "exchange_selection"
    API_KEY_CONFIGURATION = "api_key_configuration"
    RISK_MANAGEMENT = "risk_management"
    ML_CONFIGURATION = "ml_configuration"
    NOTIFICATION_SETUP = "notification_setup"
    VALIDATION = "validation"
    COMPLETION = "completion"

@dataclass
class ExchangeConfig:
    """Exchange configuration structure"""
    name: str
    enabled: bool
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    testnet: bool = True
    validated: bool = False
    permissions: List[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.04  # 4%
    max_daily_trades: int = 10
    max_drawdown_percentage: float = 0.05  # 5%
    enable_risk_limits: bool = True

@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_type: str = "xgboost"  # xgboost, lightgbm, ensemble
    prediction_confidence_threshold: float = 0.65
    retrain_frequency_hours: int = 24
    feature_importance_threshold: float = 0.01
    enable_sentiment_analysis: bool = True

@dataclass
class NotificationConfig:
    """Notification configuration"""
    email_enabled: bool = False
    email_address: str = ""
    webhook_enabled: bool = False
    webhook_url: str = ""
    trade_notifications: bool = True
    error_notifications: bool = True

class SetupWizard:
    """
    Enhanced setup wizard with complete configuration
    Real-time validation and secure storage
    """
    
    def __init__(self):
        self.config_manager = SecureConfigManager()
        self.api_validator = APIKeyValidator()
        self.permission_manager = PermissionManager(self.api_validator)
        
        # Configuration storage
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.risk_config = RiskConfig()
        self.ml_config = MLConfig()
        self.notification_config = NotificationConfig()
        
        # Setup state
        self.current_step = SetupStep.WELCOME
        self.setup_complete = False
        self.master_password = None
        
        # Available exchanges
        self.available_exchanges = {
            ExchangeType.BYBIT: {
                'name': 'Bybit',
                'description': 'Popular derivatives exchange with futures and spot trading',
                'requires_passphrase': False,
                'testnet_available': True
            },
            ExchangeType.BINANCE: {
                'name': 'Binance',
                'description': 'World\'s largest cryptocurrency exchange by volume',
                'requires_passphrase': False,
                'testnet_available': True
            },
            ExchangeType.OKX: {
                'name': 'OKX',
                'description': 'Advanced trading platform with derivatives support',
                'requires_passphrase': True,
                'testnet_available': True
            },
            ExchangeType.COINBASE: {
                'name': 'Coinbase Pro',
                'description': 'US-based exchange with regulatory compliance',
                'requires_passphrase': True,
                'testnet_available': False
            }
        }
    
    async def run_setup(self) -> bool:
        """
        Run the complete setup wizard
        
        Returns:
            bool: True if setup completed successfully
        """
        try:
            print("🚀 Bybit Trading Bot - Enhanced Setup Wizard")
            print("=" * 50)
            
            # Step 1: Welcome and overview
            if not await self._step_welcome():
                return False
            
            # Step 2: Security setup
            if not await self._step_security_setup():
                return False
            
            # Step 3: Exchange selection
            if not await self._step_exchange_selection():
                return False
            
            # Step 4: API key configuration
            if not await self._step_api_key_configuration():
                return False
            
            # Step 5: Risk management setup
            if not await self._step_risk_management():
                return False
            
            # Step 6: ML configuration
            if not await self._step_ml_configuration():
                return False
            
            # Step 7: Notification setup
            if not await self._step_notification_setup():
                return False
            
            # Step 8: Validation
            if not await self._step_validation():
                return False
            
            # Step 9: Completion
            if not await self._step_completion():
                return False
            
            self.setup_complete = True
            return True
            
        except KeyboardInterrupt:
            print("\n\n❌ Setup cancelled by user")
            return False
        except Exception as e:
            print(f"\n\n❌ Setup failed: {e}")
            return False
    
    async def _step_welcome(self) -> bool:
        """Welcome step with overview"""
        self.current_step = SetupStep.WELCOME
        
        print("\n🎯 Welcome to the Advanced Trading Bot Setup!")
        print("\nThis wizard will guide you through:")
        print("  • Security configuration and encryption setup")
        print("  • Exchange API key configuration and validation")
        print("  • Risk management parameter configuration")
        print("  • Machine learning model configuration")
        print("  • Notification and monitoring setup")
        print("  • Complete system validation")
        
        print("\n⚠️  Important Security Notes:")
        print("  • All API keys will be encrypted with AES-256-GCM")
        print("  • Use testnet/sandbox mode initially for safety")
        print("  • Never share your API keys or master password")
        print("  • Enable only required API permissions")
        
        response = input("\n📋 Ready to begin setup? (y/n): ").lower().strip()
        return response in ['y', 'yes']
    
    async def _step_security_setup(self) -> bool:
        """Security and encryption setup"""
        self.current_step = SetupStep.SECURITY_SETUP
        
        print("\n🔐 Security Configuration")
        print("=" * 30)
        
        # Check if secure storage is already configured
        if self.config_manager.is_initialized():
            print("✅ Secure storage is already configured")
            use_existing = input("Use existing configuration? (y/n): ").lower().strip()
            if use_existing in ['y', 'yes']:
                # Verify master password
                password = getpass.getpass("Enter master password: ")
                if self.config_manager.verify_master_password(password):
                    self.master_password = password
                    print("✅ Master password verified")
                    return True
                else:
                    print("❌ Invalid master password")
                    return False
        
        print("\n📝 Creating new secure configuration...")
        
        # Get master password
        while True:
            password = getpass.getpass("Create master password (min 12 chars): ")
            if len(password) < 12:
                print("❌ Password must be at least 12 characters")
                continue
            
            confirm_password = getpass.getpass("Confirm master password: ")
            if password != confirm_password:
                print("❌ Passwords don't match")
                continue
            
            break
        
        # Initialize secure storage
        try:
            self.config_manager.initialize_storage(password)
            self.master_password = password
            print("✅ Secure storage initialized with AES-256-GCM encryption")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize secure storage: {e}")
            return False
    
    async def _step_exchange_selection(self) -> bool:
        """Exchange selection step"""
        self.current_step = SetupStep.EXCHANGE_SELECTION
        
        print("\n🏦 Exchange Selection")
        print("=" * 25)
        
        print("\nAvailable exchanges:")
        for i, (exchange_type, info) in enumerate(self.available_exchanges.items(), 1):
            testnet_note = " (Testnet available)" if info['testnet_available'] else ""
            print(f"  {i}. {info['name']}{testnet_note}")
            print(f"     {info['description']}")
        
        print("\nSelect exchanges to configure (e.g., 1,2,3 or 'all'):")
        selection = input("Selection: ").strip().lower()
        
        if selection == 'all':
            selected_exchanges = list(self.available_exchanges.keys())
        else:
            try:
                indices = [int(x.strip()) for x in selection.split(',')]
                exchange_list = list(self.available_exchanges.keys())
                selected_exchanges = [exchange_list[i-1] for i in indices if 1 <= i <= len(exchange_list)]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return False
        
        if not selected_exchanges:
            print("❌ No exchanges selected")
            return False
        
        # Initialize exchange configurations
        for exchange_type in selected_exchanges:
            info = self.available_exchanges[exchange_type]
            self.exchanges[exchange_type.value] = ExchangeConfig(
                name=info['name'],
                enabled=True,
                testnet=info['testnet_available']  # Default to testnet if available
            )
        
        print(f"✅ Selected {len(selected_exchanges)} exchanges for configuration")
        return True
    
    async def _step_api_key_configuration(self) -> bool:
        """API key configuration with real-time validation"""
        self.current_step = SetupStep.API_KEY_CONFIGURATION
        
        print("\n🔑 API Key Configuration")
        print("=" * 30)
        
        for exchange_name, config in self.exchanges.items():
            print(f"\n📊 Configuring {config.name}")
            print("-" * 20)
            
            # Get exchange info
            exchange_type = ExchangeType(exchange_name)
            exchange_info = self.available_exchanges[exchange_type]
            
            # Testnet selection
            if exchange_info['testnet_available']:
                use_testnet = input(f"Use testnet/sandbox for {config.name}? (recommended) (y/n): ").lower().strip()
                config.testnet = use_testnet in ['y', 'yes']
                
                if config.testnet:
                    print("✅ Using testnet - safe for testing")
                else:
                    print("⚠️  Using mainnet - real money at risk!")
                    confirm_mainnet = input("Are you sure? (yes/no): ").lower().strip()
                    if confirm_mainnet != 'yes':
                        config.testnet = True
                        print("✅ Switched to testnet for safety")
            
            # API key input
            print(f"\n🔐 Enter API credentials for {config.name}:")
            config.api_key = input("API Key: ").strip()
            config.api_secret = getpass.getpass("API Secret: ")
            
            if exchange_info['requires_passphrase']:
                config.passphrase = getpass.getpass("Passphrase: ")
            
            # Validate API key
            print(f"\n🔍 Validating API key for {config.name}...")
            api_key_info = APIKeyInfo(
                exchange=exchange_name,
                key=config.api_key,
                secret=config.api_secret,
                passphrase=config.passphrase if exchange_info['requires_passphrase'] else None,
                testnet=config.testnet
            )
            
            validation_result = await self.api_validator.validate_api_key(api_key_info)
            
            if validation_result.valid:
                print(f"✅ API key valid for {config.name}")
                print(f"   Permissions: {', '.join(validation_result.permissions)}")
                if validation_result.balance_available:
                    print("   ✅ Balance access confirmed")
                
                config.validated = True
                config.permissions = validation_result.permissions
                
                # Check required permissions
                required_perms = [p.value for p in self.permission_manager.get_permission_requirements(exchange_type)]
                missing_perms = [p for p in required_perms if p not in validation_result.permissions]
                
                if missing_perms:
                    print(f"   ⚠️  Missing permissions: {', '.join(missing_perms)}")
                    print("   Please update API key permissions in your exchange account")
                    
                    continue_anyway = input("   Continue with limited permissions? (y/n): ").lower().strip()
                    if continue_anyway not in ['y', 'yes']:
                        return False
                
            else:
                print(f"❌ API key validation failed for {config.name}")
                if validation_result.error_message:
                    print(f"   Error: {validation_result.error_message}")
                
                retry = input("Retry API key configuration? (y/n): ").lower().strip()
                if retry in ['y', 'yes']:
                    # Recursive retry
                    return await self._step_api_key_configuration()
                else:
                    config.enabled = False
                    print(f"   {config.name} disabled due to validation failure")
        
        # Check if any exchanges are successfully configured
        valid_exchanges = [name for name, config in self.exchanges.items() if config.enabled and config.validated]
        
        if not valid_exchanges:
            print("\n❌ No exchanges successfully configured")
            retry = input("Retry exchange configuration? (y/n): ").lower().strip()
            if retry in ['y', 'yes']:
                return await self._step_exchange_selection()
            return False
        
        print(f"\n✅ Successfully configured {len(valid_exchanges)} exchanges")
        return True
    
    async def _step_risk_management(self) -> bool:
        """Risk management configuration"""
        self.current_step = SetupStep.RISK_MANAGEMENT
        
        print("\n⚠️  Risk Management Configuration")
        print("=" * 35)
        
        print("Configure risk management parameters to protect your capital:")
        
        # Position size
        while True:
            try:
                position_size = input(f"Max position size (% of portfolio) [{self.risk_config.max_position_size*100:.1f}%]: ").strip()
                if not position_size:
                    break
                
                value = float(position_size.rstrip('%')) / 100
                if 0 < value <= 1:
                    self.risk_config.max_position_size = value
                    break
                else:
                    print("❌ Position size must be between 0% and 100%")
            except ValueError:
                print("❌ Invalid number format")
        
        # Stop loss
        while True:
            try:
                stop_loss = input(f"Stop loss percentage [{self.risk_config.stop_loss_percentage*100:.1f}%]: ").strip()
                if not stop_loss:
                    break
                
                value = float(stop_loss.rstrip('%')) / 100
                if 0 < value <= 0.5:  # Max 50% stop loss
                    self.risk_config.stop_loss_percentage = value
                    break
                else:
                    print("❌ Stop loss must be between 0% and 50%")
            except ValueError:
                print("❌ Invalid number format")
        
        # Take profit
        while True:
            try:
                take_profit = input(f"Take profit percentage [{self.risk_config.take_profit_percentage*100:.1f}%]: ").strip()
                if not take_profit:
                    break
                
                value = float(take_profit.rstrip('%')) / 100
                if 0 < value <= 2:  # Max 200% take profit
                    self.risk_config.take_profit_percentage = value
                    break
                else:
                    print("❌ Take profit must be between 0% and 200%")
            except ValueError:
                print("❌ Invalid number format")
        
        # Daily trade limit
        while True:
            try:
                daily_trades = input(f"Max daily trades [{self.risk_config.max_daily_trades}]: ").strip()
                if not daily_trades:
                    break
                
                value = int(daily_trades)
                if 1 <= value <= 1000:
                    self.risk_config.max_daily_trades = value
                    break
                else:
                    print("❌ Daily trades must be between 1 and 1000")
            except ValueError:
                print("❌ Invalid number format")
        
        # Max drawdown
        while True:
            try:
                drawdown = input(f"Max drawdown percentage [{self.risk_config.max_drawdown_percentage*100:.1f}%]: ").strip()
                if not drawdown:
                    break
                
                value = float(drawdown.rstrip('%')) / 100
                if 0 < value <= 0.5:  # Max 50% drawdown
                    self.risk_config.max_drawdown_percentage = value
                    break
                else:
                    print("❌ Max drawdown must be between 0% and 50%")
            except ValueError:
                print("❌ Invalid number format")
        
        print("\n✅ Risk management configuration complete")
        print(f"   Max position size: {self.risk_config.max_position_size*100:.1f}%")
        print(f"   Stop loss: {self.risk_config.stop_loss_percentage*100:.1f}%")
        print(f"   Take profit: {self.risk_config.take_profit_percentage*100:.1f}%")
        print(f"   Max daily trades: {self.risk_config.max_daily_trades}")
        print(f"   Max drawdown: {self.risk_config.max_drawdown_percentage*100:.1f}%")
        
        return True
    
    async def _step_ml_configuration(self) -> bool:
        """Machine learning configuration"""
        self.current_step = SetupStep.ML_CONFIGURATION
        
        print("\n🤖 Machine Learning Configuration")
        print("=" * 35)
        
        # Model selection
        print("\nAvailable ML models:")
        print("  1. XGBoost (recommended for beginners)")
        print("  2. LightGBM (faster training)")
        print("  3. Ensemble (XGBoost + LightGBM)")
        
        while True:
            model_choice = input("Select model [1]: ").strip()
            if not model_choice:
                model_choice = "1"
            
            model_map = {"1": "xgboost", "2": "lightgbm", "3": "ensemble"}
            if model_choice in model_map:
                self.ml_config.model_type = model_map[model_choice]
                break
            else:
                print("❌ Invalid selection")
        
        # Prediction confidence threshold
        while True:
            try:
                confidence = input(f"Prediction confidence threshold [{self.ml_config.prediction_confidence_threshold:.2f}]: ").strip()
                if not confidence:
                    break
                
                value = float(confidence)
                if 0.5 <= value <= 0.95:
                    self.ml_config.prediction_confidence_threshold = value
                    break
                else:
                    print("❌ Confidence threshold must be between 0.5 and 0.95")
            except ValueError:
                print("❌ Invalid number format")
        
        # Retrain frequency
        while True:
            try:
                retrain = input(f"Model retrain frequency (hours) [{self.ml_config.retrain_frequency_hours}]: ").strip()
                if not retrain:
                    break
                
                value = int(retrain)
                if 1 <= value <= 168:  # Max once per week
                    self.ml_config.retrain_frequency_hours = value
                    break
                else:
                    print("❌ Retrain frequency must be between 1 and 168 hours")
            except ValueError:
                print("❌ Invalid number format")
        
        # Sentiment analysis
        sentiment = input(f"Enable sentiment analysis? (y/n) [{'y' if self.ml_config.enable_sentiment_analysis else 'n'}]: ").lower().strip()
        if sentiment in ['y', 'yes']:
            self.ml_config.enable_sentiment_analysis = True
        elif sentiment in ['n', 'no']:
            self.ml_config.enable_sentiment_analysis = False
        
        print("\n✅ ML configuration complete")
        print(f"   Model: {self.ml_config.model_type}")
        print(f"   Confidence threshold: {self.ml_config.prediction_confidence_threshold:.2f}")
        print(f"   Retrain frequency: {self.ml_config.retrain_frequency_hours} hours")
        print(f"   Sentiment analysis: {'enabled' if self.ml_config.enable_sentiment_analysis else 'disabled'}")
        
        return True
    
    async def _step_notification_setup(self) -> bool:
        """Notification setup"""
        self.current_step = SetupStep.NOTIFICATION_SETUP
        
        print("\n📢 Notification Configuration")
        print("=" * 30)
        
        # Email notifications
        email_enabled = input("Enable email notifications? (y/n): ").lower().strip()
        if email_enabled in ['y', 'yes']:
            self.notification_config.email_enabled = True
            
            while True:
                email = input("Email address: ").strip()
                if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    self.notification_config.email_address = email
                    break
                else:
                    print("❌ Invalid email format")
        
        # Webhook notifications
        webhook_enabled = input("Enable webhook notifications? (y/n): ").lower().strip()
        if webhook_enabled in ['y', 'yes']:
            self.notification_config.webhook_enabled = True
            webhook_url = input("Webhook URL: ").strip()
            if webhook_url:
                self.notification_config.webhook_url = webhook_url
        
        # Notification types
        trade_notifications = input("Enable trade notifications? (y/n) [y]: ").lower().strip()
        self.notification_config.trade_notifications = trade_notifications != 'n'
        
        error_notifications = input("Enable error notifications? (y/n) [y]: ").lower().strip()
        self.notification_config.error_notifications = error_notifications != 'n'
        
        print("\n✅ Notification configuration complete")
        return True
    
    async def _step_validation(self) -> bool:
        """Complete system validation"""
        self.current_step = SetupStep.VALIDATION
        
        print("\n🔍 System Validation")
        print("=" * 20)
        
        validation_passed = True
        
        # Validate exchanges
        print("\n1. Exchange validation:")
        for exchange_name, config in self.exchanges.items():
            if config.enabled:
                print(f"   ✅ {config.name}: API validated, permissions OK")
            else:
                print(f"   ❌ {config.name}: Disabled or validation failed")
        
        # Validate risk management
        print("\n2. Risk management validation:")
        if self.risk_config.stop_loss_percentage > 0:
            print("   ✅ Stop loss configured")
        else:
            print("   ⚠️  Stop loss not configured")
            validation_passed = False
        
        if self.risk_config.max_position_size <= 0.5:  # Max 50% position size
            print("   ✅ Position sizing reasonable")
        else:
            print("   ⚠️  Position size may be too large")
        
        # Validate ML configuration
        print("\n3. ML configuration validation:")
        if self.ml_config.prediction_confidence_threshold >= 0.6:
            print("   ✅ Confidence threshold set appropriately")
        else:
            print("   ⚠️  Low confidence threshold may increase false signals")
        
        # Validate security
        print("\n4. Security validation:")
        if self.config_manager.is_initialized():
            print("   ✅ Secure storage initialized")
        else:
            print("   ❌ Secure storage not initialized")
            validation_passed = False
        
        if validation_passed:
            print("\n✅ All validations passed")
        else:
            print("\n⚠️  Some validations failed")
            continue_anyway = input("Continue with setup completion? (y/n): ").lower().strip()
            if continue_anyway not in ['y', 'yes']:
                return False
        
        return True
    
    async def _step_completion(self) -> bool:
        """Complete setup and save configuration"""
        self.current_step = SetupStep.COMPLETION
        
        print("\n💾 Saving Configuration")
        print("=" * 25)
        
        try:
            # Save all configurations to secure storage
            config_data = {
                'exchanges': {name: asdict(config) for name, config in self.exchanges.items()},
                'risk_management': asdict(self.risk_config),
                'ml_configuration': asdict(self.ml_config),
                'notifications': asdict(self.notification_config),
                'setup_completed_at': datetime.now().isoformat(),
                'version': '2.0.0'
            }
            
            # Store configuration securely
            self.config_manager.set_config('trading_bot_config', config_data)
            
            # Store individual API keys securely
            for exchange_name, config in self.exchanges.items():
                if config.enabled and config.api_key:
                    api_key_data = {
                        'api_key': config.api_key,
                        'api_secret': config.api_secret,
                        'passphrase': config.passphrase,
                        'testnet': config.testnet,
                        'validated': config.validated,
                        'permissions': config.permissions
                    }
                    self.config_manager.set_config(f'api_keys_{exchange_name}', api_key_data)
            
            print("✅ Configuration saved securely")
            
            # Setup completion summary
            print("\n🎉 Setup Complete!")
            print("=" * 20)
            
            enabled_exchanges = [config.name for config in self.exchanges.values() if config.enabled]
            print(f"✅ Exchanges configured: {', '.join(enabled_exchanges)}")
            print(f"✅ Model: {self.ml_config.model_type}")
            print(f"✅ Max position size: {self.risk_config.max_position_size*100:.1f}%")
            print(f"✅ Notifications: {'enabled' if (self.notification_config.email_enabled or self.notification_config.webhook_enabled) else 'disabled'}")
            
            print("\n📋 Next Steps:")
            print("  1. Review the generated configuration")
            print("  2. Start with paper trading to test the system")
            print("  3. Monitor performance and adjust parameters")
            print("  4. Enable live trading when comfortable")
            
            print("\n⚠️  Remember:")
            print("  • Always start with testnet/sandbox mode")
            print("  • Monitor your trades closely")
            print("  • Never risk more than you can afford to lose")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False


# Main execution
async def main():
    """Run the enhanced setup wizard"""
    wizard = SetupWizard()
    
    try:
        success = await wizard.run_setup()
        if success:
            print("\n🚀 Setup completed successfully!")
            print("You can now start the trading bot with your secure configuration.")
        else:
            print("\n❌ Setup incomplete")
            print("Please run the setup wizard again to complete configuration.")
        
        return success
        
    except Exception as e:
        print(f"\n💥 Setup wizard crashed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(main())