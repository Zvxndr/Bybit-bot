"""
Modern Integration Tests for Unified Trading Bot System

Tests the complete integration of:
- Phase 1: Unified Risk Management 
- Phase 2.5: ML Integration Layer
- Phase 3: Unified API System  
- Phase 4: Unified Configuration Management

Created: September 2025
Status: Production Ready Integration Tests
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import our unified systems
from src.bot.core.config.manager import UnifiedConfigurationManager
from src.bot.core.config.schema import UnifiedConfigurationSchema, Environment
from src.bot.risk.core.unified_risk_manager import UnifiedRiskManager
from src.bot.integration.ml_integration_controller import MLIntegrationController
from src.bot.api.unified_client import UnifiedBybitClient


class TestUnifiedSystemIntegration:
    """Test complete integration of all unified systems"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_config_data(self):
        """Sample configuration data for testing"""
        return {
            "environment": "development",
            "trading": {
                "mode": "paper_and_live",
                "max_position_size": 0.02,
                "stop_loss_percentage": 0.02,
                "take_profit_percentage": 0.04
            },
            "risk_management": {
                "max_drawdown": 0.15,
                "position_sizing_method": "kelly_criterion",
                "volatility_target": 0.20,
                "correlation_threshold": 0.7,
                "concentration_limit": 0.25
            },
            "ml_integration": {
                "feature_engineering": {
                    "enabled": True,
                    "lookback_periods": [5, 10, 20, 50]
                },
                "model_management": {
                    "auto_retrain": True,
                    "validation_threshold": 0.6
                }
            },
            "api": {
                "bybit": {
                    "testnet_enabled": True,
                    "mainnet_enabled": False,
                    "rate_limits": {
                        "orders_per_second": 10,
                        "requests_per_minute": 120
                    }
                }
            },
            "security": {
                "enable_encryption": False,
                "api_keys": {
                    "testnet_key": "test_key",
                    "testnet_secret": "test_secret"
                }
            }
        }
    
    @pytest.fixture
    def config_manager(self, temp_config_dir, mock_config_data):
        """Initialize configuration manager with test data"""
        config_file = temp_config_dir / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(mock_config_data, f)
        
        manager = UnifiedConfigurationManager(config_file)
        return manager
    
    def test_configuration_loading(self, config_manager):
        """Test unified configuration loads correctly"""
        config = config_manager.get_configuration()
        
        assert config is not None
        assert config.environment == Environment.DEVELOPMENT
        assert config.trading.max_position_size == 0.02
        assert config.risk_management.max_drawdown == 0.15
        assert config.ml_integration.feature_engineering.enabled == True
        assert config.api.bybit.testnet_enabled == True
    
    def test_risk_manager_integration(self, config_manager):
        """Test risk manager integrates with unified config"""
        config = config_manager.get_configuration()
        
        # Initialize risk manager with unified config
        risk_manager = UnifiedRiskManager(config.risk_management)
        
        # Test risk manager uses config values
        assert risk_manager.max_drawdown == 0.15
        assert risk_manager.position_sizing_method == "kelly_criterion"
        assert risk_manager.volatility_target == 0.20
        
        # Test position sizing calculation
        test_params = {
            'portfolio_value': 10000,
            'expected_return': 0.05,
            'volatility': 0.15,
            'risk_free_rate': 0.02
        }
        
        position_size = risk_manager.calculate_position_size(**test_params)
        assert 0 < position_size <= config.trading.max_position_size
    
    @pytest.mark.asyncio
    async def test_ml_integration_controller(self, config_manager):
        """Test ML integration controller with unified config"""
        config = config_manager.get_configuration()
        
        # Mock ML components to avoid actual ML dependencies
        with patch('src.bot.integration.ml_integration_controller.MLFeaturePipeline') as mock_pipeline, \
             patch('src.bot.integration.ml_integration_controller.MLModelManager') as mock_model_mgr:
            
            mock_pipeline.return_value = Mock()
            mock_model_mgr.return_value = Mock()
            
            # Initialize ML controller
            ml_controller = MLIntegrationController(config.ml_integration)
            
            # Test configuration integration
            assert ml_controller.config.feature_engineering.enabled == True
            assert ml_controller.config.feature_engineering.lookback_periods == [5, 10, 20, 50]
            assert ml_controller.config.model_management.auto_retrain == True
    
    @pytest.mark.asyncio
    async def test_api_client_integration(self, config_manager):
        """Test unified API client with configuration"""
        config = config_manager.get_configuration()
        
        # Mock the actual Bybit client to avoid real API calls
        with patch('src.bot.api.unified_client.BybitClient') as mock_bybit:
            mock_bybit.return_value = AsyncMock()
            
            # Initialize unified client
            api_client = UnifiedBybitClient(config.api.bybit)
            
            # Test configuration integration
            assert api_client.config.testnet_enabled == True
            assert api_client.config.mainnet_enabled == False
            assert api_client.config.rate_limits.orders_per_second == 10
    
    def test_configuration_validation(self, config_manager):
        """Test configuration validation works correctly"""
        config = config_manager.get_configuration()
        
        # Test that configuration validates successfully
        validation_result = config_manager.validate_configuration()
        assert validation_result.is_valid == True
        assert len(validation_result.errors) == 0
    
    def test_configuration_updates(self, config_manager):
        """Test configuration updates propagate correctly"""
        # Update risk management settings
        updates = {
            'risk_management': {
                'max_drawdown': 0.10,
                'volatility_target': 0.25
            }
        }
        
        config_manager.update_configuration(updates)
        updated_config = config_manager.get_configuration()
        
        assert updated_config.risk_management.max_drawdown == 0.10
        assert updated_config.risk_management.volatility_target == 0.25
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self, config_manager):
        """Test all systems work together end-to-end"""
        config = config_manager.get_configuration()
        
        # Mock external dependencies
        with patch('src.bot.api.unified_client.BybitClient') as mock_bybit, \
             patch('src.bot.integration.ml_integration_controller.MLFeaturePipeline') as mock_pipeline, \
             patch('src.bot.integration.ml_integration_controller.MLModelManager') as mock_model_mgr:
            
            # Setup mocks
            mock_bybit.return_value = AsyncMock()
            mock_pipeline.return_value = Mock()
            mock_model_mgr.return_value = Mock()
            
            # Initialize all systems
            risk_manager = UnifiedRiskManager(config.risk_management)
            ml_controller = MLIntegrationController(config.ml_integration)
            api_client = UnifiedBybitClient(config.api.bybit)
            
            # Test systems can interact
            assert risk_manager.max_drawdown == config.risk_management.max_drawdown
            assert ml_controller.config.feature_engineering.enabled == config.ml_integration.feature_engineering.enabled
            assert api_client.config.testnet_enabled == config.api.bybit.testnet_enabled
            
            # Test configuration changes propagate
            config_manager.update_configuration({
                'risk_management': {'max_drawdown': 0.12}
            })
            
            # Risk manager should pick up new config
            updated_config = config_manager.get_configuration()
            new_risk_manager = UnifiedRiskManager(updated_config.risk_management)
            assert new_risk_manager.max_drawdown == 0.12


class TestConfigurationEnvironments:
    """Test configuration behavior across different environments"""
    
    @pytest.fixture
    def production_config_data(self):
        """Production environment configuration"""
        return {
            "environment": "production",
            "trading": {
                "mode": "live_only",
                "max_position_size": 0.01  # More conservative in production
            },
            "security": {
                "enable_encryption": True,
                "api_keys": {
                    "mainnet_key": "encrypted_prod_key",
                    "mainnet_secret": "encrypted_prod_secret"
                }
            }
        }
    
    def test_production_environment_validation(self, temp_config_dir, production_config_data):
        """Test production environment has proper validation"""
        config_file = temp_config_dir / "prod_config.json"
        with open(config_file, 'w') as f:
            json.dump(production_config_data, f)
        
        manager = UnifiedConfigurationManager(config_file)
        config = manager.get_configuration()
        
        # Production should have stricter settings
        assert config.environment == Environment.PRODUCTION
        assert config.security.enable_encryption == True
        assert config.trading.max_position_size <= 0.01


if __name__ == "__main__":
    """Run integration tests"""
    print("🧪 Running Unified System Integration Tests...")
    print("=" * 60)
    
    # Run with pytest for detailed output
    pytest.main([__file__, "-v", "--tb=short"])