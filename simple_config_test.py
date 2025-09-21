"""
Simplified Configuration Integration Test - Phase 4 Testing

Lightweight test to verify the unified configuration system without import conflicts.
"""

import os
import json
import yaml
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import sys

# Add the specific config path to avoid conflicts
config_path = Path(__file__).parent / "src" / "bot" / "core" / "config"
sys.path.insert(0, str(config_path))

def test_basic_imports():
    """Test basic imports work"""
    print("Testing basic configuration imports...")
    
    try:
        from schema import (
            UnifiedConfigurationSchema, Environment, TradingMode,
            ExchangeCredentials, DatabaseConfig
        )
        
        from manager import (
            UnifiedConfigurationManager, ValidationResult
        )
        
        from integrations import (
            RiskManagementConfigAdapter,
            MLIntegrationConfigAdapter,
            APISystemConfigAdapter
        )
        
        print("✓ All imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_schema_creation():
    """Test configuration schema creation"""
    print("Testing configuration schema creation...")
    
    try:
        from schema import UnifiedConfigurationSchema, Environment, TradingMode
        
        # Create test configuration
        config_data = {
            'environment': 'development',
            'trading': {
                'mode': 'conservative',
                'base_balance': 10000.0,
            },
            'enable_trading': False
        }
        
        config = UnifiedConfigurationSchema(**config_data)
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.trading.mode == TradingMode.CONSERVATIVE
        assert config.trading.base_balance == 10000.0
        assert config.enable_trading == False
        
        print("✓ Schema creation successful")
        return True
        
    except Exception as e:
        print(f"✗ Schema creation failed: {e}")
        return False

def test_manager_functionality():
    """Test configuration manager basic functionality"""
    print("Testing configuration manager functionality...")
    
    try:
        from manager import UnifiedConfigurationManager
        from schema import UnifiedConfigurationSchema
        
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = UnifiedConfigurationManager(workspace_root=temp_dir)
            
            # Create default configuration
            config = manager.create_default_configuration()
            assert isinstance(config, UnifiedConfigurationSchema)
            
            # Test configuration summary
            summary = config.get_summary()
            assert 'environment' in summary
            assert 'version' in summary
            
            print("✓ Manager functionality successful")
            return True
            
    except Exception as e:
        print(f"✗ Manager functionality failed: {e}")
        return False

def test_integration_adapters():
    """Test Phase 1-3 integration adapters"""
    print("Testing integration adapters...")
    
    try:
        from schema import UnifiedConfigurationSchema
        from integrations import (
            RiskManagementConfigAdapter,
            MLIntegrationConfigAdapter,
            APISystemConfigAdapter
        )
        
        # Create test configuration
        test_config = UnifiedConfigurationSchema(
            environment='development',
            enable_trading=False
        )
        
        # Test Phase 1 adapter
        risk_adapter = RiskManagementConfigAdapter(test_config)
        risk_config = risk_adapter.get_risk_config()
        assert 'trading_mode' in risk_config
        assert 'base_balance' in risk_config
        
        # Test Phase 2.5 adapter
        ml_adapter = MLIntegrationConfigAdapter(test_config)
        ml_config = ml_adapter.get_ml_config()
        assert 'enable_ml_integration' in ml_config
        assert 'lookback_periods' in ml_config
        
        # Test Phase 3 adapter
        api_adapter = APISystemConfigAdapter(test_config)
        api_config = api_adapter.get_api_config()
        assert 'enable_trading' in api_config
        assert 'rate_limits' in api_config
        
        print("✓ Integration adapters successful")
        return True
        
    except Exception as e:
        print(f"✗ Integration adapters failed: {e}")
        return False

def test_config_file_operations():
    """Test configuration file operations"""
    print("Testing configuration file operations...")
    
    try:
        from manager import UnifiedConfigurationManager
        from schema import UnifiedConfigurationSchema
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = UnifiedConfigurationManager(workspace_root=temp_dir)
            
            # Create and save configuration
            config = manager.create_default_configuration()
            config_path = os.path.join(temp_dir, "test_config.yaml")
            
            manager.save_configuration(config, config_path)
            assert os.path.exists(config_path)
            
            # Load configuration
            loaded_config = manager.load_configuration(config_path)
            assert isinstance(loaded_config, UnifiedConfigurationSchema)
            assert loaded_config.environment == config.environment
            
            print("✓ Config file operations successful")
            return True
            
    except Exception as e:
        print(f"✗ Config file operations failed: {e}")
        return False

def test_validation_system():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    try:
        from manager import UnifiedConfigurationManager
        from schema import UnifiedConfigurationSchema
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = UnifiedConfigurationManager(workspace_root=temp_dir)
            
            # Test valid configuration
            valid_config = manager.create_default_configuration()
            manager._config = valid_config
            
            result = manager.validate_current_configuration()
            assert result.is_valid == True
            assert result.score >= 0.0
            
            print("✓ Validation system successful")
            return True
            
    except Exception as e:
        print(f"✗ Validation system failed: {e}")
        return False

def run_simplified_tests():
    """Run simplified integration tests"""
    print("="*60)
    print("SIMPLIFIED CONFIGURATION INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Schema Creation", test_schema_creation),
        ("Manager Functionality", test_manager_functionality),
        ("Integration Adapters", test_integration_adapters),
        ("Config File Operations", test_config_file_operations),
        ("Validation System", test_validation_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{passed + 1}/{total}] {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"Test {test_name} failed!")
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Configuration system is working correctly.")
    else:
        print(f"❌ {total - passed} tests failed. Review the errors above.")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_simplified_tests()
    exit(0 if success else 1)