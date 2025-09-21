"""
Minimal Configuration Test - Phase 4 Testing

Simplified configuration system test to verify basic functionality without circular references.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

# Minimal configuration schema for testing
class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class TradingMode(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"

class ExchangeCredentials(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    is_testnet: bool = True

class MinimalConfigurationSchema(BaseModel):
    environment: Environment = Environment.DEVELOPMENT
    version: str = "1.0.0"
    enable_trading: bool = False
    exchange: Dict[Environment, ExchangeCredentials] = Field(default_factory=dict)

def test_minimal_configuration():
    """Test minimal configuration system"""
    print("Testing minimal configuration system...")
    
    try:
        # Test 1: Basic schema creation
        config_data = {
            'environment': 'development',
            'version': '1.0.0',
            'enable_trading': False
        }
        
        config = MinimalConfigurationSchema(**config_data)
        assert config.environment == Environment.DEVELOPMENT
        assert config.version == "1.0.0"
        assert config.enable_trading == False
        print("✓ Basic schema creation successful")
        
        # Test 2: With exchange credentials
        config_with_creds = MinimalConfigurationSchema(
            environment=Environment.DEVELOPMENT,
            exchange={
                Environment.DEVELOPMENT: ExchangeCredentials(
                    api_key="test_key",
                    api_secret="test_secret",
                    is_testnet=True
                )
            }
        )
        
        dev_creds = config_with_creds.exchange.get(Environment.DEVELOPMENT)
        assert dev_creds is not None
        assert dev_creds.api_key == "test_key"
        print("✓ Exchange credentials configuration successful")
        
        # Test 3: Configuration summary
        summary = {
            'environment': config.environment.value,
            'version': config.version,
            'trading_enabled': config.enable_trading,
            'has_credentials': len(config.exchange) > 0
        }
        assert summary['environment'] == 'development'
        print("✓ Configuration summary successful")
        
        # Test 4: Environment switching
        prod_config = MinimalConfigurationSchema(
            environment=Environment.PRODUCTION,
            enable_trading=True
        )
        assert prod_config.environment == Environment.PRODUCTION
        print("✓ Environment switching successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Minimal configuration test failed: {e}")
        return False

def test_configuration_file_operations():
    """Test configuration file operations"""
    print("Testing configuration file operations...")
    
    try:
        import yaml
        
        # Create test configuration
        config = MinimalConfigurationSchema(
            environment=Environment.DEVELOPMENT,
            version="1.0.0",
            enable_trading=False
        )
        
        # Test saving to file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")
            
            # Save configuration
            config_dict = config.model_dump()
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f)
            
            assert os.path.exists(config_path)
            print("✓ Configuration file saving successful")
            
            # Load configuration
            with open(config_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            loaded_config = MinimalConfigurationSchema(**loaded_data)
            assert loaded_config.environment == config.environment
            assert loaded_config.version == config.version
            print("✓ Configuration file loading successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration file operations test failed: {e}")
        return False

def test_integration_adapter_pattern():
    """Test integration adapter pattern"""
    print("Testing integration adapter pattern...")
    
    try:
        # Create test configuration
        config = MinimalConfigurationSchema(
            environment=Environment.DEVELOPMENT,
            enable_trading=False
        )
        
        # Mock Phase 1 adapter
        class MockRiskAdapter:
            def __init__(self, config):
                self.config = config
            
            def get_risk_config(self):
                return {
                    'environment': self.config.environment.value,
                    'enable_trading': self.config.enable_trading,
                    'version': self.config.version
                }
        
        # Test adapter
        adapter = MockRiskAdapter(config)
        risk_config = adapter.get_risk_config()
        
        assert 'environment' in risk_config
        assert 'enable_trading' in risk_config
        assert risk_config['environment'] == 'development'
        print("✓ Integration adapter pattern successful")
        
        # Mock Phase 3 adapter
        class MockAPIAdapter:
            def __init__(self, config):
                self.config = config
            
            def get_api_config(self):
                return {
                    'enable_trading': self.config.enable_trading,
                    'credentials': self.config.exchange.get(self.config.environment, {})
                }
        
        api_adapter = MockAPIAdapter(config)
        api_config = api_adapter.get_api_config()
        
        assert 'enable_trading' in api_config
        assert 'credentials' in api_config
        print("✓ API adapter pattern successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration adapter pattern test failed: {e}")
        return False

def run_minimal_tests():
    """Run minimal configuration tests"""
    print("="*60)
    print("MINIMAL CONFIGURATION SYSTEM TESTS")
    print("="*60)
    
    tests = [
        ("Minimal Configuration", test_minimal_configuration),
        ("Configuration File Operations", test_configuration_file_operations),
        ("Integration Adapter Pattern", test_integration_adapter_pattern),
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
    print("MINIMAL TEST RESULTS")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All minimal tests passed! Core configuration concepts are working.")
        print("📋 Ready to proceed with Phase 1-3 system integration.")
    else:
        print(f"❌ {total - passed} tests failed.")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_minimal_tests()
    exit(0 if success else 1)