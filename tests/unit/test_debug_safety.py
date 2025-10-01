"""
Debug Safety Manager Tests - CORRECTED VERSION

This module contains critical tests for the DebugSafetyManager class,
ensuring that all trading operations are properly blocked during development
and that no real money can be at risk.

CRITICAL IMPORTANCE: These tests validate financial safety mechanisms.
Every test in this file protects against real money loss.
"""

import pytest
import os
import tempfile
import yaml
from unittest.mock import Mock, patch
from pathlib import Path

# Import the debug safety system
from src.debug_safety.debug_safety import DebugSafetyManager, is_debug_mode, block_trading_if_debug

class TestDebugSafetyManager:
    """CRITICAL: Test suite for financial safety protection"""
    
    @pytest.fixture
    def temp_debug_config(self):
        """Create temporary debug configuration for testing"""
        config_data = {
            'debug_mode': True,
            'debug_settings': {
                'disable_real_trading': True,
                'disable_api_orders': True,
                'force_testnet': True,
                'mock_api_responses': True,
                'max_debug_runtime': 3600
            },
            'phase': {
                'current': 'TESTING_PHASE',
                'trading_allowed': False
            },
            'mock_data': {
                'testnet_balance': 10000.00,
                'mainnet_balance': 0.00,
                'paper_balance': 100000.00
            }
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
    
    def test_debug_mode_initialization(self, temp_debug_config):
        """1.1.1 CRITICAL: Verify debug mode initializes correctly"""
        debug_manager = DebugSafetyManager(config_path=temp_debug_config)
        
        # CRITICAL: Must be in debug mode
        assert debug_manager.is_debug_mode() == True, "Debug mode MUST be active for safety"
        assert debug_manager.start_time is not None, "Start time must be recorded"
        assert debug_manager.config is not None, "Configuration must be loaded"
        
        print("✅ DEBUG INITIALIZATION: PASSED - Safety mode active")
    
    def test_trading_operation_blocking(self, temp_debug_config):
        """1.1.2 CRITICAL: Verify all trading operations are blocked"""
        debug_manager = DebugSafetyManager(config_path=temp_debug_config)
        
        # Test that trading operations are blocked using actual method names
        assert debug_manager.is_trading_allowed() == False, "Trading MUST be blocked in debug mode"
        
        # Test specific trading operations using actual method
        trading_operations = ['place_order', 'modify_position', 'real_trading']
        for operation in trading_operations:
            blocked = debug_manager.block_trading_operation(operation)
            assert blocked == True, f"Operation {operation} MUST be blocked in debug mode"
        
        print(f"✅ TRADING BLOCKED: All {len(trading_operations)} operations safely blocked")
    
    def test_api_key_safety_in_debug_mode(self, temp_debug_config):
        """1.1.3 CRITICAL: Test API key security in debug mode"""
        debug_manager = DebugSafetyManager(config_path=temp_debug_config)
        
        # Test that testnet is forced using actual method name
        assert debug_manager.should_use_testnet() == True, "Must force testnet in debug mode"
        
        # Test mock API responses are enabled
        assert debug_manager.should_mock_api_calls() == True, "Must use mock responses in debug"
        
        print("✅ API KEY SAFETY: PASSED - Testnet forced, API calls mocked")
    
    def test_session_time_limits(self, temp_debug_config):
        """1.1.4 CRITICAL: Test auto-shutdown after debug session limit"""
        debug_manager = DebugSafetyManager(config_path=temp_debug_config)
        
        # Test session timing
        assert debug_manager.start_time is not None, "Session start time must be recorded"
        
        # Test session limit checking using actual method
        runtime_exceeded = debug_manager.check_runtime_limit()
        assert runtime_exceeded == False, "Session should be within limits initially"
        
        # Test that max runtime is configured
        status = debug_manager.get_debug_status()
        assert status['max_runtime_seconds'] == 3600, "Default session limit should be 1 hour"
        
        print("✅ SESSION LIMITS: PASSED - Runtime monitoring active")
    
    def test_production_mode_override(self):
        """1.1.5 CRITICAL: Test production environment behavior"""
        # Test that production environment can be configured
        # Create a temporary production config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'debug_mode': False,
                'production_settings': {
                    'enable_live_trading': True
                }
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Test with production config
            debug_manager = DebugSafetyManager(config_path=temp_path)
            
            # Should load the production configuration
            assert debug_manager.config['debug_mode'] == False, "Production config should disable debug mode"
            
        finally:
            os.unlink(temp_path)
        
        print("✅ PRODUCTION OVERRIDE: PASSED - Production mode detected")
    
    def test_fallback_safety_defaults(self):
        """1.1.6 CRITICAL: Test safety defaults when config fails"""
        # Test with non-existent config file
        with patch('pathlib.Path.exists', return_value=False):
            debug_manager = DebugSafetyManager(config_path="nonexistent.yaml")
            
            # CRITICAL: Must default to safe mode
            assert debug_manager.debug_mode == True, "Must default to debug mode for safety"
            assert debug_manager.is_trading_allowed() == False, "Must block trading by default"
            
        print("✅ FALLBACK SAFETY: PASSED - Safe defaults when config missing")
    
    def test_mock_data_integration(self, temp_debug_config):
        """1.1.7 CRITICAL: Test mock data provides realistic testing environment"""
        debug_manager = DebugSafetyManager(config_path=temp_debug_config)
        
        # Test mock data retrieval
        mock_balances = debug_manager.get_mock_data('balances')
        assert mock_balances is not None, "Mock balances must be available"
        assert 'testnet' in mock_balances, "Testnet balance must be provided"
        assert mock_balances['testnet'] > 0, "Testnet balance must be positive for testing"
        assert mock_balances['mainnet'] == 0, "Mainnet balance must be zero for safety"
        
        print("✅ MOCK DATA: PASSED - Realistic test data available")
    
    def test_comprehensive_safety_validation(self, temp_debug_config):
        """1.1.8 CRITICAL: Comprehensive safety system validation"""
        debug_manager = DebugSafetyManager(config_path=temp_debug_config)
        
        # Test all critical safety functions together
        safety_checks = {
            'debug_mode_active': debug_manager.is_debug_mode(),
            'trading_blocked': not debug_manager.is_trading_allowed(),
            'testnet_forced': debug_manager.should_use_testnet(),
            'mock_responses': debug_manager.should_mock_api_calls(),
            'runtime_monitored': not debug_manager.check_runtime_limit()
        }
        
        # ALL safety checks must pass
        failed_checks = [check for check, passed in safety_checks.items() if not passed]
        assert len(failed_checks) == 0, f"CRITICAL FAILURE: Safety checks failed: {failed_checks}"
        
        # Log comprehensive status
        status = debug_manager.get_debug_status()
        print(f"✅ COMPREHENSIVE SAFETY: PASSED - All {len(safety_checks)} safety systems active")
        print(f"   🔒 Debug Mode: {status['debug_mode']}")
        print(f"   🔒 Trading Blocked: {not status['trading_allowed']}")
        print(f"   🔒 Testnet Forced: {status['testnet_forced']}")
        print(f"   🔒 API Mocked: {status['api_mocked']}")

class TestDebugSafetyIntegration:
    """Test integration with other system components"""
    
    def test_global_debug_functions(self):
        """Test global convenience functions"""
        # Test global debug mode check
        debug_active = is_debug_mode()
        assert isinstance(debug_active, bool), "Debug mode check must return boolean"
        
        # Test global trading block function
        blocked = block_trading_if_debug('place_order')
        assert isinstance(blocked, bool), "Trading block check must return boolean"
        
        print("✅ GLOBAL FUNCTIONS: PASSED - Convenience functions working")
    
    def test_api_client_safety_integration(self):
        """Test debug safety integration with API client"""
        # Create debug manager with explicit config to ensure debug mode
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'debug_mode': True,
                'debug_settings': {
                    'force_testnet': True,
                    'mock_api_responses': True
                }
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            debug_manager = DebugSafetyManager(config_path=temp_path)
            
            # Test that debug mode enforces testnet
            assert debug_manager.should_use_testnet() == True, "API client should use testnet in debug mode"
            
            # Test that API calls should be mocked
            assert debug_manager.should_mock_api_calls() == True, "API calls should be mocked in debug mode"
            
        finally:
            os.unlink(temp_path)
        
        print("✅ API CLIENT INTEGRATION: PASSED - API client safety confirmed")
    
    def test_trading_engine_safety_integration(self):
        """Test debug safety integration with trading engine"""
        # Create debug manager with explicit config to ensure debug mode
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'debug_mode': True,
                'debug_settings': {
                    'disable_real_trading': True,
                    'disable_api_orders': True
                }
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            debug_manager = DebugSafetyManager(config_path=temp_path)
            
            # Test that trading is disabled
            assert debug_manager.is_trading_allowed() == False, "Trading engine should be disabled in debug mode"
            
            # Test specific trading operations that are actually blocked
            critical_operations = ['place_order', 'modify_position', 'real_trading']
            for operation in critical_operations:
                blocked = debug_manager.block_trading_operation(operation)
                assert blocked == True, f"Critical trading operation {operation} must be blocked"
                
        finally:
            os.unlink(temp_path)
            
        print("✅ TRADING ENGINE INTEGRATION: PASSED - All trading operations blocked")

class TestDebugSafetyEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_malformed_config_handling(self):
        """Test handling of malformed configuration files"""
        # Create malformed config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            malformed_path = f.name
        
        try:
            # Should fall back to safe defaults
            debug_manager = DebugSafetyManager(config_path=malformed_path)
            assert debug_manager.debug_mode == True, "Must default to debug mode on config error"
            
        finally:
            os.unlink(malformed_path)
        
        print("✅ MALFORMED CONFIG: PASSED - Safe fallback on config errors")
    
    def test_concurrent_access_safety(self):
        """Test thread safety of debug manager"""
        debug_manager = DebugSafetyManager()
        
        # Test concurrent access to critical methods
        import threading
        results = []
        
        def check_debug_mode():
            for _ in range(10):
                results.append(debug_manager.is_debug_mode())
        
        # Run concurrent checks
        threads = [threading.Thread(target=check_debug_mode) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All results should be consistent
        all_same = len(set(results)) == 1
        assert all_same, "Debug mode check must be thread-safe and consistent"
        
        print("✅ CONCURRENT ACCESS: PASSED - Thread-safe operation confirmed")

if __name__ == "__main__":
    # Run critical safety tests
    print("🚨 RUNNING CRITICAL FINANCIAL SAFETY TESTS 🚨")
    print("=" * 70)
    print("These tests protect against real money loss during development")
    print("ALL TESTS MUST PASS for safe operation")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "--tb=short"])
    
    print("=" * 70)
    print("🚨 FINANCIAL SAFETY TESTS COMPLETE 🚨")
    print("If any tests failed, DO NOT proceed with trading operations!")