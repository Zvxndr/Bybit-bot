#!/usr/bin/env python3
"""
Test Dynamic Risk Scaling
=========================

Quick test to verify the dynamic risk scaling system is working correctly
with high risk for low balances and falloff to 100k.
"""

import sys
import os
import yaml
import math
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from private_mode_launcher import calculate_dynamic_risk_ratio

def load_config():
    """Load the private use configuration"""
    config_path = Path("config/private_use.yaml")
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return None
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_dynamic_risk_scaling():
    """Test the dynamic risk scaling calculations"""
    print("🧮 Testing Dynamic Risk Scaling System")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    if not config:
        return False
    
    # Check if dynamic scaling is enabled
    private_mode = config.get('trading', {}).get('private_mode', {})
    scaling_config = private_mode.get('dynamic_risk_scaling', {})
    
    if not scaling_config.get('enabled', False):
        print("❌ Dynamic risk scaling is DISABLED in config")
        print("   Check config/private_use.yaml - dynamic_risk_scaling.enabled should be true")
        return False
    
    print("✅ Dynamic risk scaling is ENABLED")
    print(f"   Small account risk: {scaling_config.get('base_risk_small_accounts', 0.02) * 100:.1f}%")
    print(f"   Large account risk: {scaling_config.get('base_risk_large_accounts', 0.005) * 100:.1f}%")
    print(f"   Transition range: ${scaling_config.get('transition_start', 10000):,} → ${scaling_config.get('transition_end', 100000):,}")
    print()
    
    # Test various balance levels
    test_balances = [1000, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 150000, 200000]
    
    print("💰 Risk Scaling Examples:")
    print("Balance".ljust(12) + "Risk %".ljust(10) + "Risk Amount*".ljust(15) + "Position Size**")
    print("-" * 50)
    
    for balance in test_balances:
        risk_ratio = calculate_dynamic_risk_ratio(balance, config)
        risk_amount = balance * risk_ratio
        position_size = risk_amount / 0.02  # Assuming 2% stop loss
        
        print(f"${balance:,}".ljust(12) + f"{risk_ratio * 100:.2f}%".ljust(10) + f"${risk_amount:.0f}".ljust(15) + f"${position_size:.0f}")
    
    print("\n* Risk amount per trade")
    print("** Position size with 2% stop loss")
    print()
    
    # Verify the scaling behavior
    small_risk = calculate_dynamic_risk_ratio(5000, config)
    medium_risk = calculate_dynamic_risk_ratio(50000, config)  
    large_risk = calculate_dynamic_risk_ratio(150000, config)
    
    print("🔍 Scaling Verification:")
    print(f"   Small account (${5000:,}): {small_risk * 100:.2f}% - Should be ~2.00%")
    print(f"   Medium account (${50000:,}): {medium_risk * 100:.2f}% - Should be between 0.5% and 2.0%")
    print(f"   Large account (${150000:,}): {large_risk * 100:.2f}% - Should be ~0.50%")
    
    # Validation checks
    success = True
    
    if small_risk < 0.015:  # Should be close to 2%
        print("❌ Small account risk too low")
        success = False
    
    if large_risk > 0.008:  # Should be close to 0.5%
        print("❌ Large account risk too high")
        success = False
        
    if medium_risk <= large_risk or medium_risk >= small_risk:
        print("❌ Medium account risk not scaling properly")
        success = False
    
    if success:
        print("\n✅ Dynamic risk scaling is working correctly!")
        print("   • High risk for small accounts to enable growth")
        print("   • Smooth exponential decay transition")
        print("   • Conservative risk for large accounts to preserve wealth")
    else:
        print("\n❌ Dynamic risk scaling has issues - check configuration")
    
    return success

if __name__ == "__main__":
    print("🚀 Bybit Trading Bot - Dynamic Risk Scaling Test")
    print()
    
    try:
        success = test_dynamic_risk_scaling()
        
        if success:
            print("\n🎯 RESULT: Dynamic risk scaling system is ACTIVE and working correctly")
            print("   Your bot will use higher risk for small balances and scale down to 0.5% at $100k+")
        else:
            print("\n⚠️  RESULT: Dynamic risk scaling system needs attention")
            
        print(f"\nTest completed at {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()