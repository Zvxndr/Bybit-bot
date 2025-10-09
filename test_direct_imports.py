#!/usr/bin/env python3
"""
Direct test of the import strategies without running the full application
"""

import sys
import os

# Simulate Docker environment
sys.path.insert(0, os.getcwd())

def test_imports_directly():
    print("🧪 DIRECT IMPORT TEST")
    print("=" * 30)
    
    # Test Multi-exchange provider import
    print("\n🔍 Testing Multi-exchange provider import...")
    try:
        # Test the relative import strategy
        from src.data.multi_exchange_provider import MultiExchangeDataManager
        print("✅ Multi-exchange provider import successful (src.data)")
    except ImportError as e:
        print(f"❌ Multi-exchange provider import failed: {e}")
    
    # Test AI Pipeline Manager import
    print("\n🔍 Testing AI Pipeline Manager import...")
    try:
        # Test the relative import strategy
        from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager
        print("✅ AI Pipeline Manager import successful (src.bot.pipeline)")
    except ImportError as e:
        print(f"❌ AI Pipeline Manager import failed: {e}")
    
    print("\n📋 Import test complete!")

if __name__ == "__main__":
    test_imports_directly()