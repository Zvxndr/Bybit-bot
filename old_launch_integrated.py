#!/usr/bin/env python3
"""
Integrated Trading Application Launcher
=====================================

Launch script for the fully integrated Bybit trading application.
This handles environment setup and starts the integrated backend.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup environment variables for the integrated application"""
    print("🔧 Setting up environment...")
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Set Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    # Set default environment variables if not already set
    env_vars = {
        'BYBIT_TESTNET': 'true',  # Force testnet for safety
        'BYBIT_ENVIRONMENT': 'testnet',
        'PAPER_TRADING_BALANCE': '100000',
        'LOG_LEVEL': 'INFO'
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"✅ Set {key}={value}")
    
    # Check for API credentials
    api_key = os.getenv('BYBIT_API_KEY') or os.getenv('BYBIT_TESTNET_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET') or os.getenv('BYBIT_TESTNET_API_SECRET')
    
    if api_key and api_secret:
        print(f"✅ API credentials found: {api_key[:8]}...")
        print("🔗 Will connect to Bybit testnet API for real data")
    else:
        print("⚠️ No API credentials found - will run in paper trading mode")
        print("💡 To connect real API, set environment variables:")
        print("   BYBIT_API_KEY=your_testnet_api_key")
        print("   BYBIT_API_SECRET=your_testnet_api_secret")
    
    print("✅ Environment setup complete")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔧 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'aiohttp',
        'pydantic'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is missing")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("📦 Install with: pip install fastapi uvicorn aiohttp pydantic")
        return False
    
    print("✅ All dependencies are installed")
    return True

def launch_integrated_app():
    """Launch the integrated application"""
    print("\n" + "="*60)
    print("🚀 LAUNCHING INTEGRATED BYBIT TRADING APPLICATION")
    print("="*60)
    
    # Check if we should use the integrated version
    integrated_main = Path("src/main_integrated.py")
    
    if integrated_main.exists():
        print("✅ Using integrated backend with real Bybit API support")
        app_path = "src.main_integrated:main"
    else:
        print("⚠️ Fallback to simplified main")
        app_path = "src.main:main"
    
    print(f"📱 Frontend will be available at: http://localhost:8080")
    print(f"📊 API documentation at: http://localhost:8080/docs")
    print(f"🔍 Health check at: http://localhost:8080/health")
    print(f"📈 Dashboard API at: http://localhost:8080/api/dashboard")
    print("\n🚀 Starting application...")
    
    try:
        # Import and run the integrated application
        if integrated_main.exists():
            from src.main_integrated import main
            main()
        else:
            from src.main import main
            main()
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Application failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main launcher function"""
    try:
        print("🔥 Bybit Trading Bot - Integrated Launcher")
        print("=" * 50)
        
        # Setup environment
        setup_environment()
        
        # Check dependencies
        if not check_dependencies():
            print("\n❌ Please install missing dependencies before launching")
            sys.exit(1)
        
        # Launch the application
        success = launch_integrated_app()
        
        if success:
            print("✅ Application launched successfully")
        else:
            print("❌ Application failed to launch")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Launcher error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()