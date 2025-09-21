#!/usr/bin/env python3
"""
Complete Deployment Validation & Guide
Comprehensive validation and step-by-step deployment instructions
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def print_header(title: str, char: str = "="):
    """Print formatted header"""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")

def print_status(item: str, status: bool, description: str = "", indent: int = 0):
    """Print status with formatting"""
    spaces = "  " * indent
    emoji = "✅" if status else "❌"
    desc = f" - {description}" if description else ""
    print(f"{spaces}{emoji} {item}{desc}")

def check_command(command: str) -> bool:
    """Check if command is available"""
    try:
        result = subprocess.run(command.split(), capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_docker_status() -> Tuple[bool, str]:
    """Check Docker installation and status"""
    if not check_command("docker --version"):
        return False, "Docker not installed"
    
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            return True, "Docker running"
        else:
            return False, "Docker not running"
    except Exception:
        return False, "Docker not accessible"

def validate_environment() -> Dict[str, bool]:
    """Validate environment setup"""
    print_header("🔍 ENVIRONMENT VALIDATION")
    
    checks = {}
    
    # Python version
    python_ok = sys.version_info >= (3, 8)
    print_status(f"Python {sys.version.split()[0]}", python_ok, "Required: >= 3.8")
    checks["python"] = python_ok
    
    # Required files
    required_files = [
        ".env", "config.py", "requirements.txt", 
        "Dockerfile", "docker-compose.yml", "start_api.py"
    ]
    
    for file in required_files:
        exists = Path(file).exists()
        print_status(file, exists, "Configuration file")
        checks[f"file_{file}"] = exists
    
    # Required directories
    required_dirs = ["src", "logs", "data", "backups"]
    for dir_name in required_dirs:
        exists = Path(dir_name).exists()
        print_status(f"{dir_name}/", exists, "Required directory")
        checks[f"dir_{dir_name}"] = exists
    
    # Docker status
    docker_available, docker_msg = check_docker_status()
    print_status("Docker", docker_available, docker_msg)
    checks["docker"] = docker_available
    
    # Configuration test
    try:
        from config import Config
        config = Config()
        config_ok = True
        print_status("Configuration System", True, "Loaded successfully")
    except Exception as e:
        config_ok = False
        print_status("Configuration System", False, f"Error: {e}")
    checks["config"] = config_ok
    
    return checks

def validate_dependencies() -> Dict[str, bool]:
    """Validate Python dependencies"""
    print_header("📦 DEPENDENCY VALIDATION")
    
    critical_deps = [
        ("fastapi", "API framework"),
        ("uvicorn", "ASGI server"),
        ("ccxt", "Exchange connectivity"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("sqlalchemy", "Database ORM"),
        ("streamlit", "Dashboard"),
        ("requests", "HTTP client")
    ]
    
    results = {}
    for package, description in critical_deps:
        try:
            __import__(package)
            print_status(package, True, description)
            results[package] = True
        except ImportError:
            print_status(package, False, f"MISSING - {description}")
            results[package] = False
    
    return results

def create_deployment_options() -> None:
    """Show deployment options"""
    print_header("🚀 DEPLOYMENT OPTIONS")
    
    print("1. 🐳 DOCKER DEPLOYMENT (Recommended)")
    print("   ├── Automated service management")
    print("   ├── Isolated environment") 
    print("   ├── Easy scaling and updates")
    print("   └── Production-ready setup")
    print("   ")
    print("   Commands:")
    print("   docker-compose up -d    # Start all services")
    print("   docker-compose logs -f  # View logs")
    print("   docker-compose down     # Stop services")
    print()
    
    print("2. 🔧 MANUAL DEPLOYMENT")
    print("   ├── Direct Python execution")
    print("   ├── Full control over processes")
    print("   ├── Easy debugging")
    print("   └── Suitable for development")
    print("   ")
    print("   Commands:")
    print("   python start_api.py     # Start API server")
    print("   streamlit run dashboard/main.py  # Start dashboard")
    print()
    
    print("3. 🖥️ SYSTEMD SERVICE (Linux Only)")
    print("   ├── System service integration")
    print("   ├── Automatic startup on boot")
    print("   ├── Process monitoring")
    print("   └── Production server deployment")
    print("   ")
    print("   Commands:")
    print("   sudo systemctl start bybit-trading-bot")
    print("   sudo systemctl enable bybit-trading-bot")

def create_step_by_step_guide() -> None:
    """Create comprehensive deployment guide"""
    print_header("📋 STEP-BY-STEP DEPLOYMENT GUIDE")
    
    print("🎯 PHASE 1: PREPARATION")
    print("  1. Ensure all files are present:")
    print("     ✓ .env file with your configuration")
    print("     ✓ config.py system validated")
    print("     ✓ All dependencies installed")
    print()
    
    print("  2. Configure API credentials:")
    print("     ✓ Get Bybit API keys (READ + TRADE only, NO WITHDRAW)")
    print("     ✓ Update .env with your actual API keys")
    print("     ✓ Set BYBIT_TESTNET=true for initial testing")
    print()
    
    print("🎯 PHASE 2: TESTING")
    print("  1. Test API connection:")
    print("     python -c \"from config import Config; Config().validate()\"")
    print()
    
    print("  2. Test individual components:")
    print("     python start_api.py  # Test API server")
    print("     # In another terminal:")
    print("     curl http://localhost:8001/health  # Verify API")
    print()
    
    print("🎯 PHASE 3: DEPLOYMENT")
    print("  Choose your deployment method:")
    print()
    
    print("  OPTION A: Docker (Recommended)")
    print("  ├── 1. Start Docker Desktop")
    print("  ├── 2. Build and run: docker-compose up -d")
    print("  ├── 3. Check health: curl http://localhost:8001/health")
    print("  └── 4. Monitor logs: docker-compose logs -f")
    print()
    
    print("  OPTION B: Manual")
    print("  ├── 1. Activate virtual environment")
    print("  ├── 2. Start API: python start_api.py")
    print("  ├── 3. Start dashboard: streamlit run dashboard/main.py")
    print("  └── 4. Monitor in separate terminals")
    print()
    
    print("🎯 PHASE 4: LIVE TRADING")
    print("  1. ⚠️  ONLY after successful testnet trading!")
    print("  2. Update .env: BYBIT_TESTNET=false")
    print("  3. Update .env: TRADING_ENABLED=true")
    print("  4. Start with small positions: POSITION_SIZE=0.001")
    print("  5. Monitor continuously for first 24 hours")
    print()

def create_troubleshooting_guide() -> None:
    """Create troubleshooting guide"""
    print_header("🔧 TROUBLESHOOTING GUIDE")
    
    print("❌ COMMON ISSUES & SOLUTIONS:")
    print()
    
    print("1. 'Docker not running'")
    print("   └── Start Docker Desktop application")
    print("   └── Wait for Docker to fully initialize")
    print("   └── Test: docker --version")
    print()
    
    print("2. 'Configuration errors'")
    print("   └── Check .env file exists and has correct values")
    print("   └── Verify API keys are valid")
    print("   └── Test: python -c \"from config import Config; Config().validate()\"")
    print()
    
    print("3. 'Import errors'")
    print("   └── Install dependencies: pip install -r requirements.txt")
    print("   └── Activate virtual environment if using one")
    print("   └── Check Python path includes project directory")
    print()
    
    print("4. 'API connection failed'")
    print("   └── Verify API keys are correct")
    print("   └── Check internet connection")
    print("   └── Verify Bybit API permissions (READ + TRADE)")
    print("   └── Check if IP restriction is blocking connection")
    print()
    
    print("5. 'Port already in use'")
    print("   └── Kill existing processes: pkill -f start_api.py")
    print("   └── Change port in configuration")
    print("   └── Use Docker which handles port mapping")
    print()

def run_quick_health_check() -> bool:
    """Run quick health check"""
    print_header("💊 QUICK HEALTH CHECK")
    
    checks = []
    
    # Test configuration
    try:
        from config import Config
        config = Config()
        print_status("Configuration", True, "System loaded")
        checks.append(True)
    except Exception as e:
        print_status("Configuration", False, f"Error: {e}")
        checks.append(False)
    
    # Test critical imports
    critical_imports = ["fastapi", "uvicorn", "ccxt", "pandas"]
    for package in critical_imports:
        try:
            __import__(package)
            print_status(f"{package} import", True)
            checks.append(True)
        except ImportError:
            print_status(f"{package} import", False, "Package missing")
            checks.append(False)
    
    # Test file structure
    critical_files = [".env", "config.py", "start_api.py"]
    for file in critical_files:
        exists = Path(file).exists()
        print_status(f"{file} exists", exists)
        checks.append(exists)
    
    success_rate = sum(checks) / len(checks) * 100
    overall_status = success_rate >= 90
    
    print_status(f"Overall Health", overall_status, f"{success_rate:.1f}% checks passed")
    
    return overall_status

def main():
    """Main validation and deployment guide"""
    print("🤖 BYBIT TRADING BOT - DEPLOYMENT VALIDATOR")
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run validations
    env_results = validate_environment()
    dep_results = validate_dependencies()
    
    # Calculate scores
    env_score = sum(env_results.values()) / len(env_results) * 100
    dep_score = sum(dep_results.values()) / len(dep_results) * 100
    overall_score = (env_score + dep_score) / 2
    
    print_header("📊 VALIDATION SUMMARY")
    print(f"Environment Score: {env_score:.1f}%")
    print(f"Dependencies Score: {dep_score:.1f}%")
    print(f"Overall Score: {overall_score:.1f}%")
    
    if overall_score >= 90:
        print_status("DEPLOYMENT STATUS", True, "READY FOR PRODUCTION")
    elif overall_score >= 70:
        print_status("DEPLOYMENT STATUS", True, "READY WITH MINOR ISSUES")
    else:
        print_status("DEPLOYMENT STATUS", False, "NOT READY - RESOLVE ISSUES")
    
    # Show guides
    create_deployment_options()
    create_step_by_step_guide()
    create_troubleshooting_guide()
    
    # Quick health check
    health_ok = run_quick_health_check()
    
    print_header("🎉 NEXT STEPS")
    if health_ok and overall_score >= 90:
        print("✅ Your system is ready for deployment!")
        print("📋 Follow the step-by-step guide above")
        print("⚠️  Remember to test on testnet first!")
        print("🚀 Configure your API keys and start trading!")
    else:
        print("⚠️  Please resolve the issues identified above")
        print("🔧 Use the troubleshooting guide for help")
        print("🔄 Run this script again after fixing issues")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_score": overall_score,
        "environment": env_results,
        "dependencies": dep_results,
        "production_ready": health_ok and overall_score >= 90
    }
    
    with open("deployment_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: deployment_validation_report.json")
    return health_ok and overall_score >= 90

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)