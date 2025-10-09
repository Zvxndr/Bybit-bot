#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE DEPLOYMENT STATUS
=================================

1. Validates local code fixes
2. Checks deployment status  
3. Provides troubleshooting guidance
"""

import sys
import os
import importlib.util
from pathlib import Path

def validate_local_fixes():
    """Validate that our production fixes are in place"""
    print("🔧 VALIDATING LOCAL FIXES")
    print("=" * 50)
    
    fixes_status = {}
    
    # Check 1: main.py imports fixed
    main_py_path = Path("src/main.py")
    if main_py_path.exists():
        content = main_py_path.read_text(encoding='utf-8')
        
        # Check for enhanced import logic
        has_import_fallback = "try:" in content and "from src.bot.risk.ml_risk_manager" in content
        fixes_status["import_fallback"] = has_import_fallback
        print(f"   {'✅' if has_import_fallback else '❌'} Import fallback logic: {has_import_fallback}")
        
        # Check for CircuitBreakerType configuration
        has_circuit_breaker = "CircuitBreakerType" in content and "circuit_breaker_thresholds" in content
        fixes_status["circuit_breaker_config"] = has_circuit_breaker
        print(f"   {'✅' if has_circuit_breaker else '❌'} CircuitBreakerType config: {has_circuit_breaker}")
        
    else:
        print("   ❌ main.py not found")
        fixes_status["main_py_exists"] = False
    
    # Check 2: Frontend dashboard unified
    dashboard_path = Path("frontend/unified_dashboard.html") 
    if dashboard_path.exists():
        content = dashboard_path.read_text(encoding='utf-8')
        
        # Check for unified styling
        has_unified_styling = content.count("#10b981") >= 2  # Both charts should have this color
        fixes_status["unified_styling"] = has_unified_styling
        print(f"   {'✅' if has_unified_styling else '❌'} Unified chart styling: {has_unified_styling}")
        
        # Check for save configuration
        has_save_config = "savePipelineConfig" in content
        fixes_status["save_config"] = has_save_config
        print(f"   {'✅' if has_save_config else '❌'} Save configuration: {has_save_config}")
        
    else:
        print("   ❌ unified_dashboard.html not found")
        fixes_status["dashboard_exists"] = False
    
    # Check 3: Production tooling
    production_files = [
        "production_audit.py",
        "startup_tests.py", 
        "health_check.py",
        "deployment_monitor.py"
    ]
    
    for file_name in production_files:
        exists = Path(file_name).exists()
        fixes_status[f"{file_name}_exists"] = exists
        print(f"   {'✅' if exists else '❌'} {file_name}: {exists}")
    
    return fixes_status

def check_python_imports():
    """Check if critical imports work locally"""
    print("\n🐍 TESTING PYTHON IMPORTS")
    print("=" * 50)
    
    import_results = {}
    
    # Test critical imports
    test_imports = [
        "src.bot.risk.ml_risk_manager",
        "src.data.multi_exchange_provider", 
        "src.bot.data",
        "fastapi",
        "uvicorn"
    ]
    
    for import_name in test_imports:
        try:
            if "." in import_name:
                # Handle module imports
                parts = import_name.split(".")
                module = importlib.import_module(import_name)
                success = True
            else:
                # Handle package imports
                importlib.import_module(import_name)
                success = True
                
            import_results[import_name] = success
            print(f"   ✅ {import_name}: OK")
            
        except ImportError as e:
            import_results[import_name] = False
            print(f"   ❌ {import_name}: {e}")
        except Exception as e:
            import_results[import_name] = False
            print(f"   ❓ {import_name}: {e}")
    
    return import_results

def check_git_status():
    """Check git status for deployment sync"""
    print("\n📦 GIT STATUS CHECK")
    print("=" * 50)
    
    import subprocess
    
    try:
        # Check if we have uncommitted changes
        result = subprocess.run(["git", "status", "--porcelain"], 
                               capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print("   ⚠️ Uncommitted changes detected:")
            print(f"   {result.stdout}")
            print("   Consider committing and pushing latest changes.")
            return False
        else:
            print("   ✅ Working tree clean")
            
        # Check last commit
        result = subprocess.run(["git", "log", "-1", "--oneline"], 
                               capture_output=True, text=True, check=True)
        print(f"   📝 Last commit: {result.stdout.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Git error: {e}")
        return False
    except FileNotFoundError:
        print("   ❌ Git not found")
        return False

def generate_troubleshooting_guide(fixes_status, import_results):
    """Generate troubleshooting guidance"""
    print("\n🔧 TROUBLESHOOTING GUIDE")
    print("=" * 50)
    
    issues_found = []
    
    # Check for critical issues
    if not fixes_status.get("import_fallback", False):
        issues_found.append("Import fallback logic missing in main.py")
        
    if not fixes_status.get("circuit_breaker_config", False):
        issues_found.append("CircuitBreakerType configuration missing")
        
    if not any(import_results.values()):
        issues_found.append("Critical Python imports failing")
    
    if issues_found:
        print("🚨 ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
            
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("   1. Re-run the production fixes if imports are failing")
        print("   2. Check Python environment and dependencies")
        print("   3. Verify DigitalOcean build logs in App Platform console")
        print("   4. Consider running local startup tests: python startup_tests.py")
        
    else:
        print("✅ LOCAL ENVIRONMENT LOOKS GOOD")
        print("\nIf deployment still not working:")
        print("   1. Check DigitalOcean App Platform console")
        print("   2. Review build and runtime logs")
        print("   3. Verify app name and domain settings") 
        print("   4. Consider redeploying from DigitalOcean console")

def main():
    """Main validation function"""
    print("🔍 COMPREHENSIVE DEPLOYMENT STATUS CHECK")
    print("=" * 60)
    print()
    
    # Run all checks
    fixes_status = validate_local_fixes()
    import_results = check_python_imports()
    git_clean = check_git_status()
    
    # Generate guidance
    generate_troubleshooting_guide(fixes_status, import_results)
    
    # Overall status
    print("\n" + "=" * 60)
    critical_fixes = ["import_fallback", "circuit_breaker_config"]
    fixes_ok = all(fixes_status.get(fix, False) for fix in critical_fixes)
    imports_ok = any(import_results.values())
    
    if fixes_ok and imports_ok and git_clean:
        print("🟢 LOCAL STATUS: READY FOR DEPLOYMENT")
        return 0
    elif fixes_ok and imports_ok:
        print("🟡 LOCAL STATUS: MOSTLY READY (check git sync)")
        return 0  
    else:
        print("🔴 LOCAL STATUS: ISSUES DETECTED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)