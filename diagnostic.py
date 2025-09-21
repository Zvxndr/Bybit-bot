#!/usr/bin/env python3
"""
Bybit Trading Bot - Diagnostic Script
Quick health check and issue identification
"""

import sys
import os
from pathlib import Path

def check_python_path():
    """Check if src directory is in Python path"""
    print("🔍 Python Path Analysis")
    print("-" * 30)
    
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    
    print(f"Current directory: {current_dir}")
    print(f"Source directory: {src_dir}")
    print(f"Source exists: {src_dir.exists()}")
    
    if str(src_dir) not in sys.path:
        print("⚠️ Source directory not in Python path")
        print("Adding to sys.path...")
        sys.path.insert(0, str(src_dir))
    else:
        print("✅ Source directory in Python path")
    
    return str(src_dir)

def check_file_structure():
    """Check if critical files exist"""
    print("\n📁 File Structure Check")
    print("-" * 30)
    
    critical_files = [
        "src/bot/__init__.py",
        "src/bot/main.py", 
        "src/bot/core.py",
        "src/bot/core_components/__init__.py",
        "src/bot/core_components/config/__init__.py",
        "src/bot/core_components/config/manager.py",
        "src/bot/core_components/config/schema.py",
        "src/bot/api/__init__.py",
        "src/bot/risk/__init__.py"
    ]
    
    missing_files = []
    
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return missing_files

def test_imports():
    """Test critical imports"""
    print("\n🔌 Import Tests")
    print("-" * 30)
    
    import_tests = [
        ("bot", "Basic bot module"),
        ("bot.main", "Main entry point"),
        ("bot.core", "Core module"),
        ("bot.core_components.config", "Config module"),
        ("bot.core_components.config.manager", "Config manager"),
        ("bot.core_components.config.schema", "Config schema"),
        ("bot.api", "API module"),
        ("bot.risk", "Risk module")
    ]
    
    failed_imports = []
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - {description}")
        except ImportError as e:
            print(f"❌ {module_name} - {description}: {e}")
            failed_imports.append((module_name, str(e)))
        except SyntaxError as e:
            print(f"🚨 {module_name} - SYNTAX ERROR: {e}")
            failed_imports.append((module_name, f"SYNTAX: {e}"))
        except Exception as e:
            print(f"⚠️ {module_name} - OTHER ERROR: {e}")
            failed_imports.append((module_name, f"OTHER: {e}"))
    
    return failed_imports

def check_api_module_syntax():
    """Check API module for syntax issues"""
    print("\n🔍 API Module Syntax Check")
    print("-" * 30)
    
    api_init_path = Path("src/bot/api/__init__.py")
    
    if not api_init_path.exists():
        print("❌ API module __init__.py not found")
        return False
    
    try:
        with open(api_init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for common syntax issues
        lines = content.split('\n')
        
        for i, line in enumerate(lines[:10], 1):  # Check first 10 lines
            if line.strip().startswith('"""') and line.strip().endswith('"""') and len(line.strip()) > 6:
                continue
            elif '""""""' in line:
                print(f"🚨 Line {i}: Malformed docstring - {line}")
                return False
            elif line.strip().startswith('#') and 'API' in line and 'Consolidation' in line:
                if not line.strip().startswith('# '):
                    print(f"⚠️ Line {i}: Possible comment parsing issue - {line}")
        
        print("✅ No obvious syntax issues found in API module")
        return True
        
    except Exception as e:
        print(f"❌ Error reading API module: {e}")
        return False

def generate_fix_recommendations():
    """Generate specific fix recommendations"""
    print("\n🔧 Fix Recommendations")
    print("-" * 30)
    
    recommendations = []
    
    # Check if main.py has import issues
    main_path = Path("src/bot/main.py")
    if main_path.exists():
        try:
            with open(main_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "from bot.core.config.manager import" in content:
                recommendations.append({
                    "file": "src/bot/main.py",
                    "issue": "Absolute import should be relative",
                    "fix": "Change 'from bot.core.config.manager' to 'from .core.config.manager'"
                })
                
            if "DatabaseManager(None)" in content:
                recommendations.append({
                    "file": "src/bot/main.py", 
                    "issue": "Database manager initialized with None",
                    "fix": "Add proper config handling or make database optional"
                })
        except Exception as e:
            print(f"⚠️ Could not analyze main.py: {e}")
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['file']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Fix: {rec['fix']}")
        print()
    
    return recommendations

def main():
    """Run complete diagnostic"""
    print("🩺 Bybit Trading Bot - Diagnostic Report")
    print("=" * 50)
    
    # Set up environment
    src_dir = check_python_path()
    
    # Check file structure
    missing_files = check_file_structure()
    
    # Test imports
    failed_imports = test_imports()
    
    # Check API syntax
    api_syntax_ok = check_api_module_syntax()
    
    # Generate recommendations
    recommendations = generate_fix_recommendations()
    
    # Summary
    print("\n📊 Diagnostic Summary")
    print("=" * 30)
    
    total_issues = len(missing_files) + len(failed_imports) + (0 if api_syntax_ok else 1)
    
    if total_issues == 0:
        print("🎉 No critical issues found! Bot should be able to start.")
        return 0
    else:
        print(f"⚠️ Found {total_issues} critical issues:")
        print(f"   - {len(missing_files)} missing files")
        print(f"   - {len(failed_imports)} import failures")
        print(f"   - {'0' if api_syntax_ok else '1'} syntax errors")
        
        print("\n🚨 Bot will NOT start until these issues are resolved.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)