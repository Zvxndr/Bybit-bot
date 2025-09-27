"""
ML Integration Verification Script

Quick verification that all ML components mentioned in the SAR exist and are accessible.
This validates the accuracy of our System Architecture Reference.

Usage: python verify_ml_components.py
"""

import sys
from pathlib import Path
import importlib.util
from typing import Dict, List, Tuple, Optional

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))


def check_file_exists(file_path: str) -> Tuple[bool, str]:
    """Check if a file exists and return status."""
    path = Path(file_path)
    if path.exists():
        try:
            lines = len(path.read_text(encoding='utf-8').splitlines()) if path.suffix == '.py' else 0
            return True, f"✅ EXISTS ({lines} lines)" if lines > 0 else "✅ EXISTS"
        except UnicodeDecodeError:
            # Fallback for files with different encoding
            try:
                lines = len(path.read_text(encoding='latin-1').splitlines()) if path.suffix == '.py' else 0
                return True, f"✅ EXISTS ({lines} lines)" if lines > 0 else "✅ EXISTS"
            except:
                return True, "✅ EXISTS (encoding issue)"
    else:
        return False, "❌ NOT FOUND"


def check_import(module_path: str, component_name: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        if '/' in module_path:
            # File path
            spec = importlib.util.spec_from_file_location("test_module", module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if component_name:
                    return hasattr(module, component_name), f"✅ IMPORTS ({component_name} found)" if hasattr(module, component_name) else f"⚠️ IMPORTS (no {component_name})"
                return True, "✅ IMPORTS"
        else:
            # Module import
            module = importlib.import_module(module_path)
            return True, "✅ IMPORTS"
    except Exception as e:
        return False, f"❌ IMPORT ERROR: {str(e)[:50]}..."
    
    return False, "❌ UNKNOWN ERROR"


def main():
    """Verify all ML components mentioned in SAR."""
    
    print("🔍 ML COMPONENT VERIFICATION")
    print("="*50)
    print("Verifying all components mentioned in System Architecture Reference...")
    print()
    
    # Core ML components to verify
    components = {
        "ML Integration Files": [
            "src/ml_dashboard_integration.py",
            "src/fire_ml_dashboard.py", 
            "activate_ml_engine.py"
        ],
        
        "ML Engine Components (From SAR)": [
            "src/bot/ml/ensemble_manager.py",
            "src/bot/machine_learning/prediction_engine.py",
            "src/bot/integration/ml_strategy_orchestrator.py",
            "src/bot/ml_strategy_discovery/ml_engine.py",
            "src/bot/strategy_graduation.py",
            "src/bot/backtesting/enhanced_backtester.py"
        ],
        
        "ML Model Files": [
            "src/bot/ml/models.py",
            "src/bot/ml/__init__.py",
            "src/bot/integration/__init__.py",
            "src/bot/integration/ml_model_manager.py"
        ],
        
        "Dashboard & Theme": [
            "src/static/css/fire-cybersigilism.css",
            "src/shared_state.py",
            "src/frontend_server.py"
        ],
        
        "Speed Demon Components": [
            "src/bot/data/historical_data_manager.py",
            "src/bot/speed_demon_integration.py",
            "scripts/speed_demon_deploy.py"
        ]
    }
    
    # Verification results
    total_components = 0
    found_components = 0
    
    for category, files in components.items():
        print(f"📂 {category}:")
        
        for file_path in files:
            total_components += 1
            exists, status = check_file_exists(file_path)
            
            if exists:
                found_components += 1
                
            print(f"   {status} {file_path}")
        
        print()
    
    # Summary
    success_rate = (found_components / total_components) * 100
    print("="*50)
    print(f"📊 VERIFICATION SUMMARY")
    print(f"✅ Found: {found_components}/{total_components} components")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🚀 SAR ACCURACY: EXCELLENT - All critical components verified")
    elif success_rate >= 75:
        print("✅ SAR ACCURACY: GOOD - Most components verified") 
    elif success_rate >= 50:
        print("⚠️ SAR ACCURACY: PARTIAL - Some components missing")
    else:
        print("❌ SAR ACCURACY: POOR - Many components missing")
    
    print()
    
    # Check key classes exist in files
    print("🔍 KEY CLASS VERIFICATION:")
    key_classes = [
        ("src/bot/ml/ensemble_manager.py", "EnsembleModelManager"),
        ("src/bot/machine_learning/prediction_engine.py", "PredictionEngine"),
        ("src/bot/integration/ml_strategy_orchestrator.py", "MLStrategyOrchestrator"),
        ("src/ml_dashboard_integration.py", "MLDashboardIntegration"),
        ("src/fire_ml_dashboard.py", "FireMLDashboard")
    ]
    
    class_found = 0
    for file_path, class_name in key_classes:
        exists, status = check_import(file_path, class_name)
        if exists:
            class_found += 1
        print(f"   {status} {file_path} → {class_name}")
    
    print()
    print(f"📊 Key Classes: {class_found}/{len(key_classes)} found")
    
    # Final recommendation
    print("="*50)
    if success_rate >= 85 and class_found >= len(key_classes) * 0.8:
        print("🔥 RECOMMENDATION: PROCEED WITH ML ACTIVATION")
        print("   → Run: python activate_ml_engine.py")
        print("   → All critical ML components are in place")
    else:
        print("⚠️ RECOMMENDATION: CHECK MISSING COMPONENTS")
        print("   → Some ML components may need attention")
        print("   → Verify file paths and implementations")
    
    print("="*50)


if __name__ == "__main__":
    main()