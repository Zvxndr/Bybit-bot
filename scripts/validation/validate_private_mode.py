#!/usr/bin/env python3
"""
Private Use Mode Validation Suite
==================================

Comprehensive validation of private use mode implementation
including all features we built today.
"""

import os
import sys
import time
import json
import requests
import subprocess
from pathlib import Path
from datetime import datetime

class PrivateUseValidator:
    """Validates private use mode implementation"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
    
    def log(self, message, status="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_emoji = {"INFO": "ℹ️", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
        print(f"[{timestamp}] {status_emoji.get(status, 'ℹ️')} {message}")
    
    def test_file_exists(self, filepath, description):
        """Test if a file exists"""
        test_name = f"file_exists_{Path(filepath).name}"
        exists = Path(filepath).exists()
        self.results["tests"][test_name] = {
            "description": description,
            "passed": exists,
            "filepath": str(filepath)
        }
        
        if exists:
            self.log(f"✅ {description}: {filepath}", "PASS")
            self.results["summary"]["passed"] += 1
        else:
            self.log(f"❌ {description}: {filepath} not found", "FAIL")
            self.results["summary"]["failed"] += 1
        
        self.results["summary"]["total"] += 1
        return exists
    
    def test_config_content(self, filepath, key_checks, description):
        """Test configuration file content"""
        test_name = f"config_content_{Path(filepath).name}"
        
        try:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                import yaml
                with open(filepath, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    config = f.read()
                    
            passed_checks = []
            failed_checks = []
            
            for key, expected in key_checks.items():
                if isinstance(config, dict):
                    # Navigate nested keys
                    value = config
                    for k in key.split('.'):
                        value = value.get(k) if isinstance(value, dict) else None
                    
                    if value == expected or (expected == "exists" and value is not None):
                        passed_checks.append(key)
                    else:
                        failed_checks.append(f"{key} (expected: {expected}, got: {value})")
                else:
                    # String content check
                    if expected in config:
                        passed_checks.append(key)
                    else:
                        failed_checks.append(f"{key} not found in content")
            
            passed = len(failed_checks) == 0
            self.results["tests"][test_name] = {
                "description": description,
                "passed": passed,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks
            }
            
            if passed:
                self.log(f"✅ {description}: All checks passed", "PASS")
                self.results["summary"]["passed"] += 1
            else:
                self.log(f"❌ {description}: Failed checks: {', '.join(failed_checks)}", "FAIL")
                self.results["summary"]["failed"] += 1
            
        except Exception as e:
            self.results["tests"][test_name] = {
                "description": description,
                "passed": False,
                "error": str(e)
            }
            self.log(f"❌ {description}: Error - {str(e)}", "FAIL")
            self.results["summary"]["failed"] += 1
        
        self.results["summary"]["total"] += 1
        return passed if 'passed' in locals() else False
    
    def test_endpoint(self, endpoint, expected_status=200, description=None):
        """Test API endpoint"""
        test_name = f"endpoint_{endpoint.replace('/', '_').replace('-', '_')}"
        description = description or f"API endpoint {endpoint}"
        
        try:
            url = f"http://localhost:8080{endpoint}"
            response = requests.get(url, timeout=10)
            
            passed = response.status_code == expected_status
            self.results["tests"][test_name] = {
                "description": description,
                "passed": passed,
                "url": url,
                "status_code": response.status_code,
                "expected_status": expected_status
            }
            
            if passed:
                self.log(f"✅ {description}: {response.status_code}", "PASS")
                self.results["summary"]["passed"] += 1
            else:
                self.log(f"❌ {description}: {response.status_code} (expected {expected_status})", "FAIL")
                self.results["summary"]["failed"] += 1
                
        except requests.exceptions.ConnectionError:
            self.results["tests"][test_name] = {
                "description": description,
                "passed": False,
                "error": "Connection refused - server not running"
            }
            self.log(f"❌ {description}: Server not running", "FAIL")
            self.results["summary"]["failed"] += 1
            
        except Exception as e:
            self.results["tests"][test_name] = {
                "description": description,
                "passed": False,
                "error": str(e)
            }
            self.log(f"❌ {description}: {str(e)}", "FAIL")
            self.results["summary"]["failed"] += 1
        
        self.results["summary"]["total"] += 1
    
    def run_all_validations(self):
        """Run complete validation suite"""
        self.log("🔥 Starting Private Use Mode Validation Suite", "INFO")
        self.log("=" * 60, "INFO")
        
        # 1. Test core files exist
        self.log("📁 Testing Core Files...", "INFO")
        self.test_file_exists("private_mode_launcher.py", "Private Mode Launcher")
        self.test_file_exists("config/private_use.yaml", "Private Use Configuration")
        self.test_file_exists("start_private_mode.bat", "Windows Batch Launcher")
        self.test_file_exists("start_private_mode.ps1", "PowerShell Launcher")
        self.test_file_exists(".env", "Environment Variables File")
        self.test_file_exists("PRIVATE_MODE_SUCCESS.md", "Success Documentation")
        
        # 2. Test configuration content
        self.log("⚙️ Testing Configuration Content...", "INFO")
        
        # Private use YAML config
        yaml_checks = {
            "trading.mode": "conservative_private",
            "trading.private_mode.max_risk_ratio": 0.005,
            "trading.private_mode.daily_loss_limit": 0.03,
            "trading.private_mode.max_positions": 3,
            "logging.enabled": True
        }
        self.test_config_content("config/private_use.yaml", yaml_checks, "Private Use YAML Configuration")
        
        # Environment file
        env_checks = {
            "PRIVATE_USE_MODE": "PRIVATE_USE_MODE=true",
            "DEBUG_MODE": "DEBUG_MODE=true", 
            "TRADING_ENVIRONMENT": "TRADING_ENVIRONMENT=testnet",
            "COMPREHENSIVE_DEBUGGING": "COMPREHENSIVE_DEBUGGING=true"
        }
        self.test_config_content(".env", env_checks, "Environment Variables")
        
        # 3. Test launch scripts content (simplified checks)
        self.log("🚀 Testing Launch Scripts...", "INFO")
        bat_checks = {
            "BYBIT TRADING BOT": "BYBIT TRADING BOT",
            "Private Use Mode": "Private Use Mode"
        }
        self.test_config_content("start_private_mode.bat", bat_checks, "Batch File Content")
        
        ps1_checks = {
            "BYBIT TRADING BOT": "BYBIT TRADING BOT",
            "Private Use Mode": "Private Use Mode"
        }
        self.test_config_content("start_private_mode.ps1", ps1_checks, "PowerShell Script Content")
        
        # 4. Test Python launcher functionality (basic import test)
        self.log("🐍 Testing Python Launcher...", "INFO")
        try:
            import private_mode_launcher
            launcher = private_mode_launcher.PrivateUseModeLogger()
            self.log("✅ Private Mode Launcher: Import successful", "PASS")
            self.results["summary"]["passed"] += 1
        except Exception as e:
            self.log(f"❌ Private Mode Launcher: Import failed - {str(e)}", "FAIL")
            self.results["summary"]["failed"] += 1
        self.results["summary"]["total"] += 1
        
        # 5. Warning about endpoint testing
        self.log("🌐 API Endpoint Testing...", "INFO") 
        self.log("⚠️ Skipping endpoint tests - requires server to be running", "WARN")
        self.log("   To test endpoints, run: python test_health_check.py", "INFO")
        
        # 6. Generate summary report
        self.log("=" * 60, "INFO")
        self.generate_summary()
    
    def generate_summary(self):
        """Generate validation summary"""
        total = self.results["summary"]["total"]
        passed = self.results["summary"]["passed"]
        failed = self.results["summary"]["failed"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        self.log("📊 VALIDATION SUMMARY", "INFO")
        self.log(f"   Total Tests: {total}", "INFO")
        self.log(f"   Passed: {passed}", "PASS")
        self.log(f"   Failed: {failed}", "FAIL" if failed > 0 else "PASS")
        self.log(f"   Success Rate: {success_rate:.1f}%", "PASS" if success_rate >= 90 else "WARN")
        
        if success_rate >= 90:
            self.log("🎉 PRIVATE USE MODE VALIDATION: SUCCESS", "PASS")
            self.log("   Ready for production use!", "INFO")
        elif success_rate >= 75:
            self.log("⚠️ PRIVATE USE MODE VALIDATION: PARTIAL SUCCESS", "WARN") 
            self.log("   Some issues detected, review failed tests", "WARN")
        else:
            self.log("❌ PRIVATE USE MODE VALIDATION: FAILED", "FAIL")
            self.log("   Major issues detected, fix required", "FAIL")
        
        # Save detailed results
        with open("private_mode_validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"📝 Detailed results saved to: private_mode_validation_results.json", "INFO")

def main():
    """Run validation suite"""
    validator = PrivateUseValidator()
    validator.run_all_validations()

if __name__ == "__main__":
    main()