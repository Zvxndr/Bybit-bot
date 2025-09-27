#!/usr/bin/env python3
"""
Essential Pre-Deployment Audit
==============================

Focuses on critical components needed for deployment.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

def audit_essential_components():
    """Audit essential components for deployment"""
    print("🔍 ESSENTIAL PRE-DEPLOYMENT AUDIT")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    audit_results = {
        'critical_issues': [],
        'warnings': [],
        'passed_checks': 0,
        'total_checks': 0
    }
    
    def check(name, condition, critical_error=None, warning=None):
        audit_results['total_checks'] += 1
        if condition:
            print(f"✅ {name}")
            audit_results['passed_checks'] += 1
            return True
        else:
            if critical_error:
                print(f"❌ {name} - {critical_error}")
                audit_results['critical_issues'].append(f"{name}: {critical_error}")
            elif warning:
                print(f"⚠️ {name} - {warning}")
                audit_results['warnings'].append(f"{name}: {warning}")
            return False
    
    # 1. Environment Setup
    print("🔧 ENVIRONMENT SETUP")
    print("-" * 30)
    
    # Check .env file
    env_exists = Path('.env').exists()
    check("Environment file (.env)", env_exists, 
          "Missing .env file - copy from .env.example and configure")
    
    if env_exists:
        load_dotenv()
        
        # Check critical environment variables
        critical_vars = {
            'SECRET_KEY': 'Application secret key',
            'JWT_SECRET': 'JWT signing key', 
            'ENCRYPTION_KEY': 'Data encryption key',
            'MFA_ENCRYPTION_KEY': 'MFA secret encryption'
        }
        
        for var, desc in critical_vars.items():
            value = os.getenv(var)
            check(f"{desc} ({var})", bool(value), 
                  f"Missing {var} in .env file")
    
    # 2. Security Components
    print("\n🔒 SECURITY COMPONENTS")
    print("-" * 30)
    
    # Check MFA system
    try:
        from security.mfa_manager import MFAManager, MFASession
        from cryptography.fernet import Fernet
        
        # Test MFA manager
        key = Fernet.generate_key().decode()
        mfa = MFAManager(key)
        session = MFASession()
        
        check("MFA Manager", True)
        check("MFA Session Manager", True)
        
    except ImportError as e:
        check("MFA System", False, f"Import failed: {str(e)}")
    except Exception as e:
        check("MFA System", False, f"Runtime error: {str(e)}")
    
    # Check encryption
    try:
        from security.encryption_manager import EncryptionManager
        enc = EncryptionManager()
        test_data = "test"
        encrypted = enc.encrypt_data(test_data)
        decrypted = enc.decrypt_data(encrypted)
        check("Encryption Manager", decrypted == test_data,
              "Encryption/decryption test failed")
    except ImportError:
        check("Encryption Manager", False, "Import failed")
    except Exception as e:
        check("Encryption Manager", False, f"Error: {str(e)}")
    
    # Check security middleware
    try:
        from security.security_middleware import SecurityLevel, RateLimitRule
        level = SecurityLevel.ADMIN
        rule = RateLimitRule(100, 3600)
        check("Security Middleware", True)
    except ImportError:
        check("Security Middleware", False, "Import failed")
    except Exception as e:
        check("Security Middleware", False, f"Error: {str(e)}")
    
    # 3. Infrastructure Components
    print("\n☁️ INFRASTRUCTURE")
    print("-" * 30)
    
    # Check DigitalOcean manager
    try:
        from infrastructure.digitalocean_manager import DigitalOceanManager
        check("DigitalOcean Manager", True)
    except ImportError:
        check("DigitalOcean Manager", False, None, "Import failed")
    
    # Check SendGrid manager
    try:
        from notifications.sendgrid_manager import SendGridManager
        check("SendGrid Manager", True)
    except ImportError:
        check("SendGrid Manager", False, None, "Import failed")
    
    # Check notification scheduler
    try:
        from notifications.notification_scheduler import NotificationScheduler
        check("Notification Scheduler", True)
    except ImportError:
        check("Notification Scheduler", False, None, "Import failed")
    
    # 4. Essential Dependencies
    print("\n📦 CRITICAL DEPENDENCIES")
    print("-" * 30)
    
    essential_deps = {
        'pyotp': 'Multi-factor authentication',
        'qrcode': 'QR code generation',
        'cryptography': 'Encryption',
        'redis': 'Session storage'
    }
    
    for dep, desc in essential_deps.items():
        try:
            __import__(dep)
            check(f"{desc} ({dep})", True)
        except ImportError:
            check(f"{desc} ({dep})", False, f"Missing dependency: {dep}")
    
    # 5. Configuration Files
    print("\n📄 CONFIGURATION")
    print("-" * 30)
    
    config_files = {
        'requirements.txt': 'Python dependencies',
        'Dockerfile': 'Container configuration',
        'docker-compose.yml': 'Multi-container setup',
        '.gitignore': 'Git ignore rules'
    }
    
    for file, desc in config_files.items():
        exists = Path(file).exists()
        if file == '.gitignore':
            if exists:
                with open(file, 'r') as f:
                    content = f.read()
                    has_env = '.env' in content
                    check(f"{desc} (includes .env)", has_env,
                          ".env not in .gitignore - security risk!")
            else:
                check(f"{desc}", False, "Missing .gitignore file")
        else:
            check(f"{desc} ({file})", exists, None, f"Missing {file}")
    
    # 6. Documentation
    print("\n📚 DOCUMENTATION")
    print("-" * 30)
    
    doc_files = {
        'README.md': 'Main documentation',
        'docs/deployment/DEPLOYMENT_GUIDE.md': 'Deployment guide',
        'docs/deployment/QUICK_START.md': 'Quick start guide',
        'utils/quick_setup.py': 'Setup automation'
    }
    
    for file, desc in doc_files.items():
        exists = Path(file).exists()
        check(f"{desc}", exists, None, f"Missing {file}")
    
    # 7. Sensitive File Security
    print("\n🔐 SECURITY CHECK")
    print("-" * 30)
    
    sensitive_files = ['.env', '.secrets_key']
    gitignore_path = Path('.gitignore')
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        for sensitive in sensitive_files:
            if Path(sensitive).exists():
                ignored = sensitive in gitignore_content
                check(f"{sensitive} properly ignored", ignored,
                      f"Security risk: {sensitive} exists but not in .gitignore")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 AUDIT SUMMARY")
    print("=" * 50)
    
    passed = audit_results['passed_checks']
    total = audit_results['total_checks']
    critical = len(audit_results['critical_issues'])
    warnings = len(audit_results['warnings'])
    
    print(f"Checks Passed: {passed}/{total}")
    print(f"Critical Issues: {critical}")
    print(f"Warnings: {warnings}")
    
    if critical > 0:
        print(f"\n🚨 CRITICAL ISSUES ({critical}):")
        for issue in audit_results['critical_issues']:
            print(f"  ❌ {issue}")
    
    if warnings > 0:
        print(f"\n⚠️ WARNINGS ({warnings}):")
        for warning in audit_results['warnings']:
            print(f"  ⚠️ {warning}")
    
    # Final verdict
    print("\n" + "=" * 50)
    if critical == 0:
        if warnings == 0:
            print("🎉 READY FOR DEPLOYMENT!")
            print("All essential components are working correctly.")
            status = "READY"
        else:
            print("⚠️ DEPLOYMENT POSSIBLE WITH CAUTION")
            print("Address warnings for optimal deployment.")
            status = "CAUTION"
    else:
        print("🚨 NOT READY FOR DEPLOYMENT")
        print("Fix critical issues before deploying.")
        status = "NOT_READY"
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"essential_audit_{timestamp}.json"
    
    import json
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'passed_checks': passed,
            'total_checks': total,
            'critical_issues': audit_results['critical_issues'],
            'warnings': audit_results['warnings']
        }, f, indent=2)
    
    print(f"\n📄 Audit report saved: {report_file}")
    
    return status

if __name__ == "__main__":
    status = audit_essential_components()
    
    # Exit codes for CI/CD
    exit_codes = {
        "READY": 0,
        "CAUTION": 1,
        "NOT_READY": 2
    }
    
    sys.exit(exit_codes.get(status, 2))