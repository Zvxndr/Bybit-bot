#!/usr/bin/env python3
"""
Pre-Deployment Comprehensive Audit
==================================

Performs a full system audit to verify all components are properly integrated
and working before production deployment.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import importlib.util

# Add src to path for imports
sys.path.append('src')

class PreDeploymentAuditor:
    """Comprehensive pre-deployment audit system"""
    
    def __init__(self):
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'components': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        self.repo_root = Path.cwd()
        
    def log_result(self, component: str, status: str, details: str, issues: List[str] = None):
        """Log audit result for a component"""
        self.audit_results['components'][component] = {
            'status': status,
            'details': details,
            'issues': issues or []
        }
        
        if status == 'CRITICAL':
            self.audit_results['critical_issues'].extend(issues or [])
        elif status == 'WARNING':
            self.audit_results['warnings'].extend(issues or [])
    
    def check_python_environment(self) -> bool:
        """Audit Python environment and dependencies"""
        print("🐍 AUDITING PYTHON ENVIRONMENT")
        print("-" * 40)
        
        issues = []
        
        try:
            # Check Python version
            python_version = sys.version_info
            print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            if python_version < (3, 8):
                issues.append("Python version too old (requires 3.8+)")
            
            # Check virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                print("✅ Virtual environment detected")
            else:
                issues.append("Not running in virtual environment")
            
            # Check critical dependencies
            critical_deps = [
                'pyotp', 'qrcode', 'cryptography', 'redis', 'fastapi', 
                'uvicorn', 'pandas', 'numpy', 'ccxt', 'sqlalchemy'
            ]
            
            missing_deps = []
            for dep in critical_deps:
                try:
                    __import__(dep)
                    print(f"✅ {dep} installed")
                except ImportError:
                    missing_deps.append(dep)
                    print(f"❌ {dep} missing")
            
            if missing_deps:
                issues.append(f"Missing dependencies: {', '.join(missing_deps)}")
            
            status = 'CRITICAL' if issues else 'PASS'
            self.log_result('python_environment', status, 
                          f"Python {python_version.major}.{python_version.minor}.{python_version.micro}, {len(critical_deps)-len(missing_deps)}/{len(critical_deps)} dependencies", 
                          issues)
            
            return len(issues) == 0
            
        except Exception as e:
            self.log_result('python_environment', 'CRITICAL', f"Environment check failed: {str(e)}", [str(e)])
            return False
    
    def check_security_components(self) -> bool:
        """Audit security implementations"""
        print("\n🔒 AUDITING SECURITY COMPONENTS")
        print("-" * 40)
        
        issues = []
        security_modules = [
            'security.mfa_manager',
            'security.encryption_manager', 
            'security.security_middleware'
        ]
        
        working_modules = 0
        
        for module in security_modules:
            try:
                # Try to import and basic instantiation test
                if module == 'security.mfa_manager':
                    from security.mfa_manager import MFAManager, MFASession
                    from cryptography.fernet import Fernet
                    key = Fernet.generate_key().decode()
                    mfa = MFAManager(key)
                    session = MFASession()
                    print(f"✅ {module} - Functional")
                    working_modules += 1
                    
                elif module == 'security.encryption_manager':
                    from security.encryption_manager import EncryptionManager
                    encryption = EncryptionManager()
                    test_data = "test_data"
                    encrypted = encryption.encrypt_data(test_data)
                    decrypted = encryption.decrypt_data(encrypted)
                    if decrypted == test_data:
                        print(f"✅ {module} - Functional")
                        working_modules += 1
                    else:
                        issues.append(f"{module}: Encryption/decryption failed")
                        
                elif module == 'security.security_middleware':
                    from security.security_middleware import SecurityLevel, RateLimitRule
                    # Test basic instantiation
                    level = SecurityLevel.ADMIN
                    rule = RateLimitRule(100, 3600)
                    print(f"✅ {module} - Functional")
                    working_modules += 1
                    
            except ImportError as e:
                issues.append(f"{module}: Import failed - {str(e)}")
                print(f"❌ {module} - Import failed")
            except Exception as e:
                issues.append(f"{module}: Runtime error - {str(e)}")
                print(f"⚠️ {module} - Runtime issue")
        
        # Check MFA configuration
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            mfa_key = os.getenv('MFA_ENCRYPTION_KEY')
            if mfa_key:
                print("✅ MFA encryption key configured")
            else:
                issues.append("MFA_ENCRYPTION_KEY not set in environment")
                
        except Exception as e:
            issues.append(f"Environment configuration check failed: {str(e)}")
        
        status = 'CRITICAL' if len(issues) > 0 else 'PASS'
        self.log_result('security_components', status, 
                      f"{working_modules}/{len(security_modules)} security modules functional", 
                      issues)
        
        return len(issues) == 0
    
    def check_infrastructure_components(self) -> bool:
        """Audit infrastructure implementations"""
        print("\n☁️ AUDITING INFRASTRUCTURE COMPONENTS")
        print("-" * 40)
        
        issues = []
        
        # Check DigitalOcean integration
        try:
            from infrastructure.digitalocean_manager import DigitalOceanManager
            # Test basic instantiation (won't connect without API key)
            print("✅ DigitalOcean manager - Import successful")
        except ImportError as e:
            issues.append(f"DigitalOcean manager import failed: {str(e)}")
            print("❌ DigitalOcean manager - Import failed")
        
        # Check SendGrid integration
        try:
            from notifications.sendgrid_manager import SendGridManager
            print("✅ SendGrid manager - Import successful")
        except ImportError as e:
            issues.append(f"SendGrid manager import failed: {str(e)}")
            print("❌ SendGrid manager - Import failed")
        
        # Check notification scheduler
        try:
            from notifications.notification_scheduler import NotificationScheduler
            print("✅ Notification scheduler - Import successful")
        except ImportError as e:
            issues.append(f"Notification scheduler import failed: {str(e)}")
            print("❌ Notification scheduler - Import failed")
        
        status = 'WARNING' if len(issues) > 0 else 'PASS'
        self.log_result('infrastructure_components', status, 
                      f"Infrastructure modules checked", issues)
        
        return len(issues) == 0
    
    def check_configuration_files(self) -> bool:
        """Audit configuration files"""
        print("\n📄 AUDITING CONFIGURATION FILES")
        print("-" * 40)
        
        issues = []
        
        # Check .env file
        env_file = self.repo_root / '.env'
        if env_file.exists():
            print("✅ .env file exists")
            
            # Check for critical environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            critical_vars = [
                'SECRET_KEY', 'JWT_SECRET', 'ENCRYPTION_KEY',
                'MFA_ENCRYPTION_KEY', 'DATABASE_URL'
            ]
            
            missing_vars = []
            for var in critical_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
                else:
                    print(f"✅ {var} configured")
            
            if missing_vars:
                issues.append(f"Missing environment variables: {', '.join(missing_vars)}")
                
        else:
            issues.append(".env file not found")
        
        # Check requirements.txt
        req_file = self.repo_root / 'requirements.txt'
        if req_file.exists():
            print("✅ requirements.txt exists")
            with open(req_file, 'r') as f:
                content = f.read()
                if 'pyotp' in content and 'cryptography' in content:
                    print("✅ Security dependencies in requirements.txt")
                else:
                    issues.append("Missing security dependencies in requirements.txt")
        else:
            issues.append("requirements.txt not found")
        
        # Check Docker files
        dockerfile = self.repo_root / 'Dockerfile'
        docker_compose = self.repo_root / 'docker-compose.yml'
        
        if dockerfile.exists():
            print("✅ Dockerfile exists")
        else:
            issues.append("Dockerfile not found")
            
        if docker_compose.exists():
            print("✅ docker-compose.yml exists")
        else:
            issues.append("docker-compose.yml not found")
        
        status = 'CRITICAL' if any('not found' in issue for issue in issues) else 'WARNING' if issues else 'PASS'
        self.log_result('configuration_files', status, "Configuration files checked", issues)
        
        return len(issues) == 0
    
    def check_documentation(self) -> bool:
        """Audit documentation completeness"""
        print("\n📚 AUDITING DOCUMENTATION")
        print("-" * 40)
        
        issues = []
        
        # Check main documentation files
        doc_files = [
            'README.md',
            'docs/deployment/DEPLOYMENT_GUIDE.md',
            'docs/deployment/QUICK_START.md'
        ]
        
        for doc_file in doc_files:
            file_path = self.repo_root / doc_file
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"✅ {doc_file} exists ({size} bytes)")
                if size < 1000:  # Less than 1KB might be incomplete
                    issues.append(f"{doc_file} seems incomplete (very small)")
            else:
                issues.append(f"{doc_file} not found")
        
        # Check utility scripts
        utils_dir = self.repo_root / 'utils'
        if utils_dir.exists():
            print("✅ utils/ directory exists")
            quick_setup = utils_dir / 'quick_setup.py'
            if quick_setup.exists():
                print("✅ quick_setup.py exists")
            else:
                issues.append("quick_setup.py not found in utils/")
        else:
            issues.append("utils/ directory not found")
        
        status = 'WARNING' if issues else 'PASS'
        self.log_result('documentation', status, "Documentation files checked", issues)
        
        return len(issues) == 0
    
    def check_testing_framework(self) -> bool:
        """Audit testing setup"""
        print("\n🧪 AUDITING TESTING FRAMEWORK")
        print("-" * 40)
        
        issues = []
        
        # Check tests directory
        tests_dir = self.repo_root / 'tests'
        if tests_dir.exists():
            print("✅ tests/ directory exists")
            
            # Count test files
            test_files = list(tests_dir.rglob('test_*.py'))
            print(f"✅ Found {len(test_files)} test files")
            
            if len(test_files) == 0:
                issues.append("No test files found in tests/ directory")
                
        else:
            issues.append("tests/ directory not found")
        
        # Check if pytest is available
        try:
            import pytest
            print("✅ pytest available")
        except ImportError:
            issues.append("pytest not installed")
        
        # Check test configuration
        pytest_ini = self.repo_root / 'pytest.ini'
        if pytest_ini.exists():
            print("✅ pytest.ini configuration exists")
        
        status = 'WARNING' if issues else 'PASS'
        self.log_result('testing_framework', status, f"Testing setup checked", issues)
        
        return len(issues) == 0
    
    def check_git_repository(self) -> bool:
        """Audit Git repository status"""
        print("\n📦 AUDITING GIT REPOSITORY")
        print("-" * 40)
        
        issues = []
        
        # Check if .git directory exists
        git_dir = self.repo_root / '.git'
        if git_dir.exists():
            print("✅ Git repository initialized")
            
            # Check .gitignore
            gitignore = self.repo_root / '.gitignore'
            if gitignore.exists():
                with open(gitignore, 'r') as f:
                    content = f.read()
                    critical_ignores = ['.env', '__pycache__', '*.pyc', '.venv']
                    missing_ignores = [ignore for ignore in critical_ignores if ignore not in content]
                    
                    if missing_ignores:
                        issues.append(f"Missing .gitignore entries: {', '.join(missing_ignores)}")
                    else:
                        print("✅ .gitignore properly configured")
            else:
                issues.append(".gitignore file not found")
            
            # Check for uncommitted changes
            try:
                result = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True, cwd=self.repo_root)
                if result.returncode == 0:
                    if result.stdout.strip():
                        issues.append("Uncommitted changes detected")
                        print("⚠️ Uncommitted changes found")
                    else:
                        print("✅ Working directory clean")
                else:
                    issues.append("Could not check git status")
            except Exception as e:
                issues.append(f"Git status check failed: {str(e)}")
                
        else:
            issues.append("Not a Git repository")
        
        status = 'WARNING' if issues else 'PASS'
        self.log_result('git_repository', status, "Git repository checked", issues)
        
        return len(issues) == 0
    
    def check_deployment_readiness(self) -> bool:
        """Check deployment readiness"""
        print("\n🚀 AUDITING DEPLOYMENT READINESS")
        print("-" * 40)
        
        issues = []
        recommendations = []
        
        # Check if deployment guides exist and are accessible
        deployment_files = [
            'docs/deployment/DEPLOYMENT_GUIDE.md',
            'docs/deployment/QUICK_START.md',
            'utils/quick_setup.py'
        ]
        
        for file_path in deployment_files:
            full_path = self.repo_root / file_path
            if full_path.exists():
                print(f"✅ {file_path} ready")
            else:
                issues.append(f"Deployment file missing: {file_path}")
        
        # Check environment template
        env_example = self.repo_root / '.env.example'
        if env_example.exists():
            print("✅ .env.example exists")
        else:
            recommendations.append("Consider adding .env.example template")
        
        # Check if sensitive files are properly ignored
        sensitive_files = ['.env', '.secrets_key']
        for sensitive in sensitive_files:
            file_path = self.repo_root / sensitive
            if file_path.exists():
                # Check if it's in .gitignore
                gitignore = self.repo_root / '.gitignore'
                if gitignore.exists():
                    with open(gitignore, 'r') as f:
                        if sensitive not in f.read():
                            issues.append(f"Sensitive file {sensitive} not in .gitignore")
                        else:
                            print(f"✅ {sensitive} properly ignored")
        
        # Check for production configurations
        production_indicators = [
            'MFA_ENCRYPTION_KEY',
            'SECURITY_REQUIRE_MFA_FOR_ADMIN',
            'TRADING_ENVIRONMENT'
        ]
        
        from dotenv import load_dotenv
        load_dotenv()
        
        for indicator in production_indicators:
            if os.getenv(indicator):
                print(f"✅ {indicator} configured")
            else:
                recommendations.append(f"Configure {indicator} for production")
        
        if recommendations:
            self.audit_results['recommendations'].extend(recommendations)
        
        status = 'CRITICAL' if issues else 'PASS'
        self.log_result('deployment_readiness', status, "Deployment readiness checked", issues)
        
        return len(issues) == 0
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run complete audit"""
        print("🔍 COMPREHENSIVE PRE-DEPLOYMENT AUDIT")
        print("=" * 50)
        print(f"Repository: {self.repo_root}")
        print(f"Timestamp: {self.audit_results['timestamp']}")
        print()
        
        # Run all audit checks
        checks = [
            ('Python Environment', self.check_python_environment),
            ('Security Components', self.check_security_components),
            ('Infrastructure Components', self.check_infrastructure_components),
            ('Configuration Files', self.check_configuration_files),
            ('Documentation', self.check_documentation),
            ('Testing Framework', self.check_testing_framework),
            ('Git Repository', self.check_git_repository),
            ('Deployment Readiness', self.check_deployment_readiness)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            try:
                if check_func():
                    passed_checks += 1
            except Exception as e:
                self.log_result(check_name.lower().replace(' ', '_'), 'CRITICAL', 
                              f"Check failed with exception: {str(e)}", [str(e)])
        
        # Determine overall status
        if self.audit_results['critical_issues']:
            self.audit_results['overall_status'] = 'CRITICAL'
        elif self.audit_results['warnings']:
            self.audit_results['overall_status'] = 'WARNING'
        else:
            self.audit_results['overall_status'] = 'PASS'
        
        # Generate summary
        print("\n" + "=" * 50)
        print("📊 AUDIT SUMMARY")
        print("=" * 50)
        
        print(f"Overall Status: {self.audit_results['overall_status']}")
        print(f"Checks Passed: {passed_checks}/{total_checks}")
        print(f"Critical Issues: {len(self.audit_results['critical_issues'])}")
        print(f"Warnings: {len(self.audit_results['warnings'])}")
        print(f"Recommendations: {len(self.audit_results['recommendations'])}")
        
        if self.audit_results['critical_issues']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in self.audit_results['critical_issues']:
                print(f"  ❌ {issue}")
        
        if self.audit_results['warnings']:
            print("\n⚠️ WARNINGS:")
            for warning in self.audit_results['warnings']:
                print(f"  ⚠️ {warning}")
        
        if self.audit_results['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in self.audit_results['recommendations']:
                print(f"  💡 {rec}")
        
        # Final verdict
        print("\n" + "=" * 50)
        if self.audit_results['overall_status'] == 'PASS':
            print("🎉 DEPLOYMENT READY!")
            print("All critical components are functional and properly configured.")
        elif self.audit_results['overall_status'] == 'WARNING':
            print("⚠️ DEPLOYMENT POSSIBLE WITH CAUTION")
            print("Address warnings before production deployment.")
        else:
            print("🚨 NOT READY FOR DEPLOYMENT")
            print("Critical issues must be resolved before deployment.")
        
        return self.audit_results

def save_audit_report(audit_results: Dict[str, Any]):
    """Save audit results to file"""
    report_file = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(audit_results, f, indent=2)
    
    print(f"\n📄 Audit report saved to: {report_file}")

if __name__ == "__main__":
    auditor = PreDeploymentAuditor()
    results = auditor.run_comprehensive_audit()
    save_audit_report(results)
    
    # Exit with appropriate code
    if results['overall_status'] == 'PASS':
        sys.exit(0)
    elif results['overall_status'] == 'WARNING':
        sys.exit(1)
    else:
        sys.exit(2)