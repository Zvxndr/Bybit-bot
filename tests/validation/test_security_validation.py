"""
Bybit API Security Validation Suite

This module validates security practices for API key management,
authentication mechanisms, and secure communication with Bybit API.

Security Areas Covered:
- API key storage and handling
- Request signing and authentication
- Environment variable management
- Credential rotation practices
- Network security
"""

import os
import re
import hmac
import hashlib
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import inspect
import ast

from src.bot.exchange.bybit_client import BybitClient
from src.bot.config_manager import ConfigurationManager


@dataclass
class SecurityIssue:
    """Represents a security issue found during validation."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    recommendation: str
    file_location: Optional[str] = None
    line_number: Optional[int] = None


@dataclass 
class SecurityValidationResult:
    """Result of security validation."""
    test_name: str
    passed: bool
    issues: List[SecurityIssue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class BybitSecurityValidator:
    """Comprehensive security validator for Bybit API integration."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
    async def run_complete_security_audit(self) -> Dict[str, Any]:
        """Run complete security audit."""
        self.logger.info("Starting comprehensive security audit")
        
        security_tests = [
            ("api_key_management", self._validate_api_key_management),
            ("credential_storage", self._validate_credential_storage),
            ("request_authentication", self._validate_request_authentication),
            ("environment_security", self._validate_environment_security),
            ("code_security", self._validate_code_security),
            ("network_security", self._validate_network_security),
            ("logging_security", self._validate_logging_security)
        ]
        
        results = {}
        all_issues = []
        
        for test_name, test_func in security_tests:
            try:
                self.logger.info(f"Running security test: {test_name}")
                result = await test_func()
                results[test_name] = result
                all_issues.extend(result.issues)
                
            except Exception as e:
                self.logger.error(f"Security test {test_name} failed: {str(e)}")
                results[test_name] = SecurityValidationResult(
                    test_name=test_name,
                    passed=False,
                    issues=[SecurityIssue(
                        severity="HIGH",
                        category="Test Execution",
                        description=f"Security test failed: {str(e)}",
                        recommendation="Fix test execution issues"
                    )]
                )
        
        return self._generate_security_report(results, all_issues)
    
    async def _validate_api_key_management(self) -> SecurityValidationResult:
        """Validate API key management practices."""
        result = SecurityValidationResult(
            test_name="api_key_management",
            passed=True
        )
        
        try:
            credentials = self.config_manager.get_current_credentials()
            
            # Check API key format and length
            if not credentials.api_key or len(credentials.api_key) < 10:
                result.issues.append(SecurityIssue(
                    severity="HIGH",
                    category="API Key",
                    description="API key is missing or too short",
                    recommendation="Ensure valid API key is configured"
                ))
            
            # Check for hardcoded keys (basic patterns)
            hardcoded_patterns = [
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'API_KEY\s*=\s*["\'][^"\']+["\']',
                r'apikey.*["\'][^"\']+["\']'
            ]
            
            # Check if API key looks hardcoded
            for pattern in hardcoded_patterns:
                if re.search(pattern, credentials.api_key, re.IGNORECASE):
                    result.issues.append(SecurityIssue(
                        severity="CRITICAL",
                        category="API Key",
                        description="API key appears to be hardcoded",
                        recommendation="Use environment variables for API keys"
                    ))
            
            # Check API secret
            if not credentials.api_secret or len(credentials.api_secret) < 20:
                result.issues.append(SecurityIssue(
                    severity="HIGH",
                    category="API Secret",
                    description="API secret is missing or too short",
                    recommendation="Ensure valid API secret is configured"
                ))
            
            # Validate key permissions (if accessible)
            result.details['api_key_length'] = len(credentials.api_key)
            result.details['api_secret_length'] = len(credentials.api_secret)
            result.details['is_testnet'] = credentials.is_testnet
            
        except Exception as e:
            result.issues.append(SecurityIssue(
                severity="HIGH",
                category="API Key Management",
                description=f"Failed to validate API key management: {str(e)}",
                recommendation="Check configuration manager and credential handling"
            ))
        
        result.passed = len(result.issues) == 0
        return result
    
    async def _validate_credential_storage(self) -> SecurityValidationResult:
        """Validate secure credential storage practices."""
        result = SecurityValidationResult(
            test_name="credential_storage",
            passed=True
        )
        
        # Check environment variables
        env_keys = [
            'BYBIT_API_KEY', 'BYBIT_API_SECRET',
            'BYBIT_TESTNET_API_KEY', 'BYBIT_TESTNET_API_SECRET'
        ]
        
        env_vars_found = {}
        for key in env_keys:
            value = os.getenv(key)
            env_vars_found[key] = {
                'present': value is not None,
                'length': len(value) if value else 0
            }
            
            if value and len(value) < 10:
                result.issues.append(SecurityIssue(
                    severity="MEDIUM",
                    category="Environment Variables",
                    description=f"Environment variable {key} is too short",
                    recommendation="Ensure environment variables contain valid credentials"
                ))
        
        result.details['environment_variables'] = env_vars_found
        
        # Check for credentials in config files
        config_files = [
            'config/config.yaml',
            'config/config.yml', 
            '.env',
            'config.json'
        ]
        
        credential_patterns = [
            r'api_key\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']',
            r'api_secret\s*[:=]\s*["\'][A-Za-z0-9]{30,}["\']',
            r'password\s*[:=]\s*["\'][^"\']+["\']'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        
                    for pattern in credential_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            result.issues.append(SecurityIssue(
                                severity="CRITICAL",
                                category="Credential Storage",
                                description=f"Potential credentials found in {config_file}",
                                recommendation="Remove credentials from config files, use environment variables",
                                file_location=config_file
                            ))
                            
                except Exception as e:
                    self.logger.warning(f"Could not check {config_file}: {str(e)}")
        
        result.passed = len(result.issues) == 0
        return result
    
    async def _validate_request_authentication(self) -> SecurityValidationResult:
        """Validate request signing and authentication."""
        result = SecurityValidationResult(
            test_name="request_authentication",
            passed=True
        )
        
        try:
            client = BybitClient(self.config_manager)
            
            # Check if signature generation is implemented
            if not hasattr(client, '_generate_signature'):
                result.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    category="Authentication",
                    description="Request signature generation not implemented",
                    recommendation="Implement proper HMAC signature generation"
                ))
            else:
                # Test signature generation
                test_signature = await self._test_signature_generation(client)
                result.details['signature_test'] = test_signature
                
                if not test_signature['valid']:
                    result.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Authentication",
                        description="Signature generation appears invalid",
                        recommendation="Review and fix signature generation algorithm"
                    ))
            
            # Check required headers
            required_headers = ['X-BAPI-API-KEY', 'X-BAPI-TIMESTAMP', 'X-BAPI-SIGN']
            
            # Test with actual request
            try:
                response = await client.get_server_time()
                result.details['authentication_test'] = {
                    'success': True,
                    'response_code': response.get('retCode', 'unknown')
                }
                
            except Exception as e:
                if "authentication" in str(e).lower() or "signature" in str(e).lower():
                    result.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Authentication",
                        description=f"Authentication failed: {str(e)}",
                        recommendation="Check API key, secret, and signature generation"
                    ))
        
        except Exception as e:
            result.issues.append(SecurityIssue(
                severity="HIGH",
                category="Request Authentication",
                description=f"Failed to validate authentication: {str(e)}",
                recommendation="Check client implementation and authentication logic"
            ))
        
        result.passed = len(result.issues) == 0
        return result
    
    async def _test_signature_generation(self, client: BybitClient) -> Dict[str, Any]:
        """Test HMAC signature generation."""
        try:
            # Test with known values
            test_params = {
                'symbol': 'BTCUSDT',
                'side': 'Buy',
                'orderType': 'Limit'
            }
            
            timestamp = str(int(datetime.now().timestamp() * 1000))
            
            # Generate signature using client method
            if hasattr(client, '_generate_signature'):
                signature = client._generate_signature(test_params, timestamp)
                
                # Basic validation - signature should be base64 encoded
                try:
                    decoded = base64.b64decode(signature)
                    return {
                        'valid': len(decoded) == 32,  # HMAC-SHA256 produces 32 bytes
                        'signature_length': len(signature),
                        'decoded_length': len(decoded)
                    }
                except Exception:
                    return {
                        'valid': False,
                        'error': 'Signature not valid base64'
                    }
            
            return {'valid': False, 'error': 'Signature method not found'}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_environment_security(self) -> SecurityValidationResult:
        """Validate environment and deployment security."""
        result = SecurityValidationResult(
            test_name="environment_security",
            passed=True
        )
        
        # Check for .env file security
        if os.path.exists('.env'):
            try:
                # Check .env file permissions (Unix-like systems)
                import stat
                file_stat = os.stat('.env')
                permissions = oct(file_stat.st_mode)[-3:]
                
                if permissions != '600':  # Should be readable only by owner
                    result.issues.append(SecurityIssue(
                        severity="MEDIUM",
                        category="File Permissions",
                        description=f".env file has permissive permissions: {permissions}",
                        recommendation="Set .env file permissions to 600 (owner read/write only)"
                    ))
                    
            except Exception:
                pass  # Skip permission check on Windows
        
        # Check for debug mode
        debug_indicators = [
            os.getenv('DEBUG'),
            os.getenv('FLASK_DEBUG'),
            os.getenv('DJANGO_DEBUG')
        ]
        
        if any(debug_indicators):
            result.issues.append(SecurityIssue(
                severity="MEDIUM",
                category="Environment",
                description="Debug mode enabled in environment",
                recommendation="Disable debug mode in production"
            ))
        
        # Check SSL/TLS configuration
        credentials = self.config_manager.get_current_credentials()
        if not credentials.base_url.startswith('https://'):
            result.issues.append(SecurityIssue(
                severity="HIGH",
                category="Network Security",
                description="API endpoint not using HTTPS",
                recommendation="Ensure all API communications use HTTPS"
            ))
        
        result.passed = len(result.issues) == 0
        return result
    
    async def _validate_code_security(self) -> SecurityValidationResult:
        """Validate code security practices."""
        result = SecurityValidationResult(
            test_name="code_security", 
            passed=True
        )
        
        # Check for sensitive data in code
        sensitive_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Password in code"),
            (r'secret\s*=\s*["\'][A-Za-z0-9]{20,}["\']', "Secret in code"),
            (r'api_key\s*=\s*["\'][A-Za-z0-9]{10,}["\']', "API key in code"),
            (r'private.*key\s*=\s*["\'][^"\']+["\']', "Private key in code")
        ]
        
        # Scan Python files
        python_files = self._find_python_files('src/')
        
        for file_path in python_files[:10]:  # Limit to avoid too many files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in sensitive_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        result.issues.append(SecurityIssue(
                            severity="HIGH",
                            category="Code Security", 
                            description=description,
                            recommendation="Remove sensitive data from code, use environment variables",
                            file_location=file_path,
                            line_number=line_num
                        ))
                        
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        result.details['files_scanned'] = len(python_files)
        result.passed = len(result.issues) == 0
        return result
    
    def _find_python_files(self, directory: str) -> List[str]:
        """Find Python files for security scanning."""
        python_files = []
        
        try:
            for root, dirs, files in os.walk(directory):
                # Skip common non-source directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules']]
                
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
                        
        except Exception as e:
            self.logger.warning(f"Error finding Python files: {str(e)}")
        
        return python_files
    
    async def _validate_network_security(self) -> SecurityValidationResult:
        """Validate network security practices."""
        result = SecurityValidationResult(
            test_name="network_security",
            passed=True
        )
        
        try:
            client = BybitClient(self.config_manager)
            
            # Check SSL certificate verification
            import ssl
            import aiohttp
            
            # Test SSL connection
            credentials = self.config_manager.get_current_credentials()
            base_url = credentials.base_url
            
            try:
                # Create SSL context
                ssl_context = ssl.create_default_context()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{base_url}/v5/market/time", ssl=ssl_context) as response:
                        result.details['ssl_verification'] = {
                            'success': True,
                            'status_code': response.status
                        }
                        
            except ssl.SSLError as e:
                result.issues.append(SecurityIssue(
                    severity="HIGH",
                    category="SSL/TLS",
                    description=f"SSL verification failed: {str(e)}",
                    recommendation="Check SSL certificate configuration"
                ))
            
            # Check for secure communication
            if not base_url.startswith('https://'):
                result.issues.append(SecurityIssue(
                    severity="CRITICAL", 
                    category="Network Security",
                    description="Using insecure HTTP connection",
                    recommendation="Use HTTPS for all API communications"
                ))
        
        except Exception as e:
            result.issues.append(SecurityIssue(
                severity="MEDIUM",
                category="Network Security",
                description=f"Network security validation failed: {str(e)}",
                recommendation="Check network configuration and SSL setup"
            ))
        
        result.passed = len(result.issues) == 0
        return result
    
    async def _validate_logging_security(self) -> SecurityValidationResult:
        """Validate logging security practices."""
        result = SecurityValidationResult(
            test_name="logging_security",
            passed=True
        )
        
        # Check for sensitive data in logs
        log_patterns = [
            (r'api_key.*[A-Za-z0-9]{10,}', "API key in logs"),
            (r'secret.*[A-Za-z0-9]{20,}', "Secret in logs"),
            (r'password.*[A-Za-z0-9]{8,}', "Password in logs")
        ]
        
        # Check recent log files
        log_files = [
            'trading_bot.log',
            'integrated_trading_bot.log',
            'app.log',
            'debug.log'
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    # Only check last 1000 lines to avoid memory issues
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[-1000:]
                        content = ''.join(lines)
                    
                    for pattern, description in log_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            result.issues.append(SecurityIssue(
                                severity="HIGH",
                                category="Logging Security",
                                description=f"{description} found in {log_file}",
                                recommendation="Remove sensitive data from log outputs",
                                file_location=log_file
                            ))
                            
                except Exception as e:
                    self.logger.warning(f"Could not check log file {log_file}: {str(e)}")
        
        result.passed = len(result.issues) == 0
        return result
    
    def _generate_security_report(self, results: Dict[str, SecurityValidationResult], 
                                 all_issues: List[SecurityIssue]) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        # Categorize issues by severity
        severity_counts = {
            'CRITICAL': sum(1 for issue in all_issues if issue.severity == 'CRITICAL'),
            'HIGH': sum(1 for issue in all_issues if issue.severity == 'HIGH'),
            'MEDIUM': sum(1 for issue in all_issues if issue.severity == 'MEDIUM'),
            'LOW': sum(1 for issue in all_issues if issue.severity == 'LOW')
        }
        
        passed_tests = sum(1 for result in results.values() if result.passed)
        total_tests = len(results)
        
        # Determine overall security status
        overall_secure = (
            severity_counts['CRITICAL'] == 0 and
            severity_counts['HIGH'] == 0 and
            passed_tests == total_tests
        )
        
        return {
            'overall_security_status': 'SECURE' if overall_secure else 'VULNERABLE',
            'summary': {
                'total_tests': total_tests,
                'tests_passed': passed_tests,
                'total_issues': len(all_issues),
                'critical_issues': severity_counts['CRITICAL'],
                'high_issues': severity_counts['HIGH'],
                'medium_issues': severity_counts['MEDIUM'],
                'low_issues': severity_counts['LOW']
            },
            'test_results': {
                name: {
                    'passed': result.passed,
                    'issues_count': len(result.issues),
                    'details': result.details
                }
                for name, result in results.items()
            },
            'security_issues': [
                {
                    'severity': issue.severity,
                    'category': issue.category,
                    'description': issue.description,
                    'recommendation': issue.recommendation,
                    'location': issue.file_location,
                    'line': issue.line_number
                }
                for issue in all_issues
            ],
            'recommendations': self._generate_security_recommendations(severity_counts, all_issues)
        }
    
    def _generate_security_recommendations(self, severity_counts: Dict[str, int], 
                                         all_issues: List[SecurityIssue]) -> List[Dict[str, str]]:
        """Generate security recommendations."""
        recommendations = []
        
        if severity_counts['CRITICAL'] > 0:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'category': 'Critical Security',
                'action': 'Address all critical security issues before deployment',
                'description': f"{severity_counts['CRITICAL']} critical security issues found"
            })
        
        if severity_counts['HIGH'] > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'High Risk Security',
                'action': 'Fix high-risk security issues within 24 hours',
                'description': f"{severity_counts['HIGH']} high-risk security issues found"
            })
        
        # Category-specific recommendations
        categories = {}
        for issue in all_issues:
            if issue.category not in categories:
                categories[issue.category] = []
            categories[issue.category].append(issue)
        
        for category, issues in categories.items():
            if len(issues) > 1:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': category,
                    'action': f'Review and fix {len(issues)} issues in {category}',
                    'description': f'Multiple security issues found in {category}'
                })
        
        return recommendations


# Main execution function
async def run_security_validation_suite():
    """Run complete security validation suite."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        config_manager.load_config()
        
        validator = BybitSecurityValidator(config_manager)
        
        logger.info("Starting comprehensive security validation suite...")
        
        # Run complete security audit
        report = await validator.run_complete_security_audit()
        
        # Display results
        print("\n" + "="*60)
        print("BYBIT API SECURITY VALIDATION RESULTS")
        print("="*60)
        
        status_emoji = "🔒" if report['overall_security_status'] == 'SECURE' else "⚠️"
        print(f"Overall Security Status: {status_emoji} {report['overall_security_status']}")
        
        print(f"\nSummary:")
        print(f"  Tests Passed: {report['summary']['tests_passed']}/{report['summary']['total_tests']}")
        print(f"  Total Issues: {report['summary']['total_issues']}")
        print(f"  🔴 Critical: {report['summary']['critical_issues']}")
        print(f"  🟡 High: {report['summary']['high_issues']}")
        print(f"  🟠 Medium: {report['summary']['medium_issues']}")
        print(f"  🟢 Low: {report['summary']['low_issues']}")
        
        if report['security_issues']:
            print(f"\nSecurity Issues Found:")
            for issue in report['security_issues']:
                severity_emoji = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟡', 
                    'MEDIUM': '🟠',
                    'LOW': '🟢'
                }.get(issue['severity'], '🔵')
                
                print(f"  {severity_emoji} {issue['category']}: {issue['description']}")
                if issue['location']:
                    location = f"{issue['location']}"
                    if issue['line']:
                        location += f":{issue['line']}"
                    print(f"    📍 {location}")
                print(f"    💡 {issue['recommendation']}")
        
        if report['recommendations']:
            print(f"\nSecurity Recommendations:")
            for rec in report['recommendations']:
                priority_emoji = {
                    'IMMEDIATE': '🚨',
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(rec['priority'], '🔵')
                
                print(f"  {priority_emoji} {rec['priority']}: {rec['action']}")
                print(f"    {rec['description']}")
        
        return report
        
    except Exception as e:
        logger.error(f"Security validation suite failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_security_validation_suite())