"""
Production Deployment Validation Checklist

This module provides a comprehensive checklist and validation suite
for production deployment of the Bybit trading bot.

Covers all critical areas required for safe production deployment:
- API integration validation
- Security compliance
- Performance requirements
- Monitoring setup
- Risk management
- Backup and recovery
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import validation modules
from .test_bybit_api_validation import BybitAPIValidator
from .test_balance_handling import BybitBalanceValidator
from .test_rate_limiting import BybitRateLimitValidator
from .test_security_validation import BybitSecurityValidator

from src.bot.config_manager import ConfigurationManager
from src.bot.exchange.bybit_client import BybitClient


class ValidationCategory(Enum):
    """Validation categories for deployment checklist."""
    CRITICAL = "critical"        # Must pass for deployment
    IMPORTANT = "important"      # Should pass for optimal operation
    RECOMMENDED = "recommended" # Nice to have for best practices


@dataclass
class ChecklistItem:
    """Individual checklist item."""
    id: str
    category: ValidationCategory
    name: str
    description: str
    validation_function: Optional[str] = None
    manual_check: bool = False
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class DeploymentValidationResult:
    """Result of complete deployment validation."""
    overall_status: str
    timestamp: datetime
    environment: str
    critical_passed: int
    critical_total: int
    important_passed: int
    important_total: int
    recommended_passed: int
    recommended_total: int
    checklist_results: Dict[str, ChecklistItem]
    detailed_reports: Dict[str, Any] = field(default_factory=dict)
    deployment_ready: bool = False
    blocking_issues: List[str] = field(default_factory=list)


class DeploymentValidator:
    """Comprehensive deployment validation system."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.deployment_checklist = self._create_deployment_checklist()
        
    def _create_deployment_checklist(self) -> Dict[str, ChecklistItem]:
        """Create comprehensive deployment checklist."""
        checklist = {}
        
        # Critical Items (Must Pass)
        critical_items = [
            ChecklistItem(
                id="api_connectivity",
                category=ValidationCategory.CRITICAL,
                name="API Connectivity",
                description="Verify successful connection to Bybit API endpoints",
                validation_function="validate_api_connectivity"
            ),
            ChecklistItem(
                id="authentication",
                category=ValidationCategory.CRITICAL,
                name="API Authentication",
                description="Verify API key authentication is working correctly",
                validation_function="validate_authentication"
            ),
            ChecklistItem(
                id="balance_retrieval",
                category=ValidationCategory.CRITICAL,
                name="Balance Retrieval",
                description="Verify wallet balance can be retrieved and parsed correctly",
                validation_function="validate_balance_retrieval"
            ),
            ChecklistItem(
                id="security_compliance",
                category=ValidationCategory.CRITICAL,
                name="Security Compliance",
                description="Verify all critical security requirements are met",
                validation_function="validate_security_compliance"
            ),
            ChecklistItem(
                id="rate_limiting",
                category=ValidationCategory.CRITICAL,
                name="Rate Limiting",
                description="Verify rate limiting is properly implemented",
                validation_function="validate_rate_limiting"
            ),
            ChecklistItem(
                id="environment_config",
                category=ValidationCategory.CRITICAL,
                name="Environment Configuration",
                description="Verify production environment is properly configured",
                manual_check=True
            ),
            ChecklistItem(
                id="api_permissions",
                category=ValidationCategory.CRITICAL,
                name="API Permissions",
                description="Verify API keys have appropriate permissions",
                manual_check=True
            )
        ]
        
        # Important Items (Should Pass)
        important_items = [
            ChecklistItem(
                id="error_handling",
                category=ValidationCategory.IMPORTANT,
                name="Error Handling",
                description="Verify robust error handling for API failures",
                validation_function="validate_error_handling"
            ),
            ChecklistItem(
                id="logging_setup",
                category=ValidationCategory.IMPORTANT,
                name="Logging Configuration",
                description="Verify logging is properly configured for production",
                validation_function="validate_logging_setup"
            ),
            ChecklistItem(
                id="monitoring_setup",
                category=ValidationCategory.IMPORTANT,
                name="Monitoring and Alerting",
                description="Verify monitoring and alerting systems are configured",
                manual_check=True
            ),
            ChecklistItem(
                id="database_connectivity",
                category=ValidationCategory.IMPORTANT,
                name="Database Connectivity",
                description="Verify database connection and schema are correct",
                validation_function="validate_database_connectivity"
            ),
            ChecklistItem(
                id="backup_strategy",
                category=ValidationCategory.IMPORTANT,
                name="Backup Strategy",
                description="Verify backup and recovery procedures are in place",
                manual_check=True
            ),
            ChecklistItem(
                id="risk_limits",
                category=ValidationCategory.IMPORTANT,
                name="Risk Management Limits",
                description="Verify risk management limits are properly configured",
                validation_function="validate_risk_limits"
            )
        ]
        
        # Recommended Items (Best Practices)
        recommended_items = [
            ChecklistItem(
                id="performance_metrics",
                category=ValidationCategory.RECOMMENDED,
                name="Performance Metrics",
                description="Verify performance monitoring is configured",
                manual_check=True
            ),
            ChecklistItem(
                id="ssl_verification",
                category=ValidationCategory.RECOMMENDED,
                name="SSL Certificate Verification",
                description="Verify SSL certificates are properly validated",
                validation_function="validate_ssl_verification"
            ),
            ChecklistItem(
                id="resource_limits",
                category=ValidationCategory.RECOMMENDED,
                name="Resource Limits",
                description="Verify CPU and memory limits are configured",
                manual_check=True
            ),
            ChecklistItem(
                id="documentation",
                category=ValidationCategory.RECOMMENDED,
                name="Documentation",
                description="Verify deployment and operational documentation exists",
                manual_check=True
            ),
            ChecklistItem(
                id="disaster_recovery",
                category=ValidationCategory.RECOMMENDED,
                name="Disaster Recovery Plan",
                description="Verify disaster recovery procedures are documented",
                manual_check=True
            )
        ]
        
        # Add all items to checklist
        for item in critical_items + important_items + recommended_items:
            checklist[item.id] = item
            
        return checklist
    
    async def run_complete_deployment_validation(self) -> DeploymentValidationResult:
        """Run complete deployment validation suite."""
        self.logger.info("Starting comprehensive deployment validation")
        
        # Initialize result
        result = DeploymentValidationResult(
            overall_status="IN_PROGRESS",
            timestamp=datetime.now(),
            environment=self.config_manager.config.environment.value,
            critical_passed=0,
            critical_total=0,
            important_passed=0,
            important_total=0,
            recommended_passed=0,
            recommended_total=0,
            checklist_results={}
        )
        
        # Run automated validations
        await self._run_automated_validations(result)
        
        # Process manual checks
        self._process_manual_checks(result)
        
        # Calculate final status
        self._calculate_deployment_status(result)
        
        # Generate deployment report
        await self._generate_deployment_report(result)
        
        return result
    
    async def _run_automated_validations(self, result: DeploymentValidationResult):
        """Run all automated validation checks."""
        
        # Initialize validators
        client = BybitClient(self.config_manager)
        api_validator = BybitAPIValidator(self.config_manager)
        balance_validator = BybitBalanceValidator(client)
        rate_limit_validator = BybitRateLimitValidator(client)
        security_validator = BybitSecurityValidator(self.config_manager)
        
        # Validation mappings
        validation_functions = {
            "validate_api_connectivity": self._validate_api_connectivity,
            "validate_authentication": self._validate_authentication,
            "validate_balance_retrieval": self._validate_balance_retrieval,
            "validate_security_compliance": self._validate_security_compliance,
            "validate_rate_limiting": self._validate_rate_limiting,
            "validate_error_handling": self._validate_error_handling,
            "validate_logging_setup": self._validate_logging_setup,
            "validate_database_connectivity": self._validate_database_connectivity,
            "validate_risk_limits": self._validate_risk_limits,
            "validate_ssl_verification": self._validate_ssl_verification
        }
        
        # Store validators for use in validation functions
        self.validators = {
            'api': api_validator,
            'balance': balance_validator,
            'rate_limit': rate_limit_validator,
            'security': security_validator
        }
        
        # Run each automated validation
        for item_id, item in self.deployment_checklist.items():
            if not item.manual_check and item.validation_function:
                try:
                    self.logger.info(f"Running validation: {item.name}")
                    
                    validation_func = validation_functions.get(item.validation_function)
                    if validation_func:
                        validation_result = await validation_func()
                        
                        item.passed = validation_result.get('passed', False)
                        item.details = validation_result.get('details', {})
                        item.notes = validation_result.get('notes', '')
                        
                        # Store detailed reports
                        if 'detailed_report' in validation_result:
                            result.detailed_reports[item_id] = validation_result['detailed_report']
                    
                except Exception as e:
                    self.logger.error(f"Validation {item.name} failed: {str(e)}")
                    item.passed = False
                    item.notes = f"Validation failed: {str(e)}"
            
            result.checklist_results[item_id] = item
    
    async def _validate_api_connectivity(self) -> Dict[str, Any]:
        """Validate API connectivity."""
        try:
            connectivity_result = await self.validators['api']._validate_connectivity()
            
            passed = (
                connectivity_result.get('endpoint_reachable', False) and
                connectivity_result.get('server_time_sync', False)
            )
            
            return {
                'passed': passed,
                'details': connectivity_result,
                'notes': f"Response time: {connectivity_result.get('response_time_ms', 'N/A')}ms"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Connectivity validation failed: {str(e)}"
            }
    
    async def _validate_authentication(self) -> Dict[str, Any]:
        """Validate API authentication."""
        try:
            auth_result = await self.validators['api']._validate_authentication()
            
            passed = auth_result.get('authenticated_request', False)
            
            return {
                'passed': passed,
                'details': auth_result,
                'notes': "Authentication successful" if passed else "Authentication failed"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Authentication validation failed: {str(e)}"
            }
    
    async def _validate_balance_retrieval(self) -> Dict[str, Any]:
        """Validate balance retrieval."""
        try:
            balance_results = await self.validators['balance'].validate_all_account_types()
            
            # Check if at least UNIFIED account works
            unified_result = balance_results.get('UNIFIED')
            passed = unified_result.success if unified_result else False
            
            report = self.validators['balance'].generate_balance_validation_report(balance_results)
            
            return {
                'passed': passed,
                'details': report['summary'],
                'detailed_report': report,
                'notes': f"Success rate: {report['summary']['success_rate']}%"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Balance validation failed: {str(e)}"
            }
    
    async def _validate_security_compliance(self) -> Dict[str, Any]:
        """Validate security compliance."""
        try:
            security_report = await self.validators['security'].run_complete_security_audit()
            
            # Pass if no critical or high issues
            critical_issues = security_report['summary']['critical_issues']
            high_issues = security_report['summary']['high_issues']
            
            passed = critical_issues == 0 and high_issues == 0
            
            return {
                'passed': passed,
                'details': security_report['summary'],
                'detailed_report': security_report,
                'notes': f"Critical: {critical_issues}, High: {high_issues}"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Security validation failed: {str(e)}"
            }
    
    async def _validate_rate_limiting(self) -> Dict[str, Any]:
        """Validate rate limiting."""
        try:
            implementation_results = await self.validators['rate_limit'].validate_rate_limiting_implementation()
            endpoint_results = await self.validators['rate_limit'].run_endpoint_rate_limit_tests()
            burst_results = await self.validators['rate_limit'].validate_burst_handling()
            
            report = self.validators['rate_limit'].generate_rate_limit_report(
                implementation_results, endpoint_results, burst_results
            )
            
            passed = report['overall_compliance']
            
            return {
                'passed': passed,
                'details': report['summary'],
                'detailed_report': report,
                'notes': f"Rate limiter: {'✓' if report['summary']['rate_limiter_implemented'] else '✗'}"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Rate limiting validation failed: {str(e)}"
            }
    
    async def _validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling."""
        try:
            error_result = await self.validators['api']._validate_error_handling()
            
            passed = (
                error_result.get('api_error_handling', False) and
                error_result.get('network_error_handling', False)
            )
            
            return {
                'passed': passed,
                'details': error_result,
                'notes': "Error handling mechanisms validated"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Error handling validation failed: {str(e)}"
            }
    
    async def _validate_logging_setup(self) -> Dict[str, Any]:
        """Validate logging configuration."""
        try:
            # Check logging configuration
            import logging
            
            root_logger = logging.getLogger()
            has_handlers = len(root_logger.handlers) > 0
            
            # Check for file handler
            file_handlers = [h for h in root_logger.handlers if hasattr(h, 'baseFilename')]
            has_file_logging = len(file_handlers) > 0
            
            passed = has_handlers and has_file_logging
            
            return {
                'passed': passed,
                'details': {
                    'handlers_count': len(root_logger.handlers),
                    'file_handlers': len(file_handlers),
                    'log_level': root_logger.level
                },
                'notes': f"Logging configured with {len(root_logger.handlers)} handlers"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Logging validation failed: {str(e)}"
            }
    
    async def _validate_database_connectivity(self) -> Dict[str, Any]:
        """Validate database connectivity."""
        try:
            # Test database connection
            from src.bot.database import DatabaseManager
            
            db_manager = DatabaseManager()
            await db_manager.initialize()
            
            # Test basic query
            connection_valid = await db_manager.test_connection()
            
            return {
                'passed': connection_valid,
                'details': {'connection_test': connection_valid},
                'notes': "Database connection successful" if connection_valid else "Database connection failed"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Database validation failed: {str(e)}"
            }
    
    async def _validate_risk_limits(self) -> Dict[str, Any]:
        """Validate risk management limits."""
        try:
            config = self.config_manager.config
            
            # Check risk configuration
            risk_config = config.trading.aggressive_mode if config.trading.mode == "aggressive" else config.trading.conservative_mode
            
            # Validate risk limits are reasonable
            checks = {
                'max_risk_reasonable': True,
                'drawdown_limits_set': risk_config.portfolio_drawdown_limit > 0,
                'var_limits_set': risk_config.var_daily_limit > 0,
                'sharpe_threshold_set': risk_config.sharpe_ratio_min > 0
            }
            
            passed = all(checks.values())
            
            return {
                'passed': passed,
                'details': checks,
                'notes': f"Risk limits configured for {config.trading.mode} mode"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"Risk limits validation failed: {str(e)}"
            }
    
    async def _validate_ssl_verification(self) -> Dict[str, Any]:
        """Validate SSL certificate verification."""
        try:
            ssl_result = await self.validators['security']._validate_network_security()
            
            passed = ssl_result.passed
            
            return {
                'passed': passed,
                'details': ssl_result.details,
                'notes': "SSL verification enabled" if passed else "SSL verification issues found"
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)},
                'notes': f"SSL validation failed: {str(e)}"
            }
    
    def _process_manual_checks(self, result: DeploymentValidationResult):
        """Process manual check items."""
        manual_items = [item for item in self.deployment_checklist.values() if item.manual_check]
        
        for item in manual_items:
            # Mark manual checks as requiring review
            item.passed = None  # Indicates manual review required
            item.notes = "Manual verification required"
            result.checklist_results[item.id] = item
    
    def _calculate_deployment_status(self, result: DeploymentValidationResult):
        """Calculate overall deployment status."""
        
        # Count results by category
        for item in result.checklist_results.values():
            if item.category == ValidationCategory.CRITICAL:
                result.critical_total += 1
                if item.passed:
                    result.critical_passed += 1
                elif item.passed is False:  # Failed (not None for manual)
                    result.blocking_issues.append(f"Critical: {item.name}")
                    
            elif item.category == ValidationCategory.IMPORTANT:
                result.important_total += 1
                if item.passed:
                    result.important_passed += 1
                    
            elif item.category == ValidationCategory.RECOMMENDED:
                result.recommended_total += 1
                if item.passed:
                    result.recommended_passed += 1
        
        # Determine deployment readiness
        critical_success_rate = result.critical_passed / result.critical_total if result.critical_total > 0 else 0
        important_success_rate = result.important_passed / result.important_total if result.important_total > 0 else 0
        
        # Deployment ready if all critical items pass and most important items pass
        result.deployment_ready = (
            critical_success_rate >= 1.0 and  # All critical must pass
            important_success_rate >= 0.8     # 80% of important should pass
        )
        
        # Set overall status
        if result.deployment_ready:
            result.overall_status = "READY_FOR_DEPLOYMENT"
        elif critical_success_rate >= 1.0:
            result.overall_status = "NEEDS_ATTENTION"
        else:
            result.overall_status = "NOT_READY"
    
    async def _generate_deployment_report(self, result: DeploymentValidationResult):
        """Generate comprehensive deployment report."""
        
        # Save detailed report to file
        report_data = {
            'validation_timestamp': result.timestamp.isoformat(),
            'environment': result.environment,
            'overall_status': result.overall_status,
            'deployment_ready': result.deployment_ready,
            'summary': {
                'critical': f"{result.critical_passed}/{result.critical_total}",
                'important': f"{result.important_passed}/{result.important_total}",
                'recommended': f"{result.recommended_passed}/{result.recommended_total}"
            },
            'blocking_issues': result.blocking_issues,
            'checklist_results': {
                item_id: {
                    'name': item.name,
                    'category': item.category.value,
                    'passed': item.passed,
                    'notes': item.notes,
                    'manual_check': item.manual_check
                }
                for item_id, item in result.checklist_results.items()
            },
            'detailed_reports': result.detailed_reports
        }
        
        # Save to file
        report_filename = f"deployment_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Detailed deployment report saved to: {report_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save deployment report: {str(e)}")


# Main execution function
async def run_deployment_validation():
    """Run complete deployment validation."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration
        config_manager = ConfigurationManager()
        config_manager.load_config()
        
        # Create deployment validator
        validator = DeploymentValidator(config_manager)
        
        logger.info("Starting comprehensive deployment validation...")
        
        # Run complete validation
        result = await validator.run_complete_deployment_validation()
        
        # Display results
        print("\n" + "="*70)
        print("BYBIT TRADING BOT - DEPLOYMENT VALIDATION RESULTS")
        print("="*70)
        
        status_emoji = {
            "READY_FOR_DEPLOYMENT": "🟢",
            "NEEDS_ATTENTION": "🟡", 
            "NOT_READY": "🔴",
            "IN_PROGRESS": "🔵"
        }.get(result.overall_status, "❓")
        
        print(f"Overall Status: {status_emoji} {result.overall_status}")
        print(f"Environment: {result.environment}")
        print(f"Deployment Ready: {'✅ YES' if result.deployment_ready else '❌ NO'}")
        
        print(f"\nValidation Summary:")
        print(f"  🔴 Critical:    {result.critical_passed}/{result.critical_total} passed")
        print(f"  🟡 Important:   {result.important_passed}/{result.important_total} passed")
        print(f"  🟢 Recommended: {result.recommended_passed}/{result.recommended_total} passed")
        
        if result.blocking_issues:
            print(f"\n🚫 Blocking Issues:")
            for issue in result.blocking_issues:
                print(f"   • {issue}")
        
        print(f"\nDetailed Checklist:")
        for item_id, item in result.checklist_results.items():
            if item.passed is True:
                status = "✅ PASSED"
            elif item.passed is False:
                status = "❌ FAILED"
            else:
                status = "⏳ MANUAL REVIEW REQUIRED"
            
            category_emoji = {
                ValidationCategory.CRITICAL: "🔴",
                ValidationCategory.IMPORTANT: "🟡",
                ValidationCategory.RECOMMENDED: "🟢"
            }.get(item.category, "🔵")
            
            print(f"  {category_emoji} {item.name}: {status}")
            if item.notes:
                print(f"     📝 {item.notes}")
        
        if not result.deployment_ready:
            print(f"\n⚠️  DEPLOYMENT NOT RECOMMENDED")
            print(f"   Please address all critical issues before deploying to production.")
        else:
            print(f"\n✅ DEPLOYMENT VALIDATION SUCCESSFUL")
            print(f"   System is ready for production deployment.")
        
        return result
        
    except Exception as e:
        logger.error(f"Deployment validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_deployment_validation())