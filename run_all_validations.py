"""
Master Validation Runner

This script runs all Bybit API validation steps comprehensively,
providing a single entry point for complete system validation.

Usage:
    python run_all_validations.py [--environment ENV] [--output-format FORMAT]

Features:
- Runs all validation suites in sequence
- Generates comprehensive reports
- Provides deployment readiness assessment
- Supports multiple output formats (console, json, html)
"""

import asyncio
import argparse
import logging
import json
import sys
from datetime import datetime
from pathlib import Path

# Import all validation modules
from tests.validation.test_bybit_api_validation import BybitAPIValidator
from tests.validation.test_balance_handling import run_balance_validation_suite
from tests.validation.test_rate_limiting import run_rate_limit_validation_suite
from tests.validation.test_security_validation import run_security_validation_suite
from tests.validation.deployment_validation import run_deployment_validation

from src.bot.config_manager import ConfigurationManager


class MasterValidationRunner:
    """Master validation runner for complete system validation."""
    
    def __init__(self, environment: str = None, output_format: str = "console"):
        self.environment = environment
        self.output_format = output_format
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        
    async def run_all_validations(self):
        """Run all validation suites."""
        self.logger.info("Starting comprehensive Bybit API validation suite")
        
        validation_suites = [
            ("API Connectivity & Authentication", self._run_api_validation),
            ("Balance Handling Validation", self._run_balance_validation),
            ("Rate Limiting Compliance", self._run_rate_limit_validation),
            ("Security Validation", self._run_security_validation),
            ("Deployment Readiness", self._run_deployment_validation)
        ]
        
        overall_success = True
        
        for suite_name, validation_func in validation_suites:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Running: {suite_name}")
                self.logger.info(f"{'='*60}")
                
                result = await validation_func()
                self.validation_results[suite_name] = result
                
                # Check if this suite passed
                suite_passed = self._determine_suite_success(result, suite_name)
                if not suite_passed:
                    overall_success = False
                    
            except Exception as e:
                self.logger.error(f"Validation suite '{suite_name}' failed: {str(e)}")
                self.validation_results[suite_name] = {
                    'error': str(e),
                    'success': False
                }
                overall_success = False
        
        # Generate final report
        final_report = self._generate_final_report(overall_success)
        
        # Output results
        if self.output_format == "console":
            self._output_console_report(final_report)
        elif self.output_format == "json":
            self._output_json_report(final_report)
        elif self.output_format == "html":
            self._output_html_report(final_report)
        
        return final_report
    
    async def _run_api_validation(self):
        """Run API connectivity and authentication validation."""
        try:
            config_manager = ConfigurationManager()
            config_manager.load_config()
            
            validator = BybitAPIValidator(config_manager)
            results = await validator.run_full_validation()
            
            return results
            
        except Exception as e:
            self.logger.error(f"API validation failed: {str(e)}")
            return {'error': str(e), 'overall_status': 'FAILED'}
    
    async def _run_balance_validation(self):
        """Run balance handling validation."""
        try:
            results = await run_balance_validation_suite()
            return results
            
        except Exception as e:
            self.logger.error(f"Balance validation failed: {str(e)}")
            return {'error': str(e), 'summary': {'success_rate': 0}}
    
    async def _run_rate_limit_validation(self):
        """Run rate limiting validation."""
        try:
            results = await run_rate_limit_validation_suite()
            return results
            
        except Exception as e:
            self.logger.error(f"Rate limiting validation failed: {str(e)}")
            return {'error': str(e), 'overall_compliance': False}
    
    async def _run_security_validation(self):
        """Run security validation."""
        try:
            results = await run_security_validation_suite()
            return results
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {str(e)}")
            return {'error': str(e), 'overall_security_status': 'VULNERABLE'}
    
    async def _run_deployment_validation(self):
        """Run deployment readiness validation."""
        try:
            results = await run_deployment_validation()
            return results
            
        except Exception as e:
            self.logger.error(f"Deployment validation failed: {str(e)}")
            return {'error': str(e), 'deployment_ready': False}
    
    def _determine_suite_success(self, result, suite_name):
        """Determine if a validation suite was successful."""
        if 'error' in result:
            return False
        
        if suite_name == "API Connectivity & Authentication":
            return result.get('overall_status') == 'PASSED'
        elif suite_name == "Balance Handling Validation":
            return result.get('summary', {}).get('success_rate', 0) >= 80
        elif suite_name == "Rate Limiting Compliance":
            return result.get('overall_compliance', False)
        elif suite_name == "Security Validation":
            return result.get('overall_security_status') == 'SECURE'
        elif suite_name == "Deployment Readiness":
            return result.get('deployment_ready', False)
        
        return False
    
    def _generate_final_report(self, overall_success):
        """Generate comprehensive final report."""
        
        # Count successful suites
        successful_suites = sum(
            1 for suite_name, result in self.validation_results.items()
            if self._determine_suite_success(result, suite_name)
        )
        total_suites = len(self.validation_results)
        
        # Extract key metrics
        deployment_ready = False
        critical_issues = []
        recommendations = []
        
        if "Deployment Readiness" in self.validation_results:
            deployment_result = self.validation_results["Deployment Readiness"]
            deployment_ready = deployment_result.get('deployment_ready', False)
            critical_issues.extend(deployment_result.get('blocking_issues', []))
        
        # Collect security issues
        if "Security Validation" in self.validation_results:
            security_result = self.validation_results["Security Validation"]
            security_issues = security_result.get('security_issues', [])
            critical_security = [
                issue for issue in security_issues 
                if issue.get('severity') in ['CRITICAL', 'HIGH']
            ]
            critical_issues.extend([issue['description'] for issue in critical_security])
        
        return {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_success': overall_success,
            'deployment_ready': deployment_ready,
            'summary': {
                'successful_suites': successful_suites,
                'total_suites': total_suites,
                'success_rate': round((successful_suites / total_suites) * 100, 1) if total_suites > 0 else 0
            },
            'suite_results': {
                suite_name: {
                    'success': self._determine_suite_success(result, suite_name),
                    'summary': self._extract_suite_summary(result, suite_name)
                }
                for suite_name, result in self.validation_results.items()
            },
            'critical_issues': critical_issues,
            'recommendations': recommendations,
            'detailed_results': self.validation_results
        }
    
    def _extract_suite_summary(self, result, suite_name):
        """Extract summary from suite result."""
        if 'error' in result:
            return f"Failed: {result['error']}"
        
        if suite_name == "API Connectivity & Authentication":
            status = result.get('overall_status', 'UNKNOWN')
            success_rate = result.get('summary', {}).get('success_rate', 0)
            return f"Status: {status}, Success Rate: {success_rate}%"
            
        elif suite_name == "Balance Handling Validation":
            success_rate = result.get('summary', {}).get('success_rate', 0)
            return f"Success Rate: {success_rate}%"
            
        elif suite_name == "Rate Limiting Compliance":
            compliance = result.get('overall_compliance', False)
            return f"Compliant: {'Yes' if compliance else 'No'}"
            
        elif suite_name == "Security Validation":
            status = result.get('overall_security_status', 'UNKNOWN')
            critical = result.get('summary', {}).get('critical_issues', 0)
            high = result.get('summary', {}).get('high_issues', 0)
            return f"Status: {status}, Critical: {critical}, High: {high}"
            
        elif suite_name == "Deployment Readiness":
            ready = result.get('deployment_ready', False)
            status = result.get('overall_status', 'UNKNOWN')
            return f"Ready: {'Yes' if ready else 'No'}, Status: {status}"
        
        return "Unknown"
    
    def _output_console_report(self, report):
        """Output results to console."""
        print("\n" + "="*80)
        print("COMPREHENSIVE BYBIT API VALIDATION RESULTS")
        print("="*80)
        
        overall_emoji = "✅" if report['overall_success'] else "❌"
        deployment_emoji = "🚀" if report['deployment_ready'] else "⏸️"
        
        print(f"Overall Validation: {overall_emoji} {'PASSED' if report['overall_success'] else 'FAILED'}")
        print(f"Deployment Ready: {deployment_emoji} {'YES' if report['deployment_ready'] else 'NO'}")
        print(f"Success Rate: {report['summary']['success_rate']}% ({report['summary']['successful_suites']}/{report['summary']['total_suites']} suites)")
        
        print(f"\nValidation Suite Results:")
        for suite_name, suite_result in report['suite_results'].items():
            status_emoji = "✅" if suite_result['success'] else "❌"
            print(f"  {status_emoji} {suite_name}")
            print(f"    📊 {suite_result['summary']}")
        
        if report['critical_issues']:
            print(f"\n🚨 Critical Issues Found:")
            for issue in report['critical_issues']:
                print(f"  • {issue}")
        
        if not report['deployment_ready']:
            print(f"\n⚠️  DEPLOYMENT NOT RECOMMENDED")
            print(f"   Please address critical issues before production deployment.")
        else:
            print(f"\n✅ SYSTEM READY FOR DEPLOYMENT")
            print(f"   All critical validations passed successfully.")
        
        print(f"\nValidation completed at: {report['validation_timestamp']}")
    
    def _output_json_report(self, report):
        """Output results to JSON file."""
        filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"Validation report saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON report: {str(e)}")
    
    def _output_html_report(self, report):
        """Output results to HTML file."""
        filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = self._generate_html_report(report)
        
        try:
            with open(filename, 'w') as f:
                f.write(html_content)
            
            print(f"HTML validation report saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save HTML report: {str(e)}")
    
    def _generate_html_report(self, report):
        """Generate HTML report content."""
        
        overall_color = "#28a745" if report['overall_success'] else "#dc3545"
        deployment_color = "#28a745" if report['deployment_ready'] else "#ffc107"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bybit API Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; }}
        .summary {{ background-color: {overall_color}; color: white; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .suite {{ background-color: #ffffff; border: 1px solid #dee2e6; margin: 10px 0; padding: 15px; border-radius: 8px; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .issues {{ background-color: #f8d7da; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        ul {{ padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Bybit Trading Bot - API Validation Report</h1>
        <p><strong>Generated:</strong> {report['validation_timestamp']}</p>
    </div>
    
    <div class="summary">
        <h2>Overall Results</h2>
        <p><strong>Validation Status:</strong> {'PASSED' if report['overall_success'] else 'FAILED'}</p>
        <p><strong>Deployment Ready:</strong> {'YES' if report['deployment_ready'] else 'NO'}</p>
        <p><strong>Success Rate:</strong> {report['summary']['success_rate']}% ({report['summary']['successful_suites']}/{report['summary']['total_suites']} suites)</p>
    </div>
"""
        
        # Add suite results
        html += "<h2>Validation Suite Details</h2>"
        for suite_name, suite_result in report['suite_results'].items():
            status_class = "success" if suite_result['success'] else "failure"
            status_text = "PASSED" if suite_result['success'] else "FAILED"
            
            html += f"""
    <div class="suite">
        <h3 class="{status_class}">{suite_name} - {status_text}</h3>
        <p>{suite_result['summary']}</p>
    </div>
"""
        
        # Add critical issues if any
        if report['critical_issues']:
            html += """
    <div class="issues">
        <h2>Critical Issues</h2>
        <ul>
"""
            for issue in report['critical_issues']:
                html += f"            <li>{issue}</li>\n"
            
            html += """
        </ul>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run comprehensive Bybit API validation")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       help="Environment to validate")
    parser.add_argument("--output-format", choices=["console", "json", "html"], 
                       default="console", help="Output format for results")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create validation runner
    runner = MasterValidationRunner(
        environment=args.environment,
        output_format=args.output_format
    )
    
    try:
        # Run all validations
        final_report = await runner.run_all_validations()
        
        # Exit with appropriate code
        exit_code = 0 if final_report['overall_success'] else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logging.error(f"Validation runner failed: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())