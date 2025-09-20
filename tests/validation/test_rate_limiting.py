"""
Bybit Rate Limiting Compliance Validator

This module validates that the trading bot properly implements rate limiting
according to Bybit's official API documentation and best practices.

Reference: https://bybit-exchange.github.io/docs/v5/rate-limit
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

from src.bot.exchange.bybit_client import BybitClient
from src.bot.utils.rate_limiter import RateLimiter
from src.bot.config_manager import ConfigurationManager


@dataclass
class RateLimit:
    """Rate limit specification."""
    endpoint_pattern: str
    requests_per_second: int
    requests_per_minute: int
    burst_limit: Optional[int] = None
    weight_per_request: int = 1


@dataclass
class RateLimitTest:
    """Rate limit test configuration."""
    name: str
    endpoint: str
    rate_limit: RateLimit
    test_duration_seconds: int = 60
    request_count: int = 100
    expected_compliance: bool = True


@dataclass
class RateLimitResult:
    """Result of rate limit test."""
    test_name: str
    compliance_status: bool
    actual_rps: float
    actual_rpm: float
    expected_rps: int
    expected_rpm: int
    violations: List[str] = field(default_factory=list)
    timing_analysis: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class BybitRateLimitValidator:
    """Validates rate limiting compliance with Bybit API limits."""
    
    # Bybit API rate limits as per documentation
    BYBIT_RATE_LIMITS = {
        'get_wallet_balance': RateLimit(
            endpoint_pattern='/v5/account/wallet-balance',
            requests_per_second=10,
            requests_per_minute=600,
            weight_per_request=1
        ),
        'get_positions': RateLimit(
            endpoint_pattern='/v5/position/list',
            requests_per_second=10,
            requests_per_minute=600,
            weight_per_request=1
        ),
        'place_order': RateLimit(
            endpoint_pattern='/v5/order/create',
            requests_per_second=10,
            requests_per_minute=600,
            weight_per_request=1
        ),
        'cancel_order': RateLimit(
            endpoint_pattern='/v5/order/cancel',
            requests_per_second=10,
            requests_per_minute=600,
            weight_per_request=1
        ),
        'get_orderbook': RateLimit(
            endpoint_pattern='/v5/market/orderbook',
            requests_per_second=50,
            requests_per_minute=3000,
            weight_per_request=1
        ),
        'get_kline': RateLimit(
            endpoint_pattern='/v5/market/kline',
            requests_per_second=20,
            requests_per_minute=1200,
            weight_per_request=1
        )
    }
    
    def __init__(self, client: BybitClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.request_history = defaultdict(deque)
        
    async def validate_rate_limiting_implementation(self) -> Dict[str, Any]:
        """Validate the overall rate limiting implementation."""
        results = {
            'rate_limiter_present': False,
            'configuration_valid': False,
            'enforcement_working': False,
            'backoff_mechanism': False,
            'details': {}
        }
        
        # Check if rate limiter is present
        if hasattr(self.client, 'rate_limiter') and self.client.rate_limiter:
            results['rate_limiter_present'] = True
            rate_limiter = self.client.rate_limiter
            
            # Validate configuration
            config_result = await self._validate_rate_limiter_config(rate_limiter)
            results['configuration_valid'] = config_result['valid']
            results['details']['configuration'] = config_result
            
            # Test enforcement
            enforcement_result = await self._test_rate_limit_enforcement(rate_limiter)
            results['enforcement_working'] = enforcement_result['working']
            results['details']['enforcement'] = enforcement_result
            
            # Test backoff mechanism
            backoff_result = await self._test_backoff_mechanism(rate_limiter)
            results['backoff_mechanism'] = backoff_result['present']
            results['details']['backoff'] = backoff_result
        
        return results
    
    async def _validate_rate_limiter_config(self, rate_limiter: RateLimiter) -> Dict[str, Any]:
        """Validate rate limiter configuration."""
        result = {
            'valid': False,
            'issues': [],
            'settings': {}
        }
        
        try:
            # Check if rate limiter has proper configuration
            if hasattr(rate_limiter, 'requests_per_second'):
                rps = getattr(rate_limiter, 'requests_per_second')
                result['settings']['requests_per_second'] = rps
                
                # Validate against Bybit limits
                if rps > 50:  # Most restrictive Bybit limit
                    result['issues'].append(f"RPS too high: {rps} (Bybit max: 50 for some endpoints)")
            
            if hasattr(rate_limiter, 'requests_per_minute'):
                rpm = getattr(rate_limiter, 'requests_per_minute')
                result['settings']['requests_per_minute'] = rpm
                
                if rpm > 3000:  # Most permissive Bybit limit
                    result['issues'].append(f"RPM too high: {rpm} (Bybit max: 3000)")
            
            # Check for burst handling
            if hasattr(rate_limiter, 'burst_capacity'):
                result['settings']['burst_capacity'] = getattr(rate_limiter, 'burst_capacity')
            
            result['valid'] = len(result['issues']) == 0
            
        except Exception as e:
            result['issues'].append(f"Configuration validation error: {str(e)}")
        
        return result
    
    async def _test_rate_limit_enforcement(self, rate_limiter: RateLimiter) -> Dict[str, Any]:
        """Test if rate limiting is actually enforced."""
        result = {
            'working': False,
            'test_results': [],
            'timing_analysis': {}
        }
        
        try:
            # Test rapid requests to see if they're limited
            request_times = []
            test_count = 10
            
            for i in range(test_count):
                start_time = time.time()
                await rate_limiter.acquire()
                end_time = time.time()
                
                delay = end_time - start_time
                request_times.append(delay)
            
            # Analyze timing
            avg_delay = statistics.mean(request_times)
            min_delay = min(request_times)
            max_delay = max(request_times)
            
            result['timing_analysis'] = {
                'average_delay_ms': round(avg_delay * 1000, 2),
                'min_delay_ms': round(min_delay * 1000, 2),
                'max_delay_ms': round(max_delay * 1000, 2),
                'requests_tested': test_count
            }
            
            # If there's consistent delay, rate limiting is working
            result['working'] = avg_delay > 0.01  # At least 10ms average delay
            
        except Exception as e:
            result['test_results'].append(f"Enforcement test error: {str(e)}")
        
        return result
    
    async def _test_backoff_mechanism(self, rate_limiter: RateLimiter) -> Dict[str, Any]:
        """Test exponential backoff mechanism."""
        result = {
            'present': False,
            'effectiveness': 'unknown',
            'backoff_pattern': []
        }
        
        try:
            # Test multiple rapid acquisitions to trigger backoff
            delays = []
            
            for i in range(5):
                start_time = time.time()
                await rate_limiter.acquire()
                delay = time.time() - start_time
                delays.append(delay)
                
                # Make rapid requests to potentially trigger backoff
                for _ in range(3):
                    await rate_limiter.acquire()
            
            result['backoff_pattern'] = [round(d * 1000, 2) for d in delays]
            
            # Check if delays increase (indicating backoff)
            if len(delays) > 2:
                increasing_pattern = all(
                    delays[i] <= delays[i+1] * 1.1  # Allow 10% variance
                    for i in range(len(delays)-1)
                )
                result['present'] = increasing_pattern
                result['effectiveness'] = 'working' if increasing_pattern else 'not_detected'
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    async def run_endpoint_rate_limit_tests(self) -> Dict[str, RateLimitResult]:
        """Run rate limit tests for specific API endpoints."""
        results = {}
        
        test_cases = [
            RateLimitTest(
                name="wallet_balance_rate_test",
                endpoint="get_wallet_balance",
                rate_limit=self.BYBIT_RATE_LIMITS['get_wallet_balance'],
                request_count=30,
                test_duration_seconds=10
            ),
            RateLimitTest(
                name="orderbook_rate_test", 
                endpoint="get_orderbook",
                rate_limit=self.BYBIT_RATE_LIMITS['get_orderbook'],
                request_count=60,
                test_duration_seconds=10
            )
        ]
        
        for test_case in test_cases:
            self.logger.info(f"Running rate limit test: {test_case.name}")
            result = await self._run_single_rate_limit_test(test_case)
            results[test_case.name] = result
        
        return results
    
    async def _run_single_rate_limit_test(self, test_case: RateLimitTest) -> RateLimitResult:
        """Run a single rate limit test."""
        result = RateLimitResult(
            test_name=test_case.name,
            compliance_status=False,
            actual_rps=0.0,
            actual_rpm=0.0,
            expected_rps=test_case.rate_limit.requests_per_second,
            expected_rpm=test_case.rate_limit.requests_per_minute
        )
        
        request_timestamps = []
        errors = []
        
        try:
            start_time = time.time()
            
            # Make requests for test duration
            for i in range(test_case.request_count):
                request_start = time.time()
                
                try:
                    # Make actual API request based on endpoint
                    if test_case.endpoint == "get_wallet_balance":
                        await self.client.get_wallet_balance(accountType="UNIFIED")
                    elif test_case.endpoint == "get_orderbook":
                        await self.client.get_orderbook(symbol="BTCUSDT", limit=25)
                    
                    request_timestamps.append(request_start)
                    
                except Exception as e:
                    errors.append(f"Request {i+1} failed: {str(e)}")
                    
                    # Check if it's a rate limit error
                    if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                        result.violations.append(f"Rate limit violation at request {i+1}")
                
                # Check if we've exceeded test duration
                if time.time() - start_time > test_case.test_duration_seconds:
                    break
            
            # Analyze request timing
            if request_timestamps:
                total_duration = request_timestamps[-1] - request_timestamps[0]
                request_count = len(request_timestamps)
                
                result.actual_rps = request_count / total_duration if total_duration > 0 else 0
                result.actual_rpm = result.actual_rps * 60
                
                # Check compliance
                rps_compliant = result.actual_rps <= test_case.rate_limit.requests_per_second * 1.1  # 10% tolerance
                rpm_compliant = result.actual_rpm <= test_case.rate_limit.requests_per_minute * 1.1
                
                result.compliance_status = rps_compliant and rpm_compliant and len(result.violations) == 0
                
                # Timing analysis
                if len(request_timestamps) > 1:
                    intervals = [
                        request_timestamps[i+1] - request_timestamps[i] 
                        for i in range(len(request_timestamps)-1)
                    ]
                    
                    result.timing_analysis = {
                        'avg_interval_ms': round(statistics.mean(intervals) * 1000, 2),
                        'min_interval_ms': round(min(intervals) * 1000, 2),
                        'max_interval_ms': round(max(intervals) * 1000, 2),
                        'total_requests': request_count,
                        'total_duration_s': round(total_duration, 2)
                    }
                
                # Generate recommendations
                if not result.compliance_status:
                    if result.actual_rps > test_case.rate_limit.requests_per_second:
                        result.recommendations.append(
                            f"Reduce request rate: actual {result.actual_rps:.1f} RPS > limit {test_case.rate_limit.requests_per_second} RPS"
                        )
                    
                    if result.violations:
                        result.recommendations.append(
                            "Implement exponential backoff for rate limit violations"
                        )
            
        except Exception as e:
            result.violations.append(f"Test execution error: {str(e)}")
        
        return result
    
    async def validate_burst_handling(self) -> Dict[str, Any]:
        """Validate handling of burst requests."""
        result = {
            'burst_supported': False,
            'recovery_time_ms': None,
            'burst_capacity': None,
            'test_results': []
        }
        
        try:
            # Make burst of requests
            burst_size = 20
            burst_start = time.time()
            burst_times = []
            
            for i in range(burst_size):
                request_start = time.time()
                
                try:
                    await self.client.get_server_time()  # Lightweight request
                    burst_times.append(time.time() - request_start)
                except Exception as e:
                    result['test_results'].append(f"Burst request {i+1} failed: {str(e)}")
            
            total_burst_time = time.time() - burst_start
            
            if burst_times:
                avg_response_time = statistics.mean(burst_times) * 1000
                result['recovery_time_ms'] = round(avg_response_time, 2)
                result['burst_capacity'] = len(burst_times)
                result['burst_supported'] = avg_response_time < 1000  # Under 1 second average
                
        except Exception as e:
            result['test_results'].append(f"Burst test error: {str(e)}")
        
        return result
    
    def generate_rate_limit_report(self, 
                                   implementation_results: Dict[str, Any],
                                   endpoint_results: Dict[str, RateLimitResult],
                                   burst_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive rate limiting report."""
        
        # Count successful endpoint tests
        endpoint_passes = sum(1 for result in endpoint_results.values() if result.compliance_status)
        total_endpoint_tests = len(endpoint_results)
        
        overall_compliance = (
            implementation_results.get('rate_limiter_present', False) and
            implementation_results.get('enforcement_working', False) and
            endpoint_passes == total_endpoint_tests
        )
        
        return {
            'overall_compliance': overall_compliance,
            'summary': {
                'rate_limiter_implemented': implementation_results.get('rate_limiter_present', False),
                'enforcement_working': implementation_results.get('enforcement_working', False),
                'endpoint_tests_passed': f"{endpoint_passes}/{total_endpoint_tests}",
                'burst_handling_supported': burst_results.get('burst_supported', False)
            },
            'implementation_details': implementation_results,
            'endpoint_test_results': {
                name: {
                    'compliance_status': result.compliance_status,
                    'actual_rps': result.actual_rps,
                    'expected_rps': result.expected_rps,
                    'violations': result.violations,
                    'recommendations': result.recommendations
                }
                for name, result in endpoint_results.items()
            },
            'burst_test_results': burst_results,
            'recommendations': self._generate_rate_limit_recommendations(
                implementation_results, endpoint_results, burst_results
            )
        }
    
    def _generate_rate_limit_recommendations(self,
                                           implementation_results: Dict[str, Any],
                                           endpoint_results: Dict[str, RateLimitResult], 
                                           burst_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate rate limiting recommendations."""
        recommendations = []
        
        if not implementation_results.get('rate_limiter_present', False):
            recommendations.append({
                'category': 'Implementation',
                'priority': 'HIGH',
                'issue': 'No rate limiter detected',
                'solution': 'Implement rate limiting to comply with Bybit API limits'
            })
        
        if not implementation_results.get('enforcement_working', False):
            recommendations.append({
                'category': 'Enforcement',
                'priority': 'HIGH', 
                'issue': 'Rate limiting not enforced properly',
                'solution': 'Fix rate limiter to actually delay requests when limits are approached'
            })
        
        for name, result in endpoint_results.items():
            if not result.compliance_status:
                recommendations.append({
                    'category': 'Endpoint Compliance',
                    'priority': 'MEDIUM',
                    'issue': f'{name} exceeds rate limits',
                    'solution': f'Reduce request rate to {result.expected_rps} RPS or implement queueing'
                })
        
        if not burst_results.get('burst_supported', False):
            recommendations.append({
                'category': 'Burst Handling',
                'priority': 'LOW',
                'issue': 'Poor burst request handling',
                'solution': 'Implement burst capacity and recovery mechanisms'
            })
        
        return recommendations


# Main execution function
async def run_rate_limit_validation_suite():
    """Run complete rate limiting validation suite."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        config_manager.load_config()
        
        client = BybitClient(config_manager)
        validator = BybitRateLimitValidator(client)
        
        logger.info("Starting Bybit rate limiting validation suite...")
        
        # Run all validation tests
        implementation_results = await validator.validate_rate_limiting_implementation()
        endpoint_results = await validator.run_endpoint_rate_limit_tests()
        burst_results = await validator.validate_burst_handling()
        
        # Generate comprehensive report
        report = validator.generate_rate_limit_report(
            implementation_results, endpoint_results, burst_results
        )
        
        # Display results
        print("\n" + "="*60)
        print("BYBIT RATE LIMITING VALIDATION RESULTS")
        print("="*60)
        
        print(f"Overall Compliance: {'✅ PASSED' if report['overall_compliance'] else '❌ FAILED'}")
        print(f"Rate Limiter Implemented: {'✅' if report['summary']['rate_limiter_implemented'] else '❌'}")
        print(f"Enforcement Working: {'✅' if report['summary']['enforcement_working'] else '❌'}")
        print(f"Endpoint Tests: {report['summary']['endpoint_tests_passed']}")
        print(f"Burst Handling: {'✅' if report['summary']['burst_handling_supported'] else '❌'}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec.get('priority', 'LOW'), "🔵")
                print(f"  {priority_emoji} {rec['category']}: {rec['solution']}")
        
        return report
        
    except Exception as e:
        logger.error(f"Rate limiting validation suite failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_rate_limit_validation_suite())