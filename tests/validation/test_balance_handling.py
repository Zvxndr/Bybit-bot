"""
Bybit Balance Handling Validator

This module validates that wallet balance retrieval and handling 
matches Bybit's official API documentation specifications exactly.

Reference: https://bybit-exchange.github.io/docs/v5/account/wallet-balance
"""

import asyncio
import logging
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.bot.exchange.bybit_client import BybitClient
from src.bot.config_manager import ConfigurationManager


class AccountType(Enum):
    """Bybit account types as per documentation."""
    UNIFIED = "UNIFIED"
    CONTRACT = "CONTRACT" 
    SPOT = "SPOT"
    INVESTMENT = "INVESTMENT"
    OPTION = "OPTION"
    FUND = "FUND"


@dataclass
class BalanceValidationResult:
    """Result of balance validation test."""
    account_type: str
    success: bool
    response_structure_valid: bool
    decimal_precision_correct: bool
    required_fields_present: bool
    sample_data: Optional[Dict] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class BybitBalanceValidator:
    """Validates Bybit balance handling against official documentation."""
    
    # Required fields as per Bybit API documentation
    REQUIRED_RESPONSE_FIELDS = [
        'retCode', 'retMsg', 'result', 'retExtInfo', 'time'
    ]
    
    REQUIRED_RESULT_FIELDS = [
        'list'
    ]
    
    REQUIRED_BALANCE_FIELDS = [
        'totalEquity',
        'totalWalletBalance', 
        'totalAvailableBalance',
        'totalPerpUPL',
        'totalInitialMargin',
        'totalMaintenanceMargin',
        'coin',
        'accountType'
    ]
    
    def __init__(self, client: BybitClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
        
    async def validate_all_account_types(self) -> Dict[str, BalanceValidationResult]:
        """Validate balance handling for all supported account types."""
        results = {}
        
        for account_type in AccountType:
            self.logger.info(f"Validating balance handling for {account_type.value}")
            result = await self._validate_account_balance(account_type.value)
            results[account_type.value] = result
            
        return results
    
    async def _validate_account_balance(self, account_type: str) -> BalanceValidationResult:
        """Validate balance handling for specific account type."""
        result = BalanceValidationResult(
            account_type=account_type,
            success=False,
            response_structure_valid=False,
            decimal_precision_correct=False,
            required_fields_present=False
        )
        
        try:
            # Make API request
            response = await self.client.get_wallet_balance(
                accountType=account_type
            )
            
            # Validate response structure
            if not self._validate_response_structure(response, result):
                return result
            
            # Extract balance data
            balance_list = response['result']['list']
            if not balance_list:
                result.errors.append("Empty balance list returned")
                return result
            
            # Validate each balance entry
            for balance_data in balance_list:
                if not self._validate_balance_fields(balance_data, result):
                    continue
                    
                if not self._validate_decimal_precision(balance_data, result):
                    continue
                    
                # Store sample data for inspection
                if not result.sample_data:
                    result.sample_data = self._extract_sample_data(balance_data)
            
            result.success = (
                result.response_structure_valid and
                result.required_fields_present and 
                result.decimal_precision_correct
            )
            
        except Exception as e:
            result.errors.append(f"Exception during validation: {str(e)}")
            self.logger.error(f"Balance validation failed for {account_type}: {str(e)}")
            
        return result
    
    def _validate_response_structure(self, response: Dict, result: BalanceValidationResult) -> bool:
        """Validate the overall response structure matches documentation."""
        
        # Check top-level required fields
        missing_fields = [
            field for field in self.REQUIRED_RESPONSE_FIELDS 
            if field not in response
        ]
        
        if missing_fields:
            result.errors.append(f"Missing required response fields: {missing_fields}")
            return False
        
        # Check retCode indicates success
        if response.get('retCode') != 0:
            result.errors.append(f"API returned error: {response.get('retMsg', 'Unknown error')}")
            return False
        
        # Check result structure
        result_data = response.get('result', {})
        missing_result_fields = [
            field for field in self.REQUIRED_RESULT_FIELDS
            if field not in result_data
        ]
        
        if missing_result_fields:
            result.errors.append(f"Missing required result fields: {missing_result_fields}")
            return False
        
        result.response_structure_valid = True
        return True
    
    def _validate_balance_fields(self, balance_data: Dict, result: BalanceValidationResult) -> bool:
        """Validate balance data contains all required fields."""
        
        missing_fields = [
            field for field in self.REQUIRED_BALANCE_FIELDS
            if field not in balance_data
        ]
        
        if missing_fields:
            result.errors.append(f"Missing required balance fields: {missing_fields}")
            return False
        
        result.required_fields_present = True
        return True
    
    def _validate_decimal_precision(self, balance_data: Dict, result: BalanceValidationResult) -> bool:
        """Validate decimal precision handling for financial values."""
        
        decimal_fields = [
            'totalEquity', 'totalWalletBalance', 'totalAvailableBalance',
            'totalPerpUPL', 'totalInitialMargin', 'totalMaintenanceMargin'
        ]
        
        precision_errors = []
        
        for field in decimal_fields:
            value = balance_data.get(field)
            if value is None:
                continue
                
            try:
                # Attempt to convert to Decimal for precise handling
                decimal_value = Decimal(str(value))
                
                # Check for reasonable precision (max 8 decimal places for crypto)
                if decimal_value.as_tuple().exponent < -8:
                    precision_errors.append(f"{field} has excessive precision: {value}")
                    
                # Verify no floating point precision issues
                if float(decimal_value) != float(value):
                    precision_errors.append(f"{field} has floating point precision issues")
                    
            except (InvalidOperation, ValueError) as e:
                precision_errors.append(f"{field} cannot be converted to Decimal: {value} ({str(e)})")
        
        if precision_errors:
            result.errors.extend(precision_errors)
            return False
        
        result.decimal_precision_correct = True
        return True
    
    def _extract_sample_data(self, balance_data: Dict) -> Dict:
        """Extract sample data for manual inspection."""
        return {
            'account_type': balance_data.get('accountType'),
            'total_equity': balance_data.get('totalEquity'),
            'wallet_balance': balance_data.get('totalWalletBalance'),
            'available_balance': balance_data.get('totalAvailableBalance'),
            'unrealized_pnl': balance_data.get('totalPerpUPL'),
            'initial_margin': balance_data.get('totalInitialMargin'),
            'maintenance_margin': balance_data.get('totalMaintenanceMargin'),
            'coin_count': len(balance_data.get('coin', []))
        }
    
    async def validate_specific_coin_balance(self, account_type: str, coin: str) -> Dict[str, Any]:
        """Validate balance retrieval for specific coin."""
        try:
            response = await self.client.get_wallet_balance(
                accountType=account_type,
                coin=coin
            )
            
            if response.get('retCode') != 0:
                return {
                    'success': False,
                    'error': response.get('retMsg', 'Unknown error')
                }
            
            balance_list = response['result']['list']
            if not balance_list:
                return {
                    'success': False,
                    'error': 'No balance data returned for specified coin'
                }
            
            # Find coin-specific data
            coin_data = None
            for balance in balance_list:
                coin_list = balance.get('coin', [])
                for coin_info in coin_list:
                    if coin_info.get('coin') == coin:
                        coin_data = coin_info
                        break
                if coin_data:
                    break
            
            if not coin_data:
                return {
                    'success': False,
                    'error': f'No data found for coin {coin}'
                }
            
            # Validate coin-specific fields
            required_coin_fields = [
                'coin', 'equity', 'walletBalance', 'availableBalance'
            ]
            
            missing_fields = [
                field for field in required_coin_fields
                if field not in coin_data
            ]
            
            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing coin fields: {missing_fields}'
                }
            
            return {
                'success': True,
                'coin_data': coin_data,
                'decimal_validation': self._validate_coin_decimal_precision(coin_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Exception: {str(e)}'
            }
    
    def _validate_coin_decimal_precision(self, coin_data: Dict) -> Dict[str, Any]:
        """Validate decimal precision for coin-specific data."""
        decimal_fields = ['equity', 'walletBalance', 'availableBalance', 'locked']
        validation_results = {}
        
        for field in decimal_fields:
            value = coin_data.get(field)
            if value is None:
                validation_results[field] = "missing"
                continue
            
            try:
                decimal_value = Decimal(str(value))
                validation_results[field] = {
                    'original': value,
                    'decimal': str(decimal_value),
                    'precision_ok': decimal_value.as_tuple().exponent >= -8
                }
            except Exception as e:
                validation_results[field] = f"error: {str(e)}"
        
        return validation_results
    
    def generate_balance_validation_report(self, results: Dict[str, BalanceValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive balance validation report."""
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.success)
        
        return {
            'summary': {
                'total_account_types': total_tests,
                'successful_validations': passed_tests,
                'success_rate': round((passed_tests / total_tests) * 100, 1) if total_tests > 0 else 0
            },
            'detailed_results': {
                account_type: {
                    'success': result.success,
                    'response_structure_valid': result.response_structure_valid,
                    'required_fields_present': result.required_fields_present,
                    'decimal_precision_correct': result.decimal_precision_correct,
                    'sample_data': result.sample_data,
                    'errors': result.errors
                }
                for account_type, result in results.items()
            },
            'recommendations': self._generate_balance_recommendations(results)
        }
    
    def _generate_balance_recommendations(self, results: Dict[str, BalanceValidationResult]) -> List[Dict[str, str]]:
        """Generate recommendations based on balance validation results."""
        recommendations = []
        
        failed_results = {k: v for k, v in results.items() if not v.success}
        
        for account_type, result in failed_results.items():
            if not result.response_structure_valid:
                recommendations.append({
                    'category': 'Response Structure',
                    'account_type': account_type,
                    'issue': 'API response structure does not match documentation',
                    'solution': 'Update response parsing to handle current Bybit API structure'
                })
            
            if not result.required_fields_present:
                recommendations.append({
                    'category': 'Required Fields',
                    'account_type': account_type,
                    'issue': 'Missing required balance fields in response',
                    'solution': 'Update field mapping to include all documented fields'
                })
            
            if not result.decimal_precision_correct:
                recommendations.append({
                    'category': 'Decimal Precision',
                    'account_type': account_type,
                    'issue': 'Decimal precision handling issues detected',
                    'solution': 'Use Decimal type for all financial calculations'
                })
        
        return recommendations


# Test integration
async def run_balance_validation_suite():
    """Run complete balance validation suite."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        config_manager.load_config()
        
        client = BybitClient(config_manager)
        validator = BybitBalanceValidator(client)
        
        logger.info("Starting Bybit balance validation suite...")
        
        # Run validation for all account types
        results = await validator.validate_all_account_types()
        
        # Test specific coin balance (USDT)
        usdt_result = await validator.validate_specific_coin_balance("UNIFIED", "USDT")
        
        # Generate report
        report = validator.generate_balance_validation_report(results)
        
        # Display results
        print("\n" + "="*60)
        print("BYBIT BALANCE VALIDATION RESULTS")
        print("="*60)
        
        print(f"Overall Success Rate: {report['summary']['success_rate']}%")
        print(f"Successful Account Types: {report['summary']['successful_validations']}/{report['summary']['total_account_types']}")
        
        print("\nAccount Type Results:")
        for account_type, result in report['detailed_results'].items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"  {account_type}: {status}")
            
            if result['errors']:
                for error in result['errors']:
                    print(f"    ⚠️  {error}")
        
        print(f"\nUSDT Balance Test: {'✅ PASSED' if usdt_result['success'] else '❌ FAILED'}")
        if not usdt_result['success']:
            print(f"    ⚠️  {usdt_result['error']}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec['category']} ({rec['account_type']}): {rec['solution']}")
        
        return report
        
    except Exception as e:
        logger.error(f"Balance validation suite failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_balance_validation_suite())