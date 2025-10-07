"""
Advanced Risk Analytics Engine.
Provides comprehensive risk analysis including VaR, stress testing, and scenario analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class VaRMethod(Enum):
    """Value at Risk calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    FILTERED_HISTORICAL = "filtered_historical"
    EXTREME_VALUE = "extreme_value"

class StressTestType(Enum):
    """Types of stress tests."""
    HISTORICAL_SCENARIO = "historical_scenario"
    MONTE_CARLO_SCENARIO = "monte_carlo_scenario"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    WORST_CASE = "worst_case"
    TAIL_RISK = "tail_risk"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    skewness: float
    kurtosis: float
    downside_deviation: float
    beta: Optional[float]
    tracking_error: Optional[float]
    information_ratio: Optional[float]
    timestamp: datetime

@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    confidence_level: float
    var_value: float
    expected_shortfall: float
    method: VaRMethod
    horizon_days: int
    portfolio_value: float
    var_dollar: float
    expected_shortfall_dollar: float
    calculation_date: datetime
    model_parameters: Dict[str, Any]
    backtesting_results: Optional[Dict[str, float]]

@dataclass
class StressTestResult:
    """Stress test analysis result."""
    stress_type: StressTestType
    scenario_name: str
    base_portfolio_value: float
    stressed_portfolio_value: float
    loss_amount: float
    loss_percentage: float
    probability: Optional[float]
    stress_factors: Dict[str, float]
    affected_positions: List[str]
    recovery_time_estimate: Optional[int]
    metadata: Dict[str, Any]

class VaRCalculator:
    """Advanced Value at Risk calculator with multiple methodologies."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # VaR configuration
        self.var_config = {
            'confidence_levels': [0.95, 0.99, 0.999],
            'horizon_days': 1,
            'min_observations': 100,
            'monte_carlo_simulations': 10000,
            'garch_params': {'p': 1, 'q': 1},
            'ewma_lambda': 0.94,
            'bootstrap_samples': 1000
        }
        
        self.logger.info("VaRCalculator initialized")
    
    async def calculate_var(self, 
                          returns: pd.Series,
                          confidence_level: float = 0.95,
                          method: VaRMethod = VaRMethod.HISTORICAL,
                          horizon_days: int = 1,
                          portfolio_value: float = 1000000) -> VaRResult:
        """Calculate Value at Risk using specified method."""
        try:
            if len(returns) < self.var_config['min_observations']:
                raise ValueError(f"Insufficient data: need at least {self.var_config['min_observations']} observations")
            
            # Calculate VaR based on method
            if method == VaRMethod.HISTORICAL:
                var_result = await self._historical_var(returns, confidence_level, horizon_days)
            elif method == VaRMethod.PARAMETRIC:
                var_result = await self._parametric_var(returns, confidence_level, horizon_days)
            elif method == VaRMethod.MONTE_CARLO:
                var_result = await self._monte_carlo_var(returns, confidence_level, horizon_days)
            elif method == VaRMethod.FILTERED_HISTORICAL:
                var_result = await self._filtered_historical_var(returns, confidence_level, horizon_days)
            else:
                raise ValueError(f"Unsupported VaR method: {method}")
            
            var_value, expected_shortfall, model_params = var_result
            
            # Convert to dollar amounts
            var_dollar = abs(var_value) * portfolio_value
            es_dollar = abs(expected_shortfall) * portfolio_value
            
            # Perform backtesting
            backtesting = await self._backtest_var(returns, var_value, confidence_level)
            
            result = VaRResult(
                confidence_level=confidence_level,
                var_value=var_value,
                expected_shortfall=expected_shortfall,
                method=method,
                horizon_days=horizon_days,
                portfolio_value=portfolio_value,
                var_dollar=var_dollar,
                expected_shortfall_dollar=es_dollar,
                calculation_date=datetime.now(),
                model_parameters=model_params,
                backtesting_results=backtesting
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            raise
    
    async def _historical_var(self, returns: pd.Series, confidence_level: float, 
                            horizon_days: int) -> Tuple[float, float, Dict]:
        """Calculate historical simulation VaR."""
        # Sort returns
        sorted_returns = returns.sort_values()
        
        # Calculate percentile position
        percentile_pos = int((1 - confidence_level) * len(sorted_returns))
        
        # VaR is the percentile return
        var_value = sorted_returns.iloc[percentile_pos]
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = sorted_returns.iloc[:percentile_pos]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_value
        
        # Scale for horizon
        if horizon_days > 1:
            var_value *= np.sqrt(horizon_days)
            expected_shortfall *= np.sqrt(horizon_days)
        
        model_params = {
            'observations': len(returns),
            'percentile_position': percentile_pos,
            'horizon_scaling': np.sqrt(horizon_days)
        }
        
        return var_value, expected_shortfall, model_params
    
    async def _parametric_var(self, returns: pd.Series, confidence_level: float,
                            horizon_days: int) -> Tuple[float, float, Dict]:
        """Calculate parametric (normal distribution) VaR."""
        # Calculate mean and standard deviation
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation
        var_value = mean_return + z_score * std_return
        
        # Expected Shortfall for normal distribution
        phi_z = stats.norm.pdf(z_score)
        expected_shortfall = mean_return - std_return * phi_z / (1 - confidence_level)
        
        # Scale for horizon
        if horizon_days > 1:
            var_value = mean_return * horizon_days + z_score * std_return * np.sqrt(horizon_days)
            expected_shortfall = mean_return * horizon_days - std_return * np.sqrt(horizon_days) * phi_z / (1 - confidence_level)
        
        model_params = {
            'mean_return': mean_return,
            'std_return': std_return,
            'z_score': z_score,
            'distribution': 'normal'
        }
        
        return var_value, expected_shortfall, model_params
    
    async def _monte_carlo_var(self, returns: pd.Series, confidence_level: float,
                             horizon_days: int) -> Tuple[float, float, Dict]:
        """Calculate Monte Carlo simulation VaR."""
        # Estimate parameters
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate Monte Carlo simulations
        n_simulations = self.var_config['monte_carlo_simulations']
        
        # Generate random returns
        if horizon_days == 1:
            simulated_returns = np.random.normal(
                mean_return, std_return, n_simulations
            )
        else:
            # Multi-period simulation
            daily_sims = np.random.normal(
                mean_return, std_return, (n_simulations, horizon_days)
            )
            simulated_returns = (1 + daily_sims).prod(axis=1) - 1
        
        # Calculate VaR and ES
        sorted_sims = np.sort(simulated_returns)
        percentile_pos = int((1 - confidence_level) * n_simulations)
        
        var_value = sorted_sims[percentile_pos]
        expected_shortfall = sorted_sims[:percentile_pos].mean()
        
        model_params = {
            'simulations': n_simulations,
            'mean_return': mean_return,
            'std_return': std_return,
            'horizon_days': horizon_days
        }
        
        return var_value, expected_shortfall, model_params
    
    async def _filtered_historical_var(self, returns: pd.Series, confidence_level: float,
                                     horizon_days: int) -> Tuple[float, float, Dict]:
        """Calculate VaR using EWMA filtered historical simulation."""
        # Calculate EWMA volatility
        lambda_param = self.var_config['ewma_lambda']
        
        # Initialize EWMA variance
        ewma_var = returns.var()
        volatilities = [np.sqrt(ewma_var)]
        
        # Calculate EWMA volatilities
        for i in range(1, len(returns)):
            ewma_var = lambda_param * ewma_var + (1 - lambda_param) * returns.iloc[i-1]**2
            volatilities.append(np.sqrt(ewma_var))
        
        # Current volatility
        current_vol = volatilities[-1]
        
        # Scale historical returns by volatility ratio
        historical_vol = returns.std()
        vol_scaling = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        scaled_returns = returns * vol_scaling
        
        # Calculate historical VaR on scaled returns
        var_value, expected_shortfall, _ = await self._historical_var(
            scaled_returns, confidence_level, horizon_days
        )
        
        model_params = {
            'ewma_lambda': lambda_param,
            'current_volatility': current_vol,
            'historical_volatility': historical_vol,
            'vol_scaling_factor': vol_scaling
        }
        
        return var_value, expected_shortfall, model_params
    
    async def _backtest_var(self, returns: pd.Series, var_value: float, 
                          confidence_level: float) -> Dict[str, float]:
        """Backtest VaR model performance."""
        try:
            # Count violations (returns worse than VaR)
            violations = (returns < var_value).sum()
            total_observations = len(returns)
            
            # Violation rate
            violation_rate = violations / total_observations
            expected_rate = 1 - confidence_level
            
            # Kupiec test (likelihood ratio test)
            if violations > 0 and violations < total_observations:
                lr_stat = -2 * np.log(
                    (expected_rate**violations * (1-expected_rate)**(total_observations-violations)) /
                    (violation_rate**violations * (1-violation_rate)**(total_observations-violations))
                )
                p_value = 1 - stats.chi2.cdf(lr_stat, 1)
            else:
                lr_stat = np.inf
                p_value = 0.0
            
            # Average violation magnitude
            violation_returns = returns[returns < var_value]
            avg_violation = violation_returns.mean() if len(violation_returns) > 0 else 0.0
            
            return {
                'violations': int(violations),
                'violation_rate': violation_rate,
                'expected_rate': expected_rate,
                'kupiec_lr_stat': lr_stat,
                'kupiec_p_value': p_value,
                'avg_violation_magnitude': avg_violation,
                'model_accuracy': 1 - abs(violation_rate - expected_rate)
            }
            
        except Exception as e:
            self.logger.error(f"VaR backtesting failed: {e}")
            return {}

class StressTester:
    """Advanced stress testing and scenario analysis."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Stress test scenarios
        self.historical_scenarios = {
            'covid_crash_2020': {
                'start_date': '2020-02-19',
                'end_date': '2020-03-23',
                'description': 'COVID-19 market crash',
                'severity': 'extreme'
            },
            'crypto_winter_2018': {
                'start_date': '2017-12-15',
                'end_date': '2018-12-15',
                'description': 'Cryptocurrency winter 2018',
                'severity': 'severe'
            },
            'may_2021_crypto_crash': {
                'start_date': '2021-05-12',
                'end_date': '2021-05-23',
                'description': 'May 2021 crypto market crash',
                'severity': 'moderate'
            }
        }
        
        self.logger.info("StressTester initialized")
    
    async def run_stress_test(self, 
                            portfolio_returns: pd.Series,
                            portfolio_value: float,
                            stress_type: StressTestType,
                            scenario_params: Dict[str, Any] = None) -> StressTestResult:
        """Run comprehensive stress test."""
        try:
            if stress_type == StressTestType.HISTORICAL_SCENARIO:
                result = await self._historical_scenario_test(
                    portfolio_returns, portfolio_value, scenario_params
                )
            elif stress_type == StressTestType.MONTE_CARLO_SCENARIO:
                result = await self._monte_carlo_stress_test(
                    portfolio_returns, portfolio_value, scenario_params
                )
            elif stress_type == StressTestType.SENSITIVITY_ANALYSIS:
                result = await self._sensitivity_analysis(
                    portfolio_returns, portfolio_value, scenario_params
                )
            elif stress_type == StressTestType.WORST_CASE:
                result = await self._worst_case_scenario(
                    portfolio_returns, portfolio_value
                )
            elif stress_type == StressTestType.TAIL_RISK:
                result = await self._tail_risk_analysis(
                    portfolio_returns, portfolio_value, scenario_params
                )
            else:
                raise ValueError(f"Unsupported stress test type: {stress_type}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            raise
    
    async def _historical_scenario_test(self, returns: pd.Series, portfolio_value: float,
                                      scenario_params: Dict[str, Any]) -> StressTestResult:
        """Apply historical crisis scenarios."""
        scenario_name = scenario_params.get('scenario', 'covid_crash_2020')
        
        if scenario_name not in self.historical_scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.historical_scenarios[scenario_name]
        
        # Get worst consecutive returns from the scenario period
        # For demonstration, we'll simulate the scenario impact
        if scenario_name == 'covid_crash_2020':
            stress_factor = -0.35  # 35% drop
        elif scenario_name == 'crypto_winter_2018':
            stress_factor = -0.80  # 80% drop
        elif scenario_name == 'may_2021_crypto_crash':
            stress_factor = -0.50  # 50% drop
        else:
            stress_factor = -0.30  # Default 30% drop
        
        # Calculate stressed portfolio value
        stressed_value = portfolio_value * (1 + stress_factor)
        loss_amount = portfolio_value - stressed_value
        loss_percentage = stress_factor * 100
        
        # Estimate recovery time based on historical data
        recovery_times = {
            'covid_crash_2020': 180,  # days
            'crypto_winter_2018': 365,
            'may_2021_crypto_crash': 90
        }
        
        return StressTestResult(
            stress_type=StressTestType.HISTORICAL_SCENARIO,
            scenario_name=scenario_name,
            base_portfolio_value=portfolio_value,
            stressed_portfolio_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            probability=scenario_params.get('probability'),
            stress_factors={'market_shock': stress_factor},
            affected_positions=['all'],
            recovery_time_estimate=recovery_times.get(scenario_name, 180),
            metadata={
                'scenario_description': scenario['description'],
                'severity': scenario['severity']
            }
        )
    
    async def _monte_carlo_stress_test(self, returns: pd.Series, portfolio_value: float,
                                     scenario_params: Dict[str, Any]) -> StressTestResult:
        """Monte Carlo simulation for extreme scenarios."""
        n_simulations = scenario_params.get('simulations', 10000)
        confidence_level = scenario_params.get('confidence_level', 0.01)  # 1% worst case
        
        # Generate extreme scenarios
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Use higher volatility for stress scenarios
        stress_vol_multiplier = scenario_params.get('volatility_multiplier', 2.0)
        stress_std = std_return * stress_vol_multiplier
        
        # Generate stressed returns
        stressed_returns = np.random.normal(
            mean_return, stress_std, n_simulations
        )
        
        # Find worst-case scenario
        worst_percentile = int(confidence_level * n_simulations)
        sorted_returns = np.sort(stressed_returns)
        worst_return = sorted_returns[worst_percentile]
        
        # Calculate impact
        stressed_value = portfolio_value * (1 + worst_return)
        loss_amount = portfolio_value - stressed_value
        loss_percentage = worst_return * 100
        
        return StressTestResult(
            stress_type=StressTestType.MONTE_CARLO_SCENARIO,
            scenario_name="Monte Carlo Stress",
            base_portfolio_value=portfolio_value,
            stressed_portfolio_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            probability=confidence_level,
            stress_factors={
                'volatility_multiplier': stress_vol_multiplier,
                'worst_return': worst_return
            },
            affected_positions=['all'],
            recovery_time_estimate=None,
            metadata={
                'simulations': n_simulations,
                'confidence_level': confidence_level
            }
        )
    
    async def _sensitivity_analysis(self, returns: pd.Series, portfolio_value: float,
                                  scenario_params: Dict[str, Any]) -> StressTestResult:
        """Sensitivity analysis for key risk factors."""
        # Define stress factors
        factors = scenario_params.get('factors', {
            'volatility_shock': 2.0,
            'correlation_shock': 0.5,
            'liquidity_shock': 0.1
        })
        
        # Calculate combined impact
        base_return = returns.mean()
        base_vol = returns.std()
        
        # Apply shocks
        vol_impact = -factors.get('volatility_shock', 1.0) * base_vol
        correlation_impact = -factors.get('correlation_shock', 0.0) * 0.05  # 5% additional loss
        liquidity_impact = -factors.get('liquidity_shock', 0.0) * 0.02  # 2% liquidity discount
        
        total_impact = vol_impact + correlation_impact + liquidity_impact
        
        stressed_value = portfolio_value * (1 + total_impact)
        loss_amount = portfolio_value - stressed_value
        loss_percentage = total_impact * 100
        
        return StressTestResult(
            stress_type=StressTestType.SENSITIVITY_ANALYSIS,
            scenario_name="Multi-Factor Sensitivity",
            base_portfolio_value=portfolio_value,
            stressed_portfolio_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            probability=None,
            stress_factors=factors,
            affected_positions=['all'],
            recovery_time_estimate=None,
            metadata={
                'factor_impacts': {
                    'volatility': vol_impact,
                    'correlation': correlation_impact,
                    'liquidity': liquidity_impact
                }
            }
        )
    
    async def _worst_case_scenario(self, returns: pd.Series, 
                                 portfolio_value: float) -> StressTestResult:
        """Worst-case scenario based on historical data."""
        # Find worst single-day return
        worst_return = returns.min()
        
        # Find worst consecutive period
        rolling_periods = [5, 10, 20]  # days
        worst_period_return = worst_return
        worst_period_days = 1
        
        for period in rolling_periods:
            if len(returns) >= period:
                rolling_returns = returns.rolling(period).sum()
                period_worst = rolling_returns.min()
                if period_worst < worst_period_return:
                    worst_period_return = period_worst
                    worst_period_days = period
        
        # Use worst period return
        stressed_value = portfolio_value * (1 + worst_period_return)
        loss_amount = portfolio_value - stressed_value
        loss_percentage = worst_period_return * 100
        
        return StressTestResult(
            stress_type=StressTestType.WORST_CASE,
            scenario_name="Historical Worst Case",
            base_portfolio_value=portfolio_value,
            stressed_portfolio_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            probability=None,
            stress_factors={'worst_return': worst_period_return},
            affected_positions=['all'],
            recovery_time_estimate=worst_period_days * 10,  # Estimated recovery
            metadata={
                'worst_single_day': worst_return,
                'worst_period_days': worst_period_days,
                'worst_period_return': worst_period_return
            }
        )
    
    async def _tail_risk_analysis(self, returns: pd.Series, portfolio_value: float,
                                scenario_params: Dict[str, Any]) -> StressTestResult:
        """Tail risk analysis using extreme value theory."""
        # Parameters
        tail_percentile = scenario_params.get('tail_percentile', 0.05)  # 5% tail
        
        # Extract tail returns
        tail_threshold = returns.quantile(tail_percentile)
        tail_returns = returns[returns <= tail_threshold]
        
        if len(tail_returns) == 0:
            tail_returns = returns.head(max(1, int(len(returns) * tail_percentile)))
        
        # Fit extreme value distribution (simplified)
        tail_mean = tail_returns.mean()
        tail_std = tail_returns.std()
        
        # Expected tail loss (Expected Shortfall at tail level)
        expected_tail_loss = tail_mean
        
        # Calculate impact
        stressed_value = portfolio_value * (1 + expected_tail_loss)
        loss_amount = portfolio_value - stressed_value
        loss_percentage = expected_tail_loss * 100
        
        return StressTestResult(
            stress_type=StressTestType.TAIL_RISK,
            scenario_name="Tail Risk Analysis",
            base_portfolio_value=portfolio_value,
            stressed_portfolio_value=stressed_value,
            loss_amount=loss_amount,
            loss_percentage=loss_percentage,
            probability=tail_percentile,
            stress_factors={
                'tail_threshold': tail_threshold,
                'expected_tail_loss': expected_tail_loss
            },
            affected_positions=['all'],
            recovery_time_estimate=None,
            metadata={
                'tail_observations': len(tail_returns),
                'tail_mean': tail_mean,
                'tail_std': tail_std
            }
        )

class RiskAnalyzer:
    """Comprehensive risk analysis engine."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        self.var_calculator = VaRCalculator()
        self.stress_tester = StressTester()
        
        self.logger.info("RiskAnalyzer initialized")
    
    async def comprehensive_risk_analysis(self, 
                                        portfolio_returns: pd.Series,
                                        benchmark_returns: pd.Series = None,
                                        portfolio_value: float = 1000000) -> Dict[str, Any]:
        """Perform comprehensive risk analysis."""
        try:
            # Calculate basic risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                portfolio_returns, benchmark_returns
            )
            
            # Calculate VaR using multiple methods
            var_results = {}
            for method in [VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC, VaRMethod.MONTE_CARLO]:
                try:
                    var_result = await self.var_calculator.calculate_var(
                        portfolio_returns, method=method, portfolio_value=portfolio_value
                    )
                    var_results[method.value] = var_result
                except Exception as e:
                    self.logger.warning(f"VaR calculation failed for {method}: {e}")
            
            # Run stress tests
            stress_results = {}
            for stress_type in [StressTestType.HISTORICAL_SCENARIO, 
                              StressTestType.WORST_CASE, 
                              StressTestType.TAIL_RISK]:
                try:
                    stress_result = await self.stress_tester.run_stress_test(
                        portfolio_returns, portfolio_value, stress_type
                    )
                    stress_results[stress_type.value] = stress_result
                except Exception as e:
                    self.logger.warning(f"Stress test failed for {stress_type}: {e}")
            
            return {
                'risk_metrics': risk_metrics,
                'var_analysis': var_results,
                'stress_tests': stress_results,
                'analysis_timestamp': datetime.now(),
                'portfolio_value': portfolio_value
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive risk analysis failed: {e}")
            raise
    
    async def _calculate_risk_metrics(self, returns: pd.Series, 
                                    benchmark_returns: pd.Series = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            # Basic statistics
            mean_return = returns.mean()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # VaR and Expected Shortfall
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            tail_95 = returns[returns <= var_95]
            tail_99 = returns[returns <= var_99]
            
            es_95 = tail_95.mean() if len(tail_95) > 0 else var_95
            es_99 = tail_99.mean() if len(tail_99) > 0 else var_99
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe Ratio (assuming risk-free rate from config)
            risk_free_rate = self.config.get('risk_free_rate', 0.02) / 252  # Daily
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            # Sortino Ratio
            downside_returns = returns[returns < risk_free_rate]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (mean_return * 252 - self.config.get('risk_free_rate', 0.02)) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar Ratio
            calmar_ratio = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Higher moments
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Benchmark-relative metrics
            beta = None
            tracking_error = None
            information_ratio = None
            
            if benchmark_returns is not None:
                # Align data
                common_index = returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 10:
                    ret_aligned = returns.loc[common_index]
                    bench_aligned = benchmark_returns.loc[common_index]
                    
                    # Beta
                    covariance = np.cov(ret_aligned, bench_aligned)[0, 1]
                    benchmark_variance = np.var(bench_aligned)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else None
                    
                    # Tracking Error
                    active_returns = ret_aligned - bench_aligned
                    tracking_error = active_returns.std() * np.sqrt(252)
                    
                    # Information Ratio
                    information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                skewness=skewness,
                kurtosis=kurtosis,
                downside_deviation=downside_deviation,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            raise