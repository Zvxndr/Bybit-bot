"""
Stress testing framework for model validation and risk assessment.

This module provides comprehensive stress testing capabilities including:
- Historical scenario replays
- Hypothetical stress scenarios
- Extreme value analysis
- Tail risk assessment
- Multi-factor stress tests
- Regime change simulations
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class StressType(Enum):
    """Types of stress tests."""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    EXTREME_VALUE = "extreme_value"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    REGIME_CHANGE = "regime_change"
    TAIL_RISK = "tail_risk"
    MULTI_FACTOR = "multi_factor"

class Severity(Enum):
    """Stress test severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"

@dataclass
class StressScenario:
    """Definition of a stress test scenario."""
    name: str
    stress_type: StressType
    severity: Severity
    description: str
    
    # Scenario parameters
    market_shock: float = 0.0  # Market return shock
    volatility_multiplier: float = 1.0  # Volatility scaling
    correlation_adjustment: float = 1.0  # Correlation scaling
    liquidity_impact: float = 0.0  # Liquidity penalty
    
    # Factor-specific shocks
    factor_shocks: Dict[str, float] = field(default_factory=dict)
    
    # Temporal parameters
    duration_days: int = 1  # Scenario duration
    probability: float = 0.01  # Annual probability
    
    # Advanced parameters
    regime_shift: bool = False
    contagion_effects: bool = False
    tail_dependency: float = 0.0
    
    # Metadata
    historical_reference: Optional[str] = None
    data_source: Optional[str] = None

@dataclass
class StressResult:
    """Results from a single stress test."""
    scenario_name: str
    stress_type: StressType
    severity: Severity
    
    # Performance metrics
    stressed_return: float
    unstressed_return: float
    return_impact: float
    
    # Risk metrics
    stressed_var: float
    unstressed_var: float
    var_impact: float
    
    stressed_cvar: float
    unstressed_cvar: float
    cvar_impact: float
    
    # Drawdown analysis
    max_drawdown: float
    time_to_recovery: int
    underwater_periods: int
    
    # Portfolio-specific metrics
    portfolio_value: float
    portfolio_impact: float
    position_impacts: Dict[str, float] = field(default_factory=dict)
    
    # Risk factor contributions
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Statistical measures
    tail_expectation: float
    conditional_drawdown: float
    stress_ratio: float  # Stressed return / unstressed return
    
    # Execution metrics
    computation_time: float
    scenario_probability: float
    confidence_level: float = 0.95

@dataclass
class StressSummary:
    """Summary of all stress test results."""
    test_id: str
    timestamp: datetime
    total_scenarios: int
    
    # Aggregate impacts
    worst_case_return: float
    worst_case_scenario: str
    average_impact: float
    
    # Risk measure impacts
    max_var_impact: float
    max_cvar_impact: float
    average_var_impact: float
    average_cvar_impact: float
    
    # Portfolio resilience
    scenarios_passed: int  # Below loss threshold
    scenarios_failed: int  # Above loss threshold
    resilience_score: float  # Percentage of scenarios passed
    
    # Tail risk analysis
    tail_scenarios: List[str]
    extreme_loss_probability: float
    expected_tail_loss: float
    
    # Factor analysis
    most_impactful_factors: Dict[str, float]
    factor_correlation_breakdown: Dict[str, float]
    
    # Regime analysis
    regime_sensitivity: Dict[str, float]
    contagion_effects: Dict[str, float]
    
    # Model validation
    stress_model_validity: Dict[str, float]
    scenario_realism: Dict[str, float]

class HistoricalScenarioBuilder:
    """Build stress scenarios from historical data."""
    
    def __init__(self):
        self.historical_events = {
            "dot_com_crash": {
                "start_date": "2000-03-10",
                "end_date": "2002-10-09",
                "market_shock": -0.49,
                "volatility_multiplier": 2.5,
                "description": "Dot-com bubble burst"
            },
            "september_11": {
                "start_date": "2001-09-11",
                "end_date": "2001-09-21",
                "market_shock": -0.11,
                "volatility_multiplier": 3.0,
                "description": "September 11 attacks market impact"
            },
            "financial_crisis": {
                "start_date": "2007-10-09",
                "end_date": "2009-03-09",
                "market_shock": -0.57,
                "volatility_multiplier": 2.8,
                "description": "2008 Financial Crisis"
            },
            "flash_crash": {
                "start_date": "2010-05-06",
                "end_date": "2010-05-06",
                "market_shock": -0.09,
                "volatility_multiplier": 5.0,
                "description": "Flash Crash of 2010"
            },
            "covid_crash": {
                "start_date": "2020-02-19",
                "end_date": "2020-03-23",
                "market_shock": -0.34,
                "volatility_multiplier": 4.0,
                "description": "COVID-19 market crash"
            },
            "crypto_winter": {
                "start_date": "2022-01-01",
                "end_date": "2022-12-31",
                "market_shock": -0.65,
                "volatility_multiplier": 2.2,
                "description": "Crypto winter 2022"
            }
        }
    
    def build_historical_scenario(self, event_name: str, 
                                severity: Severity = Severity.SEVERE) -> StressScenario:
        """Build stress scenario from historical event."""
        if event_name not in self.historical_events:
            raise ValueError(f"Unknown historical event: {event_name}")
        
        event_data = self.historical_events[event_name]
        
        # Adjust severity
        severity_multipliers = {
            Severity.MILD: 0.5,
            Severity.MODERATE: 0.75,
            Severity.SEVERE: 1.0,
            Severity.EXTREME: 1.5
        }
        
        multiplier = severity_multipliers[severity]
        
        scenario = StressScenario(
            name=f"{event_name}_{severity.value}",
            stress_type=StressType.HISTORICAL,
            severity=severity,
            description=f"{event_data['description']} ({severity.value} impact)",
            market_shock=event_data["market_shock"] * multiplier,
            volatility_multiplier=event_data["volatility_multiplier"] * multiplier,
            historical_reference=event_name,
            probability=self._estimate_probability(event_data["market_shock"])
        )
        
        return scenario
    
    def _estimate_probability(self, market_shock: float) -> float:
        """Estimate annual probability based on shock magnitude."""
        # Rough estimation based on historical frequency
        shock_magnitude = abs(market_shock)
        
        if shock_magnitude < 0.1:
            return 0.1  # 10% annual probability
        elif shock_magnitude < 0.2:
            return 0.05  # 5% annual probability
        elif shock_magnitude < 0.3:
            return 0.02  # 2% annual probability
        elif shock_magnitude < 0.4:
            return 0.01  # 1% annual probability
        else:
            return 0.005  # 0.5% annual probability
    
    def build_all_historical_scenarios(self) -> List[StressScenario]:
        """Build all available historical scenarios."""
        scenarios = []
        
        for event_name in self.historical_events.keys():
            for severity in [Severity.MODERATE, Severity.SEVERE, Severity.EXTREME]:
                scenario = self.build_historical_scenario(event_name, severity)
                scenarios.append(scenario)
        
        return scenarios

class HypotheticalScenarioBuilder:
    """Build hypothetical stress scenarios."""
    
    def __init__(self):
        self.scenario_templates = {
            "interest_rate_shock": {
                "description": "Sudden interest rate changes",
                "base_factors": ["interest_rate", "credit_spread", "currency"]
            },
            "commodity_crisis": {
                "description": "Commodity price shock",
                "base_factors": ["commodity", "inflation", "currency"]
            },
            "geopolitical_crisis": {
                "description": "Geopolitical instability",
                "base_factors": ["equity", "currency", "commodity", "volatility"]
            },
            "tech_disruption": {
                "description": "Technology sector disruption",
                "base_factors": ["tech_equity", "growth_premium", "volatility"]
            },
            "liquidity_freeze": {
                "description": "Market liquidity crisis",
                "base_factors": ["liquidity", "credit_spread", "volatility"]
            }
        }
    
    def build_hypothetical_scenario(self, scenario_type: str,
                                  severity: Severity,
                                  custom_factors: Optional[Dict[str, float]] = None) -> StressScenario:
        """Build hypothetical stress scenario."""
        if scenario_type not in self.scenario_templates:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        template = self.scenario_templates[scenario_type]
        
        # Severity-based parameter scaling
        severity_params = {
            Severity.MILD: {
                "market_shock": -0.05,
                "volatility_multiplier": 1.5,
                "correlation_adjustment": 1.2
            },
            Severity.MODERATE: {
                "market_shock": -0.10,
                "volatility_multiplier": 2.0,
                "correlation_adjustment": 1.4
            },
            Severity.SEVERE: {
                "market_shock": -0.20,
                "volatility_multiplier": 2.5,
                "correlation_adjustment": 1.6
            },
            Severity.EXTREME: {
                "market_shock": -0.35,
                "volatility_multiplier": 3.5,
                "correlation_adjustment": 2.0
            }
        }
        
        params = severity_params[severity]
        
        # Build factor shocks
        factor_shocks = custom_factors or {}
        if not factor_shocks:
            # Default factor shocks based on scenario type
            base_shock = abs(params["market_shock"]) * 0.5
            for factor in template["base_factors"]:
                factor_shocks[factor] = -base_shock * np.random.uniform(0.5, 1.5)
        
        scenario = StressScenario(
            name=f"{scenario_type}_{severity.value}",
            stress_type=StressType.HYPOTHETICAL,
            severity=severity,
            description=f"{template['description']} ({severity.value} impact)",
            market_shock=params["market_shock"],
            volatility_multiplier=params["volatility_multiplier"],
            correlation_adjustment=params["correlation_adjustment"],
            factor_shocks=factor_shocks,
            probability=self._estimate_hypothetical_probability(severity)
        )
        
        return scenario
    
    def _estimate_hypothetical_probability(self, severity: Severity) -> float:
        """Estimate probability for hypothetical scenarios."""
        probability_map = {
            Severity.MILD: 0.05,      # 5% annual
            Severity.MODERATE: 0.02,  # 2% annual
            Severity.SEVERE: 0.01,    # 1% annual
            Severity.EXTREME: 0.005   # 0.5% annual
        }
        return probability_map[severity]
    
    def build_multi_factor_scenario(self, factor_shocks: Dict[str, float],
                                  severity: Severity) -> StressScenario:
        """Build multi-factor stress scenario."""
        scenario = StressScenario(
            name=f"multi_factor_{severity.value}",
            stress_type=StressType.MULTI_FACTOR,
            severity=severity,
            description=f"Multi-factor stress test ({severity.value} impact)",
            factor_shocks=factor_shocks,
            probability=self._estimate_hypothetical_probability(severity)
        )
        
        # Calculate aggregate market shock
        if factor_shocks:
            scenario.market_shock = sum(factor_shocks.values()) / len(factor_shocks)
            scenario.volatility_multiplier = 1.0 + abs(scenario.market_shock) * 5
        
        return scenario

class TailRiskAnalyzer:
    """Analyze tail risk and extreme value scenarios."""
    
    def __init__(self):
        self.extreme_quantiles = [0.01, 0.005, 0.001]  # 1%, 0.5%, 0.1%
    
    def build_extreme_value_scenarios(self, returns_data: np.ndarray,
                                    confidence_levels: List[float] = None) -> List[StressScenario]:
        """Build scenarios based on extreme value theory."""
        if confidence_levels is None:
            confidence_levels = [0.99, 0.995, 0.999]
        
        scenarios = []
        
        # Fit extreme value distributions
        # Use block maxima method
        block_size = 21  # Approximate monthly blocks
        block_minima = []
        
        for i in range(0, len(returns_data) - block_size, block_size):
            block = returns_data[i:i + block_size]
            block_minima.append(np.min(block))
        
        if len(block_minima) < 10:
            logger.warning("Insufficient data for extreme value analysis")
            return scenarios
        
        # Fit Generalized Extreme Value distribution
        try:
            gev_params = stats.genextreme.fit(block_minima)
            
            for conf_level in confidence_levels:
                # Calculate extreme quantile
                extreme_quantile = stats.genextreme.ppf(1 - conf_level, *gev_params)
                
                # Determine severity based on quantile
                if extreme_quantile > -0.05:
                    severity = Severity.MILD
                elif extreme_quantile > -0.10:
                    severity = Severity.MODERATE
                elif extreme_quantile > -0.20:
                    severity = Severity.SEVERE
                else:
                    severity = Severity.EXTREME
                
                scenario = StressScenario(
                    name=f"extreme_value_{conf_level:.1%}",
                    stress_type=StressType.EXTREME_VALUE,
                    severity=severity,
                    description=f"Extreme value scenario at {conf_level:.1%} confidence",
                    market_shock=extreme_quantile,
                    volatility_multiplier=2.0 + abs(extreme_quantile) * 10,
                    probability=1 - conf_level
                )
                
                scenarios.append(scenario)
                
        except Exception as e:
            logger.error(f"Error fitting extreme value distribution: {e}")
        
        return scenarios
    
    def build_tail_dependency_scenario(self, returns_data: pd.DataFrame) -> StressScenario:
        """Build scenario accounting for tail dependencies."""
        try:
            # Calculate tail dependency coefficients
            if returns_data.shape[1] < 2:
                raise ValueError("Need at least 2 assets for tail dependency analysis")
            
            # Focus on lower tail (losses)
            lower_quantile = 0.05
            tail_data = returns_data[returns_data.iloc[:, 0] <= returns_data.iloc[:, 0].quantile(lower_quantile)]
            
            if len(tail_data) < 10:
                logger.warning("Insufficient tail data")
                tail_correlation = 0.8
            else:
                tail_correlation = tail_data.corr().iloc[0, 1] if len(tail_data.columns) > 1 else 0.8
            
            # Build scenario with elevated tail correlation
            scenario = StressScenario(
                name="tail_dependency",
                stress_type=StressType.TAIL_RISK,
                severity=Severity.SEVERE,
                description="Tail dependency stress scenario",
                market_shock=-0.15,
                volatility_multiplier=2.5,
                correlation_adjustment=1.8,
                tail_dependency=tail_correlation,
                probability=0.01
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error building tail dependency scenario: {e}")
            return StressScenario(
                name="tail_dependency_fallback",
                stress_type=StressType.TAIL_RISK,
                severity=Severity.MODERATE,
                description="Fallback tail dependency scenario",
                market_shock=-0.10,
                volatility_multiplier=2.0,
                probability=0.02
            )

class StressTester:
    """Main stress testing engine."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "stress_test_results.db"
        self.historical_builder = HistoricalScenarioBuilder()
        self.hypothetical_builder = HypotheticalScenarioBuilder()
        self.tail_analyzer = TailRiskAnalyzer()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database for storing results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Stress test results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stress_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    stress_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    stressed_return REAL,
                    unstressed_return REAL,
                    return_impact REAL,
                    stressed_var REAL,
                    unstressed_var REAL,
                    var_impact REAL,
                    max_drawdown REAL,
                    portfolio_impact REAL,
                    scenario_probability REAL,
                    computation_time REAL,
                    factor_contributions TEXT,
                    position_impacts TEXT
                )
            """)
            
            # Stress test summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stress_test_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    total_scenarios INTEGER,
                    worst_case_return REAL,
                    worst_case_scenario TEXT,
                    resilience_score REAL,
                    max_var_impact REAL,
                    tail_loss_probability REAL,
                    execution_time REAL,
                    scenario_distribution TEXT,
                    risk_factor_analysis TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def run_stress_test(self, portfolio_data: pd.DataFrame,
                       scenarios: Optional[List[StressScenario]] = None,
                       test_id: Optional[str] = None) -> StressSummary:
        """Run comprehensive stress test."""
        start_time = datetime.now()
        test_id = test_id or f"stress_test_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting stress test: {test_id}")
        
        try:
            # Generate scenarios if not provided
            if scenarios is None:
                scenarios = self._generate_default_scenarios(portfolio_data)
            
            # Run individual stress tests
            results = []
            for scenario in scenarios:
                result = self._run_single_stress_test(portfolio_data, scenario)
                results.append(result)
                self._save_stress_result(result, test_id)
            
            # Generate summary
            summary = self._generate_stress_summary(results, test_id)
            
            # Save summary
            self._save_stress_summary(summary)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Stress test completed in {execution_time:.2f}s")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in stress test: {e}")
            raise
    
    def _generate_default_scenarios(self, portfolio_data: pd.DataFrame) -> List[StressScenario]:
        """Generate default set of stress scenarios."""
        scenarios = []
        
        # Historical scenarios
        historical_scenarios = self.historical_builder.build_all_historical_scenarios()
        scenarios.extend(historical_scenarios[:6])  # Limit to avoid too many
        
        # Hypothetical scenarios
        scenario_types = ["interest_rate_shock", "geopolitical_crisis", "liquidity_freeze"]
        for scenario_type in scenario_types:
            for severity in [Severity.MODERATE, Severity.SEVERE]:
                hyp_scenario = self.hypothetical_builder.build_hypothetical_scenario(
                    scenario_type, severity
                )
                scenarios.append(hyp_scenario)
        
        # Extreme value scenarios
        if 'returns' in portfolio_data.columns:
            returns_data = portfolio_data['returns'].dropna().values
            extreme_scenarios = self.tail_analyzer.build_extreme_value_scenarios(returns_data)
            scenarios.extend(extreme_scenarios[:3])  # Top 3 extreme scenarios
        
        # Tail dependency scenario
        if portfolio_data.shape[1] > 1:
            tail_scenario = self.tail_analyzer.build_tail_dependency_scenario(portfolio_data)
            scenarios.append(tail_scenario)
        
        return scenarios
    
    def _run_single_stress_test(self, portfolio_data: pd.DataFrame,
                              scenario: StressScenario) -> StressResult:
        """Run single stress test scenario."""
        start_time = datetime.now()
        
        try:
            # Calculate baseline metrics
            if 'returns' in portfolio_data.columns:
                base_returns = portfolio_data['returns'].dropna()
                unstressed_return = base_returns.mean()
                unstressed_var = base_returns.quantile(0.05)
                unstressed_cvar = base_returns[base_returns <= unstressed_var].mean()
            else:
                # Use first column as proxy
                base_series = portfolio_data.iloc[:, 0].dropna()
                base_returns = base_series.pct_change().dropna()
                unstressed_return = base_returns.mean()
                unstressed_var = base_returns.quantile(0.05)
                unstressed_cvar = base_returns[base_returns <= unstressed_var].mean()
            
            # Apply stress scenario
            stressed_returns = self._apply_stress_scenario(base_returns, scenario)
            
            # Calculate stressed metrics
            stressed_return = stressed_returns.mean()
            stressed_var = stressed_returns.quantile(0.05)
            stressed_cvar = stressed_returns[stressed_returns <= stressed_var].mean()
            
            # Calculate impacts
            return_impact = stressed_return - unstressed_return
            var_impact = stressed_var - unstressed_var
            cvar_impact = stressed_cvar - unstressed_cvar
            
            # Drawdown analysis
            cumulative_returns = (1 + stressed_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Time to recovery (simplified)
            underwater_periods = len(drawdowns[drawdowns < -0.01])
            time_to_recovery = underwater_periods  # Simplified estimate
            
            # Portfolio impact (assuming equal weights)
            portfolio_value = 100.0  # Base value
            portfolio_impact = portfolio_value * return_impact
            
            # Factor contributions (simplified)
            factor_contributions = {}
            if scenario.factor_shocks:
                total_shock = sum(abs(shock) for shock in scenario.factor_shocks.values())
                for factor, shock in scenario.factor_shocks.items():
                    contribution = (abs(shock) / total_shock) * abs(return_impact)
                    factor_contributions[factor] = contribution
            
            # Statistical measures
            tail_expectation = stressed_returns[stressed_returns <= stressed_returns.quantile(0.01)].mean()
            conditional_drawdown = drawdowns[drawdowns <= drawdowns.quantile(0.05)].mean()
            stress_ratio = stressed_return / unstressed_return if unstressed_return != 0 else 0.0
            
            # Create result
            result = StressResult(
                scenario_name=scenario.name,
                stress_type=scenario.stress_type,
                severity=scenario.severity,
                stressed_return=stressed_return,
                unstressed_return=unstressed_return,
                return_impact=return_impact,
                stressed_var=stressed_var,
                unstressed_var=unstressed_var,
                var_impact=var_impact,
                stressed_cvar=stressed_cvar,
                unstressed_cvar=unstressed_cvar,
                cvar_impact=cvar_impact,
                max_drawdown=max_drawdown,
                time_to_recovery=time_to_recovery,
                underwater_periods=underwater_periods,
                portfolio_value=portfolio_value + portfolio_impact,
                portfolio_impact=portfolio_impact,
                factor_contributions=factor_contributions,
                tail_expectation=tail_expectation,
                conditional_drawdown=conditional_drawdown,
                stress_ratio=stress_ratio,
                computation_time=(datetime.now() - start_time).total_seconds(),
                scenario_probability=scenario.probability,
                confidence_level=0.95
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single stress test {scenario.name}: {e}")
            # Return minimal result
            return StressResult(
                scenario_name=scenario.name,
                stress_type=scenario.stress_type,
                severity=scenario.severity,
                stressed_return=0.0,
                unstressed_return=0.0,
                return_impact=0.0,
                stressed_var=0.0,
                unstressed_var=0.0,
                var_impact=0.0,
                stressed_cvar=0.0,
                unstressed_cvar=0.0,
                cvar_impact=0.0,
                max_drawdown=0.0,
                time_to_recovery=0,
                underwater_periods=0,
                portfolio_value=100.0,
                portfolio_impact=0.0,
                tail_expectation=0.0,
                conditional_drawdown=0.0,
                stress_ratio=1.0,
                computation_time=(datetime.now() - start_time).total_seconds(),
                scenario_probability=scenario.probability
            )
    
    def _apply_stress_scenario(self, base_returns: pd.Series,
                             scenario: StressScenario) -> pd.Series:
        """Apply stress scenario to returns."""
        stressed_returns = base_returns.copy()
        
        # Apply market shock
        if scenario.market_shock != 0:
            # Apply as one-time shock to the first return
            if len(stressed_returns) > 0:
                stressed_returns.iloc[0] += scenario.market_shock
        
        # Apply volatility scaling
        if scenario.volatility_multiplier != 1.0:
            mean_return = stressed_returns.mean()
            excess_returns = stressed_returns - mean_return
            stressed_returns = mean_return + excess_returns * scenario.volatility_multiplier
        
        # Apply factor shocks (simplified)
        if scenario.factor_shocks:
            # Apply as distributed shocks
            total_factor_shock = sum(scenario.factor_shocks.values())
            shocked_returns = stressed_returns + total_factor_shock / len(stressed_returns)
            stressed_returns = shocked_returns
        
        # Apply correlation effects (simplified)
        if scenario.correlation_adjustment != 1.0:
            # Increase correlation by reducing idiosyncratic component
            market_component = stressed_returns * 0.7  # Assume 70% market component
            idiosyncratic_component = stressed_returns - market_component
            
            # Reduce idiosyncratic component to increase correlation
            correlation_factor = 1.0 / scenario.correlation_adjustment
            stressed_returns = market_component + idiosyncratic_component * correlation_factor
        
        return stressed_returns
    
    def _generate_stress_summary(self, results: List[StressResult], 
                               test_id: str) -> StressSummary:
        """Generate summary from stress test results."""
        try:
            # Basic statistics
            total_scenarios = len(results)
            successful_results = [r for r in results if r.computation_time > 0]
            
            if not successful_results:
                raise ValueError("No successful stress test results")
            
            # Worst case analysis
            worst_result = min(successful_results, key=lambda x: x.return_impact)
            worst_case_return = worst_result.return_impact
            worst_case_scenario = worst_result.scenario_name
            
            # Average impact
            return_impacts = [r.return_impact for r in successful_results]
            average_impact = np.mean(return_impacts)
            
            # Risk measure impacts
            var_impacts = [r.var_impact for r in successful_results]
            cvar_impacts = [r.cvar_impact for r in successful_results]
            
            max_var_impact = min(var_impacts) if var_impacts else 0.0
            max_cvar_impact = min(cvar_impacts) if cvar_impacts else 0.0
            average_var_impact = np.mean(var_impacts) if var_impacts else 0.0
            average_cvar_impact = np.mean(cvar_impacts) if cvar_impacts else 0.0
            
            # Resilience analysis (assuming -5% loss threshold)
            loss_threshold = -0.05
            scenarios_passed = sum(1 for r in successful_results if r.return_impact > loss_threshold)
            scenarios_failed = total_scenarios - scenarios_passed
            resilience_score = scenarios_passed / total_scenarios if total_scenarios > 0 else 0.0
            
            # Tail risk analysis
            tail_threshold = np.percentile(return_impacts, 5)
            tail_scenarios = [r.scenario_name for r in successful_results if r.return_impact <= tail_threshold]
            
            extreme_losses = [r.return_impact for r in successful_results if r.return_impact <= tail_threshold]
            extreme_loss_probability = sum(r.scenario_probability for r in successful_results if r.return_impact <= tail_threshold)
            expected_tail_loss = np.mean(extreme_losses) if extreme_losses else 0.0
            
            # Factor analysis
            all_factors = set()
            for r in successful_results:
                all_factors.update(r.factor_contributions.keys())
            
            most_impactful_factors = {}
            for factor in all_factors:
                factor_impacts = [r.factor_contributions.get(factor, 0.0) for r in successful_results]
                most_impactful_factors[factor] = np.mean(factor_impacts)
            
            # Sort by impact
            most_impactful_factors = dict(sorted(most_impactful_factors.items(), 
                                               key=lambda x: abs(x[1]), reverse=True)[:5])
            
            # Regime analysis (simplified)
            regime_impacts = {}
            stress_types = set(r.stress_type for r in successful_results)
            for stress_type in stress_types:
                type_results = [r for r in successful_results if r.stress_type == stress_type]
                type_impacts = [r.return_impact for r in type_results]
                regime_impacts[stress_type.value] = np.mean(type_impacts) if type_impacts else 0.0
            
            # Create summary
            summary = StressSummary(
                test_id=test_id,
                timestamp=datetime.now(),
                total_scenarios=total_scenarios,
                worst_case_return=worst_case_return,
                worst_case_scenario=worst_case_scenario,
                average_impact=average_impact,
                max_var_impact=max_var_impact,
                max_cvar_impact=max_cvar_impact,
                average_var_impact=average_var_impact,
                average_cvar_impact=average_cvar_impact,
                scenarios_passed=scenarios_passed,
                scenarios_failed=scenarios_failed,
                resilience_score=resilience_score,
                tail_scenarios=tail_scenarios,
                extreme_loss_probability=extreme_loss_probability,
                expected_tail_loss=expected_tail_loss,
                most_impactful_factors=most_impactful_factors,
                factor_correlation_breakdown={},  # Placeholder
                regime_sensitivity=regime_impacts,
                contagion_effects={},  # Placeholder
                stress_model_validity={},  # Placeholder
                scenario_realism={}  # Placeholder
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating stress summary: {e}")
            raise
    
    def _save_stress_result(self, result: StressResult, test_id: str):
        """Save individual stress result to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO stress_test_results (
                    test_id, timestamp, scenario_name, stress_type, severity,
                    stressed_return, unstressed_return, return_impact,
                    stressed_var, unstressed_var, var_impact, max_drawdown,
                    portfolio_impact, scenario_probability, computation_time,
                    factor_contributions, position_impacts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                datetime.now().isoformat(),
                result.scenario_name,
                result.stress_type.value,
                result.severity.value,
                result.stressed_return,
                result.unstressed_return,
                result.return_impact,
                result.stressed_var,
                result.unstressed_var,
                result.var_impact,
                result.max_drawdown,
                result.portfolio_impact,
                result.scenario_probability,
                result.computation_time,
                json.dumps(result.factor_contributions),
                json.dumps(result.position_impacts)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving stress result: {e}")
    
    def _save_stress_summary(self, summary: StressSummary):
        """Save stress test summary to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            scenario_distribution = {}
            risk_factor_analysis = {
                'most_impactful_factors': summary.most_impactful_factors,
                'regime_sensitivity': summary.regime_sensitivity
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO stress_test_summary (
                    test_id, timestamp, total_scenarios, worst_case_return,
                    worst_case_scenario, resilience_score, max_var_impact,
                    tail_loss_probability, execution_time, scenario_distribution,
                    risk_factor_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.test_id,
                summary.timestamp.isoformat(),
                summary.total_scenarios,
                summary.worst_case_return,
                summary.worst_case_scenario,
                summary.resilience_score,
                summary.max_var_impact,
                summary.extreme_loss_probability,
                0.0,  # execution_time placeholder
                json.dumps(scenario_distribution),
                json.dumps(risk_factor_analysis)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving stress summary: {e}")
    
    def get_stress_results(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve stress test results."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if test_id:
                summary_query = "SELECT * FROM stress_test_summary WHERE test_id = ?"
                summary_df = pd.read_sql_query(summary_query, conn, params=[test_id])
                
                results_query = "SELECT * FROM stress_test_results WHERE test_id = ?"
                results_df = pd.read_sql_query(results_query, conn, params=[test_id])
            else:
                # Get latest test
                summary_query = "SELECT * FROM stress_test_summary ORDER BY timestamp DESC LIMIT 1"
                summary_df = pd.read_sql_query(summary_query, conn)
                
                if not summary_df.empty:
                    latest_test_id = summary_df.iloc[0]['test_id']
                    results_query = "SELECT * FROM stress_test_results WHERE test_id = ?"
                    results_df = pd.read_sql_query(results_query, conn, params=[latest_test_id])
                else:
                    results_df = pd.DataFrame()
            
            conn.close()
            
            return {
                'summary': summary_df.to_dict('records'),
                'results': results_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error retrieving stress results: {e}")
            return {'summary': [], 'results': []}

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample portfolio data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Generate synthetic portfolio returns
    returns = np.random.normal(0.0005, 0.015, n_samples)
    
    # Add some persistence and regime changes
    for i in range(1, n_samples):
        returns[i] += 0.05 * returns[i-1]
    
    # Add some extreme events
    crash_indices = np.random.choice(n_samples, 5, replace=False)
    returns[crash_indices] -= np.random.uniform(0.05, 0.15, 5)
    
    portfolio_data = pd.DataFrame({
        'returns': returns
    }, index=dates)
    
    # Run stress test
    stress_tester = StressTester()
    
    try:
        summary = stress_tester.run_stress_test(portfolio_data)
        
        print("\n=== Stress Test Summary ===")
        print(f"Test ID: {summary.test_id}")
        print(f"Total Scenarios: {summary.total_scenarios}")
        print(f"Worst Case Return: {summary.worst_case_return:.4f}")
        print(f"Worst Case Scenario: {summary.worst_case_scenario}")
        print(f"Average Impact: {summary.average_impact:.4f}")
        print(f"Resilience Score: {summary.resilience_score:.2%}")
        print(f"Max VaR Impact: {summary.max_var_impact:.4f}")
        print(f"Expected Tail Loss: {summary.expected_tail_loss:.4f}")
        
        print("\n--- Most Impactful Risk Factors ---")
        for factor, impact in summary.most_impactful_factors.items():
            print(f"{factor}: {impact:.4f}")
        
        print("\n--- Regime Sensitivity ---")
        for regime, impact in summary.regime_sensitivity.items():
            print(f"{regime}: {impact:.4f}")
        
    except Exception as e:
        logger.error(f"Error in stress test example: {e}")