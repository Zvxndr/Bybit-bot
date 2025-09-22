"""
Advanced Risk Budgeting Engine.
Provides comprehensive risk budgeting, attribution, and constraint management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class RiskBudgetType(Enum):
    """Risk budget types."""
    VOLATILITY = "volatility"
    VAR = "var"                    # Value at Risk
    CVAR = "cvar"                  # Conditional Value at Risk
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    TRACKING_ERROR = "tracking_error"
    DOWNSIDE_DEVIATION = "downside_deviation"
    TAIL_RISK = "tail_risk"
    CONCENTRATION_RISK = "concentration_risk"

class RiskConstraintType(Enum):
    """Risk constraint types."""
    POSITION_LIMIT = "position_limit"
    SECTOR_LIMIT = "sector_limit"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_LIMIT = "volatility_limit"
    VAR_LIMIT = "var_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    LEVERAGE_LIMIT = "leverage_limit"

@dataclass
class RiskBudget:
    """Risk budget definition."""
    budget_type: RiskBudgetType
    total_budget: float
    asset_budgets: Dict[str, float]
    utilized_budget: Dict[str, float]
    remaining_budget: Dict[str, float]
    budget_utilization: float  # 0-1
    risk_contributions: Dict[str, float]
    marginal_risks: Dict[str, float]
    budget_violations: List[str]
    last_updated: datetime

@dataclass
class RiskConstraint:
    """Risk constraint definition."""
    constraint_type: RiskConstraintType
    constraint_value: float
    current_value: float
    utilization: float  # current/constraint
    is_violated: bool
    violation_amount: float
    description: str
    priority: int  # 1=high, 2=medium, 3=low
    last_checked: datetime

@dataclass
class RiskAttribution:
    """Risk attribution analysis."""
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_contributions: Dict[str, float]
    asset_contributions: Dict[str, float]
    sector_contributions: Dict[str, float]
    style_contributions: Dict[str, float]
    interaction_effects: Dict[str, float]
    attribution_quality: float  # R-squared of attribution
    analysis_date: datetime

@dataclass
class RiskReport:
    """Comprehensive risk report."""
    portfolio_risk_metrics: Dict[str, float]
    risk_budgets: List[RiskBudget]
    risk_constraints: List[RiskConstraint]
    risk_attribution: RiskAttribution
    risk_warnings: List[str]
    risk_recommendations: List[str]
    stress_test_results: Dict[str, float]
    scenario_analysis: Dict[str, Dict[str, float]]
    risk_adjusted_performance: Dict[str, float]
    report_date: datetime
    confidence_level: float

class RiskBudgeter:
    """Advanced risk budgeting and constraint management engine."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Risk budgeting configuration
        self.risk_config = {
            'confidence_levels': [0.95, 0.99],
            'lookback_periods': {
                'short': 60,
                'medium': 120,
                'long': 252
            },
            'risk_budget_limits': {
                'volatility': {'total': 0.20, 'individual': 0.05},
                'var': {'total': 0.05, 'individual': 0.02},
                'cvar': {'total': 0.08, 'individual': 0.03},
                'max_drawdown': {'total': 0.15, 'individual': 0.05},
                'concentration': {'total': 1.0, 'individual': 0.25}
            },
            'constraint_limits': {
                'position_limit': 0.20,      # Max 20% in single position
                'sector_limit': 0.30,        # Max 30% in single sector
                'correlation_limit': 0.80,   # Max correlation with benchmark
                'volatility_limit': 0.25,    # Max 25% annualized volatility
                'var_limit': 0.05,           # Max 5% daily VaR
                'drawdown_limit': 0.20,      # Max 20% drawdown
                'leverage_limit': 1.0        # No leverage
            },
            'stress_scenarios': {
                'market_crash': {'equity_shock': -0.30, 'vol_spike': 2.0},
                'correlation_spike': {'correlation_increase': 0.50},
                'liquidity_crisis': {'bid_ask_widening': 5.0},
                'sector_rotation': {'sector_dispersion': 0.20}
            }
        }
        
        # Risk history
        self.risk_history = []
        self.constraint_violations = []
        
        # Risk factor loadings (simplified)
        self.risk_factors = {
            'market': {'BTC': 0.8, 'ETH': 0.7, 'ADA': 0.6, 'SOL': 0.7},
            'size': {'BTC': 1.0, 'ETH': 0.8, 'ADA': 0.3, 'SOL': 0.4},
            'momentum': {'BTC': 0.5, 'ETH': 0.6, 'ADA': 0.3, 'SOL': 0.8},
            'volatility': {'BTC': 0.6, 'ETH': 0.7, 'ADA': 0.8, 'SOL': 0.9}
        }
        
        self.logger.info("RiskBudgeter initialized")
    
    async def create_risk_budgets(self, 
                                weights: Dict[str, float],
                                returns: pd.DataFrame,
                                target_risk: float = None) -> List[RiskBudget]:
        """Create comprehensive risk budgets for portfolio."""
        budgets = []
        
        try:
            # Default target risk
            if target_risk is None:
                target_risk = self.risk_config['risk_budget_limits']['volatility']['total']
            
            # Volatility budget
            vol_budget = await self._create_volatility_budget(weights, returns, target_risk)
            budgets.append(vol_budget)
            
            # VaR budget
            var_budget = await self._create_var_budget(weights, returns)
            budgets.append(var_budget)
            
            # CVaR budget
            cvar_budget = await self._create_cvar_budget(weights, returns)
            budgets.append(cvar_budget)
            
            # Maximum drawdown budget
            dd_budget = await self._create_drawdown_budget(weights, returns)
            budgets.append(dd_budget)
            
            # Concentration risk budget
            conc_budget = await self._create_concentration_budget(weights)
            budgets.append(conc_budget)
            
            return budgets
            
        except Exception as e:
            self.logger.error(f"Failed to create risk budgets: {e}")
            return []
    
    async def check_risk_constraints(self, 
                                   weights: Dict[str, float],
                                   returns: pd.DataFrame,
                                   portfolio_metrics: Dict[str, Any]) -> List[RiskConstraint]:
        """Check all risk constraints."""
        constraints = []
        
        try:
            current_time = datetime.now()
            
            # Position limit constraints
            position_constraint = await self._check_position_limits(weights)
            if position_constraint:
                constraints.append(position_constraint)
            
            # Volatility constraint
            vol_constraint = await self._check_volatility_constraint(portfolio_metrics)
            if vol_constraint:
                constraints.append(vol_constraint)
            
            # VaR constraint
            var_constraint = await self._check_var_constraint(returns, weights)
            if var_constraint:
                constraints.append(var_constraint)
            
            # Drawdown constraint
            dd_constraint = await self._check_drawdown_constraint(portfolio_metrics)
            if dd_constraint:
                constraints.append(dd_constraint)
            
            # Concentration constraint
            conc_constraint = await self._check_concentration_constraint(weights)
            if conc_constraint:
                constraints.append(conc_constraint)
            
            # Correlation constraint
            corr_constraint = await self._check_correlation_constraint(returns)
            if corr_constraint:
                constraints.append(corr_constraint)
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"Failed to check risk constraints: {e}")
            return []
    
    async def perform_risk_attribution(self, 
                                     weights: Dict[str, float],
                                     returns: pd.DataFrame) -> RiskAttribution:
        """Perform comprehensive risk attribution analysis."""
        try:
            # Calculate portfolio returns
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = returns @ weight_array
            
            # Total portfolio risk
            total_risk = portfolio_returns.std() * np.sqrt(252)  # Annualized
            
            # Systematic vs idiosyncratic risk decomposition
            systematic_risk, idiosyncratic_risk = await self._decompose_systematic_risk(
                weights, returns
            )
            
            # Factor contributions
            factor_contributions = await self._calculate_factor_contributions(
                weights, returns
            )
            
            # Asset contributions
            asset_contributions = await self._calculate_asset_risk_contributions(
                weights, returns
            )
            
            # Sector contributions (simplified - based on asset groupings)
            sector_contributions = await self._calculate_sector_contributions(
                weights, returns
            )
            
            # Style contributions
            style_contributions = await self._calculate_style_contributions(
                weights, returns
            )
            
            # Interaction effects
            interaction_effects = await self._calculate_interaction_effects(
                weights, returns
            )
            
            # Attribution quality (simplified R-squared)
            attribution_quality = await self._calculate_attribution_quality(
                total_risk, systematic_risk, idiosyncratic_risk
            )
            
            return RiskAttribution(
                total_risk=float(total_risk),
                systematic_risk=float(systematic_risk),
                idiosyncratic_risk=float(idiosyncratic_risk),
                factor_contributions=factor_contributions,
                asset_contributions=asset_contributions,
                sector_contributions=sector_contributions,
                style_contributions=style_contributions,
                interaction_effects=interaction_effects,
                attribution_quality=float(attribution_quality),
                analysis_date=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Risk attribution failed: {e}")
            return RiskAttribution(
                total_risk=0.0,
                systematic_risk=0.0,
                idiosyncratic_risk=0.0,
                factor_contributions={},
                asset_contributions={},
                sector_contributions={},
                style_contributions={},
                interaction_effects={},
                attribution_quality=0.0,
                analysis_date=datetime.now()
            )
    
    async def generate_risk_report(self, 
                                 weights: Dict[str, float],
                                 returns: pd.DataFrame,
                                 portfolio_metrics: Dict[str, Any]) -> RiskReport:
        """Generate comprehensive risk report."""
        try:
            # Create risk budgets
            risk_budgets = await self.create_risk_budgets(weights, returns)
            
            # Check constraints
            risk_constraints = await self.check_risk_constraints(weights, returns, portfolio_metrics)
            
            # Perform attribution
            risk_attribution = await self.perform_risk_attribution(weights, returns)
            
            # Calculate portfolio risk metrics
            portfolio_risk_metrics = await self._calculate_portfolio_risk_metrics(
                weights, returns, portfolio_metrics
            )
            
            # Generate warnings and recommendations
            risk_warnings = await self._generate_risk_warnings(risk_budgets, risk_constraints)
            risk_recommendations = await self._generate_risk_recommendations(
                risk_budgets, risk_constraints, risk_attribution
            )
            
            # Perform stress tests
            stress_results = await self._perform_stress_tests(weights, returns)
            
            # Scenario analysis
            scenario_analysis = await self._perform_scenario_analysis(weights, returns)
            
            # Risk-adjusted performance
            risk_adjusted_performance = await self._calculate_risk_adjusted_performance(
                returns, weights, portfolio_metrics
            )
            
            return RiskReport(
                portfolio_risk_metrics=portfolio_risk_metrics,
                risk_budgets=risk_budgets,
                risk_constraints=risk_constraints,
                risk_attribution=risk_attribution,
                risk_warnings=risk_warnings,
                risk_recommendations=risk_recommendations,
                stress_test_results=stress_results,
                scenario_analysis=scenario_analysis,
                risk_adjusted_performance=risk_adjusted_performance,
                report_date=datetime.now(),
                confidence_level=0.95
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate risk report: {e}")
            raise
    
    async def _create_volatility_budget(self, 
                                      weights: Dict[str, float],
                                      returns: pd.DataFrame,
                                      target_vol: float) -> RiskBudget:
        """Create volatility risk budget."""
        try:
            # Calculate current portfolio volatility
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            cov_matrix = returns.cov() * 252  # Annualized
            
            portfolio_var = np.dot(weight_array.T, np.dot(cov_matrix.values, weight_array))
            portfolio_vol = np.sqrt(portfolio_var)
            
            if portfolio_vol == 0:
                # Equal budget allocation if no volatility
                n_assets = len(weights)
                equal_budget = target_vol / n_assets
                asset_budgets = {symbol: equal_budget for symbol in weights}
                utilized_budget = {symbol: 0.0 for symbol in weights}
                remaining_budget = {symbol: equal_budget for symbol in weights}
                risk_contributions = {symbol: 0.0 for symbol in weights}
                marginal_risks = {symbol: 0.0 for symbol in weights}
            else:
                # Calculate marginal risk contributions
                marginal_risks = {}
                risk_contributions = {}
                
                for i, symbol in enumerate(returns.columns):
                    if symbol in weights:
                        marginal_risk = np.dot(cov_matrix.values[i, :], weight_array) / portfolio_vol
                        risk_contribution = weights[symbol] * marginal_risk
                        
                        marginal_risks[symbol] = float(marginal_risk)
                        risk_contributions[symbol] = float(risk_contribution)
                
                # Allocate budgets proportional to target weights
                asset_budgets = {symbol: target_vol * weights[symbol] for symbol in weights}
                
                # Calculate utilized and remaining budgets
                total_risk_contribution = sum(risk_contributions.values())
                scale_factor = target_vol / total_risk_contribution if total_risk_contribution > 0 else 1.0
                
                utilized_budget = {symbol: risk_contributions[symbol] * scale_factor 
                                 for symbol in weights}
                remaining_budget = {symbol: asset_budgets[symbol] - utilized_budget[symbol] 
                                  for symbol in weights}
            
            # Calculate budget utilization
            total_budget = sum(asset_budgets.values())
            total_utilized = sum(utilized_budget.values())
            budget_utilization = total_utilized / total_budget if total_budget > 0 else 0
            
            # Check for violations
            violations = []
            individual_limit = self.risk_config['risk_budget_limits']['volatility']['individual']
            
            for symbol in weights:
                if utilized_budget[symbol] > individual_limit:
                    violations.append(f"{symbol} exceeds individual volatility budget: "
                                    f"{utilized_budget[symbol]:.1%} > {individual_limit:.1%}")
            
            return RiskBudget(
                budget_type=RiskBudgetType.VOLATILITY,
                total_budget=target_vol,
                asset_budgets=asset_budgets,
                utilized_budget=utilized_budget,
                remaining_budget=remaining_budget,
                budget_utilization=budget_utilization,
                risk_contributions=risk_contributions,
                marginal_risks=marginal_risks,
                budget_violations=violations,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create volatility budget: {e}")
            return RiskBudget(
                budget_type=RiskBudgetType.VOLATILITY,
                total_budget=target_vol,
                asset_budgets={},
                utilized_budget={},
                remaining_budget={},
                budget_utilization=0.0,
                risk_contributions={},
                marginal_risks={},
                budget_violations=[],
                last_updated=datetime.now()
            )
    
    async def _create_var_budget(self, 
                               weights: Dict[str, float],
                               returns: pd.DataFrame,
                               confidence_level: float = 0.95) -> RiskBudget:
        """Create Value at Risk budget."""
        try:
            # Calculate portfolio returns
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = returns @ weight_array
            
            # Calculate VaR
            var_quantile = stats.scoreatpercentile(portfolio_returns, (1 - confidence_level) * 100)
            portfolio_var = abs(var_quantile)
            
            # Target VaR budget
            target_var = self.risk_config['risk_budget_limits']['var']['total']
            
            # Simple VaR contribution approximation (using volatility contributions)
            vol_budget = await self._create_volatility_budget(weights, returns, 0.20)
            
            # Scale volatility contributions to VaR
            var_scale = portfolio_var / vol_budget.total_budget if vol_budget.total_budget > 0 else 1.0
            
            asset_budgets = {symbol: target_var * weights[symbol] for symbol in weights}
            utilized_budget = {symbol: vol_budget.risk_contributions.get(symbol, 0) * var_scale 
                             for symbol in weights}
            remaining_budget = {symbol: asset_budgets[symbol] - utilized_budget[symbol] 
                              for symbol in weights}
            
            # Risk contributions (approximated)
            risk_contributions = utilized_budget.copy()
            marginal_risks = {symbol: risk_contributions[symbol] / weights[symbol] 
                            if weights[symbol] > 0 else 0 for symbol in weights}
            
            # Budget utilization
            total_utilized = sum(utilized_budget.values())
            budget_utilization = total_utilized / target_var if target_var > 0 else 0
            
            # Check violations
            violations = []
            individual_limit = self.risk_config['risk_budget_limits']['var']['individual']
            
            for symbol in weights:
                if utilized_budget[symbol] > individual_limit:
                    violations.append(f"{symbol} exceeds individual VaR budget: "
                                    f"{utilized_budget[symbol]:.1%} > {individual_limit:.1%}")
            
            return RiskBudget(
                budget_type=RiskBudgetType.VAR,
                total_budget=target_var,
                asset_budgets=asset_budgets,
                utilized_budget=utilized_budget,
                remaining_budget=remaining_budget,
                budget_utilization=budget_utilization,
                risk_contributions=risk_contributions,
                marginal_risks=marginal_risks,
                budget_violations=violations,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create VaR budget: {e}")
            return RiskBudget(
                budget_type=RiskBudgetType.VAR,
                total_budget=0.05,
                asset_budgets={},
                utilized_budget={},
                remaining_budget={},
                budget_utilization=0.0,
                risk_contributions={},
                marginal_risks={},
                budget_violations=[],
                last_updated=datetime.now()
            )
    
    async def _create_cvar_budget(self, 
                                weights: Dict[str, float],
                                returns: pd.DataFrame,
                                confidence_level: float = 0.95) -> RiskBudget:
        """Create Conditional Value at Risk budget."""
        try:
            # Calculate portfolio returns
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = returns @ weight_array
            
            # Calculate CVaR (Expected Shortfall)
            var_threshold = stats.scoreatpercentile(portfolio_returns, (1 - confidence_level) * 100)
            tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
            cvar = abs(tail_losses.mean()) if len(tail_losses) > 0 else 0
            
            # Target CVaR budget
            target_cvar = self.risk_config['risk_budget_limits']['cvar']['total']
            
            # Approximate CVaR contributions using tail correlation
            asset_budgets = {symbol: target_cvar * weights[symbol] for symbol in weights}
            
            # Simple approximation: scale VaR contributions
            var_budget = await self._create_var_budget(weights, returns, confidence_level)
            cvar_scale = cvar / var_budget.total_budget if var_budget.total_budget > 0 else 1.0
            
            utilized_budget = {symbol: var_budget.utilized_budget.get(symbol, 0) * cvar_scale 
                             for symbol in weights}
            remaining_budget = {symbol: asset_budgets[symbol] - utilized_budget[symbol] 
                              for symbol in weights}
            
            risk_contributions = utilized_budget.copy()
            marginal_risks = {symbol: risk_contributions[symbol] / weights[symbol] 
                            if weights[symbol] > 0 else 0 for symbol in weights}
            
            # Budget utilization
            total_utilized = sum(utilized_budget.values())
            budget_utilization = total_utilized / target_cvar if target_cvar > 0 else 0
            
            # Check violations
            violations = []
            individual_limit = self.risk_config['risk_budget_limits']['cvar']['individual']
            
            for symbol in weights:
                if utilized_budget[symbol] > individual_limit:
                    violations.append(f"{symbol} exceeds individual CVaR budget: "
                                    f"{utilized_budget[symbol]:.1%} > {individual_limit:.1%}")
            
            return RiskBudget(
                budget_type=RiskBudgetType.CVAR,
                total_budget=target_cvar,
                asset_budgets=asset_budgets,
                utilized_budget=utilized_budget,
                remaining_budget=remaining_budget,
                budget_utilization=budget_utilization,
                risk_contributions=risk_contributions,
                marginal_risks=marginal_risks,
                budget_violations=violations,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create CVaR budget: {e}")
            return RiskBudget(
                budget_type=RiskBudgetType.CVAR,
                total_budget=0.08,
                asset_budgets={},
                utilized_budget={},
                remaining_budget={},
                budget_utilization=0.0,
                risk_contributions={},
                marginal_risks={},
                budget_violations=[],
                last_updated=datetime.now()
            )
    
    async def _create_drawdown_budget(self, 
                                    weights: Dict[str, float],
                                    returns: pd.DataFrame) -> RiskBudget:
        """Create maximum drawdown budget."""
        try:
            # Calculate portfolio returns
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = returns @ weight_array
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Target drawdown budget
            target_dd = self.risk_config['risk_budget_limits']['max_drawdown']['total']
            
            # Approximate drawdown contributions
            asset_budgets = {symbol: target_dd * weights[symbol] for symbol in weights}
            
            # Simple approximation: proportional to volatility contribution
            vol_budget = await self._create_volatility_budget(weights, returns, 0.20)
            dd_scale = max_drawdown / vol_budget.total_budget if vol_budget.total_budget > 0 else 1.0
            
            utilized_budget = {symbol: vol_budget.risk_contributions.get(symbol, 0) * dd_scale 
                             for symbol in weights}
            remaining_budget = {symbol: asset_budgets[symbol] - utilized_budget[symbol] 
                              for symbol in weights}
            
            risk_contributions = utilized_budget.copy()
            marginal_risks = {symbol: risk_contributions[symbol] / weights[symbol] 
                            if weights[symbol] > 0 else 0 for symbol in weights}
            
            # Budget utilization
            total_utilized = sum(utilized_budget.values())
            budget_utilization = total_utilized / target_dd if target_dd > 0 else 0
            
            # Check violations
            violations = []
            individual_limit = self.risk_config['risk_budget_limits']['max_drawdown']['individual']
            
            for symbol in weights:
                if utilized_budget[symbol] > individual_limit:
                    violations.append(f"{symbol} exceeds individual drawdown budget: "
                                    f"{utilized_budget[symbol]:.1%} > {individual_limit:.1%}")
            
            return RiskBudget(
                budget_type=RiskBudgetType.MAXIMUM_DRAWDOWN,
                total_budget=target_dd,
                asset_budgets=asset_budgets,
                utilized_budget=utilized_budget,
                remaining_budget=remaining_budget,
                budget_utilization=budget_utilization,
                risk_contributions=risk_contributions,
                marginal_risks=marginal_risks,
                budget_violations=violations,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create drawdown budget: {e}")
            return RiskBudget(
                budget_type=RiskBudgetType.MAXIMUM_DRAWDOWN,
                total_budget=0.15,
                asset_budgets={},
                utilized_budget={},
                remaining_budget={},
                budget_utilization=0.0,
                risk_contributions={},
                marginal_risks={},
                budget_violations=[],
                last_updated=datetime.now()
            )
    
    async def _create_concentration_budget(self, weights: Dict[str, float]) -> RiskBudget:
        """Create concentration risk budget."""
        try:
            # Target concentration (Herfindahl-Hirschman Index)
            target_concentration = self.risk_config['risk_budget_limits']['concentration']['total']
            
            # Calculate current concentration
            current_hhi = sum(w**2 for w in weights.values())
            
            # Asset budgets (max weight squared)
            individual_limit = self.risk_config['risk_budget_limits']['concentration']['individual']
            asset_budgets = {symbol: individual_limit for symbol in weights}
            
            # Utilized budget (actual weight squared)
            utilized_budget = {symbol: weights[symbol]**2 for symbol in weights}
            
            # Remaining budget
            remaining_budget = {symbol: asset_budgets[symbol] - utilized_budget[symbol] 
                              for symbol in weights}
            
            # Risk contributions (weight squared contribution to HHI)
            risk_contributions = utilized_budget.copy()
            
            # Marginal risks (derivative of HHI w.r.t. weight)
            marginal_risks = {symbol: 2 * weights[symbol] for symbol in weights}
            
            # Budget utilization
            budget_utilization = current_hhi / target_concentration if target_concentration > 0 else 0
            
            # Check violations
            violations = []
            for symbol in weights:
                if utilized_budget[symbol] > individual_limit:
                    violations.append(f"{symbol} exceeds concentration budget: "
                                    f"{np.sqrt(utilized_budget[symbol]):.1%} position > "
                                    f"{np.sqrt(individual_limit):.1%} limit")
            
            return RiskBudget(
                budget_type=RiskBudgetType.CONCENTRATION_RISK,
                total_budget=target_concentration,
                asset_budgets=asset_budgets,
                utilized_budget=utilized_budget,
                remaining_budget=remaining_budget,
                budget_utilization=budget_utilization,
                risk_contributions=risk_contributions,
                marginal_risks=marginal_risks,
                budget_violations=violations,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create concentration budget: {e}")
            return RiskBudget(
                budget_type=RiskBudgetType.CONCENTRATION_RISK,
                total_budget=1.0,
                asset_budgets={},
                utilized_budget={},
                remaining_budget={},
                budget_utilization=0.0,
                risk_contributions={},
                marginal_risks={},
                budget_violations=[],
                last_updated=datetime.now()
            )
    
    async def _check_position_limits(self, weights: Dict[str, float]) -> Optional[RiskConstraint]:
        """Check position limit constraints."""
        try:
            limit = self.risk_config['constraint_limits']['position_limit']
            max_position = max(weights.values()) if weights else 0
            max_symbol = max(weights.items(), key=lambda x: x[1])[0] if weights else ""
            
            utilization = max_position / limit if limit > 0 else 0
            is_violated = max_position > limit
            violation_amount = max(0, max_position - limit)
            
            if is_violated or utilization > 0.8:  # Report if close to limit
                return RiskConstraint(
                    constraint_type=RiskConstraintType.POSITION_LIMIT,
                    constraint_value=limit,
                    current_value=max_position,
                    utilization=utilization,
                    is_violated=is_violated,
                    violation_amount=violation_amount,
                    description=f"Maximum position size: {max_symbol} at {max_position:.1%}",
                    priority=1 if is_violated else 2,
                    last_checked=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check position limits: {e}")
            return None
    
    async def _check_volatility_constraint(self, portfolio_metrics: Dict[str, Any]) -> Optional[RiskConstraint]:
        """Check portfolio volatility constraint."""
        try:
            limit = self.risk_config['constraint_limits']['volatility_limit']
            current_vol = portfolio_metrics.get('volatility', 0.0)
            
            utilization = current_vol / limit if limit > 0 else 0
            is_violated = current_vol > limit
            violation_amount = max(0, current_vol - limit)
            
            if is_violated or utilization > 0.8:
                return RiskConstraint(
                    constraint_type=RiskConstraintType.VOLATILITY_LIMIT,
                    constraint_value=limit,
                    current_value=current_vol,
                    utilization=utilization,
                    is_violated=is_violated,
                    violation_amount=violation_amount,
                    description=f"Portfolio volatility: {current_vol:.1%} vs {limit:.1%} limit",
                    priority=1 if is_violated else 2,
                    last_checked=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check volatility constraint: {e}")
            return None
    
    async def _check_var_constraint(self, 
                                  returns: pd.DataFrame,
                                  weights: Dict[str, float]) -> Optional[RiskConstraint]:
        """Check VaR constraint."""
        try:
            limit = self.risk_config['constraint_limits']['var_limit']
            
            # Calculate portfolio VaR
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = returns @ weight_array
            
            var_95 = abs(stats.scoreatpercentile(portfolio_returns, 5))
            
            utilization = var_95 / limit if limit > 0 else 0
            is_violated = var_95 > limit
            violation_amount = max(0, var_95 - limit)
            
            if is_violated or utilization > 0.8:
                return RiskConstraint(
                    constraint_type=RiskConstraintType.VAR_LIMIT,
                    constraint_value=limit,
                    current_value=var_95,
                    utilization=utilization,
                    is_violated=is_violated,
                    violation_amount=violation_amount,
                    description=f"Portfolio VaR (95%): {var_95:.1%} vs {limit:.1%} limit",
                    priority=1 if is_violated else 2,
                    last_checked=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check VaR constraint: {e}")
            return None
    
    async def _check_drawdown_constraint(self, portfolio_metrics: Dict[str, Any]) -> Optional[RiskConstraint]:
        """Check maximum drawdown constraint."""
        try:
            limit = self.risk_config['constraint_limits']['drawdown_limit']
            current_dd = abs(portfolio_metrics.get('max_drawdown', 0.0))
            
            utilization = current_dd / limit if limit > 0 else 0
            is_violated = current_dd > limit
            violation_amount = max(0, current_dd - limit)
            
            if is_violated or utilization > 0.8:
                return RiskConstraint(
                    constraint_type=RiskConstraintType.DRAWDOWN_LIMIT,
                    constraint_value=limit,
                    current_value=current_dd,
                    utilization=utilization,
                    is_violated=is_violated,
                    violation_amount=violation_amount,
                    description=f"Maximum drawdown: {current_dd:.1%} vs {limit:.1%} limit",
                    priority=1 if is_violated else 2,
                    last_checked=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check drawdown constraint: {e}")
            return None
    
    async def _check_concentration_constraint(self, weights: Dict[str, float]) -> Optional[RiskConstraint]:
        """Check concentration constraint."""
        try:
            # Using HHI as concentration measure
            current_hhi = sum(w**2 for w in weights.values())
            
            # Maximum allowable concentration (lower HHI = better diversification)
            # For equal weights: HHI = 1/n, so set limit based on minimum diversification
            min_assets = 4  # Assume at least 4 assets for diversification
            limit = 1.0 / min_assets
            
            utilization = current_hhi / 1.0  # Compare to maximum concentration (1.0)
            is_violated = current_hhi > limit
            violation_amount = max(0, current_hhi - limit)
            
            if is_violated or utilization > 0.8:
                return RiskConstraint(
                    constraint_type=RiskConstraintType.CONCENTRATION_LIMIT,
                    constraint_value=limit,
                    current_value=current_hhi,
                    utilization=utilization,
                    is_violated=is_violated,
                    violation_amount=violation_amount,
                    description=f"Portfolio concentration (HHI): {current_hhi:.3f} vs {limit:.3f} limit",
                    priority=2 if is_violated else 3,
                    last_checked=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check concentration constraint: {e}")
            return None
    
    async def _check_correlation_constraint(self, returns: pd.DataFrame) -> Optional[RiskConstraint]:
        """Check correlation constraint."""
        try:
            # Calculate average pairwise correlation
            corr_matrix = returns.corr()
            n = len(corr_matrix)
            
            if n < 2:
                return None
            
            # Get upper triangle correlations (excluding diagonal)
            upper_triangle = corr_matrix.values[np.triu_indices(n, k=1)]
            avg_correlation = np.nanmean(np.abs(upper_triangle))
            
            limit = self.risk_config['constraint_limits']['correlation_limit']
            
            utilization = avg_correlation / limit if limit > 0 else 0
            is_violated = avg_correlation > limit
            violation_amount = max(0, avg_correlation - limit)
            
            if is_violated or utilization > 0.8:
                return RiskConstraint(
                    constraint_type=RiskConstraintType.CORRELATION_LIMIT,
                    constraint_value=limit,
                    current_value=avg_correlation,
                    utilization=utilization,
                    is_violated=is_violated,
                    violation_amount=violation_amount,
                    description=f"Average correlation: {avg_correlation:.2f} vs {limit:.2f} limit",
                    priority=2 if is_violated else 3,
                    last_checked=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check correlation constraint: {e}")
            return None
    
    # Risk attribution helper methods
    async def _decompose_systematic_risk(self, 
                                       weights: Dict[str, float],
                                       returns: pd.DataFrame) -> Tuple[float, float]:
        """Decompose risk into systematic and idiosyncratic components."""
        try:
            # Simplified decomposition using market factor
            # In practice, would use more sophisticated factor models
            
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = returns @ weight_array
            
            # Use equal-weighted market as proxy for systematic factor
            market_returns = returns.mean(axis=1)
            
            # Calculate beta to market
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance > 0:
                beta = covariance / market_variance
                systematic_var = beta**2 * market_variance
            else:
                systematic_var = 0
            
            # Total portfolio variance
            total_var = np.var(portfolio_returns)
            idiosyncratic_var = max(0, total_var - systematic_var)
            
            systematic_risk = np.sqrt(systematic_var) * np.sqrt(252)  # Annualized
            idiosyncratic_risk = np.sqrt(idiosyncratic_var) * np.sqrt(252)
            
            return systematic_risk, idiosyncratic_risk
            
        except Exception as e:
            self.logger.error(f"Risk decomposition failed: {e}")
            return 0.0, 0.0
    
    async def _calculate_factor_contributions(self, 
                                            weights: Dict[str, float],
                                            returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate factor contributions to risk."""
        try:
            factor_contributions = {}
            
            for factor_name, factor_loadings in self.risk_factors.items():
                factor_contribution = 0
                
                for symbol in weights:
                    if symbol in factor_loadings:
                        loading = factor_loadings[symbol]
                        weight = weights[symbol]
                        factor_contribution += weight * loading
                
                factor_contributions[factor_name] = float(factor_contribution)
            
            return factor_contributions
            
        except Exception as e:
            self.logger.error(f"Factor contribution calculation failed: {e}")
            return {}
    
    async def _calculate_asset_risk_contributions(self, 
                                                weights: Dict[str, float],
                                                returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate individual asset risk contributions."""
        try:
            # Use volatility budget calculation
            vol_budget = await self._create_volatility_budget(weights, returns, 0.20)
            return vol_budget.risk_contributions
            
        except Exception as e:
            self.logger.error(f"Asset risk contribution calculation failed: {e}")
            return {}
    
    async def _calculate_sector_contributions(self, 
                                            weights: Dict[str, float],
                                            returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate sector risk contributions."""
        try:
            # Simplified sector grouping for crypto
            sectors = {
                'large_cap': ['BTC', 'ETH'],
                'alt_coins': ['ADA', 'SOL', 'AVAX', 'DOT'],
                'defi': ['UNI', 'LINK'],
                'others': []
            }
            
            sector_contributions = {}
            asset_contributions = await self._calculate_asset_risk_contributions(weights, returns)
            
            for sector, symbols in sectors.items():
                sector_risk = sum(asset_contributions.get(symbol, 0) for symbol in symbols)
                sector_contributions[sector] = float(sector_risk)
            
            return sector_contributions
            
        except Exception as e:
            self.logger.error(f"Sector contribution calculation failed: {e}")
            return {}
    
    async def _calculate_style_contributions(self, 
                                           weights: Dict[str, float],
                                           returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate style factor contributions."""
        try:
            # Use factor contributions as proxy for style
            factor_contributions = await self._calculate_factor_contributions(weights, returns)
            
            # Map factors to styles
            style_mapping = {
                'momentum': 'momentum',
                'volatility': 'low_volatility',
                'size': 'market_cap',
                'market': 'beta'
            }
            
            style_contributions = {}
            for factor, contribution in factor_contributions.items():
                style = style_mapping.get(factor, factor)
                style_contributions[style] = contribution
            
            return style_contributions
            
        except Exception as e:
            self.logger.error(f"Style contribution calculation failed: {e}")
            return {}
    
    async def _calculate_interaction_effects(self, 
                                           weights: Dict[str, float],
                                           returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate interaction effects between factors."""
        try:
            # Simplified interaction effect calculation
            # In practice, would use more sophisticated cross-factor analysis
            
            corr_matrix = returns.corr()
            interaction_effects = {}
            
            symbols = list(weights.keys())
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
                        correlation = corr_matrix.loc[symbol1, symbol2]
                        weight1 = weights[symbol1]
                        weight2 = weights[symbol2]
                        
                        interaction = correlation * weight1 * weight2
                        pair_name = f"{symbol1}_{symbol2}"
                        interaction_effects[pair_name] = float(interaction)
            
            return interaction_effects
            
        except Exception as e:
            self.logger.error(f"Interaction effect calculation failed: {e}")
            return {}
    
    async def _calculate_attribution_quality(self, 
                                           total_risk: float,
                                           systematic_risk: float,
                                           idiosyncratic_risk: float) -> float:
        """Calculate attribution quality (R-squared)."""
        try:
            if total_risk == 0:
                return 0.0
            
            explained_variance = systematic_risk**2
            total_variance = total_risk**2
            
            r_squared = explained_variance / total_variance if total_variance > 0 else 0
            return float(np.clip(r_squared, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Attribution quality calculation failed: {e}")
            return 0.0
    
    # Risk report helper methods
    async def _calculate_portfolio_risk_metrics(self, 
                                              weights: Dict[str, float],
                                              returns: pd.DataFrame,
                                              portfolio_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            # Get existing metrics
            metrics = portfolio_metrics.copy()
            
            # Add additional risk metrics
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = returns @ weight_array
            
            # Skewness and kurtosis
            metrics['skewness'] = float(stats.skew(portfolio_returns))
            metrics['kurtosis'] = float(stats.kurtosis(portfolio_returns))
            
            # VaR and CVaR at different confidence levels
            for confidence in [0.95, 0.99]:
                var = abs(stats.scoreatpercentile(portfolio_returns, (1-confidence)*100))
                metrics[f'var_{int(confidence*100)}'] = float(var)
                
                tail_losses = portfolio_returns[portfolio_returns <= -var]
                cvar = abs(tail_losses.mean()) if len(tail_losses) > 0 else var
                metrics[f'cvar_{int(confidence*100)}'] = float(cvar)
            
            # Concentration metrics
            metrics['hhi'] = float(sum(w**2 for w in weights.values()))
            metrics['effective_assets'] = float(1 / metrics['hhi']) if metrics['hhi'] > 0 else 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Portfolio risk metrics calculation failed: {e}")
            return portfolio_metrics
    
    async def _generate_risk_warnings(self, 
                                    risk_budgets: List[RiskBudget],
                                    risk_constraints: List[RiskConstraint]) -> List[str]:
        """Generate risk warnings."""
        warnings = []
        
        try:
            # Budget violation warnings
            for budget in risk_budgets:
                if budget.budget_violations:
                    warnings.extend([f"[{budget.budget_type.value.upper()}] {violation}" 
                                   for violation in budget.budget_violations])
                
                if budget.budget_utilization > 0.9:
                    warnings.append(f"[{budget.budget_type.value.upper()}] "
                                  f"High budget utilization: {budget.budget_utilization:.1%}")
            
            # Constraint violation warnings
            for constraint in risk_constraints:
                if constraint.is_violated:
                    warnings.append(f"[CONSTRAINT] {constraint.description} - "
                                  f"VIOLATED by {constraint.violation_amount:.1%}")
                elif constraint.utilization > 0.8:
                    warnings.append(f"[CONSTRAINT] {constraint.description} - "
                                  f"High utilization: {constraint.utilization:.1%}")
            
        except Exception as e:
            self.logger.error(f"Risk warning generation failed: {e}")
            warnings.append("Unable to generate complete risk warnings")
        
        return warnings
    
    async def _generate_risk_recommendations(self, 
                                           risk_budgets: List[RiskBudget],
                                           risk_constraints: List[RiskConstraint],
                                           risk_attribution: RiskAttribution) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        try:
            # High utilization recommendations
            for budget in risk_budgets:
                if budget.budget_utilization > 0.8:
                    recommendations.append(f"Consider reducing {budget.budget_type.value} exposure")
            
            # Constraint recommendations
            for constraint in risk_constraints:
                if constraint.is_violated:
                    if constraint.constraint_type == RiskConstraintType.POSITION_LIMIT:
                        recommendations.append("Reduce largest position size to comply with limits")
                    elif constraint.constraint_type == RiskConstraintType.VOLATILITY_LIMIT:
                        recommendations.append("Reduce portfolio volatility through diversification")
                    elif constraint.constraint_type == RiskConstraintType.CORRELATION_LIMIT:
                        recommendations.append("Add assets with lower correlation to improve diversification")
            
            # Attribution-based recommendations
            if risk_attribution.systematic_risk > risk_attribution.idiosyncratic_risk * 2:
                recommendations.append("Portfolio is heavily exposed to systematic risk - consider diversification")
            
            if risk_attribution.attribution_quality < 0.5:
                recommendations.append("Risk attribution quality is low - review factor model")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Risk profile appears within acceptable parameters")
            
        except Exception as e:
            self.logger.error(f"Risk recommendation generation failed: {e}")
            recommendations.append("Unable to generate complete risk recommendations")
        
        return recommendations
    
    async def _perform_stress_tests(self, 
                                  weights: Dict[str, float],
                                  returns: pd.DataFrame) -> Dict[str, float]:
        """Perform portfolio stress tests."""
        try:
            stress_results = {}
            
            # Calculate baseline portfolio metrics
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            baseline_returns = returns @ weight_array
            baseline_vol = baseline_returns.std() * np.sqrt(252)
            
            for scenario_name, scenario_params in self.risk_config['stress_scenarios'].items():
                if scenario_name == 'market_crash':
                    # Apply negative shock to all assets
                    shock = scenario_params['equity_shock']
                    stressed_return = baseline_returns.mean() * 252 + shock  # Annual return
                    stress_results[f"{scenario_name}_return"] = float(stressed_return)
                    
                    vol_multiplier = scenario_params['vol_spike']
                    stressed_vol = baseline_vol * vol_multiplier
                    stress_results[f"{scenario_name}_volatility"] = float(stressed_vol)
                
                elif scenario_name == 'correlation_spike':
                    # Increase correlations
                    corr_increase = scenario_params['correlation_increase']
                    original_corr = returns.corr()
                    
                    # Approximate portfolio vol with higher correlations
                    avg_corr = original_corr.values[np.triu_indices_from(original_corr.values, k=1)].mean()
                    new_avg_corr = min(0.95, avg_corr + corr_increase)
                    
                    # Simplified calculation
                    corr_effect = new_avg_corr / avg_corr if avg_corr > 0 else 1.0
                    stressed_vol = baseline_vol * np.sqrt(corr_effect)
                    stress_results[f"{scenario_name}_volatility"] = float(stressed_vol)
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"Stress testing failed: {e}")
            return {}
    
    async def _perform_scenario_analysis(self, 
                                       weights: Dict[str, float],
                                       returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Perform scenario analysis."""
        try:
            scenarios = {
                'bull_market': {'return_multiplier': 1.5, 'vol_multiplier': 0.8},
                'bear_market': {'return_multiplier': -0.5, 'vol_multiplier': 1.5},
                'crisis': {'return_multiplier': -1.0, 'vol_multiplier': 2.0},
                'recovery': {'return_multiplier': 1.2, 'vol_multiplier': 1.2}
            }
            
            scenario_results = {}
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            baseline_returns = returns @ weight_array
            
            for scenario_name, scenario_params in scenarios.items():
                scenario_result = {}
                
                # Calculate scenario return
                baseline_return = baseline_returns.mean() * 252  # Annualized
                scenario_return = baseline_return * scenario_params['return_multiplier']
                scenario_result['expected_return'] = float(scenario_return)
                
                # Calculate scenario volatility
                baseline_vol = baseline_returns.std() * np.sqrt(252)
                scenario_vol = baseline_vol * scenario_params['vol_multiplier']
                scenario_result['volatility'] = float(scenario_vol)
                
                # Calculate scenario Sharpe ratio
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                scenario_sharpe = (scenario_return - risk_free_rate) / scenario_vol if scenario_vol > 0 else 0
                scenario_result['sharpe_ratio'] = float(scenario_sharpe)
                
                scenario_results[scenario_name] = scenario_result
            
            return scenario_results
            
        except Exception as e:
            self.logger.error(f"Scenario analysis failed: {e}")
            return {}
    
    async def _calculate_risk_adjusted_performance(self, 
                                                 returns: pd.DataFrame,
                                                 weights: Dict[str, float],
                                                 portfolio_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        try:
            performance_metrics = {}
            
            # Get portfolio returns
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = returns @ weight_array
            
            # Basic metrics
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            
            # Risk-adjusted returns
            risk_free_rate = 0.02
            
            # Sharpe ratio
            sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            performance_metrics['sharpe_ratio'] = float(sharpe_ratio)
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_vol
            sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
            performance_metrics['sortino_ratio'] = float(sortino_ratio)
            
            # Calmar ratio (return / max drawdown)
            max_dd = abs(portfolio_metrics.get('max_drawdown', 0.01))
            calmar_ratio = annual_return / max_dd if max_dd > 0 else 0
            performance_metrics['calmar_ratio'] = float(calmar_ratio)
            
            # Information ratio (assume benchmark return = 0)
            tracking_error = annual_vol  # Simplified
            information_ratio = annual_return / tracking_error if tracking_error > 0 else 0
            performance_metrics['information_ratio'] = float(information_ratio)
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Risk-adjusted performance calculation failed: {e}")
            return {}
    
    def get_risk_history(self, lookback_days: int = None) -> List[RiskReport]:
        """Get risk report history."""
        if lookback_days is None:
            return self.risk_history.copy()
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        return [
            report for report in self.risk_history 
            if report.report_date >= cutoff_date
        ]
    
    def get_constraint_violations(self) -> List[RiskConstraint]:
        """Get current constraint violations."""
        return self.constraint_violations.copy()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary."""
        if not self.risk_history:
            return {
                'total_reports': 0,
                'active_violations': 0,
                'last_report': None,
                'risk_trend': None
            }
        
        latest = self.risk_history[-1]
        
        return {
            'total_reports': len(self.risk_history),
            'active_violations': len([c for c in latest.risk_constraints if c.is_violated]),
            'active_warnings': len(latest.risk_warnings),
            'last_report': latest.report_date,
            'portfolio_risk_level': latest.portfolio_risk_metrics.get('volatility', 0.0),
            'risk_budget_utilization': np.mean([b.budget_utilization for b in latest.risk_budgets]),
            'supported_budget_types': [bt.value for bt in RiskBudgetType],
            'supported_constraint_types': [ct.value for ct in RiskConstraintType]
        }