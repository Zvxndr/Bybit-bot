"""
Dynamic Parameter Optimization System

This module provides continuous parameter optimization capabilities that adapt
trading parameters based on market conditions and performance for the Bybit trading bot.

Key Features:
- Walk-forward optimization with rolling windows
- Parameter drift detection and monitoring
- Automatic recalibration triggers
- Multi-objective optimization (return, risk, drawdown)
- Market regime-aware parameter adjustment
- Bayesian optimization for hyperparameter tuning
- Parameter stability analysis
- Performance attribution analysis

Author: Trading Bot Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import optuna
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

# Configure logging
logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Parameter optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    WALK_FORWARD = "walk_forward"
    ROLLING_WINDOW = "rolling_window"


class ObjectiveFunction(Enum):
    """Optimization objective functions"""
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_RETURN = "max_return"
    MIN_VOLATILITY = "min_volatility"
    MIN_DRAWDOWN = "min_drawdown"
    CUSTOM_UTILITY = "custom_utility"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class ParameterBounds:
    """Parameter bounds for optimization"""
    name: str
    min_value: float
    max_value: float
    step_size: Optional[float] = None
    parameter_type: str = 'continuous'  # 'continuous', 'discrete', 'categorical'
    values: Optional[List] = None  # For categorical parameters


@dataclass
class OptimizationResult:
    """Optimization result data structure"""
    optimal_parameters: Dict[str, Any]
    objective_value: float
    optimization_method: str
    optimization_time: float
    iterations: int
    convergence_achieved: bool
    parameter_stability: Dict[str, float]
    performance_metrics: Dict[str, float]
    confidence_interval: Optional[Tuple[float, float]] = None
    parameter_sensitivity: Optional[Dict[str, float]] = None


@dataclass
class ParameterDrift:
    """Parameter drift detection result"""
    parameter_name: str
    current_value: Any
    historical_mean: float
    drift_score: float
    drift_threshold: float
    is_drifting: bool
    drift_direction: str  # 'increasing', 'decreasing', 'stable'
    confidence: float


class ParameterOptimizer:
    """
    Dynamic Parameter Optimization System
    
    This class provides comprehensive parameter optimization with
    continuous adaptation and drift detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the parameter optimizer
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config or self._get_default_config()
        self.optimization_history = []
        self.parameter_history = {}
        self.performance_data = {}
        self.drift_detectors = {}
        self.current_parameters = {}
        self.optimization_cache = {}
        
        # Initialize scalers for normalization
        self.parameter_scaler = StandardScaler()
        self.objective_scaler = StandardScaler()
        
        logger.info("ParameterOptimizer initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for parameter optimization"""
        return {
            'optimization': {
                'method': 'bayesian',
                'max_iterations': 100,
                'convergence_tolerance': 1e-6,
                'improvement_threshold': 0.01,
                'n_jobs': -1,
                'random_state': 42
            },
            'walk_forward': {
                'training_window_days': 252,
                'test_window_days': 63,
                'step_size_days': 21,
                'minimum_trades': 50,
                'performance_threshold': 0.1
            },
            'drift_detection': {
                'lookback_window': 50,
                'drift_threshold': 2.0,  # Standard deviations
                'stability_threshold': 0.15,
                'monitoring_frequency': 'daily'
            },
            'objectives': {
                'primary': 'sharpe_ratio',
                'secondary': 'calmar_ratio',
                'risk_penalty': 0.1,
                'drawdown_penalty': 0.2,
                'volatility_penalty': 0.05
            },
            'constraints': {
                'max_drawdown': 0.25,
                'min_sharpe_ratio': 0.5,
                'max_volatility': 0.40,
                'min_trades_per_month': 10
            },
            'bayesian': {
                'n_initial_points': 10,
                'acquisition_function': 'ei',  # 'ei', 'pi', 'ucb'
                'kernel': 'matern',
                'alpha': 1e-6,
                'normalize_y': True
            }
        }
    
    def define_parameter_space(self, parameter_bounds: List[ParameterBounds]) -> Dict[str, Any]:
        """
        Define the parameter space for optimization
        
        Args:
            parameter_bounds: List of parameter bounds
            
        Returns:
            Dictionary defining the parameter space
        """
        try:
            parameter_space = {}
            
            for bound in parameter_bounds:
                if bound.parameter_type == 'continuous':
                    parameter_space[bound.name] = {
                        'type': 'continuous',
                        'bounds': (bound.min_value, bound.max_value),
                        'step_size': bound.step_size
                    }
                elif bound.parameter_type == 'discrete':
                    if bound.step_size:
                        values = np.arange(bound.min_value, bound.max_value + bound.step_size, bound.step_size)
                        parameter_space[bound.name] = {
                            'type': 'discrete',
                            'values': values.tolist()
                        }
                    else:
                        parameter_space[bound.name] = {
                            'type': 'discrete',
                            'bounds': (int(bound.min_value), int(bound.max_value))
                        }
                elif bound.parameter_type == 'categorical':
                    parameter_space[bound.name] = {
                        'type': 'categorical',
                        'values': bound.values or []
                    }
            
            self.parameter_space = parameter_space
            logger.info(f"Defined parameter space with {len(parameter_bounds)} parameters")
            return parameter_space
            
        except Exception as e:
            logger.error(f"Error defining parameter space: {e}")
            return {}
    
    def optimize_parameters(self, 
                          objective_function: Callable,
                          parameter_space: Dict[str, Any],
                          method: OptimizationMethod = OptimizationMethod.BAYESIAN,
                          max_iterations: int = None) -> OptimizationResult:
        """
        Optimize parameters using specified method
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            method: Optimization method
            max_iterations: Maximum iterations (overrides config)
            
        Returns:
            OptimizationResult object
        """
        try:
            start_time = datetime.now()
            max_iter = max_iterations or self.config['optimization']['max_iterations']
            
            logger.info(f"Starting parameter optimization using {method.value}")
            
            if method == OptimizationMethod.BAYESIAN:
                result = self._optimize_bayesian(objective_function, parameter_space, max_iter)
            elif method == OptimizationMethod.GRID_SEARCH:
                result = self._optimize_grid_search(objective_function, parameter_space)
            elif method == OptimizationMethod.RANDOM_SEARCH:
                result = self._optimize_random_search(objective_function, parameter_space, max_iter)
            elif method == OptimizationMethod.GENETIC_ALGORITHM:
                result = self._optimize_genetic(objective_function, parameter_space, max_iter)
            elif method == OptimizationMethod.WALK_FORWARD:
                result = self._optimize_walk_forward(objective_function, parameter_space)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate parameter stability
            stability = self._calculate_parameter_stability(result['optimal_parameters'])
            
            # Calculate sensitivity analysis
            sensitivity = self._calculate_parameter_sensitivity(
                objective_function, 
                result['optimal_parameters'], 
                parameter_space
            )
            
            # Create optimization result
            opt_result = OptimizationResult(
                optimal_parameters=result['optimal_parameters'],
                objective_value=result['objective_value'],
                optimization_method=method.value,
                optimization_time=optimization_time,
                iterations=result.get('iterations', max_iter),
                convergence_achieved=result.get('converged', False),
                parameter_stability=stability,
                performance_metrics=result.get('performance_metrics', {}),
                parameter_sensitivity=sensitivity
            )
            
            # Store in history
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'result': opt_result,
                'method': method.value
            })
            
            # Update current parameters
            self.current_parameters.update(result['optimal_parameters'])
            
            logger.info(f"Optimization completed in {optimization_time:.2f} seconds. "
                       f"Objective value: {result['objective_value']:.4f}")
            
            return opt_result
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return OptimizationResult(
                optimal_parameters={},
                objective_value=0.0,
                optimization_method=method.value,
                optimization_time=0.0,
                iterations=0,
                convergence_achieved=False,
                parameter_stability={},
                performance_metrics={}
            )
    
    def _optimize_bayesian(self, objective_function: Callable, parameter_space: Dict, max_iterations: int) -> Dict:
        """Bayesian optimization using Optuna"""
        try:
            def optuna_objective(trial):
                params = {}
                for param_name, param_config in parameter_space.items():
                    if param_config['type'] == 'continuous':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['bounds'][0],
                            param_config['bounds'][1]
                        )
                    elif param_config['type'] == 'discrete':
                        if 'values' in param_config:
                            params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
                        else:
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config['bounds'][0],
                                param_config['bounds'][1]
                            )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
                
                return objective_function(params)
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.config['optimization']['random_state'])
            )
            
            # Optimize
            study.optimize(optuna_objective, n_trials=max_iterations, show_progress_bar=False)
            
            return {
                'optimal_parameters': study.best_params,
                'objective_value': study.best_value,
                'iterations': len(study.trials),
                'converged': study.best_trial.state == optuna.trial.TrialState.COMPLETE
            }
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return {
                'optimal_parameters': {},
                'objective_value': 0.0,
                'iterations': 0,
                'converged': False
            }
    
    def _optimize_grid_search(self, objective_function: Callable, parameter_space: Dict) -> Dict:
        """Grid search optimization"""
        try:
            # Create parameter grid
            param_grid = {}
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'continuous':
                    # Create grid points for continuous parameters
                    min_val, max_val = param_config['bounds']
                    n_points = 10  # Default grid points
                    param_grid[param_name] = np.linspace(min_val, max_val, n_points)
                elif param_config['type'] == 'discrete':
                    if 'values' in param_config:
                        param_grid[param_name] = param_config['values']
                    else:
                        min_val, max_val = param_config['bounds']
                        param_grid[param_name] = list(range(min_val, max_val + 1))
                elif param_config['type'] == 'categorical':
                    param_grid[param_name] = param_config['values']
            
            # Generate all parameter combinations
            grid = ParameterGrid(param_grid)
            
            best_params = None
            best_score = -np.inf
            
            for params in grid:
                try:
                    score = objective_function(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                except Exception as e:
                    logger.warning(f"Error evaluating parameters {params}: {e}")
                    continue
            
            return {
                'optimal_parameters': best_params or {},
                'objective_value': best_score,
                'iterations': len(list(grid)),
                'converged': True
            }
            
        except Exception as e:
            logger.error(f"Error in grid search optimization: {e}")
            return {
                'optimal_parameters': {},
                'objective_value': 0.0,
                'iterations': 0,
                'converged': False
            }
    
    def _optimize_random_search(self, objective_function: Callable, parameter_space: Dict, max_iterations: int) -> Dict:
        """Random search optimization"""
        try:
            best_params = None
            best_score = -np.inf
            
            np.random.seed(self.config['optimization']['random_state'])
            
            for i in range(max_iterations):
                params = {}
                
                for param_name, param_config in parameter_space.items():
                    if param_config['type'] == 'continuous':
                        min_val, max_val = param_config['bounds']
                        params[param_name] = np.random.uniform(min_val, max_val)
                    elif param_config['type'] == 'discrete':
                        if 'values' in param_config:
                            params[param_name] = np.random.choice(param_config['values'])
                        else:
                            min_val, max_val = param_config['bounds']
                            params[param_name] = np.random.randint(min_val, max_val + 1)
                    elif param_config['type'] == 'categorical':
                        params[param_name] = np.random.choice(param_config['values'])
                
                try:
                    score = objective_function(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                except Exception as e:
                    logger.warning(f"Error evaluating parameters {params}: {e}")
                    continue
            
            return {
                'optimal_parameters': best_params or {},
                'objective_value': best_score,
                'iterations': max_iterations,
                'converged': True
            }
            
        except Exception as e:
            logger.error(f"Error in random search optimization: {e}")
            return {
                'optimal_parameters': {},
                'objective_value': 0.0,
                'iterations': 0,
                'converged': False
            }
    
    def _optimize_genetic(self, objective_function: Callable, parameter_space: Dict, max_iterations: int) -> Dict:
        """Genetic algorithm optimization using scipy"""
        try:
            # Convert parameter space to bounds for scipy
            bounds = []
            param_names = []
            
            for param_name, param_config in parameter_space.items():
                if param_config['type'] in ['continuous', 'discrete']:
                    bounds.append(param_config['bounds'])
                    param_names.append(param_name)
            
            if not bounds:
                raise ValueError("No continuous or discrete parameters for genetic algorithm")
            
            def scipy_objective(x):
                params = {name: value for name, value in zip(param_names, x)}
                return -objective_function(params)  # Scipy minimizes, so negate
            
            result = differential_evolution(
                scipy_objective,
                bounds,
                maxiter=max_iterations,
                seed=self.config['optimization']['random_state'],
                workers=1  # Avoid multiprocessing issues
            )
            
            optimal_params = {name: value for name, value in zip(param_names, result.x)}
            
            return {
                'optimal_parameters': optimal_params,
                'objective_value': -result.fun,  # Convert back to maximization
                'iterations': result.nit,
                'converged': result.success
            }
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            return {
                'optimal_parameters': {},
                'objective_value': 0.0,
                'iterations': 0,
                'converged': False
            }
    
    def _optimize_walk_forward(self, objective_function: Callable, parameter_space: Dict) -> Dict:
        """Walk-forward optimization"""
        try:
            # This would require historical data and is more complex
            # For now, fall back to Bayesian optimization
            logger.info("Walk-forward optimization requested, falling back to Bayesian optimization")
            return self._optimize_bayesian(objective_function, parameter_space, 50)
            
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            return {
                'optimal_parameters': {},
                'objective_value': 0.0,
                'iterations': 0,
                'converged': False
            }
    
    def detect_parameter_drift(self, parameter_name: str, current_value: Any, significance_level: float = 0.05) -> ParameterDrift:
        """
        Detect parameter drift using statistical methods
        
        Args:
            parameter_name: Name of the parameter
            current_value: Current parameter value
            significance_level: Statistical significance level
            
        Returns:
            ParameterDrift object
        """
        try:
            # Get historical values
            if parameter_name not in self.parameter_history:
                self.parameter_history[parameter_name] = []
            
            history = self.parameter_history[parameter_name]
            
            # Add current value to history
            history.append({
                'timestamp': datetime.now(),
                'value': current_value
            })
            
            # Keep only recent history
            lookback_window = self.config['drift_detection']['lookback_window']
            if len(history) > lookback_window:
                history = history[-lookback_window:]
                self.parameter_history[parameter_name] = history
            
            # Need sufficient history for drift detection
            if len(history) < 10:
                return ParameterDrift(
                    parameter_name=parameter_name,
                    current_value=current_value,
                    historical_mean=current_value,
                    drift_score=0.0,
                    drift_threshold=self.config['drift_detection']['drift_threshold'],
                    is_drifting=False,
                    drift_direction='stable',
                    confidence=0.0
                )
            
            # Extract values
            values = [h['value'] for h in history if isinstance(h['value'], (int, float))]
            
            if len(values) < 5:
                return ParameterDrift(
                    parameter_name=parameter_name,
                    current_value=current_value,
                    historical_mean=current_value,
                    drift_score=0.0,
                    drift_threshold=self.config['drift_detection']['drift_threshold'],
                    is_drifting=False,
                    drift_direction='stable',
                    confidence=0.0
                )
            
            # Calculate statistics
            historical_mean = np.mean(values[:-1])  # Exclude current value
            historical_std = np.std(values[:-1])
            
            # Drift score (standardized difference)
            if historical_std > 0:
                drift_score = abs(current_value - historical_mean) / historical_std
            else:
                drift_score = 0.0
            
            # Determine drift direction
            if current_value > historical_mean + historical_std:
                drift_direction = 'increasing'
            elif current_value < historical_mean - historical_std:
                drift_direction = 'decreasing'
            else:
                drift_direction = 'stable'
            
            # Statistical test for drift (using t-test)
            if len(values) >= 10:
                # Split into two halves
                mid_point = len(values) // 2
                first_half = values[:mid_point]
                second_half = values[mid_point:]
                
                if len(first_half) > 1 and len(second_half) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(first_half, second_half)
                        confidence = 1 - p_value
                    except:
                        confidence = 0.0
                else:
                    confidence = 0.0
            else:
                confidence = 0.0
            
            # Determine if drifting
            drift_threshold = self.config['drift_detection']['drift_threshold']
            is_drifting = drift_score > drift_threshold and confidence > (1 - significance_level)
            
            drift_result = ParameterDrift(
                parameter_name=parameter_name,
                current_value=current_value,
                historical_mean=float(historical_mean),
                drift_score=float(drift_score),
                drift_threshold=drift_threshold,
                is_drifting=is_drifting,
                drift_direction=drift_direction,
                confidence=float(confidence)
            )
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Error detecting parameter drift for {parameter_name}: {e}")
            return ParameterDrift(
                parameter_name=parameter_name,
                current_value=current_value,
                historical_mean=current_value,
                drift_score=0.0,
                drift_threshold=2.0,
                is_drifting=False,
                drift_direction='stable',
                confidence=0.0
            )
    
    def should_reoptimize(self, 
                         current_performance: Dict[str, float],
                         drift_results: List[ParameterDrift] = None) -> Tuple[bool, str]:
        """
        Determine if parameters should be reoptimized
        
        Args:
            current_performance: Current performance metrics
            drift_results: List of parameter drift results
            
        Returns:
            Tuple of (should_reoptimize, reason)
        """
        try:
            reasons = []
            
            # Check performance degradation
            if len(self.optimization_history) > 0:
                last_optimization = self.optimization_history[-1]
                last_performance = last_optimization['result'].performance_metrics
                
                # Check Sharpe ratio degradation
                if 'sharpe_ratio' in current_performance and 'sharpe_ratio' in last_performance:
                    sharpe_degradation = (last_performance['sharpe_ratio'] - current_performance['sharpe_ratio']) / last_performance['sharpe_ratio']
                    if sharpe_degradation > 0.2:  # 20% degradation
                        reasons.append(f"Sharpe ratio degraded by {sharpe_degradation:.1%}")
                
                # Check drawdown increase
                if 'max_drawdown' in current_performance and 'max_drawdown' in last_performance:
                    drawdown_increase = current_performance['max_drawdown'] - last_performance['max_drawdown']
                    if drawdown_increase > 0.05:  # 5% absolute increase
                        reasons.append(f"Maximum drawdown increased by {drawdown_increase:.1%}")
            
            # Check parameter drift
            if drift_results:
                drifting_params = [d for d in drift_results if d.is_drifting]
                if len(drifting_params) >= 2:  # Multiple parameters drifting
                    param_names = [d.parameter_name for d in drifting_params]
                    reasons.append(f"Multiple parameters drifting: {', '.join(param_names)}")
                elif len(drifting_params) == 1 and drifting_params[0].confidence > 0.95:
                    reasons.append(f"High-confidence drift in {drifting_params[0].parameter_name}")
            
            # Check time since last optimization
            if len(self.optimization_history) > 0:
                last_optimization_time = self.optimization_history[-1]['timestamp']
                days_since_optimization = (datetime.now() - last_optimization_time).days
                
                if days_since_optimization > 30:  # Monthly reoptimization
                    reasons.append(f"Time-based reoptimization (last: {days_since_optimization} days ago)")
            
            # Check constraint violations
            constraints = self.config['constraints']
            for constraint_name, constraint_value in constraints.items():
                if constraint_name in current_performance:
                    if constraint_name.startswith('max_') and current_performance[constraint_name] > constraint_value:
                        reasons.append(f"Constraint violation: {constraint_name} = {current_performance[constraint_name]:.3f} > {constraint_value}")
                    elif constraint_name.startswith('min_') and current_performance[constraint_name] < constraint_value:
                        reasons.append(f"Constraint violation: {constraint_name} = {current_performance[constraint_name]:.3f} < {constraint_value}")
            
            should_reoptimize = len(reasons) > 0
            reason = "; ".join(reasons) if reasons else "No reoptimization needed"
            
            return should_reoptimize, reason
            
        except Exception as e:
            logger.error(f"Error determining reoptimization need: {e}")
            return False, "Error in reoptimization check"
    
    def _calculate_parameter_stability(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate parameter stability scores"""
        try:
            stability_scores = {}
            
            for param_name, param_value in parameters.items():
                if param_name in self.parameter_history:
                    history = self.parameter_history[param_name]
                    values = [h['value'] for h in history if isinstance(h['value'], (int, float))]
                    
                    if len(values) > 3:
                        # Calculate coefficient of variation as stability measure
                        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                        stability_score = 1 / (1 + cv)  # Higher is more stable
                        stability_scores[param_name] = float(stability_score)
                    else:
                        stability_scores[param_name] = 0.5  # Neutral for insufficient data
                else:
                    stability_scores[param_name] = 0.0  # No history
            
            return stability_scores
            
        except Exception as e:
            logger.error(f"Error calculating parameter stability: {e}")
            return {}
    
    def _calculate_parameter_sensitivity(self, 
                                       objective_function: Callable,
                                       optimal_parameters: Dict[str, Any],
                                       parameter_space: Dict[str, Any]) -> Dict[str, float]:
        """Calculate parameter sensitivity analysis"""
        try:
            sensitivity_scores = {}
            base_score = objective_function(optimal_parameters)
            
            for param_name, param_config in parameter_space.items():
                if param_name not in optimal_parameters:
                    continue
                
                if param_config['type'] == 'continuous':
                    base_value = optimal_parameters[param_name]
                    min_val, max_val = param_config['bounds']
                    
                    # Test small perturbations
                    perturbation = (max_val - min_val) * 0.01  # 1% of range
                    
                    # Test positive perturbation
                    test_params = optimal_parameters.copy()
                    test_params[param_name] = min(max_val, base_value + perturbation)
                    try:
                        score_up = objective_function(test_params)
                    except:
                        score_up = base_score
                    
                    # Test negative perturbation
                    test_params[param_name] = max(min_val, base_value - perturbation)
                    try:
                        score_down = objective_function(test_params)
                    except:
                        score_down = base_score
                    
                    # Calculate sensitivity (derivative approximation)
                    if perturbation != 0:
                        sensitivity = abs((score_up - score_down) / (2 * perturbation))
                        sensitivity_scores[param_name] = float(sensitivity)
                    else:
                        sensitivity_scores[param_name] = 0.0
                else:
                    # For discrete/categorical parameters, use different approach
                    sensitivity_scores[param_name] = 0.5  # Default moderate sensitivity
            
            return sensitivity_scores
            
        except Exception as e:
            logger.error(f"Error calculating parameter sensitivity: {e}")
            return {}
    
    def create_multi_objective_function(self, 
                                      performance_function: Callable,
                                      weights: Dict[str, float] = None) -> Callable:
        """
        Create multi-objective function for optimization
        
        Args:
            performance_function: Function that returns performance metrics
            weights: Weights for different objectives
            
        Returns:
            Multi-objective function
        """
        try:
            default_weights = {
                'return': 0.4,
                'sharpe_ratio': 0.3,
                'max_drawdown': -0.2,  # Negative because we want to minimize
                'volatility': -0.1     # Negative because we want to minimize
            }
            
            objective_weights = weights or default_weights
            
            def multi_objective(parameters: Dict[str, Any]) -> float:
                try:
                    metrics = performance_function(parameters)
                    
                    if not isinstance(metrics, dict):
                        return 0.0
                    
                    objective_value = 0.0
                    
                    for metric_name, weight in objective_weights.items():
                        if metric_name in metrics:
                            metric_value = metrics[metric_name]
                            
                            # Normalize metrics to similar scales
                            if metric_name == 'return':
                                normalized_value = metric_value  # Already in reasonable range
                            elif metric_name == 'sharpe_ratio':
                                normalized_value = metric_value / 3.0  # Scale to ~0-1 range
                            elif metric_name == 'max_drawdown':
                                normalized_value = metric_value  # Already negative
                            elif metric_name == 'volatility':
                                normalized_value = -metric_value  # Make negative for minimization
                            else:
                                normalized_value = metric_value
                            
                            objective_value += weight * normalized_value
                    
                    return objective_value
                    
                except Exception as e:
                    logger.warning(f"Error in multi-objective evaluation: {e}")
                    return -999.0  # Large negative penalty for errors
            
            return multi_objective
            
        except Exception as e:
            logger.error(f"Error creating multi-objective function: {e}")
            return lambda x: 0.0
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history and current state"""
        try:
            if not self.optimization_history:
                return {'message': 'No optimization history available'}
            
            # Get recent optimizations
            recent_optimizations = self.optimization_history[-10:]
            
            # Calculate statistics
            objective_values = [opt['result'].objective_value for opt in recent_optimizations]
            optimization_times = [opt['result'].optimization_time for opt in recent_optimizations]
            
            # Get parameter stability
            current_stability = {}
            if self.current_parameters:
                current_stability = self._calculate_parameter_stability(self.current_parameters)
            
            # Drift detection summary
            drift_summary = {}
            for param_name in self.current_parameters:
                if param_name in self.parameter_history:
                    drift_result = self.detect_parameter_drift(param_name, self.current_parameters[param_name])
                    drift_summary[param_name] = {
                        'is_drifting': drift_result.is_drifting,
                        'drift_score': drift_result.drift_score,
                        'direction': drift_result.drift_direction
                    }
            
            summary = {
                'total_optimizations': len(self.optimization_history),
                'recent_optimizations': len(recent_optimizations),
                'best_objective_value': float(max(objective_values)) if objective_values else 0.0,
                'latest_objective_value': float(objective_values[-1]) if objective_values else 0.0,
                'average_optimization_time': float(np.mean(optimization_times)) if optimization_times else 0.0,
                'current_parameters': self.current_parameters,
                'parameter_stability': current_stability,
                'drift_summary': drift_summary,
                'last_optimization': recent_optimizations[-1]['timestamp'].isoformat() if recent_optimizations else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {'error': str(e)}
    
    def save_optimization_state(self, filepath: str) -> bool:
        """Save optimization state to file"""
        try:
            import pickle
            
            state = {
                'optimization_history': self.optimization_history,
                'parameter_history': self.parameter_history,
                'current_parameters': self.current_parameters,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Optimization state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving optimization state: {e}")
            return False
    
    def load_optimization_state(self, filepath: str) -> bool:
        """Load optimization state from file"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.optimization_history = state.get('optimization_history', [])
            self.parameter_history = state.get('parameter_history', {})
            self.current_parameters = state.get('current_parameters', {})
            
            # Update config with loaded values
            loaded_config = state.get('config', {})
            self.config.update(loaded_config)
            
            logger.info(f"Optimization state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading optimization state: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test the parameter optimizer
    print("Testing Dynamic Parameter Optimization System")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer()
    
    # Define sample parameter space
    parameter_bounds = [
        ParameterBounds("lookback_period", 10, 100, 1, "discrete"),
        ParameterBounds("risk_multiplier", 0.5, 3.0, None, "continuous"),
        ParameterBounds("stop_loss", 0.01, 0.1, None, "continuous"),
        ParameterBounds("take_profit", 0.02, 0.2, None, "continuous"),
        ParameterBounds("strategy_type", None, None, None, "categorical", ["momentum", "mean_reversion", "trend"])
    ]
    
    parameter_space = optimizer.define_parameter_space(parameter_bounds)
    print(f"Defined parameter space with {len(parameter_space)} parameters")
    
    # Sample objective function (Sharpe ratio simulation)
    def sample_objective_function(params):
        try:
            # Simulate strategy performance based on parameters
            lookback = params.get('lookback_period', 50)
            risk_mult = params.get('risk_multiplier', 1.0)
            stop_loss = params.get('stop_loss', 0.05)
            take_profit = params.get('take_profit', 0.1)
            strategy_type = params.get('strategy_type', 'momentum')
            
            # Simulate returns based on parameters
            np.random.seed(42)  # For reproducible results
            
            # Strategy-specific base performance
            strategy_multipliers = {
                'momentum': 1.2,
                'mean_reversion': 0.9,
                'trend': 1.1
            }
            
            base_return = 0.1 * strategy_multipliers.get(strategy_type, 1.0)
            base_volatility = 0.15
            
            # Parameter effects
            lookback_effect = 1.0 - abs(lookback - 50) / 100  # Optimal around 50
            risk_effect = 1.0 - abs(risk_mult - 1.5) / 2.0    # Optimal around 1.5
            stop_loss_effect = 1.0 - abs(stop_loss - 0.03) / 0.05  # Optimal around 3%
            take_profit_effect = 1.0 - abs(take_profit - 0.06) / 0.1  # Optimal around 6%
            
            # Calculate adjusted metrics
            adjusted_return = base_return * lookback_effect * risk_effect * stop_loss_effect * take_profit_effect
            adjusted_volatility = base_volatility * (1 + (risk_mult - 1) * 0.2)
            
            # Add some noise
            noise = np.random.normal(0, 0.02)
            adjusted_return += noise
            
            # Calculate Sharpe ratio
            sharpe_ratio = adjusted_return / adjusted_volatility if adjusted_volatility > 0 else 0
            
            # Add penalty for extreme parameters
            penalty = 0
            if risk_mult > 2.5:
                penalty += (risk_mult - 2.5) * 0.5
            if stop_loss > 0.08:
                penalty += (stop_loss - 0.08) * 2
            
            return max(0, sharpe_ratio - penalty)
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 0.0
    
    # Test different optimization methods
    methods = [
        OptimizationMethod.BAYESIAN,
        OptimizationMethod.RANDOM_SEARCH,
        OptimizationMethod.GRID_SEARCH
    ]
    
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.value} optimization:")
        
        max_iter = 50 if method != OptimizationMethod.GRID_SEARCH else None
        result = optimizer.optimize_parameters(sample_objective_function, parameter_space, method, max_iter)
        
        results[method.value] = result
        
        print(f"  Optimal Parameters: {result.optimal_parameters}")
        print(f"  Objective Value: {result.objective_value:.4f}")
        print(f"  Optimization Time: {result.optimization_time:.2f}s")
        print(f"  Converged: {result.convergence_achieved}")
        print(f"  Iterations: {result.iterations}")
        
        if result.parameter_sensitivity:
            print("  Parameter Sensitivity:")
            for param, sensitivity in result.parameter_sensitivity.items():
                print(f"    {param}: {sensitivity:.4f}")
    
    # Test drift detection
    print(f"\nTesting Parameter Drift Detection:")
    print("-" * 30)
    
    # Simulate parameter evolution
    test_param = "risk_multiplier"
    base_value = 1.5
    
    # Add some history
    for i in range(20):
        # Simulate gradual drift
        drift_value = base_value + (i * 0.02) + np.random.normal(0, 0.05)
        drift_result = optimizer.detect_parameter_drift(test_param, drift_value)
        
        if i % 5 == 0:  # Print every 5th iteration
            print(f"  Step {i}: Value={drift_value:.3f}, Drift Score={drift_result.drift_score:.3f}, "
                  f"Drifting={drift_result.is_drifting}, Direction={drift_result.drift_direction}")
    
    # Test reoptimization decision
    print(f"\nTesting Reoptimization Decision:")
    print("-" * 30)
    
    # Simulate performance degradation
    current_performance = {
        'sharpe_ratio': 0.8,  # Lower than optimal
        'max_drawdown': 0.15,
        'return': 0.06
    }
    
    should_reopt, reason = optimizer.should_reoptimize(current_performance)
    print(f"Should Reoptimize: {should_reopt}")
    print(f"Reason: {reason}")
    
    # Test multi-objective optimization
    print(f"\nTesting Multi-Objective Optimization:")
    print("-" * 30)
    
    def sample_performance_function(params):
        # Simulate multiple performance metrics
        sharpe = sample_objective_function(params)
        return {
            'return': sharpe * 0.15,  # Approximate return
            'sharpe_ratio': sharpe,
            'max_drawdown': -0.1 - (sharpe - 1) * 0.05,  # Better sharpe = lower drawdown
            'volatility': 0.15 + max(0, 2 - sharpe) * 0.05  # Better sharpe = lower volatility
        }
    
    multi_obj_func = optimizer.create_multi_objective_function(sample_performance_function)
    mo_result = optimizer.optimize_parameters(multi_obj_func, parameter_space, OptimizationMethod.BAYESIAN, 30)
    
    print(f"Multi-Objective Optimal Parameters: {mo_result.optimal_parameters}")
    print(f"Multi-Objective Value: {mo_result.objective_value:.4f}")
    
    # Get optimization summary
    print(f"\nOptimization Summary:")
    print("-" * 20)
    summary = optimizer.get_optimization_summary()
    
    print(f"Total Optimizations: {summary.get('total_optimizations', 0)}")
    print(f"Best Objective Value: {summary.get('best_objective_value', 0):.4f}")
    print(f"Latest Objective Value: {summary.get('latest_objective_value', 0):.4f}")
    print(f"Average Optimization Time: {summary.get('average_optimization_time', 0):.2f}s")
    
    if summary.get('parameter_stability'):
        print("Parameter Stability:")
        for param, stability in summary['parameter_stability'].items():
            print(f"  {param}: {stability:.3f}")
    
    print(f"\nDynamic Parameter Optimization Testing Complete!")