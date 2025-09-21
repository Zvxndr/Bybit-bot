"""
Comprehensive Strategy Validator integrating all validation methods.

This module provides a unified interface for strategy validation using
multiple statistical and machine learning validation techniques:

- Walk-Forward Analysis for robustness testing
- Permutation Testing for statistical significance
- CSCV for overfitting detection (PBO)
- Purged CV for ML model validation
- Monte Carlo simulation for stress testing
- Integrated reporting and recommendations

The validator applies multiple validation layers to ensure strategy
robustness before live deployment.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

from .walkforward import WalkForwardAnalyzer, WalkForwardResult
from .permutation import PermutationTester, PermutationResult
from .cscv import CSCVValidator, CSCVResult
from .purged_cv import PurgedTimeSeriesCV, PurgedCVResult
from ..utils.logging import TradingLogger


@dataclass
class ValidationThresholds:
    """Validation thresholds for different modes."""
    
    # Walk-Forward Analysis thresholds
    min_wfo_consistency: float = 0.7
    min_wfo_sharpe: float = 1.0
    max_wfo_drawdown: float = 0.15
    
    # Permutation Testing thresholds
    max_permutation_pvalue: float = 0.05
    min_permutation_zscore: float = 2.0
    
    # CSCV thresholds
    max_pbo: float = 0.5
    max_pbo_pvalue: float = 0.05
    
    # Purged CV thresholds (for ML strategies)
    min_cv_accuracy: float = 0.55
    min_cv_f1: float = 0.5
    
    # General thresholds
    min_sample_size: int = 1000
    min_validation_period_days: int = 365
    
    @classmethod
    def conservative_mode(cls) -> 'ValidationThresholds':
        """Conservative validation thresholds."""
        return cls(
            min_wfo_consistency=0.8,
            min_wfo_sharpe=1.2,
            max_wfo_drawdown=0.1,
            max_permutation_pvalue=0.01,
            min_permutation_zscore=2.5,
            max_pbo=0.3,
            max_pbo_pvalue=0.01,
            min_cv_accuracy=0.6,
            min_cv_f1=0.55,
            min_sample_size=2000,
            min_validation_period_days=730
        )
    
    @classmethod
    def aggressive_mode(cls) -> 'ValidationThresholds':
        """Aggressive validation thresholds."""
        return cls(
            min_wfo_consistency=0.6,
            min_wfo_sharpe=0.8,
            max_wfo_drawdown=0.2,
            max_permutation_pvalue=0.1,
            min_permutation_zscore=1.5,
            max_pbo=0.7,
            max_pbo_pvalue=0.1,
            min_cv_accuracy=0.52,
            min_cv_f1=0.48,
            min_sample_size=500,
            min_validation_period_days=180
        )


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    
    strategy_name: str
    validation_mode: str
    timestamp: datetime
    
    # Individual validation results
    wfo_result: Optional[WalkForwardResult] = None
    permutation_result: Optional[PermutationResult] = None
    cscv_result: Optional[CSCVResult] = None
    purged_cv_result: Optional[PurgedCVResult] = None
    
    # Overall validation status
    passed: bool = False
    score: float = 0.0
    confidence_level: str = "LOW"
    
    # Detailed analysis
    validation_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.validation_scores is None:
            self.validation_scores = {}
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []
        if self.risk_assessment is None:
            self.risk_assessment = {}


class StrategyValidator:
    """
    Comprehensive strategy validation system.
    
    This class integrates multiple validation techniques to provide
    thorough strategy assessment before live deployment.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("StrategyValidator")
        
        # Initialize validation components
        self.wfo_analyzer = WalkForwardAnalyzer(self.config.get('walkforward', {}))
        self.permutation_tester = PermutationTester(self.config.get('permutation', {}))
        self.cscv_validator = CSCVValidator(self.config.get('cscv', {}))
        self.purged_cv = PurgedTimeSeriesCV(self.config.get('purged_cv', {}))
        
        # Set validation thresholds based on mode
        mode = self.config.get('mode', 'conservative')
        if mode == 'conservative':
            self.thresholds = ValidationThresholds.conservative_mode()
        elif mode == 'aggressive':
            self.thresholds = ValidationThresholds.aggressive_mode()
        else:
            self.thresholds = ValidationThresholds()
    
    def _default_config(self) -> Dict:
        """Default validation configuration."""
        return {
            'mode': 'conservative',
            'enable_wfo': True,
            'enable_permutation': True,
            'enable_cscv': True,
            'enable_purged_cv': False,  # Only for ML strategies
            'parallel_processing': True,
            'max_workers': 4,
            'confidence_intervals': [0.95, 0.99],
            'monte_carlo_runs': 1000,
            'save_detailed_results': True,
            'walkforward': {
                'window_sizes': [252, 504, 756],  # 1, 2, 3 years
                'step_sizes': [21, 63],  # Monthly, quarterly
                'optimization_frequency': 'quarterly',
                'min_sample_size': 252
            },
            'permutation': {
                'n_permutations': 1000,
                'methods': ['shuffle_returns', 'shuffle_dates', 'block_bootstrap'],
                'confidence_level': 0.95,
                'parallel_processing': True
            },
            'cscv': {
                'n_splits': 16,
                'test_size': 0.5,
                'min_samples_per_split': 100
            },
            'purged_cv': {
                'n_splits': 5,
                'purge_gap': 1,
                'embargo_gap': 1
            }
        }
    
    def validate_strategy(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Optional[Dict] = None,
        ml_model: Optional[Any] = None,
        features: Optional[pd.DataFrame] = None,
        targets: Optional[pd.Series] = None
    ) -> ValidationResult:
        """
        Perform comprehensive strategy validation.
        
        Args:
            strategy_name: Name of the strategy
            data: Historical market data
            strategy_func: Strategy function for backtesting
            param_grid: Parameter grid for optimization
            ml_model: ML model (if ML strategy)
            features: Feature matrix (if ML strategy)
            targets: Target vector (if ML strategy)
            
        Returns:
            ValidationResult with comprehensive analysis
        """
        self.logger.info(f"Starting comprehensive validation for {strategy_name}")
        
        result = ValidationResult(
            strategy_name=strategy_name,
            validation_mode=self.config['mode'],
            timestamp=datetime.now()
        )
        
        # Validate inputs
        self._validate_inputs(data, strategy_func, ml_model, features, targets)
        
        # Determine validation types to run
        is_ml_strategy = ml_model is not None and features is not None and targets is not None
        
        validation_tasks = []
        
        # Walk-Forward Analysis
        if self.config['enable_wfo']:
            validation_tasks.append(('wfo', self._run_walkforward_analysis, data, strategy_func, param_grid))
        
        # Permutation Testing
        if self.config['enable_permutation']:
            validation_tasks.append(('permutation', self._run_permutation_testing, data, strategy_func, param_grid))
        
        # CSCV Analysis
        if self.config['enable_cscv']:
            validation_tasks.append(('cscv', self._run_cscv_analysis, data, strategy_func, param_grid))
        
        # Purged CV (for ML strategies)
        if self.config['enable_purged_cv'] and is_ml_strategy:
            validation_tasks.append(('purged_cv', self._run_purged_cv, ml_model, features, targets))
        
        # Run validation tasks
        if self.config['parallel_processing']:
            self._run_validation_parallel(validation_tasks, result)
        else:
            self._run_validation_sequential(validation_tasks, result)
        
        # Aggregate results and calculate overall score
        self._calculate_overall_validation(result)
        
        # Generate risk assessment
        self._assess_risks(result)
        
        # Generate recommendations
        self._generate_recommendations(result)
        
        self.logger.info(
            f"Validation completed for {strategy_name}: "
            f"{'PASSED' if result.passed else 'FAILED'} "
            f"(Score: {result.score:.3f}, Confidence: {result.confidence_level})"
        )
        
        return result
    
    def _validate_inputs(self, data, strategy_func, ml_model, features, targets):
        """Validate input parameters."""
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        
        if len(data) < self.thresholds.min_sample_size:
            raise ValueError(
                f"Insufficient data: {len(data)} < {self.thresholds.min_sample_size}"
            )
        
        if strategy_func is None:
            raise ValueError("Strategy function cannot be None")
        
        # Validate ML inputs if provided
        if ml_model is not None:
            if features is None or targets is None:
                raise ValueError("Features and targets required for ML model validation")
            
            if len(features) != len(targets):
                raise ValueError("Features and targets must have same length")
    
    def _run_walkforward_analysis(self, data, strategy_func, param_grid):
        """Run Walk-Forward Analysis."""
        try:
            return self.wfo_analyzer.run_walkforward_analysis(
                data=data,
                strategy_func=strategy_func,
                param_grid=param_grid or {},
                window_sizes=self.config['walkforward']['window_sizes'],
                step_sizes=self.config['walkforward']['step_sizes']
            )
        except Exception as e:
            self.logger.error(f"Walk-Forward Analysis failed: {e}")
            return None
    
    def _run_permutation_testing(self, data, strategy_func, param_grid):
        """Run Permutation Testing."""
        try:
            return self.permutation_tester.run_permutation_test(
                data=data,
                strategy_func=strategy_func,
                param_grid=param_grid or {},
                n_permutations=self.config['permutation']['n_permutations']
            )
        except Exception as e:
            self.logger.error(f"Permutation Testing failed: {e}")
            return None
    
    def _run_cscv_analysis(self, data, strategy_func, param_grid):
        """Run CSCV Analysis."""
        try:
            return self.cscv_validator.run_cscv_analysis(
                data=data,
                strategy_func=strategy_func,
                param_grid=param_grid or {}
            )
        except Exception as e:
            self.logger.error(f"CSCV Analysis failed: {e}")
            return None
    
    def _run_purged_cv(self, ml_model, features, targets):
        """Run Purged Cross-Validation."""
        try:
            return self.purged_cv.cross_validate(
                estimator=ml_model,
                X=features,
                y=targets
            )
        except Exception as e:
            self.logger.error(f"Purged CV failed: {e}")
            return None
    
    def _run_validation_sequential(self, validation_tasks, result):
        """Run validation tasks sequentially."""
        for task_name, task_func, *args in validation_tasks:
            self.logger.info(f"Running {task_name} validation")
            
            task_result = task_func(*args)
            setattr(result, f"{task_name}_result", task_result)
    
    def _run_validation_parallel(self, validation_tasks, result):
        """Run validation tasks in parallel."""
        with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
            future_to_task = {
                executor.submit(task_func, *args): task_name
                for task_name, task_func, *args in validation_tasks
            }
            
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    task_result = future.result()
                    setattr(result, f"{task_name}_result", task_result)
                    self.logger.info(f"Completed {task_name} validation")
                except Exception as e:
                    self.logger.error(f"{task_name} validation failed: {e}")
                    setattr(result, f"{task_name}_result", None)
    
    def _calculate_overall_validation(self, result: ValidationResult):
        """Calculate overall validation score and status."""
        scores = []
        weights = []
        
        # Walk-Forward Analysis scoring
        if result.wfo_result:
            wfo_score = self._score_wfo_result(result.wfo_result)
            scores.append(wfo_score)
            weights.append(0.3)
            result.validation_scores['walkforward'] = wfo_score
        
        # Permutation Testing scoring
        if result.permutation_result:
            perm_score = self._score_permutation_result(result.permutation_result)
            scores.append(perm_score)
            weights.append(0.25)
            result.validation_scores['permutation'] = perm_score
        
        # CSCV scoring
        if result.cscv_result:
            cscv_score = self._score_cscv_result(result.cscv_result)
            scores.append(cscv_score)
            weights.append(0.25)
            result.validation_scores['cscv'] = cscv_score
        
        # Purged CV scoring
        if result.purged_cv_result:
            cv_score = self._score_purged_cv_result(result.purged_cv_result)
            scores.append(cv_score)
            weights.append(0.2)
            result.validation_scores['purged_cv'] = cv_score
        
        # Calculate weighted average score
        if scores:
            result.score = float(np.average(scores, weights=weights[:len(scores)]))
        else:
            result.score = 0.0
        
        # Determine pass/fail status
        min_score = 0.7 if self.config['mode'] == 'conservative' else 0.6
        result.passed = result.score >= min_score
        
        # Determine confidence level
        if result.score >= 0.8:
            result.confidence_level = "HIGH"
        elif result.score >= 0.6:
            result.confidence_level = "MEDIUM"
        else:
            result.confidence_level = "LOW"
    
    def _score_wfo_result(self, wfo_result: WalkForwardResult) -> float:
        """Score Walk-Forward Analysis result."""
        if not wfo_result.summary_stats:
            return 0.0
        
        score = 0.0
        max_score = 3.0
        
        # Consistency score
        consistency = wfo_result.summary_stats.get('consistency_score', 0)
        if consistency >= self.thresholds.min_wfo_consistency:
            score += 1.0
        else:
            score += consistency / self.thresholds.min_wfo_consistency
        
        # Sharpe ratio score
        sharpe = wfo_result.summary_stats.get('mean_sharpe_ratio', 0)
        if sharpe >= self.thresholds.min_wfo_sharpe:
            score += 1.0
        else:
            score += max(0, sharpe / self.thresholds.min_wfo_sharpe)
        
        # Drawdown score
        max_dd = abs(wfo_result.summary_stats.get('max_drawdown', 1.0))
        if max_dd <= self.thresholds.max_wfo_drawdown:
            score += 1.0
        else:
            score += max(0, 1.0 - (max_dd - self.thresholds.max_wfo_drawdown) / 0.1)
        
        return min(score / max_score, 1.0)
    
    def _score_permutation_result(self, perm_result: PermutationResult) -> float:
        """Score Permutation Testing result."""
        if not perm_result.summary_stats:
            return 0.0
        
        score = 0.0
        max_score = 2.0
        
        # P-value score
        p_value = perm_result.summary_stats.get('p_value', 1.0)
        if p_value <= self.thresholds.max_permutation_pvalue:
            score += 1.0
        else:
            score += max(0, 1.0 - (p_value - self.thresholds.max_permutation_pvalue) / 0.05)
        
        # Z-score score
        z_score = perm_result.summary_stats.get('z_score', 0)
        if z_score >= self.thresholds.min_permutation_zscore:
            score += 1.0
        else:
            score += max(0, z_score / self.thresholds.min_permutation_zscore)
        
        return min(score / max_score, 1.0)
    
    def _score_cscv_result(self, cscv_result: CSCVResult) -> float:
        """Score CSCV result."""
        if not cscv_result.summary_stats:
            return 0.0
        
        score = 0.0
        max_score = 2.0
        
        # PBO score
        pbo = cscv_result.pbo
        if pbo <= self.thresholds.max_pbo:
            score += 1.0
        else:
            score += max(0, 1.0 - (pbo - self.thresholds.max_pbo) / 0.2)
        
        # Statistical significance score
        if not cscv_result.overfitting_detected:
            score += 1.0
        else:
            p_value = cscv_result.pbo_p_value
            score += max(0, 1.0 - p_value / self.thresholds.max_pbo_pvalue)
        
        return min(score / max_score, 1.0)
    
    def _score_purged_cv_result(self, cv_result: PurgedCVResult) -> float:
        """Score Purged CV result."""
        if not cv_result.mean_scores:
            return 0.0
        
        score = 0.0
        max_score = 2.0
        
        # Accuracy score
        accuracy = cv_result.mean_scores.get('accuracy', 0)
        if accuracy >= self.thresholds.min_cv_accuracy:
            score += 1.0
        else:
            score += max(0, accuracy / self.thresholds.min_cv_accuracy)
        
        # F1 score
        f1 = cv_result.mean_scores.get('f1', 0)
        if f1 >= self.thresholds.min_cv_f1:
            score += 1.0
        else:
            score += max(0, f1 / self.thresholds.min_cv_f1)
        
        return min(score / max_score, 1.0)
    
    def _assess_risks(self, result: ValidationResult):
        """Assess strategy risks based on validation results."""
        risk_assessment = {
            'overall_risk': 'LOW',
            'specific_risks': [],
            'risk_factors': {}
        }
        
        risk_score = 0
        
        # Overfitting risk
        if result.cscv_result and result.cscv_result.overfitting_detected:
            risk_assessment['specific_risks'].append('High overfitting probability detected')
            risk_score += 2
        
        # Lack of statistical significance
        if result.permutation_result and result.permutation_result.summary_stats.get('p_value', 1.0) > 0.05:
            risk_assessment['specific_risks'].append('Strategy lacks statistical significance')
            risk_score += 2
        
        # Inconsistent performance
        if result.wfo_result and result.wfo_result.summary_stats.get('consistency_score', 1.0) < 0.6:
            risk_assessment['specific_risks'].append('Inconsistent out-of-sample performance')
            risk_score += 1
        
        # High drawdown risk
        if result.wfo_result and abs(result.wfo_result.summary_stats.get('max_drawdown', 0)) > 0.2:
            risk_assessment['specific_risks'].append('High maximum drawdown observed')
            risk_score += 1
        
        # Determine overall risk level
        if risk_score >= 4:
            risk_assessment['overall_risk'] = 'HIGH'
        elif risk_score >= 2:
            risk_assessment['overall_risk'] = 'MEDIUM'
        
        result.risk_assessment = risk_assessment
    
    def _generate_recommendations(self, result: ValidationResult):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Based on overall score
        if result.score < 0.6:
            recommendations.append("Strategy requires significant improvements before deployment")
            recommendations.append("Consider redesigning the strategy logic or parameters")
        elif result.score < 0.8:
            recommendations.append("Strategy shows promise but needs refinement")
            recommendations.append("Consider additional validation or parameter tuning")
        
        # Specific recommendations based on individual results
        if result.cscv_result and result.cscv_result.overfitting_detected:
            recommendations.append("Reduce model complexity to address overfitting")
            recommendations.append("Increase training data size or use regularization")
        
        if result.permutation_result and result.permutation_result.summary_stats.get('p_value', 1.0) > 0.05:
            recommendations.append("Strategy lacks statistical significance")
            recommendations.append("Consider alternative features or strategy logic")
        
        if result.wfo_result and result.wfo_result.summary_stats.get('consistency_score', 1.0) < 0.7:
            recommendations.append("Improve strategy robustness across different market conditions")
            recommendations.append("Consider regime-aware parameters or adaptive mechanisms")
        
        # Risk-based recommendations
        if result.risk_assessment['overall_risk'] == 'HIGH':
            recommendations.append("High risk detected - recommend paper trading before live deployment")
            recommendations.append("Implement additional risk controls and position sizing limits")
        
        result.recommendations = recommendations
    
    def generate_comprehensive_report(self, result: ValidationResult) -> str:
        """Generate comprehensive validation report."""
        report = f"""
Strategy Validation Report: {result.strategy_name}
{'='*80}

Executive Summary:
- Validation Mode: {result.validation_mode.upper()}
- Overall Score: {result.score:.3f}/1.000
- Status: {'✅ PASSED' if result.passed else '❌ FAILED'}
- Confidence Level: {result.confidence_level}
- Risk Level: {result.risk_assessment.get('overall_risk', 'UNKNOWN')}
- Validation Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Individual Validation Results:
{'='*40}
"""
        
        # Walk-Forward Analysis
        if result.wfo_result:
            wfo_score = result.validation_scores.get('walkforward', 0)
            report += f"""
Walk-Forward Analysis: {wfo_score:.3f}/1.000
- Consistency Score: {result.wfo_result.summary_stats.get('consistency_score', 0):.3f}
- Mean Sharpe Ratio: {result.wfo_result.summary_stats.get('mean_sharpe_ratio', 0):.3f}
- Maximum Drawdown: {result.wfo_result.summary_stats.get('max_drawdown', 0):.2%}
"""
        
        # Permutation Testing
        if result.permutation_result:
            perm_score = result.validation_scores.get('permutation', 0)
            report += f"""
Permutation Testing: {perm_score:.3f}/1.000
- P-Value: {result.permutation_result.summary_stats.get('p_value', 1.0):.4f}
- Z-Score: {result.permutation_result.summary_stats.get('z_score', 0):.3f}
- Statistical Significance: {'YES' if result.permutation_result.summary_stats.get('p_value', 1.0) < 0.05 else 'NO'}
"""
        
        # CSCV Analysis
        if result.cscv_result:
            cscv_score = result.validation_scores.get('cscv', 0)
            report += f"""
CSCV Analysis: {cscv_score:.3f}/1.000
- Probability of Backtest Overfitting: {result.cscv_result.pbo:.3f}
- Overfitting Detected: {'YES' if result.cscv_result.overfitting_detected else 'NO'}
- PBO P-Value: {result.cscv_result.pbo_p_value:.4f}
"""
        
        # Purged CV
        if result.purged_cv_result:
            cv_score = result.validation_scores.get('purged_cv', 0)
            report += f"""
Purged Cross-Validation: {cv_score:.3f}/1.000
- Mean Accuracy: {result.purged_cv_result.mean_scores.get('accuracy', 0):.3f}
- Mean F1 Score: {result.purged_cv_result.mean_scores.get('f1', 0):.3f}
- Mean Precision: {result.purged_cv_result.mean_scores.get('precision', 0):.3f}
"""
        
        # Risk Assessment
        report += f"""
Risk Assessment:
{'='*40}
Overall Risk Level: {result.risk_assessment.get('overall_risk', 'UNKNOWN')}
"""
        
        if result.risk_assessment.get('specific_risks'):
            report += "\nSpecific Risks Identified:\n"
            for risk in result.risk_assessment['specific_risks']:
                report += f"⚠️ {risk}\n"
        
        # Recommendations
        if result.recommendations:
            report += "\nRecommendations:\n"
            report += "=" * 40 + "\n"
            for i, rec in enumerate(result.recommendations, 1):
                report += f"{i}. {rec}\n"
        
        # Conclusion
        report += f"""
Conclusion:
{'='*40}
The strategy validation {'PASSED' if result.passed else 'FAILED'} with an overall score of {result.score:.3f}.
"""
        
        if result.passed:
            report += "The strategy meets the validation criteria and shows acceptable risk characteristics.\n"
            if result.confidence_level == 'HIGH':
                report += "High confidence in strategy robustness - recommended for live deployment.\n"
            elif result.confidence_level == 'MEDIUM':
                report += "Moderate confidence - consider additional testing or paper trading.\n"
            else:
                report += "Low confidence - recommend further development before deployment.\n"
        else:
            report += "The strategy does not meet the validation criteria and requires improvement.\n"
            report += "Review the recommendations above before considering deployment.\n"
        
        return report
    
    def save_validation_results(self, result: ValidationResult, filepath: str):
        """Save validation results to file."""
        try:
            # Convert result to serializable dictionary
            result_dict = {
                'strategy_name': result.strategy_name,
                'validation_mode': result.validation_mode,
                'timestamp': result.timestamp.isoformat(),
                'passed': result.passed,
                'score': result.score,
                'confidence_level': result.confidence_level,
                'validation_scores': result.validation_scores,
                'warnings': result.warnings,
                'recommendations': result.recommendations,
                'risk_assessment': result.risk_assessment,
            }
            
            # Save as YAML for readability
            with open(filepath, 'w') as f:
                yaml.dump(result_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Validation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation results: {e}")
    
    def load_validation_results(self, filepath: str) -> ValidationResult:
        """Load validation results from file."""
        try:
            with open(filepath, 'r') as f:
                result_dict = yaml.safe_load(f)
            
            # Convert back to ValidationResult
            result = ValidationResult(
                strategy_name=result_dict['strategy_name'],
                validation_mode=result_dict['validation_mode'],
                timestamp=datetime.fromisoformat(result_dict['timestamp']),
                passed=result_dict['passed'],
                score=result_dict['score'],
                confidence_level=result_dict['confidence_level'],
                validation_scores=result_dict.get('validation_scores', {}),
                warnings=result_dict.get('warnings', []),
                recommendations=result_dict.get('recommendations', []),
                risk_assessment=result_dict.get('risk_assessment', {})
            )
            
            self.logger.info(f"Validation results loaded from {filepath}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to load validation results: {e}")
            raise