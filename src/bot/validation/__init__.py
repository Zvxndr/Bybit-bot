"""
Main validation pipeline integration module.

This module provides the main interface for running comprehensive validation
pipelines including walk-forward analysis, Monte Carlo simulation, stress testing,
and model validation in an integrated workflow.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import validation modules
from .walk_forward import WalkForwardAnalyzer, WalkForwardConfig, ModelValidator as WFModelValidator
from .monte_carlo import MonteCarloSimulator, MonteCarloConfig
from .stress_testing import StressTester, StressScenario, Severity
from .model_validation import ModelValidator, ValidationConfig, ModelType

logger = logging.getLogger(__name__)

@dataclass
class ValidationPipelineConfig:
    """Configuration for the complete validation pipeline."""
    # Pipeline components to run
    run_walk_forward: bool = True
    run_monte_carlo: bool = True
    run_stress_testing: bool = True
    run_model_validation: bool = True
    
    # Walk-forward configuration
    wf_config: Optional[WalkForwardConfig] = None
    
    # Monte Carlo configuration
    mc_config: Optional[MonteCarloConfig] = None
    
    # Model validation configuration
    mv_config: Optional[ValidationConfig] = None
    
    # Stress testing scenarios
    stress_scenarios: Optional[List[StressScenario]] = None
    
    # Data configuration
    validation_start_date: Optional[str] = None
    validation_end_date: Optional[str] = None
    min_data_points: int = 500
    
    # Execution settings
    parallel_execution: bool = True
    max_workers: int = 4
    save_intermediate_results: bool = True
    
    # Output settings
    generate_reports: bool = True
    report_format: str = "html"  # html, pdf, json
    output_directory: str = "validation_results"
    
    # Database settings
    db_path: str = "validation_pipeline.db"
    
    # Risk thresholds
    max_acceptable_var: float = -0.05
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -0.20
    min_hit_rate: float = 0.45

@dataclass
class ValidationPipelineResult:
    """Results from the complete validation pipeline."""
    pipeline_id: str
    timestamp: datetime
    config: ValidationPipelineConfig
    
    # Individual component results
    walk_forward_summary: Optional[Any] = None
    monte_carlo_result: Optional[Any] = None
    stress_test_summary: Optional[Any] = None
    model_validation_results: List[Any] = field(default_factory=list)
    
    # Aggregate analysis
    overall_assessment: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    model_rankings: Dict[str, float] = field(default_factory=dict)
    
    # Validation status
    validation_passed: bool = False
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Execution metrics
    total_execution_time: float = 0.0
    component_execution_times: Dict[str, float] = field(default_factory=dict)
    
    # Report paths
    generated_reports: Dict[str, str] = field(default_factory=dict)

class ValidationPipeline:
    """Main validation pipeline orchestrator."""
    
    def __init__(self, config: ValidationPipelineConfig):
        self.config = config
        
        # Initialize component validators with default configs if not provided
        self._initialize_components()
        
        # Initialize database
        self._init_database()
        
        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize validation components with configurations."""
        # Walk-forward analyzer
        if self.config.run_walk_forward:
            wf_config = self.config.wf_config or WalkForwardConfig()
            self.walk_forward_analyzer = WalkForwardAnalyzer(wf_config, self.config.db_path)
        
        # Monte Carlo simulator
        if self.config.run_monte_carlo:
            mc_config = self.config.mc_config or MonteCarloConfig()
            self.monte_carlo_simulator = MonteCarloSimulator(mc_config, self.config.db_path)
        
        # Stress tester
        if self.config.run_stress_testing:
            self.stress_tester = StressTester(self.config.db_path)
        
        # Model validator
        if self.config.run_model_validation:
            mv_config = self.config.mv_config or ValidationConfig()
            self.model_validator = ModelValidator(mv_config, self.config.db_path)
    
    def _init_database(self):
        """Initialize main pipeline database."""
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            # Pipeline results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_pipeline_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    validation_passed INTEGER,
                    overall_assessment TEXT,
                    risk_assessment TEXT,
                    model_rankings TEXT,
                    critical_issues TEXT,
                    warnings TEXT,
                    recommendations TEXT,
                    total_execution_time REAL,
                    component_times TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing pipeline database: {e}")
    
    def run_validation_pipeline(self, data: pd.DataFrame, 
                              models: Dict[str, Any],
                              pipeline_id: Optional[str] = None) -> ValidationPipelineResult:
        """Run the complete validation pipeline."""
        start_time = datetime.now()
        pipeline_id = pipeline_id or f"validation_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting validation pipeline: {pipeline_id}")
        
        # Initialize result object
        result = ValidationPipelineResult(
            pipeline_id=pipeline_id,
            timestamp=start_time,
            config=self.config
        )
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Prepare data for validation
            validation_data = self._prepare_validation_data(data)
            
            # Run validation components
            if self.config.parallel_execution:
                self._run_parallel_validation(validation_data, models, result)
            else:
                self._run_sequential_validation(validation_data, models, result)
            
            # Perform aggregate analysis
            self._perform_aggregate_analysis(result)
            
            # Generate assessment
            self._generate_overall_assessment(result)
            
            # Generate reports
            if self.config.generate_reports:
                self._generate_reports(result)
            
            # Save results
            self._save_pipeline_results(result)
            
            result.total_execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Validation pipeline completed in {result.total_execution_time:.2f}s")
            logger.info(f"Validation passed: {result.validation_passed}")
            
        except Exception as e:
            error_msg = f"Error in validation pipeline: {str(e)}"
            logger.error(error_msg)
            result.critical_issues.append(error_msg)
            result.validation_passed = False
            result.total_execution_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data quality and completeness."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        if len(data) < self.config.min_data_points:
            raise ValueError(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
        
        # Check for required columns
        required_columns = ['returns']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing recommended columns: {missing_columns}")
        
        # Check data quality
        null_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if null_percentage > 0.1:
            logger.warning(f"High percentage of null values: {null_percentage:.2%}")
    
    def _prepare_validation_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for validation components."""
        # Filter by date range if specified
        if self.config.validation_start_date or self.config.validation_end_date:
            if self.config.validation_start_date:
                data = data[data.index >= self.config.validation_start_date]
            if self.config.validation_end_date:
                data = data[data.index <= self.config.validation_end_date]
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Add derived features if not present
        if 'returns' not in data.columns and len(data.columns) > 0:
            price_col = data.columns[0]
            data['returns'] = data[price_col].pct_change()
        
        return data
    
    def _run_parallel_validation(self, data: pd.DataFrame, 
                               models: Dict[str, Any], 
                               result: ValidationPipelineResult):
        """Run validation components in parallel."""
        futures = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit validation tasks
            if self.config.run_walk_forward:
                for model_name, model in models.items():
                    future = executor.submit(self._run_walk_forward_analysis, data, model, model_name)
                    futures[f'walk_forward_{model_name}'] = future
            
            if self.config.run_monte_carlo:
                future = executor.submit(self._run_monte_carlo_simulation, data)
                futures['monte_carlo'] = future
            
            if self.config.run_stress_testing:
                future = executor.submit(self._run_stress_testing, data)
                futures['stress_testing'] = future
            
            if self.config.run_model_validation:
                for model_name, model in models.items():
                    future = executor.submit(self._run_model_validation, data, model, model_name)
                    futures[f'model_validation_{model_name}'] = future
            
            # Collect results
            for task_name, future in futures.items():
                try:
                    task_result = future.result(timeout=3600)  # 1 hour timeout
                    self._process_component_result(task_name, task_result, result)
                except Exception as e:
                    logger.error(f"Error in {task_name}: {e}")
                    result.critical_issues.append(f"Failed to complete {task_name}: {str(e)}")
    
    def _run_sequential_validation(self, data: pd.DataFrame, 
                                 models: Dict[str, Any], 
                                 result: ValidationPipelineResult):
        """Run validation components sequentially."""
        # Walk-forward analysis
        if self.config.run_walk_forward:
            for model_name, model in models.items():
                try:
                    component_result = self._run_walk_forward_analysis(data, model, model_name)
                    self._process_component_result(f'walk_forward_{model_name}', component_result, result)
                except Exception as e:
                    logger.error(f"Error in walk-forward analysis for {model_name}: {e}")
                    result.critical_issues.append(f"Walk-forward analysis failed for {model_name}")
        
        # Monte Carlo simulation
        if self.config.run_monte_carlo:
            try:
                component_result = self._run_monte_carlo_simulation(data)
                self._process_component_result('monte_carlo', component_result, result)
            except Exception as e:
                logger.error(f"Error in Monte Carlo simulation: {e}")
                result.critical_issues.append("Monte Carlo simulation failed")
        
        # Stress testing
        if self.config.run_stress_testing:
            try:
                component_result = self._run_stress_testing(data)
                self._process_component_result('stress_testing', component_result, result)
            except Exception as e:
                logger.error(f"Error in stress testing: {e}")
                result.critical_issues.append("Stress testing failed")
        
        # Model validation
        if self.config.run_model_validation:
            for model_name, model in models.items():
                try:
                    component_result = self._run_model_validation(data, model, model_name)
                    self._process_component_result(f'model_validation_{model_name}', component_result, result)
                except Exception as e:
                    logger.error(f"Error in model validation for {model_name}: {e}")
                    result.critical_issues.append(f"Model validation failed for {model_name}")
    
    def _run_walk_forward_analysis(self, data: pd.DataFrame, model: Any, model_name: str) -> Any:
        """Run walk-forward analysis for a model."""
        start_time = datetime.now()
        
        # Create model validator wrapper
        class ModelValidatorWrapper(WFModelValidator):
            def __init__(self, model, model_name):
                self.model = model
                self.model_name = model_name
                self.feature_names = []
            
            def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
                self.feature_names = list(X_train.columns)
                self.model.fit(X_train, y_train)
            
            def predict(self, X_test: pd.DataFrame) -> np.ndarray:
                return self.model.predict(X_test)
            
            def get_feature_importance(self) -> Dict[str, float]:
                if hasattr(self.model, 'feature_importances_'):
                    return dict(zip(self.feature_names, self.model.feature_importances_))
                elif hasattr(self.model, 'coef_'):
                    coef = self.model.coef_
                    if coef.ndim == 1:
                        return dict(zip(self.feature_names, np.abs(coef)))
                    else:
                        return dict(zip(self.feature_names, np.abs(coef[0])))
                return {}
            
            def get_model_params(self) -> Dict[str, Any]:
                return self.model.get_params() if hasattr(self.model, 'get_params') else {}
        
        validator = ModelValidatorWrapper(model, model_name)
        analysis_id = f"wf_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        summary = self.walk_forward_analyzer.run_analysis(data, validator, analysis_id)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        return {'summary': summary, 'execution_time': execution_time, 'model_name': model_name}
    
    def _run_monte_carlo_simulation(self, data: pd.DataFrame) -> Any:
        """Run Monte Carlo simulation."""
        start_time = datetime.now()
        
        simulation_id = f"mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = self.monte_carlo_simulator.run_simulation(data, simulation_id=simulation_id)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        return {'result': result, 'execution_time': execution_time}
    
    def _run_stress_testing(self, data: pd.DataFrame) -> Any:
        """Run stress testing."""
        start_time = datetime.now()
        
        test_id = f"stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        scenarios = self.config.stress_scenarios
        summary = self.stress_tester.run_stress_test(data, scenarios, test_id)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        return {'summary': summary, 'execution_time': execution_time}
    
    def _run_model_validation(self, data: pd.DataFrame, model: Any, model_name: str) -> Any:
        """Run model validation."""
        start_time = datetime.now()
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col.startswith('feature_')]
        if not feature_cols:
            # Create simple features from returns
            if 'returns' in data.columns:
                data_copy = data.copy()
                data_copy['feature_momentum'] = data_copy['returns'].rolling(10).mean()
                data_copy['feature_volatility'] = data_copy['returns'].rolling(20).std()
                data_copy['feature_rsi'] = self._calculate_rsi(data_copy['returns'])
                feature_cols = ['feature_momentum', 'feature_volatility', 'feature_rsi']
                data = data_copy
        
        if 'returns' not in data.columns:
            raise ValueError("No returns column found for model validation")
        
        # Clean data
        clean_data = data[feature_cols + ['returns']].dropna()
        
        X = clean_data[feature_cols].values
        y = clean_data['returns'].values
        
        timestamps = clean_data.index if isinstance(clean_data.index, pd.DatetimeIndex) else None
        
        # Run validation
        validation_result = self.model_validator.validate_model(
            model=model,
            X=X,
            y=y,
            model_id=model_name,
            model_type=ModelType.REGRESSION,
            timestamps=timestamps
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        return {'result': validation_result, 'execution_time': execution_time, 'model_name': model_name}
    
    def _calculate_rsi(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = returns.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _process_component_result(self, task_name: str, task_result: Any, result: ValidationPipelineResult):
        """Process results from individual validation components."""
        execution_time = task_result.get('execution_time', 0.0)
        result.component_execution_times[task_name] = execution_time
        
        if task_name.startswith('walk_forward'):
            if result.walk_forward_summary is None:
                result.walk_forward_summary = {}
            model_name = task_result['model_name']
            result.walk_forward_summary[model_name] = task_result['summary']
            
        elif task_name == 'monte_carlo':
            result.monte_carlo_result = task_result['result']
            
        elif task_name == 'stress_testing':
            result.stress_test_summary = task_result['summary']
            
        elif task_name.startswith('model_validation'):
            result.model_validation_results.append(task_result['result'])
    
    def _perform_aggregate_analysis(self, result: ValidationPipelineResult):
        """Perform aggregate analysis across all validation components."""
        analysis = {}
        
        try:
            # Walk-forward analysis summary
            if result.walk_forward_summary:
                wf_scores = []
                for model_name, summary in result.walk_forward_summary.items():
                    if hasattr(summary, 'mean_oos_sharpe'):
                        wf_scores.append(summary.mean_oos_sharpe)
                        result.model_rankings[f'{model_name}_wf_sharpe'] = summary.mean_oos_sharpe
                
                if wf_scores:
                    analysis['walk_forward'] = {
                        'mean_sharpe': np.mean(wf_scores),
                        'best_sharpe': max(wf_scores),
                        'worst_sharpe': min(wf_scores),
                        'sharpe_stability': 1.0 - np.std(wf_scores) / (abs(np.mean(wf_scores)) + 1e-8)
                    }
            
            # Monte Carlo analysis summary
            if result.monte_carlo_result:
                mc_result = result.monte_carlo_result
                analysis['monte_carlo'] = {
                    'mean_return': mc_result.mean_return,
                    'volatility': mc_result.volatility,
                    'var_95': mc_result.var_estimates.get(0.95, np.nan),
                    'cvar_95': mc_result.cvar_estimates.get(0.95, np.nan),
                    'max_drawdown_mean': np.mean(mc_result.max_drawdown_dist)
                }
            
            # Stress testing analysis summary
            if result.stress_test_summary:
                st_summary = result.stress_test_summary
                analysis['stress_testing'] = {
                    'worst_case_return': st_summary.worst_case_return,
                    'resilience_score': st_summary.resilience_score,
                    'tail_loss_probability': st_summary.extreme_loss_probability,
                    'expected_tail_loss': st_summary.expected_tail_loss
                }
            
            # Model validation analysis summary
            if result.model_validation_results:
                validation_scores = [r.overall_score for r in result.model_validation_results]
                passed_validations = sum(1 for r in result.model_validation_results if r.validation_passed)
                
                analysis['model_validation'] = {
                    'mean_score': np.mean(validation_scores),
                    'pass_rate': passed_validations / len(result.model_validation_results),
                    'best_score': max(validation_scores),
                    'worst_score': min(validation_scores)
                }
                
                # Add individual model scores to rankings
                for val_result in result.model_validation_results:
                    result.model_rankings[f'{val_result.model_id}_validation_score'] = val_result.overall_score
            
            result.overall_assessment = analysis
            
        except Exception as e:
            logger.error(f"Error in aggregate analysis: {e}")
            result.warnings.append(f"Aggregate analysis partially failed: {str(e)}")
    
    def _generate_overall_assessment(self, result: ValidationPipelineResult):
        """Generate overall assessment and recommendations."""
        try:
            # Risk assessment
            risk_metrics = {}
            
            # Extract risk metrics from different components
            if result.monte_carlo_result:
                risk_metrics['var_95'] = result.monte_carlo_result.var_estimates.get(0.95, 0.0)
                risk_metrics['max_drawdown'] = np.mean(result.monte_carlo_result.max_drawdown_dist)
            
            if result.walk_forward_summary:
                sharpe_ratios = []
                for summary in result.walk_forward_summary.values():
                    if hasattr(summary, 'mean_oos_sharpe'):
                        sharpe_ratios.append(summary.mean_oos_sharpe)
                
                if sharpe_ratios:
                    risk_metrics['mean_sharpe'] = np.mean(sharpe_ratios)
            
            if result.stress_test_summary:
                risk_metrics['stress_resilience'] = result.stress_test_summary.resilience_score
                risk_metrics['worst_case_loss'] = result.stress_test_summary.worst_case_return
            
            result.risk_assessment = risk_metrics
            
            # Overall validation status
            validation_checks = []
            
            # Check VaR threshold
            if 'var_95' in risk_metrics:
                var_check = risk_metrics['var_95'] >= self.config.max_acceptable_var
                validation_checks.append(var_check)
                if not var_check:
                    result.critical_issues.append(f"VaR exceeds threshold: {risk_metrics['var_95']:.4f} < {self.config.max_acceptable_var}")
            
            # Check Sharpe ratio
            if 'mean_sharpe' in risk_metrics:
                sharpe_check = risk_metrics['mean_sharpe'] >= self.config.min_sharpe_ratio
                validation_checks.append(sharpe_check)
                if not sharpe_check:
                    result.warnings.append(f"Low Sharpe ratio: {risk_metrics['mean_sharpe']:.4f} < {self.config.min_sharpe_ratio}")
            
            # Check maximum drawdown
            if 'max_drawdown' in risk_metrics:
                dd_check = risk_metrics['max_drawdown'] >= self.config.max_drawdown_threshold
                validation_checks.append(dd_check)
                if not dd_check:
                    result.critical_issues.append(f"Max drawdown exceeds threshold: {risk_metrics['max_drawdown']:.4f} < {self.config.max_drawdown_threshold}")
            
            # Check model validation pass rate
            if result.model_validation_results:
                passed = sum(1 for r in result.model_validation_results if r.validation_passed)
                pass_rate = passed / len(result.model_validation_results)
                
                if pass_rate < 0.5:
                    result.critical_issues.append(f"Low model validation pass rate: {pass_rate:.2%}")
                    validation_checks.append(False)
                elif pass_rate < 0.8:
                    result.warnings.append(f"Moderate model validation pass rate: {pass_rate:.2%}")
                    validation_checks.append(True)
                else:
                    validation_checks.append(True)
            
            # Overall validation status
            result.validation_passed = len(validation_checks) > 0 and all(validation_checks) and len(result.critical_issues) == 0
            
            # Generate recommendations
            self._generate_recommendations(result)
            
        except Exception as e:
            logger.error(f"Error generating overall assessment: {e}")
            result.critical_issues.append(f"Assessment generation failed: {str(e)}")
            result.validation_passed = False
    
    def _generate_recommendations(self, result: ValidationPipelineResult):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        try:
            # Model performance recommendations
            if result.model_validation_results:
                best_model = max(result.model_validation_results, key=lambda x: x.overall_score)
                worst_model = min(result.model_validation_results, key=lambda x: x.overall_score)
                
                recommendations.append(f"Best performing model: {best_model.model_id} (score: {best_model.overall_score:.3f})")
                
                if worst_model.overall_score < 0.5:
                    recommendations.append(f"Consider replacing {worst_model.model_id} (low score: {worst_model.overall_score:.3f})")
            
            # Risk management recommendations
            if result.risk_assessment:
                if result.risk_assessment.get('var_95', 0) < self.config.max_acceptable_var:
                    recommendations.append("Consider reducing position sizes or implementing additional hedging")
                
                if result.risk_assessment.get('mean_sharpe', 0) < self.config.min_sharpe_ratio:
                    recommendations.append("Focus on improving risk-adjusted returns through better signal generation")
                
                if result.risk_assessment.get('stress_resilience', 1.0) < 0.7:
                    recommendations.append("Implement stress-testing based position sizing and risk controls")
            
            # Walk-forward recommendations
            if result.walk_forward_summary:
                for model_name, summary in result.walk_forward_summary.items():
                    if hasattr(summary, 'overfitting_score') and summary.overfitting_score > 0.3:
                        recommendations.append(f"Model {model_name} shows signs of overfitting - consider regularization")
                    
                    if hasattr(summary, 'regime_consistency') and summary.regime_consistency < 0.6:
                        recommendations.append(f"Model {model_name} is not regime-aware - consider regime-based modeling")
            
            # Monte Carlo recommendations
            if result.monte_carlo_result:
                if result.monte_carlo_result.skewness < -1.0:
                    recommendations.append("Returns distribution shows negative skew - consider tail risk hedging")
                
                if result.monte_carlo_result.kurtosis > 5.0:
                    recommendations.append("Returns show fat tails - standard risk models may underestimate risk")
            
            result.recommendations = recommendations
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
    
    def _generate_reports(self, result: ValidationPipelineResult):
        """Generate validation reports."""
        try:
            timestamp = result.timestamp.strftime('%Y%m%d_%H%M%S')
            
            # JSON report (always generated)
            json_path = Path(self.config.output_directory) / f"validation_report_{timestamp}.json"
            self._generate_json_report(result, json_path)
            result.generated_reports['json'] = str(json_path)
            
            # HTML report
            if self.config.report_format in ['html', 'all']:
                html_path = Path(self.config.output_directory) / f"validation_report_{timestamp}.html"
                self._generate_html_report(result, html_path)
                result.generated_reports['html'] = str(html_path)
            
            logger.info(f"Generated validation reports in {self.config.output_directory}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            result.warnings.append(f"Report generation failed: {str(e)}")
    
    def _generate_json_report(self, result: ValidationPipelineResult, output_path: Path):
        """Generate JSON validation report."""
        report_data = {
            'pipeline_id': result.pipeline_id,
            'timestamp': result.timestamp.isoformat(),
            'validation_passed': result.validation_passed,
            'overall_assessment': result.overall_assessment,
            'risk_assessment': result.risk_assessment,
            'model_rankings': result.model_rankings,
            'critical_issues': result.critical_issues,
            'warnings': result.warnings,
            'recommendations': result.recommendations,
            'execution_times': result.component_execution_times,
            'total_execution_time': result.total_execution_time
        }
        
        # Add component results (simplified)
        if result.walk_forward_summary:
            report_data['walk_forward_results'] = {
                model: {
                    'mean_oos_returns': getattr(summary, 'mean_oos_returns', 0),
                    'mean_oos_sharpe': getattr(summary, 'mean_oos_sharpe', 0),
                    'performance_stability': getattr(summary, 'performance_stability', 0)
                }
                for model, summary in result.walk_forward_summary.items()
            }
        
        if result.monte_carlo_result:
            report_data['monte_carlo_results'] = {
                'mean_return': result.monte_carlo_result.mean_return,
                'volatility': result.monte_carlo_result.volatility,
                'var_95': result.monte_carlo_result.var_estimates.get(0.95, 0),
                'max_drawdown_mean': float(np.mean(result.monte_carlo_result.max_drawdown_dist))
            }
        
        if result.stress_test_summary:
            report_data['stress_test_results'] = {
                'worst_case_return': result.stress_test_summary.worst_case_return,
                'resilience_score': result.stress_test_summary.resilience_score,
                'scenarios_passed': result.stress_test_summary.scenarios_passed,
                'scenarios_failed': result.stress_test_summary.scenarios_failed
            }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _generate_html_report(self, result: ValidationPipelineResult, output_path: Path):
        """Generate HTML validation report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Pipeline Report - {result.pipeline_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
                .error {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Validation Pipeline Report</h1>
                <p><strong>Pipeline ID:</strong> {result.pipeline_id}</p>
                <p><strong>Timestamp:</strong> {result.timestamp}</p>
                <p><strong>Validation Status:</strong> {'✅ PASSED' if result.validation_passed else '❌ FAILED'}</p>
                <p><strong>Execution Time:</strong> {result.total_execution_time:.2f} seconds</p>
            </div>
        """
        
        # Risk Assessment
        if result.risk_assessment:
            html_content += f"""
            <div class="section">
                <h2>Risk Assessment</h2>
                {self._format_metrics_html(result.risk_assessment)}
            </div>
            """
        
        # Critical Issues
        if result.critical_issues:
            html_content += f"""
            <div class="section error">
                <h2>Critical Issues</h2>
                <ul>
                    {''.join(f'<li>{issue}</li>' for issue in result.critical_issues)}
                </ul>
            </div>
            """
        
        # Warnings
        if result.warnings:
            html_content += f"""
            <div class="section warning">
                <h2>Warnings</h2>
                <ul>
                    {''.join(f'<li>{warning}</li>' for warning in result.warnings)}
                </ul>
            </div>
            """
        
        # Recommendations
        if result.recommendations:
            html_content += f"""
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in result.recommendations)}
                </ul>
            </div>
            """
        
        # Model Rankings
        if result.model_rankings:
            html_content += f"""
            <div class="section">
                <h2>Model Rankings</h2>
                <table>
                    <tr><th>Model/Metric</th><th>Score</th></tr>
                    {''.join(f'<tr><td>{model}</td><td>{score:.4f}</td></tr>' 
                            for model, score in sorted(result.model_rankings.items(), key=lambda x: x[1], reverse=True))}
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _format_metrics_html(self, metrics: Dict[str, float]) -> str:
        """Format metrics for HTML display."""
        return ''.join(f'<div class="metric"><strong>{key}:</strong> {value:.4f}</div>' 
                      for key, value in metrics.items())
    
    def _save_pipeline_results(self, result: ValidationPipelineResult):
        """Save pipeline results to database."""
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO validation_pipeline_results (
                    pipeline_id, timestamp, config_json, validation_passed,
                    overall_assessment, risk_assessment, model_rankings,
                    critical_issues, warnings, recommendations,
                    total_execution_time, component_times
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.pipeline_id,
                result.timestamp.isoformat(),
                json.dumps(result.config.__dict__, default=str),
                1 if result.validation_passed else 0,
                json.dumps(result.overall_assessment, default=str),
                json.dumps(result.risk_assessment),
                json.dumps(result.model_rankings),
                json.dumps(result.critical_issues),
                json.dumps(result.warnings),
                json.dumps(result.recommendations),
                result.total_execution_time,
                json.dumps(result.component_execution_times)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
    
    def get_pipeline_results(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve pipeline results from database."""
        try:
            conn = sqlite3.connect(self.config.db_path)
            
            if pipeline_id:
                query = "SELECT * FROM validation_pipeline_results WHERE pipeline_id = ?"
                df = pd.read_sql_query(query, conn, params=[pipeline_id])
            else:
                query = "SELECT * FROM validation_pipeline_results ORDER BY timestamp DESC LIMIT 1"
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            return df.to_dict('records') if not df.empty else []
            
        except Exception as e:
            logger.error(f"Error retrieving pipeline results: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Generate synthetic returns with realistic properties
    returns = np.random.normal(0.0005, 0.015, n_samples)
    
    # Add momentum and mean reversion
    for i in range(1, n_samples):
        returns[i] += 0.02 * returns[i-1] - 0.01 * np.mean(returns[max(0, i-20):i])
    
    # Create feature data
    data = pd.DataFrame({
        'returns': returns,
        'feature_momentum': pd.Series(returns).rolling(10).mean(),
        'feature_volatility': pd.Series(returns).rolling(20).std(),
        'feature_trend': pd.Series(returns).rolling(5).sum()
    }, index=dates)
    
    data = data.dropna()
    
    # Define models to validate
    models = {
        'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'linear_regression': LinearRegression()
    }
    
    # Configure validation pipeline
    config = ValidationPipelineConfig(
        run_walk_forward=True,
        run_monte_carlo=True,
        run_stress_testing=True,
        run_model_validation=True,
        parallel_execution=False,  # Set to False for simpler debugging
        generate_reports=True,
        output_directory="validation_output"
    )
    
    # Run validation pipeline
    pipeline = ValidationPipeline(config)
    
    try:
        result = pipeline.run_validation_pipeline(data, models)
        
        print("\n" + "="*60)
        print("VALIDATION PIPELINE RESULTS")
        print("="*60)
        
        print(f"Pipeline ID: {result.pipeline_id}")
        print(f"Validation Status: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}")
        print(f"Total Execution Time: {result.total_execution_time:.2f}s")
        
        print(f"\n--- Risk Assessment ---")
        for metric, value in result.risk_assessment.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\n--- Model Rankings ---")
        sorted_rankings = sorted(result.model_rankings.items(), key=lambda x: x[1], reverse=True)
        for model, score in sorted_rankings[:5]:  # Top 5
            print(f"{model}: {score:.4f}")
        
        if result.critical_issues:
            print(f"\n--- Critical Issues ---")
            for issue in result.critical_issues:
                print(f"❌ {issue}")
        
        if result.warnings:
            print(f"\n--- Warnings ---")
            for warning in result.warnings:
                print(f"⚠️  {warning}")
        
        if result.recommendations:
            print(f"\n--- Recommendations ---")
            for rec in result.recommendations:
                print(f"💡 {rec}")
        
        print(f"\n--- Component Execution Times ---")
        for component, time_taken in result.component_execution_times.items():
            print(f"{component}: {time_taken:.2f}s")
        
        if result.generated_reports:
            print(f"\n--- Generated Reports ---")
            for report_type, path in result.generated_reports.items():
                print(f"{report_type.upper()}: {path}")
        
    except Exception as e:
        logger.error(f"Error in validation pipeline example: {e}")
        import traceback
        traceback.print_exc()