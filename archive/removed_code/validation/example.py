"""
Example demonstrating the complete Phase 7 validation pipeline.

This example shows how to use all validation components together:
- Walk-forward analysis
- Monte Carlo simulation  
- Stress testing
- Model validation
- Integrated pipeline
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

# Import validation modules
from src.bot.validation import ValidationPipeline, ValidationPipelineConfig
from src.bot.validation.walk_forward import WalkForwardConfig
from src.bot.validation.monte_carlo import MonteCarloConfig
from src.bot.validation.stress_testing import StressScenario, Severity, StressType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AdvancedModelValidator:
    """Advanced example with multiple models and comprehensive validation."""
    
    def __init__(self):
        self.models = {}
        self.data = None
        self.pipeline = None
        
    def create_synthetic_market_data(self, start_date='2020-01-01', 
                                   end_date='2023-12-31', 
                                   freq='D') -> pd.DataFrame:
        """Create realistic synthetic market data for validation."""
        logger.info("Creating synthetic market data...")
        
        dates = pd.date_range(start_date, end_date, freq=freq)
        n_samples = len(dates)
        
        # Initialize price series
        np.random.seed(42)
        
        # Generate base returns with regime switching
        regime_1_length = n_samples // 3
        regime_2_length = n_samples // 3
        regime_3_length = n_samples - regime_1_length - regime_2_length
        
        # Regime 1: Bull market (higher returns, lower volatility)
        returns_1 = np.random.normal(0.0008, 0.012, regime_1_length)
        
        # Regime 2: Bear market (negative returns, higher volatility)  
        returns_2 = np.random.normal(-0.0003, 0.025, regime_2_length)
        
        # Regime 3: Sideways market (low returns, moderate volatility)
        returns_3 = np.random.normal(0.0002, 0.018, regime_3_length)
        
        # Combine regimes
        returns = np.concatenate([returns_1, returns_2, returns_3])
        
        # Add serial correlation and volatility clustering
        for i in range(1, len(returns)):
            # Serial correlation
            returns[i] += 0.05 * returns[i-1]
            
            # Volatility clustering (GARCH-like)
            if i > 20:
                vol_proxy = np.std(returns[i-20:i])
                returns[i] += np.random.normal(0, vol_proxy * 0.3)
        
        # Add extreme events (market crashes)
        crash_dates = np.random.choice(n_samples, 5, replace=False)
        for crash_idx in crash_dates:
            # Simulate market crash: large negative return followed by high volatility
            returns[crash_idx] = -np.random.uniform(0.08, 0.20)
            
            # Increased volatility for next 10 days
            for j in range(1, min(11, n_samples - crash_idx)):
                if crash_idx + j < len(returns):
                    returns[crash_idx + j] += np.random.normal(0, 0.04)
        
        # Create price series
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create comprehensive feature set
        returns_series = pd.Series(returns, index=dates)
        prices_series = pd.Series(prices, index=dates)
        
        data = pd.DataFrame(index=dates)
        
        # Basic features
        data['price'] = prices_series
        data['returns'] = returns_series
        data['log_returns'] = np.log(prices_series / prices_series.shift(1))
        
        # Technical indicators
        data['feature_sma_5'] = prices_series.rolling(5).mean()
        data['feature_sma_20'] = prices_series.rolling(20).mean()
        data['feature_sma_50'] = prices_series.rolling(50).mean()
        
        # Momentum features
        data['feature_momentum_5'] = returns_series.rolling(5).sum()
        data['feature_momentum_10'] = returns_series.rolling(10).sum()
        data['feature_momentum_20'] = returns_series.rolling(20).sum()
        
        # Volatility features
        data['feature_vol_5'] = returns_series.rolling(5).std()
        data['feature_vol_20'] = returns_series.rolling(20).std()
        data['feature_vol_60'] = returns_series.rolling(60).std()
        
        # Mean reversion features
        data['feature_mean_rev_5'] = (prices_series - data['feature_sma_5']) / data['feature_sma_5']
        data['feature_mean_rev_20'] = (prices_series - data['feature_sma_20']) / data['feature_sma_20']
        
        # Trend features
        data['feature_trend_5'] = data['feature_sma_5'] / data['feature_sma_5'].shift(5) - 1
        data['feature_trend_20'] = data['feature_sma_20'] / data['feature_sma_20'].shift(20) - 1
        
        # RSI-like feature
        delta = returns_series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data['feature_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands features
        bb_middle = data['feature_sma_20']
        bb_std = prices_series.rolling(20).std()
        data['feature_bb_upper'] = bb_middle + (bb_std * 2)
        data['feature_bb_lower'] = bb_middle - (bb_std * 2)
        data['feature_bb_position'] = (prices_series - bb_middle) / (bb_std * 2)
        
        # Volume proxy (synthetic)
        data['feature_volume'] = np.abs(returns_series) * 1000000 + np.random.exponential(500000, n_samples)
        
        # Market microstructure features
        data['feature_high_low'] = np.random.uniform(0.005, 0.02, n_samples)  # Synthetic bid-ask spread proxy
        
        # Regime detection features
        data['feature_regime_vol'] = returns_series.rolling(60).std()
        data['feature_regime_skew'] = returns_series.rolling(60).skew()
        data['feature_regime_kurt'] = returns_series.rolling(60).kurtosis()
        
        # Cross-sectional features (synthetic multi-asset)
        data['feature_correlation'] = np.random.uniform(0.3, 0.8, n_samples)  # Market correlation proxy
        
        # Economic features (synthetic)
        data['feature_interest_rate'] = 0.02 + 0.03 * np.sin(np.arange(n_samples) * 2 * np.pi / 252) + np.random.normal(0, 0.005, n_samples)
        data['feature_vix_proxy'] = data['feature_vol_20'] * 100 + np.random.normal(0, 5, n_samples)
        
        # Seasonal features
        data['feature_month'] = data.index.month
        data['feature_day_of_week'] = data.index.dayofweek
        data['feature_quarter'] = data.index.quarter
        
        # Drop initial NaN values
        data = data.dropna()
        
        logger.info(f"Created synthetic data: {len(data)} samples, {len(data.columns)} features")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Features: {[col for col in data.columns if col.startswith('feature_')][:10]}...")
        
        return data
    
    def create_model_ensemble(self) -> Dict[str, Any]:
        """Create ensemble of different model types for validation."""
        logger.info("Creating model ensemble...")
        
        models = {
            # Linear models
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0, random_state=42),
            
            # Tree-based models
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            
            # Non-linear models
            'support_vector': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
        }
        
        logger.info(f"Created {len(models)} models: {list(models.keys())}")
        return models
    
    def create_custom_stress_scenarios(self) -> List[StressScenario]:
        """Create custom stress test scenarios."""
        logger.info("Creating custom stress scenarios...")
        
        scenarios = []
        
        # Historical-style scenarios
        scenarios.append(StressScenario(
            name="crypto_winter_2022",
            stress_type=StressType.HISTORICAL,
            severity=Severity.EXTREME,
            description="Crypto winter 2022-style crash",
            market_shock=-0.75,
            volatility_multiplier=3.5,
            correlation_adjustment=1.8,
            duration_days=180,
            probability=0.005
        ))
        
        scenarios.append(StressScenario(
            name="covid_crash_2020",
            stress_type=StressType.HISTORICAL,
            severity=Severity.SEVERE,
            description="COVID-19 market crash",
            market_shock=-0.35,
            volatility_multiplier=4.0,
            duration_days=30,
            probability=0.01
        ))
        
        # Hypothetical scenarios
        scenarios.append(StressScenario(
            name="regulatory_crackdown",
            stress_type=StressType.HYPOTHETICAL,
            severity=Severity.SEVERE,
            description="Major regulatory crackdown on crypto",
            market_shock=-0.50,
            volatility_multiplier=3.0,
            factor_shocks={
                'regulatory_risk': -0.30,
                'liquidity_risk': -0.20,
                'sentiment_risk': -0.25
            },
            probability=0.02
        ))
        
        scenarios.append(StressScenario(
            name="liquidity_crisis",
            stress_type=StressType.LIQUIDITY_CRISIS,
            severity=Severity.EXTREME,
            description="Major liquidity crisis",
            market_shock=-0.25,
            volatility_multiplier=2.5,
            liquidity_impact=-0.15,
            correlation_adjustment=2.0,
            probability=0.005
        ))
        
        # Tail risk scenarios
        scenarios.append(StressScenario(
            name="black_swan",
            stress_type=StressType.TAIL_RISK,
            severity=Severity.EXTREME,
            description="Black swan event",
            market_shock=-0.60,
            volatility_multiplier=5.0,
            tail_dependency=0.9,
            probability=0.001
        ))
        
        logger.info(f"Created {len(scenarios)} stress scenarios")
        return scenarios
    
    def configure_validation_pipeline(self) -> ValidationPipelineConfig:
        """Configure the comprehensive validation pipeline."""
        logger.info("Configuring validation pipeline...")
        
        # Walk-forward configuration
        wf_config = WalkForwardConfig(
            train_window=504,  # 2 years
            test_window=63,    # 3 months
            step_size=21,      # 1 month
            n_splits=15,
            min_train_size=252,
            confidence_level=0.95,
            use_regime_detection=True,
            parallel_processing=True,
            max_workers=4
        )
        
        # Monte Carlo configuration
        mc_config = MonteCarloConfig(
            n_simulations=10000,
            simulation_horizon=252,  # 1 year
            confidence_levels=[0.90, 0.95, 0.99],
            bootstrap_samples=1000,
            use_parametric=True,
            distribution_type="auto",
            parallel_processing=True,
            max_workers=4
        )
        
        # Model validation configuration
        mv_config = ValidationConfig(
            cv_folds=5,
            time_series_cv=True,
            significance_level=0.05,
            stability_window=252,
            stability_step=21,
            bootstrap_iterations=1000,
            parallel_processing=True,
            max_workers=4
        )
        
        # Stress scenarios
        stress_scenarios = self.create_custom_stress_scenarios()
        
        # Pipeline configuration
        config = ValidationPipelineConfig(
            run_walk_forward=True,
            run_monte_carlo=True,
            run_stress_testing=True,
            run_model_validation=True,
            wf_config=wf_config,
            mc_config=mc_config,
            mv_config=mv_config,
            stress_scenarios=stress_scenarios,
            parallel_execution=True,
            max_workers=4,
            save_intermediate_results=True,
            generate_reports=True,
            report_format="html",
            output_directory="validation_output",
            db_path="comprehensive_validation.db",
            # Risk thresholds
            max_acceptable_var=-0.05,
            min_sharpe_ratio=0.8,
            max_drawdown_threshold=-0.15,
            min_hit_rate=0.48
        )
        
        logger.info("Validation pipeline configured")
        return config
    
    def run_comprehensive_validation(self):
        """Run the complete validation pipeline."""
        logger.info("Starting comprehensive validation pipeline...")
        
        try:
            # 1. Create synthetic market data
            self.data = self.create_synthetic_market_data()
            
            # 2. Create model ensemble  
            self.models = self.create_model_ensemble()
            
            # 3. Configure validation pipeline
            config = self.configure_validation_pipeline()
            
            # 4. Initialize and run pipeline
            self.pipeline = ValidationPipeline(config)
            
            logger.info("Running validation pipeline - this may take several minutes...")
            start_time = datetime.now()
            
            result = self.pipeline.run_validation_pipeline(
                data=self.data,
                models=self.models,
                pipeline_id=f"comprehensive_validation_{start_time.strftime('%Y%m%d_%H%M%S')}"
            )
            
            # 5. Display results
            self.display_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def display_results(self, result):
        """Display comprehensive validation results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION PIPELINE RESULTS")
        print("="*80)
        
        print(f"\n🔍 Pipeline ID: {result.pipeline_id}")
        print(f"📅 Timestamp: {result.timestamp}")
        print(f"⏱️  Total Execution Time: {result.total_execution_time:.2f} seconds")
        print(f"📊 Validation Status: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}")
        
        # Overall Assessment
        if result.overall_assessment:
            print(f"\n📈 OVERALL ASSESSMENT")
            print("-" * 40)
            
            if 'walk_forward' in result.overall_assessment:
                wf = result.overall_assessment['walk_forward']
                print(f"Walk-Forward Analysis:")
                print(f"  • Mean Sharpe Ratio: {wf.get('mean_sharpe', 0):.3f}")
                print(f"  • Best Sharpe Ratio: {wf.get('best_sharpe', 0):.3f}")
                print(f"  • Sharpe Stability: {wf.get('sharpe_stability', 0):.3f}")
            
            if 'monte_carlo' in result.overall_assessment:
                mc = result.overall_assessment['monte_carlo']
                print(f"\nMonte Carlo Simulation:")
                print(f"  • Mean Return: {mc.get('mean_return', 0):.4f}")
                print(f"  • Volatility: {mc.get('volatility', 0):.4f}")
                print(f"  • VaR (95%): {mc.get('var_95', 0):.4f}")
                print(f"  • CVaR (95%): {mc.get('cvar_95', 0):.4f}")
            
            if 'stress_testing' in result.overall_assessment:
                st = result.overall_assessment['stress_testing']
                print(f"\nStress Testing:")
                print(f"  • Worst Case Return: {st.get('worst_case_return', 0):.4f}")
                print(f"  • Resilience Score: {st.get('resilience_score', 0):.3f}")
                print(f"  • Tail Loss Probability: {st.get('tail_loss_probability', 0):.4f}")
            
            if 'model_validation' in result.overall_assessment:
                mv = result.overall_assessment['model_validation']
                print(f"\nModel Validation:")
                print(f"  • Mean Score: {mv.get('mean_score', 0):.3f}")
                print(f"  • Pass Rate: {mv.get('pass_rate', 0):.2%}")
                print(f"  • Best Score: {mv.get('best_score', 0):.3f}")
        
        # Risk Assessment
        if result.risk_assessment:
            print(f"\n⚠️  RISK ASSESSMENT")
            print("-" * 40)
            for metric, value in result.risk_assessment.items():
                print(f"  • {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Model Rankings
        if result.model_rankings:
            print(f"\n🏆 MODEL RANKINGS")
            print("-" * 40)
            sorted_rankings = sorted(result.model_rankings.items(), key=lambda x: x[1], reverse=True)
            for i, (model, score) in enumerate(sorted_rankings[:10], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
                print(f"  {emoji} {model}: {score:.4f}")
        
        # Critical Issues
        if result.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES")
            print("-" * 40)
            for issue in result.critical_issues:
                print(f"  ❌ {issue}")
        
        # Warnings
        if result.warnings:
            print(f"\n⚠️  WARNINGS")
            print("-" * 40)
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
        
        # Recommendations
        if result.recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Component Execution Times
        if result.component_execution_times:
            print(f"\n⏱️  COMPONENT EXECUTION TIMES")
            print("-" * 40)
            for component, time_taken in result.component_execution_times.items():
                print(f"  • {component}: {time_taken:.2f}s")
        
        # Generated Reports
        if result.generated_reports:
            print(f"\n📄 GENERATED REPORTS")
            print("-" * 40)
            for report_type, path in result.generated_reports.items():
                print(f"  • {report_type.upper()}: {path}")
        
        print("\n" + "="*80)
        
        # Summary recommendations based on results
        self.provide_actionable_insights(result)
    
    def provide_actionable_insights(self, result):
        """Provide actionable insights based on validation results."""
        print("\n🎯 ACTIONABLE INSIGHTS")
        print("="*40)
        
        insights = []
        
        # Model selection insights
        if result.model_rankings:
            best_models = sorted(result.model_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
            best_model_names = [model.split('_')[0] for model, _ in best_models]
            insights.append(f"Focus on these top-performing models: {', '.join(set(best_model_names))}")
        
        # Risk management insights
        if result.risk_assessment:
            if result.risk_assessment.get('var_95', 0) < -0.10:
                insights.append("High VaR detected - consider implementing position sizing limits")
            
            if result.risk_assessment.get('mean_sharpe', 0) < 0.5:
                insights.append("Low Sharpe ratio - focus on improving signal quality and noise reduction")
            
            if result.risk_assessment.get('stress_resilience', 1.0) < 0.6:
                insights.append("Poor stress resilience - implement dynamic hedging strategies")
        
        # Validation-specific insights
        if result.validation_passed:
            insights.append("✅ Models passed validation - ready for paper trading phase")
        else:
            insights.append("❌ Models failed validation - requires model improvement before deployment")
        
        # Regime-based insights
        if result.walk_forward_summary:
            insights.append("Consider implementing regime-aware position sizing based on market conditions")
        
        # Portfolio construction insights
        if len(result.model_rankings) > 1:
            insights.append("Consider ensemble approach combining top-performing models")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
        
        if not insights:
            print("    No specific insights available from current results.")

def main():
    """Main execution function."""
    print("🚀 Starting Phase 7 Validation Pipeline Example")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("validation_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize validator
    validator = AdvancedModelValidator()
    
    try:
        # Run comprehensive validation
        result = validator.run_comprehensive_validation()
        
        print(f"\n✅ Validation pipeline completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
        if result.generated_reports:
            print(f"📊 View detailed reports:")
            for report_type, path in result.generated_reports.items():
                print(f"   {report_type}: {path}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Validation pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()