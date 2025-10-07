"""
Phase 9 Advanced Features Integration Example

This example demonstrates how all Phase 9 advanced features work together:
- Market regime detection and filtering
- Portfolio optimization systems
- Automated reporting with performance analysis
- News sentiment analysis and blackout rules
- Dynamic parameter optimization

This integration showcases the sophisticated trading system capabilities
with adaptive behavior based on market conditions.

Author: Trading Bot Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import Phase 9 components
from regime_detector import RegimeDetector, MarketRegime
from portfolio_optimizer import PortfolioOptimizer, OptimizationMethod, OptimizationConstraints
from automated_reporter import AutomatedReporter, PerformanceData, ReportConfig, ReportType, ReportFormat
from news_analyzer import NewsAnalyzer, NewsArticle, NewsCategory, SentimentLevel
from parameter_optimizer import ParameterOptimizer, ParameterBounds, OptimizationMethod as ParamOptMethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTradingSystem:
    """
    Integrated Advanced Trading System
    
    This class demonstrates the integration of all Phase 9 advanced features
    into a cohesive trading system with sophisticated market analysis and
    adaptive behavior.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the advanced trading system"""
        self.config = config or self._get_default_config()
        
        # Initialize all Phase 9 components
        self.regime_detector = RegimeDetector(self.config.get('regime_detection', {}))
        self.portfolio_optimizer = PortfolioOptimizer(self.config.get('portfolio_optimization', {}))
        self.automated_reporter = AutomatedReporter(self.config.get('automated_reporting', {}))
        self.news_analyzer = NewsAnalyzer(self.config.get('news_analysis', {}))
        self.parameter_optimizer = ParameterOptimizer(self.config.get('parameter_optimization', {}))
        
        # System state
        self.current_regime = None
        self.current_sentiment = None
        self.current_portfolio = {}
        self.current_parameters = {}
        self.trading_halted = False
        self.halt_reason = ""
        
        # Performance tracking
        self.performance_history = []
        self.regime_performance = {}
        self.optimization_schedule = []
        
        logger.info("AdvancedTradingSystem initialized with all Phase 9 components")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the integrated system"""
        return {
            'system': {
                'update_frequency_minutes': 15,
                'regime_update_frequency_minutes': 60,
                'portfolio_rebalance_frequency_hours': 24,
                'parameter_optimization_frequency_days': 7,
                'news_analysis_frequency_minutes': 30
            },
            'integration': {
                'regime_weight': 0.3,
                'sentiment_weight': 0.2,
                'performance_weight': 0.3,
                'optimization_weight': 0.2,
                'use_regime_filtering': True,
                'use_sentiment_filtering': True,
                'use_adaptive_parameters': True
            },
            'thresholds': {
                'regime_confidence_threshold': 0.7,
                'sentiment_confidence_threshold': 0.6,
                'parameter_drift_threshold': 2.0,
                'performance_degradation_threshold': 0.2
            }
        }
    
    async def run_comprehensive_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive market analysis using all Phase 9 components
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            Dictionary with analysis results from all components
        """
        try:
            logger.info("Starting comprehensive market analysis")
            
            # 1. Market Regime Detection
            logger.info("Analyzing market regime...")
            regime_classification = self.regime_detector.detect_regime(market_data, method='combined')
            self.current_regime = regime_classification.regime
            
            # 2. News Sentiment Analysis
            logger.info("Analyzing news sentiment...")
            articles = await self.news_analyzer.fetch_news_articles(hours_back=24)
            sentiment_analysis = self.news_analyzer.analyze_sentiment(articles)
            self.current_sentiment = sentiment_analysis
            
            # 3. Check for trading halt conditions
            should_halt_regime = not self.regime_detector.should_trade_in_regime(
                regime_classification.regime, 
                'trend_following'  # Example strategy type
            )
            
            should_halt_sentiment, sentiment_reason = self.news_analyzer.should_halt_trading('conservative')
            
            self.trading_halted = should_halt_regime or should_halt_sentiment
            self.halt_reason = "Regime filter" if should_halt_regime else sentiment_reason if should_halt_sentiment else ""
            
            # 4. Portfolio Optimization (if not halted)
            portfolio_result = None
            if not self.trading_halted:
                logger.info("Optimizing portfolio allocation...")
                
                # Create sample returns data for optimization
                returns_data = market_data['close'].pct_change().dropna()
                returns_df = pd.DataFrame({
                    'BTCUSDT': returns_data,
                    'ETHUSDT': returns_data * 0.8 + np.random.normal(0, 0.01, len(returns_data)),
                    'SOLUSDT': returns_data * 1.2 + np.random.normal(0, 0.015, len(returns_data))
                })
                
                self.portfolio_optimizer.load_data(returns_df)
                
                # Regime-aware optimization method selection
                if regime_classification.regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]:
                    optimization_method = OptimizationMethod.MIN_VARIANCE
                elif regime_classification.regime in [MarketRegime.BULL_MARKET, MarketRegime.BUBBLE]:
                    optimization_method = OptimizationMethod.MAX_SHARPE
                else:
                    optimization_method = OptimizationMethod.RISK_PARITY
                
                constraints = OptimizationConstraints(
                    min_weight=0.0,
                    max_weight=0.5,
                    max_turnover=0.3
                )
                
                portfolio_result = self.portfolio_optimizer.optimize_portfolio(
                    optimization_method,
                    constraints,
                    self.current_portfolio
                )
                
                self.current_portfolio = portfolio_result.weights
            
            # 5. Parameter Optimization Check
            logger.info("Checking parameter optimization needs...")
            
            # Simulate current performance metrics
            current_performance = {
                'sharpe_ratio': 1.2 + np.random.normal(0, 0.1),
                'max_drawdown': -0.08 + np.random.normal(0, 0.02),
                'return': 0.15 + np.random.normal(0, 0.03),
                'volatility': 0.18 + np.random.normal(0, 0.02)
            }
            
            # Check if parameter reoptimization is needed
            should_reoptimize, reopt_reason = self.parameter_optimizer.should_reoptimize(current_performance)
            
            parameter_result = None
            if should_reoptimize:
                logger.info(f"Reoptimizing parameters: {reopt_reason}")
                parameter_result = await self._optimize_parameters()
            
            # Compile comprehensive analysis
            analysis_result = {
                'timestamp': datetime.now(),
                'regime_analysis': {
                    'regime': regime_classification.regime.value,
                    'confidence': regime_classification.confidence,
                    'persistence_score': regime_classification.persistence_score,
                    'should_trade': not should_halt_regime
                },
                'sentiment_analysis': {
                    'overall_sentiment': sentiment_analysis.overall_sentiment.value,
                    'sentiment_score': sentiment_analysis.sentiment_score,
                    'confidence': sentiment_analysis.confidence,
                    'impact_score': sentiment_analysis.impact_score,
                    'blackout_recommended': sentiment_analysis.blackout_recommended,
                    'article_count': sentiment_analysis.article_count
                },
                'portfolio_optimization': {
                    'method_used': optimization_method.value if portfolio_result else None,
                    'optimal_weights': portfolio_result.weights if portfolio_result else {},
                    'expected_return': portfolio_result.expected_return if portfolio_result else 0.0,
                    'expected_volatility': portfolio_result.expected_volatility if portfolio_result else 0.0,
                    'sharpe_ratio': portfolio_result.sharpe_ratio if portfolio_result else 0.0,
                    'diversification_ratio': portfolio_result.diversification_ratio if portfolio_result else 0.0
                },
                'parameter_optimization': {
                    'reoptimization_needed': should_reoptimize,
                    'reason': reopt_reason,
                    'optimal_parameters': parameter_result.optimal_parameters if parameter_result else {},
                    'objective_value': parameter_result.objective_value if parameter_result else 0.0
                },
                'trading_decision': {
                    'trading_halted': self.trading_halted,
                    'halt_reason': self.halt_reason,
                    'recommended_action': self._generate_trading_recommendation(
                        regime_classification, sentiment_analysis, portfolio_result
                    )
                },
                'performance_metrics': current_performance
            }
            
            # Store in history
            self.performance_history.append(analysis_result)
            
            # Keep history manageable
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
            
            logger.info("Comprehensive analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    async def _optimize_parameters(self) -> Any:
        """Optimize trading parameters based on current conditions"""
        try:
            # Define parameter space for optimization
            parameter_bounds = [
                ParameterBounds("lookback_period", 10, 100, 1, "discrete"),
                ParameterBounds("risk_multiplier", 0.5, 3.0, None, "continuous"),
                ParameterBounds("volatility_threshold", 0.01, 0.05, None, "continuous"),
                ParameterBounds("rebalance_threshold", 0.05, 0.20, None, "continuous")
            ]
            
            parameter_space = self.parameter_optimizer.define_parameter_space(parameter_bounds)
            
            # Create objective function that considers current regime and sentiment
            def regime_aware_objective(params):
                base_score = 1.0
                
                # Adjust score based on current regime
                if self.current_regime == MarketRegime.HIGH_VOLATILITY:
                    # Favor conservative parameters
                    if params.get('risk_multiplier', 1.5) < 1.5:
                        base_score += 0.2
                    if params.get('volatility_threshold', 0.03) > 0.03:
                        base_score += 0.1
                elif self.current_regime == MarketRegime.BULL_MARKET:
                    # Favor more aggressive parameters
                    if params.get('risk_multiplier', 1.5) > 1.5:
                        base_score += 0.2
                
                # Adjust score based on sentiment
                if self.current_sentiment and self.current_sentiment.sentiment_score < -0.5:
                    # Negative sentiment - be more conservative
                    if params.get('risk_multiplier', 1.5) < 1.2:
                        base_score += 0.15
                elif self.current_sentiment and self.current_sentiment.sentiment_score > 0.5:
                    # Positive sentiment - can be more aggressive
                    if params.get('risk_multiplier', 1.5) > 1.3:
                        base_score += 0.15
                
                # Add some noise for realistic optimization
                base_score += np.random.normal(0, 0.05)
                
                return max(0, base_score)
            
            # Optimize parameters
            result = self.parameter_optimizer.optimize_parameters(
                regime_aware_objective,
                parameter_space,
                ParamOptMethod.BAYESIAN,
                max_iterations=30
            )
            
            self.current_parameters.update(result.optimal_parameters)
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return None
    
    def _generate_trading_recommendation(self, regime_classification, sentiment_analysis, portfolio_result) -> str:
        """Generate comprehensive trading recommendation"""
        try:
            recommendations = []
            
            # Regime-based recommendations
            if regime_classification.confidence > self.config['thresholds']['regime_confidence_threshold']:
                if regime_classification.regime == MarketRegime.BULL_MARKET:
                    recommendations.append("Strong bull market detected - consider increasing exposure")
                elif regime_classification.regime == MarketRegime.BEAR_MARKET:
                    recommendations.append("Bear market detected - reduce exposure and consider hedging")
                elif regime_classification.regime == MarketRegime.HIGH_VOLATILITY:
                    recommendations.append("High volatility regime - reduce position sizes")
                elif regime_classification.regime == MarketRegime.CRASH:
                    recommendations.append("Market crash detected - halt trading and preserve capital")
                elif regime_classification.regime == MarketRegime.SIDEWAYS:
                    recommendations.append("Sideways market - use mean reversion strategies")
            
            # Sentiment-based recommendations
            if sentiment_analysis.confidence > self.config['thresholds']['sentiment_confidence_threshold']:
                if sentiment_analysis.overall_sentiment == SentimentLevel.VERY_POSITIVE:
                    recommendations.append("Very positive news sentiment - monitor for FOMO conditions")
                elif sentiment_analysis.overall_sentiment == SentimentLevel.VERY_NEGATIVE:
                    recommendations.append("Very negative news sentiment - exercise extreme caution")
                
                if sentiment_analysis.blackout_recommended:
                    recommendations.append("News blackout recommended - halt trading temporarily")
            
            # Portfolio optimization recommendations
            if portfolio_result and portfolio_result.sharpe_ratio > 1.5:
                recommendations.append(f"Optimal portfolio identified with Sharpe ratio of {portfolio_result.sharpe_ratio:.2f}")
            elif portfolio_result and portfolio_result.sharpe_ratio < 0.5:
                recommendations.append("Portfolio optimization suggests poor risk-adjusted returns - review strategy")
            
            # Combined analysis
            if self.trading_halted:
                recommendations.append(f"Trading halted: {self.halt_reason}")
            else:
                recommendations.append("Conditions favorable for trading")
            
            return "; ".join(recommendations) if recommendations else "No specific recommendations at this time"
            
        except Exception as e:
            logger.error(f"Error generating trading recommendation: {e}")
            return "Unable to generate recommendation due to analysis error"
    
    async def generate_comprehensive_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a comprehensive report using the automated reporter"""
        try:
            # Create performance data from analysis result
            performance_data = PerformanceData(
                portfolio_returns=pd.Series([0.01, -0.005, 0.02, 0.008]),  # Sample data
                benchmark_returns=pd.Series([0.008, -0.003, 0.015, 0.006]),  # Sample data
                regime_data=analysis_result.get('regime_analysis', {}),
                optimization_results=analysis_result.get('portfolio_optimization', {})
            )
            
            # Generate daily report
            report_data = self.automated_reporter.generate_daily_report(performance_data)
            
            # Add advanced analysis sections
            report_data['advanced_analysis'] = {
                'regime_analysis': analysis_result.get('regime_analysis', {}),
                'sentiment_analysis': analysis_result.get('sentiment_analysis', {}),
                'parameter_optimization': analysis_result.get('parameter_optimization', {}),
                'trading_decision': analysis_result.get('trading_decision', {})
            }
            
            # Create HTML report
            html_report = self.automated_reporter.create_html_report(report_data)
            
            # Save report
            report_config = ReportConfig(
                report_type=ReportType.DAILY,
                report_format=ReportFormat.HTML,
                include_regime_analysis=True
            )
            
            output_path = self.automated_reporter.save_report(report_data, report_config)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return ""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get component statuses
            regime_stats = self.regime_detector.get_regime_statistics()
            sentiment_summary = self.news_analyzer.get_sentiment_summary()
            optimization_summary = self.parameter_optimizer.get_optimization_summary()
            
            # Calculate system health score
            health_score = 1.0
            
            # Regime detection health
            if regime_stats.get('current_persistence', 0) < 0.5:
                health_score -= 0.1
            
            # Sentiment analysis health
            if sentiment_summary.get('latest_confidence', 0) < 0.5:
                health_score -= 0.1
            
            # Parameter optimization health
            if optimization_summary.get('best_objective_value', 0) < 0.5:
                health_score -= 0.1
            
            # Trading status
            trading_status = "ACTIVE" if not self.trading_halted else "HALTED"
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_health_score': float(max(0, health_score)),
                'trading_status': trading_status,
                'halt_reason': self.halt_reason,
                'current_regime': self.current_regime.value if self.current_regime else None,
                'current_sentiment': self.current_sentiment.overall_sentiment.value if self.current_sentiment else None,
                'component_status': {
                    'regime_detector': {
                        'status': 'active',
                        'statistics': regime_stats
                    },
                    'news_analyzer': {
                        'status': 'active',
                        'summary': sentiment_summary
                    },
                    'portfolio_optimizer': {
                        'status': 'active',
                        'current_portfolio': self.current_portfolio
                    },
                    'parameter_optimizer': {
                        'status': 'active',
                        'summary': optimization_summary
                    },
                    'automated_reporter': {
                        'status': 'active',
                        'last_report': None  # Would track last report generation
                    }
                },
                'performance_history_length': len(self.performance_history),
                'current_parameters': self.current_parameters
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}


async def main():
    """Main function demonstrating the integrated advanced trading system"""
    print("Phase 9 Advanced Features Integration Demo")
    print("=" * 60)
    
    # Initialize the integrated system
    system = AdvancedTradingSystem()
    
    # Create sample market data
    print("\nGenerating sample market data...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate Bitcoin price data with different regime periods
    prices = []
    base_price = 30000
    
    for i, date in enumerate(dates):
        # Simulate different market regimes
        if i < 100:  # Bull market
            daily_return = np.random.normal(0.002, 0.015)
        elif i < 200:  # Sideways
            daily_return = np.random.normal(0, 0.012)
        elif i < 220:  # Crash
            daily_return = np.random.normal(-0.008, 0.04)
        elif i < 300:  # Recovery
            daily_return = np.random.normal(0.003, 0.025)
        else:  # High volatility
            daily_return = np.random.normal(0.001, 0.035)
        
        base_price *= (1 + daily_return)
        prices.append(base_price)
    
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'volume': np.random.uniform(1000000, 5000000, len(prices))
    })
    market_data['open'] = market_data['close'].shift(1)
    market_data = market_data.dropna()
    
    print(f"Generated {len(market_data)} days of market data")
    
    # Run comprehensive analysis
    print("\nRunning comprehensive market analysis...")
    analysis_result = await system.run_comprehensive_analysis(market_data)
    
    if 'error' not in analysis_result:
        print("\n" + "="*50)
        print("COMPREHENSIVE ANALYSIS RESULTS")
        print("="*50)
        
        # Display regime analysis
        regime_analysis = analysis_result.get('regime_analysis', {})
        print(f"\n📊 MARKET REGIME ANALYSIS:")
        print(f"   Current Regime: {regime_analysis.get('regime', 'Unknown')}")
        print(f"   Confidence: {regime_analysis.get('confidence', 0):.3f}")
        print(f"   Persistence Score: {regime_analysis.get('persistence_score', 0):.3f}")
        print(f"   Should Trade: {'✓' if regime_analysis.get('should_trade', False) else '✗'}")
        
        # Display sentiment analysis
        sentiment_analysis = analysis_result.get('sentiment_analysis', {})
        print(f"\n📰 NEWS SENTIMENT ANALYSIS:")
        print(f"   Overall Sentiment: {sentiment_analysis.get('overall_sentiment', 'Unknown')}")
        print(f"   Sentiment Score: {sentiment_analysis.get('sentiment_score', 0):+.3f}")
        print(f"   Confidence: {sentiment_analysis.get('confidence', 0):.3f}")
        print(f"   Impact Score: {sentiment_analysis.get('impact_score', 0):.3f}")
        print(f"   Articles Analyzed: {sentiment_analysis.get('article_count', 0)}")
        print(f"   Blackout Recommended: {'⚠️ YES' if sentiment_analysis.get('blackout_recommended', False) else '✓ NO'}")
        
        # Display portfolio optimization
        portfolio_opt = analysis_result.get('portfolio_optimization', {})
        print(f"\n💼 PORTFOLIO OPTIMIZATION:")
        print(f"   Method Used: {portfolio_opt.get('method_used', 'None')}")
        print(f"   Expected Return: {portfolio_opt.get('expected_return', 0):.4f}")
        print(f"   Expected Volatility: {portfolio_opt.get('expected_volatility', 0):.4f}")
        print(f"   Sharpe Ratio: {portfolio_opt.get('sharpe_ratio', 0):.3f}")
        print(f"   Diversification Ratio: {portfolio_opt.get('diversification_ratio', 0):.3f}")
        
        if portfolio_opt.get('optimal_weights'):
            print(f"   Optimal Weights:")
            for asset, weight in portfolio_opt['optimal_weights'].items():
                print(f"     {asset}: {weight:.3f} ({weight*100:.1f}%)")
        
        # Display parameter optimization
        param_opt = analysis_result.get('parameter_optimization', {})
        print(f"\n⚙️ PARAMETER OPTIMIZATION:")
        print(f"   Reoptimization Needed: {'⚠️ YES' if param_opt.get('reoptimization_needed', False) else '✓ NO'}")
        print(f"   Reason: {param_opt.get('reason', 'N/A')}")
        print(f"   Objective Value: {param_opt.get('objective_value', 0):.4f}")
        
        if param_opt.get('optimal_parameters'):
            print(f"   Optimal Parameters:")
            for param, value in param_opt['optimal_parameters'].items():
                print(f"     {param}: {value}")
        
        # Display trading decision
        trading_decision = analysis_result.get('trading_decision', {})
        print(f"\n🎯 TRADING DECISION:")
        print(f"   Trading Status: {'🛑 HALTED' if trading_decision.get('trading_halted', False) else '✅ ACTIVE'}")
        if trading_decision.get('halt_reason'):
            print(f"   Halt Reason: {trading_decision['halt_reason']}")
        print(f"   Recommendation: {trading_decision.get('recommended_action', 'N/A')}")
        
        # Display performance metrics
        performance = analysis_result.get('performance_metrics', {})
        print(f"\n📈 PERFORMANCE METRICS:")
        for metric, value in performance.items():
            print(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\n" + "="*50)
        
        # Generate comprehensive report
        print("\nGenerating comprehensive report...")
        report_path = await system.generate_comprehensive_report(analysis_result)
        if report_path:
            print(f"Report saved to: {report_path}")
        else:
            print("Report generation failed")
        
        # Get system status
        print("\nGetting system status...")
        status = system.get_system_status()
        
        print(f"\n🔧 SYSTEM STATUS:")
        print(f"   System Health Score: {status.get('system_health_score', 0):.3f}")
        print(f"   Trading Status: {status.get('trading_status', 'Unknown')}")
        print(f"   Performance History: {status.get('performance_history_length', 0)} records")
        
        component_status = status.get('component_status', {})
        print(f"   Component Status:")
        for component, comp_status in component_status.items():
            print(f"     {component.replace('_', ' ').title()}: {comp_status.get('status', 'unknown')}")
        
    else:
        print(f"❌ Analysis failed: {analysis_result['error']}")
    
    print(f"\n🎉 Phase 9 Advanced Features Integration Demo Complete!")
    print(f"    ✅ Market Regime Detection - Implemented")
    print(f"    ✅ Portfolio Optimization - Implemented") 
    print(f"    ✅ Automated Reporting - Implemented")
    print(f"    ✅ News Sentiment Analysis - Implemented")
    print(f"    ✅ Dynamic Parameter Optimization - Implemented")
    print(f"    ✅ System Integration - Implemented")
    
    print(f"\n💡 The system now provides:")
    print(f"    • Sophisticated market regime detection with 8 different regimes")
    print(f"    • Advanced portfolio optimization using multiple methods")
    print(f"    • Comprehensive automated reporting with performance analytics")
    print(f"    • Real-time news sentiment analysis with trading halt triggers")
    print(f"    • Dynamic parameter optimization with drift detection")
    print(f"    • Integrated decision-making based on all analysis components")


if __name__ == "__main__":
    asyncio.run(main())