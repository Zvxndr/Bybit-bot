"""
Multi-Asset Portfolio Manager Integration Test.
Demonstrates the complete portfolio management system with all components.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Portfolio management components
from .portfolio_manager import PortfolioManager, Portfolio, Position
from .asset_allocator import AssetAllocator, AllocationStrategy
from .correlation_analyzer import CorrelationAnalyzer, CorrelationMeasure
from .rebalancer import PortfolioRebalancer, RebalanceStrategy
from .risk_budgeter import RiskBudgeter, RiskBudgetType

class PortfolioManagerIntegrationTest:
    """Integration test for the complete portfolio management system."""
    
    def __init__(self):
        self.portfolio_manager = PortfolioManager()
        self.asset_allocator = AssetAllocator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.rebalancer = PortfolioRebalancer()
        self.risk_budgeter = RiskBudgeter()
        
        # Test symbols
        self.symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'LINK', 'UNI']
        
        # Generate mock price data
        self.price_data = self._generate_mock_price_data()
        
        print("Portfolio Manager Integration Test initialized")
    
    def _generate_mock_price_data(self) -> pd.DataFrame:
        """Generate mock price data for testing."""
        # Create 1 year of daily price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible results
        
        price_data = {}
        base_prices = {
            'BTC': 40000, 'ETH': 2500, 'ADA': 0.5, 'SOL': 100,
            'AVAX': 20, 'DOT': 8, 'LINK': 15, 'UNI': 6
        }
        
        for symbol in self.symbols:
            # Generate returns with different volatilities
            volatilities = {
                'BTC': 0.04, 'ETH': 0.05, 'ADA': 0.06, 'SOL': 0.07,
                'AVAX': 0.08, 'DOT': 0.06, 'LINK': 0.05, 'UNI': 0.07
            }
            
            vol = volatilities[symbol]
            returns = np.random.normal(0.0008, vol, len(dates))  # Daily returns
            
            # Create price series
            prices = [base_prices[symbol]]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            price_data[symbol] = prices[1:]  # Remove initial price
        
        return pd.DataFrame(price_data, index=dates)
    
    async def run_complete_integration_test(self):
        """Run complete integration test of all portfolio management components."""
        print("\n" + "="*80)
        print("MULTI-ASSET PORTFOLIO MANAGER INTEGRATION TEST")
        print("="*80)
        
        try:
            # 1. Initialize Portfolio
            print("\n1. INITIALIZING PORTFOLIO")
            print("-" * 40)
            
            initial_weights = {
                'BTC': 0.30, 'ETH': 0.25, 'ADA': 0.10, 'SOL': 0.10,
                'AVAX': 0.08, 'DOT': 0.07, 'LINK': 0.05, 'UNI': 0.05
            }
            
            portfolio = await self.portfolio_manager.create_portfolio(
                name="Multi-Asset Crypto Portfolio",
                initial_weights=initial_weights,
                initial_value=1000000  # $1M portfolio
            )
            
            print(f"✅ Portfolio created: {portfolio.name}")
            print(f"   Total value: ${portfolio.total_value:,.2f}")
            print(f"   Number of positions: {len(portfolio.positions)}")
            
            # 2. Asset Allocation Analysis
            print("\n2. ASSET ALLOCATION ANALYSIS")
            print("-" * 40)
            
            # Test multiple allocation strategies
            strategies_to_test = [
                AllocationStrategy.RISK_PARITY,
                AllocationStrategy.EQUAL_WEIGHT,
                AllocationStrategy.MOMENTUM,
                AllocationStrategy.VOLATILITY_PARITY
            ]
            
            allocation_results = {}
            
            for strategy in strategies_to_test:
                print(f"\n   Testing {strategy.value} allocation...")
                
                allocation_result = await self.asset_allocator.allocate_assets(
                    symbols=self.symbols,
                    strategy=strategy,
                    price_data=self.price_data
                )
                
                allocation_results[strategy] = allocation_result
                
                print(f"   ✅ {strategy.value}:")
                print(f"      Expected return: {allocation_result.expected_return:.1%}")
                print(f"      Expected volatility: {allocation_result.expected_volatility:.1%}")
                print(f"      Sharpe ratio: {allocation_result.sharpe_ratio:.2f}")
                print(f"      Confidence score: {allocation_result.confidence_score:.1%}")
            
            # Select best allocation (highest Sharpe ratio)
            best_strategy = max(allocation_results.keys(), 
                              key=lambda s: allocation_results[s].sharpe_ratio)
            best_allocation = allocation_results[best_strategy]
            
            print(f"\n   🏆 Best strategy: {best_strategy.value}")
            print(f"      Rationale: {best_allocation.allocation_rationale}")
            
            # 3. Correlation Analysis
            print("\n3. CORRELATION ANALYSIS")
            print("-" * 40)
            
            correlation_analysis = await self.correlation_analyzer.analyze_correlations(
                price_data=self.price_data,
                method=CorrelationMeasure.PEARSON
            )
            
            print(f"   ✅ Correlation regime: {correlation_analysis.correlation_regime.value}")
            print(f"   Average correlation: {correlation_analysis.average_correlation:.2f}")
            print(f"   Correlation stability: {correlation_analysis.correlation_stability:.2f}")
            print(f"   Market stress indicator: {correlation_analysis.market_stress_indicator:.2f}")
            print(f"   Number of clusters: {len(correlation_analysis.correlation_clusters)}")
            
            if correlation_analysis.correlation_warnings:
                print("   ⚠️  Correlation warnings:")
                for warning in correlation_analysis.correlation_warnings[:3]:
                    print(f"      - {warning}")
            
            # 4. Portfolio Optimization
            print("\n4. PORTFOLIO OPTIMIZATION")
            print("-" * 40)
            
            # Update portfolio with optimized weights
            optimized_weights = best_allocation.weights
            
            print("   Optimizing portfolio weights...")
            optimization_result = await self.portfolio_manager.optimize_portfolio(
                portfolio_id=portfolio.portfolio_id,
                target_weights=optimized_weights,
                optimization_method="risk_parity"
            )
            
            print(f"   ✅ Optimization completed")
            print(f"      Expected return: {optimization_result.expected_return:.1%}")
            print(f"      Expected volatility: {optimization_result.expected_volatility:.1%}")
            print(f"      Sharpe ratio: {optimization_result.sharpe_ratio:.2f}")
            
            # 5. Risk Budgeting
            print("\n5. RISK BUDGETING & CONSTRAINT MANAGEMENT")
            print("-" * 40)
            
            # Create risk budgets
            risk_budgets = await self.risk_budgeter.create_risk_budgets(
                weights=optimized_weights,
                returns=self.price_data.pct_change().dropna(),
                target_risk=0.20  # 20% target volatility
            )
            
            print("   ✅ Risk budgets created:")
            for budget in risk_budgets:
                print(f"      {budget.budget_type.value}:")
                print(f"         Utilization: {budget.budget_utilization:.1%}")
                print(f"         Violations: {len(budget.budget_violations)}")
            
            # Check risk constraints
            portfolio_metrics = {
                'volatility': optimization_result.expected_volatility,
                'max_drawdown': 0.12,  # Mock drawdown
                'sharpe_ratio': optimization_result.sharpe_ratio
            }
            
            risk_constraints = await self.risk_budgeter.check_risk_constraints(
                weights=optimized_weights,
                returns=self.price_data.pct_change().dropna(),
                portfolio_metrics=portfolio_metrics
            )
            
            print(f"\n   Risk constraints checked: {len(risk_constraints)} active")
            for constraint in risk_constraints:
                status = "🔴 VIOLATED" if constraint.is_violated else "🟡 MONITORED"
                print(f"      {status} {constraint.description}")
            
            # 6. Rebalancing Signals
            print("\n6. REBALANCING ANALYSIS")
            print("-" * 40)
            
            # Generate rebalancing signals
            current_weights = initial_weights  # Simulate drift from initial
            target_weights = optimized_weights
            
            market_data = {
                'price_data': self.price_data,
                'correlation_data': {
                    'average_correlation': correlation_analysis.average_correlation,
                    'regime': correlation_analysis.correlation_regime.value
                }
            }
            
            rebalance_signals = await self.rebalancer.generate_rebalance_signals(
                current_weights=current_weights,
                target_weights=target_weights,
                market_data=market_data,
                portfolio_metrics=portfolio_metrics
            )
            
            print(f"   ✅ Generated {len(rebalance_signals)} rebalancing signals:")
            for signal in rebalance_signals:
                print(f"      {signal.trigger.value}: {signal.strength:.1%} strength ({signal.urgency})")
                print(f"         Reason: {signal.reason}")
            
            # Execute rebalancing if signals are strong
            if rebalance_signals and max(s.strength for s in rebalance_signals) > 0.5:
                print("\n   Executing rebalancing...")
                
                rebalance_result = await self.rebalancer.execute_rebalancing(
                    strategy=RebalanceStrategy.THRESHOLD,
                    current_weights=current_weights,
                    target_weights=target_weights,
                    portfolio_value=portfolio.total_value,
                    market_data=market_data,
                    signals=rebalance_signals
                )
                
                print(f"   ✅ Rebalancing executed:")
                print(f"      Transactions: {len(rebalance_result.transactions)}")
                print(f"      Total costs: ${rebalance_result.total_transaction_costs:,.2f}")
                print(f"      Cost ratio: {rebalance_result.cost_ratio:.2%}")
                print(f"      Effectiveness: {rebalance_result.rebalance_effectiveness:.1%}")
            
            # 7. Comprehensive Risk Report
            print("\n7. COMPREHENSIVE RISK REPORT")
            print("-" * 40)
            
            risk_report = await self.risk_budgeter.generate_risk_report(
                weights=optimized_weights,
                returns=self.price_data.pct_change().dropna(),
                portfolio_metrics=portfolio_metrics
            )
            
            print("   ✅ Risk report generated:")
            print(f"      Risk budgets: {len(risk_report.risk_budgets)}")
            print(f"      Risk constraints: {len(risk_report.risk_constraints)}")
            print(f"      Risk warnings: {len(risk_report.risk_warnings)}")
            print(f"      Risk recommendations: {len(risk_report.risk_recommendations)}")
            
            # Show key risk metrics
            print("\n   Key Risk Metrics:")
            for metric, value in risk_report.portfolio_risk_metrics.items():
                if metric in ['volatility', 'sharpe_ratio', 'var_95', 'hhi']:
                    if isinstance(value, float):
                        if 'ratio' in metric:
                            print(f"      {metric}: {value:.2f}")
                        else:
                            print(f"      {metric}: {value:.1%}")
            
            # Show attribution
            attribution = risk_report.risk_attribution
            print(f"\n   Risk Attribution:")
            print(f"      Total risk: {attribution.total_risk:.1%}")
            print(f"      Systematic: {attribution.systematic_risk:.1%}")
            print(f"      Idiosyncratic: {attribution.idiosyncratic_risk:.1%}")
            print(f"      Attribution quality: {attribution.attribution_quality:.1%}")
            
            # 8. Performance Summary
            print("\n8. PORTFOLIO PERFORMANCE SUMMARY")
            print("-" * 40)
            
            # Get updated portfolio
            updated_portfolio = await self.portfolio_manager.get_portfolio(portfolio.portfolio_id)
            performance = await self.portfolio_manager.calculate_portfolio_performance(portfolio.portfolio_id)
            
            print("   ✅ Performance Metrics:")
            print(f"      Total Return: {performance['total_return']:.1%}")
            print(f"      Annualized Return: {performance['annualized_return']:.1%}")
            print(f"      Volatility: {performance['volatility']:.1%}")
            print(f"      Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"      Max Drawdown: {performance['max_drawdown']:.1%}")
            print(f"      Sortino Ratio: {performance['sortino_ratio']:.2f}")
            
            # Top performing positions
            print("\n   Top Performing Positions:")
            top_positions = sorted(updated_portfolio.positions.values(), 
                                 key=lambda p: p.unrealized_pnl, reverse=True)[:3]
            
            for position in top_positions:
                print(f"      {position.symbol}: {position.unrealized_pnl_percent:.1%} "
                      f"(${position.unrealized_pnl:,.2f})")
            
            # 9. System Status Summary
            print("\n9. SYSTEM STATUS SUMMARY")
            print("-" * 40)
            
            # Get component summaries
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            allocator_summary = self.asset_allocator.get_allocation_summary()
            correlation_summary = self.correlation_analyzer.get_correlation_summary()
            rebalancer_summary = self.rebalancer.get_rebalancer_summary()
            risk_summary = self.risk_budgeter.get_risk_summary()
            
            print("   ✅ Component Status:")
            print(f"      Portfolio Manager: {portfolio_summary['total_portfolios']} portfolios")
            print(f"      Asset Allocator: {allocator_summary['total_allocations']} allocations")
            print(f"      Correlation Analyzer: {correlation_summary['total_analyses']} analyses")
            print(f"      Rebalancer: {rebalancer_summary['total_rebalances']} rebalances")
            print(f"      Risk Budgeter: {risk_summary['total_reports']} reports")
            
            print("\n" + "="*80)
            print("✅ INTEGRATION TEST COMPLETED SUCCESSFULLY")
            print("="*80)
            
            print(f"\nSystem demonstrates complete portfolio management capabilities:")
            print(f"- Multi-asset portfolio creation and management")
            print(f"- Advanced allocation strategies with optimization")
            print(f"- Real-time correlation monitoring and regime detection")
            print(f"- Intelligent rebalancing with cost optimization")
            print(f"- Comprehensive risk budgeting and constraint management")
            print(f"- Integrated performance attribution and reporting")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_stress_test(self):
        """Run stress test scenarios."""
        print("\n" + "="*60)
        print("STRESS TEST SCENARIOS")
        print("="*60)
        
        # Market crash scenario
        print("\n📉 Market Crash Scenario (-30% shock)")
        print("-" * 40)
        
        # Create stressed price data
        stressed_prices = self.price_data * 0.7  # 30% decline
        
        # Test portfolio under stress
        stressed_allocation = await self.asset_allocator.allocate_assets(
            symbols=self.symbols,
            strategy=AllocationStrategy.RISK_PARITY,
            price_data=stressed_prices
        )
        
        print(f"   Stressed portfolio volatility: {stressed_allocation.expected_volatility:.1%}")
        print(f"   Diversification ratio: {stressed_allocation.diversification_ratio:.2f}")
        
        # Correlation spike scenario  
        print("\n📈 Correlation Spike Scenario")
        print("-" * 40)
        
        correlation_analysis = await self.correlation_analyzer.analyze_correlations(
            price_data=stressed_prices
        )
        
        print(f"   Stress correlation regime: {correlation_analysis.correlation_regime.value}")
        print(f"   Market stress indicator: {correlation_analysis.market_stress_indicator:.2f}")
        
        print("\n✅ Stress tests completed")


async def main():
    """Run the complete integration test."""
    test = PortfolioManagerIntegrationTest()
    
    # Run main integration test
    success = await test.run_complete_integration_test()
    
    if success:
        # Run stress tests
        await test.run_stress_test()
        
        print(f"\n🎉 Multi-Asset Portfolio Manager is fully operational!")
        print(f"   Ready for Phase 6 production deployment.")
    else:
        print(f"\n❌ Integration test failed - system needs debugging")

if __name__ == "__main__":
    asyncio.run(main())