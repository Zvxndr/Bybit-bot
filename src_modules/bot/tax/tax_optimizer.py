"""
Advanced Tax Loss Harvesting and Optimization System

This module provides sophisticated tax optimization strategies including:
- Real-time tax-loss harvesting opportunities
- Multi-asset portfolio optimization
- Wash sale rule compliance
- Tax-efficient rebalancing
- Long-term vs short-term optimization
- Cross-asset correlation analysis
- Dynamic tax rate optimization
- Regulatory compliance automation
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import scipy.optimize as optimize
from scipy.stats import norm
import warnings

from .tax_engine import TaxEngine, TaxEvent, TaxEventType, HoldingPeriod, Transaction, TransactionType
from .tax_reporter import TaxReporter

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Tax optimization strategies."""
    TAX_LOSS_HARVEST = "tax_loss_harvest"
    LONG_TERM_OPTIMIZATION = "long_term_optimization"
    WASH_SALE_AVOIDANCE = "wash_sale_avoidance"
    REBALANCING_OPTIMIZATION = "rebalancing_optimization"
    MULTI_ASSET_HARVEST = "multi_asset_harvest"
    DYNAMIC_TAX_RATE = "dynamic_tax_rate"

class RiskTolerance(Enum):
    """Risk tolerance levels for optimization."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class OptimizationObjective(Enum):
    """Optimization objectives."""
    MAXIMIZE_TAX_SAVINGS = "maximize_tax_savings"
    MINIMIZE_WASH_SALES = "minimize_wash_sales"
    OPTIMIZE_HOLDING_PERIODS = "optimize_holding_periods"
    BALANCE_RISK_RETURN = "balance_risk_return"

@dataclass
class TaxOptimizationConfig:
    """Configuration for tax optimization system."""
    strategy: OptimizationStrategy
    objective: OptimizationObjective
    risk_tolerance: RiskTolerance
    
    # Tax parameters
    current_tax_bracket: Decimal = Decimal('0.24')
    projected_tax_bracket: Decimal = Decimal('0.24')
    state_tax_rate: Decimal = Decimal('0.05')
    
    # Optimization constraints
    max_wash_sale_risk: Decimal = Decimal('0.10')  # Maximum acceptable wash sale risk
    min_holding_period: int = 30  # Minimum days before considering sale
    max_portfolio_turnover: Decimal = Decimal('0.50')  # Maximum annual turnover
    
    # Rebalancing parameters
    rebalancing_threshold: Decimal = Decimal('0.05')  # 5% deviation threshold
    target_allocations: Dict[str, Decimal] = field(default_factory=dict)
    
    # Advanced features
    consider_transaction_costs: bool = True
    factor_market_conditions: bool = True
    use_predictive_modeling: bool = False
    parallel_processing: bool = True
    max_workers: int = 4

@dataclass
class OptimizationRecommendation:
    """Tax optimization recommendation."""
    id: str
    timestamp: datetime
    strategy: OptimizationStrategy
    asset: str
    action: str  # "SELL", "HOLD", "BUY", "REBALANCE"
    quantity: Decimal
    expected_tax_savings: Decimal
    risk_score: Decimal
    confidence: Decimal
    rationale: str
    
    # Implementation details
    execution_timeline: str  # "IMMEDIATE", "WITHIN_7_DAYS", "BEFORE_YEAR_END"
    wash_sale_risk: bool
    alternative_assets: List[str] = field(default_factory=list)
    
    # Impact analysis
    portfolio_impact: Dict[str, Any] = field(default_factory=dict)
    tax_impact: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HarvestingOpportunity:
    """Tax loss harvesting opportunity."""
    asset: str
    current_position: Decimal
    unrealized_loss: Decimal
    cost_basis: Decimal
    current_value: Decimal
    days_held: int
    
    # Tax impact
    potential_tax_savings: Decimal
    effective_tax_rate: Decimal
    holding_period_impact: str
    
    # Risk assessment
    wash_sale_risk_score: Decimal
    liquidity_risk: Decimal
    correlation_risk: Decimal
    
    # Execution details
    recommended_quantity: Decimal
    alternative_investments: List[str]
    execution_priority: int

class TaxOptimizer:
    """Advanced tax optimization and loss harvesting system."""
    
    def __init__(self, tax_engine: TaxEngine, config: TaxOptimizationConfig):
        self.tax_engine = tax_engine
        self.config = config
        
        # Market data and analytics
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_data: Dict[str, Decimal] = {}
        
        # Optimization results
        self.current_recommendations: List[OptimizationRecommendation] = []
        self.harvesting_opportunities: List[HarvestingOpportunity] = []
        self.optimization_history: List[Dict] = []
        
        # Performance tracking
        self.realized_tax_savings: Decimal = Decimal('0')
        self.optimization_score: Decimal = Decimal('0')
        
        logger.info(f"TaxOptimizer initialized with {config.strategy.value} strategy")
    
    async def analyze_optimization_opportunities(self, 
                                               current_date: Optional[datetime] = None) -> List[OptimizationRecommendation]:
        """Analyze and identify tax optimization opportunities."""
        if current_date is None:
            current_date = datetime.now()
        
        logger.info("Analyzing tax optimization opportunities...")
        
        try:
            # Update market data and analytics
            await self._update_market_analytics(current_date)
            
            # Generate recommendations based on strategy
            if self.config.strategy == OptimizationStrategy.TAX_LOSS_HARVEST:
                recommendations = await self._analyze_tax_loss_harvesting(current_date)
            elif self.config.strategy == OptimizationStrategy.LONG_TERM_OPTIMIZATION:
                recommendations = await self._analyze_long_term_optimization(current_date)
            elif self.config.strategy == OptimizationStrategy.WASH_SALE_AVOIDANCE:
                recommendations = await self._analyze_wash_sale_avoidance(current_date)
            elif self.config.strategy == OptimizationStrategy.REBALANCING_OPTIMIZATION:
                recommendations = await self._analyze_rebalancing_optimization(current_date)
            elif self.config.strategy == OptimizationStrategy.MULTI_ASSET_HARVEST:
                recommendations = await self._analyze_multi_asset_harvesting(current_date)
            else:
                recommendations = await self._analyze_comprehensive_optimization(current_date)
            
            # Filter and rank recommendations
            recommendations = self._filter_and_rank_recommendations(recommendations, current_date)
            
            # Update tracking
            self.current_recommendations = recommendations
            self._update_optimization_history(recommendations, current_date)
            
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to analyze optimization opportunities: {e}")
            raise
    
    async def _update_market_analytics(self, current_date: datetime):
        """Update market data and analytics for optimization."""
        logger.debug("Updating market analytics...")
        
        # Get current portfolio holdings
        portfolio = self.tax_engine.get_portfolio_summary()
        assets = list(portfolio['holdings'].keys())
        
        if not assets:
            return
        
        # Update price data (placeholder - would integrate with real market data)
        await self._fetch_price_data(assets, current_date)
        
        # Calculate correlation matrix
        if len(assets) > 1:
            self.correlation_matrix = await self._calculate_correlation_matrix(assets)
        
        # Update volatility estimates
        await self._update_volatility_estimates(assets)
    
    async def _fetch_price_data(self, assets: List[str], current_date: datetime):
        """Fetch historical price data for assets."""
        # Placeholder for market data integration
        # In production, this would fetch from exchanges or data providers
        
        for asset in assets:
            # Simulate price data with some realistic patterns
            dates = pd.date_range(
                start=current_date - timedelta(days=365),
                end=current_date,
                freq='D'
            )
            
            # Generate synthetic price data
            np.random.seed(hash(asset) % 2**32)  # Consistent seed per asset
            returns = np.random.normal(0.001, 0.02, len(dates))
            
            # Add some market regime changes
            regime_changes = np.random.choice(len(dates), 3, replace=False)
            for change_point in regime_changes:
                if change_point < len(returns) - 30:
                    returns[change_point:change_point+30] *= 2.0  # Increased volatility
            
            prices = 100 * np.exp(np.cumsum(returns))
            
            self.price_data[asset] = pd.DataFrame({
                'date': dates,
                'price': prices,
                'returns': np.concatenate([[0], returns[1:]])
            }).set_index('date')
    
    async def _calculate_correlation_matrix(self, assets: List[str]) -> pd.DataFrame:
        """Calculate asset correlation matrix."""
        if len(assets) < 2:
            return pd.DataFrame()
        
        returns_data = {}
        for asset in assets:
            if asset in self.price_data and not self.price_data[asset].empty:
                returns_data[asset] = self.price_data[asset]['returns']
        
        if len(returns_data) < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()
    
    async def _update_volatility_estimates(self, assets: List[str]):
        """Update volatility estimates for assets."""
        for asset in assets:
            if asset in self.price_data and not self.price_data[asset].empty:
                returns = self.price_data[asset]['returns']
                
                # Calculate EWMA volatility with more weight on recent observations
                ewma_vol = returns.ewm(span=30).std().iloc[-1]
                self.volatility_data[asset] = Decimal(str(ewma_vol))
    
    async def _analyze_tax_loss_harvesting(self, current_date: datetime) -> List[OptimizationRecommendation]:
        """Analyze tax loss harvesting opportunities."""
        logger.debug("Analyzing tax loss harvesting opportunities...")
        
        recommendations = []
        portfolio = self.tax_engine.get_portfolio_summary()
        
        for asset, holding in portfolio['holdings'].items():
            if holding['quantity'] <= 0:
                continue
            
            # Get current market price (placeholder)
            current_price = await self._get_current_price(asset)
            if current_price is None:
                continue
            
            current_value = holding['quantity'] * current_price
            unrealized_loss = current_value - holding['cost_basis']
            
            if unrealized_loss < 0:  # We have an unrealized loss
                opportunity = await self._evaluate_harvesting_opportunity(
                    asset, holding, current_price, unrealized_loss, current_date
                )
                
                if opportunity and opportunity.potential_tax_savings > 0:
                    recommendation = OptimizationRecommendation(
                        id=f"tlh_{asset}_{current_date.strftime('%Y%m%d')}",
                        timestamp=current_date,
                        strategy=OptimizationStrategy.TAX_LOSS_HARVEST,
                        asset=asset,
                        action="SELL",
                        quantity=opportunity.recommended_quantity,
                        expected_tax_savings=opportunity.potential_tax_savings,
                        risk_score=opportunity.wash_sale_risk_score,
                        confidence=self._calculate_recommendation_confidence(opportunity),
                        rationale=f"Harvest ${abs(unrealized_loss):,.2f} loss for tax savings",
                        execution_timeline="WITHIN_7_DAYS" if opportunity.wash_sale_risk_score < 0.3 else "IMMEDIATE",
                        wash_sale_risk=opportunity.wash_sale_risk_score > 0.5,
                        alternative_assets=opportunity.alternative_investments,
                        portfolio_impact=self._assess_portfolio_impact(asset, opportunity.recommended_quantity),
                        tax_impact=self._assess_tax_impact(opportunity)
                    )
                    
                    recommendations.append(recommendation)
        
        return recommendations
    
    async def _evaluate_harvesting_opportunity(self, asset: str, holding: Dict[str, Any], 
                                             current_price: Decimal, unrealized_loss: Decimal,
                                             current_date: datetime) -> Optional[HarvestingOpportunity]:
        """Evaluate a specific tax loss harvesting opportunity."""
        
        # Calculate potential tax savings
        effective_tax_rate = self._calculate_effective_tax_rate(asset)
        potential_savings = abs(unrealized_loss) * effective_tax_rate
        
        # Assess wash sale risk
        wash_sale_risk = await self._assess_wash_sale_risk(asset, current_date)
        
        # Assess liquidity and correlation risks
        liquidity_risk = await self._assess_liquidity_risk(asset)
        correlation_risk = await self._assess_correlation_risk(asset)
        
        # Find alternative investments
        alternatives = await self._find_alternative_investments(asset)
        
        # Calculate recommended quantity (may be partial position)
        recommended_quantity = self._calculate_optimal_harvest_quantity(
            holding, unrealized_loss, wash_sale_risk
        )
        
        if potential_savings < Decimal('10'):  # Minimum threshold
            return None
        
        return HarvestingOpportunity(
            asset=asset,
            current_position=holding['quantity'],
            unrealized_loss=unrealized_loss,
            cost_basis=holding['cost_basis'],
            current_value=holding['quantity'] * current_price,
            days_held=self._estimate_avg_holding_days(asset),
            potential_tax_savings=potential_savings,
            effective_tax_rate=effective_tax_rate,
            holding_period_impact="SHORT_TERM" if self._estimate_avg_holding_days(asset) <= 365 else "LONG_TERM",
            wash_sale_risk_score=wash_sale_risk,
            liquidity_risk=liquidity_risk,
            correlation_risk=correlation_risk,
            recommended_quantity=recommended_quantity,
            alternative_investments=alternatives,
            execution_priority=self._calculate_execution_priority(potential_savings, wash_sale_risk)
        )
    
    async def _analyze_long_term_optimization(self, current_date: datetime) -> List[OptimizationRecommendation]:
        """Analyze opportunities to optimize for long-term capital gains treatment."""
        logger.debug("Analyzing long-term optimization opportunities...")
        
        recommendations = []
        
        # Find positions approaching long-term status
        for asset_lots in self.tax_engine.tax_lots.values():
            for lot in asset_lots:
                days_held = (current_date - lot.acquisition_date).days
                days_to_long_term = 365 - days_held
                
                if 0 < days_to_long_term <= 90:  # Within 90 days of long-term
                    # Check if holding for long-term treatment would be beneficial
                    current_price = await self._get_current_price(lot.asset)
                    if current_price is None:
                        continue
                    
                    current_value = lot.quantity * current_price
                    unrealized_gain = current_value - lot.adjusted_total_cost_basis
                    
                    if unrealized_gain > 0:  # Only for gains
                        # Calculate tax savings from waiting
                        short_term_tax = unrealized_gain * self.config.current_tax_bracket
                        long_term_tax = unrealized_gain * self._get_long_term_rate()
                        potential_savings = short_term_tax - long_term_tax
                        
                        if potential_savings > Decimal('50'):  # Minimum threshold
                            recommendation = OptimizationRecommendation(
                                id=f"lto_{lot.asset}_{lot.transaction_id}",
                                timestamp=current_date,
                                strategy=OptimizationStrategy.LONG_TERM_OPTIMIZATION,
                                asset=lot.asset,
                                action="HOLD",
                                quantity=lot.quantity,
                                expected_tax_savings=potential_savings,
                                risk_score=self._assess_holding_risk(lot.asset, days_to_long_term),
                                confidence=Decimal('0.8'),
                                rationale=f"Hold {days_to_long_term} more days for long-term treatment",
                                execution_timeline=f"HOLD_UNTIL_{(current_date + timedelta(days=days_to_long_term)).strftime('%Y-%m-%d')}",
                                wash_sale_risk=False
                            )
                            
                            recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_wash_sale_avoidance(self, current_date: datetime) -> List[OptimizationRecommendation]:
        """Analyze positions at risk of wash sale violations."""
        logger.debug("Analyzing wash sale avoidance opportunities...")
        
        recommendations = []
        
        # Check recent transactions for wash sale risks
        recent_sales = [
            tx for tx in self.tax_engine.transactions
            if (tx.transaction_type == TransactionType.SELL and
                (current_date - tx.timestamp).days <= 30)
        ]
        
        for sale_tx in recent_sales:
            # Find corresponding tax event
            tax_event = next(
                (event for event in self.tax_engine.tax_events 
                 if event.transaction_id == sale_tx.id and event.gain_loss < 0),
                None
            )
            
            if tax_event and not tax_event.wash_sale_affected:
                # Check if there are any plans to repurchase
                wash_sale_window_end = sale_tx.timestamp + timedelta(days=30)
                days_remaining = (wash_sale_window_end - current_date).days
                
                if days_remaining > 0:
                    recommendation = OptimizationRecommendation(
                        id=f"wsa_{sale_tx.asset}_{sale_tx.id}",
                        timestamp=current_date,
                        strategy=OptimizationStrategy.WASH_SALE_AVOIDANCE,
                        asset=sale_tx.asset,
                        action="AVOID_PURCHASE",
                        quantity=sale_tx.quantity,
                        expected_tax_savings=abs(tax_event.gain_loss) * self.config.current_tax_bracket,
                        risk_score=Decimal('0.2'),
                        confidence=Decimal('0.9'),
                        rationale=f"Avoid purchasing {sale_tx.asset} for {days_remaining} more days",
                        execution_timeline=f"AVOID_UNTIL_{wash_sale_window_end.strftime('%Y-%m-%d')}",
                        wash_sale_risk=True
                    )
                    
                    recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_rebalancing_optimization(self, current_date: datetime) -> List[OptimizationRecommendation]:
        """Analyze tax-efficient rebalancing opportunities."""
        logger.debug("Analyzing rebalancing optimization opportunities...")
        
        if not self.config.target_allocations:
            return []
        
        recommendations = []
        portfolio = self.tax_engine.get_portfolio_summary()
        
        # Calculate current allocations
        total_value = Decimal('0')
        current_values = {}
        
        for asset, holding in portfolio['holdings'].items():
            current_price = await self._get_current_price(asset)
            if current_price is not None:
                current_value = holding['quantity'] * current_price
                current_values[asset] = current_value
                total_value += current_value
        
        if total_value == 0:
            return recommendations
        
        # Compare with target allocations
        for asset, target_pct in self.config.target_allocations.items():
            current_value = current_values.get(asset, Decimal('0'))
            current_pct = current_value / total_value
            deviation = abs(current_pct - target_pct)
            
            if deviation > self.config.rebalancing_threshold:
                target_value = total_value * target_pct
                rebalance_amount = target_value - current_value
                
                # Determine action and tax implications
                if rebalance_amount > 0:  # Need to buy more
                    action = "BUY"
                    quantity = rebalance_amount / (await self._get_current_price(asset) or Decimal('1'))
                else:  # Need to sell some
                    action = "SELL"
                    quantity = abs(rebalance_amount) / (await self._get_current_price(asset) or Decimal('1'))
                
                # Calculate tax implications of the trade
                tax_impact = await self._calculate_rebalancing_tax_impact(asset, rebalance_amount)
                
                recommendation = OptimizationRecommendation(
                    id=f"rebal_{asset}_{current_date.strftime('%Y%m%d')}",
                    timestamp=current_date,
                    strategy=OptimizationStrategy.REBALANCING_OPTIMIZATION,
                    asset=asset,
                    action=action,
                    quantity=abs(quantity),
                    expected_tax_savings=tax_impact.get('net_tax_impact', Decimal('0')),
                    risk_score=Decimal('0.3'),
                    confidence=Decimal('0.7'),
                    rationale=f"Rebalance to target allocation: {deviation:.1%} deviation",
                    execution_timeline="WITHIN_30_DAYS",
                    wash_sale_risk=False,
                    portfolio_impact={'rebalancing_benefit': True},
                    tax_impact=tax_impact
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _analyze_multi_asset_harvesting(self, current_date: datetime) -> List[OptimizationRecommendation]:
        """Analyze multi-asset tax loss harvesting with correlation considerations."""
        logger.debug("Analyzing multi-asset harvesting opportunities...")
        
        recommendations = []
        
        # Get all harvesting opportunities
        individual_opportunities = await self._analyze_tax_loss_harvesting(current_date)
        
        if len(individual_opportunities) < 2:
            return individual_opportunities
        
        # Group by correlation clusters
        correlation_clusters = await self._group_assets_by_correlation()
        
        # Optimize across clusters to maximize tax savings while managing correlation risk
        for cluster_assets in correlation_clusters:
            cluster_recommendations = [
                rec for rec in individual_opportunities 
                if rec.asset in cluster_assets
            ]
            
            if len(cluster_recommendations) > 1:
                # Optimize harvesting within this cluster
                optimized_recs = await self._optimize_cluster_harvesting(
                    cluster_recommendations, current_date
                )
                recommendations.extend(optimized_recs)
            else:
                recommendations.extend(cluster_recommendations)
        
        return recommendations
    
    async def _analyze_comprehensive_optimization(self, current_date: datetime) -> List[OptimizationRecommendation]:
        """Perform comprehensive tax optimization analysis."""
        logger.debug("Performing comprehensive optimization analysis...")
        
        # Combine all optimization strategies
        all_recommendations = []
        
        # Get recommendations from each strategy
        strategies = [
            self._analyze_tax_loss_harvesting,
            self._analyze_long_term_optimization,
            self._analyze_wash_sale_avoidance,
            self._analyze_rebalancing_optimization
        ]
        
        if self.config.parallel_processing:
            # Run strategies in parallel
            tasks = [strategy(current_date) for strategy in strategies]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_recommendations.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Strategy failed: {result}")
        else:
            # Run strategies sequentially
            for strategy in strategies:
                try:
                    recommendations = await strategy(current_date)
                    all_recommendations.extend(recommendations)
                except Exception as e:
                    logger.error(f"Strategy failed: {e}")
        
        # Remove conflicts and optimize combined recommendations
        optimized_recommendations = await self._resolve_recommendation_conflicts(
            all_recommendations, current_date
        )
        
        return optimized_recommendations
    
    def _filter_and_rank_recommendations(self, recommendations: List[OptimizationRecommendation],
                                       current_date: datetime) -> List[OptimizationRecommendation]:
        """Filter and rank recommendations based on configuration and constraints."""
        
        # Filter based on risk tolerance
        filtered_recommendations = []
        
        for rec in recommendations:
            # Apply risk tolerance filter
            if self.config.risk_tolerance == RiskTolerance.CONSERVATIVE:
                if rec.risk_score <= Decimal('0.3') and rec.confidence >= Decimal('0.8'):
                    filtered_recommendations.append(rec)
            elif self.config.risk_tolerance == RiskTolerance.MODERATE:
                if rec.risk_score <= Decimal('0.6') and rec.confidence >= Decimal('0.6'):
                    filtered_recommendations.append(rec)
            else:  # AGGRESSIVE
                if rec.confidence >= Decimal('0.4'):
                    filtered_recommendations.append(rec)
        
        # Apply additional filters
        final_recommendations = []
        
        for rec in filtered_recommendations:
            # Minimum tax savings threshold
            if rec.expected_tax_savings >= Decimal('10'):
                # Check wash sale constraints
                if rec.wash_sale_risk and self.config.max_wash_sale_risk < Decimal('0.5'):
                    if rec.risk_score <= self.config.max_wash_sale_risk:
                        final_recommendations.append(rec)
                else:
                    final_recommendations.append(rec)
        
        # Rank by expected tax savings and confidence
        final_recommendations.sort(
            key=lambda x: (x.expected_tax_savings * x.confidence, -x.risk_score),
            reverse=True
        )
        
        return final_recommendations
    
    async def execute_recommendation(self, recommendation: OptimizationRecommendation,
                                   dry_run: bool = True) -> Dict[str, Any]:
        """Execute a tax optimization recommendation."""
        logger.info(f"Executing recommendation: {recommendation.id} ({recommendation.action})")
        
        execution_result = {
            'recommendation_id': recommendation.id,
            'executed': False,
            'dry_run': dry_run,
            'execution_date': datetime.now(),
            'errors': [],
            'warnings': []
        }
        
        try:
            if recommendation.action == "SELL":
                result = await self._execute_sell_recommendation(recommendation, dry_run)
            elif recommendation.action == "BUY":
                result = await self._execute_buy_recommendation(recommendation, dry_run)
            elif recommendation.action == "HOLD":
                result = await self._execute_hold_recommendation(recommendation, dry_run)
            elif recommendation.action == "AVOID_PURCHASE":
                result = await self._execute_avoid_recommendation(recommendation, dry_run)
            else:
                raise ValueError(f"Unknown action: {recommendation.action}")
            
            execution_result.update(result)
            
            if not dry_run and result.get('executed', False):
                # Update tracking
                self.realized_tax_savings += recommendation.expected_tax_savings
                
                # Log execution
                logger.info(f"Successfully executed recommendation {recommendation.id}")
            
        except Exception as e:
            logger.error(f"Failed to execute recommendation {recommendation.id}: {e}")
            execution_result['errors'].append(str(e))
        
        return execution_result
    
    # Helper methods
    async def _get_current_price(self, asset: str) -> Optional[Decimal]:
        """Get current market price for an asset."""
        # Placeholder for real price feed integration
        if asset in self.price_data and not self.price_data[asset].empty:
            latest_price = self.price_data[asset]['price'].iloc[-1]
            return Decimal(str(latest_price))
        return None
    
    def _calculate_effective_tax_rate(self, asset: str) -> Decimal:
        """Calculate effective tax rate for an asset."""
        # Consider federal, state, and NIIT
        federal_rate = self.config.current_tax_bracket
        state_rate = self.config.state_tax_rate
        niit_rate = Decimal('0.038')  # Net Investment Income Tax
        
        # Simplified calculation
        return federal_rate + state_rate + niit_rate
    
    async def _assess_wash_sale_risk(self, asset: str, current_date: datetime) -> Decimal:
        """Assess wash sale risk for an asset."""
        # Check for recent purchases within 30 days
        lookback_start = current_date - timedelta(days=30)
        
        recent_purchases = [
            tx for tx in self.tax_engine.transactions
            if (tx.asset == asset and 
                tx.transaction_type == TransactionType.BUY and
                lookback_start <= tx.timestamp <= current_date)
        ]
        
        # Simple risk scoring based on recent activity
        if len(recent_purchases) >= 3:
            return Decimal('0.8')  # High risk
        elif len(recent_purchases) >= 2:
            return Decimal('0.6')  # Medium risk
        elif len(recent_purchases) >= 1:
            return Decimal('0.4')  # Low-medium risk
        else:
            return Decimal('0.1')  # Low risk
    
    async def _assess_liquidity_risk(self, asset: str) -> Decimal:
        """Assess liquidity risk for an asset."""
        # Placeholder - would use order book data, volumes, spreads
        if asset in ['BTC', 'ETH']:
            return Decimal('0.1')  # Low liquidity risk for major assets
        else:
            return Decimal('0.3')  # Higher risk for smaller assets
    
    async def _assess_correlation_risk(self, asset: str) -> Decimal:
        """Assess correlation risk for an asset."""
        if self.correlation_matrix is None or asset not in self.correlation_matrix.columns:
            return Decimal('0.2')  # Default moderate risk
        
        # Calculate average correlation with other holdings
        correlations = self.correlation_matrix[asset].drop(asset)
        if correlations.empty:
            return Decimal('0.1')
        
        avg_correlation = abs(correlations.mean())
        return Decimal(str(min(avg_correlation, 1.0)))
    
    async def _find_alternative_investments(self, asset: str) -> List[str]:
        """Find alternative investments for tax-loss harvesting."""
        alternatives = []
        
        if self.correlation_matrix is not None and asset in self.correlation_matrix.columns:
            # Find assets with moderate correlation (0.3-0.7) - similar but not substantially identical
            correlations = self.correlation_matrix[asset].drop(asset)
            suitable_alternatives = correlations[
                (correlations.abs() >= 0.3) & (correlations.abs() <= 0.7)
            ]
            
            alternatives = suitable_alternatives.index.tolist()[:3]  # Top 3 alternatives
        
        # Add some default alternatives based on asset category
        if asset in ['BTC', 'ETH']:
            alternatives.extend(['CRYPTO_INDEX', 'BLOCKCHAIN_ETF'])
        
        return alternatives[:5]  # Maximum 5 alternatives
    
    def _calculate_optimal_harvest_quantity(self, holding: Dict[str, Any], 
                                          unrealized_loss: Decimal, 
                                          wash_sale_risk: Decimal) -> Decimal:
        """Calculate optimal quantity to harvest."""
        total_quantity = holding['quantity']
        
        # Adjust based on risk
        if wash_sale_risk > Decimal('0.7'):
            # High wash sale risk - harvest smaller portion
            return total_quantity * Decimal('0.5')
        elif wash_sale_risk > Decimal('0.4'):
            # Medium risk - harvest moderate portion
            return total_quantity * Decimal('0.75')
        else:
            # Low risk - can harvest full position
            return total_quantity
    
    def _estimate_avg_holding_days(self, asset: str) -> int:
        """Estimate average holding days for an asset."""
        # Simplified calculation - would use actual lot data
        asset_lots = self.tax_engine.tax_lots.get(asset, [])
        if not asset_lots:
            return 180  # Default estimate
        
        current_date = datetime.now()
        total_days = sum(
            (current_date - lot.acquisition_date).days * float(lot.quantity)
            for lot in asset_lots
        )
        total_quantity = sum(float(lot.quantity) for lot in asset_lots)
        
        return int(total_days / total_quantity) if total_quantity > 0 else 180
    
    def _calculate_execution_priority(self, potential_savings: Decimal, 
                                    wash_sale_risk: Decimal) -> int:
        """Calculate execution priority (1 = highest, 10 = lowest)."""
        # Higher savings and lower risk = higher priority
        priority_score = float(potential_savings) / 100 - float(wash_sale_risk) * 5
        
        if priority_score >= 10:
            return 1
        elif priority_score >= 5:
            return 2
        elif priority_score >= 2:
            return 3
        elif priority_score >= 1:
            return 4
        else:
            return 5 + min(int(abs(priority_score)), 5)
    
    def _get_long_term_rate(self) -> Decimal:
        """Get long-term capital gains tax rate."""
        # Simplified - would use actual brackets
        return Decimal('0.15')  # Typical long-term rate
    
    def _assess_holding_risk(self, asset: str, days_to_hold: int) -> Decimal:
        """Assess risk of holding an asset for specified days."""
        # Use volatility to estimate holding risk
        volatility = self.volatility_data.get(asset, Decimal('0.02'))
        
        # Risk increases with volatility and holding period
        risk_factor = volatility * Decimal(str(days_to_hold / 365))
        
        return min(risk_factor, Decimal('1.0'))
    
    def _calculate_recommendation_confidence(self, opportunity: HarvestingOpportunity) -> Decimal:
        """Calculate confidence score for a recommendation."""
        # Base confidence on multiple factors
        base_confidence = Decimal('0.8')
        
        # Adjust for risks
        confidence = base_confidence
        confidence -= opportunity.wash_sale_risk_score * Decimal('0.3')
        confidence -= opportunity.liquidity_risk * Decimal('0.2')
        confidence -= opportunity.correlation_risk * Decimal('0.1')
        
        # Boost for larger tax savings
        if opportunity.potential_tax_savings > Decimal('1000'):
            confidence += Decimal('0.1')
        
        return max(min(confidence, Decimal('1.0')), Decimal('0.0'))
    
    def _assess_portfolio_impact(self, asset: str, quantity: Decimal) -> Dict[str, Any]:
        """Assess impact of a transaction on portfolio."""
        portfolio = self.tax_engine.get_portfolio_summary()
        
        if asset not in portfolio['holdings']:
            return {'impact': 'minimal'}
        
        current_quantity = portfolio['holdings'][asset]['quantity']
        impact_pct = float(quantity / current_quantity) if current_quantity > 0 else 0
        
        return {
            'impact_percentage': impact_pct,
            'impact_level': 'high' if impact_pct > 0.5 else 'medium' if impact_pct > 0.2 else 'low',
            'remaining_quantity': current_quantity - quantity
        }
    
    def _assess_tax_impact(self, opportunity: HarvestingOpportunity) -> Dict[str, Any]:
        """Assess tax impact of a harvesting opportunity."""
        return {
            'tax_savings': opportunity.potential_tax_savings,
            'effective_rate': opportunity.effective_tax_rate,
            'holding_period_benefit': opportunity.holding_period_impact,
            'wash_sale_risk': opportunity.wash_sale_risk_score > Decimal('0.5')
        }
    
    async def _calculate_rebalancing_tax_impact(self, asset: str, rebalance_amount: Decimal) -> Dict[str, Any]:
        """Calculate tax impact of rebalancing."""
        # Placeholder for detailed rebalancing tax calculation
        return {
            'net_tax_impact': Decimal('0'),  # Would calculate actual impact
            'short_term_impact': Decimal('0'),
            'long_term_impact': Decimal('0')
        }
    
    async def _group_assets_by_correlation(self) -> List[List[str]]:
        """Group assets by correlation for multi-asset optimization."""
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return []
        
        # Simple clustering based on correlation threshold
        assets = self.correlation_matrix.columns.tolist()
        clusters = []
        used_assets = set()
        
        for asset in assets:
            if asset in used_assets:
                continue
            
            # Find highly correlated assets
            correlations = self.correlation_matrix[asset]
            cluster = [asset]
            
            for other_asset in assets:
                if (other_asset != asset and 
                    other_asset not in used_assets and
                    abs(correlations[other_asset]) > 0.7):  # High correlation threshold
                    cluster.append(other_asset)
                    used_assets.add(other_asset)
            
            clusters.append(cluster)
            used_assets.add(asset)
        
        return clusters
    
    async def _optimize_cluster_harvesting(self, cluster_recommendations: List[OptimizationRecommendation],
                                         current_date: datetime) -> List[OptimizationRecommendation]:
        """Optimize tax harvesting within a correlation cluster."""
        # For highly correlated assets, avoid harvesting all at once
        # Select the most tax-efficient combination
        
        if len(cluster_recommendations) <= 1:
            return cluster_recommendations
        
        # Sort by tax savings per unit of risk
        cluster_recommendations.sort(
            key=lambda x: float(x.expected_tax_savings / (x.risk_score + Decimal('0.01'))),
            reverse=True
        )
        
        # Select up to 2 assets from highly correlated cluster to avoid overconcentration
        return cluster_recommendations[:2]
    
    async def _resolve_recommendation_conflicts(self, recommendations: List[OptimizationRecommendation],
                                              current_date: datetime) -> List[OptimizationRecommendation]:
        """Resolve conflicts between different optimization recommendations."""
        
        # Group by asset
        asset_recommendations = {}
        for rec in recommendations:
            if rec.asset not in asset_recommendations:
                asset_recommendations[rec.asset] = []
            asset_recommendations[rec.asset].append(rec)
        
        resolved_recommendations = []
        
        for asset, asset_recs in asset_recommendations.items():
            if len(asset_recs) == 1:
                resolved_recommendations.extend(asset_recs)
            else:
                # Resolve conflicts for this asset
                # Priority: Tax loss harvesting > Long-term optimization > Rebalancing
                priority_order = [
                    OptimizationStrategy.TAX_LOSS_HARVEST,
                    OptimizationStrategy.LONG_TERM_OPTIMIZATION,
                    OptimizationStrategy.WASH_SALE_AVOIDANCE,
                    OptimizationStrategy.REBALANCING_OPTIMIZATION
                ]
                
                for strategy in priority_order:
                    strategy_recs = [rec for rec in asset_recs if rec.strategy == strategy]
                    if strategy_recs:
                        # Take the best recommendation for this strategy
                        best_rec = max(strategy_recs, 
                                     key=lambda x: x.expected_tax_savings * x.confidence)
                        resolved_recommendations.append(best_rec)
                        break
        
        return resolved_recommendations
    
    # Execution methods (placeholders for actual trading integration)
    async def _execute_sell_recommendation(self, recommendation: OptimizationRecommendation, 
                                         dry_run: bool) -> Dict[str, Any]:
        """Execute a sell recommendation."""
        if dry_run:
            return {'executed': True, 'dry_run': True, 'action': 'SELL', 'details': 'Dry run execution'}
        
        # Would integrate with actual trading system
        logger.info(f"Would sell {recommendation.quantity} {recommendation.asset}")
        return {'executed': False, 'reason': 'Trading integration not implemented'}
    
    async def _execute_buy_recommendation(self, recommendation: OptimizationRecommendation, 
                                        dry_run: bool) -> Dict[str, Any]:
        """Execute a buy recommendation."""
        if dry_run:
            return {'executed': True, 'dry_run': True, 'action': 'BUY', 'details': 'Dry run execution'}
        
        # Would integrate with actual trading system
        logger.info(f"Would buy {recommendation.quantity} {recommendation.asset}")
        return {'executed': False, 'reason': 'Trading integration not implemented'}
    
    async def _execute_hold_recommendation(self, recommendation: OptimizationRecommendation, 
                                         dry_run: bool) -> Dict[str, Any]:
        """Execute a hold recommendation."""
        # Hold recommendations are informational
        return {'executed': True, 'action': 'HOLD', 'details': 'Position held as recommended'}
    
    async def _execute_avoid_recommendation(self, recommendation: OptimizationRecommendation, 
                                          dry_run: bool) -> Dict[str, Any]:
        """Execute an avoid purchase recommendation."""
        # Avoidance recommendations are informational
        return {'executed': True, 'action': 'AVOID', 'details': 'Purchase avoided as recommended'}
    
    def _update_optimization_history(self, recommendations: List[OptimizationRecommendation], 
                                   current_date: datetime):
        """Update optimization history for tracking and analysis."""
        history_entry = {
            'date': current_date,
            'recommendations_count': len(recommendations),
            'total_expected_savings': sum(rec.expected_tax_savings for rec in recommendations),
            'strategies_used': list(set(rec.strategy.value for rec in recommendations)),
            'avg_confidence': (sum(rec.confidence for rec in recommendations) / len(recommendations)
                             if recommendations else Decimal('0'))
        }
        
        self.optimization_history.append(history_entry)
        
        # Keep only last 365 days of history
        cutoff_date = current_date - timedelta(days=365)
        self.optimization_history = [
            entry for entry in self.optimization_history 
            if entry['date'] >= cutoff_date
        ]
    
    def get_optimization_performance(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        if not self.optimization_history:
            return {'performance': 'No optimization history available'}
        
        total_opportunities = sum(entry['recommendations_count'] for entry in self.optimization_history)
        total_expected_savings = sum(entry['total_expected_savings'] for entry in self.optimization_history)
        avg_confidence = (sum(entry['avg_confidence'] for entry in self.optimization_history) / 
                         len(self.optimization_history))
        
        return {
            'total_opportunities_identified': total_opportunities,
            'total_expected_tax_savings': total_expected_savings,
            'realized_tax_savings': self.realized_tax_savings,
            'avg_recommendation_confidence': avg_confidence,
            'optimization_periods': len(self.optimization_history),
            'savings_realization_rate': (float(self.realized_tax_savings / total_expected_savings) 
                                       if total_expected_savings > 0 else 0.0)
        }

# Example usage
if __name__ == "__main__":
    # Example usage of the tax optimizer
    from .tax_engine import TaxEngine, TaxConfiguration
    
    # Create tax engine
    tax_config = TaxConfiguration("US")
    tax_engine = TaxEngine(tax_config)
    
    # Create optimizer
    opt_config = TaxOptimizationConfig(
        strategy=OptimizationStrategy.TAX_LOSS_HARVEST,
        objective=OptimizationObjective.MAXIMIZE_TAX_SAVINGS,
        risk_tolerance=RiskTolerance.MODERATE
    )
    
    optimizer = TaxOptimizer(tax_engine, opt_config)
    
    # This would run optimization analysis
    print("Tax optimizer initialized successfully")