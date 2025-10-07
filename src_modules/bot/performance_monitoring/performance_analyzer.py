"""
Performance Analyzer
Advanced performance analysis with Australian-specific metrics,
benchmarking against ASX indices, and tax-adjusted returns
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import numpy as np
from collections import defaultdict, deque

# Performance analysis libraries
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# Import system components
from ..australian_compliance.ato_integration import AustralianTaxCalculator
from ..ml_strategy_discovery.ml_engine import StrategySignal, StrategyType
from ..arbitrage_engine.arbitrage_detector import ArbitrageOpportunity
from ..risk_management.portfolio_risk_controller import PortfolioRiskController

logger = logging.getLogger(__name__)

class PerformancePeriod(Enum):
    """Performance analysis periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION = "inception"

class BenchmarkType(Enum):
    """Benchmark comparison types"""
    ASX_200 = "asx_200"
    ASX_300 = "asx_300"
    AUD_CASH_RATE = "aud_cash_rate"
    BITCOIN_USD = "bitcoin_usd"
    BITCOIN_AUD = "bitcoin_aud"
    ETHEREUM_AUD = "ethereum_aud"
    CRYPTO_INDEX = "crypto_index"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Basic returns
    total_return_aud: Decimal
    annualized_return: Decimal
    volatility: Decimal
    
    # Risk-adjusted metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    
    # Drawdown metrics
    max_drawdown: Decimal
    max_drawdown_duration_days: int
    current_drawdown: Decimal
    
    # Win/loss metrics
    win_rate: Decimal
    profit_factor: Decimal
    avg_win_aud: Decimal
    avg_loss_aud: Decimal
    
    # Australian-specific metrics
    tax_adjusted_return: Decimal
    cgt_discount_benefit: Decimal
    alpha_vs_asx200: Decimal
    
    # Strategy performance
    ml_strategy_return: Decimal
    arbitrage_return: Decimal
    strategy_correlation: Decimal
    
    # Benchmark comparisons
    benchmark_outperformance: Dict[BenchmarkType, Decimal] = field(default_factory=dict)

@dataclass
class TradeAnalysis:
    """Individual trade analysis"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    holding_period_days: int
    
    # Financial metrics
    entry_price_aud: Decimal
    exit_price_aud: Optional[Decimal]
    size_aud: Decimal
    pnl_aud: Decimal
    pnl_percentage: Decimal
    
    # Tax analysis
    gross_pnl_aud: Decimal
    tax_liability_aud: Decimal
    net_pnl_aud: Decimal
    cgt_discount_applied: bool
    
    # Strategy attribution
    strategy_type: str
    signal_strength: Optional[float]
    confidence: Optional[float]
    
    # Execution analysis
    slippage_bps: int
    execution_time_ms: int
    fees_aud: Decimal

class AustralianBenchmarkProvider:
    """
    Provides Australian and international benchmark data
    Simulates ASX indices and AUD-denominated crypto benchmarks
    """
    
    def __init__(self):
        # Benchmark configurations
        self.benchmarks = {
            BenchmarkType.ASX_200: {
                'name': 'ASX 200 Index',
                'currency': 'AUD',
                'annual_return': Decimal('0.08'),  # 8% historical
                'volatility': Decimal('0.15')      # 15% volatility
            },
            BenchmarkType.AUD_CASH_RATE: {
                'name': 'RBA Cash Rate',
                'currency': 'AUD',
                'annual_return': Decimal('0.04'),  # 4% current rate
                'volatility': Decimal('0.02')      # Low volatility
            },
            BenchmarkType.BITCOIN_AUD: {
                'name': 'Bitcoin (AUD)',
                'currency': 'AUD',
                'annual_return': Decimal('0.25'),  # 25% historical
                'volatility': Decimal('0.80')      # High volatility
            },
            BenchmarkType.ETHEREUM_AUD: {
                'name': 'Ethereum (AUD)',
                'currency': 'AUD',
                'annual_return': Decimal('0.20'),  # 20% historical
                'volatility': Decimal('0.85')      # High volatility
            }
        }
        
        # Generated benchmark data cache
        self.benchmark_data = {}
        self.last_update = None
        
        logger.info("Initialized Australian Benchmark Provider")
    
    async def get_benchmark_returns(
        self,
        benchmark: BenchmarkType,
        period_start: date,
        period_end: date
    ) -> List[Tuple[date, Decimal]]:
        """Get benchmark returns for specified period"""
        
        try:
            if benchmark not in self.benchmarks:
                logger.error(f"Unknown benchmark: {benchmark}")
                return []
            
            config = self.benchmarks[benchmark]
            
            # Generate synthetic benchmark data (in practice, would fetch real data)
            returns = []
            current_date = period_start
            
            while current_date <= period_end:
                # Generate daily return using random walk
                daily_return = np.random.normal(
                    float(config['annual_return']) / 252,  # Daily mean
                    float(config['volatility']) / np.sqrt(252)  # Daily std
                )
                
                returns.append((current_date, Decimal(str(daily_return))))
                current_date += timedelta(days=1)
            
            return returns
            
        except Exception as e:
            logger.error(f"Benchmark data retrieval failed for {benchmark}: {e}")
            return []
    
    def get_benchmark_info(self, benchmark: BenchmarkType) -> Dict[str, Any]:
        """Get benchmark information"""
        
        if benchmark not in self.benchmarks:
            return {}
        
        config = self.benchmarks[benchmark]
        return {
            'name': config['name'],
            'currency': config['currency'],
            'expected_annual_return': float(config['annual_return']),
            'expected_volatility': float(config['volatility'])
        }

class StrategyPerformanceAnalyzer:
    """
    Analyzes performance of ML strategies and arbitrage opportunities
    Provides strategy attribution and optimization insights
    """
    
    def __init__(self):
        self.strategy_performance = defaultdict(list)
        self.trade_history = []
        
        logger.info("Initialized Strategy Performance Analyzer")
    
    def add_trade_result(self, trade: TradeAnalysis):
        """Add trade result for analysis"""
        
        self.trade_history.append(trade)
        self.strategy_performance[trade.strategy_type].append(trade)
        
        logger.debug(f"Added trade result: {trade.symbol} {trade.strategy_type} "
                    f"P&L: ${trade.pnl_aud:.2f}")
    
    def analyze_strategy_performance(
        self,
        strategy_type: str,
        period_days: int = 90
    ) -> Dict[str, Any]:
        """Analyze performance of specific strategy type"""
        
        if strategy_type not in self.strategy_performance:
            return {'error': f'No data for strategy type: {strategy_type}'}
        
        trades = self.strategy_performance[strategy_type]
        
        # Filter by period
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_trades = [
            trade for trade in trades
            if trade.entry_time >= cutoff_date
        ]
        
        if not recent_trades:
            return {'error': f'No recent trades for strategy: {strategy_type}'}
        
        # Calculate strategy metrics
        total_trades = len(recent_trades)
        winning_trades = [t for t in recent_trades if t.pnl_aud > 0]
        losing_trades = [t for t in recent_trades if t.pnl_aud < 0]
        
        total_pnl = sum(trade.pnl_aud for trade in recent_trades)
        gross_profits = sum(trade.pnl_aud for trade in winning_trades)
        gross_losses = abs(sum(trade.pnl_aud for trade in losing_trades))
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Average holding period
        avg_holding_period = statistics.mean([
            trade.holding_period_days for trade in recent_trades
        ]) if recent_trades else 0
        
        # Signal analysis
        signal_strengths = [
            trade.signal_strength for trade in recent_trades
            if trade.signal_strength is not None
        ]
        
        avg_signal_strength = statistics.mean(signal_strengths) if signal_strengths else 0
        
        # Tax efficiency
        total_tax_liability = sum(trade.tax_liability_aud for trade in recent_trades)
        tax_efficiency = (total_pnl - total_tax_liability) / total_pnl if total_pnl > 0 else 0
        
        return {
            'strategy_type': strategy_type,
            'period_days': period_days,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl_aud': float(total_pnl),
            'gross_profits_aud': float(gross_profits),
            'gross_losses_aud': float(gross_losses),
            'profit_factor': profit_factor,
            'avg_holding_period_days': avg_holding_period,
            'avg_signal_strength': avg_signal_strength,
            'tax_efficiency': tax_efficiency,
            'avg_trade_size_aud': float(statistics.mean([t.size_aud for t in recent_trades])) if recent_trades else 0
        }
    
    def compare_strategies(self, period_days: int = 90) -> Dict[str, Any]:
        """Compare performance across all strategy types"""
        
        strategy_types = list(self.strategy_performance.keys())
        
        if len(strategy_types) < 2:
            return {'error': 'Need at least 2 strategy types for comparison'}
        
        comparison = {}
        
        for strategy_type in strategy_types:
            analysis = self.analyze_strategy_performance(strategy_type, period_days)
            if 'error' not in analysis:
                comparison[strategy_type] = analysis
        
        # Calculate strategy rankings
        if len(comparison) >= 2:
            comparison['rankings'] = {
                'by_total_pnl': sorted(
                    comparison.keys(),
                    key=lambda s: comparison[s]['total_pnl_aud'],
                    reverse=True
                ),
                'by_win_rate': sorted(
                    comparison.keys(),
                    key=lambda s: comparison[s]['win_rate'],
                    reverse=True
                ),
                'by_profit_factor': sorted(
                    comparison.keys(),
                    key=lambda s: comparison[s]['profit_factor'],
                    reverse=True
                )
            }
        
        return comparison
    
    def get_strategy_correlation_matrix(self, period_days: int = 90) -> Dict[str, Any]:
        """Calculate correlation matrix between strategies"""
        
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Group trades by date and strategy
        daily_returns = defaultdict(lambda: defaultdict(Decimal))
        
        for trade in self.trade_history:
            if trade.entry_time >= cutoff_date and trade.exit_time:
                trade_date = trade.exit_time.date()
                strategy = trade.strategy_type
                daily_returns[trade_date][strategy] += trade.pnl_aud
        
        # Convert to correlation matrix
        strategy_types = list(self.strategy_performance.keys())
        
        if len(strategy_types) < 2:
            return {'error': 'Need at least 2 strategies for correlation'}
        
        # Create returns matrix
        dates = sorted(daily_returns.keys())
        returns_matrix = []
        
        for date in dates:
            day_returns = []
            for strategy in strategy_types:
                day_returns.append(float(daily_returns[date].get(strategy, Decimal('0'))))
            returns_matrix.append(day_returns)
        
        if len(returns_matrix) < 10:  # Need minimum data points
            return {'error': 'Insufficient data for correlation analysis'}
        
        # Calculate correlation matrix using numpy
        correlation_matrix = np.corrcoef(np.array(returns_matrix).T)
        
        # Convert to dictionary format
        correlation_dict = {}
        for i, strategy1 in enumerate(strategy_types):
            correlation_dict[strategy1] = {}
            for j, strategy2 in enumerate(strategy_types):
                correlation_dict[strategy1][strategy2] = float(correlation_matrix[i][j])
        
        return {
            'correlation_matrix': correlation_dict,
            'strategy_types': strategy_types,
            'data_points': len(returns_matrix),
            'period_days': period_days
        }

class AustralianPerformanceAnalyzer:
    """
    Main performance analyzer with Australian-specific calculations
    Provides comprehensive performance analysis, benchmarking, and optimization insights
    """
    
    def __init__(
        self,
        tax_calculator: AustralianTaxCalculator,
        risk_controller: PortfolioRiskController
    ):
        self.tax_calculator = tax_calculator
        self.risk_controller = risk_controller
        
        # Analysis components
        self.benchmark_provider = AustralianBenchmarkProvider()
        self.strategy_analyzer = StrategyPerformanceAnalyzer()
        
        # Performance tracking
        self.portfolio_history = deque(maxlen=1000)  # Daily portfolio values
        self.return_history = deque(maxlen=1000)     # Daily returns
        self.trade_history = []
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_expiry = timedelta(hours=1)
        
        logger.info("Initialized Australian Performance Analyzer")
    
    def add_portfolio_snapshot(
        self,
        date: date,
        portfolio_value_aud: Decimal,
        cash_balance_aud: Decimal,
        positions: Dict[str, Decimal]
    ):
        """Add daily portfolio snapshot"""
        
        snapshot = {
            'date': date,
            'portfolio_value_aud': portfolio_value_aud,
            'cash_balance_aud': cash_balance_aud,
            'positions': positions.copy(),
            'timestamp': datetime.now()
        }
        
        self.portfolio_history.append(snapshot)
        
        # Calculate daily return if we have previous data
        if len(self.portfolio_history) >= 2:
            prev_value = self.portfolio_history[-2]['portfolio_value_aud']
            current_value = portfolio_value_aud
            
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else Decimal('0')
            
            self.return_history.append({
                'date': date,
                'return_pct': daily_return * Decimal('100'),
                'portfolio_value': current_value
            })
    
    def add_trade_result(self, trade: TradeAnalysis):
        """Add completed trade for analysis"""
        
        self.trade_history.append(trade)
        self.strategy_analyzer.add_trade_result(trade)
        
        # Clear relevant cache entries
        self._clear_cache_by_prefix('trade_analysis')
        self._clear_cache_by_prefix('strategy_performance')
    
    async def calculate_comprehensive_performance(
        self,
        period: PerformancePeriod = PerformancePeriod.QUARTERLY
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        cache_key = f"performance_{period.value}"
        
        if self._is_cache_valid(cache_key):
            return self.analysis_cache[cache_key]['data']
        
        try:
            # Determine analysis period
            end_date = date.today()
            
            if period == PerformancePeriod.DAILY:
                start_date = end_date - timedelta(days=1)
            elif period == PerformancePeriod.WEEKLY:
                start_date = end_date - timedelta(weeks=1)
            elif period == PerformancePeriod.MONTHLY:
                start_date = end_date - timedelta(days=30)
            elif period == PerformancePeriod.QUARTERLY:
                start_date = end_date - timedelta(days=90)
            elif period == PerformancePeriod.YEARLY:
                start_date = end_date - timedelta(days=365)
            else:  # INCEPTION
                start_date = self.portfolio_history[0]['date'] if self.portfolio_history else end_date
            
            # Get returns for period
            period_returns = [
                r for r in self.return_history
                if start_date <= r['date'] <= end_date
            ]
            
            if not period_returns:
                # Return default metrics if no data
                return self._create_default_metrics()
            
            # Calculate basic metrics
            returns_pct = [r['return_pct'] for r in period_returns]
            total_return = self._calculate_total_return(period_returns)
            annualized_return = self._annualize_return(total_return, len(returns_pct))
            volatility = self._calculate_volatility(returns_pct)
            
            # Risk-adjusted metrics
            sharpe_ratio = await self._calculate_sharpe_ratio(returns_pct)
            sortino_ratio = self._calculate_sortino_ratio(returns_pct)
            calmar_ratio = self._calculate_calmar_ratio(annualized_return, returns_pct)
            
            # Drawdown analysis
            max_drawdown, max_dd_duration, current_dd = self._calculate_drawdown_metrics(period_returns)
            
            # Trade analysis
            period_trades = [
                t for t in self.trade_history
                if start_date <= t.entry_time.date() <= end_date
            ]
            
            win_rate, profit_factor, avg_win, avg_loss = self._calculate_trade_metrics(period_trades)
            
            # Australian-specific metrics
            tax_adjusted_return = await self._calculate_tax_adjusted_return(period_trades)
            cgt_discount_benefit = self._calculate_cgt_discount_benefit(period_trades)
            alpha_vs_asx200 = await self._calculate_alpha_vs_benchmark(
                returns_pct, BenchmarkType.ASX_200, start_date, end_date
            )
            
            # Strategy attribution
            ml_return, arb_return, strategy_corr = self._calculate_strategy_attribution(period_trades)
            
            # Benchmark comparisons
            benchmark_outperformance = await self._calculate_benchmark_outperformance(
                returns_pct, start_date, end_date
            )
            
            # Create comprehensive metrics
            metrics = PerformanceMetrics(
                total_return_aud=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration_days=max_dd_duration,
                current_drawdown=current_dd,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win_aud=avg_win,
                avg_loss_aud=avg_loss,
                tax_adjusted_return=tax_adjusted_return,
                cgt_discount_benefit=cgt_discount_benefit,
                alpha_vs_asx200=alpha_vs_asx200,
                ml_strategy_return=ml_return,
                arbitrage_return=arb_return,
                strategy_correlation=strategy_corr,
                benchmark_outperformance=benchmark_outperformance
            )
            
            # Cache results
            self.analysis_cache[cache_key] = {
                'data': metrics,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Calculated comprehensive performance for {period.value}: "
                       f"Return={annualized_return:.2%}, Sharpe={sharpe_ratio:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return self._create_default_metrics()
    
    def _calculate_total_return(self, returns: List[Dict[str, Any]]) -> Decimal:
        """Calculate total return for period"""
        
        if not returns:
            return Decimal('0')
        
        start_value = returns[0]['portfolio_value']
        end_value = returns[-1]['portfolio_value']
        
        return (end_value - start_value) / start_value if start_value > 0 else Decimal('0')
    
    def _annualize_return(self, total_return: Decimal, periods: int) -> Decimal:
        """Annualize return based on number of periods"""
        
        if periods <= 0:
            return Decimal('0')
        
        # Assume daily periods
        periods_per_year = Decimal('252')  # Trading days
        
        if periods >= periods_per_year:
            return total_return  # Already annualized
        
        # Compound to annual
        return ((Decimal('1') + total_return) ** (periods_per_year / periods)) - Decimal('1')
    
    def _calculate_volatility(self, returns_pct: List[Decimal]) -> Decimal:
        """Calculate annualized volatility"""
        
        if len(returns_pct) < 2:
            return Decimal('0')
        
        returns_float = [float(r) for r in returns_pct]
        daily_std = Decimal(str(statistics.stdev(returns_float)))
        
        # Annualize (252 trading days)
        return daily_std * Decimal('252').sqrt()
    
    async def _calculate_sharpe_ratio(self, returns_pct: List[Decimal]) -> Decimal:
        """Calculate Sharpe ratio using AUD cash rate"""
        
        if len(returns_pct) < 2:
            return Decimal('0')
        
        # Get risk-free rate (AUD cash rate)
        risk_free_rate = Decimal('0.04')  # 4% cash rate
        daily_rf = risk_free_rate / Decimal('252')
        
        # Calculate excess returns
        excess_returns = [r / Decimal('100') - daily_rf for r in returns_pct]
        
        if not excess_returns:
            return Decimal('0')
        
        mean_excess = Decimal(str(statistics.mean([float(r) for r in excess_returns])))
        
        if len(excess_returns) < 2:
            return Decimal('0')
        
        std_excess = Decimal(str(statistics.stdev([float(r) for r in excess_returns])))
        
        if std_excess == 0:
            return Decimal('0')
        
        # Annualized Sharpe ratio
        return (mean_excess * Decimal('252')) / (std_excess * Decimal('252').sqrt())
    
    def _calculate_sortino_ratio(self, returns_pct: List[Decimal]) -> Decimal:
        """Calculate Sortino ratio (downside deviation)"""
        
        if len(returns_pct) < 2:
            return Decimal('0')
        
        returns_float = [float(r) for r in returns_pct]
        
        # Calculate downside returns only
        downside_returns = [r for r in returns_float if r < 0]
        
        if not downside_returns:
            return Decimal('100')  # No downside risk
        
        mean_return = statistics.mean(returns_float)
        downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0
        
        if downside_std == 0:
            return Decimal('100')
        
        # Annualized Sortino ratio
        return Decimal(str(mean_return * 252 / (downside_std * np.sqrt(252))))
    
    def _calculate_calmar_ratio(self, annualized_return: Decimal, returns_pct: List[Decimal]) -> Decimal:
        """Calculate Calmar ratio (return / max drawdown)"""
        
        max_dd, _, _ = self._calculate_drawdown_metrics([
            {'date': date.today(), 'return_pct': r, 'portfolio_value': Decimal('100000')}
            for r in returns_pct
        ])
        
        if max_dd == 0:
            return Decimal('100')  # No drawdown
        
        return annualized_return / (max_dd / Decimal('100'))
    
    def _calculate_drawdown_metrics(
        self,
        returns: List[Dict[str, Any]]
    ) -> Tuple[Decimal, int, Decimal]:
        """Calculate drawdown metrics"""
        
        if not returns:
            return Decimal('0'), 0, Decimal('0')
        
        # Calculate running maximum and drawdowns
        running_max = Decimal('0')
        max_drawdown = Decimal('0')
        current_drawdown = Decimal('0')
        max_dd_duration = 0
        current_dd_duration = 0
        
        for return_data in returns:
            value = return_data['portfolio_value']
            
            if value > running_max:
                running_max = value
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            
            drawdown = (running_max - value) / running_max * Decimal('100') if running_max > 0 else Decimal('0')
            max_drawdown = max(max_drawdown, drawdown)
            
            # Current drawdown is the last calculated drawdown
            current_drawdown = drawdown
        
        return max_drawdown, max_dd_duration, current_drawdown
    
    def _calculate_trade_metrics(
        self,
        trades: List[TradeAnalysis]
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """Calculate trade-based metrics"""
        
        if not trades:
            return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')
        
        winning_trades = [t for t in trades if t.pnl_aud > 0]
        losing_trades = [t for t in trades if t.pnl_aud < 0]
        
        win_rate = Decimal(str(len(winning_trades) / len(trades))) * Decimal('100')
        
        gross_profits = sum(t.pnl_aud for t in winning_trades)
        gross_losses = abs(sum(t.pnl_aud for t in losing_trades))
        
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else Decimal('inf')
        
        avg_win = gross_profits / len(winning_trades) if winning_trades else Decimal('0')
        avg_loss = gross_losses / len(losing_trades) if losing_trades else Decimal('0')
        
        return win_rate, profit_factor, avg_win, avg_loss
    
    async def _calculate_tax_adjusted_return(self, trades: List[TradeAnalysis]) -> Decimal:
        """Calculate tax-adjusted return"""
        
        if not trades:
            return Decimal('0')
        
        gross_pnl = sum(t.gross_pnl_aud for t in trades)
        tax_liability = sum(t.tax_liability_aud for t in trades)
        
        return gross_pnl - tax_liability
    
    def _calculate_cgt_discount_benefit(self, trades: List[TradeAnalysis]) -> Decimal:
        """Calculate benefit from CGT discount"""
        
        if not trades:
            return Decimal('0')
        
        # Calculate what tax would have been without CGT discount
        total_benefit = Decimal('0')
        
        for trade in trades:
            if trade.cgt_discount_applied and trade.gross_pnl_aud > 0:
                # CGT discount saves 50% on tax for gains
                benefit = trade.gross_pnl_aud * Decimal('0.45') * Decimal('0.5')  # 45% tax rate * 50% discount
                total_benefit += benefit
        
        return total_benefit
    
    async def _calculate_alpha_vs_benchmark(
        self,
        returns_pct: List[Decimal],
        benchmark: BenchmarkType,
        start_date: date,
        end_date: date
    ) -> Decimal:
        """Calculate alpha vs benchmark"""
        
        try:
            benchmark_returns = await self.benchmark_provider.get_benchmark_returns(
                benchmark, start_date, end_date
            )
            
            if not benchmark_returns or len(benchmark_returns) != len(returns_pct):
                return Decimal('0')
            
            portfolio_return = statistics.mean([float(r) for r in returns_pct])
            benchmark_return = statistics.mean([float(r[1]) for r in benchmark_returns])
            
            # Alpha = Portfolio Return - Benchmark Return (annualized)
            alpha = (Decimal(str(portfolio_return)) - Decimal(str(benchmark_return))) * Decimal('252')
            
            return alpha
            
        except Exception as e:
            logger.error(f"Alpha calculation failed: {e}")
            return Decimal('0')
    
    def _calculate_strategy_attribution(
        self,
        trades: List[TradeAnalysis]
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate strategy attribution and correlation"""
        
        if not trades:
            return Decimal('0'), Decimal('0'), Decimal('0')
        
        ml_trades = [t for t in trades if 'ml' in t.strategy_type.lower()]
        arb_trades = [t for t in trades if 'arbitrage' in t.strategy_type.lower()]
        
        ml_return = sum(t.pnl_aud for t in ml_trades)
        arb_return = sum(t.pnl_aud for t in arb_trades)
        
        # Calculate strategy correlation (simplified)
        if len(ml_trades) > 0 and len(arb_trades) > 0:
            ml_returns = [float(t.pnl_percentage) for t in ml_trades]
            arb_returns = [float(t.pnl_percentage) for t in arb_trades]
            
            if len(ml_returns) == len(arb_returns):
                correlation = Decimal(str(np.corrcoef(ml_returns, arb_returns)[0, 1]))
            else:
                correlation = Decimal('0')
        else:
            correlation = Decimal('0')
        
        return ml_return, arb_return, correlation
    
    async def _calculate_benchmark_outperformance(
        self,
        returns_pct: List[Decimal],
        start_date: date,
        end_date: date
    ) -> Dict[BenchmarkType, Decimal]:
        """Calculate outperformance vs all benchmarks"""
        
        outperformance = {}
        
        portfolio_return = statistics.mean([float(r) for r in returns_pct]) * 252  # Annualized
        
        for benchmark in BenchmarkType:
            try:
                benchmark_returns = await self.benchmark_provider.get_benchmark_returns(
                    benchmark, start_date, end_date
                )
                
                if benchmark_returns:
                    benchmark_return = statistics.mean([float(r[1]) for r in benchmark_returns]) * 252
                    outperformance[benchmark] = Decimal(str(portfolio_return - benchmark_return))
                
            except Exception as e:
                logger.error(f"Benchmark comparison failed for {benchmark}: {e}")
                outperformance[benchmark] = Decimal('0')
        
        return outperformance
    
    def _create_default_metrics(self) -> PerformanceMetrics:
        """Create default performance metrics when no data available"""
        
        return PerformanceMetrics(
            total_return_aud=Decimal('0'),
            annualized_return=Decimal('0'),
            volatility=Decimal('0'),
            sharpe_ratio=Decimal('0'),
            sortino_ratio=Decimal('0'),
            calmar_ratio=Decimal('0'),
            max_drawdown=Decimal('0'),
            max_drawdown_duration_days=0,
            current_drawdown=Decimal('0'),
            win_rate=Decimal('0'),
            profit_factor=Decimal('0'),
            avg_win_aud=Decimal('0'),
            avg_loss_aud=Decimal('0'),
            tax_adjusted_return=Decimal('0'),
            cgt_discount_benefit=Decimal('0'),
            alpha_vs_asx200=Decimal('0'),
            ml_strategy_return=Decimal('0'),
            arbitrage_return=Decimal('0'),
            strategy_correlation=Decimal('0')
        )
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        
        if cache_key not in self.analysis_cache:
            return False
        
        cache_entry = self.analysis_cache[cache_key]
        return datetime.now() - cache_entry['timestamp'] < self.cache_expiry
    
    def _clear_cache_by_prefix(self, prefix: str):
        """Clear cache entries with specified prefix"""
        
        keys_to_remove = [key for key in self.analysis_cache.keys() if key.startswith(prefix)]
        for key in keys_to_remove:
            del self.analysis_cache[key]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        return {
            'data_points': {
                'portfolio_snapshots': len(self.portfolio_history),
                'daily_returns': len(self.return_history),
                'completed_trades': len(self.trade_history)
            },
            'strategy_performance': self.strategy_analyzer.compare_strategies(),
            'recent_performance': self.return_history[-30:] if len(self.return_history) >= 30 else list(self.return_history),
            'top_trades': sorted(self.trade_history, key=lambda t: t.pnl_aud, reverse=True)[:10],
            'worst_trades': sorted(self.trade_history, key=lambda t: t.pnl_aud)[:5]
        }

# Usage example
async def main():
    """Example usage of Australian Performance Analyzer"""
    
    print("Australian Performance Analyzer Example")
    
    # This would be initialized with actual system components
    # analyzer = AustralianPerformanceAnalyzer(tax_calculator, risk_controller)
    
    print("Performance Analysis Features:")
    print("- AUD-denominated returns and volatility")
    print("- Tax-adjusted performance metrics")
    print("- Benchmark comparison vs ASX indices")
    print("- Strategy attribution (ML vs Arbitrage)")
    print("- CGT discount benefit calculation")
    print("- Risk-adjusted metrics (Sharpe, Sortino, Calmar)")
    print("- Australian-specific alpha calculations")

if __name__ == "__main__":
    asyncio.run(main())