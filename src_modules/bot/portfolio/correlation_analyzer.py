"""
Advanced Correlation Analysis Engine.
Provides comprehensive correlation monitoring, analysis, and regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class CorrelationRegime(Enum):
    """Correlation regimes."""
    LOW_CORRELATION = "low_correlation"        # < 0.3 average
    MODERATE_CORRELATION = "moderate_correlation"  # 0.3 - 0.6
    HIGH_CORRELATION = "high_correlation"      # 0.6 - 0.8
    CRISIS_CORRELATION = "crisis_correlation"  # > 0.8

class CorrelationMeasure(Enum):
    """Correlation measurement methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ROLLING_CORRELATION = "rolling_correlation"
    DYNAMIC_CORRELATION = "dynamic_correlation"
    TAIL_CORRELATION = "tail_correlation"

@dataclass
class CorrelationAnalysis:
    """Correlation analysis result."""
    correlation_matrix: pd.DataFrame
    correlation_regime: CorrelationRegime
    average_correlation: float
    correlation_clusters: Dict[str, List[str]]
    correlation_stability: float
    regime_probability: Dict[CorrelationRegime, float]
    analysis_date: datetime
    lookback_period: int
    correlation_breakdown: Dict[str, Dict[str, float]]
    tail_correlations: Dict[str, Dict[str, float]]
    rolling_correlations: pd.DataFrame
    diversification_benefits: Dict[str, float]
    correlation_warnings: List[str]
    market_stress_indicator: float

class CorrelationAnalyzer:
    """Advanced correlation analysis and monitoring engine."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Correlation configuration
        self.correlation_config = {
            'default_lookback': 252,  # 1 year
            'rolling_window': 30,     # 30 days
            'clustering_threshold': 0.7,
            'regime_thresholds': {
                'low': 0.3,
                'moderate': 0.6,
                'high': 0.8
            },
            'tail_quantile': 0.05,    # 5% tails
            'min_observations': 60,   # Minimum data points
            'stability_lookback': 90  # Days for stability analysis
        }
        
        # Correlation history
        self.correlation_history = []
        
        # Market stress indicators
        self.stress_indicators = {
            'correlation_surge': 0.15,    # Threshold for rapid correlation increase
            'dispersion_collapse': 0.1,   # Threshold for low cross-asset dispersion
            'tail_dependence': 0.3        # Threshold for elevated tail correlations
        }
        
        self.logger.info("CorrelationAnalyzer initialized")
    
    async def analyze_correlations(self, 
                                 price_data: pd.DataFrame,
                                 method: CorrelationMeasure = CorrelationMeasure.PEARSON,
                                 lookback_period: int = None) -> CorrelationAnalysis:
        """Perform comprehensive correlation analysis."""
        try:
            # Validate inputs
            self._validate_correlation_inputs(price_data)
            
            # Set lookback period
            lookback = lookback_period or self.correlation_config['default_lookback']
            lookback = min(lookback, len(price_data))
            
            # Get data for analysis
            analysis_data = price_data.tail(lookback)
            returns = analysis_data.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix(returns, method)
            
            # Determine correlation regime
            regime, regime_probs = await self._determine_correlation_regime(correlation_matrix)
            
            # Calculate average correlation
            avg_correlation = await self._calculate_average_correlation(correlation_matrix)
            
            # Perform correlation clustering
            clusters = await self._perform_correlation_clustering(correlation_matrix)
            
            # Calculate correlation stability
            stability = await self._calculate_correlation_stability(price_data, method)
            
            # Calculate rolling correlations
            rolling_corr = await self._calculate_rolling_correlations(returns)
            
            # Calculate tail correlations
            tail_corr = await self._calculate_tail_correlations(returns)
            
            # Calculate diversification benefits
            div_benefits = await self._calculate_diversification_benefits(correlation_matrix)
            
            # Generate correlation breakdown
            breakdown = await self._generate_correlation_breakdown(correlation_matrix)
            
            # Detect correlation warnings
            warnings = await self._detect_correlation_warnings(correlation_matrix, returns)
            
            # Calculate market stress indicator
            stress_indicator = await self._calculate_market_stress_indicator(
                correlation_matrix, returns
            )
            
            # Create analysis result
            analysis = CorrelationAnalysis(
                correlation_matrix=correlation_matrix,
                correlation_regime=regime,
                average_correlation=avg_correlation,
                correlation_clusters=clusters,
                correlation_stability=stability,
                regime_probability=regime_probs,
                analysis_date=datetime.now(),
                lookback_period=lookback,
                correlation_breakdown=breakdown,
                tail_correlations=tail_corr,
                rolling_correlations=rolling_corr,
                diversification_benefits=div_benefits,
                correlation_warnings=warnings,
                market_stress_indicator=stress_indicator
            )
            
            # Store analysis
            self._store_correlation_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            raise
    
    async def _calculate_correlation_matrix(self, 
                                          returns: pd.DataFrame,
                                          method: CorrelationMeasure) -> pd.DataFrame:
        """Calculate correlation matrix using specified method."""
        try:
            if method == CorrelationMeasure.PEARSON:
                return returns.corr(method='pearson')
            elif method == CorrelationMeasure.SPEARMAN:
                return returns.corr(method='spearman')
            elif method == CorrelationMeasure.KENDALL:
                return returns.corr(method='kendall')
            elif method == CorrelationMeasure.ROLLING_CORRELATION:
                # Use current period for base correlation
                return returns.corr(method='pearson')
            else:
                # Default to Pearson
                return returns.corr(method='pearson')
            
        except Exception as e:
            self.logger.error(f"Failed to calculate correlation matrix: {e}")
            # Return identity matrix as fallback
            n_assets = len(returns.columns)
            return pd.DataFrame(
                np.eye(n_assets), 
                index=returns.columns, 
                columns=returns.columns
            )
    
    async def _determine_correlation_regime(self, 
                                          correlation_matrix: pd.DataFrame) -> Tuple[CorrelationRegime, Dict[CorrelationRegime, float]]:
        """Determine current correlation regime."""
        try:
            # Calculate average correlation (excluding diagonal)
            corr_values = correlation_matrix.values
            n = len(corr_values)
            
            # Get upper triangle values (excluding diagonal)
            upper_triangle = corr_values[np.triu_indices(n, k=1)]
            avg_corr = np.nanmean(upper_triangle)
            
            # Determine regime based on thresholds
            thresholds = self.correlation_config['regime_thresholds']
            
            if avg_corr < thresholds['low']:
                primary_regime = CorrelationRegime.LOW_CORRELATION
            elif avg_corr < thresholds['moderate']:
                primary_regime = CorrelationRegime.MODERATE_CORRELATION
            elif avg_corr < thresholds['high']:
                primary_regime = CorrelationRegime.HIGH_CORRELATION
            else:
                primary_regime = CorrelationRegime.CRISIS_CORRELATION
            
            # Calculate regime probabilities (fuzzy boundaries)
            regime_probs = {}
            
            # Low correlation probability
            if avg_corr <= thresholds['low']:
                low_prob = 1.0
            else:
                low_prob = max(0, (thresholds['moderate'] - avg_corr) / 
                              (thresholds['moderate'] - thresholds['low']))
            
            # Moderate correlation probability
            if thresholds['low'] <= avg_corr <= thresholds['moderate']:
                mod_prob = 1.0 - abs(avg_corr - (thresholds['low'] + thresholds['moderate'])/2) / \
                          ((thresholds['moderate'] - thresholds['low'])/2)
            else:
                mod_prob = max(0, 1.0 - abs(avg_corr - (thresholds['low'] + thresholds['moderate'])/2) / 0.2)
            
            # High correlation probability
            if thresholds['moderate'] <= avg_corr <= thresholds['high']:
                high_prob = 1.0 - abs(avg_corr - (thresholds['moderate'] + thresholds['high'])/2) / \
                           ((thresholds['high'] - thresholds['moderate'])/2)
            else:
                high_prob = max(0, 1.0 - abs(avg_corr - (thresholds['moderate'] + thresholds['high'])/2) / 0.2)
            
            # Crisis correlation probability
            if avg_corr >= thresholds['high']:
                crisis_prob = 1.0
            else:
                crisis_prob = max(0, (avg_corr - thresholds['moderate']) / 
                                 (thresholds['high'] - thresholds['moderate']))
            
            # Normalize probabilities
            total_prob = low_prob + mod_prob + high_prob + crisis_prob
            if total_prob > 0:
                regime_probs = {
                    CorrelationRegime.LOW_CORRELATION: low_prob / total_prob,
                    CorrelationRegime.MODERATE_CORRELATION: mod_prob / total_prob,
                    CorrelationRegime.HIGH_CORRELATION: high_prob / total_prob,
                    CorrelationRegime.CRISIS_CORRELATION: crisis_prob / total_prob
                }
            else:
                regime_probs = {regime: 0.25 for regime in CorrelationRegime}
            
            return primary_regime, regime_probs
            
        except Exception as e:
            self.logger.error(f"Failed to determine correlation regime: {e}")
            return CorrelationRegime.MODERATE_CORRELATION, {regime: 0.25 for regime in CorrelationRegime}
    
    async def _calculate_average_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate average pairwise correlation."""
        try:
            corr_values = correlation_matrix.values
            n = len(corr_values)
            
            # Get upper triangle values (excluding diagonal)
            upper_triangle = corr_values[np.triu_indices(n, k=1)]
            return float(np.nanmean(upper_triangle))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate average correlation: {e}")
            return 0.0
    
    async def _perform_correlation_clustering(self, 
                                            correlation_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """Perform hierarchical clustering based on correlations."""
        try:
            if not HAS_SCIPY:
                # Simple clustering without scipy
                return await self._simple_correlation_clustering(correlation_matrix)
            
            # Convert correlation to distance
            distance_matrix = 1 - correlation_matrix.abs()
            
            # Perform hierarchical clustering
            condensed_distances = squareform(distance_matrix.values, checks=False)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Get clusters
            threshold = 1 - self.correlation_config['clustering_threshold']
            cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
            
            # Organize clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                cluster_name = f"Cluster_{label}"
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(correlation_matrix.index[i])
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Correlation clustering failed: {e}")
            return await self._simple_correlation_clustering(correlation_matrix)
    
    async def _simple_correlation_clustering(self, 
                                           correlation_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """Simple correlation-based clustering without scipy."""
        try:
            threshold = self.correlation_config['clustering_threshold']
            symbols = correlation_matrix.index.tolist()
            clusters = {}
            assigned = set()
            cluster_id = 1
            
            for symbol in symbols:
                if symbol in assigned:
                    continue
                
                # Start new cluster
                cluster_name = f"Cluster_{cluster_id}"
                cluster_members = [symbol]
                assigned.add(symbol)
                
                # Find highly correlated assets
                correlations = correlation_matrix[symbol]
                for other_symbol in symbols:
                    if (other_symbol != symbol and 
                        other_symbol not in assigned and
                        abs(correlations[other_symbol]) >= threshold):
                        cluster_members.append(other_symbol)
                        assigned.add(other_symbol)
                
                clusters[cluster_name] = cluster_members
                cluster_id += 1
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Simple correlation clustering failed: {e}")
            return {"Cluster_1": correlation_matrix.index.tolist()}
    
    async def _calculate_correlation_stability(self, 
                                             price_data: pd.DataFrame,
                                             method: CorrelationMeasure) -> float:
        """Calculate correlation stability over time."""
        try:
            stability_period = self.correlation_config['stability_lookback']
            window_size = 60  # 60-day rolling window for stability
            
            if len(price_data) < stability_period + window_size:
                return 0.5  # Default moderate stability
            
            # Calculate rolling correlation matrices
            returns = price_data.pct_change().dropna()
            correlations_over_time = []
            
            for i in range(window_size, len(returns) - stability_period + window_size, 10):  # Every 10 days
                window_returns = returns.iloc[i-window_size:i]
                if len(window_returns) >= 30:  # Minimum observations
                    corr_matrix = await self._calculate_correlation_matrix(window_returns, method)
                    
                    # Get average correlation
                    avg_corr = await self._calculate_average_correlation(corr_matrix)
                    correlations_over_time.append(avg_corr)
            
            if len(correlations_over_time) < 3:
                return 0.5
            
            # Calculate stability as inverse of standard deviation
            corr_std = np.std(correlations_over_time)
            stability = 1.0 / (1.0 + corr_std * 10)  # Scale factor
            
            return float(np.clip(stability, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate correlation stability: {e}")
            return 0.5
    
    async def _calculate_rolling_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlations."""
        try:
            window = self.correlation_config['rolling_window']
            
            if len(returns) < window * 2:
                # Return current correlations if insufficient data
                return returns.corr()
            
            # Calculate rolling correlations for each pair
            symbols = returns.columns.tolist()
            rolling_corr_data = {}
            
            # Sample every 5 days to reduce computation
            step_size = max(1, len(returns) // 50)  # Max 50 data points
            
            for i in range(window, len(returns), step_size):
                window_returns = returns.iloc[i-window:i]
                corr_matrix = window_returns.corr()
                
                timestamp = returns.index[i]
                
                # Store upper triangle correlations
                for j, symbol1 in enumerate(symbols):
                    for k, symbol2 in enumerate(symbols[j+1:], j+1):
                        pair_name = f"{symbol1}_{symbol2}"
                        if pair_name not in rolling_corr_data:
                            rolling_corr_data[pair_name] = []
                        
                        correlation = corr_matrix.iloc[j, k]
                        rolling_corr_data[pair_name].append({
                            'timestamp': timestamp,
                            'correlation': correlation
                        })
            
            # Convert to DataFrame
            if rolling_corr_data:
                # Create multi-index DataFrame
                all_data = []
                for pair, data in rolling_corr_data.items():
                    for entry in data:
                        all_data.append({
                            'timestamp': entry['timestamp'],
                            'pair': pair,
                            'correlation': entry['correlation']
                        })
                
                if all_data:
                    df = pd.DataFrame(all_data)
                    rolling_corr = df.pivot(index='timestamp', columns='pair', values='correlation')
                    return rolling_corr
            
            # Fallback to current correlation matrix
            return returns.corr()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate rolling correlations: {e}")
            return returns.corr()
    
    async def _calculate_tail_correlations(self, returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate tail correlations (correlations in extreme scenarios)."""
        try:
            quantile = self.correlation_config['tail_quantile']
            symbols = returns.columns.tolist()
            tail_correlations = {}
            
            for symbol in symbols:
                tail_correlations[symbol] = {}
                
                # Get extreme negative returns for this symbol
                symbol_returns = returns[symbol]
                tail_threshold = symbol_returns.quantile(quantile)
                extreme_days = symbol_returns <= tail_threshold
                
                if extreme_days.sum() < 5:  # Need minimum observations
                    for other_symbol in symbols:
                        tail_correlations[symbol][other_symbol] = 0.0
                    continue
                
                # Calculate correlations during extreme days
                tail_returns = returns[extreme_days]
                
                if len(tail_returns) >= 5:
                    tail_corr_matrix = tail_returns.corr()
                    
                    for other_symbol in symbols:
                        if symbol == other_symbol:
                            tail_correlations[symbol][other_symbol] = 1.0
                        else:
                            correlation = tail_corr_matrix.loc[symbol, other_symbol]
                            tail_correlations[symbol][other_symbol] = float(correlation) if not np.isnan(correlation) else 0.0
                else:
                    for other_symbol in symbols:
                        tail_correlations[symbol][other_symbol] = 0.0
            
            return tail_correlations
            
        except Exception as e:
            self.logger.error(f"Failed to calculate tail correlations: {e}")
            return {}
    
    async def _calculate_diversification_benefits(self, 
                                                correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate diversification benefits for each asset."""
        try:
            symbols = correlation_matrix.index.tolist()
            div_benefits = {}
            
            for symbol in symbols:
                # Calculate average correlation with other assets
                correlations = correlation_matrix[symbol].drop(symbol)
                avg_corr = correlations.abs().mean()
                
                # Diversification benefit is inverse of correlation
                # Higher correlation = lower diversification benefit
                div_benefit = 1.0 - avg_corr
                div_benefits[symbol] = float(np.clip(div_benefit, 0, 1))
            
            return div_benefits
            
        except Exception as e:
            self.logger.error(f"Failed to calculate diversification benefits: {e}")
            return {symbol: 0.5 for symbol in correlation_matrix.index}
    
    async def _generate_correlation_breakdown(self, 
                                            correlation_matrix: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate detailed correlation breakdown."""
        try:
            breakdown = {}
            symbols = correlation_matrix.index.tolist()
            
            for symbol in symbols:
                symbol_breakdown = {}
                correlations = correlation_matrix[symbol]
                
                # Positive correlations
                positive_corrs = correlations[correlations > 0].drop(symbol, errors='ignore')
                symbol_breakdown['avg_positive_correlation'] = float(positive_corrs.mean()) if len(positive_corrs) > 0 else 0.0
                symbol_breakdown['max_positive_correlation'] = float(positive_corrs.max()) if len(positive_corrs) > 0 else 0.0
                
                # Negative correlations
                negative_corrs = correlations[correlations < 0]
                symbol_breakdown['avg_negative_correlation'] = float(negative_corrs.mean()) if len(negative_corrs) > 0 else 0.0
                symbol_breakdown['min_negative_correlation'] = float(negative_corrs.min()) if len(negative_corrs) > 0 else 0.0
                
                # Overall statistics
                other_corrs = correlations.drop(symbol, errors='ignore')
                symbol_breakdown['avg_absolute_correlation'] = float(other_corrs.abs().mean()) if len(other_corrs) > 0 else 0.0
                symbol_breakdown['correlation_volatility'] = float(other_corrs.std()) if len(other_corrs) > 0 else 0.0
                
                # High correlation count
                high_corr_threshold = 0.7
                symbol_breakdown['high_correlation_count'] = int((other_corrs.abs() > high_corr_threshold).sum())
                
                breakdown[symbol] = symbol_breakdown
            
            return breakdown
            
        except Exception as e:
            self.logger.error(f"Failed to generate correlation breakdown: {e}")
            return {}
    
    async def _detect_correlation_warnings(self, 
                                         correlation_matrix: pd.DataFrame,
                                         returns: pd.DataFrame) -> List[str]:
        """Detect correlation-related warnings and risks."""
        warnings = []
        
        try:
            # Average correlation
            avg_corr = await self._calculate_average_correlation(correlation_matrix)
            
            # Warning 1: Very high correlations (crisis mode)
            if avg_corr > self.correlation_config['regime_thresholds']['high']:
                warnings.append(f"Crisis-level correlations detected (avg: {avg_corr:.2f}). Diversification benefits severely reduced.")
            
            # Warning 2: Sudden correlation spike
            if len(self.correlation_history) >= 2:
                prev_avg_corr = self.correlation_history[-1].average_correlation
                corr_change = avg_corr - prev_avg_corr
                
                if corr_change > self.stress_indicators['correlation_surge']:
                    warnings.append(f"Rapid correlation increase detected (+{corr_change:.2f}). Potential market stress.")
            
            # Warning 3: Extreme correlations in individual pairs
            corr_values = correlation_matrix.values
            n = len(corr_values)
            upper_triangle = corr_values[np.triu_indices(n, k=1)]
            extreme_corr_count = np.sum(np.abs(upper_triangle) > 0.9)
            
            if extreme_corr_count > len(upper_triangle) * 0.1:  # More than 10% extreme
                warnings.append(f"{extreme_corr_count} asset pairs show extreme correlation (>0.9). Review portfolio construction.")
            
            # Warning 4: Low diversification benefits
            div_benefits = await self._calculate_diversification_benefits(correlation_matrix)
            low_div_assets = [asset for asset, benefit in div_benefits.items() if benefit < 0.2]
            
            if len(low_div_assets) > len(div_benefits) * 0.3:  # More than 30% of assets
                warnings.append(f"Multiple assets ({len(low_div_assets)}) provide low diversification benefits.")
            
            # Warning 5: Correlation instability
            recent_vol = returns.tail(30).std().mean()  # Recent volatility
            long_vol = returns.std().mean()  # Long-term volatility
            
            if recent_vol > long_vol * 1.5:
                warnings.append("High recent volatility may indicate correlation regime shift.")
            
        except Exception as e:
            self.logger.error(f"Failed to detect correlation warnings: {e}")
            warnings.append("Unable to perform complete correlation risk assessment.")
        
        return warnings
    
    async def _calculate_market_stress_indicator(self, 
                                               correlation_matrix: pd.DataFrame,
                                               returns: pd.DataFrame) -> float:
        """Calculate market stress indicator based on correlation patterns."""
        try:
            stress_components = []
            
            # Component 1: Average correlation level
            avg_corr = await self._calculate_average_correlation(correlation_matrix)
            corr_stress = min(1.0, avg_corr / self.correlation_config['regime_thresholds']['high'])
            stress_components.append(corr_stress)
            
            # Component 2: Correlation volatility
            recent_returns = returns.tail(60) if len(returns) >= 60 else returns
            if len(recent_returns) >= 30:
                recent_corr = recent_returns.corr()
                recent_avg_corr = await self._calculate_average_correlation(recent_corr)
                
                if len(self.correlation_history) > 0:
                    historical_avg = np.mean([h.average_correlation for h in self.correlation_history[-10:]])
                    corr_volatility = abs(recent_avg_corr - historical_avg)
                    vol_stress = min(1.0, corr_volatility / 0.3)
                    stress_components.append(vol_stress)
            
            # Component 3: Tail correlation stress
            tail_corr = await self._calculate_tail_correlations(recent_returns)
            if tail_corr:
                tail_values = []
                for symbol, corrs in tail_corr.items():
                    for other_symbol, corr in corrs.items():
                        if symbol != other_symbol:
                            tail_values.append(abs(corr))
                
                if tail_values:
                    avg_tail_corr = np.mean(tail_values)
                    tail_stress = min(1.0, avg_tail_corr / self.stress_indicators['tail_dependence'])
                    stress_components.append(tail_stress)
            
            # Component 4: Market dispersion collapse
            symbol_returns = recent_returns.mean()  # Average returns by symbol
            return_dispersion = symbol_returns.std()
            dispersion_stress = 1.0 - min(1.0, return_dispersion / self.stress_indicators['dispersion_collapse'])
            stress_components.append(dispersion_stress)
            
            # Combine stress components
            if stress_components:
                market_stress = np.mean(stress_components)
            else:
                market_stress = 0.0
            
            return float(np.clip(market_stress, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate market stress indicator: {e}")
            return 0.0
    
    def _validate_correlation_inputs(self, price_data: pd.DataFrame):
        """Validate inputs for correlation analysis."""
        if price_data.empty:
            raise ValueError("No price data provided for correlation analysis")
        
        if len(price_data.columns) < 2:
            raise ValueError("Need at least 2 assets for correlation analysis")
        
        if len(price_data) < self.correlation_config['min_observations']:
            raise ValueError(f"Insufficient data: need at least {self.correlation_config['min_observations']} observations")
    
    def _store_correlation_analysis(self, analysis: CorrelationAnalysis):
        """Store correlation analysis in history."""
        self.correlation_history.append(analysis)
        
        # Limit history size
        if len(self.correlation_history) > 100:
            self.correlation_history = self.correlation_history[-50:]
    
    def get_correlation_history(self, lookback_days: int = None) -> List[CorrelationAnalysis]:
        """Get correlation analysis history."""
        if lookback_days is None:
            return self.correlation_history.copy()
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        return [
            analysis for analysis in self.correlation_history 
            if analysis.analysis_date >= cutoff_date
        ]
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of correlation analyzer."""
        if not self.correlation_history:
            return {
                'total_analyses': 0,
                'current_regime': None,
                'average_correlation': None,
                'last_analysis': None
            }
        
        latest = self.correlation_history[-1]
        
        return {
            'total_analyses': len(self.correlation_history),
            'current_regime': latest.correlation_regime.value,
            'average_correlation': latest.average_correlation,
            'correlation_stability': latest.correlation_stability,
            'market_stress_level': latest.market_stress_indicator,
            'active_warnings': len(latest.correlation_warnings),
            'last_analysis': latest.analysis_date,
            'supported_methods': [method.value for method in CorrelationMeasure]
        }
    
    async def get_regime_transition_probability(self) -> Dict[str, float]:
        """Calculate probability of correlation regime transitions."""
        try:
            if len(self.correlation_history) < 10:
                return {regime.value: 0.1 for regime in CorrelationRegime}
            
            # Analyze regime transitions
            recent_regimes = [analysis.correlation_regime for analysis in self.correlation_history[-10:]]
            current_regime = recent_regimes[-1]
            
            # Simple transition probability based on recent history
            regime_counts = {}
            for regime in CorrelationRegime:
                regime_counts[regime.value] = recent_regimes.count(regime) / len(recent_regimes)
            
            # Adjust based on current market stress
            latest_analysis = self.correlation_history[-1]
            stress_level = latest_analysis.market_stress_indicator
            
            # Higher stress increases probability of crisis regime
            if stress_level > 0.5:
                regime_counts[CorrelationRegime.CRISIS_CORRELATION.value] *= (1 + stress_level)
                regime_counts[CorrelationRegime.HIGH_CORRELATION.value] *= (1 + stress_level * 0.5)
            
            # Normalize probabilities
            total_prob = sum(regime_counts.values())
            if total_prob > 0:
                regime_counts = {regime: prob / total_prob for regime, prob in regime_counts.items()}
            
            return regime_counts
            
        except Exception as e:
            self.logger.error(f"Failed to calculate regime transition probabilities: {e}")
            return {regime.value: 0.25 for regime in CorrelationRegime}