"""
Portfolio Risk Analysis for Multi-Asset Trading Strategies.

This module provides comprehensive portfolio risk assessment including:

- Correlation analysis and clustering
- Sector/category exposure monitoring
- Risk factor decomposition
- Portfolio concentration metrics
- Tail risk and extreme scenario analysis
- Dynamic risk monitoring and alerts
- Risk attribution and contribution analysis
- Stress testing and scenario analysis

All analyses are designed for cryptocurrency markets with their
unique risk characteristics and high volatility patterns.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats, cluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging import TradingLogger


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskAlert:
    """Container for risk alerts."""
    
    alert_type: str
    risk_level: RiskLevel
    asset: Optional[str]
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    action_required: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PortfolioRiskMetrics:
    """Container for portfolio risk metrics."""
    
    total_portfolio_value: float
    portfolio_volatility: float
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    concentration_index: float
    correlation_risk: float
    tail_risk: float
    liquidity_risk: float
    risk_alerts: List[RiskAlert]
    timestamp: datetime
    
    def __post_init__(self):
        if self.risk_alerts is None:
            self.risk_alerts = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CorrelationAnalyzer:
    """
    Advanced correlation analysis for portfolio risk assessment.
    
    This class provides sophisticated correlation analysis including
    dynamic correlations, correlation clustering, and tail dependence.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("CorrelationAnalyzer")
        
    def _default_config(self) -> Dict:
        """Default configuration for correlation analyzer."""
        return {
            'correlation_window': 60,        # Days for correlation calculation
            'min_periods': 30,               # Minimum periods for correlation
            'correlation_threshold': 0.7,    # High correlation threshold
            'tail_threshold': 0.05,          # Tail dependence threshold
            'clustering_method': 'ward',     # Hierarchical clustering method
            'n_clusters': 5,                 # Number of correlation clusters
            'dynamic_window': 20,            # Window for dynamic correlation
            'significance_level': 0.05,      # Statistical significance level
        }
    
    def calculate_correlation_matrix(
        self,
        returns_data: pd.DataFrame,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive correlation matrix with statistical tests.
        
        Args:
            returns_data: DataFrame with asset returns
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Dictionary with correlation results
        """
        # Clean data
        clean_data = returns_data.dropna()
        
        if len(clean_data) < self.config['min_periods']:
            self.logger.warning("Insufficient data for correlation analysis")
            return self._empty_correlation_result(returns_data.columns)
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = clean_data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = clean_data.corr(method='spearman')
        else:  # kendall
            corr_matrix = clean_data.corr(method='kendall')
        
        # Calculate p-values for correlations
        p_values = self._calculate_correlation_pvalues(clean_data, method)
        
        # Identify high correlations
        high_corr_pairs = self._find_high_correlations(corr_matrix, p_values)
        
        # Calculate dynamic correlations
        dynamic_corr = self._calculate_dynamic_correlations(clean_data)
        
        # Perform correlation clustering
        clusters = self._perform_correlation_clustering(corr_matrix)
        
        return {
            'correlation_matrix': corr_matrix,
            'p_values': p_values,
            'high_correlations': high_corr_pairs,
            'dynamic_correlations': dynamic_corr,
            'correlation_clusters': clusters,
            'average_correlation': self._calculate_average_correlation(corr_matrix),
            'correlation_eigenvalues': self._calculate_correlation_eigenvalues(corr_matrix)
        }
    
    def _calculate_correlation_pvalues(
        self,
        data: pd.DataFrame,
        method: str
    ) -> pd.DataFrame:
        """Calculate p-values for correlation coefficients."""
        n_assets = len(data.columns)
        p_values = np.ones((n_assets, n_assets))
        
        for i, asset1 in enumerate(data.columns):
            for j, asset2 in enumerate(data.columns):
                if i != j:
                    if method == 'pearson':
                        _, p_val = stats.pearsonr(data[asset1].dropna(), data[asset2].dropna())
                    elif method == 'spearman':
                        _, p_val = stats.spearmanr(data[asset1].dropna(), data[asset2].dropna())
                    else:  # kendall
                        _, p_val = stats.kendalltau(data[asset1].dropna(), data[asset2].dropna())
                    
                    p_values[i, j] = p_val
        
        return pd.DataFrame(p_values, index=data.columns, columns=data.columns)
    
    def _find_high_correlations(
        self,
        corr_matrix: pd.DataFrame,
        p_values: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Find statistically significant high correlations."""
        high_corr_pairs = []
        
        for i, asset1 in enumerate(corr_matrix.columns):
            for j, asset2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.iloc[i, j]
                    p_val = p_values.iloc[i, j]
                    
                    if (abs(corr_val) >= self.config['correlation_threshold'] and 
                        p_val < self.config['significance_level']):
                        
                        high_corr_pairs.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'correlation ': corr_val,
                            'p_value': p_val,
                            'significant': True
                        })
        
        return high_corr_pairs
    
    def _calculate_dynamic_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlations over time."""
        window = self.config['dynamic_window']
        
        if len(data) < window * 2:
            return pd.DataFrame()
        
        # Calculate rolling correlation for the first two assets as example
        if len(data.columns) >= 2:
            asset1, asset2 = data.columns[0], data.columns[1]
            rolling_corr = data[asset1].rolling(window).corr(data[asset2])
            
            return pd.DataFrame({
                'date': data.index[window-1:],
                'correlation': rolling_corr.dropna(),
                'asset_pair': f"{asset1}_{asset2}"
            })
        
        return pd.DataFrame()
    
    def _perform_correlation_clustering(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform hierarchical clustering based on correlations."""
        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix.abs()
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
        
        # Convert to condensed distance matrix
        condensed_distances = squareform(distance_matrix.values)
        
        # Perform clustering
        linkage_matrix = linkage(condensed_distances, method=self.config['clustering_method'])
        
        # Get cluster assignments
        cluster_labels = fcluster(linkage_matrix, self.config['n_clusters'], criterion='maxclust')
        
        # Create cluster dictionary
        clusters = {}
        for i, asset in enumerate(corr_matrix.columns):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(asset)
        
        return {
            'clusters': clusters,
            'linkage_matrix': linkage_matrix,
            'n_clusters': len(clusters)
        }
    
    def _calculate_average_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate average pairwise correlation."""
        # Get upper triangle of correlation matrix (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.values[mask]
        
        return np.mean(correlations)
    
    def _calculate_correlation_eigenvalues(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate eigenvalues of correlation matrix for risk analysis."""
        eigenvalues = np.linalg.eigvals(corr_matrix.values)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        return {
            'largest_eigenvalue': eigenvalues[0],
            'smallest_eigenvalue': eigenvalues[-1],
            'condition_number': eigenvalues[0] / eigenvalues[-1],
            'explained_variance_ratio': eigenvalues[0] / np.sum(eigenvalues)
        }
    
    def _empty_correlation_result(self, assets: List[str]) -> Dict[str, Any]:
        """Return empty correlation result for insufficient data."""
        n_assets = len(assets)
        identity_matrix = pd.DataFrame(
            np.eye(n_assets),
            index=assets,
            columns=assets
        )
        
        return {
            'correlation_matrix': identity_matrix,
            'p_values': identity_matrix,
            'high_correlations': [],
            'dynamic_correlations': pd.DataFrame(),
            'correlation_clusters': {'clusters': {1: list(assets)}, 'n_clusters': 1},
            'average_correlation': 0.0,
            'correlation_eigenvalues': {
                'largest_eigenvalue': 1.0,
                'smallest_eigenvalue': 1.0,
                'condition_number': 1.0,
                'explained_variance_ratio': 1.0
            }
        }


class SectorExposureAnalyzer:
    """
    Sector and category exposure analysis for cryptocurrency portfolios.
    
    This class analyzes portfolio exposure across different cryptocurrency
    sectors and categories to identify concentration risks.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("SectorExposureAnalyzer")
        
        # Define cryptocurrency sectors/categories
        self.crypto_sectors = self._define_crypto_sectors()
        
    def _default_config(self) -> Dict:
        """Default configuration for sector exposure analyzer."""
        return {
            'max_sector_exposure': 0.4,      # Maximum 40% exposure to any sector
            'max_single_asset': 0.2,         # Maximum 20% in single asset
            'min_diversification': 3,        # Minimum 3 sectors
            'concentration_threshold': 0.6,   # Concentration warning threshold
            'rebalance_threshold': 0.05,     # 5% threshold for rebalancing alerts
        }
    
    def _define_crypto_sectors(self) -> Dict[str, List[str]]:
        """Define cryptocurrency sectors and their constituent assets."""
        return {
            'layer1': ['BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'ATOM', 'NEAR'],
            'defi': ['UNI', 'AAVE', 'COMP', 'MKR', 'SUSHI', 'CRV', '1INCH', 'YFI'],
            'exchange_tokens': ['BNB', 'CRO', 'FTT', 'KCS', 'HT', 'OKB'],
            'payment': ['XRP', 'LTC', 'BCH', 'XLM', 'NANO', 'DASH'],
            'smart_contracts': ['ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'ALGO', 'TEZOS'],
            'gaming_nft': ['AXS', 'SAND', 'MANA', 'ENJ', 'GALA', 'ILV'],
            'oracles': ['LINK', 'BAND', 'TRB', 'API3'],
            'privacy': ['XMR', 'ZEC', 'DASH', 'SCRT'],
            'infrastructure': ['FIL', 'AR', 'GRT', 'THETA', 'STORJ'],
            'stablecoins': ['USDT', 'USDC', 'BUSD', 'DAI', 'FRAX', 'LUSD'],
            'meme': ['DOGE', 'SHIB', 'FLOKI', 'PEPE'],
            'web3': ['FIL', 'AR', 'GRT', 'THETA', 'BAT', 'STORJ']
        }
    
    def analyze_sector_exposure(
        self,
        portfolio_weights: Dict[str, float],
        custom_sectors: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze portfolio exposure across cryptocurrency sectors.
        
        Args:
            portfolio_weights: Dictionary of asset weights
            custom_sectors: Optional custom sector definitions
            
        Returns:
            Dictionary with sector exposure analysis
        """
        sectors = custom_sectors or self.crypto_sectors
        
        # Calculate sector exposures
        sector_exposures = {}
        unclassified_exposure = 0.0
        
        for asset, weight in portfolio_weights.items():
            asset_upper = asset.upper()
            classified = False
            
            for sector, assets in sectors.items():
                if asset_upper in [a.upper() for a in assets]:
                    if sector not in sector_exposures:
                        sector_exposures[sector] = 0.0
                    sector_exposures[sector] += weight
                    classified = True
                    break
            
            if not classified:
                unclassified_exposure += weight
        
        if unclassified_exposure > 0:
            sector_exposures['unclassified'] = unclassified_exposure
        
        # Calculate risk metrics
        risk_metrics = self._calculate_sector_risk_metrics(sector_exposures, portfolio_weights)
        
        # Generate alerts
        alerts = self._generate_sector_alerts(sector_exposures, portfolio_weights)
        
        return {
            'sector_exposures': sector_exposures,
            'risk_metrics': risk_metrics,
            'alerts': alerts,
            'diversification_score': self._calculate_diversification_score(sector_exposures),
            'concentration_metrics': self._calculate_concentration_metrics(portfolio_weights),
            'rebalancing_suggestions': self._generate_rebalancing_suggestions(
                sector_exposures, portfolio_weights
            )
        }
    
    def _calculate_sector_risk_metrics(
        self,
        sector_exposures: Dict[str, float],
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate sector-based risk metrics."""
        # Herfindahl-Hirschman Index for sector concentration
        hhi_sectors = sum(exposure ** 2 for exposure in sector_exposures.values())
        
        # Herfindahl-Hirschman Index for individual assets
        hhi_assets = sum(weight ** 2 for weight in portfolio_weights.values())
        
        # Number of effective sectors
        effective_sectors = 1 / hhi_sectors if hhi_sectors > 0 else 0
        
        # Maximum sector exposure
        max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0
        
        # Sector entropy (diversification measure)
        sector_entropy = -sum(
            exposure * np.log(exposure) 
            for exposure in sector_exposures.values() 
            if exposure > 0
        )
        
        return {
            'hhi_sectors': hhi_sectors,
            'hhi_assets': hhi_assets,
            'effective_sectors': effective_sectors,
            'max_sector_exposure': max_sector_exposure,
            'sector_entropy': sector_entropy,
            'diversification_ratio': effective_sectors / len(sector_exposures) if sector_exposures else 0
        }
    
    def _generate_sector_alerts(
        self,
        sector_exposures: Dict[str, float],
        portfolio_weights: Dict[str, float]
    ) -> List[RiskAlert]:
        """Generate alerts based on sector exposure analysis."""
        alerts = []
        
        # Check maximum sector exposure
        for sector, exposure in sector_exposures.items():
            if exposure > self.config['max_sector_exposure']:
                alerts.append(RiskAlert(
                    alert_type='sector_concentration',
                    risk_level=RiskLevel.HIGH,
                    asset=None,
                    metric_name=f'{sector}_exposure',
                    current_value=exposure,
                    threshold_value=self.config['max_sector_exposure'],
                    message=f"Sector {sector} exposure ({exposure:.1%}) exceeds maximum ({self.config['max_sector_exposure']:.1%})",
                    action_required=True,
                    timestamp=datetime.now()
                ))
        
        # Check individual asset concentration
        for asset, weight in portfolio_weights.items():
            if weight > self.config['max_single_asset']:
                alerts.append(RiskAlert(
                    alert_type='asset_concentration',
                    risk_level=RiskLevel.MEDIUM,
                    asset=asset,
                    metric_name='asset_weight',
                    current_value=weight,
                    threshold_value=self.config['max_single_asset'],
                    message=f"Asset {asset} weight ({weight:.1%}) exceeds maximum ({self.config['max_single_asset']:.1%})",
                    action_required=True,
                    timestamp=datetime.now()
                ))
        
        # Check minimum diversification
        if len(sector_exposures) < self.config['min_diversification']:
            alerts.append(RiskAlert(
                alert_type='insufficient_diversification',
                risk_level=RiskLevel.MEDIUM,
                asset=None,
                metric_name='sector_count',
                current_value=len(sector_exposures),
                threshold_value=self.config['min_diversification'],
                message=f"Portfolio has only {len(sector_exposures)} sectors, minimum is {self.config['min_diversification']}",
                action_required=True,
                timestamp=datetime.now()
            ))
        
        return alerts
    
    def _calculate_diversification_score(self, sector_exposures: Dict[str, float]) -> float:
        """Calculate portfolio diversification score (0-1 scale)."""
        if not sector_exposures:
            return 0.0
        
        # Perfect diversification would be equal weights across all sectors
        n_sectors = len(sector_exposures)
        ideal_weight = 1.0 / n_sectors
        
        # Calculate deviation from ideal diversification
        deviations = [abs(exposure - ideal_weight) for exposure in sector_exposures.values()]
        avg_deviation = np.mean(deviations)
        
        # Convert to score (lower deviation = higher score)
        diversification_score = max(0, 1 - (avg_deviation * n_sectors))
        
        return diversification_score
    
    def _calculate_concentration_metrics(self, portfolio_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate various concentration metrics."""
        if not portfolio_weights:
            return {}
        
        weights = list(portfolio_weights.values())
        
        # Gini coefficient
        gini_coeff = self._calculate_gini_coefficient(weights)
        
        # Top N concentration ratios
        sorted_weights = sorted(weights, reverse=True)
        cr1 = sorted_weights[0] if len(sorted_weights) >= 1 else 0
        cr3 = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
        cr5 = sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)
        
        return {
            'gini_coefficient': gini_coeff,
            'cr1': cr1,  # Largest holding
            'cr3': cr3,  # Top 3 holdings
            'cr5': cr5,  # Top 5 holdings
            'effective_assets': 1 / sum(w**2 for w in weights)
        }
    
    def _calculate_gini_coefficient(self, weights: List[float]) -> float:
        """Calculate Gini coefficient for portfolio concentration."""
        if not weights:
            return 0.0
        
        sorted_weights = sorted(weights)
        n = len(sorted_weights)
        
        # Calculate Gini coefficient
        gini = (2 * sum((i + 1) * w for i, w in enumerate(sorted_weights))) / (n * sum(sorted_weights)) - (n + 1) / n
        
        return gini
    
    def _generate_rebalancing_suggestions(
        self,
        sector_exposures: Dict[str, float],
        portfolio_weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate rebalancing suggestions based on sector analysis."""
        suggestions = []
        
        # Suggest reducing overweight sectors
        for sector, exposure in sector_exposures.items():
            if exposure > self.config['max_sector_exposure']:
                target_exposure = self.config['max_sector_exposure']
                reduction_needed = exposure - target_exposure
                
                suggestions.append({
                    'action': 'reduce',
                    'sector': sector,
                    'current_exposure': exposure,
                    'target_exposure': target_exposure,
                    'reduction_needed': reduction_needed,
                    'priority': 'high'
                })
        
        # Suggest reducing overweight individual assets
        for asset, weight in portfolio_weights.items():
            if weight > self.config['max_single_asset']:
                target_weight = self.config['max_single_asset']
                reduction_needed = weight - target_weight
                
                suggestions.append({
                    'action': 'reduce',
                    'asset': asset,
                    'current_weight': weight,
                    'target_weight': target_weight,
                    'reduction_needed': reduction_needed,
                    'priority': 'medium'
                })
        
        return suggestions


class TailRiskAnalyzer:
    """
    Tail risk and extreme scenario analysis for cryptocurrency portfolios.
    
    This class provides comprehensive tail risk analysis including:
    - Value at Risk (VaR) and Conditional VaR (CVaR)
    - Extreme value theory analysis
    - Tail dependence analysis
    - Stress testing scenarios
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("TailRiskAnalyzer")
        
    def _default_config(self) -> Dict:
        """Default configuration for tail risk analyzer."""
        return {
            'confidence_levels': [0.95, 0.99, 0.995],  # VaR confidence levels
            'var_method': 'historical',                  # historical, parametric, monte_carlo
            'tail_threshold': 0.1,                      # Threshold for tail events
            'extreme_threshold': 0.05,                  # Threshold for extreme events
            'monte_carlo_simulations': 10000,           # Number of MC simulations
            'stress_scenarios': {                       # Predefined stress scenarios
                'crypto_crash': -0.5,                   # 50% market crash
                'flash_crash': -0.3,                    # 30% flash crash
                'bear_market': -0.4,                    # 40% bear market
                'liquidity_crisis': -0.35               # 35% liquidity crisis
            }
        }
    
    def calculate_portfolio_var(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: Optional[Dict[str, float]] = None,
        asset_returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR for portfolio.
        
        Args:
            portfolio_returns: Historical portfolio returns
            portfolio_weights: Current portfolio weights
            asset_returns: Individual asset returns (for component VaR)
            
        Returns:
            Dictionary with VaR analysis results
        """
        results = {}
        
        for confidence_level in self.config['confidence_levels']:
            if self.config['var_method'] == 'historical':
                var_result = self._historical_var(portfolio_returns, confidence_level)
            elif self.config['var_method'] == 'parametric':
                var_result = self._parametric_var(portfolio_returns, confidence_level)
            else:  # monte_carlo
                var_result = self._monte_carlo_var(portfolio_returns, confidence_level)
            
            results[f'var_{int(confidence_level*100)}'] = var_result
        
        # Calculate component VaR if asset data is available
        if portfolio_weights and asset_returns is not None:
            component_var = self._calculate_component_var(
                asset_returns, portfolio_weights, self.config['confidence_levels'][0]
            )
            results['component_var'] = component_var
        
        # Tail statistics
        results['tail_statistics'] = self._calculate_tail_statistics(portfolio_returns)
        
        return results
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> Dict[str, float]:
        """Calculate historical VaR and CVaR."""
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 100:
            self.logger.warning("Insufficient data for reliable VaR calculation")
        
        # Calculate VaR
        var = np.percentile(clean_returns, (1 - confidence_level) * 100)
        
        # Calculate CVaR (Expected Shortfall)
        tail_returns = clean_returns[clean_returns <= var]
        cvar = tail_returns.mean() if len(tail_returns) > 0 else var
        
        return {
            'var': var,
            'cvar': cvar,
            'method': 'historical',
            'confidence_level': confidence_level,
            'sample_size': len(clean_returns)
        }
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> Dict[str, float]:
        """Calculate parametric VaR assuming normal distribution."""
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:
            self.logger.warning("Insufficient data for parametric VaR")
            return self._historical_var(returns, confidence_level)
        
        # Calculate statistics
        mean_return = clean_returns.mean()
        std_return = clean_returns.std()
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean_return + z_score * std_return
        
        # Calculate CVaR for normal distribution
        cvar = mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level)
        
        return {
            'var': var,
            'cvar': cvar,
            'method': 'parametric',
            'confidence_level': confidence_level,
            'mean_return': mean_return,
            'volatility': std_return
        }
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float) -> Dict[str, float]:
        """Calculate Monte Carlo VaR using historical bootstrap."""
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 50:
            return self._historical_var(returns, confidence_level)
        
        # Bootstrap simulations
        simulated_returns = np.random.choice(
            clean_returns.values,
            size=self.config['monte_carlo_simulations'],
            replace=True
        )
        
        # Calculate VaR and CVaR from simulations
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        tail_returns = simulated_returns[simulated_returns <= var]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var
        
        return {
            'var': var,
            'cvar': cvar,
            'method': 'monte_carlo',
            'confidence_level': confidence_level,
            'simulations': self.config['monte_carlo_simulations']
        }
    
    def _calculate_component_var(
        self,
        asset_returns: pd.DataFrame,
        portfolio_weights: Dict[str, float],
        confidence_level: float
    ) -> Dict[str, Any]:
        """Calculate component VaR for individual assets."""
        # Clean data and align with portfolio weights
        clean_returns = asset_returns.dropna()
        common_assets = set(clean_returns.columns) & set(portfolio_weights.keys())
        
        if not common_assets:
            return {}
        
        clean_returns = clean_returns[list(common_assets)]
        weights = np.array([portfolio_weights[asset] for asset in clean_returns.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (clean_returns * weights).sum(axis=1)
        
        # Calculate portfolio VaR
        portfolio_var = self._historical_var(portfolio_returns, confidence_level)['var']
        
        # Calculate marginal VaR for each asset
        marginal_vars = {}
        component_vars = {}
        
        for i, asset in enumerate(clean_returns.columns):
            # Calculate marginal VaR (sensitivity of portfolio VaR to asset weight)
            asset_returns_series = clean_returns.iloc[:, i]
            
            # Approximate marginal VaR using correlation
            asset_var = self._historical_var(asset_returns_series, confidence_level)['var']
            correlation = portfolio_returns.corr(asset_returns_series)
            
            marginal_var = correlation * asset_var
            marginal_vars[asset] = marginal_var
            
            # Component VaR = weight * marginal VaR
            component_vars[asset] = weights[i] * marginal_var
        
        return {
            'portfolio_var': portfolio_var,
            'marginal_vars': marginal_vars,
            'component_vars': component_vars,
            'var_contributions': {
                asset: comp_var / portfolio_var if portfolio_var != 0 else 0
                for asset, comp_var in component_vars.items()
            }
        }
    
    def _calculate_tail_statistics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate various tail risk statistics."""
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 50:
            return {}
        
        # Sort returns
        sorted_returns = clean_returns.sort_values()
        
        # Tail threshold
        tail_size = int(len(sorted_returns) * self.config['tail_threshold'])
        tail_returns = sorted_returns.iloc[:tail_size]
        
        # Calculate tail statistics
        tail_mean = tail_returns.mean()
        tail_std = tail_returns.std()
        tail_skewness = stats.skew(tail_returns)
        tail_kurtosis = stats.kurtosis(tail_returns)
        
        # Maximum drawdown in tail
        tail_cumulative = (1 + tail_returns).cumprod()
        tail_running_max = tail_cumulative.expanding().max()
        tail_drawdowns = (tail_cumulative - tail_running_max) / tail_running_max
        max_tail_drawdown = abs(tail_drawdowns.min())
        
        return {
            'tail_mean': tail_mean,
            'tail_volatility': tail_std,
            'tail_skewness': tail_skewness,
            'tail_kurtosis': tail_kurtosis,
            'max_tail_drawdown': max_tail_drawdown,
            'tail_sample_size': len(tail_returns)
        }
    
    def stress_test_portfolio(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio under various scenarios.
        
        Args:
            portfolio_weights: Current portfolio weights
            asset_returns: Historical asset returns
            custom_scenarios: Custom stress scenarios
            
        Returns:
            Dictionary with stress test results
        """
        scenarios = custom_scenarios or self._create_default_stress_scenarios(asset_returns)
        
        stress_results = {}
        
        for scenario_name, scenario_shocks in scenarios.items():
            # Calculate portfolio impact
            portfolio_impact = 0.0
            
            for asset, weight in portfolio_weights.items():
                if asset in scenario_shocks:
                    asset_shock = scenario_shocks[asset]
                    portfolio_impact += weight * asset_shock
            
            # Calculate risk metrics under stress
            stress_metrics = self._calculate_stress_metrics(
                portfolio_impact, scenario_name, asset_returns, portfolio_weights
            )
            
            stress_results[scenario_name] = {
                'portfolio_impact': portfolio_impact,
                'scenario_description': self._get_scenario_description(scenario_name),
                'risk_metrics': stress_metrics
            }
        
        return {
            'stress_scenarios': stress_results,
            'worst_case_scenario': min(stress_results.items(), key=lambda x: x[1]['portfolio_impact']),
            'stress_summary': self._summarize_stress_results(stress_results)
        }
    
    def _create_default_stress_scenarios(self, asset_returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Create default stress scenarios based on historical data."""
        scenarios = {}
        
        # Market-wide stress scenarios
        for scenario_name, market_shock in self.config['stress_scenarios'].items():
            scenarios[scenario_name] = {
                asset: market_shock for asset in asset_returns.columns
            }
        
        # Historical worst-case scenarios
        if len(asset_returns) >= 252:  # At least 1 year of data
            # Find worst single day for each asset
            worst_days = {}
            for asset in asset_returns.columns:
                worst_day_return = asset_returns[asset].min()
                worst_days[asset] = worst_day_return
            
            scenarios['historical_worst_day'] = worst_days
            
            # Find worst week (5-day rolling sum)
            worst_weeks = {}
            for asset in asset_returns.columns:
                rolling_returns = asset_returns[asset].rolling(5).sum()
                worst_week_return = rolling_returns.min()
                worst_weeks[asset] = worst_week_return
            
            scenarios['historical_worst_week'] = worst_weeks
        
        return scenarios
    
    def _calculate_stress_metrics(
        self,
        portfolio_impact: float,
        scenario_name: str,
        asset_returns: pd.DataFrame,
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk metrics under stress scenario."""
        # Estimate post-stress portfolio value
        post_stress_value = 1 + portfolio_impact
        
        # Estimate recovery time (simplified)
        historical_returns = (asset_returns * list(portfolio_weights.values())).sum(axis=1)
        avg_positive_return = historical_returns[historical_returns > 0].mean()
        
        recovery_time = abs(portfolio_impact) / avg_positive_return if avg_positive_return > 0 else float('inf')
        
        return {
            'portfolio_impact': portfolio_impact,
            'post_stress_value': post_stress_value,
            'estimated_recovery_days': recovery_time,
            'risk_level': self._assess_stress_risk_level(portfolio_impact)
        }
    
    def _get_scenario_description(self, scenario_name: str) -> str:
        """Get description for stress scenario."""
        descriptions = {
            'crypto_crash': 'Major cryptocurrency market crash (-50%)',
            'flash_crash': 'Sudden market flash crash (-30%)',
            'bear_market': 'Extended bear market conditions (-40%)',
            'liquidity_crisis': 'Market liquidity crisis (-35%)',
            'historical_worst_day': 'Historical worst single-day performance',
            'historical_worst_week': 'Historical worst 5-day performance'
        }
        
        return descriptions.get(scenario_name, f'Custom scenario: {scenario_name}')
    
    def _assess_stress_risk_level(self, portfolio_impact: float) -> str:
        """Assess risk level based on portfolio impact."""
        if portfolio_impact >= -0.1:
            return 'low'
        elif portfolio_impact >= -0.2:
            return 'medium'
        elif portfolio_impact >= -0.3:
            return 'high'
        else:
            return 'extreme'
    
    def _summarize_stress_results(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize stress test results."""
        impacts = [result['portfolio_impact'] for result in stress_results.values()]
        
        return {
            'average_impact': np.mean(impacts),
            'worst_impact': min(impacts),
            'best_impact': max(impacts),
            'impact_volatility': np.std(impacts),
            'scenarios_tested': len(stress_results),
            'extreme_scenarios': len([impact for impact in impacts if impact < -0.3])
        }


class PortfolioRiskMonitor:
    """
    Comprehensive portfolio risk monitoring system.
    
    This class integrates all risk analysis components to provide
    real-time risk monitoring, alerting, and reporting.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("PortfolioRiskMonitor")
        
        # Initialize analyzers
        self.correlation_analyzer = CorrelationAnalyzer(self.config.get('correlation', {}))
        self.sector_analyzer = SectorExposureAnalyzer(self.config.get('sector', {}))
        self.tail_risk_analyzer = TailRiskAnalyzer(self.config.get('tail_risk', {}))
        
        # Risk alert history
        self.alert_history: List[RiskAlert] = []
        
    def _default_config(self) -> Dict:
        """Default configuration for portfolio risk monitor."""
        return {
            'monitoring_frequency': 'daily',    # Monitoring frequency
            'alert_cooldown': 3600,             # Seconds between similar alerts
            'risk_limits': {
                'max_portfolio_var': 0.05,      # 5% daily VaR limit
                'max_correlation': 0.8,         # Maximum average correlation
                'max_sector_exposure': 0.4,     # Maximum sector exposure
                'min_liquidity_ratio': 0.1,     # Minimum liquid asset ratio
            },
            'auto_rebalance': False,            # Automatic rebalancing
            'notification_channels': ['log'],   # Alert channels
            'correlation': {},
            'sector': {},
            'tail_risk': {},
        }
    
    def monitor_portfolio_risk(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        portfolio_value: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> PortfolioRiskMetrics:
        """
        Comprehensive portfolio risk monitoring.
        
        Args:
            portfolio_weights: Current portfolio weights
            asset_returns: Historical asset returns
            portfolio_value: Current portfolio value
            market_data: Additional market data
            
        Returns:
            PortfolioRiskMetrics with comprehensive risk assessment
        """
        self.logger.info("Starting comprehensive portfolio risk monitoring")
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, asset_returns)
        
        # Correlation analysis
        correlation_results = self.correlation_analyzer.calculate_correlation_matrix(asset_returns)
        
        # Sector exposure analysis
        sector_results = self.sector_analyzer.analyze_sector_exposure(portfolio_weights)
        
        # Tail risk analysis
        var_results = self.tail_risk_analyzer.calculate_portfolio_var(portfolio_returns)
        
        # Stress testing
        stress_results = self.tail_risk_analyzer.stress_test_portfolio(
            portfolio_weights, asset_returns
        )
        
        # Calculate comprehensive risk metrics
        risk_metrics = self._calculate_comprehensive_metrics(
            portfolio_returns, correlation_results, sector_results, 
            var_results, stress_results, portfolio_value
        )
        
        # Generate risk alerts
        all_alerts = self._generate_comprehensive_alerts(
            risk_metrics, correlation_results, sector_results, var_results
        )
        
        # Update alert history
        self._update_alert_history(all_alerts)
        
        return risk_metrics
    
    def _calculate_portfolio_returns(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns from asset returns and weights."""
        # Align weights with available returns data
        common_assets = set(portfolio_weights.keys()) & set(asset_returns.columns)
        
        if not common_assets:
            self.logger.warning("No common assets between portfolio and returns data")
            return pd.Series(dtype=float)
        
        # Create weight vector
        weights = pd.Series([portfolio_weights[asset] for asset in common_assets], 
                          index=list(common_assets))
        
        # Calculate weighted returns
        aligned_returns = asset_returns[list(common_assets)]
        portfolio_returns = (aligned_returns * weights).sum(axis=1)
        
        return portfolio_returns
    
    def _calculate_comprehensive_metrics(
        self,
        portfolio_returns: pd.Series,
        correlation_results: Dict[str, Any],
        sector_results: Dict[str, Any],
        var_results: Dict[str, Any],
        stress_results: Dict[str, Any],
        portfolio_value: float
    ) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        # Basic portfolio metrics
        if len(portfolio_returns) > 0:
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            calmar_ratio = self._calculate_calmar_ratio(portfolio_returns, max_drawdown)
        else:
            portfolio_volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
            calmar_ratio = 0.0
        
        # VaR metrics
        portfolio_var = var_results.get('var_95', {}).get('var', 0.0)
        portfolio_cvar = var_results.get('var_95', {}).get('cvar', 0.0)
        
        # Concentration and correlation metrics
        concentration_index = sector_results['risk_metrics']['hhi_assets']
        correlation_risk = correlation_results['average_correlation']
        
        # Tail risk
        tail_stats = var_results.get('tail_statistics', {})
        tail_risk = tail_stats.get('tail_mean', 0.0)
        
        # Liquidity risk (simplified - would need market data for full analysis)
        liquidity_risk = 0.1  # Placeholder - crypto generally has liquidity risk
        
        return PortfolioRiskMetrics(
            total_portfolio_value=portfolio_value,
            portfolio_volatility=portfolio_volatility,
            portfolio_var=abs(portfolio_var),
            portfolio_cvar=abs(portfolio_cvar),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            concentration_index=concentration_index,
            correlation_risk=correlation_risk,
            tail_risk=abs(tail_risk),
            liquidity_risk=liquidity_risk,
            risk_alerts=[],  # Will be populated by alert generation
            timestamp=datetime.now()
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return excess_returns / volatility
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if excess_returns > 0 else 0.0
        
        downside_deviation = negative_returns.std() * np.sqrt(252)
        
        return excess_returns / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0 or max_drawdown == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        
        return annual_return / max_drawdown
    
    def _generate_comprehensive_alerts(
        self,
        risk_metrics: PortfolioRiskMetrics,
        correlation_results: Dict[str, Any],
        sector_results: Dict[str, Any],
        var_results: Dict[str, Any]
    ) -> List[RiskAlert]:
        """Generate comprehensive risk alerts."""
        alerts = []
        
        # VaR alerts
        if risk_metrics.portfolio_var > self.config['risk_limits']['max_portfolio_var']:
            alerts.append(RiskAlert(
                alert_type='var_breach',
                risk_level=RiskLevel.HIGH,
                asset=None,
                metric_name='portfolio_var',
                current_value=risk_metrics.portfolio_var,
                threshold_value=self.config['risk_limits']['max_portfolio_var'],
                message=f"Portfolio VaR ({risk_metrics.portfolio_var:.2%}) exceeds limit ({self.config['risk_limits']['max_portfolio_var']:.2%})",
                action_required=True,
                timestamp=datetime.now()
            ))
        
        # Correlation alerts
        if correlation_results['average_correlation'] > self.config['risk_limits']['max_correlation']:
            alerts.append(RiskAlert(
                alert_type='high_correlation',
                risk_level=RiskLevel.MEDIUM,
                asset=None,
                metric_name='average_correlation',
                current_value=correlation_results['average_correlation'],
                threshold_value=self.config['risk_limits']['max_correlation'],
                message=f"Average correlation ({correlation_results['average_correlation']:.2f}) is high",
                action_required=True,
                timestamp=datetime.now()
            ))
        
        # Add sector alerts
        alerts.extend(sector_results['alerts'])
        
        # Liquidity alerts
        if risk_metrics.liquidity_risk > (1 - self.config['risk_limits']['min_liquidity_ratio']):
            alerts.append(RiskAlert(
                alert_type='liquidity_risk',
                risk_level=RiskLevel.MEDIUM,
                asset=None,
                metric_name='liquidity_risk',
                current_value=risk_metrics.liquidity_risk,
                threshold_value=1 - self.config['risk_limits']['min_liquidity_ratio'],
                message=f"Portfolio liquidity risk is elevated",
                action_required=False,
                timestamp=datetime.now()
            ))
        
        return alerts
    
    def _update_alert_history(self, alerts: List[RiskAlert]) -> None:
        """Update alert history and manage duplicates."""
        current_time = datetime.now()
        
        # Filter out recent similar alerts (cooldown period)
        new_alerts = []
        for alert in alerts:
            is_duplicate = False
            
            for historical_alert in self.alert_history:
                time_diff = (current_time - historical_alert.timestamp).total_seconds()
                
                if (time_diff < self.config['alert_cooldown'] and
                    alert.alert_type == historical_alert.alert_type and
                    alert.asset == historical_alert.asset):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                new_alerts.append(alert)
        
        # Add new alerts to history
        self.alert_history.extend(new_alerts)
        
        # Trim history to last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Log new alerts
        for alert in new_alerts:
            self.logger.warning(f"Risk Alert: {alert.message}")
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for risk dashboard display."""
        if not hasattr(self, '_last_risk_metrics'):
            return {}
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if (datetime.now() - alert.timestamp).total_seconds() < 86400  # Last 24 hours
        ]
        
        return {
            'current_metrics': {
                'portfolio_var': self._last_risk_metrics.portfolio_var,
                'portfolio_volatility': self._last_risk_metrics.portfolio_volatility,
                'sharpe_ratio': self._last_risk_metrics.sharpe_ratio,
                'max_drawdown': self._last_risk_metrics.max_drawdown,
                'concentration_index': self._last_risk_metrics.concentration_index
            },
            'recent_alerts': [
                {
                    'type': alert.alert_type,
                    'level': alert.risk_level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in recent_alerts
            ],
            'alert_summary': {
                'total_alerts': len(recent_alerts),
                'high_risk_alerts': len([a for a in recent_alerts if a.risk_level == RiskLevel.HIGH]),
                'action_required': len([a for a in recent_alerts if a.action_required])
            }
        }