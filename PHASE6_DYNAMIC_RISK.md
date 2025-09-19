# Phase 6: Dynamic Risk Management System

## Overview

The Dynamic Risk Management System provides sophisticated, adaptive risk management that automatically adjusts to changing market conditions. This system combines advanced volatility monitoring, correlation analysis, and dynamic hedging to provide institutional-quality risk management.

## Key Features

### 🎯 Adaptive Volatility Monitoring
- **Multi-Method Volatility Estimation**: GARCH, EWMA, Realized, Parkinson, Garman-Klass
- **Volatility Regime Detection**: Automatic classification (Very Low, Low, Normal, High, Very High, Extreme)
- **Real-Time Forecasting**: 1-hour, 4-hour, and 24-hour volatility forecasts
- **Trend Analysis**: Volatility trend detection with strength measurement

### 📊 Dynamic Correlation Analysis
- **Real-Time Correlation Matrices**: Multi-timeframe correlation calculation
- **Regime Detection**: Low, Normal, High, and Crisis correlation regimes
- **Portfolio Analysis**: Diversification metrics, effective asset count, concentration risk
- **Cross-Asset Analysis**: Correlation clustering and principal component analysis

### 🛡️ Dynamic Hedging System
- **Multiple Hedge Types**: Delta, Beta, Correlation, Volatility, Pairs, Cross-Asset
- **Optimal Hedge Ratios**: Minimum variance, beta regression, correlation-based
- **Automatic Rebalancing**: Real-time effectiveness monitoring and rebalancing
- **Cost-Aware Hedging**: Hedge cost optimization and PnL attribution

### ⚖️ Integrated Risk Management
- **Risk Regime Detection**: Combined volatility and correlation regime classification
- **Adaptive Position Sizing**: Automatic position adjustments based on market conditions
- **Real-Time Monitoring**: Continuous risk assessment and alert systems
- **Performance Attribution**: Comprehensive risk and hedging performance tracking

## Architecture

```
Dynamic Risk System
├── Volatility Monitoring
│   ├── AdaptiveVolatilityMonitor
│   ├── VolatilityEstimator (GARCH, EWMA, etc.)
│   └── VolatilityRegimeDetector
├── Correlation Analysis
│   ├── DynamicCorrelationAnalyzer
│   ├── DynamicCorrelationCalculator
│   └── PortfolioCorrelationAnalyzer
├── Dynamic Hedging
│   ├── DynamicHedgingSystem
│   ├── HedgeRatioCalculator
│   └── HedgePositionManager
└── Risk Integration
    ├── RiskRegimeDetector
    ├── RiskAdjustmentCalculator
    └── DynamicRiskSystem (Main)
```

## Core Components

### 1. Volatility Monitoring

**VolatilityEstimator**: Multi-method volatility calculation
- Realized volatility (multiple timeframes)
- GARCH(1,1) model with maximum likelihood estimation
- EWMA with configurable decay factors
- Parkinson and Garman-Klass OHLC estimators
- Downside/upside volatility separation

**VolatilityRegimeDetector**: Regime classification and transitions
- Historical percentile-based regime detection
- Minimum duration constraints to prevent rapid switching
- Trend analysis with statistical significance testing
- Confidence scoring for regime classifications

### 2. Correlation Analysis

**DynamicCorrelationCalculator**: Multiple correlation methodologies
- Pearson, Spearman, and Kendall correlations
- Tail correlation during extreme market moves
- Dynamic conditional correlation (DCC-GARCH style)
- Beta coefficients and tracking error calculation

**PortfolioCorrelationAnalyzer**: Portfolio-wide analysis
- Correlation matrix eigenvalue decomposition
- Hierarchical clustering for asset grouping
- Principal component analysis for factor exposure
- Diversification ratio and effective asset calculation

### 3. Dynamic Hedging

**HedgeRatioCalculator**: Optimal hedge ratio calculation
- Minimum variance hedge ratios
- Beta-based hedge ratios with regression analysis
- Correlation and volatility-adjusted ratios
- Dynamic rolling window estimation

**HedgePositionManager**: Hedge lifecycle management
- Automated hedge creation and sizing
- Real-time effectiveness monitoring
- Intelligent rebalancing with multiple signal types
- Position tracking and performance attribution

### 4. Risk Integration

**RiskRegimeDetector**: Overall portfolio risk assessment
- Combined volatility and correlation regime scoring
- Regime transition probability estimation
- Smoothed regime changes with minimum duration
- Multi-factor risk classification

**RiskAdjustmentCalculator**: Position sizing adjustments
- Volatility-based position scaling
- Correlation penalty adjustments
- Regime-specific multipliers
- Comprehensive risk adjustment recommendations

## Usage Examples

### Basic Setup

```python
from bot.dynamic_risk import DynamicRiskSystem

# Configure the risk system
config = {
    'portfolio_symbols': ['BTC', 'ETH', 'ADA', 'SOL'],
    'monitoring_interval': 300,  # 5 minutes
    'auto_hedge_enabled': True,
    'auto_adjust_enabled': True
}

# Create risk system
risk_system = DynamicRiskSystem(config)

# Setup callbacks for risk events
def risk_callback(metrics):
    print(f"Risk regime: {metrics.risk_regime.value}")
    print(f"Portfolio volatility: {metrics.portfolio_volatility:.3f}")

risk_system.add_risk_callback(risk_callback)

# Start monitoring
risk_system.start_monitoring()
```

### Adding Market Data

```python
# Feed real-time market data
risk_system.add_market_data(
    symbol='BTC',
    price=45000.0,
    position_size=0.5,  # Your position size
    open_price=44800.0,
    high=45200.0,
    low=44600.0,
    volume=1000.0
)
```

### Manual Risk Assessment

```python
# Calculate portfolio risk metrics
symbols = ['BTC', 'ETH', 'ADA', 'SOL']
positions = {'BTC': 0.5, 'ETH': 2.0, 'ADA': 1000, 'SOL': 10}

risk_metrics = risk_system.calculate_portfolio_risk_metrics(symbols, positions)

if risk_metrics:
    print(f"Risk Regime: {risk_metrics.risk_regime.value}")
    print(f"Portfolio Volatility: {risk_metrics.portfolio_volatility:.3f}")
    print(f"Average Correlation: {risk_metrics.average_correlation:.3f}")
    print(f"Effective Positions: {risk_metrics.effective_positions:.1f}")
```

### Manual Hedging

```python
# Create hedge for a specific position
hedge_id = risk_system.hedging_system.create_hedge(
    primary_symbol='BTC',
    hedge_symbols=['ETH', 'ADA'],  # Hedge instruments
    hedge_type=HedgeType.BETA_HEDGE,
    hedge_method='minimum_variance'
)

if hedge_id:
    print(f"Created hedge: {hedge_id}")
    
    # Monitor hedge performance
    hedge_position = risk_system.hedging_system.get_hedge_position(hedge_id)
    print(f"Hedge effectiveness: {hedge_position.effectiveness:.3f}")
```

## Configuration

### Volatility Monitor Configuration

```python
volatility_config = {
    'update_interval': 300,
    'min_observations': 100,
    'garch_update_freq': 3600,
    'ewma_lambda': 0.94,
    'annualization_factor': 365.25 * 24,
    'outlier_threshold': 5.0,
    'regime_lookback': 2160
}
```

### Correlation Analyzer Configuration

```python
correlation_config = {
    'update_interval': 300,
    'min_observations': 30,
    'tail_threshold': 0.05,
    'correlation_threshold': 0.3,
    'cluster_method': 'ward',
    'n_clusters': 3
}
```

### Hedging System Configuration

```python
hedging_config = {
    'monitoring_interval': 300,
    'max_hedge_positions': 10,
    'effectiveness_threshold': 0.6,
    'rebalance_threshold': 0.15,
    'max_hedge_ratio': 2.0,
    'min_position_size': 100
}
```

## Risk Metrics

### Portfolio Risk Metrics

- **Risk Regime**: Current portfolio risk classification
- **Portfolio Volatility**: Overall portfolio volatility estimate
- **Correlation Metrics**: Average and maximum correlations
- **Diversification**: Effective number of independent positions
- **Hedge Effectiveness**: Overall hedging performance

### Individual Asset Adjustments

- **Volatility Scalar**: Position size adjustment for volatility
- **Correlation Scalar**: Adjustment for correlation regime
- **Total Adjustment**: Combined position size multiplier
- **Adaptation Signal**: Recommended action (reduce, increase, hedge, rebalance)

## Risk Regimes

### Volatility Regimes
- **Very Low**: <10th percentile of historical volatility
- **Low**: 10th-25th percentile
- **Normal**: 25th-75th percentile  
- **High**: 75th-90th percentile
- **Very High**: 90th-99th percentile
- **Extreme**: >99th percentile

### Correlation Regimes
- **Low Correlation**: <25th percentile of historical correlations
- **Normal Correlation**: 25th-75th percentile
- **High Correlation**: 75th-95th percentile
- **Crisis Correlation**: >95th percentile or absolute correlation >0.8

### Overall Risk Regimes
- **Low Risk**: Low volatility, normal correlations
- **Normal Risk**: Normal volatility and correlations
- **High Risk**: High volatility or high correlations
- **Crisis Risk**: Extreme volatility and high correlations

## Database Schema

The system persists data in SQLite databases:

### Volatility Data
- `volatility_metrics`: Time series of volatility metrics
- Indexes on symbol and timestamp

### Correlation Data  
- `correlation_metrics`: Pairwise correlation metrics
- `portfolio_metrics`: Portfolio-wide correlation analysis
- Indexes for efficient querying

### Hedge Data
- `hedge_positions`: Hedge position details and status
- `hedge_performance`: Time series hedge performance data
- Comprehensive hedge tracking

### Risk Data
- `risk_metrics`: Portfolio risk assessments
- `risk_adjustments`: Individual asset risk adjustments
- Full audit trail of risk decisions

## Performance Characteristics

### Computational Efficiency
- **Real-time Processing**: Sub-second risk calculations
- **Memory Efficient**: Rolling windows with configurable limits
- **Scalable**: Handles 100+ assets efficiently

### Statistical Robustness
- **Multiple Estimators**: Reduces model risk through diversification
- **Confidence Scoring**: All estimates include confidence measures
- **Outlier Handling**: Robust statistical methods

### Practical Features
- **Missing Data Handling**: Graceful degradation with incomplete data
- **Regime Smoothing**: Prevents excessive regime switching
- **Cost Awareness**: Considers transaction costs in recommendations

## Integration Points

The Dynamic Risk System integrates with:

- **Phase 4 Risk Management**: Enhances static risk controls with dynamic adaptation
- **Phase 5 Execution Engine**: Provides risk-adjusted execution parameters
- **Database Layer**: Persistent storage of all risk metrics and decisions
- **Logging System**: Comprehensive audit trail of risk events

## Testing and Validation

Comprehensive testing includes:

- **Unit Tests**: All components individually tested
- **Integration Tests**: End-to-end system testing
- **Market Simulation**: Realistic market condition testing
- **Stress Testing**: Extreme market scenario validation
- **Performance Testing**: Computational efficiency validation

## Future Enhancements

Planned improvements include:

- **Alternative Risk Models**: VaR, Expected Shortfall, Coherent Risk Measures
- **Machine Learning Integration**: ML-enhanced regime detection
- **Multi-Asset Class Support**: Traditional assets, derivatives, options
- **Advanced Hedging**: Options strategies, exotic derivatives
- **Real-time Optimization**: Continuous portfolio optimization

---

## Complete Implementation

Phase 6 delivers a production-ready dynamic risk management system with:

✅ **Adaptive Volatility Monitoring** - Multi-method volatility estimation with regime detection  
✅ **Dynamic Correlation analysis** - Real-time correlation monitoring and portfolio analysis  
✅ **Automated Dynamic Hedging** - Intelligent hedge creation, management, and rebalancing  
✅ **Integrated Risk Assessment** - Combined risk regime detection and position adjustments  
✅ **Real-time Monitoring** - Continuous risk assessment with alerting and callbacks  
✅ **Comprehensive Testing** - Full test coverage with realistic market simulations  
✅ **Database Integration** - Persistent storage with efficient querying  
✅ **Production Ready** - Robust error handling, logging, and performance optimization  

The system provides institutional-quality dynamic risk management that automatically adapts to changing market conditions, protecting portfolios during volatile periods while maximizing returns during stable conditions.