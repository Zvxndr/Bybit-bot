# AI Trading Bot Architecture & Goals

## 🎯 Core Mission
**Transform a small balance into $100K through AI-driven automated trading with dynamic risk management.**

## 🚀 Three-Phase Automated Pipeline System

### Phase 1: ML Strategy Discovery & Historical Backtesting
- **ML Engine**: `MLStrategyDiscoveryEngine` generates new strategies (3/hour configurable)
- **Historical Data**: Downloads up to 10 years of OHLCV data with sanitization
- **Backtest Engine**: `BybitEnhancedBacktestEngine` validates strategy performance
- **Graduation Criteria**: 
  - Minimum Sharpe ratio (configurable)
  - Maximum drawdown thresholds
  - Minimum return percentage
  - Trade count validation
- **Database Tracking**: `StrategyPipeline` model tracks all metrics and progression

### Phase 2: Paper Trading Validation (Bybit Testnet)
- **Real-time Testing**: Live market simulation without capital risk
- **Performance Monitoring**: Tracks paper P&L, trade count, success rates
- **Duration Requirements**: Minimum validation period before live consideration
- **Graduation Criteria**:
  - Minimum paper trading return (2%+)
  - Minimum validation period (14+ days)
  - Strategy stability metrics (0.8+ consistency)
  - Maximum paper drawdown limits (-5%)

### Phase 3: Live Trading Deployment (Bybit Live API)
- **Automated Promotion**: Strategies auto-graduate based on paper performance
- **Live Capital Allocation**: Dynamic allocation based on strategy confidence
- **Continuous Monitoring**: Real-time performance tracking with auto-retirement
- **Risk Integration**: ML Risk Manager controls position sizing and stops

## 🧠 ML-First Dynamic Risk System

### Core Dynamic Risk Algorithm
```python
# Small accounts: 2% risk (aggressive growth)
# Large accounts: 0.5% risk (wealth preservation)  
# Exponential decay between $10K-$100K transition
```

### Risk Components
- **MLRiskManager** (848 lines): Advanced ML-driven risk calculations
- **MLSelfAdjustingRiskManager** (731 lines): Self-optimizing risk parameters
- **UnifiedRiskManager**: Base risk framework with Australian tax optimization
- **Dynamic Scaling**: Balance-based risk adjustment (no hardcoded limits)

### ML Risk Features
- Portfolio risk assessment with correlation analysis
- Market regime detection and adaptation
- Volatility-adjusted position sizing
- Tax-optimized holding periods (CGT discount optimization)
- Real-time risk metric calculation and reporting

## 🏗️ Technical Architecture Details

### Database Schema
- **StrategyPipeline**: Complete strategy lifecycle tracking
- **Trade**: Comprehensive trade records with Australian tax fields
- **StrategyPerformance**: Time-series performance metrics
- **MarketData**: Historical and real-time market data storage
- **TaxLogEntry**: ATO-compliant transaction logging

### Core Components
- **AutomatedPipelineManager**: Orchestrates 3-phase progression system
- **StrategyGraduationManager**: Handles automated promotion/retirement logic
- **AustralianTimezoneManager**: NSW timezone and tax compliance
- **HistoricalDataManager**: Multi-exchange data collection and sanitization

### API Endpoints
```
/api/pipeline/strategies/{phase} - Get strategies by phase
/api/pipeline/metrics - Pipeline performance metrics
/api/ml-risk-metrics - Real-time ML risk engine data
/api/tax/export - Australian tax report generation
/api/graduation/* - Strategy graduation controls
```

## 💱 Exchange Integration Architecture

### Primary Trading Platform
- **Bybit**: Full integration (Testnet → Live progression)
- **Real-time WebSocket**: Market data and order execution
- **API Rate Limiting**: Intelligent request management
- **Error Handling**: Comprehensive connection monitoring

### Data-Only Sources (Optional)
- **OKX**: Additional market data for ML training
- **Binance**: Supplementary price feeds and volume data
- **Multi-source Validation**: Cross-exchange data verification

### Asset Management
- **USDT Pairs Only**: Simplified asset universe for focus
- **Dynamic Pair Selection**: ML-driven pair rotation based on opportunity
- **Liquidity Filtering**: Automatic filtering of low-liquidity pairs

## 🇦🇺 Australian Compliance System

### Tax Optimization Features
- **CGT Discount Optimization**: Automatic >365 day holding optimization
- **FIFO Cost Base Calculation**: ATO-compliant cost basis tracking
- **AUD Conversion**: Real-time RBA exchange rate integration
- **Financial Year Reporting**: June 30 year-end tax report generation
- **Private Use Optimization**: Tailored for NSW individual traders

### Compliance Tracking
- **Audit Trails**: Complete transaction history with timestamps
- **Tax Event Detection**: Automatic CGT event identification
- **Export Capabilities**: CSV/JSON export for accountant integration
- **Sydney Timezone**: All operations in Australian timezone

## 📊 User Interface & Controls

### Dashboard Features
- **Three-Column Pipeline View**: Backtest | Paper | Live strategy display
- **Real-time Performance Charts**: Separate paper/live performance tracking
- **ML Risk Engine Metrics**: Live confidence, risk scores, daily budgets
- **Emergency Controls**: Immediate stop/pause/start system-wide controls

### Monitoring Systems
- **WebSocket Updates**: Real-time frontend notifications
- **Strategy Naming Engine**: Unique ID generation (BTC_MR_A4F2D format)
- **Activity Logging**: Comprehensive debug logging (default enabled)
- **Email Reporting**: Optional weekly performance summaries

## 🏗 Deployment Architecture

### Production Environment
- **Platform**: DigitalOcean App Platform
- **Runtime**: `python -m src.main`
- **Database**: SQLite with automatic backups
- **Security**: Environment variable secrets management
- **Monitoring**: Built-in health checks and alerting

### Configuration Management
- **Environment-Specific**: Development/Staging/Production configs
- **Unified Configuration**: Single config system across all components
- **ML Parameter Storage**: Graduation/retirement criteria only (no risk limits)
- **Secret Management**: API keys and credentials via environment variables

## 🔮 Growth & Expansion Roadmap

### Phase 4: Post-$100K Expansion
1. **ASX Integration**: Australian stock and bond markets
2. **Arbitrage Engine**: Cross-exchange opportunity detection
3. **Advanced ML**: News sentiment analysis integration
4. **Institutional Features**: Enhanced reporting and compliance tools

### Scaling Architecture
- **Performance Optimization**: Speed Demon caching and optimization
- **Multi-Asset Support**: Preparation for stocks, bonds, commodities
- **Advanced Risk Models**: Machine learning risk parameter optimization
- **Regulatory Compliance**: Enhanced ATO reporting and audit capabilities

---

## 🎯 Key Architectural Principles

1. **AI-First Design**: ML algorithms determine all trading and risk decisions
2. **Australian Focus**: Built specifically for NSW private use compliance
3. **Growth Optimization**: Designed for aggressive small account scaling to $100K
4. **Automated Pipeline**: Minimal manual intervention in strategy lifecycle
5. **Tax Efficiency**: Optimized for Australian CGT and reporting requirements
6. **Clean Architecture**: Modular design supporting future expansion

*This system represents a complete end-to-end automated trading pipeline optimized for Australian private traders seeking aggressive account growth through AI-driven strategies.*