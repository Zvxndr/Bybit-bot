# Copilot Instructions: Production-Ready ML Trading Bot

## Project Overview
This is a production-ready, enterprise-grade algorithmic trading system that combines multiple data sources (cross-exchange, sentiment, on-chain) with advanced machine learning models for cryptocurrency trading on Bybit. The system has completed all 4 implementation phases and includes comprehensive infrastructure, monitoring, and deployment capabilities.

## Architecture Philosophy
- **Multi-Data Integration:** Combine exchange data, sentiment, and alternative data sources
- **Model Diversity:** Use different ML models that make uncorrelated errors
- **Data Quality First:** Robust monitoring and fallbacks for all external data
- **Uncertainty Awareness:** Prefer models that provide confidence estimates
- **Statistical First:** Every decision must be backed by statistical evidence
- **Database-Centric:** All actions, decisions, and results must be persisted for audit trails

## Code Style & Quality Guidelines

### General Principles
- Use Python 3.11+ features (pattern matching, typing extensions, zoneinfo)
- Follow PEP 8 with Black formatting (line length: 100)
- Type hints for all function signatures and major variables
- Descriptive variable names (avoid abbreviations unless widely accepted)
- Docstrings for all classes, methods, and functions using Google style
- Modular architecture with single responsibility principles

### Specific Patterns
```python
# Use dataclasses for configuration objects
@dataclass
class RiskParameters:
    portfolio_drawdown_limit: float
    strategy_drawdown_limit: float
    sharpe_ratio_min: float
    # ...

# Use context managers for resource handling
class DatabaseSession:
    def __enter__(self):
        return self.session
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

# Use abstract base classes for strategy interface
class TradingStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        pass
```

### Error Handling
- Use custom exceptions for domain-specific errors (`RiskLimitExceededError`, `StrategyValidationError`)
- Implement comprehensive logging with structured JSON format
- Use circuit breakers for exchange connectivity issues
- Always include context in error messages

## Database Schema & Models

### Core Tables
```python
# Key models for trading operations
class Trade(Base):
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    exchange = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # 'buy' or 'sell'
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    fee_currency = Column(String, nullable=False)
    strategy_id = Column(String, nullable=False)
    
    # Tax tracking fields (Australia specific)
    cost_base_aud = Column(Float)
    proceeds_aud = Column(Float)
    is_cgt_event = Column(Boolean, default=False)
    aud_conversion_rate = Column(Float)

class StrategyPerformance(Base):
    strategy_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    equity = Column(Float)
    drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    mode = Column(String)  # 'conservative' or 'aggressive'
    risk_parameters = Column(JSON)  # Snapshot of parameters used
```

## Data Implementation Guidelines

### Multi-Exchange Data (`src/bot/data/multi_exchange_fetcher.py`)
```python
class MultiExchangeDataFetcher:
    def __init__(self, config):
        self.exchanges = {
            'bybit': ccxt.bybit(config['bybit']),
            'binance': ccxt.binance(config['binance']),
            'okx': ccxt.okx(config['okx'])
        }
    
    def fetch_cross_exchange_features(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """
        Fetch data from multiple exchanges and calculate comparative features.
        """
        features = {}
        data = {}
        
        # Fetch data from all exchanges in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                exchange: executor.submit(
                    self.exchanges[exchange].fetch_ohlcv, symbol, timeframe
                )
                for exchange in self.exchanges
            }
            for exchange, future in futures.items():
                try:
                    data[exchange] = self._process_ohlcv(future.result())
                except Exception as e:
                    logging.warning(f"Failed to fetch from {exchange}: {e}")
                    data[exchange] = None
        
        # Calculate cross-exchange features
        if all(data.values()):
            features['volume_ratio_binance_bybit'] = (
                data['binance']['volume'] / data['bybit']['volume']
            )
            features['price_discrepancy_okx_bybit'] = (
                data['okx']['close'] / data['bybit']['close'] - 1
            )
        
        return features
```

### Sentiment Data Integration (`src/bot/data/sentiment_fetcher.py`)
```python
class SentimentDataFetcher:
    def __init__(self, config):
        self.cryptopanic_api_key = config['cryptopanic']['api_key']
        self.fear_greed_url = "https://api.alternative.me/fng/"
    
    def fetch_sentiment_features(self) -> Dict[str, float]:
        """
        Fetch sentiment data from various sources.
        """
        features = {}
        
        # Fetch from CryptoPanic
        try:
            response = requests.get(
                f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_api_key}&currencies=BTC"
            )
            news_data = response.json()
            features['news_sentiment_score'] = self._calculate_sentiment_score(news_data)
            features['news_volume_24h'] = len(news_data['results'])
        except Exception as e:
            logging.warning(f"Failed to fetch CryptoPanic data: {e}")
        
        # Fetch Fear & Greed Index
        try:
            response = requests.get(self.fear_greed_url)
            fgi_data = response.json()
            features['fear_greed_index'] = float(fgi_data['data'][0]['value'])
            features['fear_greed_classification'] = fgi_data['data'][0]['value_classification']
        except Exception as e:
            logging.warning(f"Failed to fetch Fear & Greed Index: {e}")
        
        return features
```

## Machine Learning Implementation

### Model Diversity Strategy
Implement multiple model types that capture different patterns:

1. **LightGBM/XGBoost:** For tabular feature-based predictions
2. **Temporal Convolutional Networks:** For multivariate time series patterns
3. **Transformers:** For complex long-range dependencies in sequential data
4. **Gaussian Processes:** For uncertainty-aware predictions

### Example Multi-Model Setup (`src/bot/ml_engine/model_factory.py`)
```python
class ModelFactory:
    def create_model(self, model_type: str, config: Dict) -> BaseModel:
        if model_type == 'lightgbm':
            return LightGBMModel(config)
        elif model_type == 'tcn':
            return TCNModel(config)
        elif model_type == 'transformer':
            return TransformerModel(config)
        elif model_type == 'gaussian_process':
            return GaussianProcessModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class BaseModel(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Tuple[float, float]:
        """Return prediction and confidence estimate"""
        pass
```

## Risk Management with Sentiment Integration

### Sentiment-Aware Risk Management (`src/bot/risk/sentiment_risk.py`)
```python
class SentimentAwareRiskManager:
    def adjust_risk_parameters(self, sentiment_data: Dict, base_params: Dict) -> Dict:
        """
        Adjust risk parameters based on market sentiment.
        """
        adjusted_params = base_params.copy()
        
        # Adjust based on Fear & Greed Index
        fgi = sentiment_data.get('fear_greed_index', 50)
        if fgi > 75:  # Extreme greed
            adjusted_params['risk_per_trade'] *= 0.5  # Reduce position size
            adjusted_params['max_drawdown_limit'] *= 0.8  # Tighter drawdown
        elif fgi < 25:  # Extreme fear
            adjusted_params['risk_per_trade'] *= 1.2  # Increase position size cautiously
            adjusted_params['var_daily_limit'] *= 1.2  # Slightly higher daily risk
        
        # Adjust based on news sentiment
        news_sentiment = sentiment_data.get('news_sentiment_score', 0)
        if news_sentiment < -0.5:  # Very negative news
            adjusted_params['risk_per_trade'] *= 0.7
        elif news_sentiment > 0.5:  # Very positive news
            adjusted_params['risk_per_trade'] *= 0.8
        
        return adjusted_params
```

## Data Quality & Monitoring

### API Health Monitoring (`src/bot/monitoring/api_health.py`)
```python
class APIHealthMonitor:
    def __init__(self):
        self.api_status = {
            'bybit': {'status': 'healthy', 'last_check': None},
            'binance': {'status': 'healthy', 'last_check': None},
            'cryptopanic': {'status': 'healthy', 'last_check': None}
        }
    
    def check_api_health(self):
        """Regularly check all API endpoints"""
        for api_name in self.api_status:
            try:
                response_time = self._ping_api(api_name)
                self.api_status[api_name] = {
                    'status': 'healthy',
                    'response_time': response_time,
                    'last_check': datetime.now()
                }
            except Exception as e:
                self.api_status[api_name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'last_check': datetime.now()
                }
                logging.error(f"API {api_name} is unhealthy: {e}")
    
    def get_fallback_strategy(self, api_name: str) -> Dict:
        """Get fallback strategy for unhealthy APIs"""
        fallbacks = {
            'binance': {'use': 'okx', 'weight': 0.7},
            'cryptopanic': {'use': 'fear_greed_only', 'weight': 0.5}
        }
        return fallbacks.get(api_name, {})
```

## ✅ Production Implementation Status

### ✅ Completed Phases
1. **✅ Multi-Exchange Data Pipeline:** Complete cross-exchange features with Bybit, Binance, OKX integration
2. **✅ Sentiment Integration:** Fear & Greed Index, CryptoPanic API, and social sentiment fully integrated
3. **✅ Advanced ML Models:** LightGBM, XGBoost, Neural Networks, and Transformer models deployed
4. **✅ Production Infrastructure:** FastAPI service, Streamlit dashboard, Kubernetes deployment, CI/CD pipelines

### 🏗️ Current Architecture
- **Production API**: FastAPI with WebSocket streaming, JWT authentication, rate limiting
- **Monitoring Dashboard**: Real-time Streamlit dashboard with performance analytics
- **Container Orchestration**: Kubernetes deployment with auto-scaling and health checks
- **CI/CD Pipeline**: Automated testing, security scanning, and deployment
- **Configuration Management**: Environment-based configuration with encrypted secrets

## Testing Strategy

1. **Data Quality Tests:** Verify all API endpoints and data freshness
2. **Cross-Validation with Multiple Data Sources:** Ensure models work with partial data
3. **Fallback Testing:** Test all failure scenarios and fallback mechanisms
4. **Sentiment Impact Analysis:** Measure how sentiment features affect model performance

## Monitoring & Alerting

1. **API Health Dashboard:** Monitor all external data sources
2. **Feature Importance Tracking:** Monitor which features are driving predictions
3. **Model Performance Comparison:** Track which models perform best in which conditions
4. **Sentiment-Risk Correlation:** Monitor how sentiment affects risk adjustments

### Tax Calculation (Australia Specific) (src/bot/tax/cgt_calculator.py)
- Use FIFO method for CGT calculations (required by ATO)
- Integrate with RBA API for historical AUD/USD rates
- Calculate CGT events on position closure
- Generate comprehensive reports for financial year
- Track separate performance for aggressive vs conservative modes

## Configuration Management

### config/config.yaml Structure
```yaml
trading:
  mode: aggressive
  base_balance: 10000
  max_risk_ratio: 0.02
  min_risk_ratio: 0.005
  balance_thresholds:
    low: 10000
    high: 100000
  risk_decay: exponential

exchange:
  name: bybit
  sandbox: true
  timeframe: 1h
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT]

database:
  dialect: postgresql
  host: localhost
  port: 5432
  name: trading_bot
```

## Testing Requirements

### Unit Tests
- Test all utility functions and data processing
- Verify statistical calculations (Sharpe ratio, drawdown, etc.)
- Test database models and queries
- Verify configuration loading and validation

### Integration Tests
- Test strategy validation pipeline
- Verify exchange API interactions (use sandbox)
- Test risk management rules enforcement
- Validate tax calculation logic

### Performance Tests
- Benchmark backtesting engine speed
- Test ML model training performance
- Verify database query performance
- Stress test under high volatility scenarios

## Deployment & Monitoring

### Docker Configuration
- Use multi-stage builds for optimized image size
- Include health checks in Docker configuration
- Set up proper volume mounting for database persistence
- Configure resource limits (CPU, memory)

### Monitoring
- Implement structured JSON logging
- Create performance dashboards with Streamlit
- Set up alerting for critical events
- Monitor system health and resource usage

## Important Development Notes

### Strategy Development
1. **Always validate** with multiple testing methodologies (WFO, permutation, CSCV)
2. **Start simple** with technical strategies before implementing ML
3. **Test thoroughly** in paper trading before live deployment
4. **Monitor continuously** for regime changes and performance decay

### Risk Management
1. **Never override** risk limits programmatically
2. **Always preserve** capital as the primary objective
3. **Implement circuit breakers** for extreme market conditions
4. **Maintain audit trails** of all risk decisions

### Tax Compliance
1. **Record everything** required for Australian tax reporting
2. **Use official rates** from RBA for AUD conversions
3. **Generate regular reports** for compliance tracking
4. **Separate tracking** for different trading modes

## Common Patterns & Anti-Patterns

### Do This:
```python
# Use context managers for database sessions
with DatabaseSession() as session:
    result = session.query(Trade).filter_by(strategy_id=strategy_id).all()

## 🚀 Production-Ready System Guidelines

### Current System Architecture
The system is now production-ready with enterprise-grade infrastructure:

```
Production Stack:
├── FastAPI Prediction Service (src/bot/api/)
├── Streamlit Monitoring Dashboard (src/bot/dashboard/)
├── Kubernetes Deployment (k8s/)
├── CI/CD Pipelines (.github/workflows/)
├── Configuration Management (config/)
└── Health Monitoring & Alerting (scripts/)
```

### Production Development Guidelines
1. **Maintain Production Standards**: All code must meet enterprise-grade standards
2. **Security First**: Encrypted secrets, JWT authentication, secure API endpoints
3. **Monitoring Required**: All new features must include health checks and metrics
4. **Container Ready**: New services must be containerized and Kubernetes-ready
5. **CI/CD Integration**: All changes must pass automated testing and security scanning

### Production Deployment Commands
```bash
# Health Check
python scripts/health_check.py --environment production

# Deploy to Kubernetes
python deploy.py deploy --environment production

# Monitor System Health
kubectl get pods -l app=trading-bot
kubectl logs -f deployment/trading-bot-api
```

### Production Monitoring
- **API Health**: `/health` endpoint with comprehensive system checks
- **Performance Metrics**: Prometheus metrics integration
- **Real-time Dashboard**: Streamlit dashboard at port 8501
- **Grafana Monitoring**: Advanced metrics visualization

This system is now production-ready with enterprise-grade infrastructure, comprehensive monitoring, and automated deployment capabilities. All further development should maintain these production standards while adding new features and capabilities.