# 📡 **API REFERENCE DOCUMENTATION**
**Backend Server:** `src/frontend_server.py`  
**Total Endpoints:** 27 (23 original + 4 email system)  
**Framework:** Flask with comprehensive error handling  
**Status:** All endpoints operational and tested

---

## 📊 **API ENDPOINT OVERVIEW**

### **System APIs** (6 endpoints)
| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/api/status` | GET | System health check | ✅ Active |
| `/api/system-info` | GET | System information | ✅ Active |
| `/api/system-overview` | GET | Dashboard overview data | ✅ Active |
| `/api/debug-safety` | GET | Safety system status | ✅ Active |
| `/api/config` | GET | Configuration data | ✅ Active |
| `/api/logs` | GET | System log access | ✅ Active |

### **Trading APIs** (5 endpoints) 
| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/api/trading/balance` | GET | Account balance (mock) | ✅ Active (Safe) |
| `/api/trading/positions` | GET | Open positions (mock) | ✅ Active (Safe) |
| `/api/trading/orders` | GET | Order history (mock) | ✅ Active (Safe) |
| `/api/trading/performance` | GET | Trading performance | ✅ Active (Safe) |
| `/api/trading/place-order` | POST | Place order (BLOCKED) | 🛡️ Safely Blocked |

### **Market Data APIs** (4 endpoints)
| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/api/market/prices` | GET | Current market prices | ✅ Active |
| `/api/market/historical` | GET | Historical price data | ✅ Active |
| `/api/market/analysis` | GET | Technical analysis | ✅ Active |
| `/api/market/indicators` | GET | Technical indicators | ✅ Active |

### **AI Strategy APIs** (3 endpoints)
| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/api/ai/strategies` | GET | Available strategies | ✅ Active |
| `/api/ai/backtest` | POST | Run backtesting (5-year max) | ✅ Active |
| `/api/ai/performance` | GET | Strategy performance | ✅ Active |

### **Risk Management APIs** (3 endpoints)
| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/api/risk/portfolio` | GET | Portfolio risk metrics | ✅ Active |
| `/api/risk/position-sizing` | POST | Calculate position size | ✅ Active |
| `/api/risk/limits` | GET | Risk limit configuration | ✅ Active |

### **Portfolio APIs** (2 endpoints)
| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/api/portfolio/holdings` | GET | Current holdings | ✅ Active |
| `/api/portfolio/allocation` | GET | Asset allocation | ✅ Active |

### **Email Notification APIs** (4 endpoints) 🆕 **NEW**
| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/api/email/test-config` | POST | Test email configuration | ✅ Active |
| `/api/email/send-test` | POST | Send test email | ✅ Active |
| `/api/email/daily-report` | POST | Generate daily report | ✅ Active |
| `/api/email/status` | GET | Email system status | ✅ Active |

---

## 🔧 **API IMPLEMENTATION DETAILS**

### **System Status API** - `/api/status`
```python
@app.route('/api/status')
def get_system_status():
    """Comprehensive system health check"""
    return {
        'status': 'operational',
        'timestamp': datetime.now().isoformat(),
        'debug_mode': True,  # Always True in private mode
        'safety_systems': {
            'trading_blocked': True,
            'testnet_only': True,
            'ultra_safe_mode': True
        },
        'api_connections': {
            'bybit': check_bybit_connection(),
            'websocket': check_websocket_connection(),
            'market_data': check_market_data_connection(),
            'trading': False,  # Always False (blocked)
            'risk_manager': check_risk_manager_connection(),
            'email': check_email_connection()
        }
    }
```

### **AI Strategy Backtesting API** - `/api/ai/backtest` 🆕 **ENHANCED**
```python
@app.route('/api/ai/backtest', methods=['POST'])
def run_backtest():
    """Enhanced backtesting with 5-year maximum period"""
    data = request.json
    
    # Validate timeframe (max 5 years for private use)
    timeframe = data.get('timeframe', '30d')
    max_days = {
        '30d': 30,
        '3m': 90,
        '6m': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825  # 5 years maximum
    }
    
    if timeframe not in max_days:
        return {'error': 'Invalid timeframe'}, 400
        
    # Run safe backtesting (no real money involved)
    results = run_safe_backtest(
        strategy=data.get('strategy'),
        timeframe=timeframe,
        risk_params=get_ultra_safe_params()
    )
    
    return {
        'backtest_results': results,
        'timeframe': timeframe,
        'safety_note': 'Backtesting only - no real trading'
    }
```

### **Email Configuration Test API** - `/api/email/test-config` 🆕 **NEW**
```python
const ws = new WebSocket('ws://localhost:8000/ws/predictions?token=eyJ0eXAiOiJKV1QiLCJh...');
```

## Rate Limits

| User Type | Rate Limit | Burst Limit |
|-----------|------------|-------------|
| Unauthenticated | 100/hour | 10/minute |
| Authenticated | 1000/hour | 50/minute |
| Admin | 5000/hour | 100/minute |

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642694400
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Rate Limited |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request was invalid",
    "details": {
      "field": "parameter_name",
      "issue": "missing required field"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123abc"
  }
}
```

## 🤖 Core Prediction Endpoints

### Real-time ML Prediction

Get real-time trading predictions from ensemble ML models.

**Endpoint**: `POST /predict`  
**Authentication**: JWT token required  
**Rate Limit**: 100/minute

#### Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Authorization: Bearer your_jwt_token" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "BTCUSDT",
       "timeframe": "1h",
       "features": {
         "rsi": 65.2,
         "macd": 0.015,
         "volume_ratio": 1.2,
         "sentiment_score": 0.3
       }
     }'
```

#### Response

```json
{
  "prediction": {
    "signal": "buy",
    "confidence": 0.85,
    "expected_return": 0.023,
    "risk_score": 0.15,
    "model_ensemble": {
      "lightgbm": {"prediction": 0.025, "weight": 0.4},
      "xgboost": {"prediction": 0.021, "weight": 0.3},
      "neural_network": {"prediction": 0.024, "weight": 0.3}
    },
    "feature_importance": {
      "sentiment_score": 0.35,
      "rsi": 0.25,
      "volume_ratio": 0.20,
      "macd": 0.20
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "model_version": "v1.2.3",
    "processing_time_ms": 45,
    "request_id": "pred_123abc"
  }
}
```

### Batch Predictions

Process multiple predictions in a single request for efficiency.

**Endpoint**: `POST /predict/batch`  
**Authentication**: JWT token required  
**Rate Limit**: 10/minute

#### Request

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Authorization: Bearer your_jwt_token" \
     -H "Content-Type: application/json" \
     -d '{
       "requests": [
         {
           "symbol": "BTCUSDT",
           "timeframe": "1h",
           "features": {"rsi": 65.2, "macd": 0.015}
         },
         {
           "symbol": "ETHUSDT", 
           "timeframe": "1h",
           "features": {"rsi": 72.1, "macd": -0.008}
         }
       ]
     }'
```

#### Response

```json
{
  "predictions": [
    {
      "symbol": "BTCUSDT",
      "prediction": {
        "signal": "buy",
        "confidence": 0.85,
        "expected_return": 0.023
      }
    },
    {
      "symbol": "ETHUSDT",
      "prediction": {
        "signal": "sell", 
        "confidence": 0.78,
        "expected_return": -0.018
      }
    }
  ],
  "metadata": {
    "total_requests": 2,
    "successful": 2,
    "failed": 0,
    "processing_time_ms": 127
  }
}
```

### WebSocket Streaming Predictions

Subscribe to real-time prediction streams via WebSocket.

**Endpoint**: `WS /ws/predictions`  
**Authentication**: JWT token in query parameter  
**Rate Limit**: 1 connection per user

#### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/predictions?token=your_jwt_token');

ws.onopen = function(event) {
    // Subscribe to specific symbols
    ws.send(JSON.stringify({
        "action": "subscribe",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "timeframe": "1m"
    }));
};

ws.onmessage = function(event) {
    const prediction = JSON.parse(event.data);
    console.log('New prediction:', prediction);
};
```

#### Message Format

```json
{
  "type": "prediction",
  "symbol": "BTCUSDT",
  "prediction": {
    "signal": "buy",
    "confidence": 0.82,
    "expected_return": 0.019,
    "timestamp": "2024-01-15T10:30:15Z"
  },
  "market_data": {
    "price": 42500.50,
    "volume": 1250000,
    "change_24h": 0.025
  }
}
```

## 🏥 System Health Endpoints

### Comprehensive Health Check

Check detailed system health including all components.

**Endpoint**: `GET /health`  
**Authentication**: None required  
**Rate Limit**: 1000/hour

#### Request

```bash
curl "http://localhost:8000/health"
```

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "api": {"status": "healthy", "response_time_ms": 12},
    "database": {"status": "healthy", "connection_pool": "8/20"},
    "ml_models": {"status": "healthy", "loaded_models": 4},
    "exchange_api": {"status": "healthy", "latency_ms": 45},
    "redis_cache": {"status": "healthy", "memory_usage": "45%"},
    "prediction_service": {"status": "healthy", "queue_size": 3}
  },
  "metrics": {
    "predictions_per_minute": 25.3,
    "api_requests_per_minute": 120.5,
    "system_cpu_percent": 35.8,
    "system_memory_percent": 62.4
  }
}
```

### System Information

Get comprehensive system information and configuration.

**Endpoint**: `GET /info`  
**Authentication**: JWT token required  
**Rate Limit**: 100/hour

#### Request

```bash
curl -H "Authorization: Bearer your_jwt_token" \
     "http://localhost:8000/info"
```

#### Response

```json
{
  "system": {
    "name": "ML Trading Bot",
    "version": "1.0.0",
    "environment": "production",
    "uptime_seconds": 86400.5,
    "start_time": "2024-01-14T10:30:00Z"
  },
  "api": {
    "version": "v1",
    "docs_url": "/docs",
    "openapi_url": "/openapi.json"
  },
  "ml_models": {
    "ensemble_version": "v1.2.3",
    "models_loaded": 4,
    "last_training": "2024-01-15T08:00:00Z",
    "model_types": ["lightgbm", "xgboost", "neural_network", "transformer"]
  },
  "data_sources": {
    "exchanges": ["bybit", "binance", "okx"],
    "sentiment": ["cryptopanic", "fear_greed_index"],
    "features": 127
  }
}
```

## 📊 Metrics & Monitoring

### Prometheus Metrics

Get Prometheus-formatted metrics for monitoring integration.

**Endpoint**: `GET /metrics`  
**Authentication**: None required (internal endpoint)  
**Rate Limit**: 1000/hour

#### Request

```bash
curl "http://localhost:8000/metrics"
```

#### Response (Prometheus Format)

```
# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total{model="ensemble"} 15420

# HELP prediction_latency_seconds Time taken to generate predictions
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.01"} 1245
prediction_latency_seconds_bucket{le="0.05"} 12890
prediction_latency_seconds_bucket{le="0.1"} 15420

# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="200"} 12458

# HELP model_accuracy Current model accuracy
# TYPE model_accuracy gauge
model_accuracy{model="lightgbm"} 0.847
model_accuracy{model="xgboost"} 0.832
model_accuracy{model="neural_network"} 0.856
```

### System Metrics

Get detailed system metrics and performance data.

**Endpoint**: `GET /metrics`  
**Authentication**: Read-only permission required  
**Rate Limit**: 100/hour

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hours | integer | 1 | Historical data period |

#### Request

```bash
curl -H "Authorization: Bearer your_token" \
     "https://api.tradingbot.com/metrics?hours=24"
```

#### Response

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": {
    "system_cpu_percent": 45.2,
    "system_memory_percent": 62.1,
    "system_disk_percent": 23.8,
    "trading_portfolio_value": 10000.50,
    "trading_open_positions": 3,
    "trading_daily_pnl": 150.75,
    "api_requests_per_minute": 25.3
  },
  "statistics": {
    "system_cpu_percent": {
      "avg": 42.1,
      "min": 15.3,
      "max": 78.9,
      "p50": 41.2,
      "p95": 65.4,
      "p99": 72.8
    },
    "trading_portfolio_value": {
      "avg": 9850.25,
      "min": 9500.00,
      "max": 10500.00,
      "p50": 9825.50,
      "p95": 10200.00,
      "p99": 10350.00
    }
  }
}
```

## 🔐 Authentication Endpoints

### User Login

Authenticate user and receive JWT access token.

**Endpoint**: `POST /auth/login`  
**Authentication**: None required  
**Rate Limit**: 10/minute per IP

#### Request

```bash
curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "your_username",
       "password": "your_password"
     }'
```

#### Response

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user_123",
    "username": "your_username",
    "role": "trader",
    "permissions": ["predict", "monitor", "models"]
  }
}
```

### Refresh Access Token

Get a new access token using refresh token.

**Endpoint**: `POST /auth/refresh`  
**Authentication**: Valid refresh token required  
**Rate Limit**: 100/hour

#### Request

```bash
curl -X POST "http://localhost:8000/auth/refresh" \
     -H "Content-Type: application/json" \
     -d '{
       "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
     }'
```

#### Response

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Get Current User

Get information about the currently authenticated user.

**Endpoint**: `GET /auth/me`  
**Authentication**: JWT token required  
**Rate Limit**: 100/hour

#### Request

```bash
curl -H "Authorization: Bearer your_jwt_token" \
     "http://localhost:8000/auth/me"
```

#### Response

```json
{
  "user": {
    "id": "user_123",
    "username": "your_username",
    "email": "user@example.com",
    "role": "trader",
    "permissions": ["predict", "monitor", "models"],
    "created_at": "2024-01-01T00:00:00Z",
    "last_login": "2024-01-15T10:30:00Z"
  },
  "session": {
    "token_issued_at": "2024-01-15T10:30:00Z",
    "token_expires_at": "2024-01-15T11:30:00Z",
    "requests_made": 45
  }
}
```

## 🤖 Model Management Endpoints

### List Available Models

Get information about all available ML models.

**Endpoint**: `GET /models`  
**Authentication**: JWT token required  
**Rate Limit**: 100/hour

#### Request

```bash
curl -H "Authorization: Bearer your_jwt_token" \
     "http://localhost:8000/models"
```

#### Response

```json
{
  "models": [
    {
      "id": "lightgbm_v1.2.3",
      "name": "LightGBM Ensemble",
      "type": "lightgbm",
      "version": "1.2.3",
      "status": "active",
      "accuracy": 0.847,
      "training_date": "2024-01-15T08:00:00Z",
      "features_count": 127,
      "weight_in_ensemble": 0.4
    },
    {
      "id": "xgboost_v1.2.3",
      "name": "XGBoost Classifier",
      "type": "xgboost", 
      "version": "1.2.3",
      "status": "active",
      "accuracy": 0.832,
      "training_date": "2024-01-15T08:00:00Z",
      "features_count": 127,
      "weight_in_ensemble": 0.3
    }
  ],
  "ensemble": {
    "version": "v1.2.3",
    "models_count": 4,
    "combined_accuracy": 0.891,
    "last_updated": "2024-01-15T08:00:00Z"
  }
}
```

### Get Model Performance

Get detailed performance metrics for a specific model.

**Endpoint**: `GET /models/{model_id}/metrics`  
**Authentication**: JWT token required  
**Rate Limit**: 100/hour

#### Request

```bash
curl -H "Authorization: Bearer your_jwt_token" \
     "http://localhost:8000/models/lightgbm_v1.2.3/metrics"
```

#### Response

```json
{
  "model": {
    "id": "lightgbm_v1.2.3",
    "name": "LightGBM Ensemble",
    "version": "1.2.3"
  },
  "performance": {
    "accuracy": 0.847,
    "precision": 0.852,
    "recall": 0.843,
    "f1_score": 0.847,
    "auc_roc": 0.891,
    "sharpe_ratio": 1.84,
    "max_drawdown": 0.12
  },
  "predictions": {
    "total_predictions": 15420,
    "correct_predictions": 13056,
    "accuracy_trend_7d": 0.851,
    "predictions_per_hour": 24.3
  },
  "feature_importance": {
    "sentiment_score": 0.35,
    "rsi_14": 0.25,
    "volume_ratio": 0.20,
    "macd_signal": 0.20
  }
}
```

### Trigger Model Retraining

Start the model retraining process with latest data.

**Endpoint**: `POST /models/retrain`  
**Authentication**: Admin role required  
**Rate Limit**: 5/hour

#### Request

```bash
curl -X POST "http://localhost:8000/models/retrain" \
     -H "Authorization: Bearer your_admin_jwt_token" \
     -H "Content-Type: application/json" \
     -d '{
       "models": ["lightgbm", "xgboost"],
       "training_data_days": 30,
       "validation_split": 0.2,
       "notify_completion": true
     }'
```

#### Response

```json
{
  "training_job": {
    "id": "train_job_456",
    "status": "started",
    "models": ["lightgbm", "xgboost"],
    "started_at": "2024-01-15T10:30:00Z",
    "estimated_completion": "2024-01-15T12:30:00Z"
  },
  "message": "Model retraining started. Check job status at /models/training/{job_id}"
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| active_only | boolean | true | Show only active alerts |
| level | string | all | Filter by alert level (info, warning, critical) |
| limit | integer | 100 | Maximum number of alerts |

#### Request

```bash
curl -H "Authorization: Bearer your_token" \
     "https://api.tradingbot.com/alerts?active_only=true&level=critical"
```

#### Response

```json
[
  {
    "id": "alert_123abc",
    "level": "critical",
    "title": "High CPU Usage",
    "message": "CPU usage has exceeded 90% for more than 5 minutes",
    "timestamp": "2024-01-15T10:25:00Z",
    "source": "system_monitor",
    "resolved": false,
    "metadata": {
      "cpu_percent": 92.5,
      "threshold": 90.0,
      "duration_minutes": 7
    }
  }
]
```

## Configuration Endpoints

### Get Configuration

Retrieve current system configuration.

**Endpoint**: `GET /config`  
**Authentication**: Read-only permission required  
**Rate Limit**: 50/hour

#### Request

```bash
curl -H "Authorization: Bearer your_token" \
     https://api.tradingbot.com/config
```

#### Response

```json
{
  "environment": "production",
  "version": "1.0.0",
  "last_updated": "2024-01-15T09:00:00Z",
  "settings": {
    "trading": {
      "max_positions": 10,
      "base_currency": "USDT",
      "trading_pairs": ["BTCUSDT", "ETHUSDT"]
    },
    "risk": {
      "max_portfolio_risk": 0.02,
      "stop_loss_percent": 0.02
    }
  }
}
```

### Update Configuration

Update system configuration.

**Endpoint**: `POST /config`  
**Authentication**: Admin permission required  
**Rate Limit**: 10/hour

#### Request Body

```json
{
  "settings": {
    "trading": {
      "max_positions": 15
    }
  },
  "validate_only": false,
  "restart_required": false
}
```

#### Request

```bash
curl -X POST \
     -H "Authorization: Bearer your_admin_token" \
     -H "Content-Type: application/json" \
     -d '{"settings": {"trading": {"max_positions": 15}}}' \
     https://api.tradingbot.com/config
```

#### Response

```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "changes_applied": {
    "trading.max_positions": {
      "old": 10,
      "new": 15
    }
  },
  "restart_required": false
}
```

## Trading Endpoints

### Trading Status

Get current trading system status.

**Endpoint**: `GET /trading/status`  
**Authentication**: Read-only permission required  
**Rate Limit**: 100/hour

#### Request

```bash
curl -H "Authorization: Bearer your_token" \
     https://api.tradingbot.com/trading/status
```

#### Response

```json
{
  "active": true,
  "mode": "live",
  "portfolio_value": 10000.50,
  "available_balance": 2500.25,
  "open_positions": 3,
  "daily_pnl": 150.75,
  "total_trades": 127,
  "last_trade_time": "2024-01-15T10:15:00Z",
  "active_strategies": ["momentum", "mean_reversion"],
  "exchange_status": {
    "bybit": "connected",
    "last_update": "2024-01-15T10:29:00Z"
  }
}
```

### Position Information

Get detailed position information.

**Endpoint**: `GET /trading/positions`  
**Authentication**: Read-only permission required  
**Rate Limit**: 100/hour

#### Request

```bash
curl -H "Authorization: Bearer your_token" \
     https://api.tradingbot.com/trading/positions
```

#### Response

```json
[
  {
    "symbol": "BTCUSDT",
    "side": "long",
    "size": 0.1,
    "entry_price": 45000.00,
    "current_price": 45500.00,
    "unrealized_pnl": 50.00,
    "percentage_pnl": 1.11,
    "open_time": "2024-01-15T09:30:00Z",
    "stop_loss": 44100.00,
    "take_profit": 46800.00
  }
]
```

## System Control Endpoints

### Execute System Command

Execute system control commands.

**Endpoint**: `POST /system/command`  
**Authentication**: Control permission required  
**Rate Limit**: 20/hour

#### Request Body

```json
{
  "command": "start",
  "parameters": {
    "mode": "live"
  },
  "reason": "Starting trading after maintenance"
}
```

#### Available Commands

| Command | Description | Parameters |
|---------|-------------|------------|
| start | Start trading system | mode (live/paper) |
| stop | Stop trading system | close_positions (boolean) |
| restart | Restart trading system | - |
| pause | Pause trading | - |
| resume | Resume trading | - |

#### Request

```bash
curl -X POST \
     -H "Authorization: Bearer your_control_token" \
     -H "Content-Type: application/json" \
     -d '{"command": "start", "reason": "Manual start"}' \
     https://api.tradingbot.com/system/command
```

#### Response

```json
{
  "success": true,
  "message": "Trading system start initiated",
  "command_id": "cmd_123abc",
  "estimated_completion": "2024-01-15T10:32:00Z"
}
```

## API Management Endpoints

### List API Keys

List all API keys (admin only).

**Endpoint**: `GET /api/keys`  
**Authentication**: Admin permission required  
**Rate Limit**: 10/hour

#### Request

```bash
curl -H "Authorization: Bearer your_admin_token" \
     https://api.tradingbot.com/api/keys
```

#### Response

```json
[
  {
    "key_id": "key_123abc",
    "name": "Production API Key",
    "permissions": ["read_only", "control"],
    "created_at": "2024-01-01T00:00:00Z",
    "last_used": "2024-01-15T10:28:00Z",
    "enabled": true,
    "rate_limit": 1000
  }
]
```

### Create API Key

Create a new API key (super admin only).

**Endpoint**: `POST /api/keys`  
**Authentication**: Super admin permission required  
**Rate Limit**: 5/hour

#### Request Body

```json
{
  "name": "New Production Key",
  "permissions": ["read_only", "control"],
  "rate_limit": 1000,
  "expires_days": 365
}
```

#### Request

```bash
curl -X POST \
     -H "Authorization: Bearer your_super_admin_token" \
     -H "Content-Type: application/json" \
     -d '{"name": "New Key", "permissions": ["read_only"]}' \
     https://api.tradingbot.com/api/keys
```

#### Response

```json
{
  "api_key": "tb_new_secure_api_key_here",
  "key_id": "key_456def",
  "message": "API key created successfully. Store this key securely - it cannot be retrieved again.",
  "permissions": ["read_only"],
  "rate_limit": 1000,
  "expires_at": "2025-01-15T10:30:00Z"
}
```

## WebSocket API

### Connection

Connect to the WebSocket endpoint for real-time updates.

**Endpoint**: `ws://localhost:8000/ws/{session_id}`  
**Authentication**: Session-based (implement as needed)

#### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session_123');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
    
    // Subscribe to system status updates
    ws.send(JSON.stringify({
        type: 'subscribe',
        subscription: 'system_status'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
    
    if (data.type === 'system_status') {
        updateSystemStatus(data.data);
    }
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function(event) {
    console.log('WebSocket closed:', event.code, event.reason);
};
```

#### Python Example

```python
import asyncio
import websockets
import json

async def websocket_client():
    uri = "ws://localhost:8000/ws/session_123"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to metrics updates
        await websocket.send(json.dumps({
            "type": "subscribe",
            "subscription": "metrics_update"
        }))
        
        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

# Run the client
asyncio.run(websocket_client())
```

### Message Types

#### Subscription Request

```json
{
  "type": "subscribe",
  "subscription": "system_status"
}
```

#### Available Subscriptions

| Subscription | Description | Frequency |
|--------------|-------------|-----------|
| system_status | System health updates | 10 seconds |
| metrics_update | Performance metrics | 10 seconds |
| alert_notification | New alerts | Real-time |
| trading_update | Trading activity | Real-time |
| health_update | Health check results | 30 seconds |

#### Unsubscribe Request

```json
{
  "type": "unsubscribe",
  "subscription": "system_status"
}
```

#### Ping/Pong

```json
{
  "type": "ping"
}
```

Response:
```json
{
  "type": "pong",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### WebSocket Message Examples

#### System Status Update

```json
{
  "type": "system_status",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "status": "healthy",
    "uptime_seconds": 86400,
    "active_alerts": 0,
    "components": {
      "trading_engine": "healthy",
      "database": "healthy"
    }
  }
}
```

#### Metrics Update

```json
{
  "type": "metrics_update",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "system_cpu_percent": 45.2,
    "system_memory_percent": 62.1,
    "trading_portfolio_value": 10000.50,
    "trading_open_positions": 3
  }
}
```

#### Alert Notification

```json
{
  "type": "alert_notification",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "id": "alert_789xyz",
    "level": "warning",
    "title": "High Memory Usage",
    "message": "Memory usage is at 85%",
    "source": "system_monitor"
  }
}
```

## SDK Examples

### Python SDK

```python
import aiohttp
import asyncio
from typing import Dict, Any

class TradingBotAPI:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def get_status(self) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/status",
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def get_metrics(self, hours: int = 1) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/metrics?hours={hours}",
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def execute_command(self, command: str, reason: str = None) -> Dict[str, Any]:
        payload = {"command": command}
        if reason:
            payload["reason"] = reason
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/system/command",
                headers=self.headers,
                json=payload
            ) as response:
                return await response.json()

# Usage
async def main():
    api = TradingBotAPI("https://api.tradingbot.com", "your_api_key")
    
    status = await api.get_status()
    print(f"System status: {status['status']}")
    
    metrics = await api.get_metrics(hours=24)
    print(f"Portfolio value: ${metrics['metrics']['trading_portfolio_value']:.2f}")

asyncio.run(main())
```

### JavaScript SDK

```javascript
class TradingBotAPI {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async getStatus() {
        const response = await fetch(`${this.baseUrl}/status`, {
            headers: this.headers
        });
        return await response.json();
    }
    
    async getMetrics(hours = 1) {
        const response = await fetch(`${this.baseUrl}/metrics?hours=${hours}`, {
            headers: this.headers
        });
        return await response.json();
    }
    
    async executeCommand(command, reason = null) {
        const payload = { command };
        if (reason) payload.reason = reason;
        
        const response = await fetch(`${this.baseUrl}/system/command`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(payload)
        });
        return await response.json();
    }
}

// Usage
const api = new TradingBotAPI('https://api.tradingbot.com', 'your_api_key');

api.getStatus().then(status => {
    console.log(`System status: ${status.status}`);
});

api.getMetrics(24).then(metrics => {
    console.log(`Portfolio: $${metrics.metrics.trading_portfolio_value}`);
});
```

## Webhooks

### Webhook Configuration

Configure webhooks to receive real-time notifications.

```json
{
  "webhooks": {
    "alerts": {
      "url": "https://your-service.com/webhooks/alerts",
      "secret": "your_webhook_secret",
      "events": ["alert.created", "alert.resolved"]
    },
    "trading": {
      "url": "https://your-service.com/webhooks/trading",
      "secret": "your_webhook_secret",
      "events": ["trade.executed", "position.opened", "position.closed"]
    }
  }
}
```

### Webhook Payload

```json
{
  "event": "alert.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "id": "alert_123abc",
    "level": "critical",
    "title": "High CPU Usage",
    "message": "CPU usage exceeded threshold"
  },
  "signature": "sha256=webhook_signature_here"
}
```

### Webhook Verification

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected}", signature)
```

---

This API reference provides comprehensive documentation for integrating with the Trading Bot system. For additional support or feature requests, please refer to the main documentation or contact the development team.