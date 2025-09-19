# Trading Bot API Reference

## Overview

The Trading Bot API provides comprehensive REST and WebSocket interfaces for system management, monitoring, and control. This document covers all available endpoints, authentication methods, and integration examples.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

### API Key Authentication

API keys are the preferred method for programmatic access:

```bash
curl -H "Authorization: Bearer tb_your_api_key_here" \
     https://api.tradingbot.com/status
```

### JWT Token Authentication

JWT tokens are used for web applications:

```bash
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
     https://api.tradingbot.com/status
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

## System Endpoints

### Health Check

Check basic system health.

**Endpoint**: `GET /health`  
**Authentication**: None required  
**Rate Limit**: 1000/hour

#### Request

```bash
curl https://api.tradingbot.com/health
```

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### System Status

Get comprehensive system status information.

**Endpoint**: `GET /status`  
**Authentication**: Read-only permission required  
**Rate Limit**: 100/hour

#### Request

```bash
curl -H "Authorization: Bearer your_token" \
     https://api.tradingbot.com/status
```

#### Response

```json
{
  "status": "healthy",
  "uptime_seconds": 86400.5,
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "trading_engine": "healthy",
    "risk_manager": "healthy",
    "database": "healthy",
    "exchange_connection": "healthy"
  },
  "active_alerts": 2,
  "message": "System operational with minor alerts"
}
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

## Alert Endpoints

### Get Alerts

Retrieve system alerts.

**Endpoint**: `GET /alerts`  
**Authentication**: Read-only permission required  
**Rate Limit**: 100/hour

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