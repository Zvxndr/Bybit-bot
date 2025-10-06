# 🔌 **API REFERENCE DOCUMENTATION**

**Last Updated:** October 6, 2025  
**Version:** 1.0.0  
**Base URL:** `http://localhost:8000` (development) | `https://your-domain.com` (production)

---

## 🚀 **API OVERVIEW**

The Bybit Trading Bot provides a comprehensive RESTful API with WebSocket support for real-time trading bot management. Built with **FastAPI**, featuring automatic OpenAPI documentation, JWT authentication, and role-based access control.

### **Key Features:**
- 🔐 **JWT Authentication** with role-based permissions
- ⚡ **WebSocket** real-time streaming  
- 🛡️ **Rate limiting** per endpoint and user
- 📊 **Comprehensive system monitoring**
- 🔄 **Trading system control**
- ⚙️ **Dynamic configuration management**

---

## 🔑 **AUTHENTICATION**

### **Authentication Methods:**
1. **JWT Tokens** (recommended)
2. **API Keys** (for service-to-service)

### **Permission Levels:**
- `READ_ONLY` - View system status and metrics
- `CONTROL` - Execute trading commands  
- `ADMIN` - Modify configuration and manage users
- `SUPER_ADMIN` - Full system control including API key management

### **Getting JWT Token:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### **Using Authentication:**
Include the JWT token in the Authorization header for all authenticated requests:
```bash
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## 📊 **SYSTEM ENDPOINTS**

### **Health Check**
```http
GET /health
```
Basic health check - no authentication required.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-06T10:30:15.123456"
}
```

**Rate Limit:** Unlimited  
**Authentication:** None

---

### **System Status**
```http
GET /status
```
Get comprehensive system status including trading bot health, exchange connectivity, and system metrics.

**Headers:**
```http
Authorization: Bearer {token}
```

**Response:**
```json
{
  "system": {
    "status": "operational",
    "uptime_seconds": 3600,
    "version": "1.0.0",
    "environment": "production"
  },
  "trading_bot": {
    "active": true,
    "mode": "live",
    "status": "running",
    "last_update": "2025-10-06T10:30:15Z"
  },
  "exchange": {
    "connected": true,
    "name": "bybit",
    "testnet": false,
    "last_ping": "2025-10-06T10:30:10Z"
  },
  "performance": {
    "cpu_usage": 15.2,
    "memory_usage": 45.7,
    "disk_usage": 23.1
  }
}
```

**Rate Limit:** 100/minute  
**Permission:** READ_ONLY

---

### **System Metrics**
```http
GET /metrics?hours={hours}
```
Get detailed system metrics for the specified time period.

**Query Parameters:**
- `hours` (optional): Number of hours of metrics to retrieve (default: 1)

**Headers:**
```http
Authorization: Bearer {token}
```

**Response:**
```json
{
  "timestamp": "2025-10-06T10:30:15Z",
  "period_hours": 1,
  "metrics": {
    "trading": {
      "total_trades": 45,
      "successful_trades": 42,
      "failed_trades": 3,
      "total_pnl": 125.50,
      "win_rate": 0.675
    },
    "system": {
      "avg_cpu_usage": 12.5,
      "avg_memory_usage": 43.2,
      "max_latency_ms": 45,
      "uptime_percentage": 99.98
    },
    "exchange": {
      "api_calls": 1250,
      "failed_calls": 2,
      "avg_response_time_ms": 85
    }
  }
}
```

**Rate Limit:** 60/minute  
**Permission:** READ_ONLY

---

### **System Alerts**
```http
GET /alerts?active_only={bool}
```
Get system alerts and warnings.

**Query Parameters:**
- `active_only` (optional): Return only active alerts (default: true)

**Headers:**
```http
Authorization: Bearer {token}
```

**Response:**
```json
{
  "alerts": [
    {
      "id": "alert_001",
      "type": "warning",
      "title": "High CPU Usage",
      "message": "CPU usage above 80% for 5 minutes",
      "severity": "medium",
      "active": true,
      "created_at": "2025-10-06T10:25:00Z",
      "resolved_at": null
    }
  ],
  "total_alerts": 1,
  "active_alerts": 1
}
```

**Rate Limit:** 60/minute  
**Permission:** READ_ONLY

---

## ⚙️ **CONFIGURATION ENDPOINTS**

### **Get Configuration**
```http
GET /config
```
Retrieve current system configuration.

**Headers:**
```http
Authorization: Bearer {token}
```

**Response:**
```json
{
  "environment": "production",
  "version": "1.0.0",
  "last_updated": "2025-10-06T09:15:30Z",
  "settings": {
    "trading": {
      "max_position_size": 0.1,
      "trading_pairs": ["BTCUSDT", "ETHUSDT"],
      "base_currency": "USDT"
    },
    "risk": {
      "max_portfolio_risk": 0.02,
      "max_daily_loss": 0.05,
      "stop_loss_percent": 0.03
    }
  }
}
```

**Rate Limit:** 60/minute  
**Permission:** READ_ONLY

---

### **Update Configuration**
```http
POST /config
```
Update system configuration settings.

**Headers:**
```http
Authorization: Bearer {token}
Content-Type: application/json
```

**Request Body:**
```json
{
  "settings": {
    "trading": {
      "max_position_size": 0.15
    },
    "risk": {
      "stop_loss_percent": 0.025
    }
  },
  "validate_only": false,
  "restart_required": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "restart_required": false,
  "updated_settings": ["trading.max_position_size", "risk.stop_loss_percent"]
}
```

**Rate Limit:** 30/minute  
**Permission:** ADMIN

---

## 📈 **TRADING ENDPOINTS**

### **Trading Status**
```http
GET /trading/status
```
Get current trading system status and performance.

**Headers:**
```http
Authorization: Bearer {token}
```

**Response:**
```json
{
  "active": true,
  "mode": "live",
  "portfolio_value": 10125.50,
  "open_positions": 3,
  "daily_pnl": 125.50,
  "total_trades": 847,
  "last_trade_time": "2025-10-06T10:28:45Z",
  "positions": [
    {
      "symbol": "BTCUSDT",
      "side": "long",
      "size": 0.05,
      "entry_price": 45000.00,
      "current_price": 45150.00,
      "unrealized_pnl": 7.50
    }
  ]
}
```

**Rate Limit:** 120/minute  
**Permission:** READ_ONLY

---

## 🎮 **SYSTEM CONTROL ENDPOINTS**

### **Execute System Command**
```http
POST /system/command
```
Execute system control commands (start, stop, pause, resume, restart).

**Headers:**
```http
Authorization: Bearer {token}
Content-Type: application/json
```

**Request Body:**
```json
{
  "command": "start",
  "parameters": {},
  "reason": "Manual start via API"
}
```

**Available Commands:**
- `start` - Start trading operations
- `stop` - Stop trading operations
- `pause` - Pause trading temporarily
- `resume` - Resume paused trading
- `restart` - Restart trading system

**Response:**
```json
{
  "success": true,
  "message": "Trading bot start initiated",
  "command": "start",
  "timestamp": "2025-10-06T10:30:15Z"
}
```

**Rate Limit:** 30/minute  
**Permission:** CONTROL

---

## 🔐 **API KEY MANAGEMENT**

### **List API Keys**
```http
GET /api/keys
```
List all API keys (admin only).

**Headers:**
```http
Authorization: Bearer {token}
```

**Response:**
```json
{
  "api_keys": [
    {
      "key_id": "key_123",
      "name": "Production Dashboard",
      "permissions": ["READ_ONLY", "CONTROL"],
      "created_at": "2025-10-01T10:00:00Z",
      "last_used": "2025-10-06T09:45:30Z",
      "enabled": true
    }
  ]
}
```

**Rate Limit:** 30/minute  
**Permission:** ADMIN

---

### **Create API Key**
```http
POST /api/keys?name={name}&permissions={permissions}&rate_limit={rate_limit}
```
Create a new API key (super admin only).

**Query Parameters:**
- `name`: Descriptive name for the API key
- `permissions`: Comma-separated list of permissions
- `rate_limit` (optional): Rate limit per hour (default: 1000)
- `expires_days` (optional): Expiration in days

**Headers:**
```http
Authorization: Bearer {token}
```

**Response:**
```json
{
  "key_id": "key_456",
  "api_key": "sk_live_abc123...",
  "name": "New Dashboard Key",
  "permissions": ["READ_ONLY"],
  "rate_limit": 1000,
  "created_at": "2025-10-06T10:30:15Z",
  "expires_at": "2025-11-06T10:30:15Z"
}
```

**Rate Limit:** 10/hour  
**Permission:** SUPER_ADMIN

---

## 🔌 **WEBSOCKET API**

### **Connection Endpoint**
```
ws://localhost:8000/ws/{session_id}
```
Real-time WebSocket connection for live updates.

**Connection Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/dashboard_001');

ws.onopen = () => {
  console.log('WebSocket connected');
  
  // Subscribe to system updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    topics: ['system_status', 'trading_updates', 'alerts']
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  handleRealtimeUpdate(message);
};
```

### **Message Types:**

#### **Subscription Request:**
```json
{
  "type": "subscribe",
  "topics": ["system_status", "trading_updates", "alerts", "metrics"]
}
```

#### **System Status Update:**
```json
{
  "type": "system_status",
  "timestamp": "2025-10-06T10:30:15Z",
  "data": {
    "trading_active": true,
    "cpu_usage": 15.2,
    "memory_usage": 45.7,
    "open_positions": 3
  }
}
```

#### **Trading Update:**
```json
{
  "type": "trading_update",
  "timestamp": "2025-10-06T10:30:15Z",
  "data": {
    "event": "position_opened",
    "symbol": "BTCUSDT",
    "side": "long",
    "size": 0.05,
    "price": 45000.00
  }
}
```

#### **Alert Notification:**
```json
{
  "type": "alert",
  "timestamp": "2025-10-06T10:30:15Z",
  "data": {
    "id": "alert_002",
    "severity": "high",
    "title": "Trading Stopped",
    "message": "Emergency stop triggered due to high losses"
  }
}
```

### **WebSocket Statistics:**
```http
GET /websocket/stats
```
Get WebSocket connection statistics.

**Response:**
```json
{
  "total_connections": 5,
  "active_sessions": ["dashboard_001", "mobile_app_002"],
  "subscription_stats": {
    "system_status": 3,
    "trading_updates": 5,
    "alerts": 4,
    "metrics": 2
  },
  "messages_sent_today": 15847,
  "uptime_hours": 24.5
}
```

**Rate Limit:** 60/minute  
**Permission:** READ_ONLY

---

## 📊 **RATE LIMITING**

### **Rate Limit Headers:**
All API responses include rate limiting information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1696598415
X-RateLimit-Retry-After: 60
```

### **Rate Limits by Permission:**
- **Unauthenticated:** 20/minute
- **READ_ONLY:** 100/minute
- **CONTROL:** 60/minute  
- **ADMIN:** 30/minute
- **SUPER_ADMIN:** 10/hour (API key creation only)

### **Rate Limit Exceeded Response:**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 60,
  "limit": 100,
  "reset": 1696598415
}
```

---

## ❌ **ERROR RESPONSES**

### **Standard Error Format:**
```json
{
  "error": "error_code",
  "message": "Human-readable error description",
  "details": {
    "field": "Additional error context"
  },
  "timestamp": "2025-10-06T10:30:15Z",
  "request_id": "req_123456"
}
```

### **HTTP Status Codes:**
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error
- `503` - Service Unavailable

### **Common Error Codes:**
- `invalid_token` - JWT token invalid or expired
- `insufficient_permissions` - User lacks required permissions
- `rate_limit_exceeded` - Rate limit exceeded
- `validation_error` - Request validation failed
- `system_unavailable` - Trading system temporarily unavailable
- `configuration_error` - Configuration update failed

---

## 🔧 **DEVELOPMENT TOOLS**

### **OpenAPI Documentation**
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

### **Postman Collection**
A Postman collection is available with pre-configured requests for all endpoints.

### **SDK Examples**

#### **Python SDK:**
```python
import asyncio
import aiohttp

class TradingBotClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    async def get_status(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/status", 
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def start_trading(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/system/command",
                headers=self.headers,
                json={"command": "start", "reason": "API start"}
            ) as response:
                return await response.json()
```

#### **JavaScript SDK:**
```javascript
class TradingBotAPI {
  constructor(baseURL, token) {
    this.baseURL = baseURL;
    this.headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }
  
  async getStatus() {
    const response = await fetch(`${this.baseURL}/status`, {
      headers: this.headers
    });
    return response.json();
  }
  
  async executeCommand(command, reason = 'API command') {
    const response = await fetch(`${this.baseURL}/system/command`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ command, reason })
    });
    return response.json();
  }
}
```

---

## 🚀 **GETTING STARTED**

### **1. Start the API Server:**
```bash
cd src/
python main.py
```

### **2. Verify API is Running:**
```bash
curl http://localhost:8000/health
```

### **3. Authenticate:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

### **4. Make Your First API Call:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/status
```

### **5. Connect to WebSocket:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/test_session');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

---

## 📝 **CHANGELOG**

### **v1.0.0 (2025-10-06)**
- Initial API release
- JWT authentication system
- Role-based permissions
- WebSocket real-time updates  
- Comprehensive system monitoring
- Trading control endpoints
- Configuration management
- Rate limiting implementation

---

**📖 Complete API documentation with examples and best practices. Ready for frontend integration.**