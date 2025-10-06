"""
Ultra Simple Backend - No Dependencies
=====================================
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
import sys
import os

class TradingBotHandler(BaseHTTPRequestHandler):
    """Simple HTTP request handler for trading bot API"""
    
    def do_GET(self):
        """Handle GET requests"""
        print(f"📡 GET request: {self.path}")
        
        if self.path == '/':
            self.send_html_response("""
            <html>
                <head><title>Bybit Trading Bot</title></head>
                <body>
                    <h1>🤖 Bybit Trading Bot Backend</h1>
                    <h2>Status: ✅ Running</h2>
                    <h3>API Endpoints:</h3>
                    <ul>
                        <li><a href="/api/status">/api/status</a> - System status</li>
                        <li><a href="/api/portfolio">/api/portfolio</a> - Portfolio data</li>
                        <li><a href="/api/strategies">/api/strategies</a> - Trading strategies</li>
                        <li><a href="/api/trades">/api/trades</a> - Recent trades</li>
                    </ul>
                </body>
            </html>
            """)
            
        elif self.path == '/api/status':
            self.send_json_response({
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "trading_enabled": False,
                "api_connected": False,
                "system_health": "good",
                "uptime": "0d 0h 5m"
            })
            
        elif self.path == '/api/portfolio':
            self.send_json_response({
                "balance": {
                    "USDT": 1000.0,
                    "BTC": 0.025,
                    "ETH": 0.5
                },
                "positions": [
                    {"symbol": "BTCUSDT", "side": "long", "size": 0.01, "pnl": 12.5},
                    {"symbol": "ETHUSDT", "side": "short", "size": 0.1, "pnl": -5.2}
                ],
                "total_value": 2247.83,
                "total_pnl": 7.3,
                "profit_24h": 15.75
            })
            
        elif self.path == '/api/strategies':
            self.send_json_response({
                "strategies": [
                    {
                        "name": "Momentum Scalper",
                        "status": "active",
                        "profit": 12.5,
                        "trades": 23,
                        "win_rate": 65.2
                    },
                    {
                        "name": "Mean Reversion",
                        "status": "paused",
                        "profit": -3.2,
                        "trades": 8,
                        "win_rate": 37.5
                    },
                    {
                        "name": "ML Trend Following",
                        "status": "active",
                        "profit": 8.7,
                        "trades": 15,
                        "win_rate": 73.3
                    }
                ]
            })
            
        elif self.path == '/api/trades':
            self.send_json_response({
                "trades": [
                    {
                        "id": "trade_001",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "amount": 0.01,
                        "price": 42500.0,
                        "timestamp": "2025-01-01T10:30:00Z",
                        "pnl": 25.0
                    },
                    {
                        "id": "trade_002", 
                        "symbol": "ETHUSDT",
                        "side": "sell",
                        "amount": 0.1,
                        "price": 3200.0,
                        "timestamp": "2025-01-01T09:15:00Z",
                        "pnl": -12.5
                    }
                ]
            })
            
        elif self.path == '/health':
            # Health check endpoint for DigitalOcean
            self.send_json_response({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "bybit-trading-bot"
            })
            
        else:
            self.send_error(404, f"Endpoint not found: {self.path}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_json_response(self, data):
        """Send JSON response with CORS headers"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        json_data = json.dumps(data, indent=2)
        self.wfile.write(json_data.encode('utf-8'))
    
    def send_html_response(self, html):
        """Send HTML response"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"🌐 {self.address_string()} - {format % args}")

def main():
    """Start the simple HTTP server"""
    host = '0.0.0.0'  # Allow external connections for DigitalOcean
    port = int(os.getenv('PORT', 8080))  # Use DigitalOcean's PORT env var
    
    print("🚀 Starting Ultra Simple Backend...")
    print(f"📡 Server: http://{host}:{port}")
    print("📊 API Endpoints:")
    print(f"   - http://{host}:{port}/api/status")
    print(f"   - http://{host}:{port}/api/portfolio")
    print(f"   - http://{host}:{port}/api/strategies")
    print(f"   - http://{host}:{port}/api/trades")
    print("🌐 Web interface: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        server = HTTPServer((host, port), TradingBotHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()