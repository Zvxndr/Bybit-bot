#!/usr/bin/env python3
"""
Simple Backend Test Script
Test what's working and what's broken in our backend
"""

import sys
import os
from pathlib import Path

print("🔍 Backend Diagnostic Test")
print("=" * 50)

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"📁 Source path: {src_path}")
print(f"📁 Files in src: {list(src_path.glob('*.py'))}")

# Test 1: Basic imports
print("\n1️⃣ Testing basic imports...")
try:
    import shared_state
    print("✅ shared_state imported")
except Exception as e:
    print(f"❌ shared_state failed: {e}")

try:
    import bybit_api  
    print("✅ bybit_api imported")
except Exception as e:
    print(f"❌ bybit_api failed: {e}")

# Test 2: ML dependencies
print("\n2️⃣ Testing ML dependencies...")
try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__}")
except Exception as e:
    print(f"❌ scikit-learn failed: {e}")

try:
    import lightgbm
    print(f"✅ lightgbm {lightgbm.__version__}")
except Exception as e:
    print(f"❌ lightgbm failed: {e}")

# Test 3: FastAPI
print("\n3️⃣ Testing FastAPI...")
try:
    from fastapi import FastAPI
    print("✅ FastAPI imported")
except Exception as e:
    print(f"❌ FastAPI failed: {e}")

# Test 4: Simple server
print("\n4️⃣ Testing simple server...")
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    class SimpleHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<h1>Backend is working!</h1>')
            elif self.path == '/api/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                status = {
                    "status": "running",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "message": "Backend test server"
                }
                self.wfile.write(json.dumps(status).encode())
        
        def log_message(self, format, *args):
            return  # Suppress logs
    
    print("✅ HTTP server components ready")
    print("\n🚀 Starting test server on http://localhost:8080")
    print("📡 API endpoint: http://localhost:8080/api/status")
    print("🛑 Press Ctrl+C to stop")
    
    server = HTTPServer(('localhost', 8080), SimpleHandler)
    server.serve_forever()
    
except KeyboardInterrupt:
    print("\n🛑 Server stopped")
except Exception as e:
    print(f"❌ Server failed: {e}")
    import traceback
    traceback.print_exc()