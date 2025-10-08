"""
Enhanced Security Manager for Trading Bot
========================================

Implements authentication hardening, session management, and security monitoring.
Add this to your application for production-grade security.
"""

import hashlib
import hmac
import time
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from collections import defaultdict, deque

class SecurityManager:
    """Enhanced security manager with rate limiting and intrusion detection"""
    
    def __init__(self):
        self.failed_attempts = defaultdict(list)
        self.locked_ips = {}
        self.session_tokens = {}
        self.suspicious_ips = set()
        
        # Security thresholds
        self.max_attempts = 5
        self.lockout_duration = 3600  # 1 hour
        self.attempt_window = 900     # 15 minutes
        
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if IP is rate limited"""
        now = time.time()
        
        # Check if IP is currently locked
        if client_ip in self.locked_ips:
            if now < self.locked_ips[client_ip]:
                return False  # Still locked
            else:
                del self.locked_ips[client_ip]  # Lock expired
                
        # Check failed attempts in recent window
        if client_ip in self.failed_attempts:
            # Remove old attempts
            recent_attempts = [t for t in self.failed_attempts[client_ip] 
                             if now - t < self.attempt_window]
            self.failed_attempts[client_ip] = recent_attempts
            
            # Lock if too many recent attempts
            if len(recent_attempts) >= self.max_attempts:
                self.locked_ips[client_ip] = now + self.lockout_duration
                self.suspicious_ips.add(client_ip)
                return False
                
        return True
    
    def record_failed_attempt(self, client_ip: str):
        """Record a failed authentication attempt"""
        self.failed_attempts[client_ip].append(time.time())
        
        # Log security event
        logging.warning(f"Failed authentication attempt from {client_ip}")
    
    def is_suspicious_ip(self, client_ip: str) -> bool:
        """Check if IP has shown suspicious behavior"""
        return client_ip in self.suspicious_ips


class SessionManager:
    """Secure session management with token validation"""
    
    def __init__(self):
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 hour
        self.secret_key = os.getenv('SECRET_KEY', 'fallback_key_change_in_production')
        
    def create_session(self, user_id: str, client_ip: str) -> str:
        """Create a secure session token"""
        timestamp = str(time.time())
        token_data = f"{user_id}:{client_ip}:{timestamp}"
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key.encode(),
            token_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        session_token = f"{token_data}:{signature}"
        
        # Store session info
        self.active_sessions[session_token] = {
            'user_id': user_id,
            'client_ip': client_ip,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        
        return session_token
    
    def validate_session(self, token: str, client_ip: str) -> bool:
        """Validate session token"""
        if not token or token not in self.active_sessions:
            return False
            
        session = self.active_sessions[token]
        now = time.time()
        
        # Check if session expired
        if now - session['last_activity'] > self.session_timeout:
            del self.active_sessions[token]
            return False
            
        # Check IP consistency (prevent session hijacking)
        if session['client_ip'] != client_ip:
            del self.active_sessions[token]
            logging.warning(f"Session hijacking attempt: token from {session['client_ip']} used by {client_ip}")
            return False
            
        # Update last activity
        session['last_activity'] = now
        return True
    
    def invalidate_session(self, token: str):
        """Invalidate a session token"""
        if token in self.active_sessions:
            del self.active_sessions[token]


class SecurityAuditLogger:
    """Comprehensive security event logging"""
    
    def __init__(self):
        # Create security logger
        self.security_logger = logging.getLogger('security_audit')
        self.security_logger.setLevel(logging.INFO)
        
        # Create file handler (cross-platform path)
        import os
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'trading-security-audit.log')
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
        
    def log_security_event(self, event_type: str, details: dict):
        """Log security event in structured format"""
        security_event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'server_id': os.getenv('SERVER_NAME', 'trading-bot')
        }
        
        self.security_logger.info(json.dumps(security_event))
        
    def log_api_access(self, client_ip: str, endpoint: str, method: str, 
                      status_code: int, user_id: Optional[str] = None):
        """Log API access for audit trail"""
        self.log_security_event('API_ACCESS', {
            'client_ip': client_ip,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'user_id': user_id
        })
        
    def log_trading_action(self, action_type: str, symbol: str, 
                          quantity: float, price: float, user_id: Optional[str] = None):
        """Log trading actions for compliance"""
        self.log_security_event('TRADING_ACTION', {
            'action_type': action_type,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'user_id': user_id
        })


class TradingSecurityMonitor:
    """Monitor trading activity for security issues"""
    
    def __init__(self):
        self.trade_history = []
        self.alert_thresholds = {
            'max_hourly_trades': 50,
            'max_daily_volume': 100000,  # USDT
            'max_position_size': 10000,  # USDT
            'suspicious_profit_rate': 50  # % per day
        }
        
    async def monitor_trade(self, trade_data: dict):
        """Monitor individual trades for security issues"""
        
        # Check trade size
        if trade_data.get('volume_usdt', 0) > self.alert_thresholds['max_position_size']:
            await self.send_security_alert(
                f"Large trade detected: {trade_data['volume_usdt']} USDT"
            )
            
        # Check for rapid trading
        recent_trades = [t for t in self.trade_history 
                        if t['timestamp'] > datetime.now() - timedelta(hours=1)]
        
        if len(recent_trades) > self.alert_thresholds['max_hourly_trades']:
            await self.send_security_alert(
                f"High frequency trading detected: {len(recent_trades)} trades in 1 hour"
            )
            
        # Store trade for monitoring
        self.trade_history.append({
            'timestamp': datetime.now(),
            'volume_usdt': trade_data.get('volume_usdt', 0),
            'symbol': trade_data.get('symbol', ''),
            'side': trade_data.get('side', '')
        })
        
        # Keep only recent trades in memory (24 hours)
        cutoff = datetime.now() - timedelta(days=1)
        self.trade_history = [t for t in self.trade_history if t['timestamp'] > cutoff]
        
    async def send_security_alert(self, message: str):
        """Send security alert"""
        logging.critical(f"TRADING SECURITY ALERT: {message}")
        # TODO: Add email/webhook notification here


# Initialize security components
security_manager = SecurityManager()
session_manager = SessionManager()  
security_audit = SecurityAuditLogger()
trading_monitor = TradingSecurityMonitor()