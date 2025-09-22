#!/usr/bin/env python3
"""
Advanced Threat Detection System
Real-time threat detection with behavioral analytics and automated response
Addresses: Professional Audit Finding - Advanced Threat Detection Requirements
"""

import asyncio
import logging
import json
import time
import hashlib
import ipaddress
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import sqlite3
import threading
import geoip2.database
import geoip2.errors
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import secrets
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of threats"""
    BRUTE_FORCE = "brute_force"
    ANOMALOUS_LOGIN = "anomalous_login"
    SUSPICIOUS_API_USAGE = "suspicious_api_usage"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    TRADING_PATTERN_DEVIATION = "trading_pattern_deviation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_IP = "malicious_ip"
    TIME_BASED_ANOMALY = "time_based_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"

class ResponseAction(Enum):
    """Automated response actions"""
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"
    BLOCK_IP = "block_ip"
    LOCK_ACCOUNT = "lock_account"
    ALERT_ADMIN = "alert_admin"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

@dataclass
class ThreatEvent:
    """Threat event structure"""
    event_id: str = field(default_factory=lambda: secrets.token_hex(16))
    timestamp: datetime = field(default_factory=datetime.now)
    threat_type: ThreatType = ThreatType.SUSPICIOUS_API_USAGE
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.5
    confidence: float = 0.7
    false_positive_probability: float = 0.1
    mitigation_applied: List[ResponseAction] = field(default_factory=list)
    resolved: bool = False

@dataclass
class UserBehaviorProfile:
    """User behavior profile for anomaly detection"""
    user_id: str
    login_locations: Set[str] = field(default_factory=set)
    login_times: List[int] = field(default_factory=list)  # Hours of day
    api_call_patterns: Dict[str, int] = field(default_factory=dict)
    trading_volume_history: List[float] = field(default_factory=list)
    typical_session_duration: float = 3600  # seconds
    last_update: datetime = field(default_factory=datetime.now)
    anomaly_score_history: List[float] = field(default_factory=list)

@dataclass
class IPIntelligence:
    """IP address intelligence data"""
    ip_address: str
    country: Optional[str] = None
    city: Optional[str] = None
    is_vpn: bool = False
    is_tor: bool = False
    is_malicious: bool = False
    reputation_score: float = 0.5
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    request_count: int = 0
    threat_events: int = 0

class BehaviorAnalyzer:
    """Behavioral analytics engine"""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        self._lock = threading.Lock()
        
        # Behavior patterns
        self.normal_login_hours = set(range(6, 23))  # 6 AM to 11 PM
        self.max_api_calls_per_minute = 100
        self.max_session_duration = 8 * 3600  # 8 hours
        
    async def analyze_login_behavior(self, user_id: str, ip_address: str, 
                                   timestamp: datetime) -> Tuple[float, Dict[str, Any]]:
        """Analyze login behavior for anomalies"""
        with self._lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
            
            profile = self.user_profiles[user_id]
            anomaly_details = {}
            risk_score = 0.0
            
            # Geographic anomaly
            if ip_address not in profile.login_locations:
                if len(profile.login_locations) > 0:
                    risk_score += 0.3
                    anomaly_details['new_location'] = True
                profile.login_locations.add(ip_address)
            
            # Time-based anomaly
            login_hour = timestamp.hour
            if login_hour not in self.normal_login_hours:
                risk_score += 0.2
                anomaly_details['unusual_time'] = True
            
            profile.login_times.append(login_hour)
            if len(profile.login_times) > 100:
                profile.login_times.pop(0)
            
            # Historical pattern analysis
            if len(profile.login_times) >= 10:
                hour_frequency = {}
                for hour in profile.login_times:
                    hour_frequency[hour] = hour_frequency.get(hour, 0) + 1
                
                expected_frequency = hour_frequency.get(login_hour, 0) / len(profile.login_times)
                if expected_frequency < 0.05:  # Less than 5% of historical logins
                    risk_score += 0.25
                    anomaly_details['unusual_pattern'] = True
            
            profile.last_update = timestamp
            return min(risk_score, 1.0), anomaly_details
    
    async def analyze_api_behavior(self, user_id: str, endpoint: str, 
                                 timestamp: datetime) -> Tuple[float, Dict[str, Any]]:
        """Analyze API usage behavior"""
        with self._lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
            
            profile = self.user_profiles[user_id]
            anomaly_details = {}
            risk_score = 0.0
            
            # Update API call patterns
            profile.api_call_patterns[endpoint] = profile.api_call_patterns.get(endpoint, 0) + 1
            
            # Check for unusual API usage
            total_calls = sum(profile.api_call_patterns.values())
            endpoint_frequency = profile.api_call_patterns[endpoint] / max(total_calls, 1)
            
            # Rare endpoint usage
            if total_calls > 100 and endpoint_frequency < 0.01:
                risk_score += 0.3
                anomaly_details['rare_endpoint'] = True
            
            # High frequency usage detection (simple rate limiting check)
            recent_calls = self._count_recent_api_calls(user_id, timestamp)
            if recent_calls > self.max_api_calls_per_minute:
                risk_score += 0.5
                anomaly_details['high_frequency'] = True
            
            return min(risk_score, 1.0), anomaly_details
    
    async def analyze_trading_behavior(self, user_id: str, volume: float, 
                                     timestamp: datetime) -> Tuple[float, Dict[str, Any]]:
        """Analyze trading behavior patterns"""
        with self._lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
            
            profile = self.user_profiles[user_id]
            anomaly_details = {}
            risk_score = 0.0
            
            profile.trading_volume_history.append(volume)
            if len(profile.trading_volume_history) > 1000:
                profile.trading_volume_history.pop(0)
            
            # Volume anomaly detection
            if len(profile.trading_volume_history) >= 10:
                volumes = np.array(profile.trading_volume_history)
                mean_volume = np.mean(volumes)
                std_volume = np.std(volumes)
                
                if std_volume > 0:
                    z_score = abs(volume - mean_volume) / std_volume
                    if z_score > 3:  # 3 standard deviations
                        risk_score += min(z_score / 10, 0.6)
                        anomaly_details['volume_anomaly'] = {
                            'z_score': z_score,
                            'volume': volume,
                            'mean_volume': mean_volume
                        }
            
            return min(risk_score, 1.0), anomaly_details
    
    def _count_recent_api_calls(self, user_id: str, timestamp: datetime, 
                               window_minutes: int = 1) -> int:
        """Count recent API calls for rate limiting"""
        # Simplified implementation - in production, use time-series database
        return 0  # Placeholder

class GeographicAnalyzer:
    """Geographic threat analysis"""
    
    def __init__(self, geoip_db_path: Optional[str] = None):
        self.geoip_db_path = geoip_db_path or "GeoLite2-City.mmdb"
        self.geoip_reader = None
        self._init_geoip()
        
        # High-risk countries/regions (example list)
        self.high_risk_countries = {
            'CN', 'RU', 'KP', 'IR', 'SY'  # ISO country codes
        }
        
        # VPN/Proxy detection patterns
        self.vpn_indicators = [
            'vpn', 'proxy', 'datacenter', 'hosting', 'cloud'
        ]
    
    def _init_geoip(self):
        """Initialize GeoIP database"""
        try:
            if Path(self.geoip_db_path).exists():
                self.geoip_reader = geoip2.database.Reader(self.geoip_db_path)
                logger.info("GeoIP database loaded successfully")
            else:
                logger.warning(f"GeoIP database not found: {self.geoip_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize GeoIP: {e}")
    
    async def analyze_ip_location(self, ip_address: str) -> Tuple[float, Dict[str, Any]]:
        """Analyze IP geolocation for threats"""
        try:
            if not self.geoip_reader:
                return 0.1, {'error': 'GeoIP database not available'}
            
            # Parse IP address
            ip = ipaddress.ip_address(ip_address)
            if ip.is_private or ip.is_loopback:
                return 0.0, {'type': 'private_ip'}
            
            # Get geolocation data
            response = self.geoip_reader.city(ip_address)
            
            details = {
                'country': response.country.iso_code,
                'country_name': response.country.name,
                'city': response.city.name,
                'latitude': float(response.location.latitude) if response.location.latitude else None,
                'longitude': float(response.location.longitude) if response.location.longitude else None
            }
            
            risk_score = 0.0
            
            # High-risk country check
            if response.country.iso_code in self.high_risk_countries:
                risk_score += 0.4
                details['high_risk_country'] = True
            
            # VPN/Proxy detection (simplified)
            org_name = response.traits.autonomous_system_organization or ''
            if any(indicator in org_name.lower() for indicator in self.vpn_indicators):
                risk_score += 0.3
                details['possible_vpn'] = True
            
            return min(risk_score, 1.0), details
            
        except geoip2.errors.AddressNotFoundError:
            return 0.2, {'error': 'IP not found in database'}
        except Exception as e:
            logger.error(f"Geographic analysis failed for {ip_address}: {e}")
            return 0.1, {'error': str(e)}

class ThreatIntelligence:
    """Threat intelligence integration"""
    
    def __init__(self):
        self.ip_intelligence: Dict[str, IPIntelligence] = {}
        self.malicious_ips: Set[str] = set()
        self.reputation_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_ttl = timedelta(hours=24)
        
        # Threat intelligence feeds (examples)
        self.threat_feeds = [
            "https://reputation.alienvault.com/reputation.data",
            "https://rules.emergingthreats.net/fwrules/emerging-Block-IPs.txt"
        ]
    
    async def check_ip_reputation(self, ip_address: str) -> Tuple[float, Dict[str, Any]]:
        """Check IP reputation against threat intelligence"""
        try:
            # Check cache first
            if ip_address in self.reputation_cache:
                reputation, timestamp = self.reputation_cache[ip_address]
                if datetime.now() - timestamp < self.cache_ttl:
                    return reputation, {'source': 'cache'}
            
            # Check local intelligence
            if ip_address in self.malicious_ips:
                self.reputation_cache[ip_address] = (0.9, datetime.now())
                return 0.9, {'source': 'local_intelligence', 'status': 'malicious'}
            
            # Check external threat intelligence (simplified)
            reputation_score = await self._query_external_intelligence(ip_address)
            
            details = {
                'source': 'external_intelligence',
                'reputation_score': reputation_score
            }
            
            # Cache result
            self.reputation_cache[ip_address] = (reputation_score, datetime.now())
            
            return reputation_score, details
            
        except Exception as e:
            logger.error(f"IP reputation check failed for {ip_address}: {e}")
            return 0.1, {'error': str(e)}
    
    async def _query_external_intelligence(self, ip_address: str) -> float:
        """Query external threat intelligence APIs"""
        # Simplified implementation - in production, integrate with real threat intelligence APIs
        try:
            # Example: Check against a basic reputation service
            # This is a placeholder - real implementation would use actual threat intel APIs
            
            # Simulate reputation check
            ip_hash = hashlib.md5(ip_address.encode()).hexdigest()
            reputation = int(ip_hash[:2], 16) / 255.0  # Simulate reputation score
            
            return reputation
            
        except Exception as e:
            logger.error(f"External intelligence query failed: {e}")
            return 0.1
    
    async def update_threat_feeds(self):
        """Update threat intelligence feeds"""
        try:
            for feed_url in self.threat_feeds:
                await self._process_threat_feed(feed_url)
            
            logger.info(f"Updated threat intelligence: {len(self.malicious_ips)} malicious IPs")
            
        except Exception as e:
            logger.error(f"Threat feed update failed: {e}")
    
    async def _process_threat_feed(self, feed_url: str):
        """Process individual threat feed"""
        try:
            # Simplified feed processing
            # In production, implement proper feed parsing for different formats
            pass
        except Exception as e:
            logger.error(f"Failed to process threat feed {feed_url}: {e}")

class AutomatedResponse:
    """Automated threat response system"""
    
    def __init__(self):
        self.response_rules: Dict[ThreatType, Dict[ThreatLevel, List[ResponseAction]]] = {
            ThreatType.BRUTE_FORCE: {
                ThreatLevel.LOW: [ResponseAction.LOG_ONLY],
                ThreatLevel.MEDIUM: [ResponseAction.RATE_LIMIT],
                ThreatLevel.HIGH: [ResponseAction.BLOCK_IP, ResponseAction.ALERT_ADMIN],
                ThreatLevel.CRITICAL: [ResponseAction.BLOCK_IP, ResponseAction.LOCK_ACCOUNT, ResponseAction.ALERT_ADMIN]
            },
            ThreatType.SUSPICIOUS_API_USAGE: {
                ThreatLevel.LOW: [ResponseAction.LOG_ONLY],
                ThreatLevel.MEDIUM: [ResponseAction.RATE_LIMIT],
                ThreatLevel.HIGH: [ResponseAction.RATE_LIMIT, ResponseAction.ALERT_ADMIN],
                ThreatLevel.CRITICAL: [ResponseAction.LOCK_ACCOUNT, ResponseAction.ALERT_ADMIN]
            },
            ThreatType.GEOGRAPHIC_ANOMALY: {
                ThreatLevel.LOW: [ResponseAction.LOG_ONLY],
                ThreatLevel.MEDIUM: [ResponseAction.LOG_ONLY],
                ThreatLevel.HIGH: [ResponseAction.ALERT_ADMIN],
                ThreatLevel.CRITICAL: [ResponseAction.LOCK_ACCOUNT, ResponseAction.ALERT_ADMIN]
            },
            ThreatType.TRADING_PATTERN_DEVIATION: {
                ThreatLevel.LOW: [ResponseAction.LOG_ONLY],
                ThreatLevel.MEDIUM: [ResponseAction.LOG_ONLY],
                ThreatLevel.HIGH: [ResponseAction.ALERT_ADMIN],
                ThreatLevel.CRITICAL: [ResponseAction.EMERGENCY_SHUTDOWN, ResponseAction.ALERT_ADMIN]
            }
        }
        
        self.blocked_ips: Set[str] = set()
        self.rate_limited_ips: Dict[str, datetime] = {}
        self.locked_accounts: Set[str] = set()
    
    async def execute_response(self, threat_event: ThreatEvent) -> List[ResponseAction]:
        """Execute automated response to threat"""
        try:
            # Get response actions for threat type and level
            actions = self.response_rules.get(threat_event.threat_type, {}).get(
                threat_event.threat_level, [ResponseAction.LOG_ONLY]
            )
            
            executed_actions = []
            
            for action in actions:
                success = await self._execute_action(action, threat_event)
                if success:
                    executed_actions.append(action)
            
            return executed_actions
            
        except Exception as e:
            logger.error(f"Response execution failed: {e}")
            return []
    
    async def _execute_action(self, action: ResponseAction, threat_event: ThreatEvent) -> bool:
        """Execute individual response action"""
        try:
            if action == ResponseAction.LOG_ONLY:
                logger.warning(f"Threat detected: {threat_event.threat_type.value} - {threat_event.details}")
                return True
            
            elif action == ResponseAction.RATE_LIMIT:
                if threat_event.source_ip:
                    self.rate_limited_ips[threat_event.source_ip] = datetime.now() + timedelta(hours=1)
                    logger.info(f"Rate limited IP: {threat_event.source_ip}")
                return True
            
            elif action == ResponseAction.BLOCK_IP:
                if threat_event.source_ip:
                    self.blocked_ips.add(threat_event.source_ip)
                    logger.warning(f"Blocked IP: {threat_event.source_ip}")
                return True
            
            elif action == ResponseAction.LOCK_ACCOUNT:
                if threat_event.user_id:
                    self.locked_accounts.add(threat_event.user_id)
                    logger.warning(f"Locked account: {threat_event.user_id}")
                return True
            
            elif action == ResponseAction.ALERT_ADMIN:
                await self._send_admin_alert(threat_event)
                return True
            
            elif action == ResponseAction.EMERGENCY_SHUTDOWN:
                logger.critical("EMERGENCY SHUTDOWN triggered by threat detection")
                # In production, this would trigger actual system shutdown
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
            return False
    
    async def _send_admin_alert(self, threat_event: ThreatEvent):
        """Send alert to administrators"""
        # Simplified alert - in production, integrate with notification systems
        alert_message = {
            'timestamp': threat_event.timestamp.isoformat(),
            'threat_type': threat_event.threat_type.value,
            'threat_level': threat_event.threat_level.value,
            'source_ip': threat_event.source_ip,
            'user_id': threat_event.user_id,
            'risk_score': threat_event.risk_score,
            'details': threat_event.details
        }
        
        logger.critical(f"ADMIN ALERT: {json.dumps(alert_message, indent=2)}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    def is_ip_rate_limited(self, ip_address: str) -> bool:
        """Check if IP is rate limited"""
        if ip_address in self.rate_limited_ips:
            if datetime.now() < self.rate_limited_ips[ip_address]:
                return True
            else:
                del self.rate_limited_ips[ip_address]
        return False
    
    def is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked"""
        return user_id in self.locked_accounts

class ThreatDetectionEngine:
    """
    Main threat detection engine
    Coordinates all threat detection components
    """
    
    def __init__(self, db_path: str = "threat_detection.db"):
        self.db_path = db_path
        self._init_database()
        
        # Initialize components
        self.behavior_analyzer = BehaviorAnalyzer()
        self.geographic_analyzer = GeographicAnalyzer()
        self.threat_intelligence = ThreatIntelligence()
        self.automated_response = AutomatedResponse()
        
        # Threat detection configuration
        self.detection_enabled = True
        self.detection_sensitivity = 0.7  # 0.0 to 1.0
        self.response_enabled = True
        
        # Event processing
        self.event_queue = asyncio.Queue()
        self.processing_tasks = []
        
        # Statistics
        self.stats = {
            'threats_detected': 0,
            'false_positives': 0,
            'responses_executed': 0,
            'last_threat_feed_update': None
        }
        
        self._lock = threading.Lock()
    
    def _init_database(self):
        """Initialize threat detection database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                threat_type TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                source_ip TEXT,
                user_id TEXT,
                session_id TEXT,
                details TEXT,
                risk_score REAL,
                confidence REAL,
                false_positive_probability REAL,
                mitigation_applied TEXT,
                resolved INTEGER
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON threat_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_threat_type ON threat_events(threat_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_ip ON threat_events(source_ip)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON threat_events(user_id)')
        
        conn.commit()
        conn.close()
    
    async def start_detection(self):
        """Start threat detection engine"""
        try:
            logger.info("Starting threat detection engine...")
            
            # Start event processing tasks
            for i in range(3):  # 3 worker tasks
                task = asyncio.create_task(self._process_events())
                self.processing_tasks.append(task)
            
            # Start threat intelligence updates
            update_task = asyncio.create_task(self._periodic_intelligence_update())
            self.processing_tasks.append(update_task)
            
            logger.info("Threat detection engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start threat detection engine: {e}")
    
    async def stop_detection(self):
        """Stop threat detection engine"""
        try:
            logger.info("Stopping threat detection engine...")
            
            # Cancel all processing tasks
            for task in self.processing_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
            logger.info("Threat detection engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping threat detection engine: {e}")
    
    async def analyze_login_attempt(self, user_id: str, ip_address: str, 
                                  success: bool, timestamp: Optional[datetime] = None) -> Optional[ThreatEvent]:
        """Analyze login attempt for threats"""
        if not self.detection_enabled:
            return None
        
        timestamp = timestamp or datetime.now()
        
        try:
            # Behavioral analysis
            behavior_risk, behavior_details = await self.behavior_analyzer.analyze_login_behavior(
                user_id, ip_address, timestamp
            )
            
            # Geographic analysis
            geo_risk, geo_details = await self.geographic_analyzer.analyze_ip_location(ip_address)
            
            # Threat intelligence
            intel_risk, intel_details = await self.threat_intelligence.check_ip_reputation(ip_address)
            
            # Combine risk scores
            combined_risk = (behavior_risk * 0.4 + geo_risk * 0.3 + intel_risk * 0.3)
            
            # Failed login attempts increase risk
            if not success:
                combined_risk = min(combined_risk + 0.3, 1.0)
            
            # Determine threat level
            if combined_risk >= 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif combined_risk >= 0.6:
                threat_level = ThreatLevel.HIGH
            elif combined_risk >= 0.4:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW
            
            # Create threat event if above threshold
            if combined_risk >= self.detection_sensitivity:
                threat_event = ThreatEvent(
                    timestamp=timestamp,
                    threat_type=ThreatType.ANOMALOUS_LOGIN,
                    threat_level=threat_level,
                    source_ip=ip_address,
                    user_id=user_id,
                    risk_score=combined_risk,
                    confidence=0.8,
                    details={
                        'login_success': success,
                        'behavior_analysis': behavior_details,
                        'geographic_analysis': geo_details,
                        'intelligence_analysis': intel_details
                    }
                )
                
                # Queue for processing
                await self.event_queue.put(threat_event)
                return threat_event
            
            return None
            
        except Exception as e:
            logger.error(f"Login analysis failed: {e}")
            return None
    
    async def analyze_api_request(self, user_id: str, ip_address: str, 
                                endpoint: str, timestamp: Optional[datetime] = None) -> Optional[ThreatEvent]:
        """Analyze API request for threats"""
        if not self.detection_enabled:
            return None
        
        timestamp = timestamp or datetime.now()
        
        try:
            # Check if IP is already blocked or rate limited
            if self.automated_response.is_ip_blocked(ip_address):
                return ThreatEvent(
                    timestamp=timestamp,
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=ip_address,
                    user_id=user_id,
                    risk_score=0.9,
                    details={'reason': 'blocked_ip', 'endpoint': endpoint}
                )
            
            # Behavioral analysis
            behavior_risk, behavior_details = await self.behavior_analyzer.analyze_api_behavior(
                user_id, endpoint, timestamp
            )
            
            # Geographic analysis (cached from recent analysis)
            geo_risk = 0.1  # Simplified for API requests
            
            # Combine risk scores
            combined_risk = behavior_risk * 0.8 + geo_risk * 0.2
            
            # Determine threat level
            if combined_risk >= 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif combined_risk >= 0.6:
                threat_level = ThreatLevel.HIGH
            elif combined_risk >= 0.4:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW
            
            # Create threat event if above threshold
            if combined_risk >= self.detection_sensitivity:
                threat_event = ThreatEvent(
                    timestamp=timestamp,
                    threat_type=ThreatType.SUSPICIOUS_API_USAGE,
                    threat_level=threat_level,
                    source_ip=ip_address,
                    user_id=user_id,
                    risk_score=combined_risk,
                    confidence=0.7,
                    details={
                        'endpoint': endpoint,
                        'behavior_analysis': behavior_details
                    }
                )
                
                await self.event_queue.put(threat_event)
                return threat_event
            
            return None
            
        except Exception as e:
            logger.error(f"API request analysis failed: {e}")
            return None
    
    async def analyze_trading_activity(self, user_id: str, volume: float, 
                                     timestamp: Optional[datetime] = None) -> Optional[ThreatEvent]:
        """Analyze trading activity for threats"""
        if not self.detection_enabled:
            return None
        
        timestamp = timestamp or datetime.now()
        
        try:
            # Behavioral analysis
            behavior_risk, behavior_details = await self.behavior_analyzer.analyze_trading_behavior(
                user_id, volume, timestamp
            )
            
            # Determine threat level
            if behavior_risk >= 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif behavior_risk >= 0.6:
                threat_level = ThreatLevel.HIGH
            elif behavior_risk >= 0.4:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW
            
            # Create threat event if above threshold
            if behavior_risk >= self.detection_sensitivity:
                threat_event = ThreatEvent(
                    timestamp=timestamp,
                    threat_type=ThreatType.TRADING_PATTERN_DEVIATION,
                    threat_level=threat_level,
                    user_id=user_id,
                    risk_score=behavior_risk,
                    confidence=0.9,
                    details={
                        'volume': volume,
                        'behavior_analysis': behavior_details
                    }
                )
                
                await self.event_queue.put(threat_event)
                return threat_event
            
            return None
            
        except Exception as e:
            logger.error(f"Trading activity analysis failed: {e}")
            return None
    
    async def _process_events(self):
        """Process threat events from queue"""
        while True:
            try:
                # Get threat event from queue
                threat_event = await self.event_queue.get()
                
                # Store event in database
                await self._store_threat_event(threat_event)
                
                # Execute automated response
                if self.response_enabled:
                    actions = await self.automated_response.execute_response(threat_event)
                    threat_event.mitigation_applied = actions
                
                # Update statistics
                with self._lock:
                    self.stats['threats_detected'] += 1
                    if actions:
                        self.stats['responses_executed'] += 1
                
                logger.info(f"Processed threat event: {threat_event.event_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing failed: {e}")
    
    async def _store_threat_event(self, threat_event: ThreatEvent):
        """Store threat event in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO threat_events
                (event_id, timestamp, threat_type, threat_level, source_ip, user_id,
                 session_id, details, risk_score, confidence, false_positive_probability,
                 mitigation_applied, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                threat_event.event_id,
                threat_event.timestamp.isoformat(),
                threat_event.threat_type.value,
                threat_event.threat_level.value,
                threat_event.source_ip,
                threat_event.user_id,
                threat_event.session_id,
                json.dumps(threat_event.details),
                threat_event.risk_score,
                threat_event.confidence,
                threat_event.false_positive_probability,
                json.dumps([action.value for action in threat_event.mitigation_applied]),
                int(threat_event.resolved)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store threat event: {e}")
    
    async def _periodic_intelligence_update(self):
        """Periodically update threat intelligence"""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour
                await self.threat_intelligence.update_threat_feeds()
                
                with self._lock:
                    self.stats['last_threat_feed_update'] = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Threat intelligence update failed: {e}")
    
    async def get_detection_stats(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        with self._lock:
            stats = self.stats.copy()
        
        # Add real-time stats
        stats.update({
            'detection_enabled': self.detection_enabled,
            'response_enabled': self.response_enabled,
            'detection_sensitivity': self.detection_sensitivity,
            'blocked_ips': len(self.automated_response.blocked_ips),
            'rate_limited_ips': len(self.automated_response.rate_limited_ips),
            'locked_accounts': len(self.automated_response.locked_accounts),
            'queue_size': self.event_queue.qsize()
        })
        
        return stats

# Example usage and testing
async def main():
    """Example usage of Advanced Threat Detection System"""
    
    print("🛡️ Advanced Threat Detection System Test")
    print("=" * 50)
    
    # Initialize threat detection engine
    detection_engine = ThreatDetectionEngine()
    
    try:
        # Start detection engine
        print("\n1. Starting threat detection engine...")
        await detection_engine.start_detection()
        print("✅ Threat detection engine started")
        
        # Simulate login attempts
        print("\n2. Simulating login attempts...")
        
        # Normal login
        threat1 = await detection_engine.analyze_login_attempt(
            user_id="user123",
            ip_address="192.168.1.100",
            success=True
        )
        
        if threat1:
            print(f"   ⚠️  Threat detected: {threat1.threat_type.value} (Risk: {threat1.risk_score:.2f})")
        else:
            print("   ✅ Normal login - no threat detected")
        
        # Suspicious login from new location
        threat2 = await detection_engine.analyze_login_attempt(
            user_id="user123",
            ip_address="1.2.3.4",  # Different IP
            success=True
        )
        
        if threat2:
            print(f"   ⚠️  Threat detected: {threat2.threat_type.value} (Risk: {threat2.risk_score:.2f})")
        else:
            print("   ✅ Login from new location - no threat detected")
        
        # Failed login attempt
        threat3 = await detection_engine.analyze_login_attempt(
            user_id="user123",
            ip_address="5.6.7.8",
            success=False
        )
        
        if threat3:
            print(f"   ⚠️  Threat detected: {threat3.threat_type.value} (Risk: {threat3.risk_score:.2f})")
        else:
            print("   ✅ Failed login - no threat detected")
        
        # Simulate API requests
        print("\n3. Simulating API requests...")
        
        threat4 = await detection_engine.analyze_api_request(
            user_id="user123",
            ip_address="192.168.1.100",
            endpoint="/api/v1/account/balance"
        )
        
        if threat4:
            print(f"   ⚠️  API threat detected: {threat4.threat_type.value} (Risk: {threat4.risk_score:.2f})")
        else:
            print("   ✅ Normal API request - no threat detected")
        
        # Simulate trading activity
        print("\n4. Simulating trading activity...")
        
        # Normal volume
        threat5 = await detection_engine.analyze_trading_activity(
            user_id="user123",
            volume=1000.0
        )
        
        if threat5:
            print(f"   ⚠️  Trading threat detected: {threat5.threat_type.value} (Risk: {threat5.risk_score:.2f})")
        else:
            print("   ✅ Normal trading volume - no threat detected")
        
        # Wait for event processing
        print("\n5. Processing threat events...")
        await asyncio.sleep(2)
        
        # Get detection statistics
        print("\n6. Detection statistics:")
        stats = await detection_engine.get_detection_stats()
        
        for key, value in stats.items():
            if isinstance(value, datetime):
                value = value.isoformat()
            print(f"   {key}: {value}")
        
        print("\n✅ Advanced Threat Detection System test completed!")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
    
    finally:
        # Stop detection engine
        await detection_engine.stop_detection()

if __name__ == "__main__":
    asyncio.run(main())