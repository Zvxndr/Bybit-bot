"""
Security Middleware
==================

Provides IP whitelisting, rate limiting, and security controls for the trading bot.
Includes Redis-backed rate limiting and IP access controls.
"""

import ipaddress
import redis
import logging
import hashlib
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different endpoints"""
    PUBLIC = "public"          # No restrictions
    PROTECTED = "protected"    # IP whitelist required
    ADMIN = "admin"           # IP whitelist + MFA required
    CRITICAL = "critical"     # Maximum security


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    requests: int              # Number of requests
    window_seconds: int        # Time window in seconds
    per_ip: bool = True       # Apply per IP or globally
    
    def __str__(self):
        scope = "per IP" if self.per_ip else "globally"
        return f"{self.requests} requests per {self.window_seconds}s {scope}"


@dataclass
class SecurityConfig:
    """Security configuration for endpoints"""
    level: SecurityLevel
    rate_limits: List[RateLimitRule]
    allowed_networks: List[str] = None
    require_mfa: bool = False
    
    def __post_init__(self):
        if self.allowed_networks is None:
            self.allowed_networks = []


class SecurityMiddleware:
    """Security middleware for request validation and rate limiting"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_config: SecurityConfig = None):
        """
        Initialize security middleware
        
        Args:
            redis_url: Redis connection URL
            default_config: Default security configuration
        """
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis.ping()
            logger.info("✅ Connected to Redis for security middleware")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {str(e)}")
            logger.info("🔄 Using in-memory rate limiting (not recommended for production)")
            self.redis = None
            self._memory_cache = {}
        
        # Default security configuration
        self.default_config = default_config or SecurityConfig(
            level=SecurityLevel.PROTECTED,
            rate_limits=[
                RateLimitRule(requests=100, window_seconds=3600),  # 100 per hour
                RateLimitRule(requests=10, window_seconds=60),     # 10 per minute
            ]
        )
        
        # Endpoint-specific configurations
        self.endpoint_configs: Dict[str, SecurityConfig] = {}
        
        # Load default IP whitelist from environment or config
        self._load_default_networks()
    
    def _load_default_networks(self):
        """Load default allowed networks"""
        # These should be configured via environment variables in production
        default_networks = [
            "127.0.0.1/32",        # Localhost
            "10.0.0.0/8",          # Private network
            "172.16.0.0/12",       # Private network
            "192.168.0.0/16",      # Private network
        ]
        
        self.default_allowed_networks = []
        for network_str in default_networks:
            try:
                network = ipaddress.ip_network(network_str)
                self.default_allowed_networks.append(network)
            except ValueError as e:
                logger.warning(f"Invalid network in default config: {network_str} - {e}")
    
    def configure_endpoint(self, endpoint: str, config: SecurityConfig):
        """Configure security for specific endpoint"""
        self.endpoint_configs[endpoint] = config
        logger.info(f"Security configured for endpoint '{endpoint}': {config.level.value}")
    
    def add_allowed_network(self, network_str: str, endpoint: str = None):
        """Add allowed network for endpoint or globally"""
        try:
            network = ipaddress.ip_network(network_str)
            
            if endpoint and endpoint in self.endpoint_configs:
                if network_str not in self.endpoint_configs[endpoint].allowed_networks:
                    self.endpoint_configs[endpoint].allowed_networks.append(network_str)
            else:
                self.default_allowed_networks.append(network)
            
            logger.info(f"Added allowed network: {network_str}" + (f" for endpoint: {endpoint}" if endpoint else " globally"))
            
        except ValueError as e:
            logger.error(f"Invalid network format: {network_str} - {e}")
            raise
    
    async def check_ip_whitelist(self, request_ip: str, endpoint: str = None) -> Dict[str, Union[bool, str]]:
        """
        Check if IP is whitelisted for endpoint
        
        Args:
            request_ip: Client IP address
            endpoint: Specific endpoint (uses default if None)
            
        Returns:
            Dictionary with check result and details
        """
        try:
            client_ip = ipaddress.ip_address(request_ip)
            
            # Get endpoint configuration
            config = self.endpoint_configs.get(endpoint, self.default_config)
            
            # Check if security level requires IP whitelisting
            if config.level == SecurityLevel.PUBLIC:
                return {'allowed': True, 'reason': 'Public endpoint'}
            
            # Build list of allowed networks
            allowed_networks = self.default_allowed_networks.copy()
            
            # Add endpoint-specific networks
            for network_str in config.allowed_networks:
                try:
                    network = ipaddress.ip_network(network_str)
                    allowed_networks.append(network)
                except ValueError as e:
                    logger.warning(f"Invalid network in endpoint config: {network_str} - {e}")
            
            # Check if IP is in any allowed network
            for network in allowed_networks:
                if client_ip in network:
                    return {
                        'allowed': True,
                        'reason': f'IP {request_ip} allowed by network {network}',
                        'network': str(network)
                    }
            
            # IP not whitelisted
            logger.warning(f"IP access denied: {request_ip} for endpoint: {endpoint}")
            return {
                'allowed': False,
                'reason': f'IP {request_ip} not in allowed networks',
                'allowed_networks': [str(net) for net in allowed_networks]
            }
            
        except ValueError as e:
            logger.error(f"Invalid IP address: {request_ip} - {e}")
            return {'allowed': False, 'reason': f'Invalid IP address: {request_ip}'}
    
    async def check_rate_limit(self, ip: str, endpoint: str, user_id: str = None) -> Dict[str, Union[bool, str, int]]:
        """
        Check rate limits for IP/endpoint combination
        
        Args:
            ip: Client IP address
            endpoint: Endpoint being accessed
            user_id: Optional user identifier
            
        Returns:
            Dictionary with rate limit check result
        """
        try:
            # Get endpoint configuration
            config = self.endpoint_configs.get(endpoint, self.default_config)
            
            current_time = datetime.now()
            
            for rule in config.rate_limits:
                # Create cache key
                if rule.per_ip:
                    cache_key = f"rate_limit:{ip}:{endpoint}:{rule.window_seconds}"
                else:
                    cache_key = f"rate_limit:global:{endpoint}:{rule.window_seconds}"
                
                # Check current usage
                if self.redis:
                    current_count = await self._check_redis_rate_limit(cache_key, rule, current_time)
                else:
                    current_count = await self._check_memory_rate_limit(cache_key, rule, current_time)
                
                # Check if limit exceeded
                if current_count > rule.requests:
                    logger.warning(f"Rate limit exceeded for {ip}:{endpoint} - {current_count}/{rule.requests}")
                    return {
                        'allowed': False,
                        'reason': f'Rate limit exceeded: {current_count}/{rule.requests} in {rule.window_seconds}s',
                        'current_count': current_count,
                        'limit': rule.requests,
                        'window_seconds': rule.window_seconds,
                        'retry_after': rule.window_seconds
                    }
            
            return {'allowed': True, 'reason': 'Within rate limits'}
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            # Fail open for rate limiting errors (but log them)
            return {'allowed': True, 'reason': f'Rate limit check error: {str(e)}'}
    
    async def _check_redis_rate_limit(self, cache_key: str, rule: RateLimitRule, current_time: datetime) -> int:
        """Check rate limit using Redis"""
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Increment counter
            pipe.incr(cache_key)
            pipe.expire(cache_key, rule.window_seconds)
            
            results = pipe.execute()
            current_count = results[0]
            
            return current_count
            
        except Exception as e:
            logger.error(f"Redis rate limit error: {str(e)}")
            # Fallback to allowing request
            return 0
    
    async def _check_memory_rate_limit(self, cache_key: str, rule: RateLimitRule, current_time: datetime) -> int:
        """Check rate limit using in-memory cache (fallback)"""
        try:
            # Clean expired entries
            cutoff_time = current_time - timedelta(seconds=rule.window_seconds)
            
            if cache_key not in self._memory_cache:
                self._memory_cache[cache_key] = []
            
            # Remove expired timestamps
            self._memory_cache[cache_key] = [
                timestamp for timestamp in self._memory_cache[cache_key]
                if timestamp > cutoff_time
            ]
            
            # Add current request
            self._memory_cache[cache_key].append(current_time)
            
            return len(self._memory_cache[cache_key])
            
        except Exception as e:
            logger.error(f"Memory rate limit error: {str(e)}")
            return 0
    
    async def validate_request(self, ip: str, endpoint: str, user_id: str = None, 
                             mfa_verified: bool = False) -> Dict[str, Union[bool, str, Dict]]:
        """
        Comprehensive request validation
        
        Args:
            ip: Client IP address
            endpoint: Endpoint being accessed
            user_id: Optional user identifier
            mfa_verified: Whether MFA has been verified
            
        Returns:
            Dictionary with validation result and details
        """
        # Get endpoint configuration
        config = self.endpoint_configs.get(endpoint, self.default_config)
        
        validation_result = {
            'allowed': True,
            'security_level': config.level.value,
            'checks': {}
        }
        
        # 1. IP Whitelist Check
        if config.level != SecurityLevel.PUBLIC:
            ip_check = await self.check_ip_whitelist(ip, endpoint)
            validation_result['checks']['ip_whitelist'] = ip_check
            
            if not ip_check['allowed']:
                validation_result['allowed'] = False
                validation_result['reason'] = ip_check['reason']
                return validation_result
        
        # 2. Rate Limit Check
        rate_check = await self.check_rate_limit(ip, endpoint, user_id)
        validation_result['checks']['rate_limit'] = rate_check
        
        if not rate_check['allowed']:
            validation_result['allowed'] = False
            validation_result['reason'] = rate_check['reason']
            validation_result['retry_after'] = rate_check.get('retry_after')
            return validation_result
        
        # 3. MFA Check (for admin/critical endpoints)
        if config.require_mfa or config.level in [SecurityLevel.ADMIN, SecurityLevel.CRITICAL]:
            if not mfa_verified:
                validation_result['allowed'] = False
                validation_result['reason'] = 'MFA verification required'
                validation_result['require_mfa'] = True
                return validation_result
            
            validation_result['checks']['mfa'] = {'verified': True}
        
        validation_result['reason'] = 'All security checks passed'
        return validation_result
    
    def get_security_stats(self) -> Dict[str, Union[int, Dict]]:
        """Get security middleware statistics"""
        stats = {
            'endpoint_configs': len(self.endpoint_configs),
            'default_networks': len(self.default_allowed_networks),
            'redis_connected': self.redis is not None,
            'endpoints': {}
        }
        
        # Add endpoint details
        for endpoint, config in self.endpoint_configs.items():
            stats['endpoints'][endpoint] = {
                'security_level': config.level.value,
                'rate_limits': len(config.rate_limits),
                'allowed_networks': len(config.allowed_networks),
                'require_mfa': config.require_mfa
            }
        
        return stats


# Example usage and configuration
if __name__ == "__main__":
    import asyncio
    
    async def test_security_middleware():
        # Initialize security middleware
        security = SecurityMiddleware()
        
        # Configure different endpoints
        security.configure_endpoint("/api/public", SecurityConfig(
            level=SecurityLevel.PUBLIC,
            rate_limits=[RateLimitRule(requests=1000, window_seconds=3600)]
        ))
        
        security.configure_endpoint("/api/admin", SecurityConfig(
            level=SecurityLevel.ADMIN,
            rate_limits=[
                RateLimitRule(requests=50, window_seconds=3600),
                RateLimitRule(requests=5, window_seconds=60)
            ],
            require_mfa=True
        ))
        
        security.configure_endpoint("/api/trading", SecurityConfig(
            level=SecurityLevel.CRITICAL,
            rate_limits=[RateLimitRule(requests=100, window_seconds=3600)],
            require_mfa=True
        ))
        
        # Test IP whitelisting
        test_ips = ["127.0.0.1", "192.168.1.100", "8.8.8.8"]
        
        for test_ip in test_ips:
            ip_result = await security.check_ip_whitelist(test_ip, "/api/admin")
            print(f"IP {test_ip}: {ip_result}")
        
        # Test rate limiting
        for i in range(3):
            rate_result = await security.check_rate_limit("127.0.0.1", "/api/admin")
            print(f"Rate limit check {i+1}: {rate_result}")
        
        # Test full validation
        validation = await security.validate_request(
            ip="127.0.0.1",
            endpoint="/api/admin",
            user_id="admin",
            mfa_verified=True
        )
        print(f"Full validation: {validation}")
        
        # Get stats
        stats = security.get_security_stats()
        print(f"Security stats: {json.dumps(stats, indent=2)}")
    
    # Run test
    asyncio.run(test_security_middleware())