"""
Rate Limiter Utility

Implements token bucket rate limiting algorithm for API requests
to ensure compliance with exchange rate limits and prevent API bans.
"""

import asyncio
import time
from collections import deque
from typing import Optional, Dict, Any
import logging


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    
    Features:
    - Token bucket algorithm with burst support
    - Configurable time windows and request limits
    - Async/await support for non-blocking operation
    - Per-endpoint rate limiting support
    - Request queue management
    """
    
    def __init__(
        self,
        max_requests: int,
        time_window: int = 60,
        burst_limit: Optional[int] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds (default: 60)
            burst_limit: Maximum burst requests (default: max_requests)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.burst_limit = burst_limit or max_requests
        
        # Token bucket state
        self.tokens = self.max_requests
        self.last_refill = time.time()
        
        # Request tracking
        self.request_times = deque()
        
        # Synchronization
        self._lock = asyncio.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens for making requests.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        async with self._lock:
            await self._wait_for_tokens(tokens)
            self._consume_tokens(tokens)
    
    async def _wait_for_tokens(self, tokens: int) -> None:
        """
        Wait until sufficient tokens are available.
        
        Args:
            tokens: Number of tokens needed
        """
        while True:
            self._refill_tokens()
            
            if self.tokens >= tokens:
                break
            
            # Calculate wait time
            wait_time = self._calculate_wait_time(tokens)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        time_elapsed = now - self.last_refill
        
        if time_elapsed <= 0:
            return
        
        # Calculate tokens to add
        tokens_to_add = (time_elapsed / self.time_window) * self.max_requests
        self.tokens = min(self.max_requests, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def _consume_tokens(self, tokens: int) -> None:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
        """
        self.tokens -= tokens
        self.request_times.append(time.time())
        
        # Clean old request times
        cutoff_time = time.time() - self.time_window
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
    
    def _calculate_wait_time(self, tokens: int) -> float:
        """
        Calculate how long to wait for tokens.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            float: Wait time in seconds
        """
        tokens_needed = tokens - self.tokens
        if tokens_needed <= 0:
            return 0
        
        # Time to generate needed tokens
        time_per_token = self.time_window / self.max_requests
        return tokens_needed * time_per_token
    
    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current rate limiter usage statistics.
        
        Returns:
            Dict[str, Any]: Usage statistics
        """
        self._refill_tokens()
        
        return {
            'tokens_available': int(self.tokens),
            'max_tokens': self.max_requests,
            'utilization_percent': ((self.max_requests - self.tokens) / self.max_requests) * 100,
            'requests_in_window': len(self.request_times),
            'time_to_full_refill': (self.max_requests - self.tokens) * (self.time_window / self.max_requests)
        }


class MultiEndpointRateLimiter:
    """
    Rate limiter that supports different limits for different endpoints.
    
    Useful for exchanges that have different rate limits for different
    types of API calls (e.g., market data vs trading operations).
    """
    
    def __init__(self, endpoint_limits: Dict[str, Dict[str, int]]):
        """
        Initialize multi-endpoint rate limiter.
        
        Args:
            endpoint_limits: Dict mapping endpoint patterns to rate limits
                Format: {
                    'endpoint_pattern': {
                        'max_requests': int,
                        'time_window': int,
                        'burst_limit': int (optional)
                    }
                }
        """
        self.limiters: Dict[str, RateLimiter] = {}
        
        for endpoint, limits in endpoint_limits.items():
            self.limiters[endpoint] = RateLimiter(
                max_requests=limits['max_requests'],
                time_window=limits.get('time_window', 60),
                burst_limit=limits.get('burst_limit')
            )
        
        # Default limiter for unspecified endpoints
        self.default_limiter = RateLimiter(max_requests=10, time_window=60)
        
        self.logger = logging.getLogger(__name__)
    
    async def acquire(self, endpoint: str, tokens: int = 1) -> None:
        """
        Acquire tokens for a specific endpoint.
        
        Args:
            endpoint: API endpoint pattern
            tokens: Number of tokens to acquire
        """
        limiter = self._get_limiter_for_endpoint(endpoint)
        await limiter.acquire(tokens)
    
    def _get_limiter_for_endpoint(self, endpoint: str) -> RateLimiter:
        """
        Get the appropriate rate limiter for an endpoint.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            RateLimiter: Rate limiter for the endpoint
        """
        # Try exact match first
        if endpoint in self.limiters:
            return self.limiters[endpoint]
        
        # Try pattern matching
        for pattern, limiter in self.limiters.items():
            if pattern in endpoint:
                return limiter
        
        # Fall back to default limiter
        self.logger.warning(f"No specific rate limiter for {endpoint}, using default")
        return self.default_limiter
    
    def get_all_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage statistics for all rate limiters.
        
        Returns:
            Dict[str, Dict[str, Any]]: Usage stats by endpoint
        """
        stats = {}
        
        for endpoint, limiter in self.limiters.items():
            stats[endpoint] = limiter.get_current_usage()
        
        stats['default'] = self.default_limiter.get_current_usage()
        
        return stats


# Bybit-specific rate limiter configuration
BYBIT_RATE_LIMITS = {
    '/v5/market/kline': {
        'max_requests': 120,
        'time_window': 60,
        'burst_limit': 10
    },
    '/v5/market/tickers': {
        'max_requests': 120,
        'time_window': 60,
        'burst_limit': 10
    },
    '/v5/market/funding/history': {
        'max_requests': 120,
        'time_window': 60,
        'burst_limit': 10
    },
    '/v5/account/wallet-balance': {
        'max_requests': 120,
        'time_window': 60,
        'burst_limit': 5
    },
    '/v5/position/list': {
        'max_requests': 120,
        'time_window': 60,
        'burst_limit': 5
    },
    '/v5/order/create': {
        'max_requests': 100,
        'time_window': 60,
        'burst_limit': 3
    },
    '/v5/order/cancel': {
        'max_requests': 100,
        'time_window': 60,
        'burst_limit': 3
    }
}


def create_bybit_rate_limiter() -> MultiEndpointRateLimiter:
    """
    Create a rate limiter configured for Bybit API limits.
    
    Returns:
        MultiEndpointRateLimiter: Configured rate limiter
    """
    return MultiEndpointRateLimiter(BYBIT_RATE_LIMITS)