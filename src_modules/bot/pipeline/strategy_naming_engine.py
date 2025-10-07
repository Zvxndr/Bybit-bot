"""
Strategy Naming Engine for AI Pipeline System

Generates unique, human-readable strategy identifiers in the format:
ASSET_TYPE_ID (e.g., BTC_MR_A4F2D, ETH_BB_C7E9A)

Components:
- ASSET: Base asset from USDT pairs (BTC, ETH, SOL, etc.)
- TYPE: Strategy type abbreviation (MR, BB, RSI, MA, MACD)
- ID: Unique 5-character alphanumeric identifier

Author: Trading Bot Team
Version: 1.0.0
"""

import hashlib
import random
import string
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from ..ml_strategy_discovery.ml_engine import StrategyType


class StrategyTypeCode(Enum):
    """Strategy type abbreviations for naming."""
    TREND_FOLLOWING = "TF"
    MEAN_REVERSION = "MR"
    MOMENTUM = "MM"
    VOLATILITY = "VL"
    MULTI_FACTOR = "MF"
    BOLLINGER_BANDS = "BB"
    RSI = "RSI"
    MACD = "MACD"
    MOVING_AVERAGE = "MA"
    STOCHASTIC = "STOC"
    WILLIAMS_R = "WR"
    CCI = "CCI"
    ADX = "ADX"
    ICHIMOKU = "ICH"
    FIBONACCI = "FIB"


@dataclass
class StrategyName:
    """Strategy name components."""
    asset: str
    type_code: str
    unique_id: str
    full_name: str
    
    @property
    def pipeline_phase(self) -> str:
        """Determine pipeline phase from ID pattern."""
        # This could be extended to encode phase in the ID
        return "backtest"  # Default for new strategies


class StrategyNamingEngine:
    """
    Engine for generating unique strategy identifiers.
    
    Maintains consistency and uniqueness across the pipeline while
    providing human-readable names that encode strategy information.
    """
    
    def __init__(self):
        self.used_ids: Set[str] = set()
        self.asset_mappings = {
            # Primary USDT pairs
            'BTCUSDT': 'BTC',
            'ETHUSDT': 'ETH', 
            'SOLUSDT': 'SOL',
            'ADAUSDT': 'ADA',
            'DOTUSDT': 'DOT',
            
            # Secondary USDT pairs
            'MATICUSDT': 'MATIC',
            'AVAXUSDT': 'AVAX',
            'LINKUSDT': 'LINK',
            'UNIUSDT': 'UNI',
            'BNBUSDT': 'BNB',
            'XRPUSDT': 'XRP',
            'LTCUSDT': 'LTC',
            'ATOMUSDT': 'ATOM',
            'NEARUSDT': 'NEAR',
            'FTMUSDT': 'FTM',
        }
        
        self.type_mappings = {
            # ML Strategy types
            StrategyType.TREND_FOLLOWING: StrategyTypeCode.TREND_FOLLOWING,
            StrategyType.MEAN_REVERSION: StrategyTypeCode.MEAN_REVERSION,
            StrategyType.MOMENTUM: StrategyTypeCode.MOMENTUM,
            StrategyType.VOLATILITY: StrategyTypeCode.VOLATILITY,
            StrategyType.MULTI_FACTOR: StrategyTypeCode.MULTI_FACTOR,
            
            # Technical indicator strategies (string matching)
            'bollinger_bands': StrategyTypeCode.BOLLINGER_BANDS,
            'rsi': StrategyTypeCode.RSI,
            'macd': StrategyTypeCode.MACD,
            'moving_average': StrategyTypeCode.MOVING_AVERAGE,
            'stochastic': StrategyTypeCode.STOCHASTIC,
            'williams_r': StrategyTypeCode.WILLIAMS_R,
            'cci': StrategyTypeCode.CCI,
            'adx': StrategyTypeCode.ADX,
            'ichimoku': StrategyTypeCode.ICHIMOKU,
            'fibonacci': StrategyTypeCode.FIBONACCI,
        }
    
    def generate_strategy_name(
        self,
        asset_pair: str,
        strategy_type: Optional[str] = None,
        strategy_description: Optional[str] = None,
        seed: Optional[str] = None
    ) -> StrategyName:
        """
        Generate a unique strategy name.
        
        Args:
            asset_pair: Trading pair (e.g., 'BTCUSDT')
            strategy_type: Strategy type or description
            strategy_description: Additional strategy details
            seed: Optional seed for reproducible ID generation
            
        Returns:
            StrategyName object with components and full name
        """
        
        # Extract asset from USDT pair
        asset = self._extract_asset(asset_pair)
        
        # Determine strategy type code
        type_code = self._determine_type_code(strategy_type, strategy_description)
        
        # Generate unique ID
        unique_id = self._generate_unique_id(asset, type_code, seed)
        
        # Construct full name
        full_name = f"{asset}_{type_code.value}_{unique_id}"
        
        # Track used ID
        self.used_ids.add(full_name)
        
        return StrategyName(
            asset=asset,
            type_code=type_code.value,
            unique_id=unique_id,
            full_name=full_name
        )
    
    def _extract_asset(self, asset_pair: str) -> str:
        """Extract base asset from USDT trading pair."""
        if asset_pair in self.asset_mappings:
            return self.asset_mappings[asset_pair]
        
        # Fallback: try to extract from USDT suffix
        if asset_pair.endswith('USDT'):
            base_asset = asset_pair.replace('USDT', '')
            return base_asset if len(base_asset) <= 6 else base_asset[:6]
        
        # Final fallback
        return asset_pair[:6].upper()
    
    def _determine_type_code(
        self, 
        strategy_type: Optional[str],
        description: Optional[str]
    ) -> StrategyTypeCode:
        """Determine strategy type code from available information."""
        
        # Try direct mapping from StrategyType enum
        if isinstance(strategy_type, StrategyType):
            return self.type_mappings[strategy_type]
        
        # Try string matching
        if strategy_type and strategy_type.lower() in self.type_mappings:
            return self.type_mappings[strategy_type.lower()]
        
        # Try description-based detection
        if description:
            desc_lower = description.lower()
            
            if 'bollinger' in desc_lower or 'bb' in desc_lower:
                return StrategyTypeCode.BOLLINGER_BANDS
            elif 'rsi' in desc_lower:
                return StrategyTypeCode.RSI
            elif 'macd' in desc_lower:
                return StrategyTypeCode.MACD
            elif 'moving average' in desc_lower or 'ma' in desc_lower:
                return StrategyTypeCode.MOVING_AVERAGE
            elif 'mean reversion' in desc_lower:
                return StrategyTypeCode.MEAN_REVERSION
            elif 'momentum' in desc_lower:
                return StrategyTypeCode.MOMENTUM
            elif 'trend' in desc_lower:
                return StrategyTypeCode.TREND_FOLLOWING
            elif 'volatility' in desc_lower:
                return StrategyTypeCode.VOLATILITY
        
        # Default to multi-factor for complex strategies
        return StrategyTypeCode.MULTI_FACTOR
    
    def _generate_unique_id(
        self, 
        asset: str, 
        type_code: StrategyTypeCode,
        seed: Optional[str] = None
    ) -> str:
        """Generate unique 5-character alphanumeric ID."""
        
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            if seed:
                # Deterministic generation for reproducibility
                hash_input = f"{asset}_{type_code.value}_{seed}_{attempt}"
                hash_obj = hashlib.md5(hash_input.encode())
                hex_hash = hash_obj.hexdigest()
                
                # Convert to alphanumeric
                unique_id = self._hex_to_alphanumeric(hex_hash[:5])
            else:
                # Random generation
                unique_id = ''.join(
                    random.choices(
                        string.ascii_uppercase + string.digits,
                        k=5
                    )
                )
            
            # Check uniqueness
            full_name = f"{asset}_{type_code.value}_{unique_id}"
            if full_name not in self.used_ids:
                return unique_id
        
        # Fallback: use timestamp-based ID
        timestamp = str(int(datetime.now().timestamp()))[-5:]
        return self._hex_to_alphanumeric(timestamp)
    
    def _hex_to_alphanumeric(self, hex_string: str) -> str:
        """Convert hex string to alphanumeric."""
        # Map hex digits to alphanumeric
        mapping = {
            'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F',
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8', '9': '9'
        }
        
        result = ''
        for char in hex_string.lower():
            if char in mapping:
                result += mapping[char]
            else:
                result += random.choice(string.ascii_uppercase)
        
        return result.upper()
    
    def parse_strategy_name(self, strategy_name: str) -> Optional[StrategyName]:
        """Parse an existing strategy name into components."""
        
        parts = strategy_name.split('_')
        if len(parts) != 3:
            return None
        
        asset, type_code, unique_id = parts
        
        # Validate format
        if (len(asset) <= 6 and 
            len(type_code) <= 4 and 
            len(unique_id) == 5 and 
            unique_id.isalnum()):
            
            return StrategyName(
                asset=asset,
                type_code=type_code,
                unique_id=unique_id,
                full_name=strategy_name
            )
        
        return None
    
    def is_valid_name(self, strategy_name: str) -> bool:
        """Check if a strategy name follows the correct format."""
        return self.parse_strategy_name(strategy_name) is not None
    
    def add_existing_name(self, strategy_name: str) -> bool:
        """Add an existing strategy name to the used set."""
        if self.is_valid_name(strategy_name):
            self.used_ids.add(strategy_name)
            return True
        return False
    
    def get_asset_strategies(self, asset: str) -> List[str]:
        """Get all strategy names for a specific asset."""
        return [
            name for name in self.used_ids 
            if name.startswith(f"{asset}_")
        ]
    
    def get_type_strategies(self, type_code: str) -> List[str]:
        """Get all strategy names for a specific type."""
        return [
            name for name in self.used_ids 
            if f"_{type_code}_" in name
        ]
    
    def get_stats(self) -> Dict[str, int]:
        """Get naming engine statistics."""
        assets = set()
        types = set()
        
        for name in self.used_ids:
            parsed = self.parse_strategy_name(name)
            if parsed:
                assets.add(parsed.asset)
                types.add(parsed.type_code)
        
        return {
            'total_strategies': len(self.used_ids),
            'unique_assets': len(assets),
            'unique_types': len(types),
            'assets': sorted(list(assets)),
            'types': sorted(list(types))
        }


# Global instance
strategy_naming_engine = StrategyNamingEngine()


# Example usage and testing
if __name__ == "__main__":
    # Example strategy name generation
    engine = StrategyNamingEngine()
    
    # Test different strategy types
    test_cases = [
        ('BTCUSDT', 'mean_reversion', 'Bollinger Band mean reversion strategy'),
        ('ETHUSDT', 'rsi', 'RSI momentum strategy'),
        ('SOLUSDT', StrategyType.MOMENTUM, 'ML momentum strategy'),
        ('ADAUSDT', 'moving_average', 'Moving average crossover'),
        ('DOTUSDT', 'macd', 'MACD signal strategy'),
    ]
    
    print("Strategy Naming Engine Test:")
    print("=" * 50)
    
    for asset_pair, strategy_type, description in test_cases:
        name = engine.generate_strategy_name(
            asset_pair=asset_pair,
            strategy_type=strategy_type,
            strategy_description=description
        )
        
        print(f"Asset: {asset_pair}")
        print(f"Type: {strategy_type}")
        print(f"Description: {description}")
        print(f"Generated Name: {name.full_name}")
        print(f"Components: {name.asset} | {name.type_code} | {name.unique_id}")
        print("-" * 50)
    
    print(f"\nEngine Stats: {engine.get_stats()}")