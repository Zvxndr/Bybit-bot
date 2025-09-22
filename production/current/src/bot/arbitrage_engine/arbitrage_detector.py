"""
Opportunistic Arbitrage Detection Engine
Designed for Australian traders with balance-tiered activation
Focuses on capital efficiency as secondary strategy to ML-first approach
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import aiohttp

logger = logging.getLogger(__name__)

class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    SIMPLE_ARBITRAGE = "simple"          # Buy low exchange, sell high exchange
    TRIANGULAR_ARBITRAGE = "triangular"  # Three-currency arbitrage loop
    FUNDING_ARBITRAGE = "funding"        # Spot vs perpetual funding rate arbitrage
    CROSS_EXCHANGE = "cross_exchange"    # Inter-exchange price differences

class OpportunityTier(Enum):
    """Opportunity tiers based on balance and risk"""
    MICRO = "micro"          # < $10,000 AUD
    SMALL = "small"          # $10,000 - $50,000 AUD  
    MEDIUM = "medium"        # $50,000 - $200,000 AUD
    LARGE = "large"          # $200,000+ AUD

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    tier: OpportunityTier
    
    # Exchange and symbol information
    buy_exchange: str
    sell_exchange: str
    symbol: str
    
    # Price and profit information
    buy_price: Decimal
    sell_price: Decimal
    price_difference: Decimal
    gross_profit_percentage: Decimal
    
    # Cost analysis
    transfer_costs: Decimal
    trading_fees: Decimal
    total_costs: Decimal
    net_profit_percentage: Decimal
    
    # Execution requirements
    minimum_amount: Decimal
    maximum_amount: Decimal 
    estimated_execution_time: int  # minutes
    
    # Risk factors
    liquidity_score: float  # 0-1, higher is better
    volatility_risk: float  # 0-1, higher is riskier
    australian_friendly: bool
    
    # Timestamps
    detected_at: datetime
    expires_at: datetime
    
    def is_profitable(self) -> bool:
        """Check if opportunity is profitable after all costs"""
        return self.net_profit_percentage > Decimal('0.1')  # Minimum 0.1% profit
    
    def is_valid(self) -> bool:
        """Check if opportunity is still valid (not expired)"""
        return datetime.now() < self.expires_at

@dataclass
class BalanceTier:
    """Balance-based tier configuration"""
    name: OpportunityTier
    min_balance_aud: Decimal
    max_balance_aud: Optional[Decimal]
    
    # Risk parameters
    max_position_percentage: Decimal  # Max % of balance per trade
    min_profit_percentage: Decimal    # Minimum profit threshold
    max_execution_time_minutes: int   # Maximum acceptable execution time
    
    # Australian-specific parameters
    requires_australian_exchange: bool
    max_international_exposure: Decimal  # Max % on international exchanges

class AustralianArbitrageDetector:
    """
    Australian-focused arbitrage detection
    Considers local regulations, banking, and tax implications
    """
    
    def __init__(self):
        self.tier_configurations = self._initialize_tier_configs()
        self.exchange_fees = self._initialize_exchange_fees()
        self.australian_exchanges = {'btcmarkets', 'coinjar', 'swyftx'}
        self.international_exchanges = {'bybit', 'binance', 'kraken', 'coinbase'}
        
        logger.info("Initialized Australian Arbitrage Detector")
    
    def _initialize_tier_configs(self) -> Dict[OpportunityTier, BalanceTier]:
        """Initialize balance tier configurations"""
        
        return {
            OpportunityTier.MICRO: BalanceTier(
                name=OpportunityTier.MICRO,
                min_balance_aud=Decimal('1000'),
                max_balance_aud=Decimal('10000'),
                max_position_percentage=Decimal('0.30'),  # 30% max per trade
                min_profit_percentage=Decimal('0.5'),     # 0.5% minimum profit
                max_execution_time_minutes=120,           # 2 hours max
                requires_australian_exchange=True,        # Must use Australian exchange
                max_international_exposure=Decimal('0.2') # 20% international max
            ),
            
            OpportunityTier.SMALL: BalanceTier(
                name=OpportunityTier.SMALL,
                min_balance_aud=Decimal('10000'),
                max_balance_aud=Decimal('50000'),
                max_position_percentage=Decimal('0.25'),  # 25% max per trade
                min_profit_percentage=Decimal('0.3'),     # 0.3% minimum profit
                max_execution_time_minutes=90,            # 1.5 hours max
                requires_australian_exchange=True,        # Must use Australian exchange
                max_international_exposure=Decimal('0.4') # 40% international max
            ),
            
            OpportunityTier.MEDIUM: BalanceTier(
                name=OpportunityTier.MEDIUM,
                min_balance_aud=Decimal('50000'),
                max_balance_aud=Decimal('200000'),
                max_position_percentage=Decimal('0.20'),  # 20% max per trade
                min_profit_percentage=Decimal('0.2'),     # 0.2% minimum profit
                max_execution_time_minutes=60,            # 1 hour max
                requires_australian_exchange=False,       # Can use international
                max_international_exposure=Decimal('0.6') # 60% international max
            ),
            
            OpportunityTier.LARGE: BalanceTier(
                name=OpportunityTier.LARGE,
                min_balance_aud=Decimal('200000'),
                max_balance_aud=None,
                max_position_percentage=Decimal('0.15'),  # 15% max per trade
                min_profit_percentage=Decimal('0.15'),    # 0.15% minimum profit
                max_execution_time_minutes=45,            # 45 minutes max
                requires_australian_exchange=False,       # Can use international
                max_international_exposure=Decimal('0.8') # 80% international max
            )
        }
    
    def _initialize_exchange_fees(self) -> Dict[str, Dict[str, Decimal]]:
        """Initialize exchange fee structures"""
        
        return {
            # Australian exchanges
            'btcmarkets': {
                'maker': Decimal('0.0085'),    # 0.85%
                'taker': Decimal('0.0085'),    # 0.85%
                'withdrawal_btc': Decimal('0.0005'),
                'withdrawal_eth': Decimal('0.005'),
                'withdrawal_aud': Decimal('0')
            },
            
            'coinjar': {
                'maker': Decimal('0.001'),     # 0.1%
                'taker': Decimal('0.001'),     # 0.1%
                'withdrawal_btc': Decimal('0.001'),
                'withdrawal_eth': Decimal('0.01'),
                'withdrawal_aud': Decimal('0')
            },
            
            'swyftx': {
                'maker': Decimal('0.006'),     # 0.6%
                'taker': Decimal('0.006'),     # 0.6%
                'withdrawal_btc': Decimal('0.0001'),
                'withdrawal_eth': Decimal('0.005'),
                'withdrawal_aud': Decimal('0')
            },
            
            # International exchanges
            'bybit': {
                'maker': Decimal('0.001'),     # 0.1%
                'taker': Decimal('0.006'),     # 0.6%
                'withdrawal_btc': Decimal('0.0005'),
                'withdrawal_eth': Decimal('0.005'),
                'withdrawal_usdt': Decimal('1')
            },
            
            'binance': {
                'maker': Decimal('0.001'),     # 0.1%
                'taker': Decimal('0.001'),     # 0.1%
                'withdrawal_btc': Decimal('0.0005'),
                'withdrawal_eth': Decimal('0.005'),
                'withdrawal_usdt': Decimal('1')
            }
        }
    
    def determine_balance_tier(self, balance_aud: Decimal) -> OpportunityTier:
        """Determine the appropriate balance tier"""
        
        for tier, config in self.tier_configurations.items():
            if balance_aud >= config.min_balance_aud:
                if config.max_balance_aud is None or balance_aud <= config.max_balance_aud:
                    return tier
        
        return OpportunityTier.MICRO  # Default to smallest tier
    
    def calculate_trading_costs(
        self,
        buy_exchange: str,
        sell_exchange: str,
        amount: Decimal,
        symbol: str
    ) -> Tuple[Decimal, Decimal]:
        """Calculate total trading and transfer costs"""
        
        # Trading fees
        buy_fee = amount * self.exchange_fees.get(buy_exchange, {}).get('taker', Decimal('0.001'))
        sell_fee = amount * self.exchange_fees.get(sell_exchange, {}).get('taker', Decimal('0.001'))
        trading_costs = buy_fee + sell_fee
        
        # Transfer costs (simplified - would use transfer cost database)
        currency = symbol.split('/')[0] if '/' in symbol else symbol[:3]
        
        if buy_exchange in self.australian_exchanges and sell_exchange in self.australian_exchanges:
            # Between Australian exchanges - minimal transfer cost
            transfer_costs = Decimal('0.01') * amount  # 1% transfer cost
        elif buy_exchange in self.international_exchanges and sell_exchange in self.international_exchanges:
            # Between international exchanges - crypto transfer
            if currency == 'BTC':
                transfer_costs = Decimal('0.0005')  # Fixed BTC fee
            elif currency == 'ETH':
                transfer_costs = Decimal('0.005')   # Fixed ETH fee
            else:
                transfer_costs = Decimal('1')       # USDT/USDC fee
        else:
            # Between Australian and international - higher cost
            transfer_costs = Decimal('0.02') * amount  # 2% transfer cost
        
        return trading_costs, transfer_costs
    
    def assess_liquidity_and_risk(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        amount: Decimal
    ) -> Tuple[float, float]:
        """Assess liquidity score and volatility risk"""
        
        # Simplified liquidity scoring
        exchange_liquidity = {
            'binance': 0.95,
            'bybit': 0.85,
            'coinbase': 0.80,
            'kraken': 0.75,
            'btcmarkets': 0.60,
            'coinjar': 0.50,
            'swyftx': 0.45
        }
        
        # Average liquidity of both exchanges
        buy_liquidity = exchange_liquidity.get(buy_exchange, 0.3)
        sell_liquidity = exchange_liquidity.get(sell_exchange, 0.3)
        liquidity_score = (buy_liquidity + sell_liquidity) / 2
        
        # Volatility risk based on symbol
        volatility_risk = {
            'BTC': 0.4,
            'ETH': 0.5,
            'ADA': 0.7,
            'DOT': 0.6,
            'LINK': 0.6
        }
        
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol[:3]
        risk_score = volatility_risk.get(base_currency, 0.8)
        
        return liquidity_score, risk_score
    
    def detect_simple_arbitrage(
        self,
        price_data: Dict[str, Dict[str, Decimal]],
        balance_aud: Decimal,
        symbol: str
    ) -> List[ArbitrageOpportunity]:
        """Detect simple arbitrage opportunities"""
        
        opportunities = []
        tier = self.determine_balance_tier(balance_aud)
        tier_config = self.tier_configurations[tier]
        
        # Get exchanges with price data for this symbol
        exchanges_with_data = [exchange for exchange, data in price_data.items() 
                             if symbol in data]
        
        if len(exchanges_with_data) < 2:
            return opportunities
        
        # Compare all exchange pairs
        for i, buy_exchange in enumerate(exchanges_with_data):
            for sell_exchange in exchanges_with_data[i+1:]:
                
                buy_price = price_data[buy_exchange][symbol]
                sell_price = price_data[sell_exchange][symbol]
                
                # Determine which exchange has better prices
                if buy_price < sell_price:
                    actual_buy_exchange, actual_sell_exchange = buy_exchange, sell_exchange
                    actual_buy_price, actual_sell_price = buy_price, sell_price
                else:
                    actual_buy_exchange, actual_sell_exchange = sell_exchange, buy_exchange
                    actual_buy_price, actual_sell_price = sell_price, buy_price
                
                # Calculate gross profit
                price_difference = actual_sell_price - actual_buy_price
                gross_profit_pct = (price_difference / actual_buy_price) * 100
                
                # Skip if gross profit too small
                if gross_profit_pct < tier_config.min_profit_percentage:
                    continue
                
                # Check Australian exchange requirements
                if tier_config.requires_australian_exchange:
                    if (actual_buy_exchange not in self.australian_exchanges and 
                        actual_sell_exchange not in self.australian_exchanges):
                        continue
                
                # Calculate position size
                max_position = balance_aud * tier_config.max_position_percentage
                
                # Calculate costs
                trading_costs, transfer_costs = self.calculate_trading_costs(
                    actual_buy_exchange, actual_sell_exchange, max_position, symbol
                )
                total_costs = trading_costs + transfer_costs
                
                # Calculate net profit
                net_profit_pct = gross_profit_pct - ((total_costs / max_position) * 100)
                
                # Skip if not profitable after costs
                if net_profit_pct <= 0:
                    continue
                
                # Assess liquidity and risk
                liquidity_score, volatility_risk = self.assess_liquidity_and_risk(
                    actual_buy_exchange, actual_sell_exchange, symbol, max_position
                )
                
                # Calculate execution time (simplified)
                base_time = 15  # 15 minutes base
                if actual_buy_exchange in self.australian_exchanges:
                    base_time += 30  # Australian exchange settlement
                if actual_sell_exchange in self.australian_exchanges:
                    base_time += 30
                execution_time = base_time
                
                # Skip if execution time too long
                if execution_time > tier_config.max_execution_time_minutes:
                    continue
                
                # Determine if Australian-friendly
                australian_friendly = (
                    actual_buy_exchange in self.australian_exchanges or 
                    actual_sell_exchange in self.australian_exchanges
                )
                
                # Create opportunity
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"simple_{actual_buy_exchange}_{actual_sell_exchange}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    arbitrage_type=ArbitrageType.SIMPLE_ARBITRAGE,
                    tier=tier,
                    buy_exchange=actual_buy_exchange,
                    sell_exchange=actual_sell_exchange,
                    symbol=symbol,
                    buy_price=actual_buy_price,
                    sell_price=actual_sell_price,
                    price_difference=price_difference,
                    gross_profit_percentage=gross_profit_pct,
                    transfer_costs=transfer_costs,
                    trading_fees=trading_costs,
                    total_costs=total_costs,
                    net_profit_percentage=net_profit_pct,
                    minimum_amount=Decimal('100'),  # $100 AUD minimum
                    maximum_amount=max_position,
                    estimated_execution_time=execution_time,
                    liquidity_score=liquidity_score,
                    volatility_risk=volatility_risk,
                    australian_friendly=australian_friendly,
                    detected_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(minutes=10)  # 10 minute window
                )
                
                if opportunity.is_profitable():
                    opportunities.append(opportunity)
        
        return opportunities

class FundingArbitrageDetector:
    """
    Funding rate arbitrage detector
    Spot vs perpetual funding rate opportunities
    """
    
    def __init__(self):
        self.funding_threshold = Decimal('0.01')  # 1% annualized funding rate threshold
        
    def detect_funding_opportunities(
        self,
        spot_prices: Dict[str, Decimal],
        perpetual_prices: Dict[str, Decimal],
        funding_rates: Dict[str, Decimal],
        balance_aud: Decimal
    ) -> List[ArbitrageOpportunity]:
        """Detect funding rate arbitrage opportunities"""
        
        opportunities = []
        
        # Find symbols available in both spot and perpetual
        common_symbols = set(spot_prices.keys()) & set(perpetual_prices.keys()) & set(funding_rates.keys())
        
        for symbol in common_symbols:
            spot_price = spot_prices[symbol]
            perp_price = perpetual_prices[symbol]
            funding_rate = funding_rates[symbol]
            
            # Calculate annualized funding rate (assuming 8-hour funding)
            annualized_funding = funding_rate * 365 * 3  # 3 times per day
            
            # Check if funding rate is attractive
            if abs(annualized_funding) > self.funding_threshold:
                
                # Determine strategy direction
                if funding_rate > 0:
                    # Positive funding: short perpetual, long spot
                    strategy = "short_perp_long_spot"
                    expected_return = annualized_funding
                else:
                    # Negative funding: long perpetual, short spot
                    strategy = "long_perp_short_spot"
                    expected_return = abs(annualized_funding)
                
                # Create funding arbitrage opportunity
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"funding_{symbol}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    arbitrage_type=ArbitrageType.FUNDING_ARBITRAGE,
                    tier=OpportunityTier.MEDIUM,  # Funding arb typically requires more capital
                    buy_exchange="spot_market",
                    sell_exchange="perpetual_market",
                    symbol=symbol,
                    buy_price=spot_price if funding_rate > 0 else perp_price,
                    sell_price=perp_price if funding_rate > 0 else spot_price,
                    price_difference=abs(perp_price - spot_price),
                    gross_profit_percentage=expected_return * 100,
                    transfer_costs=Decimal('0'),  # Minimal for same exchange
                    trading_fees=spot_price * Decimal('0.002'),  # 0.2% combined fees
                    total_costs=spot_price * Decimal('0.002'),
                    net_profit_percentage=(expected_return - Decimal('0.002')) * 100,
                    minimum_amount=Decimal('5000'),  # Higher minimum for funding arb
                    maximum_amount=balance_aud * Decimal('0.5'),  # Max 50% of balance
                    estimated_execution_time=30,  # 30 minutes to set up
                    liquidity_score=0.8,  # Generally good liquidity
                    volatility_risk=0.3,  # Lower risk due to hedging
                    australian_friendly=True,  # Can be done on Australian exchanges
                    detected_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=8)  # Until next funding
                )
                
                if opportunity.is_profitable():
                    opportunities.append(opportunity)
        
        return opportunities

class TriangularArbitrageDetector:
    """
    Triangular arbitrage detector
    Three-currency arbitrage loops (e.g., BTC -> ETH -> AUD -> BTC)
    """
    
    def __init__(self):
        self.currency_pairs = [
            ('BTC', 'ETH', 'AUD'),
            ('BTC', 'ADA', 'AUD'),
            ('ETH', 'ADA', 'AUD'),
            ('BTC', 'USDT', 'AUD')
        ]
    
    def detect_triangular_opportunities(
        self,
        price_data: Dict[str, Dict[str, Decimal]],
        balance_aud: Decimal
    ) -> List[ArbitrageOpportunity]:
        """Detect triangular arbitrage opportunities"""
        
        opportunities = []
        
        for exchange, exchange_prices in price_data.items():
            for curr_a, curr_b, curr_c in self.currency_pairs:
                
                # Get required exchange rates
                pair_ab = f"{curr_a}/{curr_b}"
                pair_bc = f"{curr_b}/{curr_c}"
                pair_ac = f"{curr_a}/{curr_c}"
                
                # Check if all pairs are available
                if not all(pair in exchange_prices for pair in [pair_ab, pair_bc, pair_ac]):
                    continue
                
                rate_ab = exchange_prices[pair_ab]
                rate_bc = exchange_prices[pair_bc]
                rate_ac = exchange_prices[pair_ac]
                
                # Calculate triangular rate
                triangular_rate = rate_ab * rate_bc
                direct_rate = rate_ac
                
                # Check for arbitrage opportunity
                if triangular_rate > direct_rate:
                    # Opportunity: A -> B -> C, then C -> A
                    profit_pct = ((triangular_rate - direct_rate) / direct_rate) * 100
                    
                    if profit_pct > Decimal('0.5'):  # Minimum 0.5% profit
                        opportunity = ArbitrageOpportunity(
                            opportunity_id=f"triangular_{exchange}_{curr_a}_{curr_b}_{curr_c}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            arbitrage_type=ArbitrageType.TRIANGULAR_ARBITRAGE,
                            tier=OpportunityTier.MEDIUM,
                            buy_exchange=exchange,
                            sell_exchange=exchange,
                            symbol=f"{curr_a}/{curr_b}/{curr_c}",
                            buy_price=direct_rate,
                            sell_price=triangular_rate,
                            price_difference=triangular_rate - direct_rate,
                            gross_profit_percentage=profit_pct,
                            transfer_costs=Decimal('0'),  # Same exchange
                            trading_fees=balance_aud * Decimal('0.003'),  # 3 trades * 0.1%
                            total_costs=balance_aud * Decimal('0.003'),
                            net_profit_percentage=profit_pct - Decimal('0.3'),
                            minimum_amount=Decimal('1000'),
                            maximum_amount=balance_aud * Decimal('0.3'),
                            estimated_execution_time=5,  # Quick execution needed
                            liquidity_score=0.7,
                            volatility_risk=0.6,  # Higher risk due to multiple legs
                            australian_friendly=exchange in {'btcmarkets', 'coinjar', 'swyftx'},
                            detected_at=datetime.now(),
                            expires_at=datetime.now() + timedelta(minutes=2)  # Very short window
                        )
                        
                        if opportunity.is_profitable():
                            opportunities.append(opportunity)
        
        return opportunities

class OpportunisticArbitrageEngine:
    """
    Main arbitrage engine coordinating all detection methods
    Designed for Australian traders with balance-based activation
    """
    
    def __init__(self):
        self.simple_detector = AustralianArbitrageDetector()
        self.funding_detector = FundingArbitrageDetector()
        self.triangular_detector = TriangularArbitrageDetector()
        
        self.active_opportunities = []
        self.opportunity_history = []
        
        logger.info("Initialized Opportunistic Arbitrage Engine")
    
    async def scan_for_opportunities(
        self,
        price_data: Dict[str, Dict[str, Decimal]],
        funding_data: Optional[Dict[str, Decimal]],
        balance_aud: Decimal,
        symbols: List[str]
    ) -> List[ArbitrageOpportunity]:
        """Scan for all types of arbitrage opportunities"""
        
        all_opportunities = []
        
        # Simple arbitrage detection
        for symbol in symbols:
            simple_opps = self.simple_detector.detect_simple_arbitrage(
                price_data, balance_aud, symbol
            )
            all_opportunities.extend(simple_opps)
        
        # Funding arbitrage (if funding data available)
        if funding_data:
            # Separate spot and perpetual prices
            spot_prices = {}
            perp_prices = {}
            
            for exchange, exchange_data in price_data.items():
                for symbol, price in exchange_data.items():
                    if 'PERP' in symbol or 'USD' in symbol:
                        perp_prices[symbol] = price
                    else:
                        spot_prices[symbol] = price
            
            funding_opps = self.funding_detector.detect_funding_opportunities(
                spot_prices, perp_prices, funding_data, balance_aud
            )
            all_opportunities.extend(funding_opps)
        
        # Triangular arbitrage
        triangular_opps = self.triangular_detector.detect_triangular_opportunities(
            price_data, balance_aud
        )
        all_opportunities.extend(triangular_opps)
        
        # Filter valid and profitable opportunities
        valid_opportunities = [
            opp for opp in all_opportunities 
            if opp.is_valid() and opp.is_profitable()
        ]
        
        # Sort by net profit percentage
        valid_opportunities.sort(key=lambda x: x.net_profit_percentage, reverse=True)
        
        # Update active opportunities
        self.active_opportunities = valid_opportunities
        
        logger.info(f"Found {len(valid_opportunities)} arbitrage opportunities")
        
        return valid_opportunities
    
    def get_tier_appropriate_opportunities(
        self,
        balance_aud: Decimal,
        max_opportunities: int = 5
    ) -> List[ArbitrageOpportunity]:
        """Get opportunities appropriate for current balance tier"""
        
        tier = self.simple_detector.determine_balance_tier(balance_aud)
        tier_config = self.simple_detector.tier_configurations[tier]
        
        # Filter opportunities for this tier
        tier_opportunities = [
            opp for opp in self.active_opportunities
            if (opp.tier == tier and 
                opp.net_profit_percentage >= tier_config.min_profit_percentage and
                opp.estimated_execution_time <= tier_config.max_execution_time_minutes)
        ]
        
        # Additional filtering for Australian requirements
        if tier_config.requires_australian_exchange:
            tier_opportunities = [
                opp for opp in tier_opportunities
                if opp.australian_friendly
            ]
        
        return tier_opportunities[:max_opportunities]
    
    def calculate_portfolio_allocation(
        self,
        opportunities: List[ArbitrageOpportunity],
        balance_aud: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate optimal allocation across arbitrage opportunities"""
        
        tier = self.simple_detector.determine_balance_tier(balance_aud)
        tier_config = self.simple_detector.tier_configurations[tier]
        
        allocations = {}
        remaining_balance = balance_aud
        max_single_position = balance_aud * tier_config.max_position_percentage
        
        for opp in opportunities:
            # Calculate position size based on profit and risk
            base_allocation = min(max_single_position, opp.maximum_amount)
            
            # Adjust for liquidity and risk
            risk_adjusted_allocation = base_allocation * Decimal(str(opp.liquidity_score)) * (1 - Decimal(str(opp.volatility_risk)))
            
            # Ensure we don't exceed remaining balance
            final_allocation = min(risk_adjusted_allocation, remaining_balance * Decimal('0.9'))
            
            if final_allocation >= opp.minimum_amount:
                allocations[opp.opportunity_id] = final_allocation
                remaining_balance -= final_allocation
            
            # Stop if we've allocated most of the balance
            if remaining_balance < balance_aud * Decimal('0.1'):
                break
        
        return allocations
    
    def get_opportunity_summary(self) -> Dict[str, Any]:
        """Get summary of current arbitrage opportunities"""
        
        if not self.active_opportunities:
            return {
                'total_opportunities': 0,
                'average_profit_pct': 0,
                'best_opportunity': None,
                'by_type': {},
                'by_tier': {}
            }
        
        # Group by type
        by_type = {}
        for opp in self.active_opportunities:
            opp_type = opp.arbitrage_type.value
            if opp_type not in by_type:
                by_type[opp_type] = []
            by_type[opp_type].append(opp)
        
        # Group by tier  
        by_tier = {}
        for opp in self.active_opportunities:
            tier = opp.tier.value
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(opp)
        
        # Calculate averages
        avg_profit = np.mean([float(opp.net_profit_percentage) for opp in self.active_opportunities])
        best_opportunity = max(self.active_opportunities, key=lambda x: x.net_profit_percentage)
        
        return {
            'total_opportunities': len(self.active_opportunities),
            'average_profit_pct': round(avg_profit, 3),
            'best_opportunity': {
                'id': best_opportunity.opportunity_id,
                'type': best_opportunity.arbitrage_type.value,
                'profit_pct': float(best_opportunity.net_profit_percentage),
                'exchanges': f"{best_opportunity.buy_exchange} -> {best_opportunity.sell_exchange}",
                'symbol': best_opportunity.symbol
            },
            'by_type': {k: len(v) for k, v in by_type.items()},
            'by_tier': {k: len(v) for k, v in by_tier.items()}
        }

# Usage example
async def main():
    """Example usage of arbitrage engine"""
    
    # Initialize engine
    engine = OpportunisticArbitrageEngine()
    
    # Sample price data
    price_data = {
        'btcmarkets': {
            'BTC/AUD': Decimal('65500'),
            'ETH/AUD': Decimal('2650'),
            'BTC/ETH': Decimal('24.7')
        },
        'bybit': {
            'BTC/USDT': Decimal('43500'),  # Assuming AUD/USD = 0.66
            'ETH/USDT': Decimal('1760'),
            'BTC/ETH': Decimal('24.9')
        },
        'binance': {
            'BTC/USDT': Decimal('43600'),
            'ETH/USDT': Decimal('1755'),
            'BTC/ETH': Decimal('24.8')
        }
    }
    
    # Sample funding data
    funding_data = {
        'BTC/USDT': Decimal('0.0001'),  # 0.01% per 8h
        'ETH/USDT': Decimal('-0.0002')  # -0.02% per 8h
    }
    
    # Test different balance tiers
    balance_tiers = [
        Decimal('5000'),    # Micro
        Decimal('25000'),   # Small
        Decimal('100000'),  # Medium
        Decimal('500000')   # Large
    ]
    
    for balance in balance_tiers:
        print(f"\n=== Balance: ${balance:,} AUD ===")
        
        # Scan for opportunities
        opportunities = await engine.scan_for_opportunities(
            price_data=price_data,
            funding_data=funding_data,
            balance_aud=balance,
            symbols=['BTC/AUD', 'ETH/AUD', 'BTC/USDT', 'ETH/USDT']
        )
        
        # Get tier-appropriate opportunities
        tier_opps = engine.get_tier_appropriate_opportunities(balance, max_opportunities=3)
        
        print(f"Found {len(opportunities)} total opportunities")
        print(f"Tier-appropriate opportunities: {len(tier_opps)}")
        
        for opp in tier_opps:
            print(f"  {opp.arbitrage_type.value}: {opp.symbol} "
                  f"{opp.buy_exchange} -> {opp.sell_exchange} "
                  f"Profit: {opp.net_profit_percentage:.3f}% "
                  f"Amount: ${opp.maximum_amount:,.0f}")
        
        # Calculate allocations
        allocations = engine.calculate_portfolio_allocation(tier_opps, balance)
        print(f"Allocations: {len(allocations)} positions, "
              f"Total: ${sum(allocations.values()):,.0f}")
    
    # Get summary
    summary = engine.get_opportunity_summary()
    print(f"\nOverall Summary:")
    print(f"  Total opportunities: {summary['total_opportunities']}")
    print(f"  Average profit: {summary['average_profit_pct']:.3f}%")
    if summary['best_opportunity']:
        best = summary['best_opportunity']
        print(f"  Best opportunity: {best['profit_pct']:.3f}% on {best['symbol']}")

if __name__ == "__main__":
    asyncio.run(main())