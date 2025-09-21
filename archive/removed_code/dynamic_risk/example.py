"""
Dynamic Risk System Example and Integration Demo.

This example demonstrates how to use the Dynamic Risk Management System
with real market data simulation and shows all major features:

- Adaptive volatility monitoring with regime detection
- Dynamic correlation analysis and portfolio assessment
- Automated hedging with real-time rebalancing
- Risk-adjusted position sizing based on market conditions
- Cross-asset risk factor analysis
- Real-time monitoring and alerting

The example simulates various market conditions to show how the system
adapts to changing volatility regimes and correlation structures.
"""

import asyncio
import time
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from . import (
    DynamicRiskSystem,
    RiskRegime,
    AdaptationSignal,
    PortfolioRiskMetrics,
    RiskAdjustment
)
from ..utils.logging import TradingLogger


class MarketDataSimulator:
    """
    Simulate realistic market data for testing the dynamic risk system.
    
    This simulator generates correlated price movements with time-varying
    volatility and correlation to test the risk system's adaptive capabilities.
    """
    
    def __init__(self, symbols: List[str], config: Optional[Dict] = None):
        self.symbols = symbols
        self.config = config or self._default_config()
        self.logger = TradingLogger("MarketDataSimulator")
        
        # Initial prices
        self.prices = {symbol: 100.0 for symbol in symbols}
        
        # Volatility regimes (will change over time)
        self.volatilities = {symbol: 0.20 for symbol in symbols}  # 20% annual vol
        
        # Correlation matrix (will evolve)
        n = len(symbols)
        self.correlation_matrix = np.eye(n) * 0.7 + np.ones((n, n)) * 0.3
        
        # Market regime state
        self.current_regime = "normal"
        self.regime_start_time = datetime.now()
        
    def _default_config(self) -> Dict:
        """Default configuration for market simulator."""
        return {
            'base_volatility': 0.20,      # Base annual volatility
            'volatility_range': (0.10, 0.80),  # Min/max volatility
            'correlation_range': (0.1, 0.9),   # Min/max correlation
            'regime_duration': 3600,      # Average regime duration (seconds)
            'crisis_probability': 0.05,   # Probability of crisis regime
            'update_frequency': 60,       # Data update frequency (seconds)
        }
    
    def _transition_regime(self) -> None:
        """Transition to a new market regime."""
        regimes = ["low_vol", "normal", "high_vol", "crisis"]
        weights = [0.15, 0.50, 0.25, 0.10] if self.current_regime != "crisis" else [0.30, 0.50, 0.20, 0.00]
        
        # More likely to exit crisis than enter it
        if self.current_regime == "crisis":
            weights = [0.40, 0.40, 0.20, 0.00]
        
        self.current_regime = np.random.choice(regimes, p=weights)
        self.regime_start_time = datetime.now()
        
        self.logger.info(f"Market regime transition to: {self.current_regime}")
        
        # Adjust volatilities based on regime
        if self.current_regime == "low_vol":
            base_vol = 0.12
        elif self.current_regime == "normal":
            base_vol = 0.20
        elif self.current_regime == "high_vol":
            base_vol = 0.35
        else:  # crisis
            base_vol = 0.60
        
        # Add some randomness to individual asset volatilities
        for symbol in self.symbols:
            self.volatilities[symbol] = base_vol * (0.8 + 0.4 * random.random())
        
        # Adjust correlations based on regime
        n = len(self.symbols)
        if self.current_regime == "crisis":
            # High correlations during crisis
            base_corr = 0.8
            self.correlation_matrix = np.eye(n) * 0.2 + np.ones((n, n)) * base_corr
        elif self.current_regime == "high_vol":
            # Moderate correlations during high volatility
            base_corr = 0.5
            self.correlation_matrix = np.eye(n) * 0.5 + np.ones((n, n)) * base_corr
        else:
            # Lower correlations during normal/low vol
            base_corr = 0.3
            self.correlation_matrix = np.eye(n) * 0.7 + np.ones((n, n)) * base_corr
        
        # Ensure positive definite
        self.correlation_matrix = np.maximum(self.correlation_matrix, 0.05)
        np.fill_diagonal(self.correlation_matrix, 1.0)
    
    def generate_returns(self) -> Dict[str, float]:
        """Generate correlated returns for all symbols."""
        # Check if we should transition regimes
        time_in_regime = datetime.now() - self.regime_start_time
        if time_in_regime.total_seconds() > self.config['regime_duration']:
            if random.random() < 0.3:  # 30% chance to transition
                self._transition_regime()
        
        # Generate correlated random returns
        n = len(self.symbols)
        
        # Cholesky decomposition for correlation
        try:
            L = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite, use identity
            L = np.eye(n)
        
        # Generate independent random variables
        independent_returns = np.random.normal(0, 1, n)
        
        # Apply correlation structure
        correlated_returns = L @ independent_returns
        
        # Scale by volatilities (convert annual to per-update)
        time_factor = math.sqrt(self.config['update_frequency'] / (365.25 * 24 * 3600))
        
        returns = {}
        for i, symbol in enumerate(self.symbols):
            vol_scaled_return = correlated_returns[i] * self.volatilities[symbol] * time_factor
            returns[symbol] = vol_scaled_return
        
        return returns
    
    def update_prices(self) -> Dict[str, Dict[str, float]]:
        """Update prices and generate OHLC data."""
        returns = self.generate_returns()
        market_data = {}
        
        for symbol in self.symbols:
            # Update price
            old_price = self.prices[symbol]
            new_price = old_price * math.exp(returns[symbol])
            self.prices[symbol] = new_price
            
            # Generate OHLC (simplified)
            high = new_price * (1 + abs(returns[symbol]) * 0.5)
            low = new_price * (1 - abs(returns[symbol]) * 0.5)
            open_price = old_price
            
            # Generate volume (random)
            volume = random.uniform(1000, 10000)
            
            market_data[symbol] = {
                'price': new_price,
                'open': open_price,
                'high': high,
                'low': low,
                'volume': volume,
                'return': returns[symbol]
            }
        
        return market_data


class DynamicRiskSystemDemo:
    """
    Comprehensive demonstration of the Dynamic Risk Management System.
    
    This demo shows how the system works in various market conditions
    and demonstrates all key features.
    """
    
    def __init__(self):
        self.logger = TradingLogger("DynamicRiskSystemDemo")
        
        # Portfolio symbols
        self.symbols = ["BTC", "ETH", "ADA", "SOL", "DOT"]
        
        # Initial positions (random for demo)
        self.positions = {
            symbol: random.uniform(1000, 5000) * random.choice([-1, 1])
            for symbol in self.symbols
        }
        
        # Market data simulator
        self.market_simulator = MarketDataSimulator(self.symbols)
        
        # Dynamic risk system
        risk_config = {
            'monitoring_interval': 30,  # 30 seconds for demo
            'portfolio_symbols': self.symbols,
            'auto_hedge_enabled': True,
            'auto_adjust_enabled': True,
            'volatility_monitor': {
                'update_interval': 30
            },
            'correlation_analyzer': {
                'update_interval': 30
            },
            'hedging_system': {
                'monitoring_interval': 60
            }
        }
        
        self.risk_system = DynamicRiskSystem(risk_config)
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Demo state
        self.demo_running = False
        self.demo_start_time = None
        self.update_count = 0
        
    def _setup_callbacks(self) -> None:
        """Setup callbacks to monitor system behavior."""
        
        def risk_metrics_callback(metrics: PortfolioRiskMetrics):
            self.logger.info(
                f"RISK UPDATE: regime={metrics.risk_regime.value}, "
                f"portfolio_vol={metrics.portfolio_volatility:.3f}, "
                f"avg_corr={metrics.average_correlation:.3f}, "
                f"effective_positions={metrics.effective_positions:.1f}"
            )
            
            # Log any high-risk conditions
            if metrics.risk_regime in [RiskRegime.HIGH_RISK, RiskRegime.CRISIS_RISK]:
                self.logger.warning(
                    f"HIGH RISK DETECTED: {metrics.risk_regime.value} "
                    f"(confidence: {metrics.regime_confidence:.2f})"
                )
        
        def adjustment_callback(symbol: str, adjustment: RiskAdjustment):
            if adjustment.adaptation_signal != AdaptationSignal.NO_CHANGE:
                self.logger.info(
                    f"ADJUSTMENT: {symbol} -> {adjustment.adaptation_signal.value}, "
                    f"scalar={adjustment.total_adjustment:.3f}, "
                    f"hedge_rec={adjustment.hedge_recommendation}"
                )
        
        self.risk_system.add_risk_callback(risk_metrics_callback)
        self.risk_system.add_adjustment_callback(adjustment_callback)
    
    def run_demo(self, duration_minutes: int = 30) -> None:
        """Run the dynamic risk system demo."""
        self.logger.info(f"Starting Dynamic Risk System Demo for {duration_minutes} minutes")
        
        self.demo_running = True
        self.demo_start_time = datetime.now()
        
        # Start risk system monitoring
        self.risk_system.start_monitoring(self.symbols)
        
        try:
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            
            while datetime.now() < end_time and self.demo_running:
                # Generate market data
                market_data = self.market_simulator.update_prices()
                
                # Feed data to risk system
                for symbol, data in market_data.items():
                    position_size = self.positions.get(symbol, 0)
                    
                    self.risk_system.add_market_data(
                        symbol=symbol,
                        price=data['price'],
                        position_size=position_size,
                        open_price=data['open'],
                        high=data['high'],
                        low=data['low'],
                        volume=data['volume']
                    )
                
                self.update_count += 1
                
                # Simulate position changes occasionally
                if self.update_count % 10 == 0:
                    self._simulate_position_changes()
                
                # Print periodic status
                if self.update_count % 5 == 0:
                    self._print_status()
                
                # Wait for next update
                time.sleep(self.market_simulator.config['update_frequency'])
            
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        
        finally:
            self.demo_running = False
            self.risk_system.stop_monitoring()
            self._print_final_summary()
    
    def _simulate_position_changes(self) -> None:
        """Simulate position changes to test system adaptation."""
        # Randomly change one position
        symbol = random.choice(self.symbols)
        old_position = self.positions[symbol]
        
        # Random position change (±20%)
        change_factor = 1 + random.uniform(-0.2, 0.2)
        new_position = old_position * change_factor
        
        self.positions[symbol] = new_position
        
        self.logger.debug(f"Position change: {symbol} {old_position:.0f} -> {new_position:.0f}")
    
    def _print_status(self) -> None:
        """Print current system status."""
        runtime = datetime.now() - self.demo_start_time
        
        # Get current risk metrics
        risk_metrics = self.risk_system.get_current_risk_metrics()
        
        print(f"\n{'='*60}")
        print(f"DYNAMIC RISK SYSTEM STATUS (Runtime: {runtime})")
        print(f"{'='*60}")
        
        if risk_metrics:
            print(f"Risk Regime: {risk_metrics.risk_regime.value} "
                  f"(confidence: {risk_metrics.regime_confidence:.2f})")
            print(f"Portfolio Volatility: {risk_metrics.portfolio_volatility:.3f}")
            print(f"Average Correlation: {risk_metrics.average_correlation:.3f}")
            print(f"Effective Positions: {risk_metrics.effective_positions:.1f}")
            print(f"Hedge Effectiveness: {risk_metrics.total_hedge_effectiveness:.3f}")
            print(f"Hedged Exposure: {risk_metrics.hedged_exposure_pct:.1%}")
        
        # Risk adjustments
        print(f"\nRisk Adjustments:")
        for symbol in self.symbols:
            adjustment = self.risk_system.get_risk_adjustment(symbol)
            if adjustment:
                print(f"  {symbol}: {adjustment.total_adjustment:.3f} "
                      f"({adjustment.adaptation_signal.value})")
        
        # Hedge summary
        hedge_summary = self.risk_system.get_hedge_summary()
        print(f"\nHedge Summary:")
        print(f"  Active Hedges: {hedge_summary.get('active_hedges', 0)}")
        print(f"  Total Hedges: {hedge_summary.get('total_hedges', 0)}")
        print(f"  Avg Effectiveness: {hedge_summary.get('avg_effectiveness', 0):.3f}")
        
        # Current market regime
        print(f"\nMarket Simulator:")
        print(f"  Current Regime: {self.market_simulator.current_regime}")
        print(f"  Update Count: {self.update_count}")
        
        print(f"{'='*60}\n")
    
    def _print_final_summary(self) -> None:
        """Print final demo summary."""
        runtime = datetime.now() - self.demo_start_time
        
        print(f"\n{'='*80}")
        print(f"FINAL DEMO SUMMARY")
        print(f"{'='*80}")
        print(f"Total Runtime: {runtime}")
        print(f"Total Updates: {self.update_count}")
        
        # Final risk summary
        risk_summary = self.risk_system.get_risk_summary()
        print(f"\nFinal Risk Summary:")
        for key, value in risk_summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nDemo completed successfully!")
        print(f"{'='*80}\n")


def run_basic_example():
    """Run a basic example of the dynamic risk system."""
    logger = TradingLogger("BasicExample")
    
    # Create risk system with minimal configuration
    config = {
        'portfolio_symbols': ['BTC', 'ETH', 'ADA'],
        'monitoring_interval': 60,  # 1 minute
        'auto_hedge_enabled': True
    }
    
    risk_system = DynamicRiskSystem(config)
    
    # Setup a simple callback
    def risk_callback(metrics: PortfolioRiskMetrics):
        logger.info(f"Risk regime: {metrics.risk_regime.value}, "
                   f"Portfolio vol: {metrics.portfolio_volatility:.3f}")
    
    risk_system.add_risk_callback(risk_callback)
    
    # Simulate some market data
    symbols = ['BTC', 'ETH', 'ADA']
    prices = {'BTC': 45000, 'ETH': 3000, 'ADA': 1.2}
    positions = {'BTC': 0.5, 'ETH': 2.0, 'ADA': 1000}
    
    logger.info("Starting basic dynamic risk system example...")
    
    # Start monitoring
    risk_system.start_monitoring(symbols)
    
    try:
        # Feed some market data
        for i in range(10):
            for symbol in symbols:
                # Simulate price movement
                change = random.uniform(-0.02, 0.02)  # ±2% change
                prices[symbol] *= (1 + change)
                
                # Add market data
                risk_system.add_market_data(
                    symbol=symbol,
                    price=prices[symbol],
                    position_size=positions[symbol]
                )
            
            time.sleep(5)  # Wait 5 seconds between updates
        
        # Calculate final risk metrics
        final_metrics = risk_system.calculate_portfolio_risk_metrics(symbols, positions)
        if final_metrics:
            logger.info(f"Final risk assessment: {final_metrics.risk_regime.value}")
        
    finally:
        risk_system.stop_monitoring()
        logger.info("Basic example completed")


if __name__ == "__main__":
    # Run the comprehensive demo
    demo = DynamicRiskSystemDemo()
    demo.run_demo(duration_minutes=10)  # 10-minute demo
    
    print("\n" + "="*80)
    print("Running basic example...")
    print("="*80)
    
    # Also run the basic example
    run_basic_example()