"""
SQLAlchemy models for the trading bot database.

This module defines all database models including:
- Trade execution records with tax tracking
- Strategy performance metrics
- Portfolio state tracking
- Risk events and violations
- Market data storage
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, JSON, String, Text, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# Base class for all models
Base = declarative_base()


class Trade(Base):
    """
    Trade execution records with comprehensive tax tracking.
    
    This model stores all trade executions with fields required for
    Australian CGT calculations and audit trails.
    """
    
    __tablename__ = "trades"
    
    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False) 
    fee = Column(Float, nullable=False, default=0.0)
    fee_currency = Column(String(10), nullable=False)
    
    # Strategy identification
    strategy_id = Column(String(100), nullable=False)
    strategy_version = Column(String(20))
    signal_confidence = Column(Float)
    
    # Order execution details
    order_id = Column(String(100))
    order_type = Column(String(20))  # 'market', 'limit'
    time_in_force = Column(String(10))  # 'GTC', 'IOC', 'FOK'
    slippage = Column(Float)
    
    # Risk and position sizing
    risk_amount = Column(Float)  # Amount risked on this trade
    position_size_usd = Column(Float)  # USD value of position
    portfolio_balance = Column(Float)  # Portfolio balance at time of trade
    trading_mode = Column(String(20))  # 'conservative' or 'aggressive'
    
    # Tax tracking fields (Australia specific)
    cost_base_aud = Column(Float)  # Cost base in AUD
    proceeds_aud = Column(Float)  # Proceeds in AUD (for sells)
    is_cgt_event = Column(Boolean, default=False)  # CGT event flag
    aud_conversion_rate = Column(Float)  # USD/AUD rate at trade time
    holding_period_days = Column(Integer)  # Days held (for CGT discount)
    
    # Performance tracking
    unrealized_pnl = Column(Float)  # Unrealized P&L at trade time
    realized_pnl = Column(Float)  # Realized P&L (for position closures)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    notes = Column(Text)
    
    # Constraints
    __table_args__ = (
        CheckConstraint(side.in_(['buy', 'sell']), name='valid_side'),
        CheckConstraint(amount > 0, name='positive_amount'),
        CheckConstraint(price > 0, name='positive_price'),
        Index('idx_trades_timestamp', timestamp),
        Index('idx_trades_symbol', symbol),
        Index('idx_trades_strategy', strategy_id),
        Index('idx_trades_tax', is_cgt_event, timestamp),
    )
    
    def __repr__(self):
        return f"<Trade({self.id}: {self.side} {self.amount} {self.symbol} @ {self.price})>"


class StrategyPerformance(Base):
    """
    Strategy performance metrics over time.
    
    Tracks key performance indicators for each strategy including
    risk-adjusted returns and mode-specific metrics.
    """
    
    __tablename__ = "strategy_performance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    period = Column(String(10), nullable=False)  # 'daily', 'weekly', 'monthly'
    
    # Core performance metrics
    equity = Column(Float, nullable=False)
    returns = Column(Float)  # Period returns
    cumulative_returns = Column(Float)
    volatility = Column(Float)
    
    # Risk metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float) 
    calmar_ratio = Column(Float)
    max_drawdown = Column(Float)
    current_drawdown = Column(Float)
    
    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    
    # Mode-specific tracking
    trading_mode = Column(String(20))  # Mode when metrics were recorded
    risk_parameters = Column(JSON)  # Snapshot of risk parameters
    
    # Validation metrics
    oos_performance = Column(Float)  # Out-of-sample performance
    consistency_score = Column(Float)  # Performance consistency
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_strategy_perf_strategy_time', strategy_id, timestamp),
        Index('idx_strategy_perf_period', period),
        UniqueConstraint('strategy_id', 'timestamp', 'period', name='uq_strategy_period'),
    )
    
    def __repr__(self):
        return f"<StrategyPerformance({self.strategy_id}: {self.equity:.2f})>"


class Portfolio(Base):
    """
    Portfolio state snapshots over time.
    
    Tracks overall portfolio value, positions, and risk metrics.
    """
    
    __tablename__ = "portfolio"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    
    # Portfolio value
    total_value_usd = Column(Float, nullable=False)
    available_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    
    # Risk metrics
    portfolio_var = Column(Float)  # Value at Risk
    portfolio_drawdown = Column(Float)
    leverage_ratio = Column(Float)
    
    # Mode and settings
    trading_mode = Column(String(20), nullable=False)
    active_strategies = Column(Integer, default=0)
    risk_utilization = Column(Float)  # Percentage of risk budget used
    
    # Positions summary
    positions = Column(JSON)  # Current positions by symbol
    correlation_matrix = Column(JSON)  # Strategy correlation matrix
    
    # Performance
    daily_pnl = Column(Float)
    weekly_pnl = Column(Float)
    monthly_pnl = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_portfolio_timestamp', timestamp),
        Index('idx_portfolio_mode', trading_mode),
    )
    
    def __repr__(self):
        return f"<Portfolio({self.timestamp}: ${self.total_value_usd:.2f})>"


class RiskEvent(Base):
    """
    Risk management events and violations.
    
    Records all risk-related events including limit breaches,
    circuit breaker activations, and mode changes.
    """
    
    __tablename__ = "risk_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    
    # Event classification
    event_type = Column(String(50), nullable=False)  # 'limit_breach', 'circuit_breaker', etc.
    severity = Column(String(20), nullable=False)  # 'low', 'medium', 'high', 'critical'
    
    # Event details
    description = Column(Text, nullable=False)
    strategy_id = Column(String(100))  # May be null for portfolio-level events
    symbol = Column(String(20))
    
    # Risk metrics at event time
    current_value = Column(Float)
    limit_value = Column(Float)
    portfolio_balance = Column(Float)
    trading_mode = Column(String(20))
    
    # Actions taken
    action_taken = Column(String(100))  # 'position_closed', 'trading_paused', etc.
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    
    # Additional context
    event_metadata = Column(JSON)
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_risk_events_timestamp', timestamp),
        Index('idx_risk_events_type', event_type),
        Index('idx_risk_events_severity', severity),
        Index('idx_risk_events_strategy', strategy_id),
    )
    
    def __repr__(self):
        return f"<RiskEvent({self.event_type}: {self.severity})>"


class TaxEvent(Base):
    """
    Tax-related events for Australian CGT compliance.
    
    Tracks CGT events, AUD conversions, and generates data
    for tax reporting.
    """
    
    __tablename__ = "tax_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    financial_year = Column(String(10), nullable=False)  # e.g., '2024-25'
    
    # Event details
    event_type = Column(String(50), nullable=False)  # 'acquisition', 'disposal', 'income'
    trade_id = Column(Integer, ForeignKey('trades.id'))
    
    # Asset details
    symbol = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)
    
    # AUD values (required for tax)
    cost_base_aud = Column(Float)
    proceeds_aud = Column(Float)
    capital_gain_aud = Column(Float)
    
    # CGT calculations
    is_cgt_discount_eligible = Column(Boolean, default=False)
    holding_period_days = Column(Integer)
    discount_amount_aud = Column(Float)
    net_capital_gain_aud = Column(Float)
    
    # Currency conversion
    usd_aud_rate = Column(Float, nullable=False)
    rate_source = Column(String(50))  # 'RBA', 'manual', etc.
    
    # FIFO matching (for disposals)
    matched_acquisition_id = Column(Integer, ForeignKey('tax_events.id'))
    
    # Metadata
    trading_mode = Column(String(20))
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    trade = relationship("Trade", backref="tax_events")
    matched_acquisition = relationship("TaxEvent", remote_side=[id])
    
    __table_args__ = (
        Index('idx_tax_events_fy', financial_year),
        Index('idx_tax_events_type', event_type),
        Index('idx_tax_events_symbol', symbol),
        Index('idx_tax_events_timestamp', timestamp),
    )
    
    def __repr__(self):
        return f"<TaxEvent({self.event_type}: {self.symbol} FY{self.financial_year})>"


class MarketData(Base):
    """
    Market data storage for backtesting and analysis.
    
    Stores OHLCV data and derived indicators for all
    trading symbols and timeframes.
    """
    
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)  # '1h', '4h', '1d'
    timestamp = Column(DateTime, nullable=False)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Derived metrics
    vwap = Column(Float)  # Volume Weighted Average Price
    volatility = Column(Float)  # Realized volatility
    
    # Technical indicators (can be computed on-demand or stored)
    sma_20 = Column(Float)
    ema_20 = Column(Float)
    rsi_14 = Column(Float)
    atr_14 = Column(Float)
    
    # Data quality flags
    is_validated = Column(Boolean, default=False)
    has_anomalies = Column(Boolean, default=False)
    data_source = Column(String(50), default='bybit')
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_market_data'),
        Index('idx_market_data_symbol_tf', symbol, timeframe),
        Index('idx_market_data_timestamp', timestamp),
        CheckConstraint(high >= low, name='valid_high_low'),
        CheckConstraint(high >= open, name='valid_high_open'),
        CheckConstraint(high >= close, name='valid_high_close'),
        CheckConstraint(low <= open, name='valid_low_open'),
        CheckConstraint(low <= close, name='valid_low_close'),
        CheckConstraint(volume >= 0, name='non_negative_volume'),
    )
    
    def __repr__(self):
        return f"<MarketData({self.symbol} {self.timeframe} {self.timestamp}: {self.close})>"


class StrategyMetadata(Base):
    """
    Strategy metadata and configuration tracking.
    
    Stores strategy definitions, parameters, and lifecycle information.
    """
    
    __tablename__ = "strategy_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(100), nullable=False, unique=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Strategy classification
    strategy_type = Column(String(50), nullable=False)  # 'technical', 'ml', 'hybrid'
    category = Column(String(50))  # 'trend', 'mean_reversion', 'momentum'
    
    # Configuration
    parameters = Column(JSON)  # Strategy parameters
    symbols = Column(JSON)  # Traded symbols
    timeframes = Column(JSON)  # Used timeframes
    
    # Lifecycle
    status = Column(String(20), default='development')  # 'development', 'validation', 'live', 'retired'
    created_at = Column(DateTime, default=func.now())
    activated_at = Column(DateTime)
    retired_at = Column(DateTime)
    
    # Validation results
    validation_results = Column(JSON)
    last_validation = Column(DateTime)
    
    # Performance summary
    total_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    best_sharpe = Column(Float)
    max_drawdown = Column(Float)
    
    # Mode compatibility
    conservative_approved = Column(Boolean, default=False)
    aggressive_approved = Column(Boolean, default=False)
    
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_strategy_meta_status', status),
        Index('idx_strategy_meta_type', strategy_type),
    )
    
    def __repr__(self):
        return f"<StrategyMetadata({self.strategy_id}: {self.status})>"


class StrategyPipeline(Base):
    """
    AI Pipeline state tracking for automated strategy progression.
    
    Tracks strategies as they move through the three-column pipeline:
    Backtest → Paper Trading → Live Trading
    """
    
    __tablename__ = "strategy_pipeline"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(100), nullable=False, unique=True)
    strategy_name = Column(String(200), nullable=False)  # Human-readable name
    
    # Pipeline phase tracking
    current_phase = Column(String(20), nullable=False)  # 'backtest', 'paper', 'live', 'rejected'
    phase_start_time = Column(DateTime, nullable=False, default=func.now())
    phase_duration = Column(Integer)  # Duration in current phase (seconds)
    
    # Asset and strategy details
    asset_pair = Column(String(20), nullable=False)  # e.g., 'BTCUSDT'
    base_asset = Column(String(10), nullable=False)   # e.g., 'BTC'
    strategy_type = Column(String(50), nullable=False) # e.g., 'mean_reversion'
    strategy_description = Column(Text)
    
    # Backtest phase results
    backtest_score = Column(Float)     # Overall backtest score (0-100)
    backtest_return = Column(Float)    # Percentage return in backtest
    sharpe_ratio = Column(Float)       # Risk-adjusted return
    max_drawdown = Column(Float)       # Maximum drawdown percentage
    win_rate = Column(Float)           # Percentage of winning trades
    profit_factor = Column(Float)      # Profit factor ratio
    
    # Paper trading phase results
    paper_start_date = Column(DateTime)
    paper_end_date = Column(DateTime)
    paper_pnl = Column(Float)          # Paper trading P&L in USD
    paper_return_pct = Column(Float)   # Paper trading return percentage
    paper_trade_count = Column(Integer, default=0)
    paper_win_count = Column(Integer, default=0)
    
    # Live trading phase results
    live_start_date = Column(DateTime)
    live_pnl = Column(Float)           # Live trading P&L in USD
    live_return_pct = Column(Float)    # Live trading return percentage
    live_trade_count = Column(Integer, default=0)
    live_win_count = Column(Integer, default=0)
    
    # Pipeline transition criteria
    promotion_threshold = Column(Float, default=10.0)  # Percentage return needed for promotion
    graduation_threshold = Column(Float, default=10.0) # Percentage return needed for graduation
    rejection_threshold = Column(Float, default=-5.0)  # Maximum loss before rejection
    
    # Automation settings
    auto_promote = Column(Boolean, default=True)       # Auto-promote from backtest to paper
    auto_graduate = Column(Boolean, default=True)      # Auto-graduate from paper to live
    max_paper_duration = Column(Integer, default=604800) # Max paper trading time (7 days)
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    rejection_reason = Column(String(200))
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    promoted_at = Column(DateTime)      # Backtest → Paper
    graduated_at = Column(DateTime)     # Paper → Live
    rejected_at = Column(DateTime)
    
    # Risk metrics
    risk_score = Column(Float)          # Overall risk assessment (0-100)
    volatility_score = Column(Float)    # Volatility assessment
    correlation_risk = Column(Float)    # Correlation with other strategies
    
    # Performance tracking
    total_pnl = Column(Float, default=0.0)  # Combined P&L across all phases
    best_return = Column(Float)             # Best single-day return
    worst_return = Column(Float)            # Worst single-day return
    consistency_score = Column(Float)       # Performance consistency metric
    
    __table_args__ = (
        Index('idx_pipeline_strategy_id', strategy_id),
        Index('idx_pipeline_phase', current_phase),
        Index('idx_pipeline_asset', asset_pair),
        Index('idx_pipeline_active', is_active),
        Index('idx_pipeline_created', created_at),
        CheckConstraint(backtest_score >= 0, name='valid_backtest_score'),
        CheckConstraint(backtest_score <= 100, name='max_backtest_score'),
        CheckConstraint(risk_score >= 0, name='valid_risk_score'),
        CheckConstraint(risk_score <= 100, name='max_risk_score'),
    )
    
    def __repr__(self):
        return f"<StrategyPipeline({self.strategy_id}: {self.current_phase})>"
    
    @property
    def is_in_backtest(self) -> bool:
        """Check if strategy is in backtest phase."""
        return self.current_phase == 'backtest'
    
    @property
    def is_in_paper(self) -> bool:
        """Check if strategy is in paper trading phase."""
        return self.current_phase == 'paper'
    
    @property
    def is_live(self) -> bool:
        """Check if strategy is in live trading phase."""
        return self.current_phase == 'live'
    
    @property
    def is_rejected(self) -> bool:
        """Check if strategy has been rejected."""
        return self.current_phase == 'rejected'
    
    @property
    def current_phase_duration_hours(self) -> float:
        """Get current phase duration in hours."""
        if not self.phase_start_time:
            return 0.0
        
        duration = datetime.utcnow() - self.phase_start_time
        return duration.total_seconds() / 3600
    
    @property
    def paper_duration_hours(self) -> float:
        """Get paper trading duration in hours."""
        if not self.paper_start_date:
            return 0.0
        
        end_date = self.paper_end_date or datetime.utcnow()
        duration = end_date - self.paper_start_date
        return duration.total_seconds() / 3600
    
    @property
    def live_duration_hours(self) -> float:
        """Get live trading duration in hours."""
        if not self.live_start_date:
            return 0.0
        
        duration = datetime.utcnow() - self.live_start_date
        return duration.total_seconds() / 3600
    
    def ready_for_promotion(self) -> bool:
        """Check if strategy is ready for promotion to paper trading."""
        if self.current_phase != 'backtest':
            return False
        
        return (
            self.backtest_score and self.backtest_score >= 75.0 and
            self.sharpe_ratio and self.sharpe_ratio >= 1.5 and
            self.backtest_return and self.backtest_return >= 10.0
        )
    
    def ready_for_graduation(self) -> bool:
        """Check if strategy is ready for graduation to live trading."""
        if self.current_phase != 'paper':
            return False
        
        return (
            self.paper_return_pct and self.paper_return_pct >= self.graduation_threshold and
            self.paper_trade_count and self.paper_trade_count >= 5 and
            self.paper_duration_hours >= 72  # At least 3 days
        )
    
    def should_be_rejected(self) -> bool:
        """Check if strategy should be rejected from pipeline."""
        if self.current_phase == 'backtest':
            return (
                self.backtest_score and self.backtest_score < 60.0 or
                self.backtest_return and self.backtest_return <= self.rejection_threshold
            )
        elif self.current_phase == 'paper':
            return (
                self.paper_return_pct and self.paper_return_pct <= self.rejection_threshold or
                self.paper_duration_hours > (self.max_paper_duration / 3600)
            )
        
        return False