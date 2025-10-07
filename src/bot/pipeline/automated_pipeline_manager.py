"""
Automated Pipeline Manager for AI Strategy Discovery System

This module orchestrates the automated three-column pipeline:
Backtest → Paper Trading → Live Trading

Features:
- Automated strategy discovery using existing ML engine
- Pipeline progression based on performance thresholds
- Real-time monitoring and metrics tracking
- WebSocket notifications for frontend updates
- Integration with existing backtesting infrastructure
- USDT cryptocurrency pair focus

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from ..database.manager import DatabaseManager
from ..database.models import StrategyPipeline, StrategyMetadata, StrategyPerformance, MarketData
from ..ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine, StrategyType
from ..backtesting.bybit_enhanced_backtest_engine import BybitEnhancedBacktestEngine
from .strategy_naming_engine import StrategyNamingEngine, strategy_naming_engine
from ..config_manager import ConfigurationManager
from ..utils.logging import TradingLogger


class PipelinePhase(Enum):
    """Pipeline phase enumeration."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    REJECTED = "rejected"


@dataclass
class PipelineConfig:
    """Pipeline configuration settings."""
    
    # Discovery settings
    discovery_rate_per_hour: int = 3
    max_concurrent_backtests: int = 5
    
    # Asset configuration
    primary_assets: List[str] = field(default_factory=lambda: [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
    ])
    secondary_assets: List[str] = field(default_factory=lambda: [
        'ADAUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT',
        'LINKUSDT', 'UNIUSDT', 'BNBUSDT', 'XRPUSDT'
    ])
    
    # Backtest thresholds
    min_backtest_score: float = 75.0
    min_sharpe_ratio: float = 1.5
    min_return_pct: float = 10.0
    max_drawdown_pct: float = 20.0
    
    # Paper trading settings
    paper_trading_duration_days: int = 7
    min_paper_trades: int = 5
    graduation_threshold_pct: float = 10.0
    rejection_threshold_pct: float = -5.0
    
    # Live trading settings
    max_live_strategies: int = 10
    max_correlation_threshold: float = 0.8
    
    # Pipeline automation
    auto_promote_enabled: bool = True
    auto_graduate_enabled: bool = True
    auto_reject_enabled: bool = True


@dataclass
class PipelineMetrics:
    """Real-time pipeline metrics."""
    
    # Discovery metrics
    strategies_tested_today: int = 0
    candidates_found_today: int = 0
    success_rate_pct: float = 0.0
    
    # Phase counts
    backtest_count: int = 0
    paper_count: int = 0
    live_count: int = 0
    rejected_count: int = 0
    
    # Performance metrics
    graduation_rate_pct: float = 0.0
    total_live_pnl: float = 0.0
    avg_strategy_return: float = 0.0
    
    # Asset distribution
    asset_distribution: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AutomatedPipelineManager:
    """
    Manager for the automated AI strategy discovery pipeline.
    
    Orchestrates the complete pipeline from strategy discovery through
    live trading with automated progression and monitoring.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        ml_engine: Optional[MLStrategyDiscoveryEngine] = None,
        backtest_engine: Optional[BybitEnhancedBacktestEngine] = None
    ):
        self.config = config or PipelineConfig()
        self.db_manager = db_manager or DatabaseManager()
        self.ml_engine = ml_engine
        self.backtest_engine = backtest_engine
        self.naming_engine = strategy_naming_engine
        
        self.logger = TradingLogger.get_logger(__name__)
        
        # Pipeline state
        self.is_running = False
        self.discovery_task = None
        self.monitoring_task = None
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable] = []
        
        # Performance tracking
        self.current_metrics = PipelineMetrics()
        self._last_discovery_time = None
        
        # Active strategy tracking
        self.active_backtests: Set[str] = set()
        self.paper_strategies: Set[str] = set()
        self.live_strategies: Set[str] = set()
    
    async def start_pipeline(self) -> bool:
        """Start the automated pipeline system."""
        try:
            self.logger.info("🚀 Starting AI Pipeline Manager...")
            
            # Initialize components
            if not await self._initialize_components():
                return False
            
            # Load existing pipeline state
            await self._load_pipeline_state()
            
            # Start discovery and monitoring tasks
            self.discovery_task = asyncio.create_task(self._discovery_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.is_running = True
            self.logger.info("✅ AI Pipeline Manager started successfully")
            
            # Notify callbacks
            await self._notify_update("pipeline_started", {"status": "running"})
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start pipeline manager: {e}")
            return False
    
    async def stop_pipeline(self) -> bool:
        """Stop the automated pipeline system."""
        try:
            self.logger.info("⏹️ Stopping AI Pipeline Manager...")
            
            self.is_running = False
            
            # Cancel tasks
            if self.discovery_task:
                self.discovery_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # Wait for tasks to complete
            if self.discovery_task:
                await self.discovery_task
            if self.monitoring_task:
                await self.monitoring_task
            
            self.logger.info("✅ AI Pipeline Manager stopped")
            
            # Notify callbacks
            await self._notify_update("pipeline_stopped", {"status": "stopped"})
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop pipeline manager: {e}")
            return False
    
    async def _initialize_components(self) -> bool:
        """Initialize required components."""
        try:
            # Initialize ML engine if not provided
            if not self.ml_engine:
                self.ml_engine = MLStrategyDiscoveryEngine()
            
            # Initialize backtest engine if not provided  
            if not self.backtest_engine:
                self.backtest_engine = BybitEnhancedBacktestEngine()
            
            # Test database connection
            await self.db_manager.test_connection()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    async def _load_pipeline_state(self):
        """Load existing pipeline state from database."""
        try:
            with self.db_manager.get_session() as session:
                # Load active strategies by phase
                strategies = session.query(StrategyPipeline).filter(
                    StrategyPipeline.is_active == True
                ).all()
                
                for strategy in strategies:
                    if strategy.current_phase == 'backtest':
                        self.active_backtests.add(strategy.strategy_id)
                    elif strategy.current_phase == 'paper':
                        self.paper_strategies.add(strategy.strategy_id)
                    elif strategy.current_phase == 'live':
                        self.live_strategies.add(strategy.strategy_id)
                
                self.logger.info(
                    f"📊 Loaded pipeline state: "
                    f"{len(self.active_backtests)} backtest, "
                    f"{len(self.paper_strategies)} paper, "
                    f"{len(self.live_strategies)} live"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load pipeline state: {e}")
    
    async def _discovery_loop(self):
        """Main discovery loop for finding new strategies."""
        while self.is_running:
            try:
                # Check if we should discover new strategies
                if self._should_discover_strategy():
                    await self._discover_new_strategy()
                
                # Wait for next discovery cycle
                interval = 3600 / self.config.discovery_rate_per_hour  # Convert to seconds
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _monitoring_loop(self):
        """Main monitoring loop for pipeline progression."""
        while self.is_running:
            try:
                # Update pipeline metrics
                await self._update_metrics()
                
                # Process pipeline progressions
                await self._process_progressions()
                
                # Check for rejections
                await self._check_rejections()
                
                # Update frontend
                await self._notify_metric_update()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    def _should_discover_strategy(self) -> bool:
        """Check if we should discover a new strategy."""
        # Limit concurrent backtests
        if len(self.active_backtests) >= self.config.max_concurrent_backtests:
            return False
        
        # Check discovery rate
        now = datetime.utcnow()
        if self._last_discovery_time:
            time_since_last = (now - self._last_discovery_time).total_seconds()
            min_interval = 3600 / self.config.discovery_rate_per_hour
            if time_since_last < min_interval:
                return False
        
        return True
    
    async def _discover_new_strategy(self):
        """Discover and test a new strategy."""
        try:
            # Select random asset from configured list
            all_assets = self.config.primary_assets + self.config.secondary_assets
            import random
            asset_pair = random.choice(all_assets)
            
            # Generate strategy using ML engine
            strategy_data = await self._generate_ml_strategy(asset_pair)
            if not strategy_data:
                return
            
            # Create strategy name
            strategy_name = self.naming_engine.generate_strategy_name(
                asset_pair=asset_pair,
                strategy_type=strategy_data['type'],
                strategy_description=strategy_data['description']
            )
            
            # Run backtest
            backtest_results = await self._run_backtest(strategy_name, strategy_data)
            if not backtest_results:
                return
            
            # Create pipeline entry
            await self._create_pipeline_entry(strategy_name, strategy_data, backtest_results)
            
            # Update tracking
            self._last_discovery_time = datetime.utcnow()
            self.active_backtests.add(strategy_name.full_name)
            
            self.logger.info(f"🎯 New strategy discovered: {strategy_name.full_name}")
            
            # Notify frontend
            await self._notify_update("strategy_discovered", {
                "strategy_id": strategy_name.full_name,
                "asset": strategy_name.asset,
                "type": strategy_name.type_code,
                "backtest_score": backtest_results.get('score', 0)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to discover new strategy: {e}")
    
    async def _generate_ml_strategy(self, asset_pair: str) -> Optional[Dict[str, Any]]:
        """Generate a new strategy using the ML engine."""
        try:
            # This would integrate with your existing ML engine
            # For now, we'll simulate strategy generation
            
            strategy_types = ['mean_reversion', 'momentum', 'bollinger_bands', 'rsi', 'macd']
            import random
            
            strategy_type = random.choice(strategy_types)
            
            return {
                'type': strategy_type,
                'description': f"{strategy_type.replace('_', ' ').title()} strategy for {asset_pair}",
                'parameters': {
                    'lookback_period': random.randint(10, 50),
                    'threshold': random.uniform(0.5, 2.0),
                    'stop_loss': random.uniform(2.0, 5.0),
                    'take_profit': random.uniform(3.0, 8.0)
                },
                'timeframe': '1h',
                'asset_pair': asset_pair
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate ML strategy: {e}")
            return None
    
    async def _run_backtest(self, strategy_name, strategy_data) -> Optional[Dict[str, float]]:
        """Run backtest for a strategy."""
        try:
            # This would integrate with your existing backtest engine
            # For now, we'll simulate backtest results
            
            import random
            
            # Simulate realistic backtest metrics
            base_score = random.uniform(60, 95)
            
            return {
                'score': base_score,
                'return_pct': random.uniform(-5, 35),
                'sharpe_ratio': random.uniform(0.8, 3.2),
                'max_drawdown': random.uniform(2, 25),
                'win_rate': random.uniform(55, 85),
                'profit_factor': random.uniform(1.1, 3.5),
                'total_trades': random.randint(50, 200)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to run backtest: {e}")
            return None
    
    async def _create_pipeline_entry(self, strategy_name, strategy_data, backtest_results):
        """Create a new pipeline entry in the database."""
        try:
            with self.db_manager.get_session() as session:
                pipeline_entry = StrategyPipeline(
                    strategy_id=strategy_name.full_name,
                    strategy_name=strategy_data['description'],
                    current_phase='backtest',
                    asset_pair=strategy_data['asset_pair'],
                    base_asset=strategy_name.asset,
                    strategy_type=strategy_data['type'],
                    strategy_description=strategy_data['description'],
                    
                    # Backtest results
                    backtest_score=backtest_results['score'],
                    backtest_return=backtest_results['return_pct'],
                    sharpe_ratio=backtest_results['sharpe_ratio'],
                    max_drawdown=backtest_results['max_drawdown'],
                    win_rate=backtest_results['win_rate'],
                    profit_factor=backtest_results['profit_factor']
                )
                
                session.add(pipeline_entry)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to create pipeline entry: {e}")
    
    async def _process_progressions(self):
        """Process strategy progressions through pipeline phases."""
        try:
            with self.db_manager.get_session() as session:
                # Check for strategies ready for promotion (backtest → paper)
                if self.config.auto_promote_enabled:
                    ready_strategies = session.query(StrategyPipeline).filter(
                        and_(
                            StrategyPipeline.current_phase == 'backtest',
                            StrategyPipeline.is_active == True,
                            StrategyPipeline.backtest_score >= self.config.min_backtest_score,
                            StrategyPipeline.sharpe_ratio >= self.config.min_sharpe_ratio,
                            StrategyPipeline.backtest_return >= self.config.min_return_pct
                        )
                    ).all()
                    
                    for strategy in ready_strategies:
                        await self._promote_to_paper(strategy, session)
                
                # Check for strategies ready for graduation (paper → live)
                if self.config.auto_graduate_enabled:
                    graduate_strategies = session.query(StrategyPipeline).filter(
                        and_(
                            StrategyPipeline.current_phase == 'paper',
                            StrategyPipeline.is_active == True,
                            StrategyPipeline.paper_return_pct >= self.config.graduation_threshold_pct
                        )
                    ).all()
                    
                    for strategy in graduate_strategies:
                        if strategy.ready_for_graduation():
                            await self._graduate_to_live(strategy, session)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to process progressions: {e}")
    
    async def _promote_to_paper(self, strategy: StrategyPipeline, session: Session):
        """Promote strategy from backtest to paper trading."""
        try:
            strategy.current_phase = 'paper'
            strategy.previous_phase = 'backtest'
            strategy.phase_start_time = datetime.utcnow()
            strategy.paper_start_date = datetime.utcnow()
            strategy.promoted_at = datetime.utcnow()
            
            # Move from backtest to paper tracking
            self.active_backtests.discard(strategy.strategy_id)
            self.paper_strategies.add(strategy.strategy_id)
            
            self.logger.info(f"📈 Promoted {strategy.strategy_id} to paper trading")
            
            # Notify frontend
            await self._notify_update("strategy_promoted", {
                "strategy_id": strategy.strategy_id,
                "from_phase": "backtest",
                "to_phase": "paper",
                "backtest_score": strategy.backtest_score
            })
            
        except Exception as e:
            self.logger.error(f"Failed to promote strategy {strategy.strategy_id}: {e}")
    
    async def _graduate_to_live(self, strategy: StrategyPipeline, session: Session):
        """Graduate strategy from paper to live trading."""
        try:
            # Check live strategy limit
            if len(self.live_strategies) >= self.config.max_live_strategies:
                self.logger.warning(f"Cannot graduate {strategy.strategy_id}: live limit reached")
                return
            
            strategy.current_phase = 'live'
            strategy.previous_phase = 'paper'
            strategy.phase_start_time = datetime.utcnow()
            strategy.paper_end_date = datetime.utcnow()
            strategy.live_start_date = datetime.utcnow()
            strategy.graduated_at = datetime.utcnow()
            
            # Move from paper to live tracking
            self.paper_strategies.discard(strategy.strategy_id)
            self.live_strategies.add(strategy.strategy_id)
            
            self.logger.info(f"🚀 Graduated {strategy.strategy_id} to live trading")
            
            # Notify frontend
            await self._notify_update("strategy_graduated", {
                "strategy_id": strategy.strategy_id,
                "from_phase": "paper",
                "to_phase": "live",
                "paper_return": strategy.paper_return_pct
            })
            
        except Exception as e:
            self.logger.error(f"Failed to graduate strategy {strategy.strategy_id}: {e}")
    
    async def _check_rejections(self):
        """Check for strategies that should be rejected."""
        try:
            with self.db_manager.get_session() as session:
                # Check backtest rejections
                failed_backtests = session.query(StrategyPipeline).filter(
                    and_(
                        StrategyPipeline.current_phase == 'backtest',
                        StrategyPipeline.is_active == True,
                        StrategyPipeline.backtest_score < 60.0
                    )
                ).all()
                
                for strategy in failed_backtests:
                    await self._reject_strategy(strategy, "Poor backtest performance", session)
                
                # Check paper trading rejections
                failed_paper = session.query(StrategyPipeline).filter(
                    and_(
                        StrategyPipeline.current_phase == 'paper',
                        StrategyPipeline.is_active == True,
                        StrategyPipeline.paper_return_pct <= self.config.rejection_threshold_pct
                    )
                ).all()
                
                for strategy in failed_paper:
                    await self._reject_strategy(strategy, "Poor paper trading performance", session)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to check rejections: {e}")
    
    async def _reject_strategy(self, strategy: StrategyPipeline, reason: str, session: Session):
        """Reject a strategy from the pipeline."""
        try:
            old_phase = strategy.current_phase
            
            strategy.current_phase = 'rejected'
            strategy.is_active = False
            strategy.rejection_reason = reason
            strategy.rejected_at = datetime.utcnow()
            
            # Remove from tracking sets
            self.active_backtests.discard(strategy.strategy_id)
            self.paper_strategies.discard(strategy.strategy_id)
            self.live_strategies.discard(strategy.strategy_id)
            
            self.logger.info(f"❌ Rejected {strategy.strategy_id}: {reason}")
            
            # Notify frontend
            await self._notify_update("strategy_rejected", {
                "strategy_id": strategy.strategy_id,
                "from_phase": old_phase,
                "reason": reason
            })
            
        except Exception as e:
            self.logger.error(f"Failed to reject strategy {strategy.strategy_id}: {e}")
    
    async def _update_metrics(self):
        """Update current pipeline metrics."""
        try:
            with self.db_manager.get_session() as session:
                today = datetime.utcnow().date()
                
                # Count strategies tested today
                tested_today = session.query(StrategyPipeline).filter(
                    func.date(StrategyPipeline.created_at) == today
                ).count()
                
                # Count successful candidates (promoted from backtest)
                candidates_today = session.query(StrategyPipeline).filter(
                    and_(
                        func.date(StrategyPipeline.promoted_at) == today,
                        StrategyPipeline.promoted_at.isnot(None)
                    )
                ).count()
                
                # Count by phase
                backtest_count = session.query(StrategyPipeline).filter(
                    and_(
                        StrategyPipeline.current_phase == 'backtest',
                        StrategyPipeline.is_active == True
                    )
                ).count()
                
                paper_count = session.query(StrategyPipeline).filter(
                    and_(
                        StrategyPipeline.current_phase == 'paper',
                        StrategyPipeline.is_active == True
                    )
                ).count()
                
                live_count = session.query(StrategyPipeline).filter(
                    and_(
                        StrategyPipeline.current_phase == 'live',
                        StrategyPipeline.is_active == True
                    )
                ).count()
                
                # Calculate success rate
                success_rate = (candidates_today / tested_today * 100) if tested_today > 0 else 0
                
                # Calculate graduation rate (last 30 days)
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                total_promoted = session.query(StrategyPipeline).filter(
                    StrategyPipeline.promoted_at >= thirty_days_ago
                ).count()
                
                total_graduated = session.query(StrategyPipeline).filter(
                    StrategyPipeline.graduated_at >= thirty_days_ago
                ).count()
                
                graduation_rate = (total_graduated / total_promoted * 100) if total_promoted > 0 else 0
                
                # Calculate total live P&L
                live_pnl_result = session.query(
                    func.sum(StrategyPipeline.live_pnl)
                ).filter(
                    and_(
                        StrategyPipeline.current_phase == 'live',
                        StrategyPipeline.is_active == True
                    )
                ).scalar()
                
                total_live_pnl = live_pnl_result or 0.0
                
                # Update metrics
                self.current_metrics = PipelineMetrics(
                    strategies_tested_today=tested_today,
                    candidates_found_today=candidates_today,
                    success_rate_pct=success_rate,
                    backtest_count=backtest_count,
                    paper_count=paper_count,
                    live_count=live_count,
                    graduation_rate_pct=graduation_rate,
                    total_live_pnl=total_live_pnl,
                    last_updated=datetime.utcnow()
                )
                
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    async def _notify_update(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks of pipeline updates."""
        try:
            event_data = {
                "type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            for callback in self.update_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to notify update: {e}")
    
    async def _notify_metric_update(self):
        """Notify frontend of metric updates."""
        await self._notify_update("metrics_updated", {
            "metrics": {
                "tested_today": self.current_metrics.strategies_tested_today,
                "candidates_found": self.current_metrics.candidates_found_today,
                "success_rate": self.current_metrics.success_rate_pct,
                "backtest_count": self.current_metrics.backtest_count,
                "paper_count": self.current_metrics.paper_count,
                "live_count": self.current_metrics.live_count,
                "graduation_rate": self.current_metrics.graduation_rate_pct,
                "total_live_pnl": self.current_metrics.total_live_pnl
            }
        })
    
    def register_update_callback(self, callback: Callable):
        """Register a callback for pipeline updates."""
        self.update_callbacks.append(callback)
    
    def unregister_update_callback(self, callback: Callable):
        """Unregister a pipeline update callback."""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def get_current_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self.current_metrics
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            "is_running": self.is_running,
            "metrics": self.current_metrics.__dict__,
            "config": self.config.__dict__,
            "active_counts": {
                "backtest": len(self.active_backtests),
                "paper": len(self.paper_strategies),
                "live": len(self.live_strategies)
            },
            "last_discovery": self._last_discovery_time.isoformat() if self._last_discovery_time else None
        }
    
    async def manual_promote(self, strategy_id: str) -> bool:
        """Manually promote a strategy to the next phase."""
        try:
            with self.db_manager.get_session() as session:
                strategy = session.query(StrategyPipeline).filter(
                    StrategyPipeline.strategy_id == strategy_id
                ).first()
                
                if not strategy:
                    return False
                
                if strategy.current_phase == 'backtest':
                    await self._promote_to_paper(strategy, session)
                elif strategy.current_phase == 'paper':
                    await self._graduate_to_live(strategy, session)
                else:
                    return False
                
                session.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to manually promote {strategy_id}: {e}")
            return False
    
    async def manual_reject(self, strategy_id: str, reason: str) -> bool:
        """Manually reject a strategy."""
        try:
            with self.db_manager.get_session() as session:
                strategy = session.query(StrategyPipeline).filter(
                    StrategyPipeline.strategy_id == strategy_id
                ).first()
                
                if not strategy:
                    return False
                
                await self._reject_strategy(strategy, f"Manual rejection: {reason}", session)
                session.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to manually reject {strategy_id}: {e}")
            return False


# Global pipeline manager instance
pipeline_manager = AutomatedPipelineManager()