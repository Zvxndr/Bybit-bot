# Class Reference


## src.analytics.advanced_analytics.AnalyticsLevel

Analytics complexity and resource levels

**File:** `src/analytics\advanced_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BASIC`

- `STANDARD`

- `ADVANCED`

- `ENTERPRISE`


---

## src.analytics.advanced_analytics.DataFrequency

Data update frequency options

**File:** `src/analytics\advanced_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TICK`

- `SECOND`

- `MINUTE`

- `FIVE_MINUTE`

- `FIFTEEN_MINUTE`

- `HOUR`

- `FOUR_HOUR`

- `DAILY`


---

## src.analytics.advanced_analytics.AlertSeverity

Alert severity levels

**File:** `src/analytics\advanced_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `CRITICAL`

- `EMERGENCY`


---

## src.analytics.advanced_analytics.MarketDataPoint

Individual market data point

**File:** `src/analytics\advanced_analytics.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.analytics.advanced_analytics.TradingSignal

Trading signal with confidence and metadata

**File:** `src/analytics\advanced_analytics.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.analytics.advanced_analytics.PerformanceMetrics

Comprehensive performance metrics

**File:** `src/analytics\advanced_analytics.py`



### Methods



### Attributes



---

## src.analytics.advanced_analytics.RealTimeDataProcessor

High-performance real-time data processing engine

**File:** `src/analytics\advanced_analytics.py`



### Methods


#### __init__()

Method description here.


#### _get_processing_config()

Method description here.


#### _buffer_to_dataframe()

Method description here.


#### get_processing_stats()

Method description here.



### Attributes



---

## src.analytics.advanced_analytics.PredictiveAnalyticsEngine

Advanced predictive analytics using machine learning

**File:** `src/analytics\advanced_analytics.py`



### Methods


#### __init__()

Method description here.


#### get_model_performance()

Method description here.



### Attributes



---

## src.analytics.advanced_analytics.AnalyticsDashboard

Real-time analytics dashboard with web interface

**File:** `src/analytics\advanced_analytics.py`



### Methods


#### __init__()

Method description here.


#### _setup_routes()

Method description here.



### Attributes



---

## src.analytics.advanced_analytics.AdvancedAnalyticsPlatform

Main analytics platform orchestrator

**File:** `src/analytics\advanced_analytics.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.



### Attributes



---

## src.api.graduation_api.StrategyStageAPI

API representation of strategy stages

**File:** `src/api\graduation_api.py`


**Inherits from:** str, Enum


### Methods



### Attributes


- `RESEARCH`

- `PAPER_VALIDATION`

- `LIVE_CANDIDATE`

- `LIVE_TRADING`

- `UNDER_REVIEW`

- `RETIRED`


---

## src.api.graduation_api.PerformanceMetricsAPI

API model for performance metrics

**File:** `src/api\graduation_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.graduation_api.StrategyRecordAPI

API model for strategy record

**File:** `src/api\graduation_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.graduation_api.StrategyRegistrationRequest

Request model for strategy registration

**File:** `src/api\graduation_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.graduation_api.ManualGraduationRequest

Request model for manual strategy graduation

**File:** `src/api\graduation_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.graduation_api.GraduationCriteriaUpdateRequest

Request model for updating graduation criteria

**File:** `src/api\graduation_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.APIPermission

API permission levels

**File:** `src/api\trading_bot_api.py`


**Inherits from:** Enum


### Methods



### Attributes


- `READ_ONLY`

- `CONTROL`

- `ADMIN`

- `SUPER_ADMIN`


---

## src.api.trading_bot_api.SystemCommand

System control commands

**File:** `src/api\trading_bot_api.py`


**Inherits from:** Enum


### Methods



### Attributes


- `START`

- `STOP`

- `RESTART`

- `PAUSE`

- `RESUME`

- `RESET`

- `SHUTDOWN`


---

## src.api.trading_bot_api.APIKey

API key information

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.UserSession

User session information

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.SystemStatusResponse

System status API response

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.MetricsResponse

Metrics API response

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.AlertResponse

Alert API response

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.ConfigurationResponse

Configuration API response

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.TradingStatusResponse

Trading status API response

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.SystemCommandRequest

System command request

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.ConfigUpdateRequest

Configuration update request

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.WebSocketMessage

WebSocket message structure

**File:** `src/api\trading_bot_api.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.api.trading_bot_api.RateLimitConfig

Rate limiting configuration

**File:** `src/api\trading_bot_api.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.api.trading_bot_api.SecurityManager

Advanced security management system

Handles authentication, authorization, API key management,
and security policies.

**File:** `src/api\trading_bot_api.py`



### Methods


#### __init__()

Method description here.


#### _create_default_keys()

Method description here.


#### generate_api_key()

Method description here.


#### validate_api_key()

Method description here.


#### create_jwt_token()

Method description here.


#### validate_jwt_token()

Method description here.


#### create_session()

Method description here.


#### validate_session()

Method description here.


#### check_permission()

Method description here.


#### record_failed_attempt()

Method description here.


#### is_locked_out()

Method description here.



### Attributes



---

## src.api.trading_bot_api.WebSocketManager

Advanced WebSocket connection management

Handles real-time communication with authentication,
subscription management, and message broadcasting.

**File:** `src/api\trading_bot_api.py`



### Methods


#### __init__()

Method description here.


#### disconnect()

Method description here.


#### get_connection_stats()

Method description here.



### Attributes



---

## src.api.trading_bot_api.TradingBotAPI

Comprehensive Trading Bot API Server

Provides REST API and WebSocket interfaces for complete
trading bot system management and monitoring.

**File:** `src/api\trading_bot_api.py`



### Methods


#### __init__()

Method description here.


#### _setup_middleware()

Method description here.


#### _setup_routes()

Method description here.


#### _setup_websockets()

Method description here.


#### _setup_background_tasks()

Method description here.



### Attributes



---

## src.bot.config.TradingModeConfig

Configuration for a specific trading mode.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.AggressiveModeConfig

Configuration for aggressive trading mode with dynamic risk scaling.

**File:** `src/bot\config.py`


**Inherits from:** TradingModeConfig


### Methods


#### max_risk_must_be_greater_than_min()

Method description here.



### Attributes



---

## src.bot.config.ConservativeModeConfig

Configuration for conservative trading mode with fixed risk.

**File:** `src/bot\config.py`


**Inherits from:** TradingModeConfig


### Methods



### Attributes



---

## src.bot.config.TradingConfig

Main trading configuration.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.ExchangeConfig

Exchange connection and trading configuration.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.DatabaseConfig

Database configuration with development/production profiles.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.MLConfig

Machine learning and feature engineering configuration.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.BacktestConfig

Backtesting and validation configuration.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.RiskConfig

Risk management configuration.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.TaxConfig

Tax reporting configuration (Australia-specific).

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.LoggingConfig

Logging configuration.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.DashboardConfig

Dashboard configuration.

**File:** `src/bot\config.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.config.Config

Pydantic configuration.

**File:** `src/bot\config.py`



### Methods



### Attributes


- `extra`

- `validate_assignment`


---

## src.bot.config_manager.Environment

Deployment environments

**File:** `src/bot\config_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DEVELOPMENT`

- `STAGING`

- `PRODUCTION`

- `TESTING`


---

## src.bot.config_manager.ConfigurationError

Configuration-related errors

**File:** `src/bot\config_manager.py`


**Inherits from:** Exception


### Methods



### Attributes



---

## src.bot.config_manager.EnvironmentCredentials

Environment-specific exchange credentials

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.ExchangeConfig

Exchange connection configuration with environment-specific credentials

**File:** `src/bot\config_manager.py`



### Methods


#### __post_init__()

Method description here.


#### get_credentials()

Method description here.


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.TradingConfig

Trading strategy configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.RiskManagementConfig

Risk management configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.BacktestingConfig

Backtesting configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.MonitoringConfig

System monitoring configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.AlertingConfig

Alerting and notification configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.DatabaseConfig

Database configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.TaxReportingConfig

Tax and reporting configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.AdvancedFeaturesConfig

Advanced features configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.APIConfig

API interface configuration

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.ComprehensiveConfig

Main configuration class containing all subsystems

**File:** `src/bot\config_manager.py`



### Methods


#### validate()

Method description here.



### Attributes



---

## src.bot.config_manager.ConfigurationManager

Comprehensive Configuration Management System

Handles loading, validation, and management of all bot configurations
with support for multiple environments and dynamic updates.

**File:** `src/bot\config_manager.py`



### Methods


#### __init__()

Method description here.


#### load_config()

Method description here.


#### _load_from_file()

Method description here.


#### _dict_to_config()

Method description here.


#### _apply_env_overrides()

Method description here.


#### _convert_env_value()

Method description here.


#### _set_nested_value()

Method description here.


#### _load_environment_credentials()

Method description here.


#### _apply_environment_settings()

Method description here.


#### save_config()

Method description here.


#### update_config()

Method description here.


#### get_config_summary()

Method description here.


#### create_default_config_files()

Method description here.


#### _calculate_config_hash()

Method description here.


#### _unflatten_dict()

Method description here.


#### is_production()

Method description here.


#### is_testnet()

Method description here.


#### get_current_credentials()

Method description here.



### Attributes



---

## src.bot.core.TradingBot

Main trading bot class that orchestrates all components.

This class manages the complete lifecycle of the trading bot including:
- Component initialization
- Data collection and management
- Strategy execution
- Risk management
- Performance monitoring

**File:** `src/bot\core.py`



### Methods


#### __init__()

Method description here.


#### _update_stats()

Method description here.


#### get_status()

Method description here.


#### get_health()

Method description here.



### Attributes



---

## src.bot.integrated_trading_bot.BotStatus

Bot operational status

**File:** `src/bot\integrated_trading_bot.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INITIALIZING`

- `STARTING`

- `RUNNING`

- `PAUSED`

- `STOPPING`

- `STOPPED`

- `ERROR`

- `MAINTENANCE`


---

## src.bot.integrated_trading_bot.BotConfiguration

Comprehensive bot configuration

**File:** `src/bot\integrated_trading_bot.py`



### Methods



### Attributes



---

## src.bot.integrated_trading_bot.IntegratedTradingBot

Unified Trading Bot - Phase 10 Integration

This class integrates all trading system phases into a cohesive,
production-ready trading bot with enterprise-grade capabilities.

**File:** `src/bot\integrated_trading_bot.py`



### Methods


#### __init__()

Method description here.


#### _setup_logging()

Method description here.


#### _validate_credentials()

Method description here.


#### _initialize_components()

Method description here.


#### _calculate_rsi()

Method description here.


#### get_strategy_graduation_status()

Method description here.



### Attributes



---

## src.bot.main.GracefulShutdown

Handle graceful shutdown of the trading bot.

**File:** `src/bot\main.py`



### Methods


#### __init__()

Method description here.


#### signal_handler()

Method description here.


#### register_bot()

Method description here.



### Attributes



---

## src.bot.strategy_graduation.StrategyStage

Strategy lifecycle stages

**File:** `src/bot\strategy_graduation.py`


**Inherits from:** Enum


### Methods



### Attributes


- `RESEARCH`

- `PAPER_VALIDATION`

- `LIVE_CANDIDATE`

- `LIVE_TRADING`

- `UNDER_REVIEW`

- `RETIRED`


---

## src.bot.strategy_graduation.GraduationDecision

Graduation decision types

**File:** `src/bot\strategy_graduation.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PROMOTE`

- `MAINTAIN`

- `DEMOTE`

- `RETIRE`


---

## src.bot.strategy_graduation.PerformanceMetrics

Strategy performance metrics for graduation assessment

**File:** `src/bot\strategy_graduation.py`



### Methods


#### calculate_graduation_score()

Method description here.



### Attributes



---

## src.bot.strategy_graduation.GraduationCriteria

Criteria for strategy graduation between stages

**File:** `src/bot\strategy_graduation.py`



### Methods



### Attributes



---

## src.bot.strategy_graduation.StrategyRecord

Complete record of a strategy's lifecycle

**File:** `src/bot\strategy_graduation.py`



### Methods


#### add_stage_change()

Method description here.


#### add_performance_snapshot()

Method description here.


#### update_capital_allocation()

Method description here.



### Attributes



---

## src.bot.strategy_graduation.StrategyGraduationManager

Manages automatic strategy graduation from paper trading to live trading

**File:** `src/bot\strategy_graduation.py`



### Methods


#### __init__()

Method description here.


#### register_strategy()

Method description here.


#### _evaluate_paper_to_candidate()

Method description here.


#### _evaluate_candidate_to_live()

Method description here.


#### _evaluate_live_maintenance()

Method description here.


#### _evaluate_under_review()

Method description here.


#### _calculate_initial_capital()

Method description here.


#### _get_days_in_stage()

Method description here.


#### _get_recent_metrics()

Method description here.


#### _check_strategy_correlation()

Method description here.


#### _calculate_config_similarity()

Method description here.


#### _check_demotion_triggers()

Method description here.


#### _get_recent_sharpe()

Method description here.


#### _get_next_stage()

Method description here.


#### _get_previous_stage()

Method description here.


#### _load_strategy_records()

Method description here.


#### _save_strategy_record()

Method description here.


#### _serialize_strategy_record()

Method description here.


#### _deserialize_strategy_record()

Method description here.


#### _save_evaluation_summary()

Method description here.


#### _get_stage_distribution()

Method description here.


#### get_graduation_report()

Method description here.


#### _get_performance_summary()

Method description here.


#### _get_recent_graduations()

Method description here.


#### _get_capital_allocation_summary()

Method description here.



### Attributes



---

## src.bot.advanced.automated_reporter.ReportType

Report types

**File:** `src/bot\advanced\automated_reporter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DAILY`

- `WEEKLY`

- `MONTHLY`

- `QUARTERLY`

- `ANNUAL`

- `CUSTOM`

- `STRATEGY`

- `PORTFOLIO`

- `RISK`

- `TAX`


---

## src.bot.advanced.automated_reporter.ReportFormat

Report formats

**File:** `src/bot\advanced\automated_reporter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PDF`

- `HTML`

- `EMAIL`

- `INTERACTIVE`

- `JSON`

- `EXCEL`


---

## src.bot.advanced.automated_reporter.ReportConfig

Report configuration

**File:** `src/bot\advanced\automated_reporter.py`



### Methods



### Attributes



---

## src.bot.advanced.automated_reporter.PerformanceData

Performance data for reporting

**File:** `src/bot\advanced\automated_reporter.py`



### Methods



### Attributes



---

## src.bot.advanced.automated_reporter.AutomatedReporter

Comprehensive Automated Reporting System

This class provides sophisticated reporting capabilities with multiple
output formats and distribution methods.

**File:** `src/bot\advanced\automated_reporter.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _load_templates()

Method description here.


#### generate_daily_report()

Method description here.


#### generate_weekly_report()

Method description here.


#### generate_monthly_report()

Method description here.


#### _calculate_daily_metrics()

Method description here.


#### _calculate_period_metrics()

Method description here.


#### _generate_daily_charts()

Method description here.


#### _generate_weekly_charts()

Method description here.


#### _generate_monthly_charts()

Method description here.


#### _analyze_portfolio_performance()

Method description here.


#### _analyze_risk_metrics()

Method description here.


#### _analyze_strategy_performance()

Method description here.


#### _analyze_regime_impact()

Method description here.


#### _generate_alerts()

Method description here.


#### _filter_data_by_period()

Method description here.


#### create_pdf_report()

Method description here.


#### create_html_report()

Method description here.


#### send_email_report()

Method description here.


#### save_report()

Method description here.


#### _serialize_for_json()

Method description here.


#### _calculate_comparative_metrics()

Method description here.


#### _calculate_risk_attribution()

Method description here.


#### _analyze_weekly_strategy_performance()

Method description here.


#### _analyze_optimization_performance()

Method description here.


#### _calculate_ytd_analysis()

Method description here.


#### _analyze_monthly_regime_performance()

Method description here.


#### _analyze_tax_implications()

Method description here.


#### _analyze_rebalancing_activity()

Method description here.



### Attributes



---

## src.bot.advanced.example.AdvancedTradingSystem

Integrated Advanced Trading System

This class demonstrates the integration of all Phase 9 advanced features
into a cohesive trading system with sophisticated market analysis and
adaptive behavior.

**File:** `src/bot\advanced\example.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _generate_trading_recommendation()

Method description here.


#### get_system_status()

Method description here.



### Attributes



---

## src.bot.advanced.news_analyzer.SentimentLevel

Sentiment levels

**File:** `src/bot\advanced\news_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `VERY_NEGATIVE`

- `NEGATIVE`

- `NEUTRAL`

- `POSITIVE`

- `VERY_POSITIVE`


---

## src.bot.advanced.news_analyzer.NewsCategory

News categories

**File:** `src/bot\advanced\news_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CRYPTO`

- `REGULATORY`

- `MARKET`

- `ECONOMIC`

- `TECHNICAL`

- `SECURITY`

- `ADOPTION`

- `INSTITUTIONAL`


---

## src.bot.advanced.news_analyzer.EventImpact

Event impact levels

**File:** `src/bot\advanced\news_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW`

- `MEDIUM`

- `HIGH`

- `CRITICAL`


---

## src.bot.advanced.news_analyzer.NewsArticle

News article data structure

**File:** `src/bot\advanced\news_analyzer.py`



### Methods



### Attributes



---

## src.bot.advanced.news_analyzer.MarketEvent

Market event data structure

**File:** `src/bot\advanced\news_analyzer.py`



### Methods



### Attributes



---

## src.bot.advanced.news_analyzer.SentimentAnalysis

Sentiment analysis result

**File:** `src/bot\advanced\news_analyzer.py`



### Methods



### Attributes



---

## src.bot.advanced.news_analyzer.NewsAnalyzer

Comprehensive News Sentiment Analysis System

This class provides sophisticated news sentiment analysis with
market event detection and trading recommendations.

**File:** `src/bot\advanced\news_analyzer.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _initialize_news_sources()

Method description here.


#### _initialize_sentiment_analyzer()

Method description here.


#### _clean_text()

Method description here.


#### _categorize_article()

Method description here.


#### analyze_sentiment()

Method description here.


#### _calculate_article_sentiment()

Method description here.


#### _adjust_sentiment_with_keywords()

Method description here.


#### _calculate_impact_score()

Method description here.


#### _score_to_sentiment_level()

Method description here.


#### _calculate_confidence()

Method description here.


#### _generate_recommendation()

Method description here.


#### _should_recommend_blackout()

Method description here.


#### should_halt_trading()

Method description here.


#### add_blackout_period()

Method description here.


#### get_sentiment_summary()

Method description here.


#### _cleanup_old_data()

Method description here.



### Attributes



---

## src.bot.advanced.parameter_optimizer.OptimizationMethod

Parameter optimization methods

**File:** `src/bot\advanced\parameter_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `GRID_SEARCH`

- `RANDOM_SEARCH`

- `BAYESIAN`

- `GENETIC_ALGORITHM`

- `PARTICLE_SWARM`

- `SIMULATED_ANNEALING`

- `WALK_FORWARD`

- `ROLLING_WINDOW`


---

## src.bot.advanced.parameter_optimizer.ObjectiveFunction

Optimization objective functions

**File:** `src/bot\advanced\parameter_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SHARPE_RATIO`

- `CALMAR_RATIO`

- `SORTINO_RATIO`

- `MAX_RETURN`

- `MIN_VOLATILITY`

- `MIN_DRAWDOWN`

- `CUSTOM_UTILITY`

- `MULTI_OBJECTIVE`


---

## src.bot.advanced.parameter_optimizer.ParameterBounds

Parameter bounds for optimization

**File:** `src/bot\advanced\parameter_optimizer.py`



### Methods



### Attributes



---

## src.bot.advanced.parameter_optimizer.OptimizationResult

Optimization result data structure

**File:** `src/bot\advanced\parameter_optimizer.py`



### Methods



### Attributes



---

## src.bot.advanced.parameter_optimizer.ParameterDrift

Parameter drift detection result

**File:** `src/bot\advanced\parameter_optimizer.py`



### Methods



### Attributes



---

## src.bot.advanced.parameter_optimizer.ParameterOptimizer

Dynamic Parameter Optimization System

This class provides comprehensive parameter optimization with
continuous adaptation and drift detection.

**File:** `src/bot\advanced\parameter_optimizer.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### define_parameter_space()

Method description here.


#### optimize_parameters()

Method description here.


#### _optimize_bayesian()

Method description here.


#### _optimize_grid_search()

Method description here.


#### _optimize_random_search()

Method description here.


#### _optimize_genetic()

Method description here.


#### _optimize_walk_forward()

Method description here.


#### detect_parameter_drift()

Method description here.


#### should_reoptimize()

Method description here.


#### _calculate_parameter_stability()

Method description here.


#### _calculate_parameter_sensitivity()

Method description here.


#### create_multi_objective_function()

Method description here.


#### get_optimization_summary()

Method description here.


#### save_optimization_state()

Method description here.


#### load_optimization_state()

Method description here.



### Attributes



---

## src.bot.advanced.portfolio_optimizer.OptimizationMethod

Portfolio optimization methods

**File:** `src/bot\advanced\portfolio_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MEAN_VARIANCE`

- `MIN_VARIANCE`

- `MAX_SHARPE`

- `RISK_PARITY`

- `EQUAL_RISK_CONTRIB`

- `BLACK_LITTERMAN`

- `MAX_DIVERSIFICATION`

- `MIN_CORRELATION`

- `HIERARCHICAL`

- `REGIME_AWARE`


---

## src.bot.advanced.portfolio_optimizer.OptimizationConstraints

Portfolio optimization constraints

**File:** `src/bot\advanced\portfolio_optimizer.py`



### Methods



### Attributes



---

## src.bot.advanced.portfolio_optimizer.OptimizationResult

Portfolio optimization result

**File:** `src/bot\advanced\portfolio_optimizer.py`



### Methods



### Attributes



---

## src.bot.advanced.portfolio_optimizer.PortfolioMetrics

Portfolio performance metrics

**File:** `src/bot\advanced\portfolio_optimizer.py`



### Methods



### Attributes



---

## src.bot.advanced.portfolio_optimizer.PortfolioOptimizer

Advanced Portfolio Optimization System

This class provides comprehensive portfolio optimization using various
methods and sophisticated risk models.

**File:** `src/bot\advanced\portfolio_optimizer.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### load_data()

Method description here.


#### _calculate_expected_returns()

Method description here.


#### _calculate_covariance_matrix()

Method description here.


#### _calculate_factor_loadings()

Method description here.


#### optimize_portfolio()

Method description here.


#### _optimize_mean_variance()

Method description here.


#### _optimize_min_variance()

Method description here.


#### _optimize_max_sharpe()

Method description here.


#### _optimize_risk_parity()

Method description here.


#### _optimize_equal_risk_contribution()

Method description here.


#### _optimize_black_litterman()

Method description here.


#### _optimize_max_diversification()

Method description here.


#### _optimize_min_correlation()

Method description here.


#### _optimize_hierarchical()

Method description here.


#### _optimize_regime_aware()

Method description here.


#### _calculate_risk_contributions()

Method description here.


#### _calculate_portfolio_drawdown()

Method description here.


#### _calculate_diversification_ratio()

Method description here.


#### calculate_portfolio_metrics()

Method description here.


#### rebalance_portfolio()

Method description here.



### Attributes



---

## src.bot.advanced.regime_detector.MarketRegime

Market regime types

**File:** `src/bot\advanced\regime_detector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BULL_MARKET`

- `BEAR_MARKET`

- `SIDEWAYS`

- `HIGH_VOLATILITY`

- `CRASH`

- `BUBBLE`

- `RECOVERY`

- `DISTRIBUTION`


---

## src.bot.advanced.regime_detector.RegimeClassification

Regime classification result

**File:** `src/bot\advanced\regime_detector.py`



### Methods



### Attributes



---

## src.bot.advanced.regime_detector.RegimeFeatures

Market regime features

**File:** `src/bot\advanced\regime_detector.py`



### Methods



### Attributes



---

## src.bot.advanced.regime_detector.RegimeDetector

Advanced Market Regime Detection System

This class provides comprehensive market regime detection using multiple
algorithms and sophisticated statistical methods.

**File:** `src/bot\advanced\regime_detector.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _initialize_thresholds()

Method description here.


#### calculate_regime_features()

Method description here.


#### classify_regime_rule_based()

Method description here.


#### classify_regime_ml()

Method description here.


#### detect_regime()

Method description here.


#### should_trade_in_regime()

Method description here.


#### get_regime_statistics()

Method description here.


#### reset_history()

Method description here.



### Attributes



---

## src.bot.analytics.advanced_analytics.AnalyticsMode

Analytics computation modes

**File:** `src/bot\analytics\advanced_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `REAL_TIME`

- `BATCH`

- `STREAMING`

- `HISTORICAL`


---

## src.bot.analytics.advanced_analytics.MarketRegime

Market regime classifications

**File:** `src/bot\analytics\advanced_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRENDING_UP`

- `TRENDING_DOWN`

- `SIDEWAYS`

- `HIGH_VOLATILITY`

- `LOW_VOLATILITY`

- `CRISIS`

- `RECOVERY`


---

## src.bot.analytics.advanced_analytics.PerformanceMetric

Performance metrics for analysis

**File:** `src/bot\analytics\advanced_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SHARPE_RATIO`

- `SORTINO_RATIO`

- `CALMAR_RATIO`

- `MAX_DRAWDOWN`

- `VOLATILITY`

- `BETA`

- `ALPHA`

- `WIN_RATE`

- `PROFIT_FACTOR`

- `EXPECTED_VALUE`


---

## src.bot.analytics.advanced_analytics.AnalyticsConfig

Configuration for analytics engine

**File:** `src/bot\analytics\advanced_analytics.py`



### Methods



### Attributes



---

## src.bot.analytics.advanced_analytics.PerformanceSnapshot

Performance snapshot at a point in time

**File:** `src/bot\analytics\advanced_analytics.py`



### Methods



### Attributes



---

## src.bot.analytics.advanced_analytics.AnalyticsInsight

Analytics insight or alert

**File:** `src/bot\analytics\advanced_analytics.py`



### Methods



### Attributes



---

## src.bot.analytics.advanced_analytics.AdvancedAnalyticsEngine

Advanced analytics engine for comprehensive performance analysis

Features:
- Real-time performance tracking ✅
- Market regime detection ✅
- Multi-dimensional performance analysis ✅
- Risk analytics and stress testing ✅
- Predictive performance modeling ✅
- Strategy attribution analysis ✅

**File:** `src/bot\analytics\advanced_analytics.py`



### Methods


#### __init__()

Method description here.


#### _calculate_volatility()

Method description here.


#### _calculate_sharpe_ratio()

Method description here.


#### _calculate_max_drawdown()

Method description here.


#### _calculate_var()

Method description here.


#### _calculate_expected_shortfall()

Method description here.


#### _calculate_market_correlation()

Method description here.


#### _calculate_beta()

Method description here.


#### get_analytics_summary()

Method description here.



### Attributes



---

## src.bot.analytics.advanced_analytics.MarketRegimeDetector

Market regime detection system

**File:** `src/bot\analytics\advanced_analytics.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.analytics.factor_analyzer.FactorType

Types of factors in multi-factor models.

**File:** `src/bot\analytics\factor_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MARKET`

- `SIZE`

- `VALUE`

- `MOMENTUM`

- `QUALITY`

- `VOLATILITY`

- `PROFITABILITY`

- `INVESTMENT`

- `LIQUIDITY`

- `SENTIMENT`

- `MACRO`

- `SECTOR`

- `CUSTOM`


---

## src.bot.analytics.factor_analyzer.FactorLoadings

Factor loadings and statistics.

**File:** `src/bot\analytics\factor_analyzer.py`



### Methods



### Attributes



---

## src.bot.analytics.factor_analyzer.FactorModel

Complete factor model results.

**File:** `src/bot\analytics\factor_analyzer.py`



### Methods



### Attributes



---

## src.bot.analytics.factor_analyzer.FactorDecomposition

Risk and return decomposition by factors.

**File:** `src/bot\analytics\factor_analyzer.py`



### Methods



### Attributes



---

## src.bot.analytics.factor_analyzer.FactorAnalyzer

Advanced multi-factor analysis and decomposition.

**File:** `src/bot\analytics\factor_analyzer.py`



### Methods


#### __init__()

Method description here.


#### _validate_factor_inputs()

Method description here.


#### _align_factor_data()

Method description here.


#### _infer_factor_type()

Method description here.


#### get_model_summary()

Method description here.



### Attributes



---

## src.bot.analytics.performance_attribution.AttributionMethod

Performance attribution methods.

**File:** `src/bot\analytics\performance_attribution.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BRINSON`

- `BARRA`

- `FAMA_FRENCH`

- `FACTOR_MODEL`

- `CUSTOM`


---

## src.bot.analytics.performance_attribution.FactorContribution

Individual factor contribution to performance.

**File:** `src/bot\analytics\performance_attribution.py`



### Methods



### Attributes



---

## src.bot.analytics.performance_attribution.AttributionResult

Complete performance attribution result.

**File:** `src/bot\analytics\performance_attribution.py`



### Methods



### Attributes



---

## src.bot.analytics.performance_attribution.PerformanceAttributor

Advanced performance attribution and analysis engine.

**File:** `src/bot\analytics\performance_attribution.py`



### Methods


#### __init__()

Method description here.


#### _validate_attribution_inputs()

Method description here.


#### _align_data()

Method description here.


#### _store_attribution_result()

Method description here.


#### get_attribution_summary()

Method description here.



### Attributes



---

## src.bot.analytics.regime_detector.MarketRegime

Market regime types.

**File:** `src/bot\analytics\regime_detector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BULL_MARKET`

- `BEAR_MARKET`

- `SIDEWAYS`

- `HIGH_VOLATILITY`

- `LOW_VOLATILITY`

- `CRISIS`

- `RECOVERY`

- `MOMENTUM`

- `MEAN_REVERSION`


---

## src.bot.analytics.regime_detector.RegimeTransition

Market regime transition information.

**File:** `src/bot\analytics\regime_detector.py`



### Methods



### Attributes



---

## src.bot.analytics.regime_detector.RegimeAnalysis

Complete regime analysis result.

**File:** `src/bot\analytics\regime_detector.py`



### Methods



### Attributes



---

## src.bot.analytics.regime_detector.RegimeDetector

Advanced market regime detection and analysis.

**File:** `src/bot\analytics\regime_detector.py`



### Methods


#### __init__()

Method description here.


#### _calculate_regime_duration()

Method description here.


#### _estimate_regime_change_date()

Method description here.


#### _update_regime_history()

Method description here.


#### get_regime_summary()

Method description here.



### Attributes



---

## src.bot.analytics.risk_analytics.VaRMethod

Value at Risk calculation methods.

**File:** `src/bot\analytics\risk_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HISTORICAL`

- `PARAMETRIC`

- `MONTE_CARLO`

- `FILTERED_HISTORICAL`

- `EXTREME_VALUE`


---

## src.bot.analytics.risk_analytics.StressTestType

Types of stress tests.

**File:** `src/bot\analytics\risk_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HISTORICAL_SCENARIO`

- `MONTE_CARLO_SCENARIO`

- `SENSITIVITY_ANALYSIS`

- `WORST_CASE`

- `TAIL_RISK`


---

## src.bot.analytics.risk_analytics.RiskMetrics

Comprehensive risk metrics.

**File:** `src/bot\analytics\risk_analytics.py`



### Methods



### Attributes



---

## src.bot.analytics.risk_analytics.VaRResult

Value at Risk calculation result.

**File:** `src/bot\analytics\risk_analytics.py`



### Methods



### Attributes



---

## src.bot.analytics.risk_analytics.StressTestResult

Stress test analysis result.

**File:** `src/bot\analytics\risk_analytics.py`



### Methods



### Attributes



---

## src.bot.analytics.risk_analytics.VaRCalculator

Advanced Value at Risk calculator with multiple methodologies.

**File:** `src/bot\analytics\risk_analytics.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.analytics.risk_analytics.StressTester

Advanced stress testing and scenario analysis.

**File:** `src/bot\analytics\risk_analytics.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.analytics.risk_analytics.RiskAnalyzer

Comprehensive risk analysis engine.

**File:** `src/bot\analytics\risk_analytics.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.api.config.LogLevel

Logging levels

**File:** `src/bot\api\config.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DEBUG`

- `INFO`

- `WARNING`

- `ERROR`

- `CRITICAL`


---

## src.bot.api.config.AustralianTaxYear

Australian tax year periods

**File:** `src/bot\api\config.py`


**Inherits from:** Enum


### Methods



### Attributes


- `FY2023`

- `FY2024`

- `FY2025`


---

## src.bot.api.config.RateLimitConfig

Rate limiting configuration

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.api.config.ConnectionConfig

HTTP connection configuration

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.api.config.WebSocketConfig

WebSocket configuration

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.api.config.CacheConfig

Data caching configuration

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.api.config.AustralianComplianceConfig

Australian compliance and tax configuration

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.api.config.LoggingConfig

Logging configuration

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.api.config.SecurityConfig

Security configuration

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.api.config.PerformanceConfig

Performance optimization configuration

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.api.config.UnifiedAPIConfig

Unified API Configuration

Centralized configuration for all API operations

**File:** `src/bot\api\config.py`



### Methods


#### __post_init__()

Method description here.


#### _validate_config()

Method description here.


#### from_environment()

Method description here.


#### from_file()

Method description here.


#### from_dict()

Method description here.


#### to_dict()

Method description here.


#### save_to_file()

Method description here.


#### update_credentials()

Method description here.


#### validate_credentials()

Method description here.


#### get_summary()

Method description here.



### Attributes



---

## src.bot.api.config.ConfigurationManager

Configuration manager with hot-reloading support

**File:** `src/bot\api\config.py`



### Methods


#### __init__()

Method description here.


#### load_from_file()

Method description here.


#### reload_if_changed()

Method description here.


#### add_reload_callback()

Method description here.


#### get_config()

Method description here.



### Attributes



---

## src.bot.api.market_data_pipeline.DataSource

Data source types

**File:** `src/bot\api\market_data_pipeline.py`


**Inherits from:** Enum


### Methods



### Attributes


- `REST_API`

- `WEBSOCKET`

- `CACHE`

- `HISTORICAL`


---

## src.bot.api.market_data_pipeline.DataQuality

Data quality levels

**File:** `src/bot\api\market_data_pipeline.py`


**Inherits from:** Enum


### Methods



### Attributes


- `EXCELLENT`

- `GOOD`

- `FAIR`

- `POOR`

- `INVALID`


---

## src.bot.api.market_data_pipeline.CacheStrategy

Cache strategies

**File:** `src/bot\api\market_data_pipeline.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TTL`

- `LRU`

- `HYBRID`


---

## src.bot.api.market_data_pipeline.DataPoint

Standardized data point

**File:** `src/bot\api\market_data_pipeline.py`



### Methods



### Attributes



---

## src.bot.api.market_data_pipeline.PipelineConfig

Market data pipeline configuration

**File:** `src/bot\api\market_data_pipeline.py`



### Methods



### Attributes



---

## src.bot.api.market_data_pipeline.DataMetrics

Data pipeline metrics

**File:** `src/bot\api\market_data_pipeline.py`



### Methods



### Attributes



---

## src.bot.api.market_data_pipeline.UnifiedDataCache

High-performance data caching system

**File:** `src/bot\api\market_data_pipeline.py`



### Methods


#### __init__()

Method description here.


#### get()

Method description here.


#### set()

Method description here.


#### get_ticker()

Method description here.


#### set_ticker()

Method description here.


#### get_orderbook()

Method description here.


#### set_orderbook()

Method description here.


#### get_recent_trades()

Method description here.


#### add_trade()

Method description here.


#### cleanup()

Method description here.


#### get_stats()

Method description here.



### Attributes



---

## src.bot.api.market_data_pipeline.DataValidator

Data quality validation and monitoring

**File:** `src/bot\api\market_data_pipeline.py`



### Methods


#### __init__()

Method description here.


#### validate_market_data()

Method description here.


#### validate_orderbook()

Method description here.


#### get_symbol_quality_score()

Method description here.


#### get_overall_quality_score()

Method description here.



### Attributes



---

## src.bot.api.market_data_pipeline.UnifiedMarketDataPipeline

Unified Market Data Pipeline

Orchestrates all market data operations with caching, validation, and monitoring

**File:** `src/bot\api\market_data_pipeline.py`



### Methods


#### __init__()

Method description here.


#### add_callback()

Method description here.


#### remove_callback()

Method description here.


#### get_status()

Method description here.



### Attributes



---

## src.bot.api.unified_bybit_client.Environment

Trading environment types

**File:** `src/bot\api\unified_bybit_client.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TESTNET`

- `MAINNET`


---

## src.bot.api.unified_bybit_client.ConnectionStatus

Connection status types

**File:** `src/bot\api\unified_bybit_client.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DISCONNECTED`

- `CONNECTING`

- `CONNECTED`

- `RECONNECTING`

- `ERROR`


---

## src.bot.api.unified_bybit_client.OrderType

Order types

**File:** `src/bot\api\unified_bybit_client.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MARKET`

- `LIMIT`

- `STOP`

- `STOP_LIMIT`


---

## src.bot.api.unified_bybit_client.OrderSide

Order sides

**File:** `src/bot\api\unified_bybit_client.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BUY`

- `SELL`


---

## src.bot.api.unified_bybit_client.TimeInForce

Time in force options

**File:** `src/bot\api\unified_bybit_client.py`


**Inherits from:** Enum


### Methods



### Attributes


- `GTC`

- `IOC`

- `FOK`


---

## src.bot.api.unified_bybit_client.BybitCredentials

Bybit API credentials with validation

**File:** `src/bot\api\unified_bybit_client.py`



### Methods


#### __post_init__()

Method description here.


#### base_url()

Method description here.


#### ws_url()

Method description here.


#### ws_private_url()

Method description here.



### Attributes



---

## src.bot.api.unified_bybit_client.MarketData

Standardized market data structure

**File:** `src/bot\api\unified_bybit_client.py`



### Methods



### Attributes



---

## src.bot.api.unified_bybit_client.OrderBookData

Order book data structure

**File:** `src/bot\api\unified_bybit_client.py`



### Methods



### Attributes



---

## src.bot.api.unified_bybit_client.TradeData

Trade data structure

**File:** `src/bot\api\unified_bybit_client.py`



### Methods



### Attributes



---

## src.bot.api.unified_bybit_client.AccountBalance

Account balance information

**File:** `src/bot\api\unified_bybit_client.py`



### Methods



### Attributes



---

## src.bot.api.unified_bybit_client.Position

Position information

**File:** `src/bot\api\unified_bybit_client.py`



### Methods



### Attributes



---

## src.bot.api.unified_bybit_client.OrderResponse

Order response data

**File:** `src/bot\api\unified_bybit_client.py`



### Methods



### Attributes



---

## src.bot.api.unified_bybit_client.BybitRateLimiter

Rate limiter for Bybit API compliance

**File:** `src/bot\api\unified_bybit_client.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.api.unified_bybit_client.UnifiedBybitClient

Unified Bybit API Client

Consolidates all Bybit API functionality into a single, production-ready client

**File:** `src/bot\api\unified_bybit_client.py`



### Methods


#### __init__()

Method description here.


#### _load_from_unified_config()

Method description here.


#### reload_configuration()

Method description here.


#### _generate_signature()

Method description here.


#### add_market_data_callback()

Method description here.


#### add_trade_callback()

Method description here.


#### add_order_callback()

Method description here.


#### get_connection_status()

Method description here.



### Attributes



---

## src.bot.api.websocket_manager.StreamType

WebSocket stream types

**File:** `src/bot\api\websocket_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PUBLIC`

- `PRIVATE`

- `BOTH`


---

## src.bot.api.websocket_manager.SubscriptionStatus

Subscription status

**File:** `src/bot\api\websocket_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `ACTIVE`

- `FAILED`

- `CANCELLED`


---

## src.bot.api.websocket_manager.WebSocketMetrics

WebSocket connection metrics

**File:** `src/bot\api\websocket_manager.py`



### Methods



### Attributes



---

## src.bot.api.websocket_manager.Subscription

WebSocket subscription information

**File:** `src/bot\api\websocket_manager.py`



### Methods



### Attributes



---

## src.bot.api.websocket_manager.ConnectionConfig

WebSocket connection configuration

**File:** `src/bot\api\websocket_manager.py`



### Methods



### Attributes



---

## src.bot.api.websocket_manager.UnifiedWebSocketManager

Unified WebSocket Manager for Bybit

Manages all WebSocket connections, subscriptions, and real-time data streams

**File:** `src/bot\api\websocket_manager.py`



### Methods


#### __init__()

Method description here.


#### add_event_callback()

Method description here.


#### add_message_handler()

Method description here.


#### get_status()

Method description here.


#### get_subscriptions()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.arbitrage_detector.ArbitrageType

Types of arbitrage opportunities

**File:** `src/bot\arbitrage_engine\arbitrage_detector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SIMPLE_ARBITRAGE`

- `TRIANGULAR_ARBITRAGE`

- `FUNDING_ARBITRAGE`

- `CROSS_EXCHANGE`


---

## src.bot.arbitrage_engine.arbitrage_detector.OpportunityTier

Opportunity tiers based on balance and risk

**File:** `src/bot\arbitrage_engine\arbitrage_detector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MICRO`

- `SMALL`

- `MEDIUM`

- `LARGE`


---

## src.bot.arbitrage_engine.arbitrage_detector.ArbitrageOpportunity

Detected arbitrage opportunity

**File:** `src/bot\arbitrage_engine\arbitrage_detector.py`



### Methods


#### is_profitable()

Method description here.


#### is_valid()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.arbitrage_detector.BalanceTier

Balance-based tier configuration

**File:** `src/bot\arbitrage_engine\arbitrage_detector.py`



### Methods



### Attributes



---

## src.bot.arbitrage_engine.arbitrage_detector.AustralianArbitrageDetector

Australian-focused arbitrage detection
Considers local regulations, banking, and tax implications

**File:** `src/bot\arbitrage_engine\arbitrage_detector.py`



### Methods


#### __init__()

Method description here.


#### _initialize_tier_configs()

Method description here.


#### _initialize_exchange_fees()

Method description here.


#### determine_balance_tier()

Method description here.


#### calculate_trading_costs()

Method description here.


#### assess_liquidity_and_risk()

Method description here.


#### detect_simple_arbitrage()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.arbitrage_detector.FundingArbitrageDetector

Funding rate arbitrage detector
Spot vs perpetual funding rate opportunities

**File:** `src/bot\arbitrage_engine\arbitrage_detector.py`



### Methods


#### __init__()

Method description here.


#### detect_funding_opportunities()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.arbitrage_detector.TriangularArbitrageDetector

Triangular arbitrage detector
Three-currency arbitrage loops (e.g., BTC -> ETH -> AUD -> BTC)

**File:** `src/bot\arbitrage_engine\arbitrage_detector.py`



### Methods


#### __init__()

Method description here.


#### detect_triangular_opportunities()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.arbitrage_detector.OpportunisticArbitrageEngine

Main arbitrage engine coordinating all detection methods
Designed for Australian traders with balance-based activation

**File:** `src/bot\arbitrage_engine\arbitrage_detector.py`



### Methods


#### __init__()

Method description here.


#### get_tier_appropriate_opportunities()

Method description here.


#### calculate_portfolio_allocation()

Method description here.


#### get_opportunity_summary()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.execution_engine.ExecutionStatus

Execution status tracking

**File:** `src/bot\arbitrage_engine\execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `EXECUTING`

- `COMPLETED`

- `FAILED`

- `CANCELLED`


---

## src.bot.arbitrage_engine.execution_engine.ExecutionStage

Stages of arbitrage execution

**File:** `src/bot\arbitrage_engine\execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `VALIDATION`

- `BUY_ORDER`

- `TRANSFER`

- `SELL_ORDER`

- `SETTLEMENT`


---

## src.bot.arbitrage_engine.execution_engine.ExecutionRecord

Record of arbitrage execution

**File:** `src/bot\arbitrage_engine\execution_engine.py`



### Methods



### Attributes



---

## src.bot.arbitrage_engine.execution_engine.ArbitrageExecutionValidator

Validates arbitrage opportunities before execution
Ensures compliance with Australian regulations and risk limits

**File:** `src/bot\arbitrage_engine\execution_engine.py`



### Methods


#### __init__()

Method description here.


#### _reset_daily_tracking_if_needed()

Method description here.


#### reserve_daily_volume()

Method description here.


#### release_daily_volume()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.execution_engine.OrderManager

Manages individual orders within arbitrage execution
Handles order placement, monitoring, and cancellation

**File:** `src/bot\arbitrage_engine\execution_engine.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.execution_engine.TransferManager

Manages transfers between exchanges during arbitrage execution

**File:** `src/bot\arbitrage_engine\execution_engine.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.arbitrage_engine.execution_engine.ArbitrageExecutionEngine

Main execution engine for arbitrage opportunities
Coordinates validation, orders, transfers, and compliance

**File:** `src/bot\arbitrage_engine\execution_engine.py`



### Methods


#### __init__()

Method description here.


#### get_execution_status()

Method description here.


#### get_performance_summary()

Method description here.



### Attributes



---

## src.bot.australian_compliance.ato_integration.CGTMethod

Capital Gains Tax calculation methods

**File:** `src/bot\australian_compliance\ato_integration.py`


**Inherits from:** Enum


### Methods



### Attributes


- `FIFO`

- `LIFO`

- `SPECIFIC_PARCEL`


---

## src.bot.australian_compliance.ato_integration.AssetType

Asset types for tax purposes

**File:** `src/bot\australian_compliance\ato_integration.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CRYPTOCURRENCY`

- `TRADITIONAL_SECURITY`

- `DERIVATIVE`

- `FOREIGN_CURRENCY`


---

## src.bot.australian_compliance.ato_integration.Trade

Trade record for tax calculations

**File:** `src/bot\australian_compliance\ato_integration.py`



### Methods


#### value_aud()

Method description here.


#### net_value_aud()

Method description here.



### Attributes



---

## src.bot.australian_compliance.ato_integration.CGTEvent

Capital Gains Tax event record

**File:** `src/bot\australian_compliance\ato_integration.py`



### Methods


#### is_long_term()

Method description here.


#### discount_eligible_gain()

Method description here.



### Attributes



---

## src.bot.australian_compliance.ato_integration.AustralianTaxCalculator

Australian Tax Office compliant tax calculator for cryptocurrency trading

Implements:
- FIFO cost base calculation (ATO requirement)
- CGT discount for assets held >12 months
- Record keeping for 5 years
- ATO-compliant reporting

**File:** `src/bot\australian_compliance\ato_integration.py`



### Methods


#### __init__()

Method description here.


#### _get_financial_year_start()

Method description here.


#### _get_financial_year_end()

Method description here.


#### add_trade()

Method description here.


#### _process_sale()

Method description here.


#### calculate_cgt_events()

Method description here.


#### generate_ato_report()

Method description here.


#### _generate_income_statement()

Method description here.


#### _generate_cgt_schedule()

Method description here.


#### _generate_foreign_income_report()

Method description here.


#### _generate_trading_expenses()

Method description here.


#### _generate_record_keeping_summary()

Method description here.


#### _check_compliance_status()

Method description here.


#### export_for_accountant()

Method description here.


#### validate_records()

Method description here.



### Attributes



---

## src.bot.australian_compliance.banking_manager.AustralianBank

Major Australian banks

**File:** `src/bot\australian_compliance\banking_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `COMMONWEALTH`

- `NAB`

- `ANZ`

- `WESTPAC`

- `MACQUARIE`

- `BENDIGO`

- `ING`


---

## src.bot.australian_compliance.banking_manager.TransferType

Types of bank transfers

**File:** `src/bot\australian_compliance\banking_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DOMESTIC`

- `INTERNATIONAL`

- `SWIFT`

- `RTGS`

- `OSKO`


---

## src.bot.australian_compliance.banking_manager.BankAccount

Australian bank account details

**File:** `src/bot\australian_compliance\banking_manager.py`



### Methods



### Attributes



---

## src.bot.australian_compliance.banking_manager.BankTransferCost

Calculate costs for bank transfers

**File:** `src/bot\australian_compliance\banking_manager.py`



### Methods


#### __init__()

Method description here.


#### _get_base_fee()

Method description here.


#### _get_fx_spread()

Method description here.


#### _get_processing_days()

Method description here.


#### calculate_total_cost()

Method description here.



### Attributes



---

## src.bot.australian_compliance.banking_manager.AustralianBankingManager

Manages Australian banking relationships and transfer optimization

**File:** `src/bot\australian_compliance\banking_manager.py`



### Methods


#### __init__()

Method description here.


#### add_bank_account()

Method description here.


#### get_current_fx_rate()

Method description here.


#### calculate_aud_conversion_costs()

Method description here.


#### optimize_transfer_route()

Method description here.


#### check_transfer_limits()

Method description here.


#### _get_limit_recommendation()

Method description here.


#### manage_banking_risks()

Method description here.


#### generate_banking_report()

Method description here.


#### schedule_transfer()

Method description here.



### Attributes



---

## src.bot.australian_compliance.regulatory_compliance.ComplianceFramework

Australian compliance frameworks

**File:** `src/bot\australian_compliance\regulatory_compliance.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ASIC`

- `AUSTRAC`

- `ATO`

- `ACCC`


---

## src.bot.australian_compliance.regulatory_compliance.LicenseType

Financial license types

**File:** `src/bot\australian_compliance\regulatory_compliance.py`


**Inherits from:** Enum


### Methods



### Attributes


- `AFSL`

- `ACL`

- `AUSTRAC_DCE`


---

## src.bot.australian_compliance.regulatory_compliance.ReportingObligation

Reporting obligations

**File:** `src/bot\australian_compliance\regulatory_compliance.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SUSPICIOUS_MATTER`

- `THRESHOLD_TRANSACTION`

- `INTERNATIONAL_FUNDS`

- `LARGE_CASH`


---

## src.bot.australian_compliance.regulatory_compliance.ComplianceEvent

Compliance event record

**File:** `src/bot\australian_compliance\regulatory_compliance.py`



### Methods



### Attributes



---

## src.bot.australian_compliance.regulatory_compliance.BusinessStructure

Business structure for compliance purposes

**File:** `src/bot\australian_compliance\regulatory_compliance.py`



### Methods



### Attributes



---

## src.bot.australian_compliance.regulatory_compliance.AustralianComplianceManager

Manages Australian regulatory compliance for cryptocurrency trading

Covers:
- AFSL licensing requirements
- AUSTRAC reporting obligations
- GST calculations and reporting
- AML/CTF compliance
- Consumer protection requirements

**File:** `src/bot\australian_compliance\regulatory_compliance.py`



### Methods


#### __init__()

Method description here.


#### _assess_licensing_requirements()

Method description here.


#### check_licensing_requirements()

Method description here.


#### assess_transaction_reporting()

Method description here.


#### _assess_suspicious_indicators()

Method description here.


#### calculate_gst_obligations()

Method description here.


#### _calculate_next_bas_due()

Method description here.


#### generate_compliance_reports()

Method description here.


#### _generate_asic_report()

Method description here.


#### _generate_austrac_report()

Method description here.


#### _generate_ato_report()

Method description here.


#### _generate_consumer_protection_report()

Method description here.


#### monitor_regulatory_changes()

Method description here.


#### generate_compliance_calendar()

Method description here.



### Attributes



---

## src.bot.backtesting.backtest_engine.BacktestTrade

Individual backtest trade record.

**File:** `src/bot\backtesting\backtest_engine.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.backtest_engine.BacktestResults

Comprehensive backtest results.

**File:** `src/bot\backtesting\backtest_engine.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.backtest_engine.BacktestEngine

Comprehensive backtesting engine for strategy validation.

Features:
- Historical data simulation with realistic execution
- Commission and slippage modeling
- Risk management integration
- Portfolio-level backtesting
- Performance analytics and reporting
- Walk-forward analysis support

**File:** `src/bot\backtesting\backtest_engine.py`



### Methods


#### __init__()

Method description here.


#### _apply_slippage()

Method description here.


#### _calculate_sharpe_ratio()

Method description here.


#### _calculate_sortino_ratio()

Method description here.


#### _reset_state()

Method description here.


#### generate_report()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_enhanced_backtest_engine.BybitVIPTier

Bybit VIP tier levels affecting fee rates.

**File:** `src/bot\backtesting\bybit_enhanced_backtest_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `NO_VIP`

- `VIP1`

- `VIP2`

- `VIP3`

- `PRO1`

- `PRO2`

- `PRO3`


---

## src.bot.backtesting.bybit_enhanced_backtest_engine.BybitContractType

Bybit contract types with different fee structures.

**File:** `src/bot\backtesting\bybit_enhanced_backtest_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LINEAR_PERPETUAL`

- `INVERSE_PERPETUAL`

- `LINEAR_FUTURES`

- `INVERSE_FUTURES`

- `SPOT`


---

## src.bot.backtesting.bybit_enhanced_backtest_engine.BybitFeeStructure

Bybit fee structure for different VIP tiers and contract types.

**File:** `src/bot\backtesting\bybit_enhanced_backtest_engine.py`



### Methods



### Attributes


- `LINEAR_PERP_FEES`

- `SPOT_FEES`


---

## src.bot.backtesting.bybit_enhanced_backtest_engine.BybitMarginRequirements

Bybit margin requirements for different symbols and tiers.

**File:** `src/bot\backtesting\bybit_enhanced_backtest_engine.py`



### Methods



### Attributes


- `INITIAL_MARGIN_TIERS`


---

## src.bot.backtesting.bybit_enhanced_backtest_engine.BybitTrade

Enhanced trade record with Bybit-specific data.

**File:** `src/bot\backtesting\bybit_enhanced_backtest_engine.py`


**Inherits from:** BacktestTrade


### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_enhanced_backtest_engine.BybitBacktestResults

Enhanced backtest results with Bybit-specific metrics.

**File:** `src/bot\backtesting\bybit_enhanced_backtest_engine.py`


**Inherits from:** BacktestResults


### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_enhanced_backtest_engine.BybitEnhancedBacktestEngine

Enhanced backtesting engine specifically designed for Bybit trading.

This engine provides:
1. Bybit-specific fee calculation with VIP tier support
2. Funding rate impact on perpetual positions
3. Liquidation risk modeling and margin requirements
4. Enhanced market impact and execution simulation
5. Historical data integration with funding rates
6. Realistic order execution with latency modeling

**File:** `src/bot\backtesting\bybit_enhanced_backtest_engine.py`


**Inherits from:** BacktestEngine


### Methods


#### __init__()

Method description here.


#### _initialize_delay_model()

Method description here.


#### _calculate_bybit_commission()

Method description here.


#### _calculate_liquidation_price()

Method description here.


#### _get_max_leverage()

Method description here.


#### _calculate_volatility()

Method description here.


#### _simulate_maker_probability()

Method description here.


#### _simulate_execution_delay()

Method description here.


#### _reset_enhanced_state()

Method description here.


#### _calculate_vip_savings()

Method description here.


#### generate_enhanced_report()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_execution_simulator.OrderType

Order types supported by Bybit.

**File:** `src/bot\backtesting\bybit_execution_simulator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MARKET`

- `LIMIT`

- `STOP_MARKET`

- `STOP_LIMIT`

- `TAKE_PROFIT_MARKET`

- `TAKE_PROFIT_LIMIT`

- `CONDITIONAL`


---

## src.bot.backtesting.bybit_execution_simulator.ExecutionStrategy

Advanced execution strategies.

**File:** `src/bot\backtesting\bybit_execution_simulator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `AGGRESSIVE`

- `PASSIVE`

- `TWAP`

- `VWAP`

- `ICEBERG`

- `SNIPER`

- `ADAPTIVE`


---

## src.bot.backtesting.bybit_execution_simulator.FillType

Types of order fills.

**File:** `src/bot\backtesting\bybit_execution_simulator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `FULL`

- `PARTIAL`

- `REJECTED`


---

## src.bot.backtesting.bybit_execution_simulator.OrderBookLevel

Single level of order book.

**File:** `src/bot\backtesting\bybit_execution_simulator.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_execution_simulator.OrderBookSnapshot

Complete order book snapshot.

**File:** `src/bot\backtesting\bybit_execution_simulator.py`



### Methods


#### best_bid()

Method description here.


#### best_ask()

Method description here.


#### spread()

Method description here.


#### mid_price()

Method description here.


#### get_depth()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_execution_simulator.ExecutionResult

Result of order execution simulation.

**File:** `src/bot\backtesting\bybit_execution_simulator.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_execution_simulator.MarketConditions

Current market conditions affecting execution.

**File:** `src/bot\backtesting\bybit_execution_simulator.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_execution_simulator.BybitExecutionSimulator

Advanced trade execution simulator for Bybit trading.

This simulator provides realistic execution modeling including:
1. Order book depth simulation and market impact
2. Latency modeling with network and processing delays
3. Partial fill simulation with time-based execution
4. Execution strategy implementation (TWAP, VWAP, etc.)
5. Bybit-specific execution characteristics
6. Execution quality measurement and optimization

**File:** `src/bot\backtesting\bybit_execution_simulator.py`



### Methods


#### __init__()

Method description here.


#### _default_latency_model()

Method description here.


#### _initialize_bybit_parameters()

Method description here.


#### simulate_order_execution()

Method description here.


#### _assess_market_conditions()

Method description here.


#### _generate_order_book()

Method description here.


#### _calculate_execution_latency()

Method description here.


#### _execute_aggressive()

Method description here.


#### _execute_passive()

Method description here.


#### _execute_twap()

Method description here.


#### _execute_vwap()

Method description here.


#### _execute_iceberg()

Method description here.


#### _calculate_market_impact()

Method description here.


#### _calculate_total_market_impact()

Method description here.


#### _should_partial_fill()

Method description here.


#### _determine_maker_status()

Method description here.


#### _calculate_passive_fill_probability()

Method description here.


#### _calculate_execution_quality()

Method description here.


#### get_execution_statistics()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_fee_simulator.VIPTierRequirements

Bybit VIP tier requirements and benefits.

**File:** `src/bot\backtesting\bybit_fee_simulator.py`



### Methods



### Attributes


- `TIER_REQUIREMENTS`


---

## src.bot.backtesting.bybit_fee_simulator.FeeCalculationDetails

Detailed fee calculation breakdown.

**File:** `src/bot\backtesting\bybit_fee_simulator.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_fee_simulator.FeeAnalysisReport

Comprehensive fee analysis and optimization report.

**File:** `src/bot\backtesting\bybit_fee_simulator.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_fee_simulator.BybitFeeCalculator

Advanced Bybit fee calculation and optimization system.

This class provides:
1. Accurate fee calculation for all Bybit contract types
2. VIP tier progression simulation and benefits analysis
3. Maker/taker optimization strategies
4. Cost analysis and savings identification
5. Real-time fee optimization recommendations

**File:** `src/bot\backtesting\bybit_fee_simulator.py`



### Methods


#### __init__()

Method description here.


#### _initialize_fee_structures()

Method description here.


#### calculate_fee()

Method description here.


#### _get_base_fee_rate()

Method description here.


#### _calculate_vip_discount()

Method description here.


#### _calculate_volume_discount()

Method description here.


#### _track_trading_volume()

Method description here.


#### _update_rolling_volume()

Method description here.


#### _check_vip_tier_progression()

Method description here.


#### get_current_vip_status()

Method description here.


#### simulate_fee_savings()

Method description here.


#### generate_fee_analysis_report()

Method description here.


#### _calculate_next_tier_savings()

Method description here.


#### _generate_optimization_suggestions()

Method description here.


#### _estimate_annual_savings()

Method description here.


#### export_fee_history()

Method description here.


#### get_fee_summary()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_liquidation_risk_manager.LiquidationRiskLevel

Liquidation risk severity levels.

**File:** `src/bot\backtesting\bybit_liquidation_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SAFE`

- `LOW`

- `MEDIUM`

- `HIGH`

- `CRITICAL`

- `IMMINENT`


---

## src.bot.backtesting.bybit_liquidation_risk_manager.MarginType

Bybit margin types.

**File:** `src/bot\backtesting\bybit_liquidation_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CROSS_MARGIN`

- `ISOLATED_MARGIN`


---

## src.bot.backtesting.bybit_liquidation_risk_manager.MarginTier

Bybit margin tier configuration.

**File:** `src/bot\backtesting\bybit_liquidation_risk_manager.py`



### Methods



### Attributes



---

## src.bot.backtesting.bybit_liquidation_risk_manager.LiquidationRiskAssessment

Comprehensive liquidation risk assessment.

**File:** `src/bot\backtesting\bybit_liquidation_risk_manager.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.backtesting.bybit_liquidation_risk_manager.PortfolioLiquidationRisk

Portfolio-level liquidation risk analysis.

**File:** `src/bot\backtesting\bybit_liquidation_risk_manager.py`



### Methods



### Attributes



---

## src.bot.backtesting.bybit_liquidation_risk_manager.BybitLiquidationRiskManager

Advanced liquidation risk management system for Bybit trading.

This system provides:
1. Accurate liquidation price calculations for all contract types
2. Real-time risk monitoring and alerting
3. Portfolio-level risk assessment
4. Stress testing and scenario analysis
5. Dynamic position sizing with risk constraints
6. Liquidation avoidance strategies

**File:** `src/bot\backtesting\bybit_liquidation_risk_manager.py`



### Methods


#### __init__()

Method description here.


#### _initialize_margin_tiers()

Method description here.


#### calculate_liquidation_price()

Method description here.


#### _get_margin_tier()

Method description here.


#### assess_liquidation_risk()

Method description here.


#### _classify_risk_level()

Method description here.


#### _estimate_volatility()

Method description here.


#### _estimate_time_to_liquidation()

Method description here.


#### _generate_risk_recommendation()

Method description here.


#### _calculate_safe_position_size()

Method description here.


#### _recommend_stop_loss()

Method description here.


#### _generate_hedge_recommendation()

Method description here.


#### simulate_liquidation_cascade()

Method description here.


#### generate_risk_report()

Method description here.


#### get_safe_leverage()

Method description here.



### Attributes



---

## src.bot.cloud.alert_manager.AlertSeverity

Alert severity levels.

**File:** `src/bot\cloud\alert_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CRITICAL`

- `HIGH`

- `MEDIUM`

- `LOW`

- `INFO`


---

## src.bot.cloud.alert_manager.AlertState

Alert states.

**File:** `src/bot\cloud\alert_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `FIRING`

- `RESOLVED`

- `SILENCED`

- `SUPPRESSED`


---

## src.bot.cloud.alert_manager.NotificationChannel

Notification channels.

**File:** `src/bot\cloud\alert_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `EMAIL`

- `SLACK`

- `WEBHOOK`

- `SMS`

- `PAGERDUTY`

- `DISCORD`


---

## src.bot.cloud.alert_manager.EscalationPolicy

Escalation policies.

**File:** `src/bot\cloud\alert_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `IMMEDIATE`

- `DELAYED`

- `BUSINESS_HOURS`

- `WEEKDAYS_ONLY`


---

## src.bot.cloud.alert_manager.AlertRule

Alert rule definition.

**File:** `src/bot\cloud\alert_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.alert_manager.AlertInstance

Active alert instance.

**File:** `src/bot\cloud\alert_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.alert_manager.NotificationConfig

Notification configuration.

**File:** `src/bot\cloud\alert_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.alert_manager.Silence

Alert silence configuration.

**File:** `src/bot\cloud\alert_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.alert_manager.EscalationRule

Escalation rule configuration.

**File:** `src/bot\cloud\alert_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.alert_manager.IncidentTicket

Incident ticket for alert grouping.

**File:** `src/bot\cloud\alert_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.alert_manager.AlertManager

Comprehensive alert management system.

**File:** `src/bot\cloud\alert_manager.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_notifications()

Method description here.


#### _setup_default_escalations()

Method description here.


#### _generate_fingerprint()

Method description here.


#### _is_silenced()

Method description here.


#### _should_notify()

Method description here.


#### _find_matching_notifications()

Method description here.


#### _check_rate_limit()

Method description here.


#### _is_in_time_interval()

Method description here.


#### _get_slack_color()

Method description here.


#### _template_string()

Method description here.


#### add_alert_rule()

Method description here.


#### remove_alert_rule()

Method description here.


#### create_silence()

Method description here.


#### remove_silence()

Method description here.


#### get_alert_manager_summary()

Method description here.



### Attributes



---

## src.bot.cloud.auto_scaler.ScalingDirection

Scaling direction.

**File:** `src/bot\cloud\auto_scaler.py`


**Inherits from:** Enum


### Methods



### Attributes


- `UP`

- `DOWN`

- `NONE`


---

## src.bot.cloud.auto_scaler.MetricType

Metric types for scaling decisions.

**File:** `src/bot\cloud\auto_scaler.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CPU_UTILIZATION`

- `MEMORY_UTILIZATION`

- `REQUEST_RATE`

- `RESPONSE_TIME`

- `QUEUE_LENGTH`

- `ERROR_RATE`

- `THROUGHPUT`

- `CUSTOM`


---

## src.bot.cloud.auto_scaler.ScalingPolicy

Scaling policies.

**File:** `src/bot\cloud\auto_scaler.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TARGET_TRACKING`

- `STEP_SCALING`

- `PREDICTIVE`

- `THRESHOLD`


---

## src.bot.cloud.auto_scaler.MetricData

Metric data point.

**File:** `src/bot\cloud\auto_scaler.py`



### Methods



### Attributes



---

## src.bot.cloud.auto_scaler.ScalingRule

Scaling rule configuration.

**File:** `src/bot\cloud\auto_scaler.py`



### Methods



### Attributes



---

## src.bot.cloud.auto_scaler.ScalingTarget

Scaling target configuration.

**File:** `src/bot\cloud\auto_scaler.py`



### Methods



### Attributes



---

## src.bot.cloud.auto_scaler.ScalingEvent

Scaling event record.

**File:** `src/bot\cloud\auto_scaler.py`



### Methods



### Attributes



---

## src.bot.cloud.auto_scaler.PredictiveModel

Predictive scaling model.

**File:** `src/bot\cloud\auto_scaler.py`



### Methods



### Attributes



---

## src.bot.cloud.auto_scaler.AutoScaler

Dynamic auto-scaling system for containers and pods.

**File:** `src/bot\cloud\auto_scaler.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_scaling_rules()

Method description here.


#### set_scaling_callbacks()

Method description here.


#### _store_metric()

Method description here.


#### _is_in_cooldown()

Method description here.


#### _get_recent_metrics()

Method description here.


#### add_scaling_target()

Method description here.


#### remove_scaling_target()

Method description here.


#### add_scaling_rule()

Method description here.


#### get_scaling_status()

Method description here.


#### get_scaling_events()

Method description here.


#### get_auto_scaler_summary()

Method description here.



### Attributes



---

## src.bot.cloud.cloud_storage.StorageProvider

Supported storage providers.

**File:** `src/bot\cloud\cloud_storage.py`


**Inherits from:** Enum


### Methods



### Attributes


- `AWS_S3`

- `GOOGLE_CLOUD`

- `AZURE_BLOB`

- `LOCAL_FS`

- `MINIO`


---

## src.bot.cloud.cloud_storage.StorageClass

Storage classes for cost optimization.

**File:** `src/bot\cloud\cloud_storage.py`


**Inherits from:** Enum


### Methods



### Attributes


- `STANDARD`

- `REDUCED_REDUNDANCY`

- `COLD`

- `ARCHIVE`

- `DEEP_ARCHIVE`


---

## src.bot.cloud.cloud_storage.AccessLevel

Storage access levels.

**File:** `src/bot\cloud\cloud_storage.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PRIVATE`

- `PUBLIC_READ`

- `PUBLIC_READ_WRITE`

- `AUTHENTICATED_READ`


---

## src.bot.cloud.cloud_storage.StorageConfig

Storage configuration.

**File:** `src/bot\cloud\cloud_storage.py`



### Methods



### Attributes



---

## src.bot.cloud.cloud_storage.StorageObject

Storage object metadata.

**File:** `src/bot\cloud\cloud_storage.py`



### Methods



### Attributes



---

## src.bot.cloud.cloud_storage.UploadProgress

Upload progress tracking.

**File:** `src/bot\cloud\cloud_storage.py`



### Methods



### Attributes



---

## src.bot.cloud.cloud_storage.SyncResult

Synchronization result.

**File:** `src/bot\cloud\cloud_storage.py`



### Methods



### Attributes



---

## src.bot.cloud.cloud_storage.CloudStorage

Comprehensive multi-cloud storage manager.

**File:** `src/bot\cloud\cloud_storage.py`



### Methods


#### __init__()

Method description here.


#### _initialize_default_storage()

Method description here.


#### _matches_pattern()

Method description here.


#### _map_storage_class_s3()

Method description here.


#### _map_storage_class_gcs()

Method description here.


#### get_upload_progress()

Method description here.


#### get_download_progress()

Method description here.


#### get_storage_summary()

Method description here.



### Attributes



---

## src.bot.cloud.container_orchestrator.ContainerState

Container states.

**File:** `src/bot\cloud\container_orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CREATED`

- `RUNNING`

- `PAUSED`

- `RESTARTING`

- `REMOVING`

- `EXITED`

- `DEAD`


---

## src.bot.cloud.container_orchestrator.RestartPolicy

Container restart policies.

**File:** `src/bot\cloud\container_orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `NO`

- `ON_FAILURE`

- `ALWAYS`

- `UNLESS_STOPPED`


---

## src.bot.cloud.container_orchestrator.NetworkMode

Container network modes.

**File:** `src/bot\cloud\container_orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BRIDGE`

- `HOST`

- `NONE`

- `CONTAINER`

- `CUSTOM`


---

## src.bot.cloud.container_orchestrator.ContainerConfig

Container configuration.

**File:** `src/bot\cloud\container_orchestrator.py`



### Methods



### Attributes



---

## src.bot.cloud.container_orchestrator.ContainerInfo

Container information and status.

**File:** `src/bot\cloud\container_orchestrator.py`



### Methods



### Attributes



---

## src.bot.cloud.container_orchestrator.ContainerMetrics

Container performance metrics.

**File:** `src/bot\cloud\container_orchestrator.py`



### Methods



### Attributes



---

## src.bot.cloud.container_orchestrator.ContainerEvent

Container event for monitoring.

**File:** `src/bot\cloud\container_orchestrator.py`



### Methods



### Attributes



---

## src.bot.cloud.container_orchestrator.ContainerOrchestrator

Container lifecycle management and orchestration.

**File:** `src/bot\cloud\container_orchestrator.py`



### Methods


#### __init__()

Method description here.


#### start_monitoring()

Method description here.


#### stop_monitoring()

Method description here.


#### add_event_handler()

Method description here.


#### _monitor_events()

Method description here.


#### _update_container_info()

Method description here.


#### _find_container_by_name()

Method description here.


#### _parse_memory_limit()

Method description here.


#### get_orchestration_summary()

Method description here.



### Attributes



---

## src.bot.cloud.deployment_manager.DeploymentStrategy

Deployment strategies.

**File:** `src/bot\cloud\deployment_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ROLLING_UPDATE`

- `BLUE_GREEN`

- `CANARY`

- `RECREATE`

- `A_B_TESTING`


---

## src.bot.cloud.deployment_manager.DeploymentEnvironment

Deployment environments.

**File:** `src/bot\cloud\deployment_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DEVELOPMENT`

- `STAGING`

- `PRODUCTION`

- `TESTING`

- `PREVIEW`


---

## src.bot.cloud.deployment_manager.DeploymentStatus

Deployment status.

**File:** `src/bot\cloud\deployment_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `IN_PROGRESS`

- `SUCCESS`

- `FAILED`

- `ROLLED_BACK`

- `CANCELLED`


---

## src.bot.cloud.deployment_manager.PipelineStage

CI/CD pipeline stages.

**File:** `src/bot\cloud\deployment_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BUILD`

- `TEST`

- `SECURITY_SCAN`

- `DEPLOY_STAGING`

- `INTEGRATION_TEST`

- `DEPLOY_PRODUCTION`

- `POST_DEPLOY_TEST`

- `ROLLBACK`


---

## src.bot.cloud.deployment_manager.DeploymentConfig

Deployment configuration.

**File:** `src/bot\cloud\deployment_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.deployment_manager.PipelineConfig

CI/CD pipeline configuration.

**File:** `src/bot\cloud\deployment_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.deployment_manager.DeploymentJob

Deployment job tracking.

**File:** `src/bot\cloud\deployment_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.deployment_manager.TestSuite

Test suite configuration.

**File:** `src/bot\cloud\deployment_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.deployment_manager.RollbackPlan

Rollback plan configuration.

**File:** `src/bot\cloud\deployment_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.deployment_manager.DeploymentManager

Comprehensive deployment management and CI/CD system.

**File:** `src/bot\cloud\deployment_manager.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_environments()

Method description here.


#### _setup_default_pipelines()

Method description here.


#### _setup_default_tests()

Method description here.


#### get_deployment_status()

Method description here.


#### _job_to_dict()

Method description here.


#### get_deployment_manager_summary()

Method description here.



### Attributes



---

## src.bot.cloud.grafana_manager.PanelType

Grafana panel types.

**File:** `src/bot\cloud\grafana_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TIMESERIES`

- `STAT`

- `GAUGE`

- `BAR_GAUGE`

- `TABLE`

- `HEATMAP`

- `PIE_CHART`

- `GRAPH`

- `SINGLESTAT`

- `TEXT`

- `LOGS`

- `NODE_GRAPH`


---

## src.bot.cloud.grafana_manager.VisualizationType

Visualization types.

**File:** `src/bot\cloud\grafana_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LINE`

- `BARS`

- `POINTS`

- `AREA`

- `STACKED`


---

## src.bot.cloud.grafana_manager.AlertState

Alert states for panels.

**File:** `src/bot\cloud\grafana_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `NO_DATA`

- `ALERTING`

- `OK`

- `PENDING`

- `UNKNOWN`


---

## src.bot.cloud.grafana_manager.PanelTarget

Panel query target.

**File:** `src/bot\cloud\grafana_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.grafana_manager.PanelThreshold

Panel threshold configuration.

**File:** `src/bot\cloud\grafana_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.grafana_manager.PanelAlert

Panel alert configuration.

**File:** `src/bot\cloud\grafana_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.grafana_manager.Panel

Grafana panel configuration.

**File:** `src/bot\cloud\grafana_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.grafana_manager.DashboardVariable

Dashboard template variable.

**File:** `src/bot\cloud\grafana_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.grafana_manager.DashboardAnnotation

Dashboard annotation.

**File:** `src/bot\cloud\grafana_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.grafana_manager.DashboardTime

Dashboard time range.

**File:** `src/bot\cloud\grafana_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.grafana_manager.GrafanaDashboardManager

Grafana dashboard management system.

**File:** `src/bot\cloud\grafana_manager.py`



### Methods


#### __init__()

Method description here.


#### create_panel()

Method description here.


#### create_trading_overview_dashboard()

Method description here.


#### create_system_performance_dashboard()

Method description here.


#### create_ml_models_dashboard()

Method description here.


#### _create_dashboard_config()

Method description here.


#### _convert_panel_to_grafana()

Method description here.


#### _convert_variable_to_grafana()

Method description here.


#### _convert_annotation_to_grafana()

Method description here.


#### _convert_alert_to_grafana()

Method description here.


#### get_dashboard_summary()

Method description here.



### Attributes



---

## src.bot.cloud.kubernetes_manager.DeploymentType

Kubernetes deployment types.

**File:** `src/bot\cloud\kubernetes_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRADING_ENGINE`

- `MARKET_DATA`

- `RISK_MANAGER`

- `ANALYTICS`

- `ML_ENGINE`

- `HFT_MODULE`

- `API_GATEWAY`

- `DATABASE`

- `REDIS_CACHE`

- `MONITORING`


---

## src.bot.cloud.kubernetes_manager.ServiceType

Kubernetes service types.

**File:** `src/bot\cloud\kubernetes_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CLUSTER_IP`

- `NODE_PORT`

- `LOAD_BALANCER`

- `EXTERNAL_NAME`


---

## src.bot.cloud.kubernetes_manager.ScalingPolicy

Auto-scaling policies.

**File:** `src/bot\cloud\kubernetes_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CPU_BASED`

- `MEMORY_BASED`

- `CUSTOM_METRICS`

- `QUEUE_LENGTH`

- `LATENCY_BASED`


---

## src.bot.cloud.kubernetes_manager.KubernetesResource

Kubernetes resource definition.

**File:** `src/bot\cloud\kubernetes_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.kubernetes_manager.PodMetrics

Pod performance metrics.

**File:** `src/bot\cloud\kubernetes_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.kubernetes_manager.ClusterInfo

Kubernetes cluster information.

**File:** `src/bot\cloud\kubernetes_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.kubernetes_manager.KubernetesManager

Kubernetes cluster management and orchestration.

**File:** `src/bot\cloud\kubernetes_manager.py`



### Methods


#### __init__()

Method description here.


#### _initialize_kubernetes_client()

Method description here.


#### _create_deployment_spec()

Method description here.


#### _create_service_spec()

Method description here.


#### generate_deployment_yaml()

Method description here.


#### get_kubernetes_summary()

Method description here.



### Attributes



---

## src.bot.cloud.load_balancer.LoadBalancingAlgorithm

Load balancing algorithms.

**File:** `src/bot\cloud\load_balancer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ROUND_ROBIN`

- `WEIGHTED_ROUND_ROBIN`

- `LEAST_CONNECTIONS`

- `WEIGHTED_LEAST_CONNECTIONS`

- `IP_HASH`

- `LEAST_RESPONSE_TIME`

- `RESOURCE_BASED`

- `RANDOM`

- `WEIGHTED_RANDOM`


---

## src.bot.cloud.load_balancer.HealthCheckType

Health check types.

**File:** `src/bot\cloud\load_balancer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HTTP`

- `HTTPS`

- `TCP`

- `UDP`

- `CUSTOM`


---

## src.bot.cloud.load_balancer.BackendStatus

Backend server status.

**File:** `src/bot\cloud\load_balancer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HEALTHY`

- `UNHEALTHY`

- `DRAINING`

- `MAINTENANCE`

- `UNKNOWN`


---

## src.bot.cloud.load_balancer.Backend

Backend server configuration.

**File:** `src/bot\cloud\load_balancer.py`



### Methods



### Attributes



---

## src.bot.cloud.load_balancer.HealthCheck

Health check configuration.

**File:** `src/bot\cloud\load_balancer.py`



### Methods



### Attributes



---

## src.bot.cloud.load_balancer.LoadBalancerPool

Load balancer pool configuration.

**File:** `src/bot\cloud\load_balancer.py`



### Methods



### Attributes



---

## src.bot.cloud.load_balancer.RequestMetrics

Request metrics.

**File:** `src/bot\cloud\load_balancer.py`



### Methods



### Attributes



---

## src.bot.cloud.load_balancer.LoadBalancerStats

Load balancer statistics.

**File:** `src/bot\cloud\load_balancer.py`



### Methods



### Attributes



---

## src.bot.cloud.load_balancer.LoadBalancer

High-performance load balancer for trading system.

**File:** `src/bot\cloud\load_balancer.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_pools()

Method description here.


#### _select_round_robin()

Method description here.


#### _select_weighted_round_robin()

Method description here.


#### _select_least_connections()

Method description here.


#### _select_weighted_least_connections()

Method description here.


#### _select_least_response_time()

Method description here.


#### _select_ip_hash()

Method description here.


#### _select_resource_based()

Method description here.


#### _select_weighted_random()

Method description here.


#### add_backend()

Method description here.


#### remove_backend()

Method description here.


#### drain_backend()

Method description here.


#### get_pool_status()

Method description here.


#### get_load_balancer_summary()

Method description here.



### Attributes



---

## src.bot.cloud.log_aggregator.LogLevel

Log levels.

**File:** `src/bot\cloud\log_aggregator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRACE`

- `DEBUG`

- `INFO`

- `WARN`

- `ERROR`

- `FATAL`


---

## src.bot.cloud.log_aggregator.LogFormat

Log formats.

**File:** `src/bot\cloud\log_aggregator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `JSON`

- `STRUCTURED`

- `PLAIN`

- `CEF`

- `GELF`


---

## src.bot.cloud.log_aggregator.LogSource

Log sources.

**File:** `src/bot\cloud\log_aggregator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `APPLICATION`

- `SYSTEM`

- `AUDIT`

- `SECURITY`

- `PERFORMANCE`

- `TRADING`

- `RISK`

- `ML`


---

## src.bot.cloud.log_aggregator.LogEntry

Structured log entry.

**File:** `src/bot\cloud\log_aggregator.py`



### Methods



### Attributes



---

## src.bot.cloud.log_aggregator.LogFilter

Log filtering criteria.

**File:** `src/bot\cloud\log_aggregator.py`



### Methods



### Attributes



---

## src.bot.cloud.log_aggregator.LogAlert

Log-based alert configuration.

**File:** `src/bot\cloud\log_aggregator.py`



### Methods



### Attributes



---

## src.bot.cloud.log_aggregator.LogMetrics

Log metrics and statistics.

**File:** `src/bot\cloud\log_aggregator.py`



### Methods



### Attributes



---

## src.bot.cloud.log_aggregator.LogParsingRule

Log parsing rule.

**File:** `src/bot\cloud\log_aggregator.py`



### Methods



### Attributes



---

## src.bot.cloud.log_aggregator.LogBuffer

Thread-safe log buffer.

**File:** `src/bot\cloud\log_aggregator.py`



### Methods


#### __init__()

Method description here.


#### add()

Method description here.


#### get_batch()

Method description here.


#### size()

Method description here.


#### clear()

Method description here.



### Attributes



---

## src.bot.cloud.log_aggregator.LogAggregator

Centralized log aggregation and processing system.

**File:** `src/bot\cloud\log_aggregator.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_parsing_rules()

Method description here.


#### _setup_default_filters()

Method description here.


#### _setup_default_alerts()

Method description here.


#### _initialize_elasticsearch()

Method description here.


#### _setup_elasticsearch_template()

Method description here.


#### _initialize_kafka()

Method description here.


#### ingest_log()

Method description here.


#### ingest_raw_log()

Method description here.


#### _parse_raw_log()

Method description here.


#### _create_log_entry_from_json()

Method description here.


#### _create_log_entry_from_fields()

Method description here.


#### _should_filter_log()

Method description here.


#### _matches_filter()

Method description here.


#### _format_log_entry()

Method description here.


#### _matches_alert_query()

Method description here.


#### _update_metrics()

Method description here.


#### _collect_metrics()

Method description here.


#### search_logs()

Method description here.


#### _matches_search_query()

Method description here.


#### get_log_aggregator_summary()

Method description here.



### Attributes



---

## src.bot.cloud.monitoring_stack.AlertSeverity

Alert severity levels.

**File:** `src/bot\cloud\monitoring_stack.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CRITICAL`

- `WARNING`

- `INFO`

- `DEBUG`


---

## src.bot.cloud.monitoring_stack.MetricType

Metric types.

**File:** `src/bot\cloud\monitoring_stack.py`


**Inherits from:** Enum


### Methods



### Attributes


- `COUNTER`

- `GAUGE`

- `HISTOGRAM`

- `SUMMARY`


---

## src.bot.cloud.monitoring_stack.MonitoringComponent

Monitoring components.

**File:** `src/bot\cloud\monitoring_stack.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PROMETHEUS`

- `GRAFANA`

- `ELASTICSEARCH`

- `ALERTMANAGER`

- `JAEGER`


---

## src.bot.cloud.monitoring_stack.MetricDefinition

Metric definition.

**File:** `src/bot\cloud\monitoring_stack.py`



### Methods



### Attributes



---

## src.bot.cloud.monitoring_stack.AlertRule

Alert rule configuration.

**File:** `src/bot\cloud\monitoring_stack.py`



### Methods



### Attributes



---

## src.bot.cloud.monitoring_stack.Dashboard

Grafana dashboard configuration.

**File:** `src/bot\cloud\monitoring_stack.py`



### Methods



### Attributes



---

## src.bot.cloud.monitoring_stack.LogEntry

Structured log entry.

**File:** `src/bot\cloud\monitoring_stack.py`



### Methods



### Attributes



---

## src.bot.cloud.monitoring_stack.TraceSpan

Distributed tracing span.

**File:** `src/bot\cloud\monitoring_stack.py`



### Methods



### Attributes



---

## src.bot.cloud.monitoring_stack.AlertInstance

Active alert instance.

**File:** `src/bot\cloud\monitoring_stack.py`



### Methods



### Attributes



---

## src.bot.cloud.monitoring_stack.MonitoringStack

Comprehensive monitoring and observability stack.

**File:** `src/bot\cloud\monitoring_stack.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_metrics()

Method description here.


#### _create_prometheus_metric()

Method description here.


#### _setup_default_alerts()

Method description here.


#### _setup_default_dashboards()

Method description here.


#### _initialize_elasticsearch()

Method description here.


#### record_metric()

Method description here.


#### start_timing()

Method description here.


#### add_log_entry()

Method description here.


#### start_trace()

Method description here.


#### finish_trace()

Method description here.


#### get_dashboard_config()

Method description here.


#### get_monitoring_summary()

Method description here.



### Attributes



---

## src.bot.cloud.secrets_manager.SecretProvider

Supported secret providers.

**File:** `src/bot\cloud\secrets_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `AWS_SECRETS_MANAGER`

- `AZURE_KEY_VAULT`

- `GOOGLE_SECRET_MANAGER`

- `HASHICORP_VAULT`

- `KUBERNETES_SECRETS`

- `LOCAL_ENCRYPTED`


---

## src.bot.cloud.secrets_manager.SecretType

Types of secrets.

**File:** `src/bot\cloud\secrets_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `API_KEY`

- `PASSWORD`

- `CERTIFICATE`

- `PRIVATE_KEY`

- `CONNECTION_STRING`

- `JWT_SECRET`

- `ENCRYPTION_KEY`

- `OAUTH_TOKEN`

- `DATABASE_CREDENTIALS`

- `WEBHOOK_SECRET`


---

## src.bot.cloud.secrets_manager.RotationStatus

Secret rotation status.

**File:** `src/bot\cloud\secrets_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `NOT_SCHEDULED`

- `SCHEDULED`

- `IN_PROGRESS`

- `COMPLETED`

- `FAILED`


---

## src.bot.cloud.secrets_manager.SecretConfig

Secret configuration.

**File:** `src/bot\cloud\secrets_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.secrets_manager.SecretValue

Secret value with metadata.

**File:** `src/bot\cloud\secrets_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.secrets_manager.SecretRotation

Secret rotation configuration.

**File:** `src/bot\cloud\secrets_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.secrets_manager.AccessAudit

Secret access audit log.

**File:** `src/bot\cloud\secrets_manager.py`



### Methods



### Attributes



---

## src.bot.cloud.secrets_manager.SecretsManager

Comprehensive secrets management system.

**File:** `src/bot\cloud\secrets_manager.py`



### Methods


#### __init__()

Method description here.


#### _initialize_default_providers()

Method description here.


#### _initialize_master_key()

Method description here.


#### _clear_secret_cache()

Method description here.


#### get_secret_list()

Method description here.


#### get_rotation_status()

Method description here.


#### get_audit_logs()

Method description here.


#### get_secrets_manager_summary()

Method description here.



### Attributes



---

## src.bot.cloud.service_mesh.ServiceMeshProtocol

Service mesh protocols.

**File:** `src/bot\cloud\service_mesh.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HTTP`

- `HTTPS`

- `GRPC`

- `TCP`

- `WEBSOCKET`


---

## src.bot.cloud.service_mesh.TrafficPolicy

Traffic management policies.

**File:** `src/bot\cloud\service_mesh.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ROUND_ROBIN`

- `LEAST_CONN`

- `RANDOM`

- `WEIGHTED`

- `CIRCUIT_BREAKER`


---

## src.bot.cloud.service_mesh.SecurityPolicy

Security policies.

**File:** `src/bot\cloud\service_mesh.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MTLS`

- `JWT`

- `OAUTH2`

- `API_KEY`

- `NONE`


---

## src.bot.cloud.service_mesh.ServiceStatus

Service status.

**File:** `src/bot\cloud\service_mesh.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HEALTHY`

- `UNHEALTHY`

- `DEGRADED`

- `MAINTENANCE`

- `UNKNOWN`


---

## src.bot.cloud.service_mesh.ServiceEndpoint

Service endpoint definition.

**File:** `src/bot\cloud\service_mesh.py`



### Methods



### Attributes



---

## src.bot.cloud.service_mesh.ServiceRoute

Service routing configuration.

**File:** `src/bot\cloud\service_mesh.py`



### Methods



### Attributes



---

## src.bot.cloud.service_mesh.CircuitBreakerConfig

Circuit breaker configuration.

**File:** `src/bot\cloud\service_mesh.py`



### Methods



### Attributes



---

## src.bot.cloud.service_mesh.RateLimitConfig

Rate limiting configuration.

**File:** `src/bot\cloud\service_mesh.py`



### Methods



### Attributes



---

## src.bot.cloud.service_mesh.SecurityConfig

Security configuration.

**File:** `src/bot\cloud\service_mesh.py`



### Methods



### Attributes



---

## src.bot.cloud.service_mesh.ServiceMetrics

Service communication metrics.

**File:** `src/bot\cloud\service_mesh.py`



### Methods



### Attributes



---

## src.bot.cloud.service_mesh.ServiceCall

Service call record for tracing.

**File:** `src/bot\cloud\service_mesh.py`



### Methods



### Attributes



---

## src.bot.cloud.service_mesh.ServiceMesh

Service mesh for microservices communication.

**File:** `src/bot\cloud\service_mesh.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_services()

Method description here.


#### _setup_default_routes()

Method description here.


#### _setup_default_security()

Method description here.


#### _find_route()

Method description here.


#### _is_circuit_breaker_open()

Method description here.


#### _record_circuit_breaker_failure()

Method description here.


#### _record_circuit_breaker_success()

Method description here.


#### _check_rate_limit()

Method description here.


#### _apply_security_headers()

Method description here.


#### get_service_status()

Method description here.


#### get_service_mesh_summary()

Method description here.



### Attributes



---

## src.bot.config.production.Environment

Deployment environments.

**File:** `src/bot\config\production.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DEVELOPMENT`

- `TESTING`

- `STAGING`

- `PRODUCTION`


---

## src.bot.config.production.LogLevel

Logging levels.

**File:** `src/bot\config\production.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DEBUG`

- `INFO`

- `WARNING`

- `ERROR`

- `CRITICAL`


---

## src.bot.config.production.DatabaseConfig

Database configuration.

**File:** `src/bot\config\production.py`



### Methods


#### get_url()

Method description here.



### Attributes



---

## src.bot.config.production.RedisConfig

Redis cache configuration.

**File:** `src/bot\config\production.py`



### Methods


#### get_url()

Method description here.



### Attributes



---

## src.bot.config.production.APIConfig

API service configuration.

**File:** `src/bot\config\production.py`



### Methods



### Attributes



---

## src.bot.config.production.DashboardConfig

Dashboard configuration.

**File:** `src/bot\config\production.py`



### Methods



### Attributes



---

## src.bot.config.production.LoggingConfig

Logging configuration.

**File:** `src/bot\config\production.py`



### Methods



### Attributes



---

## src.bot.config.production.TradingConfig

Trading-specific configuration.

**File:** `src/bot\config\production.py`



### Methods



### Attributes



---

## src.bot.config.production.MonitoringConfig

Monitoring and alerting configuration.

**File:** `src/bot\config\production.py`



### Methods



### Attributes



---

## src.bot.config.production.SecurityConfig

Security configuration.

**File:** `src/bot\config\production.py`



### Methods



### Attributes



---

## src.bot.config.production.SecretsManager

Secure secrets management with encryption.

**File:** `src/bot\config\production.py`



### Methods


#### __init__()

Method description here.


#### _generate_master_key()

Method description here.


#### _create_cipher()

Method description here.


#### encrypt_secret()

Method description here.


#### decrypt_secret()

Method description here.


#### rotate_master_key()

Method description here.



### Attributes



---

## src.bot.config.production.ProductionConfigManager

Production-grade configuration management system.

Handles environment-based configuration, secrets management,
validation, and secure credential handling.

**File:** `src/bot\config\production.py`



### Methods


#### __init__()

Method description here.


#### _load_configuration()

Method description here.


#### _load_default_config()

Method description here.


#### _load_config_file()

Method description here.


#### _load_secrets_file()

Method description here.


#### _load_environment_variables()

Method description here.


#### _deep_merge()

Method description here.


#### _set_nested_value()

Method description here.


#### _get_nested_value()

Method description here.


#### _create_database_config()

Method description here.


#### _create_redis_config()

Method description here.


#### _create_api_config()

Method description here.


#### _create_dashboard_config()

Method description here.


#### _create_logging_config()

Method description here.


#### _create_trading_config()

Method description here.


#### _create_monitoring_config()

Method description here.


#### _create_security_config()

Method description here.


#### _validate_configuration()

Method description here.


#### get_config_summary()

Method description here.


#### export_config()

Method description here.


#### save_secrets_file()

Method description here.



### Attributes



---

## src.bot.core_components.config.cli.CLIContext

CLI context for sharing state between commands

**File:** `src/bot\core_components\config\cli.py`



### Methods


#### __init__()

Method description here.


#### get_manager()

Method description here.



### Attributes



---

## src.bot.core_components.config.integrations.RiskManagementConfigAdapter

Adapter for Phase 1 Unified Risk Management System

**File:** `src/bot\core_components\config\integrations.py`



### Methods


#### __init__()

Method description here.


#### get_risk_config()

Method description here.


#### get_australian_compliance_config()

Method description here.



### Attributes



---

## src.bot.core_components.config.integrations.MLIntegrationConfigAdapter

Adapter for Phase 2.5 ML Integration Layer

**File:** `src/bot\core_components\config\integrations.py`



### Methods


#### __init__()

Method description here.


#### get_ml_config()

Method description here.


#### get_backtesting_config()

Method description here.



### Attributes



---

## src.bot.core_components.config.integrations.APISystemConfigAdapter

Adapter for Phase 3 Unified API System

**File:** `src/bot\core_components\config\integrations.py`



### Methods


#### __init__()

Method description here.


#### get_api_config()

Method description here.



### Attributes



---

## src.bot.core_components.config.integrations.LegacyConfigurationBridge

Bridge for legacy configuration systems

**File:** `src/bot\core_components\config\integrations.py`



### Methods


#### __init__()

Method description here.


#### get_legacy_config_dict()

Method description here.


#### _get_legacy_exchange_config()

Method description here.



### Attributes



---

## src.bot.core_components.config.integrations.ConfigurationIntegrationManager

Manager for configuration integration operations

**File:** `src/bot\core_components\config\integrations.py`



### Methods


#### __init__()

Method description here.


#### update_phase1_risk_system()

Method description here.


#### update_phase25_ml_system()

Method description here.


#### update_phase3_api_system()

Method description here.


#### validate_integration_compatibility()

Method description here.


#### generate_integration_report()

Method description here.



### Attributes



---

## src.bot.core_components.config.manager.ValidationResult

Configuration validation result

**File:** `src/bot\core_components\config\manager.py`



### Methods



### Attributes



---

## src.bot.core_components.config.manager.ConfigurationValidator

Configuration validation and health checking

**File:** `src/bot\core_components\config\manager.py`



### Methods


#### __init__()

Method description here.


#### validate_schema()

Method description here.


#### validate_connectivity()

Method description here.



### Attributes



---

## src.bot.core_components.config.manager.ConfigurationMigrator

Migration utilities for existing configuration files

**File:** `src/bot\core_components\config\manager.py`



### Methods


#### __init__()

Method description here.


#### detect_existing_configs()

Method description here.


#### migrate_yaml_config()

Method description here.


#### migrate_env_file()

Method description here.


#### _transform_to_unified_schema()

Method description here.


#### backup_existing_configs()

Method description here.



### Attributes



---

## src.bot.core_components.config.manager.UnifiedConfigurationManager

Main configuration manager that provides unified configuration management

**File:** `src/bot\core_components\config\manager.py`



### Methods


#### __init__()

Method description here.


#### _initialize_secrets_manager()

Method description here.


#### load_configuration()

Method description here.


#### _find_config_file()

Method description here.


#### _load_config_data()

Method description here.


#### save_configuration()

Method description here.


#### get_configuration()

Method description here.


#### reload_configuration()

Method description here.


#### enable_hot_reload()

Method description here.


#### disable_hot_reload()

Method description here.


#### add_reload_callback()

Method description here.


#### _on_config_changed()

Method description here.


#### _calculate_config_hash()

Method description here.


#### migrate_existing_configs()

Method description here.


#### create_default_configuration()

Method description here.


#### validate_current_configuration()

Method description here.


#### get_configuration_summary()

Method description here.


#### export_configuration()

Method description here.


#### _remove_secrets()

Method description here.


#### __enter__()

Method description here.


#### __exit__()

Method description here.



### Attributes



---

## src.bot.core_components.config.schema.Environment

Deployment environments

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DEVELOPMENT`

- `STAGING`

- `PRODUCTION`

- `TESTING`


---

## src.bot.core_components.config.schema.TradingMode

Trading operation modes

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CONSERVATIVE`

- `AGGRESSIVE`

- `HYBRID`


---

## src.bot.core_components.config.schema.LogLevel

Logging levels

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DEBUG`

- `INFO`

- `WARNING`

- `ERROR`

- `CRITICAL`


---

## src.bot.core_components.config.schema.DatabaseDialect

Database dialects

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** Enum


### Methods



### Attributes


- `POSTGRESQL`

- `DUCKDB`

- `SQLITE`


---

## src.bot.core_components.config.schema.CacheBackend

Cache backend types

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** Enum


### Methods



### Attributes


- `REDIS`

- `MEMORY`

- `HYBRID`


---

## src.bot.core_components.config.schema.ExchangeCredentials

Exchange API credentials

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods


#### validate_credentials()

Method description here.



### Attributes



---

## src.bot.core_components.config.schema.DatabaseConfig

Database configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods


#### get_connection_string()

Method description here.



### Attributes



---

## src.bot.core_components.config.schema.RateLimitConfig

Rate limiting configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.ConnectionConfig

HTTP connection configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.WebSocketConfig

WebSocket configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.CacheConfig

Caching configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.TradingModeConfig

Base trading mode configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.AggressiveModeConfig

Aggressive trading mode configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** TradingModeConfig


### Methods


#### validate_risk_ratios()

Method description here.



### Attributes



---

## src.bot.core_components.config.schema.ConservativeModeConfig

Conservative trading mode configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** TradingModeConfig


### Methods



### Attributes



---

## src.bot.core_components.config.schema.TradingConfig

Trading configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.MLConfig

Machine learning configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.BacktestingConfig

Backtesting configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.AustralianComplianceConfig

Australian compliance configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.LoggingConfig

Logging configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.SecurityConfig

Security configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.MonitoringConfig

Monitoring and alerting configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.APIConfig

API server configuration

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods



### Attributes



---

## src.bot.core_components.config.schema.UnifiedConfigurationSchema

Unified Configuration Schema

Master configuration model that consolidates all component configurations

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** BaseModel


### Methods


#### validate_configuration()

Method description here.


#### get_current_credentials()

Method description here.


#### validate_credentials()

Method description here.


#### get_trading_mode_config()

Method description here.


#### to_dict()

Method description here.


#### get_summary()

Method description here.



### Attributes



---

## src.bot.core_components.config.schema.SecretsManager

Secure secrets management with encryption

**File:** `src/bot\core_components\config\schema.py`



### Methods


#### __init__()

Method description here.


#### _generate_key()

Method description here.


#### encrypt_value()

Method description here.


#### decrypt_value()

Method description here.


#### encrypt_secrets()

Method description here.


#### decrypt_secrets()

Method description here.


#### save_key()

Method description here.


#### load_key()

Method description here.



### Attributes



---

## src.bot.core_components.config.schema.EnvironmentManager

Environment-specific configuration management

**File:** `src/bot\core_components\config\schema.py`



### Methods


#### __init__()

Method description here.


#### _detect_environment()

Method description here.


#### load_environment_variables()

Method description here.


#### apply_environment_overrides()

Method description here.



### Attributes



---

## src.bot.core_components.config.schema.ConfigurationWatcher

File system watcher for configuration changes

**File:** `src/bot\core_components\config\schema.py`


**Inherits from:** FileSystemEventHandler


### Methods


#### __init__()

Method description here.


#### on_modified()

Method description here.



### Attributes



---

## src.bot.core_components.config.schema.Config

Pydantic configuration

**File:** `src/bot\core_components\config\schema.py`



### Methods



### Attributes


- `use_enum_values`

- `validate_assignment`

- `extra`


---

## src.bot.data.collector.DataCollector

Handles market data collection from exchanges.

Supports both historical data fetching and real-time streaming
with automatic data sanitization and storage.

**File:** `src/bot\data\collector.py`



### Methods


#### __init__()

Method description here.


#### _initialize_exchange()

Method description here.


#### get_collection_stats()

Method description here.



### Attributes



---

## src.bot.data.historical_data_manager.DataQualityMetrics

Data quality assessment metrics.

**File:** `src/bot\data\historical_data_manager.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.data.historical_data_manager.DataFetchRequest

Request specification for historical data.

**File:** `src/bot\data\historical_data_manager.py`



### Methods



### Attributes



---

## src.bot.data.historical_data_manager.EnhancedMarketData

Enhanced market data with additional Bybit-specific information.

**File:** `src/bot\data\historical_data_manager.py`



### Methods


#### merge_funding_data()

Method description here.



### Attributes



---

## src.bot.data.historical_data_manager.HistoricalDataManager

Comprehensive historical data management system.

Features:
- Intelligent data fetching with gap detection
- Bybit-specific funding rate integration
- Data quality validation and scoring
- Efficient caching and database storage
- Multi-timeframe data synchronization
- Real-time data updates

**File:** `src/bot\data\historical_data_manager.py`



### Methods


#### __init__()

Method description here.


#### _get_interval_ms()

Method description here.


#### _assess_data_quality()

Method description here.


#### _get_cached_data()

Method description here.


#### _cache_data()

Method description here.


#### get_quality_report()

Method description here.


#### clear_cache()

Method description here.



### Attributes



---

## src.bot.data.multi_exchange_fetcher.ExchangeConfig

Configuration for exchange connections.

**File:** `src/bot\data\multi_exchange_fetcher.py`



### Methods



### Attributes



---

## src.bot.data.multi_exchange_fetcher.MarketData

Standardized market data structure.

**File:** `src/bot\data\multi_exchange_fetcher.py`



### Methods



### Attributes



---

## src.bot.data.multi_exchange_fetcher.MultiExchangeDataFetcher

Multi-exchange data fetcher with cross-exchange feature calculation.

Fetches data from multiple exchanges in parallel and calculates comparative
features for enhanced trading signal generation.

**File:** `src/bot\data\multi_exchange_fetcher.py`



### Methods


#### __init__()

Method description here.


#### _load_exchange_configs()

Method description here.


#### _initialize_exchanges()

Method description here.


#### fetch_cross_exchange_features()

Method description here.


#### _fetch_parallel_data()

Method description here.


#### _fetch_exchange_data()

Method description here.


#### _calculate_cross_exchange_features()

Method description here.


#### _update_quality_metrics()

Method description here.


#### get_data_quality_report()

Method description here.


#### __enter__()

Method description here.


#### __exit__()

Method description here.



### Attributes



---

## src.bot.data.provider.DataProvider

Unified interface for market data access.

Provides seamless access to both historical data from the database
and live data from exchanges.

**File:** `src/bot\data\provider.py`



### Methods


#### __init__()

Method description here.


#### get_historical_data()

Method description here.


#### _get_data_from_database()

Method description here.


#### get_latest_price()

Method description here.


#### get_available_symbols()

Method description here.


#### get_available_timeframes()

Method description here.


#### get_data_coverage()

Method description here.


#### ensure_data_availability()

Method description here.


#### _collect_missing_data()

Method description here.


#### clear_cache()

Method description here.


#### get_cache_stats()

Method description here.



### Attributes



---

## src.bot.data.quality_monitor.DataQualityLevel

Data quality assessment levels.

**File:** `src/bot\data\quality_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `EXCELLENT`

- `GOOD`

- `ACCEPTABLE`

- `POOR`

- `CRITICAL`


---

## src.bot.data.quality_monitor.DataQualityIssue

Types of data quality issues.

**File:** `src/bot\data\quality_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MISSING_DATA`

- `STALE_DATA`

- `OUTLIER_DETECTED`

- `INCONSISTENT_DATA`

- `INVALID_FORMAT`

- `DUPLICATE_DATA`

- `INCOMPLETE_RECORD`

- `ANOMALY_DETECTED`


---

## src.bot.data.quality_monitor.QualityMetrics

Data quality metrics for a data source.

**File:** `src/bot\data\quality_monitor.py`



### Methods



### Attributes



---

## src.bot.data.quality_monitor.DataSource

Configuration for a monitored data source.

**File:** `src/bot\data\quality_monitor.py`



### Methods



### Attributes



---

## src.bot.data.quality_monitor.DataQualityMonitor

Comprehensive data quality monitoring system.

Continuously monitors data quality across all sources, detects issues,
applies automatic fixes where possible, and provides quality reports.

**File:** `src/bot\data\quality_monitor.py`



### Methods


#### __init__()

Method description here.


#### _initialize_data_sources()

Method description here.


#### _initialize_improvement_actions()

Method description here.


#### _initialize_quality_metrics()

Method description here.


#### _assess_completeness()

Method description here.


#### _assess_freshness()

Method description here.


#### _assess_accuracy()

Method description here.


#### _assess_consistency()

Method description here.


#### _check_cross_field_consistency()

Method description here.


#### _determine_quality_level()

Method description here.


#### _identify_issues()

Method description here.


#### _count_outliers()

Method description here.


#### _detect_anomalies()

Method description here.


#### _generate_quality_details()

Method description here.


#### _update_statistical_models()

Method description here.


#### _generate_quality_report()

Method description here.


#### _handle_missing_data()

Method description here.


#### _handle_outliers()

Method description here.


#### update_data()

Method description here.


#### get_quality_metrics()

Method description here.


#### get_all_quality_metrics()

Method description here.


#### get_quality_history()

Method description here.


#### get_system_quality_report()

Method description here.



### Attributes



---

## src.bot.data.quality_monitor.MockStats

No description available.

**File:** `src/bot\data\quality_monitor.py`



### Methods


#### zscore()

Method description here.



### Attributes



---

## src.bot.data.sanitizer.DataSanitizer

Comprehensive data sanitization for market data.

This class implements multiple validation and cleaning methods
to ensure data quality for trading algorithms.

**File:** `src/bot\data\sanitizer.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### sanitize_data()

Method description here.


#### _validate_structure()

Method description here.


#### _normalize_timestamps()

Method description here.


#### _validate_ohlcv()

Method description here.


#### _detect_outliers()

Method description here.


#### _detect_volume_anomalies()

Method description here.


#### _handle_missing_data()

Method description here.


#### _cross_validate_data()

Method description here.


#### _final_validation()

Method description here.


#### get_sanitization_stats()

Method description here.


#### reset_stats()

Method description here.



### Attributes



---

## src.bot.data.sentiment_fetcher.SentimentData

Standardized sentiment data structure.

**File:** `src/bot\data\sentiment_fetcher.py`



### Methods



### Attributes



---

## src.bot.data.sentiment_fetcher.NewsItem

Individual news item structure.

**File:** `src/bot\data\sentiment_fetcher.py`



### Methods



### Attributes



---

## src.bot.data.sentiment_fetcher.FearGreedData

Fear & Greed Index data structure.

**File:** `src/bot\data\sentiment_fetcher.py`



### Methods



### Attributes



---

## src.bot.data.sentiment_fetcher.SentimentDataFetcher

Comprehensive sentiment data fetcher with multiple source integration.

Provides real-time sentiment analysis from news, social media, and market indicators
to enhance trading signal generation and risk management.

**File:** `src/bot\data\sentiment_fetcher.py`



### Methods


#### __init__()

Method description here.


#### fetch_sentiment_features()

Method description here.


#### _fetch_fear_greed_features()

Method description here.


#### _fetch_news_sentiment_features()

Method description here.


#### _fetch_cryptopanic_news()

Method description here.


#### _calculate_text_sentiment()

Method description here.


#### _calculate_news_features()

Method description here.


#### _calculate_aggregated_sentiment()

Method description here.


#### get_sentiment_summary()

Method description here.


#### __del__()

Method description here.



### Attributes



---

## src.bot.database.manager.DatabaseManager

Manages database connections and operations.

Supports both DuckDB for development and PostgreSQL for production
with automatic connection pooling and session management.

**File:** `src/bot\database\manager.py`



### Methods


#### __init__()

Method description here.


#### initialize()

Method description here.


#### _create_engine()

Method description here.


#### _create_postgresql_engine()

Method description here.


#### _create_duckdb_engine()

Method description here.


#### _setup_postgresql_events()

Method description here.


#### _setup_duckdb_events()

Method description here.


#### _create_tables()

Method description here.


#### get_session()

Method description here.


#### execute_query()

Method description here.


#### get_table_info()

Method description here.


#### backup_database()

Method description here.


#### reset_database()

Method description here.


#### get_connection_info()

Method description here.


#### health_check()

Method description here.


#### close()

Method description here.



### Attributes



---

## src.bot.database.manager.DatabaseSession

Context manager for database sessions.

Provides a more convenient interface for database operations
with automatic session management.

**File:** `src/bot\database\manager.py`



### Methods


#### __init__()

Method description here.


#### __enter__()

Method description here.


#### __exit__()

Method description here.



### Attributes



---

## src.bot.database.models.Trade

Trade execution records with comprehensive tax tracking.

This model stores all trade executions with fields required for
Australian CGT calculations and audit trails.

**File:** `src/bot\database\models.py`


**Inherits from:** Base


### Methods


#### __repr__()

Method description here.



### Attributes


- `__tablename__`

- `id`

- `timestamp`

- `exchange`

- `symbol`

- `side`

- `amount`

- `price`

- `fee`

- `fee_currency`

- `strategy_id`

- `strategy_version`

- `signal_confidence`

- `order_id`

- `order_type`

- `time_in_force`

- `slippage`

- `risk_amount`

- `position_size_usd`

- `portfolio_balance`

- `trading_mode`

- `cost_base_aud`

- `proceeds_aud`

- `is_cgt_event`

- `aud_conversion_rate`

- `holding_period_days`

- `unrealized_pnl`

- `realized_pnl`

- `created_at`

- `updated_at`

- `notes`

- `__table_args__`


---

## src.bot.database.models.StrategyPerformance

Strategy performance metrics over time.

Tracks key performance indicators for each strategy including
risk-adjusted returns and mode-specific metrics.

**File:** `src/bot\database\models.py`


**Inherits from:** Base


### Methods


#### __repr__()

Method description here.



### Attributes


- `__tablename__`

- `id`

- `strategy_id`

- `timestamp`

- `period`

- `equity`

- `returns`

- `cumulative_returns`

- `volatility`

- `sharpe_ratio`

- `sortino_ratio`

- `calmar_ratio`

- `max_drawdown`

- `current_drawdown`

- `total_trades`

- `winning_trades`

- `losing_trades`

- `win_rate`

- `profit_factor`

- `trading_mode`

- `risk_parameters`

- `oos_performance`

- `consistency_score`

- `created_at`

- `__table_args__`


---

## src.bot.database.models.Portfolio

Portfolio state snapshots over time.

Tracks overall portfolio value, positions, and risk metrics.

**File:** `src/bot\database\models.py`


**Inherits from:** Base


### Methods


#### __repr__()

Method description here.



### Attributes


- `__tablename__`

- `id`

- `timestamp`

- `total_value_usd`

- `available_balance`

- `unrealized_pnl`

- `realized_pnl`

- `portfolio_var`

- `portfolio_drawdown`

- `leverage_ratio`

- `trading_mode`

- `active_strategies`

- `risk_utilization`

- `positions`

- `correlation_matrix`

- `daily_pnl`

- `weekly_pnl`

- `monthly_pnl`

- `created_at`

- `__table_args__`


---

## src.bot.database.models.RiskEvent

Risk management events and violations.

Records all risk-related events including limit breaches,
circuit breaker activations, and mode changes.

**File:** `src/bot\database\models.py`


**Inherits from:** Base


### Methods


#### __repr__()

Method description here.



### Attributes


- `__tablename__`

- `id`

- `timestamp`

- `event_type`

- `severity`

- `description`

- `strategy_id`

- `symbol`

- `current_value`

- `limit_value`

- `portfolio_balance`

- `trading_mode`

- `action_taken`

- `resolved`

- `resolved_at`

- `event_metadata`

- `created_at`

- `__table_args__`


---

## src.bot.database.models.TaxEvent

Tax-related events for Australian CGT compliance.

Tracks CGT events, AUD conversions, and generates data
for tax reporting.

**File:** `src/bot\database\models.py`


**Inherits from:** Base


### Methods


#### __repr__()

Method description here.



### Attributes


- `__tablename__`

- `id`

- `timestamp`

- `financial_year`

- `event_type`

- `trade_id`

- `symbol`

- `amount`

- `cost_base_aud`

- `proceeds_aud`

- `capital_gain_aud`

- `is_cgt_discount_eligible`

- `holding_period_days`

- `discount_amount_aud`

- `net_capital_gain_aud`

- `usd_aud_rate`

- `rate_source`

- `matched_acquisition_id`

- `trading_mode`

- `notes`

- `created_at`

- `trade`

- `matched_acquisition`

- `__table_args__`


---

## src.bot.database.models.MarketData

Market data storage for backtesting and analysis.

Stores OHLCV data and derived indicators for all
trading symbols and timeframes.

**File:** `src/bot\database\models.py`


**Inherits from:** Base


### Methods


#### __repr__()

Method description here.



### Attributes


- `__tablename__`

- `id`

- `symbol`

- `timeframe`

- `timestamp`

- `open`

- `high`

- `low`

- `close`

- `volume`

- `vwap`

- `volatility`

- `sma_20`

- `ema_20`

- `rsi_14`

- `atr_14`

- `is_validated`

- `has_anomalies`

- `data_source`

- `created_at`

- `__table_args__`


---

## src.bot.database.models.StrategyMetadata

Strategy metadata and configuration tracking.

Stores strategy definitions, parameters, and lifecycle information.

**File:** `src/bot\database\models.py`


**Inherits from:** Base


### Methods


#### __repr__()

Method description here.



### Attributes


- `__tablename__`

- `id`

- `strategy_id`

- `name`

- `description`

- `strategy_type`

- `category`

- `parameters`

- `symbols`

- `timeframes`

- `status`

- `created_at`

- `activated_at`

- `retired_at`

- `validation_results`

- `last_validation`

- `total_trades`

- `total_pnl`

- `best_sharpe`

- `max_drawdown`

- `conservative_approved`

- `aggressive_approved`

- `updated_at`

- `__table_args__`


---

## src.bot.deployment.infrastructure.DeploymentConfig

Deployment configuration settings.

**File:** `src/bot\deployment\infrastructure.py`



### Methods



### Attributes



---

## src.bot.deployment.infrastructure.KubernetesDeploymentGenerator

Generate Kubernetes deployment manifests.

**File:** `src/bot\deployment\infrastructure.py`



### Methods


#### __init__()

Method description here.


#### generate_all_manifests()

Method description here.


#### _generate_namespace()

Method description here.


#### _generate_configmap()

Method description here.


#### _generate_secret_template()

Method description here.


#### _generate_deployment()

Method description here.


#### _generate_service()

Method description here.


#### _generate_ingress()

Method description here.


#### _generate_hpa()

Method description here.


#### _generate_pvc()

Method description here.


#### _generate_service_account()

Method description here.


#### _generate_rbac()

Method description here.


#### _generate_network_policy()

Method description here.



### Attributes



---

## src.bot.deployment.infrastructure.CIPipelineGenerator

Generate CI/CD pipeline configurations.

**File:** `src/bot\deployment\infrastructure.py`



### Methods


#### __init__()

Method description here.


#### generate_github_actions()

Method description here.


#### generate_gitlab_ci()

Method description here.



### Attributes



---

## src.bot.deployment.infrastructure.MonitoringSetup

Setup monitoring and alerting infrastructure.

**File:** `src/bot\deployment\infrastructure.py`



### Methods


#### __init__()

Method description here.


#### generate_prometheus_config()

Method description here.


#### generate_alert_rules()

Method description here.


#### generate_grafana_dashboard()

Method description here.



### Attributes



---

## src.bot.exchange.bybit_client.BybitCredentials

Bybit API credentials configuration.

**File:** `src/bot\exchange\bybit_client.py`



### Methods


#### base_url()

Method description here.



### Attributes



---

## src.bot.exchange.bybit_client.MarketDataResponse

Standardized market data response.

**File:** `src/bot\exchange\bybit_client.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.exchange.bybit_client.FundingRateData

Funding rate historical data.

**File:** `src/bot\exchange\bybit_client.py`



### Methods



### Attributes



---

## src.bot.exchange.bybit_client.BybitAPIError

Custom exception for Bybit API errors.

**File:** `src/bot\exchange\bybit_client.py`


**Inherits from:** Exception


### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.exchange.bybit_client.BybitClient

Comprehensive Bybit API v5 client implementation.

Features:
- Full authentication with HMAC SHA256 signatures
- Automatic rate limiting compliance
- Historical data fetching with caching
- Real-time WebSocket connections
- Error handling with exponential backoff
- Trading operations for all account types

**File:** `src/bot\exchange\bybit_client.py`



### Methods


#### __init__()

Method description here.


#### _generate_signature()

Method description here.


#### _get_headers()

Method description here.



### Attributes



---

## src.bot.execution.execution_analytics.BenchmarkType

Benchmark type enumeration.

**File:** `src/bot\execution\execution_analytics.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ARRIVAL_PRICE`

- `TWAP`

- `VWAP`

- `OPEN_PRICE`

- `CLOSE_PRICE`

- `MID_PRICE`


---

## src.bot.execution.execution_analytics.ExecutionMetrics

Container for execution performance metrics.

**File:** `src/bot\execution\execution_analytics.py`



### Methods


#### __post_init__()

Method description here.


#### calculate_slippage()

Method description here.


#### calculate_implementation_shortfall()

Method description here.


#### to_dict()

Method description here.



### Attributes



---

## src.bot.execution.execution_analytics.StrategyMetrics

Aggregated metrics for an execution strategy.

**File:** `src/bot\execution\execution_analytics.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.execution.execution_analytics.MarketImpactModel

Market impact model parameters.

**File:** `src/bot\execution\execution_analytics.py`



### Methods


#### predict_impact()

Method description here.



### Attributes



---

## src.bot.execution.execution_analytics.ExecutionAnalyzer

Comprehensive execution analysis and performance tracking.

This class analyzes order execution performance, calculates
various metrics, and provides insights for optimization.

**File:** `src/bot\execution\execution_analytics.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### _init_database()

Method description here.


#### analyze_order_execution()

Method description here.


#### analyze_execution_plan()

Method description here.


#### _get_arrival_price()

Method description here.


#### _calculate_benchmark_prices()

Method description here.


#### _calculate_market_impact()

Method description here.


#### _calculate_aggressiveness_score()

Method description here.


#### _calculate_execution_quality_score()

Method description here.


#### _analyze_participation_rate()

Method description here.


#### _update_strategy_metrics()

Method description here.


#### _recalculate_strategy_statistics()

Method description here.


#### update_market_data()

Method description here.


#### get_execution_metrics()

Method description here.


#### get_strategy_metrics()

Method description here.


#### get_symbol_performance()

Method description here.


#### generate_execution_report()

Method description here.


#### _save_execution_metrics()

Method description here.


#### export_metrics()

Method description here.



### Attributes



---

## src.bot.execution.liquidity_seeker.LiquidityVenue

Supported liquidity venues

**File:** `src/bot\execution\liquidity_seeker.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BYBIT_SPOT`

- `BYBIT_DERIVATIVES`

- `EXTERNAL_SPOT`

- `DARK_POOLS`


---

## src.bot.execution.liquidity_seeker.LiquidityType

Types of liquidity

**File:** `src/bot\execution\liquidity_seeker.py`


**Inherits from:** Enum


### Methods



### Attributes


- `VISIBLE`

- `HIDDEN`

- `DARK`

- `SWEPT`


---

## src.bot.execution.liquidity_seeker.LiquiditySource

Information about a liquidity source

**File:** `src/bot\execution\liquidity_seeker.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.execution.liquidity_seeker.LiquiditySnapshot

Complete liquidity snapshot across venues

**File:** `src/bot\execution\liquidity_seeker.py`



### Methods



### Attributes



---

## src.bot.execution.liquidity_seeker.LiquidityStrategy

Liquidity seeking strategy

**File:** `src/bot\execution\liquidity_seeker.py`



### Methods



### Attributes



---

## src.bot.execution.liquidity_seeker.LiquiditySeeker

Advanced liquidity seeking and optimization engine

Key Features:
- Multi-venue liquidity discovery ✅
- Hidden liquidity detection ✅  
- Optimal execution routing ✅
- Dynamic strategy adaptation ✅
- >98% fill rate achievement ✅

**File:** `src/bot\execution\liquidity_seeker.py`



### Methods


#### __init__()

Method description here.


#### _initialize_strategies()

Method description here.


#### _convert_to_derivatives_symbol()

Method description here.


#### _calculate_strategy_performance()

Method description here.



### Attributes



---

## src.bot.execution.optimized_execution.OrderType

Advanced order types supported by optimized execution

**File:** `src/bot\execution\optimized_execution.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MARKET`

- `LIMIT`

- `STOP_LOSS`

- `TAKE_PROFIT`

- `OCO`

- `TRAILING_STOP`

- `ICEBERG`

- `TWAP`

- `VWAP`


---

## src.bot.execution.optimized_execution.ExecutionStrategy

Execution strategy selection

**File:** `src/bot\execution\optimized_execution.py`


**Inherits from:** Enum


### Methods



### Attributes


- `AGGRESSIVE`

- `PASSIVE`

- `BALANCED`

- `STEALTH`

- `LIQUIDITY_SEEKING`


---

## src.bot.execution.optimized_execution.OptimizedOrder

Optimized order with advanced features

**File:** `src/bot\execution\optimized_execution.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.execution.optimized_execution.ExecutionResult

Result of order execution

**File:** `src/bot\execution\optimized_execution.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.execution.optimized_execution.OptimizedExecutionEngine

Advanced execution engine with optimization features

Key Features:
- Sub-80ms execution times ✅
- Advanced order types support ✅
- Slippage minimization ✅
- Liquidity seeking algorithms ✅
- Real-time quality monitoring ✅

**File:** `src/bot\execution\optimized_execution.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.execution.order_management.OrderType

Order type enumeration.

**File:** `src/bot\execution\order_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MARKET`

- `LIMIT`

- `STOP_MARKET`

- `STOP_LIMIT`

- `TAKE_PROFIT`

- `TAKE_PROFIT_LIMIT`

- `OCO`

- `ICEBERG`

- `TWAP`

- `VWAP`


---

## src.bot.execution.order_management.OrderSide

Order side enumeration.

**File:** `src/bot\execution\order_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BUY`

- `SELL`


---

## src.bot.execution.order_management.OrderStatus

Order status enumeration.

**File:** `src/bot\execution\order_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `SUBMITTED`

- `OPEN`

- `PARTIALLY_FILLED`

- `FILLED`

- `CANCELLED`

- `REJECTED`

- `EXPIRED`

- `FAILED`


---

## src.bot.execution.order_management.TimeInForce

Time in force enumeration.

**File:** `src/bot\execution\order_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `GTC`

- `IOC`

- `FOK`

- `GTD`

- `DAY`


---

## src.bot.execution.order_management.OrderPriority

Order priority enumeration.

**File:** `src/bot\execution\order_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW`

- `NORMAL`

- `HIGH`

- `URGENT`

- `CRITICAL`


---

## src.bot.execution.order_management.OrderFill

Container for order fill information.

**File:** `src/bot\execution\order_management.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.execution.order_management.Order

Core order representation with comprehensive tracking.

**File:** `src/bot\execution\order_management.py`



### Methods


#### __post_init__()

Method description here.


#### is_buy()

Method description here.


#### is_sell()

Method description here.


#### is_market_order()

Method description here.


#### is_limit_order()

Method description here.


#### is_stop_order()

Method description here.


#### is_filled()

Method description here.


#### is_partially_filled()

Method description here.


#### is_open()

Method description here.


#### is_closed()

Method description here.


#### fill_percentage()

Method description here.


#### total_fees()

Method description here.


#### add_fill()

Method description here.


#### update_status()

Method description here.


#### to_dict()

Method description here.



### Attributes



---

## src.bot.execution.order_management.OrderValidator

Order validation system with comprehensive checks.

This class provides extensive validation for orders before submission
including risk checks, balance validation, and market constraints.

**File:** `src/bot\execution\order_management.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### validate_order()

Method description here.


#### _validate_basic_order()

Method description here.


#### _validate_price()

Method description here.


#### _validate_size()

Method description here.


#### _validate_symbol()

Method description here.


#### _validate_balance()

Method description here.


#### _validate_position_limits()

Method description here.


#### _validate_order_type_specific()

Method description here.



### Attributes



---

## src.bot.execution.order_management.OrderBook

Simple order book representation for price calculations.

This class maintains a basic order book for price impact calculations
and optimal order placement strategies.

**File:** `src/bot\execution\order_management.py`



### Methods


#### __init__()

Method description here.


#### update()

Method description here.


#### best_bid()

Method description here.


#### best_ask()

Method description here.


#### mid_price()

Method description here.


#### spread()

Method description here.


#### spread_bps()

Method description here.


#### calculate_slippage()

Method description here.


#### get_optimal_limit_price()

Method description here.



### Attributes



---

## src.bot.execution.order_management.OrderManager

Core order management system with comprehensive tracking and monitoring.

This class provides centralized order management including creation,
validation, tracking, and lifecycle management.

**File:** `src/bot\execution\order_management.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### create_order()

Method description here.


#### get_order()

Method description here.


#### get_orders_by_symbol()

Method description here.


#### get_orders_by_strategy()

Method description here.


#### get_open_orders()

Method description here.


#### update_order_status()

Method description here.


#### add_order_fill()

Method description here.


#### cancel_order()

Method description here.


#### cancel_all_orders()

Method description here.


#### update_order_book()

Method description here.


#### get_order_book()

Method description here.


#### calculate_order_slippage()

Method description here.


#### get_optimal_price()

Method description here.


#### add_order_callback()

Method description here.


#### remove_order_callback()

Method description here.


#### get_order_statistics()

Method description here.


#### _validate_order()

Method description here.


#### _check_order_limits()

Method description here.


#### _notify_callbacks()

Method description here.


#### cleanup_old_orders()

Method description here.



### Attributes



---

## src.bot.execution.position_management.PositionSide

Position side enumeration.

**File:** `src/bot\execution\position_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LONG`

- `SHORT`

- `FLAT`


---

## src.bot.execution.position_management.PositionStatus

Position status enumeration.

**File:** `src/bot\execution\position_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `OPEN`

- `CLOSING`

- `CLOSED`

- `HEDGED`


---

## src.bot.execution.position_management.PositionSnapshot

Snapshot of position state at a specific time.

**File:** `src/bot\execution\position_management.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.execution.position_management.Position

Comprehensive position tracking with real-time updates.

This class maintains detailed position state including size,
average price, PnL, and risk metrics.

**File:** `src/bot\execution\position_management.py`



### Methods


#### side()

Method description here.


#### is_long()

Method description here.


#### is_short()

Method description here.


#### is_flat()

Method description here.


#### abs_size()

Method description here.


#### notional_value()

Method description here.


#### total_pnl()

Method description here.


#### pnl_percentage()

Method description here.


#### return_on_margin()

Method description here.


#### update_market_price()

Method description here.


#### add_trade()

Method description here.


#### _calculate_realized_pnl()

Method description here.


#### take_snapshot()

Method description here.


#### calculate_var()

Method description here.


#### get_performance_metrics()

Method description here.


#### _calculate_win_rate()

Method description here.


#### _calculate_profit_factor()

Method description here.


#### _calculate_sharpe_ratio()

Method description here.


#### to_dict()

Method description here.



### Attributes



---

## src.bot.execution.position_management.PositionRiskMonitor

Position-based risk monitoring and alerting.

This class monitors positions for risk violations and
triggers alerts or automatic actions.

**File:** `src/bot\execution\position_management.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### check_position_risk()

Method description here.


#### _trigger_alerts()

Method description here.


#### add_alert_callback()

Method description here.


#### get_violation_summary()

Method description here.



### Attributes



---

## src.bot.execution.position_management.PositionManager

Comprehensive position management system.

This class provides centralized position tracking, monitoring,
and management across multiple symbols and exchanges.

**File:** `src/bot\execution\position_management.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### _init_database()

Method description here.


#### get_position()

Method description here.


#### update_position_from_fill()

Method description here.


#### update_market_price()

Method description here.


#### update_all_market_prices()

Method description here.


#### get_positions()

Method description here.


#### get_open_positions()

Method description here.


#### get_portfolio_value()

Method description here.


#### get_portfolio_pnl()

Method description here.


#### get_exposure_analysis()

Method description here.


#### calculate_portfolio_var()

Method description here.


#### check_risk_limits()

Method description here.


#### close_position()

Method description here.


#### start_monitoring()

Method description here.


#### stop_monitoring()

Method description here.


#### _monitoring_loop()

Method description here.


#### _save_position_to_db()

Method description here.


#### _save_snapshot_to_db()

Method description here.


#### _load_position_from_db()

Method description here.


#### add_position_callback()

Method description here.


#### remove_position_callback()

Method description here.


#### _notify_callbacks()

Method description here.


#### get_position_summary()

Method description here.


#### export_positions()

Method description here.



### Attributes



---

## src.bot.execution.slippage_minimizer.SlippageAnalysis

Comprehensive slippage analysis result

**File:** `src/bot\execution\slippage_minimizer.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.execution.slippage_minimizer.OrderBookSnapshot

Detailed order book snapshot for analysis

**File:** `src/bot\execution\slippage_minimizer.py`



### Methods



### Attributes



---

## src.bot.execution.slippage_minimizer.SlippageMinimizer

Advanced slippage minimization engine

Features:
- Real-time slippage prediction ✅
- Adaptive execution strategies ✅
- ML-based timing optimization ✅
- Order book impact modeling ✅
- Sub-5bps average slippage ✅

**File:** `src/bot\execution\slippage_minimizer.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.execution.smart_routing.ExecutionStrategy

Execution strategy enumeration.

**File:** `src/bot\execution\smart_routing.py`


**Inherits from:** Enum


### Methods



### Attributes


- `IMMEDIATE`

- `PASSIVE`

- `AGGRESSIVE`

- `TWAP`

- `VWAP`

- `ICEBERG`

- `ADAPTIVE`

- `POV`

- `ARRIVAL_PRICE`


---

## src.bot.execution.smart_routing.MarketCondition

Market condition enumeration.

**File:** `src/bot\execution\smart_routing.py`


**Inherits from:** Enum


### Methods



### Attributes


- `QUIET`

- `ACTIVE`

- `VOLATILE`

- `STRESSED`

- `TRENDING`

- `RANGING`


---

## src.bot.execution.smart_routing.ExecutionPlan

Execution plan for complex orders.

This class defines how a large order should be broken down
and executed over time to minimize market impact.

**File:** `src/bot\execution\smart_routing.py`



### Methods


#### __post_init__()

Method description here.


#### completion_rate()

Method description here.


#### is_complete()

Method description here.


#### update_execution()

Method description here.



### Attributes



---

## src.bot.execution.smart_routing.MarketMetrics

Market microstructure metrics for execution decisions.

**File:** `src/bot\execution\smart_routing.py`



### Methods



### Attributes



---

## src.bot.execution.smart_routing.MarketAnalyzer

Market microstructure analyzer for execution optimization.

This class analyzes real-time market data to assess execution
conditions and optimize routing strategies.

**File:** `src/bot\execution\smart_routing.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### analyze_market_conditions()

Method description here.


#### _calculate_market_depth()

Method description here.


#### _calculate_volume_metrics()

Method description here.


#### _calculate_volatility_metrics()

Method description here.


#### _calculate_order_flow_metrics()

Method description here.


#### _classify_market_condition()

Method description here.


#### _calculate_liquidity_score()

Method description here.


#### _calculate_execution_difficulty()

Method description here.


#### update_price_history()

Method description here.


#### update_volume_history()

Method description here.


#### update_trade_history()

Method description here.



### Attributes



---

## src.bot.execution.smart_routing.SmartRouter

Smart order routing system for optimal execution.

This class determines the best execution strategy based on
order characteristics and current market conditions.

**File:** `src/bot\execution\smart_routing.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### route_order()

Method description here.


#### _create_simple_order()

Method description here.


#### _select_execution_strategy()

Method description here.


#### _create_execution_plan()

Method description here.


#### _start_execution_plan()

Method description here.


#### start_execution_engine()

Method description here.


#### stop_execution_engine()

Method description here.


#### _execution_loop()

Method description here.


#### _process_execution_plan()

Method description here.


#### _execute_twap()

Method description here.


#### _execute_vwap()

Method description here.


#### _execute_iceberg()

Method description here.


#### _execute_adaptive()

Method description here.


#### _execute_pov()

Method description here.


#### _create_child_order()

Method description here.


#### get_execution_status()

Method description here.


#### cancel_execution_plan()

Method description here.



### Attributes



---

## src.bot.features.cross_exchange_calculator.CrossExchangeFeatures

Container for cross-exchange features.

**File:** `src/bot\features\cross_exchange_calculator.py`



### Methods



### Attributes



---

## src.bot.features.cross_exchange_calculator.ArbitrageOpportunity

Represents a detected arbitrage opportunity.

**File:** `src/bot\features\cross_exchange_calculator.py`



### Methods



### Attributes



---

## src.bot.features.cross_exchange_calculator.CrossExchangeFeatureCalculator

Advanced cross-exchange feature calculator for cryptocurrency trading.

Analyzes market data across multiple exchanges to generate sophisticated
features for trading algorithms, including arbitrage detection, market
efficiency analysis, and liquidity assessment.

**File:** `src/bot\features\cross_exchange_calculator.py`



### Methods


#### __init__()

Method description here.


#### calculate_features()

Method description here.


#### _validate_market_data()

Method description here.


#### _extract_prices()

Method description here.


#### _extract_volumes()

Method description here.


#### _extract_depths()

Method description here.


#### _update_history()

Method description here.


#### _calculate_price_features()

Method description here.


#### _calculate_spread_features()

Method description here.


#### _calculate_volume_features()

Method description here.


#### _calculate_arbitrage_features()

Method description here.


#### _calculate_arbitrage_confidence()

Method description here.


#### _get_historical_spreads()

Method description here.


#### _calculate_depth_features()

Method description here.


#### _calculate_correlation_features()

Method description here.


#### _calculate_price_correlation()

Method description here.


#### _calculate_volume_correlation()

Method description here.


#### _calculate_efficiency_features()

Method description here.


#### _calculate_price_efficiency_score()

Method description here.


#### _calculate_leadership_scores()

Method description here.


#### _cleanup_cache()

Method description here.


#### get_recent_features()

Method description here.


#### get_arbitrage_opportunities()

Method description here.


#### get_market_efficiency_report()

Method description here.


#### get_exchange_comparison()

Method description here.



### Attributes



---

## src.bot.hft.arbitrage_detector.ArbitrageType

Types of arbitrage opportunities.

**File:** `src/bot\hft\arbitrage_detector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SPATIAL`

- `TEMPORAL`

- `TRIANGULAR`

- `STATISTICAL`

- `MERGER`

- `CALENDAR`

- `VOLATILITY`

- `FUNDING_RATE`

- `BASIS`

- `DIVIDEND`


---

## src.bot.hft.arbitrage_detector.ArbitrageStatus

Arbitrage opportunity status.

**File:** `src/bot\hft\arbitrage_detector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DETECTED`

- `VALIDATED`

- `EXECUTING`

- `EXECUTED`

- `EXPIRED`

- `FAILED`


---

## src.bot.hft.arbitrage_detector.RiskLevel

Risk levels for arbitrage opportunities.

**File:** `src/bot\hft\arbitrage_detector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW`

- `MEDIUM`

- `HIGH`

- `EXTREME`


---

## src.bot.hft.arbitrage_detector.ArbitrageOpportunity

Arbitrage opportunity data structure.

**File:** `src/bot\hft\arbitrage_detector.py`



### Methods



### Attributes



---

## src.bot.hft.arbitrage_detector.MarketPrice

Market price data structure.

**File:** `src/bot\hft\arbitrage_detector.py`



### Methods



### Attributes



---

## src.bot.hft.arbitrage_detector.ArbitrageMetrics

Arbitrage detection metrics.

**File:** `src/bot\hft\arbitrage_detector.py`



### Methods



### Attributes



---

## src.bot.hft.arbitrage_detector.ArbitrageDetector

Advanced arbitrage detection engine.

**File:** `src/bot\hft\arbitrage_detector.py`



### Methods


#### __init__()

Method description here.


#### _assess_risk_level()

Method description here.


#### get_active_opportunities()

Method description here.


#### get_detection_metrics()

Method description here.



### Attributes



---

## src.bot.hft.arbitrage_detector.SpatialArbitrageDetector

Specialized spatial arbitrage detector.

**File:** `src/bot\hft\arbitrage_detector.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.arbitrage_detector.TriangularArbitrageDetector

Specialized triangular arbitrage detector.

**File:** `src/bot\hft\arbitrage_detector.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.arbitrage_detector.StatisticalArbitrageDetector

Specialized statistical arbitrage detector.

**File:** `src/bot\hft\arbitrage_detector.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.hft_execution_engine.ExecutionStrategy

Execution strategies.

**File:** `src/bot\hft\hft_execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `IMMEDIATE`

- `TWAP`

- `VWAP`

- `ICEBERG`

- `SNIPER`

- `AGGRESSIVE`

- `STEALTH`

- `LIQUIDITY_SEEKING`

- `MOMENTUM`

- `MEAN_REVERSION`


---

## src.bot.hft.hft_execution_engine.OrderStatus

Order execution status.

**File:** `src/bot\hft\hft_execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `ROUTING`

- `SENT`

- `PARTIALLY_FILLED`

- `FILLED`

- `CANCELLED`

- `REJECTED`

- `EXPIRED`

- `FAILED`


---

## src.bot.hft.hft_execution_engine.ExecutionVenue

Execution venues.

**File:** `src/bot\hft\hft_execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PRIMARY_EXCHANGE`

- `DARK_POOL`

- `CROSS_NETWORK`

- `SMART_ROUTER`


---

## src.bot.hft.hft_execution_engine.UrgencyLevel

Execution urgency levels.

**File:** `src/bot\hft\hft_execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW`

- `MEDIUM`

- `HIGH`

- `CRITICAL`


---

## src.bot.hft.hft_execution_engine.ExecutionOrder

HFT execution order.

**File:** `src/bot\hft\hft_execution_engine.py`



### Methods



### Attributes



---

## src.bot.hft.hft_execution_engine.ExecutionFill

Order fill information.

**File:** `src/bot\hft\hft_execution_engine.py`



### Methods



### Attributes



---

## src.bot.hft.hft_execution_engine.MarketData

Real-time market data for execution.

**File:** `src/bot\hft\hft_execution_engine.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.hft.hft_execution_engine.ExecutionMetrics

Execution performance metrics.

**File:** `src/bot\hft\hft_execution_engine.py`



### Methods



### Attributes



---

## src.bot.hft.hft_execution_engine.HFTExecutionEngine

Ultra-low latency execution engine.

**File:** `src/bot\hft\hft_execution_engine.py`



### Methods


#### __init__()

Method description here.


#### get_execution_summary()

Method description here.


#### get_order_status()

Method description here.



### Attributes



---

## src.bot.hft.hft_risk_manager.RiskLevel

Risk severity levels.

**File:** `src/bot\hft\hft_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW`

- `MEDIUM`

- `HIGH`

- `CRITICAL`

- `EXTREME`


---

## src.bot.hft.hft_risk_manager.RiskAction

Risk control actions.

**File:** `src/bot\hft\hft_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MONITOR`

- `WARN`

- `REDUCE_POSITION`

- `CANCEL_ORDERS`

- `HALT_TRADING`

- `EMERGENCY_EXIT`


---

## src.bot.hft.hft_risk_manager.RiskMetricType

Types of risk metrics.

**File:** `src/bot\hft\hft_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `POSITION_SIZE`

- `NOTIONAL_EXPOSURE`

- `PNL`

- `DRAWDOWN`

- `VOLATILITY`

- `CORRELATION`

- `LIQUIDITY`

- `CONCENTRATION`

- `LEVERAGE`


---

## src.bot.hft.hft_risk_manager.CircuitBreakerType

Circuit breaker types.

**File:** `src/bot\hft\hft_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DAILY_LOSS`

- `POSITION_LIMIT`

- `VOLATILITY_SPIKE`

- `CORRELATION_BREAKDOWN`

- `LIQUIDITY_CRISIS`

- `ORDER_REJECTION_RATE`

- `LATENCY_SPIKE`


---

## src.bot.hft.hft_risk_manager.RiskLimit

Risk limit definition.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods



### Attributes



---

## src.bot.hft.hft_risk_manager.RiskAlert

Risk alert.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods



### Attributes



---

## src.bot.hft.hft_risk_manager.PositionRisk

Position-level risk metrics.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods



### Attributes



---

## src.bot.hft.hft_risk_manager.PortfolioRisk

Portfolio-level risk metrics.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods



### Attributes



---

## src.bot.hft.hft_risk_manager.RiskScenario

Risk scenario for stress testing.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods



### Attributes



---

## src.bot.hft.hft_risk_manager.HFTRiskManager

High-frequency trading risk manager.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods


#### __init__()

Method description here.


#### _initialize_default_limits()

Method description here.


#### get_risk_summary()

Method description here.



### Attributes



---

## src.bot.hft.hft_risk_manager.VaRCalculator

Value at Risk calculator.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.hft_risk_manager.StressTester

Stress testing engine.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.hft_risk_manager.CorrelationMonitor

Correlation monitoring system.

**File:** `src/bot\hft\hft_risk_manager.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.latency_engine.LatencyType

Types of latency measurements.

**File:** `src/bot\hft\latency_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `NETWORK_RTT`

- `ORDER_PROCESSING`

- `MARKET_DATA`

- `EXECUTION`

- `WEBSOCKET`

- `API_CALL`

- `INTERNAL_PROCESSING`

- `END_TO_END`


---

## src.bot.hft.latency_engine.OptimizationLevel

Latency optimization levels.

**File:** `src/bot\hft\latency_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `STANDARD`

- `AGGRESSIVE`

- `ULTRA_LOW`

- `CUSTOM`


---

## src.bot.hft.latency_engine.LatencyMeasurement

Individual latency measurement.

**File:** `src/bot\hft\latency_engine.py`



### Methods



### Attributes



---

## src.bot.hft.latency_engine.LatencyMetrics

Aggregated latency metrics.

**File:** `src/bot\hft\latency_engine.py`



### Methods



### Attributes



---

## src.bot.hft.latency_engine.NetworkPath

Network path information.

**File:** `src/bot\hft\latency_engine.py`



### Methods



### Attributes



---

## src.bot.hft.latency_engine.LatencyEngine

Ultra-low latency engine for HFT operations.

**File:** `src/bot\hft\latency_engine.py`



### Methods


#### __init__()

Method description here.


#### measure_latency_sync()

Method description here.


#### get_current_metrics()

Method description here.


#### _check_latency_alert()

Method description here.


#### get_optimization_recommendations()

Method description here.


#### get_latency_summary()

Method description here.



### Attributes



---

## src.bot.hft.latency_engine.NetworkMonitor

Network performance monitor for HFT operations.

**File:** `src/bot\hft\latency_engine.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.latency_engine.LatencyOptimizer

Advanced latency optimization engine.

**File:** `src/bot\hft\latency_engine.py`



### Methods


#### __init__()

Method description here.


#### _analyze_improvement()

Method description here.



### Attributes



---

## src.bot.hft.market_making_engine.MarketMakingStrategy

Market making strategies.

**File:** `src/bot\hft\market_making_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SIMPLE_SPREAD`

- `ADAPTIVE_SPREAD`

- `INVENTORY_BASED`

- `VOLATILITY_ADJUSTED`

- `MOMENTUM_AWARE`

- `ORDERBOOK_IMBALANCE`

- `ADVERSE_SELECTION`

- `OPTIMAL_EXECUTION`


---

## src.bot.hft.market_making_engine.QuoteStatus

Quote status.

**File:** `src/bot\hft\market_making_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ACTIVE`

- `FILLED`

- `CANCELLED`

- `REJECTED`

- `PENDING`


---

## src.bot.hft.market_making_engine.InventoryDirection

Inventory direction.

**File:** `src/bot\hft\market_making_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LONG`

- `SHORT`

- `NEUTRAL`


---

## src.bot.hft.market_making_engine.MarketQuote

Market making quote.

**File:** `src/bot\hft\market_making_engine.py`



### Methods



### Attributes



---

## src.bot.hft.market_making_engine.InventoryPosition

Inventory position tracking.

**File:** `src/bot\hft\market_making_engine.py`



### Methods



### Attributes



---

## src.bot.hft.market_making_engine.SpreadMetrics

Spread performance metrics.

**File:** `src/bot\hft\market_making_engine.py`



### Methods



### Attributes



---

## src.bot.hft.market_making_engine.MarketMakingEngine

Advanced market making engine with multiple strategies.

**File:** `src/bot\hft\market_making_engine.py`



### Methods


#### __init__()

Method description here.


#### _should_update_quote()

Method description here.


#### get_market_making_summary()

Method description here.



### Attributes



---

## src.bot.hft.market_making_engine.SpreadManager

Manages spread calculations and optimizations.

**File:** `src/bot\hft\market_making_engine.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.market_making_engine.InventoryManager

Manages inventory positions and risk.

**File:** `src/bot\hft\market_making_engine.py`



### Methods


#### __init__()

Method description here.


#### check_inventory_limits()

Method description here.



### Attributes



---

## src.bot.hft.market_making_engine.QuoteManager

Manages quote lifecycle and execution.

**File:** `src/bot\hft\market_making_engine.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.order_flow_analyzer.FlowDirection

Order flow direction.

**File:** `src/bot\hft\order_flow_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BULLISH`

- `BEARISH`

- `NEUTRAL`

- `AGGRESSIVE_BUY`

- `AGGRESSIVE_SELL`


---

## src.bot.hft.order_flow_analyzer.OrderType

Order types for flow analysis.

**File:** `src/bot\hft\order_flow_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MARKET_BUY`

- `MARKET_SELL`

- `LIMIT_BUY`

- `LIMIT_SELL`

- `ICEBERG`

- `HIDDEN`


---

## src.bot.hft.order_flow_analyzer.VolumeProfile

Volume profile types.

**File:** `src/bot\hft\order_flow_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `POINT_OF_CONTROL`

- `VALUE_AREA_HIGH`

- `VALUE_AREA_LOW`

- `VOLUME_WEIGHTED_AVERAGE`


---

## src.bot.hft.order_flow_analyzer.MarketRegime

Market regime identification.

**File:** `src/bot\hft\order_flow_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRENDING_UP`

- `TRENDING_DOWN`

- `RANGE_BOUND`

- `HIGH_VOLATILITY`

- `LOW_VOLATILITY`

- `BREAKOUT`

- `REVERSAL`


---

## src.bot.hft.order_flow_analyzer.Trade

Individual trade data.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.hft.order_flow_analyzer.OrderBookLevel

Order book level data.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods



### Attributes



---

## src.bot.hft.order_flow_analyzer.OrderBookSnapshot

Complete order book snapshot.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.hft.order_flow_analyzer.FlowMetrics

Order flow metrics.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods



### Attributes



---

## src.bot.hft.order_flow_analyzer.VolumeProfileData

Volume profile analysis data.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods



### Attributes



---

## src.bot.hft.order_flow_analyzer.MarketMicrostructure

Market microstructure analysis.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods



### Attributes



---

## src.bot.hft.order_flow_analyzer.OrderFlowAnalyzer

Advanced order flow analysis engine.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods


#### __init__()

Method description here.


#### _determine_flow_direction()

Method description here.


#### _calculate_flow_strength()

Method description here.


#### get_flow_analysis()

Method description here.


#### get_analysis_summary()

Method description here.



### Attributes



---

## src.bot.hft.order_flow_analyzer.TickAggregator

Aggregates tick data for analysis.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.order_flow_analyzer.FlowDetector

Detects order flow patterns.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.order_flow_analyzer.RegimeAnalyzer

Analyzes market regimes.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.hft.order_flow_analyzer.LiquidityAnalyzer

Analyzes liquidity conditions.

**File:** `src/bot\hft\order_flow_analyzer.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.indicators.sentiment_enhanced.SentimentStrength

Sentiment strength levels.

**File:** `src/bot\indicators\sentiment_enhanced.py`


**Inherits from:** Enum


### Methods



### Attributes


- `EXTREMELY_BEARISH`

- `BEARISH`

- `NEUTRAL`

- `BULLISH`

- `EXTREMELY_BULLISH`


---

## src.bot.indicators.sentiment_enhanced.SignalStrength

Signal strength levels.

**File:** `src/bot\indicators\sentiment_enhanced.py`


**Inherits from:** Enum


### Methods



### Attributes


- `VERY_WEAK`

- `WEAK`

- `MODERATE`

- `STRONG`

- `VERY_STRONG`


---

## src.bot.indicators.sentiment_enhanced.SentimentIndicatorResult

Result from a sentiment-enhanced indicator.

**File:** `src/bot\indicators\sentiment_enhanced.py`



### Methods



### Attributes



---

## src.bot.indicators.sentiment_enhanced.TechnicalSignal

Technical analysis signal with sentiment enhancement.

**File:** `src/bot\indicators\sentiment_enhanced.py`



### Methods



### Attributes



---

## src.bot.indicators.sentiment_enhanced.SentimentEnhancedIndicators

Comprehensive sentiment-enhanced technical indicators system.

Combines traditional technical analysis with sentiment data to provide
more accurate and timely trading signals that account for market psychology
and news-driven price movements.

**File:** `src/bot\indicators\sentiment_enhanced.py`



### Methods


#### __init__()

Method description here.


#### update_data()

Method description here.


#### calculate_all_indicators()

Method description here.


#### _calculate_sentiment_weighted_ma()

Method description here.


#### _calculate_sentiment_enhanced_rsi()

Method description here.


#### _calculate_news_momentum_macd()

Method description here.


#### _calculate_fear_greed_bollinger()

Method description here.


#### _calculate_sentiment_trend_strength()

Method description here.


#### _calculate_social_volume_price_trend()

Method description here.


#### _calculate_sentiment_divergence()

Method description here.


#### _get_current_sentiment()

Method description here.


#### _calculate_sentiment_weights()

Method description here.


#### _calculate_news_momentum()

Method description here.


#### _get_fear_greed_adjustment()

Method description here.


#### _calculate_sentiment_trend()

Method description here.


#### _calculate_social_volume_factor()

Method description here.


#### _calculate_sentiment_momentum()

Method description here.


#### _calculate_indicator_confidence()

Method description here.


#### generate_trading_signals()

Method description here.


#### _interpret_indicator_signal()

Method description here.


#### _create_consensus_signal()

Method description here.


#### get_indicator_value()

Method description here.


#### get_active_signals()

Method description here.


#### get_signal_history()

Method description here.


#### get_indicator_summary()

Method description here.



### Attributes



---

## src.bot.integration.ml_execution_optimizer.ExecutionStrategy

Execution strategy types

**File:** `src/bot\integration\ml_execution_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `IMMEDIATE`

- `PASSIVE`

- `AGGRESSIVE`

- `ICEBERG`

- `TWAP`

- `VWAP`

- `ML_OPTIMAL`


---

## src.bot.integration.ml_execution_optimizer.LiquidityCondition

Market liquidity conditions

**File:** `src/bot\integration\ml_execution_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HIGH`

- `MEDIUM`

- `LOW`

- `VERY_LOW`


---

## src.bot.integration.ml_execution_optimizer.MarketImpactLevel

Expected market impact levels

**File:** `src/bot\integration\ml_execution_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MINIMAL`

- `LOW`

- `MODERATE`

- `HIGH`

- `SEVERE`


---

## src.bot.integration.ml_execution_optimizer.ExecutionPrediction

ML prediction for execution optimization

**File:** `src/bot\integration\ml_execution_optimizer.py`



### Methods



### Attributes



---

## src.bot.integration.ml_execution_optimizer.OrderSlice

Individual slice of a larger order

**File:** `src/bot\integration\ml_execution_optimizer.py`



### Methods



### Attributes



---

## src.bot.integration.ml_execution_optimizer.ExecutionPlan

Complete execution plan for an order

**File:** `src/bot\integration\ml_execution_optimizer.py`



### Methods



### Attributes



---

## src.bot.integration.ml_execution_optimizer.ExecutionPerformance

Execution performance metrics

**File:** `src/bot\integration\ml_execution_optimizer.py`



### Methods



### Attributes



---

## src.bot.integration.ml_execution_optimizer.MLExecutionOptimizer

ML-Enhanced Order Execution Optimizer

Uses machine learning to optimize trade execution quality and minimize costs

**File:** `src/bot\integration\ml_execution_optimizer.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _initialize_performance_tracking()

Method description here.


#### _classify_liquidity()

Method description here.


#### _estimate_market_impact_factor()

Method description here.


#### _prepare_execution_features()

Method description here.


#### _calculate_time_to_market_close()

Method description here.


#### _create_baseline_predictions()

Method description here.


#### _select_execution_strategy()

Method description here.


#### _calculate_execution_costs()

Method description here.


#### _estimate_completion_time()

Method description here.


#### _calculate_execution_risks()

Method description here.


#### _impact_level_to_float()

Method description here.


#### record_execution_performance()

Method description here.


#### get_execution_analytics()

Method description here.



### Attributes



---

## src.bot.integration.ml_execution_optimizer.MockExecutionModel

Mock ML model for execution predictions

**File:** `src/bot\integration\ml_execution_optimizer.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.integration.ml_feature_pipeline.MLSignalType

Types of ML trading signals

**File:** `src/bot\integration\ml_feature_pipeline.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BUY`

- `SELL`

- `HOLD`

- `STRONG_BUY`

- `STRONG_SELL`


---

## src.bot.integration.ml_feature_pipeline.MLStrategyType

Types of ML-enhanced strategies

**File:** `src/bot\integration\ml_feature_pipeline.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TREND_FOLLOWING`

- `MEAN_REVERSION`

- `VOLATILITY_BREAKOUT`

- `REGIME_SWITCHING`

- `ARBITRAGE`

- `MOMENTUM`


---

## src.bot.integration.ml_feature_pipeline.MLConfidenceLevel

ML prediction confidence levels

**File:** `src/bot\integration\ml_feature_pipeline.py`


**Inherits from:** Enum


### Methods



### Attributes


- `VERY_LOW`

- `LOW`

- `MODERATE`

- `HIGH`

- `VERY_HIGH`


---

## src.bot.integration.ml_feature_pipeline.MLFeatures

Unified ML features from all systems

**File:** `src/bot\integration\ml_feature_pipeline.py`



### Methods



### Attributes



---

## src.bot.integration.ml_feature_pipeline.MLPrediction

ML model prediction with metadata

**File:** `src/bot\integration\ml_feature_pipeline.py`



### Methods



### Attributes



---

## src.bot.integration.ml_feature_pipeline.MLTradingDecision

ML-enhanced trading decision

**File:** `src/bot\integration\ml_feature_pipeline.py`



### Methods



### Attributes



---

## src.bot.integration.ml_feature_pipeline.MLFeaturePipeline

Unified feature engineering pipeline combining capabilities from
both ml/ and machine_learning/ packages

**File:** `src/bot\integration\ml_feature_pipeline.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _initialize_components()

Method description here.


#### _calculate_correlation()

Method description here.


#### _count_features()

Method description here.


#### _calculate_orderbook_features()

Method description here.


#### _calculate_trade_flow_features()

Method description here.


#### _calculate_volume_profile_features()

Method description here.



### Attributes



---

## src.bot.integration.ml_integration_controller.SystemStatus

ML system status

**File:** `src/bot\integration\ml_integration_controller.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INITIALIZING`

- `ACTIVE`

- `DEGRADED`

- `MAINTENANCE`

- `ERROR`


---

## src.bot.integration.ml_integration_controller.TradingMode

Trading mode options

**File:** `src/bot\integration\ml_integration_controller.py`


**Inherits from:** Enum


### Methods



### Attributes


- `FULLY_AUTOMATED`

- `SEMI_AUTOMATED`

- `ADVISORY_ONLY`

- `BACKTESTING`

- `PAPER_TRADING`


---

## src.bot.integration.ml_integration_controller.TradingDecision

Complete trading decision from ML system

**File:** `src/bot\integration\ml_integration_controller.py`



### Methods



### Attributes



---

## src.bot.integration.ml_integration_controller.SystemHealthMetrics

Overall system health metrics

**File:** `src/bot\integration\ml_integration_controller.py`



### Methods



### Attributes



---

## src.bot.integration.ml_integration_controller.MLIntegrationController

Main ML Integration Controller

Coordinates all ML components to provide unified ML-enhanced trading

**File:** `src/bot\integration\ml_integration_controller.py`



### Methods


#### __init__()

Method description here.


#### _load_config()

Method description here.


#### _load_unified_config()

Method description here.


#### _get_default_config()

Method description here.


#### _calculate_overall_confidence()

Method description here.


#### _calculate_expected_outcome()

Method description here.


#### _determine_recommended_action()

Method description here.


#### _create_error_decision()

Method description here.


#### get_system_status()

Method description here.



### Attributes



---

## src.bot.integration.ml_model_manager.ModelType

Types of ML models supported

**File:** `src/bot\integration\ml_model_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LIGHTGBM`

- `XGBOOST`

- `NEURAL_NETWORK`

- `ENSEMBLE`

- `REGIME_AWARE`


---

## src.bot.integration.ml_model_manager.ModelStatus

Status of ML models

**File:** `src/bot\integration\ml_model_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ACTIVE`

- `INACTIVE`

- `TRAINING`

- `ERROR`

- `MAINTENANCE`


---

## src.bot.integration.ml_model_manager.ModelMetadata

Metadata for ML models

**File:** `src/bot\integration\ml_model_manager.py`



### Methods



### Attributes



---

## src.bot.integration.ml_model_manager.EnsemblePrediction

Ensemble prediction combining multiple models

**File:** `src/bot\integration\ml_model_manager.py`



### Methods



### Attributes



---

## src.bot.integration.ml_model_manager.MLModelManager

Unified ML Model Manager for trading predictions

Coordinates multiple ML models and provides unified prediction interface

**File:** `src/bot\integration\ml_model_manager.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _convert_features_for_model()

Method description here.


#### _convert_raw_prediction()

Method description here.


#### _extract_signal_type()

Method description here.


#### _numeric_to_signal_type()

Method description here.


#### _signal_to_numeric()

Method description here.


#### _calculate_consensus_strength()

Method description here.


#### _calculate_uncertainty()

Method description here.


#### _combine_feature_importance()

Method description here.


#### _create_empty_ensemble_prediction()

Method description here.


#### get_model_status()

Method description here.



### Attributes



---

## src.bot.integration.ml_performance_monitor.PerformanceMetric

Performance metric types

**File:** `src/bot\integration\ml_performance_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PREDICTION_ACCURACY`

- `RETURN_ATTRIBUTION`

- `SHARPE_RATIO`

- `SORTINO_RATIO`

- `MAX_DRAWDOWN`

- `WIN_RATE`

- `PROFIT_FACTOR`

- `CALMAR_RATIO`


---

## src.bot.integration.ml_performance_monitor.AlertLevel

Performance alert levels

**File:** `src/bot\integration\ml_performance_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `CRITICAL`


---

## src.bot.integration.ml_performance_monitor.PredictionOutcome

Outcome of a prediction for performance tracking

**File:** `src/bot\integration\ml_performance_monitor.py`



### Methods



### Attributes



---

## src.bot.integration.ml_performance_monitor.TradePerformance

Performance metrics for a completed trade

**File:** `src/bot\integration\ml_performance_monitor.py`



### Methods



### Attributes



---

## src.bot.integration.ml_performance_monitor.ModelPerformanceMetrics

Comprehensive performance metrics for an ML model

**File:** `src/bot\integration\ml_performance_monitor.py`



### Methods



### Attributes



---

## src.bot.integration.ml_performance_monitor.PerformanceAlert

Performance monitoring alert

**File:** `src/bot\integration\ml_performance_monitor.py`



### Methods



### Attributes



---

## src.bot.integration.ml_performance_monitor.PerformanceReport

Comprehensive performance report

**File:** `src/bot\integration\ml_performance_monitor.py`



### Methods



### Attributes



---

## src.bot.integration.ml_performance_monitor.MLPerformanceMonitor

Comprehensive ML Performance Monitor

Tracks and analyzes performance of ML-enhanced trading strategies

**File:** `src/bot\integration\ml_performance_monitor.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _initialize_model_metrics()

Method description here.


#### record_prediction_outcome()

Method description here.


#### _price_change_to_signal()

Method description here.


#### _calculate_prediction_accuracy()

Method description here.


#### _signal_to_direction()

Method description here.


#### record_trade_performance()

Method description here.


#### _calculate_sharpe_ratio()

Method description here.


#### _calculate_max_drawdown()

Method description here.


#### _cleanup_old_alerts()

Method description here.


#### _generate_recommendations()

Method description here.


#### get_real_time_metrics()

Method description here.


#### get_active_alerts()

Method description here.



### Attributes



---

## src.bot.integration.ml_strategy_orchestrator.StrategyType

Types of trading strategies

**File:** `src/bot\integration\ml_strategy_orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ML_ONLY`

- `TRADITIONAL_ONLY`

- `ML_TRADITIONAL_COMBINED`

- `ML_ENHANCED_TRADITIONAL`

- `REGIME_ADAPTIVE`


---

## src.bot.integration.ml_strategy_orchestrator.MarketRegime

Market regime types

**File:** `src/bot\integration\ml_strategy_orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRENDING_UP`

- `TRENDING_DOWN`

- `SIDEWAYS`

- `HIGH_VOLATILITY`

- `LOW_VOLATILITY`

- `UNCERTAIN`


---

## src.bot.integration.ml_strategy_orchestrator.SignalStrength

Signal strength levels

**File:** `src/bot\integration\ml_strategy_orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `VERY_WEAK`

- `WEAK`

- `MODERATE`

- `STRONG`

- `VERY_STRONG`


---

## src.bot.integration.ml_strategy_orchestrator.TraditionalSignal

Traditional strategy signal

**File:** `src/bot\integration\ml_strategy_orchestrator.py`



### Methods



### Attributes



---

## src.bot.integration.ml_strategy_orchestrator.CombinedSignal

Combined ML and traditional signal

**File:** `src/bot\integration\ml_strategy_orchestrator.py`



### Methods



### Attributes



---

## src.bot.integration.ml_strategy_orchestrator.StrategyPerformance

Strategy performance metrics

**File:** `src/bot\integration\ml_strategy_orchestrator.py`



### Methods



### Attributes



---

## src.bot.integration.ml_strategy_orchestrator.MLStrategyOrchestrator

Unified ML Strategy Orchestrator

Combines ML predictions with traditional strategies for optimal trading decisions

**File:** `src/bot\integration\ml_strategy_orchestrator.py`



### Methods


#### __init__()

Method description here.


#### _get_default_config()

Method description here.


#### _initialize_mock_strategies()

Method description here.


#### _initialize_performance_tracking()

Method description here.


#### _prepare_traditional_strategy_data()

Method description here.


#### _get_regime_weights()

Method description here.


#### _combine_signals()

Method description here.


#### update_strategy_performance()

Method description here.


#### get_strategy_performance_summary()

Method description here.


#### get_current_market_regime()

Method description here.



### Attributes



---

## src.bot.integration.ml_strategy_orchestrator.MockStrategy

No description available.

**File:** `src/bot\integration\ml_strategy_orchestrator.py`



### Methods


#### __init__()

Method description here.


#### generate_signal()

Method description here.



### Attributes



---

## src.bot.live_trading.alert_system.AlertType

Types of alerts in the system.

**File:** `src/bot\live_trading\alert_system.py`


**Inherits from:** Enum


### Methods



### Attributes


- `RISK_LIMIT_BREACH`

- `POSITION_LOSS`

- `STRATEGY_PERFORMANCE`

- `EXECUTION_ERROR`

- `SYSTEM_ERROR`

- `CONNECTION_LOSS`

- `BALANCE_LOW`

- `DRAWDOWN_LIMIT`

- `UNUSUAL_ACTIVITY`

- `HEALTH_CHECK`


---

## src.bot.live_trading.alert_system.AlertSeverity

Alert severity levels.

**File:** `src/bot\live_trading\alert_system.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `ERROR`

- `CRITICAL`


---

## src.bot.live_trading.alert_system.AlertStatus

Alert status states.

**File:** `src/bot\live_trading\alert_system.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ACTIVE`

- `ACKNOWLEDGED`

- `RESOLVED`

- `SUPPRESSED`


---

## src.bot.live_trading.alert_system.NotificationChannel

Available notification channels.

**File:** `src/bot\live_trading\alert_system.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DASHBOARD`

- `EMAIL`

- `WEBHOOK`

- `SMS`

- `CONSOLE`


---

## src.bot.live_trading.alert_system.AlertRule

Alert rule configuration.

**File:** `src/bot\live_trading\alert_system.py`



### Methods



### Attributes



---

## src.bot.live_trading.alert_system.Alert

Alert instance.

**File:** `src/bot\live_trading\alert_system.py`



### Methods


#### acknowledge()

Method description here.


#### resolve()

Method description here.



### Attributes



---

## src.bot.live_trading.alert_system.NotificationConfig

Notification channel configuration.

**File:** `src/bot\live_trading\alert_system.py`



### Methods



### Attributes



---

## src.bot.live_trading.alert_system.AlertSystem

Comprehensive alert system for trading bot monitoring.

Features:
- Configurable alert rules with conditions and thresholds
- Multiple severity levels and notification channels
- Alert escalation and acknowledgment workflows
- Rate limiting and cooldown periods
- Historical alert tracking and analytics
- Integration with all system components

**File:** `src/bot\live_trading\alert_system.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_alert_rules()

Method description here.


#### _setup_notification_channels()

Method description here.


#### _format_alert_message()

Method description here.


#### get_active_alerts()

Method description here.


#### get_alert_statistics()

Method description here.



### Attributes



---

## src.bot.live_trading.live_execution_engine.TradingMode

Trading execution modes.

**File:** `src/bot\live_trading\live_execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PAPER`

- `LIVE`

- `HYBRID`


---

## src.bot.live_trading.live_execution_engine.ExecutionStatus

Execution status for orders and trades.

**File:** `src/bot\live_trading\live_execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `EXECUTING`

- `EXECUTED`

- `PARTIALLY_EXECUTED`

- `FAILED`

- `CANCELLED`


---

## src.bot.live_trading.live_execution_engine.PositionSyncStatus

Position synchronization status.

**File:** `src/bot\live_trading\live_execution_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `IN_SYNC`

- `OUT_OF_SYNC`

- `SYNCING`

- `SYNC_ERROR`


---

## src.bot.live_trading.live_execution_engine.ExecutionResult

Result of trade execution.

**File:** `src/bot\live_trading\live_execution_engine.py`



### Methods


#### fill_ratio()

Method description here.


#### slippage_bps()

Method description here.



### Attributes



---

## src.bot.live_trading.live_execution_engine.VirtualPosition

Virtual position for paper trading.

**File:** `src/bot\live_trading\live_execution_engine.py`



### Methods


#### update_price()

Method description here.



### Attributes



---

## src.bot.live_trading.live_execution_engine.ExecutionMetrics

Execution performance metrics.

**File:** `src/bot\live_trading\live_execution_engine.py`



### Methods


#### success_rate()

Method description here.


#### update()

Method description here.



### Attributes



---

## src.bot.live_trading.live_execution_engine.LiveExecutionEngine

Comprehensive live trading execution engine.

Features:
- Multi-mode execution: paper, live, and hybrid trading
- Real-time position tracking and synchronization
- Execution quality monitoring and optimization
- Risk management integration with pre-trade checks
- Strategy graduation from paper to live trading
- Comprehensive execution analytics and reporting

**File:** `src/bot\live_trading\live_execution_engine.py`



### Methods


#### __init__()

Method description here.


#### _setup_websocket_handlers()

Method description here.


#### _handle_market_data()

Method description here.


#### _handle_execution_update()

Method description here.


#### _handle_position_update()

Method description here.


#### get_execution_metrics()

Method description here.



### Attributes



---

## src.bot.live_trading.monitoring_dashboard.MetricType

Types of performance metrics.

**File:** `src/bot\live_trading\monitoring_dashboard.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PORTFOLIO`

- `STRATEGY`

- `EXECUTION`

- `RISK`

- `SYSTEM`


---

## src.bot.live_trading.monitoring_dashboard.AlertSeverity

Alert severity levels.

**File:** `src/bot\live_trading\monitoring_dashboard.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `ERROR`

- `CRITICAL`


---

## src.bot.live_trading.monitoring_dashboard.PerformanceMetrics

Comprehensive performance metrics.

**File:** `src/bot\live_trading\monitoring_dashboard.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.live_trading.monitoring_dashboard.SystemHealth

System health indicators.

**File:** `src/bot\live_trading\monitoring_dashboard.py`



### Methods



### Attributes



---

## src.bot.live_trading.monitoring_dashboard.DashboardAlert

Dashboard alert message.

**File:** `src/bot\live_trading\monitoring_dashboard.py`



### Methods



### Attributes



---

## src.bot.live_trading.monitoring_dashboard.MonitoringDashboard

Real-time performance monitoring dashboard.

Features:
- Real-time metrics collection and aggregation
- Web-based dashboard with live updates via WebSocket
- Configurable alerts and notifications
- Historical data storage and analysis
- Performance analytics and reporting
- System health monitoring and diagnostics

**File:** `src/bot\live_trading\monitoring_dashboard.py`



### Methods


#### __init__()

Method description here.


#### _setup_routes()

Method description here.


#### _get_dashboard_html()

Method description here.



### Attributes



---

## src.bot.live_trading.phase5_orchestrator.Phase5LiveTradingOrchestrator

Main orchestrator for Phase 5 live trading operations.

This class coordinates all live trading components and provides
a unified interface for starting, stopping, and managing the
live trading system.

Features:
- Unified component lifecycle management
- Graceful startup and shutdown sequences
- Error handling and recovery
- Health monitoring and reporting
- Configuration management
- Service orchestration

**File:** `src/bot\live_trading\phase5_orchestrator.py`



### Methods


#### __init__()

Method description here.


#### _setup_signal_handlers()

Method description here.



### Attributes



---

## src.bot.live_trading.production_deployment.DeploymentEnvironment

Deployment environment types.

**File:** `src/bot\live_trading\production_deployment.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DEVELOPMENT`

- `STAGING`

- `PRODUCTION`

- `TESTING`


---

## src.bot.live_trading.production_deployment.DeploymentStatus

Deployment status states.

**File:** `src/bot\live_trading\production_deployment.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `IN_PROGRESS`

- `COMPLETED`

- `FAILED`

- `ROLLED_BACK`


---

## src.bot.live_trading.production_deployment.ServiceStatus

Service status states.

**File:** `src/bot\live_trading\production_deployment.py`


**Inherits from:** Enum


### Methods



### Attributes


- `STOPPED`

- `STARTING`

- `RUNNING`

- `STOPPING`

- `ERROR`


---

## src.bot.live_trading.production_deployment.DeploymentConfig

Deployment configuration.

**File:** `src/bot\live_trading\production_deployment.py`



### Methods



### Attributes



---

## src.bot.live_trading.production_deployment.ServiceConfig

Service configuration.

**File:** `src/bot\live_trading\production_deployment.py`



### Methods



### Attributes



---

## src.bot.live_trading.production_deployment.DeploymentRecord

Deployment record.

**File:** `src/bot\live_trading\production_deployment.py`



### Methods



### Attributes



---

## src.bot.live_trading.production_deployment.ProductionDeploymentPipeline

Production deployment pipeline for trading bot.

Features:
- Automated deployment with validation
- Blue-green deployment strategy
- Configuration management per environment
- Service orchestration and monitoring
- Health checks and rollback capabilities
- Docker container support
- CI/CD integration hooks

**File:** `src/bot\live_trading\production_deployment.py`



### Methods


#### __init__()

Method description here.


#### _create_directories()

Method description here.


#### _load_service_configs()

Method description here.


#### _get_pre_deploy_checks()

Method description here.


#### _get_post_deploy_checks()

Method description here.


#### _get_service_start_order()

Method description here.


#### _find_deployment_by_version()

Method description here.


#### _get_last_successful_deployment()

Method description here.



### Attributes



---

## src.bot.live_trading.websocket_manager.WebSocketStreamType

WebSocket stream types.

**File:** `src/bot\live_trading\websocket_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PUBLIC`

- `PRIVATE`


---

## src.bot.live_trading.websocket_manager.ConnectionStatus

WebSocket connection status.

**File:** `src/bot\live_trading\websocket_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DISCONNECTED`

- `CONNECTING`

- `CONNECTED`

- `RECONNECTING`

- `ERROR`


---

## src.bot.live_trading.websocket_manager.WebSocketConfig

WebSocket configuration settings.

**File:** `src/bot\live_trading\websocket_manager.py`



### Methods



### Attributes



---

## src.bot.live_trading.websocket_manager.WebSocketMessage

Structured WebSocket message.

**File:** `src/bot\live_trading\websocket_manager.py`



### Methods



### Attributes



---

## src.bot.live_trading.websocket_manager.ConnectionMetrics

WebSocket connection performance metrics.

**File:** `src/bot\live_trading\websocket_manager.py`



### Methods



### Attributes



---

## src.bot.live_trading.websocket_manager.WebSocketManager

Comprehensive WebSocket manager for Bybit real-time data streams.

Features:
- Automatic connection management with reconnection logic
- Support for both public and private streams
- Message validation and error handling
- Performance metrics and health monitoring
- Subscription management for multiple symbols/topics
- Event-driven message processing with callbacks

**File:** `src/bot\live_trading\websocket_manager.py`



### Methods


#### __init__()

Method description here.


#### add_message_handler()

Method description here.


#### add_error_handler()

Method description here.


#### _create_websocket_message()

Method description here.


#### _topic_matches_pattern()

Method description here.


#### get_connection_status()

Method description here.



### Attributes



---

## src.bot.machine_learning.feature_engineering.FeatureType

Types of features.

**File:** `src/bot\machine_learning\feature_engineering.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TECHNICAL`

- `MICROSTRUCTURE`

- `SENTIMENT`

- `MACRO`

- `CROSS_ASSET`

- `CUSTOM`


---

## src.bot.machine_learning.feature_engineering.FeatureSelectionMethod

Feature selection methods.

**File:** `src/bot\machine_learning\feature_engineering.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CORRELATION`

- `MUTUAL_INFO`

- `F_TEST`

- `RFE`

- `RFECV`

- `TREE_IMPORTANCE`

- `PCA`

- `ICA`


---

## src.bot.machine_learning.feature_engineering.FeatureSet

Feature set definition.

**File:** `src/bot\machine_learning\feature_engineering.py`



### Methods



### Attributes



---

## src.bot.machine_learning.feature_engineering.FeatureImportance

Feature importance scores.

**File:** `src/bot\machine_learning\feature_engineering.py`



### Methods



### Attributes



---

## src.bot.machine_learning.feature_engineering.FeatureEngineering

Advanced Feature Engineering for Trading ML Models.

This class provides comprehensive feature engineering capabilities
including technical indicators, market microstructure features,
sentiment analysis, and cross-asset features.

Features:
- 200+ technical indicators using TA-Lib
- Market microstructure features from order book data
- Cross-asset correlation and spread features
- Sentiment features from news and social media
- Macro economic features
- Feature selection and dimensionality reduction
- Real-time feature computation

**File:** `src/bot\machine_learning\feature_engineering.py`



### Methods


#### __init__()

Method description here.


#### _remove_correlated_features()

Method description here.


#### _initialize_default_feature_sets()

Method description here.


#### _is_cached()

Method description here.


#### _cache_result()

Method description here.



### Attributes



---

## src.bot.machine_learning.ml_engine.ModelType

Types of ML models supported.

**File:** `src/bot\machine_learning\ml_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LSTM`

- `TRANSFORMER`

- `RANDOM_FOREST`

- `XGBOOST`

- `LINEAR_REGRESSION`

- `RIDGE_REGRESSION`

- `GRADIENT_BOOSTING`

- `ENSEMBLE`


---

## src.bot.machine_learning.ml_engine.PredictionType

Types of predictions.

**File:** `src/bot\machine_learning\ml_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PRICE`

- `DIRECTION`

- `VOLATILITY`

- `RETURN`

- `REGIME`

- `RISK`


---

## src.bot.machine_learning.ml_engine.ModelStatus

Model status states.

**File:** `src/bot\machine_learning\ml_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRAINING`

- `TRAINED`

- `DEPLOYED`

- `DEPRECATED`

- `ERROR`


---

## src.bot.machine_learning.ml_engine.PredictionResult

ML model prediction result.

**File:** `src/bot\machine_learning\ml_engine.py`



### Methods



### Attributes



---

## src.bot.machine_learning.ml_engine.ModelPerformance

Model performance metrics.

**File:** `src/bot\machine_learning\ml_engine.py`



### Methods



### Attributes



---

## src.bot.machine_learning.ml_engine.MLEngine

Core Machine Learning Engine for intelligent trading.

This class provides a unified interface for training, evaluating,
and deploying machine learning models for trading applications.

Features:
- Multi-framework support (TensorFlow, PyTorch, scikit-learn)
- Real-time prediction capabilities
- Online learning and model adaptation
- Performance monitoring and evaluation
- Model versioning and persistence
- Feature importance analysis

**File:** `src/bot\machine_learning\ml_engine.py`



### Methods


#### __init__()

Method description here.


#### _get_scaler()

Method description here.



### Attributes



---

## src.bot.machine_learning.model_manager.ModelStatus

Model lifecycle status.

**File:** `src/bot\machine_learning\model_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRAINING`

- `TRAINED`

- `DEPLOYED`

- `DEPRECATED`

- `FAILED`

- `ARCHIVED`


---

## src.bot.machine_learning.model_manager.ModelType

Types of models supported.

**File:** `src/bot\machine_learning\model_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LSTM`

- `TRANSFORMER`

- `RANDOM_FOREST`

- `XGBOOST`

- `LINEAR_REGRESSION`

- `SVM`

- `ENSEMBLE`


---

## src.bot.machine_learning.model_manager.ModelMetadata

Metadata for a trained model.

**File:** `src/bot\machine_learning\model_manager.py`



### Methods



### Attributes



---

## src.bot.machine_learning.model_manager.TrainingConfig

Configuration for model training.

**File:** `src/bot\machine_learning\model_manager.py`



### Methods



### Attributes



---

## src.bot.machine_learning.model_manager.ModelPerformance

Model performance tracking.

**File:** `src/bot\machine_learning\model_manager.py`



### Methods



### Attributes



---

## src.bot.machine_learning.model_manager.ModelManager

Advanced model lifecycle management system.

**File:** `src/bot\machine_learning\model_manager.py`



### Methods


#### __init__()

Method description here.


#### list_models()

Method description here.


#### get_deployed_models()

Method description here.


#### _load_model_registry()

Method description here.


#### _generate_model_id()

Method description here.


#### _calculate_data_hash()

Method description here.


#### _increment_version()

Method description here.


#### get_registry_summary()

Method description here.



### Attributes



---

## src.bot.machine_learning.prediction_engine.PredictionType

Types of predictions supported by the engine.

**File:** `src/bot\machine_learning\prediction_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PRICE`

- `DIRECTION`

- `VOLATILITY`

- `VOLUME`

- `SUPPORT_RESISTANCE`

- `TREND_STRENGTH`

- `MARKET_REGIME`


---

## src.bot.machine_learning.prediction_engine.PredictionResult

Result of a prediction operation.

**File:** `src/bot\machine_learning\prediction_engine.py`



### Methods



### Attributes



---

## src.bot.machine_learning.prediction_engine.EnsemblePrediction

Result of ensemble prediction combining multiple models.

**File:** `src/bot\machine_learning\prediction_engine.py`



### Methods



### Attributes



---

## src.bot.machine_learning.prediction_engine.PredictionEngine

Advanced prediction engine with ensemble methods and confidence scoring.

**File:** `src/bot\machine_learning\prediction_engine.py`



### Methods


#### __init__()

Method description here.


#### _calculate_ensemble_prediction()

Method description here.


#### _calculate_confidence()

Method description here.


#### _calculate_model_consensus()

Method description here.


#### _is_cached_valid()

Method description here.


#### get_performance_summary()

Method description here.



### Attributes



---

## src.bot.ml.ensemble.EnsembleWeight

Container for ensemble weights with metadata.

**File:** `src/bot\ml\ensemble.py`



### Methods



### Attributes



---

## src.bot.ml.ensemble.EnsembleResult

Container for ensemble prediction results.

**File:** `src/bot\ml\ensemble.py`



### Methods



### Attributes



---

## src.bot.ml.ensemble.DynamicEnsemble

Dynamic ensemble with adaptive weighting based on performance and market conditions.

This ensemble automatically adjusts model weights based on recent performance,
market regimes, and prediction confidence.

**File:** `src/bot\ml\ensemble.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### add_model()

Method description here.


#### fit()

Method description here.


#### predict()

Method description here.


#### _calculate_initial_weights()

Method description here.


#### _calculate_dynamic_weights()

Method description here.


#### _performance_based_weights()

Method description here.


#### _regime_aware_weights()

Method description here.


#### _exponential_decay_weights()

Method description here.


#### _apply_weight_constraints()

Method description here.


#### _combine_predictions()

Method description here.


#### _calculate_confidence()

Method description here.


#### _calculate_uncertainty()

Method description here.


#### _update_weights_online()

Method description here.


#### _calculate_ensemble_metrics()

Method description here.


#### get_model_performance_summary()

Method description here.


#### plot_weight_evolution()

Method description here.



### Attributes



---

## src.bot.ml.ensemble.StackingEnsemble

Stacking ensemble that uses a meta-model to combine base model predictions.

**File:** `src/bot\ml\ensemble.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### add_base_model()

Method description here.


#### fit()

Method description here.


#### predict()

Method description here.


#### predict_proba()

Method description here.


#### _generate_meta_features()

Method description here.


#### _get_meta_features_predict()

Method description here.



### Attributes



---

## src.bot.ml.ensemble_manager.ModelType

Types of ML models.

**File:** `src/bot\ml\ensemble_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LIGHTGBM`

- `XGBOOST`

- `CATBOOST`

- `RANDOM_FOREST`

- `EXTRA_TREES`

- `MLP`

- `RIDGE`

- `LASSO`

- `ELASTIC_NET`


---

## src.bot.ml.ensemble_manager.EnsembleStrategy

Ensemble combination strategies.

**File:** `src/bot\ml\ensemble_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SIMPLE_AVERAGE`

- `WEIGHTED_AVERAGE`

- `PERFORMANCE_WEIGHTED`

- `BAYESIAN_AVERAGE`

- `STACKING`

- `DYNAMIC_SELECTION`

- `REGIME_BASED`


---

## src.bot.ml.ensemble_manager.ModelStatus

Model status tracking.

**File:** `src/bot\ml\ensemble_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ACTIVE`

- `TRAINING`

- `VALIDATION`

- `INACTIVE`

- `FAILED`

- `DEPRECATED`


---

## src.bot.ml.ensemble_manager.ModelMetrics

Comprehensive model performance metrics.

**File:** `src/bot\ml\ensemble_manager.py`



### Methods



### Attributes



---

## src.bot.ml.ensemble_manager.EnsemblePrediction

Ensemble prediction with uncertainty quantification.

**File:** `src/bot\ml\ensemble_manager.py`



### Methods



### Attributes



---

## src.bot.ml.ensemble_manager.ModelConfiguration

Configuration for individual models.

**File:** `src/bot\ml\ensemble_manager.py`



### Methods



### Attributes



---

## src.bot.ml.ensemble_manager.BaseModelWrapper

Base wrapper for ML models with common interface.

**File:** `src/bot\ml\ensemble_manager.py`



### Methods


#### __init__()

Method description here.


#### _initialize_model()

Method description here.


#### _initialize_scaler()

Method description here.


#### fit()

Method description here.


#### predict()

Method description here.


#### get_feature_importance()

Method description here.


#### save_model()

Method description here.


#### load_model()

Method description here.



### Attributes



---

## src.bot.ml.ensemble_manager.EnsembleModelManager

Sophisticated ensemble learning system for cryptocurrency trading.

Manages multiple ML models with dynamic weighting, performance tracking,
and adaptive ensemble composition.

**File:** `src/bot\ml\ensemble_manager.py`



### Methods


#### __init__()

Method description here.


#### _initialize_default_models()

Method description here.


#### update_training_data()

Method description here.


#### add_model()

Method description here.


#### _remove_worst_model()

Method description here.


#### remove_model()

Method description here.


#### train_models()

Method description here.


#### _train_single_model()

Method description here.


#### _validate_models()

Method description here.


#### _update_ensemble_weights()

Method description here.


#### _normalize_ensemble_weights()

Method description here.


#### predict()

Method description here.


#### _combine_predictions()

Method description here.


#### _combine_predictions_weighted_average()

Method description here.


#### optimize_hyperparameters()

Method description here.


#### get_model_performance_summary()

Method description here.


#### save_ensemble()

Method description here.


#### load_ensemble()

Method description here.



### Attributes



---

## src.bot.ml.features.TechnicalIndicators

Technical indicator calculation with comprehensive TA-Lib integration.

This class provides a unified interface for calculating technical indicators
while ensuring proper handling of NaN values and forward-looking bias.

**File:** `src/bot\ml\features.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### calculate_momentum_features()

Method description here.


#### calculate_trend_features()

Method description here.


#### calculate_volatility_features()

Method description here.


#### calculate_volume_features()

Method description here.



### Attributes



---

## src.bot.ml.features.FeatureEngineer

Comprehensive feature engineering for trading strategies.

This class orchestrates the entire feature engineering pipeline,
including technical indicators, lagged features, transformations,
and feature selection.

**File:** `src/bot\ml\features.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### engineer_features()

Method description here.


#### _calculate_all_technical_features()

Method description here.


#### _create_lagged_features()

Method description here.


#### _calculate_rolling_statistics()

Method description here.


#### _create_regime_features()

Method description here.


#### _create_cross_asset_features()

Method description here.


#### _apply_transformations()

Method description here.


#### _handle_missing_values()

Method description here.


#### _remove_outliers()

Method description here.


#### _select_features()

Method description here.


#### _remove_correlated_features()

Method description here.


#### _scale_features()

Method description here.


#### transform_new_data()

Method description here.



### Attributes



---

## src.bot.ml.feature_selector.SelectionMethod

Feature selection methods.

**File:** `src/bot\ml\feature_selector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MUTUAL_INFO`

- `RFE`

- `STABILITY`

- `CORRELATION`

- `UNIVARIATE`

- `L1_REGULARIZATION`

- `TREE_IMPORTANCE`

- `PCA_ANALYSIS`

- `COMBINED`


---

## src.bot.ml.feature_selector.FeatureType

Types of features for categorization.

**File:** `src/bot\ml\feature_selector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TECHNICAL`

- `SENTIMENT`

- `CROSS_EXCHANGE`

- `TEMPORAL`

- `VOLATILITY`

- `MACRO`

- `DERIVED`

- `INTERACTION`


---

## src.bot.ml.feature_selector.MarketRegime

Market regimes for regime-aware selection.

**File:** `src/bot\ml\feature_selector.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BULL`

- `BEAR`

- `SIDEWAYS`

- `HIGH_VOLATILITY`

- `LOW_VOLATILITY`


---

## src.bot.ml.feature_selector.FeatureMetrics

Comprehensive feature quality metrics.

**File:** `src/bot\ml\feature_selector.py`



### Methods



### Attributes



---

## src.bot.ml.feature_selector.FeatureSelectionResult

Results from feature selection process.

**File:** `src/bot\ml\feature_selector.py`



### Methods



### Attributes



---

## src.bot.ml.feature_selector.RegimeFeatureSet

Feature set optimized for specific market regime.

**File:** `src/bot\ml\feature_selector.py`



### Methods



### Attributes



---

## src.bot.ml.feature_selector.AdvancedFeatureSelector

Advanced feature selection system for cryptocurrency trading.

Implements multiple sophisticated feature selection techniques and
provides intelligent feature engineering and redundancy removal.

**File:** `src/bot\ml\feature_selector.py`



### Methods


#### __init__()

Method description here.


#### _initialize_selection_models()

Method description here.


#### update_feature_data()

Method description here.


#### _update_feature_metadata()

Method description here.


#### _infer_feature_type()

Method description here.


#### _calculate_correlation_matrix()

Method description here.


#### _identify_correlation_clusters()

Method description here.


#### _update_feature_metrics()

Method description here.


#### select_features()

Method description here.


#### _apply_basic_filtering()

Method description here.


#### _mutual_info_selection()

Method description here.


#### _rfe_selection()

Method description here.


#### _stability_selection()

Method description here.


#### _correlation_selection()

Method description here.


#### _tree_importance_selection()

Method description here.


#### _l1_selection()

Method description here.


#### _combined_selection()

Method description here.


#### _run_single_method()

Method description here.


#### _calculate_cv_performance()

Method description here.


#### _calculate_selection_stability()

Method description here.


#### _count_redundancy_removed()

Method description here.


#### _update_metrics_from_selection()

Method description here.


#### _calculate_selection_probability()

Method description here.


#### get_feature_metrics()

Method description here.


#### get_selection_history()

Method description here.


#### get_recommended_features()

Method description here.


#### analyze_feature_importance()

Method description here.



### Attributes



---

## src.bot.ml.models.ModelResult

Container for ML model results.

**File:** `src/bot\ml\models.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.ml.models.LightGBMTrader

LightGBM implementation optimized for financial data.

This class provides a LightGBM wrapper with financial-specific
optimizations and proper validation techniques.

**File:** `src/bot\ml\models.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### fit()

Method description here.


#### predict()

Method description here.


#### predict_proba()

Method description here.


#### get_feature_importance()

Method description here.


#### optimize_hyperparameters()

Method description here.


#### _default_param_grid()

Method description here.



### Attributes



---

## src.bot.ml.models.XGBoostTrader

XGBoost implementation optimized for financial data.

**File:** `src/bot\ml\models.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### fit()

Method description here.


#### predict()

Method description here.


#### predict_proba()

Method description here.


#### get_feature_importance()

Method description here.



### Attributes



---

## src.bot.ml.models.EnsembleTrader

Ensemble of multiple ML models for robust predictions.

**File:** `src/bot\ml\models.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### fit()

Method description here.


#### predict()

Method description here.


#### predict_proba()

Method description here.


#### _weighted_average_predictions()

Method description here.


#### _voting_predictions()

Method description here.


#### _optimize_ensemble_weights()

Method description here.


#### _get_cv_predictions()

Method description here.


#### _combine_cv_predictions()

Method description here.



### Attributes



---

## src.bot.ml.models.MLModelFactory

Factory for creating and managing ML models.

**File:** `src/bot\ml\models.py`



### Methods


#### __init__()

Method description here.


#### create_model()

Method description here.


#### train_and_validate_model()

Method description here.


#### _calculate_validation_scores()

Method description here.


#### compare_models()

Method description here.


#### save_model()

Method description here.


#### load_model()

Method description here.



### Attributes



---

## src.bot.ml.model_monitor.DriftType

Types of model drift.

**File:** `src/bot\ml\model_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DATA_DRIFT`

- `CONCEPT_DRIFT`

- `PREDICTION_DRIFT`

- `PERFORMANCE_DRIFT`


---

## src.bot.ml.model_monitor.AlertSeverity

Alert severity levels.

**File:** `src/bot\ml\model_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW`

- `MEDIUM`

- `HIGH`

- `CRITICAL`


---

## src.bot.ml.model_monitor.ModelStatus

Model lifecycle status.

**File:** `src/bot\ml\model_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ACTIVE`

- `CHAMPION`

- `CHALLENGER`

- `SHADOW`

- `DEPRECATED`

- `FAILED`

- `RETRAINING`


---

## src.bot.ml.model_monitor.DriftTestResult

Drift test results.

**File:** `src/bot\ml\model_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `NO_DRIFT`

- `DRIFT_DETECTED`

- `INCONCLUSIVE`


---

## src.bot.ml.model_monitor.DriftDetectionResult

Results from drift detection analysis.

**File:** `src/bot\ml\model_monitor.py`



### Methods



### Attributes



---

## src.bot.ml.model_monitor.PerformanceAlert

Model performance alert.

**File:** `src/bot\ml\model_monitor.py`



### Methods



### Attributes



---

## src.bot.ml.model_monitor.ModelHealthScore

Comprehensive model health assessment.

**File:** `src/bot\ml\model_monitor.py`



### Methods



### Attributes



---

## src.bot.ml.model_monitor.ABTestResult

A/B test comparison results.

**File:** `src/bot\ml\model_monitor.py`



### Methods



### Attributes



---

## src.bot.ml.model_monitor.ModelPerformanceMonitor

Comprehensive model monitoring system for cryptocurrency trading.

Provides real-time performance monitoring, drift detection, alerting,
and automated model lifecycle management.

**File:** `src/bot\ml\model_monitor.py`



### Methods


#### __init__()

Method description here.


#### register_model()

Method description here.


#### log_prediction()

Method description here.


#### calculate_performance_metrics()

Method description here.


#### detect_data_drift()

Method description here.


#### _ks_drift_test()

Method description here.


#### _psi_drift_test()

Method description here.


#### _js_drift_test()

Method description here.


#### _spc_drift_test()

Method description here.


#### check_performance_degradation()

Method description here.


#### calculate_model_health_score()

Method description here.


#### should_retrain_model()

Method description here.


#### start_ab_test()

Method description here.


#### evaluate_ab_test()

Method description here.


#### get_monitoring_summary()

Method description here.


#### start_monitoring()

Method description here.


#### stop_monitoring()

Method description here.


#### _monitoring_loop()

Method description here.



### Attributes



---

## src.bot.ml.regimes.RegimeInfo

Information about a market regime.

**File:** `src/bot\ml\regimes.py`



### Methods



### Attributes



---

## src.bot.ml.regimes.HMMRegimeDetector

Hidden Markov Model based regime detection.

Uses HMM to identify latent market states based on observable
market features like returns, volatility, and volume.

**File:** `src/bot\ml\regimes.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### fit()

Method description here.


#### predict_regimes()

Method description here.


#### _prepare_features()

Method description here.


#### _calculate_rsi()

Method description here.


#### _calculate_macd()

Method description here.


#### _analyze_regime_characteristics()

Method description here.


#### _calculate_max_drawdown()

Method description here.


#### _calculate_avg_duration()

Method description here.


#### _classify_regime()

Method description here.


#### _describe_regime()

Method description here.



### Attributes



---

## src.bot.ml.regimes.VolatilityRegimeDetector

Volatility-based regime detection using threshold or clustering methods.

**File:** `src/bot\ml\regimes.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### fit()

Method description here.


#### predict_regimes()

Method description here.


#### _classify_volatility()

Method description here.


#### _analyze_volatility_regimes()

Method description here.



### Attributes



---

## src.bot.ml.regimes.TrendRegimeDetector

Trend-based regime detection using moving averages and trend indicators.

**File:** `src/bot\ml\regimes.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### detect_regimes()

Method description here.


#### _apply_min_duration_filter()

Method description here.



### Attributes



---

## src.bot.ml.regimes.MultiFactorRegimeDetector

Multi-factor regime detection combining multiple indicators.

**File:** `src/bot\ml\regimes.py`



### Methods


#### __init__()

Method description here.


#### _default_config()

Method description here.


#### fit()

Method description here.


#### predict_regimes()

Method description here.


#### _weighted_average_combination()

Method description here.


#### _majority_vote_combination()

Method description here.



### Attributes



---

## src.bot.ml.regimes.RegimeAnalyzer

Comprehensive regime analysis and reporting.

**File:** `src/bot\ml\regimes.py`



### Methods


#### __init__()

Method description here.


#### analyze_regime_performance()

Method description here.


#### _calculate_max_drawdown()

Method description here.


#### generate_regime_report()

Method description here.



### Attributes



---

## src.bot.ml.time_series_forecaster.ForecastModel

Types of forecasting models.

**File:** `src/bot\ml\time_series_forecaster.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LSTM`

- `GRU`

- `TRANSFORMER`

- `CNN_LSTM`

- `ARIMA`

- `PROPHET`

- `EXPONENTIAL_SMOOTHING`

- `VAR`

- `LINEAR_TREND`

- `RANDOM_FOREST`


---

## src.bot.ml.time_series_forecaster.ForecastHorizon

Forecast horizons.

**File:** `src/bot\ml\time_series_forecaster.py`


**Inherits from:** Enum


### Methods



### Attributes


- `H1`

- `H4`

- `H24`

- `D7`

- `D30`


---

## src.bot.ml.time_series_forecaster.ForecastResult

Time series forecast result.

**File:** `src/bot\ml\time_series_forecaster.py`



### Methods



### Attributes



---

## src.bot.ml.time_series_forecaster.TimeSeriesFeatures

Time series feature engineering results.

**File:** `src/bot\ml\time_series_forecaster.py`



### Methods



### Attributes



---

## src.bot.ml.time_series_forecaster.TimeSeriesDataset

PyTorch Dataset for time series data.

**File:** `src/bot\ml\time_series_forecaster.py`


**Inherits from:** Dataset


### Methods


#### __init__()

Method description here.


#### __len__()

Method description here.


#### __getitem__()

Method description here.



### Attributes



---

## src.bot.ml.time_series_forecaster.TimeSeriesForecaster

Advanced time series forecasting system for cryptocurrency trading.

Implements multiple forecasting models with multi-horizon predictions,
uncertainty quantification, and regime-aware forecasting.

**File:** `src/bot\ml\time_series_forecaster.py`



### Methods


#### __init__()

Method description here.


#### _initialize_available_models()

Method description here.


#### update_data()

Method description here.


#### _engineer_time_series_features()

Method description here.


#### _initialize_scalers()

Method description here.


#### _prepare_sequences()

Method description here.


#### _horizon_to_steps()

Method description here.


#### train_model()

Method description here.


#### _train_deep_learning_model()

Method description here.


#### _train_arima_model()

Method description here.


#### _train_prophet_model()

Method description here.


#### _train_random_forest_model()

Method description here.


#### _train_linear_model()

Method description here.


#### _evaluate_model()

Method description here.


#### forecast()

Method description here.


#### _predict_deep_learning()

Method description here.


#### _predict_sklearn()

Method description here.


#### _predict_arima()

Method description here.


#### _predict_prophet()

Method description here.


#### train_all_models()

Method description here.


#### get_ensemble_forecast()

Method description here.


#### get_model_performance_summary()

Method description here.



### Attributes



---

## src.bot.ml.time_series_forecaster.LSTMForecaster

LSTM-based time series forecaster.

**File:** `src/bot\ml\time_series_forecaster.py`


**Inherits from:** <ast.Attribute object at 0x0000021F09FDBF10>


### Methods


#### __init__()

Method description here.


#### forward()

Method description here.



### Attributes



---

## src.bot.ml.time_series_forecaster.TransformerForecaster

Transformer-based time series forecaster.

**File:** `src/bot\ml\time_series_forecaster.py`


**Inherits from:** <ast.Attribute object at 0x0000021F0A0C3510>


### Methods


#### __init__()

Method description here.


#### _create_positional_encoding()

Method description here.


#### forward()

Method description here.



### Attributes



---

## src.bot.ml.optimization.inference_optimizer.InferenceMode

Inference execution modes

**File:** `src/bot\ml\optimization\inference_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SINGLE`

- `BATCH`

- `STREAMING`

- `CACHED`


---

## src.bot.ml.optimization.inference_optimizer.AcceleratorType

Hardware accelerator types

**File:** `src/bot\ml\optimization\inference_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CPU`

- `GPU_CUDA`

- `GPU_MPS`

- `TENSORRT`

- `OPENVINO`


---

## src.bot.ml.optimization.inference_optimizer.InferenceRequest

Individual inference request

**File:** `src/bot\ml\optimization\inference_optimizer.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.ml.optimization.inference_optimizer.InferenceResult

Inference result with metadata

**File:** `src/bot\ml\optimization\inference_optimizer.py`



### Methods



### Attributes



---

## src.bot.ml.optimization.inference_optimizer.BatchInferenceRequest

Batch inference request

**File:** `src/bot\ml\optimization\inference_optimizer.py`



### Methods



### Attributes



---

## src.bot.ml.optimization.inference_optimizer.InferenceOptimizer

Advanced inference optimization and acceleration engine

Features:
- Async/parallel inference processing ✅
- Intelligent batching and caching ✅
- Hardware acceleration management ✅
- Sub-20ms inference times ✅
- Memory optimization ✅

**File:** `src/bot\ml\optimization\inference_optimizer.py`



### Methods


#### __init__()

Method description here.


#### _detect_best_accelerator()

Method description here.


#### _setup_hardware_optimization()

Method description here.


#### _check_cache()

Method description here.


#### _update_cache()

Method description here.


#### get_performance_metrics()

Method description here.


#### get_memory_usage()

Method description here.



### Attributes



---

## src.bot.ml.optimization.model_compressor.OptimizationTechnique

Available optimization techniques

**File:** `src/bot\ml\optimization\model_compressor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `QUANTIZATION_INT8`

- `QUANTIZATION_FP16`

- `PRUNING_STRUCTURED`

- `PRUNING_UNSTRUCTURED`

- `KNOWLEDGE_DISTILLATION`

- `ONNX_OPTIMIZATION`

- `TENSORRT_OPTIMIZATION`


---

## src.bot.ml.optimization.model_compressor.InferenceEngine

Supported inference engines

**File:** `src/bot\ml\optimization\model_compressor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PYTORCH`

- `ONNX`

- `TENSORRT`

- `OPENVINO`


---

## src.bot.ml.optimization.model_compressor.OptimizationConfig

Model optimization configuration

**File:** `src/bot\ml\optimization\model_compressor.py`



### Methods



### Attributes



---

## src.bot.ml.optimization.model_compressor.OptimizationResult

Result of model optimization

**File:** `src/bot\ml\optimization\model_compressor.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.ml.optimization.model_compressor.ModelCompressor

Advanced model compression and optimization engine

Features:
- Multi-technique optimization pipeline ✅
- Automatic technique selection ✅
- Inference engine optimization ✅
- Real-time performance monitoring ✅
- <20ms inference achievement ✅

**File:** `src/bot\ml\optimization\model_compressor.py`



### Methods


#### __init__()

Method description here.


#### _get_model_size_mb()

Method description here.


#### _get_sample_input()

Method description here.


#### _get_calibration_data()

Method description here.


#### _generate_optimization_recommendations()

Method description here.


#### _copy_model()

Method description here.


#### _create_student_model()

Method description here.



### Attributes



---

## src.bot.ml.transfer_learning.transfer_learning_engine.TransferLearningStrategy

Transfer learning strategies

**File:** `src/bot\ml\transfer_learning\transfer_learning_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `FEATURE_EXTRACTION`

- `FINE_TUNING`

- `DOMAIN_ADAPTATION`

- `META_LEARNING`

- `ENSEMBLE_TRANSFER`


---

## src.bot.ml.transfer_learning.transfer_learning_engine.MarketRegime

Market regime classifications

**File:** `src/bot\ml\transfer_learning\transfer_learning_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRENDING_UP`

- `TRENDING_DOWN`

- `SIDEWAYS`

- `HIGH_VOLATILITY`

- `LOW_VOLATILITY`

- `BREAKOUT`

- `REVERSAL`


---

## src.bot.ml.transfer_learning.transfer_learning_engine.TransferLearningTask

Transfer learning task definition

**File:** `src/bot\ml\transfer_learning\transfer_learning_engine.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.ml.transfer_learning.transfer_learning_engine.TransferResults

Results of transfer learning process

**File:** `src/bot\ml\transfer_learning\transfer_learning_engine.py`



### Methods



### Attributes



---

## src.bot.ml.transfer_learning.transfer_learning_engine.MarketSimilarityAnalyzer

Analyze similarity between markets for transfer learning

**File:** `src/bot\ml\transfer_learning\transfer_learning_engine.py`



### Methods


#### __init__()

Method description here.


#### _calculate_feature_similarity()

Method description here.



### Attributes



---

## src.bot.ml.transfer_learning.transfer_learning_engine.TransferLearningEngine

Advanced transfer learning engine for cross-market knowledge transfer

Features:
- Multi-strategy transfer learning ✅
- Market similarity analysis ✅
- Adaptive transfer optimization ✅
- Meta-learning capabilities ✅
- Ensemble transfer methods ✅

**File:** `src/bot\ml\transfer_learning\transfer_learning_engine.py`



### Methods


#### __init__()

Method description here.


#### _create_feature_extraction_model()

Method description here.


#### _create_fine_tuning_model()

Method description here.


#### _get_model_size_mb()

Method description here.


#### get_transfer_learning_metrics()

Method description here.


#### _analyze_strategy_performance()

Method description here.



### Attributes



---

## src.bot.ml_strategy_discovery.data_infrastructure.ExchangeName

Supported exchanges

**File:** `src/bot\ml_strategy_discovery\data_infrastructure.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BYBIT`

- `BINANCE`

- `COINBASE`

- `KRAKEN`

- `BTCMARKETS`

- `COINJAR`

- `SWYFTX`


---

## src.bot.ml_strategy_discovery.data_infrastructure.DataType

Types of market data

**File:** `src/bot\ml_strategy_discovery\data_infrastructure.py`


**Inherits from:** Enum


### Methods



### Attributes


- `OHLCV`

- `ORDERBOOK`

- `TRADES`

- `FUNDING_RATES`

- `OPEN_INTEREST`


---

## src.bot.ml_strategy_discovery.data_infrastructure.TransferCost

Transfer cost between exchanges

**File:** `src/bot\ml_strategy_discovery\data_infrastructure.py`



### Methods



### Attributes



---

## src.bot.ml_strategy_discovery.data_infrastructure.ExchangeInfo

Exchange information and capabilities

**File:** `src/bot\ml_strategy_discovery\data_infrastructure.py`



### Methods



### Attributes



---

## src.bot.ml_strategy_discovery.data_infrastructure.MarketData

Market data container

**File:** `src/bot\ml_strategy_discovery\data_infrastructure.py`



### Methods



### Attributes



---

## src.bot.ml_strategy_discovery.data_infrastructure.AustralianDataProvider

Australian-specific data provider
RBA, ASX, and macro data relevant to crypto trading

**File:** `src/bot\ml_strategy_discovery\data_infrastructure.py`



### Methods


#### __init__()

Method description here.


#### _is_cached_valid()

Method description here.



### Attributes



---

## src.bot.ml_strategy_discovery.data_infrastructure.TransferCostDatabase

Database for tracking transfer costs between exchanges
Critical for Australian arbitrage where bank transfer costs matter

**File:** `src/bot\ml_strategy_discovery\data_infrastructure.py`



### Methods


#### __init__()

Method description here.


#### _initialize_database()

Method description here.


#### _populate_australian_costs()

Method description here.


#### add_transfer_cost()

Method description here.


#### add_exchange_info()

Method description here.


#### get_transfer_cost()

Method description here.


#### get_australian_friendly_routes()

Method description here.


#### calculate_transfer_cost()

Method description here.



### Attributes



---

## src.bot.ml_strategy_discovery.data_infrastructure.MultiExchangeDataCollector

Multi-exchange data collection optimized for Australian traders

**File:** `src/bot\ml_strategy_discovery\data_infrastructure.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.ml_strategy_discovery.ml_engine.StrategyType

ML Strategy types

**File:** `src/bot\ml_strategy_discovery\ml_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TREND_FOLLOWING`

- `MEAN_REVERSION`

- `MOMENTUM`

- `VOLATILITY`

- `MULTI_FACTOR`


---

## src.bot.ml_strategy_discovery.ml_engine.ModelType

Machine learning model types

**File:** `src/bot\ml_strategy_discovery\ml_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `RANDOM_FOREST`

- `GRADIENT_BOOSTING`

- `RIDGE_REGRESSION`

- `ELASTIC_NET`

- `ENSEMBLE`


---

## src.bot.ml_strategy_discovery.ml_engine.FeatureSet

Feature engineering configuration

**File:** `src/bot\ml_strategy_discovery\ml_engine.py`



### Methods



### Attributes



---

## src.bot.ml_strategy_discovery.ml_engine.ModelConfiguration

ML model configuration

**File:** `src/bot\ml_strategy_discovery\ml_engine.py`



### Methods



### Attributes



---

## src.bot.ml_strategy_discovery.ml_engine.StrategySignal

ML-generated trading signal

**File:** `src/bot\ml_strategy_discovery\ml_engine.py`



### Methods



### Attributes



---

## src.bot.ml_strategy_discovery.ml_engine.FeatureEngineer

Advanced feature engineering for Australian crypto trading

**File:** `src/bot\ml_strategy_discovery\ml_engine.py`



### Methods


#### __init__()

Method description here.


#### create_technical_features()

Method description here.


#### create_australian_features()

Method description here.


#### create_macro_features()

Method description here.


#### create_target_variables()

Method description here.


#### engineer_features()

Method description here.



### Attributes



---

## src.bot.ml_strategy_discovery.ml_engine.MLStrategyModel

Individual ML model for strategy generation

**File:** `src/bot\ml_strategy_discovery\ml_engine.py`



### Methods


#### __init__()

Method description here.


#### _initialize_model()

Method description here.


#### prepare_features()

Method description here.


#### train()

Method description here.


#### predict()

Method description here.


#### generate_signal()

Method description here.


#### save_model()

Method description here.


#### load_model()

Method description here.



### Attributes



---

## src.bot.ml_strategy_discovery.ml_engine.MLStrategyDiscoveryEngine

Main ML Strategy Discovery Engine
Coordinates multiple ML models for comprehensive strategy generation

**File:** `src/bot\ml_strategy_discovery\ml_engine.py`



### Methods


#### __init__()

Method description here.


#### _initialize_models()

Method description here.


#### train_models()

Method description here.


#### generate_signals()

Method description here.


#### evaluate_model_performance()

Method description here.


#### get_strategy_recommendations()

Method description here.



### Attributes



---

## src.bot.monitoring.api_health.HealthStatus

API health status levels.

**File:** `src/bot\monitoring\api_health.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HEALTHY`

- `DEGRADED`

- `UNHEALTHY`

- `CRITICAL`

- `UNKNOWN`


---

## src.bot.monitoring.api_health.AlertLevel

Alert severity levels.

**File:** `src/bot\monitoring\api_health.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `ERROR`

- `CRITICAL`


---

## src.bot.monitoring.api_health.HealthMetrics

Health metrics for an API endpoint.

**File:** `src/bot\monitoring\api_health.py`



### Methods



### Attributes



---

## src.bot.monitoring.api_health.HealthCheck

Configuration for a health check.

**File:** `src/bot\monitoring\api_health.py`



### Methods



### Attributes



---

## src.bot.monitoring.api_health.Alert

Alert message structure.

**File:** `src/bot\monitoring\api_health.py`



### Methods



### Attributes



---

## src.bot.monitoring.api_health.APIHealthMonitor

Comprehensive API health monitoring system.

Monitors all external APIs, tracks performance metrics, implements fallback
strategies, and provides alerting for critical issues.

**File:** `src/bot\monitoring\api_health.py`



### Methods


#### __init__()

Method description here.


#### _initialize_health_checks()

Method description here.


#### _initialize_fallback_strategies()

Method description here.


#### _initialize_metrics()

Method description here.


#### _should_check_api()

Method description here.


#### _update_health_metrics()

Method description here.


#### _determine_health_status()

Method description here.


#### _check_for_alerts()

Method description here.


#### _update_system_health()

Method description here.


#### _cleanup_old_alerts()

Method description here.


#### _validate_bybit_response()

Method description here.


#### _validate_cryptopanic_response()

Method description here.


#### _validate_fear_greed_response()

Method description here.


#### get_api_health_status()

Method description here.


#### get_all_health_metrics()

Method description here.


#### get_system_health_report()

Method description here.


#### get_fallback_strategy()

Method description here.


#### get_recent_alerts()

Method description here.



### Attributes



---

## src.bot.monitoring.enhanced.performance_tracker.MetricType

Types of performance metrics

**File:** `src/bot\monitoring\enhanced\performance_tracker.py`


**Inherits from:** Enum


### Methods



### Attributes


- `EXECUTION_TIME`

- `SLIPPAGE`

- `FILL_RATE`

- `ML_INFERENCE`

- `LIQUIDITY`

- `ACCURACY`

- `MEMORY_USAGE`

- `THROUGHPUT`


---

## src.bot.monitoring.enhanced.performance_tracker.AlertLevel

Alert severity levels

**File:** `src/bot\monitoring\enhanced\performance_tracker.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `CRITICAL`

- `EMERGENCY`


---

## src.bot.monitoring.enhanced.performance_tracker.PerformanceMetric

Individual performance metric

**File:** `src/bot\monitoring\enhanced\performance_tracker.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.monitoring.enhanced.performance_tracker.PerformanceAlert

Performance alert

**File:** `src/bot\monitoring\enhanced\performance_tracker.py`



### Methods



### Attributes



---

## src.bot.monitoring.enhanced.performance_tracker.ComponentStatus

Status of a monitored component

**File:** `src/bot\monitoring\enhanced\performance_tracker.py`



### Methods



### Attributes



---

## src.bot.monitoring.enhanced.performance_tracker.EnhancedPerformanceTracker

Enhanced performance tracking for optimized components

Features:
- Real-time metric collection ✅
- Performance threshold monitoring ✅
- Component health tracking ✅
- Advanced analytics and alerting ✅
- Dashboard integration ready ✅

**File:** `src/bot\monitoring\enhanced\performance_tracker.py`



### Methods


#### __init__()

Method description here.


#### _get_metric_type_from_name()

Method description here.


#### _create_alert_message()

Method description here.


#### add_alert_callback()

Method description here.


#### get_health_status()

Method description here.



### Attributes



---

## src.bot.optimization.bayesian_optimizer.AcquisitionFunction

Acquisition functions for Bayesian optimization

**File:** `src/bot\optimization\bayesian_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `EXPECTED_IMPROVEMENT`

- `UPPER_CONFIDENCE_BOUND`

- `PROBABILITY_OF_IMPROVEMENT`

- `ENTROPY_SEARCH`

- `KNOWLEDGE_GRADIENT`


---

## src.bot.optimization.bayesian_optimizer.OptimizationObjective

Optimization objectives

**File:** `src/bot\optimization\bayesian_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MAXIMIZE_SHARPE`

- `MAXIMIZE_RETURN`

- `MINIMIZE_DRAWDOWN`

- `MAXIMIZE_WIN_RATE`

- `MINIMIZE_SLIPPAGE`

- `MAXIMIZE_FILL_RATE`

- `MULTI_OBJECTIVE`


---

## src.bot.optimization.bayesian_optimizer.OptimizationParameter

Parameter definition for optimization

**File:** `src/bot\optimization\bayesian_optimizer.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.optimization.bayesian_optimizer.OptimizationResult

Result of a single optimization trial

**File:** `src/bot\optimization\bayesian_optimizer.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.optimization.bayesian_optimizer.OptimizationConfig

Configuration for Bayesian optimization

**File:** `src/bot\optimization\bayesian_optimizer.py`



### Methods



### Attributes



---

## src.bot.optimization.bayesian_optimizer.BayesianOptimizer

Advanced Bayesian optimization engine

Features:
- Gaussian Process surrogate models ✅
- Multiple acquisition functions ✅
- Multi-objective optimization ✅
- Adaptive hyperparameter tuning ✅
- Contextual optimization ✅

**File:** `src/bot\optimization\bayesian_optimizer.py`



### Methods


#### __init__()

Method description here.


#### _create_kernel()

Method description here.


#### _extract_parameter_bounds()

Method description here.


#### _get_acquisition_function()

Method description here.


#### _generate_random_parameters()

Method description here.


#### _fit_gaussian_process()

Method description here.


#### _parameters_to_vector()

Method description here.


#### _vector_to_parameters()

Method description here.


#### _expected_improvement()

Method description here.


#### _upper_confidence_bound()

Method description here.


#### _probability_of_improvement()

Method description here.


#### _update_optimization_state()

Method description here.


#### _update_pareto_frontier()

Method description here.


#### _dominates()

Method description here.


#### get_optimization_summary()

Method description here.


#### _calculate_improvement_achieved()

Method description here.



### Attributes



---

## src.bot.optimization.phase2_integration.Phase2Status

Phase 2 implementation status

**File:** `src/bot\optimization\phase2_integration.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INITIALIZING`

- `ACTIVE`

- `OPTIMIZING`

- `LEARNING`

- `ANALYZING`

- `COMPLETED`

- `ERROR`


---

## src.bot.optimization.phase2_integration.IntegrationMode

Integration operation modes

**File:** `src/bot\optimization\phase2_integration.py`


**Inherits from:** Enum


### Methods



### Attributes


- `UNIFIED`

- `SEQUENTIAL`

- `PARALLEL`

- `ADAPTIVE`


---

## src.bot.optimization.phase2_integration.Phase2Metrics

Phase 2 performance metrics tracking

**File:** `src/bot\optimization\phase2_integration.py`



### Methods


#### calculate_overall_improvement()

Method description here.



### Attributes



---

## src.bot.optimization.phase2_integration.IntegrationConfig

Configuration for Phase 2 integration

**File:** `src/bot\optimization\phase2_integration.py`



### Methods



### Attributes



---

## src.bot.optimization.phase2_integration.Phase2IntegrationManager

Phase 2 Integration Manager - Unified Advanced ML System

Orchestrates all Phase 2 components for seamless operation:
- Transfer Learning Engine ✅
- Bayesian Optimizer ✅
- Strategy Optimizer ✅
- Advanced Analytics Engine ✅
- Performance Tracking Integration ✅

**File:** `src/bot\optimization\phase2_integration.py`



### Methods


#### __init__()

Method description here.


#### _calculate_workflow_improvement()

Method description here.


#### get_phase2_status_report()

Method description here.



### Attributes



---

## src.bot.optimization.strategy_optimizer.StrategyType

Strategy types for optimization

**File:** `src/bot\optimization\strategy_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TREND_FOLLOWING`

- `MEAN_REVERSION`

- `MOMENTUM`

- `ARBITRAGE`

- `MARKET_MAKING`

- `BREAKOUT`

- `ENSEMBLE`


---

## src.bot.optimization.strategy_optimizer.OptimizationMode

Optimization modes

**File:** `src/bot\optimization\strategy_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SINGLE_OBJECTIVE`

- `MULTI_OBJECTIVE`

- `RISK_ADJUSTED`

- `ROBUST`

- `ADAPTIVE`


---

## src.bot.optimization.strategy_optimizer.StrategyConfig

Strategy configuration for optimization

**File:** `src/bot\optimization\strategy_optimizer.py`



### Methods



### Attributes



---

## src.bot.optimization.strategy_optimizer.OptimizationSession

Optimization session tracking

**File:** `src/bot\optimization\strategy_optimizer.py`



### Methods



### Attributes



---

## src.bot.optimization.strategy_optimizer.StrategyOptimizationManager

Advanced strategy optimization and hyperparameter tuning manager

Features:
- Multi-strategy optimization ✅
- Bayesian hyperparameter tuning ✅
- Transfer learning integration ✅
- Risk-adjusted optimization ✅
- Real-time adaptation ✅

**File:** `src/bot\optimization\strategy_optimizer.py`



### Methods


#### __init__()

Method description here.


#### _initialize_strategy_templates()

Method description here.


#### _calculate_improvement()

Method description here.


#### get_optimization_metrics()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.compliance_reporter.ReportType

Types of compliance reports

**File:** `src/bot\performance_monitoring\compliance_reporter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ATO_CGT_SCHEDULE`

- `ATO_BUSINESS_ACTIVITY`

- `ASIC_DERIVATIVE_REPORT`

- `AUSTRAC_TRANSACTION_REPORT`

- `INTERNAL_AUDIT_TRAIL`

- `TAX_OPTIMIZATION_REPORT`

- `PROFESSIONAL_TRADER_ASSESSMENT`

- `QUARTERLY_COMPLIANCE_SUMMARY`


---

## src.bot.performance_monitoring.compliance_reporter.ReportStatus

Report generation status

**File:** `src/bot\performance_monitoring\compliance_reporter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `GENERATING`

- `COMPLETED`

- `FAILED`

- `SUBMITTED`


---

## src.bot.performance_monitoring.compliance_reporter.ComplianceReport

Compliance report metadata and content

**File:** `src/bot\performance_monitoring\compliance_reporter.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.compliance_reporter.ATOReportGenerator

Australian Taxation Office report generator
Creates CGT schedules, business activity statements, and tax optimization reports

**File:** `src/bot\performance_monitoring\compliance_reporter.py`



### Methods


#### __init__()

Method description here.


#### _calculate_cgt_summary()

Method description here.


#### _get_financial_year()

Method description here.


#### _generate_cgt_csv()

Method description here.


#### _validate_cgt_schedule()

Method description here.


#### _get_quarter()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.compliance_reporter.ASICReportGenerator

Australian Securities and Investments Commission report generator
Creates derivative transaction reports and market participation reports

**File:** `src/bot\performance_monitoring\compliance_reporter.py`



### Methods


#### __init__()

Method description here.


#### _format_asic_trades()

Method description here.


#### _generate_asic_csv()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.compliance_reporter.AustralianComplianceReporter

Main compliance reporting coordinator
Manages all Australian regulatory reporting requirements

**File:** `src/bot\performance_monitoring\compliance_reporter.py`



### Methods


#### __init__()

Method description here.


#### _generate_report_id()

Method description here.


#### save_report_to_file()

Method description here.


#### get_report_summary()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.dashboard.DashboardMetricType

Types of dashboard metrics

**File:** `src/bot\performance_monitoring\dashboard.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PORTFOLIO_VALUE`

- `DAILY_PNL`

- `TRADE_COUNT`

- `RISK_SCORE`

- `COMPLIANCE_STATUS`

- `ML_PERFORMANCE`

- `ARBITRAGE_PERFORMANCE`

- `TAX_LIABILITY`

- `DRAWDOWN`

- `SHARPE_RATIO`


---

## src.bot.performance_monitoring.dashboard.AlertLevel

Alert severity levels

**File:** `src/bot\performance_monitoring\dashboard.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `CRITICAL`

- `EMERGENCY`


---

## src.bot.performance_monitoring.dashboard.PerformanceMetric

Performance metric data point

**File:** `src/bot\performance_monitoring\dashboard.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.dashboard.DashboardAlert

Dashboard alert/notification

**File:** `src/bot\performance_monitoring\dashboard.py`



### Methods


#### to_dict()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.dashboard.AustralianPerformanceCalculator

Calculates Australian-specific performance metrics
Handles AUD-denominated returns, tax-adjusted performance, and compliance metrics

**File:** `src/bot\performance_monitoring\dashboard.py`



### Methods


#### __init__()

Method description here.


#### add_daily_return()

Method description here.


#### calculate_sharpe_ratio()

Method description here.


#### calculate_max_drawdown()

Method description here.


#### calculate_tax_adjusted_return()

Method description here.


#### calculate_australian_alpha()

Method description here.


#### get_performance_summary()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.dashboard.RealTimeDashboard

Real-time web dashboard for Australian trading system monitoring
Provides live updates of performance, compliance, and system health

**File:** `src/bot\performance_monitoring\dashboard.py`



### Methods


#### __init__()

Method description here.


#### _generate_alert_id()

Method description here.


#### _calculate_overall_risk_level()

Method description here.


#### _calculate_uptime_hours()

Method description here.


#### get_dashboard_data()

Method description here.


#### _get_alert_counts_by_level()

Method description here.


#### acknowledge_alert()

Method description here.


#### clear_acknowledged_alerts()

Method description here.


#### generate_portfolio_chart()

Method description here.


#### generate_pnl_chart()

Method description here.


#### generate_risk_metrics_chart()

Method description here.


#### stop_dashboard()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.performance_analyzer.PerformancePeriod

Performance analysis periods

**File:** `src/bot\performance_monitoring\performance_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DAILY`

- `WEEKLY`

- `MONTHLY`

- `QUARTERLY`

- `YEARLY`

- `INCEPTION`


---

## src.bot.performance_monitoring.performance_analyzer.BenchmarkType

Benchmark comparison types

**File:** `src/bot\performance_monitoring\performance_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ASX_200`

- `ASX_300`

- `AUD_CASH_RATE`

- `BITCOIN_USD`

- `BITCOIN_AUD`

- `ETHEREUM_AUD`

- `CRYPTO_INDEX`


---

## src.bot.performance_monitoring.performance_analyzer.PerformanceMetrics

Comprehensive performance metrics

**File:** `src/bot\performance_monitoring\performance_analyzer.py`



### Methods



### Attributes



---

## src.bot.performance_monitoring.performance_analyzer.TradeAnalysis

Individual trade analysis

**File:** `src/bot\performance_monitoring\performance_analyzer.py`



### Methods



### Attributes



---

## src.bot.performance_monitoring.performance_analyzer.AustralianBenchmarkProvider

Provides Australian and international benchmark data
Simulates ASX indices and AUD-denominated crypto benchmarks

**File:** `src/bot\performance_monitoring\performance_analyzer.py`



### Methods


#### __init__()

Method description here.


#### get_benchmark_info()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.performance_analyzer.StrategyPerformanceAnalyzer

Analyzes performance of ML strategies and arbitrage opportunities
Provides strategy attribution and optimization insights

**File:** `src/bot\performance_monitoring\performance_analyzer.py`



### Methods


#### __init__()

Method description here.


#### add_trade_result()

Method description here.


#### analyze_strategy_performance()

Method description here.


#### compare_strategies()

Method description here.


#### get_strategy_correlation_matrix()

Method description here.



### Attributes



---

## src.bot.performance_monitoring.performance_analyzer.AustralianPerformanceAnalyzer

Main performance analyzer with Australian-specific calculations
Provides comprehensive performance analysis, benchmarking, and optimization insights

**File:** `src/bot\performance_monitoring\performance_analyzer.py`



### Methods


#### __init__()

Method description here.


#### add_portfolio_snapshot()

Method description here.


#### add_trade_result()

Method description here.


#### _calculate_total_return()

Method description here.


#### _annualize_return()

Method description here.


#### _calculate_volatility()

Method description here.


#### _calculate_sortino_ratio()

Method description here.


#### _calculate_calmar_ratio()

Method description here.


#### _calculate_drawdown_metrics()

Method description here.


#### _calculate_trade_metrics()

Method description here.


#### _calculate_cgt_discount_benefit()

Method description here.


#### _calculate_strategy_attribution()

Method description here.


#### _create_default_metrics()

Method description here.


#### _is_cache_valid()

Method description here.


#### _clear_cache_by_prefix()

Method description here.


#### get_performance_summary()

Method description here.



### Attributes



---

## src.bot.portfolio.asset_allocator.AllocationStrategy

Asset allocation strategies.

**File:** `src/bot\portfolio\asset_allocator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `EQUAL_WEIGHT`

- `MARKET_CAP_WEIGHT`

- `VOLATILITY_PARITY`

- `RISK_PARITY`

- `MOMENTUM`

- `MEAN_REVERSION`

- `BLACK_LITTERMAN`

- `HIERARCHICAL_RISK_PARITY`

- `ADAPTIVE`

- `FACTOR_BASED`


---

## src.bot.portfolio.asset_allocator.OptimizationMethod

Optimization methods for allocation.

**File:** `src/bot\portfolio\asset_allocator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MEAN_VARIANCE`

- `MINIMUM_VARIANCE`

- `MAXIMUM_SHARPE`

- `MAXIMUM_DIVERSIFICATION`

- `RISK_BUDGETING`

- `CVaR_OPTIMIZATION`


---

## src.bot.portfolio.asset_allocator.AllocationResult

Asset allocation result.

**File:** `src/bot\portfolio\asset_allocator.py`



### Methods



### Attributes



---

## src.bot.portfolio.asset_allocator.AssetAllocator

Advanced asset allocation engine with multiple strategies.

**File:** `src/bot\portfolio\asset_allocator.py`



### Methods


#### __init__()

Method description here.


#### _calculate_confidence_score()

Method description here.


#### _determine_rebalance_frequency()

Method description here.


#### _calculate_next_rebalance_date()

Method description here.


#### _generate_allocation_rationale()

Method description here.


#### _validate_allocation_inputs()

Method description here.


#### _store_allocation_result()

Method description here.


#### get_allocation_history()

Method description here.


#### get_allocation_summary()

Method description here.



### Attributes



---

## src.bot.portfolio.correlation_analyzer.CorrelationRegime

Correlation regimes.

**File:** `src/bot\portfolio\correlation_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW_CORRELATION`

- `MODERATE_CORRELATION`

- `HIGH_CORRELATION`

- `CRISIS_CORRELATION`


---

## src.bot.portfolio.correlation_analyzer.CorrelationMeasure

Correlation measurement methods.

**File:** `src/bot\portfolio\correlation_analyzer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PEARSON`

- `SPEARMAN`

- `KENDALL`

- `ROLLING_CORRELATION`

- `DYNAMIC_CORRELATION`

- `TAIL_CORRELATION`


---

## src.bot.portfolio.correlation_analyzer.CorrelationAnalysis

Correlation analysis result.

**File:** `src/bot\portfolio\correlation_analyzer.py`



### Methods



### Attributes



---

## src.bot.portfolio.correlation_analyzer.CorrelationAnalyzer

Advanced correlation analysis and monitoring engine.

**File:** `src/bot\portfolio\correlation_analyzer.py`



### Methods


#### __init__()

Method description here.


#### _validate_correlation_inputs()

Method description here.


#### _store_correlation_analysis()

Method description here.


#### get_correlation_history()

Method description here.


#### get_correlation_summary()

Method description here.



### Attributes



---

## src.bot.portfolio.integration_test.PortfolioManagerIntegrationTest

Integration test for the complete portfolio management system.

**File:** `src/bot\portfolio\integration_test.py`



### Methods


#### __init__()

Method description here.


#### _generate_mock_price_data()

Method description here.



### Attributes



---

## src.bot.portfolio.portfolio_manager.PositionType

Position types.

**File:** `src/bot\portfolio\portfolio_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LONG`

- `SHORT`


---

## src.bot.portfolio.portfolio_manager.PortfolioStatus

Portfolio status.

**File:** `src/bot\portfolio\portfolio_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ACTIVE`

- `PAUSED`

- `LIQUIDATING`

- `CLOSED`


---

## src.bot.portfolio.portfolio_manager.Position

Individual position in the portfolio.

**File:** `src/bot\portfolio\portfolio_manager.py`



### Methods


#### total_pnl()

Method description here.


#### pnl_percentage()

Method description here.



### Attributes



---

## src.bot.portfolio.portfolio_manager.PortfolioConstraints

Portfolio optimization constraints.

**File:** `src/bot\portfolio\portfolio_manager.py`



### Methods



### Attributes



---

## src.bot.portfolio.portfolio_manager.Portfolio

Complete portfolio representation.

**File:** `src/bot\portfolio\portfolio_manager.py`



### Methods


#### invested_value()

Method description here.


#### total_pnl()

Method description here.


#### num_positions()

Method description here.



### Attributes



---

## src.bot.portfolio.portfolio_manager.PortfolioManager

Advanced multi-asset portfolio management system.

**File:** `src/bot\portfolio\portfolio_manager.py`



### Methods


#### __init__()

Method description here.


#### get_portfolio()

Method description here.


#### list_portfolios()

Method description here.


#### get_portfolio_summary()

Method description here.



### Attributes



---

## src.bot.portfolio.rebalancer.RebalanceStrategy

Rebalancing strategies.

**File:** `src/bot\portfolio\rebalancer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CALENDAR`

- `THRESHOLD`

- `VOLATILITY_TARGET`

- `RISK_BUDGET`

- `MOMENTUM`

- `ADAPTIVE`

- `COST_OPTIMIZED`

- `TACTICAL`


---

## src.bot.portfolio.rebalancer.RebalanceTrigger

Rebalancing triggers.

**File:** `src/bot\portfolio\rebalancer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TIME_BASED`

- `DRIFT_BASED`

- `VOLATILITY_BASED`

- `CORRELATION_BASED`

- `DRAWDOWN_BASED`

- `MOMENTUM_BASED`

- `COMBINED`


---

## src.bot.portfolio.rebalancer.RebalanceSignal

Rebalancing signal.

**File:** `src/bot\portfolio\rebalancer.py`



### Methods



### Attributes



---

## src.bot.portfolio.rebalancer.RebalanceTransaction

Individual rebalancing transaction.

**File:** `src/bot\portfolio\rebalancer.py`



### Methods



### Attributes



---

## src.bot.portfolio.rebalancer.RebalanceResult

Rebalancing execution result.

**File:** `src/bot\portfolio\rebalancer.py`



### Methods



### Attributes



---

## src.bot.portfolio.rebalancer.PortfolioRebalancer

Advanced portfolio rebalancing engine.

**File:** `src/bot\portfolio\rebalancer.py`



### Methods


#### __init__()

Method description here.


#### _validate_rebalancing_inputs()

Method description here.


#### _store_rebalance_result()

Method description here.


#### get_rebalance_history()

Method description here.


#### get_pending_signals()

Method description here.


#### get_rebalancer_summary()

Method description here.



### Attributes



---

## src.bot.portfolio.risk_budgeter.RiskBudgetType

Risk budget types.

**File:** `src/bot\portfolio\risk_budgeter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `VOLATILITY`

- `VAR`

- `CVAR`

- `MAXIMUM_DRAWDOWN`

- `TRACKING_ERROR`

- `DOWNSIDE_DEVIATION`

- `TAIL_RISK`

- `CONCENTRATION_RISK`


---

## src.bot.portfolio.risk_budgeter.RiskConstraintType

Risk constraint types.

**File:** `src/bot\portfolio\risk_budgeter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `POSITION_LIMIT`

- `SECTOR_LIMIT`

- `CORRELATION_LIMIT`

- `VOLATILITY_LIMIT`

- `VAR_LIMIT`

- `DRAWDOWN_LIMIT`

- `CONCENTRATION_LIMIT`

- `LEVERAGE_LIMIT`


---

## src.bot.portfolio.risk_budgeter.RiskBudget

Risk budget definition.

**File:** `src/bot\portfolio\risk_budgeter.py`



### Methods



### Attributes



---

## src.bot.portfolio.risk_budgeter.RiskConstraint

Risk constraint definition.

**File:** `src/bot\portfolio\risk_budgeter.py`



### Methods



### Attributes



---

## src.bot.portfolio.risk_budgeter.RiskAttribution

Risk attribution analysis.

**File:** `src/bot\portfolio\risk_budgeter.py`



### Methods



### Attributes



---

## src.bot.portfolio.risk_budgeter.RiskReport

Comprehensive risk report.

**File:** `src/bot\portfolio\risk_budgeter.py`



### Methods



### Attributes



---

## src.bot.portfolio.risk_budgeter.RiskBudgeter

Advanced risk budgeting and constraint management engine.

**File:** `src/bot\portfolio\risk_budgeter.py`



### Methods


#### __init__()

Method description here.


#### get_risk_history()

Method description here.


#### get_constraint_violations()

Method description here.


#### get_risk_summary()

Method description here.



### Attributes



---

## src.bot.risk.core.unified_risk_manager.RiskLevel

Risk level classification

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `VERY_LOW`

- `LOW`

- `MODERATE`

- `HIGH`

- `VERY_HIGH`

- `EXTREME`


---

## src.bot.risk.core.unified_risk_manager.PositionSizeMethod

Position sizing methods

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `FIXED_PERCENTAGE`

- `KELLY_CRITERION`

- `VOLATILITY_TARGETING`

- `RISK_PARITY`

- `MAX_DRAWDOWN`

- `TAX_OPTIMIZED`


---

## src.bot.risk.core.unified_risk_manager.MarketRegime

Market regime classification

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW_VOLATILITY`

- `NORMAL`

- `HIGH_VOLATILITY`

- `CRISIS`

- `TRENDING_UP`

- `TRENDING_DOWN`


---

## src.bot.risk.core.unified_risk_manager.AlertLevel

Alert severity levels

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `CRITICAL`

- `EMERGENCY`


---

## src.bot.risk.core.unified_risk_manager.RiskParameters

Unified risk parameters configuration

**File:** `src/bot\risk\core\unified_risk_manager.py`



### Methods


#### __post_init__()

Method description here.


#### from_unified_config()

Method description here.



### Attributes



---

## src.bot.risk.core.unified_risk_manager.PositionRisk

Risk assessment for a specific position

**File:** `src/bot\risk\core\unified_risk_manager.py`



### Methods



### Attributes



---

## src.bot.risk.core.unified_risk_manager.PortfolioRiskMetrics

Comprehensive portfolio risk metrics

**File:** `src/bot\risk\core\unified_risk_manager.py`



### Methods



### Attributes



---

## src.bot.risk.core.unified_risk_manager.PositionSizer

Abstract base class for position sizing methods

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** ABC


### Methods


#### calculate_size()

Method description here.



### Attributes



---

## src.bot.risk.core.unified_risk_manager.KellyCriterionSizer

Kelly Criterion position sizing

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** PositionSizer


### Methods


#### calculate_size()

Method description here.



### Attributes



---

## src.bot.risk.core.unified_risk_manager.RiskParitySizer

Risk Parity position sizing

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** PositionSizer


### Methods


#### calculate_size()

Method description here.



### Attributes



---

## src.bot.risk.core.unified_risk_manager.VolatilityTargetingSizer

Volatility targeting position sizing

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** PositionSizer


### Methods


#### calculate_size()

Method description here.



### Attributes



---

## src.bot.risk.core.unified_risk_manager.TaxOptimizedSizer

Australian tax-optimized position sizing

**File:** `src/bot\risk\core\unified_risk_manager.py`


**Inherits from:** PositionSizer


### Methods


#### __init__()

Method description here.


#### calculate_size()

Method description here.



### Attributes



---

## src.bot.risk.core.unified_risk_manager.MarketRegimeDetector

Detects market regimes for dynamic risk adjustment

**File:** `src/bot\risk\core\unified_risk_manager.py`



### Methods


#### __init__()

Method description here.


#### detect_regime()

Method description here.



### Attributes



---

## src.bot.risk.core.unified_risk_manager.UnifiedRiskManager

Unified Risk Management System

Combines Australian tax-aware risk management with advanced algorithms
and dynamic risk adjustment capabilities.

**File:** `src/bot\risk\core\unified_risk_manager.py`



### Methods


#### __init__()

Method description here.


#### reload_configuration()

Method description here.


#### _determine_risk_level()

Method description here.


#### _apply_risk_limits()

Method description here.


#### _calculate_portfolio_returns()

Method description here.


#### _calculate_max_drawdown()

Method description here.


#### _calculate_sharpe_ratio()

Method description here.


#### _calculate_sortino_ratio()

Method description here.


#### _calculate_concentration_risk()

Method description here.


#### _calculate_portfolio_risk_score()

Method description here.


#### _generate_portfolio_alerts()

Method description here.


#### _empty_portfolio_metrics()

Method description here.


#### get_risk_summary()

Method description here.


#### should_reject_trade()

Method description here.



### Attributes



---

## src.bot.tax.example.ComprehensiveTaxExample

Comprehensive example of the tax system capabilities.

**File:** `src/bot\tax\example.py`



### Methods


#### __init__()

Method description here.


#### run_complete_example()

Method description here.


#### initialize_tax_system()

Method description here.


#### create_sample_trading_data()

Method description here.


#### process_transactions_with_analysis()

Method description here.


#### generate_tax_reports()

Method description here.


#### generate_specialized_reports()

Method description here.


#### demonstrate_real_time_analysis()

Method description here.


#### show_advanced_portfolio_analysis()

Method description here.


#### show_asset_specific_analysis()

Method description here.


#### export_tax_data()

Method description here.


#### show_system_status()

Method description here.



### Attributes



---

## src.bot.tax.tax_engine.AccountingMethod

Supported accounting methods for cost basis calculation.

**File:** `src/bot\tax\tax_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `FIFO`

- `LIFO`

- `SPECIFIC_ID`

- `AVERAGE_COST`

- `HIFO`


---

## src.bot.tax.tax_engine.TransactionType

Types of cryptocurrency transactions.

**File:** `src/bot\tax\tax_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BUY`

- `SELL`

- `TRANSFER_IN`

- `TRANSFER_OUT`

- `STAKE`

- `UNSTAKE`

- `REWARD`

- `AIRDROP`

- `FORK`

- `MINING`

- `DeFi_YIELD`

- `NFT_MINT`

- `NFT_SALE`


---

## src.bot.tax.tax_engine.TaxEventType

Types of taxable events.

**File:** `src/bot\tax\tax_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CAPITAL_GAIN`

- `CAPITAL_LOSS`

- `ORDINARY_INCOME`

- `WASH_SALE_ADJUSTMENT`

- `TAX_LOSS_HARVEST`


---

## src.bot.tax.tax_engine.HoldingPeriod

Classification of holding periods for tax purposes.

**File:** `src/bot\tax\tax_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SHORT_TERM`

- `LONG_TERM`


---

## src.bot.tax.tax_engine.Transaction

Represents a cryptocurrency transaction.

**File:** `src/bot\tax\tax_engine.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.bot.tax.tax_engine.TaxLot

Represents a tax lot (FIFO queue entry) for cost basis tracking.

**File:** `src/bot\tax\tax_engine.py`



### Methods


#### adjusted_cost_basis_per_unit()

Method description here.


#### adjusted_total_cost_basis()

Method description here.



### Attributes



---

## src.bot.tax.tax_engine.TaxEvent

Represents a taxable event.

**File:** `src/bot\tax\tax_engine.py`



### Methods


#### is_gain()

Method description here.


#### is_loss()

Method description here.



### Attributes



---

## src.bot.tax.tax_engine.WashSaleRule

Configuration for wash sale rule application.

**File:** `src/bot\tax\tax_engine.py`



### Methods



### Attributes



---

## src.bot.tax.tax_engine.TaxConfiguration

Tax calculation configuration and parameters.

**File:** `src/bot\tax\tax_engine.py`



### Methods


#### __init__()

Method description here.


#### get_tax_rate()

Method description here.



### Attributes



---

## src.bot.tax.tax_engine.TaxEngine

Advanced cryptocurrency tax calculation engine.

**File:** `src/bot\tax\tax_engine.py`



### Methods


#### __init__()

Method description here.


#### _init_database()

Method description here.


#### add_transaction()

Method description here.


#### _validate_transaction()

Method description here.


#### _process_acquisition()

Method description here.


#### _process_disposal()

Method description here.


#### _check_wash_sale()

Method description here.


#### _apply_wash_sale_adjustments()

Method description here.


#### calculate_tax_liability()

Method description here.


#### get_tax_loss_harvesting_opportunities()

Method description here.


#### _get_current_price()

Method description here.


#### _assess_wash_sale_risk()

Method description here.


#### _calculate_tax_savings()

Method description here.


#### _analyze_holding_periods()

Method description here.


#### _get_available_quantity()

Method description here.


#### _update_performance_metrics()

Method description here.


#### get_portfolio_summary()

Method description here.


#### _save_transaction_to_db()

Method description here.


#### _save_tax_lot_to_db()

Method description here.


#### _save_tax_event_to_db()

Method description here.


#### _save_wash_sale_adjustment()

Method description here.


#### _update_tax_lot_wash_sale()

Method description here.



### Attributes



---

## src.bot.tax.tax_optimizer.OptimizationStrategy

Tax optimization strategies.

**File:** `src/bot\tax\tax_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TAX_LOSS_HARVEST`

- `LONG_TERM_OPTIMIZATION`

- `WASH_SALE_AVOIDANCE`

- `REBALANCING_OPTIMIZATION`

- `MULTI_ASSET_HARVEST`

- `DYNAMIC_TAX_RATE`


---

## src.bot.tax.tax_optimizer.RiskTolerance

Risk tolerance levels for optimization.

**File:** `src/bot\tax\tax_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CONSERVATIVE`

- `MODERATE`

- `AGGRESSIVE`


---

## src.bot.tax.tax_optimizer.OptimizationObjective

Optimization objectives.

**File:** `src/bot\tax\tax_optimizer.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MAXIMIZE_TAX_SAVINGS`

- `MINIMIZE_WASH_SALES`

- `OPTIMIZE_HOLDING_PERIODS`

- `BALANCE_RISK_RETURN`


---

## src.bot.tax.tax_optimizer.TaxOptimizationConfig

Configuration for tax optimization system.

**File:** `src/bot\tax\tax_optimizer.py`



### Methods



### Attributes



---

## src.bot.tax.tax_optimizer.OptimizationRecommendation

Tax optimization recommendation.

**File:** `src/bot\tax\tax_optimizer.py`



### Methods



### Attributes



---

## src.bot.tax.tax_optimizer.HarvestingOpportunity

Tax loss harvesting opportunity.

**File:** `src/bot\tax\tax_optimizer.py`



### Methods



### Attributes



---

## src.bot.tax.tax_optimizer.TaxOptimizer

Advanced tax optimization and loss harvesting system.

**File:** `src/bot\tax\tax_optimizer.py`



### Methods


#### __init__()

Method description here.


#### _filter_and_rank_recommendations()

Method description here.


#### _calculate_effective_tax_rate()

Method description here.


#### _calculate_optimal_harvest_quantity()

Method description here.


#### _estimate_avg_holding_days()

Method description here.


#### _calculate_execution_priority()

Method description here.


#### _get_long_term_rate()

Method description here.


#### _assess_holding_risk()

Method description here.


#### _calculate_recommendation_confidence()

Method description here.


#### _assess_portfolio_impact()

Method description here.


#### _assess_tax_impact()

Method description here.


#### _update_optimization_history()

Method description here.


#### get_optimization_performance()

Method description here.



### Attributes



---

## src.bot.tax.tax_reporter.ReportFormat

Supported report formats.

**File:** `src/bot\tax\tax_reporter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PDF`

- `HTML`

- `CSV`

- `EXCEL`

- `JSON`


---

## src.bot.tax.tax_reporter.ReportType

Types of tax reports.

**File:** `src/bot\tax\tax_reporter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `FORM_8949`

- `SCHEDULE_D`

- `SCHEDULE_1`

- `FORM_8938`

- `COMPREHENSIVE`

- `TAX_SUMMARY`

- `AUDIT_TRAIL`

- `WASH_SALE_REPORT`

- `TAX_LOSS_HARVEST`


---

## src.bot.tax.tax_reporter.JurisdictionCompliance

Supported tax jurisdictions.

**File:** `src/bot\tax\tax_reporter.py`


**Inherits from:** Enum


### Methods



### Attributes


- `US_FEDERAL`

- `US_STATE`

- `CANADA`

- `UK`

- `AUSTRALIA`

- `GERMANY`

- `JAPAN`


---

## src.bot.tax.tax_reporter.ReportConfiguration

Configuration for tax report generation.

**File:** `src/bot\tax\tax_reporter.py`



### Methods



### Attributes



---

## src.bot.tax.tax_reporter.TaxReporter

Advanced tax reporting and compliance system.

**File:** `src/bot\tax\tax_reporter.py`



### Methods


#### __init__()

Method description here.


#### generate_report()

Method description here.


#### _gather_tax_data()

Method description here.


#### _generate_form_8949()

Method description here.


#### _format_8949_transactions()

Method description here.


#### _get_acquisition_date()

Method description here.


#### _calculate_8949_totals()

Method description here.


#### _generate_schedule_d()

Method description here.


#### _generate_comprehensive_report()

Method description here.


#### _create_executive_summary()

Method description here.


#### _create_transaction_summary()

Method description here.


#### _create_wash_sale_analysis()

Method description here.


#### _create_tax_optimization_analysis()

Method description here.


#### _create_compliance_checklist()

Method description here.


#### _create_supporting_documentation()

Method description here.


#### _generate_audit_trail()

Method description here.


#### _calculate_avg_holding_period()

Method description here.


#### _analyze_holding_periods()

Method description here.


#### _analyze_accounting_methods()

Method description here.


#### _check_fatca_requirement()

Method description here.


#### _create_form_8949_pdf()

Method description here.


#### _create_comprehensive_pdf()

Method description here.


#### _create_comprehensive_html()

Method description here.


#### export_to_csv()

Method description here.


#### export_to_excel()

Method description here.


#### generate_multi_year_comparison()

Method description here.


#### _calculate_trends()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.australian_trading_engine.ExecutionPriority

Execution priority levels

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `EMERGENCY`

- `HIGH`

- `MEDIUM`

- `LOW`

- `TAX_OPTIMIZATION`


---

## src.bot.trading_engine_integration.australian_trading_engine.TradeSource

Source of trading signal

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ML_STRATEGY`

- `ARBITRAGE`

- `RISK_MANAGEMENT`

- `TAX_OPTIMIZATION`

- `MANUAL`


---

## src.bot.trading_engine_integration.australian_trading_engine.AustralianTradeRequest

Enhanced trade request with Australian compliance data

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`



### Methods



### Attributes



---

## src.bot.trading_engine_integration.australian_trading_engine.AustralianExecutionResult

Enhanced execution result with Australian compliance tracking

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`



### Methods



### Attributes



---

## src.bot.trading_engine_integration.australian_trading_engine.AustralianOrderRouter

Smart order router optimized for Australian traders
Routes orders to appropriate exchanges considering costs and compliance

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`



### Methods


#### __init__()

Method description here.


#### select_optimal_exchange()

Method description here.


#### calculate_execution_costs()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.australian_trading_engine.AustralianComplianceExecutor

Execution engine with integrated Australian compliance checking
Ensures all trades comply with ATO, ASIC, and AUSTRAC requirements

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.australian_trading_engine.MLStrategyExecutor

Executor for ML strategy signals with Australian compliance

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.australian_trading_engine.ArbitrageExecutor

Executor for arbitrage opportunities with Australian compliance

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.australian_trading_engine.AustralianTradingEngineIntegration

Main integration engine coordinating ML strategies, arbitrage, and existing trading engine
with comprehensive Australian compliance

**File:** `src/bot\trading_engine_integration\australian_trading_engine.py`



### Methods


#### __init__()

Method description here.


#### get_integration_status()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.integration_coordinator.SystemConfiguration

Configuration for the Australian trading system

**File:** `src/bot\trading_engine_integration\integration_coordinator.py`



### Methods



### Attributes



---

## src.bot.trading_engine_integration.integration_coordinator.SystemStatus

Current status of the Australian trading system

**File:** `src/bot\trading_engine_integration\integration_coordinator.py`



### Methods



### Attributes



---

## src.bot.trading_engine_integration.integration_coordinator.AustralianTradingSystemCoordinator

Main coordinator for the comprehensive Australian cryptocurrency trading system
Orchestrates ML strategies, arbitrage detection, risk management, and compliance

**File:** `src/bot\trading_engine_integration\integration_coordinator.py`



### Methods


#### __init__()

Method description here.


#### request_shutdown()

Method description here.


#### get_system_status()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.signal_processing_manager.SignalConflictType

Types of signal conflicts

**File:** `src/bot\trading_engine_integration\signal_processing_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `OPPOSING_DIRECTIONS`

- `EXCESSIVE_EXPOSURE`

- `TIMING_CONFLICT`

- `RESOURCE_CONFLICT`

- `COMPLIANCE_CONFLICT`


---

## src.bot.trading_engine_integration.signal_processing_manager.SignalStatus

Status of signals in processing queue

**File:** `src/bot\trading_engine_integration\signal_processing_manager.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `PROCESSING`

- `EXECUTED`

- `REJECTED`

- `EXPIRED`

- `CONFLICTED`


---

## src.bot.trading_engine_integration.signal_processing_manager.ProcessingSignal

Wrapper for signals with processing metadata

**File:** `src/bot\trading_engine_integration\signal_processing_manager.py`



### Methods


#### is_expired()

Method description here.


#### get_signal_strength()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.signal_processing_manager.SignalConflictResolver

Resolves conflicts between different trading signals
Prioritizes based on Australian compliance, profitability, and risk

**File:** `src/bot\trading_engine_integration\signal_processing_manager.py`



### Methods


#### __init__()

Method description here.


#### detect_conflicts()

Method description here.


#### _detect_symbol_conflicts()

Method description here.


#### _detect_resource_conflicts()

Method description here.


#### _get_signal_direction()

Method description here.


#### resolve_conflicts()

Method description here.


#### _resolve_opposing_directions()

Method description here.


#### _resolve_excessive_exposure()

Method description here.


#### _resolve_timing_conflict()

Method description here.


#### _resolve_resource_conflict()

Method description here.


#### _resolve_compliance_conflict()

Method description here.



### Attributes



---

## src.bot.trading_engine_integration.signal_processing_manager.SignalProcessingManager

Manages the processing queue and coordination of ML and arbitrage signals
Handles conflict resolution, priority scheduling, and execution coordination

**File:** `src/bot\trading_engine_integration\signal_processing_manager.py`



### Methods


#### __init__()

Method description here.


#### _generate_signal_id()

Method description here.


#### add_ml_signals()

Method description here.


#### add_arbitrage_opportunities()

Method description here.


#### _cleanup_expired_signals()

Method description here.


#### get_queue_status()

Method description here.


#### get_execution_history()

Method description here.



### Attributes



---

## src.bot.utils.logging.JSONFormatter

Custom JSON formatter for structured logging.

**File:** `src/bot\utils\logging.py`



### Methods


#### __init__()

Method description here.


#### format()

Method description here.



### Attributes



---

## src.bot.utils.logging.TradingLogger

Enhanced logger with trading-specific context.

This class provides convenient methods for logging trading-related
events with automatic context injection.

**File:** `src/bot\utils\logging.py`



### Methods


#### __init__()

Method description here.


#### bind()

Method description here.


#### _log()

Method description here.


#### debug()

Method description here.


#### info()

Method description here.


#### warning()

Method description here.


#### error()

Method description here.


#### critical()

Method description here.


#### trade_executed()

Method description here.


#### signal_generated()

Method description here.


#### risk_limit_hit()

Method description here.


#### strategy_performance()

Method description here.


#### mode_switch()

Method description here.



### Attributes



---

## src.bot.utils.rate_limiter.RateLimiter

Token bucket rate limiter for API requests.

Features:
- Token bucket algorithm with burst support
- Configurable time windows and request limits
- Async/await support for non-blocking operation
- Per-endpoint rate limiting support
- Request queue management

**File:** `src/bot\utils\rate_limiter.py`



### Methods


#### __init__()

Method description here.


#### _refill_tokens()

Method description here.


#### _consume_tokens()

Method description here.


#### _calculate_wait_time()

Method description here.


#### get_current_usage()

Method description here.



### Attributes



---

## src.bot.utils.rate_limiter.MultiEndpointRateLimiter

Rate limiter that supports different limits for different endpoints.

Useful for exchanges that have different rate limits for different
types of API calls (e.g., market data vs trading operations).

**File:** `src/bot\utils\rate_limiter.py`



### Methods


#### __init__()

Method description here.


#### _get_limiter_for_endpoint()

Method description here.


#### get_all_usage_stats()

Method description here.



### Attributes



---

## src.config.secure_storage.SecureConfigManager

Secure configuration manager with AES-256-GCM encryption
Addresses audit finding: Configuration encryption at rest

**File:** `src/config\secure_storage.py`



### Methods


#### __init__()

Method description here.


#### _initialize_encryption_key()

Method description here.


#### _derive_key_from_password()

Method description here.


#### _save_key()

Method description here.


#### encrypt_config()

Method description here.


#### decrypt_config()

Method description here.


#### save_secure_config()

Method description here.


#### load_secure_config()

Method description here.


#### update_config_field()

Method description here.


#### get_config_field()

Method description here.


#### delete_config_field()

Method description here.


#### rotate_encryption_key()

Method description here.


#### verify_integrity()

Method description here.


#### get_security_info()

Method description here.



### Attributes



---

## src.config.secure_storage.SecureEnvironmentManager

Manage environment variables securely
Complement to SecureConfigManager for runtime secrets

**File:** `src/config\secure_storage.py`



### Methods


#### get_secure_env()

Method description here.


#### set_secure_env()

Method description here.


#### clear_secure_env()

Method description here.



### Attributes



---

## src.dashboard.backend.config.DatabaseSettings

Database configuration

**File:** `src/dashboard\backend\config.py`


**Inherits from:** BaseSettings


### Methods



### Attributes



---

## src.dashboard.backend.config.APISettings

API configuration

**File:** `src/dashboard\backend\config.py`


**Inherits from:** BaseSettings


### Methods



### Attributes



---

## src.dashboard.backend.config.WebSocketSettings

WebSocket configuration

**File:** `src/dashboard\backend\config.py`


**Inherits from:** BaseSettings


### Methods



### Attributes



---

## src.dashboard.backend.config.MonitoringSettings

Monitoring and alerting configuration

**File:** `src/dashboard\backend\config.py`


**Inherits from:** BaseSettings


### Methods



### Attributes



---

## src.dashboard.backend.config.MLSettings

ML component configuration

**File:** `src/dashboard\backend\config.py`


**Inherits from:** BaseSettings


### Methods



### Attributes



---

## src.dashboard.backend.config.SecuritySettings

Security configuration

**File:** `src/dashboard\backend\config.py`


**Inherits from:** BaseSettings


### Methods



### Attributes



---

## src.dashboard.backend.config.DashboardSettings

Main dashboard settings container

**File:** `src/dashboard\backend\config.py`


**Inherits from:** BaseSettings


### Methods


#### __init__()

Method description here.



### Attributes



---

## src.dashboard.backend.config.Config

No description available.

**File:** `src/dashboard\backend\config.py`



### Methods



### Attributes


- `env_file`

- `env_file_encoding`


---

## src.dashboard.backend.database.DatabaseConfig

Database configuration

**File:** `src/dashboard\backend\database.py`



### Methods



### Attributes



---

## src.dashboard.backend.database.DatabaseManager

Manages database connections and operations for dashboard

**File:** `src/dashboard\backend\database.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.dashboard.backend.integration.Phase1Integration

Integration with Phase 1 execution optimization components

**File:** `src/dashboard\backend\integration.py`



### Methods


#### __init__()

Method description here.


#### _get_mock_data()

Method description here.



### Attributes



---

## src.dashboard.backend.integration.Phase2Integration

Integration with Phase 2 advanced ML components

**File:** `src/dashboard\backend\integration.py`



### Methods


#### __init__()

Method description here.


#### _get_mock_insights()

Method description here.



### Attributes



---

## src.dashboard.backend.main.DashboardBackend

Main dashboard backend application

**File:** `src/dashboard\backend\main.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.dashboard.backend.monitoring.PerformanceMetric

Individual performance metric

**File:** `src/dashboard\backend\monitoring.py`



### Methods



### Attributes



---

## src.dashboard.backend.monitoring.SystemHealth

System health snapshot

**File:** `src/dashboard\backend\monitoring.py`



### Methods



### Attributes



---

## src.dashboard.backend.monitoring.PerformanceMonitor

Advanced performance monitoring system

**File:** `src/dashboard\backend\monitoring.py`



### Methods


#### __init__()

Method description here.


#### record_request()

Method description here.


#### record_component_response_time()

Method description here.


#### get_uptime()

Method description here.



### Attributes



---

## src.dashboard.backend.websocket.WebSocketManager

Manages WebSocket connections and real-time data broadcasting

**File:** `src/dashboard\backend\websocket.py`



### Methods


#### __init__()

Method description here.


#### get_connection_count()

Method description here.



### Attributes



---

## src.dashboard.backend.routers.analytics_router.AnalyticsService

Service for analytics operations

**File:** `src/dashboard\backend\routers\analytics_router.py`



### Methods



### Attributes



---

## src.dashboard.backend.routers.health_router.HealthService

Service for health monitoring operations

**File:** `src/dashboard\backend\routers\health_router.py`



### Methods



### Attributes



---

## src.dashboard.backend.routers.ml_router.MLService

Service for ML operations and insights

**File:** `src/dashboard\backend\routers\ml_router.py`



### Methods



### Attributes



---

## src.dashboard.backend.routers.trading_router.TradingDataService

Service for trading data operations

**File:** `src/dashboard\backend\routers\trading_router.py`



### Methods



### Attributes



---

## src.deployment.deployment_manager.DeploymentManager

Production Deployment Manager

Handles complete deployment automation including containerization,
orchestration, and production configuration.

**File:** `src/deployment\deployment_manager.py`



### Methods


#### __init__()

Method description here.


#### create_docker_files()

Method description here.


#### create_kubernetes_manifests()

Method description here.


#### create_deployment_scripts()

Method description here.


#### create_monitoring_config()

Method description here.


#### create_nginx_config()

Method description here.


#### create_environment_template()

Method description here.


#### create_database_init()

Method description here.


#### generate_deployment_package()

Method description here.


#### create_deployment_guide()

Method description here.



### Attributes



---

## src.documentation.knowledge_base.SimpleLogger

No description available.

**File:** `src/documentation\knowledge_base.py`



### Methods


#### __init__()

Method description here.


#### bind()

Method description here.


#### info()

Method description here.


#### error()

Method description here.



### Attributes



---

## src.documentation.knowledge_base.DocumentationType

Types of documentation

**File:** `src/documentation\knowledge_base.py`


**Inherits from:** Enum


### Methods



### Attributes


- `API_REFERENCE`

- `USER_GUIDE`

- `DEVELOPER_GUIDE`

- `TROUBLESHOOTING`

- `TUTORIAL`

- `FAQ`

- `CHANGELOG`

- `ARCHITECTURE`


---

## src.documentation.knowledge_base.ContentFormat

Documentation output formats

**File:** `src/documentation\knowledge_base.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MARKDOWN`

- `HTML`

- `PDF`

- `JSON`

- `DOCX`

- `CONFLUENCE`


---

## src.documentation.knowledge_base.SearchResultType

Search result types

**File:** `src/documentation\knowledge_base.py`


**Inherits from:** Enum


### Methods



### Attributes


- `DOCUMENTATION`

- `CODE_EXAMPLE`

- `API_ENDPOINT`

- `TROUBLESHOOTING`

- `FAQ`


---

## src.documentation.knowledge_base.DocumentationPage

Individual documentation page

**File:** `src/documentation\knowledge_base.py`



### Methods



### Attributes



---

## src.documentation.knowledge_base.APIEndpoint

API endpoint documentation

**File:** `src/documentation\knowledge_base.py`



### Methods



### Attributes



---

## src.documentation.knowledge_base.TroubleshootingEntry

Troubleshooting knowledge base entry

**File:** `src/documentation\knowledge_base.py`



### Methods



### Attributes



---

## src.documentation.knowledge_base.CodeAnalyzer

Analyze code for automatic documentation generation

**File:** `src/documentation\knowledge_base.py`



### Methods


#### __init__()

Method description here.


#### analyze_codebase()

Method description here.


#### _analyze_python_file()

Method description here.


#### _get_module_name()

Method description here.


#### _extract_imports()

Method description here.


#### _analyze_class()

Method description here.


#### _analyze_function()

Method description here.


#### _extract_class_attributes()

Method description here.


#### _extract_return_annotation()

Method description here.


#### _is_api_endpoint()

Method description here.


#### _extract_api_endpoint()

Method description here.



### Attributes



---

## src.documentation.knowledge_base.DocumentationGenerator

Generate documentation from code analysis and templates

**File:** `src/documentation\knowledge_base.py`



### Methods


#### __init__()

Method description here.


#### _create_default_templates()

Method description here.


#### generate_api_documentation()

Method description here.


#### _generate_api_docs_simple()

Method description here.


#### generate_user_guide()

Method description here.


#### _generate_user_guide_simple()

Method description here.


#### generate_class_reference()

Method description here.


#### _generate_class_reference_simple()

Method description here.


#### generate_troubleshooting_guide()

Method description here.


#### _generate_troubleshooting_simple()

Method description here.


#### convert_to_html()

Method description here.


#### _simple_markdown_to_html()

Method description here.



### Attributes



---

## src.documentation.knowledge_base.SearchEngine

Search engine for documentation and knowledge base (simplified in-memory implementation)

**File:** `src/documentation\knowledge_base.py`



### Methods


#### __init__()

Method description here.


#### index_document()

Method description here.


#### search()

Method description here.


#### get_suggestions()

Method description here.



### Attributes



---

## src.documentation.knowledge_base.InteractiveExampleGenerator

Generate interactive code examples

**File:** `src/documentation\knowledge_base.py`



### Methods


#### __init__()

Method description here.


#### generate_trading_examples()

Method description here.


#### generate_configuration_examples()

Method description here.



### Attributes



---

## src.documentation.knowledge_base.DocumentationPlatform

Main documentation platform orchestrator

**File:** `src/documentation\knowledge_base.py`



### Methods


#### __init__()

Method description here.


#### _initialize_troubleshooting_entries()

Method description here.


#### _index_documentation_for_search()

Method description here.


#### search_documentation()

Method description here.


#### get_documentation_stats()

Method description here.



### Attributes



---

## src.ml.pipeline_enhancement.ModelType

Types of ML models

**File:** `src/ml\pipeline_enhancement.py`


**Inherits from:** Enum


### Methods



### Attributes


- `TRADITIONAL`

- `ENSEMBLE`

- `NEURAL_NETWORK`

- `DEEP_LEARNING`

- `TRANSFORMER`


---

## src.ml.pipeline_enhancement.OptimizationObjective

Optimization objectives

**File:** `src/ml\pipeline_enhancement.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ACCURACY`

- `PRECISION`

- `RECALL`

- `F1_SCORE`

- `ROC_AUC`

- `RMSE`

- `MAE`

- `CUSTOM`


---

## src.ml.pipeline_enhancement.TrainingStrategy

Training strategies

**File:** `src/ml\pipeline_enhancement.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SINGLE_THREAD`

- `MULTI_THREAD`

- `MULTI_PROCESS`

- `DISTRIBUTED`

- `GPU_ACCELERATED`


---

## src.ml.pipeline_enhancement.ModelConfig

Model configuration

**File:** `src/ml\pipeline_enhancement.py`



### Methods



### Attributes



---

## src.ml.pipeline_enhancement.TrainingResult

Training result with metrics

**File:** `src/ml\pipeline_enhancement.py`



### Methods



### Attributes



---

## src.ml.pipeline_enhancement.NeuralArchitectureSearch

Advanced Neural Architecture Search using evolutionary algorithms

**File:** `src/ml\pipeline_enhancement.py`



### Methods


#### __init__()

Method description here.


#### initialize_population()

Method description here.


#### _generate_random_architecture()

Method description here.


#### evaluate_architecture()

Method description here.


#### _evaluate_pytorch_architecture()

Method description here.


#### _build_pytorch_model()

Method description here.


#### _evaluate_sklearn_architecture()

Method description here.


#### evolve_population()

Method description here.


#### _tournament_selection()

Method description here.


#### _crossover()

Method description here.


#### _mutate()

Method description here.


#### search()

Method description here.



### Attributes



---

## src.ml.pipeline_enhancement.HyperparameterOptimizer

Advanced hyperparameter optimization using Bayesian methods

**File:** `src/ml\pipeline_enhancement.py`



### Methods


#### __init__()

Method description here.


#### optimize()

Method description here.


#### get_optimization_results()

Method description here.



### Attributes



---

## src.ml.pipeline_enhancement.AdvancedFeatureEngineering

Automated feature engineering and selection

**File:** `src/ml\pipeline_enhancement.py`



### Methods


#### __init__()

Method description here.


#### engineer_features()

Method description here.


#### _is_trading_data()

Method description here.


#### _add_trading_features()

Method description here.


#### _has_datetime_columns()

Method description here.


#### _add_time_features()

Method description here.


#### _add_statistical_features()

Method description here.


#### _add_interaction_features()

Method description here.


#### _add_polynomial_features()

Method description here.


#### select_features()

Method description here.



### Attributes



---

## src.ml.pipeline_enhancement.ModelEnsemble

Advanced model ensemble with dynamic weighting

**File:** `src/ml\pipeline_enhancement.py`



### Methods


#### __init__()

Method description here.


#### add_model()

Method description here.


#### fit()

Method description here.


#### predict()

Method description here.


#### update_weights()

Method description here.


#### get_model_weights()

Method description here.



### Attributes



---

## src.ml.pipeline_enhancement.MLPipelineOrchestrator

Main ML pipeline orchestrator

**File:** `src/ml\pipeline_enhancement.py`



### Methods


#### __init__()

Method description here.


#### _init_database()

Method description here.


#### train_comprehensive_pipeline()

Method description here.


#### _train_with_optimization()

Method description here.


#### _get_param_space()

Method description here.


#### _create_model_with_params()

Method description here.


#### _run_neural_architecture_search()

Method description here.


#### _calculate_improvement()

Method description here.


#### _calculate_time_reduction()

Method description here.


#### _store_training_result()

Method description here.



### Attributes



---

## src.monitoring.comprehensive_health_monitor.HealthStatus

System health status levels

**File:** `src/monitoring\comprehensive_health_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HEALTHY`

- `WARNING`

- `CRITICAL`

- `UNKNOWN`


---

## src.monitoring.comprehensive_health_monitor.MetricType

Metric data types

**File:** `src/monitoring\comprehensive_health_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `GAUGE`

- `COUNTER`

- `HISTOGRAM`

- `SUMMARY`


---

## src.monitoring.comprehensive_health_monitor.AlertLevel

Alert severity levels

**File:** `src/monitoring\comprehensive_health_monitor.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `CRITICAL`

- `EMERGENCY`


---

## src.monitoring.comprehensive_health_monitor.Metric

System metric data structure

**File:** `src/monitoring\comprehensive_health_monitor.py`



### Methods



### Attributes



---

## src.monitoring.comprehensive_health_monitor.HealthCheck

Health check configuration

**File:** `src/monitoring\comprehensive_health_monitor.py`



### Methods



### Attributes



---

## src.monitoring.comprehensive_health_monitor.Alert

Alert information

**File:** `src/monitoring\comprehensive_health_monitor.py`



### Methods



### Attributes



---

## src.monitoring.comprehensive_health_monitor.SystemStatus

Overall system status

**File:** `src/monitoring\comprehensive_health_monitor.py`



### Methods



### Attributes



---

## src.monitoring.comprehensive_health_monitor.MetricsCollector

Advanced metrics collection system

Collects and stores system, application, and trading-specific metrics
with configurable retention and aggregation.

**File:** `src/monitoring\comprehensive_health_monitor.py`



### Methods


#### __init__()

Method description here.


#### record_metric()

Method description here.


#### get_metric_history()

Method description here.


#### get_current_metrics()

Method description here.


#### collect_system_metrics()

Method description here.


#### collect_trading_metrics()

Method description here.


#### collect_application_metrics()

Method description here.


#### get_metric_statistics()

Method description here.



### Attributes



---

## src.monitoring.comprehensive_health_monitor.HealthCheckManager

Advanced health check management system

Manages multiple health checks with failure tracking,
automatic recovery detection, and escalation policies.

**File:** `src/monitoring\comprehensive_health_monitor.py`



### Methods


#### __init__()

Method description here.


#### register_health_check()

Method description here.


#### _handle_check_success()

Method description here.


#### _handle_check_failure()

Method description here.


#### get_overall_status()

Method description here.


#### get_status_summary()

Method description here.



### Attributes



---

## src.monitoring.comprehensive_health_monitor.AlertManager

Intelligent alerting system with escalation policies

Manages alerts with different severity levels, escalation policies,
and multiple notification channels.

**File:** `src/monitoring\comprehensive_health_monitor.py`



### Methods


#### __init__()

Method description here.


#### _setup_notification_channels()

Method description here.


#### get_active_alerts()

Method description here.


#### get_alert_summary()

Method description here.



### Attributes



---

## src.monitoring.comprehensive_health_monitor.ComprehensiveHealthMonitor

Master health monitoring system

Coordinates all monitoring components and provides unified
system health status with intelligent analysis and automated responses.

**File:** `src/monitoring\comprehensive_health_monitor.py`



### Methods


#### __init__()

Method description here.


#### _setup_default_health_checks()

Method description here.


#### _check_cpu_usage()

Method description here.


#### _check_memory_usage()

Method description here.


#### _check_disk_usage()

Method description here.


#### get_comprehensive_status()

Method description here.


#### get_monitoring_summary()

Method description here.



### Attributes



---

## src.performance.optimization_engine.PerformanceLevel

Performance optimization levels

**File:** `src/performance\optimization_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `CONSERVATIVE`

- `BALANCED`

- `AGGRESSIVE`

- `EXTREME`


---

## src.performance.optimization_engine.CacheStrategy

Cache eviction strategies

**File:** `src/performance\optimization_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LRU`

- `LFU`

- `TTL`

- `ADAPTIVE`


---

## src.performance.optimization_engine.MetricType

Types of performance metrics

**File:** `src/performance\optimization_engine.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LATENCY`

- `THROUGHPUT`

- `MEMORY`

- `CPU`

- `CACHE`

- `ERROR_RATE`


---

## src.performance.optimization_engine.PerformanceMetric

Performance metric data point

**File:** `src/performance\optimization_engine.py`



### Methods



### Attributes



---

## src.performance.optimization_engine.CacheConfig

Cache configuration settings

**File:** `src/performance\optimization_engine.py`



### Methods



### Attributes



---

## src.performance.optimization_engine.ConnectionPoolConfig

Connection pool configuration

**File:** `src/performance\optimization_engine.py`



### Methods



### Attributes



---

## src.performance.optimization_engine.AdvancedCache

High-performance adaptive cache with compression and persistence

**File:** `src/performance\optimization_engine.py`



### Methods


#### __init__()

Method description here.


#### get()

Method description here.


#### set()

Method description here.


#### delete()

Method description here.


#### _update_access_stats()

Method description here.


#### _evict_items()

Method description here.


#### get_stats()

Method description here.


#### clear()

Method description here.



### Attributes



---

## src.performance.optimization_engine.ConnectionPool

High-performance connection pool with health monitoring

**File:** `src/performance\optimization_engine.py`



### Methods


#### __init__()

Method description here.


#### _update_connection_stats()

Method description here.


#### get_stats()

Method description here.



### Attributes



---

## src.performance.optimization_engine.PerformanceMonitor

Comprehensive performance monitoring system

**File:** `src/performance\optimization_engine.py`



### Methods


#### __init__()

Method description here.


#### _init_database()

Method description here.


#### _init_prometheus()

Method description here.


#### record_metric()

Method description here.


#### _check_alerts()

Method description here.


#### get_metrics_summary()

Method description here.


#### get_system_metrics()

Method description here.



### Attributes



---

## src.performance.optimization_engine.PerformanceOptimizer

Main performance optimization orchestrator

**File:** `src/performance\optimization_engine.py`



### Methods


#### __init__()

Method description here.


#### _apply_optimizations()

Method description here.


#### performance_timer()

Method description here.


#### cached_method()

Method description here.


#### get_connection_pool()

Method description here.


#### get_performance_report()

Method description here.



### Attributes



---

## src.reliability.chaos_engineering.FaultType

Types of faults that can be injected

**File:** `src/reliability\chaos_engineering.py`


**Inherits from:** Enum


### Methods



### Attributes


- `NETWORK_LATENCY`

- `NETWORK_PARTITION`

- `NETWORK_LOSS`

- `CPU_STRESS`

- `MEMORY_STRESS`

- `DISK_STRESS`

- `SERVICE_FAILURE`

- `DATABASE_FAILURE`

- `API_SLOWDOWN`

- `TIMEOUT_INJECTION`


---

## src.reliability.chaos_engineering.FaultSeverity

Severity levels for fault injection

**File:** `src/reliability\chaos_engineering.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW`

- `MEDIUM`

- `HIGH`

- `CRITICAL`


---

## src.reliability.chaos_engineering.SystemHealth

System health states

**File:** `src/reliability\chaos_engineering.py`


**Inherits from:** Enum


### Methods



### Attributes


- `HEALTHY`

- `DEGRADED`

- `FAILING`

- `CRITICAL`

- `RECOVERING`


---

## src.reliability.chaos_engineering.FaultInjectionConfig

Configuration for fault injection

**File:** `src/reliability\chaos_engineering.py`



### Methods



### Attributes



---

## src.reliability.chaos_engineering.ReliabilityMetric

Reliability measurement data point

**File:** `src/reliability\chaos_engineering.py`



### Methods



### Attributes



---

## src.reliability.chaos_engineering.FailureEvent

Record of a system failure event

**File:** `src/reliability\chaos_engineering.py`



### Methods



### Attributes



---

## src.reliability.chaos_engineering.FaultInjector

Advanced fault injection system

**File:** `src/reliability\chaos_engineering.py`



### Methods


#### __init__()

Method description here.


#### _setup_fault_executors()

Method description here.


#### _setup_recovery_strategies()

Method description here.


#### get_active_faults()

Method description here.


#### get_fault_history()

Method description here.



### Attributes



---

## src.reliability.chaos_engineering.ReliabilityMonitor

Comprehensive system reliability monitoring

**File:** `src/reliability\chaos_engineering.py`



### Methods


#### __init__()

Method description here.


#### _init_database()

Method description here.


#### _setup_default_health_checks()

Method description here.


#### add_circuit_breaker()

Method description here.


#### get_reliability_summary()

Method description here.



### Attributes



---

## src.reliability.chaos_engineering.CircuitBreaker

Circuit breaker implementation for fault tolerance

**File:** `src/reliability\chaos_engineering.py`



### Methods


#### __init__()

Method description here.


#### call()

Method description here.


#### is_open()

Method description here.


#### reset()

Method description here.



### Attributes



---

## src.reliability.chaos_engineering.MTBFTracker

Mean Time Between Failures tracker

**File:** `src/reliability\chaos_engineering.py`



### Methods


#### __init__()

Method description here.


#### record_failure()

Method description here.


#### get_current_mtbf()

Method description here.



### Attributes



---

## src.reliability.chaos_engineering.AvailabilityTracker

System availability tracker

**File:** `src/reliability\chaos_engineering.py`



### Methods


#### __init__()

Method description here.


#### record_downtime_start()

Method description here.


#### record_downtime_end()

Method description here.


#### get_current_availability()

Method description here.



### Attributes



---

## src.reliability.chaos_engineering.ChaosEngineeringOrchestrator

Main chaos engineering orchestrator

**File:** `src/reliability\chaos_engineering.py`



### Methods


#### __init__()

Method description here.


#### _load_test_scenarios()

Method description here.


#### _evaluate_experiment_success()

Method description here.


#### get_reliability_report()

Method description here.



### Attributes



---

## src.security.advanced_key_management.HSMType

Supported HSM types

**File:** `src/security\advanced_key_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `SOFTWARE`

- `PKCS11`

- `AZURE_KEY_VAULT`

- `AWS_KMS`

- `HASHICORP_VAULT`


---

## src.security.advanced_key_management.KeyStatus

Key lifecycle status

**File:** `src/security\advanced_key_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ACTIVE`

- `PENDING_ROTATION`

- `ROTATED`

- `COMPROMISED`

- `EXPIRED`

- `REVOKED`


---

## src.security.advanced_key_management.AuditEventType

Audit event types

**File:** `src/security\advanced_key_management.py`


**Inherits from:** Enum


### Methods



### Attributes


- `KEY_CREATED`

- `KEY_ACCESSED`

- `KEY_ROTATED`

- `KEY_COMPROMISED`

- `KEY_REVOKED`

- `HSM_CONNECTION`

- `UNAUTHORIZED_ACCESS`

- `POLICY_VIOLATION`


---

## src.security.advanced_key_management.KeyMetadata

Key metadata structure

**File:** `src/security\advanced_key_management.py`



### Methods



### Attributes



---

## src.security.advanced_key_management.AuditEvent

Audit event structure

**File:** `src/security\advanced_key_management.py`



### Methods



### Attributes



---

## src.security.advanced_key_management.HSMConfig

HSM configuration

**File:** `src/security\advanced_key_management.py`



### Methods



### Attributes



---

## src.security.advanced_key_management.HSMInterface

Abstract HSM interface

**File:** `src/security\advanced_key_management.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.security.advanced_key_management.SoftwareHSM

Software-based HSM implementation for development/testing

**File:** `src/security\advanced_key_management.py`


**Inherits from:** HSMInterface


### Methods


#### __init__()

Method description here.



### Attributes



---

## src.security.advanced_key_management.AuditTrail

Immutable audit trail system

**File:** `src/security\advanced_key_management.py`



### Methods


#### __init__()

Method description here.


#### _init_database()

Method description here.


#### _calculate_hash()

Method description here.



### Attributes



---

## src.security.advanced_key_management.AdvancedKeyManager

Advanced Key Management System
Enterprise-grade key management with HSM integration

**File:** `src/security\advanced_key_management.py`



### Methods


#### __init__()

Method description here.


#### _create_hsm_client()

Method description here.


#### _init_metadata_db()

Method description here.



### Attributes



---

## src.security.api_validator.ExchangeType

Supported exchange types

**File:** `src/security\api_validator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BYBIT`

- `BINANCE`

- `OKX`

- `COINBASE`

- `KUCOIN`


---

## src.security.api_validator.PermissionLevel

API permission levels

**File:** `src/security\api_validator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `READ`

- `TRADE`

- `WITHDRAW`

- `FUTURES`

- `MARGIN`


---

## src.security.api_validator.APIKeyInfo

API key information structure

**File:** `src/security\api_validator.py`



### Methods



### Attributes



---

## src.security.api_validator.ValidationResult

API key validation result

**File:** `src/security\api_validator.py`



### Methods



### Attributes



---

## src.security.api_validator.APIKeyValidator

Comprehensive API key validation system
Validates keys with real exchange API calls

**File:** `src/security\api_validator.py`



### Methods


#### __init__()

Method description here.


#### _create_exchange_client()

Method description here.


#### _is_cached_result_valid()

Method description here.


#### clear_validation_cache()

Method description here.



### Attributes


- `REQUIRED_PERMISSIONS`

- `EXCHANGE_CONFIG`


---

## src.security.api_validator.PermissionManager

API permission management and monitoring
Addresses audit finding: Permission validation missing

**File:** `src/security\api_validator.py`



### Methods


#### __init__()

Method description here.


#### get_permission_requirements()

Method description here.


#### get_permission_history()

Method description here.



### Attributes



---

## src.security.threat_detection.ThreatLevel

Threat severity levels

**File:** `src/security\threat_detection.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOW`

- `MEDIUM`

- `HIGH`

- `CRITICAL`


---

## src.security.threat_detection.ThreatType

Types of threats

**File:** `src/security\threat_detection.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BRUTE_FORCE`

- `ANOMALOUS_LOGIN`

- `SUSPICIOUS_API_USAGE`

- `GEOGRAPHIC_ANOMALY`

- `TRADING_PATTERN_DEVIATION`

- `UNAUTHORIZED_ACCESS`

- `DATA_EXFILTRATION`

- `MALICIOUS_IP`

- `TIME_BASED_ANOMALY`

- `VOLUME_ANOMALY`


---

## src.security.threat_detection.ResponseAction

Automated response actions

**File:** `src/security\threat_detection.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOG_ONLY`

- `RATE_LIMIT`

- `BLOCK_IP`

- `LOCK_ACCOUNT`

- `ALERT_ADMIN`

- `EMERGENCY_SHUTDOWN`


---

## src.security.threat_detection.ThreatEvent

Threat event structure

**File:** `src/security\threat_detection.py`



### Methods



### Attributes



---

## src.security.threat_detection.UserBehaviorProfile

User behavior profile for anomaly detection

**File:** `src/security\threat_detection.py`



### Methods



### Attributes



---

## src.security.threat_detection.IPIntelligence

IP address intelligence data

**File:** `src/security\threat_detection.py`



### Methods



### Attributes



---

## src.security.threat_detection.BehaviorAnalyzer

Behavioral analytics engine

**File:** `src/security\threat_detection.py`



### Methods


#### __init__()

Method description here.


#### _count_recent_api_calls()

Method description here.



### Attributes



---

## src.security.threat_detection.GeographicAnalyzer

Geographic threat analysis

**File:** `src/security\threat_detection.py`



### Methods


#### __init__()

Method description here.


#### _init_geoip()

Method description here.



### Attributes



---

## src.security.threat_detection.ThreatIntelligence

Threat intelligence integration

**File:** `src/security\threat_detection.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.security.threat_detection.AutomatedResponse

Automated threat response system

**File:** `src/security\threat_detection.py`



### Methods


#### __init__()

Method description here.


#### is_ip_blocked()

Method description here.


#### is_ip_rate_limited()

Method description here.


#### is_account_locked()

Method description here.



### Attributes



---

## src.security.threat_detection.ThreatDetectionEngine

Main threat detection engine
Coordinates all threat detection components

**File:** `src/security\threat_detection.py`



### Methods


#### __init__()

Method description here.


#### _init_database()

Method description here.



### Attributes



---

## src.security.zero_trust.TrustLevel

Trust levels for zero trust evaluation

**File:** `src/security\zero_trust.py`


**Inherits from:** Enum


### Methods



### Attributes


- `UNTRUSTED`

- `LOW`

- `MEDIUM`

- `HIGH`

- `CRITICAL`


---

## src.security.zero_trust.AccessDecision

Access control decisions

**File:** `src/security\zero_trust.py`


**Inherits from:** Enum


### Methods



### Attributes


- `ALLOW`

- `DENY`

- `CHALLENGE`

- `MONITOR`


---

## src.security.zero_trust.ResourceType

Types of protected resources

**File:** `src/security\zero_trust.py`


**Inherits from:** Enum


### Methods



### Attributes


- `API_ENDPOINT`

- `DATABASE`

- `FILE_SYSTEM`

- `NETWORK_SEGMENT`

- `CONFIGURATION`

- `CRYPTOGRAPHIC_KEY`


---

## src.security.zero_trust.Identity

Represents an authenticated identity

**File:** `src/security\zero_trust.py`



### Methods



### Attributes



---

## src.security.zero_trust.AccessRequest

Represents a resource access request

**File:** `src/security\zero_trust.py`



### Methods



### Attributes



---

## src.security.zero_trust.PolicyRule

Zero trust policy rule

**File:** `src/security\zero_trust.py`



### Methods



### Attributes



---

## src.security.zero_trust.ContinuousAuthenticator

Manages continuous authentication and trust scoring

**File:** `src/security\zero_trust.py`



### Methods


#### __init__()

Method description here.


#### _init_database()

Method description here.


#### _generate_device_fingerprint()

Method description here.


#### _calculate_time_decay()

Method description here.



### Attributes



---

## src.security.zero_trust.PolicyEngine

Zero trust policy evaluation engine

**File:** `src/security\zero_trust.py`



### Methods


#### __init__()

Method description here.


#### add_policy()

Method description here.


#### remove_policy()

Method description here.


#### _find_applicable_policies()

Method description here.


#### _matches_pattern()

Method description here.


#### _check_time_range()

Method description here.


#### _check_ip_range()

Method description here.


#### _generate_cache_key()

Method description here.



### Attributes



---

## src.security.zero_trust.MicroSegmentation

Network micro-segmentation manager

**File:** `src/security\zero_trust.py`



### Methods


#### __init__()

Method description here.


#### create_segment()

Method description here.


#### assign_to_segment()

Method description here.


#### allow_communication()

Method description here.


#### deny_communication()

Method description here.


#### can_communicate()

Method description here.



### Attributes



---

## src.security.zero_trust.ZeroTrustOrchestrator

Main zero trust architecture orchestrator

**File:** `src/security\zero_trust.py`



### Methods


#### __init__()

Method description here.


#### _setup_audit_logging()

Method description here.


#### _setup_default_policies()

Method description here.


#### _setup_default_segments()

Method description here.


#### _determine_segment()

Method description here.


#### _determine_resource_type()

Method description here.


#### get_trust_metrics()

Method description here.



### Attributes



---

## src.services.orchestrator.ServiceStatus

Service status enumeration

**File:** `src/services\orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `STOPPED`

- `STARTING`

- `RUNNING`

- `STOPPING`

- `ERROR`

- `CIRCUIT_BREAKER_OPEN`


---

## src.services.orchestrator.TradingMode

Trading mode enumeration

**File:** `src/services\orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PAPER`

- `LIVE`

- `BACKTEST`


---

## src.services.orchestrator.OrderType

Order type enumeration

**File:** `src/services\orchestrator.py`


**Inherits from:** Enum


### Methods



### Attributes


- `MARKET`

- `LIMIT`

- `STOP_LOSS`

- `TAKE_PROFIT`


---

## src.services.orchestrator.TradingSignal

Trading signal structure

**File:** `src/services\orchestrator.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.services.orchestrator.RiskAssessment

Risk assessment result

**File:** `src/services\orchestrator.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.services.orchestrator.ExecutionResult

Trade execution result

**File:** `src/services\orchestrator.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.services.orchestrator.CircuitBreaker

Circuit breaker pattern implementation for service protection

**File:** `src/services\orchestrator.py`



### Methods


#### __init__()

Method description here.


#### call()

Method description here.


#### _should_attempt_reset()

Method description here.


#### _on_success()

Method description here.


#### _on_failure()

Method description here.



### Attributes



---

## src.services.orchestrator.BaseService

Base service class with common functionality

**File:** `src/services\orchestrator.py`


**Inherits from:** ABC


### Methods


#### __init__()

Method description here.


#### is_healthy()

Method description here.



### Attributes



---

## src.services.orchestrator.MLPredictionService

Machine Learning prediction service

**File:** `src/services\orchestrator.py`


**Inherits from:** BaseService


### Methods


#### __init__()

Method description here.


#### _make_prediction()

Method description here.



### Attributes



---

## src.services.orchestrator.RiskManagementService

Risk management service

**File:** `src/services\orchestrator.py`


**Inherits from:** BaseService


### Methods


#### __init__()

Method description here.


#### _perform_risk_assessment()

Method description here.


#### _calculate_risk_score()

Method description here.



### Attributes



---

## src.services.orchestrator.ExecutionService

Trade execution service

**File:** `src/services\orchestrator.py`


**Inherits from:** BaseService


### Methods


#### __init__()

Method description here.


#### _perform_execution()

Method description here.


#### _paper_trade()

Method description here.


#### _live_trade()

Method description here.


#### _start_order_processor()

Method description here.



### Attributes



---

## src.services.orchestrator.TradingOrchestrator

Main trading orchestration service
Coordinates all trading services

**File:** `src/services\orchestrator.py`



### Methods


#### __init__()

Method description here.


#### get_service_status()

Method description here.


#### _get_start_order()

Method description here.


#### _all_services_healthy()

Method description here.



### Attributes



---

## src.setup.interactive_setup.SetupPhase

Setup wizard phases

**File:** `src/setup\interactive_setup.py`


**Inherits from:** Enum


### Methods



### Attributes


- `WELCOME`

- `ENVIRONMENT_CHECK`

- `DEPENDENCY_INSTALL`

- `BASIC_CONFIG`

- `ADVANCED_CONFIG`

- `API_SETUP`

- `SECURITY_CONFIG`

- `TESTING`

- `FINALIZATION`

- `COMPLETE`


---

## src.setup.interactive_setup.ConfigurationLevel

Configuration complexity levels

**File:** `src/setup\interactive_setup.py`


**Inherits from:** Enum


### Methods



### Attributes


- `BEGINNER`

- `INTERMEDIATE`

- `ADVANCED`

- `EXPERT`


---

## src.setup.interactive_setup.ValidationSeverity

Validation message severity

**File:** `src/setup\interactive_setup.py`


**Inherits from:** Enum


### Methods



### Attributes


- `INFO`

- `WARNING`

- `ERROR`

- `CRITICAL`


---

## src.setup.interactive_setup.SetupStep

Individual setup step definition

**File:** `src/setup\interactive_setup.py`



### Methods



### Attributes



---

## src.setup.interactive_setup.ValidationResult

Configuration validation result

**File:** `src/setup\interactive_setup.py`



### Methods



### Attributes



---

## src.setup.interactive_setup.UserPreferences

User preferences and settings

**File:** `src/setup\interactive_setup.py`



### Methods



### Attributes



---

## src.setup.interactive_setup.EnvironmentValidator

System environment validation and checking

**File:** `src/setup\interactive_setup.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.setup.interactive_setup.DependencyManager

Automated dependency management and installation

**File:** `src/setup\interactive_setup.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.setup.interactive_setup.ConfigurationWizard

Interactive configuration wizard

**File:** `src/setup\interactive_setup.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.setup.interactive_setup.SetupTester

Test configuration and validate setup

**File:** `src/setup\interactive_setup.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.setup.interactive_setup.ConfigurationManager

Manage configuration files and backups

**File:** `src/setup\interactive_setup.py`



### Methods


#### __init__()

Method description here.


#### backup_existing_config()

Method description here.


#### save_configuration()

Method description here.


#### load_configuration()

Method description here.



### Attributes



---

## src.setup.interactive_setup.InteractiveSetupOrchestrator

Main setup orchestrator managing the entire setup process

**File:** `src/setup\interactive_setup.py`



### Methods


#### __init__()

Method description here.


#### _initialize_setup_steps()

Method description here.


#### _filter_steps_by_preferences()

Method description here.



### Attributes



---

## src.setup.setup_wizard.SetupStep

Setup wizard steps

**File:** `src/setup\setup_wizard.py`


**Inherits from:** Enum


### Methods



### Attributes


- `WELCOME`

- `SECURITY_SETUP`

- `EXCHANGE_SELECTION`

- `API_KEY_CONFIGURATION`

- `RISK_MANAGEMENT`

- `ML_CONFIGURATION`

- `NOTIFICATION_SETUP`

- `VALIDATION`

- `COMPLETION`


---

## src.setup.setup_wizard.ExchangeConfig

Exchange configuration structure

**File:** `src/setup\setup_wizard.py`



### Methods


#### __post_init__()

Method description here.



### Attributes



---

## src.setup.setup_wizard.RiskConfig

Risk management configuration

**File:** `src/setup\setup_wizard.py`



### Methods



### Attributes



---

## src.setup.setup_wizard.MLConfig

Machine learning configuration

**File:** `src/setup\setup_wizard.py`



### Methods



### Attributes



---

## src.setup.setup_wizard.NotificationConfig

Notification configuration

**File:** `src/setup\setup_wizard.py`



### Methods



### Attributes



---

## src.setup.setup_wizard.SetupWizard

Enhanced setup wizard with complete configuration
Real-time validation and secure storage

**File:** `src/setup\setup_wizard.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.testing.integration_testing.TestLevel

Test execution levels

**File:** `src/testing\integration_testing.py`


**Inherits from:** Enum


### Methods



### Attributes


- `UNIT`

- `INTEGRATION`

- `SYSTEM`

- `PERFORMANCE`

- `SECURITY`

- `REGRESSION`


---

## src.testing.integration_testing.TestResult

Test execution results

**File:** `src/testing\integration_testing.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PASSED`

- `FAILED`

- `SKIPPED`

- `ERROR`

- `TIMEOUT`


---

## src.testing.integration_testing.CoverageType

Code coverage types

**File:** `src/testing\integration_testing.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LINE`

- `BRANCH`

- `FUNCTION`

- `STATEMENT`


---

## src.testing.integration_testing.TestCase

Individual test case definition

**File:** `src/testing\integration_testing.py`



### Methods



### Attributes



---

## src.testing.integration_testing.TestExecution

Test execution results

**File:** `src/testing\integration_testing.py`



### Methods



### Attributes



---

## src.testing.integration_testing.CoverageReport

Code coverage analysis results

**File:** `src/testing\integration_testing.py`



### Methods



### Attributes



---

## src.testing.integration_testing.TestDataFactory

Factory for generating test data

**File:** `src/testing\integration_testing.py`



### Methods


#### __init__()

Method description here.


#### generate_market_data()

Method description here.


#### generate_trading_signals()

Method description here.


#### generate_user_config()

Method description here.


#### generate_edge_cases()

Method description here.



### Attributes



---

## src.testing.integration_testing.MockTradingEnvironment

Mock trading environment for safe testing

**File:** `src/testing\integration_testing.py`



### Methods


#### __init__()

Method description here.


#### inject_fault()

Method description here.



### Attributes



---

## src.testing.integration_testing.TestCoverageAnalyzer

Comprehensive test coverage analysis

**File:** `src/testing\integration_testing.py`



### Methods


#### __init__()

Method description here.


#### start_coverage()

Method description here.


#### stop_coverage()

Method description here.


#### generate_report()

Method description here.


#### _calculate_complexity_metrics()

Method description here.


#### export_html_report()

Method description here.


#### export_xml_report()

Method description here.



### Attributes



---

## src.testing.integration_testing.PerformanceBenchmarkSuite

Performance benchmarking and regression detection

**File:** `src/testing\integration_testing.py`



### Methods


#### __init__()

Method description here.


#### benchmark_function()

Method description here.


#### set_baseline()

Method description here.


#### detect_regressions()

Method description here.


#### generate_performance_report()

Method description here.



### Attributes



---

## src.testing.integration_testing.IntegrationTestSuite

Comprehensive integration testing framework

**File:** `src/testing\integration_testing.py`



### Methods


#### __init__()

Method description here.


#### _initialize_test_cases()

Method description here.


#### export_test_reports()

Method description here.



### Attributes



---

## src.testing.performance_testing.TestType

Performance test types

**File:** `src/testing\performance_testing.py`


**Inherits from:** Enum


### Methods



### Attributes


- `LOAD_TEST`

- `STRESS_TEST`

- `SPIKE_TEST`

- `VOLUME_TEST`

- `ENDURANCE_TEST`

- `SCALABILITY_TEST`


---

## src.testing.performance_testing.TestStatus

Test execution status

**File:** `src/testing\performance_testing.py`


**Inherits from:** Enum


### Methods



### Attributes


- `PENDING`

- `RUNNING`

- `COMPLETED`

- `FAILED`

- `CANCELLED`


---

## src.testing.performance_testing.PerformanceMetric

Performance metrics to track

**File:** `src/testing\performance_testing.py`


**Inherits from:** Enum


### Methods



### Attributes


- `RESPONSE_TIME`

- `THROUGHPUT`

- `ERROR_RATE`

- `CPU_USAGE`

- `MEMORY_USAGE`

- `DISK_IO`

- `NETWORK_IO`

- `CONCURRENT_USERS`


---

## src.testing.performance_testing.TestResult

Individual test result

**File:** `src/testing\performance_testing.py`



### Methods



### Attributes



---

## src.testing.performance_testing.TestScenario

Performance test scenario configuration

**File:** `src/testing\performance_testing.py`



### Methods



### Attributes



---

## src.testing.performance_testing.SystemSnapshot

System resource snapshot

**File:** `src/testing\performance_testing.py`



### Methods



### Attributes



---

## src.testing.performance_testing.PerformanceReport

Comprehensive performance test report

**File:** `src/testing\performance_testing.py`



### Methods



### Attributes



---

## src.testing.performance_testing.LoadGenerator

Advanced load generation system

Generates realistic load patterns with configurable user behavior,
think times, and request patterns.

**File:** `src/testing\performance_testing.py`



### Methods


#### __init__()

Method description here.


#### get_session()

Method description here.



### Attributes



---

## src.testing.performance_testing.SystemMonitor

Advanced system resource monitoring during performance tests

Tracks CPU, memory, disk, network, and application-specific metrics
during test execution.

**File:** `src/testing\performance_testing.py`



### Methods


#### __init__()

Method description here.


#### take_snapshot()

Method description here.


#### establish_baseline()

Method description here.


#### get_monitoring_summary()

Method description here.


#### _compare_to_baseline()

Method description here.



### Attributes



---

## src.testing.performance_testing.PerformanceAnalyzer

Advanced performance analysis and reporting system

Analyzes test results, identifies bottlenecks, and generates
comprehensive reports with recommendations.

**File:** `src/testing\performance_testing.py`



### Methods


#### __init__()

Method description here.


#### analyze_test_results()

Method description here.


#### _create_empty_report()

Method description here.


#### _calculate_test_duration()

Method description here.


#### _calculate_peak_throughput()

Method description here.


#### _analyze_system_resources()

Method description here.


#### _assess_performance()

Method description here.


#### generate_html_report()

Method description here.



### Attributes



---

## src.testing.performance_testing.ComprehensivePerformanceTester

Master performance testing system

Coordinates load generation, system monitoring, and performance analysis
to provide comprehensive performance testing capabilities.

**File:** `src/testing\performance_testing.py`



### Methods


#### __init__()

Method description here.


#### create_load_test_scenarios()

Method description here.


#### generate_performance_summary()

Method description here.



### Attributes



---

## src.tests.security_tests.SecurityTestResults

Security test results aggregator

**File:** `src/tests\security_tests.py`



### Methods


#### __init__()

Method description here.


#### add_test_result()

Method description here.


#### add_vulnerability()

Method description here.


#### get_summary()

Method description here.


#### _calculate_security_score()

Method description here.



### Attributes



---

## src.tests.security_tests.EncryptionSecurityTests

Test encryption and secure storage security

**File:** `src/tests\security_tests.py`


**Inherits from:** <ast.Attribute object at 0x0000021F0A135390>


### Methods


#### setUp()

Method description here.


#### tearDown()

Method description here.


#### test_encryption_strength()

Method description here.


#### test_key_derivation_security()

Method description here.


#### test_password_strength_requirements()

Method description here.


#### test_data_integrity_verification()

Method description here.



### Attributes



---

## src.tests.security_tests.APISecurityTests

Test API key validation and security

**File:** `src/tests\security_tests.py`


**Inherits from:** <ast.Attribute object at 0x0000021F0A0A4090>


### Methods


#### setUp()

Method description here.


#### test_permission_validation_security()

Method description here.


#### test_rate_limiting_security()

Method description here.



### Attributes



---

## src.tests.security_tests.SetupWizardSecurityTests

Test setup wizard security

**File:** `src/tests\security_tests.py`


**Inherits from:** <ast.Attribute object at 0x0000021F0A1EE450>


### Methods


#### setUp()

Method description here.


#### tearDown()

Method description here.


#### test_setup_wizard_input_validation()

Method description here.


#### test_setup_wizard_password_security()

Method description here.



### Attributes



---

## src.tests.security_tests.ServiceLayerSecurityTests

Test service layer security

**File:** `src/tests\security_tests.py`


**Inherits from:** <ast.Attribute object at 0x0000021F0A1BAA50>


### Methods


#### setUp()

Method description here.


#### test_circuit_breaker_security()

Method description here.


#### test_service_isolation_security()

Method description here.



### Attributes



---

## src.tests.security_tests.PenetrationTests

Penetration testing for security vulnerabilities

**File:** `src/tests\security_tests.py`



### Methods


#### __init__()

Method description here.



### Attributes



---

## src.tests.security_tests.SecurityTestSuite

Main security test suite coordinator

**File:** `src/tests\security_tests.py`



### Methods


#### __init__()

Method description here.


#### _merge_unittest_results()

Method description here.


#### _merge_results()

Method description here.



### Attributes



---

## src.tests.test_integration.TestDataGenerator

Generate test data for integration tests

**File:** `src/tests\test_integration.py`



### Methods


#### generate_market_data()

Method description here.


#### generate_trade_data()

Method description here.


#### generate_news_articles()

Method description here.



### Attributes



---

## src.tests.test_integration.IntegrationTestBase

Base class for integration tests

**File:** `src/tests\test_integration.py`


**Inherits from:** <ast.Attribute object at 0x0000021F0A05F690>


### Methods



### Attributes



---

## src.tests.test_integration.TestConfigurationManager

Test configuration management system

**File:** `src/tests\test_integration.py`


**Inherits from:** <ast.Attribute object at 0x0000021F0A0B37D0>


### Methods


#### setUp()

Method description here.


#### tearDown()

Method description here.


#### test_create_default_configs()

Method description here.


#### test_load_configuration()

Method description here.


#### test_configuration_validation()

Method description here.


#### test_configuration_updates()

Method description here.



### Attributes



---

## src.tests.test_integration.TestIntegratedTradingBot

Test the integrated trading bot system

**File:** `src/tests\test_integration.py`


**Inherits from:** IntegrationTestBase


### Methods



### Attributes



---

## src.tests.test_integration.TestEndToEndWorkflows

Test complete end-to-end trading workflows

**File:** `src/tests\test_integration.py`


**Inherits from:** IntegrationTestBase


### Methods



### Attributes



---

## src.tests.test_integration.TestPerformanceAndStress

Test system performance and stress scenarios

**File:** `src/tests\test_integration.py`


**Inherits from:** IntegrationTestBase


### Methods



### Attributes



---

## src.tests.test_integration.TestComponentInteraction

Test interactions between different system components

**File:** `src/tests\test_integration.py`


**Inherits from:** IntegrationTestBase


### Methods



### Attributes



---

## src.tests.test_integration.TestSystemReliability

Test system reliability and error handling

**File:** `src/tests\test_integration.py`


**Inherits from:** IntegrationTestBase


### Methods



### Attributes



---

## src.tests.test_integration.TestRunner

Test runner for integration tests

**File:** `src/tests\test_integration.py`



### Methods


#### run_all_tests()

Method description here.



### Attributes



---
