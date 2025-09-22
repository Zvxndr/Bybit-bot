"""
Log Aggregation System.
Centralized logging, collection, processing, and analysis system.
"""

import asyncio
import json
import time
import gzip
import re
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import statistics
from collections import defaultdict, deque
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

try:
    import elasticsearch
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk, scan
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False

try:
    import kafka
    from kafka import KafkaProducer, KafkaConsumer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class LogLevel(Enum):
    """Log levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info" 
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"

class LogFormat(Enum):
    """Log formats."""
    JSON = "json"
    STRUCTURED = "structured"
    PLAIN = "plain"
    CEF = "cef"  # Common Event Format
    GELF = "gelf"  # Graylog Extended Log Format

class LogSource(Enum):
    """Log sources."""
    APPLICATION = "application"
    SYSTEM = "system"
    AUDIT = "audit"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TRADING = "trading"
    RISK = "risk"
    ML = "ml"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    source: LogSource
    service: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    thread_id: Optional[str] = None
    hostname: str = ""
    process_id: str = ""
    version: str = ""
    environment: str = "production"
    labels: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None

@dataclass
class LogFilter:
    """Log filtering criteria."""
    level: Optional[LogLevel] = None
    source: Optional[LogSource] = None
    service: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    contains: Optional[str] = None
    regex: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)
    exclude: bool = False

@dataclass
class LogAlert:
    """Log-based alert configuration."""
    name: str
    query: str
    threshold: Union[int, float]
    time_window: timedelta
    severity: str = "warning"
    enabled: bool = True
    description: str = ""
    actions: List[str] = field(default_factory=list)

@dataclass
class LogMetrics:
    """Log metrics and statistics."""
    total_logs: int = 0
    logs_by_level: Dict[str, int] = field(default_factory=dict)
    logs_by_source: Dict[str, int] = field(default_factory=dict)
    logs_by_service: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    avg_log_size: float = 0.0
    ingestion_rate: float = 0.0
    processing_lag: float = 0.0
    storage_size: int = 0

@dataclass
class LogParsingRule:
    """Log parsing rule."""
    name: str
    pattern: str
    format: LogFormat
    fields: List[str]
    enabled: bool = True
    priority: int = 100
    test_cases: List[str] = field(default_factory=list)

class LogBuffer:
    """Thread-safe log buffer."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.total_added = 0
        self.total_dropped = 0
    
    def add(self, log_entry: LogEntry):
        """Add log entry to buffer."""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.total_dropped += 1
            
            self.buffer.append(log_entry)
            self.total_added += 1
    
    def get_batch(self, batch_size: int = 100) -> List[LogEntry]:
        """Get batch of log entries."""
        with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
            return batch
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()

class LogAggregator:
    """Centralized log aggregation and processing system."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Configuration
        self.config = {
            'collection': {
                'batch_size': 100,
                'flush_interval': 5,
                'buffer_size': 10000,
                'max_field_length': 1024,
                'compression': True
            },
            'storage': {
                'elasticsearch': {
                    'enabled': True,
                    'hosts': ['localhost:9200'],
                    'index_pattern': 'trading-logs-%Y.%m.%d',
                    'template_name': 'trading-logs-template',
                    'shards': 1,
                    'replicas': 0,
                    'refresh_interval': '5s'
                },
                'local': {
                    'enabled': True,
                    'path': './logs',
                    'rotation_size': '100MB',
                    'retention_days': 30,
                    'compression': True
                }
            },
            'processing': {
                'parsing_enabled': True,
                'enrichment_enabled': True,
                'filtering_enabled': True,
                'sampling_rate': 1.0,
                'max_processing_time': 1.0
            },
            'streaming': {
                'kafka': {
                    'enabled': False,
                    'bootstrap_servers': ['localhost:9092'],
                    'topic': 'trading-logs',
                    'compression_type': 'gzip'
                }
            }
        }
        
        # Storage backends
        self.elasticsearch_client: Optional[Elasticsearch] = None
        self.kafka_producer: Optional[KafkaProducer] = None
        
        # Processing components
        self.log_buffer = LogBuffer(self.config['collection']['buffer_size'])
        self.parsing_rules: Dict[str, LogParsingRule] = {}
        self.log_filters: List[LogFilter] = []
        self.log_alerts: Dict[str, LogAlert] = {}
        
        # Metrics and statistics
        self.metrics = LogMetrics()
        self.metrics_history: List[Tuple[datetime, LogMetrics]] = []
        
        # Processing state
        self.aggregator_active = False
        self.collection_task = None
        self.processing_task = None
        self.metrics_task = None
        
        # Recent logs for real-time monitoring
        self.recent_logs = deque(maxlen=1000)
        
        # Initialize components
        self._setup_default_parsing_rules()
        self._setup_default_filters()
        self._setup_default_alerts()
        
        if HAS_ELASTICSEARCH and self.config['storage']['elasticsearch']['enabled']:
            self._initialize_elasticsearch()
        
        if HAS_KAFKA and self.config['streaming']['kafka']['enabled']:
            self._initialize_kafka()
        
        self.logger.info("LogAggregator initialized")
    
    def _setup_default_parsing_rules(self):
        """Setup default log parsing rules."""
        try:
            # JSON log parsing
            json_rule = LogParsingRule(
                name="json_logs",
                pattern=r'^{.*}$',
                format=LogFormat.JSON,
                fields=['timestamp', 'level', 'message', 'service'],
                priority=10
            )
            
            # Structured log parsing (key=value format)
            structured_rule = LogParsingRule(
                name="structured_logs",
                pattern=r'(\w+)=([^\s]+)',
                format=LogFormat.STRUCTURED,
                fields=['key', 'value'],
                priority=20
            )
            
            # Common log format (Apache/Nginx style)
            common_rule = LogParsingRule(
                name="common_log_format",
                pattern=r'^(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+|-)',
                format=LogFormat.PLAIN,
                fields=['client_ip', 'timestamp', 'method', 'path', 'protocol', 'status', 'size'],
                priority=30
            )
            
            # Trading specific log format
            trading_rule = LogParsingRule(
                name="trading_logs",
                pattern=r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+) - (\w+): (.+)',
                format=LogFormat.STRUCTURED,
                fields=['timestamp', 'level', 'component', 'message'],
                priority=5
            )
            
            self.parsing_rules["json_logs"] = json_rule
            self.parsing_rules["structured_logs"] = structured_rule
            self.parsing_rules["common_log_format"] = common_rule
            self.parsing_rules["trading_logs"] = trading_rule
            
        except Exception as e:
            self.logger.error(f"Failed to setup default parsing rules: {e}")
    
    def _setup_default_filters(self):
        """Setup default log filters."""
        try:
            # Filter out debug logs in production
            debug_filter = LogFilter(
                level=LogLevel.DEBUG,
                exclude=True
            )
            
            # Filter out health check logs
            health_filter = LogFilter(
                contains="health_check",
                exclude=True
            )
            
            # Keep only error and above for certain services
            service_filter = LogFilter(
                service="background_service",
                level=LogLevel.INFO,
                exclude=True
            )
            
            self.log_filters.extend([debug_filter, health_filter, service_filter])
            
        except Exception as e:
            self.logger.error(f"Failed to setup default filters: {e}")
    
    def _setup_default_alerts(self):
        """Setup default log-based alerts."""
        try:
            # High error rate alert
            error_rate_alert = LogAlert(
                name="high_error_rate",
                query="level:error",
                threshold=10,
                time_window=timedelta(minutes=5),
                severity="critical",
                description="High error rate detected",
                actions=["email", "slack"]
            )
            
            # Critical error alert
            critical_alert = LogAlert(
                name="critical_errors",
                query="level:fatal OR message:*CRITICAL*",
                threshold=1,
                time_window=timedelta(minutes=1),
                severity="critical",
                description="Critical error detected",
                actions=["email", "slack", "pagerduty"]
            )
            
            # Trading system alert
            trading_alert = LogAlert(
                name="trading_errors",
                query="source:trading AND level:error",
                threshold=5,
                time_window=timedelta(minutes=10),
                severity="warning",
                description="Trading system errors",
                actions=["slack"]
            )
            
            # Performance alert
            performance_alert = LogAlert(
                name="slow_requests",
                query="duration:>5000",
                threshold=10,
                time_window=timedelta(minutes=5),
                severity="warning",
                description="Slow requests detected",
                actions=["slack"]
            )
            
            self.log_alerts["high_error_rate"] = error_rate_alert
            self.log_alerts["critical_errors"] = critical_alert
            self.log_alerts["trading_errors"] = trading_alert
            self.log_alerts["slow_requests"] = performance_alert
            
        except Exception as e:
            self.logger.error(f"Failed to setup default alerts: {e}")
    
    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch client."""
        try:
            if HAS_ELASTICSEARCH:
                es_config = self.config['storage']['elasticsearch']
                self.elasticsearch_client = Elasticsearch(
                    es_config['hosts'],
                    retry_on_timeout=True,
                    max_retries=3
                )
                
                # Test connection
                if self.elasticsearch_client.ping():
                    self.logger.info("Connected to Elasticsearch")
                    self._setup_elasticsearch_template()
                else:
                    self.logger.warning("Failed to connect to Elasticsearch")
                    self.elasticsearch_client = None
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize Elasticsearch: {e}")
            self.elasticsearch_client = None
    
    def _setup_elasticsearch_template(self):
        """Setup Elasticsearch index template."""
        try:
            if not self.elasticsearch_client:
                return
            
            template = {
                "index_patterns": ["trading-logs-*"],
                "template": {
                    "settings": {
                        "number_of_shards": self.config['storage']['elasticsearch']['shards'],
                        "number_of_replicas": self.config['storage']['elasticsearch']['replicas'],
                        "refresh_interval": self.config['storage']['elasticsearch']['refresh_interval'],
                        "index.mapping.total_fields.limit": 2000
                    },
                    "mappings": {
                        "properties": {
                            "@timestamp": {"type": "date"},
                            "level": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "service": {"type": "keyword"},
                            "message": {"type": "text", "analyzer": "standard"},
                            "trace_id": {"type": "keyword"},
                            "span_id": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "session_id": {"type": "keyword"},
                            "request_id": {"type": "keyword"},
                            "hostname": {"type": "keyword"},
                            "environment": {"type": "keyword"},
                            "duration": {"type": "float"},
                            "status_code": {"type": "integer"},
                            "labels": {"type": "object"},
                            "fields": {"type": "object"},
                            "exception": {
                                "properties": {
                                    "type": {"type": "keyword"},
                                    "message": {"type": "text"},
                                    "stack_trace": {"type": "text"}
                                }
                            }
                        }
                    }
                }
            }
            
            template_name = self.config['storage']['elasticsearch']['template_name']
            self.elasticsearch_client.indices.put_index_template(
                name=template_name,
                body=template
            )
            
            self.logger.info(f"Elasticsearch template {template_name} created")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Elasticsearch template: {e}")
    
    def _initialize_kafka(self):
        """Initialize Kafka producer."""
        try:
            if HAS_KAFKA:
                kafka_config = self.config['streaming']['kafka']
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=kafka_config['bootstrap_servers'],
                    compression_type=kafka_config.get('compression_type', 'gzip'),
                    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                    retry_backoff_ms=100,
                    retries=3
                )
                
                self.logger.info("Kafka producer initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka: {e}")
            self.kafka_producer = None
    
    async def start_aggregator(self):
        """Start log aggregator."""
        try:
            if self.aggregator_active:
                return
            
            self.aggregator_active = True
            
            # Start processing tasks
            self.collection_task = asyncio.create_task(self._collection_loop())
            self.processing_task = asyncio.create_task(self._processing_loop())
            self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            self.logger.info("Log Aggregator started")
            
        except Exception as e:
            self.logger.error(f"Failed to start Log Aggregator: {e}")
    
    async def stop_aggregator(self):
        """Stop log aggregator."""
        try:
            self.aggregator_active = False
            
            # Cancel tasks
            tasks = [self.collection_task, self.processing_task, self.metrics_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Flush remaining logs
            await self._flush_logs()
            
            # Cleanup resources
            if self.kafka_producer:
                self.kafka_producer.close()
            
            self.logger.info("Log Aggregator stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop Log Aggregator: {e}")
    
    async def _collection_loop(self):
        """Log collection loop."""
        try:
            while self.aggregator_active:
                batch = self.log_buffer.get_batch(self.config['collection']['batch_size'])
                
                if batch:
                    await self._process_log_batch(batch)
                
                await asyncio.sleep(self.config['collection']['flush_interval'])
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Collection loop error: {e}")
    
    async def _processing_loop(self):
        """Log processing loop."""
        try:
            while self.aggregator_active:
                # Process alerts
                await self._check_log_alerts()
                
                # Update metrics
                self._update_metrics()
                
                await asyncio.sleep(30)  # Process every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Processing loop error: {e}")
    
    async def _metrics_loop(self):
        """Metrics collection loop."""
        try:
            while self.aggregator_active:
                # Collect and store metrics
                current_metrics = self._collect_metrics()
                self.metrics_history.append((datetime.now(), current_metrics))
                
                # Keep metrics history manageable
                if len(self.metrics_history) > 1440:  # 24 hours at 1-minute intervals
                    self.metrics_history = self.metrics_history[-720:]  # Keep 12 hours
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Metrics loop error: {e}")
    
    def ingest_log(self, level: Union[str, LogLevel], source: Union[str, LogSource], 
                   service: str, message: str, **kwargs):
        """Ingest a log entry."""
        try:
            # Convert string enums
            if isinstance(level, str):
                level = LogLevel(level.lower())
            if isinstance(source, str):
                source = LogSource(source.lower())
            
            # Create log entry
            log_entry = LogEntry(
                timestamp=kwargs.get('timestamp', datetime.now()),
                level=level,
                source=source,
                service=service,
                message=message[:self.config['collection']['max_field_length']],
                trace_id=kwargs.get('trace_id'),
                span_id=kwargs.get('span_id'),
                user_id=kwargs.get('user_id'),
                session_id=kwargs.get('session_id'),
                request_id=kwargs.get('request_id'),
                thread_id=kwargs.get('thread_id'),
                hostname=kwargs.get('hostname', ''),
                process_id=kwargs.get('process_id', ''),
                version=kwargs.get('version', ''),
                environment=kwargs.get('environment', 'production'),
                labels=kwargs.get('labels', {}),
                fields=kwargs.get('fields', {}),
                exception=kwargs.get('exception'),
                duration=kwargs.get('duration'),
                status_code=kwargs.get('status_code')
            )
            
            # Apply filters
            if self._should_filter_log(log_entry):
                return
            
            # Add to buffer
            self.log_buffer.add(log_entry)
            
            # Add to recent logs for monitoring
            self.recent_logs.append(log_entry)
            
        except Exception as e:
            self.logger.error(f"Failed to ingest log: {e}")
    
    def ingest_raw_log(self, raw_log: str, source: str = "unknown"):
        """Ingest raw log string and parse it."""
        try:
            # Parse raw log
            parsed_entry = self._parse_raw_log(raw_log, source)
            
            if parsed_entry:
                self.log_buffer.add(parsed_entry)
                self.recent_logs.append(parsed_entry)
            
        except Exception as e:
            self.logger.error(f"Failed to ingest raw log: {e}")
    
    def _parse_raw_log(self, raw_log: str, source: str) -> Optional[LogEntry]:
        """Parse raw log string using parsing rules."""
        try:
            # Try parsing rules in priority order
            sorted_rules = sorted(
                self.parsing_rules.values(),
                key=lambda r: r.priority
            )
            
            for rule in sorted_rules:
                if not rule.enabled:
                    continue
                
                if rule.format == LogFormat.JSON:
                    try:
                        data = json.loads(raw_log)
                        return self._create_log_entry_from_json(data, source)
                    except json.JSONDecodeError:
                        continue
                
                elif rule.format == LogFormat.STRUCTURED:
                    matches = re.findall(rule.pattern, raw_log)
                    if matches:
                        fields = {}
                        for match in matches:
                            if len(match) >= 2:
                                fields[match[0]] = match[1]
                        
                        return self._create_log_entry_from_fields(fields, raw_log, source)
                
                elif rule.format == LogFormat.PLAIN:
                    match = re.match(rule.pattern, raw_log)
                    if match:
                        field_values = dict(zip(rule.fields, match.groups()))
                        return self._create_log_entry_from_fields(field_values, raw_log, source)
            
            # If no rule matches, create basic log entry
            return LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                source=LogSource.APPLICATION,
                service=source,
                message=raw_log
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse raw log: {e}")
            return None
    
    def _create_log_entry_from_json(self, data: Dict[str, Any], source: str) -> LogEntry:
        """Create log entry from JSON data."""
        try:
            # Parse timestamp
            timestamp = data.get('timestamp', data.get('@timestamp', datetime.now()))
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Parse level
            level_str = data.get('level', data.get('severity', 'info')).lower()
            level = LogLevel(level_str) if level_str in [l.value for l in LogLevel] else LogLevel.INFO
            
            # Parse source
            source_str = data.get('source', data.get('logger', source)).lower()
            log_source = LogSource(source_str) if source_str in [s.value for s in LogSource] else LogSource.APPLICATION
            
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=log_source,
                service=data.get('service', source),
                message=data.get('message', data.get('msg', '')),
                trace_id=data.get('trace_id', data.get('traceId')),
                span_id=data.get('span_id', data.get('spanId')),
                user_id=data.get('user_id', data.get('userId')),
                session_id=data.get('session_id', data.get('sessionId')),
                request_id=data.get('request_id', data.get('requestId')),
                hostname=data.get('hostname', data.get('host', '')),
                labels=data.get('labels', {}),
                fields=data.get('fields', data.get('extra', {})),
                duration=data.get('duration'),
                status_code=data.get('status_code', data.get('statusCode'))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create log entry from JSON: {e}")
            return None
    
    def _create_log_entry_from_fields(self, fields: Dict[str, str], raw_message: str, source: str) -> LogEntry:
        """Create log entry from parsed fields."""
        try:
            # Parse timestamp
            timestamp = datetime.now()
            if 'timestamp' in fields:
                try:
                    timestamp = datetime.fromisoformat(fields['timestamp'])
                except ValueError:
                    pass
            
            # Parse level
            level = LogLevel.INFO
            if 'level' in fields:
                level_str = fields['level'].lower()
                if level_str in [l.value for l in LogLevel]:
                    level = LogLevel(level_str)
            
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=LogSource.APPLICATION,
                service=fields.get('service', source),
                message=fields.get('message', raw_message),
                labels={k: v for k, v in fields.items() if k not in ['timestamp', 'level', 'service', 'message']}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create log entry from fields: {e}")
            return None
    
    def _should_filter_log(self, log_entry: LogEntry) -> bool:
        """Check if log should be filtered out."""
        try:
            for log_filter in self.log_filters:
                if self._matches_filter(log_entry, log_filter):
                    return log_filter.exclude
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check log filter: {e}")
            return False
    
    def _matches_filter(self, log_entry: LogEntry, log_filter: LogFilter) -> bool:
        """Check if log entry matches filter."""
        try:
            # Check level
            if log_filter.level and log_entry.level != log_filter.level:
                return False
            
            # Check source
            if log_filter.source and log_entry.source != log_filter.source:
                return False
            
            # Check service
            if log_filter.service and log_entry.service != log_filter.service:
                return False
            
            # Check time range
            if log_filter.time_range:
                start_time, end_time = log_filter.time_range
                if not (start_time <= log_entry.timestamp <= end_time):
                    return False
            
            # Check contains
            if log_filter.contains and log_filter.contains.lower() not in log_entry.message.lower():
                return False
            
            # Check regex
            if log_filter.regex and not re.search(log_filter.regex, log_entry.message):
                return False
            
            # Check labels
            for key, value in log_filter.labels.items():
                if log_entry.labels.get(key) != value:
                    return False
            
            # Check fields
            for key, value in log_filter.fields.items():
                if log_entry.fields.get(key) != value:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to match filter: {e}")
            return False
    
    async def _process_log_batch(self, batch: List[LogEntry]):
        """Process batch of log entries."""
        try:
            if not batch:
                return
            
            # Send to Elasticsearch
            if self.elasticsearch_client:
                await self._send_to_elasticsearch(batch)
            
            # Send to Kafka
            if self.kafka_producer:
                await self._send_to_kafka(batch)
            
            # Write to local files
            await self._write_to_local_files(batch)
            
            # Update metrics
            self.metrics.total_logs += len(batch)
            
        except Exception as e:
            self.logger.error(f"Failed to process log batch: {e}")
    
    async def _send_to_elasticsearch(self, batch: List[LogEntry]):
        """Send log batch to Elasticsearch."""
        try:
            if not self.elasticsearch_client:
                return
            
            actions = []
            for log_entry in batch:
                index_name = log_entry.timestamp.strftime(
                    self.config['storage']['elasticsearch']['index_pattern']
                )
                
                doc = {
                    '@timestamp': log_entry.timestamp.isoformat(),
                    'level': log_entry.level.value,
                    'source': log_entry.source.value,
                    'service': log_entry.service,
                    'message': log_entry.message,
                    'trace_id': log_entry.trace_id,
                    'span_id': log_entry.span_id,
                    'user_id': log_entry.user_id,
                    'session_id': log_entry.session_id,
                    'request_id': log_entry.request_id,
                    'hostname': log_entry.hostname,
                    'environment': log_entry.environment,
                    'labels': log_entry.labels,
                    'fields': log_entry.fields,
                    'duration': log_entry.duration,
                    'status_code': log_entry.status_code
                }
                
                if log_entry.exception:
                    doc['exception'] = log_entry.exception
                
                actions.append({
                    '_index': index_name,
                    '_source': doc
                })
            
            # Bulk index
            bulk(self.elasticsearch_client, actions)
            
        except Exception as e:
            self.logger.error(f"Failed to send logs to Elasticsearch: {e}")
    
    async def _send_to_kafka(self, batch: List[LogEntry]):
        """Send log batch to Kafka."""
        try:
            if not self.kafka_producer:
                return
            
            topic = self.config['streaming']['kafka']['topic']
            
            for log_entry in batch:
                message = {
                    'timestamp': log_entry.timestamp.isoformat(),
                    'level': log_entry.level.value,
                    'source': log_entry.source.value,
                    'service': log_entry.service,
                    'message': log_entry.message,
                    'trace_id': log_entry.trace_id,
                    'labels': log_entry.labels,
                    'fields': log_entry.fields
                }
                
                self.kafka_producer.send(topic, value=message)
            
            self.kafka_producer.flush()
            
        except Exception as e:
            self.logger.error(f"Failed to send logs to Kafka: {e}")
    
    async def _write_to_local_files(self, batch: List[LogEntry]):
        """Write log batch to local files."""
        try:
            if not self.config['storage']['local']['enabled']:
                return
            
            # Group logs by date and service
            logs_by_date_service = defaultdict(list)
            
            for log_entry in batch:
                date_str = log_entry.timestamp.strftime('%Y-%m-%d')
                key = f"{date_str}-{log_entry.service}"
                logs_by_date_service[key].append(log_entry)
            
            # Write to files
            import os
            log_dir = self.config['storage']['local']['path']
            os.makedirs(log_dir, exist_ok=True)
            
            for key, logs in logs_by_date_service.items():
                filename = f"{key}.log"
                filepath = os.path.join(log_dir, filename)
                
                with open(filepath, 'a', encoding='utf-8') as f:
                    for log_entry in logs:
                        log_line = self._format_log_entry(log_entry)
                        f.write(log_line + '\n')
            
        except Exception as e:
            self.logger.error(f"Failed to write logs to local files: {e}")
    
    def _format_log_entry(self, log_entry: LogEntry) -> str:
        """Format log entry for output."""
        try:
            if self.config['collection']['compression']:
                # JSON format for structured storage
                data = {
                    'timestamp': log_entry.timestamp.isoformat(),
                    'level': log_entry.level.value,
                    'source': log_entry.source.value,
                    'service': log_entry.service,
                    'message': log_entry.message
                }
                
                if log_entry.trace_id:
                    data['trace_id'] = log_entry.trace_id
                
                if log_entry.labels:
                    data['labels'] = log_entry.labels
                
                if log_entry.fields:
                    data['fields'] = log_entry.fields
                
                return json.dumps(data, separators=(',', ':'))
            else:
                # Plain text format
                return f"[{log_entry.timestamp.isoformat()}] {log_entry.level.value.upper()} - {log_entry.service}: {log_entry.message}"
                
        except Exception as e:
            self.logger.error(f"Failed to format log entry: {e}")
            return f"[{log_entry.timestamp.isoformat()}] ERROR - Failed to format log"
    
    async def _flush_logs(self):
        """Flush any remaining logs."""
        try:
            remaining_logs = self.log_buffer.get_batch(self.log_buffer.size())
            if remaining_logs:
                await self._process_log_batch(remaining_logs)
                
        except Exception as e:
            self.logger.error(f"Failed to flush logs: {e}")
    
    async def _check_log_alerts(self):
        """Check log-based alerts."""
        try:
            for alert in self.log_alerts.values():
                if not alert.enabled:
                    continue
                
                # Simple alert checking (would use proper query engine in production)
                await self._evaluate_log_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to check log alerts: {e}")
    
    async def _evaluate_log_alert(self, alert: LogAlert):
        """Evaluate a single log alert."""
        try:
            # Count recent logs matching query
            current_time = datetime.now()
            start_time = current_time - alert.time_window
            
            matching_count = 0
            for log_entry in self.recent_logs:
                if log_entry.timestamp < start_time:
                    continue
                
                if self._matches_alert_query(log_entry, alert.query):
                    matching_count += 1
            
            # Check threshold
            if matching_count >= alert.threshold:
                self.logger.warning(f"Log alert triggered: {alert.name} - {matching_count} matches")
                # Would send alert notification here
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate log alert {alert.name}: {e}")
    
    def _matches_alert_query(self, log_entry: LogEntry, query: str) -> bool:
        """Check if log entry matches alert query."""
        try:
            # Simple query matching (would use proper query parser in production)
            query = query.lower()
            
            if "level:error" in query and log_entry.level != LogLevel.ERROR:
                return False
            
            if "level:fatal" in query and log_entry.level != LogLevel.FATAL:
                return False
            
            if "source:trading" in query and log_entry.source != LogSource.TRADING:
                return False
            
            if "message:*critical*" in query and "critical" not in log_entry.message.lower():
                return False
            
            if "duration:>" in query:
                threshold_match = re.search(r'duration:>(\d+)', query)
                if threshold_match and log_entry.duration:
                    threshold = float(threshold_match.group(1))
                    if log_entry.duration <= threshold:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to match alert query: {e}")
            return False
    
    def _update_metrics(self):
        """Update log metrics."""
        try:
            # Update counters from recent logs
            levels = defaultdict(int)
            sources = defaultdict(int)
            services = defaultdict(int)
            error_count = 0
            total_size = 0
            
            for log_entry in self.recent_logs:
                levels[log_entry.level.value] += 1
                sources[log_entry.source.value] += 1
                services[log_entry.service] += 1
                
                if log_entry.level in [LogLevel.ERROR, LogLevel.FATAL]:
                    error_count += 1
                
                # Estimate log size
                total_size += len(log_entry.message) + 200  # Approximate overhead
            
            # Update metrics
            self.metrics.logs_by_level = dict(levels)
            self.metrics.logs_by_source = dict(sources)
            self.metrics.logs_by_service = dict(services)
            
            if len(self.recent_logs) > 0:
                self.metrics.error_rate = error_count / len(self.recent_logs)
                self.metrics.avg_log_size = total_size / len(self.recent_logs)
            
            self.metrics.ingestion_rate = self.log_buffer.total_added / max(1, time.time() - (time.time() - 3600))  # Rough estimate
            self.metrics.processing_lag = self.log_buffer.size()  # Simple lag indicator
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    def _collect_metrics(self) -> LogMetrics:
        """Collect current metrics snapshot."""
        try:
            return LogMetrics(
                total_logs=self.metrics.total_logs,
                logs_by_level=self.metrics.logs_by_level.copy(),
                logs_by_source=self.metrics.logs_by_source.copy(),
                logs_by_service=self.metrics.logs_by_service.copy(),
                error_rate=self.metrics.error_rate,
                avg_log_size=self.metrics.avg_log_size,
                ingestion_rate=self.metrics.ingestion_rate,
                processing_lag=self.metrics.processing_lag,
                storage_size=self.metrics.storage_size
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return LogMetrics()
    
    def search_logs(self, query: str, time_range: Optional[Tuple[datetime, datetime]] = None,
                   limit: int = 100) -> List[LogEntry]:
        """Search logs with query."""
        try:
            results = []
            
            # Search recent logs in memory
            for log_entry in reversed(list(self.recent_logs)):
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= log_entry.timestamp <= end_time):
                        continue
                
                if self._matches_search_query(log_entry, query):
                    results.append(log_entry)
                    
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search logs: {e}")
            return []
    
    def _matches_search_query(self, log_entry: LogEntry, query: str) -> bool:
        """Check if log entry matches search query."""
        try:
            # Simple text search (would use proper search engine in production)
            query = query.lower()
            
            if query in log_entry.message.lower():
                return True
            
            if query in log_entry.service.lower():
                return True
            
            if query in log_entry.level.value.lower():
                return True
            
            # Search in labels and fields
            for value in log_entry.labels.values():
                if query in str(value).lower():
                    return True
            
            for value in log_entry.fields.values():
                if query in str(value).lower():
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to match search query: {e}")
            return False
    
    def get_log_aggregator_summary(self) -> Dict[str, Any]:
        """Get log aggregator summary."""
        try:
            return {
                'status': 'active' if self.aggregator_active else 'inactive',
                'configuration': {
                    'elasticsearch_enabled': self.config['storage']['elasticsearch']['enabled'] and self.elasticsearch_client is not None,
                    'kafka_enabled': self.config['streaming']['kafka']['enabled'] and self.kafka_producer is not None,
                    'local_storage_enabled': self.config['storage']['local']['enabled'],
                    'batch_size': self.config['collection']['batch_size'],
                    'flush_interval': self.config['collection']['flush_interval']
                },
                'buffer': {
                    'current_size': self.log_buffer.size(),
                    'max_size': self.log_buffer.max_size,
                    'total_added': self.log_buffer.total_added,
                    'total_dropped': self.log_buffer.total_dropped
                },
                'processing': {
                    'parsing_rules': len(self.parsing_rules),
                    'filters': len(self.log_filters),
                    'alerts': len(self.log_alerts)
                },
                'metrics': {
                    'total_logs': self.metrics.total_logs,
                    'error_rate': self.metrics.error_rate,
                    'ingestion_rate': self.metrics.ingestion_rate,
                    'processing_lag': self.metrics.processing_lag,
                    'avg_log_size': self.metrics.avg_log_size
                },
                'recent_logs': len(self.recent_logs),
                'metrics_history_points': len(self.metrics_history)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate log aggregator summary: {e}")
            return {'error': 'Unable to generate summary'}