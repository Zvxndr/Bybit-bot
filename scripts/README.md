# Health Check and Monitoring Scripts

Collection of health check and monitoring scripts for production deployment
of the ML trading bot system.

## Scripts Overview

### health_check.py
Comprehensive health check script that validates:
- API service availability and response times
- Database connectivity and performance
- Model loading and prediction capability
- Cache (Redis) connectivity
- External API connectivity (Bybit)
- Resource utilization (CPU, memory, disk)
- Model drift detection

### performance_test.py
Load testing and performance benchmarking:
- API endpoint load testing
- Prediction latency benchmarking
- Concurrent request handling
- Memory leak detection
- Performance regression testing

### backup_restore.py
Automated backup and disaster recovery:
- Database backup automation
- Model artifacts backup
- Configuration backup
- Automated restore procedures
- Backup verification

### log_monitor.py
Log monitoring and alerting:
- Error pattern detection
- Performance anomaly detection
- Custom alert triggers
- Log aggregation and analysis

## Usage

### Basic Health Check
```bash
python scripts/health_check.py --environment production
```

### Performance Testing
```bash
python scripts/performance_test.py --load-test --duration 300
```

### Backup Operations
```bash
python scripts/backup_restore.py backup --type full
```

### Log Monitoring
```bash
python scripts/log_monitor.py --monitor --alert-threshold error
```

## Integration

These scripts are designed to integrate with:
- Kubernetes liveness/readiness probes
- CI/CD pipelines for deployment validation
- Monitoring systems (Prometheus, Grafana)
- Alerting systems (PagerDuty, Slack)
- Automated backup systems

## Configuration

Scripts use the same configuration system as the main application,
supporting environment-specific settings and secure secrets management.