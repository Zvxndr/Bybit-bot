"""
Production Deployment Scripts - Phase 10

This module provides comprehensive deployment automation including:
- Docker containerization
- Environment setup and configuration
- Dependency management
- Database initialization and migration
- Monitoring and logging configuration
- Security hardening
- Health checks and readiness probes

Author: Trading Bot Team
Version: 1.0.0
"""

# Dockerfile for the trading bot
DOCKERFILE_CONTENT = """
# Multi-stage Docker build for optimal production deployment
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.local/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading trading

# Create app directory and set ownership
RUN mkdir -p /app /app/logs /app/data /app/config /app/reports
RUN chown -R trading:trading /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=trading:trading src/ ./src/
COPY --chown=trading:trading config/ ./config/
COPY --chown=trading:trading requirements.txt ./

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8081

# Default command
CMD ["python", "-m", "src.bot.main", "--config-path", "config/config_production.yaml"]
"""

# Docker Compose for complete stack
DOCKER_COMPOSE_CONTENT = """
version: '3.8'

services:
  # Trading Bot Application
  trading-bot:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: bybit-trading-bot
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://trading_user:${POSTGRES_PASSWORD}@postgres:5432/trading_bot
      - REDIS_URL=redis://redis:6379/0
      # Production Trading - LIVE CREDENTIALS (REAL MONEY)
      - BYBIT_LIVE_API_KEY=${BYBIT_LIVE_API_KEY}
      - BYBIT_LIVE_API_SECRET=${BYBIT_LIVE_API_SECRET}
      # Testnet credentials (for staging/development environments)
      - BYBIT_TESTNET_API_KEY=${BYBIT_TESTNET_API_KEY}
      - BYBIT_TESTNET_API_SECRET=${BYBIT_TESTNET_API_SECRET}
      - EMAIL_PASSWORD=${EMAIL_PASSWORD}
      - JWT_SECRET=${JWT_SECRET}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./reports:/app/reports
      - ./config:/app/config:ro
    ports:
      - "8080:8080"  # REST API
      - "8081:8081"  # WebSocket
    depends_on:
      - postgres
      - redis
    networks:
      - trading-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: trading-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=trading_bot
      - POSTGRES_USER=trading_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user -d trading_bot"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for caching and session storage
  redis:
    image: redis:7-alpine
    container_name: trading-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    container_name: trading-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=15d'
      - '--web.enable-lifecycle'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - trading-network

  # Grafana for metrics visualization
  grafana:
    image: grafana/grafana:latest
    container_name: trading-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "3000:3000"
    networks:
      - trading-network
    depends_on:
      - prometheus

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: trading-nginx
    restart: unless-stopped
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    networks:
      - trading-network
    depends_on:
      - trading-bot

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  trading-network:
    driver: bridge
"""

# Requirements.txt for Python dependencies
REQUIREMENTS_CONTENT = """
# Core dependencies
aiohttp==3.8.6
aiofiles==23.2.1
asyncio-mqtt==0.16.1
pandas==2.1.3
numpy==1.25.2
scipy==1.11.4
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0

# Financial and trading
ccxt==4.1.23
ta-lib==0.4.28
yfinance==0.2.28
python-binance==1.0.19

# Database
sqlalchemy==2.0.23
alembic==1.13.1
psycopg2-binary==2.9.9
redis==5.0.1

# Web framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
pydantic==2.5.0
python-multipart==0.0.6

# Authentication and security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Configuration and environment
pyyaml==6.0.1
python-dotenv==1.0.0
click==8.1.7

# Monitoring and observability
prometheus-client==0.19.0
structlog==23.2.0
loguru==0.7.2

# Machine learning and optimization
torch==2.1.1
transformers==4.36.0
optuna==3.4.0
cvxpy==1.4.1

# News and sentiment analysis
feedparser==6.0.10
newspaper3k==0.2.8
textblob==0.17.1
vaderSentiment==3.3.2

# Utilities
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3
python-dateutil==2.8.2
pytz==2023.3
schedule==1.2.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
httpx==0.25.2

# Development tools
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.6.0
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import logging


class DeploymentManager:
    """
    Production Deployment Manager
    
    Handles complete deployment automation including containerization,
    orchestration, and production configuration.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.deployment_dir = self.project_root / "deployment"
        self.logger = logging.getLogger(__name__)
        
        # Create deployment directory structure
        self.deployment_dir.mkdir(exist_ok=True)
        (self.deployment_dir / "docker").mkdir(exist_ok=True)
        (self.deployment_dir / "kubernetes").mkdir(exist_ok=True)
        (self.deployment_dir / "scripts").mkdir(exist_ok=True)
        (self.deployment_dir / "monitoring").mkdir(exist_ok=True)
        (self.deployment_dir / "nginx").mkdir(exist_ok=True)
    
    def create_docker_files(self):
        """Create Docker configuration files"""
        try:
            # Write Dockerfile
            dockerfile_path = self.project_root / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(DOCKERFILE_CONTENT.strip())
            
            # Write docker-compose.yml
            compose_path = self.project_root / "docker-compose.yml"
            with open(compose_path, 'w') as f:
                f.write(DOCKER_COMPOSE_CONTENT.strip())
            
            # Write requirements.txt
            requirements_path = self.project_root / "requirements.txt"
            with open(requirements_path, 'w') as f:
                f.write(REQUIREMENTS_CONTENT.strip())
            
            # Create .dockerignore
            dockerignore_content = """
.git
.gitignore
.github
README.md
Dockerfile
.dockerignore
.env
.env.*
node_modules
*/node_modules
logs/
*.log
.pytest_cache
.coverage
.mypy_cache
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.db
*.sqlite3
"""
            
            dockerignore_path = self.project_root / ".dockerignore"
            with open(dockerignore_path, 'w') as f:
                f.write(dockerignore_content.strip())
            
            self.logger.info("Docker configuration files created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating Docker files: {e}")
            raise
    
    def create_kubernetes_manifests(self):
        """Create Kubernetes deployment manifests"""
        k8s_dir = self.deployment_dir / "kubernetes"
        
        # Namespace
        namespace_yaml = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {'name': 'trading-bot'}
        }
        
        # ConfigMap
        configmap_yaml = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'trading-bot-config',
                'namespace': 'trading-bot'
            },
            'data': {
                'ENVIRONMENT': 'production',
                'LOG_LEVEL': 'INFO'
            }
        }
        
        # Secret
        secret_yaml = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': 'trading-bot-secrets',
                'namespace': 'trading-bot'
            },
            'type': 'Opaque',
            'data': {
                # Environment variables will be injected at deployment time
                # These values should be set in the deployment environment
                'BYBIT_API_KEY': '${BYBIT_API_KEY}',
                'BYBIT_API_SECRET': '${BYBIT_API_SECRET}',
                'DATABASE_PASSWORD': '${DATABASE_PASSWORD}',
                'JWT_SECRET': '${JWT_SECRET}'
            }
        }
        
        # Deployment
        deployment_yaml = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'trading-bot',
                'namespace': 'trading-bot',
                'labels': {'app': 'trading-bot'}
            },
            'spec': {
                'replicas': 1,
                'selector': {'matchLabels': {'app': 'trading-bot'}},
                'template': {
                    'metadata': {'labels': {'app': 'trading-bot'}},
                    'spec': {
                        'containers': [{
                            'name': 'trading-bot',
                            'image': 'bybit-trading-bot:latest',
                            'ports': [
                                {'containerPort': 8080, 'name': 'http'},
                                {'containerPort': 8081, 'name': 'websocket'}
                            ],
                            'envFrom': [
                                {'configMapRef': {'name': 'trading-bot-config'}},
                                {'secretRef': {'name': 'trading-bot-secrets'}}
                            ],
                            'resources': {
                                'requests': {'memory': '512Mi', 'cpu': '500m'},
                                'limits': {'memory': '2Gi', 'cpu': '2000m'}
                            },
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': 8080},
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/ready', 'port': 8080},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        }
        
        # Service
        service_yaml = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'trading-bot-service',
                'namespace': 'trading-bot'
            },
            'spec': {
                'selector': {'app': 'trading-bot'},
                'ports': [
                    {'name': 'http', 'port': 80, 'targetPort': 8080},
                    {'name': 'websocket', 'port': 8081, 'targetPort': 8081}
                ],
                'type': 'ClusterIP'
            }
        }
        
        # Ingress
        ingress_yaml = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': 'trading-bot-ingress',
                'namespace': 'trading-bot',
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': ['trading.yourdomain.com'],
                    'secretName': 'trading-bot-tls'
                }],
                'rules': [{
                    'host': 'trading.yourdomain.com',
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': 'trading-bot-service',
                                    'port': {'number': 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        # Write YAML files
        manifests = {
            'namespace.yaml': namespace_yaml,
            'configmap.yaml': configmap_yaml,
            'secret.yaml': secret_yaml,
            'deployment.yaml': deployment_yaml,
            'service.yaml': service_yaml,
            'ingress.yaml': ingress_yaml
        }
        
        for filename, manifest in manifests.items():
            with open(k8s_dir / filename, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
        
        self.logger.info("Kubernetes manifests created successfully")
    
    def create_deployment_scripts(self):
        """Create deployment automation scripts"""
        scripts_dir = self.deployment_dir / "scripts"
        
        # Docker deployment script
        docker_deploy_script = """#!/bin/bash
set -e

echo "🚀 Starting Docker deployment..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t bybit-trading-bot:latest .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down || true

# Start new deployment
echo "▶️ Starting new deployment..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🔍 Running health checks..."
curl -f http://localhost:8080/health || {
    echo "❌ Health check failed"
    docker-compose logs trading-bot
    exit 1
}

echo "✅ Docker deployment completed successfully!"
"""
        
        # Kubernetes deployment script
        k8s_deploy_script = """#!/bin/bash
set -e

echo "🚀 Starting Kubernetes deployment..."

# Apply namespace first
kubectl apply -f deployment/kubernetes/namespace.yaml

# Create secrets with environment-specific credentials
echo "🔐 Creating secrets..."
kubectl create secret generic trading-bot-secrets \\
    --from-literal=BYBIT_LIVE_API_KEY="$BYBIT_LIVE_API_KEY" \\
    --from-literal=BYBIT_LIVE_API_SECRET="$BYBIT_LIVE_API_SECRET" \\
    --from-literal=BYBIT_TESTNET_API_KEY="$BYBIT_TESTNET_API_KEY" \\
    --from-literal=BYBIT_TESTNET_API_SECRET="$BYBIT_TESTNET_API_SECRET" \\
    --from-literal=DATABASE_PASSWORD="$DATABASE_PASSWORD" \\
    --from-literal=JWT_SECRET="$JWT_SECRET" \\
    --namespace=trading-bot \\
    --dry-run=client -o yaml | kubectl apply -f -

# Apply all manifests
echo "📦 Applying Kubernetes manifests..."
kubectl apply -f deployment/kubernetes/

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/trading-bot -n trading-bot

# Get service information
echo "📋 Service information:"
kubectl get services -n trading-bot

echo "✅ Kubernetes deployment completed successfully!"
"""
        
        # Database migration script
        db_migrate_script = """#!/bin/bash
set -e

echo "🗄️ Running database migrations..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Run Alembic migrations
python -m alembic upgrade head

echo "✅ Database migrations completed successfully!"
"""
        
        # Backup script
        backup_script = """#!/bin/bash
set -e

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "💾 Creating backup at $BACKUP_DIR..."

# Database backup
echo "📊 Backing up database..."
pg_dump "$DATABASE_URL" > "$BACKUP_DIR/database.sql"

# Configuration backup
echo "⚙️ Backing up configuration..."
cp -r config/ "$BACKUP_DIR/config/"

# Logs backup
echo "📋 Backing up logs..."
cp -r logs/ "$BACKUP_DIR/logs/"

# Data backup
echo "📈 Backing up data..."
cp -r data/ "$BACKUP_DIR/data/"

echo "✅ Backup completed successfully at $BACKUP_DIR"
"""
        
        # Write scripts
        scripts = {
            'deploy-docker.sh': docker_deploy_script,
            'deploy-k8s.sh': k8s_deploy_script,
            'migrate-db.sh': db_migrate_script,
            'backup.sh': backup_script
        }
        
        for filename, script in scripts.items():
            script_path = scripts_dir / filename
            with open(script_path, 'w') as f:
                f.write(script.strip())
            # Make executable
            os.chmod(script_path, 0o755)
        
        self.logger.info("Deployment scripts created successfully")
    
    def create_monitoring_config(self):
        """Create monitoring configuration files"""
        monitoring_dir = self.deployment_dir / "monitoring"
        
        # Prometheus configuration
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'scrape_configs': [
                {
                    'job_name': 'trading-bot',
                    'static_configs': [{'targets': ['trading-bot:8080']}],
                    'metrics_path': '/metrics',
                    'scrape_interval': '10s'
                },
                {
                    'job_name': 'prometheus',
                    'static_configs': [{'targets': ['localhost:9090']}]
                }
            ]
        }
        
        with open(monitoring_dir / "prometheus.yml", 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        # Grafana dashboard configuration
        grafana_dir = monitoring_dir / "grafana"
        grafana_dir.mkdir(exist_ok=True)
        (grafana_dir / "dashboards").mkdir(exist_ok=True)
        (grafana_dir / "datasources").mkdir(exist_ok=True)
        
        # Grafana datasource
        datasource_config = {
            'apiVersion': 1,
            'datasources': [{
                'name': 'Prometheus',
                'type': 'prometheus',
                'access': 'proxy',
                'url': 'http://prometheus:9090',
                'isDefault': True
            }]
        }
        
        with open(grafana_dir / "datasources" / "prometheus.yml", 'w') as f:
            yaml.dump(datasource_config, f, default_flow_style=False)
        
        self.logger.info("Monitoring configuration created successfully")
    
    def create_nginx_config(self):
        """Create Nginx reverse proxy configuration"""
        nginx_dir = self.deployment_dir / "nginx"
        
        nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream trading_bot {
        server trading-bot:8080;
    }
    
    upstream trading_bot_ws {
        server trading-bot:8081;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 80;
        server_name _;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name trading.yourdomain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://trading_bot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WebSocket endpoints
        location /ws/ {
            proxy_pass http://trading_bot_ws;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health checks
        location /health {
            proxy_pass http://trading_bot;
            access_log off;
        }
        
        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
"""
        
        with open(nginx_dir / "nginx.conf", 'w') as f:
            f.write(nginx_config.strip())
        
        # Create SSL directory
        (nginx_dir / "ssl").mkdir(exist_ok=True)
        
        # Create self-signed certificate for development
        ssl_script = """#!/bin/bash
# Generate self-signed SSL certificate for development
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\
    -keyout deployment/nginx/ssl/key.pem \\
    -out deployment/nginx/ssl/cert.pem \\
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
"""
        
        with open(nginx_dir / "generate-ssl.sh", 'w') as f:
            f.write(ssl_script.strip())
        os.chmod(nginx_dir / "generate-ssl.sh", 0o755)
        
        self.logger.info("Nginx configuration created successfully")
    
    def create_environment_template(self):
        """Create environment variables template"""
        env_template = """# Trading Bot Environment Configuration
# Copy this file to .env and fill in your actual values

# Environment
ENVIRONMENT=production

# Exchange API Keys - ENVIRONMENT SPECIFIC (REQUIRED)
# TESTNET CREDENTIALS (for development/staging)
BYBIT_TESTNET_API_KEY=your_testnet_api_key_here
BYBIT_TESTNET_API_SECRET=your_testnet_api_secret_here

# LIVE TRADING CREDENTIALS (for production - REAL MONEY!)
BYBIT_LIVE_API_KEY=your_live_api_key_here
BYBIT_LIVE_API_SECRET=your_live_api_secret_here

# Database
DATABASE_URL=postgresql://trading_user:your_db_password@localhost:5432/trading_bot
POSTGRES_PASSWORD=your_db_password_here

# Redis
REDIS_PASSWORD=your_redis_password_here

# Security
JWT_SECRET=your_jwt_secret_here

# Email Notifications
EMAIL_PASSWORD=your_email_password_here

# Monitoring
GRAFANA_PASSWORD=your_grafana_password_here

# Domain (for SSL)
DOMAIN_NAME=trading.yourdomain.com
"""
        
        with open(self.project_root / ".env.template", 'w') as f:
            f.write(env_template.strip())
        
        self.logger.info("Environment template created successfully")
    
    def create_database_init(self):
        """Create database initialization script"""
        database_dir = self.project_root / "database"
        database_dir.mkdir(exist_ok=True)
        
        init_sql = """
-- Trading Bot Database Initialization Script
-- This script sets up the initial database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS reporting;

-- Set search path
SET search_path TO trading, public;

-- Grant permissions
GRANT USAGE ON SCHEMA trading TO trading_user;
GRANT USAGE ON SCHEMA monitoring TO trading_user;
GRANT USAGE ON SCHEMA reporting TO trading_user;

GRANT CREATE ON SCHEMA trading TO trading_user;
GRANT CREATE ON SCHEMA monitoring TO trading_user;
GRANT CREATE ON SCHEMA reporting TO trading_user;

-- Create initial tables (will be managed by Alembic migrations)
-- This is just for reference - actual schema is in migrations

COMMENT ON SCHEMA trading IS 'Core trading data and operations';
COMMENT ON SCHEMA monitoring IS 'System monitoring and health data';
COMMENT ON SCHEMA reporting IS 'Reporting and analytics data';
"""
        
        with open(database_dir / "init.sql", 'w') as f:
            f.write(init_sql.strip())
        
        self.logger.info("Database initialization script created successfully")
    
    def generate_deployment_package(self):
        """Generate complete deployment package"""
        try:
            self.logger.info("Generating complete deployment package...")
            
            # Create all deployment files
            self.create_docker_files()
            self.create_kubernetes_manifests()
            self.create_deployment_scripts()
            self.create_monitoring_config()
            self.create_nginx_config()
            self.create_environment_template()
            self.create_database_init()
            
            # Create deployment guide
            self.create_deployment_guide()
            
            self.logger.info("✅ Complete deployment package generated successfully!")
            
        except Exception as e:
            self.logger.error(f"Error generating deployment package: {e}")
            raise
    
    def create_deployment_guide(self):
        """Create comprehensive deployment guide"""
        guide_content = """# Bybit Trading Bot - Production Deployment Guide

## Overview
This guide covers the complete production deployment of the Bybit Trading Bot using Docker and Kubernetes.

## Prerequisites
- Docker and Docker Compose
- Kubernetes cluster (optional)
- PostgreSQL 15+
- Redis 7+
- SSL certificates (for HTTPS)

## Quick Start with Docker

### 1. Environment Setup
```bash
# Copy environment template
cp .env.template .env

# Edit .env with your actual values
vim .env
```

### 2. Deploy with Docker Compose
```bash
# Generate SSL certificates (for development)
./deployment/nginx/generate-ssl.sh

# Deploy the stack
./deployment/scripts/deploy-docker.sh
```

### 3. Verify Deployment
```bash
# Check service health
curl http://localhost:8080/health

# View logs
docker-compose logs -f trading-bot
```

## Kubernetes Deployment

### 1. Prepare Environment
```bash
# Set environment variables with environment-specific credentials
export BYBIT_TESTNET_API_KEY="your_testnet_api_key"
export BYBIT_TESTNET_API_SECRET="your_testnet_api_secret"
export BYBIT_LIVE_API_KEY="your_live_api_key"
export BYBIT_LIVE_API_SECRET="your_live_api_secret"
export DATABASE_PASSWORD="your_db_password"
export JWT_SECRET="your_jwt_secret"
```

### 2. Deploy to Kubernetes
```bash
# Deploy to cluster
./deployment/scripts/deploy-k8s.sh

# Check deployment status
kubectl get pods -n trading-bot
```

## Configuration

### Environment Variables

#### API Credentials (Environment-Specific)
- `BYBIT_TESTNET_API_KEY`: Your Bybit testnet API key (for development/staging)
- `BYBIT_TESTNET_API_SECRET`: Your Bybit testnet API secret (for development/staging)
- `BYBIT_LIVE_API_KEY`: Your Bybit live API key (for production - REAL MONEY!)
- `BYBIT_LIVE_API_SECRET`: Your Bybit live API secret (for production - REAL MONEY!)

#### Other Configuration
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET`: Secret for JWT token generation
- `ENVIRONMENT`: Deployment environment (development, staging, production)

### Configuration Files
- `config/config_production.yaml`: Main configuration
- `deployment/monitoring/prometheus.yml`: Metrics collection
- `deployment/nginx/nginx.conf`: Reverse proxy

## Monitoring

### Metrics and Dashboards
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Bot API: http://localhost:8080

### Health Checks
- Health endpoint: `/health`
- Readiness endpoint: `/ready`
- Metrics endpoint: `/metrics`

## Backup and Recovery

### Database Backup
```bash
# Run backup script
./deployment/scripts/backup.sh

# Manual backup
pg_dump $DATABASE_URL > backup.sql
```

### Configuration Backup
```bash
# Backup configuration
tar -czf config-backup.tar.gz config/ data/ logs/
```

## Security Considerations

### Network Security
- Use HTTPS with valid SSL certificates
- Configure firewall rules
- Enable rate limiting

### API Security
- Rotate API keys regularly
- Use environment variables for secrets
- Enable authentication on all endpoints

### Database Security
- Use strong passwords
- Enable SSL connections
- Regular security updates

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Check API credentials
   - Verify network connectivity
   - Review firewall settings

2. **Database Issues**
   - Check connection string
   - Verify database permissions
   - Run migrations: `./deployment/scripts/migrate-db.sh`

3. **Performance Issues**
   - Monitor resource usage
   - Check Grafana dashboards
   - Review application logs

### Log Locations
- Application logs: `logs/trading_bot.log`
- Docker logs: `docker-compose logs trading-bot`
- Kubernetes logs: `kubectl logs -n trading-bot deployment/trading-bot`

## Scaling

### Horizontal Scaling
```bash
# Scale with Docker Compose
docker-compose up -d --scale trading-bot=3

# Scale with Kubernetes
kubectl scale deployment trading-bot --replicas=3 -n trading-bot
```

### Vertical Scaling
- Adjust resource limits in deployment configurations
- Monitor performance metrics
- Scale based on trading volume

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Consult the API documentation
4. Contact support team

## Version Information
- Bot Version: 1.0.0
- Docker Image: bybit-trading-bot:latest
- Kubernetes Version: 1.25+
- Python Version: 3.11+
"""
        
        with open(self.deployment_dir / "DEPLOYMENT_GUIDE.md", 'w') as f:
            f.write(guide_content.strip())
        
        self.logger.info("Deployment guide created successfully")


# Main deployment script
def main():
    """Main deployment package generation"""
    print("Phase 10: Production Deployment Scripts")
    print("=" * 60)
    
    # Initialize deployment manager
    project_root = Path(__file__).parent.parent.parent
    deployment_manager = DeploymentManager(str(project_root))
    
    # Generate complete deployment package
    deployment_manager.generate_deployment_package()
    
    print(f"\n🎉 Production Deployment Package Complete!")
    print(f"✅ Docker containerization")
    print(f"✅ Kubernetes orchestration")
    print(f"✅ Environment configuration")
    print(f"✅ Database initialization")
    print(f"✅ Monitoring setup")
    print(f"✅ Nginx reverse proxy")
    print(f"✅ SSL/TLS configuration")
    print(f"✅ Deployment automation scripts")
    print(f"✅ Comprehensive deployment guide")
    
    print(f"\n📁 Deployment files created in:")
    print(f"   - Dockerfile")
    print(f"   - docker-compose.yml")
    print(f"   - requirements.txt")
    print(f"   - deployment/kubernetes/")
    print(f"   - deployment/scripts/")
    print(f"   - deployment/monitoring/")
    print(f"   - deployment/nginx/")
    print(f"   - .env.template")
    
    print(f"\n🚀 Ready for production deployment!")


if __name__ == "__main__":
    main()