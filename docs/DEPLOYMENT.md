# Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Checklist](#production-checklist)
7. [Monitoring Setup](#monitoring-setup)
8. [Backup and Recovery](#backup-and-recovery)

## Overview

This guide covers deployment options for the Bybit Trading Bot, from local development to production cloud deployment. The bot supports multiple deployment methods:

- **Local Development**: Direct Python execution for development and testing
- **Docker**: Containerized deployment for consistency across environments
- **Cloud**: AWS/GCP/Azure deployment with auto-scaling and monitoring
- **Hybrid**: Local execution with cloud monitoring and data storage

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores, 2.4 GHz
- RAM: 4 GB
- Storage: 10 GB available space
- Network: Stable internet connection with low latency

**Recommended Requirements:**
- CPU: 4+ cores, 3.0+ GHz
- RAM: 8+ GB
- Storage: 20+ GB SSD
- Network: High-speed internet with < 50ms latency to Bybit

### Software Dependencies

```bash
# Python 3.11+
python --version  # Should be 3.11 or higher

# Git
git --version

# Docker (for containerized deployment)
docker --version
docker-compose --version

# Optional: Node.js (for frontend monitoring dashboard)
node --version
npm --version
```

## Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/bybit-trading-bot.git
cd bybit-trading-bot
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
# Add your Bybit API credentials and other settings
```

Example `.env` file:
```env
# Bybit API
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Database (optional for development)
DATABASE_URL=sqlite:///data/trading.db

# Monitoring (optional)
PROMETHEUS_PORT=8080
```

### 4. Initialize Database

```bash
# Run database migrations
python scripts/init_db.py

# Create initial configuration
python scripts/create_default_config.py
```

### 5. Run the Bot

```bash
# Start the trading bot
python main.py

# Or with specific configuration
python main.py --config config/development.yaml

# Run in background (Unix)
nohup python main.py > logs/bot.log 2>&1 &
```

### 6. Verify Installation

```bash
# Check bot status
curl http://localhost:8080/health

# View metrics
curl http://localhost:8080/metrics

# Check logs
tail -f logs/trading_bot.log
```

## Docker Deployment

### 1. Build Docker Image

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Create directories
RUN mkdir -p logs data

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the bot
CMD ["python", "main.py"]
```

### 2. Build and Run

```bash
# Build image
docker build -t bybit-trading-bot:latest .

# Run container
docker run -d \
  --name trading-bot \
  --env-file .env \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  bybit-trading-bot:latest

# Check container status
docker ps
docker logs trading-bot
```

### 3. Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  trading-bot:
    build: .
    container_name: trading-bot
    env_file: .env
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - trading-network

  # PostgreSQL database (optional)
  postgres:
    image: postgres:15
    container_name: trading-db
    environment:
      POSTGRES_DB: trading_bot
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - trading-network

  # Redis for caching (optional)
  redis:
    image: redis:7-alpine
    container_name: trading-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - trading-network

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: trading-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    networks:
      - trading-network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: trading-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    networks:
      - trading-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  trading-network:
    driver: bridge
```

### 4. Start Complete Stack

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f trading-bot

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup

```bash
# Launch EC2 instance (Amazon Linux 2)
# t3.medium or larger recommended for production

# Connect to instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone https://github.com/your-org/bybit-trading-bot.git
cd bybit-trading-bot

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d
```

#### 2. ECS Deployment

```yaml
# ecs-task-definition.json
{
  "family": "bybit-trading-bot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::your-account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::your-account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "trading-bot",
      "image": "your-account.dkr.ecr.region.amazonaws.com/bybit-trading-bot:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "BYBIT_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:bybit-api-key"
        },
        {
          "name": "BYBIT_API_SECRET",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:bybit-api-secret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bybit-trading-bot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### 3. CloudFormation Template

```yaml
# infrastructure/cloudformation.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Bybit Trading Bot Infrastructure'

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]
  
  InstanceType:
    Type: String
    Default: t3.medium
    Description: EC2 instance type

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-trading-bot-vpc

  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-trading-bot-public-subnet

  PrivateSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-trading-bot-private-subnet

  # Security Groups
  TradingBotSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for trading bot
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8080
          ToPort: 8080
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-trading-bot-sg

  # RDS Database
  DatabaseSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for trading bot database
      SubnetIds:
        - !Ref PublicSubnet
        - !Ref PrivateSubnet
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-trading-bot-db-subnet-group

  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: !Sub ${Environment}-trading-bot-db
      DBInstanceClass: db.t3.micro
      Engine: postgres
      EngineVersion: '15.3'
      MasterUsername: trading_user
      MasterUserPassword: !Ref DatabasePassword
      AllocatedStorage: 20
      DBSubnetGroupName: !Ref DatabaseSubnetGroup
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      BackupRetentionPeriod: 7
      MultiAZ: false
      PubliclyAccessible: false

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub ${Environment}-trading-bot-alb
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet
        - !Ref PrivateSubnet
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # Auto Scaling Group
  LaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateName: !Sub ${Environment}-trading-bot-template
      LaunchTemplateData:
        ImageId: ami-0abcdef1234567890  # Amazon Linux 2 AMI
        InstanceType: !Ref InstanceType
        SecurityGroupIds:
          - !Ref TradingBotSecurityGroup
        UserData:
          Fn::Base64: !Sub |
            #!/bin/bash
            yum update -y
            yum install -y docker
            systemctl start docker
            systemctl enable docker
            usermod -a -G docker ec2-user
            
            # Install docker-compose
            curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            chmod +x /usr/local/bin/docker-compose
            
            # Clone and start application
            cd /home/ec2-user
            git clone https://github.com/your-org/bybit-trading-bot.git
            cd bybit-trading-bot
            
            # Setup environment
            cat > .env << EOF
            ENVIRONMENT=${Environment}
            BYBIT_API_KEY=${BybitAPIKey}
            BYBIT_API_SECRET=${BybitAPISecret}
            DATABASE_URL=postgresql://trading_user:${DatabasePassword}@${Database.Endpoint.Address}:5432/trading_bot
            EOF
            
            # Start services
            docker-compose up -d

Parameters:
  DatabasePassword:
    Type: String
    NoEcho: true
    Description: Password for the RDS database
  
  BybitAPIKey:
    Type: String
    NoEcho: true
    Description: Bybit API Key
  
  BybitAPISecret:
    Type: String
    NoEcho: true
    Description: Bybit API Secret

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt ApplicationLoadBalancer.DNSName
  
  DatabaseEndpoint:
    Description: RDS database endpoint
    Value: !GetAtt Database.Endpoint.Address
```

### Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: trading-bot

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-bot-config
  namespace: trading-bot
data:
  config.yaml: |
    app:
      name: "Bybit Trading Bot"
      environment: "production"
    trading:
      enabled: true
      max_concurrent_trades: 5
    # ... rest of config

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-bot-secrets
  namespace: trading-bot
type: Opaque
data:
  bybit-api-key: <base64-encoded-api-key>
  bybit-api-secret: <base64-encoded-api-secret>
  database-password: <base64-encoded-db-password>

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-bot
  namespace: trading-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-bot
  template:
    metadata:
      labels:
        app: trading-bot
    spec:
      containers:
      - name: trading-bot
        image: bybit-trading-bot:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: BYBIT_API_KEY
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: bybit-api-key
        - name: BYBIT_API_SECRET
          valueFrom:
            secretKeyRef:
              name: trading-bot-secrets
              key: bybit-api-secret
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
      volumes:
      - name: config-volume
        configMap:
          name: trading-bot-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: trading-bot-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: trading-bot-service
  namespace: trading-bot
spec:
  selector:
    app: trading-bot
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer

---
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: trading-bot-pvc
  namespace: trading-bot
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

## Production Checklist

### Pre-Deployment

- [ ] API credentials configured securely
- [ ] Configuration files validated
- [ ] Environment variables set
- [ ] Database migrations completed
- [ ] SSL/TLS certificates configured
- [ ] Firewall rules configured
- [ ] Monitoring setup completed
- [ ] Backup strategy implemented
- [ ] Testing completed in staging environment

### Security Checklist

- [ ] API keys stored in secure vault (AWS Secrets Manager, etc.)
- [ ] Database connections encrypted
- [ ] Application runs as non-root user
- [ ] Network access restricted
- [ ] Logs don't contain sensitive information
- [ ] Regular security updates scheduled
- [ ] Access logging enabled
- [ ] Rate limiting configured

### Performance Checklist

- [ ] Resource limits configured
- [ ] Auto-scaling rules defined
- [ ] Database performance optimized
- [ ] Caching strategy implemented
- [ ] Load balancer configured
- [ ] CDN setup for static assets
- [ ] Database connection pooling enabled

### Monitoring Checklist

- [ ] Health checks configured
- [ ] Metrics collection enabled
- [ ] Alerting rules defined
- [ ] Log aggregation setup
- [ ] Dashboard created
- [ ] SLA monitoring implemented
- [ ] Error tracking enabled

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "trading_bot_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8080']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Bybit Trading Bot Dashboard",
    "panels": [
      {
        "title": "Active Trades",
        "type": "stat",
        "targets": [
          {
            "expr": "trading_bot_active_trades_total",
            "legendFormat": "Active Trades"
          }
        ]
      },
      {
        "title": "Portfolio Value",
        "type": "graph",
        "targets": [
          {
            "expr": "trading_bot_portfolio_value_usd",
            "legendFormat": "Portfolio Value (USD)"
          }
        ]
      },
      {
        "title": "Trade Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(trading_bot_successful_trades_total[5m]) / rate(trading_bot_total_trades_total[5m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ]
      }
    ]
  }
}
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_DIR="/backups/database"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/trading_bot_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Create database backup
pg_dump ${DATABASE_URL} > ${BACKUP_FILE}

# Compress backup
gzip ${BACKUP_FILE}

# Remove backups older than 30 days
find ${BACKUP_DIR} -name "*.gz" -mtime +30 -delete

echo "Database backup completed: ${BACKUP_FILE}.gz"
```

### Configuration Backup

```bash
#!/bin/bash
# scripts/backup_config.sh

BACKUP_DIR="/backups/config"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Backup configuration files
tar -czf "${BACKUP_DIR}/config_${TIMESTAMP}.tar.gz" config/

# Backup environment file (without sensitive data)
grep -v -E "(API_KEY|API_SECRET|PASSWORD)" .env > "${BACKUP_DIR}/.env.template"

echo "Configuration backup completed"
```

### Recovery Procedures

```bash
#!/bin/bash
# scripts/restore_database.sh

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "Restoring database from ${BACKUP_FILE}..."

# Stop trading bot
docker-compose stop trading-bot

# Restore database
gunzip -c ${BACKUP_FILE} | psql ${DATABASE_URL}

# Start trading bot
docker-compose start trading-bot

echo "Database restoration completed"
```

## Troubleshooting

### Common Issues

1. **Bot won't start**
   - Check environment variables
   - Verify API credentials
   - Check database connectivity
   - Review logs for errors

2. **High memory usage**
   - Check for memory leaks in logs
   - Adjust caching settings
   - Increase container memory limits

3. **API errors**
   - Verify API credentials
   - Check rate limiting
   - Confirm network connectivity
   - Review Bybit API status

4. **Database connection issues**
   - Check database server status
   - Verify connection string
   - Check network connectivity
   - Review database logs

### Log Analysis

```bash
# View recent logs
docker-compose logs --tail=100 -f trading-bot

# Search for errors
docker-compose logs trading-bot | grep ERROR

# Check specific time range
docker-compose logs --since="2024-01-01T00:00:00" trading-bot

# Export logs for analysis
docker-compose logs trading-bot > /tmp/trading-bot.log
```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats trading-bot

# Check database performance
docker-compose exec postgres psql -U trading_user -d trading_bot -c "SELECT * FROM pg_stat_activity;"

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8080/health"
```