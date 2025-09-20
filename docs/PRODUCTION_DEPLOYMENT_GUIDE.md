# 🚀 Production Deployment Guide - ML Trading Bot

## Overview

This guide covers deploying the production-ready ML Trading Bot to enterprise environments using modern container orchestration and CI/CD practices. The system is designed for Kubernetes deployment with comprehensive monitoring, security, and auto-scaling capabilities.

**Production Stack**: FastAPI + Streamlit + Kubernetes + CI/CD + Monitoring
**Infrastructure**: Container-native with horizontal scaling and zero-downtime deployments

## 📋 Table of Contents

1. [🎯 Quick Production Deployment](#quick-production-deployment)
2. [☸️ Kubernetes Production Setup](#kubernetes-production-setup)
3. [🔧 Configuration Management](#configuration-management)
4. [🛡️ Security & Secrets Management](#security--secrets-management)
5. [📊 Monitoring & Observability](#monitoring--observability)
6. [🔄 CI/CD Pipeline Setup](#cicd-pipeline-setup)
7. [📈 Scaling & Performance](#scaling--performance)
8. [🚨 Disaster Recovery & Backup](#disaster-recovery--backup)
9. [🔧 Maintenance & Operations](#maintenance--operations)
10. [🐛 Troubleshooting Guide](#troubleshooting-guide)

---

## 🎯 Quick Production Deployment

### **⚡ 5-Minute Production Setup**

**Prerequisites**: Kubernetes cluster, kubectl, docker

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Bybit-bot.git
cd Bybit-bot

# 2. Configure production environment
cp config/production.yaml.template config/production.yaml
cp config/secrets.yaml.template config/secrets.yaml

# 3. Set up secrets in Kubernetes
python deploy.py setup-secrets --environment production

# 4. Deploy to Kubernetes
python deploy.py deploy --environment production

# 5. Verify deployment
python deploy.py health-check --environment production
```

**Access your deployment**:
- **API**: `https://your-domain.com/docs`
- **Dashboard**: `https://your-domain.com/dashboard`
- **Monitoring**: `https://your-domain.com/grafana`

### **🔧 Quick Configuration**

```yaml
# config/production.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  
ml_models:
  ensemble_weight:
    lightgbm: 0.4
    xgboost: 0.3
    neural_network: 0.2
    transformer: 0.1
    
kubernetes:
  replicas: 3
  auto_scaling: true
  resources:
    requests:
      cpu: "500m"
      memory: "1Gi"
    limits:
      cpu: "2"
      memory: "4Gi"
```

---

## ☸️ Kubernetes Production Setup

### **📦 Container Registry Setup**

```bash
# Build and push production images
docker build -t your-registry/ml-trading-bot:latest .
docker push your-registry/ml-trading-bot:latest

# Update deployment with your registry
sed -i 's|ml-trading-bot:latest|your-registry/ml-trading-bot:latest|g' k8s/deployment.yaml
```

### **🔐 Secrets Management**

```bash
# Create Kubernetes secrets
kubectl create secret generic ml-trading-bot-secrets \
  --from-literal=BYBIT_API_KEY="your-api-key" \
  --from-literal=BYBIT_API_SECRET="your-api-secret" \
  --from-literal=DATABASE_URL="postgresql://user:pass@host:5432/db" \
  --from-literal=REDIS_URL="redis://redis:6379/0" \
  --from-literal=JWT_SECRET="your-jwt-secret"

# Or use the automated script
python scripts/setup_k8s_secrets.py --environment production
```

### **📊 Deploy Core Services**

```bash
# Deploy in order
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Or use the deployment script
python deploy.py deploy --environment production --wait-ready
```

### **🎯 Production Deployment Manifest**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-trading-bot
  namespace: trading-bot
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ml-trading-bot
  template:
    metadata:
      labels:
        app: ml-trading-bot
        version: v1.0.0
    spec:
      containers:
      - name: api
        image: ml-trading-bot:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        envFrom:
        - secretRef:
            name: ml-trading-bot-secrets
        - configMapRef:
            name: ml-trading-bot-config
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### **⚖️ Auto-scaling Configuration**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-trading-bot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-trading-bot
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: predictions_per_second
      target:
        type: AverageValue
        averageValue: "50"
```

### **🌐 Ingress & Load Balancing**

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-trading-bot-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "60s"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: ml-trading-bot-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-trading-bot-service
            port:
              number: 8000
```

---

## 🔄 CI/CD Pipeline Setup

### **🚀 GitHub Actions Production Pipeline**

```yaml
# .github/workflows/production-deploy.yml
name: Production Deployment

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
        
    - name: Security scan
      run: |
        bandit -r src/
        safety check
        
    - name: Code quality
      run: |
        pylint src/
        mypy src/

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    outputs:
      image: ${{ steps.image.outputs.image }}
      digest: ${{ steps.build.outputs.digest }}
    steps:
    - name: Checkout
      uses: actions/checkout@v3
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
      
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          
    - name: Build and push
      id: build
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: production
    steps:
    - name: Checkout
      uses: actions/checkout@v3
      
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
        
    - name: Deploy to Kubernetes
      run: |
        # Update image in deployment
        kubectl set image deployment/ml-trading-bot \
          api=${{ needs.build-and-push.outputs.image }}@${{ needs.build-and-push.outputs.digest }}
          
        # Wait for rollout
        kubectl rollout status deployment/ml-trading-bot --timeout=300s
        
        # Run health checks
        python scripts/health_check.py --environment production --wait-ready
```

### **🔧 Automated Deployment Scripts**

```python
# deploy.py - Production deployment automation
import click
import subprocess
import yaml
from pathlib import Path

@click.group()
def cli():
    """ML Trading Bot Deployment CLI"""
    pass

@cli.command()
@click.option('--environment', default='production', help='Deployment environment')
@click.option('--wait-ready', is_flag=True, help='Wait for deployment to be ready')
def deploy(environment, wait_ready):
    """Deploy to Kubernetes cluster"""
    click.echo(f"🚀 Deploying to {environment}")
    
    # Apply Kubernetes manifests
    manifests = [
        'k8s/namespace.yaml',
        'k8s/configmap.yaml', 
        'k8s/postgres.yaml',
        'k8s/redis.yaml',
        'k8s/deployment.yaml',
        'k8s/service.yaml',
        'k8s/ingress.yaml',
        'k8s/hpa.yaml'
    ]
    
    for manifest in manifests:
        click.echo(f"Applying {manifest}")
        subprocess.run(['kubectl', 'apply', '-f', manifest], check=True)
    
    if wait_ready:
        click.echo("⏳ Waiting for deployment to be ready...")
        subprocess.run([
            'kubectl', 'rollout', 'status', 
            'deployment/ml-trading-bot', '--timeout=300s'
        ], check=True)
        
        # Run health checks
        subprocess.run([
            'python', 'scripts/health_check.py',
            '--environment', environment,
            '--wait-ready'
        ], check=True)
        
    click.echo("✅ Deployment completed successfully!")

@cli.command()
@click.option('--environment', default='production')
def health_check(environment):
    """Run comprehensive health checks"""
    click.echo(f"🏥 Running health checks for {environment}")
    
    # Check API health
    subprocess.run([
        'kubectl', 'exec', 'deployment/ml-trading-bot', '--',
        'curl', '-f', 'http://localhost:8000/health'
    ], check=True)
    
    # Check database connectivity
    subprocess.run([
        'python', 'scripts/health_check.py',
        '--check', 'database',
        '--environment', environment
    ], check=True)
    
    click.echo("✅ All health checks passed!")

if __name__ == '__main__':
    cli()
```

---

## 📊 Monitoring & Observability

### **📈 Prometheus Monitoring Setup**

```yaml
# k8s/monitoring/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'ml-trading-bot'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: ml-trading-bot
    - job_name: 'kubernetes-nodes'
      kubernetes_sd_configs:
      - role: node
    rule_files:
    - "alert_rules.yml"
    alerting:
      alertmanagers:
      - static_configs:
        - targets: ['alertmanager:9093']
```

### **📊 Grafana Dashboard Configuration**

```json
{
  "dashboard": {
    "title": "ML Trading Bot - Production Dashboard",
    "panels": [
      {
        "title": "Predictions Per Minute",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(predictions_total[5m])*60",
            "legendFormat": "Predictions/min"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "heatmap",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### **🚨 Alerting Rules**

```yaml
# k8s/monitoring/alert-rules.yaml
groups:
- name: ml-trading-bot-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
      
  - alert: ModelAccuracyDrop
    expr: model_accuracy < 0.75
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy has dropped"
      description: "Model {{ $labels.model }} accuracy is {{ $value }}"
      
  - alert: PodMemoryHigh
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Pod memory usage high"
      description: "Pod {{ $labels.pod }} memory usage is {{ $value | humanizePercentage }}"
```

---

## 🛡️ Security & Secrets Management

### **🔐 Production Security Configuration**

```bash
# Generate production secrets
python scripts/generate_secrets.py --environment production

# Create Kubernetes secrets with proper labels
kubectl create secret generic ml-trading-bot-secrets \
  --from-env-file=config/.env.production \
  --dry-run=client -o yaml | \
  kubectl label --local -f - environment=production app=ml-trading-bot -o yaml | \
  kubectl apply -f -
```

### **🔒 Network Security Policies**

```yaml
# k8s/security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-trading-bot-network-policy
spec:
  podSelector:
    matchLabels:
      app: ml-trading-bot
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```
- [ ] Tax reporting systems configured
- [ ] Trading permissions and licenses obtained
- [ ] Insurance and liability coverage reviewed
- [ ] Terms of service and privacy policies updated

---

## 🏗️ Infrastructure Planning

### **Minimum Production Requirements**

#### **Single Server Deployment**
- **CPU**: 4 cores, 3.0 GHz+
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 100 GB SSD
- **Network**: Dedicated internet with <10ms latency to exchange
- **Uptime**: 99.9% availability with UPS backup

#### **High Availability Deployment**
- **Primary Server**: As above
- **Backup Server**: Identical specs for failover
- **Load Balancer**: Nginx or HAProxy
- **Database**: PostgreSQL cluster with replication
- **Monitoring**: Dedicated monitoring server

### **Cloud Platform Recommendations**

#### **AWS (Amazon Web Services)**
```
Instance Type: t3.large or c5.xlarge
CPU: 2-4 vCPUs
RAM: 8-16 GB
Storage: 100 GB gp3 SSD
Network: Enhanced networking enabled
```

#### **Google Cloud Platform**
```
Instance Type: n1-standard-4 or n2-standard-4
CPU: 4 vCPUs
RAM: 15 GB
Storage: 100 GB persistent SSD
Network: Premium tier
```

#### **Digital Ocean**
```
Droplet: $40/month plan or higher
CPU: 4 vCPUs
RAM: 8 GB
Storage: 160 GB SSD
Network: Dedicated CPU recommended
```

### **Network Requirements**
- **Bandwidth**: Minimum 100 Mbps up/down
- **Latency**: <50ms to Bybit servers (Singapore/AWS)
- **Redundancy**: Backup internet connection recommended
- **VPN**: Optional but recommended for security

---

## 🌍 Environment Setup

### **Linux Server Setup (Ubuntu 22.04 LTS)**

#### **1. Initial Server Configuration**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git vim htop unzip ufw

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable

# Create trading user
sudo adduser trading
sudo usermod -aG sudo trading
sudo su - trading
```

#### **2. Install Dependencies**
```bash
# Install Python 3.11
sudo apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Install Redis (optional, for caching)
sudo apt install -y redis-server

# Install Docker (for containerized deployment)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker trading

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### **3. Configure Services**
```bash
# Configure PostgreSQL
sudo -u postgres psql
CREATE USER trading_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE trading_bot_prod OWNER trading_user;
GRANT ALL PRIVILEGES ON DATABASE trading_bot_prod TO trading_user;
\q

# Configure Redis
sudo nano /etc/redis/redis.conf
# Uncomment and set: requirepass your_redis_password
sudo systemctl restart redis-server

# Enable services
sudo systemctl enable postgresql
sudo systemctl enable redis-server
```

### **Windows Server Setup**

#### **1. Windows Server 2019/2022**
```powershell
# Install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install Python
choco install python311 -y

# Install Git
choco install git -y

# Install PostgreSQL
choco install postgresql --params '/Password:your_secure_password' -y

# Refresh environment
refreshenv
```

#### **2. Configure Windows Services**
- Set up Windows Firewall rules
- Configure automatic startup for bot service
- Set up scheduled tasks for maintenance
- Configure Windows Update settings

---

## 🐳 Docker Deployment

### **1. Build Production Image**

```bash
# Clone repository
git clone https://github.com/Zvxndr/Bybit-bot.git
cd Bybit-bot

# Build production image
docker build -t bybit-trading-bot:latest .

# Verify image
docker images | grep bybit-trading-bot
```

### **2. Configure Environment Variables**

Create `.env.prod` file:
```bash
# Environment
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://trading_user:your_secure_password@postgres:5432/trading_bot_prod

# Redis
REDIS_URL=redis://:your_redis_password@redis:6379/0

# Exchange API (LIVE TRADING - BE CAREFUL!)
BYBIT_LIVE_API_KEY=your_live_api_key
BYBIT_LIVE_API_SECRET=your_live_api_secret

# Testnet (for testing)
BYBIT_TESTNET_API_KEY=your_testnet_api_key
BYBIT_TESTNET_API_SECRET=your_testnet_api_secret

# Email Alerts
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENTS=admin@yourcompany.com,trader@yourcompany.com

# Security
JWT_SECRET=your_jwt_secret_key_here_32_chars_minimum
API_KEY=your_api_access_key

# Monitoring
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_HEALTH_CHECKS=true
HEALTH_CHECK_INTERVAL=30

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=/app/logs/trading_bot.log
```

### **3. Docker Compose Configuration**

Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  # Trading Bot Application
  trading-bot:
    image: bybit-trading-bot:latest
    container_name: bybit-trading-bot-prod
    restart: unless-stopped
    env_file:
      - .env.prod
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
    image: postgres:15
    container_name: trading-postgres-prod
    restart: unless-stopped
    environment:
      POSTGRES_DB: trading_bot_prod
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: your_secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user -d trading_bot_prod"]
      interval: 30s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: trading-redis-prod
    restart: unless-stopped
    command: redis-server --requirepass your_redis_password
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 5s
      retries: 5

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: trading-nginx-prod
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

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    container_name: trading-prometheus
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - trading-network

  grafana:
    image: grafana/grafana:latest
    container_name: trading-grafana
    restart: unless-stopped
    environment:
      GF_SECURITY_ADMIN_PASSWORD: your_grafana_password
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    networks:
      - trading-network
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  trading-network:
    driver: bridge
```

### **4. Deploy with Docker Compose**

```bash
# Deploy all services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f trading-bot

# Scale if needed
docker-compose -f docker-compose.prod.yml up -d --scale trading-bot=2
```

---

## ☁️ Cloud Platform Deployment

### **AWS Deployment**

#### **1. EC2 Instance Setup**
```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type t3.large \
    --key-name your-key-pair \
    --security-group-ids sg-your-security-group \
    --subnet-id subnet-your-subnet

# Configure security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-your-security-group \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-your-security-group \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0
```

#### **2. RDS Database Setup**
```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
    --db-instance-identifier trading-bot-db \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --engine-version 15.4 \
    --master-username trading_user \
    --master-user-password your_secure_password \
    --allocated-storage 100 \
    --storage-encrypted \
    --vpc-security-group-ids sg-your-db-security-group
```

#### **3. ECS/Fargate Deployment**
```yaml
# task-definition.json
{
  "family": "bybit-trading-bot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "trading-bot",
      "image": "your-account.dkr.ecr.region.amazonaws.com/bybit-trading-bot:latest",
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
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:ssm:region:account:parameter/trading-bot/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bybit-trading-bot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### **Google Cloud Platform Deployment**

#### **1. Compute Engine Setup**
```bash
# Create VM instance
gcloud compute instances create trading-bot-vm \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --tags=http-server,https-server

# Configure firewall
gcloud compute firewall-rules create allow-trading-bot \
    --allow tcp:80,tcp:443,tcp:8080 \
    --source-ranges 0.0.0.0/0 \
    --target-tags http-server,https-server
```

#### **2. Cloud SQL Setup**
```bash
# Create PostgreSQL instance
gcloud sql instances create trading-bot-db \
    --database-version=POSTGRES_15 \
    --cpu=2 \
    --memory=7680MB \
    --storage-size=100GB \
    --storage-type=SSD \
    --region=us-central1

# Create database and user
gcloud sql databases create trading_bot_prod --instance=trading-bot-db
gcloud sql users create trading_user --instance=trading-bot-db --password=your_secure_password
```

#### **3. Cloud Run Deployment**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project/bybit-trading-bot

# Deploy to Cloud Run
gcloud run deploy bybit-trading-bot \
    --image gcr.io/your-project/bybit-trading-bot \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 3 \
    --set-env-vars ENVIRONMENT=production
```

---

## 🗄️ Database Setup

### **Production Database Configuration**

#### **1. PostgreSQL Optimization**
```sql
-- postgresql.conf optimizations
shared_buffers = 2GB                    # 25% of RAM
effective_cache_size = 6GB              # 75% of RAM
maintenance_work_mem = 512MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1                  # For SSD
effective_io_concurrency = 200          # For SSD
max_connections = 100
```

#### **2. Database Security**
```sql
-- Create dedicated schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS monitoring; 
CREATE SCHEMA IF NOT EXISTS reporting;

-- Create read-only user for reporting
CREATE USER trading_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE trading_bot_prod TO trading_readonly;
GRANT USAGE ON SCHEMA trading TO trading_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA trading TO trading_readonly;

-- Enable connection logging
log_connections = on
log_disconnections = on
log_statement = 'all'
```

#### **3. Backup Configuration**
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="trading_bot_prod"

# Create backup
pg_dump -h localhost -U trading_user -d $DB_NAME > $BACKUP_DIR/backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/backup_$DATE.sql

# Remove backups older than 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
aws s3 cp $BACKUP_DIR/backup_$DATE.sql.gz s3://your-backup-bucket/
```

### **Migration Management**

#### **1. Alembic Configuration**
```python
# alembic.ini
[alembic]
script_location = migrations
sqlalchemy.url = postgresql://trading_user:password@localhost/trading_bot_prod

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic
```

#### **2. Deployment Migration**
```bash
# Run migrations
alembic upgrade head

# Backup before major migrations
pg_dump trading_bot_prod > backup_before_migration.sql

# Run specific migration
alembic upgrade +1

# Rollback if needed
alembic downgrade -1
```

---

## 🔒 Security Configuration

### **1. SSL/TLS Configuration**

#### **Nginx SSL Configuration**
```nginx
# /etc/nginx/sites-available/trading-bot
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/your-domain.crt;
    ssl_certificate_key /etc/nginx/ssl/your-domain.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Proxy to trading bot
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://localhost:8081;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### **Let's Encrypt SSL Certificate**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **2. Firewall Configuration**

#### **UFW (Ubuntu)**
```bash
# Reset firewall
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH access (change port if not 22)
sudo ufw allow 22/tcp

# HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Database (restrict to specific IPs)
sudo ufw allow from your.server.ip.address to any port 5432

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

#### **iptables (Advanced)**
```bash
# Save current rules
iptables-save > iptables-backup.rules

# Basic security rules
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### **3. API Security**

#### **API Key Management**
```python
# Use environment variables or secrets management
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.encryption_key = os.environ.get('ENCRYPTION_KEY')
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_api_key(self, api_key: str) -> str:
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        return self.cipher.decrypt(encrypted_key.encode()).decode()
```

#### **Rate Limiting**
```python
from fastapi import FastAPI, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/status")
@limiter.limit("10/minute")
async def get_status(request: Request):
    return {"status": "healthy"}
```

---

## 📊 Monitoring & Alerting

### **1. Prometheus Configuration**

#### **prometheus.yml**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "trading_bot_rules.yml"

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8080']
    scrape_interval: 10s
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### **Alerting Rules**
```yaml
# trading_bot_rules.yml
groups:
  - name: trading_bot_alerts
    rules:
      - alert: TradingBotDown
        expr: up{job="trading-bot"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Trading bot is down"
          description: "Trading bot has been down for more than 30 seconds"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes{job="trading-bot"} > 2e9
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Trading bot memory usage is above 2GB"

      - alert: DatabaseConnectionFailed
        expr: trading_bot_database_connections_failed_total > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failed"
          description: "Trading bot cannot connect to database"
```

### **2. Grafana Dashboards**

#### **Trading Bot Dashboard**
```json
{
  "dashboard": {
    "id": null,
    "title": "Trading Bot Dashboard",
    "panels": [
      {
        "title": "Portfolio Value",
        "type": "stat",
        "targets": [
          {
            "expr": "trading_bot_portfolio_value",
            "legendFormat": "Portfolio Value"
          }
        ]
      },
      {
        "title": "Daily P&L",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(trading_bot_pnl_total[1h])",
            "legendFormat": "Hourly P&L"
          }
        ]
      },
      {
        "title": "Active Positions",
        "type": "table",
        "targets": [
          {
            "expr": "trading_bot_active_positions",
            "legendFormat": "Positions"
          }
        ]
      }
    ]
  }
}
```

### **3. Log Management**

#### **Centralized Logging with ELK Stack**
```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logs:/app/logs
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch
```

---

## 🔄 CI/CD Pipeline

### **1. GitHub Actions Workflow**

#### **.github/workflows/deploy.yml**
```yaml
name: Deploy Trading Bot

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src tests/
    
    - name: Run linting
      run: |
        flake8 src/
        black --check src/
        mypy src/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ghcr.io/${{ github.repository }}:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /opt/trading-bot
          docker-compose pull
          docker-compose up -d --force-recreate
          docker system prune -f
```

### **2. Deployment Scripts**

#### **deploy.sh**
```bash
#!/bin/bash
set -euo pipefail

# Configuration
APP_NAME="bybit-trading-bot"
DEPLOY_DIR="/opt/trading-bot"
BACKUP_DIR="/backup/deployments"
DATE=$(date +%Y%m%d_%H%M%S)

echo "🚀 Starting deployment of $APP_NAME at $(date)"

# Create backup
echo "📦 Creating backup..."
mkdir -p $BACKUP_DIR
docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml down
tar -czf $BACKUP_DIR/backup_$DATE.tar.gz -C $DEPLOY_DIR .

# Pull latest changes
echo "📥 Pulling latest changes..."
cd $DEPLOY_DIR
git pull origin main

# Pull latest Docker images
echo "🐳 Pulling Docker images..."
docker-compose -f docker-compose.prod.yml pull

# Run database migrations
echo "🗄️ Running database migrations..."
docker-compose -f docker-compose.prod.yml run --rm trading-bot alembic upgrade head

# Deploy new version
echo "🔄 Deploying new version..."
docker-compose -f docker-compose.prod.yml up -d

# Health check
echo "🏥 Running health checks..."
sleep 30
if curl -f http://localhost:8080/health; then
    echo "✅ Deployment successful!"
    
    # Clean up old backups (keep last 10)
    ls -t $BACKUP_DIR/backup_*.tar.gz | tail -n +11 | xargs -r rm
    
    # Clean up Docker
    docker system prune -f
else
    echo "❌ Health check failed, rolling back..."
    docker-compose -f docker-compose.prod.yml down
    tar -xzf $BACKUP_DIR/backup_$DATE.tar.gz -C $DEPLOY_DIR
    docker-compose -f docker-compose.prod.yml up -d
    exit 1
fi

echo "🎉 Deployment completed at $(date)"
```

#### **rollback.sh**
```bash
#!/bin/bash
set -euo pipefail

BACKUP_DIR="/backup/deployments"
DEPLOY_DIR="/opt/trading-bot"

echo "🔄 Rolling back deployment..."

# List available backups
echo "Available backups:"
ls -la $BACKUP_DIR/backup_*.tar.gz

# Get latest backup
LATEST_BACKUP=$(ls -t $BACKUP_DIR/backup_*.tar.gz | head -n 1)
echo "Using backup: $LATEST_BACKUP"

# Stop current version
docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml down

# Restore from backup
tar -xzf $LATEST_BACKUP -C $DEPLOY_DIR

# Start restored version
docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml up -d

# Health check
sleep 30
if curl -f http://localhost:8080/health; then
    echo "✅ Rollback successful!"
else
    echo "❌ Rollback failed!"
    exit 1
fi
```

---

## 🔧 Maintenance & Updates

### **1. Regular Maintenance Tasks**

#### **Daily Tasks**
```bash
#!/bin/bash
# daily_maintenance.sh

# Check system health
curl -f http://localhost:8080/health || exit 1

# Check disk space
df -h | awk '$5 > 80 {print "High disk usage: " $0}'

# Check memory usage
free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'

# Check log sizes
find /opt/trading-bot/logs -name "*.log" -size +100M -exec ls -lh {} \;

# Database maintenance
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_bot_prod -c "VACUUM ANALYZE;"

# Generate daily report
curl -X POST http://localhost:8080/api/v1/reports/daily
```

#### **Weekly Tasks**
```bash
#!/bin/bash
# weekly_maintenance.sh

# Full database backup
pg_dump -h localhost -U trading_user trading_bot_prod | gzip > /backup/weekly_$(date +%Y%m%d).sql.gz

# Update system packages
sudo apt update && sudo apt upgrade -y

# Restart services for memory cleanup
docker-compose -f docker-compose.prod.yml restart trading-bot

# Check SSL certificate expiry
echo | openssl s_client -servername your-domain.com -connect your-domain.com:443 2>/dev/null | openssl x509 -noout -dates

# Generate weekly performance report
curl -X POST http://localhost:8080/api/v1/reports/weekly
```

#### **Monthly Tasks**
```bash
#!/bin/bash
# monthly_maintenance.sh

# Security updates
sudo apt update && sudo apt list --upgradable
sudo unattended-upgrade

# Docker image updates
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedSince}}"
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# Clean up old logs and backups
find /opt/trading-bot/logs -name "*.log" -mtime +90 -delete
find /backup -name "*.sql.gz" -mtime +90 -delete

# Performance analysis
docker stats --no-stream
```

### **2. Update Procedures**

#### **Application Updates**
```bash
# 1. Create backup
./backup.sh

# 2. Pull latest code
git pull origin main

# 3. Build new image
docker build -t bybit-trading-bot:latest .

# 4. Run tests
docker run --rm bybit-trading-bot:latest pytest

# 5. Deploy
./deploy.sh

# 6. Verify deployment
./health_check.sh
```

#### **Security Updates**
```bash
# System security updates
sudo apt update
sudo apt list --upgradable | grep -i security
sudo apt upgrade

# Docker security updates
docker pull postgres:15
docker pull redis:7-alpine
docker pull nginx:alpine

# SSL certificate renewal
sudo certbot renew --dry-run
sudo certbot renew
```

---

## 🚨 Disaster Recovery

### **1. Backup Strategy**

#### **Data Backup**
```bash
# Database backup
pg_dump -h localhost -U trading_user trading_bot_prod > backup_$(date +%Y%m%d_%H%M%S).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/ .env.prod

# Application data backup
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/ logs/ reports/

# Upload to cloud storage
aws s3 sync /backup s3://trading-bot-backups/$(date +%Y%m%d)/
```

#### **Automated Backup Script**
```bash
#!/bin/bash
# backup_automation.sh

BACKUP_DIR="/backup/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Database backup
docker-compose exec postgres pg_dump -U trading_user trading_bot_prod | gzip > $BACKUP_DIR/database.sql.gz

# Application backup
tar -czf $BACKUP_DIR/application.tar.gz -C /opt/trading-bot .

# Upload to cloud
case "$CLOUD_PROVIDER" in
  "aws")
    aws s3 sync $BACKUP_DIR s3://trading-bot-backups/$(date +%Y%m%d)/
    ;;
  "gcp")
    gsutil -m rsync -r $BACKUP_DIR gs://trading-bot-backups/$(date +%Y%m%d)/
    ;;
  "azure")
    az storage blob upload-batch -d backups -s $BACKUP_DIR
    ;;
esac

# Clean up local backups older than 7 days
find /backup -type d -mtime +7 -exec rm -rf {} +
```

### **2. Recovery Procedures**

#### **Complete System Recovery**
```bash
#!/bin/bash
# disaster_recovery.sh

echo "🚨 Starting disaster recovery procedure..."

# 1. Provision new server
echo "📡 Provisioning new server..."
# (Cloud provider specific commands)

# 2. Install dependencies
echo "📦 Installing dependencies..."
sudo apt update && sudo apt install -y docker.io docker-compose postgresql-client

# 3. Restore from backup
echo "💽 Restoring from backup..."
LATEST_BACKUP=$(aws s3 ls s3://trading-bot-backups/ | sort | tail -n 1 | awk '{print $2}')
aws s3 sync s3://trading-bot-backups/$LATEST_BACKUP /tmp/restore/

# 4. Restore database
echo "🗄️ Restoring database..."
gunzip -c /tmp/restore/database.sql.gz | psql -h localhost -U trading_user trading_bot_prod

# 5. Restore application
echo "🚀 Restoring application..."
tar -xzf /tmp/restore/application.tar.gz -C /opt/trading-bot/

# 6. Start services
echo "▶️ Starting services..."
cd /opt/trading-bot
docker-compose -f docker-compose.prod.yml up -d

# 7. Verify recovery
echo "✅ Verifying recovery..."
sleep 60
curl -f http://localhost:8080/health && echo "Recovery successful!" || echo "Recovery failed!"
```

#### **Database Recovery**
```bash
# Point-in-time recovery
pg_restore --clean --no-acl --no-owner -h localhost -U trading_user -d trading_bot_prod backup.dump

# Selective table recovery
pg_restore --clean --no-acl --no-owner -h localhost -U trading_user -d trading_bot_prod -t trades backup.dump
```

### **3. High Availability Setup**

#### **Load Balancer Configuration**
```nginx
# /etc/nginx/nginx.conf
upstream trading_bot_backend {
    server 10.0.1.10:8080 weight=3 max_fails=2 fail_timeout=30s;
    server 10.0.1.11:8080 weight=3 max_fails=2 fail_timeout=30s;
    server 10.0.1.12:8080 weight=1 max_fails=2 fail_timeout=30s backup;
}

server {
    listen 80;
    location / {
        proxy_pass http://trading_bot_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 3s;
        proxy_send_timeout 3s;
        proxy_read_timeout 3s;
    }
}
```

#### **Database Clustering**
```yaml
# PostgreSQL cluster with streaming replication
version: '3.8'
services:
  postgres-primary:
    image: postgres:15
    environment:
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: repl_password
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
      - ./postgresql-primary.conf:/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"

  postgres-replica:
    image: postgres:15
    environment:
      PGUSER: postgres
      POSTGRES_MASTER_SERVICE: postgres-primary
    volumes:
      - postgres_replica_data:/var/lib/postgresql/data
    depends_on:
      - postgres-primary
```

---

## 📋 Final Deployment Checklist

### **Pre-Deployment**
- [ ] All tests passing in staging environment
- [ ] Performance benchmarks meet requirements
- [ ] Security audit completed
- [ ] Backup and recovery procedures tested
- [ ] Monitoring and alerting configured
- [ ] SSL certificates installed and configured
- [ ] DNS records configured
- [ ] Load testing completed
- [ ] Documentation updated

### **Deployment Day**
- [ ] Schedule maintenance window
- [ ] Notify stakeholders
- [ ] Create full system backup
- [ ] Deploy to production
- [ ] Run post-deployment tests
- [ ] Monitor system for 2-4 hours
- [ ] Verify all services are healthy
- [ ] Update status page
- [ ] Send deployment completion notification

### **Post-Deployment**
- [ ] Monitor system performance for 24-48 hours
- [ ] Review logs for any errors or warnings
- [ ] Verify all integrations are working
- [ ] Update monitoring dashboards
- [ ] Schedule first maintenance window
- [ ] Document any issues and resolutions
- [ ] Update runbooks with new procedures

---

## 📞 Support and Emergency Contacts

### **Emergency Response Team**
- **Primary Engineer**: [Your Contact Info]
- **DevOps Engineer**: [Contact Info]
- **Database Administrator**: [Contact Info]
- **Security Officer**: [Contact Info]

### **Service Providers**
- **Cloud Provider Support**: [Support Contact]
- **Domain/SSL Provider**: [Support Contact]
- **Exchange Support**: [Bybit Support]
- **Monitoring Service**: [Support Contact]

### **Escalation Procedures**
1. **Level 1**: Application warnings/minor issues
2. **Level 2**: Service degradation, partial outage
3. **Level 3**: Complete outage, security breach, data loss

---

**This completes the comprehensive production deployment guide. Follow each section carefully and adapt configurations to your specific environment and requirements.**

*Last Updated: September 2025*
*Version: 1.0.0*