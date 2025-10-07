# Bybit Trading Bot - Production Ready
=============================================

**⚠️ READ DEPLOYMENT_ARCHITECTURE_GUIDE.md BEFORE MAKING CHANGES ⚠️**

**Enterprise-Grade Cryptocurrency Trading Bot with Advanced Risk Management**

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Version](https://img.shields.io/badge/Version-2.0.0-blue)
![License](https://img.shields.io/badge/License-Private-red)

## 🚨 **CRITICAL ARCHITECTURE INFO**

### **⚡ Single Application Deployment**
```yaml
Entry Point: main.py (Production AI Pipeline)
Port: 8000 (DigitalOcean standard)  
Architecture: FastAPI backend with AI pipeline
Deployment: python main.py (DO NOT CHANGE)
```

### **🚫 DO NOT CREATE SEPARATE SERVERS**
- The system uses ONE application (`main.py`)
- FastAPI backend with production AI pipeline
- If imports fail, fix them in `main.py` - don't create new servers

## 🚀 Overview

This is a sophisticated cryptocurrency trading bot designed for **DigitalOcean deployment**, featuring enterprise-grade security and **integrated architecture**.

### ✅ System Status: 100% Complete & Deployment Ready

- **Application**: Single Production AI Pipeline (main.py) 
- **Frontend**: Tabler dashboard (integrated into main app)
- **Database**: SQLite with PostgreSQL upgrade path
- **Security**: Enterprise HSM integration, MFA, advanced key management
- **Deployment**: DigitalOcean App Platform (single container)
- **Monitoring**: Built-in health checks at `/health`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Trading Bot System                      │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Next.js)     │  Backend (FastAPI)   │  Database      │
│  - Dashboard            │  - Trading Engine    │  - SQLite      │
│  - Real-time UI        │  - Risk Management   │  - TimescaleDB │
│  - Configuration       │  - Strategy Engine   │  - Backups     │
├─────────────────────────────────────────────────────────────────┤
│  Security Layer         │  Monitoring          │  Deployment    │
│  - HSM Integration      │  - Health Checks     │  - Docker      │
│  - MFA Authentication  │  - Alerting          │  - Digital Ocean│
│  - Key Management      │  - Performance       │  - Auto-scaling│
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Digital Ocean Deployment (Recommended)

```bash
# Clone repository
git clone https://github.com/Zvxndr/Bybit-bot.git
cd Bybit-bot

# Deploy with Node.js support
chmod +x deploy_digital_ocean_nodejs.sh
./deploy_digital_ocean_nodejs.sh
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python src/database_init.py

# Configure environment
cp config/secrets.yaml.template config/secrets.yaml
# Edit config/secrets.yaml with your API keys

# Run application
python main.py
```

### Option 3: Docker Deployment

```bash
# Build with Node.js support
docker build -f Dockerfile.nodejs -t bybit-bot:latest .

# Run container
docker run -d \
  --name bybit-bot \
  -p 8080:8080 -p 3000:3000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  bybit-bot:latest
```

## 📊 Features

### Core Trading Features
- ✅ **Multi-Exchange Support**: Bybit testnet/mainnet
- ✅ **Strategy Engine**: ML-powered trading strategies
- ✅ **Risk Management**: Advanced position sizing and stop-loss
- ✅ **Portfolio Management**: Real-time portfolio tracking
- ✅ **Tax Compliance**: Australian CGT event tracking

### Advanced Features
- ✅ **Machine Learning**: Strategy discovery and optimization
- ✅ **Real-time Data**: WebSocket market data feeds
- ✅ **Email Notifications**: Trade alerts and system notifications
- ✅ **Health Monitoring**: Comprehensive system health checks
- ✅ **Data Persistence**: Git-safe deployment with data preservation

### Security Features
- ✅ **HSM Integration**: Hardware Security Module support
- ✅ **MFA Authentication**: Multi-factor authentication
- ✅ **Key Management**: Advanced API key encryption
- ✅ **Secure Configuration**: Environment-based secrets management

## 🛠️ Configuration

### Environment Setup

The bot supports multiple environments:

- **Development**: `config/development.yaml`
- **Staging**: `config/staging.yaml`  
- **Production**: `config/config.yaml`

### API Configuration

1. Create `config/secrets.yaml` from template
2. Add your Bybit API credentials:

```yaml
bybit:
  api_key: "your_api_key"
  api_secret: "your_api_secret"
  testnet: true  # Set to false for mainnet

email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  username: "your_email@gmail.com"
  password: "your_app_password"
```

## 📈 Monitoring & Health

### Health Endpoints

- **API Health**: `http://localhost:8080/health`
- **System Status**: `http://localhost:8080/status`
- **Metrics**: `http://localhost:8080/metrics`

### Logging

Logs are available in the `logs/` directory:
- `application.log`: Main application logs
- `trading.log`: Trading-specific logs
- `error.log`: Error and exception logs

## 🔧 Development

### Project Structure

```
├── src/                    # Application source code
│   ├── main.py            # Main application entry point
│   ├── bot/               # Trading bot logic
│   ├── api/               # FastAPI application
│   ├── dashboard/         # Next.js frontend
│   ├── security/          # Security implementations
│   └── services/          # Core services
├── config/                # Configuration files
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── tests/                 # Test suites
└── docker/                # Docker configurations
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests  
python -m pytest tests/integration/

# End-to-end tests
python -m pytest tests/e2e/
```

## 🌐 Deployment

### Digital Ocean

Use the provided deployment scripts:

- **Linux/Mac**: `deploy_digital_ocean_nodejs.sh`
- **Windows**: `deploy_digital_ocean_nodejs.ps1`

### Docker

Multiple Dockerfile options:

- **Standard**: `Dockerfile` (Python only)
- **Full Stack**: `Dockerfile.nodejs` (Python + Node.js)

### Kubernetes

Kubernetes manifests available in `kubernetes/` directory.

## 📚 Documentation

- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Detailed deployment instructions
- **[Architecture](docs/ARCHITECTURE.md)**: System architecture overview
- **[Security Guide](docs/RISK_MANAGEMENT.md)**: Security best practices

## 🔒 Security

This bot implements enterprise-grade security:

- **API Key Encryption**: AES-256 encryption for API keys
- **HSM Integration**: Hardware Security Module support
- **MFA**: Multi-factor authentication for admin access
- **Audit Logging**: Comprehensive security event logging
- **Network Security**: VPN and firewall recommendations

## 📄 License

This project is proprietary software. All rights reserved.

## 🆘 Support

For support and questions:

1. Check the [documentation](docs/)
2. Review [troubleshooting guide](docs/MAINTENANCE_TROUBLESHOOTING_GUIDE.md)
3. Create an issue in the repository

---

**⚠️ Risk Disclaimer**: Cryptocurrency trading involves significant risk. This bot is provided for educational and research purposes. Always test thoroughly in a sandbox environment before live trading.

**🏛️ Compliance**: This bot includes Australian tax compliance features. Consult with financial advisors for regulatory compliance in your jurisdiction.