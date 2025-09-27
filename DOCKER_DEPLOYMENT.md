# 🐳 Docker Deployment Guide

## Quick Docker Setup (5 minutes)

Your Australian Trust Trading Bot is **100% ready for Docker deployment**! Here's how to get it running:

### Option 1: Docker Compose (Recommended)
```bash
# Clone and enter directory
git clone https://github.com/Zvxndr/Bybit-bot.git
cd Bybit-bot

# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor

# Start the bot
docker-compose up -d
```

### Option 2: Docker Run
```bash
# Build the image
docker build -t bybit-bot .

# Run the container
docker run -d \
  --name bybit-bot \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  bybit-bot
```

## 📋 Environment Variables Required

Add these to your `.env` file:

```env
# Trading Configuration
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
TRADING_ENVIRONMENT=testnet  # or mainnet for live trading

# Security Keys (auto-generated during setup)
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
MFA_ENCRYPTION_KEY=your_mfa_key

# Database
DATABASE_URL=sqlite:///./trading_bot.db

# Optional: Email Notifications
SENDGRID_API_KEY=your_sendgrid_key
FROM_EMAIL=bot@yourdomain.com
TRUSTEE_EMAILS=trustee1@email.com,trustee2@email.com

# Optional: Infrastructure
DIGITALOCEAN_TOKEN=your_do_token
```

## 🔒 Security Features Included

- ✅ **Multi-Factor Authentication (MFA)** - TOTP tokens + backup codes
- ✅ **AES-256 Encryption** - All sensitive data encrypted
- ✅ **JWT Sessions** - Secure authentication
- ✅ **Non-root Container** - Runs as `appuser` for security
- ✅ **Health Checks** - Automatic monitoring

## 🚀 Quick Verification

After starting the container:

```bash
# Check if it's running
docker ps

# View logs
docker-compose logs -f bybit-bot

# Test the health endpoint
curl http://localhost:8080/health
```

Expected response: `{"status": "healthy", "timestamp": "..."}`

## 📱 MFA Setup

1. **Start the container** (it will generate MFA secrets)
2. **Check logs for QR code** or setup instructions:
   ```bash
   docker-compose logs bybit-bot | grep -A 20 "MFA Setup"
   ```
3. **Scan QR code** with Google Authenticator, Authy, or similar
4. **Save backup codes** shown in the logs

## 🎯 Production Deployment

For production use:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  bybit-bot:
    build: .
    ports:
      - "443:8080"  # Use HTTPS
    environment:
      - ENV=production
      - LOG_LEVEL=warning
    volumes:
      - /var/log/bybit-bot:/app/logs
      - /var/data/bybit-bot:/app/data
    restart: always
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
```

## 🛡️ Security Checklist

- [ ] Change default passwords in `.env`
- [ ] Use strong API keys
- [ ] Enable MFA for admin access
- [ ] Set `TRADING_ENVIRONMENT=testnet` initially
- [ ] Review logs for any errors
- [ ] Test with small amounts first

## 📊 Monitoring

Built-in endpoints:
- `http://localhost:8080/health` - Health check
- `http://localhost:8080/metrics` - Trading metrics
- `http://localhost:8080/admin` - Admin dashboard (requires MFA)

## 🆘 Troubleshooting

**Container won't start?**
```bash
docker-compose logs bybit-bot
```

**Permission errors?**
```bash
sudo chown -R 1000:1000 ./logs ./data
```

**Can't connect to Bybit?**
- Check API keys in `.env`
- Verify `TRADING_ENVIRONMENT` setting
- Check firewall/network settings

---

## 🎉 You're Ready!

Your **Australian Trust Trading Bot** is now running in Docker with:
- ✅ Enterprise security (MFA, encryption)
- ✅ Automated trading capabilities
- ✅ Health monitoring
- ✅ Proper logging and data persistence

**Happy Trading!** 📈💰

---
*For detailed deployment options, see `docs/deployment/DEPLOYMENT_GUIDE.md`*