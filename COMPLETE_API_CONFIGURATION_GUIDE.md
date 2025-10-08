# 🔐 COMPLETE API KEYS & SECRETS CONFIGURATION
## All Environment Variables for Your Trading Bot

This is your **complete configuration file** that goes on your DigitalOcean droplet at `/opt/trading/.env.production`

---

## 📍 **WHERE TO PUT THIS**

**On your DigitalOcean droplet** (via Console or SSH):
```bash
# Navigate to your trading bot directory
cd /opt/trading

# Create/edit the production environment file
nano .env.production

# Copy ALL the configuration below into this file
```

---

## 🔑 **COMPLETE ENVIRONMENT CONFIGURATION**

Copy and paste this entire configuration, then fill in your actual values:

```env
# =============================================================================
# 🇦🇺 AUSTRALIAN TAX COMPLIANT BYBIT TRADING BOT - PRODUCTION CONFIGURATION
# =============================================================================

# 🔴 LIVE TRADING API KEYS (HANDLE WITH EXTREME CARE)
# Get these from: https://www.bybit.com/app/user/api-management
BYBIT_API_KEY="YOUR_LIVE_BYBIT_API_KEY_HERE"
BYBIT_API_SECRET="YOUR_LIVE_BYBIT_API_SECRET_HERE"
BYBIT_TESTNET=false

# 🟡 PAPER TRADING API KEYS (for testing before going live)
# Get these from: https://testnet.bybit.com/app/user/api-management
BYBIT_TESTNET_API_KEY="YOUR_PAPER_TRADING_API_KEY_HERE"
BYBIT_TESTNET_API_SECRET="YOUR_PAPER_TRADING_API_SECRET_HERE"

# 🌐 TRADING ENVIRONMENT CONTROL
TRADING_MODE="paper"                    # Change to "live" when ready for real trading
TESTNET_MODE=true                       # Set to false for live trading

# 🇦🇺 AUSTRALIAN TAX COMPLIANCE SETTINGS
TIMEZONE="Australia/Sydney"
TAX_COMPLIANCE_MODE="production"
ATO_REPORTING_ENABLED=true
FINANCIAL_YEAR="2025-26"
TAX_JURISDICTION="AU"
CGT_METHOD="FIFO"
RECORD_RETENTION_YEARS=7

# 📧 EMAIL REPORTING & ALERTS CONFIGURATION
# Option 1: Gmail SMTP (Free, easy setup)
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT=587
SMTP_USERNAME="your_gmail@gmail.com"
SMTP_PASSWORD="your_gmail_app_specific_password"
FROM_EMAIL="your_gmail@gmail.com"
ALERT_EMAIL="your_alerts@gmail.com"

# Option 2: SendGrid API (More reliable for production)
SENDGRID_API_KEY=""                     # Get free from sendgrid.com
SENDGRID_FROM_EMAIL="reports@yourdomain.com"
SENDGRID_TEMPLATE_ID=""

# Option 3: Outlook/Hotmail SMTP
# SMTP_SERVER="smtp-mail.outlook.com"
# SMTP_PORT=587
# SMTP_USERNAME="your_outlook@hotmail.com"
# SMTP_PASSWORD="your_outlook_password"

# 🚨 EMERGENCY & ALERT CONTACTS
EMERGENCY_EMAIL="your_phone_sms@carrier.com"    # For SMS alerts via email
TRUSTEE_EMAILS="trusted_person1@email.com,trusted_person2@email.com"
BENEFICIARY_EMAILS="family@email.com"

# 🔐 DASHBOARD AUTHENTICATION (CRITICAL FOR SECURITY)
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_USERNAME="your_secure_username"
DASHBOARD_PASSWORD="YourVerySecurePassword123!@#$"
DASHBOARD_REALM="Australian Tax Compliant Trading Bot"

# 🛡️ SECURITY CONFIGURATION
SECRET_KEY="GENERATE_THIS_WITH_OPENSSL_RAND_BASE64_32"
API_RATE_LIMIT="100/hour"
MAX_LOGIN_ATTEMPTS=3
SESSION_TIMEOUT_MINUTES=30
CSRF_PROTECTION=true

# 📊 DATABASE & STORAGE
DATABASE_PATH="/opt/trading/data/trading_bot_prod.db"
DATABASE_BACKUP_ENABLED=true
DATABASE_BACKUP_INTERVAL_HOURS=6
LOG_LEVEL="INFO"
LOG_RETENTION_DAYS=2555                 # 7 years for Australian compliance

# 🚨 RISK MANAGEMENT & SAFETY LIMITS (START CONSERVATIVE!)
# These are your safety nets - start small!
MAX_DAILY_LOSS_AUD=100.00              # Conservative daily loss limit
MAX_POSITION_SIZE_PERCENT=0.5          # 0.5% maximum position size
MAX_POSITIONS_CONCURRENT=3             # Maximum open positions
EMERGENCY_STOP_THRESHOLD=2.0           # 2% daily loss triggers emergency stop
RISK_MANAGEMENT_MODE="ultra_conservative"

# 💰 PORTFOLIO & POSITION SIZING
MIN_POSITION_SIZE_USD=10.00            # Minimum trade size
POSITION_SIZE_METHOD="fixed_percentage" # or "fixed_amount", "volatility_based"
BASE_CURRENCY="AUD"                    # Australian Dollar base
SLIPPAGE_TOLERANCE_PERCENT=0.1         # 0.1% slippage tolerance

# 📈 TRADING STRATEGY SETTINGS
STRATEGY_GRADUATION_ENABLED=true       # Allow strategies to graduate from paper to live
GRADUATION_MIN_TRADES=100              # Minimum trades before graduation
GRADUATION_MIN_PROFIT_PERCENT=5.0     # Minimum profit % for graduation
GRADUATION_MAX_DRAWDOWN_PERCENT=3.0   # Maximum drawdown for graduation

# 🔍 MONITORING & HEALTH CHECKS
ENABLE_REAL_TIME_MONITORING=true
HEALTH_CHECK_INTERVAL=60               # Check system health every 60 seconds
INFRASTRUCTURE_ALERTS=true             # Enable infrastructure monitoring
API_HEALTH_CHECK_INTERVAL=300          # Check Bybit API every 5 minutes

# 📊 REPORTING & ANALYTICS
DAILY_REPORT_ENABLED=true
DAILY_REPORT_TIME="17:00"             # 5 PM Sydney time
WEEKLY_REPORT_ENABLED=true
MONTHLY_TAX_REPORT_ENABLED=true        # For Australian tax compliance
PERFORMANCE_ANALYTICS_ENABLED=true

# 🌐 NETWORK & API SETTINGS
REQUEST_TIMEOUT_SECONDS=30
MAX_RETRIES=3
RETRY_DELAY_SECONDS=1
CONNECTION_POOL_SIZE=10
BYBIT_API_RATE_LIMIT_PER_SECOND=5

# 🔄 BACKUP & RECOVERY
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"            # Daily at 2 AM
BACKUP_RETENTION_DAYS=90
BACKUP_ENCRYPTION=true

# 🚀 PERFORMANCE OPTIMIZATION
ENABLE_CACHING=true
CACHE_TTL_SECONDS=300
ENABLE_COMPRESSION=true
THREAD_POOL_SIZE=4

# 🐛 DEBUGGING & DEVELOPMENT (Set to false for production)
DEBUG_MODE=false
VERBOSE_LOGGING=false
ENABLE_PROFILING=false
MOCK_DATA_MODE=false

# 🏃‍♂️ APPLICATION SETTINGS
APP_NAME="Australian Tax Compliant Bybit Trading Bot"
APP_VERSION="1.0.0"
ENVIRONMENT="production"
WORKER_PROCESSES=1
MAX_WORKERS=4

# 🌍 LOCALIZATION
LANGUAGE="en_AU"                       # Australian English
CURRENCY_DISPLAY_FORMAT="AUD"
DATE_FORMAT="%d/%m/%Y"                 # Australian date format
TIME_FORMAT="%H:%M:%S"

# 🔔 NOTIFICATION PREFERENCES
ENABLE_EMAIL_NOTIFICATIONS=true
ENABLE_SMS_NOTIFICATIONS=false        # Set to true if you configure SMS
NOTIFICATION_FREQUENCY="immediate"     # or "daily", "weekly"
ALERT_ON_PROFIT=true
ALERT_ON_LOSS=true
ALERT_ON_EMERGENCY_STOP=true
ALERT_ON_SYSTEM_ERROR=true

# 🎯 ADDITIONAL INTEGRATIONS (Optional)
DISCORD_WEBHOOK_URL=""                 # For Discord notifications
SLACK_WEBHOOK_URL=""                   # For Slack notifications
TELEGRAM_BOT_TOKEN=""                  # For Telegram alerts
TELEGRAM_CHAT_ID=""

# 🔐 ENCRYPTION & HASHING
ENCRYPTION_KEY=""                      # Generate with: openssl rand -base64 32
HASH_ALGORITHM="SHA-256"
PASSWORD_SALT=""                       # Generate with: openssl rand -base64 16
```

---

## 📝 **HOW TO FILL IN YOUR VALUES**

### **1. 🔴 Your Paper Trading Keys (You Have These)**
```env
BYBIT_TESTNET_API_KEY="your_actual_paper_trading_key"
BYBIT_TESTNET_API_SECRET="your_actual_paper_trading_secret"
```

### **2. 📧 Email Setup (Choose One Method)**

**Method A: Gmail (Easiest)**
1. Go to Google Account settings
2. Enable 2-factor authentication
3. Generate an "App Specific Password"
4. Use that password (not your regular Gmail password)

```env
SMTP_USERNAME="youremail@gmail.com"
SMTP_PASSWORD="your_16_character_app_password"
ALERT_EMAIL="youremail@gmail.com"
```

**Method B: SendGrid (More Professional)**
1. Sign up at sendgrid.com (free tier: 100 emails/day)
2. Create API key
3. Verify your sender email

```env
SENDGRID_API_KEY="SG.your_sendgrid_api_key"
SENDGRID_FROM_EMAIL="reports@yourdomain.com"
```

### **3. 🔐 Generate Security Keys**
```bash
# On your droplet, generate secure keys:
openssl rand -base64 32    # For SECRET_KEY
openssl rand -base64 32    # For ENCRYPTION_KEY  
openssl rand -base64 16    # For PASSWORD_SALT
```

### **4. 🛡️ Dashboard Authentication**
```env
DASHBOARD_USERNAME="your_chosen_username"
DASHBOARD_PASSWORD="YourVerySecurePassword123!@#$"
```

---

## 🚨 **CRITICAL SECURITY STEPS**

### **1. Secure File Permissions**
```bash
# After creating the file, secure it:
chmod 600 /opt/trading/.env.production
chown tradingbot:tradingbot /opt/trading/.env.production

# Verify only you can read it:
ls -la /opt/trading/.env.production
# Should show: -rw------- 1 tradingbot tradingbot
```

### **2. Test Configuration**
```bash
# Test that environment loads correctly:
cd /opt/trading
source venv/bin/activate
python -c "
import os
from dotenv import load_dotenv
load_dotenv('.env.production')
print('✅ BYBIT_TESTNET_API_KEY:', 'SET' if os.getenv('BYBIT_TESTNET_API_KEY') else 'NOT SET')
print('✅ SMTP_USERNAME:', 'SET' if os.getenv('SMTP_USERNAME') else 'NOT SET')
print('✅ DASHBOARD_AUTH_ENABLED:', os.getenv('DASHBOARD_AUTH_ENABLED'))
"
```

---

## 🎯 **STARTUP SEQUENCE**

### **1. Paper Trading First (Safe)**
```env
TRADING_MODE="paper"
TESTNET_MODE=true
BYBIT_TESTNET=true
```

### **2. Live Trading (When Ready)**
```env
TRADING_MODE="live" 
TESTNET_MODE=false
BYBIT_TESTNET=false
# Make sure your BYBIT_API_KEY and BYBIT_API_SECRET are set for live trading
```

---

## 📧 **EMAIL SETUP FOR TAX REPORTS**

Your bot will automatically send:
- **Daily trading summaries**
- **Weekly performance reports**  
- **Monthly Australian tax compliance reports**
- **Emergency alerts**
- **System health notifications**

**All emails include:**
✅ ATO-ready transaction logs
✅ FIFO cost basis calculations  
✅ CGT event summaries
✅ 7-year compliant record keeping

---

**🎉 This is your complete configuration! Save it as `/opt/trading/.env.production` on your DigitalOcean droplet and you'll have everything configured for secure, compliant trading! 🇦🇺**