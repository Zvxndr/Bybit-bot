# 🔑 DigitalOcean Environment Variables - Setup Guide

## ⚠️ **YES - You Need to Update Environment Variables!**

Your DigitalOcean deployment needs several environment variables to work properly. Here's what you need to configure:

## 🚀 **How to Update Environment Variables on DigitalOcean:**

1. **Go to your DigitalOcean App Platform Dashboard**
2. **Click on your `bybit-trading-bot-production` app**
3. **Go to Settings → Environment Variables**
4. **Add/Update the following variables:**

---

## 🔑 **CRITICAL - Bybit API Keys** (Required for trading)

```bash
# 🟢 TESTNET (Recommended for initial deployment)
BYBIT_TESTNET_API_KEY=your_testnet_api_key_here
BYBIT_TESTNET_API_SECRET=your_testnet_secret_here
BYBIT_TESTNET=true

# 🔴 MAINNET (Only after testing - USE WITH CAUTION)
# BYBIT_API_KEY=your_live_api_key_here  
# BYBIT_API_SECRET=your_live_secret_here
```

**⚠️ Start with TESTNET first!** Only use mainnet after confirming everything works.

---

## 🛡️ **SECURITY** (Required)

```bash
# Generate a random 32-character string for security
SECRET_KEY=your_random_32_character_string_here

# Example: AbC123XyZ789!@#MnO456PqR890$%^
```

---

## ⚙️ **TRADING ENVIRONMENT** (Required)

```bash
TRADING_ENVIRONMENT=production
NODE_ENV=production
DEBUG=false
PYTHONPATH=/app
PYTHONUNBUFFERED=1
```

---

## 💰 **RISK MANAGEMENT** (Recommended - keeps you safe!)

```bash
MAX_DAILY_RISK=0.02          # 2% max daily loss
MAX_POSITION_SIZE=0.01       # 1% max per position  
EMERGENCY_STOP_LOSS=0.05     # 5% emergency stop
ENABLE_STOP_LOSS=true
DEFAULT_STOP_LOSS=0.02       # 2% stop loss
DEFAULT_TAKE_PROFIT=0.04     # 4% take profit
PAPER_TRADING_BALANCE=10000  # $10k virtual balance
```

---

## 📊 **DATABASE & LOGGING** (Optional - defaults work)

```bash
DATABASE_URL=sqlite:///data/trading_bot.db
LOG_LEVEL=INFO
```

---

## 📧 **EMAIL NOTIFICATIONS** (Optional)

```bash
# If you want email alerts (optional)
EMAIL_ENABLED=false
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_TO=your_alerts@gmail.com
```

---

## 🎯 **Quick Setup - Minimum Required:**

If you just want to get it running quickly, add these **essential variables**:

```bash
# 1. TESTNET API KEYS (get from Bybit testnet)
BYBIT_TESTNET_API_KEY=your_testnet_key
BYBIT_TESTNET_API_SECRET=your_testnet_secret
BYBIT_TESTNET=true

# 2. SECURITY KEY (generate random string)
SECRET_KEY=AbC123XyZ789MnO456PqR890UvW234

# 3. ENVIRONMENT 
TRADING_ENVIRONMENT=production
NODE_ENV=production
```

---

## 📝 **Step-by-Step Setup:**

### **1. Get Bybit Testnet API Keys:**
- Go to [testnet.bybit.com](https://testnet.bybit.com)
- Login/Register → API Management → Create API Key
- Copy the API Key and Secret

### **2. Generate Secret Key:**
- Use an online generator or run: `openssl rand -hex 16`
- Should be 32 characters long

### **3. Add to DigitalOcean:**
- DigitalOcean Dashboard → Your App → Settings → Environment Variables
- Click "Edit" and add each variable
- Click "Save" - **this will trigger a new deployment**

### **4. Monitor Deployment:**
- Go to "Activity" tab to watch the build
- Check "Runtime Logs" for any errors
- Your new ML dashboard should be live!

---

## ✅ **After Adding Variables:**

1. **DigitalOcean will automatically redeploy** your app
2. **Monitor the build logs** for any issues  
3. **Access your dashboard** at your DigitalOcean app URL
4. **Login with admin/password** to see the ML strategies!

## 🚨 **Important Notes:**

- **Start with TESTNET** - it's free and safe for testing
- **Keep API keys secret** - never share or commit to Git
- **Monitor your first deployment** - check logs for any errors
- **Test with small amounts** when you move to mainnet

Your ML trading system will work with just the minimum required variables, but the risk management settings help keep your funds safe! 💪