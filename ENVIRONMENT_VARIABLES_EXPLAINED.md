# 🔐 ENVIRONMENT VARIABLES EXPLAINED
## Complete Guide to All Configuration Settings

**Updated**: October 9, 2025  
**For**: DigitalOcean App Platform Deployment

---

## 📋 **REQUIRED VARIABLES**

### **🏢 Core Application Settings**
```bash
NODE_ENV=production                    # Always set to 'production' for live deployment
TRADING_ENVIRONMENT=production         # Tells the bot it's running in production
PORT=8080                             # Port for DigitalOcean App Platform (don't change)
```

### **🔑 Bybit API Keys (MOST IMPORTANT)**
```bash
# Live Trading Keys (from bybit.com)
BYBIT_API_KEY=your_live_key_here      # Your main Bybit API key
BYBIT_API_SECRET=your_live_secret     # Your main Bybit API secret

# Testnet Keys (from testnet.bybit.com) 
BYBIT_TESTNET_API_KEY=testnet_key     # For testing strategies safely
BYBIT_TESTNET_API_SECRET=testnet_secret # For testing strategies safely
```

**⚠️ CRITICAL**: 
- Create API keys with **trading permissions** but **NO withdrawal permissions**
- Keep your secrets safe - never share them
- Test with testnet first before live trading

---

## 🛡️ **RISK MANAGEMENT (PROTECTS YOUR MONEY)**

### **Daily Risk Limit**
```bash
MAX_DAILY_RISK=0.02                  # Maximum 2% daily loss allowed
```
**What this means**:
- If you have $1,000: Bot stops after $20 daily loss
- If you have $10,000: Bot stops after $200 daily loss
- **Recommendation**: Start with 0.01 (1%) or 0.02 (2%)

### **Position Size Limit**
```bash
MAX_POSITION_SIZE=0.01               # Maximum 1% per trade
```
**What this means**:
- If you have $1,000: Each trade max $10
- If you have $10,000: Each trade max $100
- **Recommendation**: Start with 0.005 (0.5%) or 0.01 (1%)

**🎯 Risk Examples**:
```
Conservative (Recommended for beginners):
MAX_DAILY_RISK=0.01      # 1% daily limit
MAX_POSITION_SIZE=0.005   # 0.5% per trade

Moderate:
MAX_DAILY_RISK=0.02      # 2% daily limit
MAX_POSITION_SIZE=0.01    # 1% per trade

Aggressive (Experienced only):
MAX_DAILY_RISK=0.05      # 5% daily limit
MAX_POSITION_SIZE=0.02    # 2% per trade
```

---

## 🔐 **SECURITY SETTINGS**

### **Secret Key (Encryption)**
```bash
SECRET_KEY=k8m2n9p4q7r1s5t3u6v8w0x2y4z9a1b3c5d7e9f2g4h6j8
```
**What this does**:
- Encrypts your session data
- Secures API communications
- Must be 32+ characters long
- Must be unique for your deployment

**Generate at**: https://www.random.org/strings/ (32 chars, alphanumeric)

### **Dashboard Login**
```bash
DASHBOARD_USERNAME=admin                    # Your username for web login
DASHBOARD_PASSWORD=MySecurePassword123!     # Your password for web login
```
**What this does**:
- When you visit your bot's URL, you'll see a login prompt
- Browser will ask for username/password
- **IMPORTANT**: Change from defaults for security!

**Security Tips**:
- Use a strong password (12+ characters)
- Mix letters, numbers, symbols
- Don't use same password as other accounts

---

## ⚙️ **OPTIONAL FEATURES**

### **External Exchange Data (Advanced Users)**
```bash
ENABLE_BINANCE_DATA=false            # Set to 'true' to enable Binance price data
ENABLE_OKX_DATA=false               # Set to 'true' to enable OKX price data
```
**Benefits when enabled**:
- Compare prices across multiple exchanges
- Better market insight
- More data for analysis

**Why OFF by default**:
- Faster startup (3-5 seconds vs 8-12 seconds)
- Uses less memory and bandwidth
- Simpler for beginners

---

## 🎯 **TYPICAL DEPLOYMENT EXAMPLES**

### **Beginner Setup (Recommended)**
```bash
# Conservative, fast, simple
MAX_DAILY_RISK=0.01
MAX_POSITION_SIZE=0.005
ENABLE_BINANCE_DATA=false
ENABLE_OKX_DATA=false
DASHBOARD_PASSWORD=MyStrongPassword123!
```

### **Intermediate Setup**
```bash
# Balanced risk with price comparison
MAX_DAILY_RISK=0.02
MAX_POSITION_SIZE=0.01
ENABLE_BINANCE_DATA=true
ENABLE_OKX_DATA=false
DASHBOARD_PASSWORD=MyStrongPassword123!
```

### **Advanced Setup**
```bash
# Full features, higher risk tolerance
MAX_DAILY_RISK=0.03
MAX_POSITION_SIZE=0.015
ENABLE_BINANCE_DATA=true
ENABLE_OKX_DATA=true
DASHBOARD_PASSWORD=MyStrongPassword123!
```

---

## 🚨 **IMPORTANT SECURITY NOTES**

### **❌ NEVER DO**:
- Share your API keys with anyone
- Use withdrawal permissions on API keys
- Keep default passwords
- Use same password for multiple services

### **✅ ALWAYS DO**:
- Change default passwords immediately
- Use strong, unique passwords
- Test with testnet first
- Start with conservative risk settings
- Monitor your bot regularly

### **🔒 Where These Are Stored**:
- Environment variables are stored securely in DigitalOcean
- Not visible in your code or logs
- Only you can see/change them through DigitalOcean dashboard

---

## 🎯 **QUICK CHECKLIST**

### **Before Deployment**:
- [ ] Got Bybit API keys (live + testnet)
- [ ] Set conservative risk limits
- [ ] Generated unique SECRET_KEY
- [ ] Changed default dashboard password
- [ ] Decided on external exchange data (recommend OFF initially)

### **After Deployment**:
- [ ] Test login to dashboard works
- [ ] Verify API connection in logs
- [ ] Check risk limits are working
- [ ] Monitor first few trades carefully

---

## 💡 **Pro Tips**

1. **Start Small**: Begin with very low risk limits
2. **Test First**: Use testnet before live trading
3. **Monitor Closely**: Watch your bot's first day carefully
4. **Gradual Increase**: Only increase risk after you're comfortable
5. **Regular Backups**: Note your settings in case you need to redeploy

**Your trading bot configuration is now crystal clear! 🚀**