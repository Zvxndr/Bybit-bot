# 🚀 Simple DigitalOcean Deployment Guide
## Get Your Trading Bot Live in 15 Minutes

**This guide is for beginners who want to deploy their trading bot quickly and safely.**

---

## 🎯 What You'll Need (5 minutes to gather)

1. **DigitalOcean Account** - [Sign up here](https://digitalocean.com) (free account)
2. **Bybit Account** - [Sign up here](https://bybit.com) (for trading)
3. **15 minutes of your time**
4. **$12/month** for server costs (you can cancel anytime)

---

## 📱 Step 1: Get Your Bybit API Keys (5 minutes)

**🔑 These keys let your bot trade for you automatically.**

### For Live Trading (Real Money):
1. Go to [bybit.com](https://bybit.com) and login
2. Click your profile → **Account & Security** → **API Management**
3. Click **Create New Key**
4. Settings to use:
   - **Key Name**: `Trading Bot Live`
   - **Permissions**: ✅ Spot Trading, ✅ Derivatives Trading, ✅ Account Info
   - **IP Restriction**: Leave blank for now (we'll add this later)
   - **Withdrawal**: ❌ **NEVER enable this** (for safety)
5. **Copy both the API Key and Secret** - save them somewhere safe!

### For Testing (Play Money - Optional):
1. Go to [testnet.bybit.com](https://testnet.bybit.com) 
2. Login with same account
3. Repeat steps above but name it `Trading Bot Test`
4. This gives you fake money to test with

**⚠️ Important: Keep these keys secret! Never share them with anyone.**

---

## 🌊 Step 2: Create Your DigitalOcean Server (3 minutes)

**🖥️ This is where your bot will live and trade 24/7.**

### Create Droplet:
1. Login to [DigitalOcean](https://digitalocean.com)
2. Click **Create** → **Droplets**
3. Choose these settings:
   - **OS**: Ubuntu 22.04 LTS
   - **Plan**: Basic $12/month (2GB RAM, 1 CPU)
   - **Datacenter**: Pick closest to you
   - **Authentication**: Choose Password (easier for beginners)
   - **Hostname**: `trading-bot`
4. Click **Create Droplet**
5. **Wait 2 minutes** for it to start up
6. **Copy the IP address** (you'll need this)

---

## 🔒 Step 3: Connect and Set Up Security (5 minutes)

**🛡️ This makes your server secure before we put your trading bot on it.**

### Connect to Your Server:
```bash
# On Windows: Use PowerShell or download PuTTY
# On Mac/Linux: Use Terminal

# Connect (replace with YOUR server IP)
ssh root@your_server_ip

# When prompted, enter the password from DigitalOcean email
```

### Run One-Time Security Setup:
```bash
# Download and run our security script
curl -o setup_security.sh https://raw.githubusercontent.com/Zvxndr/Bybit-bot-fresh/main/setup_security.sh

# Make it runnable and execute
chmod +x setup_security.sh
./setup_security.sh
```

**✅ This automatically sets up:**
- Firewall protection
- Intrusion detection  
- Web server
- Python environment
- Everything you need!

---

## 🚀 Step 4: Deploy Your Trading Bot (2 minutes)

### Download Your Bot Code:
```bash
# Go to the app directory
cd /app

# Download your trading bot
git clone https://github.com/Zvxndr/Bybit-bot-fresh.git .

# Install Python packages
source venv/bin/activate
pip install -r requirements.txt
```

### Set Up Your API Keys:
```bash
# Create your environment file
nano .env
```

**Copy this template and replace with YOUR values:**
```bash
# Your Bybit Live Trading Keys (REQUIRED)
BYBIT_API_KEY=paste_your_live_api_key_here
BYBIT_API_SECRET=paste_your_live_api_secret_here
BYBIT_TESTNET=false

# Your Bybit Test Keys (OPTIONAL)
BYBIT_TESTNET_API_KEY=paste_your_testnet_key_here
BYBIT_TESTNET_API_SECRET=paste_your_testnet_secret_here

# Basic Settings
NODE_ENV=production
TRADING_ENVIRONMENT=production
DEBUG=false
SECRET_KEY=make_up_a_long_random_password_here_32_chars_minimum

# Safety Settings (Recommended)
MAX_DAILY_RISK=0.02
MAX_POSITION_SIZE=0.01
EMERGENCY_STOP_LOSS=0.05
ENABLE_STOP_LOSS=true
DEFAULT_STOP_LOSS=0.02
DEFAULT_TAKE_PROFIT=0.04

# Storage
DATABASE_URL=sqlite:///data/trading_bot.db
LOG_LEVEL=INFO
```

**To save in nano:** Press `Ctrl+X`, then `Y`, then `Enter`

---

## 🎮 Step 5: Start Your Trading Bot (1 minute)

```bash
# Start your trading bot
sudo systemctl start trading-bot

# Check if it's running
sudo systemctl status trading-bot

# If you see "active (running)" in green - SUCCESS! 🎉
```

### View Your Bot Dashboard:
```
Open your web browser and go to:
http://your_server_ip

You should see your trading bot dashboard!
```

---

## 📊 Step 6: Verify Everything Works

### Check Your Bot is Trading:
1. **Web Dashboard**: Visit `http://your_server_ip`
2. **Check Positions**: Should show your Bybit account balance
3. **Check Logs**: `sudo journalctl -u trading-bot -f`
4. **Test with Small Amount**: Make a tiny trade first!

### Monitor Your Bot:
```bash
# View live logs (press Ctrl+C to exit)
sudo journalctl -u trading-bot -f

# Check system status
sudo systemctl status trading-bot nginx

# Restart if needed
sudo systemctl restart trading-bot
```

---

## 🛡️ Security Best Practices

### Secure Your API Keys:
1. **Add IP Restriction**: Go back to Bybit → API Management → Edit your key → Add your server IP
2. **Monitor Regularly**: Check your bot daily for first week
3. **Start Small**: Test with small amounts first
4. **Keep Keys Safe**: Never share your API keys with anyone

### Server Security:
- ✅ Firewall active (automatic)
- ✅ Intrusion detection (automatic)
- ✅ Regular updates: `sudo apt update && sudo apt upgrade`

---

## 🆘 Common Issues & Solutions

### "Permission Denied" when connecting:
```bash
# Make sure you're using the right IP and password
# Check DigitalOcean email for login details
```

### "Trading bot won't start":
```bash
# Check the logs for errors
sudo journalctl -u trading-bot --no-pager

# Usually it's an API key issue - double-check your .env file
nano /app/.env
```

### "Can't see dashboard":
```bash
# Check if nginx is running
sudo systemctl status nginx

# Restart it if needed
sudo systemctl restart nginx
```

### "No trades happening":
```bash
# Check your API keys are correct
# Check you have funds in your Bybit account
# Check the logs: sudo journalctl -u trading-bot -f
```

---

## 📞 Getting Help

### Check These First:
1. **Logs**: `sudo journalctl -u trading-bot -f` (shows what's happening)
2. **Service Status**: `sudo systemctl status trading-bot`
3. **Web Dashboard**: `http://your_server_ip` (should show data)

### Common Log Messages:
- ✅ `"API connection successful"` = Good!
- ✅ `"Portfolio loaded"` = Good!
- ❌ `"API key invalid"` = Check your Bybit API keys
- ❌ `"Connection refused"` = Check internet/firewall

---

## 🎉 Success Checklist

**Your trading bot is working if you can check all these:**

- [ ] ✅ Server created on DigitalOcean
- [ ] ✅ Connected via SSH successfully
- [ ] ✅ Security setup completed (firewall, etc.)
- [ ] ✅ Bot code downloaded to `/app`
- [ ] ✅ API keys added to `.env` file
- [ ] ✅ Trading bot service running (`systemctl status trading-bot`)
- [ ] ✅ Dashboard accessible at `http://your_server_ip`
- [ ] ✅ Shows your real Bybit account balance
- [ ] ✅ No errors in logs (`journalctl -u trading-bot -f`)

---

## 🚀 Next Steps (Optional)

### Add SSL Certificate (Secure HTTPS):
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get free SSL certificate (replace with your domain)
sudo certbot --nginx -d yourdomain.com
```

### Set Up Domain Name:
1. Buy a domain from Namecheap/GoDaddy
2. Point it to your server IP
3. Update nginx config with your domain
4. Get SSL certificate (step above)

### Add Email Notifications (Later):
- We'll add email alerts in a future update
- For now, monitor via dashboard and logs

---

## 💰 Cost Breakdown

**Monthly Costs:**
- DigitalOcean Server: $12/month
- Domain (optional): $10/year
- SSL Certificate: FREE
- **Total: ~$12-13/month**

**💡 Tip**: Your bot could pay for itself with just a few profitable trades!

---

## 🎯 You're Done!

**🎉 Congratulations! Your trading bot is now live and trading automatically!**

### What happens now:
1. **Your bot trades 24/7** using your strategies
2. **Monitor via dashboard** at `http://your_server_ip`
3. **Check logs occasionally** with `sudo journalctl -u trading-bot -f`
4. **Watch your account grow** (hopefully! 📈)

### Remember:
- **Start small** until you're confident
- **Monitor daily** for the first week
- **Keep learning** about trading strategies
- **Have fun** and don't risk money you can't afford to lose!

**🚀 Happy Trading! Your bot is now making money while you sleep! 💰**