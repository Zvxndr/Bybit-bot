# DigitalOcean Debug Deployment Guide
## Automatic Debug Script Execution on DigitalOcean

### 🌊 **DigitalOcean Compatibility - YES! ✅**

This debug setup is **fully compatible** with DigitalOcean droplets and will work seamlessly.

### 🚀 **DigitalOcean Deployment Steps:**

#### **1. SSH into your DigitalOcean Droplet:**
```bash
ssh root@your-droplet-ip
```

#### **2. Navigate to your project directory:**
```bash
cd /path/to/your/Bybit-bot-fresh
```

#### **3. Start in Debug Mode (3 options):**

**Option A: Use the debug startup script (Recommended)**
```bash
./start_debug.sh
```

**Option B: Set environment variables**
```bash
export DEBUG_MODE=true
export ENVIRONMENT=debug
python3 src/main.py --debug
```

**Option C: The .debug file is already present**
```bash
python3 src/main.py
```

### 🔧 **DigitalOcean Specific Features:**

1. **Linux Compatible**: Uses `python3` and bash scripts
2. **Background Execution**: Can run with `nohup` for persistent operation
3. **Log File Generation**: All debug output saved to log files
4. **Port 8080**: Dashboard accessible at `http://your-droplet-ip:8080`

### 📊 **DigitalOcean Debug Execution Flow:**

```
🌊 DigitalOcean Droplet Starts
    ↓
🔧 Debug Mode Detected (.debug file exists)
    ↓
🚀 Application Initializes
    ↓
🧪 test_button_functions.py runs automatically
    ↓  
🔥 debug_data_wipe.py runs automatically
    ↓
📊 All results logged to files
    ↓
🌐 Web interface available on port 8080
    ↓
📋 Complete logs ready for review
```

### 🌐 **DigitalOcean Access Points:**

Once running, you can access:
- **Dashboard**: `http://your-droplet-ip:8080`
- **API Health**: `http://your-droplet-ip:8080/health`
- **Button Tests**: Automatically executed and logged
- **Logs**: Check the generated log files

### 💡 **DigitalOcean Best Practices:**

#### **Run with nohup for persistent operation:**
```bash
nohup ./start_debug.sh > debug_deployment.log 2>&1 &
```

#### **Check real-time logs:**
```bash
tail -f debug_deployment.log
```

#### **Check application logs:**
```bash
tail -f logs/open_alpha_*.log
```

### 🔍 **DigitalOcean Firewall Setup:**

Make sure port 8080 is open in your DigitalOcean firewall:

```bash
# If using ufw
sudo ufw allow 8080

# Or in DigitalOcean control panel
# Add inbound rule: Port 8080, Source: All IPv4
```

### 📋 **Expected Log Output on DigitalOcean:**

```
🌊 Starting on DigitalOcean...
🔧 DEBUG MODE ACTIVATED - Test scripts will run automatically
🚀 Starting Open Alpha Application
🔧 Debug mode detected - running automated debug scripts
🧪 Running button function tests...
✅ Button function tests completed successfully
🔥 Running data wipe debug tests...
✅ Data wipe debug tests completed successfully
🌐 Frontend & API server starting on port 8080
📱 Dashboard: http://localhost:8080
💚 Health: http://localhost:8080/health
📊 API: http://localhost:8080/api/status
✅ Application ready!
```

### 🎯 **DigitalOcean Deployment Verification:**

After deployment, verify everything works:

1. **SSH into droplet** ✅
2. **Run debug startup script** ✅
3. **Check logs show test results** ✅
4. **Access web interface** ✅
5. **Test buttons manually** ✅
6. **Share logs for analysis** ✅

### 💰 **Cost Effective:**

- Uses minimal CPU for test scripts
- Runs once during startup
- Background execution available
- Automatic shutdown after testing

### 🔒 **Security Notes for DigitalOcean:**

- Debug mode only runs test scripts
- No real trading in debug mode
- Safe to run on public droplet
- Firewall rules recommended

---

**Ready for DigitalOcean deployment!** 🌊

Just run `./start_debug.sh` on your droplet and the debug scripts will execute automatically during startup.