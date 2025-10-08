# 🚀 **QUICK REFERENCE - AUSTRALIAN TAX COMPLIANT TRADING BOT**

## ⚡ **CORRECT COMMANDS**

### **Local Development & Testing:**
```bash
# Run the unified application (ONLY CORRECT WAY)
python -m src.main

# Access unified dashboard:
# Dashboard: http://localhost:8080/
# API Status: http://localhost:8080/api/system-status  
# Tax Exports: http://localhost:8080/api/tax/logs
# WebSocket: ws://localhost:8080/ws
```

### **Production Deployment:**
```bash
# Commit changes and push to trigger DigitalOcean deployment
git add .
git commit -m "Deploy to production with Australian tax compliance"  
git push origin main
# DigitalOcean automatically runs: python main.py → src/main.py
```

### **🇦🇺 Australian Tax Compliance Testing:**
```bash
# Test tax compliance system locally
python -m src.main
# Navigate to dashboard → Tax Compliance tab
# Download ATO-ready tax logs (CSV/JSON/ATO formats)
```

---

## 🚫 **NEVER DO THESE (WILL BREAK SYSTEM)**

```bash
# ❌ NEVER run separate servers:
python backend_server.py
python simple_dashboard.py
npm start  # No Node.js in this project

# ❌ NEVER change entry point:
CMD ["python", "backend_server.py"]  # Wrong entry point

# ❌ NEVER use wrong ports:
Frontend: 3000, Backend: 8000  # We use single port 8080
```

---

## ✅ **ALWAYS DO THESE**

### **Development:**
```bash
# ✅ ALWAYS use unified entry point:
python -m src.main

# ✅ ALWAYS use single unified system:
PORT=8080  # Everything on one port: FastAPI + Dashboard + WebSocket + API

# ✅ ALWAYS test Australian tax compliance:
# Dashboard → Tax Compliance tab → Download ATO-ready exports
```

### **🇦🇺 Production:**
```bash  
# ✅ ALWAYS use encrypted environment variables on DigitalOcean:
BYBIT_API_KEY=your_live_key      # For live trading
BYBIT_TESTNET_API_KEY=your_test_key  # For paper trading

# ✅ ALWAYS start with paper trading:
# Test strategies in Phase 2 before Phase 3 live trading
```

---

## �️ **TROUBLESHOOTING**

### **🚨 Emergency Stop Not Working:**
```bash
# Test emergency stop procedure:
1. Access dashboard at localhost:8080  
2. Click "Emergency Stop" button
3. Confirm dialog appears
4. Verify all trading halts immediately
```

### **🇦🇺 Tax Logs Not Generating:**
```bash
# Check Australian tax compliance:
1. Verify Australia/Sydney timezone is active
2. Check financial year detection (July 1 - June 30)
3. Test tax export downloads
4. Verify FIFO calculations in CSV exports
```

### **📊 Dashboard Not Loading:**
```bash
# Debug unified dashboard:
1. Check src/main.py is running on port 8080
2. Verify frontend/unified_dashboard.html exists
3. Check WebSocket connection at ws://localhost:8080/ws
4. DO NOT create separate frontend server
```

### **⚠️ API Connection Issues:**
```bash
# Debug 3-phase balance system:
1. Phase 1 (Backtest): Should always work (no API required)
2. Phase 2 (Paper): Requires BYBIT_TESTNET_API_KEY  
3. Phase 3 (Live): Requires BYBIT_API_KEY (use carefully!)
```

---

## 📋 **PRODUCTION READINESS CHECKLIST**

### **🛡️ Security Verification:**
- [ ] ✅ Emergency stop button tested and working
- [ ] ✅ API keys stored as encrypted environment variables
- [ ] ✅ Conservative position sizes configured (1-2%)
- [ ] ✅ Real-time monitoring active and alerting

### **🇦🇺 Australian Tax Compliance:**
- [ ] ✅ Australia/Sydney timezone active  
- [ ] ✅ Financial year 2025-26 detected correctly
- [ ] ✅ ATO-ready tax exports downloading successfully
- [ ] ✅ FIFO cost basis calculations working
- [ ] ✅ 7-year retention system active

### **📈 Trading System:**
- [ ] ✅ 3-phase balance system working (backtest/paper/live)
- [ ] ✅ Strategy graduation pipeline functional
- [ ] ✅ Risk management systems active
- [ ] ✅ WebSocket real-time updates working

---

**🚀 Your Australian tax-compliant trading bot is production ready for DigitalOcean deployment! 🇦🇺**