# 🎯 **QUICK REFERENCE - CORRECT WORKFLOW**

## ⚡ **DEVELOPMENT COMMANDS**

### **Local Development:**
```bash
# Run the main application (THE ONLY WAY)
cd src
python main.py

# Access everything on port 8080:
# Frontend: http://localhost:8080/
# APIs: http://localhost:8080/api/status
# Health: http://localhost:8080/health
```

### **Production Deployment:**
```bash
# Push to GitHub (DigitalOcean deploys automatically)
git add .
git commit -m "Production deployment"  
git push origin main
```

---

## 🚫 **NEVER DO THESE**

```bash
# ❌ NEVER create separate servers:
python backend_server.py
python simple_backend.py 
python -m http.server 3000

# ❌ NEVER change Dockerfile entry:
CMD ["python", "backend_server.py"]

# ❌ NEVER use multiple ports:
Frontend: 3000, Backend: 8000
```

---

## ✅ **ALWAYS DO THESE**

```bash
# ✅ ALWAYS use main.py:
python src/main.py

# ✅ ALWAYS use port 8080:
Everything on localhost:8080

# ✅ ALWAYS fix imports in place:
Edit src/main.py SharedState class
```

---

## 🔧 **IF SOMETHING BREAKS**

### **Import Error:**
1. Find missing method in src/main.py error
2. Add method to SharedState fallback class  
3. DO NOT create new server files

### **Frontend Connection Error:**
1. Check frontend uses same origin URLs
2. Verify src/main.py is serving frontend files
3. DO NOT split frontend/backend

### **Deployment Error:**
1. Check Dockerfile points to src/main.py
2. Verify port 8080 configuration
3. DO NOT change entry point

---

## 📋 **SYSTEM CHECKLIST**

Before any changes, verify:
- [ ] Dockerfile: `CMD ["python", "src/main.py"]`
- [ ] No separate server files exist
- [ ] Frontend uses same origin API calls
- [ ] Everything runs on port 8080
- [ ] TradingBotApplication is entry point

---

**🎯 Remember: Your system is ALREADY correct for DigitalOcean!**