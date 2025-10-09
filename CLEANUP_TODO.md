# 🧹 WORKSPACE CLEANUP & TODO

## 📋 IMMEDIATE TODO
### 1. Remove Hardcoded Risk Parameters from Config ✅ COMPLETED
- [x] Update `config/config.yaml` - remove all hardcoded risk ratios
- [x] Replace with ML graduation/retirement criteria only
- [x] Ensure ML Risk Manager can operate without constraints
- [ ] Test dynamic risk calculation with live ML engine

### 2. Workspace Cleanup (CRITICAL) ✅ COMPLETED
- [x] Delete bloated documentation files (17+ files removed)
- [x] Remove duplicate Dockerfiles and deployment scripts  
- [x] Clean up unnecessary configuration files (env templates, etc.)
- [x] Remove unused utility scripts and directories
- [x] Delete kubernetes/, monitoring/, docker/ directories
- [x] Remove duplicate requirements files
- **RESULT**: Reduced from 50+ root files to ~20 essential files

### 3. Core Functionality Verification
- [ ] Verify 3-phase pipeline works: Historical → Paper → Live
- [ ] Test ML risk scaling based on balance
- [ ] Validate Australian tax logging
- [ ] Check emergency controls functionality

## 🗑 FILES TO DELETE (Bloat Removal)

### Documentation Bloat
```
API_CONFIGURATION_AUDIT_REPORT.md
AUDIT_COMPLETION_SUMMARY.md
BEGINNER_COMPLETE_WALKTHROUGH.md
BEGINNER_DEPLOYMENT_GUIDE.md
COMPREHENSIVE_ARCHITECTURE_AUDIT.md
COMPREHENSIVE_PRODUCTION_TODO.md
CORRECTED_DIGITALOCEAN_DEPLOYMENT_GUIDE.md
DATABASE_PRIVATE_USE_AUDIT.md
DEPLOYMENT_READY_STATUS.md
DIGITALOCEAN_DEPLOYMENT_WITH_EMAIL.md
DIGITALOCEAN_QUICK_DEPLOY.md
ENVIRONMENT_VARIABLES_EXPLAINED.md
FUTURE_ARBITRAGE_IMPLEMENTATION.md
INTEGRATION_COMPLETE.md
LOCAL_SETUP_GUIDE.md
MULTI_EXCHANGE_DATA_INTEGRATION_COMPLETE.md
MULTI_EXCHANGE_INTEGRATION_AUDIT_COMPLETE.md
OPTIONAL_EXCHANGES_FEATURE.md
PRODUCTION_DEPLOYMENT_GUIDE.md
PRODUCTION_READINESS_AUDIT.md
PRODUCTION_READINESS_AUDIT_COMPLETE.md
QUICK_SETUP_SUMMARY.md
SECURITY_HARDENING_WALKTHROUGH.md
SIMPLE_DEPLOYMENT_GUIDE.md
SIMPLE_DIGITALOCEAN_DEPLOYMENT.md
STRATEGY_PIPELINE_ARCHITECTURE_FIX.md
STRATEGY_PIPELINE_SUCCESS.md
WORKSPACE_CLEANUP_REPORT.md
```

### Deployment Script Bloat
```
configure_auto_setup.sh
security_deploy.sh
setup_security.sh
start-container.sh
start.sh
Dockerfile.minimal
Dockerfile.production
Dockerfile.script
docker-compose.prod.yml
```

### Environment File Bloat
```
.env.digitalocean.template
.env.example
.env.production.simple
.env.production.template
.env.simple
DEPLOYMENT_CONFIG.env
```

### Utility Bloat
```
setup_monitoring.py
test_dual_env.py
nginx-security-config.txt
historical_data_downloader.py (if duplicate exists in src/)
```

## 🎯 CORE FILES TO KEEP

### Essential Configuration
- `config/config.yaml` (cleaned up)
- `config/private_use.yaml`
- `.env` (production secrets)
- `requirements.txt`

### Core Application
- `src/` (entire directory)
- `frontend/unified_dashboard.html`
- `data/` (database and models)

### Deployment (Minimal)
- `Dockerfile` (single, clean version)
- `docker-compose.yml`

### Documentation (Essential Only)
- `ARCHITECTURE.md` (new, clean)
- `README.md` (update to match goals)
- `docs/` (keep essential API docs only)

## 🔧 CONFIG.YAML CLEANUP TASKS

### Remove These Hardcoded Risk Parameters:
```yaml
# DELETE THESE SECTIONS
aggressive_mode:
  max_risk_ratio: 0.02          # ← ML should determine
  min_risk_ratio: 0.005         # ← ML should determine
  portfolio_drawdown_limit: 0.40 # ← ML should determine
  strategy_drawdown_limit: 0.25  # ← ML should determine

conservative_mode:              # ← Delete entire section (not needed)
  risk_ratio: 0.01
  # ... rest of conservative mode

risk:
  strategy:
    max_position_size: 0.1      # ← ML should determine
    stop_loss: 0.05             # ← ML should determine
    take_profit: 0.15           # ← ML should determine
  circuit_breakers:
    daily_loss_limit: 0.05      # ← ML should determine
```

### Keep Only ML Input Parameters:
```yaml
# KEEP THESE (Graduation/Retirement Criteria Only)
ml_risk_params:
  graduation_criteria:
    min_profit_consistency: 0.7
    min_sharpe_ratio: 1.0
  retirement_criteria:
    max_consecutive_losses: 10
    drawdown_threshold: 0.15
```

## 🎯 ARCHITECTURE ALIGNMENT CHECK
- ✅ Three-phase pipeline exists
- ✅ ML Risk Manager implemented
- ❌ Hardcoded config overrides ML decisions
- ❌ Workspace bloated with unnecessary files
- ❌ Multiple conflicting deployment approaches
- ❌ Documentation chaos

## 🚀 SUCCESS CRITERIA
1. **Clean Workspace**: <20 root-level files
2. **ML-First Risk**: No hardcoded risk parameters
3. **Simple Deployment**: Single Dockerfile approach
4. **Clear Documentation**: Only ARCHITECTURE.md and README.md
5. **Focused Features**: Core trading pipeline only

---
*Goal: Transform bloated, confused system into clean, focused AI trading bot*