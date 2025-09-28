# Debug Mode Deployment Setup
## Automatic Test Script Execution

### 🔧 **Setup Complete**

I've configured your Bybit bot to automatically run debug scripts during deployment when in debug mode.

### 📋 **What Was Added:**

1. **Debug Mode Detection**: Enhanced `main.py` to detect debug mode through:
   - Environment variable: `DEBUG_MODE=true`
   - Environment variable: `ENVIRONMENT=debug`
   - Debug flag file: `.debug` (exists in root directory)
   - Command line argument: `--debug`

2. **Automatic Script Execution**: Added methods to run:
   - `test_button_functions.py` - Tests all API endpoints
   - `debug_data_wipe.py` - Tests data wipe functionality specifically

3. **Startup Scripts**:
   - `start_debug.bat` - Windows batch file to start in debug mode
   - `start_debug.sh` - Linux/Mac shell script to start in debug mode
   - `.debug` - Flag file that triggers debug mode

### 🚀 **How to Use:**

#### **Option 1: Use the Debug Startup Script (Recommended)**
```powershell
# Windows
./start_debug.bat

# Linux/Mac  
./start_debug.sh
```

#### **Option 2: Set Environment Variables Manually**
```powershell
# PowerShell
$env:DEBUG_MODE = "true"
$env:ENVIRONMENT = "debug"
python src/main.py --debug
```

#### **Option 3: The .debug file already exists**
Simply run:
```powershell
python src/main.py
```

### 📊 **What Happens During Debug Deployment:**

1. **Application starts** with debug mode detection
2. **Initialization completes** (all normal startup processes)
3. **Debug scripts run automatically**:
   - Tests all button functionality (`/api/health`, `/api/bot/start`, `/api/bot/pause`, `/api/bot/emergency-stop`, `/api/admin/wipe-data`, etc.)
   - Tests data wipe functionality directly and via API
   - Logs all results to the main application log
4. **Application continues** running normally
5. **All results logged** with detailed output

### 🔍 **Log Output You'll See:**

```
🔧 DEBUG MODE ACTIVATED - Test scripts will run automatically
🔧 Debug mode detected - running automated debug scripts
🧪 Running button function tests...
✅ Button function tests completed successfully
🔥 Running data wipe debug tests...
✅ Data wipe debug tests completed successfully
```

### 📁 **Files Created/Modified:**

- `src/main.py` - Enhanced with debug script execution
- `.debug` - Debug mode flag file
- `start_debug.bat` - Windows debug startup script
- `start_debug.sh` - Unix debug startup script
- `test_button_functions.py` - Comprehensive API testing (already existed)
- `debug_data_wipe.py` - Data wipe specific testing (already existed)

### 💡 **Key Benefits:**

1. **Automatic Testing**: No manual script execution needed
2. **Comprehensive Logging**: All test results logged to main application
3. **Non-Blocking**: Scripts run as subprocesses, don't block main app
4. **Fail-Safe**: If debug scripts fail, main application continues
5. **Production Safe**: Only runs in debug mode

### 🎯 **Ready for Deployment**

Your bot is now ready for debug deployment! When you start it:

1. The debug scripts will run automatically
2. All results will be logged 
3. You can share the logs to see exactly what's working/failing
4. The main application will continue running normally

Run `./start_debug.bat` and share the logs once it's complete!