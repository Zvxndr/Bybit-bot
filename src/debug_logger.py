"""
Comprehensive Debugging and Logging Utilities
==============================================

Enhanced logging and debugging utilities for the Open Alpha bot.
Provides detailed runtime information for troubleshooting.
"""

import logging
import traceback
import sys
import os
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

class ComprehensiveLogger:
    """Enhanced logging with system diagnostics"""
    
    def __init__(self):
        self.logger = logging.getLogger('open_alpha_debug')
        self.start_time = time.time()
    
    def log_system_info(self):
        """Log comprehensive system information"""
        self.logger.info("=" * 50)
        self.logger.info("🔧 SYSTEM DIAGNOSTICS")
        self.logger.info("=" * 50)
        
        # Python info
        self.logger.info(f"🐍 Python version: {sys.version}")
        self.logger.info(f"🐍 Python executable: {sys.executable}")
        self.logger.info(f"📁 Working directory: {os.getcwd()}")
        self.logger.info(f"📁 Script location: {Path(__file__).parent}")
        
        # System resources
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.logger.info(f"💻 CPU usage: {cpu_percent:.1f}%")
            self.logger.info(f"💾 Memory: {memory.percent:.1f}% used ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
            self.logger.info(f"💽 Disk: {disk.percent:.1f}% used ({disk.used/1024/1024/1024:.1f}GB / {disk.total/1024/1024/1024:.1f}GB)")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get system info: {e}")
        
        # Environment variables (safe ones)
        self.logger.info("🔧 Environment variables:")
        safe_vars = ['PATH', 'PYTHONPATH', 'HOME', 'USER', 'USERNAME']
        for var in safe_vars:
            value = os.environ.get(var, 'Not set')
            if var == 'PATH':
                # Truncate PATH for readability
                value = value[:100] + '...' if len(value) > 100 else value
            self.logger.info(f"   {var}: {value}")
        
        # Check for important files
        self.logger.info("📁 File system check:")
        important_paths = [
            'src/main.py',
            'src/frontend_server.py', 
            'src/shared_state.py',
            'src/bybit_api.py',
            '.env',
            'requirements.txt'
        ]
        
        for path in important_paths:
            exists = "✅" if Path(path).exists() else "❌"
            self.logger.info(f"   {exists} {path}")
        
        self.logger.info("=" * 50)
    
    def log_import_diagnostics(self):
        """Log module import diagnostics"""
        self.logger.info("🔧 MODULE IMPORT DIAGNOSTICS")
        self.logger.info("-" * 30)
        
        # Check sys.path
        self.logger.info(f"🔧 sys.path entries ({len(sys.path)}):")
        for i, path in enumerate(sys.path[:10]):  # Show first 10
            self.logger.info(f"   [{i}] {path}")
        
        # Check loaded modules
        important_modules = [
            'asyncio', 'logging', 'pathlib', 'threading',
            'pybit', 'requests', 'aiohttp', 'dotenv'
        ]
        
        self.logger.info("🔧 Module availability check:")
        for module in important_modules:
            try:
                __import__(module)
                self.logger.info(f"   ✅ {module}")
            except ImportError as e:
                self.logger.warning(f"   ❌ {module}: {e}")
        
        self.logger.info("-" * 30)
    
    def log_exception_detail(self, exc: Exception, context: str = ""):
        """Log detailed exception information"""
        self.logger.error("=" * 50)
        self.logger.error(f"❌ EXCEPTION DETAILS {f'({context})' if context else ''}")
        self.logger.error("=" * 50)
        
        # Exception basic info
        self.logger.error(f"Exception type: {type(exc).__name__}")
        self.logger.error(f"Exception message: {str(exc)}")
        
        # Get traceback
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        self.logger.error("Traceback (most recent call last):")
        for line in tb_lines:
            for sub_line in line.strip().split('\n'):
                if sub_line:
                    self.logger.error(f"  {sub_line}")
        
        # Context information
        frame = sys._getframe(1)
        self.logger.error(f"Calling function: {frame.f_code.co_name}")
        self.logger.error(f"File: {frame.f_code.co_filename}:{frame.f_lineno}")
        
        # Local variables (safe ones)
        try:
            locals_dict = frame.f_locals
            safe_locals = {}
            for key, value in locals_dict.items():
                if not key.startswith('_') and len(str(value)) < 200:
                    try:
                        # Try to convert to string safely
                        str_value = str(value)
                        safe_locals[key] = str_value
                    except:
                        safe_locals[key] = f"<{type(value).__name__}>"
            
            if safe_locals:
                self.logger.error("Local variables:")
                for key, value in safe_locals.items():
                    self.logger.error(f"  {key}: {value}")
        
        except Exception as e:
            self.logger.error(f"Could not get local variables: {e}")
        
        self.logger.error("=" * 50)
    
    def log_performance_metrics(self, operation: str, start_time: float, **kwargs):
        """Log performance metrics for operations"""
        elapsed = time.time() - start_time
        self.logger.info(f"⏱️ {operation}: {elapsed:.3f}s")
        
        for key, value in kwargs.items():
            self.logger.info(f"   {key}: {value}")
    
    def log_state_dump(self, state_dict: Dict[str, Any], name: str = "State"):
        """Log comprehensive state information"""
        self.logger.debug(f"🔧 {name} Dump:")
        self.logger.debug("-" * 20)
        
        def log_dict(d: Dict, prefix: str = ""):
            for key, value in d.items():
                if isinstance(value, dict):
                    self.logger.debug(f"{prefix}{key}:")
                    log_dict(value, prefix + "  ")
                elif isinstance(value, list):
                    self.logger.debug(f"{prefix}{key}: [{len(value)} items]")
                    if value and len(value) <= 5:  # Show small lists
                        for i, item in enumerate(value):
                            item_str = str(item)[:100]
                            self.logger.debug(f"{prefix}  [{i}]: {item_str}")
                else:
                    value_str = str(value)[:100]
                    self.logger.debug(f"{prefix}{key}: {value_str}")
        
        try:
            log_dict(state_dict)
        except Exception as e:
            self.logger.error(f"Error dumping {name}: {e}")
        
        self.logger.debug("-" * 20)

# Global instance
debug_logger = ComprehensiveLogger()

def log_startup():
    """Log comprehensive startup information"""
    debug_logger.log_system_info()
    debug_logger.log_import_diagnostics()

def log_exception(exc: Exception, context: str = ""):
    """Convenience function to log exception details"""
    debug_logger.log_exception_detail(exc, context)

def log_performance(operation: str, start_time: float, **kwargs):
    """Convenience function to log performance metrics"""
    debug_logger.log_performance_metrics(operation, start_time, **kwargs)

def log_state(state_dict: Dict[str, Any], name: str = "State"):
    """Convenience function to log state"""
    debug_logger.log_state_dump(state_dict, name)