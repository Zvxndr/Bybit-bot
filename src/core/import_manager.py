"""
Import Manager - Unified Import Handling
=======================================

Provides clean, predictable import strategy for all components.
Eliminates relative import confusion and ensures consistent behavior.
"""

import sys
import os
from pathlib import Path
from typing import Any, Optional


class ImportManager:
    """Manages all application imports with predictable behavior"""
    
    def __init__(self):
        self.src_path = Path(__file__).parent.parent
        self.project_root = self.src_path.parent
        self._setup_python_path()
        
    def _setup_python_path(self):
        """Setup Python path for consistent imports"""
        paths_to_add = [
            str(self.src_path),           # For src.module imports
            str(self.project_root),       # For direct imports
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
    
    def safe_import(self, module_path: str, fallback: Optional[Any] = None) -> Any:
        """
        Safely import module with fallback
        
        Args:
            module_path: Module to import (e.g., 'debug_safety.debug_safety')
            fallback: Return value if import fails
            
        Returns:
            Imported module or fallback value
        """
        try:
            # Try standard import
            module = __import__(module_path, fromlist=[''])
            return module
        except ImportError:
            try:
                # Try with src prefix
                src_module_path = f"src.{module_path}"
                module = __import__(src_module_path, fromlist=[''])
                return module
            except ImportError:
                return fallback
    
    def get_debug_safety_functions(self):
        """Get debug safety functions with proper fallbacks"""
        try:
            # Try direct import first
            from debug_safety import get_debug_manager, is_debug_mode, block_trading_if_debug
            return get_debug_manager, is_debug_mode, block_trading_if_debug
        except ImportError:
            try:
                # Try src-prefixed import
                from src.debug_safety import get_debug_manager, is_debug_mode, block_trading_if_debug
                return get_debug_manager, is_debug_mode, block_trading_if_debug
            except ImportError:
                # Create safe fallbacks
                def safe_get_debug_manager():
                    return None
                    
                def safe_is_debug_mode():
                    return True  # Safer to assume debug mode
                    
                def safe_block_trading_if_debug():
                    return True  # Block trading if uncertain
                    
                return safe_get_debug_manager, safe_is_debug_mode, safe_block_trading_if_debug


# Global import manager instance
import_manager = ImportManager()