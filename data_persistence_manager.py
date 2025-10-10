#!/usr/bin/env python3
"""
Data Persistence Manager for Production Deployments
====================================================

Ensures data persistence across deployments and provides hard reset functionality.
Handles both historical data and strategy databases with backup/restore capabilities.
"""

import os
import shutil
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
import asyncio


class DataPersistenceManager:
    """Manages data persistence across deployments with backup/restore capabilities"""
    
    def __init__(self, data_dir: str = "data", backup_dir: str = "backups"):
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.persistent_files = [
            "historical_data.db",
            "trading_bot.db", 
            "strategies.db"
        ]
        self.persistent_dirs = [
            "models",
            "strategies", 
            "speed_demon_cache"
        ]
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔄 Data Persistence Manager initialized")
        logger.info(f"📂 Data directory: {self.data_dir.absolute()}")
        logger.info(f"💾 Backup directory: {self.backup_dir.absolute()}")
    
    def create_deployment_backup(self) -> Dict[str, any]:
        """Create backup before deployment to prevent data loss"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"pre_deployment_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        backed_up_files = []
        backed_up_dirs = []
        
        try:
            # Backup database files
            for file_name in self.persistent_files:
                file_path = self.data_dir / file_name
                if file_path.exists():
                    backup_file = backup_path / file_name
                    shutil.copy2(file_path, backup_file)
                    backed_up_files.append(file_name)
                    logger.info(f"💾 Backed up: {file_name}")
            
            # Backup directories
            for dir_name in self.persistent_dirs:
                dir_path = self.data_dir / dir_name
                if dir_path.exists():
                    backup_dir_path = backup_path / dir_name
                    shutil.copytree(dir_path, backup_dir_path, dirs_exist_ok=True)
                    backed_up_dirs.append(dir_name)
                    logger.info(f"📁 Backed up directory: {dir_name}")
            
            # Create backup manifest
            manifest = {
                "timestamp": timestamp,
                "backup_name": backup_name,
                "files": backed_up_files,
                "directories": backed_up_dirs,
                "created_at": datetime.now().isoformat()
            }
            
            manifest_path = backup_path / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.success(f"✅ Deployment backup created: {backup_name}")
            return manifest
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise
    
    def restore_from_backup(self, backup_name: Optional[str] = None) -> Dict[str, any]:
        """Restore data from backup (uses latest if backup_name not specified)"""
        try:
            # Find backup to restore
            if backup_name:
                backup_path = self.backup_dir / backup_name
            else:
                # Find latest backup
                backups = [d for d in self.backup_dir.iterdir() if d.is_dir()]
                if not backups:
                    raise ValueError("No backups found")
                backup_path = max(backups, key=lambda d: d.stat().st_mtime)
            
            if not backup_path.exists():
                raise ValueError(f"Backup not found: {backup_path}")
            
            # Load manifest
            manifest_path = backup_path / "manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            restored_files = []
            restored_dirs = []
            
            # Restore files
            for file_name in manifest.get("files", []):
                backup_file = backup_path / file_name
                if backup_file.exists():
                    target_file = self.data_dir / file_name
                    shutil.copy2(backup_file, target_file)
                    restored_files.append(file_name)
                    logger.info(f"🔄 Restored: {file_name}")
            
            # Restore directories
            for dir_name in manifest.get("directories", []):
                backup_dir_path = backup_path / dir_name
                if backup_dir_path.exists():
                    target_dir = self.data_dir / dir_name
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.copytree(backup_dir_path, target_dir)
                    restored_dirs.append(dir_name)
                    logger.info(f"📁 Restored directory: {dir_name}")
            
            restore_info = {
                "backup_used": backup_path.name,
                "files_restored": restored_files,
                "directories_restored": restored_dirs,
                "restored_at": datetime.now().isoformat()
            }
            
            logger.success(f"✅ Data restored from backup: {backup_path.name}")
            return restore_info
            
        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            raise
    
    def hard_reset_production_data(self, confirm: bool = False) -> Dict[str, any]:
        """
        HARD RESET: Completely clears all data (strategies, historical data, cache)
        WARNING: This is irreversible! Only use for production debugging.
        """
        if not confirm:
            raise ValueError("Hard reset requires explicit confirmation (confirm=True)")
        
        # Create emergency backup before reset
        logger.warning("⚠️ HARD RESET INITIATED - Creating emergency backup...")
        emergency_backup = self.create_deployment_backup()
        
        reset_info = {
            "reset_at": datetime.now().isoformat(),
            "emergency_backup": emergency_backup["backup_name"],
            "cleared_files": [],
            "cleared_directories": [],
            "cleared_cache": []
        }
        
        try:
            # Clear database files
            for file_name in self.persistent_files:
                file_path = self.data_dir / file_name
                if file_path.exists():
                    file_path.unlink()
                    reset_info["cleared_files"].append(file_name)
                    logger.warning(f"🗑️ Deleted: {file_name}")
            
            # Clear persistent directories
            for dir_name in self.persistent_dirs:
                dir_path = self.data_dir / dir_name
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    reset_info["cleared_directories"].append(dir_name)
                    logger.warning(f"🗑️ Deleted directory: {dir_name}")
            
            # Clear additional cache/temp files
            cache_patterns = ["*.tmp", "*.cache", "*.log"]
            for pattern in cache_patterns:
                for file_path in self.data_dir.glob(pattern):
                    file_path.unlink()
                    reset_info["cleared_cache"].append(file_path.name)
            
            # Recreate necessary directories
            for dir_name in self.persistent_dirs:
                (self.data_dir / dir_name).mkdir(exist_ok=True)
            
            logger.warning("💥 HARD RESET COMPLETE - All production data cleared!")
            logger.info(f"🔄 Emergency backup available: {emergency_backup['backup_name']}")
            
            return reset_info
            
        except Exception as e:
            logger.error(f"❌ Hard reset failed: {e}")
            logger.info("🔄 Attempting to restore from emergency backup...")
            self.restore_from_backup(emergency_backup["backup_name"])
            raise
    
    def get_data_status(self) -> Dict[str, any]:
        """Get current data status and statistics"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "data_directory": str(self.data_dir.absolute()),
            "files": {},
            "directories": {},
            "backups": [],
            "total_size_mb": 0
        }
        
        total_size = 0
        
        # Check files
        for file_name in self.persistent_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                size = file_path.stat().st_size
                total_size += size
                status["files"][file_name] = {
                    "exists": True,
                    "size_mb": round(size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            else:
                status["files"][file_name] = {"exists": False}
        
        # Check directories
        for dir_name in self.persistent_dirs:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                total_size += dir_size
                file_count = len([f for f in dir_path.rglob('*') if f.is_file()])
                status["directories"][dir_name] = {
                    "exists": True,
                    "size_mb": round(dir_size / 1024 / 1024, 2),
                    "file_count": file_count
                }
            else:
                status["directories"][dir_name] = {"exists": False}
        
        # List available backups
        if self.backup_dir.exists():
            for backup_dir in sorted(self.backup_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
                if backup_dir.is_dir():
                    manifest_path = backup_dir / "manifest.json"
                    if manifest_path.exists():
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                        status["backups"].append({
                            "name": backup_dir.name,
                            "created": manifest.get("created_at"),
                            "files": len(manifest.get("files", [])),
                            "directories": len(manifest.get("directories", []))
                        })
        
        status["total_size_mb"] = round(total_size / 1024 / 1024, 2)
        
        return status
    
    def validate_data_integrity(self) -> Dict[str, any]:
        """Validate integrity of all persistent data"""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "database_checks": {},
            "directory_checks": {},
            "issues": []
        }
        
        # Check database files
        for file_name in self.persistent_files:
            file_path = self.data_dir / file_name
            if file_path.exists() and file_name.endswith('.db'):
                try:
                    with sqlite3.connect(str(file_path)) as conn:
                        # Run integrity check
                        result = conn.execute("PRAGMA integrity_check").fetchone()
                        is_healthy = result[0] == "ok" if result else False
                        
                        # Get table count
                        tables = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
                        
                        validation["database_checks"][file_name] = {
                            "integrity": "ok" if is_healthy else "corrupted",
                            "table_count": tables,
                            "size_mb": round(file_path.stat().st_size / 1024 / 1024, 2)
                        }
                        
                        if not is_healthy:
                            validation["issues"].append(f"Database corruption detected: {file_name}")
                            validation["overall_status"] = "warning"
                            
                except Exception as e:
                    validation["database_checks"][file_name] = {"error": str(e)}
                    validation["issues"].append(f"Cannot access database: {file_name} - {e}")
                    validation["overall_status"] = "error"
        
        # Check directories
        for dir_name in self.persistent_dirs:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                try:
                    file_count = len([f for f in dir_path.rglob('*') if f.is_file()])
                    validation["directory_checks"][dir_name] = {
                        "accessible": True,
                        "file_count": file_count
                    }
                except Exception as e:
                    validation["directory_checks"][dir_name] = {"error": str(e)}
                    validation["issues"].append(f"Cannot access directory: {dir_name} - {e}")
                    validation["overall_status"] = "error"
        
        if validation["issues"]:
            logger.warning(f"⚠️ Data integrity issues found: {len(validation['issues'])}")
        else:
            logger.info("✅ Data integrity validation passed")
        
        return validation


# Global instance for easy access
persistence_manager = DataPersistenceManager()


async def ensure_data_persistence():
    """Ensure data persistence is properly configured for deployment"""
    try:
        # Check current data status
        status = persistence_manager.get_data_status()
        logger.info(f"📊 Current data status: {status['total_size_mb']:.2f} MB")
        
        # Validate data integrity
        validation = persistence_manager.validate_data_integrity()
        if validation["overall_status"] != "healthy":
            logger.warning("⚠️ Data integrity issues detected - consider running hard reset")
        
        # Create backup if significant data exists
        if status["total_size_mb"] > 1.0:  # Only backup if >1MB of data
            persistence_manager.create_deployment_backup()
        
        return {
            "status": status,
            "validation": validation,
            "persistence_ready": True
        }
        
    except Exception as e:
        logger.error(f"❌ Data persistence setup failed: {e}")
        return {
            "persistence_ready": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the persistence manager
    asyncio.run(ensure_data_persistence())