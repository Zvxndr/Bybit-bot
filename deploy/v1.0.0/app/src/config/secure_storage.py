#!/usr/bin/env python3
"""
Secure Configuration Storage
AES-256-GCM encryption for sensitive configuration data
Addresses critical security audit findings
"""

import os
import json
import base64
import secrets
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class SecureConfigManager:
    """
    Secure configuration manager with AES-256-GCM encryption
    Addresses audit finding: Configuration encryption at rest
    """
    
    def __init__(self, config_dir: str = "config", master_key: Optional[str] = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.secure_config_file = self.config_dir / "secure_config.enc"
        self.key_file = self.config_dir / ".key"
        
        # Initialize encryption key
        self.encryption_key = self._initialize_encryption_key(master_key)
        
    def _initialize_encryption_key(self, master_key: Optional[str] = None) -> bytes:
        """Initialize or load encryption key"""
        if master_key:
            # Derive key from master password
            return self._derive_key_from_password(master_key)
        
        # Try to load existing key
        if self.key_file.exists():
            try:
                with open(self.key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Could not load existing key: {e}")
        
        # Generate new key
        key = AESGCM.generate_key(bit_length=256)
        self._save_key(key)
        return key
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        # Use a fixed salt for password-based keys (store this securely in production)
        salt = b"bybit_trading_bot_salt_v1"  # In production, store this separately
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())
    
    def _save_key(self, key: bytes):
        """Save encryption key to file with restricted permissions"""
        try:
            with open(self.key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions (owner only)
            if os.name != 'nt':  # Unix-like systems
                os.chmod(self.key_file, 0o600)
                
        except Exception as e:
            logger.error(f"Failed to save encryption key: {e}")
            raise
    
    def encrypt_config(self, config: Dict[str, Any]) -> str:
        """
        Encrypt configuration data using AES-256-GCM
        
        Args:
            config: Configuration dictionary to encrypt
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            # Convert config to JSON
            config_json = json.dumps(config, indent=2)
            config_bytes = config_json.encode('utf-8')
            
            # Generate nonce
            nonce = os.urandom(12)  # 96-bit nonce for GCM
            
            # Encrypt with AES-GCM
            aesgcm = AESGCM(self.encryption_key)
            encrypted_data = aesgcm.encrypt(nonce, config_bytes, None)
            
            # Combine nonce + encrypted data
            combined = nonce + encrypted_data
            
            # Return base64 encoded result
            return base64.b64encode(combined).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Configuration encryption failed: {e}")
            raise
    
    def decrypt_config(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt configuration data
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted configuration dictionary
        """
        try:
            # Decode base64
            combined = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Split nonce and encrypted data
            nonce = combined[:12]  # First 12 bytes are nonce
            encrypted = combined[12:]  # Rest is encrypted data
            
            # Decrypt with AES-GCM
            aesgcm = AESGCM(self.encryption_key)
            decrypted_bytes = aesgcm.decrypt(nonce, encrypted, None)
            
            # Convert back to dictionary
            config_json = decrypted_bytes.decode('utf-8')
            return json.loads(config_json)
            
        except Exception as e:
            logger.error(f"Configuration decryption failed: {e}")
            raise
    
    def save_secure_config(self, config: Dict[str, Any]):
        """Save configuration with encryption"""
        try:
            encrypted_data = self.encrypt_config(config)
            with open(self.secure_config_file, 'w') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            if os.name != 'nt':  # Unix-like systems
                os.chmod(self.secure_config_file, 0o600)
                
            logger.info("Secure configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save secure configuration: {e}")
            raise
    
    def load_secure_config(self) -> Dict[str, Any]:
        """Load and decrypt configuration"""
        try:
            if not self.secure_config_file.exists():
                logger.warning("Secure configuration file does not exist")
                return {}
            
            with open(self.secure_config_file, 'r') as f:
                encrypted_data = f.read().strip()
            
            return self.decrypt_config(encrypted_data)
            
        except Exception as e:
            logger.error(f"Failed to load secure configuration: {e}")
            raise
    
    def update_config_field(self, field_path: str, value: Any):
        """
        Update specific configuration field
        
        Args:
            field_path: Dot-separated path (e.g., 'api.bybit.key')
            value: New value to set
        """
        try:
            config = self.load_secure_config()
            
            # Navigate to the field
            parts = field_path.split('.')
            current = config
            
            # Create nested structure if needed
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[parts[-1]] = value
            
            # Save updated config
            self.save_secure_config(config)
            
            logger.info(f"Configuration field '{field_path}' updated")
            
        except Exception as e:
            logger.error(f"Failed to update configuration field: {e}")
            raise
    
    def get_config_field(self, field_path: str, default: Any = None) -> Any:
        """
        Get specific configuration field
        
        Args:
            field_path: Dot-separated path (e.g., 'api.bybit.key')
            default: Default value if field doesn't exist
            
        Returns:
            Field value or default
        """
        try:
            config = self.load_secure_config()
            
            # Navigate to the field
            parts = field_path.split('.')
            current = config
            
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            
            return current
            
        except Exception as e:
            logger.warning(f"Failed to get configuration field '{field_path}': {e}")
            return default
    
    def delete_config_field(self, field_path: str):
        """Delete specific configuration field"""
        try:
            config = self.load_secure_config()
            
            # Navigate to the parent
            parts = field_path.split('.')
            current = config
            
            for part in parts[:-1]:
                if not isinstance(current, dict) or part not in current:
                    logger.warning(f"Field path '{field_path}' not found")
                    return
                current = current[part]
            
            # Delete the field
            if parts[-1] in current:
                del current[parts[-1]]
                self.save_secure_config(config)
                logger.info(f"Configuration field '{field_path}' deleted")
            else:
                logger.warning(f"Field '{parts[-1]}' not found")
                
        except Exception as e:
            logger.error(f"Failed to delete configuration field: {e}")
            raise
    
    def rotate_encryption_key(self, new_master_key: Optional[str] = None):
        """
        Rotate encryption key (re-encrypt all data with new key)
        
        Args:
            new_master_key: New master password, or None to generate random key
        """
        try:
            # Load current config
            current_config = self.load_secure_config()
            
            # Generate new encryption key
            if new_master_key:
                new_key = self._derive_key_from_password(new_master_key)
            else:
                new_key = AESGCM.generate_key(bit_length=256)
            
            # Backup old key and config
            backup_key_file = self.config_dir / f".key.backup.{secrets.token_hex(8)}"
            backup_config_file = self.config_dir / f"secure_config.enc.backup.{secrets.token_hex(8)}"
            
            if self.key_file.exists():
                os.rename(self.key_file, backup_key_file)
            if self.secure_config_file.exists():
                os.rename(self.secure_config_file, backup_config_file)
            
            # Set new key and save config
            self.encryption_key = new_key
            self._save_key(new_key)
            self.save_secure_config(current_config)
            
            logger.info("Encryption key rotated successfully")
            logger.info(f"Backup files: {backup_key_file}, {backup_config_file}")
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise
    
    def verify_integrity(self) -> bool:
        """Verify that configuration can be loaded and decrypted"""
        try:
            config = self.load_secure_config()
            # If we can load it, encryption/decryption is working
            return True
        except Exception as e:
            logger.error(f"Configuration integrity check failed: {e}")
            return False
    
    def get_security_info(self) -> Dict[str, Any]:
        """Get security information about the configuration"""
        return {
            "encryption_algorithm": "AES-256-GCM",
            "key_length": 256,
            "config_file_exists": self.secure_config_file.exists(),
            "key_file_exists": self.key_file.exists(),
            "integrity_check": self.verify_integrity(),
            "config_file_size": self.secure_config_file.stat().st_size if self.secure_config_file.exists() else 0,
        }


class SecureEnvironmentManager:
    """
    Manage environment variables securely
    Complement to SecureConfigManager for runtime secrets
    """
    
    @staticmethod
    def get_secure_env(key: str, default: Optional[str] = None, required: bool = True) -> Optional[str]:
        """
        Get environment variable with security logging
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value or default
        """
        value = os.getenv(key, default)
        
        if required and value is None:
            logger.error(f"Required environment variable '{key}' not found")
            raise ValueError(f"Required environment variable '{key}' not found")
        
        if value is not None:
            logger.debug(f"Environment variable '{key}' loaded (length: {len(value)})")
        
        return value
    
    @staticmethod
    def set_secure_env(key: str, value: str):
        """Set environment variable with security logging"""
        os.environ[key] = value
        logger.debug(f"Environment variable '{key}' set (length: {len(value)})")
    
    @staticmethod
    def clear_secure_env(key: str):
        """Clear environment variable securely"""
        if key in os.environ:
            del os.environ[key]
            logger.debug(f"Environment variable '{key}' cleared")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration with sensitive data
    test_config = {
        "api": {
            "bybit": {
                "key": "test_api_key_12345",
                "secret": "test_secret_67890",
                "testnet": True
            },
            "binance": {
                "key": "binance_key_12345",
                "secret": "binance_secret_67890",
                "testnet": True
            }
        },
        "trading": {
            "max_position_size": 1000,
            "stop_loss_percentage": 2.0
        }
    }
    
    # Test secure configuration manager
    print("Testing Secure Configuration Manager...")
    
    # Initialize with master password
    secure_config = SecureConfigManager(master_key="test_master_password_123")
    
    # Save configuration
    secure_config.save_secure_config(test_config)
    print("✅ Configuration encrypted and saved")
    
    # Load configuration
    loaded_config = secure_config.load_secure_config()
    print("✅ Configuration decrypted and loaded")
    
    # Verify data integrity
    assert loaded_config == test_config, "Configuration data mismatch!"
    print("✅ Data integrity verified")
    
    # Test field operations
    secure_config.update_config_field("api.bybit.testnet", False)
    value = secure_config.get_config_field("api.bybit.testnet")
    assert value == False, "Field update failed!"
    print("✅ Field operations working")
    
    # Test security info
    security_info = secure_config.get_security_info()
    print(f"✅ Security info: {security_info}")
    
    print("\n🔒 Secure Configuration Manager ready for production!")