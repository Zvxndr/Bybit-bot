"""
Encryption Manager
=================

Handles data encryption, key management, and secure storage for sensitive data.
Provides AES encryption for API keys and other sensitive information.
"""

import os
import base64
import hashlib
import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from typing import Dict, Optional, Union, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages encryption keys and provides secure data encryption/decryption"""
    
    def __init__(self, key_file_path: str = None, password: str = None):
        """
        Initialize encryption manager
        
        Args:
            key_file_path: Path to store encryption key file
            password: Password for key derivation (optional)
        """
        self.backend = default_backend()
        self.key_file_path = key_file_path or os.getenv('ENCRYPTION_KEY_FILE', '.encryption_key')
        self.password = password or os.getenv('ENCRYPTION_PASSWORD')
        
        # Initialize master key
        self.master_key = self._initialize_master_key()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        
        logger.info("✅ Encryption manager initialized")
    
    def _initialize_master_key(self) -> bytes:
        """Initialize or load master encryption key"""
        key_path = Path(self.key_file_path)
        
        if key_path.exists():
            # Load existing key
            try:
                with open(key_path, 'rb') as f:
                    key_data = f.read()
                
                if self.password:
                    # Decrypt key with password
                    key = self._derive_key_from_password(self.password, key_data[:16])  # First 16 bytes are salt
                    encrypted_key = key_data[16:]
                    
                    cipher = Cipher(algorithms.AES(key), modes.CFB(encrypted_key[:16]), backend=self.backend)
                    decryptor = cipher.decryptor()
                    master_key = decryptor.update(encrypted_key[16:]) + decryptor.finalize()
                else:
                    master_key = key_data
                
                logger.info(f"📄 Loaded encryption key from {key_path}")
                return master_key
                
            except Exception as e:
                logger.error(f"❌ Failed to load encryption key: {str(e)}")
                logger.info("🔑 Generating new encryption key")
        
        # Generate new key
        master_key = self._generate_master_key()
        self._save_master_key(master_key)
        
        return master_key
    
    def _generate_master_key(self) -> bytes:
        """Generate new master encryption key"""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _save_master_key(self, master_key: bytes):
        """Save master key to file with optional password protection"""
        key_path = Path(self.key_file_path)
        
        try:
            # Ensure directory exists
            key_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.password:
                # Encrypt key with password
                salt = secrets.token_bytes(16)
                derived_key = self._derive_key_from_password(self.password, salt)
                
                iv = secrets.token_bytes(16)
                cipher = Cipher(algorithms.AES(derived_key), modes.CFB(iv), backend=self.backend)
                encryptor = cipher.encryptor()
                encrypted_key = encryptor.update(master_key) + encryptor.finalize()
                
                # Save salt + iv + encrypted_key
                key_data = salt + iv + encrypted_key
            else:
                key_data = master_key
            
            with open(key_path, 'wb') as f:
                f.write(key_data)
            
            # Set restrictive permissions (owner read-only)
            os.chmod(key_path, 0o600)
            
            logger.info(f"💾 Saved encryption key to {key_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save encryption key: {str(e)}")
            raise
    
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
            backend=self.backend
        )
        return kdf.derive(password.encode())
    
    def encrypt_sensitive_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Encrypt sensitive data
        
        Args:
            data: Data to encrypt (string, bytes, or dict)
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode()
            elif isinstance(data, str):
                data_bytes = data.encode()
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode()
            
            # Encrypt using Fernet (includes authentication)
            encrypted_data = self.fernet.encrypt(data_bytes)
            
            # Return base64 encoded for storage
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"❌ Encryption failed: {str(e)}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str, return_type: str = 'string') -> Union[str, bytes, Dict[str, Any]]:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            return_type: Type to return ('string', 'bytes', 'dict')
            
        Returns:
            Decrypted data in specified format
        """
        try:
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            
            # Decrypt using Fernet
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            
            # Return in requested format
            if return_type == 'bytes':
                return decrypted_bytes
            elif return_type == 'dict':
                return json.loads(decrypted_bytes.decode())
            else:  # string
                return decrypted_bytes.decode()
                
        except Exception as e:
            logger.error(f"❌ Decryption failed: {str(e)}")
            raise
    
    def encrypt_api_credentials(self, credentials: Dict[str, str]) -> str:
        """
        Encrypt API credentials dictionary
        
        Args:
            credentials: Dictionary of API credentials
            
        Returns:
            Encrypted credentials as base64 string
        """
        return self.encrypt_sensitive_data(credentials)
    
    def decrypt_api_credentials(self, encrypted_credentials: str) -> Dict[str, str]:
        """
        Decrypt API credentials
        
        Args:
            encrypted_credentials: Encrypted credentials string
            
        Returns:
            Dictionary of decrypted credentials
        """
        return self.decrypt_sensitive_data(encrypted_credentials, return_type='dict')
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: bytes = None) -> Dict[str, str]:
        """
        Hash password using Scrypt
        
        Args:
            password: Password to hash
            salt: Optional salt (generates new if not provided)
            
        Returns:
            Dictionary with salt and hash
        """
        if salt is None:
            salt = secrets.token_bytes(16)
        
        # Use Scrypt for password hashing (more secure than PBKDF2 for passwords)
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            n=2**14,  # CPU/memory cost parameter
            r=8,      # Block size parameter
            p=1,      # Parallelization parameter
            backend=self.backend
        )
        
        password_hash = kdf.derive(password.encode())
        
        return {
            'salt': base64.b64encode(salt).decode(),
            'hash': base64.b64encode(password_hash).decode()
        }
    
    def verify_password(self, password: str, salt: str, stored_hash: str) -> bool:
        """
        Verify password against stored hash
        
        Args:
            password: Password to verify
            salt: Base64 encoded salt
            stored_hash: Base64 encoded stored hash
            
        Returns:
            True if password matches
        """
        try:
            salt_bytes = base64.b64decode(salt.encode())
            stored_hash_bytes = base64.b64decode(stored_hash.encode())
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                n=2**14,
                r=8,
                p=1,
                backend=self.backend
            )
            
            # This will raise InvalidKey if password doesn't match
            kdf.verify(password.encode(), stored_hash_bytes)
            return True
            
        except Exception:
            return False
    
    def rotate_master_key(self, new_password: str = None):
        """
        Rotate master encryption key
        
        Args:
            new_password: New password for key protection
        """
        logger.info("🔄 Starting master key rotation")
        
        # Generate new master key
        new_master_key = self._generate_master_key()
        
        # Backup old key file
        key_path = Path(self.key_file_path)
        if key_path.exists():
            backup_path = key_path.with_suffix(f'.backup.{int(datetime.now().timestamp())}')
            key_path.rename(backup_path)
            logger.info(f"📦 Backed up old key to {backup_path}")
        
        # Save new key
        old_password = self.password
        self.password = new_password or old_password
        self._save_master_key(new_master_key)
        
        # Update instance
        self.master_key = new_master_key
        self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        
        logger.info("✅ Master key rotation completed")
        logger.warning("⚠️  Re-encrypt all existing encrypted data with new key!")
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """Get encryption manager information"""
        return {
            'key_file_path': self.key_file_path,
            'key_file_exists': Path(self.key_file_path).exists(),
            'password_protected': bool(self.password),
            'key_size_bits': len(self.master_key) * 8,
            'cipher_algorithm': 'AES-256-CFB',
            'kdf_algorithm': 'PBKDF2-SHA256' if self.password else 'None'
        }


class SecureStorage:
    """Secure storage for sensitive configuration data"""
    
    def __init__(self, storage_file: str = '.secure_storage.json', encryption_manager: EncryptionManager = None):
        """
        Initialize secure storage
        
        Args:
            storage_file: File to store encrypted data
            encryption_manager: Encryption manager instance
        """
        self.storage_file = Path(storage_file)
        self.encryption_manager = encryption_manager or EncryptionManager()
        self._data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load encrypted data from storage file"""
        if not self.storage_file.exists():
            return {}
        
        try:
            with open(self.storage_file, 'r') as f:
                encrypted_data = f.read()
            
            if not encrypted_data.strip():
                return {}
            
            # Decrypt the entire data structure
            decrypted_data = self.encryption_manager.decrypt_sensitive_data(
                encrypted_data, return_type='dict'
            )
            
            logger.info(f"📄 Loaded secure storage from {self.storage_file}")
            return decrypted_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load secure storage: {str(e)}")
            return {}
    
    def _save_data(self):
        """Save encrypted data to storage file"""
        try:
            # Encrypt the entire data structure
            encrypted_data = self.encryption_manager.encrypt_sensitive_data(self._data)
            
            # Ensure directory exists
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_file, 'w') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.storage_file, 0o600)
            
            logger.debug(f"💾 Saved secure storage to {self.storage_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save secure storage: {str(e)}")
            raise
    
    def set(self, key: str, value: Any):
        """Set encrypted value in storage"""
        self._data[key] = value
        self._save_data()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get decrypted value from storage"""
        return self._data.get(key, default)
    
    def delete(self, key: str) -> bool:
        """Delete key from storage"""
        if key in self._data:
            del self._data[key]
            self._save_data()
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in storage"""
        return key in self._data
    
    def keys(self) -> List[str]:
        """Get all keys in storage"""
        return list(self._data.keys())
    
    def clear(self):
        """Clear all data from storage"""
        self._data.clear()
        self._save_data()


# Example usage and testing
if __name__ == "__main__":
    from datetime import datetime
    
    # Test encryption manager
    print("🔐 Testing Encryption Manager")
    
    # Initialize with password protection
    encryption = EncryptionManager(key_file_path='test_encryption.key', password='test_password_123')
    
    # Test data encryption
    test_data = {
        'api_key': 'test_api_key_12345',
        'api_secret': 'test_secret_67890',
        'timestamp': datetime.now().isoformat()
    }
    
    encrypted = encryption.encrypt_sensitive_data(test_data)
    print(f"Encrypted: {encrypted[:50]}...")
    
    decrypted = encryption.decrypt_sensitive_data(encrypted, return_type='dict')
    print(f"Decrypted: {decrypted}")
    
    # Test secure storage
    print("\n🗄️  Testing Secure Storage")
    
    storage = SecureStorage('test_storage.json', encryption)
    
    # Store API credentials
    storage.set('bybit_testnet', {
        'api_key': 'testnet_key_123',
        'api_secret': 'testnet_secret_456'
    })
    
    storage.set('sendgrid', {
        'api_key': 'sendgrid_key_789'
    })
    
    # Retrieve credentials
    bybit_creds = storage.get('bybit_testnet')
    print(f"Bybit credentials: {bybit_creds}")
    
    # List all keys
    print(f"Storage keys: {storage.keys()}")
    
    # Get encryption info
    info = encryption.get_encryption_info()
    print(f"\nEncryption info: {json.dumps(info, indent=2)}")
    
    # Cleanup test files
    Path('test_encryption.key').unlink(missing_ok=True)
    Path('test_storage.json').unlink(missing_ok=True)