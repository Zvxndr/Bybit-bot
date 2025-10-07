"""
Multi-Factor Authentication Manager
==================================

Provides MFA functionality for admin access and API endpoints.
Supports TOTP (Time-based One-Time Password) with QR code generation.
"""

import pyotp
import qrcode
import io
import base64
from cryptography.fernet import Fernet
from typing import Dict, List, Optional
import secrets
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MFAManager:
    """Multi-Factor Authentication Manager for secure access"""
    
    def __init__(self, encryption_key: str):
        """Initialize MFA manager with encryption key"""
        try:
            self.cipher = Fernet(encryption_key.encode() if len(encryption_key) == 44 else Fernet.generate_key())
        except Exception:
            # Generate new key if invalid
            self.cipher = Fernet(Fernet.generate_key())
            logger.warning("Invalid encryption key provided, generated new key")
            
        self.backup_codes_per_user = 10
        
    def generate_mfa_secret(self, user_id: str, issuer_name: str = "Bybit Trading Bot") -> Dict[str, str]:
        """
        Generate MFA secret for user
        
        Args:
            user_id: Unique identifier for user
            issuer_name: Name of the application
            
        Returns:
            Dictionary containing encrypted secret, QR code URI, and backup codes
        """
        try:
            # Generate random secret
            secret = pyotp.random_base32()
            
            # Encrypt the secret for storage
            encrypted_secret = self.cipher.encrypt(secret.encode()).decode()
            
            # Generate QR code URI for authenticator apps
            totp = pyotp.TOTP(secret)
            qr_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name=issuer_name
            )
            
            # Generate backup codes
            backup_codes = self._generate_backup_codes()
            
            logger.info(f"MFA secret generated for user: {user_id}")
            
            return {
                'encrypted_secret': encrypted_secret,
                'qr_code_uri': qr_uri,
                'backup_codes': backup_codes,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating MFA secret for {user_id}: {str(e)}")
            raise
    
    def generate_qr_code_image(self, qr_uri: str) -> str:
        """
        Generate QR code image as base64 string
        
        Args:
            qr_uri: QR code URI from provisioning_uri
            
        Returns:
            Base64 encoded PNG image
        """
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(qr_uri)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error generating QR code image: {str(e)}")
            raise
    
    def verify_mfa_token(self, encrypted_secret: str, token: str, backup_codes: List[str] = None) -> Dict[str, bool]:
        """
        Verify MFA token or backup code
        
        Args:
            encrypted_secret: Encrypted TOTP secret
            token: 6-digit TOTP token or backup code
            backup_codes: List of backup codes (optional)
            
        Returns:
            Dictionary with verification result and method used
        """
        try:
            # Decrypt the secret
            secret = self.cipher.decrypt(encrypted_secret.encode()).decode()
            totp = pyotp.TOTP(secret)
            
            # Try TOTP verification first
            if totp.verify(token, valid_window=1):  # Allow 1 window (30 seconds) tolerance
                logger.info("MFA token verified successfully")
                return {
                    'verified': True,
                    'method': 'totp',
                    'used_backup_code': None
                }
            
            # Try backup codes if provided
            if backup_codes and token in backup_codes:
                logger.info("MFA backup code verified successfully")
                return {
                    'verified': True,
                    'method': 'backup_code',
                    'used_backup_code': token
                }
            
            logger.warning("MFA token verification failed")
            return {
                'verified': False,
                'method': None,
                'used_backup_code': None
            }
            
        except Exception as e:
            logger.error(f"Error verifying MFA token: {str(e)}")
            return {
                'verified': False,
                'method': None,
                'used_backup_code': None,
                'error': str(e)
            }
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA"""
        codes = []
        for _ in range(self.backup_codes_per_user):
            # Generate 8-character alphanumeric codes
            code = ''.join(secrets.choice('ABCDEFGHIJKLMNPQRSTUVWXYZ23456789') for _ in range(8))
            codes.append(code)
        
        return codes
    
    def is_token_valid_format(self, token: str) -> bool:
        """Check if token is in valid format (6 digits or backup code)"""
        # TOTP tokens are 6 digits
        if token.isdigit() and len(token) == 6:
            return True
        
        # Backup codes are 8 alphanumeric characters
        if len(token) == 8 and token.isalnum():
            return True
            
        return False


class MFASession:
    """Session management for MFA-protected operations"""
    
    def __init__(self, session_timeout_minutes: int = 60):
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.active_sessions: Dict[str, datetime] = {}
    
    def create_session(self, user_id: str) -> str:
        """Create MFA session for user"""
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[f"{user_id}:{session_id}"] = datetime.now()
        
        logger.info(f"MFA session created for user: {user_id}")
        return session_id
    
    def validate_session(self, user_id: str, session_id: str) -> bool:
        """Validate MFA session"""
        session_key = f"{user_id}:{session_id}"
        
        if session_key not in self.active_sessions:
            return False
        
        session_time = self.active_sessions[session_key]
        if datetime.now() - session_time > self.session_timeout:
            # Session expired
            del self.active_sessions[session_key]
            logger.info(f"MFA session expired for user: {user_id}")
            return False
        
        # Extend session
        self.active_sessions[session_key] = datetime.now()
        return True
    
    def invalidate_session(self, user_id: str, session_id: str = None):
        """Invalidate MFA session(s)"""
        if session_id:
            session_key = f"{user_id}:{session_id}"
            if session_key in self.active_sessions:
                del self.active_sessions[session_key]
        else:
            # Invalidate all sessions for user
            keys_to_remove = [key for key in self.active_sessions.keys() if key.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self.active_sessions[key]
        
        logger.info(f"MFA session(s) invalidated for user: {user_id}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_keys = [
            key for key, session_time in self.active_sessions.items()
            if current_time - session_time > self.session_timeout
        ]
        
        for key in expired_keys:
            del self.active_sessions[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired MFA sessions")


# Example usage and testing
if __name__ == "__main__":
    # Test MFA functionality
    import os
    
    # Generate encryption key for testing
    encryption_key = Fernet.generate_key().decode()
    
    # Initialize MFA manager
    mfa = MFAManager(encryption_key)
    
    # Generate MFA for test user
    user_setup = mfa.generate_mfa_secret("admin@tradingbot.com")
    
    print("MFA Setup Complete:")
    print(f"QR Code URI: {user_setup['qr_code_uri']}")
    print(f"Backup Codes: {user_setup['backup_codes'][:3]}...")
    
    # Test session management
    session_manager = MFASession()
    session_id = session_manager.create_session("admin")
    
    print(f"Session created: {session_id[:16]}...")
    print(f"Session valid: {session_manager.validate_session('admin', session_id)}")