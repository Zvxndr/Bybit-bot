#!/usr/bin/env python3
"""
Simple MFA Test - Core Functionality Check
==========================================

Tests just the essential MFA functions without QR code generation.
"""

import os
import sys
sys.path.append('src')

from security.mfa_manager import MFAManager, MFASession
from cryptography.fernet import Fernet
import pyotp

def test_core_mfa():
    """Test core MFA functionality"""
    print("🔐 CORE MFA FUNCTIONALITY TEST")
    print("=" * 40)
    
    # 1. Generate encryption key
    print("\n🔑 1. Generating encryption key...")
    encryption_key = Fernet.generate_key().decode()
    print(f"✅ Key generated: {encryption_key[:20]}...")
    
    # 2. Initialize MFA Manager
    print("\n🚀 2. Testing MFA Manager...")
    mfa = MFAManager(encryption_key)
    print("✅ MFA Manager initialized")
    
    # 3. Generate MFA secret (without QR code)
    print("\n👤 3. Setting up MFA for test user...")
    user_setup = mfa.generate_mfa_secret("admin@test.com")
    
    print(f"✅ Secret generated and encrypted")
    print(f"📱 QR URI created: {len(user_setup['qr_code_uri'])} characters")
    print(f"🔐 Backup codes: {len(user_setup['backup_codes'])} codes")
    print(f"🆘 Sample backup codes: {user_setup['backup_codes'][:2]}")
    
    # 4. Test token generation (simulate what authenticator app does)
    print("\n🔢 4. Testing token generation...")
    
    # Decrypt secret to generate a test token
    secret = mfa.cipher.decrypt(user_setup['encrypted_secret'].encode()).decode()
    totp = pyotp.TOTP(secret)
    current_token = totp.now()
    
    print(f"✅ Generated test token: {current_token}")
    
    # 5. Test token verification
    print("\n✅ 5. Testing token verification...")
    verification = mfa.verify_mfa_token(
        user_setup['encrypted_secret'], 
        current_token,
        user_setup['backup_codes']
    )
    
    if verification['verified']:
        print(f"✅ Token verified successfully via {verification['method']}")
    else:
        print(f"❌ Token verification failed")
        return False
    
    # 6. Test backup code
    print("\n🆘 6. Testing backup code...")
    backup_verification = mfa.verify_mfa_token(
        user_setup['encrypted_secret'],
        user_setup['backup_codes'][0],
        user_setup['backup_codes']
    )
    
    if backup_verification['verified']:
        print(f"✅ Backup code verified via {backup_verification['method']}")
        print(f"🔄 Used backup code: {backup_verification['used_backup_code']}")
    else:
        print("❌ Backup code verification failed")
        return False
    
    # 7. Test session management
    print("\n⏰ 7. Testing session management...")
    session_mgr = MFASession(session_timeout_minutes=60)
    session_id = session_mgr.create_session("admin")
    
    print(f"✅ Session created: {session_id[:16]}...")
    
    if session_mgr.validate_session("admin", session_id):
        print("✅ Session validation working")
    else:
        print("❌ Session validation failed")
        return False
    
    # 8. Test format validation
    print("\n📝 8. Testing format validation...")
    print(f"✅ 6-digit token format: {mfa.is_token_valid_format('123456')}")
    print(f"✅ 8-char backup format: {mfa.is_token_valid_format('A1B2C3D4')}")
    print(f"❌ Invalid format: {mfa.is_token_valid_format('invalid')}")
    
    print("\n🎉 ALL CORE MFA TESTS PASSED!")
    return True, encryption_key, user_setup

def show_setup_instructions(encryption_key, user_setup):
    """Show what needs to be configured"""
    print("\n" + "=" * 50)
    print("🔧 MFA SETUP INSTRUCTIONS")
    print("=" * 50)
    
    print("\n✅ CURRENT STATUS:")
    print("- All MFA dependencies are installed")
    print("- Core MFA functionality is working")
    print("- Token generation and verification working")
    print("- Session management working")
    print("- Backup codes working")
    
    print("\n📋 TO COMPLETE SETUP:")
    
    print("\n1. 🔑 ADD TO .env FILE:")
    print(f"   MFA_ENCRYPTION_KEY={encryption_key}")
    print("   MFA_SESSION_TIMEOUT=60")
    print("   MFA_ISSUER_NAME=Australian Trust Trading Bot")
    
    print("\n2. 📱 FOR ADMIN SETUP:")
    print("   - Copy this QR code URI to a QR generator")
    print("   - Or manually enter secret in authenticator app")
    print(f"   QR URI: {user_setup['qr_code_uri']}")
    
    print("\n3. 🆘 SAVE BACKUP CODES:")
    print("   Store these codes securely (use only once each):")
    for i, code in enumerate(user_setup['backup_codes'], 1):
        print(f"   {i:2d}. {code}")
    
    print("\n4. 📱 RECOMMENDED AUTHENTICATOR APPS:")
    print("   - Google Authenticator (iOS/Android)")
    print("   - Microsoft Authenticator (iOS/Android)")
    print("   - Authy (iOS/Android/Desktop)")
    print("   - 1Password (if you use it)")
    
    print("\n5. 🔗 INTEGRATION READY:")
    print("   - MFA classes are ready to use")
    print("   - Import: from security.mfa_manager import MFAManager")
    print("   - Use in your web interface or API endpoints")
    
    print("\n6. 🧪 NEXT STEPS:")
    print("   - Test with real authenticator app")
    print("   - Integrate into your dashboard")
    print("   - Add MFA requirements to critical operations")

if __name__ == "__main__":
    print("🚀 TESTING MFA SYSTEM...")
    
    result = test_core_mfa()
    
    if isinstance(result, tuple) and result[0]:
        success, encryption_key, user_setup = result
        show_setup_instructions(encryption_key, user_setup)
        
        print("\n" + "🎉" * 20)
        print("MFA SYSTEM IS READY TO USE!")
        print("🎉" * 20)
        
    else:
        print("\n❌ MFA system has issues. Check the errors above.")