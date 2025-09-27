#!/usr/bin/env python3
"""
MFA Integration Example
======================

Shows how to integrate MFA into your trading bot application.
This demonstrates the complete flow from setup to verification.
"""

import os
import sys
sys.path.append('src')

from security.mfa_manager import MFAManager, MFASession
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class TradingBotMFA:
    """MFA integration for the trading bot"""
    
    def __init__(self):
        # Get MFA settings from environment
        self.mfa_key = os.getenv('MFA_ENCRYPTION_KEY')
        self.session_timeout = int(os.getenv('MFA_SESSION_TIMEOUT', 60))
        self.issuer_name = os.getenv('MFA_ISSUER_NAME', 'Australian Trust Trading Bot')
        
        if not self.mfa_key:
            raise ValueError("MFA_ENCRYPTION_KEY not found in environment variables")
        
        # Initialize MFA components
        self.mfa_manager = MFAManager(self.mfa_key)
        self.session_manager = MFASession(self.session_timeout)
        
        print(f"✅ MFA initialized with {self.session_timeout} minute sessions")
    
    def setup_admin_mfa(self, admin_email: str) -> dict:
        """
        Set up MFA for an admin user
        Returns QR code and backup codes
        """
        print(f"🔧 Setting up MFA for admin: {admin_email}")
        
        # Generate MFA setup
        mfa_setup = self.mfa_manager.generate_mfa_secret(
            admin_email, 
            self.issuer_name
        )
        
        # Save setup data (in production, store in database)
        setup_file = f"mfa_setup_{admin_email.replace('@', '_').replace('.', '_')}.json"
        
        setup_data = {
            'user_email': admin_email,
            'encrypted_secret': mfa_setup['encrypted_secret'],
            'backup_codes': mfa_setup['backup_codes'],
            'setup_date': mfa_setup['created_at'],
            'qr_uri': mfa_setup['qr_code_uri']
        }
        
        with open(setup_file, 'w') as f:
            json.dump(setup_data, f, indent=2)
        
        print(f"✅ MFA setup saved to: {setup_file}")
        print(f"📱 QR Code URI: {mfa_setup['qr_code_uri']}")
        print(f"🔐 Generated {len(mfa_setup['backup_codes'])} backup codes")
        
        return setup_data
    
    def authenticate_admin(self, admin_email: str, mfa_token: str) -> dict:
        """
        Authenticate admin with MFA token
        Returns session info if successful
        """
        print(f"🔐 Authenticating admin: {admin_email}")
        
        # Load admin's MFA setup (in production, get from database)
        setup_file = f"mfa_setup_{admin_email.replace('@', '_').replace('.', '_')}.json"
        
        if not os.path.exists(setup_file):
            return {
                'success': False,
                'error': 'MFA not set up for this user'
            }
        
        with open(setup_file, 'r') as f:
            setup_data = json.load(f)
        
        # Verify the token
        verification = self.mfa_manager.verify_mfa_token(
            setup_data['encrypted_secret'],
            mfa_token,
            setup_data['backup_codes']
        )
        
        if verification['verified']:
            # Create session
            session_id = self.session_manager.create_session(admin_email)
            
            # If backup code was used, remove it
            if verification['method'] == 'backup_code':
                setup_data['backup_codes'].remove(verification['used_backup_code'])
                with open(setup_file, 'w') as f:
                    json.dump(setup_data, f, indent=2)
                print(f"🔄 Backup code {verification['used_backup_code']} has been used and removed")
            
            print(f"✅ Authentication successful via {verification['method']}")
            
            return {
                'success': True,
                'session_id': session_id,
                'method': verification['method'],
                'expires_in': self.session_timeout * 60  # seconds
            }
        else:
            print("❌ Authentication failed")
            return {
                'success': False,
                'error': 'Invalid MFA token'
            }
    
    def check_admin_session(self, admin_email: str, session_id: str) -> bool:
        """Check if admin session is valid"""
        return self.session_manager.validate_session(admin_email, session_id)
    
    def logout_admin(self, admin_email: str, session_id: str = None):
        """Logout admin (invalidate session)"""
        self.session_manager.invalidate_session(admin_email, session_id)
        print(f"👋 Admin {admin_email} logged out")

# Example usage functions
def demo_mfa_flow():
    """Demonstrate complete MFA flow"""
    print("🚀 DEMONSTRATING MFA FLOW")
    print("=" * 40)
    
    try:
        # Initialize MFA system
        bot_mfa = TradingBotMFA()
        
        # Admin email
        admin_email = "admin@tradingbot.com"
        
        # Step 1: Set up MFA for admin
        print("\n📋 STEP 1: Setting up MFA...")
        setup_data = bot_mfa.setup_admin_mfa(admin_email)
        
        print(f"\n📱 ADMIN SETUP INSTRUCTIONS:")
        print(f"1. Open your authenticator app")
        print(f"2. Add account using this QR code URI:")
        print(f"   {setup_data['qr_uri']}")
        print(f"3. Save these backup codes securely:")
        for i, code in enumerate(setup_data['backup_codes'][:5], 1):
            print(f"   {i}. {code}")
        print(f"   ... and {len(setup_data['backup_codes'])-5} more")
        
        # Step 2: Simulate token generation (in reality, admin gets from app)
        print(f"\n🔢 STEP 2: Simulating authentication...")
        
        # Generate a current token for testing
        secret = bot_mfa.mfa_manager.cipher.decrypt(setup_data['encrypted_secret'].encode()).decode()
        import pyotp
        totp = pyotp.TOTP(secret)
        current_token = totp.now()
        
        print(f"📱 Simulated app token: {current_token}")
        
        # Step 3: Authenticate with token
        print(f"\n🔐 STEP 3: Authenticating admin...")
        auth_result = bot_mfa.authenticate_admin(admin_email, current_token)
        
        if auth_result['success']:
            session_id = auth_result['session_id']
            print(f"✅ Admin authenticated successfully!")
            print(f"🎫 Session ID: {session_id[:20]}...")
            print(f"⏰ Expires in: {auth_result['expires_in']} seconds")
            
            # Step 4: Check session
            print(f"\n✅ STEP 4: Checking session...")
            if bot_mfa.check_admin_session(admin_email, session_id):
                print("✅ Session is valid - admin can access protected features")
            else:
                print("❌ Session is invalid")
            
            # Step 5: Test backup code
            print(f"\n🆘 STEP 5: Testing backup code...")
            backup_code = setup_data['backup_codes'][0]
            backup_auth = bot_mfa.authenticate_admin(admin_email, backup_code)
            
            if backup_auth['success']:
                print(f"✅ Backup code authentication successful!")
                print(f"🔄 Backup code {backup_code} has been consumed")
            
        else:
            print(f"❌ Authentication failed: {auth_result['error']}")
        
        print(f"\n🎉 MFA DEMO COMPLETE!")
        
    except Exception as e:
        print(f"❌ Error in MFA demo: {e}")

def show_integration_points():
    """Show where MFA should be integrated"""
    print("\n🔗 MFA INTEGRATION POINTS")
    print("=" * 40)
    
    print("\n1. 🚪 ADMIN LOGIN:")
    print("   - User enters username/password")
    print("   - System prompts for MFA token")
    print("   - Call: bot_mfa.authenticate_admin(email, token)")
    
    print("\n2. 🔒 PROTECTED ENDPOINTS:")
    print("   - Check session before allowing access")
    print("   - Call: bot_mfa.check_admin_session(email, session_id)")
    
    print("\n3. ⚠️ CRITICAL OPERATIONS:")
    print("   - Modify trading strategy")
    print("   - Change trust beneficiaries")
    print("   - Update API keys")
    print("   - May require fresh MFA token")
    
    print("\n4. 🏛️ TRUST OPERATIONS:")
    print("   - Trustee decisions")
    print("   - Beneficiary changes")
    print("   - Distribution modifications")
    print("   - All require MFA verification")
    
    print("\n5. 📊 DASHBOARD FEATURES:")
    print("   - Admin panel access")
    print("   - Settings modifications")
    print("   - Report generation")
    print("   - API key management")

if __name__ == "__main__":
    print("🔐 MFA INTEGRATION EXAMPLE")
    print("This shows how MFA works in your trading bot")
    
    # Run demonstration
    demo_mfa_flow()
    
    # Show integration points
    show_integration_points()
    
    print("\n✅ MFA is ready for integration into your trading bot!")