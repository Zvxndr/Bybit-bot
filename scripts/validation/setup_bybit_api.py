"""
Bybit API Setup Guide
====================

Script to help users set up their Bybit API credentials for real balance integration.
"""

import os
from pathlib import Path
import yaml

def create_secrets_config():
    """Create a secrets.yaml file with Bybit API credentials"""
    print("🔧 Bybit API Configuration Setup")
    print("=" * 50)
    print()
    print("To fetch your real account balance from Bybit, you need to configure API credentials.")
    print()
    print("📋 Steps to get your Bybit API credentials:")
    print("1. Go to https://testnet.bybit.com (for testnet) or https://www.bybit.com (for mainnet)")
    print("2. Log in to your account")
    print("3. Go to Account & Security > API Management")
    print("4. Create a new API key with 'Read Only' permissions")
    print("5. Copy your API Key and Secret")
    print()
    
    # Get user input
    api_key = input("Enter your Bybit API Key: ").strip()
    if not api_key:
        print("❌ API Key is required. Exiting...")
        return False
    
    api_secret = input("Enter your Bybit API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret is required. Exiting...")
        return False
    
    is_testnet = input("Is this for testnet? (y/n) [default: y]: ").strip().lower()
    testnet = is_testnet != 'n'
    
    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create secrets.yaml
    secrets_config = {
        'bybit': {
            'api_key': api_key,
            'api_secret': api_secret,
            'testnet': testnet
        }
    }
    
    secrets_path = config_dir / "secrets.yaml"
    with open(secrets_path, 'w') as f:
        yaml.dump(secrets_config, f, default_flow_style=False)
    
    # Set file permissions (restrictive)
    try:
        os.chmod(secrets_path, 0o600)
    except:
        pass  # Windows might not support this
    
    print()
    print("✅ Configuration saved!")
    print(f"📁 File: {secrets_path}")
    print(f"🌐 Environment: {'Testnet' if testnet else 'Mainnet'}")
    print()
    print("⚠️  SECURITY NOTES:")
    print("- Never commit secrets.yaml to version control")
    print("- Keep your API credentials secure")
    print("- Use 'Read Only' API permissions for safety")
    print("- The API key only needs balance/position reading permissions")
    print()
    print("🚀 You can now run the trading bot to see your real balance!")
    
    return True

def setup_environment_variables():
    """Alternative setup using environment variables"""
    print()
    print("🔧 Alternative: Environment Variables Setup")
    print("=" * 50)
    print()
    print("You can also set environment variables instead of using the config file:")
    print()
    print("Windows (PowerShell):")
    print('$env:BYBIT_API_KEY="your-api-key-here"')
    print('$env:BYBIT_API_SECRET="your-api-secret-here"')
    print()
    print("Linux/Mac (bash):")
    print('export BYBIT_API_KEY="your-api-key-here"')
    print('export BYBIT_API_SECRET="your-api-secret-here"')
    print()

def main():
    """Main setup function"""
    print()
    print("🤖 Bybit Trading Bot - API Setup")
    print("=" * 40)
    print()
    
    # Check if config already exists
    secrets_path = Path("config/secrets.yaml")
    if secrets_path.exists():
        print("⚠️  Existing configuration found!")
        print(f"📁 {secrets_path}")
        overwrite = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("❌ Aborted. Existing configuration preserved.")
            return
    
    print()
    choice = input("Choose setup method:\n1. Config file (recommended)\n2. Environment variables\n3. Skip setup\n\nChoice (1-3): ").strip()
    
    if choice == '1':
        if create_secrets_config():
            print("✅ Setup complete! You can now run the trading bot.")
        else:
            print("❌ Setup failed.")
    elif choice == '2':
        setup_environment_variables()
    else:
        print("⏭️  Setup skipped. The bot will show 'API credentials not configured' until you set them up.")
    
    print()
    print("📖 For more information, check the README.md file.")

if __name__ == "__main__":
    main()