"""
Quick Setup Script for Australian Trust Trading Bot
==================================================

This script helps you configure your environment quickly and safely.
"""

import os
import json
import secrets
import string
from pathlib import Path
from typing import Dict, List

def generate_secure_password(length: int = 32) -> str:
    """Generate a cryptographically secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    return password

def create_env_file() -> Dict[str, str]:
    """Create .env file with user input"""
    
    print("🔧 Australian Trust Trading Bot - Environment Setup")
    print("=" * 60)
    
    env_vars = {}
    
    # DigitalOcean Configuration
    print("\n☁️ DigitalOcean Configuration:")
    print("   Get your API token from: https://cloud.digitalocean.com/account/api/tokens")
    do_token = input("   Enter DigitalOcean API Token: ").strip()
    env_vars['DIGITALOCEAN_TOKEN'] = do_token
    
    # SendGrid Configuration
    print("\n📧 SendGrid Configuration:")
    print("   Get your API key from: https://app.sendgrid.com/settings/api_keys")
    sg_key = input("   Enter SendGrid API Key: ").strip()
    from_email = input("   Enter your FROM email address: ").strip()
    from_name = input("   Enter your FROM name (e.g., 'Australian Trust Bot'): ").strip() or "Australian Trust Bot"
    
    env_vars['SENDGRID_API_KEY'] = sg_key
    env_vars['FROM_EMAIL'] = from_email
    env_vars['FROM_NAME'] = from_name
    
    # Trust Configuration
    print("\n🏛️ Australian Trust Configuration:")
    print("   Enter trustee email addresses (comma-separated)")
    trustee_emails = input("   Trustee emails: ").strip()
    
    print("   Enter beneficiary email addresses (comma-separated, optional)")
    beneficiary_emails = input("   Beneficiary emails (optional): ").strip()
    
    env_vars['TRUSTEE_EMAILS'] = trustee_emails
    env_vars['BENEFICIARY_EMAILS'] = beneficiary_emails
    
    # Security Configuration
    print("\n🔐 Security Configuration:")
    use_generated_password = input("   Generate secure master password automatically? (y/n): ").strip().lower()
    
    if use_generated_password == 'y':
        master_password = generate_secure_password(40)
        print(f"   Generated password: {master_password}")
        print("   ⚠️  SAVE THIS PASSWORD SECURELY - YOU'LL NEED IT!")
    else:
        master_password = input("   Enter master password (20+ characters recommended): ").strip()
    
    env_vars['MASTER_PASSWORD'] = master_password
    env_vars['MFA_ENABLED'] = 'true'
    
    # Trading Configuration (Optional)
    print("\n📈 Trading Configuration (Optional):")
    has_bybit = input("   Do you have Bybit API credentials? (y/n): ").strip().lower()
    
    if has_bybit == 'y':
        bybit_key = input("   Enter Bybit API Key: ").strip()
        bybit_secret = input("   Enter Bybit Secret: ").strip()
        use_testnet = input("   Use testnet for testing? (y/n): ").strip().lower() == 'y'
        
        env_vars['BYBIT_API_KEY'] = bybit_key
        env_vars['BYBIT_SECRET'] = bybit_secret
        env_vars['BYBIT_TESTNET'] = 'true' if use_testnet else 'false'
    else:
        env_vars['BYBIT_API_KEY'] = ''
        env_vars['BYBIT_SECRET'] = ''
        env_vars['BYBIT_TESTNET'] = 'true'
    
    # Default Configuration
    env_vars.update({
        'DATABASE_URL': 'postgresql://user:pass@localhost:5432/trading_bot',
        'REDIS_URL': 'redis://localhost:6379/0',
        'DEBUG': 'false',
        'LOG_LEVEL': 'INFO',
        'TIMEZONE': 'Australia/Sydney'
    })
    
    return env_vars

def create_config_files(env_vars: Dict[str, str]):
    """Create configuration files"""
    
    # Ensure config directory exists
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    # Create production.json
    production_config = {
        "environment": "production",
        "project_name": "australian-trust-bot",
        "deployment": {
            "region": "sgp1",
            "droplet_size": "s-2vcpu-4gb",
            "droplet_count": 2,
            "database_size": "db-s-1vcpu-1gb",
            "volume_size": 50,
            "enable_load_balancer": True,
            "enable_firewall": True,
            "enable_monitoring": True
        },
        "security": {
            "allowed_ips": [],
            "mfa_required": True,
            "session_timeout": 3600,
            "rate_limiting": True
        },
        "notifications": {
            "weekly_reports": True,
            "daily_summaries": True,
            "alert_thresholds": {
                "profit_alert": 5.0,
                "loss_alert": -2.0,
                "risk_alert": 80.0
            }
        },
        "compliance": {
            "australian_trust": True,
            "audit_logging": True,
            "tax_reporting": True
        }
    }
    
    with open('config/production.json', 'w') as f:
        json.dump(production_config, f, indent=2)
    
    print("📁 Created config/production.json")

def write_env_file(env_vars: Dict[str, str]):
    """Write environment variables to .env file"""
    
    env_content = """# Australian Trust Trading Bot - Environment Configuration
# Generated automatically - Keep this file secure!

# DigitalOcean Configuration
DIGITALOCEAN_TOKEN={DIGITALOCEAN_TOKEN}

# SendGrid Configuration  
SENDGRID_API_KEY={SENDGRID_API_KEY}
FROM_EMAIL={FROM_EMAIL}
FROM_NAME={FROM_NAME}

# Australian Trust Configuration
TRUSTEE_EMAILS={TRUSTEE_EMAILS}
BENEFICIARY_EMAILS={BENEFICIARY_EMAILS}

# Security Configuration
MASTER_PASSWORD={MASTER_PASSWORD}
MFA_ENABLED={MFA_ENABLED}

# Trading Configuration
BYBIT_API_KEY={BYBIT_API_KEY}
BYBIT_SECRET={BYBIT_SECRET}  
BYBIT_TESTNET={BYBIT_TESTNET}

# Database Configuration
DATABASE_URL={DATABASE_URL}

# Redis Configuration
REDIS_URL={REDIS_URL}

# Application Configuration
DEBUG={DEBUG}
LOG_LEVEL={LOG_LEVEL}
TIMEZONE={TIMEZONE}
""".format(**env_vars)
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("📄 Created .env file")

def create_test_scripts():
    """Create test scripts for validation"""
    
    # Test email script
    test_email_script = '''import os
from dotenv import load_dotenv
from src.notifications.sendgrid_manager import SendGridEmailManager

load_dotenv()

def test_email():
    try:
        email_manager = SendGridEmailManager(
            api_key=os.getenv('SENDGRID_API_KEY'),
            from_email=os.getenv('FROM_EMAIL'),
            from_name=os.getenv('FROM_NAME')
        )
        
        result = email_manager.test_email_connection()
        
        if result['connected']:
            print("✅ Email configuration successful!")
            print(f"Status Code: {result['status_code']}")
        else:
            print("❌ Email configuration failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
        return result['connected']
        
    except Exception as e:
        print(f"❌ Email test error: {str(e)}")
        return False

if __name__ == "__main__":
    test_email()
'''
    
    with open('test_email.py', 'w') as f:
        f.write(test_email_script)
    
    # Test DigitalOcean script
    test_do_script = '''import os
from dotenv import load_dotenv
from src.infrastructure.digitalocean_manager import DigitalOceanManager

load_dotenv()

def test_digitalocean():
    try:
        do_manager = DigitalOceanManager(os.getenv('DIGITALOCEAN_TOKEN'))
        result = do_manager.validate_connection()
        
        if result['connected']:
            print("✅ DigitalOcean configuration successful!")
            print(f"Account: {result['account_email']}")
            print(f"Droplet Limit: {result['droplet_limit']}")
        else:
            print("❌ DigitalOcean configuration failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
        return result['connected']
        
    except Exception as e:
        print(f"❌ DigitalOcean test error: {str(e)}")
        return False

if __name__ == "__main__":
    test_digitalocean()
'''
    
    with open('test_digitalocean.py', 'w') as f:
        f.write(test_do_script)
    
    print("🧪 Created test scripts: test_email.py, test_digitalocean.py")

def create_requirements_file():
    """Create requirements.txt if it doesn't exist"""
    
    requirements = """# Australian Trust Trading Bot Requirements
python-digitalocean>=1.17.0
sendgrid>=6.10.0
redis>=4.5.0
python-dotenv>=1.0.0
cryptography>=41.0.0
pyotp>=2.8.0
qrcode[pil]>=7.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
schedule>=1.2.0
requests>=2.31.0
flask>=2.3.0
gunicorn>=21.2.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
alembic>=1.11.0
"""
    
    if not Path('requirements.txt').exists():
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("📦 Created requirements.txt")
    else:
        print("📦 requirements.txt already exists")

def create_deployment_script():
    """Create deployment script"""
    
    deploy_script = '''import os
import json
from dotenv import load_dotenv
from src.infrastructure.digitalocean_manager import DigitalOceanManager, InfrastructureConfig, DropletSize, DatabaseEngine

load_dotenv()

def deploy_infrastructure():
    """Deploy complete infrastructure to DigitalOcean"""
    
    print("🚀 Starting Australian Trust Bot deployment...")
    
    # Initialize DigitalOcean manager
    do_manager = DigitalOceanManager(os.getenv('DIGITALOCEAN_TOKEN'))
    
    # Test connection first
    connection = do_manager.validate_connection()
    if not connection['connected']:
        print(f"❌ DigitalOcean connection failed: {connection.get('error')}")
        return False
    
    print(f"✅ Connected to DigitalOcean: {connection['account_email']}")
    
    # Load configuration
    with open('config/production.json', 'r') as f:
        config_data = json.load(f)
    
    # Create infrastructure configuration
    config = InfrastructureConfig(
        project_name=config_data['project_name'],
        region=config_data['deployment']['region'],
        app_droplet_size=DropletSize.MEDIUM,
        app_droplet_count=config_data['deployment']['droplet_count'],
        database_engine=DatabaseEngine.POSTGRESQL,
        database_size=config_data['deployment']['database_size'],
        enable_load_balancer=config_data['deployment']['enable_load_balancer'],
        enable_firewall=config_data['deployment']['enable_firewall'],
        volume_size=config_data['deployment']['volume_size'],
        enable_monitoring=config_data['deployment']['enable_monitoring']
    )
    
    print("📋 Infrastructure Configuration:")
    print(f"   Project: {config.project_name}")
    print(f"   Region: {config.region}")
    print(f"   Droplets: {config.app_droplet_count}x {config.app_droplet_size.value}")
    print(f"   Database: PostgreSQL")
    print(f"   Load Balancer: {'Yes' if config.enable_load_balancer else 'No'}")
    print(f"   Firewall: {'Yes' if config.enable_firewall else 'No'}")
    print(f"   Estimated Cost: $150-200/month")
    
    # Confirm deployment
    confirm = input("\\n❓ Deploy this infrastructure? (yes/no): ")
    if confirm.lower() != 'yes':
        print("⏹️ Deployment cancelled")
        return False
    
    # Deploy infrastructure
    print("\\n🏗️ Deploying infrastructure (this may take 10-15 minutes)...")
    
    results = do_manager.setup_complete_infrastructure(config)
    
    if results['success']:
        print("\\n🎉 Deployment successful!")
        print(f"💰 Estimated monthly cost: ${results['costs']['total_monthly']:.2f}")
        
        # Save deployment info
        with open('deployment_info.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("💾 Deployment info saved to: deployment_info.json")
        
        # Display connection info
        if 'droplets' in results['resources']:
            print("\\n🖥️ Droplet Information:")
            for droplet in results['resources']['droplets']:
                print(f"   • {droplet['name']}: {droplet['ip_address']}")
        
        if 'load_balancer' in results['resources']:
            lb = results['resources']['load_balancer']
            print(f"\\n⚖️ Load Balancer: {lb['ip']}")
            print("   This is your main application URL!")
        
        return True
        
    else:
        print(f"\\n❌ Deployment failed!")
        for error in results['errors']:
            print(f"   • {error}")
        return False

if __name__ == "__main__":
    deploy_infrastructure()
'''
    
    with open('deploy.py', 'w') as f:
        f.write(deploy_script)
    
    print("🚀 Created deploy.py")

def main():
    """Main setup function"""
    
    print("🏛️ Australian Trust Trading Bot - Quick Setup")
    print("=" * 60)
    print("This script will help you configure your trading bot quickly and securely.")
    print("Make sure you have:")
    print("  • DigitalOcean account and API token")
    print("  • SendGrid account and API key")
    print("  • Email addresses for trustees and beneficiaries")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    input()
    
    try:
        # Create environment configuration
        env_vars = create_env_file()
        
        # Write .env file
        write_env_file(env_vars)
        
        # Create config files
        create_config_files(env_vars)
        
        # Create requirements.txt
        create_requirements_file()
        
        # Create test scripts
        create_test_scripts()
        
        # Create deployment script
        create_deployment_script()
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        print("📝 Created logs directory")
        
        print("\n" + "=" * 60)
        print("✅ Setup Complete!")
        print("=" * 60)
        
        print("\n🎯 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Test email config: python test_email.py")
        print("3. Test DigitalOcean: python test_digitalocean.py")
        print("4. Review config files in config/ directory")
        print("5. Deploy to cloud: python deploy.py")
        
        print("\n📁 Files Created:")
        print("  • .env (environment variables)")
        print("  • config/production.json (deployment config)")
        print("  • requirements.txt (Python dependencies)")
        print("  • test_email.py (email testing)")
        print("  • test_digitalocean.py (DigitalOcean testing)")
        print("  • deploy.py (deployment script)")
        
        print("\n⚠️  Important Security Notes:")
        print("  • Keep your .env file secure and never share it")
        print("  • Your master password is used for encryption")
        print("  • API keys should be kept confidential")
        
        print("\n🚀 Ready for deployment!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        print("Please check your inputs and try again")

if __name__ == "__main__":
    main()