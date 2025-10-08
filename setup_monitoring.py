#!/usr/bin/env python3
"""
Trading Bot Monitoring Setup
============================

Setup script for configuring email notifications and monitoring system.
Run this script to configure your email settings and test the monitoring system.
"""

import os
import sys
import getpass
from pathlib import Path

def setup_email_configuration():
    """Interactive setup for email configuration"""
    print("🔧 Trading Bot Monitoring Email Setup")
    print("=" * 50)
    
    # Get email settings from user
    print("\n📧 Email Configuration")
    print("This will set up email notifications for system alerts and daily reports.")
    print("Leave any field empty to use existing environment variable or default.\n")
    
    smtp_username = input("SMTP Username (usually your email): ").strip()
    if smtp_username:
        os.environ['SMTP_USERNAME'] = smtp_username
    
    smtp_password = getpass.getpass("SMTP Password (for Gmail, use App Password): ").strip()
    if smtp_password:
        os.environ['SMTP_PASSWORD'] = smtp_password
    
    from_email = input("From Email (sender address): ").strip()
    if from_email:
        os.environ['FROM_EMAIL'] = from_email
    
    alert_email = input("Alert Email (where to send notifications): ").strip()
    if alert_email:
        os.environ['ALERT_EMAIL'] = alert_email
    
    print("\n✅ Email configuration completed!")
    
    # Create .env file for persistence
    env_file = Path('.env')
    with open(env_file, 'a') as f:
        f.write(f"\n# Email Configuration - Added by setup script\n")
        if smtp_username:
            f.write(f"SMTP_USERNAME={smtp_username}\n")
        if smtp_password:
            f.write(f"SMTP_PASSWORD={smtp_password}\n")
        if from_email:
            f.write(f"FROM_EMAIL={from_email}\n")
        if alert_email:
            f.write(f"ALERT_EMAIL={alert_email}\n")
    
    print(f"📝 Configuration saved to {env_file.absolute()}")
    return True

def test_monitoring_system():
    """Test the monitoring system"""
    print("\n🧪 Testing Monitoring System")
    print("=" * 30)
    
    try:
        # Import monitoring system
        sys.path.append('src')
        from src.monitoring.infrastructure_monitor import create_infrastructure_monitor
        
        monitor = create_infrastructure_monitor()
        print("✅ Monitoring system initialized successfully")
        
        # Test system metrics collection
        import asyncio
        async def test_metrics():
            await monitor._collect_system_metrics()
            print("✅ System metrics collection working")
            
            # Get current metrics
            metrics = monitor.get_current_metrics()
            print(f"📊 Current CPU: {metrics.get('system', {}).get('cpu_percent', 'N/A')}%")
            print(f"📊 Current Memory: {metrics.get('system', {}).get('memory_percent', 'N/A')}%")
            
            # Test email (if configured)
            if os.getenv('SMTP_USERNAME') and os.getenv('SMTP_PASSWORD'):
                print("\n📧 Testing email notification...")
                try:
                    await monitor._send_email(
                        subject="✅ Trading Bot Setup Complete",
                        body="Your trading bot monitoring system is now configured and ready for production use!",
                        email_type="setup_test"
                    )
                    print("✅ Test email sent successfully!")
                except Exception as e:
                    print(f"❌ Email test failed: {e}")
            else:
                print("⚠️ Email not configured - skipping email test")
        
        asyncio.run(test_metrics())
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False
    
    return True

def show_usage_instructions():
    """Show instructions for using the monitoring system"""
    print("\n📚 Usage Instructions")
    print("=" * 20)
    print("""
🚀 Your trading bot monitoring system is now ready!

📊 Monitoring Features:
   • Real-time system metrics (CPU, memory, disk)
   • Automated alerting with email notifications
   • Daily performance reports sent at 9 AM UTC
   • API performance monitoring
   • Trading system health checks

🔧 API Endpoints:
   • GET  /api/monitoring/metrics   - Current system metrics
   • GET  /api/monitoring/alerts    - Alert summary  
   • POST /api/monitoring/start     - Start monitoring
   • POST /api/monitoring/stop      - Stop monitoring
   • GET  /health                   - System health check

📧 Email Notifications:
   • Critical alerts sent immediately
   • Daily reports at 9 AM UTC with system summary
   • Test emails available via API endpoint

🔄 Next Steps:
   1. Start your trading bot: python -m src.main
   2. Access dashboard: http://localhost:8000
   3. Monitoring starts automatically
   4. Check /api/monitoring/metrics for real-time data

⚙️ Configuration:
   • Edit config/monitoring_config.yaml for advanced settings
   • Environment variables in .env file for sensitive data
   • Logs stored in logs/ directory

🆘 Support:
   • Check logs/app.log for detailed system information
   • Use /api/monitoring/send-test-email to verify email setup
   • Review monitoring_config.yaml for threshold adjustments
""")

def main():
    """Main setup function"""
    print("🤖 Trading Bot Monitoring System Setup")
    print("=" * 40)
    
    # Check if running from correct directory
    if not Path('src').exists() or not Path('config').exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    print("This script will help you configure monitoring and email notifications.\n")
    
    # Setup email configuration
    setup_email = input("Configure email notifications? (y/N): ").strip().lower()
    if setup_email in ['y', 'yes']:
        if not setup_email_configuration():
            print("❌ Email setup failed")
            sys.exit(1)
    
    # Test monitoring system
    test_system = input("\nTest monitoring system? (y/N): ").strip().lower()
    if test_system in ['y', 'yes']:
        if not test_monitoring_system():
            print("❌ Monitoring test failed")
            sys.exit(1)
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n🎉 Setup completed successfully!")
    print("Your trading bot is now production-ready with comprehensive monitoring.")

if __name__ == "__main__":
    main()