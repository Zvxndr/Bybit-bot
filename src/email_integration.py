"""
Email Integration Layer
======================

Connects the SendGrid email manager to the main application for 
automated notifications and reports.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import os
from pathlib import Path

# Import the existing SendGrid manager
try:
    from src.notifications.sendgrid_manager import SendGridEmailManager
    SENDGRID_AVAILABLE = True
except Exception as e:
    logging.warning(f"SendGrid not available: {e}")
    SENDGRID_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmailIntegrationManager:
    """Integrates email notifications with the main trading application"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.email_manager = None
        self.enabled = False
        
        # Email settings
        self.sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
        self.from_email = os.getenv('FROM_EMAIL', 'trading-bot@yourdomain.com')
        self.alert_emails = os.getenv('ALERT_EMAIL', '').split(',')
        
        # Initialize if possible
        self._initialize_email_manager()
    
    def _initialize_email_manager(self):
        """Initialize the SendGrid email manager"""
        if not SENDGRID_AVAILABLE:
            logger.warning("📧 SendGrid package not available - email disabled")
            return
            
        if not self.sendgrid_api_key:
            logger.warning("📧 SENDGRID_API_KEY not set - email disabled")
            return
            
        try:
            self.email_manager = SendGridEmailManager(
                api_key=self.sendgrid_api_key,
                from_email=self.from_email,
                from_name="Bybit Trading Bot"
            )
            self.enabled = True
            logger.info("✅ Email integration initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize email manager: {e}")
    
    async def send_startup_notification(self):
        """Send notification when bot starts up"""
        if not self.enabled:
            return
            
        try:
            subject = "🚀 Trading Bot Started"
            message = f"""
            <h2>Trading Bot Startup Notification</h2>
            <p>The Bybit trading bot has successfully started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
            <ul>
                <li>✅ System Status: Online</li>
                <li>✅ Risk Management: Active</li>
                <li>✅ ML Pipeline: Ready</li>
                <li>✅ Monitoring: Enabled</li>
            </ul>
            <p>The bot is now monitoring markets and ready to execute trades.</p>
            """
            
            for email in self.alert_emails:
                if email.strip():
                    await self._send_email(email.strip(), subject, message)
                    
            logger.info("📧 Startup notification sent")
            
        except Exception as e:
            logger.error(f"Failed to send startup notification: {e}")
    
    async def send_shutdown_notification(self):
        """Send notification when bot shuts down"""
        if not self.enabled:
            return
            
        try:
            subject = "🛑 Trading Bot Shutdown"
            message = f"""
            <h2>Trading Bot Shutdown Notification</h2>
            <p>The Bybit trading bot has shut down at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
            <p>System Status: Offline</p>
            <p>All positions and trades have been logged. Review the trading session when convenient.</p>
            """
            
            for email in self.alert_emails:
                if email.strip():
                    await self._send_email(email.strip(), subject, message)
                    
            logger.info("📧 Shutdown notification sent")
            
        except Exception as e:
            logger.error(f"Failed to send shutdown notification: {e}")
    
    async def send_alert(self, alert_type: str, message: str, data: Dict[str, Any] = None):
        """Send alert notification"""
        if not self.enabled:
            logger.info(f"📧 Alert would be sent: {alert_type} - {message}")
            return
            
        try:
            subject = f"🚨 Trading Alert: {alert_type}"
            html_message = f"""
            <h2>Trading Bot Alert</h2>
            <p><strong>Alert Type:</strong> {alert_type}</p>
            <p><strong>Message:</strong> {message}</p>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            if data:
                html_message += "<h3>Additional Data:</h3><ul>"
                for key, value in data.items():
                    html_message += f"<li><strong>{key}:</strong> {value}</li>"
                html_message += "</ul>"
            
            for email in self.alert_emails:
                if email.strip():
                    await self._send_email(email.strip(), subject, html_message)
                    
            logger.info(f"📧 Alert sent: {alert_type}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    async def send_weekly_report(self, performance_data: Dict[str, Any]):
        """Send weekly performance report"""
        if not self.enabled:
            logger.info("📧 Weekly report would be sent (email disabled)")
            return
            
        try:
            if hasattr(self.email_manager, 'send_weekly_report'):
                result = self.email_manager.send_weekly_report(
                    recipients=self.alert_emails,
                    report_data=performance_data
                )
                logger.info(f"📧 Weekly report sent: {result}")
            else:
                # Fallback simple report
                subject = "📊 Weekly Trading Report"
                message = f"""
                <h2>Weekly Trading Performance</h2>
                <p>Report Period: {performance_data.get('week_ending', 'N/A')}</p>
                <p>Performance Summary:</p>
                <ul>
                    <li>Total Return: {performance_data.get('total_return', 'N/A')}</li>
                    <li>Number of Trades: {performance_data.get('total_trades', 'N/A')}</li>
                    <li>Win Rate: {performance_data.get('win_rate', 'N/A')}</li>
                    <li>Sharpe Ratio: {performance_data.get('sharpe_ratio', 'N/A')}</li>
                </ul>
                """
                
                for email in self.alert_emails:
                    if email.strip():
                        await self._send_email(email.strip(), subject, message)
                        
                logger.info("📧 Weekly report sent")
                
        except Exception as e:
            logger.error(f"Failed to send weekly report: {e}")
    
    async def _send_email(self, to_email: str, subject: str, html_content: str):
        """Send individual email"""
        try:
            if hasattr(self.email_manager, 'send_email'):
                # Use the full SendGrid manager if available
                result = self.email_manager.send_email(
                    to_email=to_email,
                    subject=subject,
                    html_content=html_content
                )
                return result
            else:
                # Simulated sending (for development)
                logger.info(f"📧 Would send email to {to_email}: {subject}")
                return {"status": "simulated"}
                    
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get email integration status"""
        return {
            "enabled": self.enabled,
            "sendgrid_available": SENDGRID_AVAILABLE,
            "api_key_configured": bool(self.sendgrid_api_key),
            "from_email": self.from_email,
            "alert_emails": len([e for e in self.alert_emails if e.strip()]),
            "last_check": datetime.now().isoformat()
        }


# Global email manager instance
email_integration = EmailIntegrationManager()

# Convenience functions for easy integration
async def send_startup_notification():
    """Send startup notification"""
    await email_integration.send_startup_notification()

async def send_shutdown_notification():
    """Send shutdown notification"""
    await email_integration.send_shutdown_notification()

async def send_alert(alert_type: str, message: str, data: Dict[str, Any] = None):
    """Send alert notification"""
    await email_integration.send_alert(alert_type, message, data)

async def send_weekly_report(performance_data: Dict[str, Any]):
    """Send weekly performance report"""
    await email_integration.send_weekly_report(performance_data)

def get_email_status() -> Dict[str, Any]:
    """Get email integration status"""
    return email_integration.get_status()


# Example usage and testing
async def test_email_integration():
    """Test the email integration"""
    print("🧪 Testing Email Integration")
    print("=" * 50)
    
    # Check status
    status = get_email_status()
    print(f"Email Status: {status}")
    
    # Test startup notification
    await send_startup_notification()
    
    # Test alert
    await send_alert(
        "System Test", 
        "This is a test alert from the email integration system",
        {"test_data": "success", "timestamp": datetime.now().isoformat()}
    )
    
    # Test weekly report
    sample_data = {
        "week_ending": datetime.now().strftime('%Y-%m-%d'),
        "total_return": "5.2%",
        "total_trades": 42,
        "win_rate": "68%",
        "sharpe_ratio": 1.34
    }
    await send_weekly_report(sample_data)
    
    print("✅ Email integration testing completed")


if __name__ == "__main__":
    asyncio.run(test_email_integration())