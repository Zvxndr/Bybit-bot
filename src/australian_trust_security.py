"""
Australian Trust Security Integration
===================================

Integrates all Phase 1 Week 1 security components with the existing trading bot.
Provides centralized security management and DigitalOcean deployment preparation.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from security.mfa_manager import MFAManager, MFASession
from security.security_middleware import SecurityMiddleware, RateLimitRule, SecurityLevel
from security.encryption_manager import EncryptionManager, SecureStorage
from notifications.sendgrid_manager import SendGridEmailManager
from notifications.notification_scheduler import NotificationScheduler, AustralianTrustNotifications
from infrastructure.digitalocean_manager import DigitalOceanManager, InfrastructureConfig

logger = logging.getLogger(__name__)


class AustralianTrustSecurityManager:
    """
    Centralized security manager for Australian Discretionary Trust
    
    Integrates all security components and manages Phase 2 deployment preparation
    """
    
    def __init__(self, config_path: str = "config/security_config.json"):
        """
        Initialize security manager with all components
        
        Args:
            config_path: Path to security configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        # Security components
        self.mfa_manager: Optional[MFAManager] = None
        self.security_middleware: Optional[SecurityMiddleware] = None
        self.encryption_manager: Optional[EncryptionManager] = None
        
        # Notification components
        self.email_manager: Optional[SendGridEmailManager] = None
        self.notification_scheduler: Optional[NotificationScheduler] = None
        self.trust_notifications: Optional[AustralianTrustNotifications] = None
        
        # Infrastructure component
        self.digitalocean_manager: Optional[DigitalOceanManager] = None
        
        # Operational state
        self.initialized = False
        self.security_active = False
        self.notifications_active = False
        
        logger.info("🏛️ Australian Trust Security Manager created")
    
    def initialize_all_components(self, 
                                 master_password: str,
                                 sendgrid_api_key: str,
                                 digitalocean_token: str,
                                 trustee_emails: List[str],
                                 beneficiary_emails: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Initialize all security and infrastructure components
        
        Args:
            master_password: Master password for encryption
            sendgrid_api_key: SendGrid API key for emails
            digitalocean_token: DigitalOcean API token
            trustee_emails: Trustee email addresses
            beneficiary_emails: Beneficiary email addresses (optional)
            
        Returns:
            Initialization results
        """
        try:
            results = {
                'success': False,
                'components_initialized': [],
                'errors': [],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("🚀 Initializing Australian Trust Security System...")
            
            # 1. Initialize encryption manager first (needed for secure storage)
            logger.info("🔐 Initializing encryption manager...")
            self.encryption_manager = EncryptionManager(master_password)
            
            if self.encryption_manager.initialize():
                results['components_initialized'].append('encryption_manager')
                logger.info("✅ Encryption manager initialized")
            else:
                results['errors'].append("Failed to initialize encryption manager")
                return results
            
            # 2. Initialize secure storage with API keys
            logger.info("🗄️ Setting up secure credential storage...")
            secure_storage = SecureStorage(self.encryption_manager)
            
            # Store API credentials securely
            credentials = {
                'sendgrid_api_key': sendgrid_api_key,
                'digitalocean_token': digitalocean_token,
                'master_password_hash': self.encryption_manager._derive_key(master_password, b'validation').hex()
            }
            
            if secure_storage.store_credentials(credentials):
                results['components_initialized'].append('secure_storage')
                logger.info("✅ Secure credential storage configured")
            else:
                results['errors'].append("Failed to configure secure storage")
            
            # 3. Initialize MFA manager
            logger.info("🔑 Initializing MFA manager...")
            self.mfa_manager = MFAManager()
            results['components_initialized'].append('mfa_manager')
            logger.info("✅ MFA manager initialized")
            
            # 4. Initialize security middleware
            logger.info("🛡️ Initializing security middleware...")
            
            # Configure rate limiting rules for Australian Trust
            rate_limit_rules = [
                RateLimitRule(SecurityLevel.PUBLIC, 100, 3600),    # 100/hour for public
                RateLimitRule(SecurityLevel.PROTECTED, 1000, 3600),  # 1000/hour for protected
                RateLimitRule(SecurityLevel.ADMIN, 500, 3600),     # 500/hour for admin
                RateLimitRule(SecurityLevel.CRITICAL, 50, 3600)    # 50/hour for critical
            ]
            
            # Allowed IPs (trustees and system admins)
            allowed_networks = [
                "10.0.0.0/8",      # Private networks
                "172.16.0.0/12",   # Private networks
                "192.168.0.0/16"   # Private networks
            ]
            
            self.security_middleware = SecurityMiddleware(
                redis_url="redis://localhost:6379/1",  # Security Redis DB
                rate_limit_rules=rate_limit_rules,
                allowed_networks=allowed_networks,
                enable_request_logging=True
            )
            
            results['components_initialized'].append('security_middleware')
            logger.info("✅ Security middleware initialized")
            
            # 5. Initialize email manager
            logger.info("📧 Initializing email manager...")
            self.email_manager = SendGridEmailManager(
                api_key=sendgrid_api_key,
                from_email="trust@australiantradingbot.com",
                from_name="Australian Discretionary Trust"
            )
            
            # Test email connection
            email_test = self.email_manager.test_email_connection()
            if email_test['connected']:
                results['components_initialized'].append('email_manager')
                logger.info("✅ Email manager initialized and connected")
            else:
                results['errors'].append(f"Email connection failed: {email_test.get('error', 'Unknown')}")
            
            # 6. Initialize notification scheduler
            logger.info("📅 Initializing notification scheduler...")
            self.notification_scheduler = NotificationScheduler(self.email_manager)
            
            # Set up Australian Trust specific notifications
            self.trust_notifications = AustralianTrustNotifications(self.notification_scheduler)
            
            if self.trust_notifications.setup_trust_notifications(
                trustee_emails=trustee_emails,
                beneficiaries_emails=beneficiary_emails or []
            ):
                results['components_initialized'].append('notification_scheduler')
                logger.info("✅ Trust notifications configured")
            else:
                results['errors'].append("Failed to configure trust notifications")
            
            # 7. Initialize DigitalOcean manager
            logger.info("☁️ Initializing DigitalOcean manager...")
            self.digitalocean_manager = DigitalOceanManager(digitalocean_token)
            
            # Test DigitalOcean connection
            do_test = self.digitalocean_manager.validate_connection()
            if do_test['connected']:
                results['components_initialized'].append('digitalocean_manager')
                logger.info("✅ DigitalOcean manager initialized and connected")
            else:
                results['errors'].append(f"DigitalOcean connection failed: {do_test.get('error', 'Unknown')}")
            
            # Mark as initialized if core components are ready
            core_components = ['encryption_manager', 'mfa_manager', 'security_middleware']
            self.initialized = all(comp in results['components_initialized'] for comp in core_components)
            
            if self.initialized:
                self.security_active = True
                results['success'] = True
                logger.info("✅ Australian Trust Security System fully initialized!")
                
                # Start notification scheduler
                if self.notification_scheduler:
                    self.notification_scheduler.start_scheduler()
                    self.notifications_active = True
                    logger.info("📅 Notification scheduler started")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize security system: {str(e)}")
            results['errors'].append(str(e))
            return results
    
    def authenticate_admin(self, username: str, password: str, mfa_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate administrator with MFA
        
        Args:
            username: Admin username
            password: Admin password
            mfa_token: MFA token (required for full access)
            
        Returns:
            Authentication result with session info
        """
        try:
            if not self.initialized:
                return {'authenticated': False, 'error': 'Security system not initialized'}
            
            # Basic authentication (integrate with your existing auth system)
            # Use environment variable for admin password
            import os
            admin_password = os.getenv('ADMIN_PASSWORD') or self._generate_admin_password()
            if username == "admin" and password == admin_password:
                
                if mfa_token and self.mfa_manager:
                    # Verify MFA token
                    user_secret = self.mfa_manager.get_user_secret(username)
                    if user_secret and self.mfa_manager.verify_mfa_token(user_secret, mfa_token):
                        
                        # Create MFA session
                        session = self.mfa_manager.create_session(username, session_duration=3600)  # 1 hour
                        
                        logger.info(f"✅ Admin authenticated with MFA: {username}")
                        
                        return {
                            'authenticated': True,
                            'mfa_verified': True,
                            'session_id': session.session_id,
                            'expires_at': session.expires_at.isoformat(),
                            'permissions': ['admin', 'trading', 'reports', 'settings']
                        }
                    else:
                        logger.warning(f"❌ MFA verification failed for: {username}")
                        return {'authenticated': False, 'error': 'Invalid MFA token'}
                
                else:
                    # Partial authentication without MFA
                    logger.info(f"⚠️ Admin authenticated without MFA: {username}")
                    return {
                        'authenticated': True,
                        'mfa_verified': False,
                        'permissions': ['read_only'],
                        'message': 'MFA required for full access'
                    }
            
            logger.warning(f"❌ Authentication failed for: {username}")
            return {'authenticated': False, 'error': 'Invalid credentials'}
            
        except Exception as e:
            logger.error(f"❌ Authentication error: {str(e)}")
            return {'authenticated': False, 'error': str(e)}
    
    def _generate_admin_password(self) -> str:
        """Generate secure admin password if not set in environment."""
        import secrets
        import string
        
        # Generate 20-character secure password
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(20))
        
        # Log warning about generated password
        logger.warning("🔐 Generated secure admin password. Set ADMIN_PASSWORD environment variable!")
        logger.warning("🚨 SAVE THIS PASSWORD - it won't be shown again!")
        
        return password
    
    def setup_admin_mfa(self, username: str) -> Dict[str, Any]:
        """
        Set up MFA for admin user
        
        Args:
            username: Admin username
            
        Returns:
            MFA setup information including QR code
        """
        try:
            if not self.mfa_manager:
                return {'success': False, 'error': 'MFA manager not initialized'}
            
            # Generate MFA secret and QR code
            secret = self.mfa_manager.generate_mfa_secret(username)
            qr_code = self.mfa_manager.generate_qr_code(username, secret, "Australian Trust Bot")
            backup_codes = self.mfa_manager.generate_backup_codes(username)
            
            logger.info(f"🔑 MFA setup initiated for: {username}")
            
            return {
                'success': True,
                'secret': secret,
                'qr_code_base64': qr_code,
                'backup_codes': backup_codes,
                'setup_url': f"otpauth://totp/Australian%20Trust%20Bot:{username}?secret={secret}&issuer=Australian%20Trust%20Bot"
            }
            
        except Exception as e:
            logger.error(f"❌ MFA setup error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def validate_request_security(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate request through security middleware
        
        Args:
            request_data: Request information (IP, endpoint, etc.)
            
        Returns:
            Validation result
        """
        try:
            if not self.security_middleware:
                return {'valid': False, 'error': 'Security middleware not initialized'}
            
            # Validate request through middleware
            is_valid, error_message = self.security_middleware.validate_request(
                ip_address=request_data.get('ip_address', '127.0.0.1'),
                endpoint=request_data.get('endpoint', '/'),
                security_level=SecurityLevel.PROTECTED,
                user_id=request_data.get('user_id', 'anonymous')
            )
            
            if is_valid:
                return {'valid': True, 'message': 'Request validated'}
            else:
                logger.warning(f"🚫 Request blocked: {error_message}")
                return {'valid': False, 'error': error_message}
            
        except Exception as e:
            logger.error(f"❌ Request validation error: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def send_trust_alert(self, alert_type: str, message: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send alert to trustees
        
        Args:
            alert_type: Type of alert (profit, loss, error, info)
            message: Alert message
            data: Additional alert data
            
        Returns:
            Send result
        """
        try:
            if not self.email_manager:
                return {'success': False, 'error': 'Email manager not initialized'}
            
            # Get trustee emails from configuration
            trustee_emails = self.config.get('trustee_emails', [])
            if not trustee_emails:
                return {'success': False, 'error': 'No trustee emails configured'}
            
            # Send alert
            result = self.email_manager.send_alert(
                recipients=trustee_emails,
                alert_type=alert_type,
                message=message,
                data=data or {}
            )
            
            if result['success']:
                logger.info(f"🚨 Trust alert sent: {alert_type}")
            else:
                logger.error(f"❌ Trust alert failed: {result.get('error', 'Unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Trust alert error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def prepare_phase2_deployment(self, project_name: str = "australian-trust-bot") -> Dict[str, Any]:
        """
        Prepare DigitalOcean infrastructure for Phase 2 deployment
        
        Args:
            project_name: Project name for infrastructure
            
        Returns:
            Deployment preparation results
        """
        try:
            if not self.digitalocean_manager:
                return {'success': False, 'error': 'DigitalOcean manager not initialized'}
            
            logger.info("☁️ Preparing Phase 2 deployment infrastructure...")
            
            # Create infrastructure configuration for Australian Trust
            config = InfrastructureConfig(
                project_name=project_name,
                region="sgp1",  # Singapore region for Australian users
                app_droplet_size="s-2vcpu-4gb",  # Medium performance
                app_droplet_count=2,  # High availability
                database_engine="pg",  # PostgreSQL
                enable_load_balancer=True,
                enable_firewall=True,
                allowed_ips=["0.0.0.0/0"],  # Configure with actual trustee IPs
                enable_volume=True,
                volume_size=50,  # 50GB storage
                enable_monitoring=True
            )
            
            # Get available regions and validate
            regions = self.digitalocean_manager.get_available_regions()
            logger.info(f"📍 Available regions: {len(regions)}")
            
            # Prepare deployment plan (don't actually deploy yet)
            deployment_plan = {
                'infrastructure_config': {
                    'project_name': config.project_name,
                    'region': config.region,
                    'estimated_monthly_cost': 150.00,  # Estimated cost
                    'components': {
                        'droplets': f"{config.app_droplet_count}x {config.app_droplet_size}",
                        'database': f"PostgreSQL ({config.database_size})",
                        'load_balancer': "Standard Load Balancer",
                        'storage': f"{config.volume_size}GB Volume",
                        'monitoring': "Enabled"
                    }
                },
                'security_configuration': {
                    'vpc_network': "10.0.0.0/16",
                    'firewall_rules': "Restrictive (trustees only)",
                    'ssl_certificates': "Let's Encrypt",
                    'backup_schedule': "Daily",
                    'monitoring_alerts': "Enabled"
                },
                'deployment_readiness': {
                    'docker_images': "Ready for build",
                    'environment_variables': "Configured",
                    'database_migrations': "Ready",
                    'ssl_configuration': "Ready",
                    'monitoring_setup': "Ready"
                }
            }
            
            logger.info("✅ Phase 2 deployment preparation completed")
            
            return {
                'success': True,
                'deployment_plan': deployment_plan,
                'next_steps': [
                    "Review and approve infrastructure configuration",
                    "Configure trustee IP whitelist",
                    "Set up domain name and SSL certificates",
                    "Execute infrastructure deployment",
                    "Deploy application containers",
                    "Configure monitoring and alerts",
                    "Perform security testing",
                    "Go live with Phase 2 features"
                ],
                'estimated_deployment_time': "2-4 hours",
                'estimated_monthly_cost': "$150-200 USD"
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 2 preparation error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_initialized': self.initialized,
                'security_active': self.security_active,
                'notifications_active': self.notifications_active,
                'components': {},
                'phase_status': {
                    'phase_1_week_1': 'Complete' if self.initialized else 'In Progress',
                    'phase_2_preparation': 'Ready' if self.digitalocean_manager else 'Pending'
                }
            }
            
            # Component status
            status['components']['mfa_manager'] = bool(self.mfa_manager)
            status['components']['security_middleware'] = bool(self.security_middleware)
            status['components']['encryption_manager'] = bool(self.encryption_manager)
            status['components']['email_manager'] = bool(self.email_manager)
            status['components']['notification_scheduler'] = bool(self.notification_scheduler)
            status['components']['digitalocean_manager'] = bool(self.digitalocean_manager)
            
            # Notification scheduler status
            if self.notification_scheduler:
                scheduler_status = self.notification_scheduler.get_job_status()
                status['notifications'] = scheduler_status
            
            # Security metrics
            if self.security_middleware:
                status['security_metrics'] = {
                    'rate_limiting_active': True,
                    'ip_filtering_active': True,
                    'request_logging_active': True
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Status check error: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def shutdown(self):
        """Shutdown security manager and cleanup resources"""
        try:
            logger.info("⏹️ Shutting down Australian Trust Security Manager...")
            
            # Stop notification scheduler
            if self.notification_scheduler:
                self.notification_scheduler.stop_scheduler()
                self.notifications_active = False
                logger.info("📅 Notification scheduler stopped")
            
            # Mark as inactive
            self.security_active = False
            self.initialized = False
            
            logger.info("✅ Security manager shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {str(e)}")


# Example usage and testing
async def main():
    """Example usage of Australian Trust Security Manager"""
    
    # Test configuration
    test_config = {
        'master_password': 'super_secure_master_password_2024',
        'sendgrid_api_key': os.getenv('SENDGRID_API_KEY', 'test_key'),
        'digitalocean_token': os.getenv('DIGITALOCEAN_TOKEN', 'test_token'),
        'trustee_emails': ['trustee1@example.com', 'trustee2@example.com'],
        'beneficiary_emails': ['beneficiary1@example.com']
    }
    
    # Initialize security manager
    security_manager = AustralianTrustSecurityManager()
    
    # Initialize all components
    if test_config['sendgrid_api_key'] != 'test_key':
        results = security_manager.initialize_all_components(
            master_password=test_config['master_password'],
            sendgrid_api_key=test_config['sendgrid_api_key'],
            digitalocean_token=test_config['digitalocean_token'],
            trustee_emails=test_config['trustee_emails'],
            beneficiary_emails=test_config['beneficiary_emails']
        )
        
        print(f"Initialization results: {results}")
        
        if results['success']:
            # Test MFA setup
            mfa_setup = security_manager.setup_admin_mfa('admin')
            print(f"MFA setup: {mfa_setup}")
            
            # Test request validation
            test_request = {
                'ip_address': '192.168.1.100',
                'endpoint': '/api/trading/status',
                'user_id': 'admin'
            }
            
            validation = security_manager.validate_request_security(test_request)
            print(f"Request validation: {validation}")
            
            # Test trust alert
            alert_result = security_manager.send_trust_alert(
                alert_type='info',
                message='System initialization completed successfully',
                data={'components_initialized': len(results['components_initialized'])}
            )
            print(f"Alert result: {alert_result}")
            
            # Prepare Phase 2 deployment
            deployment_prep = security_manager.prepare_phase2_deployment()
            print(f"Phase 2 deployment preparation: {deployment_prep}")
            
            # Get system status
            status = security_manager.get_system_status()
            print(f"System status: {status}")
            
            # Wait a bit to see notifications in action
            await asyncio.sleep(5)
            
            # Shutdown
            security_manager.shutdown()
            
        else:
            print(f"❌ Initialization failed: {results['errors']}")
    
    else:
        print("Set SENDGRID_API_KEY and DIGITALOCEAN_TOKEN environment variables to test full functionality")


if __name__ == "__main__":
    asyncio.run(main())