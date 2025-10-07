"""
Notification Scheduler
====================

Automated scheduling system for email reports and alerts.
Integrates with SendGrid manager to deliver timely communications.
"""

import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class NotificationFrequency(Enum):
    """Notification frequency options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    IMMEDIATE = "immediate"
    CUSTOM = "custom"


@dataclass
class NotificationJob:
    """Notification job configuration"""
    name: str
    frequency: NotificationFrequency
    recipients: List[str]
    callback: Callable
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    custom_schedule: Optional[str] = None  # Cron-like schedule for custom frequency
    data_provider: Optional[Callable] = None  # Function to get data for notifications


class NotificationScheduler:
    """
    Professional notification scheduling system
    
    Handles automated email reports, alerts, and notifications
    with robust scheduling and error handling
    """
    
    def __init__(self, sendgrid_manager):
        """
        Initialize notification scheduler
        
        Args:
            sendgrid_manager: SendGridEmailManager instance
        """
        self.sendgrid_manager = sendgrid_manager
        self.jobs: Dict[str, NotificationJob] = {}
        self.running = False
        self.scheduler_thread = None
        
        # Job execution history
        self.execution_history = []
        self.max_history_size = 1000
        
        logger.info("✅ Notification scheduler initialized")
    
    def add_weekly_report_job(self, name: str, recipients: List[str], 
                             day: str = "sunday", time: str = "18:00",
                             data_provider: Callable = None) -> bool:
        """
        Add weekly report notification job
        
        Args:
            name: Job name
            recipients: List of email recipients
            day: Day of week to send (e.g., 'sunday')
            time: Time to send (HH:MM format)
            data_provider: Function that returns report data
            
        Returns:
            True if job added successfully
        """
        try:
            def report_callback():
                """Weekly report callback"""
                try:
                    # Get report data
                    if data_provider:
                        report_data = data_provider()
                    else:
                        report_data = self._get_default_report_data()
                    
                    # Send weekly report
                    result = self.sendgrid_manager.send_weekly_report(
                        recipients=recipients,
                        report_data=report_data,
                        subject_prefix="[Australian Trust] "
                    )
                    
                    # Log result
                    if result['success']:
                        logger.info(f"📧 Weekly report sent successfully: {result['sent_count']}/{result['total_recipients']}")
                    else:
                        logger.error(f"❌ Weekly report failed: {result.get('error', 'Unknown error')}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ Error in weekly report callback: {str(e)}")
                    return {'success': False, 'error': str(e)}
            
            # Create job
            job = NotificationJob(
                name=name,
                frequency=NotificationFrequency.WEEKLY,
                recipients=recipients,
                callback=report_callback,
                data_provider=data_provider
            )
            
            # Schedule job
            schedule.every().week.at(time).do(report_callback).tag(name)
            
            # Store job
            self.jobs[name] = job
            
            logger.info(f"📅 Weekly report job added: {name} -> {day} at {time}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add weekly report job: {str(e)}")
            return False
    
    def add_daily_summary_job(self, name: str, recipients: List[str],
                             time: str = "20:00", data_provider: Callable = None) -> bool:
        """
        Add daily summary notification job
        
        Args:
            name: Job name
            recipients: List of email recipients
            time: Time to send (HH:MM format)
            data_provider: Function that returns daily data
            
        Returns:
            True if job added successfully
        """
        try:
            def summary_callback():
                """Daily summary callback"""
                try:
                    # Get daily data
                    if data_provider:
                        daily_data = data_provider()
                    else:
                        daily_data = self._get_default_daily_data()
                    
                    # Send daily summary as alert
                    result = self.sendgrid_manager.send_alert(
                        recipients=recipients,
                        alert_type="info",
                        message=f"Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}",
                        data=daily_data
                    )
                    
                    if result['success']:
                        logger.info(f"📊 Daily summary sent successfully")
                    else:
                        logger.error(f"❌ Daily summary failed: {result.get('error', 'Unknown error')}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ Error in daily summary callback: {str(e)}")
                    return {'success': False, 'error': str(e)}
            
            # Create job
            job = NotificationJob(
                name=name,
                frequency=NotificationFrequency.DAILY,
                recipients=recipients,
                callback=summary_callback,
                data_provider=data_provider
            )
            
            # Schedule job
            schedule.every().day.at(time).do(summary_callback).tag(name)
            
            # Store job
            self.jobs[name] = job
            
            logger.info(f"📅 Daily summary job added: {name} at {time}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add daily summary job: {str(e)}")
            return False
    
    def add_alert_job(self, name: str, recipients: List[str], 
                     condition_checker: Callable, alert_type: str = "warning",
                     check_interval: int = 300) -> bool:
        """
        Add conditional alert job that checks periodically
        
        Args:
            name: Job name
            recipients: List of email recipients
            condition_checker: Function that returns (should_alert: bool, message: str, data: dict)
            alert_type: Type of alert (info, warning, error, profit, loss)
            check_interval: Check interval in seconds
            
        Returns:
            True if job added successfully
        """
        try:
            def alert_callback():
                """Alert condition callback"""
                try:
                    # Check condition
                    should_alert, message, data = condition_checker()
                    
                    if should_alert:
                        # Send alert
                        result = self.sendgrid_manager.send_alert(
                            recipients=recipients,
                            alert_type=alert_type,
                            message=message,
                            data=data or {}
                        )
                        
                        if result['success']:
                            logger.info(f"🚨 Alert sent: {alert_type} - {message[:50]}...")
                        else:
                            logger.error(f"❌ Alert failed: {result.get('error', 'Unknown error')}")
                        
                        return result
                    
                    return {'success': True, 'alert_sent': False}
                    
                except Exception as e:
                    logger.error(f"❌ Error in alert callback: {str(e)}")
                    return {'success': False, 'error': str(e)}
            
            # Create job
            job = NotificationJob(
                name=name,
                frequency=NotificationFrequency.CUSTOM,
                recipients=recipients,
                callback=alert_callback,
                custom_schedule=f"every {check_interval} seconds"
            )
            
            # Schedule job
            schedule.every(check_interval).seconds.do(alert_callback).tag(name)
            
            # Store job
            self.jobs[name] = job
            
            logger.info(f"🚨 Alert job added: {name} (check every {check_interval}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add alert job: {str(e)}")
            return False
    
    def add_monthly_report_job(self, name: str, recipients: List[str],
                              day: int = 1, time: str = "09:00",
                              data_provider: Callable = None) -> bool:
        """
        Add monthly report notification job
        
        Args:
            name: Job name
            recipients: List of email recipients
            day: Day of month to send (1-28)
            time: Time to send (HH:MM format)
            data_provider: Function that returns monthly data
            
        Returns:
            True if job added successfully
        """
        try:
            def monthly_callback():
                """Monthly report callback"""
                try:
                    # Get monthly data
                    if data_provider:
                        monthly_data = data_provider()
                    else:
                        monthly_data = self._get_default_monthly_data()
                    
                    # Send monthly report
                    result = self.sendgrid_manager.send_weekly_report(  # Reuse weekly template
                        recipients=recipients,
                        report_data=monthly_data,
                        subject_prefix="[Monthly Report] "
                    )
                    
                    if result['success']:
                        logger.info(f"📈 Monthly report sent successfully")
                    else:
                        logger.error(f"❌ Monthly report failed: {result.get('error', 'Unknown error')}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ Error in monthly report callback: {str(e)}")
                    return {'success': False, 'error': str(e)}
            
            # Create job  
            job = NotificationJob(
                name=name,
                frequency=NotificationFrequency.MONTHLY,
                recipients=recipients,
                callback=monthly_callback,
                data_provider=data_provider
            )
            
            # Schedule job (first day of month)
            schedule.every().month.do(monthly_callback).tag(name)
            
            # Store job
            self.jobs[name] = job
            
            logger.info(f"📅 Monthly report job added: {name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add monthly report job: {str(e)}")
            return False
    
    def start_scheduler(self) -> bool:
        """Start the notification scheduler"""
        try:
            if self.running:
                logger.warning("⚠️ Scheduler already running")
                return False
            
            self.running = True
            
            def run_scheduler():
                """Scheduler thread function"""
                logger.info("🚀 Notification scheduler started")
                
                while self.running:
                    try:
                        schedule.run_pending()
                        time.sleep(30)  # Check every 30 seconds
                        
                    except Exception as e:
                        logger.error(f"❌ Scheduler error: {str(e)}")
                        time.sleep(60)  # Wait longer on error
                
                logger.info("⏹️ Notification scheduler stopped")
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("✅ Notification scheduler thread started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {str(e)}")
            self.running = False
            return False
    
    def stop_scheduler(self) -> bool:
        """Stop the notification scheduler"""
        try:
            self.running = False
            
            # Wait for thread to finish
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=10)
            
            # Clear all scheduled jobs
            schedule.clear()
            
            logger.info("⏹️ Notification scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop scheduler: {str(e)}")
            return False
    
    def enable_job(self, job_name: str) -> bool:
        """Enable a notification job"""
        if job_name in self.jobs:
            self.jobs[job_name].enabled = True
            logger.info(f"✅ Job enabled: {job_name}")
            return True
        
        logger.warning(f"⚠️ Job not found: {job_name}")
        return False
    
    def disable_job(self, job_name: str) -> bool:
        """Disable a notification job"""
        if job_name in self.jobs:
            self.jobs[job_name].enabled = False
            # Cancel scheduled job
            schedule.clear(job_name)
            logger.info(f"⏸️ Job disabled: {job_name}")
            return True
        
        logger.warning(f"⚠️ Job not found: {job_name}")
        return False
    
    def remove_job(self, job_name: str) -> bool:
        """Remove a notification job"""
        try:
            if job_name in self.jobs:
                # Cancel scheduled job
                schedule.clear(job_name)
                # Remove from jobs dict
                del self.jobs[job_name]
                logger.info(f"🗑️ Job removed: {job_name}")
                return True
            
            logger.warning(f"⚠️ Job not found: {job_name}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to remove job {job_name}: {str(e)}")
            return False
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all notification jobs"""
        status = {
            'scheduler_running': self.running,
            'total_jobs': len(self.jobs),
            'enabled_jobs': sum(1 for job in self.jobs.values() if job.enabled),
            'jobs': {}
        }
        
        for name, job in self.jobs.items():
            status['jobs'][name] = {
                'name': job.name,
                'frequency': job.frequency.value,
                'recipients_count': len(job.recipients),
                'enabled': job.enabled,
                'last_run': job.last_run.isoformat() if job.last_run else None,
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'run_count': job.run_count
            }
        
        return status
    
    def run_job_immediately(self, job_name: str) -> Dict[str, Any]:
        """Run a specific job immediately"""
        try:
            if job_name not in self.jobs:
                return {'success': False, 'error': f'Job {job_name} not found'}
            
            job = self.jobs[job_name]
            
            if not job.enabled:
                return {'success': False, 'error': f'Job {job_name} is disabled'}
            
            # Run the job callback
            result = job.callback()
            
            # Update job statistics
            job.last_run = datetime.now()
            job.run_count += 1
            
            # Add to execution history
            self._add_to_history(job_name, result)
            
            logger.info(f"🚀 Job executed manually: {job_name}")
            return {'success': True, 'job_result': result}
            
        except Exception as e:
            logger.error(f"❌ Failed to run job {job_name}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _get_default_report_data(self) -> Dict[str, Any]:
        """Get default report data when no data provider is available"""
        return {
            'week_ending': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_value': 0.0,
            'weekly_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'daily_values': {
                'dates': [],
                'portfolio_values': []
            },
            'strategy_performance': {},
            'active_strategies': {},
            'position_risk': 0.0,
            'daily_loss_used': 0.0
        }
    
    def _get_default_daily_data(self) -> Dict[str, Any]:
        """Get default daily data when no data provider is available"""
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'daily_return': 0.0,
            'trades_count': 0,
            'profit_loss': 0.0,
            'portfolio_value': 0.0
        }
    
    def _get_default_monthly_data(self) -> Dict[str, Any]:
        """Get default monthly data when no data provider is available"""
        default_data = self._get_default_report_data()
        default_data['month_ending'] = datetime.now().strftime('%Y-%m-%d')
        default_data['monthly_return'] = 0.0
        return default_data
    
    def _add_to_history(self, job_name: str, result: Dict[str, Any]):
        """Add job execution to history"""
        history_entry = {
            'job_name': job_name,
            'timestamp': datetime.now().isoformat(),
            'success': result.get('success', False),
            'result': result
        }
        
        self.execution_history.append(history_entry)
        
        # Trim history if too large
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent job execution history"""
        return self.execution_history[-limit:] if limit else self.execution_history
    
    def export_configuration(self, file_path: str) -> bool:
        """Export scheduler configuration to file"""
        try:
            config = {
                'jobs': {},
                'export_time': datetime.now().isoformat()
            }
            
            for name, job in self.jobs.items():
                config['jobs'][name] = {
                    'name': job.name,
                    'frequency': job.frequency.value,
                    'recipients': job.recipients,
                    'enabled': job.enabled,
                    'custom_schedule': job.custom_schedule,
                    'run_count': job.run_count
                }
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"📁 Configuration exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export configuration: {str(e)}")
            return False


# Australian Trust Specific Notification Handlers
class AustralianTrustNotifications:
    """Specialized notification handlers for Australian Discretionary Trust"""
    
    def __init__(self, scheduler: NotificationScheduler):
        self.scheduler = scheduler
        
    def setup_trust_notifications(self, trustee_emails: List[str], 
                                 beneficiaries_emails: List[str] = None) -> bool:
        """
        Set up all Australian Trust specific notifications
        
        Args:
            trustee_emails: Trustee email addresses (receive all notifications)
            beneficiaries_emails: Beneficiary emails (receive reports only)
        """
        try:
            beneficiaries_emails = beneficiaries_emails or []
            all_recipients = trustee_emails + beneficiaries_emails
            
            # 1. Weekly Trust Performance Report (Sundays at 6 PM)
            self.scheduler.add_weekly_report_job(
                name="australian_trust_weekly_report",
                recipients=all_recipients,
                day="sunday",
                time="18:00",
                data_provider=self._get_trust_performance_data
            )
            
            # 2. Daily Risk Monitoring (Daily at 8 PM, trustees only)
            def risk_check():
                # Check if risk limits exceeded
                risk_data = self._get_risk_metrics()
                max_risk = max(risk_data.get('position_risk', 0), 
                              risk_data.get('daily_loss_used', 0))
                
                should_alert = max_risk > 80  # Alert if >80% of limits used
                message = f"Daily risk check - Maximum risk utilization: {max_risk:.1f}%"
                
                return should_alert, message, risk_data
            
            self.scheduler.add_alert_job(
                name="daily_risk_monitoring",
                recipients=trustee_emails,
                condition_checker=risk_check,
                alert_type="warning",
                check_interval=3600  # Check hourly
            )
            
            # 3. Monthly Compliance Report (1st of month at 9 AM)
            self.scheduler.add_monthly_report_job(
                name="monthly_compliance_report",
                recipients=trustee_emails,  # Compliance reports to trustees only
                day=1,
                time="09:00",
                data_provider=self._get_compliance_data
            )
            
            # 4. Profit Alert (Immediate when profit target hit)
            def profit_check():
                performance = self._get_current_performance()
                daily_return = performance.get('daily_return', 0)
                
                should_alert = daily_return > 5.0  # Alert on >5% daily return
                message = f"Significant profit achieved: {daily_return:.2f}% daily return"
                
                return should_alert, message, performance
            
            self.scheduler.add_alert_job(
                name="profit_target_alert",
                recipients=trustee_emails,
                condition_checker=profit_check,
                alert_type="profit",
                check_interval=1800  # Check every 30 minutes
            )
            
            # 5. Loss Protection Alert (Immediate when loss limits approached)
            def loss_check():
                risk_data = self._get_risk_metrics()
                daily_loss = risk_data.get('daily_loss_used', 0)
                
                should_alert = daily_loss > 70  # Alert at 70% of daily loss limit
                message = f"Daily loss limit warning: {daily_loss:.1f}% of 5% limit used"
                
                return should_alert, message, risk_data
            
            self.scheduler.add_alert_job(
                name="loss_protection_alert",
                recipients=trustee_emails,
                condition_checker=loss_check,
                alert_type="loss",
                check_interval=900  # Check every 15 minutes
            )
            
            logger.info("✅ Australian Trust notifications configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup trust notifications: {str(e)}")
            return False
    
    def _get_trust_performance_data(self) -> Dict[str, Any]:
        """Get performance data for trust reporting"""
        # This would integrate with your actual trading bot data
        # For now, return sample structure
        return {
            'week_ending': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_value': 125000.0,  # Would come from actual portfolio
            'weekly_return': 2.34,
            'sharpe_ratio': 1.45,
            'max_drawdown': -3.21,
            'daily_values': {
                'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7, 0, -1)],
                'portfolio_values': [122000, 123500, 124200, 123800, 125000, 124500, 125000]
            },
            'strategy_performance': {
                'BTC Momentum': 45.2,
                'ETH Mean Reversion': 32.1,
                'Arbitrage': 22.7
            },
            'active_strategies': {
                'Bitcoin Momentum': {
                    'status': 'Active',
                    'weekly_return': 3.45,
                    'positions': 2,
                    'win_rate': 67.5
                }
            },
            'position_risk': 7.5,
            'daily_loss_used': 15.2,
            'trust_specific': {
                'beneficiary_count': 5,
                'trust_capital': 100000.0,
                'capital_gains_realized': 25000.0,
                'tax_year': '2024-25'
            }
        }
    
    def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return {
            'position_risk': 7.5,  # % of portfolio at risk
            'daily_loss_used': 15.2,  # % of daily loss limit used
            'max_position_size': 10.0,  # Maximum single position %
            'correlation_risk': 0.65,  # Portfolio correlation
            'var_95': -2.1  # Value at Risk 95%
        }
    
    def _get_compliance_data(self) -> Dict[str, Any]:
        """Get compliance and regulatory data"""
        base_data = self._get_trust_performance_data()
        
        # Add compliance-specific data
        base_data.update({
            'compliance_status': 'Compliant',
            'trust_deed_compliance': True,
            'beneficiary_distributions': {
                'planned': 15000.0,
                'executed': 12000.0,
                'pending': 3000.0
            },
            'tax_obligations': {
                'capital_gains_tax': 2500.0,
                'income_tax': 1800.0,
                'estimated_total': 4300.0
            },
            'audit_trail': {
                'trades_logged': True,
                'decisions_documented': True,
                'beneficiary_notices_sent': True
            }
        })
        
        return base_data
    
    def _get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'daily_return': 2.1,
            'current_portfolio_value': 125000.0,
            'unrealized_pnl': 2500.0,
            'realized_pnl': 1200.0,
            'active_positions': 3
        }


# Example usage
if __name__ == "__main__":
    from sendgrid_manager import SendGridEmailManager
    
    # Initialize with test configuration
    sendgrid_manager = SendGridEmailManager(
        api_key=os.getenv('SENDGRID_API_KEY', 'test_key'),
        from_email="trust@yourdomain.com",
        from_name="Australian Trust Bot"
    )
    
    scheduler = NotificationScheduler(sendgrid_manager)
    trust_notifications = AustralianTrustNotifications(scheduler)
    
    # Setup notifications
    trustee_emails = ["trustee1@example.com", "trustee2@example.com"]
    beneficiary_emails = ["beneficiary1@example.com", "beneficiary2@example.com"]
    
    success = trust_notifications.setup_trust_notifications(
        trustee_emails=trustee_emails,
        beneficiaries_emails=beneficiary_emails
    )
    
    if success:
        print("✅ Trust notifications configured successfully")
        
        # Start scheduler
        scheduler.start_scheduler()
        
        # Get status
        status = scheduler.get_job_status()
        print(f"📊 Scheduler status: {status}")
        
        # Run a test job
        test_result = scheduler.run_job_immediately("australian_trust_weekly_report")
        print(f"🧪 Test result: {test_result}")
        
    else:
        print("❌ Failed to configure trust notifications")