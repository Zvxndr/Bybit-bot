#!/usr/bin/env python3
"""
Production Deployment Script for Bybit Trading Bot
Simple version without emojis for Windows compatibility
"""

import os
import sys
import json
import shutil
import secrets
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deployment.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ProductionDeployment:
    """Production deployment automation"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.deployment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check deployment prerequisites"""
        logger.info("Checking deployment prerequisites...")
        
        checks = {
            'python_version': self._check_python_version(),
            'git_available': self._check_command('git --version'),
            'docker_available': self._check_command('docker --version'),
            'node_available': self._check_command('node --version'),
            'project_structure': self._check_project_structure(),
            'requirements_file': self._check_requirements_file()
        }
        
        for check, result in checks.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"  {status} {check.replace('_', ' ').title()}")
            
        return checks
    
    def _check_python_version(self) -> bool:
        """Check Python version >= 3.8"""
        return sys.version_info >= (3, 8)
    
    def _check_command(self, command: str) -> bool:
        """Check if command is available"""
        try:
            subprocess.run(command.split(), capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_project_structure(self) -> bool:
        """Check essential project structure"""
        required_files = ['config.py', '.env.template']
        return all((self.project_root / file).exists() for file in required_files)
    
    def _check_requirements_file(self) -> bool:
        """Check requirements file exists"""
        return (self.project_root / 'requirements.txt').exists()
    
    def setup_environment(self) -> bool:
        """Set up production environment"""
        logger.info("Setting up production environment...")
        
        try:
            # Create .env file from template
            env_template = self.project_root / '.env.template'
            env_file = self.project_root / '.env'
            
            if env_template.exists() and not env_file.exists():
                logger.info("Creating .env file from template...")
                shutil.copy(env_template, env_file)
                
                # Generate secrets
                self._generate_secrets(env_file)
            
            # Create necessary directories
            directories = ['logs', 'data', 'backups', 'temp']
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                logger.info(f"Created directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def _generate_secrets(self, env_file: Path):
        """Generate secure secrets in .env file"""
        logger.info("Generating secure secrets...")
        
        secrets_to_generate = {
            'JWT_SECRET_KEY': secrets.token_urlsafe(32),
            'DATABASE_SECRET_KEY': secrets.token_urlsafe(32),
            'REDIS_PASSWORD': secrets.token_urlsafe(16),
            'WEBHOOK_SECRET': secrets.token_urlsafe(24),
            'ENCRYPTION_KEY': secrets.token_urlsafe(32),
            'SESSION_SECRET_KEY': secrets.token_urlsafe(32),
            'API_SECRET_SALT': secrets.token_urlsafe(16),
            'BACKUP_ENCRYPTION_KEY': secrets.token_urlsafe(32)
        }
        
        # Read current content
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Replace placeholder secrets
        for key, value in secrets_to_generate.items():
            placeholder = f"{key}=changeme_secure_random_string"
            replacement = f"{key}={value}"
            content = content.replace(placeholder, replacement)
        
        # Write updated content
        with open(env_file, 'w') as f:
            f.write(content)
        
        logger.info("Secure secrets generated")
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        try:
            # Install requirements.txt
            requirements_files = ['requirements.txt']
            
            for req_file in requirements_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    logger.info(f"Installing {req_file}...")
                    result = subprocess.run([
                        sys.executable, '-m', 'pip', 'install', '-r', str(req_path)
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        logger.error(f"Failed to install {req_file}: {result.stderr}")
                        return False
                    else:
                        print(result.stdout)  # Show installation output
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False
    
    def setup_database(self) -> bool:
        """Set up database configuration"""
        logger.info("Setting up database configuration...")
        
        try:
            # For now, just create database directory
            db_dir = self.project_root / 'data' / 'database'
            db_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Database configuration completed")
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Set up monitoring and logging"""
        logger.info("Setting up monitoring and logging...")
        
        try:
            # Create monitoring directory structure
            monitoring_dirs = ['logs/app', 'logs/trading', 'logs/system', 'data/metrics']
            for dir_path in monitoring_dirs:
                (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
            
            logger.info("Monitoring setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return False
    
    def create_docker_config(self) -> bool:
        """Create Docker configuration"""
        logger.info("Creating Docker configuration...")
        
        try:
            # Skip Docker setup on Windows for now
            if platform.system() == 'Windows':
                logger.info("Skipping Docker setup on Windows")
                return True
            
            docker_compose = {
                'version': '3.8',
                'services': {
                    'timescaledb': {
                        'image': 'timescale/timescaledb:latest-pg16',
                        'environment': {
                            'POSTGRES_DB': 'trading_bot',
                            'POSTGRES_USER': 'trading_user',
                            'POSTGRES_PASSWORD': '${DATABASE_PASSWORD}'
                        },
                        'ports': ['5432:5432'],
                        'volumes': ['./data/database:/var/lib/postgresql/data'],
                        'restart': 'unless-stopped'
                    },
                    'redis': {
                        'image': 'redis:7-alpine',
                        'command': 'redis-server --requirepass ${REDIS_PASSWORD}',
                        'ports': ['6379:6379'],
                        'volumes': ['./data/redis:/data'],
                        'restart': 'unless-stopped'
                    },
                    'trading-bot': {
                        'build': '.',
                        'depends_on': ['timescaledb', 'redis'],
                        'environment': {
                            'DATABASE_URL': 'postgresql://trading_user:${DATABASE_PASSWORD}@timescaledb:5432/trading_bot',
                            'REDIS_URL': 'redis://:${REDIS_PASSWORD}@redis:6379/0'
                        },
                        'volumes': [
                            './logs:/app/logs',
                            './data:/app/data',
                            './config:/app/config'
                        ],
                        'restart': 'unless-stopped'
                    }
                }
            }
            
            # Write docker-compose.yml
            with open(self.project_root / 'docker-compose.yml', 'w') as f:
                json.dump(docker_compose, f, indent=2)
            
            logger.info("Docker configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Docker configuration failed: {e}")
            return False
    
    def create_systemd_service(self) -> bool:
        """Create systemd service (Linux only)"""
        if platform.system() != 'Linux':
            logger.info("Skipping systemd service creation (not Linux)")
            return True
            
        logger.info("Creating systemd service...")
        
        try:
            service_content = f"""[Unit]
Description=Bybit Trading Bot
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory={self.project_root}
Environment=PATH={self.project_root}/.venv/bin
ExecStart={self.project_root}/.venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
            
            service_file = self.project_root / 'bybit-trading-bot.service'
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            logger.info("Systemd service file created")
            return True
            
        except Exception as e:
            logger.error(f"Systemd service creation failed: {e}")
            return False
    
    def validate_security(self) -> bool:
        """Validate security configuration"""
        logger.info("Validating security configuration...")
        
        try:
            security_checks = []
            
            # Check .env file permissions
            env_file = self.project_root / '.env'
            if env_file.exists():
                # On Windows, we can't easily check file permissions
                if platform.system() != 'Windows':
                    import stat
                    permissions = oct(env_file.stat().st_mode)[-3:]
                    if permissions != '600':
                        security_checks.append(f".env file permissions: {permissions} (should be 600)")
            
            # Check for default values in config
            try:
                from config import Config
                config = Config()
                if not config.is_production_ready():
                    security_checks.append("Configuration contains default/test values")
            except ImportError:
                security_checks.append("Config module not found")
            
            if security_checks:
                logger.warning("Security issues found:")
                for issue in security_checks:
                    logger.warning(f"  - {issue}")
                return False
            
            logger.info("Security validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return False
    
    def deploy(self) -> bool:
        """Execute complete deployment"""
        logger.info("Starting production deployment...")
        logger.info(f"Deployment ID: {self.deployment_id}")
        
        deployment_steps = {
            'prerequisites': lambda: all(self.check_prerequisites().values()),
            'environment_setup': self.setup_environment,
            'dependencies': self.install_dependencies,
            'database': self.setup_database,
            'monitoring': self.setup_monitoring,
            'docker_config': self.create_docker_config,
            'systemd_service': self.create_systemd_service,
            'security_validation': self.validate_security
        }
        
        results = {}
        
        for step_name, step_function in deployment_steps.items():
            logger.info(f"Executing: {step_name}")
            try:
                results[step_name] = step_function()
                status = "SUCCESS" if results[step_name] else "FAILED"
                logger.info(f"{step_name}: {status}")
            except Exception as e:
                logger.error(f"{step_name} failed with exception: {e}")
                results[step_name] = False
        
        # Generate deployment report
        self._generate_deployment_report(results)
        
        success = all(results.values())
        if success:
            logger.info("Deployment completed successfully!")
        else:
            logger.error("Deployment completed with errors")
        
        return success
    
    def _generate_deployment_report(self, results: Dict[str, bool]):
        """Generate deployment report"""
        report = {
            'deployment_id': self.deployment_id,
            'timestamp': datetime.now().isoformat(),
            'platform': platform.system(),
            'python_version': sys.version,
            'project_root': str(self.project_root),
            'results': results,
            'success': all(results.values())
        }
        
        report_file = self.project_root / f'deployment_report_{self.deployment_id}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Deployment report saved to: {report_file}")

def main():
    """Main deployment function"""
    print("Bybit Trading Bot - Production Deployment")
    print("=" * 50)
    
    deployer = ProductionDeployment()
    success = deployer.deploy()
    
    if success:
        print("\nDeployment completed successfully!")
        print("Next steps:")
        print("1. Configure API credentials in .env file")
        print("2. Review configuration in config.py")
        print("3. Test the system with: python -c 'from config import Config; Config().validate()'")
        print("4. Start the trading bot")
    else:
        print("\nDeployment completed with errors. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()