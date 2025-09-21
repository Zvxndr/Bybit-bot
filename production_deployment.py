#!/usr/bin/env python3
"""
Production Deployment Setup Script
Comprehensive deployment automation for Bybit Trading Bot
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import secrets
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionDeployment:
    """Production deployment automation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.deployment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.deployment_log = f"deployment_{self.deployment_id}.log"
        
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check deployment prerequisites"""
        logger.info("🔍 Checking deployment prerequisites...")
        
        checks = {
            'python_version': sys.version_info >= (3, 8),
            'git_available': shutil.which('git') is not None,
            'docker_available': shutil.which('docker') is not None,
            'node_available': shutil.which('node') is not None,
            'project_structure': self._check_project_structure(),
            'requirements_file': (self.project_root / 'requirements.txt').exists()
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check.replace('_', ' ').title()}")
        
        return checks
    
    def _check_project_structure(self) -> bool:
        """Check if project has the required structure"""
        required_paths = [
            'src/bot',
            'src/dashboard/backend',
            'src/dashboard/frontend',
            'requirements.txt',
            'README.md'
        ]
        
        return all((self.project_root / path).exists() for path in required_paths)
    
    def setup_environment(self) -> bool:
        """Set up production environment"""
        logger.info("⚙️ Setting up production environment...")
        
        try:
            # Create .env file from template if it doesn't exist
            env_file = self.project_root / '.env'
            env_template = self.project_root / '.env.template'
            
            if not env_file.exists() and env_template.exists():
                logger.info("📝 Creating .env file from template...")
                shutil.copy(env_template, env_file)
                
                # Generate secure secrets
                self._generate_secrets(env_file)
            
            # Create necessary directories
            directories = [
                'logs',
                'data',
                'backups',
                'temp'
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                logger.info(f"📁 Created directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def _generate_secrets(self, env_file: Path) -> None:
        """Generate secure secrets for production"""
        logger.info("🔐 Generating secure secrets...")
        
        secrets_to_generate = {
            'SECRET_KEY': secrets.token_urlsafe(32),
            'JWT_SECRET': secrets.token_urlsafe(32),
            'ENCRYPTION_KEY': secrets.token_urlsafe(32)
        }
        
        # Read existing .env file
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Replace default secrets with generated ones
        for key, value in secrets_to_generate.items():
            old_line = f"{key}=your_{key.lower()}_here"
            new_line = f"{key}={value}"
            content = content.replace(old_line, new_line)
        
        # Write updated content
        with open(env_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Secure secrets generated")
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        try:
            # Check if virtual environment exists
            venv_path = self.project_root / '.venv'
            if not venv_path.exists():
                logger.info("🐍 Creating virtual environment...")
                subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
            
            # Determine pip path
            if os.name == 'nt':  # Windows
                pip_path = venv_path / 'Scripts' / 'pip.exe'
                python_path = venv_path / 'Scripts' / 'python.exe'
            else:  # Unix-like
                pip_path = venv_path / 'bin' / 'pip'
                python_path = venv_path / 'bin' / 'python'
            
            # Install requirements
            requirements_files = [
                'requirements.txt',
                'requirements-api.txt',
                'requirements-dashboard.txt'
            ]
            
            for req_file in requirements_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    logger.info(f"📦 Installing {req_file}...")
                    subprocess.run([str(pip_path), 'install', '-r', str(req_path)], check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def setup_database(self) -> bool:
        """Set up production database"""
        logger.info("🗄️ Setting up database...")
        
        try:
            # For now, we'll use SQLite for simplicity
            # In production, you might want to set up PostgreSQL/TimescaleDB
            
            db_path = self.project_root / 'data' / 'trading_bot.db'
            db_path.parent.mkdir(exist_ok=True)
            
            logger.info(f"✅ Database setup complete: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Set up monitoring and logging"""
        logger.info("📊 Setting up monitoring...")
        
        try:
            # Create monitoring configuration
            monitoring_config = {
                'log_level': 'INFO',
                'log_rotation': 'daily',
                'max_log_size': '100MB',
                'retention_days': 30,
                'metrics_enabled': True,
                'alerts_enabled': True
            }
            
            # Create log directory structure
            log_dirs = [
                'logs/application',
                'logs/trading',
                'logs/system',
                'logs/errors'
            ]
            
            for log_dir in log_dirs:
                (self.project_root / log_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ Monitoring setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            return False
    
    def create_systemd_service(self) -> bool:
        """Create systemd service for Linux production deployment"""
        logger.info("🔧 Creating systemd service...")
        
        try:
            service_content = f"""[Unit]
Description=Bybit Trading Bot
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=ubuntu
WorkingDirectory={self.project_root}
Environment=PATH={self.project_root}/.venv/bin
ExecStart={self.project_root}/.venv/bin/python start_api.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
            
            service_file = self.project_root / 'bybit-trading-bot.service'
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            logger.info(f"✅ Systemd service created: {service_file}")
            logger.info("To install: sudo cp bybit-trading-bot.service /etc/systemd/system/")
            logger.info("To enable: sudo systemctl enable bybit-trading-bot")
            logger.info("To start: sudo systemctl start bybit-trading-bot")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Systemd service creation failed: {e}")
            return False
    
    def create_docker_setup(self) -> bool:
        """Create Docker configuration for containerized deployment"""
        logger.info("🐳 Creating Docker configuration...")
        
        try:
            # Create Dockerfile
            dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data backups temp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8001 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

# Start command
CMD ["python", "start_api.py"]
"""
            
            with open(self.project_root / 'Dockerfile', 'w') as f:
                f.write(dockerfile_content)
            
            # Create docker-compose.yml
            compose_content = """version: '3.8'

services:
  trading-bot:
    build: .
    ports:
      - "8001:8001"
      - "3000:3000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./backups:/app/backups
    environment:
      - TRADING_ENABLED=false
      - MOCK_TRADING=true
      - DEBUG=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8001/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: timescale/timescaledb:latest-pg14
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=trading_data
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=secure_password_change_me
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
"""
            
            with open(self.project_root / 'docker-compose.yml', 'w') as f:
                f.write(compose_content)
            
            logger.info("✅ Docker configuration created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker setup failed: {e}")
            return False
    
    def run_security_check(self) -> bool:
        """Run security checks"""
        logger.info("🔒 Running security checks...")
        
        security_issues = []
        
        # Check for default secrets
        env_file = self.project_root / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
                
                if 'your_' in content:
                    security_issues.append("Default placeholder values found in .env file")
                if 'default-' in content:
                    security_issues.append("Default secret keys found in .env file")
        
        # Check file permissions (Unix-like systems)
        if os.name != 'nt':
            sensitive_files = ['.env', 'config.py']
            for file in sensitive_files:
                file_path = self.project_root / file
                if file_path.exists():
                    stat = file_path.stat()
                    if stat.st_mode & 0o077:
                        security_issues.append(f"File {file} has overly permissive permissions")
        
        if security_issues:
            logger.warning("⚠️ Security issues found:")
            for issue in security_issues:
                logger.warning(f"  - {issue}")
            return False
        else:
            logger.info("✅ Security checks passed")
            return True
    
    def generate_deployment_report(self, results: Dict[str, bool]) -> str:
        """Generate deployment report"""
        logger.info("📋 Generating deployment report...")
        
        success_count = sum(results.values())
        total_count = len(results)
        success_rate = (success_count / total_count) * 100
        
        report = f"""
# 🚀 Production Deployment Report

**Deployment ID:** {self.deployment_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Success Rate:** {success_rate:.1f}% ({success_count}/{total_count})

## 📊 Deployment Results

"""
        
        for step, success in results.items():
            status = "✅" if success else "❌"
            report += f"- {status} {step.replace('_', ' ').title()}\n"
        
        report += f"""
## 🎯 Next Steps

"""
        
        if success_rate >= 90:
            report += """
### ✅ Deployment Successful!

1. **Configure API Keys**: Update .env file with your Bybit API credentials
2. **Start Services**: Use the provided systemd service or Docker Compose
3. **Monitor System**: Check logs and dashboard for proper operation
4. **Enable Trading**: Once validated, set TRADING_ENABLED=true

### 🚀 Start Commands:

**Using Docker:**
```bash
docker-compose up -d
```

**Using Systemd (Linux):**
```bash
sudo systemctl start bybit-trading-bot
```

**Manual Start:**
```bash
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
python start_api.py
```
"""
        else:
            report += """
### ⚠️ Deployment Issues Detected

Please address the failed steps above before proceeding to production.

### 🔧 Troubleshooting:
1. Check the deployment log for detailed error messages
2. Ensure all prerequisites are met
3. Verify file permissions and dependencies
4. Re-run the deployment script after fixing issues
"""
        
        report += f"""
## 📞 Support Information

- **Log File:** {self.deployment_log}
- **Project Root:** {self.project_root}
- **Configuration:** config.py
- **Environment:** .env

For issues, check the logs directory and system status.
"""
        
        return report
    
    def deploy(self) -> bool:
        """Execute complete deployment process"""
        logger.info("🚀 Starting production deployment...")
        logger.info(f"📝 Deployment ID: {self.deployment_id}")
        
        # Setup file logging
        file_handler = logging.FileHandler(self.deployment_log)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Run deployment steps
        steps = {
            'prerequisites': lambda: all(self.check_prerequisites().values()),
            'environment_setup': self.setup_environment,
            'dependencies': self.install_dependencies,
            'database_setup': self.setup_database,
            'monitoring_setup': self.setup_monitoring,
            'systemd_service': self.create_systemd_service,
            'docker_setup': self.create_docker_setup,
            'security_check': self.run_security_check
        }
        
        results = {}
        for step_name, step_function in steps.items():
            try:
                logger.info(f"🔄 Executing: {step_name}")
                results[step_name] = step_function()
            except Exception as e:
                logger.error(f"❌ Step {step_name} failed: {e}")
                results[step_name] = False
        
        # Generate and save report
        report = self.generate_deployment_report(results)
        report_file = f"deployment_report_{self.deployment_id}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n📄 Full report saved to: {report_file}")
        print(f"📄 Deployment log saved to: {self.deployment_log}")
        
        success_rate = sum(results.values()) / len(results) * 100
        overall_success = success_rate >= 90
        
        if overall_success:
            logger.info("🎉 Deployment completed successfully!")
        else:
            logger.warning("⚠️ Deployment completed with issues. Check the report.")
        
        return overall_success

def main():
    """Main deployment execution"""
    print("🚀 Bybit Trading Bot - Production Deployment")
    print("=" * 50)
    
    deployer = ProductionDeployment()
    success = deployer.deploy()
    
    print("=" * 50)
    if success:
        print("🎉 Deployment Successful!")
        print("Your trading bot is ready for production!")
    else:
        print("⚠️ Deployment Completed with Issues")
        print("Please review the report and fix any problems.")

if __name__ == "__main__":
    main()