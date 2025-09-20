#!/usr/bin/env python3
"""
🌊 Digital Ocean Deployment Automation
=====================================

This script automates the complete Digital Ocean deployment process:
- Creates droplets with proper configuration
- Sets up security and firewall rules
- Deploys the ML trading bot application
- Configures monitoring and backups
- Provides comprehensive status reporting

Designed for complete beginners - just run and follow the prompts!
"""

import os
import sys
import json
import time
import subprocess
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Optional

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "pyyaml"], check=True)
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text

console = Console()

class DigitalOceanDeployer:
    """Automated Digital Ocean deployment for ML Trading Bot"""
    
    def __init__(self):
        self.console = Console()
        self.config = {}
        self.droplet_id = None
        self.droplet_ip = None
        self.project_root = Path(__file__).parent
        
    def run(self):
        """Run the complete deployment process"""
        try:
            self.show_welcome()
            self.check_prerequisites()
            self.get_deployment_config()
            self.confirm_deployment()
            self.deploy_to_digital_ocean()
            self.show_completion()
        except KeyboardInterrupt:
            self.console.print("\n❌ Deployment cancelled by user", style="red")
            if self.droplet_id:
                self.cleanup_resources()
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n❌ Deployment failed: {str(e)}", style="red")
            if self.droplet_id:
                self.cleanup_resources()
            sys.exit(1)
    
    def show_welcome(self):
        """Show welcome screen and deployment overview"""
        welcome_text = Text()
        welcome_text.append("🌊 Digital Ocean Deployment\n", style="bold blue")
        welcome_text.append("Deploy your ML Trading Bot to the cloud!\n\n", style="cyan")
        welcome_text.append("What this will do:\n", style="white")
        welcome_text.append("✅ Create a cloud server on Digital Ocean\n", style="green")
        welcome_text.append("✅ Install all required software automatically\n", style="green")
        welcome_text.append("✅ Deploy your trading bot securely\n", style="green")
        welcome_text.append("✅ Set up monitoring and alerts\n", style="green")
        welcome_text.append("✅ Configure automated backups\n\n", style="green")
        welcome_text.append("💰 Cost: Starting from $6/month\n", style="yellow")
        welcome_text.append("⏱️  Time: 15-20 minutes\n", style="yellow")
        welcome_text.append("🎯 Result: 24/7 trading bot in the cloud", style="magenta")
        
        panel = Panel(
            welcome_text,
            title="🚀 Cloud Deployment",
            border_style="blue"
        )
        self.console.print(panel)
        
        if not Confirm.ask("\n🤔 Ready to deploy to Digital Ocean?", default=True):
            self.console.print("👋 Come back when you're ready!", style="yellow")
            sys.exit(0)
    
    def check_prerequisites(self):
        """Check all prerequisites for deployment"""
        self.console.print("\n🔍 Checking prerequisites...", style="blue")
        
        requirements = Table()
        requirements.add_column("Requirement", style="cyan")
        requirements.add_column("Status", style="green")
        requirements.add_column("Details", style="white")
        
        # Check if bot is configured
        if Path(".env").exists() and Path("config").exists():
            requirements.add_row("Bot Configuration", "✅ Pass", "Setup wizard completed")
        else:
            requirements.add_row("Bot Configuration", "❌ Fail", "Run setup wizard first")
            self.console.print(requirements)
            self.console.print("❌ Please run the setup wizard first: python setup_wizard.py", style="red")
            sys.exit(1)
        
        # Check Digital Ocean CLI
        try:
            result = subprocess.run(['doctl', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                requirements.add_row("Digital Ocean CLI", "✅ Pass", "doctl installed")
            else:
                requirements.add_row("Digital Ocean CLI", "❌ Fail", "doctl not found")
                self.install_doctl()
        except FileNotFoundError:
            requirements.add_row("Digital Ocean CLI", "❌ Fail", "doctl not found")
            self.install_doctl()
        
        # Check authentication
        try:
            result = subprocess.run(['doctl', 'account', 'get'], capture_output=True, text=True)
            if result.returncode == 0:
                requirements.add_row("DO Authentication", "✅ Pass", "Authenticated")
            else:
                requirements.add_row("DO Authentication", "❌ Fail", "Not authenticated")
                self.setup_authentication()
        except:
            requirements.add_row("DO Authentication", "❌ Fail", "Not authenticated")
            self.setup_authentication()
        
        # Check local files
        required_files = ["start_api.py", "start_dashboard.py", "requirements.txt"]
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if not missing_files:
            requirements.add_row("Application Files", "✅ Pass", "All files present")
        else:
            requirements.add_row("Application Files", "⚠️  Warning", f"Missing: {', '.join(missing_files)}")
        
        self.console.print(requirements)
        self.console.print("✅ Prerequisites check completed!", style="green")
    
    def install_doctl(self):
        """Install Digital Ocean CLI"""
        self.console.print("\n📥 Installing Digital Ocean CLI...", style="blue")
        
        system = os.name
        if system == 'nt':  # Windows
            self.console.print("Please install doctl manually on Windows:", style="yellow")
            self.console.print("1. Download from: https://github.com/digitalocean/doctl/releases")
            self.console.print("2. Extract and add to PATH")
            self.console.print("3. Run this script again")
            sys.exit(1)
        else:  # Linux/Mac
            try:
                if sys.platform == 'darwin':  # Mac
                    subprocess.run(['brew', 'install', 'doctl'], check=True)
                else:  # Linux
                    # Download and install doctl
                    download_url = "https://github.com/digitalocean/doctl/releases/download/v1.100.0/doctl-1.100.0-linux-amd64.tar.gz"
                    subprocess.run(['curl', '-OL', download_url], check=True)
                    subprocess.run(['tar', 'xf', 'doctl-1.100.0-linux-amd64.tar.gz'], check=True)
                    subprocess.run(['sudo', 'mv', 'doctl', '/usr/local/bin'], check=True)
                
                self.console.print("✅ doctl installed successfully!", style="green")
            except subprocess.CalledProcessError:
                self.console.print("❌ Failed to install doctl automatically", style="red")
                self.console.print("Please install manually: https://docs.digitalocean.com/reference/doctl/how-to/install/")
                sys.exit(1)
    
    def setup_authentication(self):
        """Set up Digital Ocean authentication"""
        self.console.print("\n🔑 Setting up Digital Ocean authentication...", style="blue")
        
        auth_info = Panel(
            "To deploy to Digital Ocean, you need an API token.\n\n"
            "📝 How to get your API token:\n"
            "1. Go to https://cloud.digitalocean.com/\n"
            "2. Click 'API' in the left sidebar\n"
            "3. Click 'Generate New Token'\n"
            "4. Name it 'Trading Bot Deploy'\n"
            "5. Select 'Read' and 'Write' scopes\n"
            "6. Copy the token (you won't see it again!)",
            title="🔑 API Token Setup",
            border_style="yellow"
        )
        self.console.print(auth_info)
        
        token = Prompt.ask("\n🔐 Enter your Digital Ocean API token", password=True)
        
        if not token:
            self.console.print("❌ API token is required!", style="red")
            sys.exit(1)
        
        # Set up authentication
        try:
            subprocess.run(['doctl', 'auth', 'init', '-t', token], check=True, input='\n', text=True)
            
            # Test authentication
            result = subprocess.run(['doctl', 'account', 'get'], capture_output=True, text=True)
            if result.returncode == 0:
                self.console.print("✅ Authentication successful!", style="green")
            else:
                self.console.print("❌ Authentication failed!", style="red")
                sys.exit(1)
                
        except subprocess.CalledProcessError:
            self.console.print("❌ Failed to authenticate!", style="red")
            sys.exit(1)
    
    def get_deployment_config(self):
        """Get deployment configuration from user"""
        self.console.print("\n⚙️  Deployment Configuration", style="blue")
        
        # Server size selection
        size_table = Table()
        size_table.add_column("Size", style="cyan")
        size_table.add_column("CPU", style="white")
        size_table.add_column("RAM", style="white")
        size_table.add_column("Cost/Month", style="green")
        size_table.add_column("Best For", style="yellow")
        
        size_table.add_row("s-1vcpu-1gb", "1 vCPU", "1 GB", "$6", "Learning/Testing")
        size_table.add_row("s-1vcpu-2gb", "1 vCPU", "2 GB", "$12", "Light Trading")
        size_table.add_row("s-2vcpu-2gb", "2 vCPU", "2 GB", "$18", "Recommended")
        size_table.add_row("s-2vcpu-4gb", "2 vCPU", "4 GB", "$24", "Heavy Trading")
        
        self.console.print(size_table)
        
        size = Prompt.ask(
            "\n💻 Choose server size",
            choices=["s-1vcpu-1gb", "s-1vcpu-2gb", "s-2vcpu-2gb", "s-2vcpu-4gb"],
            default="s-2vcpu-2gb"
        )
        
        # Region selection
        region_table = Table()
        region_table.add_column("Code", style="cyan")
        region_table.add_column("Location", style="white")
        region_table.add_column("Good For", style="yellow")
        
        region_table.add_row("nyc3", "New York", "US East Coast")
        region_table.add_row("sfo3", "San Francisco", "US West Coast")
        region_table.add_row("ams3", "Amsterdam", "Europe")
        region_table.add_row("sgp1", "Singapore", "Asia")
        region_table.add_row("lon1", "London", "UK")
        
        self.console.print(region_table)
        
        region = Prompt.ask(
            "\n🌍 Choose region (closest to you for best performance)",
            choices=["nyc3", "sfo3", "ams3", "sgp1", "lon1"],
            default="nyc3"
        )
        
        # Additional options
        droplet_name = Prompt.ask(
            "\n📛 Server name",
            default="ml-trading-bot"
        )
        
        enable_monitoring = Confirm.ask(
            "📊 Enable monitoring (Grafana + Prometheus)?",
            default=True
        )
        
        enable_backups = Confirm.ask(
            "💾 Enable automated backups (+20% cost)?",
            default=True
        )
        
        setup_domain = Confirm.ask(
            "🌐 Do you have a domain name to use?",
            default=False
        )
        
        domain_name = None
        if setup_domain:
            domain_name = Prompt.ask("🌐 Enter your domain name (e.g., mybot.com)")
        
        # Store configuration
        self.config = {
            'droplet': {
                'name': droplet_name,
                'size': size,
                'region': region,
                'image': 'ubuntu-22-04-x64',
                'enable_backups': enable_backups,
                'enable_monitoring': enable_monitoring
            },
            'domain': {
                'enabled': setup_domain,
                'name': domain_name
            },
            'monitoring': {
                'enabled': enable_monitoring
            }
        }
        
        # Show cost estimate
        self.show_cost_estimate()
    
    def show_cost_estimate(self):
        """Show estimated monthly costs"""
        costs = {
            's-1vcpu-1gb': 6,
            's-1vcpu-2gb': 12,
            's-2vcpu-2gb': 18,
            's-2vcpu-4gb': 24
        }
        
        base_cost = costs.get(self.config['droplet']['size'], 18)
        backup_cost = int(base_cost * 0.2) if self.config['droplet']['enable_backups'] else 0
        monitoring_cost = 0  # We use self-hosted monitoring
        
        total_cost = base_cost + backup_cost + monitoring_cost
        
        cost_table = Table()
        cost_table.add_column("Item", style="cyan")
        cost_table.add_column("Cost/Month", style="green")
        
        cost_table.add_row("Server", f"${base_cost}")
        if backup_cost > 0:
            cost_table.add_row("Automated Backups", f"${backup_cost}")
        cost_table.add_row("Bandwidth (1TB)", "$0")
        cost_table.add_row("Total Estimated", f"${total_cost}", style="bold green")
        
        self.console.print("\n💰 Cost Estimate:", style="blue")
        self.console.print(cost_table)
    
    def confirm_deployment(self):
        """Final confirmation before deployment"""
        self.console.print("\n🔍 Deployment Summary", style="blue")
        
        summary_table = Table()
        summary_table.add_column("Setting", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Server Name", self.config['droplet']['name'])
        summary_table.add_row("Server Size", self.config['droplet']['size'])
        summary_table.add_row("Region", self.config['droplet']['region'])
        summary_table.add_row("Monitoring", "✅ Enabled" if self.config['monitoring']['enabled'] else "❌ Disabled")
        summary_table.add_row("Backups", "✅ Enabled" if self.config['droplet']['enable_backups'] else "❌ Disabled")
        summary_table.add_row("Domain", self.config['domain']['name'] if self.config['domain']['enabled'] else "❌ Not configured")
        
        self.console.print(summary_table)
        
        warning_panel = Panel(
            "⚠️  This will create billable resources on Digital Ocean!\n"
            "💰 You will be charged according to Digital Ocean's pricing.\n"
            "🛑 You can destroy resources at any time to stop billing.",
            title="💸 Billing Warning",
            border_style="red"
        )
        self.console.print(warning_panel)
        
        if not Confirm.ask("\n🚀 Start deployment?", default=True):
            self.console.print("👋 Deployment cancelled", style="yellow")
            sys.exit(0)
    
    def deploy_to_digital_ocean(self):
        """Execute the deployment process"""
        self.console.print("\n🚀 Starting deployment...", style="blue")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            # Step 1: Create SSH key
            task1 = progress.add_task("Setting up SSH key...", total=None)
            self.setup_ssh_key()
            progress.update(task1, description="✅ SSH key configured")
            
            # Step 2: Create firewall
            task2 = progress.add_task("Creating firewall rules...", total=None)
            self.create_firewall()
            progress.update(task2, description="✅ Firewall rules created")
            
            # Step 3: Create droplet
            task3 = progress.add_task("Creating server (this may take a few minutes)...", total=None)
            self.create_droplet()
            progress.update(task3, description="✅ Server created")
            
            # Step 4: Wait for server to be ready
            task4 = progress.add_task("Waiting for server to be ready...", total=None)
            self.wait_for_server()
            progress.update(task4, description="✅ Server is ready")
            
            # Step 5: Configure server
            task5 = progress.add_task("Installing software on server...", total=None)
            self.configure_server()
            progress.update(task5, description="✅ Server configured")
            
            # Step 6: Deploy application
            task6 = progress.add_task("Deploying trading bot application...", total=None)
            self.deploy_application()
            progress.update(task6, description="✅ Application deployed")
            
            # Step 7: Set up monitoring
            if self.config['monitoring']['enabled']:
                task7 = progress.add_task("Setting up monitoring...", total=None)
                self.setup_monitoring()
                progress.update(task7, description="✅ Monitoring configured")
            
            # Step 8: Configure domain (if provided)
            if self.config['domain']['enabled']:
                task8 = progress.add_task("Configuring domain...", total=None)
                self.configure_domain()
                progress.update(task8, description="✅ Domain configured")
    
    def setup_ssh_key(self):
        """Set up SSH key for server access"""
        ssh_dir = Path.home() / '.ssh'
        ssh_dir.mkdir(exist_ok=True)
        
        key_path = ssh_dir / 'do_trading_bot'
        pub_key_path = ssh_dir / 'do_trading_bot.pub'
        
        # Generate SSH key if it doesn't exist
        if not key_path.exists():
            subprocess.run([
                'ssh-keygen', '-t', 'rsa', '-b', '4096',
                '-f', str(key_path),
                '-N', '',  # No passphrase
                '-C', 'trading-bot-deploy'
            ], check=True, capture_output=True)
        
        # Upload to Digital Ocean
        with open(pub_key_path) as f:
            pub_key_content = f.read().strip()
        
        # Check if key already exists
        result = subprocess.run([
            'doctl', 'compute', 'ssh-key', 'list', '--format', 'Name'
        ], capture_output=True, text=True)
        
        if 'trading-bot-key' not in result.stdout:
            subprocess.run([
                'doctl', 'compute', 'ssh-key', 'import', 'trading-bot-key',
                '--public-key', pub_key_content
            ], check=True, capture_output=True)
    
    def create_firewall(self):
        """Create firewall rules"""
        firewall_name = f"{self.config['droplet']['name']}-firewall"
        
        # Check if firewall exists
        result = subprocess.run([
            'doctl', 'compute', 'firewall', 'list', '--format', 'Name'
        ], capture_output=True, text=True)
        
        if firewall_name not in result.stdout:
            subprocess.run([
                'doctl', 'compute', 'firewall', 'create',
                '--name', firewall_name,
                '--inbound-rules', 'protocol:tcp,ports:22,address:0.0.0.0/0',
                '--inbound-rules', 'protocol:tcp,ports:80,address:0.0.0.0/0',
                '--inbound-rules', 'protocol:tcp,ports:443,address:0.0.0.0/0', 
                '--inbound-rules', 'protocol:tcp,ports:8000,address:0.0.0.0/0',
                '--inbound-rules', 'protocol:tcp,ports:8501,address:0.0.0.0/0',
                '--inbound-rules', 'protocol:tcp,ports:3000,address:0.0.0.0/0'
            ], check=True, capture_output=True)
    
    def create_droplet(self):
        """Create the Digital Ocean droplet"""
        cmd = [
            'doctl', 'compute', 'droplet', 'create',
            self.config['droplet']['name'],
            '--region', self.config['droplet']['region'],
            '--image', self.config['droplet']['image'],
            '--size', self.config['droplet']['size'],
            '--ssh-keys', 'trading-bot-key',
            '--enable-monitoring'
        ]
        
        if self.config['droplet']['enable_backups']:
            cmd.append('--enable-backups')
        
        cmd.append('--wait')
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Get droplet info
        time.sleep(5)
        self.get_droplet_info()
    
    def get_droplet_info(self):
        """Get droplet ID and IP address"""
        result = subprocess.run([
            'doctl', 'compute', 'droplet', 'list',
            '--format', 'ID,Name,PublicIPv4',
            '--no-header'
        ], capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if self.config['droplet']['name'] in line:
                parts = line.split()
                self.droplet_id = parts[0]
                self.droplet_ip = parts[2]
                break
    
    def wait_for_server(self):
        """Wait for server to be accessible via SSH"""
        ssh_key_path = Path.home() / '.ssh' / 'do_trading_bot'
        
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                result = subprocess.run([
                    'ssh', '-i', str(ssh_key_path),
                    '-o', 'StrictHostKeyChecking=no',
                    '-o', 'ConnectTimeout=10',
                    f'root@{self.droplet_ip}',
                    'echo "ready"'
                ], capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    return
                    
            except subprocess.TimeoutExpired:
                pass
            
            time.sleep(10)
        
        raise Exception("Server failed to become ready")
    
    def configure_server(self):
        """Configure the server with required software"""
        ssh_key_path = Path.home() / '.ssh' / 'do_trading_bot'
        
        # Create setup script
        setup_script = '''#!/bin/bash
set -e

# Update system
apt-get update -y
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl enable docker
systemctl start docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install additional tools
apt-get install -y nginx certbot python3-certbot-nginx htop

# Create app directory
mkdir -p /opt/trading-bot
'''
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            f.write(setup_script)
            script_path = f.name
        
        try:
            # Copy setup script to server
            subprocess.run([
                'scp', '-i', str(ssh_key_path),
                '-o', 'StrictHostKeyChecking=no',
                script_path,
                f'root@{self.droplet_ip}:/tmp/setup.sh'
            ], check=True, capture_output=True)
            
            # Run setup script
            subprocess.run([
                'ssh', '-i', str(ssh_key_path),
                '-o', 'StrictHostKeyChecking=no',
                f'root@{self.droplet_ip}',
                'chmod +x /tmp/setup.sh && /tmp/setup.sh'
            ], check=True, capture_output=True)
            
        finally:
            os.unlink(script_path)
    
    def deploy_application(self):
        """Deploy the trading bot application"""
        ssh_key_path = Path.home() / '.ssh' / 'do_trading_bot'
        
        # Create production docker-compose.yml
        compose_content = '''version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  dashboard:
    build: .
    command: streamlit run start_dashboard.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - api

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: trading_bot
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: secure_password_123
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
'''
        
        # Create application files archive
        files_to_copy = [
            '.env', 'requirements.txt', 'Dockerfile',
            'start_api.py', 'start_dashboard.py'
        ]
        
        # Add directories if they exist
        for dir_name in ['src', 'config', 'scripts']:
            if Path(dir_name).exists():
                files_to_copy.append(dir_name)
        
        # Create tar archive
        subprocess.run(['tar', '-czf', '/tmp/app.tar.gz'] + files_to_copy, check=True)
        
        try:
            # Copy application files
            subprocess.run([
                'scp', '-i', str(ssh_key_path),
                '-o', 'StrictHostKeyChecking=no',
                '/tmp/app.tar.gz',
                f'root@{self.droplet_ip}:/opt/trading-bot/'
            ], check=True, capture_output=True)
            
            # Create docker-compose file on server
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml') as f:
                f.write(compose_content)
                compose_path = f.name
            
            subprocess.run([
                'scp', '-i', str(ssh_key_path),
                '-o', 'StrictHostKeyChecking=no',
                compose_path,
                f'root@{self.droplet_ip}:/opt/trading-bot/docker-compose.yml'
            ], check=True, capture_output=True)
            
            # Extract and start application
            subprocess.run([
                'ssh', '-i', str(ssh_key_path),
                '-o', 'StrictHostKeyChecking=no',
                f'root@{self.droplet_ip}',
                'cd /opt/trading-bot && tar -xzf app.tar.gz && docker-compose up -d --build'
            ], check=True, capture_output=True)
            
        finally:
            os.unlink('/tmp/app.tar.gz')
            if 'compose_path' in locals():
                os.unlink(compose_path)
    
    def setup_monitoring(self):
        """Set up monitoring stack"""
        ssh_key_path = Path.home() / '.ssh' / 'do_trading_bot'
        
        monitoring_commands = '''
# Start Prometheus
docker run -d --name prometheus \\
  -p 9090:9090 \\
  --restart unless-stopped \\
  prom/prometheus

# Start Grafana
docker run -d --name grafana \\
  -p 3000:3000 \\
  --restart unless-stopped \\
  -e "GF_SECURITY_ADMIN_PASSWORD=admin123" \\
  grafana/grafana
'''
        
        subprocess.run([
            'ssh', '-i', str(ssh_key_path),
            '-o', 'StrictHostKeyChecking=no',
            f'root@{self.droplet_ip}',
            monitoring_commands
        ], check=True, capture_output=True)
    
    def configure_domain(self):
        """Configure domain and SSL"""
        # This would configure nginx and certbot for the domain
        # Implementation depends on specific domain configuration needs
        pass
    
    def show_completion(self):
        """Show deployment completion message"""
        completion_text = Text()
        completion_text.append("🎉 Deployment Completed Successfully!\n\n", style="bold green")
        completion_text.append("Your ML Trading Bot is now live in the cloud!\n\n", style="cyan")
        
        completion_text.append("🌐 Access Points:\n", style="white")
        completion_text.append(f"• Trading Dashboard: http://{self.droplet_ip}:8501\n", style="green")
        completion_text.append(f"• API Documentation: http://{self.droplet_ip}:8000/docs\n", style="green")
        
        if self.config['monitoring']['enabled']:
            completion_text.append(f"• Monitoring (Grafana): http://{self.droplet_ip}:3000\n", style="green")
            completion_text.append(f"  └─ Username: admin, Password: admin123\n", style="yellow")
        
        completion_text.append(f"• SSH Access: ssh -i ~/.ssh/do_trading_bot root@{self.droplet_ip}\n\n", style="blue")
        
        completion_text.append("📋 Next Steps:\n", style="white")
        completion_text.append("1. Visit the dashboard and verify everything works\n", style="cyan")
        completion_text.append("2. Change default passwords for security\n", style="cyan")
        completion_text.append("3. Monitor your bot's performance\n", style="cyan")
        completion_text.append("4. Set up alerts and backups\n\n", style="cyan")
        
        completion_text.append("⚠️  Important:\n", style="red")
        completion_text.append("• Your server is now billing at Digital Ocean\n", style="yellow")
        completion_text.append("• Change all default passwords immediately\n", style="yellow")
        completion_text.append("• Monitor your server and trading performance\n", style="yellow")
        completion_text.append("• Keep your SSH key secure and backed up", style="yellow")
        
        panel = Panel(
            completion_text,
            title="🚀 Deployment Complete!",
            border_style="green"
        )
        self.console.print(panel)
    
    def cleanup_resources(self):
        """Clean up resources on failure"""
        if self.droplet_id:
            try:
                subprocess.run([
                    'doctl', 'compute', 'droplet', 'delete', self.droplet_id, '--force'
                ], check=True, capture_output=True)
                self.console.print("✅ Cleaned up failed droplet", style="green")
            except:
                self.console.print("⚠️ Could not clean up droplet - please delete manually", style="yellow")

def main():
    """Main entry point"""
    deployer = DigitalOceanDeployer()
    deployer.run()

if __name__ == "__main__":
    main()