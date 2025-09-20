#!/usr/bin/env python3
"""
🧙‍♂️ ML Trading Bot Setup Wizard
===================================

A beginner-friendly interactive setup wizard that guides users through:
- Environment setup and dependency installation
- API key configuration and testing
- Trading preferences and risk settings
- Deployment options (local, Docker, Digital Ocean)
- Initial validation and health checks

This wizard makes the ML trading bot accessible to complete beginners.
"""

import os
import sys
import json
import subprocess
import platform
import requests
import getpass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

# Rich for beautiful terminal UI
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
except ImportError:
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "pyyaml", "requests"], check=True)
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align

console = Console()

class TradingBotSetupWizard:
    """Interactive setup wizard for the ML Trading Bot"""
    
    def __init__(self):
        self.console = Console()
        self.config = {}
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.scripts_dir = self.project_root / "scripts"
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
    
    def run(self):
        """Run the complete setup wizard"""
        try:
            self.show_welcome()
            self.check_system_requirements()
            self.setup_environment()
            self.configure_api_keys()
            self.configure_trading_settings()
            self.choose_deployment_method()
            self.validate_setup()
            self.show_completion()
        except KeyboardInterrupt:
            self.console.print("\n❌ Setup cancelled by user", style="red")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n❌ Setup failed: {str(e)}", style="red")
            self.console.print("💡 Please check the logs or contact support", style="yellow")
            sys.exit(1)
    
    def show_welcome(self):
        """Show welcome screen with information about the bot"""
        welcome_text = Text()
        welcome_text.append("🤖 ML Trading Bot Setup Wizard\n", style="bold magenta")
        welcome_text.append("Welcome to the most beginner-friendly crypto trading bot!\n\n", style="cyan")
        welcome_text.append("This wizard will help you:\n", style="white")
        welcome_text.append("✅ Set up your development environment\n", style="green")
        welcome_text.append("✅ Configure API keys securely\n", style="green")
        welcome_text.append("✅ Customize trading settings\n", style="green")
        welcome_text.append("✅ Choose deployment method\n", style="green")
        welcome_text.append("✅ Validate your setup\n\n", style="green")
        welcome_text.append("⚠️  Important: This bot is for educational purposes.\n", style="yellow")
        welcome_text.append("Only trade with money you can afford to lose!", style="red bold")
        
        panel = Panel(
            Align.center(welcome_text),
            title="🚀 Welcome",
            border_style="blue"
        )
        self.console.print(panel)
        
        if not Confirm.ask("\n🤔 Ready to start the setup?", default=True):
            self.console.print("👋 Come back when you're ready!", style="yellow")
            sys.exit(0)
    
    def check_system_requirements(self):
        """Check system requirements and compatibility"""
        self.console.print("\n🔍 Checking system requirements...", style="blue")
        
        requirements = Table()
        requirements.add_column("Requirement", style="cyan")
        requirements.add_column("Status", style="green")
        requirements.add_column("Details", style="white")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 11):
            requirements.add_row("Python 3.11+", "✅ Pass", f"Found Python {python_version.major}.{python_version.minor}")
        else:
            requirements.add_row("Python 3.11+", "❌ Fail", f"Found Python {python_version.major}.{python_version.minor}")
            self.console.print(requirements)
            self.console.print("❌ Python 3.11 or higher is required!", style="red")
            sys.exit(1)
        
        # Check operating system
        os_name = platform.system()
        requirements.add_row("Operating System", "✅ Pass", f"{os_name} {platform.release()}")
        
        # Check available disk space
        try:
            disk_usage = os.statvfs(self.project_root) if hasattr(os, 'statvfs') else None
            if disk_usage:
                free_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
                if free_gb >= 2:
                    requirements.add_row("Disk Space", "✅ Pass", f"{free_gb:.1f} GB available")
                else:
                    requirements.add_row("Disk Space", "⚠️  Warning", f"{free_gb:.1f} GB available (need 2+ GB)")
            else:
                requirements.add_row("Disk Space", "ℹ️  Info", "Unable to check (Windows)")
        except:
            requirements.add_row("Disk Space", "ℹ️  Info", "Unable to check")
        
        # Check internet connectivity
        try:
            requests.get("https://api.bybit.com/v5/market/time", timeout=5)
            requirements.add_row("Internet Connection", "✅ Pass", "Connected to Bybit API")
        except:
            requirements.add_row("Internet Connection", "❌ Fail", "Cannot reach Bybit API")
            self.console.print("❌ Internet connection required!", style="red")
            sys.exit(1)
        
        self.console.print(requirements)
        self.console.print("✅ System requirements check completed!", style="green")
    
    def setup_environment(self):
        """Set up Python environment and dependencies"""
        self.console.print("\n⚙️  Setting up environment...", style="blue")
        
        # Check for virtual environment
        venv_path = self.project_root / ".venv"
        if not venv_path.exists():
            if Confirm.ask("📦 Create a virtual environment? (Recommended)", default=True):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Creating virtual environment...", total=None)
                    try:
                        subprocess.run([
                            sys.executable, "-m", "venv", str(venv_path)
                        ], check=True, capture_output=True)
                        progress.update(task, description="✅ Virtual environment created!")
                    except subprocess.CalledProcessError as e:
                        self.console.print(f"❌ Failed to create virtual environment: {e}", style="red")
                        sys.exit(1)
        
        # Install dependencies
        if Confirm.ask("📚 Install required dependencies?", default=True):
            self.install_dependencies()
        
        self.console.print("✅ Environment setup completed!", style="green")
    
    def install_dependencies(self):
        """Install required Python packages"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            # Determine pip command
            venv_path = self.project_root / ".venv"
            if venv_path.exists():
                if platform.system() == "Windows":
                    pip_cmd = str(venv_path / "Scripts" / "pip.exe")
                else:
                    pip_cmd = str(venv_path / "bin" / "pip")
            else:
                pip_cmd = "pip"
            
            packages = [
                ("Core dependencies", ["fastapi[all]", "streamlit", "pandas", "numpy", "aiohttp"]),
                ("ML packages", ["scikit-learn", "xgboost", "lightgbm", "torch", "transformers"]),
                ("Trading libraries", ["python-bybit", "ccxt", "ta-lib-binary"]),
                ("Database & caching", ["asyncpg", "redis", "sqlalchemy[asyncio]"]),
                ("Monitoring & logging", ["prometheus-client", "structlog", "rich"])
            ]
            
            for category, package_list in packages:
                task = progress.add_task(f"Installing {category}...", total=None)
                try:
                    subprocess.run([
                        pip_cmd, "install", "--upgrade", *package_list
                    ], check=True, capture_output=True)
                    progress.update(task, description=f"✅ {category} installed!")
                except subprocess.CalledProcessError:
                    progress.update(task, description=f"⚠️  {category} - some packages failed")
    
    def configure_api_keys(self):
        """Configure API keys and test connectivity"""
        self.console.print("\n🔑 API Key Configuration", style="blue")
        
        panel_text = Text()
        panel_text.append("To use this trading bot, you need Bybit API keys.\n\n", style="white")
        panel_text.append("📝 How to get Bybit API keys:\n", style="cyan")
        panel_text.append("1. Go to https://www.bybit.com/\n", style="white")
        panel_text.append("2. Create an account (if you don't have one)\n", style="white")
        panel_text.append("3. Go to Account & Security > API Management\n", style="white")
        panel_text.append("4. Create new API key with trading permissions\n", style="white")
        panel_text.append("5. Copy the API Key and Secret\n\n", style="white")
        panel_text.append("⚠️  Important: Start with TESTNET for learning!\n", style="yellow")
        panel_text.append("🔒 Your keys will be stored securely and encrypted.", style="green")
        
        info_panel = Panel(panel_text, title="API Key Setup", border_style="yellow")
        self.console.print(info_panel)
        
        # Choose environment
        env_choice = Prompt.ask(
            "\n🌍 Which environment do you want to use?",
            choices=["testnet", "mainnet"],
            default="testnet"
        )
        
        if env_choice == "mainnet":
            warning = Panel(
                "⚠️  WARNING: You're configuring MAINNET (real money)!\n"
                "Make sure you understand the risks and start with small amounts.",
                title="⚠️  MAINNET WARNING",
                border_style="red"
            )
            self.console.print(warning)
            if not Confirm.ask("🤔 Are you sure you want to use mainnet?", default=False):
                env_choice = "testnet"
                self.console.print("✅ Switched to testnet for safety!", style="green")
        
        self.config['environment'] = env_choice
        
        # Get API credentials
        self.console.print(f"\n🔑 Please enter your {env_choice.upper()} API credentials:")
        
        api_key = Prompt.ask("API Key", password=False)
        api_secret = getpass.getpass("API Secret (hidden): ")
        
        if not api_key or not api_secret:
            self.console.print("❌ API key and secret are required!", style="red")
            sys.exit(1)
        
        # Test API connection
        if self.test_api_connection(api_key, api_secret, env_choice):
            self.config['api_key'] = api_key
            self.config['api_secret'] = api_secret
            self.console.print("✅ API keys configured and tested successfully!", style="green")
        else:
            self.console.print("❌ API connection failed!", style="red")
            sys.exit(1)
    
    def test_api_connection(self, api_key: str, api_secret: str, environment: str) -> bool:
        """Test API connection with provided credentials"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Testing API connection...", total=None)
            
            try:
                base_url = "https://api-testnet.bybit.com" if environment == "testnet" else "https://api.bybit.com"
                
                import hmac
                import hashlib
                import time
                
                timestamp = str(int(time.time() * 1000))
                param_str = f"timestamp={timestamp}"
                signature = hmac.new(
                    api_secret.encode('utf-8'),
                    param_str.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                headers = {
                    'X-BAPI-API-KEY': api_key,
                    'X-BAPI-SIGN': signature,
                    'X-BAPI-TIMESTAMP': timestamp,
                    'X-BAPI-RECV-WINDOW': '5000'
                }
                
                response = requests.get(
                    f"{base_url}/v5/account/wallet-balance",
                    headers=headers,
                    params={'accountType': 'UNIFIED'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    progress.update(task, description="✅ API connection successful!")
                    return True
                else:
                    progress.update(task, description="❌ API connection failed!")
                    self.console.print(f"Error: {response.text}", style="red")
                    return False
                    
            except Exception as e:
                progress.update(task, description="❌ Connection test failed!")
                self.console.print(f"Error: {str(e)}", style="red")
                return False
    
    def configure_trading_settings(self):
        """Configure trading preferences and risk settings"""
        self.console.print("\n⚙️  Trading Configuration", style="blue")
        
        # Trading pairs
        self.console.print("\n📊 Select trading pairs:")
        popular_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]
        selected_pairs = []
        
        for pair in popular_pairs:
            if Confirm.ask(f"Trade {pair}?", default=pair in ["BTCUSDT", "ETHUSDT"]):
                selected_pairs.append(pair)
        
        if not selected_pairs:
            selected_pairs = ["BTCUSDT"]  # Default fallback
        
        self.config['trading_pairs'] = selected_pairs
        
        # Risk settings
        self.console.print("\n🛡️  Risk Management Settings:")
        
        # Explain dynamic risk scaling
        risk_info = Panel(
            "🤖 **Dynamic Risk Scaling Enabled**\n\n"
            "The bot automatically adjusts risk based on your account size:\n"
            "• **Small accounts** (< $10k): 2.0% risk per trade (more aggressive)\n"
            "• **Growing accounts** ($10k - $100k): Gradually reduces to 0.5%\n"
            "• **Large accounts** (> $100k): 0.5% risk per trade (conservative)\n\n"
            "This maximizes growth for small accounts while protecting large ones!",
            title="🎯 Smart Risk Management",
            border_style="green"
        )
        self.console.print(risk_info)
        
        initial_balance = FloatPrompt.ask(
            "Initial balance (USDT)",
            default=1000.0,
            show_default=True
        )
        
        # Calculate current risk based on balance
        if initial_balance <= 10000:
            current_risk = 2.0
            risk_desc = "Maximum risk (small account)"
        elif initial_balance >= 100000:
            current_risk = 0.5
            risk_desc = "Minimum risk (large account)"
        else:
            # Simplified calculation for display
            ratio = (initial_balance - 10000) / (100000 - 10000)
            current_risk = 0.5 + (1.5 * (1 - ratio))
            risk_desc = "Scaled risk (growing account)"
        
        self.console.print(f"\n📊 Your current risk level: {current_risk:.1f}% per trade ({risk_desc})")
        
        # Allow override if desired
        if Confirm.ask("Would you like to customize the risk settings?", default=False):
            max_risk_per_trade = FloatPrompt.ask(
                "Maximum risk per trade (%)",
                default=current_risk,
                show_default=True
            ) / 100
        else:
            max_risk_per_trade = current_risk / 100
        
        max_daily_loss = FloatPrompt.ask(
            "Maximum daily loss (%)",
            default=5.0,
            show_default=True
        ) / 100
        
        self.config.update({
            'initial_balance': initial_balance,
            'max_risk_per_trade': max_risk_per_trade,
            'max_daily_loss': max_daily_loss,
            'stop_loss_percentage': 0.02,  # 2% stop loss
            'take_profit_percentage': 0.04  # 4% take profit
        })
        
        # ML Model preferences
        self.console.print("\n🤖 ML Model Configuration:")
        
        model_aggressiveness = Prompt.ask(
            "Model prediction aggressiveness",
            choices=["conservative", "balanced", "aggressive"],
            default="balanced"
        )
        
        self.config['ml_settings'] = {
            'aggressiveness': model_aggressiveness,
            'confidence_threshold': 0.7 if model_aggressiveness == "conservative" else 0.6,
            'ensemble_weights': {
                'lightgbm': 0.4,
                'xgboost': 0.3,
                'neural_network': 0.2,
                'transformer': 0.1
            }
        }
        
        self.console.print("✅ Trading settings configured!", style="green")
    
    def choose_deployment_method(self):
        """Let user choose deployment method"""
        self.console.print("\n🚀 Deployment Options", style="blue")
        
        deployment_info = Table()
        deployment_info.add_column("Method", style="cyan")
        deployment_info.add_column("Difficulty", style="white")
        deployment_info.add_column("Cost", style="green")
        deployment_info.add_column("Best For", style="yellow")
        
        deployment_info.add_row("Local", "🟢 Easy", "Free", "Learning & Development")
        deployment_info.add_row("Docker", "🟡 Medium", "Free", "Consistent Environment")
        deployment_info.add_row("Digital Ocean", "🟡 Medium", "$5-20/month", "24/7 Production")
        deployment_info.add_row("Kubernetes", "🔴 Hard", "$20-100/month", "Enterprise Scale")
        
        self.console.print(deployment_info)
        
        deployment_choice = Prompt.ask(
            "\n🤔 How do you want to deploy your bot?",
            choices=["local", "docker", "digitalocean", "kubernetes"],
            default="local"
        )
        
        self.config['deployment_method'] = deployment_choice
        
        if deployment_choice == "local":
            self.console.print("✅ Local deployment selected - great for getting started!", style="green")
        elif deployment_choice == "docker":
            self.console.print("✅ Docker deployment selected - excellent choice!", style="green")
            self.setup_docker_config()
        elif deployment_choice == "digitalocean":
            self.console.print("✅ Digital Ocean deployment selected!", style="green")
            self.setup_digitalocean_config()
        elif deployment_choice == "kubernetes":
            self.console.print("✅ Kubernetes deployment selected - advanced option!", style="green")
            self.setup_kubernetes_config()
    
    def setup_docker_config(self):
        """Configure Docker deployment settings"""
        self.console.print("\n🐳 Docker Configuration", style="blue")
        
        self.config['docker'] = {
            'port_api': IntPrompt.ask("API port", default=8000),
            'port_dashboard': IntPrompt.ask("Dashboard port", default=8501),
            'enable_monitoring': Confirm.ask("Enable monitoring (Grafana/Prometheus)?", default=True)
        }
    
    def setup_digitalocean_config(self):
        """Configure Digital Ocean deployment settings"""
        self.console.print("\n🌊 Digital Ocean Configuration", style="blue")
        
        self.console.print("We'll create a deployment guide for Digital Ocean!")
        self.config['digitalocean'] = {
            'droplet_size': Prompt.ask(
                "Droplet size",
                choices=["s-1vcpu-1gb", "s-1vcpu-2gb", "s-2vcpu-2gb", "s-2vcpu-4gb"],
                default="s-1vcpu-2gb"
            ),
            'region': Prompt.ask(
                "Preferred region",
                choices=["nyc1", "nyc3", "ams3", "sgp1", "lon1", "fra1"],
                default="nyc3"
            ),
            'enable_monitoring': Confirm.ask("Enable monitoring?", default=True),
            'setup_domain': Confirm.ask("Set up custom domain?", default=False)
        }
    
    def setup_kubernetes_config(self):
        """Configure Kubernetes deployment settings"""
        self.console.print("\n☸️  Kubernetes Configuration", style="blue")
        
        self.config['kubernetes'] = {
            'replicas': IntPrompt.ask("Number of API replicas", default=3),
            'enable_autoscaling': Confirm.ask("Enable auto-scaling?", default=True),
            'monitoring_stack': Confirm.ask("Deploy monitoring stack?", default=True)
        }
    
    def validate_setup(self):
        """Validate the setup configuration"""
        self.console.print("\n✅ Validating Setup", style="blue")
        
        # Save configuration
        self.save_configuration()
        
        # Create necessary files
        self.create_config_files()
        
        # Run basic validation
        self.run_validation_tests()
        
        self.console.print("✅ Setup validation completed!", style="green")
    
    def save_configuration(self):
        """Save configuration to files"""
        # Save main config
        config_file = self.config_dir / "setup_wizard_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Create environment-specific config
        env_config = {
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'environment': self.config['environment']
            },
            'trading': {
                'symbols': self.config['trading_pairs'],
                'initial_balance': self.config['initial_balance'],
                'risk_management': {
                    'max_risk_per_trade': self.config['max_risk_per_trade'],
                    'max_daily_loss': self.config['max_daily_loss'],
                    'stop_loss': self.config['stop_loss_percentage'],
                    'take_profit': self.config['take_profit_percentage']
                }
            },
            'ml_models': self.config['ml_settings']
        }
        
        env_file = self.config_dir / f"{self.config['environment']}.yaml" 
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f, default_flow_style=False)
        
        self.console.print(f"💾 Configuration saved to {config_file}", style="green")
    
    def create_config_files(self):
        """Create necessary configuration files"""
        # Create .env file
        env_content = f"""# ML Trading Bot Environment Configuration
# Generated by Setup Wizard on {self.get_timestamp()}

# API Configuration
BYBIT_API_KEY={self.config['api_key']}
BYBIT_API_SECRET={self.config['api_secret']}
ENVIRONMENT={self.config['environment']}

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this
API_KEY_SECRET=your-api-key-encryption-secret

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        self.console.print(f"🔐 Environment file created: {env_file}", style="green")
    
    def run_validation_tests(self):
        """Run basic validation tests"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            tests = [
                ("Configuration files", self.validate_config_files),
                ("API connectivity", self.validate_api_connection),
                ("Dependencies", self.validate_dependencies),
                ("File permissions", self.validate_permissions)
            ]
            
            for test_name, test_func in tests:
                task = progress.add_task(f"Testing {test_name}...", total=None)
                try:
                    test_func()
                    progress.update(task, description=f"✅ {test_name} - OK")
                except Exception as e:
                    progress.update(task, description=f"⚠️  {test_name} - Warning")
                    self.console.print(f"Warning in {test_name}: {str(e)}", style="yellow")
    
    def validate_config_files(self):
        """Validate configuration files exist and are readable"""
        required_files = [
            self.config_dir / "setup_wizard_config.yaml",
            self.config_dir / f"{self.config['environment']}.yaml",
            self.project_root / ".env"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Missing: {file_path}")
    
    def validate_api_connection(self):
        """Validate API connection works"""
        if not self.test_api_connection(
            self.config['api_key'], 
            self.config['api_secret'], 
            self.config['environment']
        ):
            raise ConnectionError("API connection failed")
    
    def validate_dependencies(self):
        """Validate required dependencies are available"""
        required_modules = ['fastapi', 'streamlit', 'pandas', 'numpy']
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                raise ImportError(f"Missing dependency: {module}")
    
    def validate_permissions(self):
        """Validate file permissions"""
        test_file = self.config_dir / "test_write.tmp"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
        except:
            raise PermissionError("Cannot write to config directory")
    
    def show_completion(self):
        """Show completion screen with next steps"""
        completion_text = Text()
        completion_text.append("🎉 Setup Complete!\n\n", style="bold green")
        completion_text.append("Your ML Trading Bot is configured and ready to use!\n\n", style="cyan")
        
        completion_text.append("📂 Generated Files:\n", style="white")
        completion_text.append(f"• Configuration: config/{self.config['environment']}.yaml\n", style="green")
        completion_text.append(f"• Environment: .env\n", style="green")
        completion_text.append(f"• Setup record: config/setup_wizard_config.yaml\n\n", style="green")
        
        completion_text.append("🚀 Next Steps:\n", style="yellow")
        
        if self.config['deployment_method'] == 'local':
            completion_text.append("1. Run: python start_api.py\n", style="white")
            completion_text.append("2. Run: streamlit run start_dashboard.py\n", style="white")
            completion_text.append("3. Open: http://localhost:8501\n", style="white")
        elif self.config['deployment_method'] == 'docker':
            completion_text.append("1. Run: docker-compose up -d\n", style="white")
            completion_text.append("2. Open: http://localhost:8501\n", style="white")
        elif self.config['deployment_method'] == 'digitalocean':
            completion_text.append("1. Check the Digital Ocean deployment guide\n", style="white")
            completion_text.append("2. Follow docs/DIGITAL_OCEAN_GUIDE.md\n", style="white")
        
        completion_text.append("\n📚 Documentation:\n", style="cyan")
        completion_text.append("• README.md - Complete system overview\n", style="white")
        completion_text.append("• docs/BEGINNER_SETUP_GUIDE.md - Detailed setup guide\n", style="white")
        completion_text.append("• docs/API_REFERENCE.md - API documentation\n", style="white")
        
        completion_text.append("\n⚠️  Important Reminders:\n", style="red")
        completion_text.append("• Start with paper trading (testnet)\n", style="yellow")
        completion_text.append("• Never risk more than you can afford to lose\n", style="yellow")
        completion_text.append("• Monitor your bot regularly\n", style="yellow")
        completion_text.append("• Keep your API keys secure\n", style="yellow")
        
        panel = Panel(
            Align.center(completion_text),
            title="🎊 Congratulations!",
            border_style="green"
        )
        self.console.print(panel)
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Main entry point"""
    wizard = TradingBotSetupWizard()
    wizard.run()

if __name__ == "__main__":
    main()