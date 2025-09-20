#!/usr/bin/env python3
"""
🔍 Setup Validation Tool
=======================

This tool helps beginners validate their trading bot setup
and troubleshoot common issues.

Run this anytime to check if your bot is properly configured!
"""

import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import yaml
except ImportError:
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "pyyaml", "requests"], check=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import yaml

console = Console()

class SetupValidator:
    """Validates trading bot setup and provides troubleshooting guidance"""
    
    def __init__(self):
        self.console = Console()
        self.project_root = Path(__file__).parent
        self.validation_results = []
        
    def run(self):
        """Run complete validation process"""
        self.show_welcome()
        self.run_all_validations()
        self.show_results()
        self.provide_recommendations()
    
    def show_welcome(self):
        """Show welcome message"""
        welcome_text = Text()
        welcome_text.append("🔍 Setup Validation Tool\n", style="bold blue")
        welcome_text.append("Checking your trading bot configuration...\n\n", style="cyan")
        welcome_text.append("This tool will validate:\n", style="white")
        welcome_text.append("✅ Python environment and dependencies\n", style="green")
        welcome_text.append("✅ Configuration files and settings\n", style="green")
        welcome_text.append("✅ API connectivity and permissions\n", style="green")
        welcome_text.append("✅ Trading bot components\n", style="green")
        welcome_text.append("✅ Security and best practices\n", style="green")
        
        panel = Panel(
            welcome_text,
            title="🔧 Validation Starting",
            border_style="blue"
        )
        self.console.print(panel)
    
    def run_all_validations(self):
        """Run all validation checks"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            
            # Python environment
            task1 = progress.add_task("Checking Python environment...", total=None)
            self.validate_python_environment()
            progress.update(task1, description="✅ Python environment checked")
            
            # Dependencies
            task2 = progress.add_task("Validating dependencies...", total=None)
            self.validate_dependencies()
            progress.update(task2, description="✅ Dependencies validated")
            
            # Configuration files
            task3 = progress.add_task("Checking configuration files...", total=None)
            self.validate_configuration_files()
            progress.update(task3, description="✅ Configuration files checked")
            
            # API connectivity
            task4 = progress.add_task("Testing API connectivity...", total=None)
            self.validate_api_connectivity()
            progress.update(task4, description="✅ API connectivity tested")
            
            # Bot components
            task5 = progress.add_task("Validating bot components...", total=None)
            self.validate_bot_components()
            progress.update(task5, description="✅ Bot components validated")
            
            # Security checks
            task6 = progress.add_task("Running security checks...", total=None)
            self.validate_security()
            progress.update(task6, description="✅ Security checks completed")
    
    def validate_python_environment(self):
        """Validate Python environment"""
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.add_result("✅", "Python Version", f"Python {python_version.major}.{python_version.minor}.{python_version.micro}", "Good")
        else:
            self.add_result("❌", "Python Version", f"Python {python_version.major}.{python_version.minor}.{python_version.micro}", "Need 3.8+")
        
        # Check virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if in_venv:
            self.add_result("✅", "Virtual Environment", "Active", "Recommended practice")
        else:
            self.add_result("⚠️", "Virtual Environment", "Not detected", "Recommended for isolation")
        
        # Check available disk space
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        if free_space_gb > 2:
            self.add_result("✅", "Disk Space", f"{free_space_gb:.1f} GB available", "Sufficient")
        else:
            self.add_result("⚠️", "Disk Space", f"{free_space_gb:.1f} GB available", "Low space")
    
    def validate_dependencies(self):
        """Validate required dependencies"""
        required_packages = [
            'fastapi', 'uvicorn', 'streamlit', 'pandas', 'numpy',
            'scikit-learn', 'requests', 'python-dotenv', 'pyyaml',
            'rich', 'redis', 'psycopg2-binary'
        ]
        
        installed_packages = {}
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if ' ' in line and not line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 2:
                        installed_packages[parts[0].lower()] = parts[1]
        except subprocess.CalledProcessError:
            self.add_result("❌", "Package Check", "Failed to check packages", "Run pip install -r requirements.txt")
            return
        
        missing_packages = []
        for package in required_packages:
            if package.lower().replace('-', '_') in installed_packages or package.lower() in installed_packages:
                self.add_result("✅", f"Package: {package}", "Installed", "Available")
            else:
                missing_packages.append(package)
                self.add_result("❌", f"Package: {package}", "Missing", "Install required")
        
        if not missing_packages:
            self.add_result("✅", "All Dependencies", "Installed", "Ready to go")
        else:
            self.add_result("❌", "Dependencies", f"{len(missing_packages)} missing", "Run quick_start.py")
    
    def validate_configuration_files(self):
        """Validate configuration files"""
        # Check .env file
        env_file = self.project_root / '.env'
        if env_file.exists():
            self.add_result("✅", ".env File", "Found", "Configuration present")
            self.validate_env_file(env_file)
        else:
            self.add_result("❌", ".env File", "Missing", "Run setup wizard")
        
        # Check config directory
        config_dir = self.project_root / 'config'
        if config_dir.exists():
            self.add_result("✅", "Config Directory", "Found", "Configuration directory present")
            self.validate_config_files(config_dir)
        else:
            self.add_result("❌", "Config Directory", "Missing", "Run setup wizard")
        
        # Check requirements.txt
        req_file = self.project_root / 'requirements.txt'
        if req_file.exists():
            self.add_result("✅", "requirements.txt", "Found", "Dependencies defined")
        else:
            self.add_result("⚠️", "requirements.txt", "Missing", "May cause dependency issues")
    
    def validate_env_file(self, env_file: Path):
        """Validate .env file contents"""
        try:
            with open(env_file) as f:
                content = f.read()
            
            required_vars = ['BYBIT_API_KEY', 'BYBIT_API_SECRET', 'RISK_LEVEL']
            found_vars = []
            
            for var in required_vars:
                if var in content and f'{var}=' in content:
                    # Check if variable has a value
                    for line in content.split('\n'):
                        if line.startswith(f'{var}=') and '=' in line:
                            value = line.split('=', 1)[1].strip()
                            if value and value not in ['your_api_key_here', 'your_secret_here', 'placeholder']:
                                found_vars.append(var)
                                break
            
            if len(found_vars) == len(required_vars):
                self.add_result("✅", "Environment Variables", "All configured", "Ready for trading")
            else:
                missing = set(required_vars) - set(found_vars)
                self.add_result("❌", "Environment Variables", f"Missing: {', '.join(missing)}", "Complete configuration")
                
        except Exception as e:
            self.add_result("❌", ".env Validation", "Failed to read", f"Error: {str(e)}")
    
    def validate_config_files(self, config_dir: Path):
        """Validate configuration files in config directory"""
        expected_files = ['trading_config.yaml', 'risk_config.yaml']
        
        for config_file in expected_files:
            file_path = config_dir / config_file
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        yaml.safe_load(f)
                    self.add_result("✅", f"Config: {config_file}", "Valid YAML", "Configuration ready")
                except yaml.YAMLError:
                    self.add_result("❌", f"Config: {config_file}", "Invalid YAML", "Fix syntax errors")
            else:
                self.add_result("⚠️", f"Config: {config_file}", "Missing", "May use defaults")
    
    def validate_api_connectivity(self):
        """Validate API connectivity"""
        env_file = self.project_root / '.env'
        if not env_file.exists():
            self.add_result("❌", "API Test", "No .env file", "Cannot test API")
            return
        
        try:
            # Load environment variables
            env_vars = {}
            with open(env_file) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
            
            api_key = env_vars.get('BYBIT_API_KEY', '').strip()
            api_secret = env_vars.get('BYBIT_API_SECRET', '').strip()
            
            if not api_key or not api_secret or api_key == 'your_api_key_here':
                self.add_result("❌", "API Credentials", "Not configured", "Add valid API keys")
                return
            
            # Test API connectivity (simple endpoint)
            headers = {
                'X-BAPI-API-KEY': api_key,
                'User-Agent': 'TradingBot/1.0'
            }
            
            response = requests.get(
                'https://api.bybit.com/v5/market/time',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.add_result("✅", "API Connectivity", "Connected", "API endpoint reachable")
                
                # Test with authenticated endpoint
                import hmac
                import hashlib
                import time
                
                timestamp = str(int(time.time() * 1000))
                param_str = f'timestamp={timestamp}'
                signature = hmac.new(
                    api_secret.encode('utf-8'),
                    param_str.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                auth_headers = {
                    'X-BAPI-API-KEY': api_key,
                    'X-BAPI-SIGN': signature,
                    'X-BAPI-TIMESTAMP': timestamp,
                    'X-BAPI-RECV-WINDOW': '5000'
                }
                
                auth_response = requests.get(
                    f'https://api.bybit.com/v5/account/wallet-balance?accountType=UNIFIED&{param_str}',
                    headers=auth_headers,
                    timeout=10
                )
                
                if auth_response.status_code == 200:
                    self.add_result("✅", "API Authentication", "Valid", "Credentials work")
                else:
                    self.add_result("❌", "API Authentication", "Failed", "Check API key permissions")
                    
            else:
                self.add_result("❌", "API Connectivity", "Failed", "Check internet connection")
                
        except requests.RequestException:
            self.add_result("❌", "API Test", "Connection failed", "Check internet/firewall")
        except Exception as e:
            self.add_result("❌", "API Test", "Error", f"Unexpected error: {str(e)}")
    
    def validate_bot_components(self):
        """Validate bot components"""
        # Check main application files
        main_files = ['start_api.py', 'start_dashboard.py']
        for file_name in main_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                self.add_result("✅", f"Component: {file_name}", "Found", "Application component ready")
            else:
                self.add_result("❌", f"Component: {file_name}", "Missing", "Critical file missing")
        
        # Check src directory
        src_dir = self.project_root / 'src'
        if src_dir.exists():
            self.add_result("✅", "Source Code", "Found", "Bot source code present")
        else:
            self.add_result("⚠️", "Source Code", "Not found", "May be in different structure")
        
        # Check logs directory
        logs_dir = self.project_root / 'logs'
        if logs_dir.exists():
            self.add_result("✅", "Logs Directory", "Found", "Logging configured")
        else:
            logs_dir.mkdir(exist_ok=True)
            self.add_result("✅", "Logs Directory", "Created", "Logging ready")
    
    def validate_security(self):
        """Validate security configuration"""
        # Check .env file permissions (Unix-like systems)
        env_file = self.project_root / '.env'
        if env_file.exists():
            if os.name != 'nt':  # Not Windows
                import stat
                file_stat = env_file.stat()
                permissions = stat.filemode(file_stat.st_mode)
                if '600' in oct(file_stat.st_mode)[-3:]:
                    self.add_result("✅", "File Permissions", "Secure", ".env has proper permissions")
                else:
                    self.add_result("⚠️", "File Permissions", "World readable", "chmod 600 .env recommended")
            else:
                self.add_result("✅", "File Permissions", "Windows", "Check NTFS permissions manually")
        
        # Check for sensitive data in git
        gitignore_file = self.project_root / '.gitignore'
        if gitignore_file.exists():
            with open(gitignore_file) as f:
                gitignore_content = f.read()
            
            if '.env' in gitignore_content:
                self.add_result("✅", "Git Security", ".env ignored", "Secrets protected from git")
            else:
                self.add_result("⚠️", "Git Security", ".env not ignored", "Add .env to .gitignore")
        else:
            self.add_result("⚠️", "Git Security", "No .gitignore", "Create .gitignore file")
    
    def add_result(self, status: str, category: str, value: str, description: str):
        """Add validation result"""
        self.validation_results.append({
            'status': status,
            'category': category,
            'value': value,
            'description': description
        })
    
    def show_results(self):
        """Show validation results"""
        self.console.print("\n📊 Validation Results", style="bold blue")
        
        results_table = Table()
        results_table.add_column("Status", style="white", width=8)
        results_table.add_column("Category", style="cyan", width=25)
        results_table.add_column("Value", style="white", width=20)
        results_table.add_column("Description", style="yellow")
        
        for result in self.validation_results:
            results_table.add_row(
                result['status'],
                result['category'],
                result['value'],
                result['description']
            )
        
        self.console.print(results_table)
    
    def provide_recommendations(self):
        """Provide recommendations based on validation results"""
        # Count status types
        passed = len([r for r in self.validation_results if r['status'] == '✅'])
        warnings = len([r for r in self.validation_results if r['status'] == '⚠️'])
        failures = len([r for r in self.validation_results if r['status'] == '❌'])
        
        total = len(self.validation_results)
        
        # Show summary
        summary_text = Text()
        summary_text.append(f"Validation Summary: {passed}/{total} passed", style="green")
        if warnings > 0:
            summary_text.append(f", {warnings} warnings", style="yellow")
        if failures > 0:
            summary_text.append(f", {failures} failures", style="red")
        
        summary_panel = Panel(
            summary_text,
            title="📈 Summary",
            border_style="blue"
        )
        self.console.print(summary_panel)
        
        # Provide specific recommendations
        recommendations = []
        
        if failures > 0:
            recommendations.append("🔴 **Critical Issues Found**")
            for result in self.validation_results:
                if result['status'] == '❌':
                    recommendations.append(f"   • {result['category']}: {result['description']}")
            recommendations.append("")
        
        if warnings > 0:
            recommendations.append("🟡 **Warnings (Optional Improvements)**")
            for result in self.validation_results:
                if result['status'] == '⚠️':
                    recommendations.append(f"   • {result['category']}: {result['description']}")
            recommendations.append("")
        
        if failures == 0 and warnings == 0:
            recommendations.append("🎉 **Excellent! Your setup is perfect!**")
            recommendations.append("✅ All validations passed")
            recommendations.append("🚀 Your trading bot is ready to run!")
            recommendations.append("")
            recommendations.append("Next steps:")
            recommendations.append("1. Start the API: `python start_api.py`")
            recommendations.append("2. Start the dashboard: `python start_dashboard.py`")
            recommendations.append("3. Visit http://localhost:8501 to see your bot!")
        elif failures == 0:
            recommendations.append("✅ **Setup is working! Just some minor improvements suggested.**")
            recommendations.append("🚀 Your trading bot should work fine as-is.")
        else:
            recommendations.append("❌ **Please fix the critical issues before running your bot.**")
            recommendations.append("")
            recommendations.append("Quick fixes:")
            recommendations.append("• Run setup wizard: `python setup_wizard.py`")
            recommendations.append("• Install dependencies: `python quick_start.py`")
            recommendations.append("• Check API credentials in .env file")
        
        if recommendations:
            rec_text = "\n".join(recommendations)
            rec_panel = Panel(
                rec_text,
                title="💡 Recommendations",
                border_style="yellow"
            )
            self.console.print(rec_panel)

def main():
    """Main entry point"""
    validator = SetupValidator()
    validator.run()

if __name__ == "__main__":
    main()