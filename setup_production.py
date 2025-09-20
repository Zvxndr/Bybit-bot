#!/usr/bin/env python3
"""
Production Environment Setup Script

Automated script for setting up production environment with proper
configuration, security settings, and dependency installation.

This script:
1. Validates system requirements
2. Sets up Python virtual environment
3. Installs production dependencies
4. Configures environment variables and secrets
5. Sets up logging directories with proper permissions
6. Validates configuration
7. Runs basic health checks

Usage:
    python setup_production.py [--environment ENV] [--force] [--dry-run]

Examples:
    # Setup production environment
    python setup_production.py --environment production
    
    # Setup staging environment
    python setup_production.py --environment staging
    
    # Dry run to see what would be done
    python setup_production.py --environment production --dry-run
"""

import os
import sys
import subprocess
import argparse
import shutil
import stat
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import platform


class ProductionSetup:
    """Production environment setup manager."""
    
    def __init__(self, environment: str = "production", force: bool = False, dry_run: bool = False):
        self.environment = environment
        self.force = force
        self.dry_run = dry_run
        self.platform = platform.system().lower()
        
        # Paths
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.logs_path = self.project_root / "logs"
        self.config_path = self.project_root / "config"
        
        # Requirements
        self.python_min_version = (3, 8)
        self.required_commands = ["git", "docker"]
        
        print(f"🚀 Production Setup for {environment} environment")
        print(f"   Platform: {platform.system()} {platform.release()}")
        print(f"   Project root: {self.project_root}")
        if dry_run:
            print("   Mode: DRY RUN (no changes will be made)")
        print("=" * 60)
    
    def run_setup(self) -> bool:
        """Run complete production setup process."""
        try:
            steps = [
                ("System Requirements", self._check_system_requirements),
                ("Python Environment", self._setup_python_environment),
                ("Dependencies", self._install_dependencies),
                ("Directory Structure", self._setup_directories),
                ("Configuration", self._setup_configuration),
                ("Environment Variables", self._setup_environment_variables),
                ("Permissions", self._setup_permissions),
                ("Health Check", self._run_health_check)
            ]
            
            for step_name, step_func in steps:
                print(f"\n📋 {step_name}...")
                if not step_func():
                    print(f"❌ {step_name} failed!")
                    return False
                print(f"✅ {step_name} completed")
            
            print(f"\n🎉 Production setup completed successfully!")
            self._print_next_steps()
            return True
            
        except Exception as e:
            print(f"❌ Setup failed with error: {e}")
            return False
    
    def _check_system_requirements(self) -> bool:
        """Check system requirements and dependencies."""
        try:
            # Check Python version
            python_version = sys.version_info[:2]
            if python_version < self.python_min_version:
                print(f"❌ Python {'.'.join(map(str, self.python_min_version))}+ required, got {'.'.join(map(str, python_version))}")
                return False
            print(f"   Python version: {'.'.join(map(str, python_version))} ✅")
            
            # Check required commands
            missing_commands = []
            for cmd in self.required_commands:
                if not shutil.which(cmd):
                    missing_commands.append(cmd)
            
            if missing_commands:
                print(f"❌ Missing required commands: {', '.join(missing_commands)}")
                return False
            
            print(f"   Required commands available: {', '.join(self.required_commands)} ✅")
            
            # Check disk space
            disk_usage = shutil.disk_usage(self.project_root)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 5.0:  # Require at least 5GB free
                print(f"❌ Insufficient disk space: {free_gb:.1f}GB free (5GB+ required)")
                return False
            print(f"   Disk space: {free_gb:.1f}GB free ✅")
            
            # Check memory
            try:
                if self.platform == "linux":
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                total_kb = int(line.split()[1])
                                total_gb = total_kb / (1024**2)
                                if total_gb < 4.0:  # Require at least 4GB RAM
                                    print(f"❌ Insufficient memory: {total_gb:.1f}GB (4GB+ required)")
                                    return False
                                print(f"   Memory: {total_gb:.1f}GB ✅")
                                break
            except Exception:
                print("   Memory check skipped (unable to determine)")
            
            return True
            
        except Exception as e:
            print(f"❌ System requirements check failed: {e}")
            return False
    
    def _setup_python_environment(self) -> bool:
        """Setup Python virtual environment."""
        try:
            if self.venv_path.exists():
                if self.force:
                    print("   Removing existing virtual environment...")
                    if not self.dry_run:
                        shutil.rmtree(self.venv_path)
                else:
                    print("   Virtual environment already exists (use --force to recreate)")
                    return True
            
            if not self.dry_run:
                print("   Creating virtual environment...")
                subprocess.run([
                    sys.executable, "-m", "venv", str(self.venv_path)
                ], check=True, capture_output=True)
                
                # Activate virtual environment and upgrade pip
                if self.platform == "windows":
                    pip_path = self.venv_path / "Scripts" / "pip.exe"
                    python_path = self.venv_path / "Scripts" / "python.exe"
                else:
                    pip_path = self.venv_path / "bin" / "pip"
                    python_path = self.venv_path / "bin" / "python"
                
                print("   Upgrading pip...")
                subprocess.run([
                    str(python_path), "-m", "pip", "install", "--upgrade", "pip"
                ], check=True, capture_output=True)
                
                print("   Installing wheel and setuptools...")
                subprocess.run([
                    str(pip_path), "install", "--upgrade", "wheel", "setuptools"
                ], check=True, capture_output=True)
            
            print("   Virtual environment ready")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup Python environment: {e}")
            if e.stdout:
                print(f"   stdout: {e.stdout.decode()}")
            if e.stderr:
                print(f"   stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"❌ Python environment setup failed: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Install production dependencies."""
        try:
            requirements_files = [
                "requirements.txt",
                f"requirements-{self.environment}.txt"
            ]
            
            if self.platform == "windows":
                pip_path = self.venv_path / "Scripts" / "pip.exe"
            else:
                pip_path = self.venv_path / "bin" / "pip"
            
            for req_file in requirements_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    print(f"   Installing dependencies from {req_file}...")
                    if not self.dry_run:
                        subprocess.run([
                            str(pip_path), "install", "-r", str(req_path)
                        ], check=True, capture_output=True)
                else:
                    print(f"   Requirements file not found: {req_file}")
            
            # Install additional production dependencies
            production_deps = [
                "gunicorn",
                "uvicorn[standard]", 
                "prometheus-client",
                "psutil",
                "cryptography"
            ]
            
            print("   Installing production-specific dependencies...")
            if not self.dry_run:
                subprocess.run([
                    str(pip_path), "install"
                ] + production_deps, check=True, capture_output=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            if e.stderr:
                print(f"   stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"❌ Dependency installation failed: {e}")
            return False
    
    def _setup_directories(self) -> bool:
        """Setup required directory structure."""
        try:
            directories = [
                self.logs_path,
                self.logs_path / "archive",
                self.config_path,
                self.project_root / "data",
                self.project_root / "models", 
                self.project_root / "backups",
                self.project_root / "tmp"
            ]
            
            for directory in directories:
                if not directory.exists():
                    print(f"   Creating directory: {directory}")
                    if not self.dry_run:
                        directory.mkdir(parents=True, exist_ok=True)
                else:
                    print(f"   Directory exists: {directory}")
            
            return True
            
        except Exception as e:
            print(f"❌ Directory setup failed: {e}")
            return False
    
    def _setup_configuration(self) -> bool:
        """Setup configuration files."""
        try:
            # Check if environment-specific config exists
            env_config = self.config_path / f"{self.environment}.yaml"
            if not env_config.exists():
                print(f"❌ Environment configuration not found: {env_config}")
                print("   Please create the configuration file before running setup")
                return False
            
            print(f"   Environment configuration found: {env_config}")
            
            # Check for secrets template
            secrets_template = self.config_path / "secrets.yaml.template"
            secrets_file = self.config_path / "secrets.yaml"
            
            if secrets_template.exists() and not secrets_file.exists():
                print("   Creating secrets file from template...")
                if not self.dry_run:
                    shutil.copy2(secrets_template, secrets_file)
                    secrets_file.chmod(0o600)
                    print(f"   ⚠️  Please edit {secrets_file} with actual secret values")
            
            # Test configuration loading
            try:
                if not self.dry_run:
                    sys.path.append(str(self.project_root))
                    from src.bot.config.production import ProductionConfigManager, Environment
                    
                    env_enum = Environment(self.environment)
                    config_manager = ProductionConfigManager(environment=env_enum)
                    print("   Configuration validation passed")
            except Exception as e:
                print(f"   ⚠️  Configuration validation warning: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration setup failed: {e}")
            return False
    
    def _setup_environment_variables(self) -> bool:
        """Setup environment variables."""
        try:
            env_file = self.project_root / f".env.{self.environment}"
            
            required_env_vars = [
                "TRADING_BOT_ENV",
                "TRADING_BOT_MASTER_KEY",
                "DATABASE_PASSWORD",
                "BYBIT_API_KEY",
                "BYBIT_API_SECRET",
                "API_SECRET_KEY",
                "JWT_SECRET_KEY"
            ]
            
            missing_vars = []
            for var in required_env_vars:
                if not os.environ.get(var):
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"   ⚠️  Missing environment variables: {', '.join(missing_vars)}")
                
                # Create example .env file
                if not env_file.exists():
                    print(f"   Creating example environment file: {env_file}")
                    if not self.dry_run:
                        with open(env_file, 'w') as f:
                            f.write(f"# Environment variables for {self.environment}\n")
                            f.write(f"TRADING_BOT_ENV={self.environment}\n")
                            for var in missing_vars:
                                f.write(f"{var}=your-{var.lower().replace('_', '-')}-here\n")
                        
                        env_file.chmod(0o600)
                        print(f"   Please edit {env_file} with actual values")
            else:
                print("   All required environment variables are set")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment variables setup failed: {e}")
            return False
    
    def _setup_permissions(self) -> bool:
        """Setup proper file and directory permissions."""
        try:
            if self.platform != "windows":
                # Set directory permissions
                permission_dirs = [
                    (self.logs_path, 0o755),
                    (self.config_path, 0o755),
                    (self.project_root / "data", 0o755),
                    (self.project_root / "models", 0o755),
                    (self.project_root / "backups", 0o700)
                ]
                
                for directory, mode in permission_dirs:
                    if directory.exists():
                        print(f"   Setting permissions for {directory}: {oct(mode)}")
                        if not self.dry_run:
                            directory.chmod(mode)
                
                # Set file permissions
                sensitive_files = [
                    self.config_path / "secrets.yaml",
                    self.project_root / f".env.{self.environment}"
                ]
                
                for file_path in sensitive_files:
                    if file_path.exists():
                        print(f"   Setting secure permissions for {file_path}: 0o600")
                        if not self.dry_run:
                            file_path.chmod(0o600)
            else:
                print("   Permission setup skipped on Windows")
            
            return True
            
        except Exception as e:
            print(f"❌ Permission setup failed: {e}")
            return False
    
    def _run_health_check(self) -> bool:
        """Run basic health checks."""
        try:
            print("   Running health checks...")
            
            # Check if we can import our modules
            try:
                if not self.dry_run:
                    sys.path.append(str(self.project_root))
                    from src.bot.config.production import ProductionConfigManager
                    print("   Module imports: ✅")
            except ImportError as e:
                print(f"   Module import failed: {e}")
                return False
            
            # Check file permissions
            sensitive_files = [
                self.config_path / "secrets.yaml",
                self.project_root / f".env.{self.environment}"
            ]
            
            for file_path in sensitive_files:
                if file_path.exists():
                    file_stat = file_path.stat()
                    if self.platform != "windows":
                        mode = stat.filemode(file_stat.st_mode)
                        if file_stat.st_mode & 0o077:  # Check if group/other can read
                            print(f"   ⚠️  File {file_path} has overly permissive permissions: {mode}")
                    print(f"   File security check: {file_path} ✅")
            
            # Check directory structure
            required_dirs = [self.logs_path, self.config_path, self.venv_path]
            for directory in required_dirs:
                if not directory.exists():
                    print(f"   ❌ Missing required directory: {directory}")
                    return False
            print("   Directory structure: ✅")
            
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def _print_next_steps(self):
        """Print next steps for the user."""
        print("\n📝 Next Steps:")
        print("=" * 40)
        print("1. Edit configuration files:")
        print(f"   - {self.config_path / f'{self.environment}.yaml'}")
        print(f"   - {self.config_path / 'secrets.yaml'}")
        print()
        print("2. Set environment variables:")
        print(f"   - Source {self.project_root / f'.env.{self.environment}'}")
        print("   - Or set variables in your deployment system")
        print()
        print("3. Test the configuration:")
        print(f"   python config_cli.py validate --environment {self.environment}")
        print()
        print("4. Start the services:")
        print("   docker-compose up -d")
        print("   # OR")
        print("   python -m src.bot.api.prediction_service")
        print()
        print("5. Monitor the logs:")
        print(f"   tail -f {self.logs_path}/trading_bot.log")


def main():
    """Main setup entry point."""
    parser = argparse.ArgumentParser(
        description="Production Environment Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'testing', 'staging', 'production'],
        default='production',
        help='Environment to setup (default: production)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force overwrite existing files and directories'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    try:
        setup = ProductionSetup(
            environment=args.environment,
            force=args.force,
            dry_run=args.dry_run
        )
        
        success = setup.run_setup()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Setup cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Setup failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())