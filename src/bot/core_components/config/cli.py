"""
Unified Configuration CLI - Phase 4 Configuration Consolidation

Command-line interface for unified configuration management that consolidates
all configuration CLI functionality into a single, comprehensive tool.

Features:
- Configuration creation and initialization
- Environment management and switching
- Secrets encryption and decryption
- Configuration validation and testing
- Migration from existing configurations
- Configuration export and import
- Health checks and diagnostics
- Integration with Phase 1-3 systems

Consolidates:
- config_cli.py functionality
- Environment variable management
- Configuration file management
- Secrets handling
- Validation and testing
"""

import os
import sys
import json
import yaml
import click
import getpass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from tabulate import tabulate
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from .manager import (
        UnifiedConfigurationManager, create_configuration_manager,
        ValidationResult
    )
    from .schema import (
        UnifiedConfigurationSchema, Environment, TradingMode,
        ExchangeCredentials
    )
except ImportError:
    from manager import (
        UnifiedConfigurationManager, create_configuration_manager,
        ValidationResult
    )
    from schema import (
        UnifiedConfigurationSchema, Environment, TradingMode,
        ExchangeCredentials
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CLI CONTEXT
# ============================================================================

class CLIContext:
    """CLI context for sharing state between commands"""
    
    def __init__(self):
        self.workspace_root = os.getcwd()
        self.config_manager: Optional[UnifiedConfigurationManager] = None
        self.verbose = False
    
    def get_manager(self) -> UnifiedConfigurationManager:
        """Get or create configuration manager"""
        if not self.config_manager:
            self.config_manager = create_configuration_manager(self.workspace_root)
        return self.config_manager

# Global CLI context
cli_context = CLIContext()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_success(message: str):
    """Print success message"""
    click.echo(click.style(f"✓ {message}", fg='green'))

def print_error(message: str):
    """Print error message"""
    click.echo(click.style(f"✗ {message}", fg='red'))

def print_warning(message: str):
    """Print warning message"""
    click.echo(click.style(f"⚠ {message}", fg='yellow'))

def print_info(message: str):
    """Print info message"""
    click.echo(click.style(f"ℹ {message}", fg='blue'))

def print_validation_result(result: ValidationResult):
    """Print validation result"""
    if result.is_valid:
        print_success(f"Configuration is valid (score: {result.score:.2f})")
    else:
        print_error("Configuration validation failed:")
        for error in result.errors:
            click.echo(f"  • {error}")
    
    if result.warnings:
        print_warning("Configuration warnings:")
        for warning in result.warnings:
            click.echo(f"  • {warning}")

def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation"""
    return click.confirm(message, default=default)

def prompt_for_credentials() -> ExchangeCredentials:
    """Prompt user for exchange credentials"""
    print_info("Enter exchange credentials:")
    
    api_key = click.prompt("API Key", hide_input=True)
    api_secret = click.prompt("API Secret", hide_input=True)
    is_testnet = click.confirm("Use testnet?", default=True)
    
    return ExchangeCredentials(
        api_key=api_key,
        api_secret=api_secret,
        is_testnet=is_testnet
    )

# ============================================================================
# MAIN CLI GROUP
# ============================================================================

@click.group()
@click.option('--workspace', '-w', default=None, help='Workspace root directory')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(workspace, verbose):
    """
    Unified Configuration Manager CLI
    
    Comprehensive configuration management for the Bybit Trading Bot.
    Consolidates all configuration operations into a single interface.
    """
    if workspace:
        cli_context.workspace_root = os.path.abspath(workspace)
    
    cli_context.verbose = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_info(f"Using workspace: {cli_context.workspace_root}")

# ============================================================================
# INITIALIZATION COMMANDS
# ============================================================================

@cli.command()
@click.option('--environment', '-e', 
              type=click.Choice(['development', 'staging', 'production']),
              default='development',
              help='Target environment')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing configuration')
def init(environment, force):
    """Initialize new configuration for the trading bot"""
    
    env = Environment(environment)
    config_path = os.path.join(cli_context.workspace_root, "src/bot/core/config/config.yaml")
    
    # Check if configuration already exists
    if os.path.exists(config_path) and not force:
        if not confirm_action(f"Configuration already exists at {config_path}. Overwrite?"):
            print_info("Configuration initialization cancelled")
            return
    
    try:
        # Create configuration manager
        manager = cli_context.get_manager()
        
        # Create default configuration
        print_info(f"Creating default configuration for {environment} environment...")
        config = manager.create_default_configuration(environment=env)
        
        # Prompt for basic settings
        if click.confirm("Configure exchange credentials now?", default=True):
            credentials = prompt_for_credentials()
            config.exchange[env] = credentials
        
        # Ask about trading mode
        trading_mode = click.prompt(
            "Trading mode",
            type=click.Choice(['conservative', 'aggressive', 'hybrid']),
            default='conservative'
        )
        config.trading.mode = TradingMode(trading_mode)
        
        # Save configuration
        manager.save_configuration(config, config_path)
        print_success(f"Configuration initialized at {config_path}")
        
        # Create environment file template
        env_path = os.path.join(cli_context.workspace_root, f".env.{environment}")
        if not os.path.exists(env_path):
            create_env_template(env_path, env)
            print_success(f"Environment template created at {env_path}")
        
        # Validate configuration
        result = manager.validate_current_configuration()
        print_validation_result(result)
        
    except Exception as e:
        print_error(f"Failed to initialize configuration: {e}")
        if cli_context.verbose:
            import traceback
            traceback.print_exc()

def create_env_template(env_path: str, environment: Environment):
    """Create environment file template"""
    template = f"""# Environment configuration for {environment.value}
# Bybit Exchange Credentials
BYBIT_{environment.value.upper()}_API_KEY=your_api_key_here
BYBIT_{environment.value.upper()}_API_SECRET=your_api_secret_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_bot_{environment.value}
DB_USER=trader
DB_PASSWORD=your_db_password

# Security
SECRET_KEY=your_secret_key_here

# Email Configuration (for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@example.com
EMAIL_PASSWORD=your_email_password

# Redis Configuration (if using Redis cache)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
"""
    
    with open(env_path, 'w') as f:
        f.write(template)

@cli.command()
def migrate():
    """Migrate existing configuration files to unified format"""
    
    try:
        manager = cli_context.get_manager()
        
        print_info("Scanning for existing configuration files...")
        
        # Detect existing configurations
        existing_configs = manager.migrator.detect_existing_configs(cli_context.workspace_root)
        
        # Show what was found
        total_files = sum(len(files) for files in existing_configs.values())
        if total_files == 0:
            print_info("No existing configuration files found")
            return
        
        print_info(f"Found {total_files} configuration files:")
        for config_type, files in existing_configs.items():
            if files:
                print(f"  {config_type}: {len(files)} files")
                if cli_context.verbose:
                    for file in files:
                        print(f"    - {file}")
        
        # Confirm migration
        if not confirm_action("Proceed with migration?", default=True):
            print_info("Migration cancelled")
            return
        
        # Perform migration
        print_info("Starting migration...")
        merged_config = manager.migrate_existing_configs()
        
        # Create unified configuration
        if merged_config:
            # Load existing or create new configuration
            try:
                config = manager.load_configuration()
            except:
                config = manager.create_default_configuration()
            
            # Update with migrated data
            config_dict = config.dict()
            config_dict.update(merged_config)
            
            # Create new configuration
            updated_config = UnifiedConfigurationSchema(**config_dict)
            
            # Save unified configuration
            unified_path = os.path.join(cli_context.workspace_root, "src/bot/core/config/config.yaml")
            manager.save_configuration(updated_config, unified_path)
            
            print_success(f"Migration completed. Unified configuration saved to {unified_path}")
        else:
            print_warning("No configuration data to migrate")
        
    except Exception as e:
        print_error(f"Migration failed: {e}")
        if cli_context.verbose:
            import traceback
            traceback.print_exc()

# ============================================================================
# ENVIRONMENT MANAGEMENT
# ============================================================================

@cli.command()
@click.argument('environment', type=click.Choice(['development', 'staging', 'production']))
def use(environment):
    """Switch to specified environment"""
    
    try:
        env = Environment(environment)
        manager = cli_context.get_manager()
        
        # Load configuration for the environment
        config = manager.load_configuration(environment=env)
        
        # Save current environment selection
        env_file = os.path.join(cli_context.workspace_root, ".current_env")
        with open(env_file, 'w') as f:
            f.write(environment)
        
        print_success(f"Switched to {environment} environment")
        
        # Show configuration summary
        summary = config.get_summary()
        print_info(f"Configuration summary:")
        print(f"  Environment: {summary['environment']}")
        print(f"  Trading enabled: {summary['trading_enabled']}")
        print(f"  Trading mode: {summary['trading_mode']}")
        print(f"  Credentials configured: {summary['has_credentials']}")
        
    except Exception as e:
        print_error(f"Failed to switch environment: {e}")

@cli.command()
def current():
    """Show current environment and configuration"""
    
    try:
        manager = cli_context.get_manager()
        config = manager.get_configuration()
        
        if not config:
            print_warning("No configuration loaded")
            return
        
        summary = manager.get_configuration_summary()
        
        # Display summary table
        data = [
            ['Environment', summary.get('environment', 'Unknown')],
            ['Version', summary.get('version', 'Unknown')],
            ['Trading Enabled', 'Yes' if summary.get('trading_enabled') else 'No'],
            ['Trading Mode', summary.get('trading_mode', 'Unknown')],
            ['Credentials Valid', 'Yes' if summary.get('credentials_valid') else 'No'],
            ['ML Enabled', 'Yes' if summary.get('ml_enabled') else 'No'],
            ['WebSockets Enabled', 'Yes' if summary.get('websockets_enabled') else 'No'],
            ['Database', summary.get('database_dialect', 'Unknown')],
            ['Cache Backend', summary.get('cache_backend', 'Unknown')],
            ['Log Level', summary.get('log_level', 'Unknown')],
            ['Validation Score', f"{summary.get('validation_score', 0):.2f}"],
            ['Last Loaded', summary.get('last_loaded', 'Never')],
        ]
        
        print("\nCurrent Configuration:")
        print(tabulate(data, headers=['Setting', 'Value'], tablefmt='grid'))
        
    except Exception as e:
        print_error(f"Failed to get current configuration: {e}")

# ============================================================================
# VALIDATION COMMANDS
# ============================================================================

@cli.command()
@click.option('--connectivity', '-c', is_flag=True, help='Test external connectivity')
def validate(connectivity):
    """Validate current configuration"""
    
    try:
        manager = cli_context.get_manager()
        
        # Schema validation
        print_info("Validating configuration schema...")
        schema_result = manager.validate_current_configuration()
        print_validation_result(schema_result)
        
        # Connectivity validation
        if connectivity:
            print_info("Testing external connectivity...")
            connectivity_result = manager.validator.validate_connectivity(manager.get_configuration())
            print_validation_result(connectivity_result)
        
        # Overall result
        overall_score = schema_result.score
        if connectivity:
            overall_score = (schema_result.score + connectivity_result.score) / 2
        
        print_info(f"\nOverall validation score: {overall_score:.2f}")
        
        if overall_score >= 0.9:
            print_success("Configuration is excellent!")
        elif overall_score >= 0.7:
            print_warning("Configuration has minor issues")
        else:
            print_error("Configuration needs attention")
        
    except Exception as e:
        print_error(f"Validation failed: {e}")

@cli.command()
def test():
    """Run comprehensive configuration tests"""
    
    try:
        manager = cli_context.get_manager()
        config = manager.get_configuration()
        
        if not config:
            print_error("No configuration loaded")
            return
        
        print_info("Running comprehensive configuration tests...\n")
        
        tests = [
            ("Schema Validation", lambda: manager.validate_current_configuration()),
            ("Credentials Test", lambda: test_credentials(config)),
            ("Database Connection", lambda: test_database(config)),
            ("Cache Configuration", lambda: test_cache(config)),
            ("Trading Configuration", lambda: test_trading(config)),
            ("ML Configuration", lambda: test_ml(config)),
        ]
        
        results = []
        for test_name, test_func in tests:
            print_info(f"Running {test_name}...")
            try:
                result = test_func()
                results.append((test_name, result))
                if result.is_valid:
                    print_success(f"{test_name}: PASS")
                else:
                    print_error(f"{test_name}: FAIL")
                    for error in result.errors:
                        print(f"  - {error}")
            except Exception as e:
                results.append((test_name, ValidationResult(False, [str(e)], [], 0.0)))
                print_error(f"{test_name}: ERROR - {e}")
        
        # Summary
        passed = sum(1 for _, result in results if result.is_valid)
        total = len(results)
        overall_score = sum(result.score for _, result in results) / total if results else 0
        
        print_info(f"\nTest Summary: {passed}/{total} tests passed")
        print_info(f"Overall score: {overall_score:.2f}")
        
    except Exception as e:
        print_error(f"Test run failed: {e}")

def test_credentials(config: UnifiedConfigurationSchema) -> ValidationResult:
    """Test exchange credentials"""
    errors = []
    warnings = []
    
    creds = config.get_current_credentials()
    if not creds:
        errors.append("No credentials configured for current environment")
    else:
        if not creds.api_key:
            errors.append("API key is empty")
        elif len(creds.api_key) < 10:
            warnings.append("API key seems too short")
        
        if not creds.api_secret:
            errors.append("API secret is empty")
        elif len(creds.api_secret) < 10:
            warnings.append("API secret seems too short")
    
    return ValidationResult(len(errors) == 0, errors, warnings, 1.0 if len(errors) == 0 else 0.0)

def test_database(config: UnifiedConfigurationSchema) -> ValidationResult:
    """Test database configuration"""
    errors = []
    warnings = []
    
    db_config = config.database
    
    if db_config.dialect.value == 'postgresql':
        if not db_config.host:
            errors.append("Database host is required for PostgreSQL")
        if not db_config.username:
            errors.append("Database username is required for PostgreSQL")
        if not db_config.password:
            warnings.append("Database password is empty")
    
    return ValidationResult(len(errors) == 0, errors, warnings, 1.0 if len(errors) == 0 else 0.5)

def test_cache(config: UnifiedConfigurationSchema) -> ValidationResult:
    """Test cache configuration"""
    errors = []
    warnings = []
    
    cache_config = config.cache
    
    if cache_config.backend.value == 'redis':
        if not cache_config.redis_host:
            errors.append("Redis host is required for Redis backend")
        if cache_config.redis_port <= 0:
            errors.append("Invalid Redis port")
    
    return ValidationResult(len(errors) == 0, errors, warnings, 1.0 if len(errors) == 0 else 0.5)

def test_trading(config: UnifiedConfigurationSchema) -> ValidationResult:
    """Test trading configuration"""
    errors = []
    warnings = []
    
    if config.enable_trading and not config.validate_credentials():
        errors.append("Trading is enabled but credentials are invalid")
    
    if config.trading.base_balance <= 0:
        errors.append("Base balance must be positive")
    
    return ValidationResult(len(errors) == 0, errors, warnings, 1.0 if len(errors) == 0 else 0.5)

def test_ml(config: UnifiedConfigurationSchema) -> ValidationResult:
    """Test ML configuration"""
    errors = []
    warnings = []
    
    if config.enable_ml_integration:
        if not config.ml.models:
            warnings.append("No ML models configured")
        
        if not config.ml.technical_indicators:
            warnings.append("No technical indicators configured")
    
    return ValidationResult(len(errors) == 0, errors, warnings, 1.0 if len(warnings) == 0 else 0.8)

# ============================================================================
# SECRETS MANAGEMENT
# ============================================================================

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path')
def encrypt(config_file, output):
    """Encrypt sensitive values in configuration file"""
    
    try:
        manager = cli_context.get_manager()
        
        # Load configuration file
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        # Encrypt secrets
        encrypted_data = manager.secrets_manager.encrypt_secrets(data)
        encrypted_data['encrypted'] = True
        
        # Save encrypted configuration
        output_file = output or config_file
        with open(output_file, 'w') as f:
            if output_file.endswith('.json'):
                json.dump(encrypted_data, f, indent=2)
            else:
                yaml.dump(encrypted_data, f, default_flow_style=False)
        
        print_success(f"Configuration encrypted and saved to {output_file}")
        
    except Exception as e:
        print_error(f"Encryption failed: {e}")

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path')
def decrypt(config_file, output):
    """Decrypt configuration file"""
    
    try:
        manager = cli_context.get_manager()
        
        # Load encrypted configuration
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        # Decrypt secrets
        decrypted_data = manager.secrets_manager.decrypt_secrets(data)
        
        # Remove encryption flag
        if 'encrypted' in decrypted_data:
            del decrypted_data['encrypted']
        
        # Save decrypted configuration
        output_file = output or f"{config_file}.decrypted"
        with open(output_file, 'w') as f:
            if output_file.endswith('.json'):
                json.dump(decrypted_data, f, indent=2)
            else:
                yaml.dump(decrypted_data, f, default_flow_style=False)
        
        print_success(f"Configuration decrypted and saved to {output_file}")
        print_warning("Decrypted file contains sensitive information!")
        
    except Exception as e:
        print_error(f"Decryption failed: {e}")

# ============================================================================
# IMPORT/EXPORT COMMANDS
# ============================================================================

@cli.command()
@click.argument('output_path', type=click.Path())
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Output format')
@click.option('--include-secrets', is_flag=True, help='Include sensitive information')
def export(output_path, format, include_secrets):
    """Export current configuration to file"""
    
    try:
        manager = cli_context.get_manager()
        
        if include_secrets and not confirm_action(
            "Export will include sensitive information. Continue?", default=False
        ):
            print_info("Export cancelled")
            return
        
        manager.export_configuration(
            output_path=output_path,
            format=format,
            include_secrets=include_secrets
        )
        
        print_success(f"Configuration exported to {output_path}")
        
        if not include_secrets:
            print_info("Sensitive information was redacted from export")
        
    except Exception as e:
        print_error(f"Export failed: {e}")

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--merge', is_flag=True, help='Merge with existing configuration')
def import_config(config_file, merge):
    """Import configuration from file"""
    
    try:
        manager = cli_context.get_manager()
        
        # Load configuration from file
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        if merge:
            # Merge with existing configuration
            existing_config = manager.get_configuration()
            if existing_config:
                existing_data = existing_config.to_dict()
                existing_data.update(data)
                data = existing_data
        
        # Create new configuration
        config = UnifiedConfigurationSchema(**data)
        
        # Save configuration
        manager.save_configuration(config)
        
        print_success(f"Configuration imported from {config_file}")
        
        # Validate imported configuration
        result = manager.validate_current_configuration()
        print_validation_result(result)
        
    except Exception as e:
        print_error(f"Import failed: {e}")

# ============================================================================
# CREDENTIALS MANAGEMENT
# ============================================================================

@cli.command()
@click.argument('environment', type=click.Choice(['development', 'staging', 'production']))
def set_credentials(environment):
    """Set exchange credentials for environment"""
    
    try:
        env = Environment(environment)
        manager = cli_context.get_manager()
        
        # Get current configuration
        config = manager.get_configuration()
        if not config:
            config = manager.create_default_configuration()
        
        # Prompt for credentials
        credentials = prompt_for_credentials()
        
        # Update configuration
        config.exchange[env] = credentials
        
        # Save configuration
        manager.save_configuration(config)
        
        print_success(f"Credentials set for {environment} environment")
        
    except Exception as e:
        print_error(f"Failed to set credentials: {e}")

@cli.command()
def list_credentials():
    """List configured credentials (without showing sensitive data)"""
    
    try:
        manager = cli_context.get_manager()
        config = manager.get_configuration()
        
        if not config:
            print_warning("No configuration loaded")
            return
        
        # Show credentials status for each environment
        data = []
        for env in Environment:
            creds = config.exchange.get(env)
            if creds:
                has_key = bool(creds.api_key)
                has_secret = bool(creds.api_secret)
                is_testnet = creds.is_testnet
                status = "Complete" if has_key and has_secret else "Incomplete"
            else:
                has_key = has_secret = is_testnet = False
                status = "Not configured"
            
            data.append([
                env.value.title(),
                "Yes" if has_key else "No",
                "Yes" if has_secret else "No", 
                "Yes" if is_testnet else "No",
                status
            ])
        
        print("\nConfigured Credentials:")
        print(tabulate(data, 
                      headers=['Environment', 'Has API Key', 'Has Secret', 'Testnet', 'Status'],
                      tablefmt='grid'))
        
    except Exception as e:
        print_error(f"Failed to list credentials: {e}")

# ============================================================================
# MONITORING AND HEALTH
# ============================================================================

@cli.command()
def health():
    """Check configuration health and system status"""
    
    try:
        manager = cli_context.get_manager()
        config = manager.get_configuration()
        
        if not config:
            print_error("No configuration loaded")
            return
        
        print_info("Configuration Health Check")
        print("=" * 50)
        
        # Configuration validation
        validation = manager.validate_current_configuration()
        print(f"Validation Score: {validation.score:.2f}/1.0")
        
        if validation.is_valid:
            print_success("Configuration is valid")
        else:
            print_error("Configuration has errors:")
            for error in validation.errors:
                print(f"  - {error}")
        
        if validation.warnings:
            print_warning("Configuration warnings:")
            for warning in validation.warnings:
                print(f"  - {warning}")
        
        print()
        
        # System status
        summary = manager.get_configuration_summary()
        
        status_items = [
            ("Environment", summary.get('environment')),
            ("Trading Status", "Enabled" if summary.get('trading_enabled') else "Disabled"),
            ("Credentials", "Valid" if summary.get('credentials_valid') else "Invalid"),
            ("Hot Reload", "Enabled" if summary.get('hot_reload_enabled') else "Disabled"),
            ("Last Updated", summary.get('updated_at', 'Never')),
        ]
        
        print("System Status:")
        for item, value in status_items:
            color = 'green' if value in ['Enabled', 'Valid'] else 'yellow' if value == 'Disabled' else 'white'
            print(f"  {item}: {click.style(str(value), fg=color)}")
        
    except Exception as e:
        print_error(f"Health check failed: {e}")

# ============================================================================
# UTILITY COMMANDS
# ============================================================================

@cli.command()
def version():
    """Show configuration manager version"""
    
    try:
        manager = cli_context.get_manager()
        config = manager.get_configuration()
        
        print(f"Unified Configuration Manager")
        print(f"Workspace: {cli_context.workspace_root}")
        
        if config:
            print(f"Configuration Version: {config.version}")
            print(f"Environment: {config.environment.value}")
        else:
            print("No configuration loaded")
        
    except Exception as e:
        print_error(f"Failed to get version: {e}")

@cli.command()
def clean():
    """Clean up temporary and backup files"""
    
    try:
        workspace = Path(cli_context.workspace_root)
        
        # Find cleanup targets
        cleanup_patterns = [
            "config_backup/*/",
            "*.decrypted",
            "*.tmp",
            ".current_env"
        ]
        
        files_to_clean = []
        for pattern in cleanup_patterns:
            files_to_clean.extend(workspace.glob(pattern))
        
        if not files_to_clean:
            print_info("No files to clean up")
            return
        
        print_info(f"Found {len(files_to_clean)} files/directories to clean:")
        for file_path in files_to_clean:
            print(f"  - {file_path}")
        
        if confirm_action("Delete these files?", default=False):
            for file_path in files_to_clean:
                try:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
                    print_success(f"Deleted {file_path}")
                except Exception as e:
                    print_error(f"Failed to delete {file_path}: {e}")
        else:
            print_info("Cleanup cancelled")
        
    except Exception as e:
        print_error(f"Cleanup failed: {e}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        print_info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if cli_context.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)