#!/usr/bin/env python3
"""
Configuration Management CLI Tool

Command-line interface for managing production configuration,
including encryption/decryption of secrets, validation,
and environment setup.

Usage:
    python config_cli.py validate [--environment ENV]
    python config_cli.py encrypt-secrets [--input FILE] [--output FILE]
    python config_cli.py decrypt-secrets [--input FILE] [--output FILE]
    python config_cli.py generate-keys
    python config_cli.py export [--environment ENV] [--include-secrets]
    python config_cli.py test [--environment ENV]

Examples:
    # Validate production configuration
    python config_cli.py validate --environment production
    
    # Encrypt secrets file
    python config_cli.py encrypt-secrets --input secrets.yaml --output secrets.encrypted.yaml
    
    # Generate new encryption keys
    python config_cli.py generate-keys
    
    # Export configuration summary
    python config_cli.py export --environment staging
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any
import secrets
import getpass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.bot.config.production import (
    ProductionConfigManager, 
    Environment, 
    SecretsManager
)


class ConfigCLI:
    """Command-line interface for configuration management."""
    
    def __init__(self):
        self.config_path = Path("config")
        self.config_path.mkdir(exist_ok=True)
    
    def validate_config(self, environment: Optional[str] = None) -> bool:
        """Validate configuration for specified environment."""
        try:
            env = Environment(environment) if environment else Environment.DEVELOPMENT
            print(f"🔍 Validating {env.value} configuration...")
            
            config_manager = ProductionConfigManager(environment=env)
            
            print("✅ Configuration validation passed!")
            
            # Show configuration summary
            summary = config_manager.get_config_summary()
            print("\n📋 Configuration Summary:")
            for section, values in summary.items():
                print(f"  {section}:")
                if isinstance(values, dict):
                    for key, value in values.items():
                        print(f"    {key}: {value}")
                else:
                    print(f"    {values}")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def encrypt_secrets(self, input_file: Optional[str] = None, output_file: Optional[str] = None) -> bool:
        """Encrypt secrets in a YAML file."""
        try:
            input_path = Path(input_file) if input_file else self.config_path / "secrets.yaml"
            output_path = Path(output_file) if output_file else input_path
            
            if not input_path.exists():
                print(f"❌ Input file not found: {input_path}")
                return False
            
            print(f"🔐 Encrypting secrets from {input_path}...")
            
            # Load secrets
            with open(input_path, 'r') as f:
                secrets_data = yaml.safe_load(f) or {}
            
            # Initialize secrets manager
            secrets_manager = SecretsManager()
            
            # Encrypt values
            encrypted_data = self._encrypt_dict_values(secrets_data, secrets_manager)
            
            # Save encrypted data
            with open(output_path, 'w') as f:
                yaml.dump(encrypted_data, f, default_flow_style=False)
            
            # Set restrictive permissions
            output_path.chmod(0o600)
            
            print(f"✅ Secrets encrypted and saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to encrypt secrets: {e}")
            return False
    
    def decrypt_secrets(self, input_file: Optional[str] = None, output_file: Optional[str] = None) -> bool:
        """Decrypt secrets in a YAML file."""
        try:
            input_path = Path(input_file) if input_file else self.config_path / "secrets.yaml"
            output_path = Path(output_file) if output_file else self.config_path / "secrets.decrypted.yaml"
            
            if not input_path.exists():
                print(f"❌ Input file not found: {input_path}")
                return False
            
            print(f"🔓 Decrypting secrets from {input_path}...")
            
            # Load encrypted secrets
            with open(input_path, 'r') as f:
                encrypted_data = yaml.safe_load(f) or {}
            
            # Initialize secrets manager
            secrets_manager = SecretsManager()
            
            # Decrypt values
            decrypted_data = self._decrypt_dict_values(encrypted_data, secrets_manager)
            
            # Save decrypted data
            with open(output_path, 'w') as f:
                yaml.dump(decrypted_data, f, default_flow_style=False)
            
            # Set restrictive permissions
            output_path.chmod(0o600)
            
            print(f"✅ Secrets decrypted and saved to {output_path}")
            print(f"⚠️  WARNING: Decrypted file contains plaintext secrets!")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to decrypt secrets: {e}")
            return False
    
    def generate_keys(self) -> bool:
        """Generate new encryption and security keys."""
        try:
            print("🔑 Generating new security keys...")
            
            # Generate various keys
            keys = {
                'master_key': secrets.token_urlsafe(32),
                'api_secret_key': secrets.token_urlsafe(32),
                'jwt_secret_key': secrets.token_urlsafe(32),
                'encryption_key': secrets.token_urlsafe(32),
                'password_salt': secrets.token_urlsafe(16)
            }
            
            print("\n🔐 Generated Keys (store these securely!):")
            print("=" * 60)
            
            for key_name, key_value in keys.items():
                env_var_name = f"TRADING_BOT_{key_name.upper()}"
                print(f"{env_var_name}={key_value}")
            
            print("=" * 60)
            print("\n📝 Next Steps:")
            print("1. Store these keys in your environment variables")
            print("2. Update your secrets.yaml file")
            print("3. Never commit these keys to version control")
            print("4. Use different keys for each environment")
            
            # Optionally save to .env file
            save_to_env = input("\n💾 Save to .env file? (y/N): ").lower().strip()
            if save_to_env == 'y':
                env_file = Path(".env")
                with open(env_file, 'a') as f:
                    f.write("\n# Generated keys - DO NOT COMMIT TO VERSION CONTROL\n")
                    for key_name, key_value in keys.items():
                        env_var_name = f"TRADING_BOT_{key_name.upper()}"
                        f.write(f"{env_var_name}={key_value}\n")
                
                env_file.chmod(0o600)
                print(f"✅ Keys saved to {env_file}")
                print("⚠️  WARNING: Add .env to your .gitignore file!")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate keys: {e}")
            return False
    
    def export_config(self, environment: Optional[str] = None, include_secrets: bool = False) -> bool:
        """Export configuration to JSON format."""
        try:
            env = Environment(environment) if environment else Environment.DEVELOPMENT
            print(f"📤 Exporting {env.value} configuration...")
            
            config_manager = ProductionConfigManager(environment=env)
            config_data = config_manager.export_config(include_secrets=include_secrets)
            
            # Save to file
            output_file = self.config_path / f"{env.value}_config_export.json"
            with open(output_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            print(f"✅ Configuration exported to {output_file}")
            
            if include_secrets:
                print("⚠️  WARNING: Export contains sensitive data!")
                output_file.chmod(0o600)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to export configuration: {e}")
            return False
    
    def test_config(self, environment: Optional[str] = None) -> bool:
        """Test configuration by attempting to load and validate all components."""
        try:
            env = Environment(environment) if environment else Environment.DEVELOPMENT
            print(f"🧪 Testing {env.value} configuration...")
            
            config_manager = ProductionConfigManager(environment=env)
            
            # Test database configuration
            print("  📊 Testing database configuration...")
            db_url = config_manager.database.get_url()
            print(f"     Database URL: {db_url[:30]}...")
            
            # Test Redis configuration
            print("  🔴 Testing Redis configuration...")
            redis_url = config_manager.redis.get_url()
            print(f"     Redis URL: {redis_url[:30]}...")
            
            # Test API configuration
            print("  🚀 Testing API configuration...")
            print(f"     API endpoint: {config_manager.api.host}:{config_manager.api.port}")
            print(f"     Workers: {config_manager.api.workers}")
            print(f"     Auth enabled: {config_manager.api.enable_auth}")
            
            # Test trading configuration
            print("  💰 Testing trading configuration...")
            print(f"     Testnet: {config_manager.trading.bybit_testnet}")
            print(f"     Paper trading: {config_manager.trading.enable_paper_trading}")
            print(f"     Symbol: {config_manager.trading.default_symbol}")
            
            print("✅ All configuration tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            return False
    
    def _encrypt_dict_values(self, data: Dict[str, Any], secrets_manager: SecretsManager) -> Dict[str, Any]:
        """Recursively encrypt dictionary values."""
        encrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                encrypted_data[key] = self._encrypt_dict_values(value, secrets_manager)
            elif isinstance(value, str) and value and not value.startswith('enc:'):
                # Only encrypt non-empty strings that aren't already encrypted
                encrypted_data[key] = f"enc:{secrets_manager.encrypt_secret(value)}"
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def _decrypt_dict_values(self, data: Dict[str, Any], secrets_manager: SecretsManager) -> Dict[str, Any]:
        """Recursively decrypt dictionary values."""
        decrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                decrypted_data[key] = self._decrypt_dict_values(value, secrets_manager)
            elif isinstance(value, str) and value.startswith('enc:'):
                # Decrypt encrypted values
                decrypted_data[key] = secrets_manager.decrypt_secret(value[4:])
            else:
                decrypted_data[key] = value
        
        return decrypted_data


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Configuration Management CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python config_cli.py validate --environment production
    python config_cli.py encrypt-secrets --input secrets.yaml
    python config_cli.py generate-keys
    python config_cli.py export --environment staging --include-secrets
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument(
        '--environment', '-e', 
        choices=['development', 'testing', 'staging', 'production'],
        help='Environment to validate'
    )
    
    # Encrypt secrets command
    encrypt_parser = subparsers.add_parser('encrypt-secrets', help='Encrypt secrets file')
    encrypt_parser.add_argument('--input', '-i', help='Input secrets file')
    encrypt_parser.add_argument('--output', '-o', help='Output file for encrypted secrets')
    
    # Decrypt secrets command
    decrypt_parser = subparsers.add_parser('decrypt-secrets', help='Decrypt secrets file')
    decrypt_parser.add_argument('--input', '-i', help='Input encrypted secrets file')
    decrypt_parser.add_argument('--output', '-o', help='Output file for decrypted secrets')
    
    # Generate keys command
    subparsers.add_parser('generate-keys', help='Generate new security keys')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration')
    export_parser.add_argument(
        '--environment', '-e',
        choices=['development', 'testing', 'staging', 'production'],
        help='Environment to export'
    )
    export_parser.add_argument(
        '--include-secrets', 
        action='store_true',
        help='Include secrets in export (WARNING: sensitive data)'
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test configuration')
    test_parser.add_argument(
        '--environment', '-e',
        choices=['development', 'testing', 'staging', 'production'],
        help='Environment to test'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = ConfigCLI()
    success = False
    
    try:
        if args.command == 'validate':
            success = cli.validate_config(args.environment)
        elif args.command == 'encrypt-secrets':
            success = cli.encrypt_secrets(args.input, args.output)
        elif args.command == 'decrypt-secrets':
            success = cli.decrypt_secrets(args.input, args.output)
        elif args.command == 'generate-keys':
            success = cli.generate_keys()
        elif args.command == 'export':
            success = cli.export_config(args.environment, args.include_secrets)
        elif args.command == 'test':
            success = cli.test_config(args.environment)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())