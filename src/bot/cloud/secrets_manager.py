"""
Secrets Manager for Secure Secret Storage and Management.
Comprehensive secret management with encryption, key rotation, and multi-provider support.
"""

import asyncio
import json
import os
import time
import hashlib
import base64
import secrets
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import hvac  # HashiCorp Vault client
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import aiofiles

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class SecretProvider(Enum):
    """Supported secret providers."""
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    GOOGLE_SECRET_MANAGER = "google_secret_manager"
    HASHICORP_VAULT = "hashicorp_vault"
    KUBERNETES_SECRETS = "kubernetes_secrets"
    LOCAL_ENCRYPTED = "local_encrypted"

class SecretType(Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    CONNECTION_STRING = "connection_string"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_TOKEN = "oauth_token"
    DATABASE_CREDENTIALS = "database_credentials"
    WEBHOOK_SECRET = "webhook_secret"

class RotationStatus(Enum):
    """Secret rotation status."""
    NOT_SCHEDULED = "not_scheduled"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SecretConfig:
    """Secret configuration."""
    name: str
    provider: SecretProvider
    secret_type: SecretType
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    rotation_enabled: bool = False
    rotation_interval_days: int = 90
    encryption_key_id: Optional[str] = None
    access_policies: List[str] = field(default_factory=list)
    allowed_applications: List[str] = field(default_factory=list)
    expiration_date: Optional[datetime] = None
    
@dataclass
class SecretValue:
    """Secret value with metadata."""
    value: str
    version: str
    created_at: datetime
    created_by: str
    description: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class SecretRotation:
    """Secret rotation configuration."""
    secret_name: str
    rotation_function: str  # Function or lambda ARN for rotation
    rotation_interval: timedelta
    last_rotation: Optional[datetime] = None
    next_rotation: Optional[datetime] = None
    status: RotationStatus = RotationStatus.NOT_SCHEDULED
    
@dataclass
class AccessAudit:
    """Secret access audit log."""
    secret_name: str
    accessed_by: str
    accessed_at: datetime
    operation: str  # read, write, delete, rotate
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
class SecretsManager:
    """Comprehensive secrets management system."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Secrets configuration
        self.provider_configs: Dict[str, Dict[str, Any]] = {}
        self.active_providers: Dict[str, Any] = {}
        
        # Local encryption
        self.master_key: Optional[bytes] = None
        self.encryption_cipher: Optional[Fernet] = None
        
        # Secret tracking
        self.secret_configs: Dict[str, SecretConfig] = {}
        self.secret_cache: Dict[str, SecretValue] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Rotation management
        self.rotation_schedules: Dict[str, SecretRotation] = {}
        
        # Audit logging
        self.access_logs: List[AccessAudit] = []
        self.max_audit_logs = 10000
        
        # Security settings
        self.max_secret_age_days = 365
        self.require_mfa = False
        self.allowed_ip_ranges: List[str] = []
        
        # Initialize providers
        self._initialize_default_providers()
        self._initialize_master_key()
        
        self.logger.info("SecretsManager initialized")
    
    def _initialize_default_providers(self):
        """Initialize default secret provider configurations."""
        try:
            # AWS Secrets Manager
            self.provider_configs[SecretProvider.AWS_SECRETS_MANAGER.value] = {
                'region': 'us-east-1',
                'access_key_id': None,
                'secret_access_key': None,
                'kms_key_id': None
            }
            
            # HashiCorp Vault
            self.provider_configs[SecretProvider.HASHICORP_VAULT.value] = {
                'url': 'http://localhost:8200',
                'token': None,
                'namespace': None,
                'mount_point': 'secret',
                'tls_verify': True
            }
            
            # Local encrypted storage
            self.provider_configs[SecretProvider.LOCAL_ENCRYPTED.value] = {
                'storage_path': './secrets',
                'backup_enabled': True,
                'backup_path': './secrets_backup'
            }
            
            # Kubernetes Secrets
            self.provider_configs[SecretProvider.KUBERNETES_SECRETS.value] = {
                'namespace': 'default',
                'kubeconfig_path': '~/.kube/config'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default providers: {e}")
    
    def _initialize_master_key(self):
        """Initialize master encryption key for local storage."""
        try:
            key_file = Path('./master.key')
            
            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    self.master_key = f.read()
            else:
                # Generate new key
                self.master_key = Fernet.generate_key()
                
                # Save key securely
                with open(key_file, 'wb') as f:
                    f.write(self.master_key)
                
                # Set restrictive permissions
                key_file.chmod(0o600)
            
            self.encryption_cipher = Fernet(self.master_key)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize master key: {e}")
            # Generate in-memory key as fallback
            self.master_key = Fernet.generate_key()
            self.encryption_cipher = Fernet(self.master_key)
    
    async def configure_provider(self, provider: SecretProvider, config: Dict[str, Any]):
        """Configure a secret provider."""
        try:
            self.provider_configs[provider.value] = config
            
            # Initialize provider connection
            await self._initialize_provider_connection(provider, config)
            
            self.logger.info(f"Provider {provider.value} configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to configure provider {provider.value}: {e}")
            raise
    
    async def _initialize_provider_connection(self, provider: SecretProvider, config: Dict[str, Any]):
        """Initialize connection to secret provider."""
        try:
            if provider == SecretProvider.AWS_SECRETS_MANAGER:
                session_kwargs = {
                    'region_name': config.get('region', 'us-east-1')
                }
                
                if config.get('access_key_id') and config.get('secret_access_key'):
                    session_kwargs.update({
                        'aws_access_key_id': config['access_key_id'],
                        'aws_secret_access_key': config['secret_access_key']
                    })
                
                session = boto3.Session(**session_kwargs)
                client = session.client('secretsmanager')
                self.active_providers[provider.value] = client
                
            elif provider == SecretProvider.HASHICORP_VAULT:
                vault_client = hvac.Client(
                    url=config['url'],
                    token=config.get('token'),
                    namespace=config.get('namespace')
                )
                
                if not vault_client.is_authenticated():
                    raise Exception("Vault authentication failed")
                
                self.active_providers[provider.value] = vault_client
                
            elif provider == SecretProvider.LOCAL_ENCRYPTED:
                storage_path = Path(config.get('storage_path', './secrets'))
                storage_path.mkdir(parents=True, exist_ok=True)
                
                backup_path = None
                if config.get('backup_enabled', True):
                    backup_path = Path(config.get('backup_path', './secrets_backup'))
                    backup_path.mkdir(parents=True, exist_ok=True)
                
                self.active_providers[provider.value] = {
                    'storage_path': storage_path,
                    'backup_path': backup_path
                }
                
            elif provider == SecretProvider.KUBERNETES_SECRETS:
                from kubernetes import client, config as k8s_config
                
                kubeconfig_path = config.get('kubeconfig_path')
                if kubeconfig_path:
                    k8s_config.load_kube_config(config_file=kubeconfig_path)
                else:
                    k8s_config.load_incluster_config()
                
                v1 = client.CoreV1Api()
                self.active_providers[provider.value] = {
                    'client': v1,
                    'namespace': config.get('namespace', 'default')
                }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize provider connection: {e}")
            raise
    
    async def create_secret(self, config: SecretConfig, value: str, created_by: str = "system") -> bool:
        """Create a new secret."""
        try:
            # Validate secret value
            if not value or len(value.strip()) == 0:
                raise ValueError("Secret value cannot be empty")
            
            # Create secret value object
            secret_value = SecretValue(
                value=value,
                version="1",
                created_at=datetime.now(),
                created_by=created_by,
                description=config.description
            )
            
            # Store secret based on provider
            success = await self._store_secret(config, secret_value)
            
            if success:
                # Store configuration
                self.secret_configs[config.name] = config
                
                # Set up rotation if enabled
                if config.rotation_enabled:
                    await self._schedule_rotation(config)
                
                # Log access
                await self._log_access(config.name, created_by, "create", True)
                
                self.logger.info(f"Secret {config.name} created successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to create secret {config.name}: {e}")
            await self._log_access(config.name, created_by, "create", False)
            return False
    
    async def _store_secret(self, config: SecretConfig, secret_value: SecretValue) -> bool:
        """Store secret in the configured provider."""
        try:
            provider = config.provider
            
            if provider == SecretProvider.AWS_SECRETS_MANAGER:
                return await self._store_secret_aws(config, secret_value)
                
            elif provider == SecretProvider.HASHICORP_VAULT:
                return await self._store_secret_vault(config, secret_value)
                
            elif provider == SecretProvider.LOCAL_ENCRYPTED:
                return await self._store_secret_local(config, secret_value)
                
            elif provider == SecretProvider.KUBERNETES_SECRETS:
                return await self._store_secret_k8s(config, secret_value)
            
            else:
                raise Exception(f"Unsupported provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to store secret: {e}")
            return False
    
    async def _store_secret_aws(self, config: SecretConfig, secret_value: SecretValue) -> bool:
        """Store secret in AWS Secrets Manager."""
        try:
            client = self.active_providers[SecretProvider.AWS_SECRETS_MANAGER.value]
            
            # Prepare secret data
            secret_data = {
                'value': secret_value.value,
                'metadata': {
                    'created_by': secret_value.created_by,
                    'description': secret_value.description,
                    'secret_type': config.secret_type.value,
                    **secret_value.metadata
                }
            }
            
            # Create or update secret
            try:
                client.create_secret(
                    Name=config.name,
                    SecretString=json.dumps(secret_data),
                    Description=config.description,
                    KmsKeyId=config.encryption_key_id,
                    Tags=[
                        {'Key': k, 'Value': v} for k, v in config.tags.items()
                    ]
                )
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceExistsException':
                    # Update existing secret
                    client.update_secret(
                        SecretId=config.name,
                        SecretString=json.dumps(secret_data),
                        Description=config.description,
                        KmsKeyId=config.encryption_key_id
                    )
                else:
                    raise
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret in AWS: {e}")
            return False
    
    async def _store_secret_vault(self, config: SecretConfig, secret_value: SecretValue) -> bool:
        """Store secret in HashiCorp Vault."""
        try:
            vault_client = self.active_providers[SecretProvider.HASHICORP_VAULT.value]
            vault_config = self.provider_configs[SecretProvider.HASHICORP_VAULT.value]
            
            # Prepare secret data
            secret_data = {
                'value': secret_value.value,
                'created_by': secret_value.created_by,
                'created_at': secret_value.created_at.isoformat(),
                'description': secret_value.description,
                'secret_type': config.secret_type.value,
                **secret_value.metadata
            }
            
            # Store in Vault
            mount_point = vault_config.get('mount_point', 'secret')
            path = f"{mount_point}/data/{config.name}"
            
            vault_client.secrets.kv.v2.create_or_update_secret(
                path=config.name,
                secret=secret_data,
                mount_point=mount_point
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret in Vault: {e}")
            return False
    
    async def _store_secret_local(self, config: SecretConfig, secret_value: SecretValue) -> bool:
        """Store secret in local encrypted storage."""
        try:
            provider_config = self.active_providers[SecretProvider.LOCAL_ENCRYPTED.value]
            storage_path = provider_config['storage_path']
            
            # Prepare secret data
            secret_data = {
                'value': secret_value.value,
                'created_by': secret_value.created_by,
                'created_at': secret_value.created_at.isoformat(),
                'description': secret_value.description,
                'secret_type': config.secret_type.value,
                'version': secret_value.version,
                'config': {
                    'rotation_enabled': config.rotation_enabled,
                    'rotation_interval_days': config.rotation_interval_days,
                    'tags': config.tags
                },
                **secret_value.metadata
            }
            
            # Encrypt secret data
            encrypted_data = self.encryption_cipher.encrypt(
                json.dumps(secret_data).encode('utf-8')
            )
            
            # Write to file
            secret_file = storage_path / f"{config.name}.secret"
            async with aiofiles.open(secret_file, 'wb') as f:
                await f.write(encrypted_data)
            
            # Set restrictive permissions
            secret_file.chmod(0o600)
            
            # Create backup if enabled
            if provider_config['backup_path']:
                backup_file = provider_config['backup_path'] / f"{config.name}.secret"
                async with aiofiles.open(backup_file, 'wb') as f:
                    await f.write(encrypted_data)
                backup_file.chmod(0o600)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret locally: {e}")
            return False
    
    async def _store_secret_k8s(self, config: SecretConfig, secret_value: SecretValue) -> bool:
        """Store secret in Kubernetes Secrets."""
        try:
            from kubernetes import client
            
            provider_config = self.active_providers[SecretProvider.KUBERNETES_SECRETS.value]
            k8s_client = provider_config['client']
            namespace = provider_config['namespace']
            
            # Prepare secret data (base64 encoded)
            secret_data = {
                'value': base64.b64encode(secret_value.value.encode('utf-8')).decode('utf-8')
            }
            
            # Create secret metadata
            metadata = client.V1ObjectMeta(
                name=config.name.lower().replace('_', '-'),  # K8s naming conventions
                namespace=namespace,
                labels={
                    'app': 'trading-bot',
                    'secret-type': config.secret_type.value,
                    **{k.replace('_', '-'): v for k, v in config.tags.items()}
                },
                annotations={
                    'created-by': secret_value.created_by,
                    'description': config.description,
                    'rotation-enabled': str(config.rotation_enabled).lower()
                }
            )
            
            # Create secret object
            secret = client.V1Secret(
                api_version="v1",
                kind="Secret",
                metadata=metadata,
                data=secret_data,
                type="Opaque"
            )
            
            # Create or update secret
            try:
                k8s_client.create_namespaced_secret(namespace=namespace, body=secret)
            except client.ApiException as e:
                if e.status == 409:  # Already exists
                    k8s_client.replace_namespaced_secret(
                        name=metadata.name,
                        namespace=namespace,
                        body=secret
                    )
                else:
                    raise
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret in Kubernetes: {e}")
            return False
    
    async def get_secret(self, secret_name: str, version: Optional[str] = None, 
                        accessed_by: str = "system") -> Optional[SecretValue]:
        """Retrieve secret value."""
        try:
            # Check cache first
            cache_key = f"{secret_name}:{version or 'latest'}"
            if (cache_key in self.secret_cache and 
                cache_key in self.cache_timestamps and
                (datetime.now() - self.cache_timestamps[cache_key]).total_seconds() < self.cache_ttl):
                
                await self._log_access(secret_name, accessed_by, "read", True)
                return self.secret_cache[cache_key]
            
            # Get secret config
            if secret_name not in self.secret_configs:
                raise Exception(f"Secret {secret_name} not configured")
            
            config = self.secret_configs[secret_name]
            
            # Retrieve from provider
            secret_value = await self._retrieve_secret(config, version)
            
            if secret_value:
                # Cache the result
                self.secret_cache[cache_key] = secret_value
                self.cache_timestamps[cache_key] = datetime.now()
                
                await self._log_access(secret_name, accessed_by, "read", True)
                return secret_value
            
            await self._log_access(secret_name, accessed_by, "read", False)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get secret {secret_name}: {e}")
            await self._log_access(secret_name, accessed_by, "read", False)
            return None
    
    async def _retrieve_secret(self, config: SecretConfig, version: Optional[str] = None) -> Optional[SecretValue]:
        """Retrieve secret from the configured provider."""
        try:
            provider = config.provider
            
            if provider == SecretProvider.AWS_SECRETS_MANAGER:
                return await self._retrieve_secret_aws(config, version)
                
            elif provider == SecretProvider.HASHICORP_VAULT:
                return await self._retrieve_secret_vault(config, version)
                
            elif provider == SecretProvider.LOCAL_ENCRYPTED:
                return await self._retrieve_secret_local(config)
                
            elif provider == SecretProvider.KUBERNETES_SECRETS:
                return await self._retrieve_secret_k8s(config)
            
            else:
                raise Exception(f"Unsupported provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret: {e}")
            return None
    
    async def _retrieve_secret_aws(self, config: SecretConfig, version: Optional[str] = None) -> Optional[SecretValue]:
        """Retrieve secret from AWS Secrets Manager."""
        try:
            client = self.active_providers[SecretProvider.AWS_SECRETS_MANAGER.value]
            
            get_args = {'SecretId': config.name}
            if version:
                get_args['VersionId'] = version
            
            response = client.get_secret_value(**get_args)
            
            secret_data = json.loads(response['SecretString'])
            
            return SecretValue(
                value=secret_data['value'],
                version=response.get('VersionId', '1'),
                created_at=response['CreatedDate'],
                created_by=secret_data.get('metadata', {}).get('created_by', 'unknown'),
                description=secret_data.get('metadata', {}).get('description', ''),
                metadata=secret_data.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret from AWS: {e}")
            return None
    
    async def _retrieve_secret_vault(self, config: SecretConfig, version: Optional[str] = None) -> Optional[SecretValue]:
        """Retrieve secret from HashiCorp Vault."""
        try:
            vault_client = self.active_providers[SecretProvider.HASHICORP_VAULT.value]
            vault_config = self.provider_configs[SecretProvider.HASHICORP_VAULT.value]
            
            mount_point = vault_config.get('mount_point', 'secret')
            
            read_args = {'path': config.name, 'mount_point': mount_point}
            if version:
                read_args['version'] = version
            
            response = vault_client.secrets.kv.v2.read_secret(**read_args)
            
            secret_data = response['data']['data']
            
            return SecretValue(
                value=secret_data['value'],
                version=str(response['data']['metadata']['version']),
                created_at=datetime.fromisoformat(secret_data.get('created_at', datetime.now().isoformat())),
                created_by=secret_data.get('created_by', 'unknown'),
                description=secret_data.get('description', ''),
                metadata={k: v for k, v in secret_data.items() 
                         if k not in ['value', 'created_by', 'created_at', 'description', 'secret_type']}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret from Vault: {e}")
            return None
    
    async def _retrieve_secret_local(self, config: SecretConfig) -> Optional[SecretValue]:
        """Retrieve secret from local encrypted storage."""
        try:
            provider_config = self.active_providers[SecretProvider.LOCAL_ENCRYPTED.value]
            storage_path = provider_config['storage_path']
            
            secret_file = storage_path / f"{config.name}.secret"
            
            if not secret_file.exists():
                return None
            
            # Read encrypted data
            async with aiofiles.open(secret_file, 'rb') as f:
                encrypted_data = await f.read()
            
            # Decrypt secret data
            decrypted_data = self.encryption_cipher.decrypt(encrypted_data)
            secret_data = json.loads(decrypted_data.decode('utf-8'))
            
            return SecretValue(
                value=secret_data['value'],
                version=secret_data.get('version', '1'),
                created_at=datetime.fromisoformat(secret_data['created_at']),
                created_by=secret_data.get('created_by', 'unknown'),
                description=secret_data.get('description', ''),
                metadata={k: v for k, v in secret_data.items() 
                         if k not in ['value', 'created_by', 'created_at', 'description', 'secret_type', 'version', 'config']}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret locally: {e}")
            return None
    
    async def _retrieve_secret_k8s(self, config: SecretConfig) -> Optional[SecretValue]:
        """Retrieve secret from Kubernetes Secrets."""
        try:
            provider_config = self.active_providers[SecretProvider.KUBERNETES_SECRETS.value]
            k8s_client = provider_config['client']
            namespace = provider_config['namespace']
            
            secret_name = config.name.lower().replace('_', '-')
            
            secret = k8s_client.read_namespaced_secret(name=secret_name, namespace=namespace)
            
            # Decode secret value
            encoded_value = secret.data.get('value', '')
            value = base64.b64decode(encoded_value).decode('utf-8')
            
            return SecretValue(
                value=value,
                version='1',  # K8s secrets don't have versions
                created_at=secret.metadata.creation_timestamp,
                created_by=secret.metadata.annotations.get('created-by', 'unknown'),
                description=secret.metadata.annotations.get('description', ''),
                metadata={}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret from Kubernetes: {e}")
            return None
    
    async def update_secret(self, secret_name: str, new_value: str, updated_by: str = "system") -> bool:
        """Update existing secret."""
        try:
            if secret_name not in self.secret_configs:
                raise Exception(f"Secret {secret_name} not configured")
            
            config = self.secret_configs[secret_name]
            
            # Create new secret value
            secret_value = SecretValue(
                value=new_value,
                version=str(int(time.time())),  # Use timestamp as version
                created_at=datetime.now(),
                created_by=updated_by,
                description=config.description
            )
            
            # Store updated secret
            success = await self._store_secret(config, secret_value)
            
            if success:
                # Clear cache
                self._clear_secret_cache(secret_name)
                
                # Log access
                await self._log_access(secret_name, updated_by, "update", True)
                
                self.logger.info(f"Secret {secret_name} updated successfully")
                return True
            
            await self._log_access(secret_name, updated_by, "update", False)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update secret {secret_name}: {e}")
            await self._log_access(secret_name, updated_by, "update", False)
            return False
    
    async def delete_secret(self, secret_name: str, deleted_by: str = "system") -> bool:
        """Delete secret."""
        try:
            if secret_name not in self.secret_configs:
                raise Exception(f"Secret {secret_name} not configured")
            
            config = self.secret_configs[secret_name]
            
            # Delete from provider
            success = await self._delete_secret_from_provider(config)
            
            if success:
                # Remove from local tracking
                del self.secret_configs[secret_name]
                
                # Clear cache
                self._clear_secret_cache(secret_name)
                
                # Remove rotation schedule
                if secret_name in self.rotation_schedules:
                    del self.rotation_schedules[secret_name]
                
                # Log access
                await self._log_access(secret_name, deleted_by, "delete", True)
                
                self.logger.info(f"Secret {secret_name} deleted successfully")
                return True
            
            await self._log_access(secret_name, deleted_by, "delete", False)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret {secret_name}: {e}")
            await self._log_access(secret_name, deleted_by, "delete", False)
            return False
    
    async def _delete_secret_from_provider(self, config: SecretConfig) -> bool:
        """Delete secret from the configured provider."""
        try:
            provider = config.provider
            
            if provider == SecretProvider.AWS_SECRETS_MANAGER:
                client = self.active_providers[provider.value]
                client.delete_secret(SecretId=config.name, ForceDeleteWithoutRecovery=True)
                
            elif provider == SecretProvider.HASHICORP_VAULT:
                vault_client = self.active_providers[provider.value]
                vault_config = self.provider_configs[provider.value]
                mount_point = vault_config.get('mount_point', 'secret')
                
                vault_client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=config.name,
                    mount_point=mount_point
                )
                
            elif provider == SecretProvider.LOCAL_ENCRYPTED:
                provider_config = self.active_providers[provider.value]
                storage_path = provider_config['storage_path']
                
                secret_file = storage_path / f"{config.name}.secret"
                if secret_file.exists():
                    secret_file.unlink()
                
                # Remove backup
                if provider_config['backup_path']:
                    backup_file = provider_config['backup_path'] / f"{config.name}.secret"
                    if backup_file.exists():
                        backup_file.unlink()
                
            elif provider == SecretProvider.KUBERNETES_SECRETS:
                provider_config = self.active_providers[provider.value]
                k8s_client = provider_config['client']
                namespace = provider_config['namespace']
                
                secret_name = config.name.lower().replace('_', '-')
                k8s_client.delete_namespaced_secret(name=secret_name, namespace=namespace)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret from provider: {e}")
            return False
    
    async def _schedule_rotation(self, config: SecretConfig):
        """Schedule secret rotation."""
        try:
            if not config.rotation_enabled:
                return
            
            rotation = SecretRotation(
                secret_name=config.name,
                rotation_function="default_rotation",  # Could be customized
                rotation_interval=timedelta(days=config.rotation_interval_days),
                next_rotation=datetime.now() + timedelta(days=config.rotation_interval_days),
                status=RotationStatus.SCHEDULED
            )
            
            self.rotation_schedules[config.name] = rotation
            
            self.logger.info(f"Rotation scheduled for secret {config.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to schedule rotation: {e}")
    
    async def rotate_secret(self, secret_name: str, rotation_function: Optional[callable] = None) -> bool:
        """Rotate secret manually or via scheduled rotation."""
        try:
            if secret_name not in self.secret_configs:
                raise Exception(f"Secret {secret_name} not configured")
            
            config = self.secret_configs[secret_name]
            
            # Update rotation status
            if secret_name in self.rotation_schedules:
                self.rotation_schedules[secret_name].status = RotationStatus.IN_PROGRESS
            
            # Generate new secret value based on type
            new_value = await self._generate_new_secret_value(config.secret_type)
            
            # Custom rotation function
            if rotation_function:
                new_value = await rotation_function(config, new_value)
            
            # Update the secret
            success = await self.update_secret(secret_name, new_value, "rotation_system")
            
            if success:
                # Update rotation tracking
                if secret_name in self.rotation_schedules:
                    rotation = self.rotation_schedules[secret_name]
                    rotation.status = RotationStatus.COMPLETED
                    rotation.last_rotation = datetime.now()
                    rotation.next_rotation = datetime.now() + rotation.rotation_interval
                
                await self._log_access(secret_name, "rotation_system", "rotate", True)
                
                self.logger.info(f"Secret {secret_name} rotated successfully")
                return True
            else:
                # Update rotation status on failure
                if secret_name in self.rotation_schedules:
                    self.rotation_schedules[secret_name].status = RotationStatus.FAILED
                
                await self._log_access(secret_name, "rotation_system", "rotate", False)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to rotate secret {secret_name}: {e}")
            
            if secret_name in self.rotation_schedules:
                self.rotation_schedules[secret_name].status = RotationStatus.FAILED
            
            await self._log_access(secret_name, "rotation_system", "rotate", False)
            return False
    
    async def _generate_new_secret_value(self, secret_type: SecretType) -> str:
        """Generate new secret value based on type."""
        try:
            if secret_type == SecretType.API_KEY:
                return secrets.token_urlsafe(32)
            
            elif secret_type == SecretType.PASSWORD:
                # Generate strong password
                import string
                chars = string.ascii_letters + string.digits + "!@#$%^&*"
                return ''.join(secrets.choice(chars) for _ in range(16))
            
            elif secret_type == SecretType.JWT_SECRET:
                return secrets.token_urlsafe(64)
            
            elif secret_type == SecretType.ENCRYPTION_KEY:
                return Fernet.generate_key().decode('utf-8')
            
            elif secret_type == SecretType.WEBHOOK_SECRET:
                return secrets.token_hex(32)
            
            else:
                # Default: generate random string
                return secrets.token_urlsafe(32)
                
        except Exception as e:
            self.logger.error(f"Failed to generate new secret value: {e}")
            return secrets.token_urlsafe(32)  # Fallback
    
    def _clear_secret_cache(self, secret_name: str):
        """Clear cache entries for a secret."""
        try:
            keys_to_remove = [key for key in self.secret_cache.keys() if key.startswith(f"{secret_name}:")]
            
            for key in keys_to_remove:
                del self.secret_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
                    
        except Exception as e:
            self.logger.error(f"Failed to clear secret cache: {e}")
    
    async def _log_access(self, secret_name: str, accessed_by: str, operation: str, success: bool):
        """Log secret access for audit purposes."""
        try:
            audit_log = AccessAudit(
                secret_name=secret_name,
                accessed_by=accessed_by,
                accessed_at=datetime.now(),
                operation=operation,
                success=success
            )
            
            self.access_logs.append(audit_log)
            
            # Trim audit logs if too many
            if len(self.access_logs) > self.max_audit_logs:
                self.access_logs = self.access_logs[-self.max_audit_logs:]
            
            # In production, this would also write to external audit log
            self.logger.info(f"Secret access: {secret_name} {operation} by {accessed_by} - {'success' if success else 'failed'}")
            
        except Exception as e:
            self.logger.error(f"Failed to log secret access: {e}")
    
    def get_secret_list(self) -> List[Dict[str, Any]]:
        """Get list of configured secrets."""
        try:
            return [
                {
                    'name': config.name,
                    'provider': config.provider.value,
                    'secret_type': config.secret_type.value,
                    'description': config.description,
                    'rotation_enabled': config.rotation_enabled,
                    'rotation_interval_days': config.rotation_interval_days,
                    'tags': config.tags,
                    'expiration_date': config.expiration_date.isoformat() if config.expiration_date else None
                }
                for config in self.secret_configs.values()
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get secret list: {e}")
            return []
    
    def get_rotation_status(self) -> List[Dict[str, Any]]:
        """Get rotation status for all secrets."""
        try:
            return [
                {
                    'secret_name': rotation.secret_name,
                    'status': rotation.status.value,
                    'last_rotation': rotation.last_rotation.isoformat() if rotation.last_rotation else None,
                    'next_rotation': rotation.next_rotation.isoformat() if rotation.next_rotation else None,
                    'rotation_interval_days': rotation.rotation_interval.days
                }
                for rotation in self.rotation_schedules.values()
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get rotation status: {e}")
            return []
    
    def get_audit_logs(self, secret_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs for secret access."""
        try:
            logs = self.access_logs
            
            if secret_name:
                logs = [log for log in logs if log.secret_name == secret_name]
            
            # Get most recent logs
            recent_logs = sorted(logs, key=lambda x: x.accessed_at, reverse=True)[:limit]
            
            return [
                {
                    'secret_name': log.secret_name,
                    'accessed_by': log.accessed_by,
                    'accessed_at': log.accessed_at.isoformat(),
                    'operation': log.operation,
                    'success': log.success,
                    'ip_address': log.ip_address,
                    'user_agent': log.user_agent
                }
                for log in recent_logs
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get audit logs: {e}")
            return []
    
    def get_secrets_manager_summary(self) -> Dict[str, Any]:
        """Get secrets manager summary."""
        try:
            total_secrets = len(self.secret_configs)
            rotation_enabled = len([c for c in self.secret_configs.values() if c.rotation_enabled])
            
            # Count by provider
            provider_counts = {}
            for config in self.secret_configs.values():
                provider = config.provider.value
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
            # Count by type
            type_counts = {}
            for config in self.secret_configs.values():
                secret_type = config.secret_type.value
                type_counts[secret_type] = type_counts.get(secret_type, 0) + 1
            
            return {
                'total_secrets': total_secrets,
                'rotation_enabled_count': rotation_enabled,
                'active_providers': list(self.active_providers.keys()),
                'provider_distribution': provider_counts,
                'secret_type_distribution': type_counts,
                'cached_secrets': len(self.secret_cache),
                'scheduled_rotations': len(self.rotation_schedules),
                'recent_access_logs': len(self.access_logs),
                'supported_providers': [provider.value for provider in SecretProvider],
                'supported_secret_types': [secret_type.value for secret_type in SecretType]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate secrets manager summary: {e}")
            return {'error': 'Unable to generate summary'}