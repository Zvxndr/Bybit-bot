"""
Cloud Storage Manager for Multi-Cloud Object Storage.
Unified interface for AWS S3, Google Cloud Storage, Azure Blob Storage, and local storage.
"""

import asyncio
import json
import os
import time
import hashlib
import mimetypes
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncIterator, IO
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import aiofiles
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient
import aiohttp

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class StorageProvider(Enum):
    """Supported storage providers."""
    AWS_S3 = "aws_s3"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_BLOB = "azure_blob"
    LOCAL_FS = "local_fs"
    MINIO = "minio"

class StorageClass(Enum):
    """Storage classes for cost optimization."""
    STANDARD = "standard"
    REDUCED_REDUNDANCY = "reduced_redundancy"
    COLD = "cold"
    ARCHIVE = "archive"
    DEEP_ARCHIVE = "deep_archive"

class AccessLevel(Enum):
    """Storage access levels."""
    PRIVATE = "private"
    PUBLIC_READ = "public-read"
    PUBLIC_READ_WRITE = "public-read-write"
    AUTHENTICATED_READ = "authenticated-read"

@dataclass
class StorageConfig:
    """Storage configuration."""
    provider: StorageProvider
    bucket_name: str
    region: str = "us-east-1"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    credentials_file: Optional[str] = None
    default_storage_class: StorageClass = StorageClass.STANDARD
    default_access_level: AccessLevel = AccessLevel.PRIVATE
    encryption_enabled: bool = True
    versioning_enabled: bool = True
    lifecycle_rules: List[Dict[str, Any]] = field(default_factory=list)
    cors_rules: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class StorageObject:
    """Storage object metadata."""
    key: str
    size: int
    last_modified: datetime
    etag: str
    storage_class: str
    content_type: str
    metadata: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    version_id: Optional[str] = None

@dataclass
class UploadProgress:
    """Upload progress tracking."""
    bytes_transferred: int
    total_bytes: int
    percentage: float
    speed_bps: float
    eta_seconds: Optional[float] = None

@dataclass
class SyncResult:
    """Synchronization result."""
    uploaded: List[str] = field(default_factory=list)
    downloaded: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_bytes_transferred: int = 0
    duration_seconds: float = 0

class CloudStorage:
    """Comprehensive multi-cloud storage manager."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Storage configuration
        self.storage_configs: Dict[str, StorageConfig] = {}
        self.active_connections: Dict[str, Any] = {}
        
        # Default configurations for different providers
        self.default_configs = {
            StorageProvider.AWS_S3: {
                'region': 'us-east-1',
                'endpoint_url': None
            },
            StorageProvider.GOOGLE_CLOUD: {
                'region': 'us-central1'
            },
            StorageProvider.AZURE_BLOB: {
                'region': 'eastus'
            },
            StorageProvider.LOCAL_FS: {
                'base_path': './storage'
            },
            StorageProvider.MINIO: {
                'endpoint_url': 'http://localhost:9000',
                'region': 'us-east-1'
            }
        }
        
        # Upload/download progress tracking
        self.upload_progress: Dict[str, UploadProgress] = {}
        self.download_progress: Dict[str, UploadProgress] = {}
        
        # Caching
        self.object_cache: Dict[str, StorageObject] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Multipart upload thresholds
        self.multipart_threshold = 100 * 1024 * 1024  # 100MB
        self.multipart_chunk_size = 10 * 1024 * 1024   # 10MB
        
        # Initialize default storage providers
        self._initialize_default_storage()
        
        self.logger.info("CloudStorage initialized")
    
    def _initialize_default_storage(self):
        """Initialize default storage configurations."""
        try:
            # AWS S3 configuration
            aws_config = StorageConfig(
                provider=StorageProvider.AWS_S3,
                bucket_name="trading-bot-storage",
                region="us-east-1",
                default_storage_class=StorageClass.STANDARD,
                encryption_enabled=True,
                versioning_enabled=True
            )
            self.storage_configs["aws"] = aws_config
            
            # Local filesystem configuration
            local_config = StorageConfig(
                provider=StorageProvider.LOCAL_FS,
                bucket_name="local-storage",
                region="local",
                default_storage_class=StorageClass.STANDARD,
                encryption_enabled=False,
                versioning_enabled=False
            )
            self.storage_configs["local"] = local_config
            
            # MinIO configuration (S3-compatible)
            minio_config = StorageConfig(
                provider=StorageProvider.MINIO,
                bucket_name="trading-bot-minio",
                endpoint_url="http://localhost:9000",
                access_key="minioadmin",
                secret_key="minioadmin",
                region="us-east-1"
            )
            self.storage_configs["minio"] = minio_config
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default storage: {e}")
    
    async def configure_storage(self, name: str, config: StorageConfig):
        """Configure a storage provider."""
        try:
            self.storage_configs[name] = config
            
            # Initialize connection
            await self._initialize_connection(name, config)
            
            self.logger.info(f"Storage {name} configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to configure storage {name}: {e}")
            raise
    
    async def _initialize_connection(self, name: str, config: StorageConfig):
        """Initialize connection to storage provider."""
        try:
            if config.provider == StorageProvider.AWS_S3:
                import boto3
                
                session_kwargs = {
                    'region_name': config.region
                }
                
                if config.access_key and config.secret_key:
                    session_kwargs.update({
                        'aws_access_key_id': config.access_key,
                        'aws_secret_access_key': config.secret_key
                    })
                
                session = boto3.Session(**session_kwargs)
                
                client_kwargs = {}
                if config.endpoint_url:
                    client_kwargs['endpoint_url'] = config.endpoint_url
                
                s3_client = session.client('s3', **client_kwargs)
                self.active_connections[name] = s3_client
                
            elif config.provider == StorageProvider.GOOGLE_CLOUD:
                if config.credentials_file:
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.credentials_file
                
                client = gcs.Client()
                self.active_connections[name] = client
                
            elif config.provider == StorageProvider.AZURE_BLOB:
                if config.access_key:  # Connection string or account key
                    blob_service_client = BlobServiceClient(
                        account_url=f"https://{config.access_key}.blob.core.windows.net",
                        credential=config.secret_key
                    )
                else:
                    # Use connection string
                    blob_service_client = BlobServiceClient.from_connection_string(
                        config.secret_key or os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
                    )
                
                self.active_connections[name] = blob_service_client
                
            elif config.provider == StorageProvider.LOCAL_FS:
                # Ensure base directory exists
                base_path = self.default_configs[StorageProvider.LOCAL_FS]['base_path']
                os.makedirs(base_path, exist_ok=True)
                self.active_connections[name] = base_path
                
            elif config.provider == StorageProvider.MINIO:
                import boto3
                
                s3_client = boto3.client(
                    's3',
                    endpoint_url=config.endpoint_url,
                    aws_access_key_id=config.access_key,
                    aws_secret_access_key=config.secret_key,
                    region_name=config.region
                )
                self.active_connections[name] = s3_client
            
            else:
                raise Exception(f"Unsupported storage provider: {config.provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize connection for {name}: {e}")
            raise
    
    async def upload_file(self, storage_name: str, local_path: str, remote_key: str, 
                         metadata: Optional[Dict[str, str]] = None,
                         storage_class: Optional[StorageClass] = None,
                         progress_callback: Optional[callable] = None) -> bool:
        """Upload file to storage."""
        try:
            if storage_name not in self.storage_configs:
                raise Exception(f"Storage {storage_name} not configured")
            
            config = self.storage_configs[storage_name]
            connection = self.active_connections[storage_name]
            
            # Get file info
            file_size = os.path.getsize(local_path)
            content_type = mimetypes.guess_type(local_path)[0] or 'application/octet-stream'
            
            # Initialize progress tracking
            progress_id = f"{storage_name}:{remote_key}"
            self.upload_progress[progress_id] = UploadProgress(
                bytes_transferred=0,
                total_bytes=file_size,
                percentage=0.0,
                speed_bps=0.0
            )
            
            start_time = time.time()
            
            if config.provider == StorageProvider.AWS_S3 or config.provider == StorageProvider.MINIO:
                await self._upload_to_s3(connection, config, local_path, remote_key, 
                                       metadata, storage_class, content_type, progress_callback)
                
            elif config.provider == StorageProvider.GOOGLE_CLOUD:
                await self._upload_to_gcs(connection, config, local_path, remote_key,
                                        metadata, storage_class, content_type, progress_callback)
                
            elif config.provider == StorageProvider.AZURE_BLOB:
                await self._upload_to_azure(connection, config, local_path, remote_key,
                                          metadata, storage_class, content_type, progress_callback)
                
            elif config.provider == StorageProvider.LOCAL_FS:
                await self._upload_to_local(connection, local_path, remote_key, metadata, progress_callback)
            
            # Clean up progress tracking
            if progress_id in self.upload_progress:
                del self.upload_progress[progress_id]
            
            duration = time.time() - start_time
            self.logger.info(f"Upload completed: {remote_key} ({file_size} bytes, {duration:.2f}s)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload file {local_path} to {remote_key}: {e}")
            return False
    
    async def _upload_to_s3(self, s3_client, config: StorageConfig, local_path: str,
                           remote_key: str, metadata: Optional[Dict[str, str]],
                           storage_class: Optional[StorageClass], content_type: str,
                           progress_callback: Optional[callable]):
        """Upload file to S3 or S3-compatible storage."""
        try:
            file_size = os.path.getsize(local_path)
            
            # Prepare upload arguments
            upload_args = {
                'Bucket': config.bucket_name,
                'Key': remote_key,
                'ContentType': content_type
            }
            
            if metadata:
                upload_args['Metadata'] = metadata
            
            if storage_class:
                upload_args['StorageClass'] = self._map_storage_class_s3(storage_class)
            
            if config.encryption_enabled:
                upload_args['ServerSideEncryption'] = 'AES256'
            
            # Use multipart upload for large files
            if file_size > self.multipart_threshold:
                await self._multipart_upload_s3(s3_client, local_path, upload_args, progress_callback)
            else:
                # Simple upload
                with open(local_path, 'rb') as f:
                    upload_args['Body'] = f.read()
                
                s3_client.put_object(**upload_args)
                
                # Update progress
                if progress_callback:
                    progress_callback(file_size, file_size)
                    
        except Exception as e:
            self.logger.error(f"S3 upload failed: {e}")
            raise
    
    async def _multipart_upload_s3(self, s3_client, local_path: str, upload_args: Dict[str, Any],
                                  progress_callback: Optional[callable]):
        """Perform multipart upload to S3."""
        try:
            bucket = upload_args['Bucket']
            key = upload_args['Key']
            
            # Initiate multipart upload
            create_args = {k: v for k, v in upload_args.items() if k not in ['Body']}
            response = s3_client.create_multipart_upload(**create_args)
            upload_id = response['UploadId']
            
            parts = []
            part_number = 1
            bytes_transferred = 0
            file_size = os.path.getsize(local_path)
            
            try:
                with open(local_path, 'rb') as f:
                    while True:
                        chunk = f.read(self.multipart_chunk_size)
                        if not chunk:
                            break
                        
                        # Upload part
                        part_response = s3_client.upload_part(
                            Bucket=bucket,
                            Key=key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=chunk
                        )
                        
                        parts.append({
                            'ETag': part_response['ETag'],
                            'PartNumber': part_number
                        })
                        
                        part_number += 1
                        bytes_transferred += len(chunk)
                        
                        # Update progress
                        if progress_callback:
                            progress_callback(bytes_transferred, file_size)
                
                # Complete multipart upload
                s3_client.complete_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={'Parts': parts}
                )
                
            except Exception as e:
                # Abort multipart upload on error
                s3_client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id
                )
                raise
                
        except Exception as e:
            self.logger.error(f"Multipart upload failed: {e}")
            raise
    
    async def _upload_to_gcs(self, gcs_client, config: StorageConfig, local_path: str,
                            remote_key: str, metadata: Optional[Dict[str, str]],
                            storage_class: Optional[StorageClass], content_type: str,
                            progress_callback: Optional[callable]):
        """Upload file to Google Cloud Storage."""
        try:
            bucket = gcs_client.bucket(config.bucket_name)
            blob = bucket.blob(remote_key)
            
            # Set metadata
            if metadata:
                blob.metadata = metadata
            
            if storage_class:
                blob.storage_class = self._map_storage_class_gcs(storage_class)
            
            # Upload file
            file_size = os.path.getsize(local_path)
            
            with open(local_path, 'rb') as f:
                blob.upload_from_file(f, content_type=content_type)
            
            # Update progress
            if progress_callback:
                progress_callback(file_size, file_size)
                
        except Exception as e:
            self.logger.error(f"GCS upload failed: {e}")
            raise
    
    async def _upload_to_azure(self, blob_client, config: StorageConfig, local_path: str,
                              remote_key: str, metadata: Optional[Dict[str, str]],
                              storage_class: Optional[StorageClass], content_type: str,
                              progress_callback: Optional[callable]):
        """Upload file to Azure Blob Storage."""
        try:
            blob_client_instance = blob_client.get_blob_client(
                container=config.bucket_name,
                blob=remote_key
            )
            
            # Prepare upload arguments
            upload_args = {
                'content_type': content_type
            }
            
            if metadata:
                upload_args['metadata'] = metadata
            
            # Upload file
            file_size = os.path.getsize(local_path)
            
            with open(local_path, 'rb') as f:
                blob_client_instance.upload_blob(f, overwrite=True, **upload_args)
            
            # Update progress
            if progress_callback:
                progress_callback(file_size, file_size)
                
        except Exception as e:
            self.logger.error(f"Azure upload failed: {e}")
            raise
    
    async def _upload_to_local(self, base_path: str, local_path: str, remote_key: str,
                              metadata: Optional[Dict[str, str]], progress_callback: Optional[callable]):
        """Upload file to local filesystem."""
        try:
            target_path = os.path.join(base_path, remote_key)
            target_dir = os.path.dirname(target_path)
            
            # Create directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy file
            file_size = os.path.getsize(local_path)
            
            shutil.copy2(local_path, target_path)
            
            # Save metadata if provided
            if metadata:
                metadata_path = f"{target_path}.metadata"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
            
            # Update progress
            if progress_callback:
                progress_callback(file_size, file_size)
                
        except Exception as e:
            self.logger.error(f"Local upload failed: {e}")
            raise
    
    async def download_file(self, storage_name: str, remote_key: str, local_path: str,
                           progress_callback: Optional[callable] = None) -> bool:
        """Download file from storage."""
        try:
            if storage_name not in self.storage_configs:
                raise Exception(f"Storage {storage_name} not configured")
            
            config = self.storage_configs[storage_name]
            connection = self.active_connections[storage_name]
            
            # Create local directory if it doesn't exist
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            
            start_time = time.time()
            
            if config.provider == StorageProvider.AWS_S3 or config.provider == StorageProvider.MINIO:
                await self._download_from_s3(connection, config, remote_key, local_path, progress_callback)
                
            elif config.provider == StorageProvider.GOOGLE_CLOUD:
                await self._download_from_gcs(connection, config, remote_key, local_path, progress_callback)
                
            elif config.provider == StorageProvider.AZURE_BLOB:
                await self._download_from_azure(connection, config, remote_key, local_path, progress_callback)
                
            elif config.provider == StorageProvider.LOCAL_FS:
                await self._download_from_local(connection, remote_key, local_path, progress_callback)
            
            duration = time.time() - start_time
            file_size = os.path.getsize(local_path)
            self.logger.info(f"Download completed: {remote_key} ({file_size} bytes, {duration:.2f}s)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download file {remote_key} to {local_path}: {e}")
            return False
    
    async def _download_from_s3(self, s3_client, config: StorageConfig, remote_key: str,
                               local_path: str, progress_callback: Optional[callable]):
        """Download file from S3."""
        try:
            # Get object info first
            response = s3_client.head_object(Bucket=config.bucket_name, Key=remote_key)
            file_size = response['ContentLength']
            
            # Download file
            with open(local_path, 'wb') as f:
                s3_client.download_fileobj(config.bucket_name, remote_key, f)
            
            # Update progress
            if progress_callback:
                progress_callback(file_size, file_size)
                
        except Exception as e:
            self.logger.error(f"S3 download failed: {e}")
            raise
    
    async def _download_from_gcs(self, gcs_client, config: StorageConfig, remote_key: str,
                                local_path: str, progress_callback: Optional[callable]):
        """Download file from Google Cloud Storage."""
        try:
            bucket = gcs_client.bucket(config.bucket_name)
            blob = bucket.blob(remote_key)
            
            # Download file
            with open(local_path, 'wb') as f:
                blob.download_to_file(f)
            
            # Update progress
            file_size = os.path.getsize(local_path)
            if progress_callback:
                progress_callback(file_size, file_size)
                
        except Exception as e:
            self.logger.error(f"GCS download failed: {e}")
            raise
    
    async def _download_from_azure(self, blob_client, config: StorageConfig, remote_key: str,
                                  local_path: str, progress_callback: Optional[callable]):
        """Download file from Azure Blob Storage."""
        try:
            blob_client_instance = blob_client.get_blob_client(
                container=config.bucket_name,
                blob=remote_key
            )
            
            # Download file
            with open(local_path, 'wb') as f:
                download_stream = blob_client_instance.download_blob()
                download_stream.readinto(f)
            
            # Update progress
            file_size = os.path.getsize(local_path)
            if progress_callback:
                progress_callback(file_size, file_size)
                
        except Exception as e:
            self.logger.error(f"Azure download failed: {e}")
            raise
    
    async def _download_from_local(self, base_path: str, remote_key: str, local_path: str,
                                  progress_callback: Optional[callable]):
        """Download file from local filesystem."""
        try:
            source_path = os.path.join(base_path, remote_key)
            
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"File not found: {source_path}")
            
            # Copy file
            shutil.copy2(source_path, local_path)
            
            # Update progress
            file_size = os.path.getsize(local_path)
            if progress_callback:
                progress_callback(file_size, file_size)
                
        except Exception as e:
            self.logger.error(f"Local download failed: {e}")
            raise
    
    async def list_objects(self, storage_name: str, prefix: str = "", max_keys: int = 1000) -> List[StorageObject]:
        """List objects in storage."""
        try:
            if storage_name not in self.storage_configs:
                raise Exception(f"Storage {storage_name} not configured")
            
            config = self.storage_configs[storage_name]
            connection = self.active_connections[storage_name]
            
            if config.provider == StorageProvider.AWS_S3 or config.provider == StorageProvider.MINIO:
                return await self._list_objects_s3(connection, config, prefix, max_keys)
                
            elif config.provider == StorageProvider.GOOGLE_CLOUD:
                return await self._list_objects_gcs(connection, config, prefix, max_keys)
                
            elif config.provider == StorageProvider.AZURE_BLOB:
                return await self._list_objects_azure(connection, config, prefix, max_keys)
                
            elif config.provider == StorageProvider.LOCAL_FS:
                return await self._list_objects_local(connection, prefix, max_keys)
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to list objects in {storage_name}: {e}")
            return []
    
    async def _list_objects_s3(self, s3_client, config: StorageConfig, prefix: str, max_keys: int) -> List[StorageObject]:
        """List objects in S3."""
        try:
            objects = []
            
            response = s3_client.list_objects_v2(
                Bucket=config.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            for obj in response.get('Contents', []):
                storage_obj = StorageObject(
                    key=obj['Key'],
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    etag=obj['ETag'].strip('"'),
                    storage_class=obj.get('StorageClass', 'STANDARD'),
                    content_type='application/octet-stream'  # Default, would need HEAD request for actual
                )
                objects.append(storage_obj)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to list S3 objects: {e}")
            return []
    
    async def _list_objects_gcs(self, gcs_client, config: StorageConfig, prefix: str, max_keys: int) -> List[StorageObject]:
        """List objects in Google Cloud Storage."""
        try:
            objects = []
            bucket = gcs_client.bucket(config.bucket_name)
            
            blobs = bucket.list_blobs(prefix=prefix, max_results=max_keys)
            
            for blob in blobs:
                storage_obj = StorageObject(
                    key=blob.name,
                    size=blob.size or 0,
                    last_modified=blob.updated or datetime.now(),
                    etag=blob.etag or '',
                    storage_class=blob.storage_class or 'STANDARD',
                    content_type=blob.content_type or 'application/octet-stream',
                    metadata=blob.metadata or {}
                )
                objects.append(storage_obj)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to list GCS objects: {e}")
            return []
    
    async def _list_objects_azure(self, blob_client, config: StorageConfig, prefix: str, max_keys: int) -> List[StorageObject]:
        """List objects in Azure Blob Storage."""
        try:
            objects = []
            container_client = blob_client.get_container_client(config.bucket_name)
            
            blobs = container_client.list_blobs(name_starts_with=prefix, results_per_page=max_keys)
            
            for blob in blobs:
                storage_obj = StorageObject(
                    key=blob.name,
                    size=blob.size,
                    last_modified=blob.last_modified,
                    etag=blob.etag,
                    storage_class='Standard',  # Azure doesn't have multiple storage classes in the same way
                    content_type=blob.properties.content_type or 'application/octet-stream',
                    metadata=blob.metadata or {}
                )
                objects.append(storage_obj)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to list Azure objects: {e}")
            return []
    
    async def _list_objects_local(self, base_path: str, prefix: str, max_keys: int) -> List[StorageObject]:
        """List objects in local filesystem."""
        try:
            objects = []
            search_path = os.path.join(base_path, prefix)
            
            if os.path.isfile(search_path):
                # Single file
                stat = os.stat(search_path)
                storage_obj = StorageObject(
                    key=os.path.relpath(search_path, base_path),
                    size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime),
                    etag=hashlib.md5(open(search_path, 'rb').read()).hexdigest(),
                    storage_class='Standard',
                    content_type=mimetypes.guess_type(search_path)[0] or 'application/octet-stream'
                )
                objects.append(storage_obj)
            else:
                # Directory or pattern
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if len(objects) >= max_keys:
                            break
                        
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, base_path)
                        
                        if rel_path.startswith(prefix):
                            stat = os.stat(file_path)
                            storage_obj = StorageObject(
                                key=rel_path.replace('\\', '/'),  # Use forward slashes
                                size=stat.st_size,
                                last_modified=datetime.fromtimestamp(stat.st_mtime),
                                etag=hashlib.md5(open(file_path, 'rb').read()).hexdigest(),
                                storage_class='Standard',
                                content_type=mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
                            )
                            objects.append(storage_obj)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to list local objects: {e}")
            return []
    
    async def delete_object(self, storage_name: str, remote_key: str) -> bool:
        """Delete object from storage."""
        try:
            if storage_name not in self.storage_configs:
                raise Exception(f"Storage {storage_name} not configured")
            
            config = self.storage_configs[storage_name]
            connection = self.active_connections[storage_name]
            
            if config.provider == StorageProvider.AWS_S3 or config.provider == StorageProvider.MINIO:
                connection.delete_object(Bucket=config.bucket_name, Key=remote_key)
                
            elif config.provider == StorageProvider.GOOGLE_CLOUD:
                bucket = connection.bucket(config.bucket_name)
                blob = bucket.blob(remote_key)
                blob.delete()
                
            elif config.provider == StorageProvider.AZURE_BLOB:
                blob_client = connection.get_blob_client(
                    container=config.bucket_name,
                    blob=remote_key
                )
                blob_client.delete_blob()
                
            elif config.provider == StorageProvider.LOCAL_FS:
                file_path = os.path.join(connection, remote_key)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                    # Remove metadata file if exists
                    metadata_path = f"{file_path}.metadata"
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
            
            self.logger.info(f"Deleted object: {remote_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete object {remote_key}: {e}")
            return False
    
    async def sync_directory(self, storage_name: str, local_dir: str, remote_prefix: str = "",
                            exclude_patterns: Optional[List[str]] = None,
                            delete_remote: bool = False) -> SyncResult:
        """Synchronize local directory with remote storage."""
        try:
            result = SyncResult()
            start_time = time.time()
            
            exclude_patterns = exclude_patterns or []
            
            # Get local files
            local_files = {}
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, local_dir).replace('\\', '/')
                    
                    # Check exclude patterns
                    if any(self._matches_pattern(rel_path, pattern) for pattern in exclude_patterns):
                        continue
                    
                    local_files[rel_path] = {
                        'path': local_path,
                        'size': os.path.getsize(local_path),
                        'mtime': os.path.getmtime(local_path)
                    }
            
            # Get remote files
            remote_objects = await self.list_objects(storage_name, remote_prefix)
            remote_files = {obj.key[len(remote_prefix):].lstrip('/'): obj for obj in remote_objects}
            
            # Compare and sync
            for rel_path, local_info in local_files.items():
                remote_key = f"{remote_prefix}/{rel_path}".strip('/')
                
                should_upload = True
                if rel_path in remote_files:
                    remote_obj = remote_files[rel_path]
                    # Compare size and modification time
                    if (local_info['size'] == remote_obj.size and 
                        local_info['mtime'] <= remote_obj.last_modified.timestamp()):
                        should_upload = False
                
                if should_upload:
                    success = await self.upload_file(storage_name, local_info['path'], remote_key)
                    if success:
                        result.uploaded.append(rel_path)
                        result.total_bytes_transferred += local_info['size']
                    else:
                        result.errors.append(f"Failed to upload {rel_path}")
            
            # Delete remote files not in local (if requested)
            if delete_remote:
                for rel_path in remote_files:
                    if rel_path not in local_files:
                        remote_key = f"{remote_prefix}/{rel_path}".strip('/')
                        success = await self.delete_object(storage_name, remote_key)
                        if success:
                            result.deleted.append(rel_path)
                        else:
                            result.errors.append(f"Failed to delete {rel_path}")
            
            result.duration_seconds = time.time() - start_time
            
            self.logger.info(f"Sync completed: {len(result.uploaded)} uploaded, "
                           f"{len(result.deleted)} deleted, {len(result.errors)} errors")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Directory sync failed: {e}")
            result.errors.append(str(e))
            return result
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches exclude pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)
    
    async def generate_presigned_url(self, storage_name: str, remote_key: str,
                                   expiration: int = 3600, method: str = 'GET') -> Optional[str]:
        """Generate presigned URL for object access."""
        try:
            if storage_name not in self.storage_configs:
                raise Exception(f"Storage {storage_name} not configured")
            
            config = self.storage_configs[storage_name]
            connection = self.active_connections[storage_name]
            
            if config.provider == StorageProvider.AWS_S3 or config.provider == StorageProvider.MINIO:
                url = connection.generate_presigned_url(
                    'get_object' if method == 'GET' else 'put_object',
                    Params={'Bucket': config.bucket_name, 'Key': remote_key},
                    ExpiresIn=expiration
                )
                return url
                
            elif config.provider == StorageProvider.GOOGLE_CLOUD:
                bucket = connection.bucket(config.bucket_name)
                blob = bucket.blob(remote_key)
                
                if method == 'GET':
                    url = blob.generate_signed_url(expiration=timedelta(seconds=expiration))
                else:
                    url = blob.generate_signed_url(
                        expiration=timedelta(seconds=expiration),
                        method='PUT'
                    )
                return url
                
            elif config.provider == StorageProvider.AZURE_BLOB:
                from azure.storage.blob import generate_blob_sas, BlobSasPermissions
                
                permissions = BlobSasPermissions(read=True) if method == 'GET' else BlobSasPermissions(write=True)
                
                sas_token = generate_blob_sas(
                    account_name=connection.account_name,
                    container_name=config.bucket_name,
                    blob_name=remote_key,
                    account_key=connection.credential.account_key,
                    permission=permissions,
                    expiry=datetime.now() + timedelta(seconds=expiration)
                )
                
                url = f"https://{connection.account_name}.blob.core.windows.net/{config.bucket_name}/{remote_key}?{sas_token}"
                return url
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def _map_storage_class_s3(self, storage_class: StorageClass) -> str:
        """Map generic storage class to S3-specific class."""
        mapping = {
            StorageClass.STANDARD: 'STANDARD',
            StorageClass.REDUCED_REDUNDANCY: 'REDUCED_REDUNDANCY',
            StorageClass.COLD: 'STANDARD_IA',
            StorageClass.ARCHIVE: 'GLACIER',
            StorageClass.DEEP_ARCHIVE: 'DEEP_ARCHIVE'
        }
        return mapping.get(storage_class, 'STANDARD')
    
    def _map_storage_class_gcs(self, storage_class: StorageClass) -> str:
        """Map generic storage class to GCS-specific class."""
        mapping = {
            StorageClass.STANDARD: 'STANDARD',
            StorageClass.REDUCED_REDUNDANCY: 'NEARLINE',
            StorageClass.COLD: 'COLDLINE',
            StorageClass.ARCHIVE: 'ARCHIVE',
            StorageClass.DEEP_ARCHIVE: 'ARCHIVE'
        }
        return mapping.get(storage_class, 'STANDARD')
    
    def get_upload_progress(self, progress_id: str) -> Optional[UploadProgress]:
        """Get upload progress for a specific upload."""
        return self.upload_progress.get(progress_id)
    
    def get_download_progress(self, progress_id: str) -> Optional[UploadProgress]:
        """Get download progress for a specific download."""
        return self.download_progress.get(progress_id)
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get cloud storage summary."""
        try:
            return {
                'configured_storages': list(self.storage_configs.keys()),
                'active_connections': list(self.active_connections.keys()),
                'active_uploads': len(self.upload_progress),
                'active_downloads': len(self.download_progress),
                'cache_size': len(self.object_cache),
                'supported_providers': [provider.value for provider in StorageProvider],
                'storage_configs': {
                    name: {
                        'provider': config.provider.value,
                        'bucket': config.bucket_name,
                        'region': config.region,
                        'encryption': config.encryption_enabled,
                        'versioning': config.versioning_enabled
                    }
                    for name, config in self.storage_configs.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate storage summary: {e}")
            return {'error': 'Unable to generate summary'}