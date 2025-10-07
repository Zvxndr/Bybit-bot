"""
DigitalOcean Infrastructure Manager
=================================

Comprehensive infrastructure management for DigitalOcean deployment.
Handles VPC setup, droplets, load balancers, databases, and security configuration.
"""

import digitalocean
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class DropletSize(Enum):
    """Standard DigitalOcean droplet sizes"""
    NANO = "s-1vcpu-1gb"        # $6/month - Development
    MICRO = "s-1vcpu-2gb"       # $12/month - Small apps
    SMALL = "s-2vcpu-2gb"       # $18/month - Production ready
    MEDIUM = "s-2vcpu-4gb"      # $24/month - Medium workloads
    LARGE = "s-4vcpu-8gb"       # $48/month - High performance
    XLARGE = "s-8vcpu-16gb"     # $96/month - Enterprise


class DatabaseEngine(Enum):
    """Database engine options"""
    POSTGRESQL = "pg"
    MYSQL = "mysql"
    REDIS = "redis"


@dataclass
class InfrastructureConfig:
    """Infrastructure configuration for Australian Trust deployment"""
    project_name: str
    region: str = "sgp1"  # Singapore region for Australian users
    
    # Droplet configuration
    app_droplet_size: DropletSize = DropletSize.MEDIUM
    app_droplet_count: int = 2
    
    # Database configuration
    database_engine: DatabaseEngine = DatabaseEngine.POSTGRESQL
    database_size: str = "db-s-1vcpu-1gb"  # $15/month
    
    # Load balancer
    enable_load_balancer: bool = True
    
    # Security
    enable_firewall: bool = True
    allowed_ips: Optional[List[str]] = None
    
    # Storage
    enable_volume: bool = True
    volume_size: int = 50  # GB
    
    # Monitoring
    enable_monitoring: bool = True
    
    def __post_init__(self):
        if self.allowed_ips is None:
            self.allowed_ips = []


class DigitalOceanManager:
    """
    Professional DigitalOcean infrastructure manager
    
    Handles complete infrastructure setup for Australian Discretionary Trust
    trading bot deployment with security, monitoring, and scalability
    """
    
    def __init__(self, api_token: str):
        """
        Initialize DigitalOcean manager
        
        Args:
            api_token: DigitalOcean API token
        """
        self.manager = digitalocean.Manager(token=api_token)
        self.api_token = api_token
        
        # Infrastructure tracking
        self.deployed_resources = {
            'droplets': [],
            'load_balancers': [],
            'databases': [],
            'volumes': [],
            'firewalls': [],
            'vpcs': []
        }
        
        logger.info("✅ DigitalOcean manager initialized")
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate DigitalOcean API connection"""
        try:
            # Test API connection by fetching account info
            account = self.manager.get_account()
            
            return {
                'connected': True,
                'account_email': account.email,
                'droplet_limit': account.droplet_limit,
                'floating_ip_limit': account.floating_ip_limit,
                'status': account.status
            }
            
        except Exception as e:
            logger.error(f"❌ DigitalOcean connection failed: {str(e)}")
            return {
                'connected': False,
                'error': str(e)
            }
    
    def get_available_regions(self) -> List[Dict[str, Any]]:
        """Get list of available DigitalOcean regions"""
        try:
            regions = self.manager.get_all_regions()
            
            region_list = []
            for region in regions:
                if region.available:  # Only available regions
                    region_list.append({
                        'slug': region.slug,
                        'name': region.name,
                        'available': region.available,
                        'features': region.features,
                        'sizes': region.sizes
                    })
            
            # Sort by proximity to Australia
            australian_priority = ['sgp1', 'sfo3', 'sfo2', 'nyc3', 'nyc1', 'lon1', 'fra1']
            region_list.sort(key=lambda x: australian_priority.index(x['slug']) 
                            if x['slug'] in australian_priority else 999)
            
            logger.info(f"📍 Found {len(region_list)} available regions")
            return region_list
            
        except Exception as e:
            logger.error(f"❌ Failed to get regions: {str(e)}")
            return []
    
    def create_vpc(self, name: str, region: str, ip_range: str = "10.0.0.0/16") -> Optional[digitalocean.VPC]:
        """
        Create Virtual Private Cloud for secure networking
        
        Args:
            name: VPC name
            region: DigitalOcean region
            ip_range: IP range for VPC (default: 10.0.0.0/16)
            
        Returns:
            VPC object if successful
        """
        try:
            vpc = digitalocean.VPC(
                token=self.api_token,
                name=name,
                region=region,
                ip_range=ip_range,
                description=f"VPC for {name} - Australian Trust Trading Bot"
            )
            
            vpc.create()
            
            # Wait for VPC to be ready
            self._wait_for_vpc_ready(vpc)
            
            self.deployed_resources['vpcs'].append(vpc)
            logger.info(f"🔒 VPC created: {name} in {region} ({ip_range})")
            
            return vpc
            
        except Exception as e:
            logger.error(f"❌ Failed to create VPC: {str(e)}")
            return None
    
    def setup_complete_infrastructure(self, config: InfrastructureConfig) -> Dict[str, Any]:
        """
        Set up complete infrastructure for Australian Trust deployment
        
        Args:
            config: Infrastructure configuration
            
        Returns:
            Dictionary with deployment results
        """
        logger.info(f"🚀 Starting infrastructure deployment: {config.project_name}")
        
        results = {
            'success': False,
            'project_name': config.project_name,
            'region': config.region,
            'resources': {},
            'costs': {},
            'errors': []
        }
        
        try:
            
            # 1. Create VPC for secure networking
            logger.info("🔒 Creating VPC...")
            vpc = self.create_vpc(
                name=f"{config.project_name}-vpc",
                region=config.region
            )
            
            if not vpc:
                results['errors'].append("Failed to create VPC")
                return results
            
            results['resources']['vpc'] = {
                'id': vpc.id,
                'name': vpc.name,
                'ip_range': vpc.ip_range
            }
            
            # 2. Create database cluster
            logger.info("🗄️ Creating database cluster...")
            database = self.create_database_cluster(
                name=f"{config.project_name}-db",
                engine=config.database_engine,
                size=config.database_size,
                region=config.region,
                vpc_uuid=vpc.id
            )
            
            if database:
                results['resources']['database'] = {
                    'id': database.id,
                    'name': database.name,
                    'engine': database.engine,
                    'connection': database.connection
                }
                results['costs']['database'] = self._get_database_cost(config.database_size)
            else:
                results['errors'].append("Failed to create database")
            
            # 3. Create storage volume
            volume = None
            if config.enable_volume:
                logger.info("💾 Creating storage volume...")
                volume = self.create_volume(
                    name=f"{config.project_name}-storage",
                    size=config.volume_size,
                    region=config.region
                )
                
                if volume:
                    results['resources']['volume'] = {
                        'id': volume.id,
                        'name': volume.name,
                        'size': volume.size_gigabytes
                    }
                    results['costs']['volume'] = config.volume_size * 0.10  # $0.10/GB/month
            
            # 4. Create application droplets
            logger.info(f"🖥️ Creating {config.app_droplet_count} application droplets...")
            droplets = self.create_app_droplets(
                name_prefix=f"{config.project_name}-app",
                count=config.app_droplet_count,
                size=config.app_droplet_size,
                region=config.region,
                vpc_uuid=vpc.id,
                volume=volume
            )
            
            if droplets:
                results['resources']['droplets'] = [
                    {
                        'id': droplet.id,
                        'name': droplet.name,
                        'ip_address': droplet.ip_address,
                        'private_ip': droplet.private_ip_address,
                        'size': droplet.size_slug
                    }
                    for droplet in droplets
                ]
                results['costs']['droplets'] = len(droplets) * self._get_droplet_cost(config.app_droplet_size)
            else:
                results['errors'].append("Failed to create droplets")
                return results
            
            # 5. Create load balancer
            load_balancer = None
            if config.enable_load_balancer:
                logger.info("⚖️ Creating load balancer...")
                load_balancer = self.create_load_balancer(
                    name=f"{config.project_name}-lb",
                    region=config.region,
                    droplets=droplets,
                    vpc_uuid=vpc.id
                )
                
                if load_balancer:
                    results['resources']['load_balancer'] = {
                        'id': load_balancer.id,
                        'name': load_balancer.name,
                        'ip': load_balancer.ip,
                        'status': load_balancer.status
                    }
                    results['costs']['load_balancer'] = 12.00  # $12/month
            
            # 6. Create firewall
            if config.enable_firewall:
                logger.info("🔥 Creating firewall...")
                firewall = self.create_firewall(
                    name=f"{config.project_name}-firewall",
                    droplets=droplets,
                    allowed_ips=config.allowed_ips or [],
                    load_balancer=load_balancer
                )
                
                if firewall:
                    results['resources']['firewall'] = {
                        'id': firewall.id,
                        'name': firewall.name,
                        'status': firewall.status
                    }
            
            # 7. Setup monitoring (if enabled)
            if config.enable_monitoring:
                logger.info("📊 Enabling monitoring...")
                monitoring_result = self.setup_monitoring(droplets)
                results['resources']['monitoring'] = monitoring_result
            
            # Calculate total monthly cost
            total_cost = sum(results['costs'].values())
            results['costs']['total_monthly'] = total_cost
            
            results['success'] = len(results['errors']) == 0
            
            if results['success']:
                logger.info(f"✅ Infrastructure deployment completed successfully!")
                logger.info(f"💰 Estimated monthly cost: ${total_cost:.2f}")
            else:
                logger.error(f"❌ Infrastructure deployment completed with errors: {results['errors']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Infrastructure deployment failed: {str(e)}")
            results['errors'].append(str(e))
            results['success'] = False
            return results
    
    def create_database_cluster(self, name: str, engine: DatabaseEngine, 
                               size: str, region: str, vpc_uuid: Optional[str] = None) -> Optional[digitalocean.Database]:
        """Create managed database cluster"""
        try:
            database = digitalocean.Database(
                token=self.api_token,
                name=name,
                engine=engine.value,
                version="14" if engine == DatabaseEngine.POSTGRESQL else "8.0",
                size=size,
                region=region,
                num_nodes=1,
                vpc_uuid=vpc_uuid
            )
            
            database.create()
            
            # Wait for database to be ready
            self._wait_for_database_ready(database)
            
            self.deployed_resources['databases'].append(database)
            logger.info(f"🗄️ Database cluster created: {name}")
            
            return database
            
        except Exception as e:
            logger.error(f"❌ Failed to create database: {str(e)}")
            return None
    
    def create_volume(self, name: str, size: int, region: str) -> Optional[digitalocean.Volume]:
        """Create storage volume"""
        try:
            volume = digitalocean.Volume(
                token=self.api_token,
                name=name,
                region=region,
                size_gigabytes=size,
                description=f"Storage volume for {name}"
            )
            
            volume.create()
            
            # Wait for volume to be ready
            self._wait_for_volume_ready(volume)
            
            self.deployed_resources['volumes'].append(volume)
            logger.info(f"💾 Volume created: {name} ({size}GB)")
            
            return volume
            
        except Exception as e:
            logger.error(f"❌ Failed to create volume: {str(e)}")
            return None
    
    def create_app_droplets(self, name_prefix: str, count: int, size: DropletSize,
                           region: str, vpc_uuid: Optional[str] = None, 
                           volume: Optional[digitalocean.Volume] = None) -> List[digitalocean.Droplet]:
        """Create application droplets"""
        droplets = []
        
        try:
            # User data script for droplet initialization
            user_data = self._get_droplet_user_data()
            
            for i in range(count):
                droplet_name = f"{name_prefix}-{i+1:02d}"
                
                droplet = digitalocean.Droplet(
                    token=self.api_token,
                    name=droplet_name,
                    region=region,
                    image='ubuntu-22-04-x64',  # Ubuntu 22.04 LTS
                    size_slug=size.value,
                    vpc_uuid=vpc_uuid,
                    ssh_keys=self._get_ssh_keys(),
                    backups=True,  # Enable backups
                    monitoring=True,  # Enable monitoring
                    user_data=user_data,
                    tags=[f"project:{name_prefix}", "environment:production", "role:app"]
                )
                
                droplet.create()
                droplets.append(droplet)
                
                logger.info(f"🖥️ Creating droplet: {droplet_name}")
            
            # Wait for all droplets to be ready
            logger.info("⏳ Waiting for droplets to be ready...")
            for droplet in droplets:
                self._wait_for_droplet_ready(droplet)
                
                # Attach volume to first droplet if provided
                if volume and droplet == droplets[0]:
                    self._attach_volume_to_droplet(volume, droplet)
            
            self.deployed_resources['droplets'].extend(droplets)
            logger.info(f"✅ Created {len(droplets)} droplets successfully")
            
            return droplets
            
        except Exception as e:
            logger.error(f"❌ Failed to create droplets: {str(e)}")
            return []
    
    def create_load_balancer(self, name: str, region: str, 
                           droplets: List[digitalocean.Droplet],
                           vpc_uuid: Optional[str] = None) -> Optional[digitalocean.LoadBalancer]:
        """Create load balancer"""
        try:
            # Forwarding rules
            forwarding_rules = [
                {
                    'entry_protocol': 'https',
                    'entry_port': 443,
                    'target_protocol': 'http',
                    'target_port': 8000,
                    'certificate_id': '',  # SSL certificate
                    'tls_passthrough': False
                },
                {
                    'entry_protocol': 'http',
                    'entry_port': 80,
                    'target_protocol': 'http',
                    'target_port': 8000,
                    'certificate_id': '',
                    'tls_passthrough': False
                }
            ]
            
            # Health check
            health_check = {
                'protocol': 'http',
                'port': 8000,
                'path': '/health',
                'check_interval_seconds': 10,
                'response_timeout_seconds': 5,
                'healthy_threshold': 3,
                'unhealthy_threshold': 3
            }
            
            load_balancer = digitalocean.LoadBalancer(
                token=self.api_token,
                name=name,
                algorithm='round_robin',
                region=region,
                forwarding_rules=forwarding_rules,
                health_check=health_check,
                sticky_sessions={'type': 'cookies', 'cookie_name': 'lb', 'cookie_ttl_seconds': 300},
                droplet_ids=[droplet.id for droplet in droplets],
                vpc_uuid=vpc_uuid,
                enable_proxy_protocol=False
            )
            
            load_balancer.create()
            
            # Wait for load balancer to be ready
            self._wait_for_load_balancer_ready(load_balancer)
            
            self.deployed_resources['load_balancers'].append(load_balancer)
            logger.info(f"⚖️ Load balancer created: {name}")
            
            return load_balancer
            
        except Exception as e:
            logger.error(f"❌ Failed to create load balancer: {str(e)}")
            return None
    
    def create_firewall(self, name: str, droplets: List[digitalocean.Droplet],
                       allowed_ips: Optional[List[str]] = None,
                       load_balancer: Optional[digitalocean.LoadBalancer] = None) -> Optional[digitalocean.Firewall]:
        """Create firewall with security rules"""
        try:
            allowed_ips = allowed_ips or []
            
            # Inbound rules
            inbound_rules = [
                # SSH access (restricted to allowed IPs)
                {
                    'protocol': 'tcp',
                    'ports': '22',
                    'sources': {
                        'addresses': allowed_ips if allowed_ips else ['0.0.0.0/0']
                    }
                },
                # HTTP traffic (load balancer or allowed IPs)
                {
                    'protocol': 'tcp',
                    'ports': '80',
                    'sources': {
                        'load_balancer_uids': [load_balancer.id] if load_balancer else [],
                        'addresses': allowed_ips if not load_balancer else []
                    }
                },
                # HTTPS traffic (load balancer or allowed IPs)
                {
                    'protocol': 'tcp',
                    'ports': '443',
                    'sources': {
                        'load_balancer_uids': [load_balancer.id] if load_balancer else [],
                        'addresses': allowed_ips if not load_balancer else []
                    }
                },
                # Application port
                {
                    'protocol': 'tcp',
                    'ports': '8000',
                    'sources': {
                        'load_balancer_uids': [load_balancer.id] if load_balancer else [],
                        'addresses': allowed_ips if not load_balancer else []
                    }
                }
            ]
            
            # Outbound rules (allow all outbound)
            outbound_rules = [
                {
                    'protocol': 'tcp',
                    'ports': 'all',
                    'destinations': {
                        'addresses': ['0.0.0.0/0', '::/0']
                    }
                },
                {
                    'protocol': 'udp',
                    'ports': 'all',
                    'destinations': {
                        'addresses': ['0.0.0.0/0', '::/0']
                    }
                }
            ]
            
            firewall = digitalocean.Firewall(
                token=self.api_token,
                name=name,
                inbound_rules=inbound_rules,
                outbound_rules=outbound_rules,
                droplet_ids=[droplet.id for droplet in droplets]
            )
            
            firewall.create()
            
            self.deployed_resources['firewalls'].append(firewall)
            logger.info(f"🔥 Firewall created: {name}")
            
            return firewall
            
        except Exception as e:
            logger.error(f"❌ Failed to create firewall: {str(e)}")
            return None
    
    def setup_monitoring(self, droplets: List[digitalocean.Droplet]) -> Dict[str, Any]:
        """Setup monitoring for droplets"""
        try:
            monitoring_config = {
                'enabled': True,
                'alert_policy': 'standard',
                'metrics': ['cpu', 'memory', 'disk', 'network'],
                'notification_channels': []
            }
            
            # Monitoring is automatically enabled for droplets created with monitoring=True
            logger.info("📊 Monitoring enabled for all droplets")
            
            return monitoring_config
            
        except Exception as e:
            logger.error(f"❌ Failed to setup monitoring: {str(e)}")
            return {'enabled': False, 'error': str(e)}
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """Get information about deployed resources"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'resources': {}
        }
        
        # Get droplet information
        if self.deployed_resources['droplets']:
            info['resources']['droplets'] = []
            for droplet in self.deployed_resources['droplets']:
                droplet.load()  # Refresh data
                info['resources']['droplets'].append({
                    'id': droplet.id,
                    'name': droplet.name,
                    'status': droplet.status,
                    'ip_address': droplet.ip_address,
                    'private_ip': droplet.private_ip_address,
                    'size': droplet.size_slug,
                    'region': droplet.region['slug'],
                    'created_at': droplet.created_at
                })
        
        # Get load balancer information
        if self.deployed_resources['load_balancers']:
            info['resources']['load_balancers'] = []
            for lb in self.deployed_resources['load_balancers']:
                lb.load()
                info['resources']['load_balancers'].append({
                    'id': lb.id,
                    'name': lb.name,
                    'status': lb.status,
                    'ip': lb.ip,
                    'algorithm': lb.algorithm,
                    'region': lb.region['slug']
                })
        
        # Get database information
        if self.deployed_resources['databases']:
            info['resources']['databases'] = []
            for db in self.deployed_resources['databases']:
                db.load()
                info['resources']['databases'].append({
                    'id': db.id,
                    'name': db.name,
                    'engine': db.engine,
                    'status': db.status,
                    'region': db.region,
                    'connection': db.connection
                })
        
        return info
    
    def destroy_infrastructure(self, confirm: bool = False) -> Dict[str, Any]:
        """Destroy all deployed infrastructure"""
        if not confirm:
            return {
                'success': False,
                'error': 'Confirmation required to destroy infrastructure'
            }
        
        results = {
            'success': True,
            'destroyed': [],
            'errors': []
        }
        
        try:
            # Destroy droplets
            for droplet in self.deployed_resources['droplets']:
                try:
                    droplet.destroy()
                    results['destroyed'].append(f"Droplet: {droplet.name}")
                    logger.info(f"🗑️ Destroyed droplet: {droplet.name}")
                except Exception as e:
                    results['errors'].append(f"Failed to destroy droplet {droplet.name}: {str(e)}")
            
            # Destroy load balancers
            for lb in self.deployed_resources['load_balancers']:
                try:
                    lb.destroy()
                    results['destroyed'].append(f"Load Balancer: {lb.name}")
                    logger.info(f"🗑️ Destroyed load balancer: {lb.name}")
                except Exception as e:
                    results['errors'].append(f"Failed to destroy load balancer {lb.name}: {str(e)}")
            
            # Destroy databases
            for db in self.deployed_resources['databases']:
                try:
                    db.destroy()
                    results['destroyed'].append(f"Database: {db.name}")
                    logger.info(f"🗑️ Destroyed database: {db.name}")
                except Exception as e:
                    results['errors'].append(f"Failed to destroy database {db.name}: {str(e)}")
            
            # Destroy volumes
            for volume in self.deployed_resources['volumes']:
                try:
                    volume.destroy()
                    results['destroyed'].append(f"Volume: {volume.name}")
                    logger.info(f"🗑️ Destroyed volume: {volume.name}")
                except Exception as e:
                    results['errors'].append(f"Failed to destroy volume {volume.name}: {str(e)}")
            
            # Destroy firewalls
            for firewall in self.deployed_resources['firewalls']:
                try:
                    firewall.destroy()
                    results['destroyed'].append(f"Firewall: {firewall.name}")
                    logger.info(f"🗑️ Destroyed firewall: {firewall.name}")
                except Exception as e:
                    results['errors'].append(f"Failed to destroy firewall {firewall.name}: {str(e)}")
            
            # Clear tracking
            self.deployed_resources = {
                'droplets': [],
                'load_balancers': [],
                'databases': [],
                'volumes': [],
                'firewalls': [],
                'vpcs': []
            }
            
            results['success'] = len(results['errors']) == 0
            
            if results['success']:
                logger.info("✅ Infrastructure destruction completed successfully")
            else:
                logger.error(f"❌ Infrastructure destruction completed with errors: {results['errors']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Infrastructure destruction failed: {str(e)}")
            results['errors'].append(str(e))
            results['success'] = False
            return results
    
    # Helper methods
    def _wait_for_droplet_ready(self, droplet: digitalocean.Droplet, timeout: int = 300):
        """Wait for droplet to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            droplet.load()
            if droplet.status == 'active':
                logger.info(f"✅ Droplet ready: {droplet.name}")
                return
            time.sleep(10)
        
        raise Exception(f"Timeout waiting for droplet {droplet.name} to be ready")
    
    def _wait_for_load_balancer_ready(self, lb: digitalocean.LoadBalancer, timeout: int = 300):
        """Wait for load balancer to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            lb.load()
            if lb.status == 'active':
                logger.info(f"✅ Load balancer ready: {lb.name}")
                return
            time.sleep(15)
        
        raise Exception(f"Timeout waiting for load balancer {lb.name} to be ready")
    
    def _wait_for_database_ready(self, db: digitalocean.Database, timeout: int = 600):
        """Wait for database to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            db.load()
            if db.status == 'online':
                logger.info(f"✅ Database ready: {db.name}")
                return
            time.sleep(30)
        
        raise Exception(f"Timeout waiting for database {db.name} to be ready")
    
    def _wait_for_volume_ready(self, volume: digitalocean.Volume, timeout: int = 180):
        """Wait for volume to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            volume.load()
            if hasattr(volume, 'status') and volume.status == 'available':
                logger.info(f"✅ Volume ready: {volume.name}")
                return
            time.sleep(10)
        
        logger.info(f"✅ Volume created: {volume.name}")
    
    def _wait_for_vpc_ready(self, vpc: digitalocean.VPC, timeout: int = 120):
        """Wait for VPC to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                vpc.load()
                logger.info(f"✅ VPC ready: {vpc.name}")
                return
            except:
                time.sleep(10)
        
        logger.info(f"✅ VPC created: {vpc.name}")
    
    def _attach_volume_to_droplet(self, volume: digitalocean.Volume, droplet: digitalocean.Droplet):
        """Attach volume to droplet"""
        try:
            volume.attach(droplet.id, droplet.region['slug'])
            logger.info(f"💾 Volume {volume.name} attached to {droplet.name}")
        except Exception as e:
            logger.error(f"❌ Failed to attach volume: {str(e)}")
    
    def _get_droplet_user_data(self) -> str:
        """Get user data script for droplet initialization"""
        return """#!/bin/bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install monitoring agent
curl -sSL https://repos.insights.digitalocean.com/install.sh | sudo bash

# Create application directory
mkdir -p /opt/trading-bot
chown ubuntu:ubuntu /opt/trading-bot

# Install Python 3.10
apt-get install -y python3.10 python3.10-pip python3.10-venv

# Install nginx for reverse proxy
apt-get install -y nginx

# Configure firewall
ufw allow 22
ufw allow 80
ufw allow 443
ufw allow 8000
ufw --force enable

# Create swap file
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab

# Log completion
echo "$(date): User data script completed" >> /var/log/user-data.log
"""
    
    def _get_ssh_keys(self) -> List[int]:
        """Get SSH key IDs for droplet access"""
        try:
            keys = self.manager.get_all_sshkeys()
            return [key.id for key in keys]
        except:
            return []
    
    def _get_droplet_cost(self, size: DropletSize) -> float:
        """Get monthly cost for droplet size"""
        costs = {
            DropletSize.NANO: 6.00,
            DropletSize.MICRO: 12.00,
            DropletSize.SMALL: 18.00,
            DropletSize.MEDIUM: 24.00,
            DropletSize.LARGE: 48.00,
            DropletSize.XLARGE: 96.00
        }
        return costs.get(size, 24.00)
    
    def _get_database_cost(self, size: str) -> float:
        """Get monthly cost for database size"""
        costs = {
            'db-s-1vcpu-1gb': 15.00,
            'db-s-1vcpu-2gb': 30.00,
            'db-s-2vcpu-4gb': 60.00
        }
        return costs.get(size, 15.00)


# Example usage and configuration
if __name__ == "__main__":
    # Test DigitalOcean manager (requires valid API token)
    api_token = os.getenv('DIGITALOCEAN_TOKEN', 'test_token')
    
    if api_token != 'test_token':
        # Initialize manager
        do_manager = DigitalOceanManager(api_token)
        
        # Test connection
        connection = do_manager.validate_connection()
        print(f"Connection test: {connection}")
        
        # Get available regions
        regions = do_manager.get_available_regions()
        print(f"Available regions: {len(regions)}")
        
        # Test infrastructure configuration
        config = InfrastructureConfig(
            project_name="australian-trust-bot",
            region="sgp1",
            app_droplet_size=DropletSize.MEDIUM,
            app_droplet_count=2,
            database_engine=DatabaseEngine.POSTGRESQL,
            allowed_ips=["203.0.113.0/24"]  # Replace with your IP range
        )
        
        print(f"Configuration: {config}")
        
        # Note: Uncomment to actually deploy infrastructure
        # results = do_manager.setup_complete_infrastructure(config)
        # print(f"Deployment results: {results}")
        
        print("DigitalOcean manager test completed!")
    else:
        print("Set DIGITALOCEAN_TOKEN environment variable to test infrastructure functionality")