"""
Network Manager for Cloud Networking and Security.
Comprehensive network management with VPC, security groups, load balancing, and traffic monitoring.
"""

import asyncio
import json
import time
import ipaddress
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import boto3
from botocore.exceptions import ClientError
import aiohttp
import socket
import subprocess
import psutil

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class NetworkProvider(Enum):
    """Supported network providers."""
    AWS_VPC = "aws_vpc"
    GOOGLE_VPC = "google_vpc"
    AZURE_VNET = "azure_vnet"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    LOCAL = "local"

class ProtocolType(Enum):
    """Network protocols."""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    HTTP = "http"
    HTTPS = "https"
    SSH = "ssh"
    FTP = "ftp"
    SMTP = "smtp"

class TrafficDirection(Enum):
    """Traffic direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"

class LoadBalancerType(Enum):
    """Load balancer types."""
    APPLICATION = "application"
    NETWORK = "network"
    CLASSIC = "classic"
    INTERNAL = "internal"

class SecurityRuleAction(Enum):
    """Security rule actions."""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"

@dataclass
class NetworkRange:
    """Network range definition."""
    cidr: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate CIDR
        try:
            ipaddress.ip_network(self.cidr, strict=False)
        except ValueError as e:
            raise ValueError(f"Invalid CIDR block: {self.cidr}")

@dataclass
class SecurityRule:
    """Network security rule."""
    name: str
    protocol: ProtocolType
    port_range: Union[int, Tuple[int, int]]
    source_cidr: str
    destination_cidr: str
    action: SecurityRuleAction
    direction: TrafficDirection
    description: str = ""
    priority: int = 100
    enabled: bool = True

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    name: str
    type: LoadBalancerType
    scheme: str = "internet-facing"  # internet-facing or internal
    subnets: List[str] = field(default_factory=list)
    security_groups: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    health_check_port: int = 8080
    health_check_protocol: str = "HTTP"
    idle_timeout: int = 60
    cross_zone_load_balancing: bool = True
    deletion_protection: bool = False

@dataclass
class TargetGroup:
    """Load balancer target group."""
    name: str
    protocol: str
    port: int
    vpc_id: str
    health_check_enabled: bool = True
    health_check_interval: int = 30
    health_check_timeout: int = 5
    healthy_threshold_count: int = 2
    unhealthy_threshold_count: int = 2
    targets: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class NetworkEndpoint:
    """Network endpoint configuration."""
    name: str
    host: str
    port: int
    protocol: str = "tcp"
    health_check_enabled: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class TrafficStats:
    """Network traffic statistics."""
    bytes_in: int
    bytes_out: int
    packets_in: int
    packets_out: int
    connections_active: int
    connections_total: int
    errors: int
    timestamp: datetime

@dataclass
class NetworkMonitoring:
    """Network monitoring configuration."""
    enabled: bool = True
    metrics_interval: int = 60  # seconds
    log_traffic: bool = False
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    retention_days: int = 30

class NetworkManager:
    """Comprehensive network management system."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Network configuration
        self.provider_configs: Dict[str, Dict[str, Any]] = {}
        self.active_providers: Dict[str, Any] = {}
        
        # Network resources
        self.vpcs: Dict[str, Dict[str, Any]] = {}
        self.subnets: Dict[str, Dict[str, Any]] = {}
        self.security_groups: Dict[str, Dict[str, Any]] = {}
        self.load_balancers: Dict[str, Dict[str, Any]] = {}
        self.target_groups: Dict[str, Dict[str, Any]] = {}
        
        # Security rules
        self.security_rules: Dict[str, SecurityRule] = {}
        self.firewall_rules: Dict[str, List[SecurityRule]] = {}
        
        # Network endpoints
        self.endpoints: Dict[str, NetworkEndpoint] = {}
        self.endpoint_health: Dict[str, Dict[str, Any]] = {}
        
        # Traffic monitoring
        self.traffic_stats: Dict[str, List[TrafficStats]] = {}
        self.monitoring_config = NetworkMonitoring()
        self.monitoring_active = False
        
        # DNS and service discovery
        self.dns_records: Dict[str, Dict[str, Any]] = {}
        self.service_registry: Dict[str, Dict[str, Any]] = {}
        
        # Network policies
        self.network_policies: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        self.logger.info("NetworkManager initialized")
    
    def _initialize_default_configs(self):
        """Initialize default network configurations."""
        try:
            # AWS VPC configuration
            self.provider_configs[NetworkProvider.AWS_VPC.value] = {
                'region': 'us-east-1',
                'availability_zones': ['us-east-1a', 'us-east-1b', 'us-east-1c'],
                'vpc_cidr': '10.0.0.0/16',
                'public_subnet_cidrs': ['10.0.1.0/24', '10.0.2.0/24', '10.0.3.0/24'],
                'private_subnet_cidrs': ['10.0.10.0/24', '10.0.20.0/24', '10.0.30.0/24'],
                'enable_dns_hostnames': True,
                'enable_dns_support': True
            }
            
            # Kubernetes networking
            self.provider_configs[NetworkProvider.KUBERNETES.value] = {
                'cluster_cidr': '10.244.0.0/16',
                'service_cidr': '10.96.0.0/12',
                'dns_domain': 'cluster.local',
                'network_plugin': 'flannel'  # flannel, calico, weave
            }
            
            # Docker networking
            self.provider_configs[NetworkProvider.DOCKER.value] = {
                'default_network': 'bridge',
                'custom_networks': {
                    'trading-network': {
                        'driver': 'bridge',
                        'subnet': '172.20.0.0/16',
                        'gateway': '172.20.0.1'
                    }
                }
            }
            
            # Local networking
            self.provider_configs[NetworkProvider.LOCAL.value] = {
                'interface': 'eth0',
                'monitoring_enabled': True,
                'firewall_enabled': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default network configs: {e}")
    
    async def configure_provider(self, provider: NetworkProvider, config: Dict[str, Any]):
        """Configure a network provider."""
        try:
            self.provider_configs[provider.value] = config
            
            # Initialize provider connection
            await self._initialize_provider_connection(provider, config)
            
            self.logger.info(f"Network provider {provider.value} configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to configure network provider {provider.value}: {e}")
            raise
    
    async def _initialize_provider_connection(self, provider: NetworkProvider, config: Dict[str, Any]):
        """Initialize connection to network provider."""
        try:
            if provider == NetworkProvider.AWS_VPC:
                session_kwargs = {
                    'region_name': config.get('region', 'us-east-1')
                }
                
                if config.get('access_key_id') and config.get('secret_access_key'):
                    session_kwargs.update({
                        'aws_access_key_id': config['access_key_id'],
                        'aws_secret_access_key': config['secret_access_key']
                    })
                
                session = boto3.Session(**session_kwargs)
                ec2_client = session.client('ec2')
                elb_client = session.client('elbv2')
                
                self.active_providers[provider.value] = {
                    'ec2': ec2_client,
                    'elb': elb_client,
                    'session': session
                }
                
            elif provider == NetworkProvider.KUBERNETES:
                from kubernetes import client, config as k8s_config
                
                kubeconfig_path = config.get('kubeconfig_path')
                if kubeconfig_path:
                    k8s_config.load_kube_config(config_file=kubeconfig_path)
                else:
                    k8s_config.load_incluster_config()
                
                v1 = client.CoreV1Api()
                networking_v1 = client.NetworkingV1Api()
                
                self.active_providers[provider.value] = {
                    'core_v1': v1,
                    'networking_v1': networking_v1
                }
                
            elif provider == NetworkProvider.DOCKER:
                import docker
                
                docker_client = docker.from_env()
                self.active_providers[provider.value] = docker_client
                
            elif provider == NetworkProvider.LOCAL:
                # Local networking - no external client needed
                self.active_providers[provider.value] = {
                    'interface': config.get('interface', 'eth0'),
                    'monitoring_enabled': config.get('monitoring_enabled', True)
                }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize provider connection: {e}")
            raise
    
    async def create_vpc(self, provider: NetworkProvider, vpc_name: str, cidr_block: str,
                        tags: Optional[Dict[str, str]] = None) -> str:
        """Create VPC/Virtual Network."""
        try:
            if provider == NetworkProvider.AWS_VPC:
                return await self._create_aws_vpc(vpc_name, cidr_block, tags or {})
            elif provider == NetworkProvider.GOOGLE_VPC:
                return await self._create_gcp_vpc(vpc_name, cidr_block, tags or {})
            elif provider == NetworkProvider.AZURE_VNET:
                return await self._create_azure_vnet(vpc_name, cidr_block, tags or {})
            else:
                raise Exception(f"VPC creation not supported for provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to create VPC {vpc_name}: {e}")
            return ""
    
    async def _create_aws_vpc(self, vpc_name: str, cidr_block: str, tags: Dict[str, str]) -> str:
        """Create AWS VPC."""
        try:
            ec2_client = self.active_providers[NetworkProvider.AWS_VPC.value]['ec2']
            
            # Create VPC
            response = ec2_client.create_vpc(
                CidrBlock=cidr_block,
                InstanceTenancy='default',
                TagSpecifications=[
                    {
                        'ResourceType': 'vpc',
                        'Tags': [
                            {'Key': 'Name', 'Value': vpc_name},
                            *[{'Key': k, 'Value': v} for k, v in tags.items()]
                        ]
                    }
                ]
            )
            
            vpc_id = response['Vpc']['VpcId']
            
            # Enable DNS support and hostnames
            ec2_client.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={'Value': True})
            ec2_client.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={'Value': True})
            
            # Store VPC info
            self.vpcs[vpc_name] = {
                'id': vpc_id,
                'cidr_block': cidr_block,
                'provider': NetworkProvider.AWS_VPC.value,
                'tags': tags,
                'created_at': datetime.now()
            }
            
            self.logger.info(f"AWS VPC {vpc_name} created: {vpc_id}")
            return vpc_id
            
        except Exception as e:
            self.logger.error(f"Failed to create AWS VPC: {e}")
            return ""
    
    async def _create_gcp_vpc(self, vpc_name: str, cidr_block: str, tags: Dict[str, str]) -> str:
        """Create Google Cloud VPC."""
        try:
            # Implementation would use Google Cloud Compute API
            # Placeholder for now
            self.logger.info(f"GCP VPC creation requested: {vpc_name}")
            return f"gcp-vpc-{vpc_name}"
            
        except Exception as e:
            self.logger.error(f"Failed to create GCP VPC: {e}")
            return ""
    
    async def _create_azure_vnet(self, vpc_name: str, cidr_block: str, tags: Dict[str, str]) -> str:
        """Create Azure Virtual Network."""
        try:
            # Implementation would use Azure SDK
            # Placeholder for now
            self.logger.info(f"Azure VNet creation requested: {vpc_name}")
            return f"azure-vnet-{vpc_name}"
            
        except Exception as e:
            self.logger.error(f"Failed to create Azure VNet: {e}")
            return ""
    
    async def create_subnet(self, vpc_name: str, subnet_name: str, cidr_block: str,
                           availability_zone: Optional[str] = None, is_public: bool = False) -> str:
        """Create subnet in VPC."""
        try:
            if vpc_name not in self.vpcs:
                raise Exception(f"VPC {vpc_name} not found")
            
            vpc_info = self.vpcs[vpc_name]
            provider = NetworkProvider(vpc_info['provider'])
            
            if provider == NetworkProvider.AWS_VPC:
                return await self._create_aws_subnet(vpc_info, subnet_name, cidr_block, availability_zone, is_public)
            else:
                raise Exception(f"Subnet creation not implemented for provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to create subnet {subnet_name}: {e}")
            return ""
    
    async def _create_aws_subnet(self, vpc_info: Dict[str, Any], subnet_name: str,
                                cidr_block: str, availability_zone: Optional[str], is_public: bool) -> str:
        """Create AWS subnet."""
        try:
            ec2_client = self.active_providers[NetworkProvider.AWS_VPC.value]['ec2']
            
            create_args = {
                'VpcId': vpc_info['id'],
                'CidrBlock': cidr_block,
                'TagSpecifications': [
                    {
                        'ResourceType': 'subnet',
                        'Tags': [
                            {'Key': 'Name', 'Value': subnet_name},
                            {'Key': 'Type', 'Value': 'public' if is_public else 'private'}
                        ]
                    }
                ]
            }
            
            if availability_zone:
                create_args['AvailabilityZone'] = availability_zone
            
            response = ec2_client.create_subnet(**create_args)
            subnet_id = response['Subnet']['SubnetId']
            
            # Enable auto-assign public IPs for public subnets
            if is_public:
                ec2_client.modify_subnet_attribute(
                    SubnetId=subnet_id,
                    MapPublicIpOnLaunch={'Value': True}
                )
            
            # Store subnet info
            self.subnets[subnet_name] = {
                'id': subnet_id,
                'vpc_id': vpc_info['id'],
                'cidr_block': cidr_block,
                'availability_zone': availability_zone,
                'is_public': is_public,
                'provider': NetworkProvider.AWS_VPC.value,
                'created_at': datetime.now()
            }
            
            self.logger.info(f"AWS subnet {subnet_name} created: {subnet_id}")
            return subnet_id
            
        except Exception as e:
            self.logger.error(f"Failed to create AWS subnet: {e}")
            return ""
    
    async def create_security_group(self, vpc_name: str, sg_name: str, description: str,
                                   rules: List[SecurityRule]) -> str:
        """Create security group."""
        try:
            if vpc_name not in self.vpcs:
                raise Exception(f"VPC {vpc_name} not found")
            
            vpc_info = self.vpcs[vpc_name]
            provider = NetworkProvider(vpc_info['provider'])
            
            if provider == NetworkProvider.AWS_VPC:
                return await self._create_aws_security_group(vpc_info, sg_name, description, rules)
            else:
                raise Exception(f"Security group creation not implemented for provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to create security group {sg_name}: {e}")
            return ""
    
    async def _create_aws_security_group(self, vpc_info: Dict[str, Any], sg_name: str,
                                        description: str, rules: List[SecurityRule]) -> str:
        """Create AWS security group."""
        try:
            ec2_client = self.active_providers[NetworkProvider.AWS_VPC.value]['ec2']
            
            # Create security group
            response = ec2_client.create_security_group(
                GroupName=sg_name,
                Description=description,
                VpcId=vpc_info['id'],
                TagSpecifications=[
                    {
                        'ResourceType': 'security-group',
                        'Tags': [
                            {'Key': 'Name', 'Value': sg_name}
                        ]
                    }
                ]
            )
            
            sg_id = response['GroupId']
            
            # Add rules
            for rule in rules:
                await self._add_aws_security_group_rule(sg_id, rule)
            
            # Store security group info
            self.security_groups[sg_name] = {
                'id': sg_id,
                'vpc_id': vpc_info['id'],
                'description': description,
                'rules': [rule.__dict__ for rule in rules],
                'provider': NetworkProvider.AWS_VPC.value,
                'created_at': datetime.now()
            }
            
            self.logger.info(f"AWS security group {sg_name} created: {sg_id}")
            return sg_id
            
        except Exception as e:
            self.logger.error(f"Failed to create AWS security group: {e}")
            return ""
    
    async def _add_aws_security_group_rule(self, sg_id: str, rule: SecurityRule):
        """Add rule to AWS security group."""
        try:
            ec2_client = self.active_providers[NetworkProvider.AWS_VPC.value]['ec2']
            
            # Prepare rule parameters
            ip_permissions = []
            
            if isinstance(rule.port_range, int):
                from_port = to_port = rule.port_range
            else:
                from_port, to_port = rule.port_range
            
            ip_permission = {
                'IpProtocol': rule.protocol.value,
                'FromPort': from_port,
                'ToPort': to_port,
                'IpRanges': [{'CidrIp': rule.source_cidr}]
            }
            
            ip_permissions.append(ip_permission)
            
            # Add rule based on direction
            if rule.direction in [TrafficDirection.INBOUND, TrafficDirection.BIDIRECTIONAL]:
                if rule.action == SecurityRuleAction.ALLOW:
                    ec2_client.authorize_security_group_ingress(
                        GroupId=sg_id,
                        IpPermissions=ip_permissions
                    )
            
            if rule.direction in [TrafficDirection.OUTBOUND, TrafficDirection.BIDIRECTIONAL]:
                if rule.action == SecurityRuleAction.ALLOW:
                    ec2_client.authorize_security_group_egress(
                        GroupId=sg_id,
                        IpPermissions=ip_permissions
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to add security group rule: {e}")
    
    async def create_load_balancer(self, lb_config: LoadBalancerConfig) -> str:
        """Create load balancer."""
        try:
            # For now, focus on AWS implementation
            if any(NetworkProvider.AWS_VPC.value in sg for sg in lb_config.security_groups):
                return await self._create_aws_load_balancer(lb_config)
            else:
                raise Exception("Load balancer creation requires AWS VPC setup")
                
        except Exception as e:
            self.logger.error(f"Failed to create load balancer {lb_config.name}: {e}")
            return ""
    
    async def _create_aws_load_balancer(self, lb_config: LoadBalancerConfig) -> str:
        """Create AWS Application Load Balancer."""
        try:
            elb_client = self.active_providers[NetworkProvider.AWS_VPC.value]['elb']
            
            # Map load balancer type
            lb_type_mapping = {
                LoadBalancerType.APPLICATION: 'application',
                LoadBalancerType.NETWORK: 'network',
                LoadBalancerType.CLASSIC: 'classic'
            }
            
            create_args = {
                'Name': lb_config.name,
                'Subnets': lb_config.subnets,
                'SecurityGroups': lb_config.security_groups,
                'Scheme': lb_config.scheme,
                'Type': lb_type_mapping.get(lb_config.type, 'application'),
                'IpAddressType': 'ipv4',
                'Tags': [
                    {'Key': 'Name', 'Value': lb_config.name}
                ]
            }
            
            response = elb_client.create_load_balancer(**create_args)
            lb_arn = response['LoadBalancers'][0]['LoadBalancerArn']
            
            # Store load balancer info
            self.load_balancers[lb_config.name] = {
                'arn': lb_arn,
                'type': lb_config.type.value,
                'scheme': lb_config.scheme,
                'subnets': lb_config.subnets,
                'security_groups': lb_config.security_groups,
                'created_at': datetime.now()
            }
            
            self.logger.info(f"AWS load balancer {lb_config.name} created: {lb_arn}")
            return lb_arn
            
        except Exception as e:
            self.logger.error(f"Failed to create AWS load balancer: {e}")
            return ""
    
    async def create_target_group(self, tg_config: TargetGroup) -> str:
        """Create target group for load balancer."""
        try:
            elb_client = self.active_providers[NetworkProvider.AWS_VPC.value]['elb']
            
            response = elb_client.create_target_group(
                Name=tg_config.name,
                Protocol=tg_config.protocol.upper(),
                Port=tg_config.port,
                VpcId=tg_config.vpc_id,
                HealthCheckEnabled=tg_config.health_check_enabled,
                HealthCheckIntervalSeconds=tg_config.health_check_interval,
                HealthCheckTimeoutSeconds=tg_config.health_check_timeout,
                HealthyThresholdCount=tg_config.healthy_threshold_count,
                UnhealthyThresholdCount=tg_config.unhealthy_threshold_count,
                Tags=[
                    {'Key': 'Name', 'Value': tg_config.name}
                ]
            )
            
            tg_arn = response['TargetGroups'][0]['TargetGroupArn']
            
            # Register targets if provided
            if tg_config.targets:
                await self._register_targets(tg_arn, tg_config.targets)
            
            # Store target group info
            self.target_groups[tg_config.name] = {
                'arn': tg_arn,
                'protocol': tg_config.protocol,
                'port': tg_config.port,
                'vpc_id': tg_config.vpc_id,
                'targets': tg_config.targets,
                'created_at': datetime.now()
            }
            
            self.logger.info(f"Target group {tg_config.name} created: {tg_arn}")
            return tg_arn
            
        except Exception as e:
            self.logger.error(f"Failed to create target group: {e}")
            return ""
    
    async def _register_targets(self, tg_arn: str, targets: List[Dict[str, Any]]):
        """Register targets with target group."""
        try:
            elb_client = self.active_providers[NetworkProvider.AWS_VPC.value]['elb']
            
            elb_client.register_targets(
                TargetGroupArn=tg_arn,
                Targets=targets
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register targets: {e}")
    
    async def add_security_rule(self, rule_name: str, rule: SecurityRule):
        """Add security rule."""
        try:
            self.security_rules[rule_name] = rule
            
            # Apply rule to relevant security groups
            for sg_name, sg_info in self.security_groups.items():
                if sg_info['provider'] == NetworkProvider.AWS_VPC.value:
                    await self._add_aws_security_group_rule(sg_info['id'], rule)
            
            self.logger.info(f"Security rule {rule_name} added")
            
        except Exception as e:
            self.logger.error(f"Failed to add security rule {rule_name}: {e}")
    
    async def register_endpoint(self, endpoint: NetworkEndpoint):
        """Register network endpoint."""
        try:
            self.endpoints[endpoint.name] = endpoint
            
            # Initialize health check
            if endpoint.health_check_enabled:
                await self._start_endpoint_health_check(endpoint)
            
            self.logger.info(f"Endpoint {endpoint.name} registered: {endpoint.host}:{endpoint.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to register endpoint {endpoint.name}: {e}")
    
    async def _start_endpoint_health_check(self, endpoint: NetworkEndpoint):
        """Start health check for endpoint."""
        try:
            # Perform initial health check
            healthy = await self._check_endpoint_health(endpoint)
            
            self.endpoint_health[endpoint.name] = {
                'healthy': healthy,
                'last_check': datetime.now(),
                'consecutive_failures': 0 if healthy else 1,
                'total_checks': 1,
                'success_rate': 1.0 if healthy else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start health check for {endpoint.name}: {e}")
    
    async def _check_endpoint_health(self, endpoint: NetworkEndpoint) -> bool:
        """Check endpoint health."""
        try:
            if endpoint.protocol.lower() in ['http', 'https']:
                return await self._check_http_endpoint(endpoint)
            else:
                return await self._check_tcp_endpoint(endpoint)
                
        except Exception as e:
            self.logger.error(f"Health check failed for {endpoint.name}: {e}")
            return False
    
    async def _check_http_endpoint(self, endpoint: NetworkEndpoint) -> bool:
        """Check HTTP/HTTPS endpoint health."""
        try:
            url = f"{endpoint.protocol}://{endpoint.host}:{endpoint.port}"
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    return response.status < 400
                    
        except Exception:
            return False
    
    async def _check_tcp_endpoint(self, endpoint: NetworkEndpoint) -> bool:
        """Check TCP endpoint health."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            result = sock.connect_ex((endpoint.host, endpoint.port))
            sock.close()
            
            return result == 0
            
        except Exception:
            return False
    
    async def start_traffic_monitoring(self):
        """Start network traffic monitoring."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            
            # Start monitoring task
            asyncio.create_task(self._monitor_traffic())
            
            self.logger.info("Traffic monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start traffic monitoring: {e}")
    
    async def _monitor_traffic(self):
        """Monitor network traffic continuously."""
        try:
            while self.monitoring_active:
                # Get network interface statistics
                net_io = psutil.net_io_counters()
                
                stats = TrafficStats(
                    bytes_in=net_io.bytes_recv,
                    bytes_out=net_io.bytes_sent,
                    packets_in=net_io.packets_recv,
                    packets_out=net_io.packets_sent,
                    connections_active=len(psutil.net_connections()),
                    connections_total=len(psutil.net_connections(kind='inet')),
                    errors=net_io.errin + net_io.errout,
                    timestamp=datetime.now()
                )
                
                # Store stats
                interface_name = "default"
                if interface_name not in self.traffic_stats:
                    self.traffic_stats[interface_name] = []
                
                self.traffic_stats[interface_name].append(stats)
                
                # Trim old stats (keep last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.traffic_stats[interface_name] = [
                    s for s in self.traffic_stats[interface_name] 
                    if s.timestamp > cutoff_time
                ]
                
                # Check alert thresholds
                await self._check_traffic_alerts(stats)
                
                await asyncio.sleep(self.monitoring_config.metrics_interval)
                
        except Exception as e:
            self.logger.error(f"Traffic monitoring error: {e}")
        finally:
            self.monitoring_active = False
    
    async def _check_traffic_alerts(self, stats: TrafficStats):
        """Check traffic against alert thresholds."""
        try:
            thresholds = self.monitoring_config.alert_thresholds
            
            # Check bytes per second threshold
            if 'bytes_per_second' in thresholds:
                total_bytes = stats.bytes_in + stats.bytes_out
                if total_bytes > thresholds['bytes_per_second']:
                    self.logger.warning(f"High traffic detected: {total_bytes} bytes/sec")
            
            # Check error rate threshold
            if 'error_rate' in thresholds:
                total_packets = stats.packets_in + stats.packets_out
                if total_packets > 0:
                    error_rate = stats.errors / total_packets
                    if error_rate > thresholds['error_rate']:
                        self.logger.warning(f"High error rate detected: {error_rate:.2%}")
            
            # Check connection count threshold
            if 'max_connections' in thresholds:
                if stats.connections_active > thresholds['max_connections']:
                    self.logger.warning(f"High connection count: {stats.connections_active}")
                    
        except Exception as e:
            self.logger.error(f"Failed to check traffic alerts: {e}")
    
    async def create_network_policy(self, policy_name: str, rules: List[SecurityRule]):
        """Create network policy."""
        try:
            policy = {
                'name': policy_name,
                'rules': [rule.__dict__ for rule in rules],
                'created_at': datetime.now(),
                'enabled': True
            }
            
            self.network_policies[policy_name] = policy
            
            # Apply policy rules
            for rule in rules:
                await self.add_security_rule(f"{policy_name}_{rule.name}", rule)
            
            self.logger.info(f"Network policy {policy_name} created with {len(rules)} rules")
            
        except Exception as e:
            self.logger.error(f"Failed to create network policy {policy_name}: {e}")
    
    def get_vpc_list(self) -> List[Dict[str, Any]]:
        """Get list of VPCs."""
        return [
            {
                'name': name,
                'id': info['id'],
                'cidr_block': info['cidr_block'],
                'provider': info['provider'],
                'created_at': info['created_at'].isoformat()
            }
            for name, info in self.vpcs.items()
        ]
    
    def get_subnet_list(self) -> List[Dict[str, Any]]:
        """Get list of subnets."""
        return [
            {
                'name': name,
                'id': info['id'],
                'vpc_id': info['vpc_id'],
                'cidr_block': info['cidr_block'],
                'availability_zone': info.get('availability_zone'),
                'is_public': info.get('is_public', False),
                'provider': info['provider'],
                'created_at': info['created_at'].isoformat()
            }
            for name, info in self.subnets.items()
        ]
    
    def get_security_groups(self) -> List[Dict[str, Any]]:
        """Get list of security groups."""
        return [
            {
                'name': name,
                'id': info['id'],
                'vpc_id': info['vpc_id'],
                'description': info['description'],
                'rules_count': len(info['rules']),
                'provider': info['provider'],
                'created_at': info['created_at'].isoformat()
            }
            for name, info in self.security_groups.items()
        ]
    
    def get_load_balancers(self) -> List[Dict[str, Any]]:
        """Get list of load balancers."""
        return [
            {
                'name': name,
                'arn': info['arn'],
                'type': info['type'],
                'scheme': info['scheme'],
                'subnets_count': len(info['subnets']),
                'security_groups_count': len(info['security_groups']),
                'created_at': info['created_at'].isoformat()
            }
            for name, info in self.load_balancers.items()
        ]
    
    def get_endpoint_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all endpoints."""
        return self.endpoint_health.copy()
    
    def get_traffic_statistics(self, interface: str = "default", hours: int = 24) -> List[Dict[str, Any]]:
        """Get traffic statistics for interface."""
        try:
            if interface not in self.traffic_stats:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_stats = [
                s for s in self.traffic_stats[interface] 
                if s.timestamp > cutoff_time
            ]
            
            return [
                {
                    'bytes_in': stats.bytes_in,
                    'bytes_out': stats.bytes_out,
                    'packets_in': stats.packets_in,
                    'packets_out': stats.packets_out,
                    'connections_active': stats.connections_active,
                    'connections_total': stats.connections_total,
                    'errors': stats.errors,
                    'timestamp': stats.timestamp.isoformat()
                }
                for stats in recent_stats
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get traffic statistics: {e}")
            return []
    
    def get_network_manager_summary(self) -> Dict[str, Any]:
        """Get network manager summary."""
        try:
            return {
                'active_providers': list(self.active_providers.keys()),
                'vpcs_count': len(self.vpcs),
                'subnets_count': len(self.subnets),
                'security_groups_count': len(self.security_groups),
                'load_balancers_count': len(self.load_balancers),
                'target_groups_count': len(self.target_groups),
                'security_rules_count': len(self.security_rules),
                'network_policies_count': len(self.network_policies),
                'registered_endpoints': len(self.endpoints),
                'healthy_endpoints': len([h for h in self.endpoint_health.values() if h.get('healthy', False)]),
                'monitoring_active': self.monitoring_active,
                'traffic_interfaces': list(self.traffic_stats.keys()),
                'supported_providers': [provider.value for provider in NetworkProvider],
                'supported_protocols': [protocol.value for protocol in ProtocolType]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate network manager summary: {e}")
            return {'error': 'Unable to generate summary'}