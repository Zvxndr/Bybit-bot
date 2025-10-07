"""
Kubernetes Manager for Container Orchestration and Deployment.
Manages Kubernetes clusters, deployments, services, and resources for scalable trading operations.
"""

import asyncio
import yaml
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import subprocess
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class DeploymentType(Enum):
    """Kubernetes deployment types."""
    TRADING_ENGINE = "trading-engine"
    MARKET_DATA = "market-data"
    RISK_MANAGER = "risk-manager"
    ANALYTICS = "analytics"
    ML_ENGINE = "ml-engine"
    HFT_MODULE = "hft-module"
    API_GATEWAY = "api-gateway"
    DATABASE = "database"
    REDIS_CACHE = "redis-cache"
    MONITORING = "monitoring"

class ServiceType(Enum):
    """Kubernetes service types."""
    CLUSTER_IP = "ClusterIP"
    NODE_PORT = "NodePort"
    LOAD_BALANCER = "LoadBalancer"
    EXTERNAL_NAME = "ExternalName"

class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CPU_BASED = "cpu"
    MEMORY_BASED = "memory"
    CUSTOM_METRICS = "custom"
    QUEUE_LENGTH = "queue"
    LATENCY_BASED = "latency"

@dataclass
class KubernetesResource:
    """Kubernetes resource definition."""
    name: str
    namespace: str
    resource_type: str  # deployment, service, configmap, etc.
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PodMetrics:
    """Pod performance metrics."""
    pod_name: str
    namespace: str
    cpu_usage: float
    memory_usage: float
    network_rx: float
    network_tx: float
    disk_usage: float
    restart_count: int
    ready: bool
    status: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ClusterInfo:
    """Kubernetes cluster information."""
    cluster_name: str
    version: str
    node_count: int
    total_cpu: float
    total_memory: float
    available_cpu: float
    available_memory: float
    pod_count: int
    service_count: int
    deployment_count: int
    health_status: str = "unknown"

class KubernetesManager:
    """Kubernetes cluster management and orchestration."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Kubernetes configuration
        self.k8s_config = {
            'cluster_name': 'trading-cluster',
            'namespace': 'trading-system',
            'image_registry': 'your-registry.com',
            'image_tag': 'latest',
            'replicas': {
                'trading_engine': 3,
                'market_data': 2,
                'risk_manager': 2,
                'analytics': 2,
                'ml_engine': 1,
                'hft_module': 5,
                'api_gateway': 3
            },
            'resource_limits': {
                'trading_engine': {'cpu': '2', 'memory': '4Gi'},
                'market_data': {'cpu': '1', 'memory': '2Gi'},
                'risk_manager': {'cpu': '1', 'memory': '2Gi'},
                'analytics': {'cpu': '2', 'memory': '8Gi'},
                'ml_engine': {'cpu': '4', 'memory': '16Gi'},
                'hft_module': {'cpu': '1', 'memory': '1Gi'},
                'api_gateway': {'cpu': '0.5', 'memory': '1Gi'}
            },
            'auto_scaling': {
                'enabled': True,
                'min_replicas': 1,
                'max_replicas': 20,
                'target_cpu_percent': 70,
                'target_memory_percent': 80,
                'scale_up_stabilization': 60,
                'scale_down_stabilization': 300
            },
            'health_checks': {
                'liveness_probe_delay': 30,
                'readiness_probe_delay': 5,
                'probe_period': 10,
                'failure_threshold': 3
            }
        }
        
        # Kubernetes clients
        self.v1 = None
        self.apps_v1 = None
        self.autoscaling_v2 = None
        self.networking_v1 = None
        
        # Resource tracking
        self.deployments: Dict[str, KubernetesResource] = {}
        self.services: Dict[str, KubernetesResource] = {}
        self.config_maps: Dict[str, KubernetesResource] = {}
        self.secrets: Dict[str, KubernetesResource] = {}
        
        # Metrics and monitoring
        self.pod_metrics: Dict[str, PodMetrics] = {}
        self.cluster_info: Optional[ClusterInfo] = None
        
        # Initialize Kubernetes client
        if HAS_KUBERNETES:
            self._initialize_kubernetes_client()
        
        self.logger.info("KubernetesManager initialized")
    
    def _initialize_kubernetes_client(self):
        """Initialize Kubernetes client configuration."""
        try:
            # Try to load in-cluster config first
            try:
                config.load_incluster_config()
                self.logger.info("Loaded in-cluster Kubernetes configuration")
            except:
                # Fall back to local kubeconfig
                config.load_kube_config()
                self.logger.info("Loaded local Kubernetes configuration")
            
            # Initialize API clients
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.autoscaling_v2 = client.AutoscalingV2Api()
            self.networking_v1 = client.NetworkingV1Api()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            self.v1 = None
    
    async def create_namespace(self, namespace: str) -> bool:
        """Create Kubernetes namespace."""
        try:
            if not self.v1:
                return False
            
            # Check if namespace exists
            try:
                self.v1.read_namespace(name=namespace)
                self.logger.info(f"Namespace {namespace} already exists")
                return True
            except ApiException as e:
                if e.status != 404:
                    raise
            
            # Create namespace
            namespace_spec = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=namespace,
                    labels={
                        'app': 'trading-system',
                        'environment': 'production'
                    }
                )
            )
            
            self.v1.create_namespace(body=namespace_spec)
            self.logger.info(f"Created namespace: {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create namespace {namespace}: {e}")
            return False
    
    async def deploy_trading_engine(self, deployment_config: Dict[str, Any]) -> bool:
        """Deploy trading engine to Kubernetes."""
        try:
            deployment_name = "trading-engine"
            namespace = self.k8s_config['namespace']
            
            # Create deployment specification
            deployment_spec = self._create_deployment_spec(
                name=deployment_name,
                image=f"{self.k8s_config['image_registry']}/trading-engine:{self.k8s_config['image_tag']}",
                replicas=self.k8s_config['replicas']['trading_engine'],
                resource_limits=self.k8s_config['resource_limits']['trading_engine'],
                env_vars=deployment_config.get('env_vars', {}),
                ports=[8000, 8001]  # HTTP and WebSocket ports
            )
            
            # Deploy to Kubernetes
            success = await self._apply_deployment(deployment_spec, namespace)
            
            if success:
                # Create service
                service_spec = self._create_service_spec(
                    name=deployment_name,
                    selector={'app': deployment_name},
                    ports=[
                        {'name': 'http', 'port': 8000, 'target_port': 8000},
                        {'name': 'websocket', 'port': 8001, 'target_port': 8001}
                    ],
                    service_type=ServiceType.CLUSTER_IP
                )
                
                await self._apply_service(service_spec, namespace)
                
                # Create HPA
                await self._create_hpa(deployment_name, namespace)
                
                self.logger.info(f"Successfully deployed {deployment_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deploy trading engine: {e}")
            return False
    
    async def deploy_market_data_service(self) -> bool:
        """Deploy market data service."""
        try:
            deployment_name = "market-data"
            namespace = self.k8s_config['namespace']
            
            deployment_spec = self._create_deployment_spec(
                name=deployment_name,
                image=f"{self.k8s_config['image_registry']}/market-data:{self.k8s_config['image_tag']}",
                replicas=self.k8s_config['replicas']['market_data'],
                resource_limits=self.k8s_config['resource_limits']['market_data'],
                env_vars={
                    'REDIS_URL': 'redis://redis-cache:6379',
                    'KAFKA_BROKERS': 'kafka:9092'
                },
                ports=[8002]
            )
            
            success = await self._apply_deployment(deployment_spec, namespace)
            
            if success:
                service_spec = self._create_service_spec(
                    name=deployment_name,
                    selector={'app': deployment_name},
                    ports=[{'name': 'http', 'port': 8002, 'target_port': 8002}],
                    service_type=ServiceType.CLUSTER_IP
                )
                
                await self._apply_service(service_spec, namespace)
                await self._create_hpa(deployment_name, namespace)
                
                self.logger.info(f"Successfully deployed {deployment_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deploy market data service: {e}")
            return False
    
    async def deploy_hft_module(self) -> bool:
        """Deploy high-frequency trading module."""
        try:
            deployment_name = "hft-module"
            namespace = self.k8s_config['namespace']
            
            # HFT requires special configuration for low latency
            deployment_spec = self._create_deployment_spec(
                name=deployment_name,
                image=f"{self.k8s_config['image_registry']}/hft-module:{self.k8s_config['image_tag']}",
                replicas=self.k8s_config['replicas']['hft_module'],
                resource_limits=self.k8s_config['resource_limits']['hft_module'],
                env_vars={
                    'LATENCY_MODE': 'ultra_low',
                    'CPU_AFFINITY': 'true',
                    'MEMORY_PREALLOC': 'true'
                },
                ports=[8003],
                node_affinity=True,  # Pin to specific nodes
                priority_class='high-priority'
            )
            
            success = await self._apply_deployment(deployment_spec, namespace)
            
            if success:
                service_spec = self._create_service_spec(
                    name=deployment_name,
                    selector={'app': deployment_name},
                    ports=[{'name': 'http', 'port': 8003, 'target_port': 8003}],
                    service_type=ServiceType.CLUSTER_IP
                )
                
                await self._apply_service(service_spec, namespace)
                
                # Custom HPA with latency-based scaling
                await self._create_custom_hpa(deployment_name, namespace, 'latency')
                
                self.logger.info(f"Successfully deployed {deployment_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deploy HFT module: {e}")
            return False
    
    async def deploy_analytics_engine(self) -> bool:
        """Deploy analytics engine."""
        try:
            deployment_name = "analytics"
            namespace = self.k8s_config['namespace']
            
            deployment_spec = self._create_deployment_spec(
                name=deployment_name,
                image=f"{self.k8s_config['image_registry']}/analytics:{self.k8s_config['image_tag']}",
                replicas=self.k8s_config['replicas']['analytics'],
                resource_limits=self.k8s_config['resource_limits']['analytics'],
                env_vars={
                    'PANDAS_COMPUTE_MODE': 'distributed',
                    'DASK_SCHEDULER': 'dask-scheduler:8786'
                },
                ports=[8004]
            )
            
            success = await self._apply_deployment(deployment_spec, namespace)
            
            if success:
                service_spec = self._create_service_spec(
                    name=deployment_name,
                    selector={'app': deployment_name},
                    ports=[{'name': 'http', 'port': 8004, 'target_port': 8004}],
                    service_type=ServiceType.CLUSTER_IP
                )
                
                await self._apply_service(service_spec, namespace)
                await self._create_hpa(deployment_name, namespace)
                
                self.logger.info(f"Successfully deployed {deployment_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deploy analytics engine: {e}")
            return False
    
    async def deploy_ml_engine(self) -> bool:
        """Deploy machine learning engine."""
        try:
            deployment_name = "ml-engine"
            namespace = self.k8s_config['namespace']
            
            deployment_spec = self._create_deployment_spec(
                name=deployment_name,
                image=f"{self.k8s_config['image_registry']}/ml-engine:{self.k8s_config['image_tag']}",
                replicas=self.k8s_config['replicas']['ml_engine'],
                resource_limits=self.k8s_config['resource_limits']['ml_engine'],
                env_vars={
                    'TENSORFLOW_SERVING_MODEL_SERVER_WORKERS': '4',
                    'CUDA_VISIBLE_DEVICES': '0'
                },
                ports=[8005, 8501],  # gRPC and REST
                gpu_required=True
            )
            
            success = await self._apply_deployment(deployment_spec, namespace)
            
            if success:
                service_spec = self._create_service_spec(
                    name=deployment_name,
                    selector={'app': deployment_name},
                    ports=[
                        {'name': 'grpc', 'port': 8005, 'target_port': 8005},
                        {'name': 'rest', 'port': 8501, 'target_port': 8501}
                    ],
                    service_type=ServiceType.CLUSTER_IP
                )
                
                await self._apply_service(service_spec, namespace)
                
                # Custom scaling based on queue length
                await self._create_custom_hpa(deployment_name, namespace, 'queue')
                
                self.logger.info(f"Successfully deployed {deployment_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deploy ML engine: {e}")
            return False
    
    def _create_deployment_spec(self, name: str, image: str, replicas: int, 
                              resource_limits: Dict[str, str], env_vars: Dict[str, str],
                              ports: List[int], node_affinity: bool = False,
                              priority_class: str = None, gpu_required: bool = False) -> Dict[str, Any]:
        """Create Kubernetes deployment specification."""
        try:
            # Container specification
            container_ports = [
                client.V1ContainerPort(container_port=port) for port in ports
            ]
            
            env_list = [
                client.V1EnvVar(name=key, value=str(value))
                for key, value in env_vars.items()
            ]
            
            # Resource requirements
            resources = client.V1ResourceRequirements(
                limits=resource_limits,
                requests={
                    'cpu': str(float(resource_limits['cpu']) * 0.5),
                    'memory': resource_limits['memory']
                }
            )
            
            # Add GPU resources if required
            if gpu_required:
                resources.limits['nvidia.com/gpu'] = '1'
                resources.requests['nvidia.com/gpu'] = '1'
            
            # Liveness and readiness probes
            liveness_probe = client.V1Probe(
                http_get=client.V1HTTPGetAction(path='/health', port=ports[0]),
                initial_delay_seconds=self.k8s_config['health_checks']['liveness_probe_delay'],
                period_seconds=self.k8s_config['health_checks']['probe_period'],
                failure_threshold=self.k8s_config['health_checks']['failure_threshold']
            )
            
            readiness_probe = client.V1Probe(
                http_get=client.V1HTTPGetAction(path='/ready', port=ports[0]),
                initial_delay_seconds=self.k8s_config['health_checks']['readiness_probe_delay'],
                period_seconds=self.k8s_config['health_checks']['probe_period'],
                failure_threshold=self.k8s_config['health_checks']['failure_threshold']
            )
            
            # Container specification
            container = client.V1Container(
                name=name,
                image=image,
                ports=container_ports,
                env=env_list,
                resources=resources,
                liveness_probe=liveness_probe,
                readiness_probe=readiness_probe,
                image_pull_policy='Always'
            )
            
            # Pod template specification
            pod_template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={'app': name, 'version': 'v1'},
                    annotations={
                        'prometheus.io/scrape': 'true',
                        'prometheus.io/port': str(ports[0])
                    }
                ),
                spec=client.V1PodSpec(
                    containers=[container],
                    restart_policy='Always'
                )
            )
            
            # Add node affinity for HFT
            if node_affinity:
                pod_template.spec.affinity = client.V1Affinity(
                    node_affinity=client.V1NodeAffinity(
                        required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                            node_selector_terms=[
                                client.V1NodeSelectorTerm(
                                    match_expressions=[
                                        client.V1NodeSelectorRequirement(
                                            key='node-type',
                                            operator='In',
                                            values=['hft-optimized']
                                        )
                                    ]
                                )
                            ]
                        )
                    )
                )
            
            # Add priority class
            if priority_class:
                pod_template.spec.priority_class_name = priority_class
            
            # Deployment specification
            deployment_spec = client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={'app': name}
                ),
                template=pod_template,
                strategy=client.V1DeploymentStrategy(
                    type='RollingUpdate',
                    rolling_update=client.V1RollingUpdateDeployment(
                        max_surge='25%',
                        max_unavailable='25%'
                    )
                )
            )
            
            deployment = client.V1Deployment(
                api_version='apps/v1',
                kind='Deployment',
                metadata=client.V1ObjectMeta(
                    name=name,
                    labels={'app': name, 'component': 'trading-system'}
                ),
                spec=deployment_spec
            )
            
            return deployment
            
        except Exception as e:
            self.logger.error(f"Failed to create deployment spec for {name}: {e}")
            return {}
    
    def _create_service_spec(self, name: str, selector: Dict[str, str], 
                           ports: List[Dict[str, Any]], service_type: ServiceType) -> Dict[str, Any]:
        """Create Kubernetes service specification."""
        try:
            service_ports = [
                client.V1ServicePort(
                    name=port_config['name'],
                    port=port_config['port'],
                    target_port=port_config['target_port'],
                    protocol='TCP'
                ) for port_config in ports
            ]
            
            service_spec = client.V1ServiceSpec(
                selector=selector,
                ports=service_ports,
                type=service_type.value
            )
            
            service = client.V1Service(
                api_version='v1',
                kind='Service',
                metadata=client.V1ObjectMeta(
                    name=name,
                    labels={'app': name, 'component': 'trading-system'}
                ),
                spec=service_spec
            )
            
            return service
            
        except Exception as e:
            self.logger.error(f"Failed to create service spec for {name}: {e}")
            return {}
    
    async def _apply_deployment(self, deployment_spec: Dict[str, Any], namespace: str) -> bool:
        """Apply deployment to Kubernetes cluster."""
        try:
            if not self.apps_v1 or not deployment_spec:
                return False
            
            deployment_name = deployment_spec.metadata.name
            
            # Check if deployment exists
            try:
                existing_deployment = self.apps_v1.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace
                )
                
                # Update existing deployment
                self.apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment_spec
                )
                self.logger.info(f"Updated deployment: {deployment_name}")
                
            except ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    self.apps_v1.create_namespaced_deployment(
                        namespace=namespace,
                        body=deployment_spec
                    )
                    self.logger.info(f"Created deployment: {deployment_name}")
                else:
                    raise
            
            # Store deployment info
            self.deployments[deployment_name] = KubernetesResource(
                name=deployment_name,
                namespace=namespace,
                resource_type='deployment',
                labels={'app': deployment_name},
                status='deployed'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply deployment: {e}")
            return False
    
    async def _apply_service(self, service_spec: Dict[str, Any], namespace: str) -> bool:
        """Apply service to Kubernetes cluster."""
        try:
            if not self.v1 or not service_spec:
                return False
            
            service_name = service_spec.metadata.name
            
            # Check if service exists
            try:
                existing_service = self.v1.read_namespaced_service(
                    name=service_name, namespace=namespace
                )
                
                # Update existing service
                self.v1.patch_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                    body=service_spec
                )
                self.logger.info(f"Updated service: {service_name}")
                
            except ApiException as e:
                if e.status == 404:
                    # Create new service
                    self.v1.create_namespaced_service(
                        namespace=namespace,
                        body=service_spec
                    )
                    self.logger.info(f"Created service: {service_name}")
                else:
                    raise
            
            # Store service info
            self.services[service_name] = KubernetesResource(
                name=service_name,
                namespace=namespace,
                resource_type='service',
                labels={'app': service_name},
                status='deployed'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply service: {e}")
            return False
    
    async def _create_hpa(self, deployment_name: str, namespace: str) -> bool:
        """Create Horizontal Pod Autoscaler."""
        try:
            if not self.autoscaling_v2:
                return False
            
            auto_scaling_config = self.k8s_config['auto_scaling']
            
            # HPA specification
            hpa_spec = client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version='apps/v1',
                    kind='Deployment',
                    name=deployment_name
                ),
                min_replicas=auto_scaling_config['min_replicas'],
                max_replicas=auto_scaling_config['max_replicas'],
                metrics=[
                    client.V2MetricSpec(
                        type='Resource',
                        resource=client.V2ResourceMetricSource(
                            name='cpu',
                            target=client.V2MetricTarget(
                                type='Utilization',
                                average_utilization=auto_scaling_config['target_cpu_percent']
                            )
                        )
                    ),
                    client.V2MetricSpec(
                        type='Resource',
                        resource=client.V2ResourceMetricSource(
                            name='memory',
                            target=client.V2MetricTarget(
                                type='Utilization',
                                average_utilization=auto_scaling_config['target_memory_percent']
                            )
                        )
                    )
                ],
                behavior=client.V2HorizontalPodAutoscalerBehavior(
                    scale_up=client.V2HPAScalingRules(
                        stabilization_window_seconds=auto_scaling_config['scale_up_stabilization']
                    ),
                    scale_down=client.V2HPAScalingRules(
                        stabilization_window_seconds=auto_scaling_config['scale_down_stabilization']
                    )
                )
            )
            
            hpa = client.V2HorizontalPodAutoscaler(
                api_version='autoscaling/v2',
                kind='HorizontalPodAutoscaler',
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-hpa",
                    namespace=namespace
                ),
                spec=hpa_spec
            )
            
            # Apply HPA
            try:
                self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                    namespace=namespace,
                    body=hpa
                )
                self.logger.info(f"Created HPA for {deployment_name}")
                return True
                
            except ApiException as e:
                if e.status == 409:  # Already exists
                    self.autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                        name=f"{deployment_name}-hpa",
                        namespace=namespace,
                        body=hpa
                    )
                    self.logger.info(f"Updated HPA for {deployment_name}")
                    return True
                else:
                    raise
            
        except Exception as e:
            self.logger.error(f"Failed to create HPA for {deployment_name}: {e}")
            return False
    
    async def _create_custom_hpa(self, deployment_name: str, namespace: str, metric_type: str) -> bool:
        """Create custom HPA with specific metrics."""
        try:
            # For now, create standard HPA
            # In production, would create custom metrics for latency, queue length, etc.
            return await self._create_hpa(deployment_name, namespace)
            
        except Exception as e:
            self.logger.error(f"Failed to create custom HPA for {deployment_name}: {e}")
            return False
    
    async def get_cluster_status(self) -> Optional[ClusterInfo]:
        """Get current cluster status and metrics."""
        try:
            if not self.v1:
                return None
            
            # Get nodes
            nodes = self.v1.list_node()
            node_count = len(nodes.items)
            
            # Calculate total resources
            total_cpu = 0
            total_memory = 0
            available_cpu = 0
            available_memory = 0
            
            for node in nodes.items:
                if node.status.capacity:
                    cpu_capacity = node.status.capacity.get('cpu', '0')
                    memory_capacity = node.status.capacity.get('memory', '0')
                    
                    # Parse CPU (can be in millicores or cores)
                    if 'm' in cpu_capacity:
                        total_cpu += int(cpu_capacity.replace('m', '')) / 1000
                    else:
                        total_cpu += int(cpu_capacity)
                    
                    # Parse memory (remove unit and convert to GB)
                    memory_value = int(memory_capacity.replace('Ki', '')) / (1024 * 1024)
                    total_memory += memory_value
                
                if node.status.allocatable:
                    cpu_allocatable = node.status.allocatable.get('cpu', '0')
                    memory_allocatable = node.status.allocatable.get('memory', '0')
                    
                    if 'm' in cpu_allocatable:
                        available_cpu += int(cpu_allocatable.replace('m', '')) / 1000
                    else:
                        available_cpu += int(cpu_allocatable)
                    
                    memory_value = int(memory_allocatable.replace('Ki', '')) / (1024 * 1024)
                    available_memory += memory_value
            
            # Get pods, services, deployments
            pods = self.v1.list_pod_for_all_namespaces()
            services = self.v1.list_service_for_all_namespaces()
            deployments = self.apps_v1.list_deployment_for_all_namespaces()
            
            # Determine health status
            health_status = "healthy"
            for node in nodes.items:
                for condition in node.status.conditions:
                    if condition.type == "Ready" and condition.status != "True":
                        health_status = "degraded"
                        break
            
            cluster_info = ClusterInfo(
                cluster_name=self.k8s_config['cluster_name'],
                version="1.25.0",  # Would get from cluster
                node_count=node_count,
                total_cpu=total_cpu,
                total_memory=total_memory,
                available_cpu=available_cpu,
                available_memory=available_memory,
                pod_count=len(pods.items),
                service_count=len(services.items),
                deployment_count=len(deployments.items),
                health_status=health_status
            )
            
            self.cluster_info = cluster_info
            return cluster_info
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster status: {e}")
            return None
    
    async def get_pod_metrics(self, namespace: str = None) -> Dict[str, PodMetrics]:
        """Get pod performance metrics."""
        try:
            if not self.v1:
                return {}
            
            target_namespace = namespace or self.k8s_config['namespace']
            
            # Get pods
            pods = self.v1.list_namespaced_pod(namespace=target_namespace)
            
            pod_metrics = {}
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                
                # Get basic status
                ready = False
                if pod.status.conditions:
                    for condition in pod.status.conditions:
                        if condition.type == "Ready":
                            ready = condition.status == "True"
                            break
                
                restart_count = 0
                if pod.status.container_statuses:
                    restart_count = sum(
                        container.restart_count for container in pod.status.container_statuses
                    )
                
                # Create pod metrics (resource usage would come from metrics server)
                metrics = PodMetrics(
                    pod_name=pod_name,
                    namespace=target_namespace,
                    cpu_usage=0.0,  # Would get from metrics server
                    memory_usage=0.0,  # Would get from metrics server
                    network_rx=0.0,
                    network_tx=0.0,
                    disk_usage=0.0,
                    restart_count=restart_count,
                    ready=ready,
                    status=pod.status.phase
                )
                
                pod_metrics[pod_name] = metrics
            
            self.pod_metrics.update(pod_metrics)
            return pod_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get pod metrics: {e}")
            return {}
    
    async def scale_deployment(self, deployment_name: str, replicas: int, namespace: str = None) -> bool:
        """Scale deployment to specified number of replicas."""
        try:
            if not self.apps_v1:
                return False
            
            target_namespace = namespace or self.k8s_config['namespace']
            
            # Update deployment replicas
            patch_body = {'spec': {'replicas': replicas}}
            
            self.apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=target_namespace,
                body=patch_body
            )
            
            self.logger.info(f"Scaled {deployment_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    async def rolling_update(self, deployment_name: str, new_image: str, namespace: str = None) -> bool:
        """Perform rolling update of deployment."""
        try:
            if not self.apps_v1:
                return False
            
            target_namespace = namespace or self.k8s_config['namespace']
            
            # Update deployment image
            patch_body = {
                'spec': {
                    'template': {
                        'spec': {
                            'containers': [
                                {
                                    'name': deployment_name,
                                    'image': new_image
                                }
                            ]
                        }
                    }
                }
            }
            
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=target_namespace,
                body=patch_body
            )
            
            self.logger.info(f"Started rolling update for {deployment_name} with image {new_image}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to perform rolling update for {deployment_name}: {e}")
            return False
    
    def generate_deployment_yaml(self, deployment_type: DeploymentType) -> str:
        """Generate YAML configuration for deployment."""
        try:
            if deployment_type == DeploymentType.TRADING_ENGINE:
                config = {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'metadata': {
                        'name': 'trading-engine',
                        'namespace': self.k8s_config['namespace'],
                        'labels': {'app': 'trading-engine'}
                    },
                    'spec': {
                        'replicas': self.k8s_config['replicas']['trading_engine'],
                        'selector': {'matchLabels': {'app': 'trading-engine'}},
                        'template': {
                            'metadata': {'labels': {'app': 'trading-engine'}},
                            'spec': {
                                'containers': [{
                                    'name': 'trading-engine',
                                    'image': f"{self.k8s_config['image_registry']}/trading-engine:{self.k8s_config['image_tag']}",
                                    'ports': [
                                        {'containerPort': 8000, 'name': 'http'},
                                        {'containerPort': 8001, 'name': 'websocket'}
                                    ],
                                    'resources': {
                                        'limits': self.k8s_config['resource_limits']['trading_engine'],
                                        'requests': {
                                            'cpu': '1',
                                            'memory': '2Gi'
                                        }
                                    },
                                    'livenessProbe': {
                                        'httpGet': {'path': '/health', 'port': 8000},
                                        'initialDelaySeconds': 30,
                                        'periodSeconds': 10
                                    },
                                    'readinessProbe': {
                                        'httpGet': {'path': '/ready', 'port': 8000},
                                        'initialDelaySeconds': 5,
                                        'periodSeconds': 10
                                    }
                                }]
                            }
                        }
                    }
                }
                
                return yaml.dump(config, default_flow_style=False)
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to generate YAML for {deployment_type}: {e}")
            return ""
    
    def get_kubernetes_summary(self) -> Dict[str, Any]:
        """Get Kubernetes deployment summary."""
        try:
            return {
                'cluster_connected': self.v1 is not None,
                'cluster_info': {
                    'name': self.cluster_info.cluster_name if self.cluster_info else 'unknown',
                    'nodes': self.cluster_info.node_count if self.cluster_info else 0,
                    'health': self.cluster_info.health_status if self.cluster_info else 'unknown'
                } if self.cluster_info else {},
                'deployments': len(self.deployments),
                'services': len(self.services),
                'active_pods': len(self.pod_metrics),
                'namespace': self.k8s_config['namespace'],
                'auto_scaling_enabled': self.k8s_config['auto_scaling']['enabled']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate Kubernetes summary: {e}")
            return {'error': 'Unable to generate summary'}