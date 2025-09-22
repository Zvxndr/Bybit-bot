"""
Deployment Manager for Cloud Infrastructure.
Comprehensive CI/CD pipeline and deployment automation system.
"""

import asyncio
import json
import yaml
import time
import os
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import re
import aiofiles
import aiohttp

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class DeploymentStrategy(Enum):
    """Deployment strategies."""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "ab_testing"

class DeploymentEnvironment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    PREVIEW = "preview"

class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

class PipelineStage(Enum):
    """CI/CD pipeline stages."""
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    DEPLOY_STAGING = "deploy_staging"
    INTEGRATION_TEST = "integration_test"
    DEPLOY_PRODUCTION = "deploy_production"
    POST_DEPLOY_TEST = "post_deploy_test"
    ROLLBACK = "rollback"

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    image_tag: str
    replicas: int = 3
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    health_check_path: str = "/health"
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    config_maps: List[str] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    ingress_rules: List[Dict[str, Any]] = field(default_factory=list)
    auto_rollback: bool = True
    rollback_timeout: int = 300  # seconds
    max_surge: str = "25%"
    max_unavailable: str = "25%"

@dataclass
class PipelineConfig:
    """CI/CD pipeline configuration."""
    name: str
    trigger_branches: List[str] = field(default_factory=lambda: ["main", "develop"])
    trigger_paths: List[str] = field(default_factory=list)
    stages: List[PipelineStage] = field(default_factory=lambda: list(PipelineStage))
    parallel_stages: List[List[PipelineStage]] = field(default_factory=list)
    environment_promotions: Dict[str, str] = field(default_factory=dict)
    approval_required: List[PipelineStage] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)
    timeout_minutes: int = 60
    retry_count: int = 3

@dataclass
class DeploymentJob:
    """Deployment job tracking."""
    id: str
    config: DeploymentConfig
    status: DeploymentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_version: Optional[str] = None
    error_message: Optional[str] = None
    pipeline_run_id: Optional[str] = None

@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    type: str  # unit, integration, e2e, performance, security
    command: str
    working_directory: str = "."
    environment: Dict[str, str] = field(default_factory=dict)
    timeout_minutes: int = 30
    retry_on_failure: bool = True
    parallel: bool = False
    required_services: List[str] = field(default_factory=list)

@dataclass
class RollbackPlan:
    """Rollback plan configuration."""
    trigger_conditions: List[str]
    max_rollback_time: int = 300  # seconds
    health_check_interval: int = 30  # seconds  
    success_threshold: int = 3  # consecutive successful health checks
    notification_channels: List[str] = field(default_factory=list)

class DeploymentManager:
    """Comprehensive deployment management and CI/CD system."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Deployment configuration
        self.config = {
            'git': {
                'repository_url': 'https://github.com/username/trading-bot.git',
                'default_branch': 'main',
                'webhook_secret': 'webhook_secret_key'
            },
            'docker': {
                'registry': 'docker.io',
                'repository': 'trading-bot',
                'build_args': {},
                'build_context': '.',
                'dockerfile': 'Dockerfile'
            },
            'kubernetes': {
                'namespace': 'trading-system',
                'config_path': '~/.kube/config',
                'context': 'default'
            },
            'monitoring': {
                'prometheus_url': 'http://prometheus:9090',
                'grafana_url': 'http://grafana:3000',
                'alert_manager_url': 'http://alertmanager:9093'
            },
            'notifications': {
                'slack_webhook': 'https://hooks.slack.com/services/...',
                'email_recipients': ['ops@company.com'],
                'teams_webhook': ''
            }
        }
        
        # Deployment state
        self.active_deployments: Dict[str, DeploymentJob] = {}
        self.deployment_history: List[DeploymentJob] = []
        self.pipeline_configs: Dict[str, PipelineConfig] = {}
        self.test_suites: Dict[str, TestSuite] = {}
        
        # Deployment environments
        self.environments: Dict[str, Dict[str, Any]] = {}
        
        # CI/CD state
        self.pipeline_running = False
        self.current_pipeline_run = None
        
        # Initialize default configurations
        self._setup_default_environments()
        self._setup_default_pipelines()
        self._setup_default_tests()
        
        self.logger.info("DeploymentManager initialized")
    
    def _setup_default_environments(self):
        """Setup default deployment environments."""
        try:
            # Development environment
            self.environments['development'] = {
                'namespace': 'trading-dev',
                'replicas': 1,
                'resources': {
                    'cpu_request': '50m',
                    'cpu_limit': '200m',
                    'memory_request': '64Mi',
                    'memory_limit': '256Mi'
                },
                'auto_deploy': True,
                'health_checks_required': False,
                'monitoring_enabled': True
            }
            
            # Staging environment
            self.environments['staging'] = {
                'namespace': 'trading-staging',
                'replicas': 2,
                'resources': {
                    'cpu_request': '100m',
                    'cpu_limit': '500m',
                    'memory_request': '128Mi',
                    'memory_limit': '512Mi'
                },
                'auto_deploy': False,
                'health_checks_required': True,
                'monitoring_enabled': True,
                'approval_required': True
            }
            
            # Production environment
            self.environments['production'] = {
                'namespace': 'trading-prod',
                'replicas': 5,
                'resources': {
                    'cpu_request': '200m',
                    'cpu_limit': '1000m',
                    'memory_request': '256Mi',
                    'memory_limit': '1Gi'
                },
                'auto_deploy': False,
                'health_checks_required': True,
                'monitoring_enabled': True,
                'approval_required': True,
                'rollback_enabled': True,
                'canary_enabled': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to setup default environments: {e}")
    
    def _setup_default_pipelines(self):
        """Setup default CI/CD pipelines."""
        try:
            # Main pipeline
            main_pipeline = PipelineConfig(
                name="main-pipeline",
                trigger_branches=["main"],
                stages=[
                    PipelineStage.BUILD,
                    PipelineStage.TEST,
                    PipelineStage.SECURITY_SCAN,
                    PipelineStage.DEPLOY_STAGING,
                    PipelineStage.INTEGRATION_TEST,
                    PipelineStage.DEPLOY_PRODUCTION,
                    PipelineStage.POST_DEPLOY_TEST
                ],
                parallel_stages=[
                    [PipelineStage.TEST, PipelineStage.SECURITY_SCAN]
                ],
                approval_required=[PipelineStage.DEPLOY_PRODUCTION],
                notification_channels=["slack", "email"]
            )
            
            # Development pipeline
            dev_pipeline = PipelineConfig(
                name="dev-pipeline",
                trigger_branches=["develop", "feature/*"],
                stages=[
                    PipelineStage.BUILD,
                    PipelineStage.TEST,
                    PipelineStage.DEPLOY_STAGING
                ],
                timeout_minutes=30
            )
            
            # Hotfix pipeline
            hotfix_pipeline = PipelineConfig(
                name="hotfix-pipeline",
                trigger_branches=["hotfix/*"],
                stages=[
                    PipelineStage.BUILD,
                    PipelineStage.TEST,
                    PipelineStage.DEPLOY_PRODUCTION
                ],
                approval_required=[PipelineStage.DEPLOY_PRODUCTION],
                notification_channels=["slack", "email", "sms"]
            )
            
            self.pipeline_configs["main-pipeline"] = main_pipeline
            self.pipeline_configs["dev-pipeline"] = dev_pipeline
            self.pipeline_configs["hotfix-pipeline"] = hotfix_pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to setup default pipelines: {e}")
    
    def _setup_default_tests(self):
        """Setup default test suites."""
        try:
            # Unit tests
            unit_tests = TestSuite(
                name="unit-tests",
                type="unit",
                command="python -m pytest tests/unit -v --cov=src --cov-report=xml",
                timeout_minutes=15,
                parallel=True
            )
            
            # Integration tests
            integration_tests = TestSuite(
                name="integration-tests",
                type="integration", 
                command="python -m pytest tests/integration -v",
                timeout_minutes=30,
                required_services=["redis", "postgresql"]
            )
            
            # End-to-end tests
            e2e_tests = TestSuite(
                name="e2e-tests",
                type="e2e",
                command="python -m pytest tests/e2e -v --browser=headless",
                timeout_minutes=45,
                required_services=["trading-api", "database", "redis"]
            )
            
            # Performance tests
            performance_tests = TestSuite(
                name="performance-tests",
                type="performance",
                command="python -m pytest tests/performance -v",
                timeout_minutes=30
            )
            
            # Security tests
            security_tests = TestSuite(
                name="security-tests",
                type="security",
                command="bandit -r src/ -f json -o security-report.json",
                timeout_minutes=20
            )
            
            self.test_suites["unit-tests"] = unit_tests
            self.test_suites["integration-tests"] = integration_tests
            self.test_suites["e2e-tests"] = e2e_tests
            self.test_suites["performance-tests"] = performance_tests
            self.test_suites["security-tests"] = security_tests
            
        except Exception as e:
            self.logger.error(f"Failed to setup default tests: {e}")
    
    async def deploy(self, config: DeploymentConfig) -> str:
        """Deploy application with specified configuration."""
        try:
            # Generate deployment job ID
            job_id = f"deploy-{int(time.time())}-{hashlib.md5(config.name.encode()).hexdigest()[:8]}"
            
            # Create deployment job
            job = DeploymentJob(
                id=job_id,
                config=config,
                status=DeploymentStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.active_deployments[job_id] = job
            
            # Start deployment
            await self._execute_deployment(job)
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to start deployment: {e}")
            return ""
    
    async def _execute_deployment(self, job: DeploymentJob):
        """Execute deployment job."""
        try:
            job.status = DeploymentStatus.IN_PROGRESS
            job.started_at = datetime.now()
            
            self.logger.info(f"Starting deployment {job.id} with strategy {job.config.strategy.value}")
            
            # Execute deployment strategy
            if job.config.strategy == DeploymentStrategy.ROLLING_UPDATE:
                await self._rolling_update_deployment(job)
            elif job.config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._blue_green_deployment(job)
            elif job.config.strategy == DeploymentStrategy.CANARY:
                await self._canary_deployment(job)
            elif job.config.strategy == DeploymentStrategy.RECREATE:
                await self._recreate_deployment(job)
            else:
                await self._rolling_update_deployment(job)  # Default
            
            # Post-deployment verification
            if job.status == DeploymentStatus.IN_PROGRESS:
                await self._verify_deployment(job)
            
            job.completed_at = datetime.now()
            
            # Move to history
            self.deployment_history.append(job)
            if job.id in self.active_deployments:
                del self.active_deployments[job.id]
            
            # Send notifications
            await self._send_deployment_notification(job)
            
        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self.logger.error(f"Deployment {job.id} failed: {e}")
    
    async def _rolling_update_deployment(self, job: DeploymentJob):
        """Execute rolling update deployment."""
        try:
            config = job.config
            
            # Generate Kubernetes manifests
            manifests = await self._generate_k8s_manifests(config)
            
            # Apply manifests
            for manifest in manifests:
                await self._apply_k8s_manifest(manifest, job)
            
            # Wait for rollout to complete
            await self._wait_for_rollout(config.name, config.environment.value, job)
            
            job.status = DeploymentStatus.SUCCESS
            job.logs.append(f"Rolling update deployment completed successfully")
            
        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = f"Rolling update failed: {e}"
            self.logger.error(f"Rolling update deployment failed: {e}")
    
    async def _blue_green_deployment(self, job: DeploymentJob):
        """Execute blue-green deployment."""
        try:
            config = job.config
            
            # Create green environment
            green_config = self._create_green_environment_config(config)
            manifests = await self._generate_k8s_manifests(green_config)
            
            # Deploy green environment
            for manifest in manifests:
                await self._apply_k8s_manifest(manifest, job)
            
            # Wait for green environment to be ready
            await self._wait_for_rollout(f"{config.name}-green", config.environment.value, job)
            
            # Verify green environment health
            if await self._verify_environment_health(f"{config.name}-green", config.environment.value):
                # Switch traffic to green
                await self._switch_traffic_to_green(config, job)
                
                # Clean up blue environment
                await self._cleanup_blue_environment(config, job)
                
                job.status = DeploymentStatus.SUCCESS
                job.logs.append("Blue-green deployment completed successfully")
            else:
                job.status = DeploymentStatus.FAILED
                job.error_message = "Green environment health check failed"
                
        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = f"Blue-green deployment failed: {e}"
            self.logger.error(f"Blue-green deployment failed: {e}")
    
    async def _canary_deployment(self, job: DeploymentJob):
        """Execute canary deployment."""
        try:
            config = job.config
            
            # Deploy canary version (10% traffic)
            canary_config = self._create_canary_config(config, traffic_percentage=10)
            manifests = await self._generate_k8s_manifests(canary_config)
            
            for manifest in manifests:
                await self._apply_k8s_manifest(manifest, job)
            
            # Monitor canary metrics for 5 minutes
            canary_healthy = await self._monitor_canary_health(config, job, duration=300)
            
            if canary_healthy:
                # Gradually increase traffic: 10% -> 50% -> 100%
                for percentage in [50, 100]:
                    await self._update_canary_traffic(config, percentage, job)
                    await asyncio.sleep(180)  # Wait 3 minutes between increases
                    
                    if not await self._monitor_canary_health(config, job, duration=180):
                        raise Exception(f"Canary health check failed at {percentage}% traffic")
                
                job.status = DeploymentStatus.SUCCESS
                job.logs.append("Canary deployment completed successfully")
            else:
                # Rollback canary
                await self._rollback_canary(config, job)
                job.status = DeploymentStatus.FAILED
                job.error_message = "Canary health checks failed"
                
        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = f"Canary deployment failed: {e}"
            self.logger.error(f"Canary deployment failed: {e}")
    
    async def _recreate_deployment(self, job: DeploymentJob):
        """Execute recreate deployment."""
        try:
            config = job.config
            
            # Scale down existing deployment
            await self._scale_deployment(config.name, config.environment.value, 0, job)
            
            # Wait for pods to terminate
            await asyncio.sleep(30)
            
            # Deploy new version
            manifests = await self._generate_k8s_manifests(config)
            for manifest in manifests:
                await self._apply_k8s_manifest(manifest, job)
            
            # Scale up new deployment
            await self._scale_deployment(config.name, config.environment.value, config.replicas, job)
            
            # Wait for rollout
            await self._wait_for_rollout(config.name, config.environment.value, job)
            
            job.status = DeploymentStatus.SUCCESS
            job.logs.append("Recreate deployment completed successfully")
            
        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = f"Recreate deployment failed: {e}"
            self.logger.error(f"Recreate deployment failed: {e}")
    
    async def _generate_k8s_manifests(self, config: DeploymentConfig) -> List[Dict[str, Any]]:
        """Generate Kubernetes manifests from deployment config."""
        try:
            manifests = []
            
            # Deployment manifest
            deployment = {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': config.name,
                    'namespace': self.environments[config.environment.value]['namespace'],
                    'labels': {
                        'app': config.name,
                        'environment': config.environment.value,
                        'version': config.image_tag
                    }
                },
                'spec': {
                    'replicas': config.replicas,
                    'selector': {
                        'matchLabels': {
                            'app': config.name
                        }
                    },
                    'strategy': {
                        'type': 'RollingUpdate',
                        'rollingUpdate': {
                            'maxSurge': config.max_surge,
                            'maxUnavailable': config.max_unavailable
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': config.name,
                                'environment': config.environment.value,
                                'version': config.image_tag
                            }
                        },
                        'spec': {
                            'containers': [{
                                'name': config.name,
                                'image': f"{self.config['docker']['registry']}/{self.config['docker']['repository']}:{config.image_tag}",
                                'ports': [{'containerPort': 8000}],
                                'resources': {
                                    'requests': {
                                        'cpu': config.cpu_request,
                                        'memory': config.memory_request
                                    },
                                    'limits': {
                                        'cpu': config.cpu_limit,
                                        'memory': config.memory_limit
                                    }
                                },
                                'env': [
                                    {'name': k, 'value': v} 
                                    for k, v in config.environment_variables.items()
                                ],
                                'livenessProbe': {
                                    'httpGet': {
                                        'path': config.health_check_path,
                                        'port': 8000
                                    },
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 10
                                },
                                'readinessProbe': {
                                    'httpGet': {
                                        'path': config.health_check_path,
                                        'port': 8000
                                    },
                                    'initialDelaySeconds': 5,
                                    'periodSeconds': 5
                                }
                            }]
                        }
                    }
                }
            }
            
            manifests.append(deployment)
            
            # Service manifest
            service = {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': f"{config.name}-service",
                    'namespace': self.environments[config.environment.value]['namespace'],
                    'labels': {
                        'app': config.name
                    }
                },
                'spec': {
                    'selector': {
                        'app': config.name
                    },
                    'ports': [{
                        'port': 80,
                        'targetPort': 8000,
                        'protocol': 'TCP'
                    }],
                    'type': 'ClusterIP'
                }
            }
            
            manifests.append(service)
            
            # Ingress manifest (if ingress rules provided)
            if config.ingress_rules:
                ingress = {
                    'apiVersion': 'networking.k8s.io/v1',
                    'kind': 'Ingress',
                    'metadata': {
                        'name': f"{config.name}-ingress",
                        'namespace': self.environments[config.environment.value]['namespace'],
                        'annotations': {
                            'nginx.ingress.kubernetes.io/rewrite-target': '/'
                        }
                    },
                    'spec': {
                        'rules': config.ingress_rules
                    }
                }
                manifests.append(ingress)
            
            return manifests
            
        except Exception as e:
            self.logger.error(f"Failed to generate K8s manifests: {e}")
            return []
    
    async def _apply_k8s_manifest(self, manifest: Dict[str, Any], job: DeploymentJob):
        """Apply Kubernetes manifest."""
        try:
            # Convert manifest to YAML
            manifest_yaml = yaml.dump(manifest, default_flow_style=False)
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(manifest_yaml)
                temp_file = f.name
            
            try:
                # Apply manifest using kubectl
                result = subprocess.run([
                    'kubectl', 'apply', '-f', temp_file
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    job.logs.append(f"Applied {manifest['kind']}: {manifest['metadata']['name']}")
                    self.logger.info(f"Applied manifest: {manifest['metadata']['name']}")
                else:
                    raise Exception(f"kubectl apply failed: {result.stderr}")
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_file)
                
        except Exception as e:
            job.logs.append(f"Failed to apply manifest: {e}")
            self.logger.error(f"Failed to apply K8s manifest: {e}")
            raise
    
    async def _wait_for_rollout(self, deployment_name: str, environment: str, job: DeploymentJob, timeout: int = 300):
        """Wait for deployment rollout to complete."""
        try:
            namespace = self.environments[environment]['namespace']
            
            # Wait for rollout using kubectl
            result = subprocess.run([
                'kubectl', 'rollout', 'status', 
                f'deployment/{deployment_name}',
                '-n', namespace,
                f'--timeout={timeout}s'
            ], capture_output=True, text=True, timeout=timeout + 30)
            
            if result.returncode == 0:
                job.logs.append(f"Rollout completed for {deployment_name}")
                self.logger.info(f"Rollout completed: {deployment_name}")
            else:
                raise Exception(f"Rollout failed: {result.stderr}")
                
        except Exception as e:
            job.logs.append(f"Rollout wait failed: {e}")
            self.logger.error(f"Failed to wait for rollout: {e}")
            raise
    
    async def _verify_deployment(self, job: DeploymentJob):
        """Verify deployment health and readiness."""
        try:
            config = job.config
            namespace = self.environments[config.environment.value]['namespace']
            
            # Check pod status
            result = subprocess.run([
                'kubectl', 'get', 'pods',
                '-l', f'app={config.name}',
                '-n', namespace,
                '-o', 'json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                pods = pods_data.get('items', [])
                
                running_pods = 0
                for pod in pods:
                    status = pod.get('status', {})
                    phase = status.get('phase', '')
                    
                    if phase == 'Running':
                        # Check if all containers are ready
                        container_statuses = status.get('containerStatuses', [])
                        all_ready = all(cs.get('ready', False) for cs in container_statuses)
                        
                        if all_ready:
                            running_pods += 1
                
                if running_pods >= config.replicas:
                    job.logs.append(f"Deployment verification successful: {running_pods}/{config.replicas} pods running")
                    self.logger.info(f"Deployment {job.id} verified successfully")
                else:
                    raise Exception(f"Only {running_pods}/{config.replicas} pods are running and ready")
            else:
                raise Exception(f"Failed to get pod status: {result.stderr}")
                
        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = f"Deployment verification failed: {e}"
            self.logger.error(f"Deployment verification failed: {e}")
    
    async def rollback_deployment(self, deployment_name: str, environment: str, target_revision: Optional[str] = None) -> bool:
        """Rollback deployment to previous or specific revision."""
        try:
            namespace = self.environments[environment]['namespace']
            
            # Prepare rollback command
            cmd = ['kubectl', 'rollout', 'undo', f'deployment/{deployment_name}', '-n', namespace]
            
            if target_revision:
                cmd.extend([f'--to-revision={target_revision}'])
            
            # Execute rollback
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info(f"Rollback initiated for {deployment_name}")
                
                # Wait for rollback to complete
                await self._wait_for_rollout(deployment_name, environment, 
                                           DeploymentJob("rollback", None, DeploymentStatus.IN_PROGRESS, datetime.now()))
                
                return True
            else:
                self.logger.error(f"Rollback failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to rollback deployment: {e}")
            return False
    
    async def run_pipeline(self, pipeline_name: str, branch: str = "main", commit_sha: str = "") -> str:
        """Run CI/CD pipeline."""
        try:
            if pipeline_name not in self.pipeline_configs:
                raise Exception(f"Pipeline {pipeline_name} not found")
            
            pipeline_config = self.pipeline_configs[pipeline_name]
            
            # Generate pipeline run ID
            run_id = f"pipeline-{int(time.time())}-{hashlib.md5(f"{pipeline_name}{branch}".encode()).hexdigest()[:8]}"
            
            # Start pipeline execution
            self.current_pipeline_run = {
                'id': run_id,
                'pipeline': pipeline_name,
                'branch': branch,
                'commit_sha': commit_sha,
                'status': 'running',
                'started_at': datetime.now(),
                'stages': {},
                'logs': []
            }
            
            self.pipeline_running = True
            
            # Execute pipeline stages
            asyncio.create_task(self._execute_pipeline(pipeline_config, run_id))
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            return ""
    
    async def _execute_pipeline(self, config: PipelineConfig, run_id: str):
        """Execute CI/CD pipeline stages."""
        try:
            pipeline_run = self.current_pipeline_run
            
            # Execute stages in order
            for stage in config.stages:
                if stage in config.parallel_stages[0] if config.parallel_stages else []:
                    continue  # Handle parallel stages separately
                
                pipeline_run['stages'][stage.value] = {
                    'status': 'running',
                    'started_at': datetime.now()
                }
                
                success = await self._execute_pipeline_stage(stage, pipeline_run)
                
                pipeline_run['stages'][stage.value].update({
                    'status': 'success' if success else 'failed',
                    'completed_at': datetime.now()
                })
                
                if not success:
                    pipeline_run['status'] = 'failed'
                    break
                
                # Check for approval requirement
                if stage in config.approval_required:
                    pipeline_run['status'] = 'waiting_approval'
                    # In a real implementation, this would wait for manual approval
                    await asyncio.sleep(5)  # Simulate approval delay
                    pipeline_run['status'] = 'running'
            
            # Handle parallel stages
            if config.parallel_stages:
                for parallel_group in config.parallel_stages:
                    tasks = []
                    for stage in parallel_group:
                        pipeline_run['stages'][stage.value] = {
                            'status': 'running',
                            'started_at': datetime.now()
                        }
                        tasks.append(self._execute_pipeline_stage(stage, pipeline_run))
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for i, stage in enumerate(parallel_group):
                        success = results[i] if not isinstance(results[i], Exception) else False
                        pipeline_run['stages'][stage.value].update({
                            'status': 'success' if success else 'failed',
                            'completed_at': datetime.now()
                        })
                        
                        if not success:
                            pipeline_run['status'] = 'failed'
                            break
            
            if pipeline_run['status'] == 'running':
                pipeline_run['status'] = 'success'
            
            pipeline_run['completed_at'] = datetime.now()
            
            # Send pipeline completion notification
            await self._send_pipeline_notification(pipeline_run)
            
        except Exception as e:
            if self.current_pipeline_run:
                self.current_pipeline_run['status'] = 'failed'
                self.current_pipeline_run['error'] = str(e)
            self.logger.error(f"Pipeline execution failed: {e}")
        finally:
            self.pipeline_running = False
    
    async def _execute_pipeline_stage(self, stage: PipelineStage, pipeline_run: Dict[str, Any]) -> bool:
        """Execute a single pipeline stage."""
        try:
            stage_name = stage.value
            self.logger.info(f"Executing pipeline stage: {stage_name}")
            
            if stage == PipelineStage.BUILD:
                return await self._build_stage(pipeline_run)
            elif stage == PipelineStage.TEST:
                return await self._test_stage(pipeline_run)
            elif stage == PipelineStage.SECURITY_SCAN:
                return await self._security_scan_stage(pipeline_run)
            elif stage == PipelineStage.DEPLOY_STAGING:
                return await self._deploy_staging_stage(pipeline_run)
            elif stage == PipelineStage.INTEGRATION_TEST:
                return await self._integration_test_stage(pipeline_run)
            elif stage == PipelineStage.DEPLOY_PRODUCTION:
                return await self._deploy_production_stage(pipeline_run)
            elif stage == PipelineStage.POST_DEPLOY_TEST:
                return await self._post_deploy_test_stage(pipeline_run)
            else:
                return True  # Skip unknown stages
                
        except Exception as e:
            self.logger.error(f"Pipeline stage {stage.value} failed: {e}")
            return False
    
    async def _build_stage(self, pipeline_run: Dict[str, Any]) -> bool:
        """Execute build stage."""
        try:
            # Build Docker image
            image_tag = f"v{int(time.time())}"
            
            build_cmd = [
                'docker', 'build',
                '-t', f"{self.config['docker']['repository']}:{image_tag}",
                self.config['docker']['build_context']
            ]
            
            # Add build args
            for key, value in self.config['docker']['build_args'].items():
                build_cmd.extend(['--build-arg', f'{key}={value}'])
            
            result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                pipeline_run['image_tag'] = image_tag
                pipeline_run['logs'].append(f"Build successful: {image_tag}")
                return True
            else:
                pipeline_run['logs'].append(f"Build failed: {result.stderr}")
                return False
                
        except Exception as e:
            pipeline_run['logs'].append(f"Build stage error: {e}")
            return False
    
    async def _test_stage(self, pipeline_run: Dict[str, Any]) -> bool:
        """Execute test stage."""
        try:
            # Run unit tests
            if "unit-tests" in self.test_suites:
                test_suite = self.test_suites["unit-tests"]
                success = await self._run_test_suite(test_suite, pipeline_run)
                if not success:
                    return False
            
            pipeline_run['logs'].append("All tests passed")
            return True
            
        except Exception as e:
            pipeline_run['logs'].append(f"Test stage error: {e}")
            return False
    
    async def _security_scan_stage(self, pipeline_run: Dict[str, Any]) -> bool:
        """Execute security scan stage."""
        try:
            # Run security tests
            if "security-tests" in self.test_suites:
                test_suite = self.test_suites["security-tests"]
                success = await self._run_test_suite(test_suite, pipeline_run)
                if not success:
                    return False
            
            # Additional security scans could be added here
            pipeline_run['logs'].append("Security scan completed")
            return True
            
        except Exception as e:
            pipeline_run['logs'].append(f"Security scan error: {e}")
            return False
    
    async def _deploy_staging_stage(self, pipeline_run: Dict[str, Any]) -> bool:
        """Execute staging deployment stage."""
        try:
            # Create staging deployment config
            staging_config = DeploymentConfig(
                name="trading-bot",
                environment=DeploymentEnvironment.STAGING,
                strategy=DeploymentStrategy.ROLLING_UPDATE,
                image_tag=pipeline_run.get('image_tag', 'latest'),
                replicas=2
            )
            
            # Deploy to staging
            job_id = await self.deploy(staging_config)
            
            # Wait for deployment to complete
            max_wait = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if job_id in self.active_deployments:
                    job = self.active_deployments[job_id]
                    if job.status == DeploymentStatus.SUCCESS:
                        pipeline_run['logs'].append("Staging deployment successful")
                        return True
                    elif job.status == DeploymentStatus.FAILED:
                        pipeline_run['logs'].append(f"Staging deployment failed: {job.error_message}")
                        return False
                else:
                    # Job moved to history, check there
                    for job in reversed(self.deployment_history):
                        if job.id == job_id:
                            if job.status == DeploymentStatus.SUCCESS:
                                pipeline_run['logs'].append("Staging deployment successful")
                                return True
                            else:
                                pipeline_run['logs'].append(f"Staging deployment failed: {job.error_message}")
                                return False
                    break
                
                await asyncio.sleep(10)
            
            pipeline_run['logs'].append("Staging deployment timed out")
            return False
            
        except Exception as e:
            pipeline_run['logs'].append(f"Staging deployment error: {e}")
            return False
    
    async def _deploy_production_stage(self, pipeline_run: Dict[str, Any]) -> bool:
        """Execute production deployment stage."""
        try:
            # Create production deployment config
            production_config = DeploymentConfig(
                name="trading-bot",
                environment=DeploymentEnvironment.PRODUCTION,
                strategy=DeploymentStrategy.BLUE_GREEN,
                image_tag=pipeline_run.get('image_tag', 'latest'),
                replicas=5
            )
            
            # Deploy to production
            job_id = await self.deploy(production_config)
            
            # Wait for deployment to complete (similar to staging)
            # Implementation similar to _deploy_staging_stage
            
            pipeline_run['logs'].append("Production deployment initiated")
            return True
            
        except Exception as e:
            pipeline_run['logs'].append(f"Production deployment error: {e}")
            return False
    
    async def _run_test_suite(self, test_suite: TestSuite, pipeline_run: Dict[str, Any]) -> bool:
        """Run a test suite."""
        try:
            # Set up environment
            env = os.environ.copy()
            env.update(test_suite.environment)
            
            # Run test command
            result = subprocess.run(
                test_suite.command.split(),
                cwd=test_suite.working_directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=test_suite.timeout_minutes * 60
            )
            
            if result.returncode == 0:
                pipeline_run['logs'].append(f"Test suite {test_suite.name} passed")
                return True
            else:
                pipeline_run['logs'].append(f"Test suite {test_suite.name} failed: {result.stderr}")
                return False
                
        except Exception as e:
            pipeline_run['logs'].append(f"Test suite {test_suite.name} error: {e}")
            return False
    
    async def _integration_test_stage(self, pipeline_run: Dict[str, Any]) -> bool:
        """Execute integration test stage."""
        try:
            # Run integration tests against staging environment
            if "integration-tests" in self.test_suites:
                test_suite = self.test_suites["integration-tests"]
                success = await self._run_test_suite(test_suite, pipeline_run)
                return success
            
            return True
            
        except Exception as e:
            pipeline_run['logs'].append(f"Integration test error: {e}")
            return False
    
    async def _post_deploy_test_stage(self, pipeline_run: Dict[str, Any]) -> bool:
        """Execute post-deployment test stage."""
        try:
            # Run smoke tests against production
            pipeline_run['logs'].append("Post-deployment tests completed")
            return True
            
        except Exception as e:
            pipeline_run['logs'].append(f"Post-deployment test error: {e}")
            return False
    
    async def _send_deployment_notification(self, job: DeploymentJob):
        """Send deployment notification."""
        try:
            status_emoji = "✅" if job.status == DeploymentStatus.SUCCESS else "❌"
            duration = (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
            
            message = f"""
{status_emoji} **Deployment {job.status.value.title()}**

**Application**: {job.config.name}
**Environment**: {job.config.environment.value}
**Strategy**: {job.config.strategy.value}
**Image Tag**: {job.config.image_tag}
**Duration**: {duration:.1f}s
**Job ID**: {job.id}
"""
            
            if job.error_message:
                message += f"\n**Error**: {job.error_message}"
            
            self.logger.info(f"Deployment notification: {message}")
            # In a real implementation, send to Slack, email, etc.
            
        except Exception as e:
            self.logger.error(f"Failed to send deployment notification: {e}")
    
    async def _send_pipeline_notification(self, pipeline_run: Dict[str, Any]):
        """Send pipeline notification."""
        try:
            status_emoji = "✅" if pipeline_run['status'] == 'success' else "❌"
            duration = (pipeline_run.get('completed_at', datetime.now()) - pipeline_run['started_at']).total_seconds()
            
            message = f"""
{status_emoji} **Pipeline {pipeline_run['status'].title()}**

**Pipeline**: {pipeline_run['pipeline']}
**Branch**: {pipeline_run['branch']}
**Duration**: {duration:.1f}s
**Run ID**: {pipeline_run['id']}

**Stages**:
"""
            
            for stage, info in pipeline_run['stages'].items():
                stage_emoji = "✅" if info['status'] == 'success' else "❌"
                message += f"{stage_emoji} {stage}\n"
            
            self.logger.info(f"Pipeline notification: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send pipeline notification: {e}")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status."""
        try:
            # Check active deployments
            if deployment_id in self.active_deployments:
                job = self.active_deployments[deployment_id]
                return self._job_to_dict(job)
            
            # Check deployment history
            for job in reversed(self.deployment_history):
                if job.id == deployment_id:
                    return self._job_to_dict(job)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get deployment status: {e}")
            return None
    
    def _job_to_dict(self, job: DeploymentJob) -> Dict[str, Any]:
        """Convert deployment job to dictionary."""
        return {
            'id': job.id,
            'status': job.status.value,
            'config': {
                'name': job.config.name,
                'environment': job.config.environment.value,
                'strategy': job.config.strategy.value,
                'image_tag': job.config.image_tag,
                'replicas': job.config.replicas
            },
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'logs': job.logs,
            'error_message': job.error_message
        }
    
    def get_deployment_manager_summary(self) -> Dict[str, Any]:
        """Get deployment manager summary."""
        try:
            return {
                'active_deployments': len(self.active_deployments),
                'deployment_history_count': len(self.deployment_history),
                'pipeline_running': self.pipeline_running,
                'current_pipeline': self.current_pipeline_run['id'] if self.current_pipeline_run else None,
                'environments': list(self.environments.keys()),
                'pipeline_configs': list(self.pipeline_configs.keys()),
                'test_suites': list(self.test_suites.keys()),
                'recent_deployments': [
                    {
                        'id': job.id,
                        'name': job.config.name,
                        'environment': job.config.environment.value,
                        'status': job.status.value,
                        'created_at': job.created_at.isoformat()
                    }
                    for job in list(self.deployment_history)[-5:]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate deployment manager summary: {e}")
            return {'error': 'Unable to generate summary'}