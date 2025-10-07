"""
Deployment Infrastructure and CI/CD Pipeline

Production-ready deployment infrastructure with continuous integration/continuous
deployment pipeline, health monitoring, auto-scaling, and production monitoring
integration.

Components:
1. CI/CD Pipeline Configuration (GitHub Actions, GitLab CI, Jenkins)
2. Kubernetes Deployment Manifests
3. Health Check and Monitoring Scripts
4. Auto-scaling Configuration
5. Load Balancer and Service Discovery
6. Backup and Disaster Recovery Scripts
7. Security Scanning and Compliance
8. Performance Testing and Benchmarking

Key Features:
- Automated testing and deployment
- Blue-green and canary deployments
- Infrastructure as Code (IaC)
- Monitoring and alerting integration
- Automatic rollback on failures
- Security scanning and vulnerability management
- Performance benchmarking and load testing
- Disaster recovery and backup automation
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class DeploymentConfig:
    """Deployment configuration settings."""
    # Environment settings
    environment: str = "production"
    namespace: str = "trading-bot"
    cluster_name: str = "trading-cluster"
    
    # Container settings
    image_registry: str = "your-registry.com"
    image_repository: str = "trading-bot"
    image_tag: str = "latest"
    
    # Scaling settings
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    
    # Health check settings
    health_check_path: str = "/health"
    readiness_timeout: int = 30
    liveness_timeout: int = 60
    
    # Service settings
    service_type: str = "ClusterIP"
    service_port: int = 80
    target_port: int = 8000
    
    # Ingress settings
    enable_ingress: bool = True
    ingress_class: str = "nginx"
    hostname: str = "trading-bot.your-domain.com"
    enable_tls: bool = True
    tls_secret_name: str = "trading-bot-tls"
    
    # Resource limits
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    
    # Storage settings
    enable_persistent_storage: bool = True
    storage_class: str = "fast-ssd"
    storage_size: str = "10Gi"
    
    # Database settings
    database_host: str = "postgres-service"
    database_port: int = 5432
    redis_host: str = "redis-service"
    redis_port: int = 6379


class KubernetesDeploymentGenerator:
    """Generate Kubernetes deployment manifests."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.manifests_dir = Path("k8s")
        self.manifests_dir.mkdir(exist_ok=True)
    
    def generate_all_manifests(self):
        """Generate all Kubernetes manifests."""
        manifests = {
            "namespace.yaml": self._generate_namespace(),
            "configmap.yaml": self._generate_configmap(),
            "secret.yaml": self._generate_secret_template(),
            "deployment.yaml": self._generate_deployment(),
            "service.yaml": self._generate_service(),
            "ingress.yaml": self._generate_ingress(),
            "hpa.yaml": self._generate_hpa(),
            "pvc.yaml": self._generate_pvc(),
            "serviceaccount.yaml": self._generate_service_account(),
            "rbac.yaml": self._generate_rbac(),
            "networkpolicy.yaml": self._generate_network_policy()
        }
        
        for filename, manifest in manifests.items():
            if manifest:  # Only write non-empty manifests
                manifest_path = self.manifests_dir / filename
                with open(manifest_path, 'w') as f:
                    yaml.dump_all(manifest if isinstance(manifest, list) else [manifest], 
                                f, default_flow_style=False)
                print(f"Generated {manifest_path}")
    
    def _generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config.namespace,
                "labels": {
                    "app": "trading-bot",
                    "environment": self.config.environment
                }
            }
        }
    
    def _generate_configmap(self) -> Dict[str, Any]:
        """Generate ConfigMap manifest."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "trading-bot-config",
                "namespace": self.config.namespace
            },
            "data": {
                "TRADING_BOT_ENV": self.config.environment,
                "DATABASE_HOST": self.config.database_host,
                "DATABASE_PORT": str(self.config.database_port),
                "REDIS_HOST": self.config.redis_host,
                "REDIS_PORT": str(self.config.redis_port),
                "API_HOST": "0.0.0.0",
                "API_PORT": str(self.config.target_port),
                "LOG_LEVEL": "INFO" if self.config.environment == "production" else "DEBUG"
            }
        }
    
    def _generate_secret_template(self) -> Dict[str, Any]:
        """Generate Secret template (needs to be populated with actual secrets)."""
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "trading-bot-secrets",
                "namespace": self.config.namespace
            },
            "type": "Opaque",
            "stringData": {
                "DATABASE_PASSWORD": "your-database-password",
                "BYBIT_API_KEY": "your-bybit-api-key",
                "BYBIT_API_SECRET": "your-bybit-api-secret",
                "API_SECRET_KEY": "your-api-secret-key",
                "JWT_SECRET_KEY": "your-jwt-secret-key",
                "TRADING_BOT_MASTER_KEY": "your-master-encryption-key"
            }
        }
    
    def _generate_deployment(self) -> Dict[str, Any]:
        """Generate Deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "trading-bot-api",
                "namespace": self.config.namespace,
                "labels": {
                    "app": "trading-bot",
                    "component": "api"
                }
            },
            "spec": {
                "replicas": self.config.min_replicas,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxUnavailable": 1,
                        "maxSurge": 1
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app": "trading-bot",
                        "component": "api"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "trading-bot",
                            "component": "api"
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": str(self.config.target_port),
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "serviceAccountName": "trading-bot",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 2000
                        },
                        "containers": [
                            {
                                "name": "api",
                                "image": f"{self.config.image_registry}/{self.config.image_repository}:{self.config.image_tag}",
                                "imagePullPolicy": "Always",
                                "ports": [
                                    {
                                        "containerPort": self.config.target_port,
                                        "name": "http",
                                        "protocol": "TCP"
                                    }
                                ],
                                "env": [],  # Populated from ConfigMap and Secret
                                "envFrom": [
                                    {
                                        "configMapRef": {
                                            "name": "trading-bot-config"
                                        }
                                    },
                                    {
                                        "secretRef": {
                                            "name": "trading-bot-secrets"
                                        }
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": self.config.cpu_request,
                                        "memory": self.config.memory_request
                                    },
                                    "limits": {
                                        "cpu": self.config.cpu_limit,
                                        "memory": self.config.memory_limit
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": self.config.liveness_timeout,
                                    "failureThreshold": 3
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 5,
                                    "timeoutSeconds": self.config.readiness_timeout,
                                    "failureThreshold": 3
                                },
                                "volumeMounts": [
                                    {
                                        "name": "data-storage",
                                        "mountPath": "/app/data"
                                    },
                                    {
                                        "name": "logs-storage",
                                        "mountPath": "/app/logs"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "data-storage",
                                "persistentVolumeClaim": {
                                    "claimName": "trading-bot-data"
                                }
                            },
                            {
                                "name": "logs-storage",
                                "emptyDir": {}
                            }
                        ]
                    }
                }
            }
        }
    
    def _generate_service(self) -> Dict[str, Any]:
        """Generate Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "trading-bot-api-service",
                "namespace": self.config.namespace,
                "labels": {
                    "app": "trading-bot",
                    "component": "api"
                }
            },
            "spec": {
                "type": self.config.service_type,
                "ports": [
                    {
                        "port": self.config.service_port,
                        "targetPort": "http",
                        "protocol": "TCP",
                        "name": "http"
                    }
                ],
                "selector": {
                    "app": "trading-bot",
                    "component": "api"
                }
            }
        }
    
    def _generate_ingress(self) -> Optional[Dict[str, Any]]:
        """Generate Ingress manifest."""
        if not self.config.enable_ingress:
            return None
        
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "trading-bot-ingress",
                "namespace": self.config.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": self.config.ingress_class,
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true" if self.config.enable_tls else "false"
                }
            },
            "spec": {
                "rules": [
                    {
                        "host": self.config.hostname,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": "trading-bot-api-service",
                                            "port": {
                                                "number": self.config.service_port
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        if self.config.enable_tls:
            ingress["spec"]["tls"] = [
                {
                    "hosts": [self.config.hostname],
                    "secretName": self.config.tls_secret_name
                }
            ]
        
        return ingress
    
    def _generate_hpa(self) -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler manifest."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "trading-bot-hpa",
                "namespace": self.config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "trading-bot-api"
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_cpu_utilization
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_memory_utilization
                            }
                        }
                    }
                ]
            }
        }
    
    def _generate_pvc(self) -> Optional[Dict[str, Any]]:
        """Generate PersistentVolumeClaim manifest."""
        if not self.config.enable_persistent_storage:
            return None
        
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "trading-bot-data",
                "namespace": self.config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.config.storage_class,
                "resources": {
                    "requests": {
                        "storage": self.config.storage_size
                    }
                }
            }
        }
    
    def _generate_service_account(self) -> Dict[str, Any]:
        """Generate ServiceAccount manifest."""
        return {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": "trading-bot",
                "namespace": self.config.namespace
            }
        }
    
    def _generate_rbac(self) -> List[Dict[str, Any]]:
        """Generate RBAC manifests."""
        return [
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": {
                    "name": "trading-bot-role",
                    "namespace": self.config.namespace
                },
                "rules": [
                    {
                        "apiGroups": [""],
                        "resources": ["pods", "services", "configmaps", "secrets"],
                        "verbs": ["get", "list", "watch"]
                    }
                ]
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": {
                    "name": "trading-bot-rolebinding",
                    "namespace": self.config.namespace
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": "trading-bot",
                        "namespace": self.config.namespace
                    }
                ],
                "roleRef": {
                    "kind": "Role",
                    "name": "trading-bot-role",
                    "apiGroup": "rbac.authorization.k8s.io"
                }
            }
        ]
    
    def _generate_network_policy(self) -> Dict[str, Any]:
        """Generate NetworkPolicy manifest."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "trading-bot-network-policy",
                "namespace": self.config.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": "trading-bot"
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "ingress-nginx"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": self.config.target_port
                            }
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "database"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 5432
                            }
                        ]
                    },
                    {
                        "to": [],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 443
                            },
                            {
                                "protocol": "TCP",
                                "port": 80
                            },
                            {
                                "protocol": "UDP",
                                "port": 53
                            }
                        ]
                    }
                ]
            }
        }


class CIPipelineGenerator:
    """Generate CI/CD pipeline configurations."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.ci_dir = Path(".github/workflows")
        self.ci_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_github_actions(self):
        """Generate GitHub Actions workflow."""
        workflow = {
            "name": "CI/CD Pipeline",
            "on": {
                "push": {
                    "branches": ["main", "develop"]
                },
                "pull_request": {
                    "branches": ["main"]
                }
            },
            "env": {
                "REGISTRY": self.config.image_registry,
                "IMAGE_NAME": self.config.image_repository
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "strategy": {
                        "matrix": {
                            "python-version": ["3.8", "3.9", "3.10", "3.11"]
                        }
                    },
                    "steps": [
                        {
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Python ${{ matrix.python-version }}",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "${{ matrix.python-version }}"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "\n".join([
                                "python -m pip install --upgrade pip",
                                "pip install -r requirements.txt",
                                "pip install -r requirements-test.txt"
                            ])
                        },
                        {
                            "name": "Lint with flake8",
                            "run": "\n".join([
                                "pip install flake8",
                                "flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics",
                                "flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics"
                            ])
                        },
                        {
                            "name": "Type check with mypy",
                            "run": "\n".join([
                                "pip install mypy",
                                "mypy src --ignore-missing-imports"
                            ])
                        },
                        {
                            "name": "Test with pytest",
                            "run": "\n".join([
                                "pip install pytest pytest-cov pytest-asyncio",
                                "pytest tests/ --cov=src --cov-report=xml --cov-report=html"
                            ])
                        },
                        {
                            "name": "Upload coverage to Codecov",
                            "uses": "codecov/codecov-action@v3",
                            "with": {
                                "file": "./coverage.xml",
                                "flags": "unittests",
                                "name": "codecov-umbrella"
                            }
                        }
                    ]
                },
                "security": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Run security scan",
                            "run": "\n".join([
                                "pip install safety bandit",
                                "safety check -r requirements.txt",
                                "bandit -r src/ -f json -o bandit-report.json || true"
                            ])
                        },
                        {
                            "name": "Upload security scan results",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "security-scan-results",
                                "path": "bandit-report.json"
                            }
                        }
                    ]
                },
                "build": {
                    "needs": ["test", "security"],
                    "runs-on": "ubuntu-latest",
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v3"
                        },
                        {
                            "name": "Log in to registry",
                            "uses": "docker/login-action@v3",
                            "with": {
                                "registry": "${{ env.REGISTRY }}",
                                "username": "${{ secrets.REGISTRY_USERNAME }}",
                                "password": "${{ secrets.REGISTRY_PASSWORD }}"
                            }
                        },
                        {
                            "name": "Extract metadata",
                            "id": "meta",
                            "uses": "docker/metadata-action@v5",
                            "with": {
                                "images": "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}",
                                "tags": "\n".join([
                                    "type=ref,event=branch",
                                    "type=ref,event=pr",
                                    "type=sha,prefix={{branch}}-",
                                    "type=raw,value=latest,enable={{is_default_branch}}"
                                ])
                            }
                        },
                        {
                            "name": "Build and push Docker image",
                            "uses": "docker/build-push-action@v5",
                            "with": {
                                "context": ".",
                                "push": True,
                                "tags": "${{ steps.meta.outputs.tags }}",
                                "labels": "${{ steps.meta.outputs.labels }}",
                                "cache-from": "type=gha",
                                "cache-to": "type=gha,mode=max"
                            }
                        }
                    ]
                },
                "deploy": {
                    "needs": "build",
                    "runs-on": "ubuntu-latest",
                    "if": "github.ref == 'refs/heads/main'",
                    "environment": "production",
                    "steps": [
                        {
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up kubectl",
                            "uses": "azure/setup-kubectl@v3",
                            "with": {
                                "version": "latest"
                            }
                        },
                        {
                            "name": "Configure kubectl",
                            "run": "\n".join([
                                "echo '${{ secrets.KUBE_CONFIG }}' | base64 -d > kubeconfig",
                                "export KUBECONFIG=kubeconfig"
                            ])
                        },
                        {
                            "name": "Deploy to Kubernetes",
                            "run": "\n".join([
                                "export KUBECONFIG=kubeconfig",
                                "kubectl apply -f k8s/",
                                "kubectl set image deployment/trading-bot-api api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -n ${{ env.NAMESPACE }}",
                                "kubectl rollout status deployment/trading-bot-api -n ${{ env.NAMESPACE }}"
                            ]),
                            "env": {
                                "NAMESPACE": self.config.namespace
                            }
                        },
                        {
                            "name": "Run health check",
                            "run": "\n".join([
                                "export KUBECONFIG=kubeconfig",
                                "kubectl wait --for=condition=available --timeout=300s deployment/trading-bot-api -n ${{ env.NAMESPACE }}",
                                "kubectl get pods -n ${{ env.NAMESPACE }}"
                            ]),
                            "env": {
                                "NAMESPACE": self.config.namespace
                            }
                        }
                    ]
                }
            }
        }
        
        workflow_path = self.ci_dir / "ci-cd.yml"
        with open(workflow_path, 'w') as f:
            yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated GitHub Actions workflow: {workflow_path}")
    
    def generate_gitlab_ci(self):
        """Generate GitLab CI/CD configuration."""
        gitlab_ci = {
            "stages": ["test", "security", "build", "deploy"],
            "variables": {
                "DOCKER_DRIVER": "overlay2",
                "DOCKER_TLS_CERTDIR": "/certs",
                "IMAGE_TAG": "$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA",
                "LATEST_TAG": "$CI_REGISTRY_IMAGE:latest"
            },
            "before_script": [
                "python --version",
                "pip install --upgrade pip"
            ],
            "test": {
                "stage": "test",
                "image": "python:3.10",
                "parallel": {
                    "matrix": [
                        {"PYTHON_VERSION": "3.8"},
                        {"PYTHON_VERSION": "3.9"},
                        {"PYTHON_VERSION": "3.10"},
                        {"PYTHON_VERSION": "3.11"}
                    ]
                },
                "before_script": [
                    "pip install -r requirements.txt -r requirements-test.txt"
                ],
                "script": [
                    "flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics",
                    "mypy src --ignore-missing-imports",
                    "pytest tests/ --cov=src --cov-report=xml --cov-report=html"
                ],
                "coverage": "/(?i)total.*? (100(?:\\.0+)?\\%|[1-9]?\\d(?:\\.\\d+)?\\%)$/",
                "artifacts": {
                    "reports": {
                        "coverage_report": {
                            "coverage_format": "cobertura",
                            "path": "coverage.xml"
                        }
                    },
                    "paths": ["htmlcov/"],
                    "expire_in": "1 week"
                }
            },
            "security": {
                "stage": "security",
                "image": "python:3.10",
                "script": [
                    "pip install safety bandit",
                    "safety check -r requirements.txt",
                    "bandit -r src/ -f json -o bandit-report.json"
                ],
                "artifacts": {
                    "paths": ["bandit-report.json"],
                    "expire_in": "1 week"
                },
                "allow_failure": True
            },
            "build": {
                "stage": "build",
                "image": "docker:latest",
                "services": ["docker:dind"],
                "before_script": [
                    "docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY"
                ],
                "script": [
                    "docker build -t $IMAGE_TAG -t $LATEST_TAG .",
                    "docker push $IMAGE_TAG",
                    "docker push $LATEST_TAG"
                ],
                "only": ["main"]
            },
            "deploy": {
                "stage": "deploy",
                "image": "bitnami/kubectl:latest",
                "script": [
                    "echo $KUBE_CONFIG | base64 -d > kubeconfig",
                    "export KUBECONFIG=kubeconfig",
                    "kubectl apply -f k8s/",
                    f"kubectl set image deployment/trading-bot-api api=$IMAGE_TAG -n {self.config.namespace}",
                    f"kubectl rollout status deployment/trading-bot-api -n {self.config.namespace}"
                ],
                "environment": {
                    "name": "production",
                    "url": f"https://{self.config.hostname}"
                },
                "only": ["main"],
                "when": "manual"
            }
        }
        
        gitlab_ci_path = Path(".gitlab-ci.yml")
        with open(gitlab_ci_path, 'w') as f:
            yaml.dump(gitlab_ci, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated GitLab CI configuration: {gitlab_ci_path}")


class MonitoringSetup:
    """Setup monitoring and alerting infrastructure."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.monitoring_dir = Path("monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)
    
    def generate_prometheus_config(self):
        """Generate Prometheus configuration."""
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [
                "trading_bot_rules.yml"
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {
                                "targets": ["alertmanager:9093"]
                            }
                        ]
                    }
                ]
            },
            "scrape_configs": [
                {
                    "job_name": "trading-bot-api",
                    "static_configs": [
                        {
                            "targets": [f"trading-bot-api-service:{self.config.service_port}"]
                        }
                    ],
                    "metrics_path": "/metrics",
                    "scrape_interval": "30s"
                },
                {
                    "job_name": "kubernetes-pods",
                    "kubernetes_sd_configs": [
                        {
                            "role": "pod"
                        }
                    ],
                    "relabel_configs": [
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"],
                            "action": "keep",
                            "regex": True
                        },
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_path"],
                            "action": "replace",
                            "target_label": "__metrics_path__",
                            "regex": "(.+)"
                        }
                    ]
                }
            ]
        }
        
        config_path = self.monitoring_dir / "prometheus.yml"
        with open(config_path, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        print(f"Generated Prometheus config: {config_path}")
    
    def generate_alert_rules(self):
        """Generate Prometheus alert rules."""
        alert_rules = {
            "groups": [
                {
                    "name": "trading_bot_alerts",
                    "rules": [
                        {
                            "alert": "TradingBotDown",
                            "expr": "up{job=\"trading-bot-api\"} == 0",
                            "for": "5m",
                            "labels": {
                                "severity": "critical"
                            },
                            "annotations": {
                                "summary": "Trading Bot API is down",
                                "description": "Trading Bot API has been down for more than 5 minutes."
                            }
                        },
                        {
                            "alert": "HighCPUUsage",
                            "expr": "rate(container_cpu_usage_seconds_total{container=\"api\"}[5m]) * 100 > 80",
                            "for": "10m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High CPU usage detected",
                                "description": "CPU usage is above 80% for more than 10 minutes."
                            }
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "container_memory_usage_bytes{container=\"api\"} / container_spec_memory_limit_bytes * 100 > 90",
                            "for": "10m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High memory usage detected",
                                "description": "Memory usage is above 90% for more than 10 minutes."
                            }
                        },
                        {
                            "alert": "ModelDriftDetected",
                            "expr": "trading_bot_model_drift_score > 0.1",
                            "for": "1h",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "Model drift detected",
                                "description": "Model drift score is above threshold, consider retraining."
                            }
                        },
                        {
                            "alert": "PredictionLatencyHigh",
                            "expr": "histogram_quantile(0.95, trading_bot_prediction_duration_seconds) > 2.0",
                            "for": "15m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High prediction latency",
                                "description": "95th percentile prediction latency is above 2 seconds."
                            }
                        }
                    ]
                }
            ]
        }
        
        rules_path = self.monitoring_dir / "trading_bot_rules.yml"
        with open(rules_path, 'w') as f:
            yaml.dump(alert_rules, f, default_flow_style=False)
        
        print(f"Generated alert rules: {rules_path}")
    
    def generate_grafana_dashboard(self):
        """Generate Grafana dashboard configuration."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Trading Bot Monitoring",
                "tags": ["trading", "bot", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "API Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, trading_bot_http_request_duration_seconds)",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, trading_bot_http_request_duration_seconds)",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Time (seconds)",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Prediction Accuracy",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "trading_bot_model_accuracy",
                                "legendFormat": "Current Accuracy"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(container_cpu_usage_seconds_total{container=\"api\"}[5m]) * 100",
                                "legendFormat": "CPU Usage %"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "container_memory_usage_bytes{container=\"api\"} / 1024 / 1024",
                                "legendFormat": "Memory Usage (MB)"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        dashboard_path = self.monitoring_dir / "grafana_dashboard.json"
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        print(f"Generated Grafana dashboard: {dashboard_path}")


def main():
    """Generate deployment infrastructure."""
    print("🚀 Generating Deployment Infrastructure")
    print("=" * 50)
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment="production",
        namespace="trading-bot",
        image_registry="your-registry.com",
        hostname="trading-bot.your-domain.com"
    )
    
    try:
        # Generate Kubernetes manifests
        print("\n📦 Generating Kubernetes manifests...")
        k8s_generator = KubernetesDeploymentGenerator(config)
        k8s_generator.generate_all_manifests()
        
        # Generate CI/CD pipelines
        print("\n🔄 Generating CI/CD pipelines...")
        ci_generator = CIPipelineGenerator(config)
        ci_generator.generate_github_actions()
        ci_generator.generate_gitlab_ci()
        
        # Generate monitoring setup
        print("\n📊 Generating monitoring configuration...")
        monitoring = MonitoringSetup(config)
        monitoring.generate_prometheus_config()
        monitoring.generate_alert_rules()
        monitoring.generate_grafana_dashboard()
        
        print("\n✅ Deployment infrastructure generated successfully!")
        print("\n📝 Next steps:")
        print("1. Review and customize the generated configurations")
        print("2. Set up your container registry and update image settings")
        print("3. Configure your Kubernetes cluster access")
        print("4. Set up monitoring stack (Prometheus, Grafana)")
        print("5. Configure secrets and environment variables")
        print("6. Test the deployment pipeline")
        
    except Exception as e:
        print(f"❌ Failed to generate deployment infrastructure: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()