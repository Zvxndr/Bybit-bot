#!/usr/bin/env python3
"""
Deployment Management CLI

Command-line interface for managing deployments, including infrastructure
setup, CI/CD pipeline configuration, health checks, and monitoring.

Usage:
    python deploy.py generate [--environment ENV]
    python deploy.py validate [--environment ENV]
    python deploy.py deploy [--environment ENV] [--dry-run]
    python deploy.py rollback [--environment ENV] [--to-version VERSION]
    python deploy.py health-check [--environment ENV]
    python deploy.py scale [--replicas COUNT] [--environment ENV]

Examples:
    # Generate deployment manifests for production
    python deploy.py generate --environment production
    
    # Deploy to staging environment
    python deploy.py deploy --environment staging
    
    # Scale production deployment
    python deploy.py scale --replicas 5 --environment production
    
    # Rollback to previous version
    python deploy.py rollback --environment production
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import requests

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.bot.deployment.infrastructure import (
    DeploymentConfig, 
    KubernetesDeploymentGenerator,
    CIPipelineGenerator,
    MonitoringSetup
)


class DeploymentManager:
    """Manage deployments and infrastructure."""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.k8s_dir = self.project_root / "k8s"
        self.monitoring_dir = self.project_root / "monitoring"
        
        # Load deployment configuration
        self.config = self._load_deployment_config()
        
        print(f"🚀 Deployment Manager - {environment} environment")
        print(f"   Project root: {self.project_root}")
        print(f"   Namespace: {self.config.namespace}")
        print("=" * 60)
    
    def _load_deployment_config(self) -> DeploymentConfig:
        """Load deployment configuration for environment."""
        config_file = self.project_root / "config" / f"deployment-{self.environment}.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        else:
            config_data = {}
        
        # Environment-specific defaults
        env_defaults = {
            "development": {
                "min_replicas": 1,
                "max_replicas": 2,
                "cpu_request": "50m",
                "memory_request": "128Mi",
                "hostname": "dev.trading-bot.local"
            },
            "staging": {
                "min_replicas": 2,
                "max_replicas": 4,
                "cpu_request": "100m",
                "memory_request": "256Mi",
                "hostname": "staging.trading-bot.your-domain.com"
            },
            "production": {
                "min_replicas": 3,
                "max_replicas": 10,
                "cpu_request": "200m",
                "memory_request": "512Mi",
                "hostname": "trading-bot.your-domain.com"
            }
        }
        
        defaults = env_defaults.get(self.environment, env_defaults["production"])
        
        return DeploymentConfig(
            environment=self.environment,
            namespace=config_data.get("namespace", f"trading-bot-{self.environment}"),
            image_registry=config_data.get("image_registry", "your-registry.com"),
            image_repository=config_data.get("image_repository", "trading-bot"),
            image_tag=config_data.get("image_tag", "latest"),
            min_replicas=config_data.get("min_replicas", defaults["min_replicas"]),
            max_replicas=config_data.get("max_replicas", defaults["max_replicas"]),
            cpu_request=config_data.get("cpu_request", defaults["cpu_request"]),
            memory_request=config_data.get("memory_request", defaults["memory_request"]),
            hostname=config_data.get("hostname", defaults["hostname"]),
            enable_ingress=config_data.get("enable_ingress", True),
            enable_tls=config_data.get("enable_tls", self.environment == "production")
        )
    
    def generate_infrastructure(self) -> bool:
        """Generate all infrastructure manifests."""
        try:
            print("📦 Generating Kubernetes manifests...")
            
            # Generate Kubernetes manifests
            k8s_generator = KubernetesDeploymentGenerator(self.config)
            k8s_generator.generate_all_manifests()
            
            # Generate CI/CD pipelines
            print("🔄 Generating CI/CD pipelines...")
            ci_generator = CIPipelineGenerator(self.config)
            ci_generator.generate_github_actions()
            ci_generator.generate_gitlab_ci()
            
            # Generate monitoring setup
            print("📊 Generating monitoring configuration...")
            monitoring = MonitoringSetup(self.config)
            monitoring.generate_prometheus_config()
            monitoring.generate_alert_rules()
            monitoring.generate_grafana_dashboard()
            
            print("✅ Infrastructure generation completed!")
            return True
            
        except Exception as e:
            print(f"❌ Infrastructure generation failed: {e}")
            return False
    
    def validate_manifests(self) -> bool:
        """Validate Kubernetes manifests."""
        try:
            print("🔍 Validating Kubernetes manifests...")
            
            if not self.k8s_dir.exists():
                print(f"❌ Kubernetes manifests directory not found: {self.k8s_dir}")
                return False
            
            # Check if kubectl is available
            if not self._check_command("kubectl"):
                print("❌ kubectl not found. Please install kubectl to validate manifests.")
                return False
            
            # Validate each manifest file
            manifest_files = list(self.k8s_dir.glob("*.yaml"))
            if not manifest_files:
                print(f"❌ No manifest files found in {self.k8s_dir}")
                return False
            
            for manifest_file in manifest_files:
                print(f"   Validating {manifest_file.name}...")
                
                result = subprocess.run([
                    "kubectl", "apply", "--dry-run=client", "-f", str(manifest_file)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"❌ Validation failed for {manifest_file.name}:")
                    print(f"   {result.stderr}")
                    return False
            
            print("✅ All manifests are valid!")
            return True
            
        except Exception as e:
            print(f"❌ Manifest validation failed: {e}")
            return False
    
    def deploy(self, dry_run: bool = False) -> bool:
        """Deploy the application to Kubernetes."""
        try:
            print(f"🚀 {'Dry run deployment' if dry_run else 'Deploying'} to {self.environment}...")
            
            # Check prerequisites
            if not self._check_prerequisites():
                return False
            
            # Apply manifests
            kubectl_args = ["kubectl", "apply"]
            if dry_run:
                kubectl_args.extend(["--dry-run=server"])
            kubectl_args.extend(["-f", str(self.k8s_dir)])
            
            print("   Applying Kubernetes manifests...")
            result = subprocess.run(kubectl_args, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Deployment failed:")
                print(f"   {result.stderr}")
                return False
            
            print(result.stdout)
            
            if not dry_run:
                # Wait for deployment to be ready
                print("   Waiting for deployment to be ready...")
                result = subprocess.run([
                    "kubectl", "wait", "--for=condition=available",
                    "--timeout=300s", f"deployment/trading-bot-api",
                    "-n", self.config.namespace
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"❌ Deployment readiness check failed:")
                    print(f"   {result.stderr}")
                    return False
                
                print("✅ Deployment completed successfully!")
                
                # Run health check
                self.health_check()
            else:
                print("✅ Dry run completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False
    
    def rollback(self, to_version: Optional[str] = None) -> bool:
        """Rollback deployment to previous version."""
        try:
            print(f"⏪ Rolling back deployment in {self.environment}...")
            
            if not self._check_prerequisites():
                return False
            
            # Get rollback target
            if to_version:
                rollback_cmd = [
                    "kubectl", "rollout", "undo", 
                    f"deployment/trading-bot-api",
                    f"--to-revision={to_version}",
                    "-n", self.config.namespace
                ]
            else:
                rollback_cmd = [
                    "kubectl", "rollout", "undo",
                    f"deployment/trading-bot-api",
                    "-n", self.config.namespace
                ]
            
            result = subprocess.run(rollback_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Rollback failed:")
                print(f"   {result.stderr}")
                return False
            
            print(result.stdout)
            
            # Wait for rollback to complete
            print("   Waiting for rollback to complete...")
            result = subprocess.run([
                "kubectl", "rollout", "status",
                f"deployment/trading-bot-api",
                "-n", self.config.namespace
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Rollback status check failed:")
                print(f"   {result.stderr}")
                return False
            
            print("✅ Rollback completed successfully!")
            
            # Run health check
            self.health_check()
            
            return True
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False
    
    def scale(self, replicas: int) -> bool:
        """Scale the deployment to specified number of replicas."""
        try:
            print(f"📏 Scaling deployment to {replicas} replicas...")
            
            if not self._check_prerequisites():
                return False
            
            result = subprocess.run([
                "kubectl", "scale", "deployment/trading-bot-api",
                f"--replicas={replicas}",
                "-n", self.config.namespace
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Scaling failed:")
                print(f"   {result.stderr}")
                return False
            
            print(result.stdout)
            
            # Wait for scaling to complete
            print("   Waiting for scaling to complete...")
            result = subprocess.run([
                "kubectl", "rollout", "status",
                f"deployment/trading-bot-api",
                "-n", self.config.namespace
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Scaling status check failed:")
                print(f"   {result.stderr}")
                return False
            
            print("✅ Scaling completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Scaling failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """Run health checks on the deployed application."""
        try:
            print("🏥 Running health checks...")
            
            if not self._check_prerequisites():
                return False
            
            # Check pod status
            print("   Checking pod status...")
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-l", "app=trading-bot",
                "-n", self.config.namespace,
                "-o", "json"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to get pod status:")
                print(f"   {result.stderr}")
                return False
            
            pods_data = json.loads(result.stdout)
            pods = pods_data.get("items", [])
            
            if not pods:
                print("❌ No pods found!")
                return False
            
            running_pods = 0
            for pod in pods:
                pod_name = pod["metadata"]["name"]
                pod_status = pod["status"]["phase"]
                
                if pod_status == "Running":
                    running_pods += 1
                    print(f"   ✅ Pod {pod_name}: {pod_status}")
                else:
                    print(f"   ❌ Pod {pod_name}: {pod_status}")
            
            if running_pods == 0:
                print("❌ No running pods found!")
                return False
            
            # Check service endpoints
            print("   Checking service endpoints...")
            result = subprocess.run([
                "kubectl", "get", "endpoints",
                "trading-bot-api-service",
                "-n", self.config.namespace,
                "-o", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                endpoints_data = json.loads(result.stdout)
                subsets = endpoints_data.get("subsets", [])
                
                if subsets and subsets[0].get("addresses"):
                    print(f"   ✅ Service endpoints: {len(subsets[0]['addresses'])} ready")
                else:
                    print("   ❌ No service endpoints ready")
                    return False
            
            # Try to access health endpoint
            if self.config.enable_ingress:
                health_url = f"https://{self.config.hostname}/health"
                print(f"   Testing health endpoint: {health_url}")
                
                try:
                    response = requests.get(health_url, timeout=10)
                    if response.status_code == 200:
                        print("   ✅ Health endpoint responding correctly")
                    else:
                        print(f"   ⚠️  Health endpoint returned status {response.status_code}")
                except requests.RequestException as e:
                    print(f"   ⚠️  Could not reach health endpoint: {e}")
            
            print("✅ Health check completed!")
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def get_logs(self, lines: int = 100, follow: bool = False) -> bool:
        """Get application logs."""
        try:
            print(f"📋 Getting logs (last {lines} lines)...")
            
            if not self._check_prerequisites():
                return False
            
            kubectl_args = [
                "kubectl", "logs",
                "-l", "app=trading-bot",
                "-n", self.config.namespace,
                "--tail", str(lines)
            ]
            
            if follow:
                kubectl_args.append("-f")
            
            # Stream logs to stdout
            subprocess.run(kubectl_args)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to get logs: {e}")
            return False
    
    def _check_prerequisites(self) -> bool:
        """Check deployment prerequisites."""
        # Check kubectl
        if not self._check_command("kubectl"):
            print("❌ kubectl not found. Please install kubectl.")
            return False
        
        # Check Kubernetes connection
        result = subprocess.run([
            "kubectl", "cluster-info"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Cannot connect to Kubernetes cluster:")
            print(f"   {result.stderr}")
            return False
        
        # Check if namespace exists
        result = subprocess.run([
            "kubectl", "get", "namespace", self.config.namespace
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️  Namespace {self.config.namespace} does not exist, it will be created.")
        
        return True
    
    def _check_command(self, command: str) -> bool:
        """Check if command is available."""
        return subprocess.run([
            "which" if os.name != "nt" else "where", command
        ], capture_output=True).returncode == 0


def main():
    """Main deployment CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deployment Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'staging', 'production'],
        default='production',
        help='Target environment (default: production)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate infrastructure manifests')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate Kubernetes manifests')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy application')
    deploy_parser.add_argument('--dry-run', action='store_true', help='Perform dry run')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback deployment')
    rollback_parser.add_argument('--to-version', help='Rollback to specific version')
    
    # Scale command
    scale_parser = subparsers.add_parser('scale', help='Scale deployment')
    scale_parser.add_argument('--replicas', type=int, required=True, help='Number of replicas')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Run health checks')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Get application logs')
    logs_parser.add_argument('--lines', type=int, default=100, help='Number of log lines')
    logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow log output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        deployment_manager = DeploymentManager(environment=args.environment)
        success = False
        
        if args.command == 'generate':
            success = deployment_manager.generate_infrastructure()
        elif args.command == 'validate':
            success = deployment_manager.validate_manifests()
        elif args.command == 'deploy':
            success = deployment_manager.deploy(dry_run=args.dry_run)
        elif args.command == 'rollback':
            success = deployment_manager.rollback(to_version=args.to_version)
        elif args.command == 'scale':
            success = deployment_manager.scale(replicas=args.replicas)
        elif args.command == 'health-check':
            success = deployment_manager.health_check()
        elif args.command == 'logs':
            success = deployment_manager.get_logs(lines=args.lines, follow=args.follow)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())