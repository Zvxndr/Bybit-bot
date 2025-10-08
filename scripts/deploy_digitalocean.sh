#!/bin/bash
# DigitalOcean Production Deployment Script
# Usage: ./deploy_digitalocean.sh

set -e

echo "🚀 DigitalOcean Production Deployment Starting..."

# Configuration
DROPLET_NAME="trading-bot-prod"
REGION="syd1"  # Sydney for Australian compliance
SIZE="s-2vcpu-4gb"
IMAGE="ubuntu-22-04-x64"
SSH_KEY_ID="your-ssh-key-id"  # Replace with your SSH key ID

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    print_error "doctl CLI not found. Please install it first:"
    echo "curl -sL https://github.com/digitalocean/doctl/releases/download/v1.94.0/doctl-1.94.0-linux-amd64.tar.gz | tar -xzv"
    echo "sudo mv doctl /usr/local/bin"
    exit 1
fi

# Authenticate doctl (user should have done this)
print_status "Checking DigitalOcean authentication..."
if ! doctl account get &> /dev/null; then
    print_error "Please authenticate doctl first:"
    echo "doctl auth init"
    exit 1
fi

# Create VPC for isolated network
print_status "Creating VPC network..."
VPC_ID=$(doctl vpcs create \
    --name "trading-vpc-prod" \
    --region "$REGION" \
    --ip-range "10.0.0.0/16" \
    --format ID --no-header)

print_status "VPC created with ID: $VPC_ID"

# Create managed database
print_status "Creating PostgreSQL database..."
DB_ID=$(doctl databases create trading-db-prod \
    --engine pg \
    --num-nodes 1 \
    --region "$REGION" \
    --size db-s-1vcpu-1gb \
    --version 15 \
    --format ID --no-header)

print_status "Database created with ID: $DB_ID"

# Create Redis cluster
print_status "Creating Redis cache..."
REDIS_ID=$(doctl databases create trading-redis-prod \
    --engine redis \
    --num-nodes 1 \
    --region "$REGION" \
    --size db-s-1vcpu-1gb \
    --version 7 \
    --format ID --no-header)

print_status "Redis created with ID: $REDIS_ID"

# Create production droplet
print_status "Creating production droplet..."
DROPLET_ID=$(doctl compute droplet create "$DROPLET_NAME" \
    --region "$REGION" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --ssh-keys "$SSH_KEY_ID" \
    --vpc-uuid "$VPC_ID" \
    --enable-monitoring \
    --enable-backups \
    --format ID --no-header)

print_status "Droplet created with ID: $DROPLET_ID"

# Wait for droplet to be ready
print_status "Waiting for droplet to be ready..."
sleep 60

# Get droplet IP
DROPLET_IP=$(doctl compute droplet get "$DROPLET_ID" --format PublicIPv4 --no-header)
print_status "Droplet IP: $DROPLET_IP"

# Create firewall rules
print_status "Creating firewall rules..."
FIREWALL_ID=$(doctl compute firewall create \
    --name "trading-firewall" \
    --inbound-rules "protocol:tcp,ports:22,sources:addresses:0.0.0.0/0 protocol:tcp,ports:80,sources:addresses:0.0.0.0/0 protocol:tcp,ports:443,sources:addresses:0.0.0.0/0" \
    --outbound-rules "protocol:tcp,ports:all,destinations:addresses:0.0.0.0/0 protocol:udp,ports:all,destinations:addresses:0.0.0.0/0" \
    --droplet-ids "$DROPLET_ID" \
    --format ID --no-header)

print_status "Firewall created with ID: $FIREWALL_ID"

# Create backup droplet
print_status "Creating backup droplet..."
BACKUP_DROPLET_ID=$(doctl compute droplet create "trading-bot-backup" \
    --region "sgp1" \
    --size "s-1vcpu-2gb" \
    --image "$IMAGE" \
    --ssh-keys "$SSH_KEY_ID" \
    --enable-monitoring \
    --format ID --no-header)

print_status "Backup droplet created with ID: $BACKUP_DROPLET_ID"

# Create load balancer
print_status "Creating load balancer..."
LB_ID=$(doctl compute load-balancer create \
    --name "trading-lb" \
    --region "$REGION" \
    --forwarding-rules "entry_protocol:https,entry_port:443,target_protocol:http,target_port:8000,certificate_id:,tls_passthrough:false entry_protocol:http,entry_port:80,target_protocol:http,target_port:8000" \
    --health-check "protocol:http,port:8000,path:/api/monitoring/health,check_interval_seconds:10,response_timeout_seconds:5,healthy_threshold:3,unhealthy_threshold:3" \
    --droplet-ids "$DROPLET_ID" \
    --format ID --no-header)

print_status "Load balancer created with ID: $LB_ID"

# Generate deployment summary
cat << EOF > deployment_summary.txt
🚀 DigitalOcean Deployment Summary
================================

Production Environment:
- Droplet ID: $DROPLET_ID
- Droplet IP: $DROPLET_IP
- Region: $REGION
- VPC ID: $VPC_ID

Database Infrastructure:
- PostgreSQL ID: $DB_ID
- Redis ID: $REDIS_ID

Security:
- Firewall ID: $FIREWALL_ID
- Load Balancer ID: $LB_ID

Backup Infrastructure:
- Backup Droplet ID: $BACKUP_DROPLET_ID

Next Steps:
1. SSH into droplet: ssh root@$DROPLET_IP
2. Run server setup: ./setup_server.sh
3. Deploy application: ./deploy_app.sh
4. Configure SSL certificates
5. Test emergency procedures

Important URLs:
- DigitalOcean Console: https://cloud.digitalocean.com/
- Droplet Console: https://cloud.digitalocean.com/droplets/$DROPLET_ID
- Database Console: https://cloud.digitalocean.com/databases/$DB_ID
EOF

print_status "Deployment summary saved to deployment_summary.txt"

print_status "✅ DigitalOcean infrastructure deployment completed!"
print_warning "⚠️  Don't forget to:"
echo "  1. Configure DNS for your domain"
echo "  2. Set up SSL certificates"
echo "  3. Configure database connections"
echo "  4. Test all security controls"
echo "  5. Verify emergency stop procedures"

echo ""
print_status "💡 Next command to run:"
echo "ssh root@$DROPLET_IP"