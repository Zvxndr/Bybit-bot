#!/bin/bash
# Emergency procedures script for live trading
# Use in case of system compromise or trading emergencies

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_emergency() {
    echo -e "${RED}[EMERGENCY]${NC} $1"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Emergency contact information
ALERT_EMAIL="your-emergency@domain.com"
SLACK_WEBHOOK="your-emergency-slack-webhook"
PHONE_NUMBER="+61-your-emergency-number"

log_emergency() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - EMERGENCY: $1" >> /opt/trading/logs/emergency.log
}

send_emergency_alert() {
    local message="$1"
    log_emergency "$message"
    
    # Send email alert
    echo "$message" | mail -s "TRADING EMERGENCY - IMMEDIATE ACTION REQUIRED" "$ALERT_EMAIL"
    
    # Send Slack alert if webhook configured
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 TRADING EMERGENCY: $message\"}" \
            "$SLACK_WEBHOOK"
    fi
    
    # Log to system journal
    logger -p user.crit "TRADING_BOT_EMERGENCY: $message"
}

# Emergency procedure functions
emergency_stop_trading() {
    print_emergency "INITIATING EMERGENCY STOP - ALL TRADING ACTIVITIES"
    
    # Call emergency stop API
    curl -X POST http://localhost:8000/api/emergency-stop \
        -H "Content-Type: application/json" \
        -d '{"reason": "Manual emergency stop", "operator": "'$(whoami)'"}' || true
    
    # Stop Docker containers
    docker-compose -f /opt/trading/docker-compose.prod.yml stop trading-app || true
    
    send_emergency_alert "Trading stopped by $(whoami) at $(date)"
    print_info "Emergency stop completed"
}

lockdown_system() {
    print_emergency "INITIATING SYSTEM LOCKDOWN"
    
    # Block all incoming connections except SSH
    ufw --force reset
    ufw default deny incoming
    ufw default deny outgoing
    ufw allow out 53     # DNS
    ufw allow out 2222   # SSH
    ufw allow in 2222    # SSH
    ufw --force enable
    
    # Stop all trading services
    docker-compose -f /opt/trading/docker-compose.prod.yml down || true
    
    send_emergency_alert "System locked down - all services stopped"
    print_info "System lockdown completed"
}

rotate_api_keys() {
    print_emergency "API KEY ROTATION REQUIRED"
    
    # Stop trading to prevent API calls with compromised keys
    emergency_stop_trading
    
    # Clear environment variables (keys need manual rotation)
    print_warning "Manual API key rotation required:"
    print_info "1. Log into Bybit account"
    print_info "2. Delete current API keys"
    print_info "3. Generate new API keys"
    print_info "4. Update .env.production file"
    print_info "5. Restart services with: docker-compose -f docker-compose.prod.yml up -d"
    
    send_emergency_alert "API key rotation initiated - manual steps required"
}

backup_critical_data() {
    print_emergency "BACKING UP CRITICAL DATA"
    
    BACKUP_DIR="/opt/trading/emergency_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup databases
    pg_dump "$DATABASE_URL" > "$BACKUP_DIR/database_backup.sql" 2>/dev/null || true
    redis-cli --rdb "$BACKUP_DIR/redis_backup.rdb" 2>/dev/null || true
    
    # Backup configuration and logs
    cp -r /opt/trading/config "$BACKUP_DIR/"
    cp -r /opt/trading/data "$BACKUP_DIR/"
    cp /opt/trading/logs/*.log "$BACKUP_DIR/" 2>/dev/null || true
    
    # Compress backup
    tar -czf "$BACKUP_DIR.tar.gz" -C "$(dirname $BACKUP_DIR)" "$(basename $BACKUP_DIR)"
    rm -rf "$BACKUP_DIR"
    
    print_info "Emergency backup created: $BACKUP_DIR.tar.gz"
    send_emergency_alert "Emergency backup completed: $BACKUP_DIR.tar.gz"
}

check_system_compromise() {
    print_emergency "CHECKING FOR SYSTEM COMPROMISE"
    
    # Check for suspicious processes
    SUSPICIOUS_PROCESSES=$(ps aux | grep -E "(bitcoin|mining|crypto)" | grep -v grep || true)
    if [ ! -z "$SUSPICIOUS_PROCESSES" ]; then
        send_emergency_alert "Suspicious processes detected: $SUSPICIOUS_PROCESSES"
    fi
    
    # Check network connections
    EXTERNAL_CONNECTIONS=$(netstat -an | grep ESTABLISHED | grep -v "127.0.0.1\|10.0.0" | wc -l)
    if [ "$EXTERNAL_CONNECTIONS" -gt 10 ]; then
        send_emergency_alert "High number of external connections: $EXTERNAL_CONNECTIONS"
    fi
    
    # Check system logs for intrusion attempts
    FAILED_LOGINS=$(grep "Failed password" /var/log/auth.log | tail -10 | wc -l)
    if [ "$FAILED_LOGINS" -gt 5 ]; then
        send_emergency_alert "Multiple failed login attempts detected"
    fi
    
    # Check file integrity
    find /opt/trading -name "*.py" -newer /opt/trading/.deploy_timestamp 2>/dev/null | head -5
    
    print_info "System compromise check completed"
}

restore_from_backup() {
    print_emergency "RESTORING FROM BACKUP"
    
    if [ -z "$1" ]; then
        print_warning "Usage: $0 restore_from_backup <backup_file>"
        ls -la /opt/trading/emergency_backup_*.tar.gz 2>/dev/null || print_info "No emergency backups found"
        return 1
    fi
    
    BACKUP_FILE="$1"
    if [ ! -f "$BACKUP_FILE" ]; then
        print_warning "Backup file not found: $BACKUP_FILE"
        return 1
    fi
    
    # Stop services
    docker-compose -f /opt/trading/docker-compose.prod.yml down
    
    # Extract backup
    RESTORE_DIR="/tmp/restore_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESTORE_DIR"
    tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"
    
    # Restore files (manual confirmation required)
    print_warning "Manual restore confirmation required for:"
    print_info "- Database: $RESTORE_DIR/*/database_backup.sql"
    print_info "- Redis: $RESTORE_DIR/*/redis_backup.rdb"
    print_info "- Config: $RESTORE_DIR/*/config/"
    print_info "- Data: $RESTORE_DIR/*/data/"
    
    send_emergency_alert "Backup restoration prepared in $RESTORE_DIR"
}

# Main emergency menu
show_emergency_menu() {
    clear
    echo "🚨 EMERGENCY PROCEDURES MENU 🚨"
    echo "================================"
    echo "1. Emergency Stop Trading"
    echo "2. System Lockdown"
    echo "3. Rotate API Keys"
    echo "4. Backup Critical Data"
    echo "5. Check System Compromise"
    echo "6. Restore from Backup"
    echo "7. View Emergency Logs"
    echo "8. System Status"
    echo "9. Exit"
    echo ""
}

# Execute based on command line argument or interactive menu
case "${1:-menu}" in
    "emergency_stop")
        emergency_stop_trading
        ;;
    "lockdown")
        lockdown_system
        ;;
    "rotate_keys")
        rotate_api_keys
        ;;
    "backup")
        backup_critical_data
        ;;
    "check_compromise")
        check_system_compromise
        ;;
    "restore_from_backup")
        restore_from_backup "$2"
        ;;
    "logs")
        tail -50 /opt/trading/logs/emergency.log 2>/dev/null || echo "No emergency logs found"
        ;;
    "status")
        print_info "System Status Check:"
        docker-compose -f /opt/trading/docker-compose.prod.yml ps
        systemctl status trading-bot || true
        ufw status
        ;;
    "menu")
        while true; do
            show_emergency_menu
            read -p "Select emergency procedure (1-9): " choice
            case $choice in
                1) emergency_stop_trading ;;
                2) lockdown_system ;;
                3) rotate_api_keys ;;
                4) backup_critical_data ;;
                5) check_system_compromise ;;
                6) 
                    echo "Available backups:"
                    ls -la /opt/trading/emergency_backup_*.tar.gz 2>/dev/null || echo "No backups found"
                    read -p "Enter backup file path: " backup_file
                    restore_from_backup "$backup_file"
                    ;;
                7) tail -50 /opt/trading/logs/emergency.log 2>/dev/null || echo "No emergency logs found" ;;
                8) 
                    print_info "System Status:"
                    docker ps
                    systemctl status trading-bot || true
                    ;;
                9) exit 0 ;;
                *) print_warning "Invalid option" ;;
            esac
            echo ""
            read -p "Press Enter to continue..."
        done
        ;;
    *)
        echo "Usage: $0 {emergency_stop|lockdown|rotate_keys|backup|check_compromise|restore_from_backup|logs|status|menu}"
        exit 1
        ;;
esac