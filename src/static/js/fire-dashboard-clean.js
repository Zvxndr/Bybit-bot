// 🔥 Fire Cybersigilism Dashboard JavaScript
// Multi-Environment Trading Bot Control System for Personal Use

class FireDashboard {
    constructor() {
        this.isAdminPanelOpen = false;
        this.botStatus = 'active';
        this.charts = {};
        this.currentEnvironment = 'testnet';
        this.balanceData = {
            testnet: { total: 0, available: 0, used: 0, unrealized: 0, history: [] },
            mainnet: { total: 0, available: 0, used: 0, unrealized: 0, history: [] },
            paper: { total: 100000, available: 100000, used: 0, unrealized: 0, history: [] }
        };
        
        this.initializeCharts();
        this.startRealTimeUpdates();
        this.showLoadingScreen();
    }

    showLoadingScreen() {
        const loadingScreen = document.getElementById('loadingScreen');
        const mainDashboard = document.getElementById('mainDashboard');
        
        setTimeout(() => {
            loadingScreen.style.display = 'none';
            mainDashboard.style.display = 'block';
            this.playFireAnimation();
        }, 3000);
    }

    playFireAnimation() {
        const cards = document.querySelectorAll('.fire-card');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(50px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.8s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 200);
        });
    }

    initializeCharts() {
        const fireGradient = (ctx) => {
            const gradient = ctx.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, 'rgba(255, 107, 53, 0.8)');
            gradient.addColorStop(1, 'rgba(255, 0, 0, 0.1)');
            return gradient;
        };

        const balanceCtx = document.getElementById('balanceChart').getContext('2d');
        this.charts.balance = new Chart(balanceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Balance',
                    data: [],
                    borderColor: '#FF6B35',
                    backgroundColor: fireGradient(balanceCtx),
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            color: '#FFFFFF',
                            font: {
                                family: 'Orbitron',
                                weight: '600'
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#CCCCCC' },
                        grid: { color: 'rgba(255, 107, 53, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#CCCCCC' },
                        grid: { color: 'rgba(255, 107, 53, 0.1)' }
                    }
                }
            }
        });

        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        this.charts.performance = new Chart(perfCtx, {
            type: 'bar',
            data: {
                labels: ['Wins', 'Losses', 'Break Even'],
                datasets: [{
                    label: 'Trades',
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(0, 255, 65, 0.8)',
                        'rgba(255, 0, 0, 0.8)',
                        'rgba(255, 184, 77, 0.8)'
                    ],
                    borderColor: [
                        '#00FF41',
                        '#FF0000',
                        '#FFB74D'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            color: '#FFFFFF',
                            font: {
                                family: 'Orbitron',
                                weight: '600'
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#CCCCCC' },
                        grid: { color: 'rgba(0, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#CCCCCC' },
                        grid: { color: 'rgba(0, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    startRealTimeUpdates() {
        setInterval(() => {
            this.fetchMultiBalance();
            this.fetchPositions();
            this.fetchTrades();
            this.fetchSystemStats();
        }, 10000);
        
        this.fetchMultiBalance();
        this.updateEnvironmentDisplay();
    }

    async fetchMultiBalance() {
        try {
            const response = await fetch('/api/multi-balance');
            const result = await response.json();
            
            if (result.success) {
                this.updateMultiEnvironmentBalance(result.data);
            }
        } catch (error) {
            console.error('Error fetching multi-balance:', error);
        }
    }

    async fetchPositions() {
        try {
            const response = await fetch(`/api/positions/${this.currentEnvironment}`);
            const result = await response.json();
            
            if (result.success) {
                this.updatePositions(result.data);
            }
        } catch (error) {
            console.error('Error fetching positions:', error);
        }
    }

    async fetchTrades() {
        try {
            const response = await fetch(`/api/trades/${this.currentEnvironment}`);
            const result = await response.json();
            
            if (result.success) {
                this.updateTrades(result.data);
            }
        } catch (error) {
            console.error('Error fetching trades:', error);
        }
    }

    async fetchSystemStats() {
        try {
            const response = await fetch('/api/system-stats');
            const result = await response.json();
            
            if (result.success) {
                this.updateSystemStats(result.data);
            }
        } catch (error) {
            console.error('Error fetching system stats:', error);
        }
    }

    updateMultiEnvironmentBalance(data) {
        Object.keys(data).forEach(env => {
            if (this.balanceData[env]) {
                this.balanceData[env] = {
                    ...this.balanceData[env],
                    ...data[env],
                    history: this.balanceData[env].history || []
                };
                
                const now = new Date().toLocaleTimeString();
                this.balanceData[env].history.push({
                    time: now,
                    balance: data[env].total || 0
                });
                
                if (this.balanceData[env].history.length > 20) {
                    this.balanceData[env].history.shift();
                }
            }
        });
        
        this.updateEnvironmentStatusIndicators();
        this.updateCurrentEnvironmentBalance();
    }

    updateEnvironmentStatusIndicators() {
        const environments = ['testnet', 'mainnet', 'paper'];
        
        environments.forEach(env => {
            const balanceElement = document.getElementById(`${env}Balance`);
            const statusElement = document.getElementById(`${env}Status`);
            
            if (balanceElement && this.balanceData[env]) {
                const balance = this.balanceData[env].total || 0;
                balanceElement.textContent = balance.toLocaleString();
                
                if (statusElement) {
                    if (env === this.currentEnvironment) {
                        statusElement.className = 'status-fire active';
                    } else if (balance > 0) {
                        statusElement.className = 'status-fire paused';
                    } else {
                        statusElement.className = 'status-fire stopped';
                    }
                }
            }
        });
    }

    updateCurrentEnvironmentBalance() {
        const envData = this.balanceData[this.currentEnvironment];
        if (!envData) return;
        
        document.getElementById('totalBalance').textContent = `$${envData.total.toLocaleString()}`;
        document.getElementById('availableBalance').textContent = `$${envData.available.toLocaleString()}`;
        document.getElementById('usedBalance').textContent = `$${envData.used.toLocaleString()}`;
        document.getElementById('unrealizedPnl').textContent = `$${envData.unrealized.toFixed(2)}`;
        document.getElementById('currentEnvironment').textContent = this.currentEnvironment.toUpperCase();
        
        if (envData.total > 0) {
            const availablePercent = ((envData.available / envData.total) * 100).toFixed(1);
            const usedPercent = ((envData.used / envData.total) * 100).toFixed(1);
            const pnlPercent = ((envData.unrealized / envData.total) * 100).toFixed(2);
            
            document.getElementById('availablePercent').textContent = `${availablePercent}%`;
            document.getElementById('usedPercent').textContent = `${usedPercent}%`;
            document.getElementById('pnlPercent').textContent = `${pnlPercent}%`;
            
            const pnlElement = document.getElementById('unrealizedPnl');
            const pnlPercentElement = document.getElementById('pnlPercent');
            if (envData.unrealized >= 0) {
                pnlElement.className = 'cyber-text';
                pnlPercentElement.className = 'cyber-text';
            } else {
                pnlElement.className = 'fire-text';
                pnlPercentElement.className = 'fire-text';
            }
        }
        
        this.updateBalanceChart();
    }

    updateBalanceChart() {
        const envData = this.balanceData[this.currentEnvironment];
        if (!envData || !envData.history) return;
        
        const labels = envData.history.map(h => h.time);
        const data = envData.history.map(h => h.balance);
        
        this.charts.balance.data.labels = labels;
        this.charts.balance.data.datasets[0].data = data;
        this.charts.balance.data.datasets[0].label = `${this.currentEnvironment.toUpperCase()} Balance`;
        
        const colors = {
            testnet: { border: '#FF6B35', bg: 'rgba(255, 107, 53, 0.2)' },
            mainnet: { border: '#FF0000', bg: 'rgba(255, 0, 0, 0.2)' },
            paper: { border: '#00FFFF', bg: 'rgba(0, 255, 255, 0.2)' }
        };
        
        const envColors = colors[this.currentEnvironment] || colors.testnet;
        this.charts.balance.data.datasets[0].borderColor = envColors.border;
        this.charts.balance.data.datasets[0].backgroundColor = envColors.bg;
        
        this.charts.balance.update('none');
    }

    updateEnvironmentDisplay() {
        document.querySelectorAll('.environment-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.env === this.currentEnvironment) {
                btn.classList.add('active');
            }
        });
        
        this.updateCurrentEnvironmentBalance();
    }

    updatePositions(positions) {
        const container = document.getElementById('positionsContainer');
        
        if (positions.length === 0) {
            container.innerHTML = `
                <div class="cyber-text" style="text-align: center; padding: 20px;">
                    <i class="fas fa-info-circle"></i> No active positions
                </div>
            `;
            return;
        }

        let html = '';
        positions.forEach(position => {
            const pnlClass = position.unrealizedPnl >= 0 ? 'cyber-text' : 'fire-text';
            const sideClass = position.side === 'buy' ? 'cyber-text' : 'fire-text';
            
            html += `
                <div class="position-card" style="border: 1px solid rgba(255, 107, 53, 0.3); border-radius: 8px; padding: 16px; margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 class="fire-text">${position.symbol}</h4>
                            <p class="${sideClass}">${position.side.toUpperCase()} ${position.size}</p>
                        </div>
                        <div style="text-align: right;">
                            <p class="cyber-text">Entry: $${position.entryPrice}</p>
                            <p class="${pnlClass}">PnL: $${position.unrealizedPnl.toFixed(2)}</p>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }

    updateTrades(trades) {
        const tbody = document.getElementById('tradesTableBody');
        
        if (trades.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="cyber-text" style="padding: 20px; text-align: center;">
                        No recent trades
                    </td>
                </tr>
            `;
            return;
        }

        let html = '';
        trades.slice(0, 10).forEach(trade => {
            const sideClass = trade.side === 'buy' ? 'cyber-text' : 'fire-text';
            const pnlClass = trade.pnl >= 0 ? 'cyber-text' : 'fire-text';
            
            html += `
                <tr style="border-bottom: 1px solid rgba(255, 107, 53, 0.1);">
                    <td class="fire-text" style="padding: 8px;">${new Date(trade.timestamp).toLocaleTimeString()}</td>
                    <td class="fire-text" style="padding: 8px;">${trade.symbol}</td>
                    <td class="${sideClass}" style="padding: 8px;">${trade.side.toUpperCase()}</td>
                    <td class="cyber-text" style="padding: 8px;">${trade.size}</td>
                    <td class="cyber-text" style="padding: 8px;">$${trade.price}</td>
                    <td class="${pnlClass}" style="padding: 8px;">$${trade.pnl.toFixed(2)}</td>
                </tr>
            `;
        });
        
        tbody.innerHTML = html;
    }

    updateSystemStats(stats) {
        document.getElementById('cpuUsage').textContent = `${stats.cpu}%`;
        document.getElementById('memoryUsage').textContent = `${stats.memory}%`;
        document.getElementById('apiStatus').textContent = stats.apiStatus;
        
        if (stats.winRate) {
            document.getElementById('winRate').textContent = `${stats.winRate}%`;
        }
        if (stats.profitFactor) {
            document.getElementById('profitFactor').textContent = stats.profitFactor;
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `fire-toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center;">
                <i class="fas fa-${this.getToastIcon(type)}" style="margin-right: 12px;"></i>
                <span>${message}</span>
            </div>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => toast.classList.add('show'), 100);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => container.removeChild(toast), 300);
        }, 5000);
    }

    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Environment switching function
async function switchEnvironment(environment) {
    if (dashboard.currentEnvironment === environment) return;
    
    try {
        const response = await fetch('/api/environment/switch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ environment })
        });
        
        const result = await response.json();
        
        if (result.success) {
            dashboard.currentEnvironment = environment;
            dashboard.updateEnvironmentDisplay();
            dashboard.showToast(`🔥 Switched to ${environment.toUpperCase()} environment`, 'success');
            
            dashboard.fetchPositions();
            dashboard.fetchTrades();
        } else {
            dashboard.showToast(`❌ Failed to switch environment: ${result.error}`, 'error');
        }
    } catch (error) {
        dashboard.showToast('❌ Network error switching environment', 'error');
    }
}

// Admin Panel Functions
function toggleAdminPanel() {
    const adminPanel = document.getElementById('adminPanel');
    const isHidden = adminPanel.style.display === 'none';
    
    adminPanel.style.display = isHidden ? 'block' : 'none';
    
    if (isHidden) {
        adminPanel.style.opacity = '0';
        adminPanel.style.transform = 'translateY(-20px)';
        
        setTimeout(() => {
            adminPanel.style.transition = 'all 0.5s ease';
            adminPanel.style.opacity = '1';
            adminPanel.style.transform = 'translateY(0)';
        }, 50);
    }
}

function closeAdminPanel() {
    document.getElementById('adminPanel').style.display = 'none';
}

// Coming Soon placeholders for PTY LTD and Trust Level features
function showComingSoon(feature) {
    dashboard.showToast(`🔥 ${feature} - Coming Soon for PTY LTD customers`, 'info');
}

// Mock admin functions for personal use
function pauseBot() {
    dashboard.showToast('🔥 Trading paused (Personal Use Mode)', 'success');
}

function resumeBot() {
    dashboard.showToast('⚡ Trading resumed (Personal Use Mode)', 'success');
}

function emergencyStop() {
    if (confirm('🚨 Emergency stop for personal trading bot?')) {
        dashboard.showToast('🛑 Emergency stop activated', 'warning');
    }
}

function clearAllData() {
    const confirmation = prompt('Type "CLEAR DATA" to confirm:');
    if (confirmation === 'CLEAR DATA') {
        dashboard.showToast('🔥 Data cleared (Personal Use Mode)', 'success');
    }
}

function exportData() {
    dashboard.showToast('🔥 Export feature - Coming Soon', 'info');
}

function updateCredentials() {
    dashboard.showToast('🔥 Credential update - Use environment variables', 'info');
}

function verifyMFA() {
    showComingSoon('Multi-Factor Authentication');
}

// Initialize dashboard when DOM is loaded
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new FireDashboard();
});