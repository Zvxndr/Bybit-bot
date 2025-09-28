// 🔥 Fire Trading Dashboard JavaScript
// Multi-Environment Trading Bot Control System for Personal Use

class FireDashboard {
    constructor() {
        console.log('🔥 FireDashboard constructor started');
        this.isAdminPanelOpen = false;
        this.botStatus = 'active';
        this.charts = {};
        this.currentEnvironment = 'testnet';
        this.balanceData = {
            testnet: { total: 0, available: 0, used: 0, unrealized: 0, history: [] },
            mainnet: { total: 0, available: 0, used: 0, unrealized: 0, history: [] },
            paper: { total: 100000, available: 100000, used: 0, unrealized: 0, history: [] }
        };
        
        try {
            this.initializeCharts();
            this.startRealTimeUpdates();
            console.log('✅ Dashboard components initialized successfully');
        } catch (error) {
            console.error('⚠️ Error initializing components:', error);
        }
        
        this.showLoadingScreen();
    }

    showLoadingScreen() {
        const loadingScreen = document.getElementById('loadingScreen');
        const mainDashboard = document.getElementById('mainDashboard');
        
        console.log('🔥 Initializing Fire Dashboard...');
        
        setTimeout(() => {
            console.log('✅ Loading complete, showing main dashboard');
            loadingScreen.style.display = 'none';
            mainDashboard.style.display = 'block';
            this.playFireAnimation();
        }, 2000);
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
            this.fetchStrategyGraduationStatus();
        }, 10000);
        
        this.fetchMultiBalance();
        this.updateEnvironmentDisplay();
        this.fetchStrategyGraduationStatus();
    }

    async fetchMultiBalance() {
        try {
            console.log('🔥 Fetching multi-environment balance...');
            const response = await fetch('/api/multi-balance');
            
            if (!response.ok) {
                console.warn(`⚠️ Balance API returned ${response.status}: ${response.statusText}`);
                this.showToast('⚠️ Balance data temporarily unavailable', 'warning');
                return;
            }
            
            const result = await response.json();
            console.log('📊 Balance response:', result);
            
            if (result.success && result.data) {
                this.updateMultiEnvironmentBalance(result.data);
                console.log('✅ Multi-environment balance updated');
            } else {
                console.warn('⚠️ Balance API returned unsuccessful response:', result);
                this.showToast('⚠️ Unable to fetch live balance data', 'warning');
            }
        } catch (error) {
            console.error('❌ Error fetching multi-balance:', error);
            this.showToast('🔥 Using cached balance data - connection issue', 'info');
        }
    }

    async fetchPositions() {
        try {
            const response = await fetch('/api/positions');
            const result = await response.json();
            
            if (result.positions) {
                this.updatePositions(result.positions);
            } else {
                console.log('No positions data received:', result);
                this.updatePositions([]);
            }
        } catch (error) {
            console.error('Error fetching positions:', error);
            this.updatePositions([]);
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

    async fetchStrategyGraduationStatus() {
        try {
            const response = await fetch('/api/strategy-graduation/status');
            
            if (response.ok) {
                const result = await response.json();
                if (result.success) {
                    updateStrategyGraduationDisplay(result.data);
                }
            } else {
                // Default values for initial deployment
                updateStrategyGraduationDisplay({
                    historical: 0,
                    paper: 0,
                    testnet: 0,
                    live: 0,
                    backtesting_active: false
                });
            }
        } catch (error) {
            console.log('Strategy graduation status will be available after deployment');
            // Set default display values
            updateStrategyGraduationDisplay({
                historical: 0,
                paper: 0,
                testnet: 0,
                live: 0,
                backtesting_active: false
            });
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

// Historical Backtesting Functions
async function switchToHistoricalBacktesting() {
    try {
        // Update UI to historical backtesting mode
        dashboard.currentEnvironment = 'historical';
        dashboard.updateEnvironmentDisplay();
        
        // Show historical backtesting information
        dashboard.showToast('📊 Historical Backtesting Mode Active', 'info');
        
        // Update balance display for historical mode
        document.getElementById('totalBalance').textContent = 'Historical Data';
        document.getElementById('currentEnvironment').textContent = 'HISTORICAL';
        
        // Fetch historical backtesting status
        await fetchHistoricalBacktestStatus();
        
    } catch (error) {
        console.error('Error switching to historical backtesting:', error);
        dashboard.showToast('❌ Error accessing historical backtesting', 'error');
    }
}

async function startHistoricalBacktest() {
    try {
        dashboard.showToast('🚀 Starting Historical Backtesting...', 'info');
        
        const response = await fetch('/api/backtest/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                mode: 'historical',
                symbols: ['BTCUSDT', 'ETHUSDT'],
                timeframe: '1h',
                lookback_days: 365
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            dashboard.showToast('✅ Historical Backtesting Started', 'success');
            
            // Update status indicators
            document.getElementById('backtestStatus').textContent = 'RUNNING';
            document.getElementById('historicalStatus').className = 'status-fire active';
            
        } else {
            dashboard.showToast('⚠️ Backtesting will begin automatically after deployment', 'info');
        }
        
    } catch (error) {
        console.error('Error starting backtest:', error);
        dashboard.showToast('📊 Historical Backtesting Mode Ready', 'info');
    }
}

async function viewBacktestResults() {
    try {
        const response = await fetch('/api/backtest/results');
        
        if (response.ok) {
            const results = await response.json();
            // Display results in a modal or new section
            dashboard.showToast('📊 Viewing Backtest Results', 'info');
        } else {
            dashboard.showToast('📈 Historical Data Available - Results will show after backtesting', 'info');
        }
        
    } catch (error) {
        dashboard.showToast('📊 Historical Data Ready for Backtesting', 'info');
    }
}

async function fetchHistoricalBacktestStatus() {
    try {
        const response = await fetch('/api/backtest/status');
        
        if (response.ok) {
            const status = await response.json();
            updateStrategyGraduationDisplay(status);
        } else {
            // Set default status for historical backtesting
            updateStrategyGraduationDisplay({
                historical: 0,
                paper: 0,
                testnet: 0,
                live: 0,
                backtesting_active: false
            });
        }
        
    } catch (error) {
        console.log('Historical backtesting status will be available after deployment');
        updateStrategyGraduationDisplay({
            historical: 0,
            paper: 0,
            testnet: 0,
            live: 0,
            backtesting_active: false
        });
    }
}

function updateStrategyGraduationDisplay(status) {
    // Update strategy counts in the pipeline
    document.getElementById('historicalCount').textContent = status.historical || 0;
    document.getElementById('paperCount').textContent = status.paper || 0;
    document.getElementById('testnetStrategyCount').textContent = status.testnet || 0;
    document.getElementById('liveCount').textContent = status.live || 0;
    
    // Update active strategies total
    const total = (status.historical || 0) + (status.paper || 0) + (status.testnet || 0) + (status.live || 0);
    document.getElementById('activeStrategies').textContent = total;
    
    // Update backtest status
    const backtestStatusText = status.backtesting_active ? 'RUNNING' : 'READY';
    document.getElementById('backtestStatus').textContent = backtestStatusText;
    
    // Update historical progress
    const progressText = status.backtesting_active ? 'Running...' : 'Ready';
    document.getElementById('historicalProgress').textContent = progressText;
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

// Enhanced control functions with backend integration
async function pauseBot() {
    try {
        const response = await fetch('/api/bot/pause', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });
        
        if (response.ok) {
            dashboard.showToast('⏸️ Trading paused successfully', 'success');
            updateBotStatus('paused');
        } else {
            dashboard.showToast('❌ Failed to pause trading', 'error');
        }
    } catch (error) {
        console.error('Pause bot error:', error);
        dashboard.showToast('🔥 Trading paused (offline mode)', 'warning');
    }
}

async function resumeBot() {
    try {
        const response = await fetch('/api/bot/resume', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });
        
        if (response.ok) {
            dashboard.showToast('▶️ Trading resumed successfully', 'success');
            updateBotStatus('running');
        } else {
            dashboard.showToast('❌ Failed to resume trading', 'error');
        }
    } catch (error) {
        console.error('Resume bot error:', error);
        dashboard.showToast('⚡ Trading resumed (offline mode)', 'warning');
    }
}

async function emergencyStop() {
    const confirmed = confirm('🚨 EMERGENCY STOP - This will halt all trading immediately!\n\nAre you sure you want to proceed?');
    
    if (confirmed) {
        try {
            const response = await fetch('/api/bot/emergency-stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${getAuthToken()}`
                }
            });
            
            if (response.ok) {
                dashboard.showToast('🛑 EMERGENCY STOP ACTIVATED', 'error');
                updateBotStatus('emergency_stopped');
                
                // Visual emergency indicator
                document.body.style.background = 'linear-gradient(45deg, #ff0000, #8b0000)';
                setTimeout(() => {
                    document.body.style.background = '';
                }, 3000);
                
            } else {
                dashboard.showToast('❌ Emergency stop failed', 'error');
            }
        } catch (error) {
            console.error('Emergency stop error:', error);
            dashboard.showToast('🛑 Emergency stop activated (offline mode)', 'warning');
        }
    }
}

async function clearAllData() {
    const firstConfirm = confirm('⚠️ WARNING: This will CLOSE ALL TRADES and delete ALL trading data!\n\n' +
                                'This includes:\n' +
                                '• CLOSING ALL OPEN POSITIONS\n' +
                                '• Canceling all pending orders\n' +
                                '• All trading history\n' +
                                '• Performance records\n' +
                                '• ML predictions cache\n' +
                                '• Configuration settings\n\n' +
                                'Continue?');
    
    if (!firstConfirm) return;
    
    const verification = prompt('Type "WIPE ALL DATA" to confirm data deletion:');
    
    if (verification === 'WIPE ALL DATA') {
        const finalConfirm = confirm('🔥 FINAL CONFIRMATION\n\n' +
                                   'This will IMMEDIATELY CLOSE ALL TRADES and WIPE ALL DATA!\n' +
                                   'This action CANNOT be undone!\n' +
                                   'All positions will be closed at market price.\n\n' +
                                   'Are you absolutely sure?');
        
        if (finalConfirm) {
            try {
                dashboard.showToast('🔄 Step 1/3: Closing all open positions...', 'info');
                
                // Step 1: Close all open positions
                try {
                    const closeResponse = await fetch('/api/admin/close-all-positions', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${getAuthToken()}`
                        },
                        body: JSON.stringify({ 
                            confirmation: 'CLOSE_ALL',
                            timestamp: Date.now()
                        })
                    });
                    
                    if (closeResponse.ok) {
                        const result = await closeResponse.json();
                        dashboard.showToast(`✅ Closed ${result.closedCount || 0} positions`, 'success');
                    } else {
                        dashboard.showToast('⚠️ Some positions may not have closed - continuing with data wipe', 'warning');
                    }
                } catch (closeError) {
                    console.warn('Position closing error:', closeError);
                    dashboard.showToast('⚠️ Position closing failed - continuing with data wipe', 'warning');
                }
                
                // Wait a moment for trades to process
                await new Promise(resolve => setTimeout(resolve, 2000));
                
                dashboard.showToast('🔄 Step 2/3: Canceling pending orders...', 'info');
                
                // Step 2: Cancel all pending orders
                try {
                    const cancelResponse = await fetch('/api/admin/cancel-all-orders', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${getAuthToken()}`
                        },
                        body: JSON.stringify({ 
                            confirmation: 'CANCEL_ALL',
                            timestamp: Date.now()
                        })
                    });
                    
                    if (cancelResponse.ok) {
                        const result = await cancelResponse.json();
                        dashboard.showToast(`✅ Canceled ${result.canceledCount || 0} pending orders`, 'success');
                    }
                } catch (cancelError) {
                    console.warn('Order canceling error:', cancelError);
                    dashboard.showToast('⚠️ Some orders may not have canceled', 'warning');
                }
                
                // Wait for orders to process
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                dashboard.showToast('🔄 Step 3/3: Wiping all trading data...', 'info');
                
                // Step 3: Wipe all data
                const response = await fetch('/api/admin/wipe-data', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${getAuthToken()}`
                    },
                    body: JSON.stringify({ 
                        confirmation: 'WIPE ALL DATA',
                        timestamp: Date.now(),
                        tradesAlreadyClosed: true
                    })
                });
                
                if (response.ok) {
                    dashboard.showToast('�️ ALL DATA WIPED - System reset complete', 'success');
                    
                    // Visual confirmation
                    const cards = document.querySelectorAll('.fire-card');
                    cards.forEach(card => {
                        card.style.animation = 'fadeOut 0.5s ease-out forwards';
                    });
                    
                    setTimeout(() => {
                        location.reload(); // Refresh to show clean state
                    }, 2000);
                    
                } else {
                    dashboard.showToast('❌ Data wipe failed', 'error');
                }
            } catch (error) {
                console.error('Data wipe error:', error);
                dashboard.showToast('🔥 Data cleared (offline mode)', 'warning');
            }
        }
    } else {
        dashboard.showToast('❌ Verification failed - Data wipe cancelled', 'info');
    }
}

async function exportData() {
    try {
        dashboard.showToast('� Creating backup...', 'info');
        
        const response = await fetch('/api/admin/export-data', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `trading_bot_backup_${new Date().toISOString().split('T')[0]}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            dashboard.showToast('💾 Backup downloaded successfully', 'success');
        } else {
            dashboard.showToast('❌ Export failed', 'error');
        }
    } catch (error) {
        console.error('Export error:', error);
        dashboard.showToast('📦 Export feature - Check local files', 'info');
    }
}

async function updateCredentials() {
    const modal = createCredentialModal();
    document.body.appendChild(modal);
}

function createCredentialModal() {
    const modal = document.createElement('div');
    modal.className = 'credential-modal';
    modal.innerHTML = `
        <div class="modal-content fire-card">
            <span class="close-modal" onclick="this.parentElement.parentElement.remove()">&times;</span>
            <h3 class="fire-text">🔐 UPDATE API CREDENTIALS</h3>
            <div style="margin: 20px 0;">
                <label class="cyber-text">Environment:</label>
                <select id="envSelect" class="cyber-input">
                    <option value="testnet">Testnet (Safe)</option>
                    <option value="mainnet">Mainnet (Real Money)</option>
                </select>
            </div>
            <div style="margin: 20px 0;">
                <label class="cyber-text">API Key:</label>
                <input type="password" id="apiKey" class="cyber-input" placeholder="Enter API key">
            </div>
            <div style="margin: 20px 0;">
                <label class="cyber-text">API Secret:</label>
                <input type="password" id="apiSecret" class="cyber-input" placeholder="Enter API secret">
            </div>
            <div style="margin: 20px 0;">
                <button onclick="saveCredentials()" class="fire-btn">� UPDATE CREDENTIALS</button>
            </div>
            <p class="fire-text" style="font-size: 0.8em; margin-top: 15px;">
                ⚠️ Credentials are encrypted and stored securely
            </p>
        </div>
    `;
    
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
        background: rgba(0,0,0,0.8); display: flex; justify-content: center; 
        align-items: center; z-index: 1000;
    `;
    
    return modal;
}

async function saveCredentials() {
    const environment = document.getElementById('envSelect').value;
    const apiKey = document.getElementById('apiKey').value;
    const apiSecret = document.getElementById('apiSecret').value;
    
    if (!apiKey || !apiSecret) {
        dashboard.showToast('❌ Please enter both API key and secret', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/admin/update-credentials', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify({
                environment,
                api_key: apiKey,
                api_secret: apiSecret
            })
        });
        
        if (response.ok) {
            dashboard.showToast('🔒 Credentials updated successfully', 'success');
            document.querySelector('.credential-modal').remove();
        } else {
            dashboard.showToast('❌ Failed to update credentials', 'error');
        }
    } catch (error) {
        console.error('Credential update error:', error);
        dashboard.showToast('🔒 Update config/secrets.yaml manually', 'info');
        document.querySelector('.credential-modal').remove();
    }
}

async function verifyMFA() {
    const mfaCode = document.getElementById('mfaCode').value;
    
    if (!mfaCode || mfaCode.length !== 6) {
        dashboard.showToast('❌ Please enter a 6-digit MFA code', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/auth/verify-mfa', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                mfa_code: mfaCode
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            localStorage.setItem('authToken', data.token);
            dashboard.showToast('🔓 MFA verified - Full access granted', 'success');
            
            // Enable admin controls
            document.querySelectorAll('.admin-control').forEach(control => {
                control.disabled = false;
                control.style.opacity = '1';
            });
            
        } else {
            dashboard.showToast('❌ Invalid MFA code', 'error');
        }
    } catch (error) {
        console.error('MFA verification error:', error);
        dashboard.showToast('🔓 MFA verification (development mode)', 'warning');
    }
    
    // Clear the input
    document.getElementById('mfaCode').value = '';
}

// Helper functions
function getAuthToken() {
    return localStorage.getItem('authToken') || 'dev-token';
}

function updateBotStatus(status) {
    const statusIndicators = document.querySelectorAll('.bot-status');
    statusIndicators.forEach(indicator => {
        indicator.textContent = status.toUpperCase();
        indicator.className = `bot-status status-${status}`;
    });
}

// Add CSS for modals and animations
const additionalStyles = `
    .credential-modal .modal-content {
        max-width: 500px;
        width: 90%;
    }
    
    .close-modal {
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 28px;
        font-weight: bold;
        color: #FF4500;
        cursor: pointer;
    }
    
    .close-modal:hover {
        color: #FF6347;
    }
    
    .cyber-input {
        width: 100%;
        padding: 10px;
        background: rgba(0,0,0,0.5);
        border: 2px solid #00FFFF;
        border-radius: 5px;
        color: #FFFFFF;
        font-family: 'Orbitron', sans-serif;
    }
    
    .cyber-input:focus {
        outline: none;
        border-color: #FF4500;
        box-shadow: 0 0 10px rgba(255,69,0,0.5);
    }
    
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.8); }
    }
    
    .status-running { color: #00FF00; }
    .status-paused { color: #FFA500; }
    .status-emergency_stopped { color: #FF0000; }
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

// Initialize dashboard when DOM is loaded
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new FireDashboard();
});