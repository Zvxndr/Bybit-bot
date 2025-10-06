/**
 * Enhanced Trading Dashboard JavaScript
 * ===================================
 * 
 * Full integration with backend API endpoints:
 * - Real-time portfolio data from Bybit testnet
 * - Dynamic risk management metrics
 * - Live position monitoring
 * - Market data updates
 * - System status monitoring
 */

class TradingDashboard {
    constructor() {
        this.updateInterval = 5000; // 5 seconds
        this.intervalId = null;
        this.isUpdating = false;
        this.apiBaseUrl = window.location.origin;
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing Trading Dashboard...');
        this.setupEventListeners();
        this.startAutoUpdate();
        this.loadInitialData();
    }
    
    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.refreshAllData();
            });
        }
        
        // Handle page visibility for performance
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.stopAutoUpdate();
            } else {
                this.startAutoUpdate();
            }
        });
        
        // Risk calculator input if exists
        const riskCalculatorInput = document.getElementById('risk-calculator-input');
        if (riskCalculatorInput) {
            riskCalculatorInput.addEventListener('input', (e) => {
                const balance = parseFloat(e.target.value);
                if (balance > 0) {
                    this.calculateRiskForBalance(balance);
                }
            });
        }
        
        console.log('✅ Event listeners setup complete');
    }
    
    async loadInitialData() {
        console.log('📊 Loading initial dashboard data...');
        await this.loadDashboardData();
        await this.loadSystemStatus();
    }
    
    startAutoUpdate() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
        }
        
        this.intervalId = setInterval(() => {
            if (!this.isUpdating) {
                this.loadDashboardData();
            }
        }, this.updateInterval);
        
        console.log(`🔄 Auto-update started (${this.updateInterval/1000}s interval)`);
    }
    
    stopAutoUpdate() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        console.log('⏸️ Auto-update stopped');
    }
    
    async refreshAllData() {
        console.log('🔄 Manual refresh triggered');
        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.classList.add('btn-loading');
        }
        
        try {
            await Promise.all([
                this.loadDashboardData(),
                this.loadSystemStatus(),
                this.loadPortfolioData(),
                this.loadRiskMetrics()
            ]);
        } finally {
            if (refreshBtn) {
                refreshBtn.classList.remove('btn-loading');
            }
        }
    }
    
    async loadDashboardData() {
        if (this.isUpdating) return;
        this.isUpdating = true;
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/dashboard`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('📊 Dashboard data received:', Object.keys(data));
            
            this.updatePortfolioDisplay(data.portfolio);
            this.updateRiskMetricsDisplay(data.risk_metrics);
            this.updateMarketDataDisplay(data.market_data);
            this.updateSystemStatusDisplay(data.system_status);
            
        } catch (error) {
            console.error('❌ Failed to load dashboard data:', error);
            this.showErrorState('Failed to load dashboard data');
        } finally {
            this.isUpdating = false;
        }
    }
    
    updatePortfolioDisplay(portfolio) {
        try {
            // Portfolio value
            const portfolioValueEl = document.getElementById('portfolio-value');
            if (portfolioValueEl && portfolio.total_balance !== undefined) {
                const balance = parseFloat(portfolio.total_balance);
                portfolioValueEl.textContent = `$${balance.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            }
            
            // Environment indicator
            this.updateEnvironmentBadge(portfolio.environment, portfolio.api_connected);
            
            // P&L
            const pnlEl = document.getElementById('today-pnl');
            if (pnlEl && portfolio.unrealized_pnl !== undefined) {
                const pnl = parseFloat(portfolio.unrealized_pnl);
                pnlEl.textContent = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`;
                pnlEl.className = `h1 mb-3 ${pnl >= 0 ? 'text-success' : 'text-danger'}`;
            }
            
            // Positions count
            const positionsEl = document.getElementById('active-strategies');
            if (positionsEl && portfolio.positions_count !== undefined) {
                positionsEl.textContent = portfolio.positions_count;
            }
            
            // Dynamic risk metrics if available
            if (portfolio.risk_metrics) {
                this.updateRiskMetricsDisplay(portfolio.risk_metrics);
            }
            
            // Update timestamp
            this.updateTimestamp(portfolio.last_updated);
            
            console.log('✅ Portfolio display updated');
            
        } catch (error) {
            console.error('❌ Error updating portfolio display:', error);
        }
    }
    
    updateRiskMetricsDisplay(riskMetrics) {
        try {
            // Dynamic risk tier badge
            const riskTierEl = document.getElementById('risk-tier-badge');
            if (riskTierEl) {
                riskTierEl.textContent = `${riskMetrics.tier?.toUpperCase() || 'UNKNOWN'} (${riskMetrics.risk_percentage || '0%'})`;
                riskTierEl.className = `badge bg-${riskMetrics.tier_color || 'secondary'}`;
            }
            
            // Risk level indicator
            const riskLevelEl = document.getElementById('risk-level');
            if (riskLevelEl) {
                riskLevelEl.textContent = riskMetrics.level || 'Unknown';
            }
            
            // Max position size
            const maxPositionEl = document.getElementById('max-position-size');
            if (maxPositionEl && riskMetrics.max_position_usd) {
                maxPositionEl.textContent = `$${riskMetrics.max_position_usd.toFixed(2)}`;
            }
            
            // Portfolio risk score (for progress bar)
            const riskScoreEl = document.getElementById('portfolio-risk-score');
            const riskProgressEl = document.getElementById('risk-progress-bar');
            if (riskScoreEl && riskMetrics.portfolio_risk_score) {
                const score = Math.min(100, riskMetrics.portfolio_risk_score);
                riskScoreEl.textContent = `${score.toFixed(1)}%`;
                
                if (riskProgressEl) {
                    riskProgressEl.style.width = `${score}%`;
                    riskProgressEl.className = `progress-bar bg-${score > 70 ? 'danger' : score > 40 ? 'warning' : 'success'}`;
                }
            }
            
            console.log('✅ Dynamic risk display updated');
            
        } catch (error) {
            console.error('❌ Error updating risk metrics display:', error);
        }
    }
    
    updatePortfolioRiskDisplay(riskMetrics) {
        try {
            // Risk warnings
            const warningsEl = document.getElementById('risk-warnings');
            if (warningsEl && riskMetrics.warnings) {
                warningsEl.innerHTML = '';
                
                if (riskMetrics.warnings.length > 0) {
                    riskMetrics.warnings.forEach(warning => {
                        const warningEl = document.createElement('div');
                        warningEl.className = 'alert alert-warning alert-dismissible fade show';
                        warningEl.innerHTML = `
                            ${warning}
                            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        `;
                        warningsEl.appendChild(warningEl);
                    });
                } else {
                    warningsEl.innerHTML = '<div class="text-success">✅ All risk metrics within acceptable ranges</div>';
                }
            }
            
            // Portfolio utilization
            const utilizationEl = document.getElementById('portfolio-utilization');
            if (utilizationEl && riskMetrics.portfolio_utilization) {
                utilizationEl.textContent = riskMetrics.portfolio_utilization;
            }
            
            // Balance trend
            const trendEl = document.getElementById('balance-trend');
            if (trendEl && riskMetrics.balance_trend) {
                const trend = riskMetrics.balance_trend;
                const trendIcon = trend === 'growing' ? '📈' : trend === 'declining' ? '📉' : '➡️';
                const trendColor = trend === 'growing' ? 'text-success' : trend === 'declining' ? 'text-danger' : 'text-muted';
                trendEl.innerHTML = `<span class="${trendColor}">${trendIcon} ${trend.charAt(0).toUpperCase() + trend.slice(1)}</span>`;
            }
            
            console.log('✅ Risk metrics display updated');
            
        } catch (error) {
            console.error('❌ Error updating risk metrics display:', error);
        }
    }
    
    updateMarketDataDisplay(marketData) {
        try {
            if (marketData && marketData.BTCUSDT) {
                const btcPriceEl = document.getElementById('btc-price');
                const btcChangeEl = document.getElementById('btc-change');
                
                if (btcPriceEl) {
                    btcPriceEl.textContent = `$${parseFloat(marketData.BTCUSDT.price).toLocaleString()}`;
                }
                
                if (btcChangeEl) {
                    const change = marketData.BTCUSDT.change_24h;
                    btcChangeEl.textContent = change;
                    btcChangeEl.className = `badge ${change.startsWith('+') ? 'bg-success' : 'bg-danger'}`;
                }
            }
            
            if (marketData && marketData.ETHUSDT) {
                const ethPriceEl = document.getElementById('eth-price');
                const ethChangeEl = document.getElementById('eth-change');
                
                if (ethPriceEl) {
                    ethPriceEl.textContent = `$${parseFloat(marketData.ETHUSDT.price).toLocaleString()}`;
                }
                
                if (ethChangeEl) {
                    const change = marketData.ETHUSDT.change_24h;
                    ethChangeEl.textContent = change;
                    ethChangeEl.className = `badge ${change.startsWith('+') ? 'bg-success' : 'bg-danger'}`;
                }
            }
            
            console.log('✅ Market data display updated');
            
        } catch (error) {
            console.error('❌ Error updating market data display:', error);
        }
    }
    
    updateSystemStatusDisplay(systemStatus) {
        try {
            const statusBanner = document.getElementById('system-status-banner');
            if (statusBanner && systemStatus) {
                const indicator = statusBanner.querySelector('.status-indicator');
                const title = statusBanner.querySelector('.status-title');
                const subtitle = statusBanner.querySelector('.status-subtitle');
                
                if (systemStatus.api_connected) {
                    if (indicator) indicator.textContent = '🟢';
                    if (title) title.textContent = `${systemStatus.environment?.toUpperCase()} CONNECTED - DYNAMIC RISK ACTIVE`;
                    if (subtitle) subtitle.textContent = 'Real-time data and dynamic risk management enabled';
                    statusBanner.className = 'system-status-banner mb-4 status-success';
                } else {
                    if (indicator) indicator.textContent = '🟡';
                    if (title) title.textContent = 'PAPER TRADING MODE';
                    if (subtitle) subtitle.textContent = 'Add BYBIT_API_KEY to environment variables for live data';
                    statusBanner.className = 'system-status-banner mb-4 status-warning';
                }
            }
            
            console.log('✅ System status display updated');
            
        } catch (error) {
            console.error('❌ Error updating system status display:', error);
        }
    }
    
    updateEnvironmentBadge(environment, apiConnected) {
        const envBadgeEl = document.getElementById('environment-badge');
        if (envBadgeEl) {
            if (apiConnected) {
                envBadgeEl.textContent = environment?.toUpperCase() || 'UNKNOWN';
                envBadgeEl.className = `badge ${environment === 'testnet' ? 'bg-warning' : 'bg-success'}`;
            } else {
                envBadgeEl.textContent = 'PAPER';
                envBadgeEl.className = 'badge bg-secondary';
            }
        }
    }
    
    updateTimestamp(timestamp) {
        const timestampEl = document.getElementById('last-update');
        if (timestampEl && timestamp) {
            const updateTime = new Date(timestamp);
            const now = new Date();
            const diffSeconds = Math.floor((now - updateTime) / 1000);
            
            let timeText;
            if (diffSeconds < 60) {
                timeText = 'Just now';
            } else if (diffSeconds < 3600) {
                timeText = `${Math.floor(diffSeconds / 60)}m ago`;
            } else {
                timeText = updateTime.toLocaleTimeString();
            }
            
            timestampEl.textContent = timeText;
        }
    }
    
    async calculateRiskForBalance(balance) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/calculate-risk/${balance}`);
            const riskData = await response.json();
            
            // Update risk calculator display
            const resultEl = document.getElementById('risk-calculator-result');
            if (resultEl) {
                resultEl.innerHTML = `
                    <div class="card">
                        <div class="card-body">
                            <h4>Dynamic Risk Analysis for $${balance.toLocaleString()}</h4>
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Risk Tier:</strong> <span class="badge bg-${riskData.tier_color || 'secondary'}">${riskData.tier?.toUpperCase()}</span></p>
                                    <p><strong>Risk Level:</strong> ${riskData.level}</p>
                                    <p><strong>Risk Percentage:</strong> ${riskData.risk_percentage}</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Max Position:</strong> $${riskData.max_position_usd?.toFixed(2)}</p>
                                    <p><strong>Daily Budget:</strong> $${riskData.daily_risk_budget?.toFixed(2)}</p>
                                    <p><strong>Max Positions:</strong> ${riskData.max_concurrent_positions}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
            
        } catch (error) {
            console.error('❌ Error calculating risk:', error);
        }
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const status = await response.json();
            
            // Update system health indicator
            const healthEl = document.getElementById('system-health');
            if (healthEl) {
                healthEl.innerHTML = status.api_connected ? 
                    '<span class="status-dot status-dot-animated bg-green"></span> Operational' :
                    '<span class="status-dot bg-yellow"></span> Paper Mode';
            }
            
        } catch (error) {
            console.error('❌ Error loading system status:', error);
            const healthEl = document.getElementById('system-health');
            if (healthEl) {
                healthEl.innerHTML = '<span class="status-dot bg-red"></span> Error';
            }
        }
    }
    
    async loadPortfolioData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/portfolio`);
            const portfolio = await response.json();
            
            this.updatePortfolioDisplay(portfolio);
            
        } catch (error) {
            console.error('❌ Error loading portfolio data:', error);
        }
    }
    
    async loadRiskMetrics() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/risk-metrics`);
            const riskMetrics = await response.json();
            
            this.updateRiskMetricsDisplay(riskMetrics);
            
        } catch (error) {
            console.error('❌ Error loading risk metrics:', error);
        }
    }
    
    showErrorState(message) {
        console.error('🚨 Error state:', message);
        
        // Show error toast or notification
        const errorToast = document.createElement('div');
        errorToast.className = 'toast align-items-center text-white bg-danger border-0';
        errorToast.setAttribute('role', 'alert');
        errorToast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        // Add to page if toast container exists
        const toastContainer = document.getElementById('toast-container') || document.body;
        toastContainer.appendChild(errorToast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (errorToast.parentNode) {
                errorToast.parentNode.removeChild(errorToast);
            }
        }, 5000);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.tradingDashboard = new TradingDashboard();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TradingDashboard;
}