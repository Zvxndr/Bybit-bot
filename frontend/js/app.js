/**
 * TradingBot Pro - Core Application
 * Main application class handling authentication, WebSocket, and API communication
 */

class TradingBotApp {
    constructor() {
        this.apiBase = window.location.origin + '/api';  // Same origin - integrated server
        this.wsConnection = null;
        this.currentUser = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        this.isAuthenticated = false;
        
        // Initialize the application
        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        console.log('🚀 Initializing TradingBot Pro...');
        
        try {
            // Check authentication first
            await this.checkAuth();
            
            if (this.isAuthenticated) {
                // Setup WebSocket connection
                this.setupWebSocket();
                
                // Load page-specific functionality
                this.loadCurrentPage();
                
                // Setup global event listeners
                this.setupEventListeners();
                
                // Start data refresh intervals
                this.startDataRefresh();
                
                console.log('✅ TradingBot Pro initialized successfully');
            }
        } catch (error) {
            console.error('❌ Failed to initialize application:', error);
            this.handleAuthError();
        }
    }

    /**
     * Check authentication status
     */
    async checkAuth() {
        const token = localStorage.getItem('jwt_token');
        
        if (!token) {
            throw new Error('No authentication token found');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/auth/verify`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Authentication failed: ${response.status}`);
            }
            
            this.currentUser = await response.json();
            this.isAuthenticated = true;
            
            console.log('✅ Authentication successful:', this.currentUser);
            
        } catch (error) {
            console.error('❌ Authentication failed:', error);
            throw error;
        }
    }

    /**
     * Handle authentication errors
     */
    handleAuthError() {
        // Clear invalid token
        localStorage.removeItem('jwt_token');
        
        // Show login prompt or redirect
        this.showError('Authentication failed. Please log in again.');
        
        // For now, we'll simulate login - in production, redirect to login page
        setTimeout(() => {
            this.promptLogin();
        }, 2000);
    }

    /**
     * Simulate login for development (replace with actual login form)
     */
    async promptLogin() {
        // For development, we'll create a temporary token
        // In production, replace this with actual login form
        
        console.log('🔑 Simulating login for development...');
        
        try {
            // Simulate login API call
            const loginData = {
                username: 'admin',
                password: 'admin' // This should come from a login form
            };
            
            const response = await fetch(`${this.apiBase}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(loginData)
            });
            
            if (response.ok) {
                const data = await response.json();
                localStorage.setItem('jwt_token', data.access_token);
                
                console.log('✅ Login successful, reloading...');
                window.location.reload();
            } else {
                // If login endpoint doesn't exist yet, create a demo token
                console.log('⚠️ Login endpoint not available, using demo mode');
                localStorage.setItem('jwt_token', 'demo-token-for-development');
                this.currentUser = { username: 'admin', role: 'ADMIN' };
                this.isAuthenticated = true;
                
                // Continue initialization
                this.setupWebSocket();
                this.loadCurrentPage();
                this.setupEventListeners();
                this.startDataRefresh();
            }
            
        } catch (error) {
            console.error('❌ Login failed:', error);
            this.showError('Login failed. Check console for details.');
        }
    }

    /**
     * Setup WebSocket connection for real-time updates
     */
    setupWebSocket() {
        const token = localStorage.getItem('jwt_token');
        const wsUrl = `ws://localhost:8000/ws?token=${encodeURIComponent(token)}`;
        
        console.log('🔌 Connecting to WebSocket:', wsUrl);
        
        try {
            this.wsConnection = new WebSocket(wsUrl);
            
            this.wsConnection.onopen = () => {
                console.log('✅ WebSocket connected');
                this.reconnectAttempts = 0;
                this.showSuccess('Real-time updates connected');
            };
            
            this.wsConnection.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleRealTimeUpdate(data);
                } catch (error) {
                    console.error('❌ Failed to parse WebSocket message:', error);
                }
            };
            
            this.wsConnection.onclose = (event) => {
                console.log('🔌 WebSocket disconnected:', event.code, event.reason);
                this.handleWebSocketDisconnect();
            };
            
            this.wsConnection.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
            };
            
        } catch (error) {
            console.error('❌ Failed to setup WebSocket:', error);
            // Continue without WebSocket - fallback to polling
            this.setupPolling();
        }
    }

    /**
     * Handle WebSocket disconnection and attempt reconnection
     */
    handleWebSocketDisconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            
            console.log(`🔄 Attempting WebSocket reconnection (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            setTimeout(() => {
                this.setupWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.log('❌ Max reconnection attempts reached, falling back to polling');
            this.showWarning('Real-time updates disconnected. Using fallback mode.');
            this.setupPolling();
        }
    }

    /**
     * Setup polling as fallback for WebSocket
     */
    setupPolling() {
        // Poll for updates every 10 seconds
        setInterval(() => {
            this.refreshSystemData();
        }, 10000);
    }

    /**
     * Handle real-time updates from WebSocket
     */
    handleRealTimeUpdate(data) {
        console.log('📡 Real-time update received:', data.type);
        
        switch(data.type) {
            case 'system_status':
                this.updateSystemStatus(data.payload);
                break;
                
            case 'trading_update':
                this.updateTradingData(data.payload);
                break;
                
            case 'pipeline_update':
                this.updatePipeline(data.payload);
                break;
                
            case 'strategy_update':
                this.updateStrategy(data.payload);
                break;
                
            case 'alert':
                this.showAlert(data.payload);
                break;
                
            case 'notification':
                this.showNotification(data.payload);
                break;
                
            default:
                console.log('❓ Unknown update type:', data.type);
        }
    }

    /**
     * Make API calls with authentication
     */
    async apiCall(endpoint, options = {}) {
        const token = localStorage.getItem('jwt_token');
        const headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            ...options.headers
        };

        const config = {
            ...options,
            headers
        };

        try {
            console.log(`📡 API Call: ${options.method || 'GET'} ${endpoint}`);
            
            const response = await fetch(`${this.apiBase}${endpoint}`, config);
            
            if (!response.ok) {
                if (response.status === 401) {
                    // Token expired or invalid
                    this.handleAuthError();
                    throw new Error('Authentication required');
                }
                
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`✅ API Response: ${endpoint}`, data);
            
            return data;
            
        } catch (error) {
            console.error(`❌ API call failed: ${endpoint}`, error);
            
            // For development, return mock data if endpoint doesn't exist
            if (error.message.includes('fetch')) {
                console.log('⚠️ Using mock data for development');
                return this.getMockData(endpoint);
            }
            
            throw error;
        }
    }

    /**
     * Get mock data for development when API endpoints are not available
     */
    getMockData(endpoint) {
        const mockData = {
            '/status': {
                system_status: 'debug',
                trading_enabled: false,
                health: 100,
                uptime: '2d 14h 32m'
            },
            '/metrics': {
                portfolio_value: 25380.50,
                portfolio_change: 2.3,
                active_strategies: 3,
                paper_strategies: 2,
                live_strategies: 1,
                today_pnl: 142.35,
                today_pnl_percent: 0.6
            },
            '/strategies': [
                {
                    id: 'BTC_M_55632',
                    phase: 'live',
                    sharpe: 1.8,
                    win_rate: 65,
                    max_drawdown: 12,
                    allocation: 25,
                    live_pnl: 8.4
                },
                {
                    id: 'ETH_H_33421',
                    phase: 'paper',
                    sharpe: 2.1,
                    win_rate: 58,
                    max_drawdown: 18,
                    days_in_phase: 8
                }
            ]
        };
        
        return mockData[endpoint] || { message: 'Mock data not available' };
    }

    /**
     * Load page-specific functionality
     */
    loadCurrentPage() {
        const path = window.location.pathname;
        const page = path.split('/').pop().replace('.html', '') || 'index';
        
        console.log(`📄 Loading page: ${page}`);
        
        // Load page-specific JavaScript if available
        const pageScript = document.createElement('script');
        pageScript.src = `js/${page}.js`;
        pageScript.onerror = () => {
            console.log(`⚠️ No specific script found for page: ${page}`);
        };
        document.head.appendChild(pageScript);
    }

    /**
     * Setup global event listeners
     */
    setupEventListeners() {
        // Emergency stop button
        const emergencyBtn = document.getElementById('emergency-stop');
        if (emergencyBtn) {
            emergencyBtn.addEventListener('click', () => this.emergencyStop());
        }
        
        // Refresh data button
        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshSystemData());
        }
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('📱 Page hidden, reducing update frequency');
            } else {
                console.log('📱 Page visible, resuming normal updates');
                this.refreshSystemData();
            }
        });
    }

    /**
     * Start automatic data refresh
     */
    startDataRefresh() {
        // Refresh system data every 30 seconds
        setInterval(() => {
            if (!document.hidden) {
                this.refreshSystemData();
            }
        }, 30000);
        
        // Initial data load
        this.refreshSystemData();
    }

    /**
     * Refresh system data
     */
    async refreshSystemData() {
        try {
            console.log('🔄 Refreshing system data...');
            
            // Get system status
            const status = await this.apiCall('/status');
            this.updateSystemStatus(status);
            
            // Get metrics
            const metrics = await this.apiCall('/metrics');
            this.updateMetrics(metrics);
            
            console.log('✅ System data refreshed');
            
        } catch (error) {
            console.error('❌ Failed to refresh system data:', error);
        }
    }

    /**
     * Update system status display
     */
    updateSystemStatus(status) {
        const banner = document.getElementById('system-status-banner');
        if (!banner) return;
        
        const indicator = banner.querySelector('.status-indicator');
        const title = banner.querySelector('.status-title');
        const subtitle = banner.querySelector('.status-subtitle');
        
        if (status.system_status === 'live') {
            banner.className = 'system-status-banner live';
            indicator.textContent = '🚀';
            title.textContent = 'LIVE TRADING ACTIVE';
            subtitle.textContent = 'Production Mode - Real money at risk';
        } else {
            banner.className = 'system-status-banner debug';
            indicator.textContent = '🛑';
            title.textContent = 'DEBUG MODE - LIVE TRADING DISABLED';
            subtitle.textContent = 'Phase 1: Testing Historical Data & Safety Systems';
        }
    }

    /**
     * Update metrics display
     */
    updateMetrics(metrics) {
        // Portfolio Value
        const portfolioValue = document.getElementById('portfolio-value');
        if (portfolioValue && metrics.portfolio_value) {
            portfolioValue.textContent = `$${metrics.portfolio_value.toLocaleString()}`;
        }
        
        // Portfolio Change
        const portfolioChange = document.getElementById('portfolio-change');
        if (portfolioChange && metrics.portfolio_change) {
            portfolioChange.textContent = `${metrics.portfolio_change > 0 ? '+' : ''}${metrics.portfolio_change}%`;
            portfolioChange.className = metrics.portfolio_change > 0 ? 'text-success ms-3' : 'text-danger ms-3';
        }
        
        // Active Strategies
        const activeStrategies = document.getElementById('active-strategies');
        if (activeStrategies && metrics.active_strategies) {
            activeStrategies.textContent = metrics.active_strategies;
        }
        
        // Today's P&L
        const todayPnl = document.getElementById('today-pnl');
        if (todayPnl && metrics.today_pnl) {
            const pnl = metrics.today_pnl;
            todayPnl.textContent = `${pnl > 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}`;
            todayPnl.className = pnl > 0 ? 'h1 mb-3 text-success' : 'h1 mb-3 text-danger';
        }
        
        // System Health
        const systemHealth = document.getElementById('system-health');
        if (systemHealth && metrics.health) {
            systemHealth.textContent = `${metrics.health}%`;
        }
        
        // Update last update time
        const lastUpdate = document.getElementById('last-update');
        if (lastUpdate) {
            lastUpdate.textContent = new Date().toLocaleTimeString();
        }
    }

    /**
     * Emergency stop all trading
     */
    async emergencyStop() {
        if (!confirm('⚠️ EMERGENCY STOP\n\nThis will immediately stop ALL trading activities.\n\nAre you sure?')) {
            return;
        }
        
        try {
            await this.apiCall('/system/emergency-stop', { method: 'POST' });
            this.showSuccess('🛑 Emergency stop activated - All trading stopped');
        } catch (error) {
            console.error('❌ Emergency stop failed:', error);
            this.showError('Failed to activate emergency stop');
        }
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        console.log('✅', message);
        // In a real implementation, you'd show a toast notification
        this.showNotification(message, 'success');
    }

    /**
     * Show error message
     */
    showError(message) {
        console.error('❌', message);
        this.showNotification(message, 'error');
    }

    /**
     * Show warning message
     */
    showWarning(message) {
        console.warn('⚠️', message);
        this.showNotification(message, 'warning');
    }

    /**
     * Show notification (basic implementation)
     */
    showNotification(message, type = 'info') {
        // Create a simple notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade-in`;
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
        `;
        
        // Add to page (you might want to create a dedicated notification container)
        document.body.insertBefore(notification, document.body.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tradingApp = new TradingBotApp();
});

// Export for use in other scripts
window.TradingBotApp = TradingBotApp;