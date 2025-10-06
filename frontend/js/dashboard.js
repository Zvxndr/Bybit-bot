/**
 * Dashboard-specific functionality
 * Handles dashboard page interactions and data display
 */

class Dashboard {
    constructor() {
        this.chartInstance = null;
        this.refreshInterval = null;
        
        this.init();
    }

    /**
     * Initialize dashboard functionality
     */
    init() {
        console.log('📊 Initializing Dashboard...');
        
        // Wait for main app to be ready
        if (window.tradingApp && window.tradingApp.isAuthenticated) {
            this.setupDashboard();
        } else {
            // Wait for app to initialize
            setTimeout(() => this.init(), 500);
        }
    }

    /**
     * Setup dashboard-specific features
     */
    setupDashboard() {
        this.setupEventListeners();
        this.loadDashboardData();
        this.setupPerformanceChart();
        this.startRealTimeUpdates();
        
        console.log('✅ Dashboard initialized');
    }

    /**
     * Setup dashboard event listeners
     */
    setupEventListeners() {
        // Timeframe selector for performance chart
        const timeframeSelector = document.getElementById('timeframe-selector');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', (e) => {
                this.updatePerformanceChart(e.target.value);
            });
        }

        // Pipeline stage clicks for navigation
        const pipelineStages = document.querySelectorAll('.pipeline-stage');
        pipelineStages.forEach(stage => {
            stage.addEventListener('click', () => {
                window.location.href = 'pipeline.html';
            });
        });

        // Quick action buttons
        this.setupQuickActions();
    }

    /**
     * Setup quick action buttons
     */
    setupQuickActions() {
        // Add quick start backtest button
        const quickBacktest = document.createElement('button');
        quickBacktest.className = 'btn btn-outline-primary me-2';
        quickBacktest.innerHTML = '🚀 Quick Backtest';
        quickBacktest.addEventListener('click', () => {
            window.location.href = 'backtesting.html';
        });

        // Add to button group if exists
        const btnList = document.querySelector('.btn-list');
        if (btnList) {
            btnList.insertBefore(quickBacktest, btnList.firstChild);
        }
    }

    /**
     * Load dashboard data
     */
    async loadDashboardData() {
        try {
            console.log('📊 Loading dashboard data...');

            // Load system status and metrics (already handled by main app)
            
            // Load recent activity
            await this.loadRecentActivity();
            
            // Load system alerts
            await this.loadSystemAlerts();
            
            // Load pipeline summary
            await this.loadPipelineSummary();

        } catch (error) {
            console.error('❌ Failed to load dashboard data:', error);
        }
    }

    /**
     * Load recent activity feed
     */
    async loadRecentActivity() {
        try {
            const activities = await window.tradingApp.apiCall('/activity/recent');
            this.updateActivityFeed(activities);
        } catch (error) {
            console.error('❌ Failed to load recent activity:', error);
            // Use mock data for development
            this.updateActivityFeed(this.getMockActivity());
        }
    }

    /**
     * Load system alerts
     */
    async loadSystemAlerts() {
        try {
            const alerts = await window.tradingApp.apiCall('/alerts');
            this.updateAlertsDisplay(alerts);
        } catch (error) {
            console.error('❌ Failed to load system alerts:', error);
            // Use mock data for development
            this.updateAlertsDisplay(this.getMockAlerts());
        }
    }

    /**
     * Load pipeline summary
     */
    async loadPipelineSummary() {
        try {
            const pipeline = await window.tradingApp.apiCall('/pipeline/summary');
            this.updatePipelineDisplay(pipeline);
        } catch (error) {
            console.error('❌ Failed to load pipeline summary:', error);
            // Use mock data for development
            this.updatePipelineDisplay(this.getMockPipeline());
        }
    }

    /**
     * Update activity feed display
     */
    updateActivityFeed(activities) {
        const activityFeed = document.getElementById('recent-activity');
        if (!activityFeed) return;

        const activitiesData = Array.isArray(activities) ? activities : activities.activities || [];

        activityFeed.innerHTML = activitiesData.map(activity => `
            <div class="row">
                <div class="col-auto">
                    <span class="status-dot status-dot-animated bg-${this.getActivityColor(activity.type)} d-block"></span>
                </div>
                <div class="col">
                    <div class="text-truncate">${activity.message}</div>
                    <div class="text-muted">${this.formatTimeAgo(activity.timestamp)}</div>
                </div>
            </div>
        `).join('');
    }

    /**
     * Update alerts display
     */
    updateAlertsDisplay(alerts) {
        const alertsDisplay = document.getElementById('system-alerts');
        if (!alertsDisplay) return;

        const alertsData = Array.isArray(alerts) ? alerts : alerts.alerts || [];

        alertsDisplay.innerHTML = alertsData.map(alert => `
            <div class="row">
                <div class="col-auto">
                    <span class="status-dot bg-${this.getAlertColor(alert.level)} d-block"></span>
                </div>
                <div class="col">
                    <div class="text-truncate">${alert.message}</div>
                    <div class="text-muted">${alert.source} - ${this.formatTimeAgo(alert.timestamp)}</div>
                </div>
            </div>
        `).join('');
    }

    /**
     * Update pipeline display
     */
    updatePipelineDisplay(pipeline) {
        // Update pipeline counts
        const backtestCount = document.getElementById('backtest-count');
        const paperCount = document.getElementById('paper-count');
        const liveCount = document.getElementById('live-count');

        if (backtestCount && pipeline.backtest_count !== undefined) {
            backtestCount.textContent = pipeline.backtest_count;
        }
        
        if (paperCount && pipeline.paper_count !== undefined) {
            paperCount.textContent = pipeline.paper_count;
        }
        
        if (liveCount && pipeline.live_count !== undefined) {
            liveCount.textContent = pipeline.live_count;
        }
    }

    /**
     * Setup performance chart
     */
    setupPerformanceChart() {
        const chartCanvas = document.getElementById('performance-chart');
        if (!chartCanvas) return;

        // For now, we'll use a simple chart library or create a placeholder
        // In production, you'd use Chart.js or similar
        this.createMockChart(chartCanvas);
    }

    /**
     * Create mock performance chart (replace with real Chart.js implementation)
     */
    createMockChart(canvas) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Set up chart style
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;

        // Generate mock data points
        const points = [];
        const numPoints = 30;
        let value = 25000;

        for (let i = 0; i < numPoints; i++) {
            value += (Math.random() - 0.5) * 500;
            points.push({
                x: (i / (numPoints - 1)) * width,
                y: height - ((value - 24000) / 2000) * height
            });
        }

        // Draw line chart
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);

        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }

        ctx.stroke();

        // Add labels
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px Arial';
        ctx.fillText('$24K', 10, height - 10);
        ctx.fillText('$26K', 10, 20);
    }

    /**
     * Update performance chart with new timeframe
     */
    updatePerformanceChart(timeframe) {
        console.log(`📊 Updating chart for timeframe: ${timeframe}`);
        
        // In production, fetch new data based on timeframe
        // For now, just recreate the mock chart
        const chartCanvas = document.getElementById('performance-chart');
        if (chartCanvas) {
            this.createMockChart(chartCanvas);
        }
    }

    /**
     * Start real-time updates for dashboard
     */
    startRealTimeUpdates() {
        // Update activity feed every minute
        this.refreshInterval = setInterval(() => {
            this.loadRecentActivity();
            this.loadSystemAlerts();
        }, 60000);
    }

    /**
     * Get activity color based on type
     */
    getActivityColor(type) {
        const colors = {
            'strategy_graduation': 'green',
            'strategy_created': 'blue',
            'strategy_stopped': 'yellow',
            'trade_executed': 'blue',
            'error': 'red',
            'warning': 'yellow',
            'info': 'blue'
        };
        
        return colors[type] || 'blue';
    }

    /**
     * Get alert color based on level
     */
    getAlertColor(level) {
        const colors = {
            'success': 'green',
            'info': 'blue',
            'warning': 'yellow',
            'error': 'red',
            'critical': 'red'
        };
        
        return colors[level] || 'blue';
    }

    /**
     * Format timestamp to relative time
     */
    formatTimeAgo(timestamp) {
        if (!timestamp) return 'Just now';
        
        const now = new Date();
        const time = new Date(timestamp);
        const diff = Math.floor((now - time) / 1000);

        if (diff < 60) return 'Just now';
        if (diff < 3600) return `${Math.floor(diff / 60)} minutes ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
        
        return `${Math.floor(diff / 86400)} days ago`;
    }

    /**
     * Mock data for development
     */
    getMockActivity() {
        return [
            {
                type: 'strategy_graduation',
                message: 'Strategy BTC_M_55632 graduated to live trading',
                timestamp: new Date(Date.now() - 10 * 60 * 1000).toISOString()
            },
            {
                type: 'strategy_created',
                message: 'ML discovery completed - 12 new strategies found',
                timestamp: new Date(Date.now() - 25 * 60 * 1000).toISOString()
            },
            {
                type: 'strategy_created',
                message: 'ETH_H_33421 moved to paper trading phase',
                timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString()
            }
        ];
    }

    /**
     * Mock alerts for development
     */
    getMockAlerts() {
        return [
            {
                level: 'success',
                message: 'All API connections stable',
                source: 'System check',
                timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString()
            },
            {
                level: 'info',
                message: 'Database backup completed successfully',
                source: 'Backup system',
                timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString()
            },
            {
                level: 'warning',
                message: 'Risk management updated position limits',
                source: 'Risk manager',
                timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
            }
        ];
    }

    /**
     * Mock pipeline data for development
     */
    getMockPipeline() {
        return {
            backtest_count: 12,
            paper_count: 2,
            live_count: 1,
            graduation_rate: 68,
            discovered_today: 47
        };
    }

    /**
     * Cleanup dashboard resources
     */
    cleanup() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Wait for main app to be available
    const initDashboard = () => {
        if (window.tradingApp) {
            window.dashboard = new Dashboard();
        } else {
            setTimeout(initDashboard, 100);
        }
    };
    
    initDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.cleanup();
    }
});