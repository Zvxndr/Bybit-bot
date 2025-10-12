
// 🔧 HISTORICAL DATA & BACKTESTING INTEGRATION FIX
// Add this to your dashboard JavaScript to fix data integration

// Enhanced data discovery function
async function discoverAvailableDataEnhanced() {
    try {
        console.log('🔍 Discovering available historical data...');
        
        const response = await fetch('/api/historical-data/discover');
        const data = await response.json();
        
        if (data.success && data.datasets && data.datasets.length > 0) {
            console.log(`✅ Found ${data.datasets.length} datasets:`, data.datasets);
            
            // Update backtesting controls with discovered data
            updateBacktestingControls(data.datasets);
            
            // Show success message
            showNotification(`📊 Historical Data Available: ${data.datasets.length} datasets ready for backtesting`, 'success');
            
        } else {
            console.log('⚠️ No historical data found:', data);
            showNotification('📥 No historical data found. Use Download controls to get data.', 'warning');
        }
        
    } catch (error) {
        console.error('❌ Data discovery error:', error);
        showNotification('❌ Error discovering data. Check API connection.', 'error');
    }
}

// Update backtesting controls with available data
function updateBacktestingControls(datasets) {
    // Update pair selector
    const pairSelect = document.getElementById('backtestPair');
    if (pairSelect) {
        pairSelect.innerHTML = '<option value="">Select Trading Pair</option>';
        
        const uniquePairs = [...new Set(datasets.map(d => d.symbol))];
        uniquePairs.forEach(pair => {
            const option = document.createElement('option');
            option.value = pair;
            option.textContent = pair;
            pairSelect.appendChild(option);
        });
    }
    
    // Update timeframe selector based on selected pair
    const updateTimeframes = () => {
        const timeframeSelect = document.getElementById('backtestTimeframe');
        const selectedPair = pairSelect.value;
        
        if (timeframeSelect && selectedPair) {
            timeframeSelect.innerHTML = '<option value="">Select Timeframe</option>';
            
            const availableTimeframes = datasets
                .filter(d => d.symbol === selectedPair)
                .map(d => d.timeframe);
            
            availableTimeframes.forEach(timeframe => {
                const option = document.createElement('option');
                option.value = timeframe;
                option.textContent = timeframe;
                timeframeSelect.appendChild(option);
            });
        }
    };
    
    if (pairSelect) {
        pairSelect.addEventListener('change', updateTimeframes);
    }
}

// Enhanced backtest results loading
async function loadBacktestResults() {
    try {
        console.log('📊 Loading backtest results...');
        
        const response = await fetch('/api/backtest/history');
        const data = await response.json();
        
        if (data.success && data.data && data.data.length > 0) {
            console.log(`✅ Found ${data.data.length} backtest results`);
            updateBacktestResultsDisplay(data.data);
        } else {
            console.log('📈 No backtest results yet - run backtests to see results');
            showEmptyBacktestResults();
        }
        
    } catch (error) {
        console.error('❌ Backtest results loading error:', error);
        showEmptyBacktestResults();
    }
}

// Update backtest results display
function updateBacktestResultsDisplay(results) {
    const container = document.getElementById('backtestResultsContainer');
    if (!container) return;
    
    container.innerHTML = results.map(result => `
        <div class="backtest-result-item border rounded p-3 mb-2">
            <div class="row">
                <div class="col-md-3">
                    <strong>${result.pair}</strong> <span class="badge bg-secondary">${result.timeframe}</span>
                </div>
                <div class="col-md-2">
                    <span class="${result.total_return >= 0 ? 'text-success' : 'text-danger'}">
                        ${result.total_return >= 0 ? '+' : ''}${result.total_return}%
                    </span>
                </div>
                <div class="col-md-2">
                    Win Rate: ${result.win_rate}%
                </div>
                <div class="col-md-2">
                    Trades: ${result.trades_count}
                </div>
                <div class="col-md-3">
                    <small class="text-muted">${new Date(result.timestamp).toLocaleString()}</small>
                </div>
            </div>
        </div>
    `).join('');
}

// Show empty state for backtest results
function showEmptyBacktestResults() {
    const container = document.getElementById('backtestResultsContainer');
    if (container) {
        container.innerHTML = `
            <div class="text-center text-muted p-4">
                <i class="bi bi-graph-up fs-1"></i>
                <h5>No Backtest Results Yet</h5>
                <p>Run historical backtests to see performance results here</p>
            </div>
        `;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Discover data immediately
    setTimeout(() => {
        discoverAvailableDataEnhanced();
        loadBacktestResults();
    }, 1000);
    
    // Refresh every 30 seconds
    setInterval(() => {
        loadBacktestResults();
    }, 30000);
});

console.log('🔧 Historical Data & Backtesting Integration Fix Loaded');
