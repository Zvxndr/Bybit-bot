// API service to connect to FastAPI backend
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api'  // DigitalOcean will serve from same domain
  : 'http://localhost:8080/api'  // Local development

class ApiService {
  async get(endpoint) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('API GET error:', error)
      throw error
    }
  }

  async post(endpoint, data) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('API POST error:', error)
      throw error
    }
  }

  async delete(endpoint) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'DELETE'
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('API DELETE error:', error)
      throw error
    }
  }

  // Historical Data endpoints (connecting to existing working_api_server.py)
  async discoverHistoricalData() {
    return this.get('/historical-data/discover')
  }

  async deleteSymbolData(symbol) {
    return this.delete(`/historical-data/symbol/${symbol}`)
  }

  async clearAllData() {
    return this.delete('/historical-data/clear-all')
  }

  // Backtest endpoints
  async getBacktestHistory() {
    return this.get('/backtest/history')
  }

  async runBacktest(config) {
    return this.post('/backtest/run', config)
  }

  // Future ML endpoints (to be added to backend later)
  async getStrategyRanking(period = 'all') {
    return this.get(`/strategies/ranking?period=${period}`)
  }

  async getMLStatus() {
    return this.get('/ml/status')
  }

  async setMLRequirements(requirements) {
    return this.post('/ml/requirements', requirements)
  }

  async setRetirementMetrics(metrics) {
    return this.post('/ml/retirement-metrics', metrics)
  }

  // API Status endpoints
  async getApiStatus() {
    return this.get('/status/apis')
  }

  async getNewsSentiment() {
    return this.get('/status/news-sentiment')
  }

  async getCorrelationData() {
    return this.get('/correlation/data')
  }

  async sendDailyReport() {
    return this.post('/reports/email', {})
  }

  async testBybitConnection(testnet = true) {
    return this.post(`/test/bybit-connection?testnet=${testnet}`, {})
  }

  async getFutureMarketsStatus() {
    return this.get('/status/future-markets')
  }
}

export const apiService = new ApiService()
export default apiService