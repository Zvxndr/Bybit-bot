export interface TradingData {
  totalPnL: number
  todayPnL: number
  winRate: number
  totalTrades: number
  activePositions: number
  availableBalance: number
}

export interface Position {
  id: string
  symbol: string
  side: 'long' | 'short'
  size: number
  entryPrice: number
  currentPrice: number
  pnl: number
  pnlPercentage: number
  timestamp: string
}

export interface MLPrediction {
  symbol: string
  direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  confidence: number
  timeframe: string
  features: Record<string, number>
}

export interface SystemHealth {
  cpuUsage: number
  memoryUsage: number
  diskUsage: number
  networkLatency: number
  uptime: string
  alertsCount: number
  componentsStatus: Record<string, 'healthy' | 'warning' | 'error'>
}

export interface WebSocketMessage {
  type: string
  topic: string
  data: any
  timestamp: string
}

export interface ChartDataPoint {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}