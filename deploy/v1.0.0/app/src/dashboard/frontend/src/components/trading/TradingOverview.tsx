'use client'

import { useState, useEffect } from 'react'
import { useWebSocket } from '@/components/providers/WebSocketProvider'
import { TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react'

interface TradingData {
  totalPnL: number
  todayPnL: number
  winRate: number
  totalTrades: number
  activePositions: number
  availableBalance: number
}

export default function TradingOverview() {
  const { isConnected, subscribe, lastMessage } = useWebSocket()
  const [tradingData, setTradingData] = useState<TradingData>({
    totalPnL: 0,
    todayPnL: 0,
    winRate: 0,
    totalTrades: 0,
    activePositions: 0,
    availableBalance: 0
  })

  useEffect(() => {
    if (isConnected) {
      subscribe('trading_overview')
      subscribe('positions')
      subscribe('performance')
    }
  }, [isConnected, subscribe])

  useEffect(() => {
    if (lastMessage && lastMessage.type === 'trading_data') {
      setTradingData(lastMessage.data)
    }
  }, [lastMessage])

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm text-gray-400">
            {isConnected ? 'Connected to Trading Backend' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Total P&L</p>
              <p className={`text-2xl font-bold ${tradingData.totalPnL >= 0 ? 'text-crypto-green' : 'text-crypto-red'}`}>
                ${tradingData.totalPnL.toFixed(2)}
              </p>
            </div>
            {tradingData.totalPnL >= 0 ? (
              <TrendingUp className="h-8 w-8 text-crypto-green" />
            ) : (
              <TrendingDown className="h-8 w-8 text-crypto-red" />
            )}
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Today's P&L</p>
              <p className={`text-2xl font-bold ${tradingData.todayPnL >= 0 ? 'text-crypto-green' : 'text-crypto-red'}`}>
                ${tradingData.todayPnL.toFixed(2)}
              </p>
            </div>
            <DollarSign className="h-8 w-8 text-crypto-yellow" />
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Win Rate</p>
              <p className="text-2xl font-bold text-white">
                {tradingData.winRate.toFixed(1)}%
              </p>
            </div>
            <Activity className="h-8 w-8 text-crypto-blue" />
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Active Positions</p>
              <p className="text-2xl font-bold text-white">
                {tradingData.activePositions}
              </p>
            </div>
            <div className="text-crypto-green text-2xl font-bold">●</div>
          </div>
        </div>
      </div>

      {/* Trading Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Positions */}
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Positions</h3>
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((_, index) => (
              <div key={index} className="flex items-center justify-between py-2 border-b border-crypto-light-gray/10">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-crypto-green rounded-full"></div>
                  <div>
                    <p className="text-sm font-medium text-white">BTC/USDT</p>
                    <p className="text-xs text-crypto-light-gray">Long • 0.05 BTC</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-crypto-green">+$234.56</p>
                  <p className="text-xs text-crypto-light-gray">+2.34%</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Trading Statistics */}
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <h3 className="text-lg font-semibold text-white mb-4">Trading Statistics</h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Total Trades</span>
              <span className="text-white font-medium">{tradingData.totalTrades}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Available Balance</span>
              <span className="text-white font-medium">${tradingData.availableBalance.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Average Trade Size</span>
              <span className="text-white font-medium">$1,234.56</span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Max Drawdown</span>
              <span className="text-crypto-red font-medium">-5.67%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Sharpe Ratio</span>
              <span className="text-crypto-green font-medium">1.72</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}