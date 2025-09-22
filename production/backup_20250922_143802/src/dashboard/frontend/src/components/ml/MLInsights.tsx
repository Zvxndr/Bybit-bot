'use client'

import { useState, useEffect } from 'react'
import { useWebSocket } from '@/components/providers/WebSocketProvider'
import { Brain, Target, TrendingUp, BarChart3 } from 'lucide-react'

interface MLData {
  predictionAccuracy: number
  modelConfidence: number
  featuresAnalyzed: number
  nextPrediction: string
  riskScore: number
  modelVersion: string
}

export default function MLInsights() {
  const { isConnected, subscribe, lastMessage } = useWebSocket()
  const [mlData, setMLData] = useState<MLData>({
    predictionAccuracy: 78.9,
    modelConfidence: 0.87,
    featuresAnalyzed: 245,
    nextPrediction: 'BULLISH',
    riskScore: 0.23,
    modelVersion: 'v2.1.3'
  })

  useEffect(() => {
    if (isConnected) {
      subscribe('ml_insights')
      subscribe('model_performance')
      subscribe('predictions')
    }
  }, [isConnected, subscribe])

  useEffect(() => {
    if (lastMessage && lastMessage.type === 'ml_data') {
      setMLData(lastMessage.data)
    }
  }, [lastMessage])

  return (
    <div className="space-y-6">
      {/* Model Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Prediction Accuracy</p>
              <p className="text-2xl font-bold text-crypto-green">
                {mlData.predictionAccuracy.toFixed(1)}%
              </p>
            </div>
            <Target className="h-8 w-8 text-crypto-green" />
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Model Confidence</p>
              <p className="text-2xl font-bold text-crypto-blue">
                {(mlData.modelConfidence * 100).toFixed(0)}%
              </p>
            </div>
            <Brain className="h-8 w-8 text-crypto-blue" />
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Features Analyzed</p>
              <p className="text-2xl font-bold text-white">
                {mlData.featuresAnalyzed}
              </p>
            </div>
            <BarChart3 className="h-8 w-8 text-crypto-yellow" />
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Risk Score</p>
              <p className={`text-2xl font-bold ${mlData.riskScore < 0.3 ? 'text-crypto-green' : mlData.riskScore < 0.7 ? 'text-crypto-yellow' : 'text-crypto-red'}`}>
                {(mlData.riskScore * 100).toFixed(0)}%
              </p>
            </div>
            <div className={`text-2xl ${mlData.riskScore < 0.3 ? 'text-crypto-green' : mlData.riskScore < 0.7 ? 'text-crypto-yellow' : 'text-crypto-red'}`}>
              ⚠
            </div>
          </div>
        </div>
      </div>

      {/* ML Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Current Predictions */}
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <h3 className="text-lg font-semibold text-white mb-4">Current Predictions</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-crypto-darker rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${mlData.nextPrediction === 'BULLISH' ? 'bg-crypto-green' : 'bg-crypto-red'}`}></div>
                <div>
                  <p className="text-sm font-medium text-white">BTC/USDT</p>
                  <p className="text-xs text-crypto-light-gray">Next 1H</p>
                </div>
              </div>
              <div className="text-right">
                <p className={`text-sm font-bold ${mlData.nextPrediction === 'BULLISH' ? 'text-crypto-green' : 'text-crypto-red'}`}>
                  {mlData.nextPrediction}
                </p>
                <p className="text-xs text-crypto-light-gray">
                  {(mlData.modelConfidence * 100).toFixed(0)}% confidence
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between p-4 bg-crypto-darker rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 rounded-full bg-crypto-green"></div>
                <div>
                  <p className="text-sm font-medium text-white">ETH/USDT</p>
                  <p className="text-xs text-crypto-light-gray">Next 1H</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm font-bold text-crypto-green">BULLISH</p>
                <p className="text-xs text-crypto-light-gray">76% confidence</p>
              </div>
            </div>

            <div className="flex items-center justify-between p-4 bg-crypto-darker rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 rounded-full bg-crypto-red"></div>
                <div>
                  <p className="text-sm font-medium text-white">SOL/USDT</p>
                  <p className="text-xs text-crypto-light-gray">Next 1H</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm font-bold text-crypto-red">BEARISH</p>
                <p className="text-xs text-crypto-light-gray">82% confidence</p>
              </div>
            </div>
          </div>
        </div>

        {/* Feature Importance */}
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <h3 className="text-lg font-semibold text-white mb-4">Feature Importance</h3>
          <div className="space-y-4">
            {[
              { name: 'Price Movement', importance: 0.89, color: 'bg-crypto-green' },
              { name: 'Volume Analysis', importance: 0.76, color: 'bg-crypto-blue' },
              { name: 'RSI Indicator', importance: 0.65, color: 'bg-crypto-yellow' },
              { name: 'MACD Signal', importance: 0.54, color: 'bg-crypto-red' },
              { name: 'Order Book Depth', importance: 0.43, color: 'bg-purple-500' },
            ].map((feature, index) => (
              <div key={index} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-crypto-light-gray">{feature.name}</span>
                  <span className="text-white">{(feature.importance * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-crypto-darker rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${feature.color}`}
                    style={{ width: `${feature.importance * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Information */}
      <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
        <h3 className="text-lg font-semibold text-white mb-4">Model Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-crypto-light-gray">Model Version</p>
            <p className="text-lg font-medium text-white">{mlData.modelVersion}</p>
          </div>
          <div>
            <p className="text-sm text-crypto-light-gray">Last Training</p>
            <p className="text-lg font-medium text-white">2 hours ago</p>
          </div>
          <div>
            <p className="text-sm text-crypto-light-gray">Training Samples</p>
            <p className="text-lg font-medium text-white">2.4M trades</p>
          </div>
        </div>
      </div>
    </div>
  )
}