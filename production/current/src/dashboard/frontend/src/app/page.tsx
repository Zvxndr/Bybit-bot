'use client'

import { useState, useEffect } from 'react'
import DashboardLayout from '@/components/layout/DashboardLayout'
import TradingOverview from '@/components/trading/TradingOverview'
import MLInsights from '@/components/ml/MLInsights'
import SystemHealth from '@/components/system/SystemHealth'
import WebSocketProvider from '@/components/providers/WebSocketProvider'

export default function Home() {
  const [activeTab, setActiveTab] = useState('trading')

  const renderContent = () => {
    switch (activeTab) {
      case 'trading':
        return <TradingOverview />
      case 'ml':
        return <MLInsights />
      case 'system':
        return <SystemHealth />
      default:
        return <TradingOverview />
    }
  }

  return (
    <WebSocketProvider>
      <DashboardLayout activeTab={activeTab} onTabChange={setActiveTab}>
        <div className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-white mb-2">
              Bybit Trading Dashboard
            </h1>
            <p className="text-crypto-light-gray">
              AI-powered cryptocurrency trading with real-time analytics
            </p>
          </div>
          
          {renderContent()}
        </div>
      </DashboardLayout>
    </WebSocketProvider>
  )
}