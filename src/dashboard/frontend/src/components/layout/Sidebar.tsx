'use client'

import { TrendingUp, Brain, Activity, Settings, BarChart3 } from 'lucide-react'

interface SidebarProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

const menuItems = [
  { id: 'trading', label: 'Trading', icon: TrendingUp },
  { id: 'ml', label: 'ML Insights', icon: Brain },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'system', label: 'System Health', icon: Activity },
  { id: 'settings', label: 'Settings', icon: Settings },
]

export default function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <div className="w-64 bg-crypto-darker border-r border-crypto-gray">
      {/* Logo */}
      <div className="p-6 border-b border-crypto-gray">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-crypto-blue rounded-lg flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Bybit AI</h1>
            <p className="text-xs text-crypto-light-gray">Trading Bot</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="p-4">
        <ul className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon
            const isActive = activeTab === item.id
            
            return (
              <li key={item.id}>
                <button
                  onClick={() => onTabChange(item.id)}
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-crypto-blue text-white'
                      : 'text-crypto-light-gray hover:bg-crypto-gray hover:text-white'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </button>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Status Indicator */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-crypto-gray">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-crypto-green rounded-full animate-pulse"></div>
          <span className="text-sm text-crypto-light-gray">Connected</span>
        </div>
      </div>
    </div>
  )
}