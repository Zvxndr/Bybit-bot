'use client'

import { useState } from 'react'
import { Bell, User, Search } from 'lucide-react'

export default function Header() {
  const [notifications] = useState(3)

  return (
    <header className="h-16 bg-crypto-darker border-b border-crypto-gray flex items-center justify-between px-6">
      {/* Search */}
      <div className="flex-1 max-w-md">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-crypto-light-gray w-4 h-4" />
          <input
            type="text"
            placeholder="Search trading pairs, strategies..."
            className="w-full pl-10 pr-4 py-2 bg-crypto-gray text-white rounded-lg border border-crypto-gray focus:border-crypto-blue focus:outline-none transition-colors"
          />
        </div>
      </div>

      {/* Status & Controls */}
      <div className="flex items-center space-x-4">
        {/* Status */}
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-crypto-green rounded-full animate-pulse"></div>
          <span className="text-sm text-crypto-light-gray">Live Trading</span>
        </div>

        {/* Notifications */}
        <div className="relative">
          <button className="p-2 text-crypto-light-gray hover:text-white transition-colors">
            <Bell className="w-5 h-5" />
            {notifications > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-crypto-red text-white text-xs rounded-full flex items-center justify-center">
                {notifications}
              </span>
            )}
          </button>
        </div>

        {/* User Profile */}
        <div className="flex items-center space-x-3">
          <div className="text-right">
            <div className="text-sm font-medium text-white">AI Trader</div>
            <div className="text-xs text-crypto-light-gray">Active Session</div>
          </div>
          <button className="w-8 h-8 bg-crypto-blue rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </button>
        </div>
      </div>
    </header>
  )
}