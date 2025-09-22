'use client'

import { useState, useEffect } from 'react'
import { useWebSocket } from '@/components/providers/WebSocketProvider'
import { Activity, Cpu, HardDrive, Wifi, AlertTriangle, CheckCircle } from 'lucide-react'

interface SystemData {
  cpuUsage: number
  memoryUsage: number
  diskUsage: number
  networkLatency: number
  uptime: string
  alertsCount: number
  componentsStatus: {
    [key: string]: 'healthy' | 'warning' | 'error'
  }
}

export default function SystemHealth() {
  const { isConnected, subscribe, lastMessage } = useWebSocket()
  const [systemData, setSystemData] = useState<SystemData>({
    cpuUsage: 45.2,
    memoryUsage: 67.8,
    diskUsage: 23.4,
    networkLatency: 12.5,
    uptime: '3d 14h 26m',
    alertsCount: 2,
    componentsStatus: {
      'Trading Engine': 'healthy',
      'ML Models': 'healthy',
      'Database': 'warning',
      'WebSocket': 'healthy',
      'API Gateway': 'healthy'
    }
  })

  useEffect(() => {
    if (isConnected) {
      subscribe('system_health')
      subscribe('performance_metrics')
      subscribe('alerts')
    }
  }, [isConnected, subscribe])

  useEffect(() => {
    if (lastMessage && lastMessage.type === 'system_data') {
      setSystemData(lastMessage.data)
    }
  }, [lastMessage])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-crypto-green'
      case 'warning': return 'text-crypto-yellow'
      case 'error': return 'text-crypto-red'
      default: return 'text-crypto-light-gray'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="w-5 h-5" />
      case 'warning': return <AlertTriangle className="w-5 h-5" />
      case 'error': return <AlertTriangle className="w-5 h-5" />
      default: return <Activity className="w-5 h-5" />
    }
  }

  const getUsageColor = (usage: number) => {
    if (usage < 50) return 'bg-crypto-green'
    if (usage < 80) return 'bg-crypto-yellow'
    return 'bg-crypto-red'
  }

  return (
    <div className="space-y-6">
      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">CPU Usage</p>
              <p className="text-2xl font-bold text-white">
                {systemData.cpuUsage.toFixed(1)}%
              </p>
            </div>
            <Cpu className="h-8 w-8 text-crypto-blue" />
          </div>
          <div className="mt-4">
            <div className="w-full bg-crypto-darker rounded-full h-2">
              <div
                className={`h-2 rounded-full ${getUsageColor(systemData.cpuUsage)}`}
                style={{ width: `${systemData.cpuUsage}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Memory Usage</p>
              <p className="text-2xl font-bold text-white">
                {systemData.memoryUsage.toFixed(1)}%
              </p>
            </div>
            <Activity className="h-8 w-8 text-crypto-green" />
          </div>
          <div className="mt-4">
            <div className="w-full bg-crypto-darker rounded-full h-2">
              <div
                className={`h-2 rounded-full ${getUsageColor(systemData.memoryUsage)}`}
                style={{ width: `${systemData.memoryUsage}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Disk Usage</p>
              <p className="text-2xl font-bold text-white">
                {systemData.diskUsage.toFixed(1)}%
              </p>
            </div>
            <HardDrive className="h-8 w-8 text-crypto-yellow" />
          </div>
          <div className="mt-4">
            <div className="w-full bg-crypto-darker rounded-full h-2">
              <div
                className={`h-2 rounded-full ${getUsageColor(systemData.diskUsage)}`}
                style={{ width: `${systemData.diskUsage}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-crypto-light-gray">Network Latency</p>
              <p className="text-2xl font-bold text-white">
                {systemData.networkLatency.toFixed(1)}ms
              </p>
            </div>
            <Wifi className="h-8 w-8 text-crypto-green" />
          </div>
          <div className="mt-4">
            <p className="text-xs text-crypto-light-gray">Avg. response time</p>
          </div>
        </div>
      </div>

      {/* Component Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <h3 className="text-lg font-semibold text-white mb-4">Component Status</h3>
          <div className="space-y-3">
            {Object.entries(systemData.componentsStatus).map(([component, status]) => (
              <div key={component} className="flex items-center justify-between p-3 bg-crypto-darker rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={getStatusColor(status)}>
                    {getStatusIcon(status)}
                  </div>
                  <span className="text-white font-medium">{component}</span>
                </div>
                <span className={`text-sm font-medium capitalize ${getStatusColor(status)}`}>
                  {status}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* System Information */}
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <h3 className="text-lg font-semibold text-white mb-4">System Information</h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Uptime</span>
              <span className="text-white font-medium">{systemData.uptime}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Active Alerts</span>
              <span className={`font-medium ${systemData.alertsCount > 0 ? 'text-crypto-red' : 'text-crypto-green'}`}>
                {systemData.alertsCount}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Last Health Check</span>
              <span className="text-white font-medium">30s ago</span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">System Load</span>
              <span className="text-white font-medium">0.45</span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Available RAM</span>
              <span className="text-white font-medium">2.4 GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-crypto-light-gray">Free Disk Space</span>
              <span className="text-white font-medium">127 GB</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      {systemData.alertsCount > 0 && (
        <div className="bg-crypto-gray rounded-lg p-6 border border-crypto-light-gray/20">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Alerts</h3>
          <div className="space-y-3">
            <div className="flex items-center p-4 bg-crypto-darker rounded-lg border-l-4 border-crypto-yellow">
              <AlertTriangle className="w-5 h-5 text-crypto-yellow mr-3" />
              <div className="flex-1">
                <p className="text-white font-medium">Database Connection Warning</p>
                <p className="text-crypto-light-gray text-sm">Increased response time detected</p>
              </div>
              <span className="text-crypto-light-gray text-sm">2 min ago</span>
            </div>
            
            <div className="flex items-center p-4 bg-crypto-darker rounded-lg border-l-4 border-crypto-red">
              <AlertTriangle className="w-5 h-5 text-crypto-red mr-3" />
              <div className="flex-1">
                <p className="text-white font-medium">Memory Usage High</p>
                <p className="text-crypto-light-gray text-sm">Memory usage exceeded 80% threshold</p>
              </div>
              <span className="text-crypto-light-gray text-sm">15 min ago</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}