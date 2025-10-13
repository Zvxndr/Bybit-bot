import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../services/api';
import {
  TrendingUp,
  TrendingDown,
  Play,
  Pause,
  Square,
  BarChart3,
  Cpu,
  Sparkles,
  AlertTriangle
} from 'lucide-react';

const Dashboard = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('all');
  const [selectedStrategy, setSelectedStrategy] = useState(null);

  // Fetch strategy rankings
  const { 
    data: strategiesData, 
    isLoading: strategiesLoading, 
    refetch: refetchStrategies 
  } = useQuery({
    queryKey: ['strategies', selectedPeriod],
    queryFn: () => apiService.getStrategyRanking(selectedPeriod),
    refetchInterval: 5000 // Refresh every 5 seconds
  });

  // Fetch ML status
  const { 
    data: mlStatus, 
    isLoading: mlStatusLoading 
  } = useQuery({
    queryKey: ['mlStatus'],
    queryFn: apiService.getMLStatus,
    refetchInterval: 3000 // Refresh every 3 seconds
  });

  const strategies = strategiesData?.strategies || [];

  const getStatusIcon = (status) => {
    switch (status) {
      case 'live': return <Play className="w-4 h-4 text-green-400" />;
      case 'paper': return <BarChart3 className="w-4 h-4 text-blue-400" />;
      case 'backtest': return <Cpu className="w-4 h-4 text-yellow-400" />;
      default: return <Square className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusBadge = (status) => {
    const baseClasses = "px-2 py-1 rounded-full text-xs font-medium uppercase tracking-wide";
    switch (status) {
      case 'live': return `${baseClasses} bg-green-900/50 text-green-300 border border-green-500/30`;
      case 'paper': return `${baseClasses} bg-blue-900/50 text-blue-300 border border-blue-500/30`;
      case 'backtest': return `${baseClasses} bg-yellow-900/50 text-yellow-300 border border-yellow-500/30`;
      default: return `${baseClasses} bg-gray-900/50 text-gray-300 border border-gray-500/30`;
    }
  };

  const formatPercentage = (value) => {
    const formatted = value > 0 ? `+${value.toFixed(1)}%` : `${value.toFixed(1)}%`;
    return (
      <span className={value >= 0 ? 'text-green-400' : 'text-red-400'}>
        {value >= 0 ? <TrendingUp className="w-4 h-4 inline mr-1" /> : <TrendingDown className="w-4 h-4 inline mr-1" />}
        {formatted}
      </span>
    );
  };

  const MLStatusWidget = () => {
    if (mlStatusLoading) return <div className="animate-pulse">Loading ML Status...</div>;
    
    const status = mlStatus;
    const isHealthy = status?.health === 'optimal';

    return (
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Sparkles className="w-6 h-6 text-purple-400" />
            <h3 className="text-lg font-semibold text-white">ML Algorithm Status</h3>
          </div>
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
            isHealthy ? 'bg-green-900/50 text-green-300' : 'bg-red-900/50 text-red-300'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              isHealthy ? 'bg-green-400' : 'bg-red-400'
            } animate-pulse`} />
            <span className="text-sm font-medium uppercase">{status?.status || 'Unknown'}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">{status?.generation_rate || 0}</div>
            <div className="text-xs text-gray-400 uppercase tracking-wide">Strategies/Hour</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">{status?.processing || 0}</div>
            <div className="text-xs text-gray-400 uppercase tracking-wide">Processing</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">{status?.queue || 0}</div>
            <div className="text-xs text-gray-400 uppercase tracking-wide">In Queue</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-400">{status?.cpu_usage || 0}%</div>
            <div className="text-xs text-gray-400 uppercase tracking-wide">CPU Usage</div>
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Recent Activity</h4>
          <div className="max-h-32 overflow-y-auto space-y-2">
            {status?.recent_activity?.map((activity, index) => (
              <div key={index} className="flex items-start space-x-3 text-sm">
                <span className="text-gray-500 min-w-12">{activity.timestamp}</span>
                <span className="text-gray-300 flex-1">{activity.message}</span>
              </div>
            )) || <div className="text-gray-500 text-sm">No recent activity</div>}
          </div>
        </div>
      </div>
    );
  };

  if (strategiesLoading && strategies.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-gray-400">Loading strategy rankings...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Strategy Dashboard</h1>
          <p className="text-gray-400">ML-generated autonomous trading strategies ranked by performance</p>
        </div>
        
        {/* Period Selector */}
        <div className="flex space-x-2 mt-4 lg:mt-0">
          {['all', 'year', 'month', 'week'].map((period) => (
            <button
              key={period}
              onClick={() => setSelectedPeriod(period)}
              className={`px-4 py-2 rounded-lg text-sm font-medium uppercase tracking-wide transition-colors ${
                selectedPeriod === period
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {period === 'all' ? 'All Time' : period}
            </button>
          ))}
        </div>
      </div>

      {/* ML Status Widget */}
      <MLStatusWidget />

      {/* Strategy Rankings */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-700">
          <h2 className="text-xl font-semibold text-white">
            Strategy Rankings - {selectedPeriod === 'all' ? 'All Time' : selectedPeriod.charAt(0).toUpperCase() + selectedPeriod.slice(1)}
          </h2>
          <p className="text-gray-400 text-sm mt-1">
            {strategies.length} active strategies • Last updated: {strategiesData?.last_updated ? new Date(strategiesData.last_updated).toLocaleTimeString() : 'Never'}
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-750">
              <tr className="text-left">
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Rank</th>
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Strategy</th>
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Return</th>
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Sharpe</th>
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Win Rate</th>
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Max DD</th>
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Trades</th>
                <th className="px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {strategies.map((strategy, index) => (
                <tr 
                  key={strategy.id}
                  className="hover:bg-gray-750 cursor-pointer transition-colors"
                  onClick={() => setSelectedStrategy(strategy)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                        strategy.rank === 1 ? 'bg-yellow-500 text-black' :
                        strategy.rank === 2 ? 'bg-gray-400 text-black' :
                        strategy.rank === 3 ? 'bg-orange-600 text-white' :
                        'bg-gray-600 text-gray-300'
                      }`}>
                        {strategy.rank}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(strategy.status)}
                      <span className="text-white font-medium">{strategy.name}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className={getStatusBadge(strategy.status)}>
                      {strategy.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {formatPercentage(strategy.return_percent)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-white">
                    {strategy.sharpe_ratio.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-white">
                    {strategy.win_rate.toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-red-400">-{strategy.max_drawdown.toFixed(1)}%</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-white">
                    {strategy.total_trades.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-400 text-sm">
                    {new Date(strategy.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 text-center">
          <Cpu className="w-12 h-12 text-blue-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">ML Configuration</h3>
          <p className="text-gray-400 text-sm mb-4">Set minimum requirements and retirement metrics</p>
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors">
            Configure ML
          </button>
        </div>

        <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 text-center">
          <BarChart3 className="w-12 h-12 text-green-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">Data Management</h3>
          <p className="text-gray-400 text-sm mb-4">Monitor data collection and database health</p>
          <button className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors">
            Manage Data
          </button>
        </div>

        <div className="bg-gray-800 rounded-xl border border-gray-700 p-6 text-center">
          <AlertTriangle className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">System Health</h3>
          <p className="text-gray-400 text-sm mb-4">Review alerts and performance metrics</p>
          <button className="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg transition-colors">
            View Health
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;