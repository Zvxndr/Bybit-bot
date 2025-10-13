import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../services/api';
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  Wifi,
  WifiOff,
  TrendingUp,
  Mail,
  Newspaper,
  BarChart3,
  TestTube,
  Send
} from 'lucide-react';

const ApiStatus = () => {
  const [testResults, setTestResults] = useState({});

  // Fetch API status
  const { data: apiStatus, isLoading, refetch } = useQuery({
    queryKey: ['apiStatus'],
    queryFn: apiService.getApiStatus,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch news sentiment
  const { data: newsSentiment } = useQuery({
    queryKey: ['newsSentiment'],
    queryFn: apiService.getNewsSentiment,
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  // Fetch correlation data
  const { data: correlationData } = useQuery({
    queryKey: ['correlationData'],
    queryFn: apiService.getCorrelationData,
    refetchInterval: 60000, // Refresh every minute
  });

  // Fetch future markets status
  const { data: futureMarkets } = useQuery({
    queryKey: ['futureMarkets'],
    queryFn: apiService.getFutureMarketsStatus,
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  const getStatusIcon = (status) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'disconnected':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'not_configured':
        return <AlertCircle className="w-5 h-5 text-yellow-400" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected':
        return 'text-green-400 bg-green-900/30 border-green-500/30';
      case 'disconnected':
        return 'text-red-400 bg-red-900/30 border-red-500/30';
      case 'not_configured':
        return 'text-yellow-400 bg-yellow-900/30 border-yellow-500/30';
      default:
        return 'text-gray-400 bg-gray-900/30 border-gray-500/30';
    }
  };

  const testConnection = async (apiName, isTestnet = false) => {
    try {
      let result;
      if (apiName === 'bybit') {
        result = await apiService.testBybitConnection(isTestnet);
      }
      
      setTestResults(prev => ({
        ...prev,
        [`${apiName}_${isTestnet ? 'testnet' : 'live'}`]: {
          success: result.success,
          message: result.message,
          timestamp: new Date().toLocaleTimeString()
        }
      }));
      
      // Refresh main status after test
      setTimeout(() => refetch(), 1000);
    } catch (error) {
      setTestResults(prev => ({
        ...prev,
        [`${apiName}_${isTestnet ? 'testnet' : 'live'}`]: {
          success: false,
          message: error.message,
          timestamp: new Date().toLocaleTimeString()
        }
      }));
    }
  };

  const sendDailyReport = async () => {
    try {
      const result = await apiService.sendDailyReport();
      alert(result.success ? 'Daily report sent successfully!' : result.message);
    } catch (error) {
      alert('Failed to send daily report: ' + error.message);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="animate-pulse text-gray-400">Loading API status...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">API Status Monitor</h2>
        <button
          onClick={() => refetch()}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          Refresh Status
        </button>
      </div>

      {/* Trading APIs */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Wifi className="w-5 h-5 mr-2 text-blue-400" />
          Trading APIs
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Bybit Testnet */}
          <div className="border border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                {getStatusIcon(apiStatus?.trading_apis?.bybit_testnet?.status)}
                <span className="text-white font-medium">Bybit Testnet</span>
              </div>
              <span className={`px-2 py-1 rounded text-xs ${getStatusColor(apiStatus?.trading_apis?.bybit_testnet?.status)}`}>
                {apiStatus?.trading_apis?.bybit_testnet?.status}
              </span>
            </div>
            <p className="text-gray-400 text-sm mb-3">
              {apiStatus?.trading_apis?.bybit_testnet?.purpose}
            </p>
            <div className="flex space-x-2">
              <button
                onClick={() => testConnection('bybit', true)}
                className="flex-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
              >
                Test Connection
              </button>
            </div>
            {testResults.bybit_testnet && (
              <div className={`mt-2 p-2 rounded text-xs ${testResults.bybit_testnet.success ? 'bg-green-900/30 text-green-300' : 'bg-red-900/30 text-red-300'}`}>
                {testResults.bybit_testnet.message} ({testResults.bybit_testnet.timestamp})
              </div>
            )}
          </div>

          {/* Bybit Live */}
          <div className="border border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                {getStatusIcon(apiStatus?.trading_apis?.bybit_live?.status)}
                <span className="text-white font-medium">Bybit Live</span>
              </div>
              <span className={`px-2 py-1 rounded text-xs ${getStatusColor(apiStatus?.trading_apis?.bybit_live?.status)}`}>
                {apiStatus?.trading_apis?.bybit_live?.status}
              </span>
            </div>
            <p className="text-gray-400 text-sm mb-3">
              {apiStatus?.trading_apis?.bybit_live?.purpose}
            </p>
            <div className="flex space-x-2">
              <button
                onClick={() => testConnection('bybit', false)}
                className="flex-1 px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
              >
                Test Connection
              </button>
            </div>
            {testResults.bybit_live && (
              <div className={`mt-2 p-2 rounded text-xs ${testResults.bybit_live.success ? 'bg-green-900/30 text-green-300' : 'bg-red-900/30 text-red-300'}`}>
                {testResults.bybit_live.message} ({testResults.bybit_live.timestamp})
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Correlation APIs */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-purple-400" />
          Cross-Exchange Correlation (Optional)
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* OKX */}
          <div className="border border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                {getStatusIcon(apiStatus?.correlation_apis?.okx?.status)}
                <span className="text-white font-medium">OKX</span>
              </div>
              <span className={`px-2 py-1 rounded text-xs ${getStatusColor(apiStatus?.correlation_apis?.okx?.status)}`}>
                {apiStatus?.correlation_apis?.okx?.status}
              </span>
            </div>
            <p className="text-gray-400 text-sm">
              {apiStatus?.correlation_apis?.okx?.purpose}
            </p>
          </div>

          {/* Binance */}
          <div className="border border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                {getStatusIcon(apiStatus?.correlation_apis?.binance?.status)}
                <span className="text-white font-medium">Binance</span>
              </div>
              <span className={`px-2 py-1 rounded text-xs ${getStatusColor(apiStatus?.correlation_apis?.binance?.status)}`}>
                {apiStatus?.correlation_apis?.binance?.status}
              </span>
            </div>
            <p className="text-gray-400 text-sm">
              {apiStatus?.correlation_apis?.binance?.purpose}
            </p>
          </div>
        </div>

        {/* Correlation Data Display */}
        {correlationData && (
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="text-white font-medium mb-3">Current Correlations</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-blue-400 font-bold text-xl">{(correlationData.correlations?.bybit_okx?.btc_correlation * 100)?.toFixed(1)}%</div>
                <div className="text-gray-400 text-sm">BTC (Bybit-OKX)</div>
              </div>
              <div>
                <div className="text-purple-400 font-bold text-xl">{(correlationData.correlations?.bybit_okx?.eth_correlation * 100)?.toFixed(1)}%</div>
                <div className="text-gray-400 text-sm">ETH (Bybit-OKX)</div>
              </div>
              <div>
                <div className="text-green-400 font-bold text-xl">{(correlationData.correlations?.bybit_binance?.btc_correlation * 100)?.toFixed(1)}%</div>
                <div className="text-gray-400 text-sm">BTC (Bybit-Binance)</div>
              </div>
              <div>
                <div className="text-yellow-400 font-bold text-xl">{(correlationData.correlations?.bybit_binance?.eth_correlation * 100)?.toFixed(1)}%</div>
                <div className="text-gray-400 text-sm">ETH (Bybit-Binance)</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Data Services */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2 text-green-400" />
          Data Services (Optional)
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* News Sentiment */}
          <div className="border border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Newspaper className="w-4 h-4 text-orange-400" />
                <span className="text-white font-medium">News Sentiment</span>
              </div>
              <span className={`px-2 py-1 rounded text-xs ${getStatusColor(apiStatus?.data_services?.news_sentiment?.status)}`}>
                {apiStatus?.data_services?.news_sentiment?.status}
              </span>
            </div>
            <p className="text-gray-400 text-sm mb-3">Market sentiment analysis</p>
            
            {newsSentiment && newsSentiment.sentiment && (
              <div className="bg-gray-700 rounded p-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-300">Overall Sentiment</span>
                  <span className={`font-bold ${newsSentiment.sentiment.sentiment_label === 'Bullish' ? 'text-green-400' : newsSentiment.sentiment.sentiment_label === 'Bearish' ? 'text-red-400' : 'text-yellow-400'}`}>
                    {newsSentiment.sentiment.sentiment_label}
                  </span>
                </div>
                <div className="text-2xl font-bold text-white">
                  {(newsSentiment.sentiment.overall_score * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-400">
                  {newsSentiment.sentiment.sources_analyzed} sources analyzed
                </div>
              </div>
            )}
          </div>

          {/* Email Reporting */}
          <div className="border border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Mail className="w-4 h-4 text-blue-400" />
                <span className="text-white font-medium">Email Reports</span>
              </div>
              <span className={`px-2 py-1 rounded text-xs ${getStatusColor(apiStatus?.data_services?.email_reporting?.status)}`}>
                {apiStatus?.data_services?.email_reporting?.status}
              </span>
            </div>
            <p className="text-gray-400 text-sm mb-3">Daily financial reports</p>
            
            <button
              onClick={sendDailyReport}
              disabled={apiStatus?.data_services?.email_reporting?.status !== 'connected'}
              className="flex items-center justify-center w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white text-sm rounded transition-colors"
            >
              <Send className="w-4 h-4 mr-2" />
              Send Test Report
            </button>
          </div>
        </div>
      </div>

      {/* Future Markets */}
      {futureMarkets && (
        <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <TestTube className="w-5 h-5 mr-2 text-yellow-400" />
            Future Market Expansion
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(futureMarkets.future_integrations || {}).map(([key, integration]) => (
              <div key={key} className="border border-gray-600 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white font-medium capitalize">{key.replace('_', ' ')}</span>
                  <span className={`px-2 py-1 rounded text-xs ${getStatusColor(integration.status)}`}>
                    {integration.status}
                  </span>
                </div>
                <p className="text-gray-400 text-sm">{integration.purpose}</p>
              </div>
            ))}
          </div>
          
          <div className="mt-4 p-3 bg-yellow-900/30 border border-yellow-500/30 rounded-lg">
            <p className="text-yellow-300 text-sm">
              <strong>Coming Soon:</strong> {futureMarkets.next_markets?.join(', ')}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ApiStatus;