# 📈 Historical Data Integration Guide

## Overview
The Portfolio Performance chart has been enhanced with real historical market data capabilities. Users can now download and use actual market data from Bybit instead of simulated data for authentic performance analysis.

## Features

### **🎯 Intelligent Data Source Selection**
The system automatically determines the best data source:
1. **Real Historical Data** - When downloaded and available
2. **Simulated Data** - Fallback when no historical data exists

### **📊 Portfolio Performance Chart Enhancement**
- **Historical Data Controls** - Configurable download interface
- **Real Market Data** - Direct integration with Bybit API
- **Professional Metrics** - Authentic price movements and returns
- **Data Source Indicators** - Clear status of current data type

### **⚙️ Download Configuration**
- **Trading Pairs**: BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT
- **Timeframes**: 1h, 4h, 1d (daily), 1w (weekly)
- **Periods**: 30, 90, 180, 365 days
- **Progress Tracking**: Real-time download progress indicator

## User Interface

### **Portfolio Performance Section**
```
📈 Portfolio Performance                    [📥 Configure]
┌─────────────────────────────────────────────────────────┐
│ Historical Data Controls (Hidden by Default)           │
├─────────────────────────────────────────────────────────┤
│ Pair: [BTCUSDT ▼] Timeframe: [1d ▼] Period: [90 days ▼]│
│                    [📥 Download & Apply]                │
│ [❌ Cancel] [🗑️ Clear Data] [████████] 100% Complete    │
├─────────────────────────────────────────────────────────┤
│              📊 Performance Chart                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
ℹ️ Real data: BTCUSDT 1d (90 days) - Last updated: Oct 8, 2025
```

### **Data Status Indicators**
- **📊 Using simulated data** - Default state, no historical data
- **✅ Real data: BTCUSDT 1d (90 days)** - Active historical data with details
- **⚠️ Download in progress** - Data acquisition in progress
- **❌ Download failed** - Error state with retry option

## Technical Architecture

### **Historical Data Downloader (`historical_data_downloader.py`)**
```python
class HistoricalDataDownloader:
    """Downloads and manages historical market data from Bybit API"""
    
    # Core Methods
    download_klines(symbol, timeframe, days) -> Dict
    get_historical_performance(symbol, timeframe, limit) -> Dict  
    clear_historical_data() -> Dict
    get_data_summary() -> Dict
```

### **Database Schema**
```sql
CREATE TABLE historical_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,               -- Trading pair (BTCUSDT, etc.)
    timeframe TEXT NOT NULL,            -- Time interval (1h, 1d, etc.)
    timestamp INTEGER NOT NULL,         -- Unix timestamp (milliseconds)
    open_price REAL NOT NULL,           -- Opening price
    high_price REAL NOT NULL,           -- Highest price
    low_price REAL NOT NULL,            -- Lowest price  
    close_price REAL NOT NULL,          -- Closing price
    volume REAL NOT NULL,               -- Trading volume
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp) -- Prevent duplicates
);
```

### **API Endpoints**

#### **POST /api/historical-data/download**
Downloads historical market data from Bybit.
```json
Request: {
    "pair": "BTCUSDT",
    "timeframe": "1d", 
    "days": 90
}

Response: {
    "success": true,
    "message": "Downloaded 90 data points",
    "data_points": 90,
    "stored_count": 90
}
```

#### **GET /api/historical-data/performance**
Retrieves processed historical data for chart display.
```json
Response: {
    "success": true,
    "message": "Retrieved 89 data points",
    "data": [
        {
            "symbol": "BTCUSDT",
            "timestamp": 1728345600000,
            "close": 62450.50,
            "volume": 1234.56
        }
    ]
}
```

#### **POST /api/historical-data/clear**
Clears all historical data from database.
```json
Response: {
    "success": true,
    "message": "Cleared all historical data",
    "deleted_count": 90
}
```

#### **GET /api/historical-data/summary**
Gets summary of available historical datasets.
```json
Response: {
    "success": true,
    "summary": [
        {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "data_points": 90,
            "earliest": "2025-07-10T00:00:00",
            "latest": "2025-10-08T00:00:00"
        }
    ]
}
```

### **Performance Data Integration**
The `/api/performance` endpoint now intelligently selects data source:

1. **Check for Historical Data** - Queries `historical_data_downloader`
2. **Calculate Real Returns** - Processes actual price movements
3. **Fallback to Simulated** - Uses generated data if no real data available

## Usage Workflow

### **📥 Downloading Historical Data**
1. Click **"📥 Configure"** button in Portfolio Performance section
2. Select desired **Trading Pair**, **Timeframe**, and **Period**
3. Click **"📥 Download & Apply"**
4. Monitor progress bar during download
5. Chart automatically updates with real data
6. Status indicator shows data source details

### **🔄 Switching Data Sources**
- **Use Real Data**: Download historical data via controls
- **Use Simulated Data**: Click **"🗑️ Clear Data"** button
- **Update Data**: Download new dataset (overwrites existing)

### **📊 Chart Behavior**
- **Real Data**: Shows actual market price movements as daily returns
- **Simulated Data**: Shows generated performance based on strategy metrics
- **Empty State**: Displays empty chart with appropriate message

## Data Processing

### **Price to Return Conversion**
Historical prices are converted to daily returns for portfolio performance display:
```python
for i in range(1, len(prices)):
    daily_return = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
    returns.append(round(daily_return, 2))
```

### **Data Quality Assurance**
- **Duplicate Prevention**: Database constraints prevent duplicate entries
- **Error Handling**: Graceful fallback to simulated data on API failures
- **Data Validation**: Price and volume data validation during storage

## Benefits

### **🎯 Authentic Analysis**
- Real market data provides accurate performance insights
- Actual volatility patterns vs. simulated approximations  
- Professional-grade historical analysis capabilities

### **📈 Professional Standards**
- Industry-standard data sources (Bybit API)
- Comprehensive time series data storage
- Flexible timeframe and period selection

### **🔧 Easy Integration**
- Seamless fallback between real and simulated data
- No disruption to existing functionality
- User-controlled data management

## Troubleshooting

### **Common Issues**

#### **Download Fails**
- **Network Issues**: Check internet connection
- **API Limits**: Wait before retrying (Bybit rate limits)
- **Invalid Parameters**: Verify pair/timeframe combination exists

#### **No Data Displayed**  
- **Empty Dataset**: Download data for selected period
- **Database Issues**: Check `data/trading_bot.db` permissions
- **Import Errors**: Ensure `historical_data_downloader.py` is accessible

#### **Performance Issues**
- **Large Datasets**: Limit download period for faster processing
- **Database Size**: Clear old data periodically
- **Memory Usage**: Monitor system resources during downloads

### **Log Messages**
```
✅ Historical data database initialized
📡 Downloading BTCUSDT 1d data for 90 days...
✅ Successfully downloaded 90 data points, stored 90
📊 Using real historical data: 89 returns from BTCUSDT 1d
📊 Using simulated data: 24 returns
```

## Configuration

### **Supported Trading Pairs**
- **BTCUSDT** - Bitcoin/USDT (Most liquid)
- **ETHUSDT** - Ethereum/USDT  
- **BNBUSDT** - Binance Coin/USDT
- **ADAUSDT** - Cardano/USDT
- **SOLUSDT** - Solana/USDT

### **Supported Timeframes**
- **1h** - Hourly candles (High resolution)
- **4h** - 4-hour candles (Medium resolution)  
- **1d** - Daily candles (Standard resolution)
- **1w** - Weekly candles (Low resolution)

### **Period Limits**
- **30 days** - Short-term analysis
- **90 days** - Standard analysis (Recommended)
- **180 days** - Extended analysis
- **365 days** - Full year analysis

---

**The historical data integration provides professional-grade market data analysis while maintaining the simplicity and reliability of the existing system architecture.**