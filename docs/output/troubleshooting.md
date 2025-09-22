# Troubleshooting Guide

## Common Issues and Solutions


### API Connection Failed

**Problem:** Unable to connect to Bybit API

**Symptoms:**

- Connection timeout errors

- 401 Unauthorized responses

- Network connection refused


**Possible Causes:**

- Invalid API credentials

- Network connectivity issues

- API rate limits exceeded

- Bybit API maintenance


**Solutions:**

1. **Verify API Credentials**
   
   Check that your API key and secret are correct and active
   
   
   Steps:
   
   - Log into your Bybit account
   
   - Go to API Management
   
   - Verify your API key is active
   
   - Check API permissions (trading, read)
   
   - Regenerate API key if necessary
   
   
   
   
   ```bash
   # Test API connection
bot = TradingBot()
try:
    account_info = await bot.get_account_info()
    print("API connection successful!")
except Exception as e:
    print(f"API connection failed: {e}")
   ```
   

2. **Check Network Connectivity**
   
   Verify internet connection and firewall settings
   
   
   Steps:
   
   - Test internet connectivity
   
   - Check firewall settings
   
   - Try different network if possible
   
   - Verify DNS resolution
   
   
   
   
   ```bash
   import requests
try:
    response = requests.get("https://api.bybit.com/v5/market/time", timeout=10)
    print(f"Bybit API accessible: {response.status_code}")
except Exception as e:
    print(f"Network issue: {e}")
   ```
   


---

### Order Execution Failed

**Problem:** Orders are not being executed or are rejected

**Symptoms:**

- Order status shows 'rejected'

- Insufficient balance errors

- Position size too small errors


**Possible Causes:**

- Insufficient account balance

- Position size below minimum

- Risk management rules violated

- Market conditions (low liquidity)


**Solutions:**

1. **Check Account Balance**
   
   Verify sufficient balance for the trade
   
   
   Steps:
   
   - Check account balance
   
   - Consider trading fees
   
   - Verify margin requirements
   
   - Check for locked funds
   
   
   
   
   ```bash
   balance = await bot.get_balance()
print(f"Available balance: {balance['available']} USDT")
print(f"Required for trade: {order_value + fees} USDT")
   ```
   

2. **Adjust Position Size**
   
   Ensure position size meets minimum requirements
   
   
   Steps:
   
   - Check minimum order size for the symbol
   
   - Adjust quantity to meet requirements
   
   - Consider price precision
   
   
   
   
   ```bash
   # Get symbol info
symbol_info = await bot.get_symbol_info("BTCUSDT")
min_qty = symbol_info['min_order_qty']
print(f"Minimum order quantity: {min_qty}")
   ```
   


---

### Slow Performance or High Latency

**Problem:** Bot is running slowly or experiencing high latency

**Symptoms:**

- Delayed order execution

- High response times

- Missed trading opportunities


**Possible Causes:**

- Network latency

- Inefficient code execution

- Resource constraints

- API rate limiting


**Solutions:**

1. **Optimize Network Connection**
   
   Improve network performance and reduce latency
   
   
   Steps:
   
   - Use a VPS close to Bybit servers
   
   - Optimize internet connection
   
   - Use wired instead of WiFi
   
   - Consider dedicated hosting
   
   
   
   

2. **Enable Performance Optimizations**
   
   Configure bot for optimal performance
   
   
   Steps:
   
   - Enable caching for market data
   
   - Optimize indicator calculations
   
   - Use connection pooling
   
   - Reduce logging verbosity
   
   
   
   
   ```bash
   # Enable performance optimizations
bot = TradingBot({
    'performance': {
        'enable_caching': True,
        'cache_duration': 1000,  # ms
        'connection_pool_size': 10,
        'async_processing': True
    }
})
   ```
   


---
