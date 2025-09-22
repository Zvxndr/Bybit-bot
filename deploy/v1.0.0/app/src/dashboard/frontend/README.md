# Bybit Trading Dashboard Frontend

## Overview

Advanced Next.js 14 trading dashboard with real-time WebSocket streaming, ML insights visualization, and comprehensive system monitoring for the Bybit AI trading bot.

## Features

### 🎯 Core Dashboard Components
- **Real-time Trading Overview** - Live P&L, positions, and performance metrics
- **ML Insights Dashboard** - Model predictions, feature importance, and explainability
- **System Health Monitoring** - Resource usage, component status, and alerts
- **Advanced Charting** - Candlestick charts with technical indicators

### 🚀 Technical Stack
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom crypto theme
- **Real-time**: WebSocket client with automatic reconnection
- **Charts**: HTML5 Canvas-based trading charts
- **State Management**: React hooks with context providers

### 🌐 Architecture
- **Component-based**: Modular, reusable React components
- **WebSocket Integration**: Real-time data streaming from backend
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Type Safety**: Full TypeScript coverage with custom interfaces

## Project Structure

```
src/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout with dark theme
│   ├── page.tsx           # Main dashboard page
│   └── globals.css        # Global styles and animations
├── components/            # React Components
│   ├── layout/           # Layout components
│   │   ├── DashboardLayout.tsx
│   │   ├── Sidebar.tsx
│   │   └── Header.tsx
│   ├── trading/          # Trading dashboard
│   │   └── TradingOverview.tsx
│   ├── ml/               # ML insights
│   │   └── MLInsights.tsx
│   ├── system/           # System monitoring
│   │   └── SystemHealth.tsx
│   ├── charts/           # Chart components
│   │   └── TradingChart.tsx
│   └── providers/        # Context providers
│       └── WebSocketProvider.tsx
├── lib/                  # Utilities
│   └── api.ts           # API client
└── types/               # TypeScript definitions
    └── index.ts
```

## Key Components

### WebSocket Provider
- Automatic connection management with reconnection
- Topic-based subscription system
- Message broadcasting to dashboard components
- Connection status monitoring

### Trading Overview
- Real-time P&L tracking
- Position monitoring
- Performance statistics
- Recent trade history

### ML Insights
- Model prediction accuracy
- Feature importance visualization
- Risk score monitoring
- Prediction confidence levels

### System Health
- CPU, memory, and disk usage
- Component status monitoring
- Alert management
- Network latency tracking

## Configuration

### Environment Variables
```bash
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_WS_URL=ws://localhost:8001/ws
```

### Tailwind Theme
Custom crypto-themed color palette:
- `crypto-dark`: Primary dark background
- `crypto-darker`: Darker background variant
- `crypto-blue`: Primary accent color
- `crypto-green`: Success/profit color
- `crypto-red`: Error/loss color
- `crypto-yellow`: Warning color

## Development

### Prerequisites
- Node.js 18+ with npm/yarn
- Backend API running on port 8001

### Installation
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Development Server
```bash
npm run dev
```
Dashboard available at: http://localhost:3000

## Backend Integration

### API Endpoints
- `GET /trading/overview` - Trading statistics
- `GET /trading/positions` - Current positions
- `GET /ml/predictions` - ML model predictions
- `GET /health/system` - System health metrics

### WebSocket Topics
- `trading_overview` - Real-time trading data
- `ml_insights` - ML model updates
- `system_health` - System monitoring
- `alerts` - System alerts and notifications

## Features in Detail

### Real-time Data Streaming
- WebSocket connection to backend at `ws://localhost:8001/ws`
- Automatic reconnection on connection loss
- Topic-based data subscription
- Real-time UI updates without page refresh

### Responsive Design
- Mobile-first responsive layout
- Adaptive grid systems
- Touch-friendly interface elements
- Optimized for desktop and mobile

### Performance Optimizations
- Component-level state management
- Efficient WebSocket message handling
- Canvas-based charting for performance
- Optimized re-renders with React best practices

## Production Deployment

### Build Process
```bash
npm run build
```

### Docker Support
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Configuration
- Configure API URLs for production backend
- Set up proper CORS settings
- Configure WebSocket URLs for production

## Monitoring & Analytics

### Real-time Metrics
- Trading performance tracking
- ML model accuracy monitoring
- System resource utilization
- User interaction analytics

### Error Handling
- WebSocket connection error recovery
- API request failure handling
- Component error boundaries
- User-friendly error messages

## Security

### Best Practices
- No sensitive data in frontend code
- Secure WebSocket connections (WSS in production)
- API request validation
- CORS configuration

## Future Enhancements

### Planned Features
- Advanced charting with TradingView integration
- Real-time order book visualization
- Portfolio management interface
- Custom alert configuration
- Multi-timeframe analysis
- Social trading features

### Technical Improvements
- Progressive Web App (PWA) support
- Offline functionality
- Push notifications
- Advanced caching strategies
- Performance monitoring
- A/B testing framework

## Contributing

1. Follow TypeScript strict mode
2. Use Tailwind CSS for styling
3. Implement proper error handling
4. Add comprehensive TypeScript types
5. Test WebSocket connections
6. Maintain responsive design principles

## License

Private - Bybit Trading Bot Dashboard