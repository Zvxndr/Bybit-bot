'use client'

import { useEffect, useRef } from 'react'

interface TradingChartProps {
  data: Array<{
    timestamp: number
    open: number
    high: number
    low: number
    close: number
    volume: number
  }>
  symbol: string
  height?: number
}

export default function TradingChart({ data, symbol, height = 400 }: TradingChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current || !data.length) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = canvas.offsetWidth
    canvas.height = height

    // Clear canvas
    ctx.fillStyle = '#050C14'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw candlestick chart
    const padding = 40
    const chartWidth = canvas.width - padding * 2
    const chartHeight = canvas.height - padding * 2

    // Calculate price range
    const prices = data.flatMap(d => [d.high, d.low])
    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)
    const priceRange = maxPrice - minPrice

    // Draw candlesticks
    const candleWidth = chartWidth / data.length * 0.8
    const candleSpacing = chartWidth / data.length

    data.forEach((item, index) => {
      const x = padding + index * candleSpacing + candleSpacing / 2
      const openY = padding + (maxPrice - item.open) / priceRange * chartHeight
      const closeY = padding + (maxPrice - item.close) / priceRange * chartHeight
      const highY = padding + (maxPrice - item.high) / priceRange * chartHeight
      const lowY = padding + (maxPrice - item.low) / priceRange * chartHeight

      // Draw wick
      ctx.strokeStyle = '#6B7280'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, highY)
      ctx.lineTo(x, lowY)
      ctx.stroke()

      // Draw candle body
      const isGreen = item.close > item.open
      ctx.fillStyle = isGreen ? '#10B981' : '#EF4444'
      
      const candleHeight = Math.abs(closeY - openY)
      const candleY = Math.min(openY, closeY)
      
      ctx.fillRect(x - candleWidth / 2, candleY, candleWidth, Math.max(candleHeight, 1))
    })

    // Draw grid lines
    ctx.strokeStyle = '#374151'
    ctx.lineWidth = 0.5
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(padding + chartWidth, y)
      ctx.stroke()
    }

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding + (chartWidth / 10) * i
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, padding + chartHeight)
      ctx.stroke()
    }

    // Draw price labels
    ctx.fillStyle = '#9CA3AF'
    ctx.font = '12px monospace'
    for (let i = 0; i <= 5; i++) {
      const price = maxPrice - (priceRange / 5) * i
      const y = padding + (chartHeight / 5) * i
      ctx.fillText(price.toFixed(2), 5, y + 4)
    }

  }, [data, height])

  return (
    <div className="bg-crypto-gray rounded-lg p-4 border border-crypto-light-gray/20">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">{symbol} Price Chart</h3>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-crypto-light-gray">Real-time</span>
          <div className="w-2 h-2 bg-crypto-green rounded-full animate-pulse"></div>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: `${height}px` }}
      />
    </div>
  )
}