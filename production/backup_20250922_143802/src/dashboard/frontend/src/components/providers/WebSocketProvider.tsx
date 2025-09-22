'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

interface WebSocketContextType {
  socket: WebSocket | null
  isConnected: boolean
  lastMessage: any
  sendMessage: (message: any) => void
  subscribe: (topic: string) => void
  unsubscribe: (topic: string) => void
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

interface WebSocketProviderProps {
  children: ReactNode
}

export default function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<any>(null)
  const [subscriptions, setSubscriptions] = useState<Set<string>>(new Set())

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001/ws'
    
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(wsUrl)
        
        ws.onopen = () => {
          console.log('WebSocket connected')
          setIsConnected(true)
          setSocket(ws)
          
          // Resubscribe to previous subscriptions
          subscriptions.forEach(topic => {
            ws.send(JSON.stringify({ action: 'subscribe', topic }))
          })
        }
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            setLastMessage(data)
            console.log('WebSocket message:', data)
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }
        
        ws.onclose = () => {
          console.log('WebSocket disconnected')
          setIsConnected(false)
          setSocket(null)
          
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000)
        }
        
        ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          setIsConnected(false)
        }
        
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error)
        setTimeout(connectWebSocket, 3000)
      }
    }

    connectWebSocket()

    return () => {
      if (socket) {
        socket.close()
      }
    }
  }, [])

  const sendMessage = (message: any) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify(message))
    }
  }

  const subscribe = (topic: string) => {
    setSubscriptions(prev => new Set([...prev, topic]))
    if (socket && isConnected) {
      socket.send(JSON.stringify({ action: 'subscribe', topic }))
    }
  }

  const unsubscribe = (topic: string) => {
    setSubscriptions(prev => {
      const newSet = new Set(prev)
      newSet.delete(topic)
      return newSet
    })
    if (socket && isConnected) {
      socket.send(JSON.stringify({ action: 'unsubscribe', topic }))
    }
  }

  const value: WebSocketContextType = {
    socket,
    isConnected,
    lastMessage,
    sendMessage,
    subscribe,
    unsubscribe
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

export function useWebSocket() {
  const context = useContext(WebSocketContext)
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}