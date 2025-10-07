"""
WebSocket Manager for Real-time Dashboard Updates
Handles multiple client connections and real-time data broadcasting
"""

from fastapi import WebSocket, WebSocketDisconnect
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and real-time data broadcasting"""
    
    def __init__(self):
        # Active connections: client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Client subscriptions: client_id -> set of subscription types
        self.client_subscriptions: Dict[str, set] = {}
        
        # Message rate limiting
        self.client_message_counts: Dict[str, int] = {}
        self.rate_limit_window = 60  # seconds
        self.max_messages_per_window = 1000
        
        # Connection stats
        self.total_connections = 0
        self.total_messages_sent = 0
        self.connection_start_time = datetime.utcnow()
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            self.client_subscriptions[client_id] = {
                "trading_update", "ml_insights", "system_health", "performance_update"
            }
            self.client_message_counts[client_id] = 0
            self.total_connections += 1
            
            logger.info(f"✅ Client {client_id} connected. Total connections: {len(self.active_connections)}")
            
            # Send welcome message with available subscriptions
            await self.send_personal_message(client_id, "connection_success", {
                "message": "Connected to Bybit Trading Bot Dashboard",
                "client_id": client_id,
                "available_subscriptions": [
                    "trading_update", "ml_insights", "system_health", "performance_update"
                ],
                "server_time": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to connect client {client_id}: {e}")
            raise
    
    async def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].close()
            except:
                pass  # Connection might already be closed
            
            del self.active_connections[client_id]
            self.client_subscriptions.pop(client_id, None)
            self.client_message_counts.pop(client_id, None)
            
            logger.info(f"🔌 Client {client_id} disconnected. Remaining connections: {len(self.active_connections)}")
    
    async def disconnect_all(self):
        """Disconnect all active connections"""
        logger.info("🔄 Disconnecting all WebSocket clients...")
        
        for client_id in list(self.active_connections.keys()):
            await self.disconnect(client_id)
        
        logger.info("✅ All WebSocket connections closed")
    
    async def send_personal_message(self, client_id: str, message_type: str, data: Any):
        """Send message to specific client"""
        if client_id not in self.active_connections:
            return False
        
        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id
            }
            
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(message))
            
            self.total_messages_sent += 1
            self.client_message_counts[client_id] += 1
            
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(client_id)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send message to {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def broadcast(self, message_type: str, data: Any, subscription_filter: Optional[str] = None):
        """Broadcast message to all subscribed clients"""
        if not self.active_connections:
            return
        
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Get clients subscribed to this message type
        target_clients = []
        for client_id, subscriptions in self.client_subscriptions.items():
            if subscription_filter is None or subscription_filter in subscriptions:
                if message_type in subscriptions:
                    target_clients.append(client_id)
        
        if not target_clients:
            return
        
        # Send to all target clients concurrently
        tasks = []
        for client_id in target_clients:
            if client_id in self.active_connections:
                tasks.append(self._send_to_client(client_id, message))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful sends
            successful_sends = sum(1 for result in results if result is True)
            
            if successful_sends > 0:
                logger.debug(f"📡 Broadcast '{message_type}' to {successful_sends}/{len(tasks)} clients")
    
    async def _send_to_client(self, client_id: str, message: dict) -> bool:
        """Helper method to send message to individual client"""
        try:
            if client_id not in self.active_connections:
                return False
            
            # Rate limiting check
            if self.client_message_counts.get(client_id, 0) > self.max_messages_per_window:
                logger.warning(f"⚠️ Rate limit exceeded for client {client_id}")
                return False
            
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(message))
            
            self.total_messages_sent += 1
            self.client_message_counts[client_id] = self.client_message_counts.get(client_id, 0) + 1
            
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(client_id)
            return False
        except Exception as e:
            logger.error(f"❌ Send error to {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def handle_client_message(self, client_id: str, message: dict):
        """Handle incoming messages from clients"""
        try:
            message_type = message.get("type")
            data = message.get("data", {})
            
            if message_type == "subscribe":
                # Handle subscription changes
                subscriptions = set(data.get("subscriptions", []))
                valid_subscriptions = {
                    "trading_update", "ml_insights", "system_health", "performance_update"
                }
                
                # Filter to valid subscriptions only
                subscriptions = subscriptions.intersection(valid_subscriptions)
                self.client_subscriptions[client_id] = subscriptions
                
                await self.send_personal_message(client_id, "subscription_updated", {
                    "subscriptions": list(subscriptions),
                    "message": f"Subscribed to {len(subscriptions)} channels"
                })
                
            elif message_type == "unsubscribe":
                # Handle unsubscription
                to_remove = set(data.get("subscriptions", []))
                current_subs = self.client_subscriptions.get(client_id, set())
                new_subs = current_subs - to_remove
                
                self.client_subscriptions[client_id] = new_subs
                
                await self.send_personal_message(client_id, "subscription_updated", {
                    "subscriptions": list(new_subs),
                    "message": f"Unsubscribed from {len(to_remove)} channels"
                })
                
            elif message_type == "ping":
                # Handle ping/pong for keepalive
                await self.send_personal_message(client_id, "pong", {
                    "message": "pong",
                    "server_time": datetime.utcnow().isoformat()
                })
                
            elif message_type == "get_status":
                # Send current status to client
                status = await self.get_connection_status()
                await self.send_personal_message(client_id, "status_response", status)
                
            else:
                logger.warning(f"⚠️ Unknown message type from {client_id}: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling client message from {client_id}: {e}")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    async def get_connection_status(self) -> dict:
        """Get detailed connection status"""
        uptime = datetime.utcnow() - self.connection_start_time
        
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "uptime_seconds": uptime.total_seconds(),
            "uptime_human": str(uptime),
            "clients": {
                client_id: {
                    "subscriptions": list(subs),
                    "message_count": self.client_message_counts.get(client_id, 0)
                }
                for client_id, subs in self.client_subscriptions.items()
            }
        }
    
    async def reset_rate_limits(self):
        """Reset rate limiting counters (called periodically)"""
        self.client_message_counts = {
            client_id: 0 for client_id in self.client_message_counts
        }
        logger.debug("🔄 Rate limit counters reset")

# Periodic rate limit reset task
async def start_rate_limit_reset_task(websocket_manager: WebSocketManager):
    """Background task to reset rate limits periodically"""
    while True:
        await asyncio.sleep(websocket_manager.rate_limit_window)
        await websocket_manager.reset_rate_limits()