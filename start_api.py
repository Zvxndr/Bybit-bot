#!/usr/bin/env python3
"""
API Server Startup Script

Production startup script for the cryptocurrency trading bot API.
Handles configuration loading, logging setup, and graceful server startup.

Usage:
    python start_api.py [--config config.yaml] [--port 8000] [--host 0.0.0.0]
"""

import argparse
import sys
import os
import signal
from pathlib import Path
import asyncio
import uvicorn
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.bot.config.manager import ConfigurationManager
from src.bot.api.prediction_service import create_app
from src.bot.utils.logging import TradingLogger


class APIServer:
    """API Server manager with graceful shutdown."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigurationManager(config_path)
        self.logger = TradingLogger("APIServer")
        self.server = None
        self.shutdown_event = asyncio.Event()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Start the API server."""
        try:
            self.logger.info("Starting Cryptocurrency Trading Bot API...")
            self.logger.info(f"Server configuration: {host}:{port} with {workers} workers")
            
            # Create FastAPI app
            app = create_app(self.config_manager)
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                workers=workers,
                log_level="info",
                access_log=True,
                loop="asyncio"
            )
            
            # Create server
            self.server = uvicorn.Server(config)
            
            # Start server in background task
            server_task = asyncio.create_task(self.server.serve())
            
            self.logger.info(f"🚀 API Server running at http://{host}:{port}")
            self.logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
            self.logger.info(f"📊 Health Check: http://{host}:{port}/health")
            self.logger.info(f"📈 Metrics: http://{host}:{port}/metrics")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Graceful shutdown
            self.logger.info("Shutting down server...")
            self.server.should_exit = True
            await server_task
            
            self.logger.info("Server shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Run the server (blocking)."""
        try:
            asyncio.run(self.start_server(host, port, workers))
        except KeyboardInterrupt:
            self.logger.info("Server interrupted by user")
        except Exception as e:
            self.logger.error(f"Server failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot API Server")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Development mode with auto-reload
    if args.reload:
        print("🔄 Development mode: Auto-reload enabled")
        
        # Use uvicorn directly for development
        try:
            from src.bot.api.prediction_service import create_app
            config_manager = ConfigurationManager(args.config)
            app = create_app(config_manager)
            
            uvicorn.run(
                "src.bot.api.prediction_service:create_app",
                host=args.host,
                port=args.port,
                reload=True,
                factory=True
            )
        except ImportError as e:
            print(f"❌ Failed to import modules: {e}")
            print("Make sure you're running from the project root directory")
            sys.exit(1)
    else:
        # Production mode
        try:
            server = APIServer(args.config)
            server.run(args.host, args.port, args.workers)
        except ImportError as e:
            print(f"❌ Failed to import modules: {e}")
            print("Make sure you're running from the project root directory")
            sys.exit(1)


if __name__ == "__main__":
    main()