#!/usr/bin/env python3
"""
Dashboard Startup Script

Launch script for the Streamlit monitoring dashboard.
Provides comprehensive visualization of ML model performance,
drift detection, system health, and business metrics.

Usage:
    python start_dashboard.py [--port 8501] [--config config.yaml]
"""

import argparse
import sys
import subprocess
from pathlib import Path
import os

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for dashboard."""
    parser = argparse.ArgumentParser(description="ML Trading Bot Monitoring Dashboard")
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    
    parser.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark"],
        default="dark",
        help="Dashboard theme (default: dark)"
    )
    
    args = parser.parse_args()
    
    # Set configuration environment variable if provided
    if args.config:
        os.environ['TRADING_BOT_CONFIG'] = args.config
    
    # Build streamlit command
    dashboard_path = project_root / "src" / "bot" / "dashboard" / "monitoring_dashboard.py"
    
    cmd = [
        "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--theme.base", args.theme,
        "--server.headless", "false",
        "--server.runOnSave", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"🚀 Starting ML Trading Bot Dashboard...")
    print(f"📊 Dashboard URL: http://{args.host}:{args.port}")
    print(f"🎨 Theme: {args.theme}")
    
    if args.config:
        print(f"⚙️ Configuration: {args.config}")
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("---")
    
    try:
        # Run Streamlit
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("👋 Dashboard shutdown by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()