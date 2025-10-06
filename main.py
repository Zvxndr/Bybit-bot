#!/usr/bin/env python3
"""
Main Entry Point for DigitalOcean Deployment
Explicitly runs the production AI pipeline system
"""

if __name__ == "__main__":
    print("🚀 Starting Production AI Pipeline System...")
    print("📁 Entry point: main.py")
    print("🎯 Target: production_ai_pipeline.py")
    
    # Import and run the production pipeline
    import production_ai_pipeline
    
    # The production_ai_pipeline.py has its own if __name__ == "__main__": block
    # So we don't need to do anything else here - it will start automatically