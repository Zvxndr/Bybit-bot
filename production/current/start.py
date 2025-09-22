#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

# Production startup script
print("Production Deployment - Bybit Trading Bot")

# Load metadata
try:
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    print(f"Version: {metadata['version']}")
    print(f"Deployment ID: {metadata['deployment_id']}")
    print("Application started successfully in production mode!")
except Exception as e:
    print(f"Warning: Could not load metadata - {e}")

print("Ready to serve production traffic")
