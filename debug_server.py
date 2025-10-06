#!/usr/bin/env python3
"""
Simple test server to verify what DigitalOcean is actually running
This will help us debug the deployment issue
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import sys
from pathlib import Path

app = FastAPI(title="Debug Test Server")

@app.get("/")
def root():
    return {"message": "Debug test server is running", "file": __file__}

@app.get("/debug")
def debug():
    return {
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_path": sys.path,
        "environment_vars": {
            "PORT": os.getenv("PORT"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT"),
            "PYTHONPATH": os.getenv("PYTHONPATH")
        },
        "files_in_directory": list(Path.cwd().glob("*.py"))[:10]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🔍 Debug server starting on port {port}")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"🐍 Python executable: {sys.executable}")
    uvicorn.run(app, host="0.0.0.0", port=port)