#!/usr/bin/env python3
"""
Simple FastAPI test to verify frontend serving works
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import uvicorn

app = FastAPI(title="Simple Test Server")

# Serve frontend
frontend_dir = Path(__file__).parent / "frontend"
print(f"Frontend directory: {frontend_dir}")
print(f"Frontend exists: {frontend_dir.exists()}")

if frontend_dir.exists():
    # Mount static files
    app.mount("/css", StaticFiles(directory=frontend_dir / "css"), name="css")
    app.mount("/js", StaticFiles(directory=frontend_dir / "js"), name="js")
    app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")
    
    @app.get("/")
    async def serve_frontend():
        index_path = frontend_dir / "index.html"
        print(f"Serving index.html from: {index_path}")
        return FileResponse(index_path)

@app.get("/test")
async def test_endpoint():
    return {"message": "Test endpoint working", "status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")