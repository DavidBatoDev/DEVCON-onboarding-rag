# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv() 
from app.api.v1.routes import router as v1_router
import os
import socket
import uvicorn
import time
from contextlib import asynccontextmanager
from datetime import timedelta



# Lifespan events for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting DEVCON RAG API...")
    
    # Track startup time
    app.state.start_time = time.time()
    
    yield
    
    # Shutdown
    print("🔌 Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root_check():
    return {"status": "ok", "message": "DEVCON RAG API"}

@app.get("/health")
def health_check():
    uptime = time.time() - app.state.start_time
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "uptime_human": str(timedelta(seconds=int(uptime)))
    }



# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")

def find_free_port(start_port=8000, end_port=9000):
    """Find a free port within the given range"""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return start_port  # Fallback

if __name__ == "__main__":
    # Find available port
    port = find_free_port()
    print(f"🚀 Starting server on port {port}")
    
    # Run with auto-reload using import string format
    uvicorn.run(
        "main:app",  # Use import string format
        host="127.0.0.1",
        port=port,
        reload=True,
        timeout_keep_alive=300
    )