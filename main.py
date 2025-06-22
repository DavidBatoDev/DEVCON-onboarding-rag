# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from app.api.v1.routes import router as v1_router
import os
import uvicorn
import time
from contextlib import asynccontextmanager
from datetime import timedelta

# Lifespan events for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting DEVCON RAG API...")
    app.state.start_time = time.time()
    yield
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

# Register API routes
app.include_router(v1_router, prefix="/api/v1")

if __name__ == "__main__":
    # Use Render's port or default to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300
    )
