from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.src.api import rag_router, podcast_router

app = FastAPI(
    title="RAG API",
    description="API for building and managing FAISS indices",
    version="1.0.0"
)

# Create static directory if not exists
static_dir = Path("/Users/coconut/pp_backend/static")
static_dir.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag_router.router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(podcast_router.router, prefix="/api/v1/podcast", tags=["Podcast"])

@app.get("/")
async def root():
    return {
        "message": "RAG API is running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
