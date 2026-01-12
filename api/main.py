"""
RecognitionSystem API

A modern REST API for multi-modal emotion and sentiment analysis.

Features:
- Audio emotion recognition (Wav2Vec2)
- Video facial emotion detection (DeepFace + MediaPipe)
- Text sentiment analysis (FinBERT with NER and aspect extraction)
- Async job processing with background tasks
- Comprehensive API documentation

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routers import audio_router, video_router, text_router
from api.routers.auth import router as auth_router
from api.models.schemas import JobResponse, JobStatus, HealthResponse
from api.services.analysis_service import analysis_service


# Track startup time
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # Startup
    print("=" * 50)
    print("RecognitionSystem API Starting...")
    print("=" * 50)
    
    # Load models (set to False for faster startup during development)
    load_audio = os.getenv("LOAD_AUDIO_MODEL", "true").lower() == "true"
    load_video = os.getenv("LOAD_VIDEO_MODEL", "true").lower() == "true"
    load_text = os.getenv("LOAD_TEXT_MODEL", "true").lower() == "true"
    
    try:
        analysis_service.load_models(
            audio=load_audio,
            video=load_video,
            text=load_text
        )
    except Exception as e:
        print(f"Warning: Failed to load some models: {e}")
    
    print("=" * 50)
    print("API Ready!")
    print("=" * 50)
    
    yield
    
    # Shutdown
    print("RecognitionSystem API Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RecognitionSystem API",
    description="""
## Multi-Modal Emotion & Sentiment Analysis API

This API provides advanced emotion and sentiment analysis capabilities:

### Audio Analysis
- Speech Emotion Recognition using Wav2Vec2 transformer models
- Segment-based analysis with confidence scores
- Support for WAV, MP3, FLAC, OGG formats

### Video Analysis  
- Facial emotion detection using DeepFace
- Optional MediaPipe face mesh for enhanced tracking
- YouTube video download and analysis
- Support for MP4, AVI, MOV, MKV formats

### Text Analysis
- Financial sentiment analysis using FinBERT
- Named Entity Recognition (NER)
- Aspect-based sentiment extraction
- PDF document processing

### Job Management
- Async processing with background tasks
- Real-time progress tracking
- Job status and result retrieval
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(audio_router, prefix="/api/v1")
app.include_router(video_router, prefix="/api/v1")
app.include_router(text_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")


# ============== Core Endpoints ==============

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "name": "RecognitionSystem API",
        "version": "2.0.0",
        "description": "Multi-Modal Emotion & Sentiment Analysis",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status, loaded models, and uptime.
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded=analysis_service.get_models_status(),
        uptime=time.time() - START_TIME
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Get job status and results.
    
    Returns the current status, progress, and results (if completed) for a job.
    """
    job = analysis_service.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job["id"],
        status=JobStatus(job["status"]),
        analysis_type=job["type"],
        message=f"Job {job['status']}",
        progress=job["progress"],
        result=job["result"],
        error=job["error"],
        created_at=job["created_at"],
        completed_at=job["completed_at"]
    )


@app.get("/api/v1/jobs", tags=["Jobs"])
async def list_jobs(
    status: Optional[str] = None,
    analysis_type: Optional[str] = None,
    limit: int = 50
):
    """
    List all jobs with optional filtering.
    
    Parameters:
    - status: Filter by status (pending, processing, completed, failed)
    - analysis_type: Filter by type (audio, video, text)
    - limit: Maximum number of jobs to return
    """
    jobs = list(analysis_service.jobs.values())
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    if analysis_type:
        jobs = [j for j in jobs if j["type"] == analysis_type]
    
    # Sort by created_at descending
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "total": len(jobs),
        "jobs": jobs[:limit]
    }


@app.delete("/api/v1/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """
    Delete a job and its results.
    """
    job = analysis_service.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del analysis_service.jobs[job_id]
    
    return {"message": f"Job {job_id} deleted"}


# ============== Error Handlers ==============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

