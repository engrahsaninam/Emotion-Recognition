"""
Video Analysis Router

Endpoints for video/facial emotion recognition.
"""

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from ..models.schemas import VideoAnalysisRequest, VideoAnalysisResponse, JobResponse, JobStatus
from ..services.analysis_service import analysis_service

router = APIRouter(prefix="/video", tags=["Video Analysis"])


@router.post("/analyze", response_model=JobResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    frame_interval: int = 30,
    detector_backend: str = "opencv"
):
    """
    Analyze facial emotions in a video file.
    
    Uploads the video file and starts async processing.
    Returns a job ID that can be used to check progress and get results.
    
    Supported formats: MP4, AVI, MOV, MKV
    
    Detector backends:
    - opencv (default, fastest)
    - mtcnn (more accurate)
    - retinaface (best accuracy)
    - mediapipe (with face mesh)
    """
    # Validate file type
    allowed_types = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: MP4, AVI, MOV, MKV"
        )
    
    # Save file
    content = await file.read()
    file_path = analysis_service.save_upload(content, file.filename)
    
    # Create job
    job_id = analysis_service.create_job("video")
    
    # Start background processing
    background_tasks.add_task(
        analysis_service.analyze_video,
        job_id=job_id,
        file_path=file_path,
        frame_interval=frame_interval,
        detector_backend=detector_backend
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        analysis_type="video",
        message="Video analysis started. Use /jobs/{job_id} to check progress."
    )


@router.post("/analyze-sync", response_model=VideoAnalysisResponse)
async def analyze_video_sync(
    file: UploadFile = File(...),
    frame_interval: int = 30,
    detector_backend: str = "opencv"
):
    """
    Analyze facial emotions in a video file synchronously.
    
    Waits for processing to complete before returning results.
    Use for smaller files or when immediate results are needed.
    """
    # Validate file type
    allowed_types = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: MP4, AVI, MOV, MKV"
        )
    
    # Save file
    content = await file.read()
    file_path = analysis_service.save_upload(content, file.filename)
    
    # Create job
    job_id = analysis_service.create_job("video")
    
    try:
        result = await analysis_service.analyze_video(
            job_id=job_id,
            file_path=file_path,
            frame_interval=frame_interval,
            detector_backend=detector_backend
        )
        return VideoAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-youtube", response_model=JobResponse)
async def analyze_youtube(
    background_tasks: BackgroundTasks,
    youtube_url: str,
    frame_interval: int = 30,
    detector_backend: str = "opencv"
):
    """
    Download and analyze a YouTube video.
    
    Downloads the video using yt-dlp and analyzes facial emotions.
    """
    import yt_dlp
    import tempfile
    import os
    
    # Create job
    job_id = analysis_service.create_job("video")
    
    try:
        # Download video
        output_path = os.path.join(analysis_service.upload_dir, f"{job_id}.mp4")
        
        ydl_opts = {
            'outtmpl': output_path,
            'format': 'best[ext=mp4]/best',
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Start background processing
        background_tasks.add_task(
            analysis_service.analyze_video,
            job_id=job_id,
            file_path=output_path,
            frame_interval=frame_interval,
            detector_backend=detector_backend
        )
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            analysis_type="video",
            message="YouTube video downloaded. Analysis started."
        )
        
    except Exception as e:
        analysis_service.update_job(job_id, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")

