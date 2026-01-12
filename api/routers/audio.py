"""
Audio Analysis Router

Endpoints for audio emotion recognition.
"""

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from ..models.schemas import AudioAnalysisRequest, AudioAnalysisResponse, JobResponse, JobStatus
from ..services.analysis_service import analysis_service

router = APIRouter(prefix="/audio", tags=["Audio Analysis"])


@router.post("/analyze", response_model=JobResponse)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    segment_duration: float = 10.0,
    process_all: bool = True,
    start_time: float = 0,
    end_time: float = None
):
    """
    Analyze emotions in an audio file.
    
    Uploads the audio file and starts async processing.
    Returns a job ID that can be used to check progress and get results.
    
    Supported formats: WAV, MP3, FLAC, OGG
    """
    # Validate file type by MIME type or extension
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg', 'audio/x-wav', 'audio/wave']
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    filename_lower = file.filename.lower() if file.filename else ""
    has_valid_extension = any(filename_lower.endswith(ext) for ext in allowed_extensions)
    has_valid_mime = file.content_type in allowed_types
    
    if not (has_valid_extension or has_valid_mime):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: WAV, MP3, FLAC, OGG"
        )
    
    # Save file
    content = await file.read()
    file_path = analysis_service.save_upload(content, file.filename)
    
    # Create job
    job_id = analysis_service.create_job("audio")
    
    # Start background processing
    background_tasks.add_task(
        analysis_service.analyze_audio,
        job_id=job_id,
        file_path=file_path,
        segment_duration=segment_duration,
        process_all=process_all,
        start_time=start_time,
        end_time=end_time
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        analysis_type="audio",
        message="Audio analysis started. Use /jobs/{job_id} to check progress."
    )


@router.post("/analyze-sync", response_model=AudioAnalysisResponse)
async def analyze_audio_sync(
    file: UploadFile = File(...),
    segment_duration: float = 10.0,
    process_all: bool = True,
    start_time: float = 0,
    end_time: float = None
):
    """
    Analyze emotions in an audio file synchronously.
    
    Waits for processing to complete before returning results.
    Use for smaller files or when immediate results are needed.
    """
    # Validate file type by MIME type or extension
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg', 'audio/x-wav', 'audio/wave']
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    filename_lower = file.filename.lower() if file.filename else ""
    has_valid_extension = any(filename_lower.endswith(ext) for ext in allowed_extensions)
    has_valid_mime = file.content_type in allowed_types
    
    if not (has_valid_extension or has_valid_mime):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: WAV, MP3, FLAC, OGG"
        )
    
    # Save file
    content = await file.read()
    file_path = analysis_service.save_upload(content, file.filename)
    
    # Create job
    job_id = analysis_service.create_job("audio")
    
    try:
        result = await analysis_service.analyze_audio(
            job_id=job_id,
            file_path=file_path,
            segment_duration=segment_duration,
            process_all=process_all,
            start_time=start_time,
            end_time=end_time
        )
        return AudioAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

