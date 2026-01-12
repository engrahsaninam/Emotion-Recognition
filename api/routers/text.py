"""
Text Analysis Router

Endpoints for text/document sentiment analysis.
"""

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional

from ..models.schemas import TextAnalysisRequest, TextAnalysisResponse, JobResponse, JobStatus
from ..services.analysis_service import analysis_service

router = APIRouter(prefix="/text", tags=["Text Analysis"])


@router.post("/analyze", response_model=JobResponse)
async def analyze_text(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    paragraph_limit: int = 50,
    extract_entities: bool = True,
    extract_aspects: bool = True
):
    """
    Analyze sentiment in text or PDF document.
    
    Provide either a PDF file or direct text input.
    Returns a job ID that can be used to check progress and get results.
    
    Features:
    - FinBERT sentiment analysis (Positive/Negative/Neutral)
    - Named Entity Recognition (NER)
    - Aspect-based sentiment extraction
    - Financial entity detection (tickers, money, percentages)
    """
    if file is None and text is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file or text input"
        )
    
    file_path = None
    
    if file:
        # Validate file type
        allowed_types = ['application/pdf', 'text/plain']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: PDF, TXT"
            )
        
        # Save file
        content = await file.read()
        file_path = analysis_service.save_upload(content, file.filename)
    
    # Create job
    job_id = analysis_service.create_job("text")
    
    # Start background processing
    background_tasks.add_task(
        analysis_service.analyze_text,
        job_id=job_id,
        file_path=file_path,
        text=text,
        paragraph_limit=paragraph_limit,
        extract_entities=extract_entities,
        extract_aspects=extract_aspects
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        analysis_type="text",
        message="Text analysis started. Use /jobs/{job_id} to check progress."
    )


@router.post("/analyze-sync", response_model=TextAnalysisResponse)
async def analyze_text_sync(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    paragraph_limit: int = 50,
    extract_entities: bool = True,
    extract_aspects: bool = True
):
    """
    Analyze sentiment in text or PDF document synchronously.
    
    Waits for processing to complete before returning results.
    """
    if file is None and text is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file or text input"
        )
    
    file_path = None
    
    if file:
        # Validate file type
        allowed_types = ['application/pdf', 'text/plain']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: PDF, TXT"
            )
        
        # Save file
        content = await file.read()
        file_path = analysis_service.save_upload(content, file.filename)
    
    # Create job
    job_id = analysis_service.create_job("text")
    
    try:
        result = await analysis_service.analyze_text(
            job_id=job_id,
            file_path=file_path,
            text=text,
            paragraph_limit=paragraph_limit,
            extract_entities=extract_entities,
            extract_aspects=extract_aspects
        )
        return TextAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-quick")
async def analyze_text_quick(text: str = Form(...)):
    """
    Quick sentiment analysis for short text.
    
    Performs immediate analysis without job creation.
    Best for single sentences or short paragraphs.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        sentiment, confidence = analysis_service.text_analyzer.analyze_sentiment(text)
        entities = analysis_service.text_analyzer.extract_entities(text)
        aspects = analysis_service.text_analyzer.extract_aspects(text)
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "entities": entities,
            "aspects": aspects
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

