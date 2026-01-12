"""
Celery Tasks for Background Processing

Defines async tasks for audio, video, and text analysis.
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.celery_app import celery_app
from api.services.analysis_service import AnalysisService


# Create a dedicated service instance for Celery workers
_worker_service = None


def get_worker_service() -> AnalysisService:
    """Get or create the worker service instance."""
    global _worker_service
    if _worker_service is None:
        _worker_service = AnalysisService()
        # Load models in worker
        _worker_service.load_models(audio=True, video=True, text=True)
    return _worker_service


@celery_app.task(bind=True, name="api.tasks.analyze_audio_task")
def analyze_audio_task(
    self,
    job_id: str,
    file_path: str,
    segment_duration: float = 10.0,
    process_all: bool = True,
    start_time: float = 0,
    end_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Celery task for audio emotion analysis.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the audio file
        segment_duration: Duration of each segment in seconds
        process_all: Whether to process entire file
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Analysis results dictionary
    """
    service = get_worker_service()
    
    try:
        import librosa
        import asyncio
        
        # Update task state
        self.update_state(state="PROCESSING", meta={"progress": 0})
        
        # Get audio duration
        duration = librosa.get_duration(path=file_path)
        
        if end_time is None or end_time > duration:
            end_time = duration
        
        segments = []
        
        if process_all:
            num_segments = max(1, int(duration / segment_duration))
            
            for i in range(num_segments):
                progress = int((i + 1) / num_segments * 100)
                self.update_state(state="PROCESSING", meta={"progress": progress})
                
                seg_start = i * segment_duration
                seg_end = min((i + 1) * segment_duration, duration)
                
                # Analyze segment
                dominant, scores, predictions = service.audio_analyzer.predict_emotion(file_path)
                
                segments.append({
                    "segment_id": i,
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "dominant_emotion": dominant,
                    "confidence": float(predictions.max()) * 100,
                    "all_emotions": scores
                })
        else:
            dominant, scores, predictions = service.audio_analyzer.predict_emotion(file_path)
            segments.append({
                "segment_id": 0,
                "start_time": start_time,
                "end_time": end_time,
                "dominant_emotion": dominant,
                "confidence": float(predictions.max()) * 100,
                "all_emotions": scores
            })
        
        # Calculate overall score
        positive_count = sum(1 for s in segments if s["dominant_emotion"] in ["happy"])
        negative_count = sum(1 for s in segments if s["dominant_emotion"] in ["sad", "angry", "fear"])
        
        total = positive_count + negative_count
        overall_score = (positive_count - negative_count) / total if total > 0 else 0
        
        result = {
            "job_id": job_id,
            "status": "completed",
            "file_name": os.path.basename(file_path),
            "duration": duration,
            "segments": segments,
            "overall_score": overall_score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@celery_app.task(bind=True, name="api.tasks.analyze_video_task")
def analyze_video_task(
    self,
    job_id: str,
    file_path: str,
    frame_interval: int = 30,
    detector_backend: str = "opencv"
) -> Dict[str, Any]:
    """
    Celery task for video facial emotion analysis.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the video file
        frame_interval: Analyze every N frames
        detector_backend: Face detection backend
        
    Returns:
        Analysis results dictionary
    """
    service = get_worker_service()
    
    try:
        import cv2
        
        self.update_state(state="PROCESSING", meta={"progress": 0})
        
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frame_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % frame_interval == 0:
                progress = int(frame_count / total_frames * 100)
                self.update_state(state="PROCESSING", meta={"progress": progress})
                
                # Detect emotions
                dominant, region, confidence, emotions = service.video_analyzer.detect_emotion(frame)
                
                if dominant != "Not Detected":
                    face_detection = {
                        "face_id": 0,
                        "region": region if isinstance(region, dict) else {},
                        "dominant_emotion": dominant,
                        "confidence": confidence,
                        "emotions": emotions
                    }
                    
                    frame_results.append({
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps,
                        "faces": [face_detection]
                    })
        
        cap.release()
        
        # Calculate overall score
        all_emotions = [f["faces"][0]["dominant_emotion"] for f in frame_results if f["faces"]]
        positive_count = sum(1 for e in all_emotions if e in ["happy"])
        negative_count = sum(1 for e in all_emotions if e in ["sad", "angry", "fear"])
        
        total = positive_count + negative_count
        overall_score = (positive_count - negative_count) / total if total > 0 else 0
        
        result = {
            "job_id": job_id,
            "status": "completed",
            "file_name": os.path.basename(file_path),
            "duration": duration,
            "fps": fps,
            "frame_results": frame_results,
            "overall_score": overall_score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@celery_app.task(bind=True, name="api.tasks.analyze_text_task")
def analyze_text_task(
    self,
    job_id: str,
    file_path: Optional[str] = None,
    text: Optional[str] = None,
    paragraph_limit: int = 50,
    extract_entities: bool = True,
    extract_aspects: bool = True
) -> Dict[str, Any]:
    """
    Celery task for text/PDF sentiment analysis.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to PDF file
        text: Direct text input
        paragraph_limit: Minimum paragraph length
        extract_entities: Whether to extract entities
        extract_aspects: Whether to extract aspects
        
    Returns:
        Analysis results dictionary
    """
    service = get_worker_service()
    
    try:
        self.update_state(state="PROCESSING", meta={"progress": 0})
        
        paragraphs = []
        
        if file_path and file_path.endswith('.pdf'):
            paragraphs = service.text_analyzer.pdf_to_paragraphs(file_path)
        elif text:
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > paragraph_limit]
        
        service.text_analyzer.paragraph_limit = paragraph_limit
        
        # Analyze document
        doc_result = service.text_analyzer.analyze_document(paragraphs)
        
        # Format results
        paragraph_results = []
        for i, pr in enumerate(doc_result.paragraphs):
            progress = int((i + 1) / len(doc_result.paragraphs) * 100)
            self.update_state(state="PROCESSING", meta={"progress": progress})
            
            para_result = {
                "paragraph_id": i,
                "text_preview": pr.text[:200] + "..." if len(pr.text) > 200 else pr.text,
                "sentiment": pr.sentiment,
                "confidence": pr.confidence,
                "entities": [
                    {"text": e["text"], "entity_type": e["type"], "confidence": e.get("confidence", 1.0)}
                    for e in pr.entities
                ] if extract_entities else [],
                "aspects": [
                    {"aspect": a["aspect"], "sentence": a["sentence"], "sentiment": a["sentiment"], "confidence": a["confidence"]}
                    for a in pr.aspects
                ] if extract_aspects else []
            }
            paragraph_results.append(para_result)
        
        result = {
            "job_id": job_id,
            "status": "completed",
            "file_name": os.path.basename(file_path) if file_path else None,
            "total_paragraphs": len(paragraphs),
            "paragraph_results": paragraph_results,
            "overall_sentiment": doc_result.overall_sentiment,
            "score_index": doc_result.score_index,
            "positive_count": doc_result.positive_count,
            "negative_count": doc_result.negative_count,
            "neutral_count": doc_result.neutral_count,
            "entities_summary": doc_result.entities_summary,
            "key_aspects": [
                {"aspect": a["aspect"], "sentence": a["sentence"], "sentiment": a["sentiment"], "confidence": a["confidence"]}
                for a in doc_result.key_aspects
            ],
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


# Webhook notification task
@celery_app.task(name="api.tasks.send_webhook")
def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> bool:
    """
    Send webhook notification when job completes.
    
    Args:
        webhook_url: URL to send the notification
        payload: Job result payload
        
    Returns:
        Success status
    """
    import httpx
    
    try:
        with httpx.Client() as client:
            response = client.post(webhook_url, json=payload, timeout=30)
            return response.status_code == 200
    except Exception as e:
        print(f"Webhook failed: {e}")
        return False

