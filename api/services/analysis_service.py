"""
Analysis Service Module

Provides business logic for audio, video, and text analysis.
"""

import os
import sys
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path

# Add parent directory to path for Backend imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AnalysisService:
    """
    Service class for handling all analysis operations.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.audio_analyzer = None
        self.video_analyzer = None
        self.text_analyzer = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.upload_dir = os.path.join(tempfile.gettempdir(), "recognition_uploads")
        self.output_dir = os.path.join(tempfile.gettempdir(), "recognition_outputs")
        
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._initialized = True
    
    def load_models(self, audio: bool = True, video: bool = True, text: bool = True):
        """Load all ML models."""
        if audio and self.audio_analyzer is None:
            print("Loading audio analysis models...")
            try:
                from Backend.Audio.wav2vec_analyzer import Wav2VecEmotionRecognizer
                self.audio_analyzer = Wav2VecEmotionRecognizer()
                self.audio_analyzer.load_models()
                print("Audio models loaded successfully")
            except Exception as e:
                print(f"Failed to load audio models: {e}")
        
        if video and self.video_analyzer is None:
            print("Loading video analysis models...")
            try:
                from Backend.Video.deep import FaceEmotionDetector
                self.video_analyzer = FaceEmotionDetector()
                self.video_analyzer.initialize()
                print("Video models loaded successfully")
            except Exception as e:
                print(f"Failed to load video models: {e}")
        
        if text and self.text_analyzer is None:
            print("Loading text analysis models...")
            try:
                from Backend.Text.advanced_text_analyzer import AdvancedTextAnalyzer
                self.text_analyzer = AdvancedTextAnalyzer()
                self.text_analyzer.load_models()
                print("Text models loaded successfully")
            except Exception as e:
                print(f"Failed to load text models: {e}")
    
    def get_models_status(self) -> Dict[str, bool]:
        """Get status of loaded models."""
        return {
            "audio": self.audio_analyzer is not None,
            "video": self.video_analyzer is not None,
            "text": self.text_analyzer is not None
        }
    
    def create_job(self, analysis_type: str) -> str:
        """Create a new analysis job."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "type": analysis_type,
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
            "created_at": datetime.utcnow(),
            "completed_at": None
        }
        return job_id
    
    def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and results."""
        return self.jobs.get(job_id)
    
    def save_upload(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return path."""
        file_path = os.path.join(self.upload_dir, f"{uuid.uuid4()}_{filename}")
        with open(file_path, "wb") as f:
            f.write(file_content)
        return file_path
    
    async def analyze_audio(
        self,
        job_id: str,
        file_path: str,
        segment_duration: float = 10.0,
        process_all: bool = True,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze audio file for emotions.
        
        Args:
            job_id: Job identifier
            file_path: Path to audio file
            segment_duration: Duration of each segment
            process_all: Whether to process entire file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Analysis results dictionary
        """
        self.update_job(job_id, status="processing")
        
        try:
            import librosa
            
            # Get audio duration
            duration = librosa.get_duration(path=file_path)
            
            if end_time is None or end_time > duration:
                end_time = duration
            
            segments = []
            
            if process_all:
                num_segments = max(1, int(duration / segment_duration))
                
                for i in range(num_segments):
                    self.update_job(job_id, progress=int((i + 1) / num_segments * 100))
                    
                    seg_start = i * segment_duration
                    seg_end = min((i + 1) * segment_duration, duration)
                    
                    # Analyze segment
                    dominant, scores, predictions = self.audio_analyzer.predict_emotion(file_path)
                    
                    # Convert numpy floats to Python floats for JSON serialization
                    scores_serializable = {k: float(v) for k, v in scores.items()}
                    
                    segments.append({
                        "segment_id": i,
                        "start_time": float(seg_start),
                        "end_time": float(seg_end),
                        "dominant_emotion": dominant,
                        "confidence": float(predictions.max()) * 100,
                        "all_emotions": scores_serializable
                    })
            else:
                dominant, scores, predictions = self.audio_analyzer.predict_emotion(file_path)
                scores_serializable = {k: float(v) for k, v in scores.items()}
                segments.append({
                    "segment_id": 0,
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "dominant_emotion": dominant,
                    "confidence": float(predictions.max()) * 100,
                    "all_emotions": scores_serializable
                })
            
            # Calculate overall score (handle both short and full emotion labels)
            positive_labels = ["happy", "hap"]
            negative_labels = ["sad", "angry", "fear", "ang"]
            positive_count = sum(1 for s in segments if s["dominant_emotion"] in positive_labels)
            negative_count = sum(1 for s in segments if s["dominant_emotion"] in negative_labels)
            
            total = positive_count + negative_count
            overall_score = (positive_count - negative_count) / total if total > 0 else 0
            
            result = {
                "job_id": job_id,
                "status": "completed",
                "file_name": os.path.basename(file_path),
                "duration": float(duration),
                "segments": segments,
                "overall_score": float(overall_score),
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "created_at": self.jobs[job_id]["created_at"].isoformat() if hasattr(self.jobs[job_id]["created_at"], 'isoformat') else str(self.jobs[job_id]["created_at"])
            }
            
            self.update_job(
                job_id,
                status="completed",
                result=result,
                progress=100,
                completed_at=datetime.utcnow()
            )
            
            return result
            
        except Exception as e:
            self.update_job(
                job_id,
                status="failed",
                error=str(e),
                completed_at=datetime.utcnow()
            )
            raise
    
    async def analyze_video(
        self,
        job_id: str,
        file_path: str,
        frame_interval: int = 30,
        detector_backend: str = "opencv"
    ) -> Dict[str, Any]:
        """
        Analyze video file for facial emotions.
        
        Args:
            job_id: Job identifier
            file_path: Path to video file
            frame_interval: Analyze every N frames
            detector_backend: Face detection backend
            
        Returns:
            Analysis results dictionary
        """
        import cv2
        
        self.update_job(job_id, status="processing")
        
        try:
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
                    self.update_job(job_id, progress=int(frame_count / total_frames * 100))
                    
                    # Detect emotions
                    dominant, region, confidence, emotions = self.video_analyzer.detect_emotion(frame)
                    
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
                "created_at": self.jobs[job_id]["created_at"]
            }
            
            self.update_job(
                job_id,
                status="completed",
                result=result,
                progress=100,
                completed_at=datetime.utcnow()
            )
            
            return result
            
        except Exception as e:
            self.update_job(
                job_id,
                status="failed",
                error=str(e),
                completed_at=datetime.utcnow()
            )
            raise
    
    async def analyze_text(
        self,
        job_id: str,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        paragraph_limit: int = 50,
        extract_entities: bool = True,
        extract_aspects: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze text/PDF for sentiment.
        
        Args:
            job_id: Job identifier
            file_path: Path to PDF file
            text: Direct text input
            paragraph_limit: Minimum paragraph length
            extract_entities: Whether to extract entities
            extract_aspects: Whether to extract aspects
            
        Returns:
            Analysis results dictionary
        """
        self.update_job(job_id, status="processing")
        
        try:
            paragraphs = []
            
            if file_path and file_path.endswith('.pdf'):
                paragraphs = self.text_analyzer.pdf_to_paragraphs(file_path)
            elif text:
                paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > paragraph_limit]
            
            self.text_analyzer.paragraph_limit = paragraph_limit
            
            # Analyze document
            doc_result = self.text_analyzer.analyze_document(paragraphs)
            
            # Format results
            paragraph_results = []
            for i, pr in enumerate(doc_result.paragraphs):
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
                "created_at": self.jobs[job_id]["created_at"]
            }
            
            self.update_job(
                job_id,
                status="completed",
                result=result,
                progress=100,
                completed_at=datetime.utcnow()
            )
            
            return result
            
        except Exception as e:
            self.update_job(
                job_id,
                status="failed",
                error=str(e),
                completed_at=datetime.utcnow()
            )
            raise


# Global service instance
analysis_service = AnalysisService()

