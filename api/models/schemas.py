"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisType(str, Enum):
    """Type of analysis."""
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


# ============== Audio Models ==============

class AudioAnalysisRequest(BaseModel):
    """Request for audio emotion analysis."""
    segment_duration: float = Field(default=10.0, description="Duration of each segment in seconds")
    process_all: bool = Field(default=True, description="Process entire audio file")
    start_time: Optional[float] = Field(default=0, description="Start time in seconds")
    end_time: Optional[float] = Field(default=None, description="End time in seconds")


class EmotionScore(BaseModel):
    """Emotion detection scores."""
    emotion: str
    score: float
    confidence: float


class AudioSegmentResult(BaseModel):
    """Result for a single audio segment."""
    segment_id: int
    start_time: float
    end_time: float
    dominant_emotion: str
    confidence: float
    all_emotions: Dict[str, float]


class AudioAnalysisResponse(BaseModel):
    """Response for audio emotion analysis."""
    job_id: str
    status: JobStatus
    file_name: str
    duration: float
    segments: List[AudioSegmentResult] = []
    overall_score: float = 0
    positive_count: int = 0
    negative_count: int = 0
    visualizations: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============== Video Models ==============

class VideoAnalysisRequest(BaseModel):
    """Request for video emotion analysis."""
    youtube_url: Optional[str] = Field(default=None, description="YouTube URL to download")
    frame_interval: int = Field(default=30, description="Analyze every N frames")
    detector_backend: str = Field(default="opencv", description="Face detection backend")
    use_mediapipe: bool = Field(default=False, description="Use MediaPipe for face mesh")


class FaceDetection(BaseModel):
    """Single face detection result."""
    face_id: int
    region: Dict[str, int]
    dominant_emotion: str
    confidence: float
    emotions: Dict[str, float]


class VideoFrameResult(BaseModel):
    """Result for a single video frame."""
    frame_number: int
    timestamp: float
    faces: List[FaceDetection]


class VideoAnalysisResponse(BaseModel):
    """Response for video emotion analysis."""
    job_id: str
    status: JobStatus
    file_name: str
    duration: float
    fps: float
    frame_results: List[VideoFrameResult] = []
    overall_score: float = 0
    positive_count: int = 0
    negative_count: int = 0
    output_video: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============== Text Models ==============

class TextAnalysisRequest(BaseModel):
    """Request for text sentiment analysis."""
    text: Optional[str] = Field(default=None, description="Direct text input")
    paragraph_limit: int = Field(default=50, description="Minimum paragraph length")
    extract_entities: bool = Field(default=True, description="Extract named entities")
    extract_aspects: bool = Field(default=True, description="Extract aspect-based sentiment")


class EntityResult(BaseModel):
    """Named entity extraction result."""
    text: str
    entity_type: str
    confidence: float


class AspectResult(BaseModel):
    """Aspect-based sentiment result."""
    aspect: str
    sentence: str
    sentiment: str
    confidence: float


class ParagraphResult(BaseModel):
    """Result for a single paragraph."""
    paragraph_id: int
    text_preview: str
    sentiment: str
    confidence: float
    entities: List[EntityResult] = []
    aspects: List[AspectResult] = []


class TextAnalysisResponse(BaseModel):
    """Response for text sentiment analysis."""
    job_id: str
    status: JobStatus
    file_name: Optional[str] = None
    total_paragraphs: int
    paragraph_results: List[ParagraphResult] = []
    overall_sentiment: str
    score_index: float
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    entities_summary: Dict[str, List[str]] = {}
    key_aspects: List[AspectResult] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============== Job Models ==============

class JobResponse(BaseModel):
    """Generic job response."""
    job_id: str
    status: JobStatus
    analysis_type: AnalysisType
    message: str
    progress: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    uptime: float

