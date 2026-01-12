"""
Celery Application Configuration

Provides distributed task processing for long-running analysis jobs.

Usage:
    celery -A api.celery_app worker --loglevel=info
    celery -A api.celery_app flower  # For monitoring
"""

import os
import sys
from celery import Celery
from kombu import Queue

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "recognition_system",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["api.tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    
    # Result settings
    result_expires=3600 * 24,  # Results expire after 24 hours
    
    # Task routing
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("audio", routing_key="audio.*"),
        Queue("video", routing_key="video.*"),
        Queue("text", routing_key="text.*"),
    ),
    
    task_routes={
        "api.tasks.analyze_audio_task": {"queue": "audio"},
        "api.tasks.analyze_video_task": {"queue": "video"},
        "api.tasks.analyze_text_task": {"queue": "text"},
    },
    
    # Worker settings
    worker_concurrency=2,  # Adjust based on available resources
    worker_max_tasks_per_child=100,
)


if __name__ == "__main__":
    celery_app.start()

