# Recognition — Emotion AI Platform

A multi-modal emotion and sentiment analysis platform that extracts emotional insights from audio, video, and text using state-of-the-art transformer models.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## What It Does

Recognition analyzes human emotions across three modalities:

| Modality | Model | Capabilities |
|----------|-------|--------------|
| **Audio** | Wav2Vec2 | Speech emotion recognition from voice recordings |
| **Video** | DeepFace + MediaPipe | Facial emotion detection with multi-face tracking |
| **Text** | FinBERT + SpaCy | Financial sentiment analysis with named entity recognition |

## Project Structure

```
├── api/                    # FastAPI backend
│   ├── main.py            # Application entry point
│   ├── routers/           # API route handlers
│   ├── auth/              # JWT & API key authentication
│   ├── tasks.py           # Celery async tasks
│   └── celery_app.py      # Task queue configuration
│
├── Backend/               # Core ML processing modules
│   ├── Audio/             # Audio analysis (Wav2Vec2, librosa)
│   ├── Video/             # Video processing (DeepFace, yt-dlp)
│   └── Text/              # Text analysis (FinBERT, SpaCy NER)
│
├── frontend/              # Next.js 14 web interface
│   ├── app/               # App router pages
│   ├── components/        # React components
│   └── package.json       # Node dependencies
│
├── docker-compose.yml     # Full stack orchestration
├── Dockerfile.api         # API container
├── requirements.txt       # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Redis (for async processing)
- FFmpeg (for audio/video processing)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/engrahsaninam/Emotion-Recognition.git
cd Emotion-Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Start the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`

### Docker Setup (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

This starts:
- API server on port 8000
- Frontend on port 3000
- Redis for task queue
- Celery workers for async processing

## API Usage

### Authentication

The API supports both JWT tokens and API keys:

```bash
# Get JWT token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use API key
curl -X POST http://localhost:8000/api/v1/analyze/audio \
  -H "X-API-Key: your_api_key" \
  -F "file=@audio.wav"
```

### Analyze Audio

```bash
curl -X POST http://localhost:8000/api/v1/analyze/audio \
  -H "Authorization: Bearer <token>" \
  -F "file=@speech.wav"
```

Response:
```json
{
  "primary_emotion": "happy",
  "confidence": 87.5,
  "emotions": {
    "happy": 87.5,
    "neutral": 8.2,
    "sad": 2.8,
    "angry": 1.0,
    "fear": 0.5
  }
}
```

### Analyze Video

```bash
# From file
curl -X POST http://localhost:8000/api/v1/analyze/video \
  -F "file=@video.mp4"

# From YouTube URL
curl -X POST http://localhost:8000/api/v1/analyze/video \
  -F "youtube_url=https://youtube.com/watch?v=..."
```

### Analyze Text

```bash
# Direct text
curl -X POST http://localhost:8000/api/v1/analyze/text \
  -F "text=Apple reported record revenue, exceeding expectations."

# From PDF
curl -X POST http://localhost:8000/api/v1/analyze/text \
  -F "file=@document.pdf"
```

Response includes sentiment and named entities:
```json
{
  "primary_emotion": "positive",
  "confidence": 94.2,
  "sentiment": "positive",
  "sentiment_score": 0.89,
  "entities": [
    {"text": "Apple", "label": "ORG"}
  ]
}
```

## Models Used

| Task | Model | Source |
|------|-------|--------|
| Speech Emotion | Wav2Vec2 | HuggingFace Transformers |
| Facial Emotion | DeepFace | DeepFace library |
| Face Detection | MediaPipe | Google MediaPipe |
| Text Sentiment | FinBERT | ProsusAI/finbert |
| Named Entities | SpaCy | en_core_web_sm |

## Configuration

Environment variables (create `.env` file):

```env
# API Settings
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/recognition
```

## Frontend Features

- **Drag & drop** file uploads
- **YouTube URL** support for video analysis
- **Real-time** analysis with loading states
- **Visual charts** for emotion breakdown
- **Named entity** display for text analysis
- **Responsive** design for all devices

## Supported Formats

| Type | Formats |
|------|---------|
| Audio | WAV, MP3, FLAC, OGG, M4A |
| Video | MP4, WebM, AVI, MOV, MKV, YouTube URLs |
| Text | TXT, PDF, DOC, DOCX, direct input |

## Development

### Running Tests

```bash
# Backend tests
pytest tests/

# Frontend tests
cd frontend && npm test
```

### Code Style

```bash
# Python
black api/ Backend/
flake8 api/ Backend/

# TypeScript
cd frontend && npm run lint
```

## Roadmap

- [ ] Batch processing for multiple files
- [ ] Real-time streaming analysis
- [ ] Custom model fine-tuning
- [ ] Webhook notifications
- [ ] Dashboard analytics
- [ ] Multi-language audio support

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for facial analysis
- [HuggingFace](https://huggingface.co/) for transformer models
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Next.js](https://nextjs.org/) for the frontend framework

---

Built with ❤️ by [@engrahsaninam](https://github.com/engrahsaninam)
