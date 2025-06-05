import os
from typing import Optional

class Settings:
    """Application settings"""
    
    # Model configuration
    MODEL_NAME: str = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    DEVICE: str = "cuda" if os.environ.get("CUDA_AVAILABLE") == "true" else "cpu"
    MAX_AUDIO_DURATION: float = 30.0
    
    # Cache settings
    HF_CACHE_DIR: str = os.environ.get("HF_HOME", "/tmp/huggingface_cache")
    TRANSFORMERS_CACHE: str = os.environ.get("TRANSFORMERS_CACHE", "/tmp/huggingface_cache")
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 8000))
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
    
    # Emotion labels
    EMOTION_LABELS: list = [
        "angry", "disgust", "fearful", "happy", 
        "neutral", "sad", "surprised"
    ]

settings = Settings()
