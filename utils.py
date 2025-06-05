import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def create_temp_audio_file(audio_data: np.ndarray, sample_rate: int) -> str:
    """Create temporary audio file"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        return tmp_file.name

def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {e}")

def validate_audio_file(file_content: bytes) -> bool:
    """Validate if file content is a valid audio file"""
    try:
        # Try to read the audio content
        sf.read(sf.BytesIO(file_content))
        return True
    except Exception:
        return False

def format_emotion_results(emotions: dict, dominant_emotion: str, confidence: float) -> dict:
    """Format emotion analysis results"""
    return {
        "emotions": {k: round(v, 4) for k, v in emotions.items()},
        "dominant_emotion": dominant_emotion,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{round(confidence * 100, 2)}%"
    }
