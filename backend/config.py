"""Configuration settings for the wellness analyzer"""
import os
from typing import Dict, Any

# Frame processing configuration
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "1"))  # Process every frame by default
MAX_FPS = int(os.getenv("MAX_FPS", "5"))  # Maximum frames per second
FRAME_INTERVAL_MS = int(os.getenv("FRAME_INTERVAL_MS", "200"))  # Milliseconds between frames

# Image processing configuration
IMAGE_QUALITY = float(os.getenv("IMAGE_QUALITY", "0.7"))  # JPEG quality (0-1)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1920"))  # Max image dimension
ENABLE_IMAGE_COMPRESSION = os.getenv("ENABLE_IMAGE_COMPRESSION", "true").lower() == "true"

# Face detection configuration
MIN_FACE_CONFIDENCE = float(os.getenv("MIN_FACE_CONFIDENCE", "0.6"))
MIN_LANDMARK_QUALITY = float(os.getenv("MIN_LANDMARK_QUALITY", "0.7"))

# Analysis thresholds
EAR_BASELINE = float(os.getenv("EAR_BASELINE", "0.28"))
EAR_STD = float(os.getenv("EAR_STD", "0.03"))
OUTLIER_THRESHOLD = float(os.getenv("OUTLIER_THRESHOLD", "2.5"))

# Calibration settings
CALIBRATION_SAMPLES_EAR = int(os.getenv("CALIBRATION_SAMPLES_EAR", "50"))
CALIBRATION_SAMPLES_POSTURE = int(os.getenv("CALIBRATION_SAMPLES_POSTURE", "30"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.1"))

# Temporal smoothing
HISTORY_SIZE = int(os.getenv("HISTORY_SIZE", "30"))
SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.3"))

# Session management
SESSION_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT_SECONDS", "3600"))  # 1 hour
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"

# Performance optimization
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1"))
ENABLE_PARALLEL_PROCESSING = os.getenv("ENABLE_PARALLEL_PROCESSING", "false").lower() == "true"

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        "frame_skip": FRAME_SKIP,
        "max_fps": MAX_FPS,
        "frame_interval_ms": FRAME_INTERVAL_MS,
        "image_quality": IMAGE_QUALITY,
        "max_image_size": MAX_IMAGE_SIZE,
        "enable_image_compression": ENABLE_IMAGE_COMPRESSION,
        "min_face_confidence": MIN_FACE_CONFIDENCE,
        "min_landmark_quality": MIN_LANDMARK_QUALITY,
        "ear_baseline": EAR_BASELINE,
        "ear_std": EAR_STD,
        "outlier_threshold": OUTLIER_THRESHOLD,
        "calibration_samples_ear": CALIBRATION_SAMPLES_EAR,
        "calibration_samples_posture": CALIBRATION_SAMPLES_POSTURE,
        "learning_rate": LEARNING_RATE,
        "history_size": HISTORY_SIZE,
        "smoothing_alpha": SMOOTHING_ALPHA,
        "session_timeout_seconds": SESSION_TIMEOUT_SECONDS,
        "max_sessions": MAX_SESSIONS,
    }

