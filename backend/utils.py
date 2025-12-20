"""Utility functions for image processing and performance optimization"""
import cv2
import numpy as np
from typing import Tuple, Optional
import logging
from functools import lru_cache
import hashlib
import base64

# Performance optimization: Cache for image compression
@lru_cache(maxsize=100)
def compress_image_cached(image_hash: str, max_size: int, quality: int) -> Optional[bytes]:
    """Cached image compression - returns None as cache is for hash only"""
    return None

def compress_image(image: np.ndarray, max_size: int = 640, quality: int = 85) -> np.ndarray:
    """Compress image for faster processing while maintaining quality"""
    h, w = image.shape[:2]
    
    # Only resize if image is larger than max_size
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logging.debug(f"Compressed image from {w}x{h} to {new_w}x{new_h}")
    
    return image

def optimize_image_for_processing(image: np.ndarray) -> np.ndarray:
    """Optimize image for face detection processing"""
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Compress if too large
    image = compress_image(image, max_size=640)
    
    return image

def calculate_image_hash(image: np.ndarray) -> str:
    """Calculate hash of image for caching"""
    # Use a simple hash of image dimensions and a sample of pixels
    h, w = image.shape[:2]
    sample = image[::max(1, h//10), ::max(1, w//10)].tobytes()
    return hashlib.md5(sample).hexdigest()

def decode_image_optimized(image_data: str) -> np.ndarray:
    """Optimized image decoding with error handling"""
    try:
        # Handle data URL format
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Optimize for processing
        image = optimize_image_for_processing(image)
        
        return image
    except Exception as e:
        logging.error(f"Error decoding image: {e}", exc_info=True)
        raise

