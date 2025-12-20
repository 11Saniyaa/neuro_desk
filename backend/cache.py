"""In-memory caching for face detection results and analysis"""
from typing import Optional, Dict, Any, Tuple
import hashlib
import numpy as np
import time
import logging
from collections import OrderedDict

class LRUCache:
    """Simple LRU cache implementation"""
    def __init__(self, maxsize: int = 100):
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            # Remove oldest (first item)
            self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
    
    def size(self) -> int:
        return len(self.cache)

class FaceDetectionCache:
    """Cache for face detection results"""
    def __init__(self, maxsize: int = 50, ttl_seconds: float = 1.0):
        self.cache = LRUCache(maxsize)
        self.ttl = ttl_seconds
        self.timestamps: Dict[str, float] = {}
    
    def _get_image_hash(self, image: np.ndarray) -> str:
        """Generate hash for image"""
        # Use a sample of pixels for fast hashing
        h, w = image.shape[:2]
        sample = image[::max(1, h//10), ::max(1, w//10)].tobytes()
        return hashlib.md5(sample).hexdigest()
    
    def get(self, image: np.ndarray) -> Optional[Tuple]:
        """Get cached face detection result"""
        key = self._get_image_hash(image)
        if key in self.cache.cache:
            # Check if expired
            if time.time() - self.timestamps.get(key, 0) < self.ttl:
                return self.cache.get(key)
            else:
                # Expired, remove
                self.cache.cache.pop(key, None)
                self.timestamps.pop(key, None)
        return None
    
    def set(self, image: np.ndarray, result: Tuple):
        """Cache face detection result"""
        key = self._get_image_hash(image)
        self.cache.set(key, result)
        self.timestamps[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": self.cache.size(),
            "maxsize": self.cache.maxsize,
            "ttl": self.ttl
        }

# Global cache instance
face_detection_cache = FaceDetectionCache(maxsize=50, ttl_seconds=1.0)

