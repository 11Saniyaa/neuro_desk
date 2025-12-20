"""Unit tests for caching functionality"""
import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import LRUCache, FaceDetectionCache


class TestLRUCache:
    """Test cases for LRU cache"""
    
    def test_basic_operations(self):
        """Test basic cache operations"""
        cache = LRUCache(maxsize=3)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.size() == 3
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = LRUCache(maxsize=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_lru_reordering(self):
        """Test that accessing moves item to end"""
        cache = LRUCache(maxsize=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Access key1, should move to end
        cache.set("key3", "value3")  # Should evict key2, not key1
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_clear(self):
        """Test cache clearing"""
        cache = LRUCache(maxsize=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None


class TestFaceDetectionCache:
    """Test cases for face detection cache"""
    
    def test_cache_hit(self):
        """Test cache hit scenario"""
        cache = FaceDetectionCache(maxsize=10, ttl_seconds=1.0)
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = (None, None, None)
        
        cache.set(image, result)
        cached = cache.get(image)
        
        assert cached == result
    
    def test_cache_miss(self):
        """Test cache miss scenario"""
        cache = FaceDetectionCache(maxsize=10, ttl_seconds=1.0)
        
        image1 = np.zeros((480, 640, 3), dtype=np.uint8)
        image2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        cache.set(image1, (None, None, None))
        cached = cache.get(image2)
        
        assert cached is None
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = FaceDetectionCache(maxsize=10, ttl_seconds=0.1)  # Very short TTL
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = (None, None, None)
        
        cache.set(image, result)
        assert cache.get(image) == result  # Should be cached
        
        time.sleep(0.15)  # Wait for expiration
        assert cache.get(image) is None  # Should be expired
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = FaceDetectionCache(maxsize=10, ttl_seconds=1.0)
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cache.set(image, (None, None, None))
        
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["maxsize"] == 10
        assert stats["ttl"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

