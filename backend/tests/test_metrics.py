"""Unit tests for metrics collection"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import MetricsCollector


class TestMetricsCollector:
    """Test cases for metrics collector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.metrics = MetricsCollector()
    
    def test_record_request(self):
        """Test recording requests"""
        self.metrics.record_request("test_endpoint")
        assert self.metrics.request_count == 1
        assert self.metrics.endpoint_metrics["test_endpoint"]["count"] == 1
    
    def test_record_success(self):
        """Test recording successful requests"""
        self.metrics.record_request("test_endpoint")
        self.metrics.record_success("test_endpoint", 0.1)
        
        assert self.metrics.success_count == 1
        assert len(self.metrics.response_times) == 1
        assert self.metrics.response_times[0] == 0.1
    
    def test_record_error(self):
        """Test recording errors"""
        self.metrics.record_request("test_endpoint")
        self.metrics.record_error("test_endpoint", "ValueError")
        
        assert self.metrics.error_count == 1
        assert self.metrics.error_types["ValueError"] == 1
        assert self.metrics.endpoint_metrics["test_endpoint"]["errors"] == 1
    
    def test_get_stats(self):
        """Test getting statistics"""
        self.metrics.record_request("test")
        self.metrics.record_success("test", 0.1)
        self.metrics.record_error("test", "Error")
        
        stats = self.metrics.get_stats()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["failed_requests"] == 1
        assert "error_rate_percent" in stats
        assert "avg_response_time_ms" in stats
    
    def test_get_health(self):
        """Test health check"""
        self.metrics.record_request("test")
        self.metrics.record_success("test", 0.05)
        
        health = self.metrics.get_health()
        assert "status" in health
        assert "uptime_seconds" in health
        assert "error_rate" in health
        assert health["status"] in ["healthy", "degraded"]
    
    def test_reset(self):
        """Test resetting metrics"""
        self.metrics.record_request("test")
        self.metrics.record_success("test", 0.1)
        self.metrics.record_error("test", "Error")
        
        self.metrics.reset()
        
        assert self.metrics.request_count == 0
        assert self.metrics.success_count == 0
        assert self.metrics.error_count == 0
        assert len(self.metrics.response_times) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

