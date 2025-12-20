"""Metrics collection and monitoring for the wellness analyzer"""
from typing import Dict, List, Any
from collections import deque, defaultdict
from datetime import datetime
import time
import logging

class MetricsCollector:
    """Collect and track metrics for monitoring"""
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.success_count = 0
        self.response_times: deque = deque(maxlen=100)
        self.error_types: Dict[str, int] = defaultdict(int)
        self.frame_processing_times: deque = deque(maxlen=100)
        self.face_detection_times: deque = deque(maxlen=100)
        self.analysis_times: deque = deque(maxlen=100)
        self.start_time = datetime.now()
        
        # Per-endpoint metrics
        self.endpoint_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "errors": 0,
            "avg_time": 0.0,
            "times": deque(maxlen=50)
        })
    
    def record_request(self, endpoint: str = "unknown"):
        """Record a new request"""
        self.request_count += 1
        self.endpoint_metrics[endpoint]["count"] += 1
    
    def record_success(self, endpoint: str = "unknown", response_time: float = 0.0):
        """Record successful request"""
        self.success_count += 1
        self.response_times.append(response_time)
        self.endpoint_metrics[endpoint]["times"].append(response_time)
        if len(self.endpoint_metrics[endpoint]["times"]) > 0:
            self.endpoint_metrics[endpoint]["avg_time"] = sum(self.endpoint_metrics[endpoint]["times"]) / len(self.endpoint_metrics[endpoint]["times"])
    
    def record_error(self, endpoint: str = "unknown", error_type: str = "unknown"):
        """Record an error"""
        self.error_count += 1
        self.error_types[error_type] += 1
        self.endpoint_metrics[endpoint]["errors"] += 1
    
    def record_frame_processing_time(self, time_ms: float):
        """Record frame processing time"""
        self.frame_processing_times.append(time_ms)
    
    def record_face_detection_time(self, time_ms: float):
        """Record face detection time"""
        self.face_detection_times.append(time_ms)
    
    def record_analysis_time(self, time_ms: float):
        """Record analysis time"""
        self.analysis_times.append(time_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        avg_frame_time = sum(self.frame_processing_times) / len(self.frame_processing_times) if self.frame_processing_times else 0
        avg_face_detection_time = sum(self.face_detection_times) / len(self.face_detection_times) if self.face_detection_times else 0
        avg_analysis_time = sum(self.analysis_times) / len(self.analysis_times) if self.analysis_times else 0
        
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "error_rate_percent": round(error_rate, 2),
            "avg_response_time_ms": round(avg_response_time * 1000, 2),
            "avg_frame_processing_time_ms": round(avg_frame_time, 2),
            "avg_face_detection_time_ms": round(avg_face_detection_time, 2),
            "avg_analysis_time_ms": round(avg_analysis_time, 2),
            "error_types": dict(self.error_types),
            "endpoint_metrics": {
                endpoint: {
                    "count": metrics["count"],
                    "errors": metrics["errors"],
                    "avg_time_ms": round(metrics["avg_time"] * 1000, 2),
                    "error_rate": round((metrics["errors"] / metrics["count"] * 100) if metrics["count"] > 0 else 0, 2)
                }
                for endpoint, metrics in self.endpoint_metrics.items()
            }
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status"""
        stats = self.get_stats()
        is_healthy = (
            stats["error_rate_percent"] < 10 and
            stats["avg_response_time_ms"] < 1000 and
            self.request_count > 0
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "uptime_seconds": stats["uptime_seconds"],
            "error_rate": stats["error_rate_percent"],
            "avg_response_time_ms": stats["avg_response_time_ms"],
            "total_requests": stats["total_requests"]
        }
    
    def reset(self):
        """Reset all metrics"""
        self.request_count = 0
        self.error_count = 0
        self.success_count = 0
        self.response_times.clear()
        self.error_types.clear()
        self.frame_processing_times.clear()
        self.face_detection_times.clear()
        self.analysis_times.clear()
        self.endpoint_metrics.clear()
        self.start_time = datetime.now()

# Global metrics collector
metrics = MetricsCollector()

