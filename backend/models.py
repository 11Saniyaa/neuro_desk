"""Data models for the wellness analyzer"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class AnalyzeRequest(BaseModel):
    """Request model for image analysis"""
    image: str = Field(..., description="Base64 encoded image data")


class PostureAnalysis(BaseModel):
    """Posture analysis result"""
    slouching: bool = False
    score: Optional[float] = None
    head_angle: float = 0.0
    face_position_y: Optional[float] = None
    face_position_x: Optional[float] = None
    error: Optional[str] = None
    reason: Optional[str] = None


class EyeStrainAnalysis(BaseModel):
    """Eye strain analysis result"""
    eye_strain_risk: str = "unknown"
    score: Optional[float] = None
    blink_rate: float = 0.0
    ear_avg: float = 0.0
    error: Optional[str] = None


class EngagementAnalysis(BaseModel):
    """Engagement analysis result"""
    concentration: str = "low"
    score: Optional[float] = None
    face_visible: bool = False
    head_stability: float = 0.0
    error: Optional[str] = None


class StressAnalysis(BaseModel):
    """Stress analysis result"""
    stress_level: str = "low"
    score: Optional[float] = None
    indicators: List[str] = []
    error: Optional[str] = None


class ProductivityAnalysis(BaseModel):
    """Productivity analysis result"""
    productivity_score: Optional[float] = None
    break_needed: bool = False
    eye_exercise_needed: bool = False
    posture_reminder: bool = False
    error: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    timestamp: str
    posture: Dict[str, Any]
    eye_strain: Dict[str, Any]
    engagement: Dict[str, Any]
    stress: Dict[str, Any]
    productivity: Dict[str, Any]
    recommendations: List[str]
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    posture: Optional[Dict[str, Any]] = None
    eye_strain: Optional[Dict[str, Any]] = None
    engagement: Optional[Dict[str, Any]] = None
    stress: Optional[Dict[str, Any]] = None
    productivity: Optional[Dict[str, Any]] = None
    recommendations: List[str] = []

