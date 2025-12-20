"""Unit tests for wellness analyzer"""
import pytest
import numpy as np
from collections import deque
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import WellnessAnalyzer, SimpleLandmark


class TestWellnessAnalyzer:
    """Test cases for WellnessAnalyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = WellnessAnalyzer()
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_landmarks = [None] * 468
        
        # Create some test landmarks
        self.test_landmarks[4] = SimpleLandmark(0.5, 0.5)  # Nose tip
        self.test_landmarks[10] = SimpleLandmark(0.5, 0.3)  # Forehead
        self.test_landmarks[152] = SimpleLandmark(0.5, 0.7)  # Chin
        self.test_landmarks[33] = SimpleLandmark(0.4, 0.45)  # Left eye outer
        self.test_landmarks[133] = SimpleLandmark(0.45, 0.45)  # Left eye inner
        self.test_landmarks[159] = SimpleLandmark(0.425, 0.44)  # Left eye top
        self.test_landmarks[145] = SimpleLandmark(0.425, 0.46)  # Left eye bottom
        self.test_landmarks[362] = SimpleLandmark(0.55, 0.45)  # Right eye outer
        self.test_landmarks[386] = SimpleLandmark(0.5, 0.45)  # Right eye inner
        self.test_landmarks[380] = SimpleLandmark(0.525, 0.44)  # Right eye top
        self.test_landmarks[374] = SimpleLandmark(0.525, 0.46)  # Right eye bottom
    
    def test_calculate_eye_aspect_ratio(self):
        """Test EAR calculation"""
        ear = self.analyzer.calculate_eye_aspect_ratio(self.test_landmarks, self.analyzer.LEFT_EYE_POINTS)
        assert ear > 0, "EAR should be positive"
        assert ear < 1.0, "EAR should be less than 1.0"
    
    def test_smooth_temporal_data(self):
        """Test temporal smoothing"""
        history = deque(maxlen=10)
        new_value = 70.0
        
        smoothed = self.analyzer.smooth_temporal_data(new_value, history, alpha=0.3)
        assert smoothed == new_value, "First value should equal input"
        assert len(history) == 1, "History should contain one value"
        
        # Add another value
        smoothed2 = self.analyzer.smooth_temporal_data(80.0, history, alpha=0.3)
        assert smoothed2 != 80.0, "Smoothed value should differ from input"
        assert 70.0 < smoothed2 < 80.0, "Smoothed value should be between old and new"
    
    def test_validate_image_quality(self):
        """Test image quality validation"""
        # Test with good image (bright, sharp, good contrast)
        good_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        good_image[100:200, 100:200] = 200  # Add some contrast
        
        result = self.analyzer.validate_image_quality(good_image)
        assert result is not None, "Should return quality check result"
        if result:
            assert result.get("valid", False), "Good image should be valid"
    
    def test_analyze_posture_no_face(self):
        """Test posture analysis with no face detected"""
        result = self.analyzer.analyze_posture(self.test_image, None, None)
        assert result is None, "Should return None when no face detected"
    
    def test_calculate_productivity_score_none_inputs(self):
        """Test productivity calculation with None inputs"""
        result = self.analyzer.calculate_productivity_score(None, None, None, None)
        assert result is None, "Should return None when all inputs are None"
    
    def test_calculate_productivity_score_valid_inputs(self):
        """Test productivity calculation with valid inputs"""
        posture = {"score": 80.0, "slouching": False}
        eye_strain = {"score": 75.0, "eye_strain_risk": "low"}
        engagement = {"score": 85.0, "concentration": "high"}
        stress = {"score": 90.0, "stress_level": "low"}
        
        result = self.analyzer.calculate_productivity_score(posture, eye_strain, engagement, stress)
        assert result is not None, "Should return productivity score"
        assert "productivity_score" in result, "Should contain productivity_score"
        assert 0 <= result["productivity_score"] <= 100, "Score should be between 0 and 100"
    
    def test_get_recommendations(self):
        """Test recommendation generation"""
        analysis = {
            "posture_reminder": True,
            "eye_exercise_needed": True,
            "break_needed": True,
            "stress_level": "high",
            "blink_rate": 0.02
        }
        
        recommendations = self.analyzer.get_recommendations(analysis)
        assert isinstance(recommendations, list), "Should return list"
        assert len(recommendations) > 0, "Should have recommendations"
    
    def test_get_recommendations_empty(self):
        """Test recommendation generation with no issues"""
        analysis = {
            "posture_reminder": False,
            "eye_exercise_needed": False,
            "break_needed": False,
            "stress_level": "low",
            "blink_rate": 0.15
        }
        
        recommendations = self.analyzer.get_recommendations(analysis)
        assert isinstance(recommendations, list), "Should return list"
        # Should have at least one positive recommendation
        assert len(recommendations) > 0, "Should have at least one recommendation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

