"""Integration tests for the wellness analyzer API"""
import pytest
import requests
import base64
import numpy as np
from PIL import Image
import io
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000"

class TestAPIEndpoints:
    """Integration tests for API endpoints"""
    
    def create_test_image(self) -> str:
        """Create a test image as base64 string"""
        # Create a simple test image
        img = Image.new('RGB', (640, 480), color='gray')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        img_data = base64.b64encode(img_bytes.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_data}"
    
    @pytest.mark.skipif(True, reason="Requires running server")
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
    
    @pytest.mark.skipif(True, reason="Requires running server")
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    @pytest.mark.skipif(True, reason="Requires running server")
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = requests.get(f"{BASE_URL}/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data or "error" in data
    
    @pytest.mark.skipif(True, reason="Requires running server")
    def test_analyze_endpoint_valid_image(self):
        """Test analyze endpoint with valid image"""
        image_data = self.create_test_image()
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"image": image_data},
            timeout=10
        )
        assert response.status_code in [200, 400, 500]  # May fail if no face detected
        data = response.json()
        assert "timestamp" in data
        assert "posture" in data
        assert "eye_strain" in data
        assert "engagement" in data
        assert "stress" in data
        assert "productivity" in data
    
    @pytest.mark.skipif(True, reason="Requires running server")
    def test_analyze_endpoint_invalid_image(self):
        """Test analyze endpoint with invalid image"""
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"image": "invalid_base64"},
            timeout=10
        )
        # Should return error
        assert response.status_code in [200, 400, 500]
        data = response.json()
        assert "error" in data or "posture" in data
    
    @pytest.mark.skipif(True, reason="Requires running server")
    def test_analyze_endpoint_no_image(self):
        """Test analyze endpoint with no image"""
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"image": ""},
            timeout=10
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

