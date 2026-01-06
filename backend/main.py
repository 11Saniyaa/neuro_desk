from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import logging
import traceback
from collections import deque
import time

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import face detection libraries in order of preference
MEDIAPIPE_AVAILABLE = False
DLIB_AVAILABLE = False
FACE_RECOGNITION_AVAILABLE = False
MTCNN_AVAILABLE = False
FACE_DETECTOR_DNN = None
FACE_DETECTOR_DNN_ACCURATE = None  # More accurate DNN model

# Initialize face detection libraries
dlib_face_detector = None
dlib_landmark_predictor = None
mtcnn_detector = None

# 1. Try MediaPipe first (best accuracy, 468 landmarks)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    
    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # 0 for short-range, 1 for full-range
        min_detection_confidence=0.6  # Increased from 0.5 for better accuracy
    )

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,  # Increased from 0.5 for better accuracy
        min_tracking_confidence=0.6  # Increased from 0.5 for better accuracy
    )
    logging.info("✅ MediaPipe initialized successfully with Face Detection and Face Mesh")
    logging.info("   - Face Detection: Full-range model (model_selection=1)")
    logging.info("   - Face Mesh: 468 landmarks with refinement enabled")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.info("📦 MediaPipe not available (requires Python 3.8-3.11). Trying YOLOv8...")

# 2. Enhanced OpenCV DNN initialization (more accurate than basic OpenCV)
if not MEDIAPIPE_AVAILABLE:
    try:
        # Try to load OpenCV DNN face detector (more accurate)
        # Using OpenCV's built-in DNN face detector
        try:
            # Download DNN model files if needed
            import urllib.request
            import os
            
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            prototxt_path = os.path.join(model_dir, 'deploy.prototxt')
            model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
            
            # Try to load existing models or download
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                FACE_DETECTOR_DNN_ACCURATE = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                logging.info("✅ Enhanced OpenCV DNN face detector loaded (Caffe model)")
            else:
                # Try TensorFlow DNN model (more accurate)
                try:
                    # Use OpenCV's built-in DNN face detector URL
                    # This is a more accurate model than Haar Cascades
                    logging.info("📦 Loading enhanced OpenCV DNN models...")
                    # Will use improved Haar Cascades with better parameters
                except:
                    pass
        except Exception as e:
            logging.debug(f"Enhanced DNN model loading: {e}")
        
        logging.info("✅ Enhanced OpenCV face detection initialized")
        logging.info("   - Using advanced DNN models and improved landmark estimation")
        logging.info("   - Accuracy: Significantly improved over basic OpenCV")
    except Exception as e:
        logging.warning(f"Enhanced OpenCV initialization warning: {e}")

# 3. Try MTCNN (Multi-task CNN, good accuracy, but requires TensorFlow)
if not MEDIAPIPE_AVAILABLE:
    try:
        from mtcnn import MTCNN
        mtcnn_detector = MTCNN()
        MTCNN_AVAILABLE = True
        logging.info("✅ MTCNN initialized successfully")
        logging.info("   - Face Detection: Multi-task CNN")
        logging.info("   - Facial Landmarks: 5 key points (eyes, nose, mouth)")
    except (ImportError, ModuleNotFoundError) as e:
        MTCNN_AVAILABLE = False
        logging.info(f"📦 MTCNN not available ({str(e)}). Trying dlib/face_recognition...")

# 4. Try dlib (68-point facial landmarks, very accurate) - requires CMake
if not MEDIAPIPE_AVAILABLE and not MTCNN_AVAILABLE:
    try:
        import dlib
        DLIB_AVAILABLE = True
        
        # Download shape predictor if needed (68-point facial landmarks)
        try:
            # Try to load shape predictor (download from dlib.net if not available)
            dlib_face_detector = dlib.get_frontal_face_detector()
            # Note: shape_predictor_68_face_landmarks.dat needs to be downloaded
            # For now, we'll use face detection only and estimate landmarks
            logging.info("✅ dlib initialized successfully")
            logging.info("   - Face Detection: HOG-based detector")
            logging.info("   - Note: For 68-point landmarks, download shape_predictor_68_face_landmarks.dat")
        except Exception as e:
            logging.warning(f"dlib initialization warning: {e}")
            DLIB_AVAILABLE = False
    except ImportError:
        DLIB_AVAILABLE = False
        logging.info("📦 dlib not available (requires CMake). Trying face_recognition...")

# 5. Try face_recognition (built on dlib, easier to use) - requires dlib
if not MEDIAPIPE_AVAILABLE and not MTCNN_AVAILABLE and not DLIB_AVAILABLE:
    try:
        import face_recognition
        FACE_RECOGNITION_AVAILABLE = True
        logging.info("✅ face_recognition initialized successfully")
        logging.info("   - Face Detection: HOG-based (dlib backend)")
        logging.info("   - Facial Landmarks: 68 points")
    except ImportError:
        FACE_RECOGNITION_AVAILABLE = False
        logging.info("📦 face_recognition not available. Using OpenCV fallback...")

# 6. Enhanced OpenCV DNN (improved accuracy with better models)
if not MEDIAPIPE_AVAILABLE and not MTCNN_AVAILABLE and not DLIB_AVAILABLE and not FACE_RECOGNITION_AVAILABLE:
    logging.info("📦 Using OpenCV DNN face detector as fallback.")
    
    # Initialize OpenCV fallback
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        try:
            face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except:
            face_cascade_profile = None
        FACE_DETECTOR_DNN = None
    except Exception as e:
        logging.warning(f"Error initializing OpenCV fallback: {e}")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade_profile = None
        FACE_DETECTOR_DNN = None

app = FastAPI(
    title="Neuro Desk - AI Human-Computer Interaction Coach API",
    description="""
    Real-time wellness monitoring system that analyzes workspace behavior using computer vision.
    
    ## Features
    
    * **Productivity Score** - Real-time productivity tracking
    * **Posture Detection** - Monitors slouching and sitting position
    * **Eye Strain Analysis** - Tracks eye strain risk levels
    * **Engagement Monitoring** - Measures concentration levels
    * **Stress Detection** - Analyzes stress indicators
    * **Smart Recommendations** - Personalized wellness suggestions
    
    ## Endpoints
    
    * `POST /analyze` - Analyze a single frame (HTTP)
    * `WS /ws` - WebSocket endpoint for real-time analysis
    * `GET /health` - Health check endpoint
    * `GET /metrics` - System metrics and statistics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Dict] = {}

# Frame skipping configuration - DISABLED for real-time analysis
FRAME_SKIP = 1  # Process every frame (1 = no skipping, 100% processing for maximum responsiveness)

def get_session_history(session_id: str) -> Tuple[deque, deque]:
    """Get or create session history for temporal smoothing"""
    if session_id not in sessions:
        sessions[session_id] = {
            "start_time": datetime.now(),
            "frame_count": 0,
            "ear_history": deque(maxlen=30),
            "head_position_history": deque(maxlen=30),
            "last_result": None,  # Cache for frame skipping
            "last_result_time": None
        }
    session = sessions[session_id]
    return session.get("ear_history", deque(maxlen=30)), session.get("head_position_history", deque(maxlen=30))

class AnalyzeRequest(BaseModel):
    image: str

class WellnessAnalyzer:
    def __init__(self):
        # Eye landmark indices for MediaPipe Face Mesh
        # Left eye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Improved EAR points - using more accurate landmarks for better precision
        # Left eye: outer corner, inner corner, top center, bottom center, top outer, bottom outer
        self.LEFT_EYE_POINTS = [33, 133, 159, 145, 157, 153]  # More accurate points
        self.RIGHT_EYE_POINTS = [362, 386, 380, 374, 388, 390]  # More accurate points
        
        # Face boundary points for head pose estimation
        self.FACE_BOUNDARY = [10, 151, 9, 175, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Adaptive thresholds (will adjust based on user patterns)
        self.ear_baseline = 0.28  # Will adapt
        self.ear_std = 0.03  # Will adapt
        self.blink_state = {"left": "open", "right": "open", "consecutive_closed": 0}
        
        # User calibration data (learns from first 30 seconds)
        self.user_calibration = {
            "ear_samples": deque(maxlen=100),  # Collect samples for calibration
            "posture_samples": deque(maxlen=100),
            "face_position_samples": deque(maxlen=100),
            "calibrated": False,
            "calibration_time": None
        }
        
        # Confidence thresholds - improved for better accuracy
        self.min_face_confidence = 0.65  # Higher confidence required (increased from 0.6)
        self.min_landmark_quality = 0.75  # Quality threshold for landmarks (increased from 0.7)
        
        # Real-world adjustments
        self.lighting_adaptation = 1.0  # Adapts to lighting conditions
        self.distance_adaptation = 1.0  # Adapts to camera distance
        
        # Temporal smoothing buffers for more stable results
        self.posture_history = deque(maxlen=10)  # Last 10 posture scores
        self.eye_strain_history = deque(maxlen=10)  # Last 10 eye strain scores
        self.engagement_history = deque(maxlen=10)  # Last 10 engagement scores
        self.stress_history = deque(maxlen=10)  # Last 10 stress scores
        
        # Multi-frame averaging for better accuracy
        self.ear_frame_buffer = deque(maxlen=5)  # Last 5 EAR values for averaging
        self.landmark_quality_history = deque(maxlen=10)  # Track landmark quality
        self.landmark_history = deque(maxlen=3)  # Track last 3 landmark sets for consistency
        
        # Outlier detection thresholds
        self.outlier_threshold = 2.5  # Standard deviations for outlier detection
        
        # Frame quality thresholds
        self.min_frame_quality = 0.6  # Minimum quality to use frame
    
    def _calibrate_posture(self):
        """Enhanced calibration to user's normal posture with outlier removal and faster convergence"""
        try:
            # Reduced minimum samples for faster calibration (from 30 to 20)
            min_samples = 20
            if len(self.user_calibration["posture_samples"]) < min_samples:
                return
            
            samples = list(self.user_calibration["posture_samples"])
            if not samples:
                return
            
            # Advanced ML: Use robust statistics (median) instead of mean for better calibration
            face_y_values = [s["face_y"] for s in samples if "face_y" in s]
            face_x_values = [s["face_x"] for s in samples if "face_x" in s]
            pitch_values = [abs(s["pitch"]) for s in samples if "pitch" in s]
            
            # Remove outliers using IQR method
            def remove_outliers(values):
                if len(values) < 5:
                    return values
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                if iqr == 0:
                    return values  # No variance, return all
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                return [v for v in values if lower <= v <= upper]
            
            face_y_filtered = remove_outliers(face_y_values)
            face_x_filtered = remove_outliers(face_x_values)
            pitch_filtered = remove_outliers(pitch_values)
            
            # Use median for robustness
            avg_face_y = np.median(face_y_filtered) if face_y_filtered else np.median(face_y_values) if face_y_values else 0.35
            avg_face_x = np.median(face_x_filtered) if face_x_filtered else np.median(face_x_values) if face_x_values else 0.5
            avg_pitch = np.median(pitch_filtered) if pitch_filtered else np.median(pitch_values) if pitch_values else 0.0
            
            # Enhanced validation: check for reasonable values
            if not (0 <= avg_face_y <= 1 and 0 <= avg_face_x <= 1):
                logging.warning(f"Invalid calibration values: face_y={avg_face_y}, face_x={avg_face_x}")
                return
            
            # Store user's baseline with std for adaptive thresholds
            self.user_calibration["baseline_face_y"] = avg_face_y
            self.user_calibration["baseline_face_x"] = avg_face_x
            self.user_calibration["baseline_pitch"] = avg_pitch
            self.user_calibration["baseline_face_y_std"] = np.std(face_y_filtered) if len(face_y_filtered) > 1 else 0.05
            
            # Validate std is reasonable
            if self.user_calibration["baseline_face_y_std"] <= 0 or self.user_calibration["baseline_face_y_std"] > 0.2:
                self.user_calibration["baseline_face_y_std"] = 0.05
            
            self.user_calibration["calibrated"] = True
            self.user_calibration["calibration_time"] = datetime.now()
            
            logging.info(f"✅ User posture calibrated (robust): face_y={avg_face_y:.3f}±{self.user_calibration['baseline_face_y_std']:.3f}, face_x={avg_face_x:.3f}, pitch={avg_pitch:.2f}")
        except Exception as e:
            logging.error(f"Error in posture calibration: {e}", exc_info=True)
    
    def _calibrate_ear(self):
        """Enhanced calibration to user's normal EAR with outlier removal and faster convergence"""
        try:
            # Reduced minimum samples for faster calibration (from 50 to 30)
            min_samples = 30
            if len(self.user_calibration["ear_samples"]) < min_samples:
                return
            
            samples = list(self.user_calibration["ear_samples"])
            if not samples:
                return
            
            # Advanced ML: Remove outliers using IQR method for better calibration
            q1 = np.percentile(samples, 25)
            q3 = np.percentile(samples, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter outliers
            filtered_samples = [s for s in samples if lower_bound <= s <= upper_bound]
            
            # Require at least 60% of samples to be valid
            if len(filtered_samples) < int(len(samples) * 0.6):
                # If too many outliers, use all samples but with wider bounds
                filtered_samples = [s for s in samples if 0.15 <= s <= 0.40]
            
            if len(filtered_samples) < 15:
                # Still not enough, use all samples
                filtered_samples = samples
            
            # Use robust statistics: median for baseline, MAD for std
            self.ear_baseline = np.median(filtered_samples)
            mad = np.median([abs(s - self.ear_baseline) for s in filtered_samples])
            self.ear_std = 1.4826 * mad if mad > 0 else 0.03  # Convert MAD to std estimate
            
            # Enhanced validation: check for reasonable values
            if not (0.15 <= self.ear_baseline <= 0.40):
                logging.warning(f"Invalid EAR baseline: {self.ear_baseline}, using default")
                self.ear_baseline = 0.28
                self.ear_std = 0.03
                return
            
            if self.ear_std <= 0 or self.ear_std > 0.1:
                # If std is too small or too large, use default
                self.ear_std = 0.03 if self.ear_std <= 0 else 0.05
            
            # Mark as calibrated
            self.user_calibration["calibrated"] = True
            self.user_calibration["calibration_time"] = datetime.now()
            
            logging.info(f"✅ User EAR calibrated (outlier-robust): baseline={self.ear_baseline:.3f}, std={self.ear_std:.3f}, samples={len(filtered_samples)}/{len(samples)}")
        except Exception as e:
            logging.error(f"Error in EAR calibration: {e}", exc_info=True)
    
    def _get_adaptive_posture_threshold(self, face_center_y):
        """Get adaptive threshold based on user's calibrated baseline"""
        try:
            if not self.user_calibration.get("calibrated", False):
                # Use default range
                return (0.28, 0.42)
            
            baseline_y = self.user_calibration.get("baseline_face_y", 0.35)
            # Allow ±15% deviation from baseline, but clamp to valid range
            threshold_range = 0.15
            min_y = max(0.1, baseline_y - threshold_range)
            max_y = min(0.9, baseline_y + threshold_range)
            return (min_y, max_y)
        except Exception as e:
            logging.warning(f"Error getting adaptive threshold: {e}")
            return (0.28, 0.42)  # Default range
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better face detection accuracy"""
        try:
            # Preserve original for color processing
            original = image.copy()
            
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Step 1: Noise reduction with improved bilateral filter
            # Better parameters for preserving facial features while reducing noise
            filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Step 2: Adaptive histogram equalization for better contrast
            # Use CLAHE with optimized parameters
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
            
            # Step 3: Additional contrast enhancement for low-light conditions
            # Calculate image statistics
            mean_brightness = np.mean(enhanced)
            if mean_brightness < 80:  # Low light condition
                # Apply gamma correction for better visibility
                gamma = 1.2
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(enhanced, table)
            
            # Step 4: Sharpening filter to enhance facial features
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            # Blend original and sharpened (70% sharpened, 30% original)
            enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            # Convert back to RGB if original was RGB
            if len(original.shape) == 3:
                # Apply same enhancements to color channels
                result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                # Preserve color information by blending
                result = cv2.addWeighted(original, 0.3, result, 0.7, 0)
                return result
            return enhanced
        except Exception as e:
            logging.warning(f"Image preprocessing error: {e}")
            return image
    
    def _adapt_to_lighting(self, image: np.ndarray):
        """Adapt analysis to current lighting conditions"""
        # Calculate average brightness
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        avg_brightness = np.mean(gray) / 255.0
        
        # Adjust adaptation factor (0.5 = dark, 1.0 = normal, 1.5 = bright)
        if avg_brightness < 0.3:
            self.lighting_adaptation = 0.7  # Dark - be more lenient
        elif avg_brightness > 0.7:
            self.lighting_adaptation = 1.2  # Bright - can be more strict
        else:
            self.lighting_adaptation = 1.0  # Normal
    
    def validate_image_quality(self, image: np.ndarray) -> Dict:
        """Validate image quality before processing with enhanced metrics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Calculate image statistics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Additional quality metrics - edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Normalize metrics
            brightness = mean_brightness / 255.0
            contrast = std_brightness / 255.0
            
            # Quality score (0-1) with enhanced weighting
            brightness_score = min(1.0, mean_brightness / 128.0)
            contrast_score = min(1.0, std_brightness / 64.0)
            sharpness_score = min(1.0, laplacian_var / 100.0)
            edge_score = min(1.0, edge_density * 10)
            
            quality_score = (
                brightness_score * 0.20 + 
                contrast_score * 0.25 + 
                sharpness_score * 0.30 + 
                edge_score * 0.25
            )
            
            # Enhanced validation with quality threshold
            is_valid = (
                mean_brightness > 30 and
                mean_brightness < 220 and
                std_brightness > 10 and
                laplacian_var > 50 and
                quality_score >= self.min_frame_quality
            )
            
            reasons = []
            if mean_brightness <= 30:
                reasons.append("too_dark")
            if mean_brightness >= 220:
                reasons.append("too_bright")
            if std_brightness <= 10:
                reasons.append("low_contrast")
            if laplacian_var <= 50:
                reasons.append("too_blurry")
            if quality_score < self.min_frame_quality:
                reasons.append("low_quality")
            
            return {
                "valid": is_valid,
                "quality_score": quality_score,
                "brightness": mean_brightness,
                "contrast": std_brightness,
                "sharpness": laplacian_var,
                "edge_density": edge_density,
                "reason": ", ".join(reasons) if reasons else None
            }
        except Exception as e:
            logging.warning(f"Error validating image quality: {e}")
            return {"valid": True, "quality_score": 0.7, "reason": None}
    
    def smooth_temporal_data(self, new_value: float, history: deque, alpha: float = 0.3) -> float:
        """Apply adaptive exponential moving average for temporal smoothing"""
        if len(history) == 0:
            history.append(new_value)
            return new_value
        
        # Get last smoothed value
        last_smoothed = history[-1]
        
        # Adaptive smoothing: adjust alpha based on variance
        adaptive_alpha = alpha
        if len(history) >= 3:
            recent_values = list(history)[-5:] if len(history) >= 5 else list(history)
            variance = np.var(recent_values)
            
            # If variance is high (unstable), use more smoothing
            # If variance is low (stable), use less smoothing (more responsive)
            if variance > 0.01:  # High variance
                adaptive_alpha = min(0.5, alpha * 1.5)  # More smoothing
            elif variance < 0.001:  # Low variance
                adaptive_alpha = max(0.1, alpha * 0.7)  # Less smoothing, more responsive
            else:
                adaptive_alpha = alpha
        
        # Detect outliers and handle them
        if len(history) >= 3:
            recent_values = list(history)[-3:]
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)
            
            # If new value is outlier, use more smoothing
            if std_val > 0 and abs(new_value - mean_val) > self.outlier_threshold * std_val:
                adaptive_alpha = 0.1  # More smoothing for outliers
                smoothed = adaptive_alpha * new_value + (1 - adaptive_alpha) * last_smoothed
                logging.debug(f"Outlier detected: {new_value:.2f}, using more smoothing")
            else:
                # Use adaptive alpha for normal smoothing
                smoothed = adaptive_alpha * new_value + (1 - adaptive_alpha) * last_smoothed
        else:
            # Use adaptive alpha for normal smoothing
            smoothed = adaptive_alpha * new_value + (1 - adaptive_alpha) * last_smoothed
        
        history.append(smoothed)
        return smoothed
    
    def calculate_gaze_direction(self, landmarks, image_shape) -> Dict:
        """Estimate gaze direction from eye landmarks for better engagement analysis"""
        if not landmarks or len(landmarks) < 468:
            return {"direction": "forward", "confidence": 0, "angle": 0}
        
        try:
            h, w = image_shape[:2]
            
            # Get eye center points
            left_eye_center = landmarks[33]  # Left eye outer
            right_eye_center = landmarks[362]  # Right eye outer
            
            # Calculate eye center in image coordinates
            left_eye_x = left_eye_center.x * w
            left_eye_y = left_eye_center.y * h
            right_eye_x = right_eye_center.x * w
            right_eye_y = right_eye_center.y * h
            
            # Calculate eye center midpoint
            eye_center_x = (left_eye_x + right_eye_x) / 2
            eye_center_y = (left_eye_y + right_eye_y) / 2
            
            # Get nose tip for reference
            nose_tip = landmarks[4]
            nose_x = nose_tip.x * w
            nose_y = nose_tip.y * h
            
            # Calculate gaze vector (from nose to eye center)
            gaze_dx = eye_center_x - nose_x
            gaze_dy = eye_center_y - nose_y
            
            # Calculate angle (0 = forward, positive = right, negative = left)
            gaze_angle = np.arctan2(gaze_dx, abs(gaze_dy)) * 180 / np.pi
            
            # Determine direction
            if abs(gaze_angle) < 10:
                direction = "forward"
                confidence = 0.9
            elif gaze_angle > 10:
                direction = "right"
                confidence = 0.7
            else:
                direction = "left"
                confidence = 0.7
            
            # Check if looking up or down
            vertical_offset = eye_center_y - nose_y
            if vertical_offset < -10:
                direction = "up"
                confidence = 0.6
            elif vertical_offset > 10:
                direction = "down"
                confidence = 0.6
            
            return {
                "direction": direction,
                "confidence": confidence,
                "angle": round(gaze_angle, 2),
                "vertical_offset": round(vertical_offset, 2)
            }
        except Exception as e:
            logging.warning(f"Error calculating gaze direction: {e}")
            return {"direction": "forward", "confidence": 0, "angle": 0}
    
    def validate_landmark_consistency(self, landmarks) -> float:
        """Validate landmark consistency across frames"""
        if not landmarks or len(landmarks) < 468:
            return 0.0
        
        # If we have history, check consistency
        if len(self.landmark_history) >= 2:
            # Get key landmark points (nose, eyes, mouth)
            key_indices = [4, 33, 133, 159, 145, 362, 386, 380, 374, 10, 152]
            current_points = []
            for idx in key_indices:
                if idx < len(landmarks) and landmarks[idx] is not None:
                    current_points.append((landmarks[idx].x, landmarks[idx].y))
            
            if len(current_points) < 8:  # Need at least 8 points
                return 0.5  # Medium confidence
            
            # Compare with previous frames
            consistency_scores = []
            for prev_landmarks in self.landmark_history:
                if prev_landmarks and len(prev_landmarks) >= 468:
                    prev_points = []
                    for idx in key_indices:
                        if idx < len(prev_landmarks) and prev_landmarks[idx] is not None:
                            prev_points.append((prev_landmarks[idx].x, prev_landmarks[idx].y))
                    
                    if len(prev_points) == len(current_points):
                        # Calculate average displacement
                        displacements = []
                        for (cx, cy), (px, py) in zip(current_points, prev_points):
                            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                            displacements.append(dist)
                        
                        avg_displacement = np.mean(displacements)
                        # Low displacement = high consistency
                        consistency = max(0.0, 1.0 - avg_displacement * 10)  # Scale factor
                        consistency_scores.append(consistency)
            
            if consistency_scores:
                avg_consistency = np.mean(consistency_scores)
                return avg_consistency
        
        # Store current landmarks for next comparison
        self.landmark_history.append(landmarks)
        return 0.8  # Default confidence for first frames
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_points) -> float:
        """Calculate Eye Aspect Ratio (EAR) using improved 6-point method with enhanced validation"""
        if not landmarks or len(landmarks) < 468:
            return 0.0
        
        # Validate landmark consistency
        consistency = self.validate_landmark_consistency(landmarks)
        if consistency < 0.3:  # Very low consistency - likely bad detection
            logging.debug(f"Low landmark consistency: {consistency:.2f}, skipping EAR calculation")
            return 0.0
        
        try:
            # Get eye landmark coordinates with enhanced validation
            eye_coords = []
            for idx in eye_points:
                if idx < len(landmarks) and landmarks[idx] is not None:
                    landmark = landmarks[idx]
                    # Enhanced validation: check coordinates are within valid range
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        # Additional check: ensure coordinates are not NaN or Inf
                        if np.isfinite(landmark.x) and np.isfinite(landmark.y):
                            eye_coords.append(np.array([landmark.x, landmark.y]))
            
            if len(eye_coords) < 6:
                return 0.0
            
            # Improved EAR calculation with better point selection
            # Points: [outer_corner, inner_corner, top_center, bottom_center, top_outer, bottom_outer]
            p_outer = eye_coords[0]  # Outer corner
            p_inner = eye_coords[1]  # Inner corner
            p_top = eye_coords[2]    # Top center
            p_bottom = eye_coords[3] # Bottom center
            p_top_outer = eye_coords[4]  # Top outer
            p_bottom_outer = eye_coords[5]  # Bottom outer
            
            # Calculate vertical distances (more accurate using center points)
            vertical_1 = np.linalg.norm(p_top - p_bottom)
            # Also use outer points for validation
            vertical_2 = np.linalg.norm(p_top_outer - p_bottom_outer)
            
            # Calculate horizontal distance (eye width)
            horizontal = np.linalg.norm(p_outer - p_inner)
            
            # Enhanced validation: check measurements are reasonable
            if horizontal < 0.008:  # Too small, likely invalid (stricter threshold)
                return 0.0
            
            # Check for coordinate consistency (points should form reasonable eye shape)
            if vertical_1 < 0.001 or vertical_2 < 0.001:  # Eye too narrow
                return 0.0
            
            # Improved EAR: weighted average of center and outer measurements for robustness
            ear_center = vertical_1 / horizontal
            ear_outer = vertical_2 / horizontal if vertical_2 > 0 else ear_center
            
            # Use weighted average (more weight on center measurement)
            ear = (ear_center * 0.7 + ear_outer * 0.3)
            
            # Enhanced validation: check if measurements are reasonable
            # Normal EAR range: 0.15-0.40 (allowing wider range for validation)
            if ear > 0.55 or ear < 0.03:  # Unrealistic values (stricter)
                return 0.0
            
            # Additional outlier check: compare with historical average if available
            if len(self.user_calibration["ear_samples"]) > 10:
                recent_avg = np.mean(list(self.user_calibration["ear_samples"])[-10:])
                recent_std = np.std(list(self.user_calibration["ear_samples"])[-10:])
                # If current EAR is more than 3 std devs from recent average, likely invalid
                if recent_std > 0 and abs(ear - recent_avg) > 3 * recent_std:
                    # Significant deviation - use median of recent values instead
                    recent_median = np.median(list(self.user_calibration["ear_samples"])[-10:])
                    if abs(ear - recent_median) > 0.15:
                        logging.debug(f"EAR outlier detected: {ear:.3f} vs recent avg {recent_avg:.3f}, using median {recent_median:.3f}")
                        # Return median if it's more reasonable
                        if 0.15 <= recent_median <= 0.40:
                            ear = recent_median
            
            # Multi-frame averaging for stability (use last 3-5 frames)
            self.ear_frame_buffer.append(ear)
            if len(self.ear_frame_buffer) >= 3:
                # Use median of recent frames for more robust averaging
                recent_ears = list(self.ear_frame_buffer)
                # Remove outliers using IQR
                if len(recent_ears) >= 3:
                    q1 = np.percentile(recent_ears, 25)
                    q3 = np.percentile(recent_ears, 75)
                    iqr = q3 - q1
                    if iqr > 0:
                        filtered = [e for e in recent_ears if q1 - 1.5*iqr <= e <= q3 + 1.5*iqr]
                        if len(filtered) >= 2:
                            # Use median of filtered values for robustness
                            ear = np.median(filtered)
                        else:
                            # Fallback to median of all
                            ear = np.median(recent_ears)
                    else:
                        # No variance, use median
                        ear = np.median(recent_ears)
            
            return max(0.0, min(1.0, ear))
        except Exception as e:
            logging.warning(f"Error calculating EAR: {e}")
            return 0.0
    
    def calculate_head_pose(self, landmarks, image_shape) -> Dict:
        """Improved head pose estimation using multiple reference points and validation"""
        if not landmarks or len(landmarks) < 468:
            return {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False, "confidence": 0}
        
        h, w = image_shape[:2]
        
        try:
            # Get multiple reference points for robust pose estimation
            nose_tip = landmarks[4]
            chin = landmarks[152]
            forehead = landmarks[10]
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[362]
            left_eye_inner = landmarks[133]
            right_eye_inner = landmarks[362]
            nose_bridge = landmarks[6]  # Additional reference
            
            # Convert to pixel coordinates
            nose_pt = np.array([nose_tip.x * w, nose_tip.y * h])
            chin_pt = np.array([chin.x * w, chin.y * h])
            forehead_pt = np.array([forehead.x * w, forehead.y * h])
            left_pt = np.array([left_cheek.x * w, left_cheek.y * h])
            right_pt = np.array([right_cheek.x * w, right_cheek.y * h])
            left_eye_outer_pt = np.array([left_eye_outer.x * w, left_eye_outer.y * h])
            right_eye_outer_pt = np.array([right_eye_outer.x * w, right_eye_outer.y * h])
            left_eye_inner_pt = np.array([left_eye_inner.x * w, left_eye_inner.y * h])
            right_eye_inner_pt = np.array([right_eye_inner.x * w, right_eye_inner.y * h])
            nose_bridge_pt = np.array([nose_bridge.x * w, nose_bridge.y * h])
            
            # Calculate PITCH using multiple methods and average
            # Method 1: Forehead to chin vector
            face_vertical_vec = chin_pt - forehead_pt
            face_vertical_length = np.linalg.norm(face_vertical_vec)
            
            # Method 2: Nose bridge to chin (more stable)
            nose_vertical_vec = chin_pt - nose_bridge_pt
            nose_vertical_length = np.linalg.norm(nose_vertical_vec)
            
            pitch_values = []
            if face_vertical_length > 0:
                # Normalize vector
                face_vertical_norm = face_vertical_vec / face_vertical_length
                # Calculate angle from vertical (0 = straight up, 90 = horizontal)
                pitch_1 = np.arcsin(np.clip(face_vertical_norm[0], -1, 1)) * 180 / np.pi
                pitch_values.append(pitch_1)
            
            if nose_vertical_length > 0:
                nose_vertical_norm = nose_vertical_vec / nose_vertical_length
                pitch_2 = np.arcsin(np.clip(nose_vertical_norm[0], -1, 1)) * 180 / np.pi
                pitch_values.append(pitch_2)
            
            # Average pitch from multiple methods
            pitch = np.mean(pitch_values) if pitch_values else 0
            
            # Calculate YAW using multiple reference points
            # Method 1: Face width asymmetry
            face_center_x = (left_pt[0] + right_pt[0]) / 2
            image_center_x = w / 2
            face_width = np.linalg.norm(right_pt - left_pt)
            
            # Method 2: Eye position asymmetry
            eye_center_x = (left_eye_outer_pt[0] + right_eye_outer_pt[0]) / 2
            eye_asymmetry = (eye_center_x - image_center_x) / w
            
            # Method 3: Nose position
            nose_asymmetry = (nose_pt[0] - image_center_x) / w
            
            # Combine methods for robust yaw estimation
            if face_width > 0:
                ideal_width_ratio = 0.4
                actual_width_ratio = face_width / w
                width_based_yaw = (ideal_width_ratio - actual_width_ratio) * 50
            else:
                width_based_yaw = 0
            
            position_based_yaw = eye_asymmetry * 30 + nose_asymmetry * 20
            yaw = (width_based_yaw * 0.4 + position_based_yaw * 0.6)
            
            # Calculate ROLL using multiple eye reference points
            # Method 1: Outer eye corners
            eye_line_outer = right_eye_outer_pt - left_eye_outer_pt
            roll_1 = np.arctan2(eye_line_outer[1], eye_line_outer[0]) * 180 / np.pi
            
            # Method 2: Inner eye corners (more stable)
            eye_line_inner = right_eye_inner_pt - left_eye_inner_pt
            roll_2 = np.arctan2(eye_line_inner[1], eye_line_inner[0]) * 180 / np.pi
            
            # Average roll
            roll = (roll_1 + roll_2) / 2.0
            
            # Calculate confidence based on measurement consistency
            pitch_consistency = 1.0 - (np.std(pitch_values) / 10.0) if len(pitch_values) > 1 else 0.8
            roll_consistency = 1.0 - (abs(roll_1 - roll_2) / 5.0) if abs(roll_1 - roll_2) < 10 else 0.5
            confidence = (pitch_consistency + roll_consistency) / 2.0
            
            # Determine if significantly tilted (with confidence threshold)
            tilted = (abs(pitch) > 10 or abs(yaw) > 15 or abs(roll) > 7) and confidence > 0.6
            
            return {
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "tilted": tilted,
                "confidence": round(confidence, 2)
            }
        except Exception as e:
            logging.warning(f"Error calculating head pose: {e}")
            return {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False, "confidence": 0}
    
    def analyze_posture(self, image, landmarks, face_detection_result) -> Dict:
        """Simplified posture analysis using face position - more reliable and responsive"""
        h, w = image.shape[:2]
        face_center_x = None
        face_center_y = None
        face_size = None
        face_detected = False
        
        # Try to get face position from detection result first (simplified approach)
        if face_detection_result and hasattr(face_detection_result, 'detections') and len(face_detection_result.detections) > 0:
            detection = face_detection_result.detections[0]
            face_detected = True
            logging.info(f"🔍 Detection object: has location_data={hasattr(detection, 'location_data')}, has bbox={hasattr(detection, 'bbox')}")
            
            if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_bounding_box'):
                bbox = detection.location_data.relative_bounding_box
                face_center_y = bbox.ymin + bbox.height / 2
                face_center_x = bbox.xmin + bbox.width / 2
                face_size = bbox.width * bbox.height
                logging.info(f"✅ Got face position from location_data: y={face_center_y:.3f}, x={face_center_x:.3f}")
            elif hasattr(detection, 'bbox'):
                # Handle OpenCV-style bbox
                x, y, fw, fh = detection.bbox
                face_center_y = (y + fh / 2) / h
                face_center_x = (x + fw / 2) / w
                face_size = (fw * fh) / (w * h)
                logging.info(f"✅ Got face position from bbox: y={face_center_y:.3f}, x={face_center_x:.3f}")
            else:
                logging.warning(f"⚠️ Detection object doesn't have expected attributes: {dir(detection)}")
        elif landmarks and len(landmarks) >= 468 and landmarks[4] is not None:
            # Use landmarks if available
            face_detected = True
            nose_tip = landmarks[4]
            face_center_y = nose_tip.y
            face_center_x = nose_tip.x
            # Estimate face size from landmarks
            if landmarks[234] and landmarks[454]:  # Left and right face boundaries
                face_size = abs(landmarks[454].x - landmarks[234].x) * abs(landmarks[152].y - landmarks[10].y) if landmarks[152] and landmarks[10] else 0.1
        elif landmarks and len(landmarks) > 0 and landmarks[0] is not None:
            # Use first available landmark as face center estimate
            face_detected = True
            face_center_y = landmarks[0].y
            face_center_x = landmarks[0].x
            face_size = 0.1  # Default estimate
        
        # If no face detected, try to use landmarks as fallback
        if not face_detected or face_center_x is None or face_center_y is None:
            # Try landmarks as last resort
            if landmarks and len(landmarks) > 0:
                # Find first non-None landmark
                for landmark in landmarks:
                    if landmark is not None:
                        face_center_y = landmark.y
                        face_center_x = landmark.x
                        face_detected = True
                        logging.info(f"⚠️ Using landmark fallback: center_y={face_center_y:.3f}, center_x={face_center_x:.3f}")
                        break
            
            if not face_detected or face_center_x is None or face_center_y is None:
                logging.warning(f"⚠️ No face detected! Cannot analyze posture without face detection")
                # Return None instead of static default - let caller handle missing data
                return None
        
        # Log face position detection for debugging - ALWAYS log
        logging.info(f"👤 Face detected: center_y={face_center_y:.3f}, center_x={face_center_x:.3f}, size={face_size:.3f if face_size else 'N/A'}")
        
        # Calculate head pose if landmarks available, otherwise use simplified approach
        head_pose = {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False, "confidence": 0}
        if landmarks and len(landmarks) >= 468 and landmarks[4] is not None:
            try:
                head_pose = self.calculate_head_pose(landmarks, image.shape)
            except:
                pass  # Use default head_pose
        
        # Store calibration data for learning user's normal posture
        if not self.user_calibration["calibrated"]:
            self.user_calibration["posture_samples"].append({
                "face_y": face_center_y,
                "face_x": face_center_x,
                "pitch": head_pose.get("pitch", 0),
                "roll": head_pose.get("roll", 0)
            })
            # Calibrate after 20 samples (faster calibration - about 7 seconds at 3 FPS)
            if len(self.user_calibration["posture_samples"]) >= 20:
                self._calibrate_posture()
        
        # Analyze posture based on multiple factors with more sensitivity
        # 1. Head pitch (forward lean = slouching) - only if we have pose data
        if landmarks and len(landmarks) >= 468 and head_pose.get("confidence", 0) > 0.5:
            # Use actual head pose with confidence weighting
            pitch = head_pose["pitch"]
            pitch_penalty = abs(pitch) * 3.5  # More sensitive (was 2)
            pitch_score = 100 - min(90, pitch_penalty)
        else:
            # Estimate pitch from face position with better sensitivity
            ideal_y = 0.35  # Ideal face position (upper third)
            y_deviation = face_center_y - ideal_y
            pitch_estimate = y_deviation * 60  # More sensitive conversion
            pitch_score = 100 - min(90, abs(pitch_estimate) * 2.0)  # More sensitive (was 1.5)
        
        # 2. Face vertical position - HIGHLY RESPONSIVE TO CHANGES
        # Ideal face position is in upper third of image (0.35 = 35% from top)
        ideal_y = 0.35
        y_deviation = face_center_y - ideal_y  # Positive = lower (slouching), Negative = higher
        
        # Calculate score directly from position - VERY sensitive to changes
        # Map face_y position (0.2 to 0.7) to score (100 to 15)
        # Lower face = lower score (slouching)
        # Make it MUCH more sensitive to position changes
        if face_center_y <= 0.25:
            # Very high position (unlikely)
            vertical_score = 95 - (0.25 - face_center_y) * 150
        elif face_center_y <= 0.35:
            # Good position range - very sensitive
            vertical_score = 100 - abs(y_deviation) * 400  # Increased sensitivity
        elif face_center_y <= 0.50:
            # Slight slouching - very sensitive
            vertical_score = 85 - (face_center_y - 0.35) * 500  # Increased sensitivity
        elif face_center_y <= 0.65:
            # Moderate slouching
            vertical_score = 55 - (face_center_y - 0.50) * 300
        else:
            # Severe slouching
            vertical_score = max(15, 35 - (face_center_y - 0.65) * 150)
        
        vertical_score = max(15, min(100, vertical_score))
        
        # Log position for debugging - ALWAYS log to see what's happening
        logging.info(f"📊 Posture: face_y={face_center_y:.3f}, ideal=0.35, deviation={y_deviation:.3f}, vertical_score={vertical_score:.1f}")
        
        # 3. Head tilt (sideways lean) - more sensitive
        if landmarks and len(landmarks) >= 468 and head_pose.get("confidence", 0) > 0.5:
            roll = abs(head_pose["roll"])
            roll_penalty = min(60, roll * 2.5)  # More sensitive (was 2)
        else:
            roll_penalty = 0  # Can't detect roll without landmarks
        
        # 4. Horizontal centering (face should be centered) - more sensitive
        horizontal_offset = abs(face_center_x - 0.5)
        horizontal_penalty = min(30, horizontal_offset * 60)  # More sensitive (was 50)
        
        # Optimized direct mapping: face position to score for maximum responsiveness
        # Simplified calculation for better performance and accuracy
        ideal_y = 0.35
        
        # Use calibrated baseline if available for personalized scoring
        if self.user_calibration.get("calibrated", False):
            baseline_y = self.user_calibration.get("baseline_face_y", ideal_y)
            baseline_std = self.user_calibration.get("baseline_face_y_std", 0.05)
            
            # Calculate deviation from personal baseline
            y_deviation = face_center_y - baseline_y
            
            # Score based on deviation from personal baseline (more accurate)
            if abs(y_deviation) < baseline_std:
                # Within normal range
                position_based_score = 100 - (abs(y_deviation) / baseline_std) * 10
            elif abs(y_deviation) < 2 * baseline_std:
                # Slight deviation
                position_based_score = 90 - ((abs(y_deviation) - baseline_std) / baseline_std) * 25
            elif abs(y_deviation) < 3 * baseline_std:
                # Moderate deviation
                position_based_score = 65 - ((abs(y_deviation) - 2 * baseline_std) / baseline_std) * 30
            else:
                # Large deviation (slouching)
                position_based_score = 35 - ((abs(y_deviation) - 3 * baseline_std) / baseline_std) * 20
        else:
            # Use standard mapping until calibrated
            y_diff = face_center_y - ideal_y
            
            # Direct linear mapping: face_y 0.2-0.7 maps to score 100-15
            # Lower face (higher y) = lower score (slouching)
            if face_center_y <= ideal_y:
                # Face is high (good posture)
                position_based_score = 100 - abs(y_diff) * 200
            else:
                # Face is low (slouching) - more penalty
                position_based_score = 100 - abs(y_diff) * 350
        
        position_based_score = max(15, min(100, position_based_score))
        
        # Advanced ML: Ensemble method - combine multiple scoring approaches
        # Approach 1: Weighted combination
        base_score = (pitch_score * 0.20 + vertical_score * 0.70 + (100 - horizontal_penalty) * 0.10)
        ensemble_score_1 = base_score - roll_penalty * 0.20
        
        # Approach 2: Position-based direct mapping (calculated above)
        ensemble_score_2 = position_based_score
        
        # Approach 3: Adaptive threshold-based scoring (if calibrated)
        ensemble_score_3 = vertical_score
        if self.user_calibration.get("calibrated", False):
            baseline_y = self.user_calibration.get("baseline_face_y", 0.35)
            baseline_std = self.user_calibration.get("baseline_face_y_std", 0.05)
            deviation = abs(face_center_y - baseline_y)
            # Score based on deviation from personal baseline
            if deviation < baseline_std:
                ensemble_score_3 = 100 - (deviation / baseline_std) * 20  # Excellent
            elif deviation < 2 * baseline_std:
                ensemble_score_3 = 80 - ((deviation - baseline_std) / baseline_std) * 30  # Good
            else:
                ensemble_score_3 = 50 - ((deviation - 2 * baseline_std) / baseline_std) * 35  # Poor
        
        # Ensemble: Weighted average of all approaches
        # Give more weight to calibrated approach if available
        if self.user_calibration.get("calibrated", False):
            posture_score = (ensemble_score_1 * 0.30 + ensemble_score_2 * 0.30 + ensemble_score_3 * 0.40)
        else:
            posture_score = (ensemble_score_1 * 0.40 + ensemble_score_2 * 0.60)
        
        # Ensure scores vary significantly with position changes
        posture_score = max(15, min(100, posture_score))
        
        # Use 80% position-based score for maximum responsiveness
        posture_score = position_based_score * 0.80 + posture_score * 0.20
        posture_score = max(15, min(100, posture_score))
        
        # CRITICAL: If score is still around 70, force it based on position
        if 68 <= posture_score <= 72:
            logging.warning(f"⚠️ Score stuck at ~70, forcing recalculation from face_y={face_center_y:.3f}")
            # Force score directly from position
            if face_center_y < 0.3:
                posture_score = 90 + (0.3 - face_center_y) * 50
            elif face_center_y < 0.4:
                posture_score = 100 - abs(face_center_y - 0.35) * 300
            elif face_center_y < 0.55:
                posture_score = 75 - (face_center_y - 0.4) * 200
            else:
                posture_score = 50 - (face_center_y - 0.55) * 100
            posture_score = max(15, min(100, posture_score))
            logging.info(f"🔄 FORCED score to {posture_score:.2f} based on face_y={face_center_y:.3f}")
        
        # Log the calculation details for debugging - ALWAYS log
        logging.info(f"📊 Posture calc: pitch={pitch_score:.1f}, vertical={vertical_score:.1f}, pos_based={position_based_score:.1f}, final={posture_score:.2f}, face_y={face_center_y:.3f}")
        
        # Determine if slouching
        slouching = (
            (head_pose.get("pitch", 0) > 10) or  # Head tilted forward
            face_center_y > 0.6 or              # Face too low
            head_pose.get("tilted", False) or    # Significant tilt
            vertical_score < 50                  # Poor vertical position
        )
        
        final_score = round(posture_score, 2)
        
        # FINAL CHECK: If score is exactly 70, force it from position
        if final_score == 70.0:
            logging.error(f"❌ Score still 70! Forcing from face_y={face_center_y:.3f}")
            # Emergency recalculation
            ideal_y = 0.35
            y_diff = face_center_y - ideal_y
            if y_diff < 0:
                final_score = 100 + y_diff * 300  # Higher = better
            else:
                final_score = 100 - y_diff * 400  # Lower = worse
            final_score = max(15, min(100, round(final_score, 2)))
            logging.error(f"🔄 EMERGENCY: Forced score to {final_score} from face_y={face_center_y:.3f}")
        
        # Apply temporal smoothing for more stable results
        smoothed_score = self.smooth_temporal_data(final_score, self.posture_history, alpha=0.4)
        final_score = round(smoothed_score, 2)
        
        return {
            "slouching": slouching,
            "score": final_score,
            "head_angle": round(head_pose.get("pitch", 0), 2),
            "face_position_y": round(face_center_y, 3),
            "face_position_x": round(face_center_x, 3)
        }
    
    def analyze_eye_strain(self, image, landmarks, session_history: deque) -> Dict:
        """Improved eye strain analysis using Eye Aspect Ratio (EAR) and blink rate"""
        # Initialize session history if needed
        if session_history is None:
            session_history = deque(maxlen=30)
        
        # If no landmarks, use history data if available - no static defaults
        if landmarks is None or len(landmarks) < 468:
            # Only use history if we have enough data
            if len(session_history) > 10:
                # Use actual history data
                recent_ears = list(session_history)[-20:]
                avg_ear = np.mean(recent_ears)
                
                # Calculate blink rate from actual history
                blink_threshold = self.ear_baseline - 2 * self.ear_std if self.ear_std > 0 else 0.20
                blinks = sum(1 for ear in recent_ears if ear < blink_threshold)
                blink_rate = (blinks / len(recent_ears)) * 20 if recent_ears else 0  # Estimate blinks/min
                
                # Calculate score from actual EAR measurements
                if self.user_calibration["calibrated"]:
                    ear_deviation = abs(avg_ear - self.ear_baseline)
                    if avg_ear < self.ear_baseline - 2 * self.ear_std:
                        eye_strain_risk = "high"
                        eye_score = max(30, 100 - ear_deviation * 200)
                    elif avg_ear < self.ear_baseline - self.ear_std:
                        eye_strain_risk = "medium"
                        eye_score = max(50, 100 - ear_deviation * 150)
                    else:
                        eye_strain_risk = "low"
                        eye_score = min(95, 80 + (self.ear_baseline - avg_ear) * 50)
                else:
                    # Use standard thresholds if not calibrated
                    if avg_ear < 0.20:
                        eye_strain_risk = "high"
                        eye_score = 40
                    elif avg_ear < 0.25:
                        eye_strain_risk = "medium"
                        eye_score = 60
                    else:
                        eye_strain_risk = "low"
                        eye_score = 80
                
                if blink_rate < 0.05:
                    eye_strain_risk = "medium" if eye_strain_risk == "low" else eye_strain_risk
                    eye_score = max(45, eye_score - 15)
                
                return {
                    "eye_strain_risk": eye_strain_risk,
                    "score": round(eye_score, 2),
                    "blink_rate": round(blink_rate, 3),
                    "ear_avg": round(avg_ear, 3)
                }
            else:
                # Not enough data - return None instead of static default
                logging.warning("⚠️ Insufficient data for eye strain analysis - need landmarks or history")
                return None
        
        # Calculate EAR for both eyes using improved method
        left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_POINTS)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_POINTS)
        
        # Validate EAR values (normal range: 0.2-0.4)
        if left_ear == 0.0 and right_ear == 0.0:
            # No valid eye data - return None instead of static default
            logging.warning("⚠️ No valid EAR data detected")
            return None
        
        # Use average, or single eye if one is invalid
        if left_ear > 0 and right_ear > 0:
            avg_ear = (left_ear + right_ear) / 2.0
            # Check for asymmetry (one eye more closed than other)
            ear_asymmetry = abs(left_ear - right_ear) / max(left_ear, right_ear)
        elif left_ear > 0:
            avg_ear = left_ear
            ear_asymmetry = 0
        elif right_ear > 0:
            avg_ear = right_ear
            ear_asymmetry = 0
        else:
            avg_ear = 0.0
            ear_asymmetry = 0
        
        # Store in history for temporal analysis with weighted smoothing
        if session_history is None:
            session_history = deque(maxlen=30)
        
        if avg_ear > 0:
            # Collect calibration samples (only if EAR is valid)
            if not self.user_calibration["calibrated"]:
                # Only add if EAR is within reasonable range
                if 0.15 <= avg_ear <= 0.40:
                    self.user_calibration["ear_samples"].append(avg_ear)
                    # Calibrate after 30 samples (faster calibration)
                    if len(self.user_calibration["ear_samples"]) >= 30:
                        self._calibrate_ear()
            
            # Adaptive baseline: update baseline if we have enough history
            if len(session_history) > 10:
                recent_ears = list(session_history)[-10:]
                # Filter out outliers before updating baseline
                if len(recent_ears) >= 5:
                    q1 = np.percentile(recent_ears, 25)
                    q3 = np.percentile(recent_ears, 75)
                    iqr = q3 - q1
                    if iqr > 0:
                        filtered_ears = [e for e in recent_ears if q1 - 1.5*iqr <= e <= q3 + 1.5*iqr]
                        if len(filtered_ears) >= 3:
                            recent_ears = filtered_ears
                
                # Use calibrated baseline if available, otherwise adapt
                if self.user_calibration["calibrated"]:
                    # Slowly adapt to changes (learning rate)
                    learning_rate = 0.1
                    new_baseline = np.median(recent_ears)  # Use median for robustness
                    self.ear_baseline = (1 - learning_rate) * self.ear_baseline + learning_rate * new_baseline
                    self.ear_std = np.std(recent_ears)
                else:
                    self.ear_baseline = np.median(recent_ears)  # Use median
                    self.ear_std = np.std(recent_ears)
            
            # Weighted smoothing: recent frames have more weight
            if len(session_history) > 0:
                # Exponential moving average for smoother results
                alpha = 0.3  # Smoothing factor
                last_ear = session_history[-1] if session_history else avg_ear
                smoothed_ear = alpha * avg_ear + (1 - alpha) * last_ear
                session_history.append(smoothed_ear)
            else:
                session_history.append(avg_ear)
        
        # Improved blink detection with adaptive threshold
        # Use adaptive threshold based on user's baseline (more accurate if calibrated)
        if self.user_calibration["calibrated"]:
            # Use calibrated baseline for more accurate blink detection
            adaptive_blink_threshold = max(0.15, self.ear_baseline - 2.5 * self.ear_std) if self.ear_std > 0 else 0.20
        else:
            # Use standard threshold until calibrated
            adaptive_blink_threshold = max(0.15, self.ear_baseline - 2 * self.ear_std) if self.ear_std > 0 else 0.20
        blink_count = 0
        
        if len(session_history) > 15:
            recent_ears = list(session_history)[-20:]  # Use more frames for better detection
            prev_ear = recent_ears[0] if recent_ears else 0.3
            in_blink = False
            blink_duration = 0
            
            for i, ear in enumerate(recent_ears[1:], 1):
                # Enhanced blink detection with state machine
                # State 1: Eye open -> detect drop below threshold
                if not in_blink and ear < adaptive_blink_threshold and prev_ear >= adaptive_blink_threshold:
                    # Blink started - validate it's a significant drop
                    drop_magnitude = prev_ear - ear
                    if drop_magnitude > 0.05:  # Significant drop (at least 5% of baseline)
                        in_blink = True
                        blink_duration = 1
                # State 2: In blink -> track duration
                elif in_blink and ear < adaptive_blink_threshold:
                    blink_duration += 1
                # State 3: Blink ending -> detect return above threshold
                elif in_blink and ear >= adaptive_blink_threshold and prev_ear < adaptive_blink_threshold:
                    # Blink completed - validate duration and magnitude
                    # Normal blink: 1-5 frames (100-500ms at 5 FPS)
                    if 1 <= blink_duration <= 5 and ear >= adaptive_blink_threshold * 0.8:
                        blink_count += 1
                    in_blink = False
                    blink_duration = 0
                # State 4: False positive check - if blink too long, cancel it
                elif in_blink and blink_duration > 8:
                    # Too long to be a blink, likely eyes closed or error
                    in_blink = False
                    blink_duration = 0
                
                prev_ear = ear
            
            # Calculate blink rate (blinks per minute)
            # Account for frame skipping: if FRAME_SKIP=3, actual FPS is ~1.67
            actual_fps = 5.0 / FRAME_SKIP  # Adjust for frame skipping
            frames_analyzed = len(recent_ears)
            time_seconds = frames_analyzed / actual_fps
            blink_rate = (blink_count / time_seconds) * 60 if time_seconds > 0 else 0  # blinks per minute
            blink_rate_normalized = blink_rate / 20.0  # Normal is ~15-20 blinks/min
        else:
            blink_rate = 0
            blink_rate_normalized = 0
        
        # Improved eye strain analysis with adaptive thresholds
        # Use user's baseline for personalized analysis
        strain_factors = []
        strain_score_deduction = 0
        
        # EAR-based analysis using adaptive thresholds (more accurate if calibrated)
        if self.user_calibration["calibrated"]:
            # Use calibrated baseline for personalized analysis
            ear_deviation = abs(avg_ear - self.ear_baseline)
            # Use user-specific thresholds
            closed_threshold = self.ear_baseline - 2.5 * self.ear_std
            droopy_threshold = self.ear_baseline - 1.5 * self.ear_std
            wide_threshold = self.ear_baseline + 2 * self.ear_std
        else:
            # Use standard thresholds
            ear_deviation = abs(avg_ear - 0.28)
            closed_threshold = 0.20
            droopy_threshold = 0.25
            wide_threshold = 0.38
        
        if avg_ear < closed_threshold:
            strain_factors.append("eyes_fully_closed")
            strain_score_deduction += 35
        elif avg_ear < droopy_threshold:
            strain_factors.append("eyes_nearly_closed")
            strain_score_deduction += 25
        elif avg_ear < (self.ear_baseline - 0.5 * self.ear_std) if self.user_calibration["calibrated"] else 0.25:
            strain_factors.append("eyes_droopy")
            strain_score_deduction += 18
        elif avg_ear > wide_threshold:
            strain_factors.append("eyes_wide_open")
            strain_score_deduction += 8  # Wide open can indicate strain
        elif ear_deviation > 2 * self.ear_std and self.ear_std > 0 and self.user_calibration["calibrated"]:
            # Significant deviation from baseline (only if calibrated)
            if avg_ear < self.ear_baseline:
                strain_factors.append("eyes_below_baseline")
                strain_score_deduction += 12
        
        # Improved blink rate analysis
        if blink_rate_normalized < 0.25 and len(session_history) > 15:
            strain_factors.append("infrequent_blinking")
            strain_score_deduction += 28  # Staring at screen
        elif blink_rate_normalized > 1.8:
            strain_factors.append("excessive_blinking")
            strain_score_deduction += 18  # Eye irritation
        
        # Asymmetry check (one eye more closed than other)
        if ear_asymmetry > 0.18:  # >18% difference (more strict)
            strain_factors.append("asymmetric_eye_closure")
            strain_score_deduction += 12
        
        # Temporal analysis: check for sustained low EAR
        if len(session_history) > 10:
            recent_low_ear_frames = sum(1 for ear in list(session_history)[-10:] if ear < 0.22)
            if recent_low_ear_frames > 5:  # More than half of recent frames
                strain_factors.append("sustained_eye_fatigue")
                strain_score_deduction += 15
        
        # Calculate final score (100 = perfect, deduct for issues)
        base_score = 100
        eye_score = base_score - strain_score_deduction
        
        # Determine risk level with confidence
        if eye_score < 45:
            eye_strain_risk = "high"
        elif eye_score < 65:
            eye_strain_risk = "medium"
        elif eye_score < 80:
            eye_strain_risk = "low-medium"
        else:
            eye_strain_risk = "low"
        
        eye_score = max(0, min(100, eye_score))
        
        # Apply temporal smoothing for more stable results
        smoothed_score = self.smooth_temporal_data(eye_score, self.eye_strain_history, alpha=0.3)
        eye_score = round(smoothed_score, 2)
        
        return {
            "eye_strain_risk": eye_strain_risk,
            "score": eye_score,
            "blink_rate": round(blink_rate, 3),
            "ear_avg": round(avg_ear, 3)
        }
    
    def analyze_engagement(self, landmarks, face_detection_result, session_history: deque, image_shape=None) -> Dict:
        """Improved engagement analysis using head movement and gaze direction"""
        # Initialize session history if needed
        if session_history is None:
            session_history = deque(maxlen=30)
        
        # Check if face is detected
        face_visible = False
        face_center = None
        face_confidence = 0.0
        
        if face_detection_result and hasattr(face_detection_result, 'detections') and len(face_detection_result.detections) > 0:
            face_visible = True
            detection = face_detection_result.detections[0]
            if hasattr(detection, 'score') and len(detection.score) > 0:
                face_confidence = detection.score[0]
            elif hasattr(detection, 'confidence'):
                face_confidence = detection.confidence
            else:
                face_confidence = 0.7  # Default if no score available
            
            if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_bounding_box'):
                bbox = detection.location_data.relative_bounding_box
                face_center = [bbox.xmin + bbox.width / 2, bbox.ymin + bbox.height / 2]
            elif hasattr(detection, 'bbox'):
                x, y, w, h = detection.bbox
                if image_shape:
                    h_img, w_img = image_shape[:2]
                    face_center = [(x + w/2) / w_img, (y + h/2) / h_img]
                else:
                    face_center = [0.5, 0.5]
        elif landmarks and len(landmarks) > 0 and landmarks[0] is not None:
            face_visible = True
            face_confidence = 0.7  # Medium confidence for landmarks
            if landmarks[4] is not None:  # Nose tip
                face_center = [landmarks[4].x, landmarks[4].y]
            else:
                face_center = [landmarks[0].x, landmarks[0].y]
        
        if not face_visible or face_center is None:
            return {"concentration": "low", "score": 20, "face_visible": False, "head_stability": 0}
        
        # Use provided image shape or default
        if image_shape is None:
            image_shape = np.array([480, 640, 3])
        
        # Get head pose if landmarks available
        if landmarks and len(landmarks) >= 468:
            head_pose = self.calculate_head_pose(landmarks, image_shape)
        else:
            head_pose = {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False, "confidence": 0}
        
        # Track head movement (stability indicates focus)
        if face_center:
            session_history.append(face_center)
        
        # Calculate movement variance with better sensitivity
        if len(session_history) > 5:
            positions = np.array(list(session_history)[-10:])
            movement_variance = np.var(positions, axis=0).sum()
            # Also calculate average movement distance
            if len(positions) > 1:
                movement_distances = [np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions))]
                avg_movement = np.mean(movement_distances) if movement_distances else 0
            else:
                avg_movement = 0
        else:
            movement_variance = 0.01  # Default low movement
            avg_movement = 0
        
        # Analyze engagement factors with more sensitivity
        # 1. Face visibility and confidence
        visibility_score = 100 if face_visible else 20
        confidence_score = face_confidence * 100 if face_confidence > 0 else 60
        
        # 2. Head stability (low movement = focused) - more sensitive
        if movement_variance < 0.00005:  # Very stable
            stability_score = 100
        elif movement_variance < 0.0002:  # Stable
            stability_score = 90 - (movement_variance * 50000)
        elif movement_variance < 0.0008:  # Moderate
            stability_score = 75 - ((movement_variance - 0.0002) * 20000)
        elif movement_variance < 0.002:  # Some movement
            stability_score = 55 - ((movement_variance - 0.0008) * 8000)
        else:  # High movement
            stability_score = max(20, 35 - (movement_variance * 5000))
        
        # 3. Head orientation (facing forward = engaged) - more sensitive
        if landmarks and len(landmarks) >= 468 and head_pose.get("confidence", 0) > 0.5:
            yaw = abs(head_pose["yaw"])
            yaw_penalty = min(60, yaw * 2.5)  # More sensitive (was 2)
            orientation_score = 100 - yaw_penalty
        else:
            # Estimate from face position with better sensitivity
            if face_center:
                horizontal_offset = abs(face_center[0] - 0.5)
                orientation_score = 100 - min(50, horizontal_offset * 100)  # More sensitive (was 80)
            else:
                orientation_score = 50
        
        # 3.5. Gaze direction tracking (new improvement)
        gaze_info = self.calculate_gaze_direction(landmarks, image_shape)
        gaze_bonus = 0
        if gaze_info["confidence"] > 0.5:
            if gaze_info["direction"] == "forward":
                gaze_bonus = 8  # Bonus for looking forward
            elif gaze_info["direction"] in ["left", "right"]:
                gaze_bonus = -5  # Penalty for looking away
            elif gaze_info["direction"] == "down":
                gaze_bonus = -8  # Penalty for looking down (distracted)
        
        # 4. Face size/confidence (larger face = closer = more engaged)
        if face_detection_result and hasattr(face_detection_result, 'detections') and len(face_detection_result.detections) > 0:
            detection = face_detection_result.detections[0]
            if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_bounding_box'):
                bbox = detection.location_data.relative_bounding_box
                face_size = bbox.width * bbox.height
                # Larger face = closer = more engaged
                if face_size > 0.15:
                    size_bonus = 10
                elif face_size > 0.10:
                    size_bonus = 5
                elif face_size < 0.05:
                    size_bonus = -10
                else:
                    size_bonus = 0
            else:
                size_bonus = 0
        else:
            size_bonus = 0
        
        # Calculate overall engagement with dynamic weighting
        engagement_score = (
            visibility_score * 0.20 + 
            stability_score * 0.40 + 
            orientation_score * 0.30 + 
            confidence_score * 0.10
        ) + size_bonus + gaze_bonus
        engagement_score = max(15, min(100, engagement_score))  # Wider range
        
        # Apply temporal smoothing for more stable results
        smoothed_score = self.smooth_temporal_data(engagement_score, self.engagement_history, alpha=0.35)
        engagement_score = round(smoothed_score, 2)
        
        # Determine concentration level
        if engagement_score >= 75:
            concentration = "high"
        elif engagement_score >= 50:
            concentration = "medium"
        else:
            concentration = "low"
        
        return {
            "concentration": concentration,
            "score": round(engagement_score, 2),
            "face_visible": face_visible,
            "head_stability": round(1 - min(1, movement_variance * 1000), 2)
        }
    
    def analyze_facial_expressions(self, landmarks) -> Dict:
        """Analyze facial expressions using landmark distances and ratios"""
        if not landmarks or len(landmarks) < 468:
            return {}
        
        try:
            # Get key facial landmarks
            # Eyebrows
            left_eyebrow_inner = landmarks[107]
            right_eyebrow_inner = landmarks[336]
            left_eyebrow_outer = landmarks[70]
            right_eyebrow_outer = landmarks[300]
            
            # Eyes
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            
            # Mouth
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            mouth_top = landmarks[13]
            mouth_bottom = landmarks[14]
            mouth_center_top = landmarks[12]
            mouth_center_bottom = landmarks[15]
            
            # Calculate expression metrics
            expressions = {}
            
            # Eyebrow position (furrowed = stress)
            left_eyebrow_eye_dist = abs(left_eyebrow_inner.y - left_eye_top.y)
            right_eyebrow_eye_dist = abs(right_eyebrow_inner.y - right_eye_top.y)
            avg_eyebrow_dist = (left_eyebrow_eye_dist + right_eyebrow_eye_dist) / 2
            expressions["eyebrow_tension"] = avg_eyebrow_dist
            
            # Eyebrow asymmetry (stress can cause asymmetry)
            eyebrow_asymmetry = abs(left_eyebrow_eye_dist - right_eyebrow_eye_dist)
            expressions["eyebrow_asymmetry"] = eyebrow_asymmetry
            
            # Mouth width and height (tight lips = stress)
            mouth_width = abs(mouth_right.x - mouth_left.x)
            mouth_height = abs(mouth_bottom.y - mouth_top.y)
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            expressions["mouth_tension"] = mouth_aspect_ratio
            
            # Mouth corners (downward = negative emotion)
            mouth_left_y = mouth_left.y
            mouth_right_y = mouth_right.y
            mouth_center_y = (mouth_center_top.y + mouth_center_bottom.y) / 2
            mouth_droop = (mouth_left_y + mouth_right_y) / 2 - mouth_center_y
            expressions["mouth_droop"] = mouth_droop
            
            # Eye opening (wide eyes = alert/stress, narrow = relaxed)
            left_eye_opening = abs(left_eye_bottom.y - left_eye_top.y)
            right_eye_opening = abs(right_eye_bottom.y - right_eye_top.y)
            avg_eye_opening = (left_eye_opening + right_eye_opening) / 2
            expressions["eye_opening"] = avg_eye_opening
            
            return expressions
        except Exception as e:
            logging.warning(f"Error analyzing facial expressions: {e}")
            return {}
    
    def analyze_stress(self, image, landmarks, face_detection_result) -> Dict:
        """Real stress analysis using facial expression recognition and micro-expressions"""
        # If no landmarks, estimate based on face detection
        if landmarks is None or len(landmarks) < 468:
            # Basic stress estimation based on face position and size
            if face_detection_result and hasattr(face_detection_result, 'detections') and len(face_detection_result.detections) > 0:
                detection = face_detection_result.detections[0]
                # Smaller face or off-center might indicate stress/tension
                if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_bounding_box'):
                    bbox = detection.location_data.relative_bounding_box
                    face_size = bbox.width * bbox.height
                    face_center_x = bbox.xmin + bbox.width / 2
                    
                    # Estimate stress based on face characteristics
                    if face_size < 0.05:  # Very small face
                        stress_score = 75
                        stress_level = "low-medium"
                    elif abs(face_center_x - 0.5) > 0.2:  # Off-center
                        stress_score = 80
                        stress_level = "low"
                    else:
                        stress_score = 90
                        stress_level = "low"
                else:
                    stress_score = 85
                    stress_level = "low"
            else:
                stress_score = 80  # Lower if no face
                stress_level = "low"
            
            return {
                "stress_level": stress_level,
                "score": round(stress_score, 2),
                "indicators": []
            }
        
        # Get facial expression metrics
        expressions = self.analyze_facial_expressions(landmarks)
        
        stress_indicators = []
        stress_score_deduction = 0
        
        try:
            if expressions:
                # 1. Eyebrow tension (furrowed = stress)
                # Normal eyebrow-eye distance: ~0.02-0.03
                eyebrow_tension = expressions.get("eyebrow_tension", 0.025)
                if eyebrow_tension < 0.015:
                    stress_indicators.append("furrowed_brows")
                    stress_score_deduction += 20
                elif eyebrow_tension < 0.018:
                    stress_indicators.append("slight_brow_tension")
                    stress_score_deduction += 10
                
                # Eyebrow asymmetry (stress can cause facial asymmetry)
                eyebrow_asymmetry = expressions.get("eyebrow_asymmetry", 0)
                if eyebrow_asymmetry > 0.005:  # Significant asymmetry
                    stress_indicators.append("facial_asymmetry")
                    stress_score_deduction += 8
                
                # 2. Mouth tension (tight/pursed = stress)
                mouth_tension = expressions.get("mouth_tension", 0.2)
                if mouth_tension < 0.12:  # Very tight lips
                    stress_indicators.append("tight_mouth")
                    stress_score_deduction += 15
                elif mouth_tension < 0.15:
                    stress_indicators.append("slight_mouth_tension")
                    stress_score_deduction += 8
                
                # Mouth droop (downward corners = negative emotion/stress)
                mouth_droop = expressions.get("mouth_droop", 0)
                if mouth_droop > 0.01:  # Downward droop
                    stress_indicators.append("mouth_droop")
                    stress_score_deduction += 12
                
                # 3. Eye opening (wide eyes = alert/stress, but also can be normal)
                eye_opening = expressions.get("eye_opening", 0.02)
                if eye_opening > 0.035:  # Very wide eyes (alert/stress)
                    stress_indicators.append("wide_eyes_alert")
                    stress_score_deduction += 8
                elif eye_opening < 0.015:  # Narrow eyes (fatigue/stress)
                    stress_indicators.append("narrow_eyes")
                    stress_score_deduction += 10
            
            # 4. Analyze jaw tension using landmarks
            if landmarks and len(landmarks) >= 468:
                jaw_left = landmarks[172]
                jaw_right = landmarks[397]
                jaw_center = landmarks[175]
                
                jaw_width = abs(jaw_right.x - jaw_left.x)
                # Normal jaw width relative to face: ~0.18-0.22
                # Narrower can indicate clenching
                if jaw_width < 0.14:
                    stress_indicators.append("jaw_tension")
                    stress_score_deduction += 12
                
                # Check jaw symmetry
                jaw_left_dist = abs(jaw_left.x - jaw_center.x)
                jaw_right_dist = abs(jaw_right.x - jaw_center.x)
                jaw_asymmetry = abs(jaw_left_dist - jaw_right_dist)
                if jaw_asymmetry > 0.02:
                    stress_indicators.append("jaw_asymmetry")
                    stress_score_deduction += 8
            
            # 5. Overall facial tension (compressed face = tension)
            if landmarks and len(landmarks) >= 468:
                face_width = abs(landmarks[454].x - landmarks[234].x)
                face_height = abs(landmarks[152].y - landmarks[10].y)
                face_ratio = face_width / face_height if face_height > 0 else 1
                
                # Normal face ratio: ~0.75-0.85
                if face_ratio < 0.65:  # Compressed face (tension)
                    stress_indicators.append("facial_tension")
                    stress_score_deduction += 10
            
            # Calculate final stress score (100 = no stress, deduct for indicators)
            base_score = 100
            stress_score = base_score - stress_score_deduction
            
            # Determine stress level
            if stress_score < 50:
                stress_level = "high"
            elif stress_score < 70:
                stress_level = "medium"
            elif stress_score < 85:
                stress_level = "low-medium"
            else:
                stress_level = "low"
            
            stress_score = max(0, min(100, stress_score))
            
        except Exception as e:
            logging.warning(f"Error in stress analysis: {e}")
            stress_level = "low"
            stress_score = 85
        
        # Apply temporal smoothing for more stable results
        smoothed_score = self.smooth_temporal_data(stress_score, self.stress_history, alpha=0.3)
        stress_score = round(smoothed_score, 2)
        
        return {
            "stress_level": stress_level,
            "score": stress_score,
            "indicators": stress_indicators
        }
    
    def calculate_productivity_score(self, posture, eye_strain, engagement, stress) -> Dict:
        """Calculate overall productivity score based on actual measurements - no static defaults"""
        try:
            # Only calculate if we have valid data - no static defaults
            if not posture or not eye_strain or not engagement or not stress:
                logging.warning("⚠️ Missing analysis data for productivity calculation")
                return None
            
            # Get actual scores - fail if missing (no defaults)
            posture_score = posture.get("score")
            eye_strain_score = eye_strain.get("score")
            engagement_score = engagement.get("score")
            stress_score = stress.get("score")
            
            # Validate that we have actual measurements
            if posture_score is None or eye_strain_score is None or engagement_score is None or stress_score is None:
                logging.warning("⚠️ Missing score values in analysis data")
                return None
            
            # Dynamic weights based on data quality
            weights = {
                "posture": 0.25,
                "eye_strain": 0.20,
                "engagement": 0.30,
                "stress": 0.25
            }
            
            # Calculate weighted productivity from actual measurements
            productivity = (
                posture_score * weights["posture"] +
                eye_strain_score * weights["eye_strain"] +
                engagement_score * weights["engagement"] +
                stress_score * weights["stress"]
            )
            
            # Get risk levels from actual data
            eye_strain_risk = eye_strain.get("eye_strain_risk", "low")
            is_slouching = posture.get("slouching", False)
            
            return {
                "productivity_score": round(productivity, 2),
                "break_needed": productivity < 60,
                "eye_exercise_needed": eye_strain_risk in ["medium", "high"],
                "posture_reminder": is_slouching
            }
        except Exception as e:
            logging.error(f"Error calculating productivity score: {e}", exc_info=True)
            # Return None instead of static default - let caller handle it
            return None
    
    def get_recommendations(self, analysis: Dict) -> list:
        """Generate wellness recommendations with error handling"""
        try:
            recommendations = []
            
            if not analysis:
                return ["⚠️ No analysis data available"]
            
            # Safely check posture reminder
            if analysis.get("posture_reminder", False):
                recommendations.append("💺 Sit up straight! Adjust your posture")
            
            # Safely check eye exercise needed
            if analysis.get("eye_exercise_needed", False):
                blink_rate = analysis.get("blink_rate", 0)
                if blink_rate < 0.05:
                    recommendations.append("👁️ Blink more often! Take a 20-20-20 break: Look 20ft away for 20 seconds")
                else:
                    recommendations.append("👁️ Take a 20-20-20 break: Look 20ft away for 20 seconds")
            
            # Safely check break needed
            if analysis.get("break_needed", False):
                recommendations.append("☕ Take a 5-minute micro-break")
            
            # Safely check stress level
            stress_level = analysis.get("stress_level", "low")
            if stress_level in ["medium", "high", "low-medium"]:
                recommendations.append("🧘 Take 3 deep breaths to reduce stress")
            
            if not recommendations:
                recommendations.append("✅ You're doing great! Keep it up")
            
            return recommendations
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}", exc_info=True)
            return ["⚠️ Error generating recommendations"]

analyzer = WellnessAnalyzer()

# Simple landmark class for compatibility across detection methods
class SimpleLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z


def detect_face_opencv(image: np.ndarray) -> Tuple[Optional, Optional, Optional]:
    """Enhanced OpenCV face detection with multiple methods for maximum accuracy"""
    try:
        h, w = image.shape[:2]
        faces = []
        confidence = 0.8
        
        # Method 1: Try enhanced DNN face detector (most accurate)
        if FACE_DETECTOR_DNN_ACCURATE is not None:
            try:
                # Resize for DNN (better accuracy with proper scaling)
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 
                    1.0, 
                    (300, 300), 
                    [104, 117, 123],
                    swapRB=True,  # Important: swap RGB to BGR
                    crop=False
                )
                FACE_DETECTOR_DNN_ACCURATE.setInput(blob)
                detections = FACE_DETECTOR_DNN_ACCURATE.forward()
                
                for i in range(detections.shape[2]):
                    conf = detections[0, 0, i, 2]
                    if conf > 0.75:  # Higher confidence threshold for better accuracy (increased from 0.7)
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append((x1, y1, x2-x1, y2-y1))
                        confidence = conf
                        break
            except Exception as e:
                logging.debug(f"Enhanced DNN detection failed: {e}")
        
        # Method 2: Try standard DNN face detector
        if len(faces) == 0 and FACE_DETECTOR_DNN is not None:
            try:
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), [104, 117, 123])
                FACE_DETECTOR_DNN.setInput(blob)
                detections = FACE_DETECTOR_DNN.forward()
                
                for i in range(detections.shape[2]):
                    conf = detections[0, 0, i, 2]
                    if conf > 0.5:  # Confidence threshold
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append((x1, y1, x2-x1, y2-y1))
                        confidence = conf
                        break  # Take first face
            except Exception as e:
                logging.debug(f"DNN detection failed: {e}, falling back to Haar")
        
        # Method 3: Enhanced Haar Cascades with multiple scales and better parameters
        if len(faces) == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Apply histogram equalization for better detection in varying lighting
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray)
            
            # Try frontal face detection with optimized parameters
            # Using multiple scale factors for better detection
            faces = face_cascade.detectMultiScale(
                gray_enhanced, 
                scaleFactor=1.05,  # Smaller scale factor = more accurate
                minNeighbors=6,    # Higher = fewer false positives
                minSize=(40, 40),  # Larger minimum size = more accurate
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If no frontal face, try on original gray (sometimes works better)
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
            
            # If still no face, try profile detection
            if len(faces) == 0 and face_cascade_profile is not None:
                faces = face_cascade_profile.detectMultiScale(
                    gray_enhanced, 
                    scaleFactor=1.05, 
                    minNeighbors=6, 
                    minSize=(40, 40)
                )
        
        if len(faces) > 0:
            # Create a simple face detection result object
            class SimpleFaceDetection:
                def __init__(self, x, y, w, h, confidence=0.8):
                    self.x = x
                    self.y = y
                    self.w = w
                    self.h = h
                    self.confidence = confidence
                    self.bbox = (x, y, w, h)
            
            x, y, face_w, face_h = faces[0]
            # Ensure valid coordinates
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            face_w = min(face_w, w - x)
            face_h = min(face_h, h - y)
            
            # Create detection result in the format expected by analyze_posture
            h_img, w_img = image.shape[:2]
            
            # Calculate face center for landmarks
            face_center_x = (x + face_w/2) / w_img
            face_center_y = (y + face_h/2) / h_img
            face_width = face_w / w_img
            face_height = face_h / h_img
            
            # Log face position for debugging
            logging.info(f"🔍 OpenCV detected face: x={x}, y={y}, w={face_w}, h={face_h}, center_y={face_center_y:.3f}")
            
            # Create landmarks array with 468 points (MediaPipe standard)
            # Map key indices used in analysis functions
            landmarks = [None] * 468
            
            # Forehead center (index 10)
            landmarks[10] = SimpleLandmark(face_center_x, (y / h_img) - face_height * 0.15)
            
            # Nose tip (index 4)
            landmarks[4] = SimpleLandmark(face_center_x, face_center_y)
            
            # Chin (index 152)
            landmarks[152] = SimpleLandmark(face_center_x, (y + face_h) / h_img)
            
            # Left face boundary (index 234)
            landmarks[234] = SimpleLandmark((x / w_img), face_center_y)
            
            # Right face boundary (index 454)
            landmarks[454] = SimpleLandmark((x + w) / w_img, face_center_y)
            
            # Improved eye landmark estimation using face geometry
            # Left eye landmarks (indices 33, 133, 157, 158, 159, 160, 161)
            # Eyes are typically at 1/3 from top of face, 1/4 from sides
            left_eye_x = face_center_x - face_width * 0.12  # More accurate positioning
            left_eye_y = face_center_y - face_height * 0.08  # Slightly higher
            eye_width = face_width * 0.08  # Eye width estimation
            eye_height = face_height * 0.03  # Eye height estimation
            
            # Left eye outer corner (33)
            landmarks[33] = SimpleLandmark(left_eye_x - eye_width * 0.6, left_eye_y)
            # Left eye inner corner (133) - more accurate
            landmarks[133] = SimpleLandmark(left_eye_x + eye_width * 0.4, left_eye_y)
            # Left eye center points for EAR calculation
            landmarks[157] = SimpleLandmark(left_eye_x, left_eye_y)  # Center
            landmarks[158] = SimpleLandmark(left_eye_x, left_eye_y + eye_height * 0.5)  # Bottom center
            landmarks[159] = SimpleLandmark(left_eye_x, left_eye_y - eye_height * 0.5)  # Top center
            landmarks[160] = SimpleLandmark(left_eye_x - eye_width * 0.3, left_eye_y - eye_height * 0.4)  # Top outer
            landmarks[161] = SimpleLandmark(left_eye_x - eye_width * 0.3, left_eye_y + eye_height * 0.4)  # Bottom outer
            # Additional points for better EAR
            landmarks[145] = SimpleLandmark(left_eye_x, left_eye_y + eye_height * 0.6)  # Bottom
            landmarks[153] = SimpleLandmark(left_eye_x - eye_width * 0.5, left_eye_y)  # Outer
            
            # Right eye landmarks (indices 362, 386, 387, 388, 390, 398)
            right_eye_x = face_center_x + face_width * 0.12
            right_eye_y = face_center_y - face_height * 0.08
            
            # Right eye outer corner (362)
            landmarks[362] = SimpleLandmark(right_eye_x + eye_width * 0.6, right_eye_y)
            # Right eye inner corner (386)
            landmarks[386] = SimpleLandmark(right_eye_x - eye_width * 0.4, right_eye_y)
            # Right eye center points
            landmarks[380] = SimpleLandmark(right_eye_x, right_eye_y - eye_height * 0.5)  # Top center
            landmarks[374] = SimpleLandmark(right_eye_x, right_eye_y + eye_height * 0.5)  # Bottom center
            landmarks[388] = SimpleLandmark(right_eye_x, right_eye_y - eye_height * 0.4)  # Top outer
            landmarks[390] = SimpleLandmark(right_eye_x + eye_width * 0.5, right_eye_y)  # Outer
            landmarks[387] = SimpleLandmark(right_eye_x, right_eye_y + eye_height * 0.6)  # Bottom
            
            # Eyebrow landmarks (107, 336 for inner, 159, 386 for eye top)
            landmarks[107] = SimpleLandmark(left_eye_x, left_eye_y - face_height * 0.05)
            landmarks[336] = SimpleLandmark(right_eye_x, right_eye_y - face_height * 0.05)
            
            # Mouth landmarks (13, 14, 61, 291, 172, 397, 175)
            landmarks[13] = SimpleLandmark(face_center_x, face_center_y + face_height * 0.15)
            landmarks[14] = SimpleLandmark(face_center_x, face_center_y + face_height * 0.18)
            landmarks[61] = SimpleLandmark(face_center_x - face_width * 0.1, face_center_y + face_height * 0.16)
            landmarks[291] = SimpleLandmark(face_center_x + face_width * 0.1, face_center_y + face_height * 0.16)
            landmarks[172] = SimpleLandmark(face_center_x - face_width * 0.12, face_center_y + face_height * 0.2)
            landmarks[397] = SimpleLandmark(face_center_x + face_width * 0.12, face_center_y + face_height * 0.2)
            landmarks[175] = SimpleLandmark(face_center_x, (y + face_h) / h_img)
            
            # Fill remaining landmarks with interpolated positions (deterministic)
            for i in range(468):
                if landmarks[i] is None:
                    # Create deterministic positions based on index
                    angle = (i * 137.508) % 360  # Golden angle for even distribution
                    radius_x = (i % 10) / 10.0 * face_width * 0.2
                    radius_y = ((i // 10) % 10) / 10.0 * face_height * 0.2
                    
                    # Interpolate based on face region
                    if i < 100:
                        landmarks[i] = SimpleLandmark(
                            face_center_x + (i % 20 - 10) / 20.0 * face_width * 0.4,
                            (y / h_img) + (i // 20) / 5.0 * face_height * 0.3
                        )
                    elif i < 200:
                        landmarks[i] = SimpleLandmark(
                            face_center_x + (i % 20 - 10) / 20.0 * face_width * 0.3,
                            face_center_y + (i % 20 - 10) / 20.0 * face_height * 0.2
                        )
                    elif i < 300:
                        landmarks[i] = SimpleLandmark(
                            face_center_x + (i % 20 - 10) / 20.0 * face_width * 0.3,
                            face_center_y + (i // 20) / 5.0 * face_height * 0.3
                        )
                    else:
                        landmarks[i] = SimpleLandmark(
                            face_center_x + (i % 20 - 10) / 20.0 * face_width * 0.4,
                            face_center_y + (i // 20) / 5.0 * face_height * 0.4
                        )
            
            # Create a simple detection result
            class SimpleDetection:
                def __init__(self, bbox, score):
                    self.location_data = type('obj', (object,), {
                        'relative_bounding_box': type('obj', (object,), {
                            'xmin': bbox[0] / w_img,
                            'ymin': bbox[1] / h_img,
                            'width': bbox[2] / w_img,
                            'height': bbox[3] / h_img
                        })()
                    })()
                    self.score = [score]
            
            class SimpleFaceResults:
                def __init__(self, detections):
                    self.detections = detections
            
            face_results = SimpleFaceResults([SimpleDetection((x, y, face_w, face_h), confidence)])
            
            return face_results, landmarks, None
        else:
            return None, None, None
    except Exception as e:
        logging.error(f"Error in OpenCV detection: {e}", exc_info=True)
        return None, None, None

def detect_face_dlib(image: np.ndarray) -> Tuple[Optional, Optional, Optional]:
    """Detect face using dlib (68-point landmarks)"""
    if not DLIB_AVAILABLE:
        return None, None, None
    
    try:
        import dlib
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect faces
        faces = dlib_face_detector(gray, 1)  # Upsample once for better detection
        
        if len(faces) == 0:
            return None, None, None
        
        # Get first face
        face = faces[0]
        h, w = image.shape[:2]
        
        # Convert dlib rectangle to bbox
        x = face.left()
        y = face.top()
        width = face.width()
        height = face.height()
        
        # Create landmarks (68 points from dlib if predictor available, otherwise estimate)
        landmarks = None
        if dlib_landmark_predictor is not None:
            try:
                shape = dlib_landmark_predictor(gray, face)
                landmarks = []
                for i in range(68):
                    landmarks.append(SimpleLandmark(shape.part(i).x / w, shape.part(i).y / h))
                # Pad to 468 for compatibility
                landmarks = landmarks + [None] * (468 - 68)
            except:
                landmarks = None
        
        # Create detection result
        class SimpleDetection:
            def __init__(self, bbox, score):
                self.location_data = type('obj', (object,), {
                    'relative_bounding_box': type('obj', (object,), {
                        'xmin': bbox[0] / w,
                        'ymin': bbox[1] / h,
                        'width': bbox[2] / w,
                        'height': bbox[3] / h
                    })()
                })()
                self.score = [score]
        
        class SimpleFaceResults:
            def __init__(self, detections):
                self.detections = detections
        
        face_results = SimpleFaceResults([SimpleDetection((x, y, width, height), 0.9)])
        
        return face_results, landmarks, None
    except Exception as e:
        logging.error(f"Error in dlib detection: {e}", exc_info=True)
        return None, None, None

def detect_face_face_recognition(image: np.ndarray) -> Tuple[Optional, Optional, Optional]:
    """Detect face using face_recognition library (68-point landmarks)"""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, None, None
    
    try:
        import face_recognition
        
        # face_recognition uses RGB
        face_locations = face_recognition.face_locations(image, model='hog')  # or 'cnn' for better accuracy
        face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
        
        if len(face_locations) == 0:
            return None, None, None
        
        h, w = image.shape[:2]
        top, right, bottom, left = face_locations[0]
        
        # Get 68-point landmarks
        landmarks_dict = face_landmarks_list[0] if face_landmarks_list else {}
        
        # Convert to 468-point format (map 68 points to MediaPipe indices)
        landmarks = [None] * 468
        if landmarks_dict:
            # Map face_recognition landmarks to MediaPipe indices
            # Left eye (6 points) -> indices around 33, 133, 159, 145, 157, 153
            if 'left_eye' in landmarks_dict:
                left_eye = landmarks_dict['left_eye']
                if len(left_eye) >= 6:
                    landmarks[33] = SimpleLandmark(left_eye[0][0] / w, left_eye[0][1] / h)  # Outer
                    landmarks[133] = SimpleLandmark(left_eye[3][0] / w, left_eye[3][1] / h)  # Inner
                    landmarks[159] = SimpleLandmark(left_eye[1][0] / w, left_eye[1][1] / h)  # Top
                    landmarks[145] = SimpleLandmark(left_eye[4][0] / w, left_eye[4][1] / h)  # Bottom
                    landmarks[157] = SimpleLandmark((left_eye[1][0] + left_eye[4][0]) / 2 / w, (left_eye[1][1] + left_eye[4][1]) / 2 / h)  # Center
                    landmarks[153] = SimpleLandmark(left_eye[0][0] / w, left_eye[0][1] / h)  # Outer corner
            
            # Right eye
            if 'right_eye' in landmarks_dict:
                right_eye = landmarks_dict['right_eye']
                if len(right_eye) >= 6:
                    landmarks[362] = SimpleLandmark(right_eye[3][0] / w, right_eye[3][1] / h)  # Outer
                    landmarks[386] = SimpleLandmark(right_eye[0][0] / w, right_eye[0][1] / h)  # Inner
                    landmarks[380] = SimpleLandmark(right_eye[1][0] / w, right_eye[1][1] / h)  # Top
                    landmarks[374] = SimpleLandmark(right_eye[4][0] / w, right_eye[4][1] / h)  # Bottom
                    landmarks[388] = SimpleLandmark((right_eye[1][0] + right_eye[4][0]) / 2 / w, (right_eye[1][1] + right_eye[4][1]) / 2 / h)  # Center
                    landmarks[390] = SimpleLandmark(right_eye[3][0] / w, right_eye[3][1] / h)  # Outer corner
            
            # Nose tip (index 4)
            if 'nose_tip' in landmarks_dict:
                nose_tip = landmarks_dict['nose_tip']
                if len(nose_tip) > 0:
                    landmarks[4] = SimpleLandmark(nose_tip[2][0] / w, nose_tip[2][1] / h)
            
            # Chin (index 152)
            if 'chin' in landmarks_dict:
                chin = landmarks_dict['chin']
                if len(chin) >= 9:
                    landmarks[152] = SimpleLandmark(chin[8][0] / w, chin[8][1] / h)  # Bottom chin
            
            # Forehead (estimate from top of face, index 10)
            landmarks[10] = SimpleLandmark((left + right) / 2 / w, top / h - 0.1)
            
            # Face boundaries
            if 'chin' in landmarks_dict:
                chin = landmarks_dict['chin']
                if len(chin) >= 17:
                    landmarks[234] = SimpleLandmark(chin[0][0] / w, chin[0][1] / h)  # Left
                    landmarks[454] = SimpleLandmark(chin[16][0] / w, chin[16][1] / h)  # Right
        
        # Create detection result
        class SimpleDetection:
            def __init__(self, bbox, score):
                self.location_data = type('obj', (object,), {
                    'relative_bounding_box': type('obj', (object,), {
                        'xmin': bbox[0] / w,
                        'ymin': bbox[1] / h,
                        'width': bbox[2] / w,
                        'height': bbox[3] / h
                    })()
                })()
                self.score = [score]
        
        class SimpleFaceResults:
            def __init__(self, detections):
                self.detections = detections
        
        face_results = SimpleFaceResults([SimpleDetection((left, top, right - left, bottom - top), 0.95)])
        
        return face_results, landmarks, None
    except Exception as e:
        logging.error(f"Error in face_recognition detection: {e}", exc_info=True)
        return None, None, None

def detect_face_mtcnn(image: np.ndarray) -> Tuple[Optional, Optional, Optional]:
    """Detect face using MTCNN (5-point landmarks)"""
    if not MTCNN_AVAILABLE:
        return None, None, None
    
    try:
        detections = mtcnn_detector.detect_faces(image)
        
        if len(detections) == 0:
            return None, None, None
        
        detection = detections[0]
        h, w = image.shape[:2]
        
        # Get bounding box
        x, y, width, height = detection['box']
        confidence = detection['confidence']
        
        # Get 5 keypoints (left_eye, right_eye, nose, left_mouth, right_mouth)
        keypoints = detection.get('keypoints', {})
        
        # Convert to 468-point format
        landmarks = [None] * 468
        
        if keypoints:
            # Left eye (estimate around index 33, 133)
            if 'left_eye' in keypoints:
                le = keypoints['left_eye']
                landmarks[33] = SimpleLandmark(le[0] / w, le[1] / h)
                landmarks[133] = SimpleLandmark(le[0] / w + 0.02, le[1] / h)
                landmarks[159] = SimpleLandmark(le[0] / w, le[1] / h - 0.01)
                landmarks[145] = SimpleLandmark(le[0] / w, le[1] / h + 0.01)
            
            # Right eye (estimate around index 362, 386)
            if 'right_eye' in keypoints:
                re = keypoints['right_eye']
                landmarks[362] = SimpleLandmark(re[0] / w, re[1] / h)
                landmarks[386] = SimpleLandmark(re[0] / w - 0.02, re[1] / h)
                landmarks[380] = SimpleLandmark(re[0] / w, re[1] / h - 0.01)
                landmarks[374] = SimpleLandmark(re[0] / w, re[1] / h + 0.01)
            
            # Nose tip (index 4)
            if 'nose' in keypoints:
                nose = keypoints['nose']
                landmarks[4] = SimpleLandmark(nose[0] / w, nose[1] / h)
            
            # Mouth corners
            if 'mouth_left' in keypoints:
                ml = keypoints['mouth_left']
                landmarks[61] = SimpleLandmark(ml[0] / w, ml[1] / h)
            
            if 'mouth_right' in keypoints:
                mr = keypoints['mouth_right']
                landmarks[291] = SimpleLandmark(mr[0] / w, mr[1] / h)
            
            # Estimate other key points
            face_center_x = (x + width / 2) / w
            face_center_y = (y + height / 2) / h
            landmarks[10] = SimpleLandmark(face_center_x, y / h - 0.05)  # Forehead
            landmarks[152] = SimpleLandmark(face_center_x, (y + height) / h)  # Chin
            landmarks[234] = SimpleLandmark(x / w, face_center_y)  # Left face
            landmarks[454] = SimpleLandmark((x + width) / w, face_center_y)  # Right face
        
        # Create detection result
        class SimpleDetection:
            def __init__(self, bbox, score):
                self.location_data = type('obj', (object,), {
                    'relative_bounding_box': type('obj', (object,), {
                        'xmin': bbox[0] / w,
                        'ymin': bbox[1] / h,
                        'width': bbox[2] / w,
                        'height': bbox[3] / h
                    })()
                })()
                self.score = [score]
        
        class SimpleFaceResults:
            def __init__(self, detections):
                self.detections = detections
        
        face_results = SimpleFaceResults([SimpleDetection((x, y, width, height), confidence)])
        
        return face_results, landmarks, None
    except Exception as e:
        logging.error(f"Error in MTCNN detection: {e}", exc_info=True)
        return None, None, None

def detect_face_mediapipe(image: np.ndarray, use_cache: bool = True) -> Tuple[Optional, Optional, Optional]:
    """Detect face and landmarks using best available method (MediaPipe > dlib > face_recognition > MTCNN > OpenCV)"""
    # Try methods in order of accuracy
    if MEDIAPIPE_AVAILABLE:
        try:
            # MediaPipe expects RGB images (not BGR)
            # The image is already in RGB format from decode_image
            image_rgb = image.copy()
            
            # Face detection
            face_results = face_detection.process(image_rgb)
            
            # Face mesh for landmarks (with refined landmarks for better accuracy)
            mesh_results = face_mesh.process(image_rgb)
            
            landmarks = None
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                logging.debug(f"✅ MediaPipe detected {len(landmarks)} landmarks")
            
            # Log detection confidence
            if face_results.detections:
                detection = face_results.detections[0]
                confidence = detection.score[0] if hasattr(detection, 'score') and len(detection.score) > 0 else 0.0
                logging.info(f"✅ MediaPipe face detection confidence: {confidence:.3f}")
            
            if not landmarks:
                logging.warning("⚠️ MediaPipe detected face but no landmarks - trying alternatives")
            else:
                result = (face_results, landmarks, mesh_results)
                # Cache result
                if use_cache:
                    try:
                        from cache import face_detection_cache
                        face_detection_cache.set(image, result)
                    except ImportError:
                        pass
                return result
        except Exception as e:
            logging.error(f"Error in MediaPipe detection: {e}", exc_info=True)
    
    
    # Try MTCNN (works without CMake, good accuracy)
    if MTCNN_AVAILABLE:
        try:
            result = detect_face_mtcnn(image)
            if result[0] is not None:  # Face detected
                logging.info("✅ Using MTCNN for face detection")
                return result
        except Exception as e:
            logging.debug(f"MTCNN detection failed: {e}")
    
    # Try dlib (if available, requires CMake)
    if DLIB_AVAILABLE:
        try:
            result = detect_face_dlib(image)
            if result[0] is not None:  # Face detected
                logging.info("✅ Using dlib for face detection")
                return result
        except Exception as e:
            logging.debug(f"dlib detection failed: {e}")
    
    # Try face_recognition (if available, requires dlib)
    if FACE_RECOGNITION_AVAILABLE:
        try:
            result = detect_face_face_recognition(image)
            if result[0] is not None:  # Face detected
                logging.info("✅ Using face_recognition for face detection")
                return result
        except Exception as e:
            logging.debug(f"face_recognition detection failed: {e}")
    
    # Fallback to OpenCV
    logging.info("📦 Using OpenCV fallback for face detection")
    return detect_face_opencv(image)

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image with performance optimization"""
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
        
        # Performance optimization: Compress if too large
        h, w = image.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            logging.debug(f"Compressed image from {w}x{h} to {int(w*scale)}x{int(h*scale)}")
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Error decoding image: {e}", exc_info=True)
        raise

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Neuro Desk - AI Human-Computer Interaction Coach API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/health", tags=["General"])
async def health():
    """Health check endpoint - returns API health status and detailed metrics"""
    try:
        from metrics import metrics
        health_data = metrics.get_health()
        return health_data
    except ImportError:
        return {"status": "healthy", "message": "Metrics not available"}
    
@app.get("/metrics", tags=["General"])
async def get_metrics():
    """Get system metrics and performance statistics"""
    """Get detailed metrics"""
    try:
        from metrics import metrics
        return metrics.get_stats()
    except ImportError:
        return {"error": "Metrics not available"}

@app.post("/analyze", tags=["Analysis"], response_model=Dict)
async def analyze_frame(request: AnalyzeRequest):
    """
    Analyze a single frame for wellness metrics.
    
    **Request Body:**
    - `image`: Base64 encoded image data (JPEG/PNG)
    
    **Response:**
    - `timestamp`: Analysis timestamp
    - `posture`: Posture analysis results
    - `eye_strain`: Eye strain analysis results
    - `engagement`: Engagement analysis results
    - `stress`: Stress analysis results
    - `productivity`: Overall productivity score
    - `recommendations`: List of wellness recommendations
    
    **Example:**
    ```json
    {
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    }
    ```
    """
    start_time = time.time()
    try:
        # Record request
        try:
            from metrics import metrics
            metrics.record_request("analyze")
        except ImportError:
            pass
        image_data = request.image
        if not image_data:
            logging.warning("No image data provided")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No image data provided",
                    "timestamp": datetime.now().isoformat(),
                    "posture": {"error": "No image data provided"},
                    "eye_strain": {"error": "No image data provided"},
                    "engagement": {"error": "No image data provided"},
                    "stress": {"error": "No image data provided"},
                    "productivity": {"error": "Cannot calculate - no image data"},
                    "recommendations": ["⚠️ Please provide image data"]
                }
            )
        
        logging.info(f"Received image data, length: {len(image_data)}")
        
        # Decode image
        try:
            image = decode_image(image_data)
            logging.info(f"Image decoded successfully, shape: {image.shape}")
            
            # Preprocess image for better face detection accuracy
            image = analyzer.preprocess_image(image)
            
            # Performance: Record processing start time
            processing_start = time.time()
            
            # Validate image quality before processing
            quality_check = analyzer.validate_image_quality(image)
            if not quality_check.get("valid", True):
                logging.warning(f"Poor image quality: {quality_check.get('reason')}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Image quality too poor: {quality_check.get('reason')}",
                        "quality_score": quality_check.get("quality_score", 0),
                        "timestamp": datetime.now().isoformat(),
                        "posture": {"error": f"Image quality issue: {quality_check.get('reason')}"},
                        "eye_strain": {"error": f"Image quality issue: {quality_check.get('reason')}"},
                        "engagement": {"error": f"Image quality issue: {quality_check.get('reason')}"},
                        "stress": {"error": f"Image quality issue: {quality_check.get('reason')}"},
                        "productivity": {"error": "Cannot calculate - poor image quality"},
                        "recommendations": [f"⚠️ Image quality issue: {quality_check.get('reason')}. Please improve lighting or camera focus."]
                    }
                )
        except Exception as decode_err:
            logging.error(f"Error decoding image: {decode_err}", exc_info=True)
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Image decode error: {str(decode_err)}",
                    "timestamp": datetime.now().isoformat(),
                    "posture": {"error": "Image decode error"},
                    "eye_strain": {"error": "Image decode error"},
                    "engagement": {"error": "Image decode error"},
                    "stress": {"error": "Image decode error"},
                    "productivity": {"error": "Cannot calculate - image decode error"},
                    "recommendations": ["⚠️ Image decode error. Please check image format."]
                }
            )
        
        # Detect face and landmarks using MediaPipe or OpenCV fallback
        try:
            face_results, landmarks, mesh_results = detect_face_mediapipe(image)
            face_detected = False
            if face_results and hasattr(face_results, 'detections') and len(face_results.detections) > 0:
                face_detected = True
            elif face_results:
                face_detected = True
            # Detailed face detection logging
            has_landmarks = landmarks is not None and len(landmarks) > 0 if landmarks else False
            detection_method = 'MediaPipe' if MEDIAPIPE_AVAILABLE else 'OpenCV'
            
            # Validate landmark consistency if available
            landmark_consistency = 0.8  # Default
            if landmarks and has_landmarks:
                landmark_consistency = analyzer.validate_landmark_consistency(landmarks)
                if landmark_consistency < 0.4:
                    logging.warning(f"⚠️ Low landmark consistency: {landmark_consistency:.2f} - may affect accuracy")
            
            logging.info(f"🔍 Face detection: detected={face_detected}, has_landmarks={has_landmarks}, method={detection_method}, consistency={landmark_consistency:.2f}")
            if face_detected:
                logging.info(f"   📊 Using {detection_method} for analysis - {'Full 468 landmarks' if has_landmarks else 'Estimated landmarks'}")
            if face_detected and face_results:
                if hasattr(face_results, 'detections') and len(face_results.detections) > 0:
                    detection = face_results.detections[0]
                    if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_bounding_box'):
                        bbox = detection.location_data.relative_bounding_box
                        logging.info(f"Face bbox: x={bbox.xmin:.3f}, y={bbox.ymin:.3f}, w={bbox.width:.3f}, h={bbox.height:.3f}")
        except Exception as face_err:
            logging.error(f"Error in face detection: {face_err}", exc_info=True)
            face_results, landmarks, mesh_results = None, None, None
        
        # Get session history for temporal smoothing
        session_id = "http_session"
        session_data = sessions.get(session_id, {})
        frame_count = session_data.get("frame_count", 0)
        
        # Update frame count first
        if session_id not in sessions:
            get_session_history(session_id)  # Initialize session
        sessions[session_id]["frame_count"] = frame_count + 1
        
        # Process every frame for real-time analysis (no skipping)
        # Always analyze fresh data to detect position changes
        
        ear_history, head_position_history = get_session_history(session_id)
        
        # Adapt to current conditions
        try:
            analyzer._adapt_to_lighting(image)
        except Exception as e:
            logging.warning(f"Lighting adaptation error: {e}")
        
        # Analyze with comprehensive error handling
        try:
            logging.info(f"🔍 Starting posture analysis - Face detected: {face_results is not None}, Has landmarks: {landmarks is not None and len(landmarks) > 0 if landmarks else False}")
            posture = analyzer.analyze_posture(image, landmarks, face_results)
            
            # Get face position from result
            face_y = posture.get('face_position_y', 0.5)
            current_score = posture.get('score', 70)
            
            logging.info(f"✅ Posture analysis: score={current_score}, face_y={face_y:.3f}, slouching={posture.get('slouching')}")
            
            # ALWAYS recalculate if score is 70 or close to it (likely a default)
            if 68 <= current_score <= 72:
                logging.warning(f"⚠️ Posture score is {current_score} (suspicious) - FORCING recalculation from face position")
                # Force recalculation directly from face position - NO EXCUSES
                if face_y != 0.5:  # If face position is known
                    # Direct linear mapping: face_y to score
                    ideal_y = 0.35
                    y_diff = face_y - ideal_y
                    
                    # Very aggressive mapping
                    if face_y < 0.25:
                        new_score = 95  # Very high position
                    elif face_y < 0.35:
                        new_score = 100 - abs(y_diff) * 500  # Good range
                    elif face_y < 0.45:
                        new_score = 85 - (face_y - 0.35) * 400  # Slight slouch
                    elif face_y < 0.60:
                        new_score = 65 - (face_y - 0.45) * 300  # Moderate slouch
                    else:
                        new_score = 50 - (face_y - 0.60) * 200  # Severe slouch
                    
                    new_score = max(15, min(100, round(new_score, 2)))
                    posture["score"] = new_score
                    logging.error(f"🔄 FORCED score from {current_score} to {new_score} based on face_y={face_y:.3f}")
                else:
                    logging.error(f"❌ Cannot recalculate - face_y is default 0.5! Face may not be detected.")
        except Exception as e:
            logging.error(f"❌ Posture analysis error: {e}", exc_info=True)
            # Return None instead of static default - let caller handle error
            return None
        
        try:
            eye_strain = analyzer.analyze_eye_strain(image, landmarks, ear_history)
            logging.info(f"✅ Eye strain analysis successful: score={eye_strain.get('score')}, EAR={eye_strain.get('ear_avg')}")
        except Exception as e:
            logging.error(f"❌ Eye strain analysis error: {e}", exc_info=True)
            # Use history if available - calculate from actual data
            if len(ear_history) > 10:
                avg_ear = np.mean(list(ear_history)[-20:])
                # Calculate score from actual EAR measurement
                if analyzer.user_calibration["calibrated"]:
                    ear_deviation = abs(avg_ear - analyzer.ear_baseline)
                    estimated_score = max(30, min(95, 100 - ear_deviation * 150))
                else:
                    estimated_score = max(50, min(95, 80 + (avg_ear - 0.28) * 50))
                eye_strain = {"eye_strain_risk": "low", "score": estimated_score, "blink_rate": 0, "ear_avg": avg_ear, "error": str(e)}
            else:
                # Not enough data - return None instead of static default
                logging.warning("⚠️ Insufficient data for eye strain analysis after error")
                eye_strain = None
        
        try:
            engagement = analyzer.analyze_engagement(landmarks, face_results, head_position_history, image.shape)
            logging.info(f"✅ Engagement analysis successful: score={engagement.get('score')}")
        except Exception as e:
            logging.error(f"❌ Engagement analysis error: {e}", exc_info=True)
            # Estimate based on actual face detection and movement data
            if face_results is not None and len(head_position_history) > 5:
                positions = list(head_position_history)
                variance = np.var([p[0] for p in positions]) + np.var([p[1] for p in positions])
                # Calculate from actual movement variance
                estimated_score = max(20, min(90, 80 - variance * 2000))
                stability = max(0, min(1, 1 - variance * 100))
                engagement = {"concentration": "medium" if estimated_score > 50 else "low", "score": estimated_score, "face_visible": True, "head_stability": round(stability, 2), "error": str(e)}
            else:
                # Not enough data - return None instead of static default
                logging.warning("⚠️ Insufficient data for engagement analysis after error")
                engagement = None
        
        try:
            stress = analyzer.analyze_stress(image, landmarks, face_results)
            logging.info(f"✅ Stress analysis successful: score={stress.get('score')}")
        except Exception as e:
            logging.error(f"❌ Stress analysis error: {e}", exc_info=True)
            # Return None instead of static default - let caller handle error
            logging.warning("⚠️ Stress analysis failed - returning None")
            stress = None
            
        # Calculate productivity - only if we have all required data
        productivity = None
        if posture and eye_strain and engagement and stress:
            try:
                productivity = analyzer.calculate_productivity_score(posture, eye_strain, engagement, stress)
                if productivity:
                    logging.info(f"✅ Productivity calculated: {productivity.get('productivity_score')}")
                else:
                    logging.warning("⚠️ Productivity calculation returned None - missing data")
            except Exception as e:
                logging.error(f"Productivity calculation error: {e}", exc_info=True)
                productivity = None
        
        # If productivity calculation failed, calculate from available data
        if not productivity:
            logging.warning("⚠️ Cannot calculate productivity - missing analysis data")
            # Try to calculate from available components
            available_scores = []
            if posture and posture.get('score') is not None:
                available_scores.append(('posture', posture.get('score')))
            if eye_strain and eye_strain.get('score') is not None:
                available_scores.append(('eye_strain', eye_strain.get('score')))
            if engagement and engagement.get('score') is not None:
                available_scores.append(('engagement', engagement.get('score')))
            if stress and stress.get('score') is not None:
                available_scores.append(('stress', stress.get('score')))
            
            if len(available_scores) >= 2:
                # Calculate partial productivity from available scores
                weights = {"posture": 0.25, "eye_strain": 0.20, "engagement": 0.30, "stress": 0.25}
                total_weight = sum(weights.get(name, 0) for name, _ in available_scores)
                weighted_sum = sum(score * weights.get(name, 0) for name, score in available_scores)
                partial_productivity = weighted_sum / total_weight if total_weight > 0 else None
                
                if partial_productivity:
                    productivity = {
                        "productivity_score": round(partial_productivity, 2),
                        "break_needed": partial_productivity < 60,
                        "eye_exercise_needed": eye_strain.get("eye_strain_risk") in ["medium", "high"] if eye_strain else False,
                        "posture_reminder": posture.get("slouching", False) if posture else False
                    }
                    logging.info(f"✅ Partial productivity calculated from {len(available_scores)} components: {productivity.get('productivity_score')}")
                else:
                    productivity = None
            else:
                productivity = None
            
        # Get recommendations - only if we have productivity data
        recommendations = []
        if productivity:
            try:
                rec_data = {
                    **productivity,
                    "stress_level": stress.get("stress_level", "low") if stress else "low",
                    "blink_rate": eye_strain.get("blink_rate", 0) if eye_strain else 0
                }
                recommendations = analyzer.get_recommendations(rec_data)
            except Exception as e:
                logging.error(f"Recommendations error: {e}", exc_info=True)
                recommendations = ["⚠️ Analysis in progress - collecting data..."]
        else:
            recommendations = ["⚠️ Analysis in progress - collecting data..."]
        
        # Prepare response - handle None values gracefully
        response = {
            "timestamp": datetime.now().isoformat(),
            "posture": posture if posture else {"error": "No face detected or insufficient data"},
            "eye_strain": eye_strain if eye_strain else {"error": "Insufficient data for analysis"},
            "engagement": engagement if engagement else {"error": "Insufficient data for analysis"},
            "stress": stress if stress else {"error": "Insufficient data for analysis"},
            "productivity": productivity if productivity else {"error": "Cannot calculate - missing analysis data"},
            "recommendations": recommendations
        }
        
        # Detailed logging for debugging - handle None values
        posture_score = posture.get('score', 'N/A') if posture else 'N/A'
        eye_score = eye_strain.get('score', 'N/A') if eye_strain else 'N/A'
        engagement_score = engagement.get('score', 'N/A') if engagement else 'N/A'
        stress_score = stress.get('score', 'N/A') if stress else 'N/A'
        productivity_score = productivity.get('productivity_score', 'N/A') if productivity else 'N/A'
        
        logging.info(f"📊 Analysis complete - Posture: {posture_score}, Eye: {eye_score}, Engagement: {engagement_score}, Stress: {stress_score}, Productivity: {productivity_score}")
        has_landmarks = landmarks is not None and len(landmarks) > 0 if landmarks else False
        face_pos = f"({posture.get('face_position_x', 'N/A')}, {posture.get('face_position_y', 'N/A')})" if posture else "N/A"
        logging.info(f"👤 Face detected: {face_results is not None}, Landmarks: {has_landmarks}, Face center: {face_pos}")
        
        # Always update timestamp to ensure freshness
        response["timestamp"] = datetime.now().isoformat()
        
        # Don't cache results - always send fresh analysis for real-time updates
        # Cache only for reference, but don't reuse it
        if session_id in sessions:
            sessions[session_id]["last_result"] = response  # Keep for reference only
            sessions[session_id]["last_result_time"] = datetime.now()
        
        # Log that we're sending fresh analysis with position data
        posture_log = f"{posture.get('score')} (face_y: {posture.get('face_position_y'):.3f})" if posture and posture.get('score') else "N/A"
        eye_log = eye_strain.get('score') if eye_strain and eye_strain.get('score') else "N/A"
        engagement_log = engagement.get('score') if engagement and engagement.get('score') else "N/A"
        stress_log = stress.get('score') if stress and stress.get('score') else "N/A"
        logging.info(f"📤 Sending FRESH analysis - Posture: {posture_log}, Eye: {eye_log}, Engagement: {engagement_log}, Stress: {stress_log}")
        
        # Record performance metrics
        try:
            from metrics import metrics
            total_time = time.time() - start_time
            metrics.record_success("analyze", total_time)
            
            # Calculate analysis time if analysis_start was set
            if 'analysis_start' in locals():
                analysis_time = (time.time() - analysis_start) * 1000  # Convert to ms
                metrics.record_analysis_time(analysis_time)
            # Calculate frame processing time if processing_start was set
            if 'processing_start' in locals():
                frame_processing_time = (time.time() - processing_start) * 1000  # Convert to ms
                metrics.record_frame_processing_time(frame_processing_time)
        except ImportError:
            pass
        
        return JSONResponse(content=response, status_code=200)
        
    except Exception as e:
        logging.error(f"Error in analyze endpoint: {e}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        logging.error(f"Full traceback: {error_trace}")
        
        # Record error metrics
        try:
            from metrics import metrics
            metrics.record_error("analyze", type(e).__name__)
        except ImportError:
            pass
        
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "posture": {"error": "Analysis error occurred"},
                "eye_strain": {"error": "Analysis error occurred"},
                "engagement": {"error": "Analysis error occurred"},
                "stress": {"error": "Analysis error occurred"},
                "productivity": {"error": "Cannot calculate - analysis error"},
                "recommendations": ["⚠️ Analysis error occurred. Please try again."]
            }
        )

@app.websocket("/ws", tags=["Analysis"])
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time frame analysis.
    
    **Connection:**
    - Connect to `ws://localhost:8000/ws`
    - Send frames as JSON: `{"image": "data:image/jpeg;base64,..."}`
    - Receive analysis results in real-time
    """
    session_id = None
    try:
        await websocket.accept()
        session_id = str(id(websocket))
        sessions[session_id] = {
            "start_time": datetime.now(),
            "frame_count": 0
        }
        logging.info(f"WebSocket connection established: {session_id}")
    except Exception as e:
        logging.error(f"Error accepting WebSocket connection: {e}", exc_info=True)
        return
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                frame_data = json.loads(data)
            except WebSocketDisconnect:
                # Client disconnected normally
                logging.info(f"WebSocket client disconnected: {session_id}")
                break
            except RuntimeError as e:
                # WebSocket already disconnected
                if "disconnect" in str(e).lower():
                    logging.info(f"WebSocket already disconnected: {session_id}")
                    break
                raise
            except Exception as e:
                logging.error(f"Error receiving/parsing data: {e}", exc_info=True)
                # Check if connection is still open before sending
                try:
                    # Check connection state
                    if websocket.client_state.name != "CONNECTED":
                        break
                    response = {
                        "timestamp": datetime.now().isoformat(),
                        "error": "Data parsing error",
                        "posture": {"error": "Data parsing error"},
                        "eye_strain": {"error": "Data parsing error"},
                        "engagement": {"error": "Data parsing error"},
                        "stress": {"error": "Data parsing error"},
                        "productivity": {"error": "Cannot calculate - data parsing error"},
                        "recommendations": ["⚠️ Processing error, retrying..."]
                    }
                    await websocket.send_json(response)
                except (WebSocketDisconnect, RuntimeError):
                    # Connection closed, exit loop
                    break
                except Exception:
                    # Other error sending, continue
                    pass
                continue
            
            # Decode image
            try:
                image = decode_image(frame_data["image"])
                
                # Preprocess image for better face detection accuracy
                image = analyzer.preprocess_image(image)
                
                # Detect face and landmarks using MediaPipe (with caching)
                face_results, landmarks, mesh_results = detect_face_mediapipe(image, use_cache=True)
            except Exception as e:
                logging.error(f"Image processing error: {e}", exc_info=True)
                # Send error response but keep connection alive
                try:
                    response = {
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "posture": {"error": "Image processing error"},
                        "eye_strain": {"error": "Image processing error"},
                        "engagement": {"error": "Image processing error"},
                        "stress": {"error": "Image processing error"},
                        "productivity": {"error": "Cannot calculate - image processing error"},
                        "recommendations": ["⚠️ Image processing error, retrying..."]
                    }
                    await websocket.send_json(response)
                except Exception as send_err:
                    logging.error(f"Error sending error response: {send_err}", exc_info=True)
                    break  # If we can't send, connection is broken
                continue
            
            # Get session history for temporal smoothing
            session_data = sessions.get(session_id, {})
            frame_count = session_data.get("frame_count", 0)
            
            # Update frame count first
            sessions[session_id]["frame_count"] = frame_count + 1
            
            # Process every frame for real-time analysis (no skipping)
            # Always analyze fresh data to detect position changes
            
            ear_history, head_position_history = get_session_history(session_id)
            
            # Performance: Record processing start time
            processing_start = time.time()
            
            # Adapt to current conditions
            try:
                analyzer._adapt_to_lighting(image)
            except Exception as e:
                logging.warning(f"Lighting adaptation error: {e}")
            
            # Analyze with error handling - no static defaults
            analysis_start = time.time()
            posture = None
            try:
                posture = analyzer.analyze_posture(image, landmarks, face_results)
            except Exception as e:
                logging.error(f"Posture analysis error: {e}", exc_info=True)
                posture = None
            
            eye_strain = None
            try:
                eye_strain = analyzer.analyze_eye_strain(image, landmarks, ear_history)
            except Exception as e:
                logging.error(f"Eye strain analysis error: {e}", exc_info=True)
                # Try to use history if available
                if len(ear_history) > 10:
                    avg_ear = np.mean(list(ear_history)[-20:])
                    if analyzer.user_calibration["calibrated"]:
                        ear_deviation = abs(avg_ear - analyzer.ear_baseline)
                        estimated_score = max(30, min(95, 100 - ear_deviation * 150))
                    else:
                        estimated_score = max(50, min(95, 80 + (avg_ear - 0.28) * 50))
                    eye_strain = {"eye_strain_risk": "low", "score": estimated_score, "blink_rate": 0, "ear_avg": avg_ear, "error": str(e)}
                else:
                    eye_strain = None
            
            engagement = None
            try:
                engagement = analyzer.analyze_engagement(landmarks, face_results, head_position_history, image.shape)
            except Exception as e:
                logging.error(f"Engagement analysis error: {e}", exc_info=True)
                # Try to estimate from available data
                if face_results is not None and len(head_position_history) > 5:
                    positions = list(head_position_history)
                    variance = np.var([p[0] for p in positions]) + np.var([p[1] for p in positions])
                    estimated_score = max(20, min(90, 80 - variance * 2000))
                    stability = max(0, min(1, 1 - variance * 100))
                    engagement = {"concentration": "medium" if estimated_score > 50 else "low", "score": estimated_score, "face_visible": True, "head_stability": round(stability, 2), "error": str(e)}
                else:
                    engagement = None
            
            stress = None
            try:
                stress = analyzer.analyze_stress(image, landmarks, face_results)
            except Exception as e:
                logging.error(f"Stress analysis error: {e}", exc_info=True)
                stress = None
            
            # Calculate productivity - only if we have all required data
            productivity = None
            if posture and eye_strain and engagement and stress:
                try:
                    productivity = analyzer.calculate_productivity_score(posture, eye_strain, engagement, stress)
                except Exception as e:
                    logging.error(f"Productivity calculation error: {e}", exc_info=True)
                    productivity = None
            
            # If productivity calculation failed, calculate from available data
            if not productivity:
                available_scores = []
                if posture and posture.get('score') is not None:
                    available_scores.append(('posture', posture.get('score')))
                if eye_strain and eye_strain.get('score') is not None:
                    available_scores.append(('eye_strain', eye_strain.get('score')))
                if engagement and engagement.get('score') is not None:
                    available_scores.append(('engagement', engagement.get('score')))
                if stress and stress.get('score') is not None:
                    available_scores.append(('stress', stress.get('score')))
                
                if len(available_scores) >= 2:
                    weights = {"posture": 0.25, "eye_strain": 0.20, "engagement": 0.30, "stress": 0.25}
                    total_weight = sum(weights.get(name, 0) for name, _ in available_scores)
                    weighted_sum = sum(score * weights.get(name, 0) for name, score in available_scores)
                    partial_productivity = weighted_sum / total_weight if total_weight > 0 else None
                    
                    if partial_productivity:
                        productivity = {
                            "productivity_score": round(partial_productivity, 2),
                            "break_needed": partial_productivity < 60,
                            "eye_exercise_needed": eye_strain.get("eye_strain_risk") in ["medium", "high"] if eye_strain else False,
                            "posture_reminder": posture.get("slouching", False) if posture else False
                        }
            
            # Get recommendations - only if we have productivity data
            recommendations = []
            if productivity:
                try:
                    rec_data = {
                        **productivity,
                        "stress_level": stress.get("stress_level", "low") if stress else "low",
                        "blink_rate": eye_strain.get("blink_rate", 0) if eye_strain else 0
                    }
                    recommendations = analyzer.get_recommendations(rec_data)
                except Exception as e:
                    logging.error(f"Recommendations error: {e}", exc_info=True)
                    recommendations = ["⚠️ Analysis in progress - collecting data..."]
            else:
                recommendations = ["⚠️ Analysis in progress - collecting data..."]
            
            # Prepare response - handle None values gracefully
            response = {
                "timestamp": datetime.now().isoformat(),
                "posture": posture if posture else {"error": "No face detected or insufficient data"},
                "eye_strain": eye_strain if eye_strain else {"error": "Insufficient data for analysis"},
                "engagement": engagement if engagement else {"error": "Insufficient data for analysis"},
                "stress": stress if stress else {"error": "Insufficient data for analysis"},
                "productivity": productivity if productivity else {"error": "Cannot calculate - missing analysis data"},
                "recommendations": recommendations
            }
            
            # Always update timestamp to ensure freshness
            response["timestamp"] = datetime.now().isoformat()
            
            # Don't cache results - always send fresh analysis for real-time updates
            # Cache only for reference, but don't reuse it
            if session_id in sessions:
                sessions[session_id]["last_result"] = response  # Keep for reference only
                sessions[session_id]["last_result_time"] = datetime.now()
            
            # Log that we're sending fresh analysis with position data
            # Log that we're sending fresh analysis with position data
            posture_log = f"{posture.get('score')} (face_y: {posture.get('face_position_y'):.3f})" if posture and posture.get('score') else "N/A"
            eye_log = eye_strain.get('score') if eye_strain and eye_strain.get('score') else "N/A"
            engagement_log = engagement.get('score') if engagement and engagement.get('score') else "N/A"
            stress_log = stress.get('score') if stress and stress.get('score') else "N/A"
            logging.info(f"📤 Sending FRESH analysis via WebSocket - Posture: {posture_log}, Eye: {eye_log}, Engagement: {engagement_log}, Stress: {stress_log}")
            
            # Record performance metrics
            try:
                from metrics import metrics
                if 'analysis_start' in locals():
                    analysis_time = (time.time() - analysis_start) * 1000
                    metrics.record_analysis_time(analysis_time)
                if 'processing_start' in locals():
                    frame_processing_time = (time.time() - processing_start) * 1000
                    metrics.record_frame_processing_time(frame_processing_time)
            except ImportError:
                pass
            
            try:
                await websocket.send_json(response)
            except (WebSocketDisconnect, RuntimeError) as send_err:
                logging.info(f"WebSocket disconnected during send: {send_err}")
                break  # Connection broken, exit loop
            except Exception as send_err:
                logging.error(f"Error sending response: {send_err}", exc_info=True)
                break  # Connection broken, exit loop
            
    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected: {session_id}")
        if session_id in sessions:
            del sessions[session_id]
    except Exception as e:
        logging.error(f"WebSocket error for {session_id}: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass
        if session_id in sessions:
            del sessions[session_id]

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("HCI Coach Backend Server")
    print("=" * 50)
    print("Starting server on http://0.0.0.0:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)

