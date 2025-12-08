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

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import MediaPipe, fallback to OpenCV if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    
    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # 0 for short-range, 1 for full-range
        min_detection_confidence=0.5
    )

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    logging.info("MediaPipe initialized successfully")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Using enhanced OpenCV-based detection.")
    # Fallback to improved OpenCV detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    except:
        face_cascade_profile = None

app = FastAPI(title="HCI Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Dict] = {}

# Frame skipping configuration
FRAME_SKIP = 3  # Process every Nth frame (3 = process every 3rd frame, 66% reduction)

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
        
        # Confidence thresholds
        self.min_face_confidence = 0.6  # Higher confidence required
        self.min_landmark_quality = 0.7  # Quality threshold for landmarks
        
    def validate_image_quality(self, image: np.ndarray) -> Dict:
        """Validate image quality before processing"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Brightness check
            brightness = np.mean(gray) / 255.0
            if brightness < 0.15:
                return {"valid": False, "reason": "too_dark", "brightness": brightness}
            if brightness > 0.95:
                return {"valid": False, "reason": "too_bright", "brightness": brightness}
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:
                return {"valid": False, "reason": "too_blurry", "sharpness": laplacian_var}
            
            # Contrast check
            contrast = np.std(gray) / 255.0
            if contrast < 0.1:
                return {"valid": False, "reason": "low_contrast", "contrast": contrast}
            
            return {
                "valid": True,
                "brightness": brightness,
                "sharpness": laplacian_var,
                "contrast": contrast,
                "quality_score": min(100, (brightness * 30 + min(1, laplacian_var/200) * 40 + contrast * 30))
            }
        except Exception as e:
            logging.warning(f"Error validating image quality: {e}")
            return {"valid": True, "quality_score": 70}  # Default to valid if check fails
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_points) -> float:
        """Calculate Eye Aspect Ratio (EAR) using improved 6-point method with validation"""
        if not landmarks or len(landmarks) < 468:
            return 0.0
        
        try:
            # Get eye landmark coordinates with validation
            eye_coords = []
            for idx in eye_points:
                if idx < len(landmarks) and landmarks[idx] is not None:
                    landmark = landmarks[idx]
                    # Validate landmark coordinates
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
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
            
            # Validate measurements
            if horizontal < 0.01:  # Too small, likely invalid
                return 0.0
            
            # Improved EAR: average of center and outer measurements for robustness
            ear_center = vertical_1 / horizontal
            ear_outer = vertical_2 / horizontal if vertical_2 > 0 else ear_center
            ear = (ear_center + ear_outer) / 2.0
            
            # Additional validation: check if measurements are reasonable
            if ear > 0.5 or ear < 0.05:  # Unrealistic values
                return 0.0
            
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
        """Improved posture analysis using head pose and position"""
        h, w = image.shape[:2]
        face_center_x = None
        face_center_y = None
        head_pose = {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False, "confidence": 0}
        face_detected = False
        
        # Try to get face position from detection result first
        if face_detection_result and hasattr(face_detection_result, 'detections') and len(face_detection_result.detections) > 0:
            detection = face_detection_result.detections[0]
            face_detected = True
            if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_bounding_box'):
                bbox = detection.location_data.relative_bounding_box
                face_center_y = bbox.ymin + bbox.height / 2
                face_center_x = bbox.xmin + bbox.width / 2
            elif hasattr(detection, 'bbox'):
                # Handle OpenCV-style bbox
                x, y, fw, fh = detection.bbox
                face_center_y = (y + fh / 2) / h
                face_center_x = (x + fw / 2) / w
        elif landmarks and len(landmarks) >= 468 and landmarks[4] is not None:
            # Use landmarks if available
            face_detected = True
            nose_tip = landmarks[4]
            face_center_y = nose_tip.y
            face_center_x = nose_tip.x
            head_pose = self.calculate_head_pose(landmarks, image.shape)
        elif landmarks and len(landmarks) > 0 and landmarks[0] is not None:
            # Use first available landmark as face center estimate
            face_detected = True
            face_center_y = landmarks[0].y
            face_center_x = landmarks[0].x
        
        # If no face detected at all, return low score
        if not face_detected or face_center_x is None or face_center_y is None:
            return {"slouching": True, "score": 35, "head_angle": 0, "face_position_y": 0.5, "face_position_x": 0.5, "reason": "no_face_detected"}
        
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
        
        # 2. Face vertical position (should be in upper portion for good posture)
        # Ideal position is around 0.3-0.4 (upper third to middle-upper)
        ideal_y_range = (0.28, 0.42)  # Slightly tighter range
        if face_center_y < ideal_y_range[0]:
            # Too high (unlikely but possible)
            vertical_penalty = (ideal_y_range[0] - face_center_y) * 80
            vertical_score = 100 - min(30, vertical_penalty)
        elif face_center_y > ideal_y_range[1]:
            # Too low (slouching)
            vertical_penalty = (face_center_y - ideal_y_range[1]) * 200  # More sensitive (was 150)
            vertical_score = 100 - min(85, vertical_penalty)
        else:
            # In ideal range - calculate score based on how centered
            center_distance = abs(face_center_y - 0.35)  # Distance from perfect center
            vertical_score = 100 - (center_distance * 50)  # Small penalty for not perfect
        
        # 3. Head tilt (sideways lean) - more sensitive
        if landmarks and len(landmarks) >= 468 and head_pose.get("confidence", 0) > 0.5:
            roll = abs(head_pose["roll"])
            roll_penalty = min(60, roll * 2.5)  # More sensitive (was 2)
        else:
            roll_penalty = 0  # Can't detect roll without landmarks
        
        # 4. Horizontal centering (face should be centered) - more sensitive
        horizontal_offset = abs(face_center_x - 0.5)
        horizontal_penalty = min(30, horizontal_offset * 60)  # More sensitive (was 50)
        
        # Calculate overall posture score with dynamic weighting
        base_score = (pitch_score * 0.35 + vertical_score * 0.45 + (100 - horizontal_penalty) * 0.2)
        posture_score = base_score - roll_penalty * 0.35  # More weight on roll (was 0.3)
        posture_score = max(15, min(100, posture_score))  # Wider range for visibility
        
        # Determine if slouching
        slouching = (
            (head_pose.get("pitch", 0) > 10) or  # Head tilted forward
            face_center_y > 0.6 or              # Face too low
            head_pose.get("tilted", False) or    # Significant tilt
            vertical_score < 50                  # Poor vertical position
        )
        
        return {
            "slouching": slouching,
            "score": round(posture_score, 2),
            "head_angle": round(head_pose.get("pitch", 0), 2),
            "face_position_y": round(face_center_y, 3),
            "face_position_x": round(face_center_x, 3)
        }
    
    def analyze_eye_strain(self, image, landmarks, session_history: deque) -> Dict:
        """Improved eye strain analysis using Eye Aspect Ratio (EAR) and blink rate"""
        # Initialize session history if needed
        if session_history is None:
            session_history = deque(maxlen=30)
        
        # If no landmarks, estimate based on face detection and time
        if landmarks is None or len(landmarks) < 468:
            # Estimate eye strain based on session duration and face detection
            if len(session_history) > 0:
                # If we have history, use average
                avg_ear = np.mean(list(session_history)) if session_history else 0.28
            else:
                # Estimate based on face detection if available
                avg_ear = 0.28  # Default
                session_history.append(avg_ear)
            
            # Estimate blink rate from history
            if len(session_history) > 5:
                recent_ears = list(session_history)[-10:]
                blink_threshold = 0.25
                blinks = sum(1 for ear in recent_ears if ear < blink_threshold)
                blink_rate = blinks / len(recent_ears) if recent_ears else 0.15
            else:
                blink_rate = 0.15  # Normal blink rate
            
            # Score based on estimated values with more variation
            base_score = 80
            if avg_ear < 0.24:
                base_score -= 18
            elif avg_ear > 0.32:
                base_score += 8
            
            if blink_rate < 0.05:
                eye_strain_risk = "medium"
                base_score -= 18
            elif blink_rate > 0.25:
                eye_strain_risk = "low-medium"
                base_score -= 10
            else:
                eye_strain_risk = "low"
            
            eye_score = max(45, min(95, base_score))
            
            return {
                "eye_strain_risk": eye_strain_risk,
                "score": round(eye_score, 2),
                "blink_rate": round(blink_rate, 3),
                "ear_avg": round(avg_ear, 3)
            }
        
        # Calculate EAR for both eyes using improved method
        left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_POINTS)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_POINTS)
        
        # Validate EAR values (normal range: 0.2-0.4)
        if left_ear == 0.0 and right_ear == 0.0:
            # No valid eye data
            return {"eye_strain_risk": "unknown", "score": 50, "blink_rate": 0, "ear_avg": 0.0}
        
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
            # Adaptive baseline: update baseline if we have enough history
            if len(session_history) > 10:
                recent_ears = list(session_history)[-10:]
                self.ear_baseline = np.mean(recent_ears)
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
        # Use adaptive threshold based on user's baseline
        adaptive_blink_threshold = max(0.15, self.ear_baseline - 2 * self.ear_std) if self.ear_std > 0 else 0.20
        blink_count = 0
        
        if len(session_history) > 15:
            recent_ears = list(session_history)[-20:]  # Use more frames for better detection
            prev_ear = recent_ears[0] if recent_ears else 0.3
            in_blink = False
            blink_duration = 0
            
            for i, ear in enumerate(recent_ears[1:], 1):
                # Detect blink start (EAR drops significantly)
                if ear < adaptive_blink_threshold and prev_ear >= adaptive_blink_threshold:
                    # Blink started
                    in_blink = True
                    blink_duration = 1
                elif ear < adaptive_blink_threshold and in_blink:
                    # Still in blink
                    blink_duration += 1
                elif ear >= adaptive_blink_threshold and prev_ear < adaptive_blink_threshold and in_blink:
                    # Blink completed (EAR returns above threshold)
                    # Validate blink: should be 1-5 frames (not too short, not too long)
                    if 1 <= blink_duration <= 5:
                        blink_count += 1
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
        
        # EAR-based analysis using adaptive thresholds
        ear_deviation = abs(avg_ear - self.ear_baseline) if self.ear_baseline > 0 else abs(avg_ear - 0.28)
        
        if avg_ear < 0.15:
            strain_factors.append("eyes_fully_closed")
            strain_score_deduction += 35
        elif avg_ear < 0.20:
            strain_factors.append("eyes_nearly_closed")
            strain_score_deduction += 25
        elif avg_ear < 0.25:
            strain_factors.append("eyes_droopy")
            strain_score_deduction += 18
        elif avg_ear > 0.38:
            strain_factors.append("eyes_wide_open")
            strain_score_deduction += 8  # Wide open can indicate strain
        elif ear_deviation > 2 * self.ear_std and self.ear_std > 0:
            # Significant deviation from baseline
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
        
        return {
            "eye_strain_risk": eye_strain_risk,
            "score": round(eye_score, 2),
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
        ) + size_bonus
        engagement_score = max(15, min(100, engagement_score))  # Wider range
        
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
                stress_score = 85
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
        
        return {
            "stress_level": stress_level,
            "score": round(stress_score, 2),
            "indicators": stress_indicators
        }
    
    def calculate_productivity_score(self, posture, eye_strain, engagement, stress) -> Dict:
        """Calculate overall productivity score"""
        weights = {
            "posture": 0.25,
            "eye_strain": 0.20,
            "engagement": 0.30,
            "stress": 0.25
        }
        
        productivity = (
            posture["score"] * weights["posture"] +
            eye_strain["score"] * weights["eye_strain"] +
            engagement["score"] * weights["engagement"] +
            stress["score"] * weights["stress"]
        )
        
        return {
            "productivity_score": round(productivity, 2),
            "break_needed": productivity < 60,
            "eye_exercise_needed": eye_strain["eye_strain_risk"] in ["medium", "high"],
            "posture_reminder": posture["slouching"]
        }
    
    def get_recommendations(self, analysis: Dict) -> list:
        """Generate wellness recommendations"""
        recommendations = []
        
        if analysis["posture_reminder"]:
            recommendations.append("💺 Sit up straight! Adjust your posture")
        
        if analysis["eye_exercise_needed"]:
            if analysis.get("blink_rate", 0) < 0.05:
                recommendations.append("👁️ Blink more often! Take a 20-20-20 break: Look 20ft away for 20 seconds")
            else:
                recommendations.append("👁️ Take a 20-20-20 break: Look 20ft away for 20 seconds")
        
        if analysis["break_needed"]:
            recommendations.append("☕ Take a 5-minute micro-break")
        
        stress_level = analysis.get("stress_level", "low")
        if stress_level in ["medium", "high", "low-medium"]:
            recommendations.append("🧘 Take 3 deep breaths to reduce stress")
        
        if not recommendations:
            recommendations.append("✅ You're doing great! Keep it up")
        
        return recommendations

analyzer = WellnessAnalyzer()

def detect_face_opencv(image: np.ndarray) -> Tuple[Optional, Optional, Optional]:
    """Detect face using OpenCV (fallback when MediaPipe unavailable)"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Try frontal face detection first
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If no frontal face, try profile
        if len(faces) == 0 and face_cascade_profile is not None:
            faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
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
            
            x, y, w, h = faces[0]
            face_det = SimpleFaceDetection(x, y, w, h)
            
            # Create a simple landmarks approximation based on face rectangle
            # This is a simplified version - in real MediaPipe we'd have 468 landmarks
            class SimpleLandmark:
                def __init__(self, x, y, z=0):
                    self.x = x
                    self.y = y
                    self.z = z
            
            h_img, w_img = image.shape[:2]
            landmarks = []
            # Create approximate landmarks based on face position
            # Key points: nose tip, eyes, mouth, chin, forehead
            face_center_x = (x + w/2) / w_img
            face_center_y = (y + h/2) / h_img
            face_width = w / w_img
            face_height = h / h_img
            
            # Create landmarks array with 468 points (MediaPipe standard)
            # Map key indices used in analysis functions
            landmarks = [None] * 468
            
            # Forehead center (index 10)
            landmarks[10] = SimpleLandmark(face_center_x, (y / h_img) - face_height * 0.15)
            
            # Nose tip (index 4)
            landmarks[4] = SimpleLandmark(face_center_x, face_center_y)
            
            # Chin (index 152)
            landmarks[152] = SimpleLandmark(face_center_x, (y + h) / h_img)
            
            # Left face boundary (index 234)
            landmarks[234] = SimpleLandmark((x / w_img), face_center_y)
            
            # Right face boundary (index 454)
            landmarks[454] = SimpleLandmark((x + w) / w_img, face_center_y)
            
            # Left eye landmarks (indices 33, 133, 157, 158, 159, 160, 161)
            left_eye_x = face_center_x - face_width * 0.15
            left_eye_y = face_center_y - face_height * 0.1
            landmarks[33] = SimpleLandmark(left_eye_x - face_width * 0.05, left_eye_y)
            landmarks[133] = SimpleLandmark(left_eye_x, left_eye_y - face_height * 0.02)
            landmarks[157] = SimpleLandmark(left_eye_x, left_eye_y)
            landmarks[158] = SimpleLandmark(left_eye_x, left_eye_y + face_height * 0.02)
            landmarks[159] = SimpleLandmark(left_eye_x + face_width * 0.05, left_eye_y)
            landmarks[160] = SimpleLandmark(left_eye_x - face_width * 0.03, left_eye_y)
            landmarks[161] = SimpleLandmark(left_eye_x + face_width * 0.03, left_eye_y)
            
            # Right eye landmarks (indices 362, 386, 387, 388, 390, 398)
            right_eye_x = face_center_x + face_width * 0.15
            right_eye_y = face_center_y - face_height * 0.1
            landmarks[362] = SimpleLandmark(right_eye_x - face_width * 0.05, right_eye_y)
            landmarks[386] = SimpleLandmark(right_eye_x, right_eye_y)
            landmarks[387] = SimpleLandmark(right_eye_x, right_eye_y + face_height * 0.02)
            landmarks[388] = SimpleLandmark(right_eye_x, right_eye_y - face_height * 0.02)
            landmarks[390] = SimpleLandmark(right_eye_x + face_width * 0.05, right_eye_y)
            landmarks[398] = SimpleLandmark(right_eye_x + face_width * 0.03, right_eye_y)
            
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
            landmarks[175] = SimpleLandmark(face_center_x, (y + h) / h_img)
            
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
            
            face_results = SimpleFaceResults([SimpleDetection((x, y, w, h), 0.8)])
            
            return face_results, landmarks, None
        else:
            return None, None, None
    except Exception as e:
        logging.error(f"Error in OpenCV detection: {e}", exc_info=True)
        return None, None, None

def detect_face_mediapipe(image: np.ndarray) -> Tuple[Optional, Optional, Optional]:
    """Detect face and landmarks using MediaPipe or OpenCV fallback"""
    if MEDIAPIPE_AVAILABLE:
        try:
            # Convert RGB to BGR for MediaPipe
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Face detection
            face_results = face_detection.process(image_bgr)
            
            # Face mesh for landmarks
            mesh_results = face_mesh.process(image_bgr)
            
            landmarks = None
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
            
            return face_results, landmarks, mesh_results
        except Exception as e:
            logging.error(f"Error in MediaPipe detection: {e}", exc_info=True)
            return detect_face_opencv(image)  # Fallback to OpenCV
    else:
        return detect_face_opencv(image)

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image"""
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
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Error decoding image: {e}", exc_info=True)
        raise

@app.get("/")
async def root():
    return {"message": "HCI Coach API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_frame(request: AnalyzeRequest):
    """HTTP endpoint for frame analysis (alternative to WebSocket)"""
    try:
        image_data = request.image
        if not image_data:
            logging.warning("No image data provided")
            return JSONResponse(
                status_code=200,
                content={
                    "error": "No image data provided",
                    "timestamp": datetime.now().isoformat(),
                    "posture": {"slouching": False, "score": 70},
                    "eye_strain": {"eye_strain_risk": "low", "score": 100},
                    "engagement": {"concentration": "low", "score": 50},
                    "stress": {"stress_level": "low", "score": 100},
                    "productivity": {"productivity_score": 70, "break_needed": False, "eye_exercise_needed": False, "posture_reminder": False},
                    "recommendations": ["⚠️ No image data"]
                }
            )
        
        logging.info(f"Received image data, length: {len(image_data)}")
        
        # Decode image
        try:
            image = decode_image(image_data)
            logging.info(f"Image decoded successfully, shape: {image.shape}")
            
            # Validate image quality before processing
            quality_check = analyzer.validate_image_quality(image)
            if not quality_check.get("valid", True):
                logging.warning(f"Poor image quality: {quality_check.get('reason')}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "error": f"Image quality too poor: {quality_check.get('reason')}",
                        "quality_score": quality_check.get("quality_score", 0),
                        "timestamp": datetime.now().isoformat(),
                        "posture": {"slouching": False, "score": 70},
                        "eye_strain": {"eye_strain_risk": "low", "score": 100},
                        "engagement": {"concentration": "low", "score": 50},
                        "stress": {"stress_level": "low", "score": 100},
                        "productivity": {"productivity_score": 70, "break_needed": False, "eye_exercise_needed": False, "posture_reminder": False},
                        "recommendations": [f"⚠️ Image quality issue: {quality_check.get('reason')}. Please improve lighting or camera focus."]
                    }
                )
        except Exception as decode_err:
            logging.error(f"Error decoding image: {decode_err}", exc_info=True)
            return JSONResponse(
                status_code=200,
                content={
                    "error": f"Image decode error: {str(decode_err)}",
                    "timestamp": datetime.now().isoformat(),
                    "posture": {"slouching": False, "score": 70},
                    "eye_strain": {"eye_strain_risk": "low", "score": 100},
                    "engagement": {"concentration": "low", "score": 50},
                    "stress": {"stress_level": "low", "score": 100},
                    "productivity": {"productivity_score": 70, "break_needed": False, "eye_exercise_needed": False, "posture_reminder": False},
                    "recommendations": ["⚠️ Image decode failed"]
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
            logging.info(f"Face detection: detected={face_detected}, has_landmarks={has_landmarks}, method={'MediaPipe' if MEDIAPIPE_AVAILABLE else 'OpenCV'}")
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
        
        # Frame skipping: Process every Nth frame, return cached result for others
        should_process = (frame_count % FRAME_SKIP == 0)
        
        # Check if we have a recent cached result (within last 1 second)
        if not should_process and session_data.get("last_result"):
            last_result_time = session_data.get("last_result_time")
            if last_result_time and (datetime.now() - last_result_time).total_seconds() < 1.0:
                logging.debug(f"Frame {frame_count}: Using cached result (frame skipping)")
                return JSONResponse(content=session_data["last_result"], status_code=200)
        
        # Update frame count
        if session_id not in sessions:
            get_session_history(session_id)  # Initialize session
        sessions[session_id]["frame_count"] = frame_count + 1
        
        ear_history, head_position_history = get_session_history(session_id)
        
        # Analyze
        try:
            posture = analyzer.analyze_posture(image, landmarks, face_results)
            eye_strain = analyzer.analyze_eye_strain(image, landmarks, ear_history)
            engagement = analyzer.analyze_engagement(landmarks, face_results, head_position_history, image.shape)
            stress = analyzer.analyze_stress(image, landmarks, face_results)
            
            # Calculate productivity
            productivity = analyzer.calculate_productivity_score(posture, eye_strain, engagement, stress)
            
            # Get recommendations
            recommendations = analyzer.get_recommendations({
                **productivity,
                "stress_level": stress["stress_level"],
                "blink_rate": eye_strain.get("blink_rate", 0)
            })
            
            # Prepare response
            response = {
                "timestamp": datetime.now().isoformat(),
                "posture": posture,
                "eye_strain": eye_strain,
                "engagement": engagement,
                "stress": stress,
                "productivity": productivity,
                "recommendations": recommendations
            }
            
            # Detailed logging for debugging constant scores
            logging.info(f"Analysis complete - Posture: {posture.get('score', 'N/A')} (face_y: {posture.get('face_position_y', 'N/A')}, head_angle: {posture.get('head_angle', 'N/A')}), "
                        f"Eye: {eye_strain.get('score', 'N/A')} (EAR: {eye_strain.get('ear_avg', 'N/A')}, blinks: {eye_strain.get('blink_rate', 'N/A')}), "
                        f"Engagement: {engagement.get('score', 'N/A')} (stability: {engagement.get('head_stability', 'N/A')}), "
                        f"Stress: {stress.get('score', 'N/A')} ({stress.get('stress_level', 'N/A')}), "
                        f"Productivity: {productivity.get('productivity_score', 'N/A')}")
            logging.info(f"Face detected: {face_results is not None}, Landmarks: {landmarks is not None and len(landmarks) > 0 if landmarks else False}, "
                        f"Face center: ({posture.get('face_position_x', 'N/A')}, {posture.get('face_position_y', 'N/A')})")
            
            # Cache result for frame skipping
            if session_id in sessions:
                sessions[session_id]["last_result"] = response
                sessions[session_id]["last_result_time"] = datetime.now()
            
            return JSONResponse(content=response, status_code=200)
        except Exception as analysis_err:
            logging.error(f"Error in analysis: {analysis_err}", exc_info=True)
            raise
        
    except Exception as e:
        logging.error(f"Error in analyze endpoint: {e}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        logging.error(f"Full traceback: {error_trace}")
        return JSONResponse(
            status_code=200,
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "posture": {"slouching": False, "score": 70},
                "eye_strain": {"eye_strain_risk": "low", "score": 100},
                "engagement": {"concentration": "low", "score": 50},
                "stress": {"stress_level": "low", "score": 100},
                "productivity": {"productivity_score": 70, "break_needed": False, "eye_exercise_needed": False, "posture_reminder": False},
                "recommendations": ["⚠️ Analysis error occurred"]
            }
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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
                        "posture": {"slouching": False, "score": 70},
                        "eye_strain": {"eye_strain_risk": "low", "score": 100},
                        "engagement": {"concentration": "low", "score": 50},
                        "stress": {"stress_level": "low", "score": 100},
                        "productivity": {"productivity_score": 70, "break_needed": False, "eye_exercise_needed": False, "posture_reminder": False},
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
                
                # Detect face and landmarks using MediaPipe
                face_results, landmarks, mesh_results = detect_face_mediapipe(image)
            except Exception as e:
                logging.error(f"Image processing error: {e}", exc_info=True)
                # Send error response but keep connection alive
                try:
                    response = {
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "posture": {"slouching": False, "score": 70},
                        "eye_strain": {"eye_strain_risk": "low", "score": 100},
                        "engagement": {"concentration": "low", "score": 50},
                        "stress": {"stress_level": "low", "score": 100},
                        "productivity": {"productivity_score": 70, "break_needed": False, "eye_exercise_needed": False, "posture_reminder": False},
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
            
            # Frame skipping: Process every Nth frame, return cached result for others
            should_process = (frame_count % FRAME_SKIP == 0)
            
            # Check if we have a recent cached result (within last 1 second)
            if not should_process and session_data.get("last_result"):
                last_result_time = session_data.get("last_result_time")
                if last_result_time and (datetime.now() - last_result_time).total_seconds() < 1.0:
                    logging.debug(f"Frame {frame_count}: Using cached result (frame skipping)")
                    try:
                        await websocket.send_json(session_data["last_result"])
                        sessions[session_id]["frame_count"] = frame_count + 1
                        continue
                    except:
                        pass  # If send fails, process normally
            
            # Update frame count
            sessions[session_id]["frame_count"] = frame_count + 1
            
            ear_history, head_position_history = get_session_history(session_id)
            
            # Analyze
            posture = analyzer.analyze_posture(image, landmarks, face_results)
            eye_strain = analyzer.analyze_eye_strain(image, landmarks, ear_history)
            engagement = analyzer.analyze_engagement(landmarks, face_results, head_position_history, image.shape)
            stress = analyzer.analyze_stress(image, landmarks, face_results)
            
            # Calculate productivity
            productivity = analyzer.calculate_productivity_score(posture, eye_strain, engagement, stress)
            
            # Get recommendations
            recommendations = analyzer.get_recommendations({
                **productivity,
                "stress_level": stress["stress_level"],
                "blink_rate": eye_strain.get("blink_rate", 0)
            })
            
            # Prepare response
            response = {
                "timestamp": datetime.now().isoformat(),
                "posture": posture,
                "eye_strain": eye_strain,
                "engagement": engagement,
                "stress": stress,
                "productivity": productivity,
                "recommendations": recommendations
            }
            
            # Cache result for frame skipping
            sessions[session_id]["last_result"] = response
            sessions[session_id]["last_result_time"] = datetime.now()
            
            try:
                await websocket.send_json(response)
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

