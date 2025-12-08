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

def get_session_history(session_id: str) -> Tuple[deque, deque]:
    """Get or create session history for temporal smoothing"""
    if session_id not in sessions:
        sessions[session_id] = {
            "start_time": datetime.now(),
            "frame_count": 0,
            "ear_history": deque(maxlen=30),
            "head_position_history": deque(maxlen=30)
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
        # Key points for eye aspect ratio (EAR) calculation
        self.LEFT_EYE_POINTS = [33, 160, 158, 153, 133, 157]  # Outer, inner corners and top/bottom
        self.RIGHT_EYE_POINTS = [362, 385, 387, 380, 374, 386]
        # Face boundary points for head pose estimation
        self.FACE_BOUNDARY = [10, 151, 9, 175, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
    def calculate_eye_aspect_ratio(self, landmarks, eye_points) -> float:
        """Calculate Eye Aspect Ratio (EAR) - lower values indicate closed eyes"""
        if not landmarks or len(landmarks) < 468:
            return 0.3  # Default value
        
        # Get eye landmark coordinates
        eye_coords = []
        for idx in eye_points:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                eye_coords.append([landmark.x, landmark.y])
        
        if len(eye_coords) < 6:
            return 0.3
        
        eye_coords = np.array(eye_coords)
        
        # Calculate distances
        # Vertical distances
        v1 = np.linalg.norm(eye_coords[1] - eye_coords[5])
        v2 = np.linalg.norm(eye_coords[2] - eye_coords[4])
        # Horizontal distance
        h = np.linalg.norm(eye_coords[0] - eye_coords[3])
        
        # EAR formula
        if h == 0:
            return 0.3
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def calculate_head_pose(self, landmarks, image_shape) -> Dict:
        """Estimate head pose (pitch, yaw, roll) from facial landmarks"""
        if not landmarks or len(landmarks) < 468:
            return {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False}
        
        h, w = image_shape[:2]
        
        # Get key facial points
        try:
            # Nose tip
            nose_tip = landmarks[4]
            # Chin
            chin = landmarks[152]
            # Forehead center
            forehead = landmarks[10]
            # Left and right face boundaries
            left_face = landmarks[234]
            right_face = landmarks[454]
            
            # Convert to pixel coordinates
            nose_pt = np.array([nose_tip.x * w, nose_tip.y * h])
            chin_pt = np.array([chin.x * w, chin.y * h])
            forehead_pt = np.array([forehead.x * w, forehead.y * h])
            left_pt = np.array([left_face.x * w, left_face.y * h])
            right_pt = np.array([right_face.x * w, right_face.y * h])
            
            # Calculate angles
            # Pitch (vertical tilt) - based on nose to chin vs forehead
            face_vertical = chin_pt - forehead_pt
            pitch = np.arctan2(face_vertical[1], abs(face_vertical[0])) * 180 / np.pi
            
            # Yaw (horizontal rotation) - based on face width
            face_width = np.linalg.norm(right_pt - left_pt)
            face_center_x = (left_pt[0] + right_pt[0]) / 2
            image_center_x = w / 2
            yaw_offset = (face_center_x - image_center_x) / w
            yaw = yaw_offset * 30  # Approximate degrees
            
            # Roll (head tilt) - based on eye/face alignment
            roll = np.arctan2(right_pt[1] - left_pt[1], right_pt[0] - left_pt[0]) * 180 / np.pi
            
            tilted = abs(pitch) > 15 or abs(yaw) > 20 or abs(roll) > 10
            
            return {
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "tilted": tilted
            }
        except Exception as e:
            logging.warning(f"Error calculating head pose: {e}")
            return {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False}
    
    def analyze_posture(self, image, landmarks, face_detection_result) -> Dict:
        """Improved posture analysis using head pose and position"""
        h, w = image.shape[:2]
        face_center_x = 0.5
        face_center_y = 0.5
        head_pose = {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False}
        
        # Try to get face position from detection result first
        if face_detection_result and hasattr(face_detection_result, 'detections') and len(face_detection_result.detections) > 0:
            detection = face_detection_result.detections[0]
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
            nose_tip = landmarks[4]
            face_center_y = nose_tip.y
            face_center_x = nose_tip.x
            head_pose = self.calculate_head_pose(landmarks, image.shape)
        elif landmarks and len(landmarks) > 0:
            # Use first available landmark as face center estimate
            if landmarks[0] is not None:
                face_center_y = landmarks[0].y
                face_center_x = landmarks[0].x
        else:
            # No face detected - return low score
            return {"slouching": True, "score": 40, "head_angle": 0, "reason": "no_face_detected"}
        
        # Analyze posture based on multiple factors
        # 1. Head pitch (forward lean = slouching) - only if we have pose data
        if landmarks and len(landmarks) >= 468:
            pitch_score = 100 - min(100, abs(head_pose["pitch"]) * 2)
        else:
            # Estimate pitch from face position
            ideal_y = 0.35  # Ideal face position (upper third)
            pitch_estimate = (face_center_y - ideal_y) * 50  # Convert to approximate pitch
            pitch_score = 100 - min(100, abs(pitch_estimate) * 1.5)
        
        # 2. Face vertical position (should be in upper portion for good posture)
        # Ideal position is around 0.3-0.4 (upper third to middle-upper)
        ideal_y_range = (0.25, 0.45)
        if face_center_y < ideal_y_range[0]:
            vertical_score = 90 - (ideal_y_range[0] - face_center_y) * 100
        elif face_center_y > ideal_y_range[1]:
            vertical_penalty = (face_center_y - ideal_y_range[1]) * 150
            vertical_score = 100 - min(80, vertical_penalty)
        else:
            vertical_score = 100  # In ideal range
        
        # 3. Head tilt (sideways lean)
        if landmarks and len(landmarks) >= 468:
            roll_penalty = min(50, abs(head_pose["roll"]) * 2)
        else:
            roll_penalty = 0  # Can't detect roll without landmarks
        
        # 4. Horizontal centering (face should be centered)
        horizontal_offset = abs(face_center_x - 0.5)
        horizontal_penalty = min(25, horizontal_offset * 50)
        
        # Calculate overall posture score with dynamic weighting
        base_score = (pitch_score * 0.4 + vertical_score * 0.4 + (100 - horizontal_penalty) * 0.2)
        posture_score = base_score - roll_penalty * 0.3
        posture_score = max(20, min(100, posture_score))  # Keep range 20-100 for visibility
        
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
            # Estimate eye strain based on session duration
            if len(session_history) > 0:
                # If we have history, use average
                avg_ear = np.mean(list(session_history)) if session_history else 0.28
            else:
                # Default moderate risk if no data
                avg_ear = 0.28
                session_history.append(avg_ear)
            
            # Estimate blink rate from history
            if len(session_history) > 5:
                recent_ears = list(session_history)[-10:]
                blink_threshold = 0.25
                blinks = sum(1 for ear in recent_ears if ear < blink_threshold)
                blink_rate = blinks / len(recent_ears) if recent_ears else 0.15
            else:
                blink_rate = 0.15  # Normal blink rate
            
            # Score based on estimated values
            if blink_rate < 0.05:
                eye_strain_risk = "medium"
                eye_score = 65
            else:
                eye_strain_risk = "low"
                eye_score = 85
            
            return {
                "eye_strain_risk": eye_strain_risk,
                "score": round(eye_score, 2),
                "blink_rate": round(blink_rate, 3),
                "ear_avg": round(avg_ear, 3)
            }
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_POINTS)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_POINTS)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Store in history for temporal analysis
        if session_history is None:
            session_history = deque(maxlen=30)
        
        session_history.append(avg_ear)
        
        # Calculate blink rate (low EAR indicates closed eyes)
        if len(session_history) > 5:
            recent_ears = list(session_history)[-10:]
            blink_threshold = 0.25
            blinks = sum(1 for ear in recent_ears if ear < blink_threshold)
            blink_rate = blinks / len(recent_ears) if recent_ears else 0
        else:
            blink_rate = 0
        
        # Analyze eye strain indicators
        # Normal EAR is around 0.25-0.35, lower indicates closed/tired eyes
        # Very low blink rate (< 0.1) indicates staring (eye strain)
        # Very high blink rate (> 0.3) indicates tiredness
        
        strain_factors = []
        
        if avg_ear < 0.2:
            strain_factors.append("eyes_closed")
        elif avg_ear < 0.25:
            strain_factors.append("eyes_tired")
        
        if blink_rate < 0.05 and len(session_history) > 10:
            strain_factors.append("staring")
        elif blink_rate > 0.3:
            strain_factors.append("excessive_blinking")
        
        # Calculate risk level
        if len(strain_factors) >= 2:
            eye_strain_risk = "high"
            eye_score = 40
        elif len(strain_factors) == 1:
            eye_strain_risk = "medium"
            eye_score = 65
        else:
            eye_strain_risk = "low"
            eye_score = 90
        
        # Adjust score based on EAR
        if avg_ear < 0.22:
            eye_score -= 20
        elif avg_ear > 0.32:
            eye_score += 5
        
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
        
        if face_detection_result and hasattr(face_detection_result, 'detections') and len(face_detection_result.detections) > 0:
            face_visible = True
            detection = face_detection_result.detections[0]
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
            if landmarks[4] is not None:  # Nose tip
                face_center = [landmarks[4].x, landmarks[4].y]
            else:
                face_center = [landmarks[0].x, landmarks[0].y]
        
        if not face_visible:
            return {"concentration": "low", "score": 25, "face_visible": False, "head_stability": 0}
        
        # Use provided image shape or default
        if image_shape is None:
            image_shape = np.array([480, 640, 3])
        
        # Get head pose if landmarks available
        if landmarks and len(landmarks) >= 468:
            head_pose = self.calculate_head_pose(landmarks, image_shape)
        else:
            head_pose = {"pitch": 0, "yaw": 0, "roll": 0, "tilted": False}
        
        # Track head movement (stability indicates focus)
        if face_center:
            session_history.append(face_center)
        
        # Calculate movement variance
        if len(session_history) > 5:
            positions = np.array(list(session_history)[-10:])
            movement_variance = np.var(positions, axis=0).sum()
        else:
            movement_variance = 0.01  # Default low movement
        
        # Analyze engagement factors
        # 1. Face visibility and confidence
        visibility_score = 100 if face_visible else 30
        
        # 2. Head stability (low movement = focused)
        if movement_variance < 0.0001:
            stability_score = 100
        elif movement_variance < 0.0005:
            stability_score = 80
        elif movement_variance < 0.001:
            stability_score = 60
        elif movement_variance < 0.002:
            stability_score = 45
        else:
            stability_score = 30
        
        # 3. Head orientation (facing forward = engaged)
        if landmarks and len(landmarks) >= 468:
            yaw_penalty = min(50, abs(head_pose["yaw"]) * 2)
            orientation_score = 100 - yaw_penalty
        else:
            # Estimate from face position
            if face_center:
                horizontal_offset = abs(face_center[0] - 0.5)
                orientation_score = 100 - min(40, horizontal_offset * 80)
            else:
                orientation_score = 50
        
        # 4. Face size/confidence (larger face = closer = more engaged)
        if face_detection_result and hasattr(face_detection_result, 'detections') and len(face_detection_result.detections) > 0:
            detection = face_detection_result.detections[0]
            if hasattr(detection, 'score') and len(detection.score) > 0:
                confidence = detection.score[0]
                confidence_score = confidence * 100
            else:
                confidence_score = 70
        else:
            confidence_score = 60
        
        # Calculate overall engagement with dynamic weighting
        engagement_score = (
            visibility_score * 0.25 + 
            stability_score * 0.35 + 
            orientation_score * 0.25 + 
            confidence_score * 0.15
        )
        engagement_score = max(20, min(100, engagement_score))  # Keep range 20-100
        
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
    
    def analyze_stress(self, image, landmarks, face_detection_result) -> Dict:
        """Improved stress analysis using facial expressions and micro-expressions"""
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
        
        stress_indicators = []
        
        try:
            # 1. Analyze eyebrow position (furrowed = stress)
            left_eyebrow = landmarks[107]  # Left eyebrow inner
            right_eyebrow = landmarks[336]  # Right eyebrow inner
            left_eye_top = landmarks[159]   # Left eye top
            right_eye_top = landmarks[386]  # Right eye top
            
            # Distance between eyebrow and eye (smaller = furrowed)
            left_eyebrow_eye_dist = abs(left_eyebrow.y - left_eye_top.y)
            right_eyebrow_eye_dist = abs(right_eyebrow.y - right_eye_top.y)
            avg_eyebrow_dist = (left_eyebrow_eye_dist + right_eyebrow_eye_dist) / 2
            
            if avg_eyebrow_dist < 0.015:  # Threshold for furrowed brows
                stress_indicators.append("furrowed_brows")
            
            # 2. Analyze mouth position (tight/pursed = stress)
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            mouth_top = landmarks[13]
            mouth_bottom = landmarks[14]
            
            mouth_width = abs(mouth_right.x - mouth_left.x)
            mouth_height = abs(mouth_bottom.y - mouth_top.y)
            mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            if mouth_ratio < 0.15:  # Tight/pursed lips
                stress_indicators.append("tight_mouth")
            
            # 3. Analyze jaw tension (clenched = stress)
            jaw_left = landmarks[172]
            jaw_right = landmarks[397]
            jaw_center = landmarks[175]
            
            jaw_width = abs(jaw_right.x - jaw_left.x)
            # Narrow jaw width can indicate clenching
            if jaw_width < 0.15:
                stress_indicators.append("jaw_tension")
            
            # 4. Overall facial tension (based on landmark distances)
            face_width = abs(landmarks[454].x - landmarks[234].x)
            face_height = abs(landmarks[152].y - landmarks[10].y)
            face_ratio = face_width / face_height if face_height > 0 else 1
            
            # Calculate stress score
            stress_count = len(stress_indicators)
            
            if stress_count >= 3:
                stress_level = "high"
                stress_score = 30
            elif stress_count == 2:
                stress_level = "medium"
                stress_score = 60
            elif stress_count == 1:
                stress_level = "low-medium"
                stress_score = 75
            else:
                stress_level = "low"
                stress_score = 90
            
            # Adjust based on facial tension
            if face_ratio < 0.7:  # Compressed face (tension)
                stress_score -= 10
            
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
            logging.info(f"Face detection: detected={face_detected}, has_landmarks={landmarks is not None and len(landmarks) > 0}, method={'MediaPipe' if MEDIAPIPE_AVAILABLE else 'OpenCV'}")
        except Exception as face_err:
            logging.error(f"Error in face detection: {face_err}", exc_info=True)
            face_results, landmarks, mesh_results = None, None, None
        
        # Get session history for temporal smoothing
        session_id = "http_session"
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
            
            logging.info(f"Analysis complete - Posture: {posture.get('score', 'N/A')}, Eye: {eye_strain.get('score', 'N/A')}, Engagement: {engagement.get('score', 'N/A')}, Stress: {stress.get('score', 'N/A')}, Productivity: {productivity.get('productivity_score', 'N/A')}")
            logging.info(f"Face detected: {face_results is not None}, Landmarks: {landmarks is not None and len(landmarks) > 0 if landmarks else False}")
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
            
            sessions[session_id]["frame_count"] += 1
            
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

