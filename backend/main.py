from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from typing import Dict
import json
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="HCI Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple face detector using OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

sessions: Dict[str, Dict] = {}

class WellnessAnalyzer:
    def __init__(self):
        pass
        
    def analyze_posture(self, image, face_rect) -> Dict:
        """Simple posture analysis based on face position"""
        if face_rect is None:
            return {"slouching": False, "score": 70}
        
        h, w = image.shape[:2]
        x, y, fw, fh = face_rect
        
        # Face should be in upper center of image for good posture
        face_center_x = x + fw / 2
        face_center_y = y + fh / 2
        
        # Check if face is centered horizontally
        horizontal_offset = abs(face_center_x - w/2) / w
        
        # Check if face is in upper portion (not slouching)
        vertical_position = face_center_y / h
        
        slouching = vertical_position > 0.6 or horizontal_offset > 0.3
        posture_score = 100 - (vertical_position * 50) - (horizontal_offset * 100)
        posture_score = max(0, min(100, posture_score))
        
        return {
            "slouching": slouching,
            "score": round(posture_score, 2)
        }
    
    def analyze_eye_strain(self, image, face_rect) -> Dict:
        """Simple eye strain estimation"""
        if face_rect is None:
            return {"eye_strain_risk": "low", "score": 100}
        
        # Simple heuristic: if face detected, assume moderate risk
        # In full version, this would analyze eye landmarks
        eye_strain_risk = "medium"
        eye_score = 75
        
        return {
            "eye_strain_risk": eye_strain_risk,
            "score": round(eye_score, 2)
        }
    
    def analyze_engagement(self, face_rect) -> Dict:
        """Analyze engagement based on face detection"""
        if face_rect is None:
            return {"concentration": "low", "score": 30}
        
        concentration = "high"
        score = 80
        
        return {
            "concentration": concentration,
            "score": score,
            "face_visible": True
        }
    
    def analyze_stress(self, image, face_rect) -> Dict:
        """Simple stress analysis"""
        if face_rect is None:
            return {"stress_level": "low", "score": 100}
        
        # Simple heuristic
        stress_level = "low"
        stress_score = 85
        
        return {
            "stress_level": stress_level,
            "score": round(stress_score, 2)
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
            recommendations.append("👁️ Take a 20-20-20 break: Look 20ft away for 20 seconds")
        
        if analysis["break_needed"]:
            recommendations.append("☕ Take a 5-minute micro-break")
        
        if analysis.get("stress_level") in ["medium", "high"]:
            recommendations.append("🧘 Take 3 deep breaths to reduce stress")
        
        if not recommendations:
            recommendations.append("✅ You're doing great! Keep it up")
        
        return recommendations

analyzer = WellnessAnalyzer()

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
            except Exception as e:
                logging.error(f"Error receiving/parsing data: {e}", exc_info=True)
                # Send error response and continue
                try:
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
                except:
                    pass
                continue
            
            # Decode image
            try:
                image = decode_image(frame_data["image"])
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Detect face
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_rect = faces[0] if len(faces) > 0 else None
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
            
            # Analyze
            posture = analyzer.analyze_posture(image, face_rect)
            eye_strain = analyzer.analyze_eye_strain(image, face_rect)
            engagement = analyzer.analyze_engagement(face_rect)
            stress = analyzer.analyze_stress(image, face_rect)
            
            # Calculate productivity
            productivity = analyzer.calculate_productivity_score(posture, eye_strain, engagement, stress)
            
            # Get recommendations
            recommendations = analyzer.get_recommendations({
                **productivity,
                "stress_level": stress["stress_level"]
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

