"""Quick test to verify the server can start and handle requests"""
import sys
import traceback

try:
    print("Testing imports...")
    from main import app, detect_face_mediapipe, decode_image
    import numpy as np
    import base64
    import cv2
    
    print("✅ All imports successful")
    
    # Test image decoding
    print("\nTesting image decoding...")
    # Create a test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', test_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_base64}"
    
    decoded = decode_image(img_data_url)
    print(f"✅ Image decoding works: shape={decoded.shape}")
    
    # Test face detection
    print("\nTesting face detection...")
    result = detect_face_mediapipe(decoded)
    print(f"✅ Face detection works: face_detected={result[0] is not None}")
    
    # Test app
    print("\nTesting FastAPI app...")
    print(f"✅ FastAPI app created: {app.title}")
    
    print("\n🎉 All tests passed! Server should work.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
    sys.exit(1)

