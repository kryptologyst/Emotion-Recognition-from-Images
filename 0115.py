# Project 115. Emotion recognition from images - MODERN IMPLEMENTATION
# Description:
# This is a comprehensive emotion recognition system using state-of-the-art deep learning techniques.
# Features include modern face detection (MediaPipe, MTCNN), advanced CNN models, real-time processing,
# web interfaces (Streamlit & Flask), database integration, and comprehensive analytics.

# This file serves as the original implementation and entry point.
# For the modern implementation, use the following files:
# - emotion_detector.py: Main detection engine with modern techniques
# - app.py: Streamlit web interface
# - api.py: Flask REST API
# - train_model.py: Model training pipeline

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from emotion_detector import EmotionDetector
    print("✅ Modern emotion detector available")
    print("🚀 Use 'python emotion_detector.py --mode webcam' for real-time detection")
    print("🌐 Use 'streamlit run app.py' for web interface")
    print("🔧 Use 'python train_model.py' to train your own model")
    print("📡 Use 'python api.py' for REST API server")
except ImportError as e:
    print(f"❌ Modern detector not available: {e}")
    print("📦 Install requirements: pip install -r requirements.txt")

# Original simple implementation (kept for reference)
def original_implementation():
    """Original simple emotion recognition implementation"""
    print("Running original implementation...")
    
    # Load pre-trained emotion recognition model
    model_path = "emotion_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("💡 Train a model first: python train_model.py")
        return
    
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Load webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))
            
            prediction = model.predict(face_input, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Draw rectangle and emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.imshow("Emotion Recognition (Original)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def modern_implementation():
    """Modern emotion recognition implementation"""
    print("Running modern implementation...")
    
    try:
        detector = EmotionDetector()
        detector.detect_from_webcam()
    except Exception as e:
        print(f"❌ Error with modern implementation: {e}")
        print("💡 Falling back to original implementation...")
        original_implementation()

def main():
    """Main function with options"""
    print("=" * 60)
    print("😊 EMOTION RECOGNITION SYSTEM")
    print("=" * 60)
    print()
    print("Choose implementation:")
    print("1. Modern implementation (recommended)")
    print("2. Original implementation")
    print("3. Exit")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        modern_implementation()
    elif choice == "2":
        original_implementation()
    elif choice == "3":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please run again.")

if __name__ == "__main__":
    main()