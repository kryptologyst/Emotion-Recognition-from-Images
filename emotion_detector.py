"""
Modern Emotion Recognition System
Uses state-of-the-art face detection and CNN models for accurate emotion recognition
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
from mtcnn import MTCNN
import sqlite3
import json
from datetime import datetime
import os
import argparse
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    """Modern emotion detection system with multiple face detection methods"""
    
    def __init__(self, model_path: str = "models/emotion_model.h5", 
                 detection_method: str = "mediapipe"):
        """
        Initialize the emotion detector
        
        Args:
            model_path: Path to the trained emotion recognition model
            detection_method: Face detection method ('mediapipe', 'mtcnn', 'haar')
        """
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.detection_method = detection_method
        
        # Initialize face detection
        self._setup_face_detection()
        
        # Load emotion recognition model
        self._load_model(model_path)
        
        # Initialize database
        self._init_database()
    
    def _setup_face_detection(self):
        """Setup face detection based on selected method"""
        if self.detection_method == "mediapipe":
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        elif self.detection_method == "mtcnn":
            self.mtcnn_detector = MTCNN()
        elif self.detection_method == "haar":
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        else:
            raise ValueError(f"Unsupported detection method: {self.detection_method}")
    
    def _load_model(self, model_path: str):
        """Load the emotion recognition model"""
        try:
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.warning(f"Model not found at {model_path}. Creating default model...")
                self.model = self._create_default_model()
                self._save_model(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self._create_default_model()
    
    def _create_default_model(self) -> keras.Model:
        """Create a default CNN model for emotion recognition"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _save_model(self, model_path: str):
        """Save the model to specified path"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def _init_database(self):
        """Initialize SQLite database for storing detection results"""
        self.db_path = "database/emotion_detections.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                emotion TEXT,
                confidence REAL,
                face_count INTEGER,
                detection_method TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image using selected method"""
        faces = []
        
        if self.detection_method == "mediapipe":
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                h, w, _ = image.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append((x, y, width, height))
        
        elif self.detection_method == "mtcnn":
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.mtcnn_detector.detect_faces(rgb_image)
            
            for detection in detections:
                x, y, w, h = detection['box']
                faces.append((x, y, w, h))
        
        elif self.detection_method == "haar":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in face_rects:
                faces.append((x, y, w, h))
        
        return faces
    
    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face region for emotion prediction"""
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        face_resized = cv2.resize(face_roi, (48, 48))
        
        # Normalize
        face_normalized = face_resized.astype("float32") / 255.0
        
        # Add batch and channel dimensions
        face_input = np.expand_dims(face_normalized, axis=(0, -1))
        
        return face_input
    
    def predict_emotion(self, face_input: np.ndarray) -> Tuple[str, float]:
        """Predict emotion from preprocessed face"""
        prediction = self.model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(prediction)
        confidence = float(prediction[0][emotion_idx])
        emotion = self.emotion_labels[emotion_idx]
        
        return emotion, confidence
    
    def detect_emotions(self, image: np.ndarray, image_path: str = None) -> Dict:
        """Detect emotions in all faces in the image"""
        faces = self.detect_faces(image)
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'face_count': len(faces),
            'detections': [],
            'detection_method': self.detection_method
        }
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Preprocess and predict
            face_input = self.preprocess_face(face_roi)
            emotion, confidence = self.predict_emotion(face_input)
            
            detection = {
                'face_id': i,
                'bbox': [x, y, w, h],
                'emotion': emotion,
                'confidence': confidence
            }
            
            results['detections'].append(detection)
        
        # Save to database
        self._save_detection_to_db(results)
        
        return results
    
    def _save_detection_to_db(self, results: Dict):
        """Save detection results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for detection in results['detections']:
            cursor.execute('''
                INSERT INTO detections 
                (image_path, emotion, confidence, face_count, detection_method, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                results['image_path'],
                detection['emotion'],
                detection['confidence'],
                results['face_count'],
                results['detection_method'],
                json.dumps(detection)
            ))
        
        conn.commit()
        conn.close()
    
    def detect_from_webcam(self):
        """Real-time emotion detection from webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        logger.info("Starting webcam detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect emotions
            results = self.detect_emotions(frame)
            
            # Draw results
            self._draw_results(frame, results)
            
            # Show frame
            cv2.imshow("Emotion Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_from_image(self, image_path: str) -> Dict:
        """Detect emotions from a single image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self.detect_emotions(image, image_path)
    
    def detect_from_folder(self, folder_path: str) -> List[Dict]:
        """Detect emotions from all images in a folder"""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        results = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(folder_path, filename)
                try:
                    result = self.detect_from_image(image_path)
                    results.append(result)
                    logger.info(f"Processed: {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        return results
    
    def _draw_results(self, image: np.ndarray, results: Dict):
        """Draw detection results on image"""
        for detection in results['detections']:
            x, y, w, h = detection['bbox']
            emotion = detection['emotion']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    def get_detection_history(self, limit: int = 100) -> List[Dict]:
        """Get recent detection history from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM detections 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        columns = ['id', 'timestamp', 'image_path', 'emotion', 'confidence', 
                  'face_count', 'detection_method', 'metadata']
        
        history = []
        for row in rows:
            record = dict(zip(columns, row))
            record['metadata'] = json.loads(record['metadata']) if record['metadata'] else {}
            history.append(record)
        
        return history


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description="Modern Emotion Recognition System")
    parser.add_argument("--mode", choices=["webcam", "image", "folder"], 
                       default="webcam", help="Detection mode")
    parser.add_argument("--path", help="Path to image or folder")
    parser.add_argument("--model", default="models/emotion_model.h5", 
                       help="Path to emotion model")
    parser.add_argument("--detection", choices=["mediapipe", "mtcnn", "haar"], 
                       default="mediapipe", help="Face detection method")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = EmotionDetector(args.model, args.detection)
    
    if args.mode == "webcam":
        detector.detect_from_webcam()
    elif args.mode == "image":
        if not args.path:
            print("Error: --path required for image mode")
            return
        results = detector.detect_from_image(args.path)
        print(json.dumps(results, indent=2))
    elif args.mode == "folder":
        if not args.path:
            print("Error: --path required for folder mode")
            return
        results = detector.detect_from_folder(args.path)
        print(f"Processed {len(results)} images")


if __name__ == "__main__":
    main()
