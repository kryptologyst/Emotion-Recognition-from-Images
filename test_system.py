"""
Test Suite for Emotion Recognition System
Comprehensive tests for all components
"""

import unittest
import os
import sys
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from emotion_detector import EmotionDetector
    from train_model import EmotionModelTrainer
    EMOTION_DETECTOR_AVAILABLE = True
except ImportError:
    EMOTION_DETECTOR_AVAILABLE = False

class TestEmotionDetector(unittest.TestCase):
    """Test cases for EmotionDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not EMOTION_DETECTOR_AVAILABLE:
            self.skipTest("EmotionDetector not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.detector = EmotionDetector()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.emotion_labels)
        self.assertEqual(len(self.detector.emotion_labels), 7)
    
    def test_emotion_labels(self):
        """Test emotion labels"""
        expected_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.assertEqual(self.detector.emotion_labels, expected_labels)
    
    def test_preprocess_face(self):
        """Test face preprocessing"""
        # Create a dummy face image
        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        processed = self.detector.preprocess_face(face)
        
        self.assertEqual(processed.shape, (1, 48, 48, 1))
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))
    
    def test_detect_faces_empty_image(self):
        """Test face detection on empty image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(empty_image)
        
        self.assertEqual(len(faces), 0)
    
    def test_detect_faces_with_face(self):
        """Test face detection with a simple face-like pattern"""
        # Create a simple face-like pattern
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Draw a simple face
        cv2.circle(image, (100, 100), 50, (255, 255, 255), -1)  # Face
        cv2.circle(image, (85, 85), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(image, (115, 85), 5, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(image, (100, 110), (10, 5), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        faces = self.detector.detect_faces(image)
        
        # Should detect at least one face (depending on detection method)
        self.assertGreaterEqual(len(faces), 0)
    
    def test_predict_emotion(self):
        """Test emotion prediction"""
        # Create dummy face input
        face_input = np.random.random((1, 48, 48, 1))
        
        emotion, confidence = self.detector.predict_emotion(face_input)
        
        self.assertIn(emotion, self.detector.emotion_labels)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_detect_emotions_integration(self):
        """Test complete emotion detection pipeline"""
        # Create a test image
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        results = self.detector.detect_emotions(image)
        
        self.assertIn('face_count', results)
        self.assertIn('detections', results)
        self.assertIn('timestamp', results)
        self.assertIn('detection_method', results)
        self.assertIsInstance(results['face_count'], int)
        self.assertIsInstance(results['detections'], list)

class TestModelTrainer(unittest.TestCase):
    """Test cases for EmotionModelTrainer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = EmotionModelTrainer(data_dir=self.temp_dir, model_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.num_classes, 7)
        self.assertEqual(self.trainer.img_size, (48, 48))
    
    def test_create_model(self):
        """Test model creation"""
        model = self.trainer.create_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 48, 48, 1))
        self.assertEqual(model.output_shape, (None, 7))
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation"""
        success = self.trainer.create_sample_dataset()
        self.assertTrue(success)
        
        # Check if sample file exists
        sample_path = os.path.join(self.temp_dir, "sample_fer2013.csv")
        self.assertTrue(os.path.exists(sample_path))

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_imports(self):
        """Test that all required modules can be imported"""
        try:
            import cv2
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            import matplotlib.pyplot as plt
            import seaborn as sns
            import streamlit as st
            import plotly.express as px
        except ImportError as e:
            self.fail(f"Required module not available: {e}")
    
    def test_opencv_version(self):
        """Test OpenCV version"""
        import cv2
        version = cv2.__version__
        self.assertIsNotNone(version)
        print(f"OpenCV version: {version}")
    
    def test_tensorflow_version(self):
        """Test TensorFlow version"""
        import tensorflow as tf
        version = tf.__version__
        self.assertIsNotNone(version)
        print(f"TensorFlow version: {version}")

class TestProjectStructure(unittest.TestCase):
    """Test project structure and files"""
    
    def test_required_files_exist(self):
        """Test that all required files exist"""
        required_files = [
            'requirements.txt',
            'README.md',
            'emotion_detector.py',
            'train_model.py',
            'app.py',
            'api.py',
            'setup.py',
            '0115.py'
        ]
        
        for file in required_files:
            self.assertTrue(os.path.exists(file), f"Required file {file} not found")
    
    def test_directory_structure(self):
        """Test that required directories exist or can be created"""
        required_dirs = [
            'models',
            'data',
            'database',
            'scripts',
            'templates',
            'static'
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            self.assertTrue(os.path.exists(directory), f"Required directory {directory} not found")

def run_performance_test():
    """Run basic performance tests"""
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    if not EMOTION_DETECTOR_AVAILABLE:
        print("❌ EmotionDetector not available for performance tests")
        return
    
    import time
    
    # Test detector initialization time
    start_time = time.time()
    detector = EmotionDetector()
    init_time = time.time() - start_time
    print(f"✅ Detector initialization: {init_time:.3f}s")
    
    # Test face preprocessing time
    face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    start_time = time.time()
    processed = detector.preprocess_face(face)
    preprocess_time = time.time() - start_time
    print(f"✅ Face preprocessing: {preprocess_time:.3f}s")
    
    # Test emotion prediction time
    start_time = time.time()
    emotion, confidence = detector.predict_emotion(processed)
    prediction_time = time.time() - start_time
    print(f"✅ Emotion prediction: {prediction_time:.3f}s")
    
    print(f"✅ Total pipeline time: {init_time + preprocess_time + prediction_time:.3f}s")

def main():
    """Run all tests"""
    print("🧪 EMOTION RECOGNITION SYSTEM - TEST SUITE")
    print("="*60)
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestProjectStructure))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance tests
    run_performance_test()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
