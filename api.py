"""
Flask API for Emotion Recognition System
RESTful API endpoints for emotion detection
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import os
from datetime import datetime
from emotion_detector import EmotionDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize emotion detector
detector = None

def initialize_detector():
    """Initialize the emotion detector"""
    global detector
    if detector is None:
        try:
            detector = EmotionDetector()
            logger.info("Emotion detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing detector: {e}")
            raise

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/detect', methods=['POST'])
def detect_emotions():
    """Detect emotions from uploaded image"""
    try:
        initialize_detector()
        
        # Check if image is provided
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read image
            image = Image.open(file.stream)
            image_array = np.array(image)
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Handle base64 encoded image
        elif 'image_data' in request.json:
            image_data = request.json['image_data']
            
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Detect emotions
        results = detector.detect_emotions(image_cv)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in detect_emotions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect/batch', methods=['POST'])
def detect_emotions_batch():
    """Detect emotions from multiple images"""
    try:
        initialize_detector()
        
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            try:
                # Read image
                image = Image.open(file.stream)
                image_array = np.array(image)
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Detect emotions
                result = detector.detect_emotions(image_cv, file.filename)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'total_images': len(files),
            'processed_images': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in detect_emotions_batch: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_detection_history():
    """Get detection history"""
    try:
        limit = request.args.get('limit', 100, type=int)
        history = detector.get_detection_history(limit)
        
        return jsonify({
            'total_records': len(history),
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get detection statistics"""
    try:
        import sqlite3
        
        if not os.path.exists(detector.db_path):
            return jsonify({'error': 'No database found'}), 404
        
        conn = sqlite3.connect(detector.db_path)
        cursor = conn.cursor()
        
        # Get total detections
        cursor.execute("SELECT COUNT(*) FROM detections")
        total_detections = cursor.fetchone()[0]
        
        # Get detections by emotion
        cursor.execute("""
            SELECT emotion, COUNT(*) as count 
            FROM detections 
            GROUP BY emotion 
            ORDER BY count DESC
        """)
        emotion_counts = dict(cursor.fetchall())
        
        # Get recent detections (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM detections 
            WHERE timestamp >= datetime('now', '-1 day')
        """)
        recent_detections = cursor.fetchone()[0]
        
        # Get average confidence
        cursor.execute("SELECT AVG(confidence) FROM detections")
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'total_detections': total_detections,
            'emotion_counts': emotion_counts,
            'recent_detections': recent_detections,
            'avg_confidence': avg_confidence
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    try:
        initialize_detector()
        
        model_info = {
            'emotion_labels': detector.emotion_labels,
            'detection_method': detector.detection_method,
            'model_path': detector.model_path if hasattr(detector, 'model_path') else 'default',
            'input_shape': detector.model.input_shape if detector.model else None
        }
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
