"""
Modern Streamlit Web Interface for Emotion Recognition
Beautiful, interactive web app for real-time emotion detection
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from datetime import datetime, timedelta
import os
import base64
from PIL import Image
import io
from emotion_detector import EmotionDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .emotion-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

class EmotionRecognitionApp:
    """Main application class for Streamlit interface"""
    
    def __init__(self):
        """Initialize the application"""
        self.detector = None
        self.db_path = "database/emotion_detections.db"
        
    def initialize_detector(self):
        """Initialize emotion detector"""
        if self.detector is None:
            try:
                self.detector = EmotionDetector()
                return True
            except Exception as e:
                st.error(f"Error initializing detector: {e}")
                return False
        return True
    
    def get_detection_stats(self) -> dict:
        """Get detection statistics from database"""
        if not os.path.exists(self.db_path):
            return {}
        
        conn = sqlite3.connect(self.db_path)
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
        
        return {
            'total_detections': total_detections,
            'emotion_counts': emotion_counts,
            'recent_detections': recent_detections,
            'avg_confidence': avg_confidence
        }
    
    def run(self):
        """Run the main application"""
        
        # Header
        st.markdown('<h1 class="main-header">😊 Emotion Recognition System</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Detection", "📸 Image Upload", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            self.render_live_detection()
        
        with tab2:
            self.render_image_upload()
        
        with tab3:
            self.render_analytics()
        
        with tab4:
            self.render_settings()
    
    def render_sidebar(self):
        """Render sidebar with statistics"""
        st.sidebar.title("📊 Statistics")
        
        stats = self.get_detection_stats()
        
        if stats:
            st.sidebar.metric("Total Detections", stats['total_detections'])
            st.sidebar.metric("Recent (24h)", stats['recent_detections'])
            st.sidebar.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")
            
            # Emotion distribution
            if stats['emotion_counts']:
                st.sidebar.subheader("Emotion Distribution")
                emotion_df = pd.DataFrame(
                    list(stats['emotion_counts'].items()),
                    columns=['Emotion', 'Count']
                )
                fig = px.pie(emotion_df, values='Count', names='Emotion', 
                           title="Detection Distribution")
                st.sidebar.plotly_chart(fig, use_container_width=True)
        else:
            st.sidebar.info("No detection data available yet.")
    
    def render_live_detection(self):
        """Render live detection interface"""
        st.header("🎥 Live Emotion Detection")
        
        if not self.initialize_detector():
            st.error("Failed to initialize emotion detector")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Webcam Feed")
            
            # Camera input
            camera_input = st.camera_input("Take a picture for emotion detection")
            
            if camera_input:
                # Process image
                image = Image.open(camera_input)
                image_array = np.array(image)
                
                # Convert PIL to OpenCV format
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Detect emotions
                with st.spinner("Detecting emotions..."):
                    results = self.detector.detect_emotions(image_cv)
                
                # Display results
                self.display_detection_results(results, image_cv)
        
        with col2:
            st.subheader("Detection Info")
            st.info("""
            **How it works:**
            1. Position your face in the camera
            2. Click the camera button to capture
            3. The system will detect your emotion
            4. Results are saved to the database
            """)
            
            # Real-time stats
            stats = self.get_detection_stats()
            if stats:
                st.metric("Total Faces Detected", stats['total_detections'])
                st.metric("Average Confidence", f"{stats['avg_confidence']:.2f}")
    
    def render_image_upload(self):
        """Render image upload interface"""
        st.header("📸 Upload Image for Analysis")
        
        if not self.initialize_detector():
            st.error("Failed to initialize emotion detector")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload an image containing faces for emotion detection"
            )
            
            if uploaded_file:
                # Process uploaded image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Convert PIL to OpenCV format
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Detect emotions
                with st.spinner("Analyzing image..."):
                    results = self.detector.detect_emotions(image_cv, uploaded_file.name)
                
                # Display results
                self.display_detection_results(results, image_cv)
        
        with col2:
            st.subheader("Batch Processing")
            
            # Folder upload (simulated)
            st.info("""
            **Batch Processing:**
            For processing multiple images, use the command line interface:
            
            ```bash
            python emotion_detector.py --mode folder --path /path/to/images/
            ```
            """)
            
            # Recent detections
            st.subheader("Recent Detections")
            self.display_recent_detections()
    
    def display_detection_results(self, results: dict, image: np.ndarray):
        """Display detection results"""
        
        if not results['detections']:
            st.warning("No faces detected in the image")
            return
        
        st.success(f"Detected {results['face_count']} face(s)")
        
        # Display image with bounding boxes
        display_image = image.copy()
        
        for detection in results['detections']:
            x, y, w, h = detection['bbox']
            emotion = detection['emotion']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(display_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Convert back to RGB for display
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        st.image(display_image_rgb, caption="Detection Results", use_column_width=True)
        
        # Display detection details
        st.subheader("Detection Details")
        
        for i, detection in enumerate(results['detections']):
            with st.expander(f"Face {i+1} - {detection['emotion']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Emotion", detection['emotion'])
                with col2:
                    st.metric("Confidence", f"{detection['confidence']:.3f}")
                with col3:
                    st.metric("Bounding Box", f"{detection['bbox']}")
    
    def display_recent_detections(self):
        """Display recent detection history"""
        if not os.path.exists(self.db_path):
            st.info("No detection history available")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, emotion, confidence, face_count 
            FROM detections 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if rows:
            df = pd.DataFrame(rows, columns=['Timestamp', 'Emotion', 'Confidence', 'Faces'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent detections found")
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        
        if not os.path.exists(self.db_path):
            st.info("No data available for analytics")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        # Emotion distribution over time
        st.subheader("Emotion Distribution Over Time")
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DATE(timestamp) as date, emotion, COUNT(*) as count
            FROM detections 
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(timestamp), emotion
            ORDER BY date DESC
        """)
        
        time_data = cursor.fetchall()
        
        if time_data:
            df_time = pd.DataFrame(time_data, columns=['Date', 'Emotion', 'Count'])
            df_time['Date'] = pd.to_datetime(df_time['Date'])
            
            fig = px.line(df_time, x='Date', y='Count', color='Emotion',
                         title="Emotion Detection Trends (Last 30 Days)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        st.subheader("Confidence Distribution")
        
        cursor.execute("SELECT emotion, confidence FROM detections")
        conf_data = cursor.fetchall()
        
        if conf_data:
            df_conf = pd.DataFrame(conf_data, columns=['Emotion', 'Confidence'])
            
            fig = px.box(df_conf, x='Emotion', y='Confidence',
                        title="Confidence Distribution by Emotion")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detection method comparison
        st.subheader("Detection Method Performance")
        
        cursor.execute("""
            SELECT detection_method, emotion, AVG(confidence) as avg_confidence
            FROM detections 
            GROUP BY detection_method, emotion
        """)
        
        method_data = cursor.fetchall()
        
        if method_data:
            df_method = pd.DataFrame(method_data, columns=['Method', 'Emotion', 'Avg_Confidence'])
            
            fig = px.bar(df_method, x='Emotion', y='Avg_Confidence', color='Method',
                        title="Average Confidence by Detection Method")
            st.plotly_chart(fig, use_container_width=True)
        
        conn.close()
    
    def render_settings(self):
        """Render settings interface"""
        st.header("⚙️ Settings")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Detection Settings")
            
            # Detection method
            detection_method = st.selectbox(
                "Face Detection Method",
                ["mediapipe", "mtcnn", "haar"],
                index=0,
                help="Choose the face detection algorithm"
            )
            
            # Model settings
            model_path = st.text_input(
                "Model Path",
                value="models/emotion_model.h5",
                help="Path to the emotion recognition model"
            )
            
            if st.button("Apply Settings"):
                st.success("Settings applied! Please refresh the page.")
        
        with col2:
            st.subheader("Database Management")
            
            # Database stats
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM detections")
                total_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT DATE(timestamp)) FROM detections")
                active_days = cursor.fetchone()[0]
                
                conn.close()
                
                st.metric("Total Records", total_records)
                st.metric("Active Days", active_days)
                
                if st.button("Clear Database"):
                    if st.checkbox("I understand this will delete all data"):
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM detections")
                        conn.commit()
                        conn.close()
                        st.success("Database cleared!")
                        st.experimental_rerun()
            else:
                st.info("No database found")
        
        # Model training section
        st.subheader("Model Training")
        
        st.info("""
        To train a new model, use the command line interface:
        
        ```bash
        python train_model.py --epochs 50 --batch_size 32
        ```
        
        Make sure you have the FER-2013 dataset in the data directory.
        """)


def main():
    """Main function to run the Streamlit app"""
    app = EmotionRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()
