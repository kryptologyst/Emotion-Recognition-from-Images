#!/bin/bash
echo "Starting Emotion Recognition System..."
echo ""
echo "Choose an option:"
echo "1. Streamlit Web Interface"
echo "2. Flask API Server"
echo "3. Command Line Interface"
echo "4. Train Model"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Starting Streamlit..."
        streamlit run app.py
        ;;
    2)
        echo "Starting Flask API..."
        python api.py
        ;;
    3)
        echo "Starting CLI..."
        python emotion_detector.py --mode webcam
        ;;
    4)
        echo "Starting model training..."
        python train_model.py
        ;;
    *)
        echo "Invalid choice"
        ;;
esac
