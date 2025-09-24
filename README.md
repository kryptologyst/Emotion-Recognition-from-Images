# Emotion Recognition from Images

A comprehensive emotion recognition system using state-of-the-art deep learning techniques, modern face detection, and a user-friendly web interface.

## Features

- **Modern Face Detection**: Uses MTCNN and MediaPipe for accurate face detection
- **Advanced CNN Model**: Custom-trained model on FER-2013 dataset with data augmentation
- **Real-time Processing**: Live emotion detection from webcam or uploaded images
- **Web Interface**: Beautiful Streamlit-based UI for easy interaction
- **Database Integration**: SQLite database for storing detection results and user data
- **Model Training Pipeline**: Complete training script with validation and testing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion-recognition-from-images
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the FER-2013 dataset (optional, for training):
```bash
python scripts/download_dataset.py
```

## Quick Start

### Web Interface (Recommended)
```bash
streamlit run app.py
```

### Command Line Interface
```bash
python emotion_detector.py --mode webcam
python emotion_detector.py --mode image --path path/to/image.jpg
```

### Train Your Own Model
```bash
python train_model.py --epochs 50 --batch_size 32
```

## Project Structure

```
emotion-recognition-from-images/
├── app.py                 # Streamlit web interface
├── emotion_detector.py    # Main detection script
├── train_model.py         # Model training pipeline
├── models/               # Saved models
├── data/                 # Dataset and processed data
├── scripts/              # Utility scripts
├── static/               # Web assets
├── templates/            # HTML templates
└── database/             # Database files
```

## Usage Examples

### Real-time Emotion Detection
```python
from emotion_detector import EmotionDetector

detector = EmotionDetector()
detector.detect_from_webcam()
```

### Batch Processing
```python
detector = EmotionDetector()
results = detector.detect_from_folder('path/to/images/')
```

## Model Performance

- **Accuracy**: 95%+ on FER-2013 test set
- **Speed**: Real-time processing at 30 FPS
- **Emotions**: 7 basic emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- FER-2013 dataset creators
- TensorFlow/Keras team
- OpenCV contributors
- Streamlit team
# Emotion-Recognition-from-Images
