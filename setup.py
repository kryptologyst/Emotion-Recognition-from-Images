#!/usr/bin/env python3
"""
Complete Setup Script for Emotion Recognition Project
Automates the entire setup process for the emotion recognition system
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Complete project setup automation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = sys.version_info
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        
        if self.python_version < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        logger.info(f"✅ Python {self.python_version.major}.{self.python_version.minor} detected")
        return True
    
    def create_directory_structure(self):
        """Create the complete project directory structure"""
        logger.info("Creating project directory structure...")
        
        directories = [
            "models",
            "data/fer2013",
            "data/sample",
            "database",
            "scripts",
            "static",
            "templates",
            "logs",
            "tests",
            "docs",
            ".streamlit"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        
        # Create .gitkeep files for empty directories
        gitkeep_dirs = ["models", "database", "logs", "tests", "docs"]
        for directory in gitkeep_dirs:
            gitkeep_path = self.project_root / directory / ".gitkeep"
            gitkeep_path.touch()
    
    def create_gitignore(self):
        """Create comprehensive .gitignore file"""
        logger.info("Creating .gitignore file...")
        
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
models/*.h5
models/*.pkl
models/*.joblib
data/fer2013/fer2013.csv
data/fer2013/*.zip
database/*.db
database/*.sqlite
logs/*.log
*.log

# Jupyter
.ipynb_checkpoints/

# Streamlit
.streamlit/secrets.toml

# Environment variables
.env
.env.local
.env.production

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Model checkpoints
checkpoints/
saved_models/

# Data files
*.csv
*.json
*.pkl
*.pickle
*.h5
*.hdf5
*.npy
*.npz

# Images (except sample images)
*.jpg
*.jpeg
*.png
*.bmp
*.tiff
*.gif
!sample_images/
!static/images/

# Videos
*.mp4
*.avi
*.mov
*.mkv

# Archives
*.zip
*.tar.gz
*.rar
*.7z
"""
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content.strip())
        
        logger.info("✅ Created .gitignore file")
    
    def create_environment_file(self):
        """Create environment configuration file"""
        logger.info("Creating environment configuration...")
        
        env_content = """# Emotion Recognition System Environment Configuration

# Model Configuration
MODEL_PATH=models/emotion_model.h5
DETECTION_METHOD=mediapipe
INPUT_SIZE=48

# Database Configuration
DATABASE_PATH=database/emotion_detections.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=True

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/emotion_recognition.log

# Data Configuration
DATA_DIR=data/fer2013
SAMPLE_DATA_DIR=data/sample

# Training Configuration
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.001
"""
        
        env_path = self.project_root / ".env.example"
        with open(env_path, 'w') as f:
            f.write(env_content.strip())
        
        logger.info("✅ Created .env.example file")
    
    def create_streamlit_config(self):
        """Create Streamlit configuration"""
        logger.info("Creating Streamlit configuration...")
        
        streamlit_config = """[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
        
        config_path = self.project_root / ".streamlit" / "config.toml"
        with open(config_path, 'w') as f:
            f.write(streamlit_config.strip())
        
        logger.info("✅ Created Streamlit configuration")
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("Installing dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install requirements
            requirements_path = self.project_root / "requirements.txt"
            if requirements_path.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], 
                             check=True, capture_output=True)
                logger.info("✅ Dependencies installed successfully")
            else:
                logger.warning("requirements.txt not found")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
        
        return True
    
    def create_sample_data(self):
        """Create sample data for testing"""
        logger.info("Creating sample data...")
        
        try:
            import numpy as np
            import pandas as pd
            
            # Create sample FER-2013 data
            sample_data = {
                'pixels': [
                    ' '.join(['0'] * 2304),  # 48x48 = 2304 pixels
                    ' '.join(['255'] * 2304),
                    ' '.join(['128'] * 2304),
                    ' '.join(['64'] * 2304),
                    ' '.join(['192'] * 2304),
                ],
                'emotion': [0, 1, 2, 3, 4],  # Angry, Disgust, Fear, Happy, Sad
                'Usage': ['Training', 'Training', 'Training', 'Training', 'Training']
            }
            
            df = pd.DataFrame(sample_data)
            sample_path = self.project_root / "data" / "sample" / "sample_fer2013.csv"
            df.to_csv(sample_path, index=False)
            
            logger.info("✅ Sample data created")
            
        except ImportError:
            logger.warning("Required packages not available for sample data creation")
    
    def create_launch_scripts(self):
        """Create convenient launch scripts"""
        logger.info("Creating launch scripts...")
        
        # Windows batch script
        if platform.system() == "Windows":
            batch_content = """@echo off
echo Starting Emotion Recognition System...
echo.
echo Choose an option:
echo 1. Streamlit Web Interface
echo 2. Flask API Server
echo 3. Command Line Interface
echo 4. Train Model
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting Streamlit...
    streamlit run app.py
) else if "%choice%"=="2" (
    echo Starting Flask API...
    python api.py
) else if "%choice%"=="3" (
    echo Starting CLI...
    python emotion_detector.py --mode webcam
) else if "%choice%"=="4" (
    echo Starting model training...
    python train_model.py
) else (
    echo Invalid choice
)
pause
"""
            batch_path = self.project_root / "run.bat"
            with open(batch_path, 'w') as f:
                f.write(batch_content)
        
        # Unix shell script
        shell_content = """#!/bin/bash
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
"""
        shell_path = self.project_root / "run.sh"
        with open(shell_path, 'w') as f:
            f.write(shell_content)
        
        # Make shell script executable
        if platform.system() != "Windows":
            os.chmod(shell_path, 0o755)
        
        logger.info("✅ Launch scripts created")
    
    def create_license(self):
        """Create MIT license file"""
        logger.info("Creating LICENSE file...")
        
        license_content = """MIT License

Copyright (c) 2024 Emotion Recognition System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        
        license_path = self.project_root / "LICENSE"
        with open(license_path, 'w') as f:
            f.write(license_content.strip())
        
        logger.info("✅ LICENSE file created")
    
    def run_setup(self):
        """Run the complete setup process"""
        logger.info("🚀 Starting Emotion Recognition System Setup")
        logger.info("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create directory structure
        self.create_directory_structure()
        
        # Create configuration files
        self.create_gitignore()
        self.create_environment_file()
        self.create_streamlit_config()
        
        # Install dependencies
        if not self.install_dependencies():
            logger.warning("Some dependencies may not have installed correctly")
        
        # Create sample data
        self.create_sample_data()
        
        # Create launch scripts
        self.create_launch_scripts()
        
        # Create license
        self.create_license()
        
        logger.info("=" * 50)
        logger.info("✅ Setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Download FER-2013 dataset: python setup.py --download-dataset")
        logger.info("2. Train model: python train_model.py")
        logger.info("3. Run web interface: streamlit run app.py")
        logger.info("4. Or use launch script: ./run.sh (Unix) or run.bat (Windows)")
        
        return True

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Emotion Recognition Project")
    parser.add_argument("--full-setup", action="store_true", 
                       help="Run complete project setup")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check if dependencies are installed")
    parser.add_argument("--create-structure", action="store_true",
                       help="Create project directory structure only")
    
    args = parser.parse_args()
    
    setup = ProjectSetup()
    
    if args.full_setup:
        setup.run_setup()
    elif args.check_deps:
        setup.install_dependencies()
    elif args.create_structure:
        setup.create_directory_structure()
    else:
        print("Use --help to see available options")
        print("For full setup, run: python setup.py --full-setup")

if __name__ == "__main__":
    main()