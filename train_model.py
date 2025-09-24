"""
Model Training Pipeline for Emotion Recognition
Trains a CNN model on the FER-2013 dataset with data augmentation
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import argparse
import logging
from typing import Tuple, Dict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionModelTrainer:
    """Trainer class for emotion recognition model"""
    
    def __init__(self, data_dir: str = "data/fer2013", 
                 model_dir: str = "models"):
        """
        Initialize the trainer
        
        Args:
            data_dir: Directory containing FER-2013 dataset
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Model parameters
        self.img_size = (48, 48)
        self.num_classes = len(self.emotion_labels)
        self.batch_size = 32
        self.epochs = 50
        
    def download_dataset(self):
        """Download FER-2013 dataset if not present"""
        csv_path = os.path.join(self.data_dir, "fer2013.csv")
        
        if not os.path.exists(csv_path):
            logger.info("Downloading FER-2013 dataset...")
            url = "https://www.kaggle.com/datasets/msambare/fer2013/download?datasetVersionNumber=1"
            
            # Note: This is a placeholder URL. In practice, you'd need to:
            # 1. Set up Kaggle API credentials
            # 2. Use kaggle datasets download command
            # 3. Or manually download and place the file
            
            logger.warning("Please download fer2013.csv from Kaggle and place it in the data directory")
            logger.info("Dataset URL: https://www.kaggle.com/datasets/msambare/fer2013")
            return False
        
        logger.info("FER-2013 dataset found")
        return True
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess FER-2013 dataset"""
        csv_path = os.path.join(self.data_dir, "fer2013.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Extract pixels and emotions
        pixels = df['pixels'].values
        emotions = df['emotion'].values
        
        # Convert pixels to images
        images = []
        for pixel_string in pixels:
            pixel_array = np.array(pixel_string.split(), dtype='uint8')
            image = pixel_array.reshape(48, 48)
            images.append(image)
        
        images = np.array(images)
        emotions = np.array(emotions)
        
        # Normalize images
        images = images.astype('float32') / 255.0
        
        # Add channel dimension
        images = np.expand_dims(images, axis=-1)
        
        # Convert emotions to categorical
        emotions_categorical = keras.utils.to_categorical(emotions, self.num_classes)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, emotions_categorical, test_size=0.2, random_state=42, stratify=emotions
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_model(self) -> keras.Model:
        """Create CNN model for emotion recognition"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_data_generators(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """Create data generators with augmentation"""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=self.batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=self.batch_size)
        
        return train_generator, val_generator
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> keras.Model:
        """Train the emotion recognition model"""
        
        # Create model
        model = self.create_model()
        
        # Print model summary
        model.summary()
        
        # Create data generators
        train_generator, val_generator = self.create_data_generators(
            X_train, y_train, X_val, y_val
        )
        
        # Define callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        logger.info("Starting model training...")
        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=val_generator,
            validation_steps=len(X_val) // self.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        model_path = os.path.join(self.model_dir, 'emotion_model.h5')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save training history
        self._save_training_history(history)
        
        return model, history
    
    def evaluate_model(self, model: keras.Model, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        report = classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        logger.info("Classification Report:")
        logger.info(classification_report(y_true_classes, y_pred_classes, 
                                        target_names=self.emotion_labels))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self._plot_confusion_matrix(cm)
        
        # Save evaluation results
        evaluation_results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        with open(os.path.join(self.model_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'))
        plt.close()
    
    def _save_training_history(self, history: keras.callbacks.History):
        """Save training history plots"""
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.close()
        
        # Save history data
        history_data = {
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        with open(os.path.join(self.model_dir, 'training_history.json'), 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def train(self):
        """Main training pipeline"""
        
        # Download dataset if needed
        if not self.download_dataset():
            logger.error("Could not download dataset. Please download manually.")
            return
        
        # Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
        
        # Train model
        model, history = self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, X_test, y_test)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test accuracy: {evaluation_results['test_accuracy']:.4f}")


def main():
    """Main function for command line training"""
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument("--data_dir", default="data/fer2013", 
                       help="Directory containing FER-2013 dataset")
    parser.add_argument("--model_dir", default="models", 
                       help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Training batch size")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = EmotionModelTrainer(args.data_dir, args.model_dir)
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch_size
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
