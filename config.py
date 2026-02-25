"""
Configuration file for Facial Emotion Recognition System
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'
TEST_DIR = DATASET_DIR / 'test'

# Output directories
OUTPUT_DIR = BASE_DIR / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
LOGS_DIR = OUTPUT_DIR / 'logs'
RESULTS_DIR = OUTPUT_DIR / 'results'

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
# Image size (for transfer learning models)
IMG_SIZE = 224
IMG_CHANNELS = 3  # RGB

# Face detection
FACE_DETECTION_METHOD = 'mtcnn'  # Options: 'haar', 'mtcnn' | MTCNN recommended for production (100% vs 92.5% accuracy)
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
FACE_PADDING = 0.2  # 20% padding around detected face
MIN_FACE_SIZE = (30, 30)  # Minimum face size to detect

# Normalization
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZATION_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Data Augmentation (for training)
AUGMENTATION = {
    'horizontal_flip': True,
    'rotation_range': 15,  # degrees
    'brightness_range': (0.8, 1.2),
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Available model architectures
AVAILABLE_MODELS = ['resnet18', 'resnet34', 'mobilenet_v2', 'efficientnet_b0']
DEFAULT_MODEL = 'mobilenet_v2'  # MobileNetV2: 92% accuracy, 2.26M params, optimal for real-time

# Training mode
TRAINING_MODE = 'finetune'  # Options: 'scratch', 'finetune' | Finetune recommended for transfer learning

# Model-specific configurations
MODEL_CONFIG = {
    'resnet18': {
        'pretrained': True,
        'feature_dim': 512,
    },
    'resnet34': {
        'pretrained': True,
        'feature_dim': 512,
    },
    'mobilenet_v2': {
        'pretrained': True,
        'feature_dim': 1280,
    },
    'efficientnet_b0': {
        'pretrained': True,
        'feature_dim': 1280,
    },
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Training hyperparameters (Optimized for MobileNetV2)
BATCH_SIZE = 32  # Optimal batch size for speed/accuracy balance
NUM_EPOCHS = 30  # 30 epochs recommended for finetuning, increase if training from scratch
LEARNING_RATE = 0.001  # Optimal learning rate for Adam optimizer
WEIGHT_DECAY = 1e-4

# Learning rate scheduler
LR_SCHEDULER = 'cosine'  # Options: 'step', 'cosine', 'plateau'
LR_STEP_SIZE = 10
LR_GAMMA = 0.1
LR_MIN = 1e-6

# Optimizer
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd'
MOMENTUM = 0.9  # For SGD

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# Data loading
NUM_WORKERS = 4
PIN_MEMORY = True

# Class weights (for handling imbalanced data)
USE_CLASS_WEIGHTS = True

# Mixed precision training
USE_AMP = True  # Automatic Mixed Precision

# Gradient clipping
GRAD_CLIP = 1.0

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================
VALIDATION_SPLIT = 0.2  # 20% of training data for validation

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.5

# Webcam settings
WEBCAM_ID = 0
WEBCAM_FPS = 30
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Temporal smoothing for video
TEMPORAL_SMOOTHING = True
SMOOTHING_WINDOW = 5  # Number of frames to average

# ============================================================================
# LOGGING AND CHECKPOINTING
# ============================================================================
# TensorBoard
USE_TENSORBOARD = True

# Model checkpointing
SAVE_BEST_ONLY = True
CHECKPOINT_METRIC = 'val_accuracy'  # Metric to monitor
CHECKPOINT_MODE = 'max'  # 'max' for accuracy, 'min' for loss

# Logging frequency
LOG_INTERVAL = 10  # Log every N batches

# ============================================================================
# PRETRAINED EMOTION MODEL
# ============================================================================
# URL or path to pretrained FER model (if available)
PRETRAINED_FER_MODEL_URL = None  # Will be populated if using pretrained model
PRETRAINED_FER_MODEL_PATH = None

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================
RANDOM_SEED = 42
