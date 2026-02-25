"""
Configuration for Text Sentiment Analysis Module
Contains all hyperparameters, paths, and settings for text processing
"""

import os
from pathlib import Path
import torch

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
TEXT_DATA_DIR = BASE_DIR / 'text_dataset'
TRAIN_TEXT_FILE = TEXT_DATA_DIR / 'train.csv'
VAL_TEXT_FILE = TEXT_DATA_DIR / 'val.csv'
TEST_TEXT_FILE = TEXT_DATA_DIR / 'test.csv'

# Create directories if they don't exist
TEXT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Output directories (shared with main config)
OUTPUT_DIR = BASE_DIR / 'output'
TEXT_MODELS_DIR = OUTPUT_DIR / 'text_models'
TEXT_LOGS_DIR = OUTPUT_DIR / 'text_logs'
TEXT_RESULTS_DIR = OUTPUT_DIR / 'text_results'

for directory in [TEXT_MODELS_DIR, TEXT_LOGS_DIR, TEXT_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SENTIMENT CLASSES
# ============================================================================
SENTIMENT_CLASSES = ['negative', 'positive']  # Binary sentiment for IMDb
NUM_SENTIMENT_CLASSES = len(SENTIMENT_CLASSES)
SENTIMENT_TO_IDX = {cls: idx for idx, cls in enumerate(SENTIMENT_CLASSES)}
IDX_TO_SENTIMENT = {idx: cls for cls, idx in SENTIMENT_TO_IDX.items()}

# For fusion with emotion classes
# Map sentiment to closest emotion equivalent
SENTIMENT_TO_EMOTION_MAP = {
    'positive': ['happy', 'surprise'],
    'negative': ['angry', 'sad', 'fear', 'disgust'],
    'neutral': ['neutral']
}

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
# Vocabulary settings
MAX_VOCAB_SIZE = 20000  # Maximum vocabulary size
MIN_WORD_FREQ = 2       # Minimum word frequency to include in vocab
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

# Sequence settings
MAX_SEQ_LENGTH = 500    # Maximum sequence length (500 recommended for sentiment analysis)
TRUNCATE_METHOD = 'post'  # 'pre' or 'post'
PADDING_METHOD = 'post'   # 'pre' or 'post'

# Text cleaning options
LOWERCASE = True
REMOVE_URLS = True
REMOVE_MENTIONS = True
REMOVE_HASHTAGS = False
REMOVE_NUMBERS = False
REMOVE_PUNCTUATION = False  # Keep for better context
REMOVE_STOPWORDS = False    # Stopwords can be important for sentiment
USE_LEMMATIZATION = False   # Set to True if using spacy

# ============================================================================
# WORD EMBEDDINGS
# ============================================================================
EMBEDDING_TYPE = 'glove'  # Options: 'trainable', 'glove', 'word2vec' | GloVe recommended for 94%+ accuracy
EMBEDDING_DIM = 300          # Dimension of word embeddings
PRETRAINED_EMBEDDINGS_PATH = None  # Path to pretrained embeddings if using

# GloVe settings (if using pretrained)
GLOVE_DIM = 300  # 50, 100, 200, or 300
GLOVE_PATH = BASE_DIR / 'embeddings' / f'glove.6B.{GLOVE_DIM}d.txt'

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# LSTM settings (Bidirectional LSTM recommended for 94%+ accuracy)
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_BIDIRECTIONAL = True  # Bidirectional for better context understanding
LSTM_DROPOUT = 0.3

# Attention settings (Attention mechanism boosts nuance capture)
USE_ATTENTION = True  # Enable attention for improved performance
ATTENTION_DIM = 128

# Classifier settings
CLASSIFIER_HIDDEN_DIMS = [256, 128]  # Hidden layer dimensions
CLASSIFIER_DROPOUT = 0.5

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TEXT_BATCH_SIZE = 64
TEXT_NUM_EPOCHS = 30
TEXT_LEARNING_RATE = 0.001
TEXT_WEIGHT_DECAY = 1e-5

# Optimizer
TEXT_OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
TEXT_MOMENTUM = 0.9      # For SGD

# Learning rate scheduler
TEXT_LR_SCHEDULER = 'cosine'  # Options: 'step', 'cosine', 'plateau'
TEXT_LR_STEP_SIZE = 10
TEXT_LR_GAMMA = 0.1
TEXT_LR_MIN = 1e-6

# Early stopping
TEXT_EARLY_STOPPING_PATIENCE = 7
TEXT_EARLY_STOPPING_MIN_DELTA = 0.001

# Gradient clipping
TEXT_GRAD_CLIP = 5.0

# Mixed precision training
TEXT_USE_AMP = True

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
TEXT_AUGMENTATION = True
AUGMENTATION_TECHNIQUES = {
    'synonym_replacement': True,  # Replace words with synonyms
    'random_insertion': True,     # Insert random synonyms
    'random_swap': True,          # Swap word positions
    'random_deletion': False,     # Delete random words
}

# Augmentation parameters
AUG_PROB = 0.1  # Probability of applying augmentation per word
AUG_NUM_CHANGES = 1  # Number of augmentations per sentence

# ============================================================================
# DATA LOADING
# ============================================================================
TEXT_NUM_WORKERS = 4
TEXT_PIN_MEMORY = True
TEXT_SHUFFLE_TRAIN = True

# Dataset split ratios (if creating from single file)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================================
# CLASS WEIGHTS
# ============================================================================
TEXT_USE_CLASS_WEIGHTS = True  # Handle imbalanced sentiment data

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================
TEXT_CONFIDENCE_THRESHOLD = 0.6
TEXT_BATCH_INFERENCE = True  # Process multiple texts at once

# ============================================================================
# LOGGING AND CHECKPOINTING
# ============================================================================
TEXT_USE_TENSORBOARD = True
TEXT_SAVE_BEST_ONLY = True
TEXT_CHECKPOINT_METRIC = 'val_accuracy'
TEXT_CHECKPOINT_MODE = 'max'
TEXT_LOG_INTERVAL = 50

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
TEXT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# RANDOM SEED
# ============================================================================
TEXT_RANDOM_SEED = 42

# ============================================================================
# DATASET DOWNLOAD SETTINGS
# ============================================================================
DATASET_NAME = 'imdb'  # Options: 'imdb', 'twitter', 'amazon'
AUTO_DOWNLOAD = True
IMDB_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
