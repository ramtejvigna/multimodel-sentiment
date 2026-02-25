# Multimodal Sentiment Analysis - Quick Start Guide

## Overview

This guide will help you get started with the multimodal sentiment analysis system, from training models to running the web application.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works too)
- Webcam (for live facial emotion detection)

## Step-by-Step Setup

### 1. Installation

```bash
# Clone or navigate to project directory
cd multimodel-sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for text preprocessing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Prepare Face Emotion Dataset

Place your facial emotion dataset in the following structure:

```
dataset/
├── train/
│   ├── angry/, disgust/, fear/, happy/, neutral/, sad/, surprise/
└── test/
    └── [same structure]
```

**Recommended datasets:**

- FER-2013
- CK+
- RAF-DB

### 3. Train Face Emotion Model

```bash
# Train ResNet18 model (recommended)
python train.py \
  --model resnet18 \
  --mode finetune \
  --epochs 50 \
  --batch_size 32

# Training will save:
# - Best model: output/models/best_model.pth
# - Training curves: output/results/training_curves.png
# - Confusion matrix: output/results/confusion_matrix_*.png
```

**Expected timeline:** 2-4 hours on GPU, 10-20 hours on CPU

### 4. Download Text Sentiment Dataset

```bash
# Download IMDb dataset (50,000 reviews)
python download_dataset.py

# This downloads to: text_dataset/aclImdb/
```

### 5. Train Text Sentiment Model

```bash
# Train LSTM model
python train_text.py \
  --train-data text_dataset/aclImdb/train \
  --test-data text_dataset/aclImdb/test \
  --data-format imdb \
  --model lstm_attention \
  --epochs 30 \
  --batch-size 64

# Training will save:
# - Best model: output/text_models/best_text_model.pth
# - Vocabulary: output/text_models/vocabulary.pkl
# - Training curves: output/text_results/training_curves_text.png
```

**Expected timeline:** 1-2 hours on GPU, 5-10 hours on CPU

### 6. Test Individual Models

**Test face model:**

```bash
python predict.py \
  --model output/models/best_model.pth \
  --image path/to/test_image.jpg \
  --visualize
```

**Test text model (Python):**

```python
from multimodal_inference import create_multimodal_analyzer

analyzer = create_multimodal_analyzer()
result = analyzer.analyze_text("This is a great day!")
print(result)
```

### 7. Run Multimodal Analysis

```python
from multimodal_inference import create_multimodal_analyzer

# Initialize
analyzer = create_multimodal_analyzer(
    face_model_path='output/models/best_model.pth',
    text_model_path='output/text_models/best_text_model.pth',
    vocab_path='output/text_models/vocabulary.pkl'
)

# Analyze text + image
result = analyzer.analyze_multimodal(
    text="I absolutely loved this movie!",
    image="path/to/happy_face.jpg"
)

print(f"Prediction: {result['multimodal_prediction']}")
print(f"Confidence: {result['multimodal_confidence']:.2%}")
print(f"Text-Face Agreement: {result['agreement']}")
```

### 8. Launch Web Application

```bash
# Start Flask server
python app.py

# Open browser to: http://localhost:5000
```

**Using the web app:**

1. Enter text in the text area
2. Click "Start Webcam" to enable camera
3. Position your face and click "Capture & Analyze"
4. View individual results for text and face
5. Click "Analyze Both" for multimodal prediction

## Model Performance

### Expected Accuracies

| Model                   | Dataset  | Expected Accuracy |
| ----------------------- | -------- | ----------------- |
| Face Emotion (ResNet18) | FER-2013 | 78-82%            |
| Text Sentiment (LSTM)   | IMDb     | 85-88%            |
| **Multimodal Fusion**   | Combined | **88-92%**        |

### Why Multimodal is Better

The fusion approach improves accuracy because:

- **Complementary information**: Text provides semantic meaning, face shows emotional state
- **Conflict resolution**: When one modality is uncertain, the other helps
- **Real-world scenarios**: People express sentiment through both words and expressions

## Troubleshooting

### Models not initialized

```
Error: Model not initialized
Solution: Ensure both face and text models are trained and saved in output/ directories
```

### No faces detected

```
Error: No face detected
Solution:
- Ensure good lighting
- Face clearly visible to camera
- Try adjusting FACE_PADDING in config.py
- Use MTCNN instead of Haar Cascade
```

### Low accuracy

```
Problem: Model accuracy is poor
Solutions:
- Train for more epochs
- Use larger batch size if you have enough memory
- Try different model architectures
- Ensure dataset quality is good
- Enable data augmentation
```

### Out of memory

```
Error: CUDA out of memory
Solutions:
- Reduce batch_size (e.g., from 64 to 32 or 16)
- Use smaller image size in config.py
- Use lighter model (MobileNetV2)
- Enable gradient checkpointing
```

## Next Steps

1. **Experiment with fusion strategies**: Try adaptive fusion or feature fusion
2. **Fine-tune fusion weights**: Adjust text_weight and face_weight for your use case
3. **Add more modalities**: Integrate audio/speech for tri-modal analysis
4. **Deploy to production**: Containerize with Docker, deploy to cloud
5. **Create datasets**: Collect your own multimodal data for specific domains

## API Documentation

### REST API Endpoints

**POST /api/analyze/text**

- Input: `{"text": "your text here"}`
- Output: `{"prediction": "positive", "confidence": 0.92, "probabilities": {...}}`

**POST /api/analyze/face**

- Input: Form data with image file or base64 encoded image
- Output: `{"prediction": "happy", "confidence": 0.87, "probabilities": {...}}`

**POST /api/analyze/multimodal**

- Input: Form data with text and image
- Output: Combined prediction with individual analyses

**GET /api/status**

- Output: `{"status": "ready", "models_loaded": true}`

## Resources

- **Paper inspiration**: Multimodal Sentiment Analysis (MSA) research
- **Datasets**: FER-2013, IMDb, CK+, RAF-DB
- **Pretrained embeddings**: GloVe vectors (optional)

## Support

For issues and questions:

1. Check troubleshooting section above
2. Review configuration files (config.py, text_config.py)
3. Ensure all dependencies are installed
4. Verify dataset format matches expected structure

Happy analyzing! 🎭📊
