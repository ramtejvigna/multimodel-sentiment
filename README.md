# Multimodal Sentiment Analysis System

A comprehensive sentiment analysis system that combines **text sentiment analysis** and **facial emotion recognition** using deep learning. The system provides multimodal fusion for enhanced accuracy by analyzing both textual content and facial expressions.

## ⚡ **NEW: Optimal Configurations Applied!**

This system is now configured with **research-backed optimal settings** for maximum accuracy and real-time performance:

- **✓ Face Detection**: MTCNN (100% accuracy vs Haar's 92.5%)
- **✓ Face Emotion Model**: MobileNetV2 (92% accuracy, 2.26M params, <100ms inference)
- **✓ Text Sentiment**: BiLSTM + GloVe + Attention (>94% accuracy)
- **✓ Fusion Strategy**: Adaptive late fusion with confidence scores

**📖 See [OPTIMAL_CONFIG.md](OPTIMAL_CONFIG.md) for detailed benchmarks, training commands, and performance tips!**

## 🌟 Features

### Facial Emotion Recognition

- **Face Detection**: MTCNN (default, most accurate) and Haar Cascade (fast alternative) support
- **Multiple Model Architectures**: **MobileNetV2 (default)**, ResNet18, ResNet34, EfficientNet, Custom CNN
- **Two Training Modes**: Train from scratch or fine-tune pretrained models (finetune recommended)
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Real-time Webcam Detection**: Live emotion detection with temporal smoothing (<100ms inference)

### Text Sentiment Analysis

- **LSTM-based Model**: Bidirectional LSTM with attention mechanism
- **Word Embeddings**: Pretrained GloVe embeddings (default) or trainable embeddings
- **Text Preprocessing**: Advanced cleaning, tokenization, and normalization
- **Binary Sentiment**: Positive/Negative classification
- **High Accuracy**: >94% accuracy with GloVe + BiLSTM + Attention

### Multimodal Fusion

- **Adaptive Fusion** (default): Learnable fusion weights with confidence-based weighting
- **Late Fusion**: Weighted combination of text and face predictions
- **Feature Fusion**: Combine embeddings before classification
- **Intelligent Conflict Resolution**: Handle disagreements between modalities
- **Confidence Scoring**: Unified confidence metrics

### Web Application

- **Modern UI**: Beautiful dark-themed interface with real-time updates
- **Text Input**: Analyze sentiment from text
- **Webcam Integration**: Live facial emotion detection
- **Multimodal Analysis**: Combined text + face analysis
- **Visual Results**: Interactive charts and confidence bars
- **REST API**: Easy integration with other applications

## Installation

1. Clone the repository or navigate to the project directory

2. Create a virtual environment (recommended):

```bash
python -m venv myvenv
source myvenv/bin/activate  # Linux/Mac
# or
myvenv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Place your dataset in the following structure:

```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── [same structure]
```

## Usage

### Quick Start (Recommended)

**Train face emotion model with optimal settings:**

```bash
python train.py --model mobilenet_v2 --mode finetune --epochs 30 --no-augmentation
```

**Train text sentiment model with GloVe embeddings:**

```bash
# First, download GloVe embeddings
mkdir -p embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip -P embeddings/
unzip embeddings/glove.6B.zip -d embeddings/

# Train on your sentiment dataset
python train_text.py --data final_sentiment_data.csv --data-format sentiment_csv --embedding-type glove --epochs 30
```

**📖 For complete training commands and optimization tips, see [OPTIMAL_CONFIG.md](OPTIMAL_CONFIG.md)**

### 1. Test Preprocessing

Test face detection and data augmentation:

```bash
python test_preprocessing.py
```

### 2. Train Face Emotion Model

**Recommended (Fine-tune MobileNetV2):**

```bash
python train.py --model mobilenet_v2 --mode finetune --epochs 30
```

**Train from scratch:**

```bash
python train.py --model mobilenet_v2 --mode scratch --epochs 50
```

**Available models:** `mobilenet_v2` (default), `resnet18`, `resnet34`, `efficientnet_b0`, `custom_cnn`

**Additional options:**

- `--batch_size 32` - Set batch size
- `--lr 0.001` - Set learning rate
- `--no-face-detection` - Disable face detection preprocessing
- `--no-augmentation` - Disable data augmentation

### 3. Inference

**Single image:**

```bash
python predict.py --model output/models/best_model.pth --image test.jpg --visualize
```

**Batch processing:**

```bash
python predict.py --model output/models/best_model.pth --folder dataset/test/happy --visualize --save-json
```

### 4. Real-time Webcam Demo

```bash
python demo.py --model output/models/best_model.pth
```

**Controls:**

- `q` - Quit
- `s` - Take screenshot
- `r` - Reset temporal smoothing buffer

## Configuration

Edit `config.py` to customize:

- Image size (default: 224x224)
- Batch size, learning rate, epochs
- Augmentation parameters
- Face detection method
- And more...

## Project Structure

```
├── config.py                    # Face model configuration
├── text_config.py               # Text model configuration
├── image_preprocessing.py       # Face detection and preprocessing
├── text_preprocessing.py        # Text cleaning and tokenization
├── data_loader.py               # PyTorch dataset for face data
├── text_data_loader.py          # PyTorch dataset for text data
├── model.py                     # Face emotion model architectures
├── text_model.py                # Text sentiment model (LSTM)
├── fusion_model.py              # Multimodal fusion architectures
├── train.py                     # Training script for face model
├── train_text.py                # Training script for text model
├── predict.py                   # Batch inference for face
├── demo.py                      # Real-time webcam face demo
├── multimodal_inference.py      # Unified multimodal pipeline
├── download_dataset.py          # IMDb dataset downloader
├── app.py                       # Flask web application
├── templates/                   # HTML templates
│   └── index.html              # Web interface
├── static/                      # CSS and JavaScript
│   ├── style.css               # Styling
│   └── app.js                  # Frontend logic
├── test_preprocessing.py        # Testing utilities
├── face_detection.py            # Standalone face detection
├── requirements.txt             # Dependencies
└── output/                      # Generated files
    ├── models/                     # Saved face models
    ├── text_models/                # Saved text models
    ├── logs/                       # TensorBoard logs
    └── results/                    # Metrics and visualizations
```

## Dataset Structure

**Face Emotion Dataset:**

```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── [same structure]
```

**Text Sentiment Dataset (IMDb):**

```
text_dataset/
└── aclImdb/
    ├── train/
    │   ├── pos/  # Positive reviews
    │   └── neg/  # Negative reviews
    └── test/
        ├── pos/
        └── neg/
```

## Training Output

Training produces:

- **Best model**: `output/models/best_model.pth`
- **Training curves**: `output/results/training_curves.png`
- **Confusion matrix**: `output/results/confusion_matrix.png`
- **Classification report**: `output/results/classification_report.txt`
- **TensorBoard logs**: `output/logs/`

View TensorBoard:

```bash
tensorboard --logdir output/logs
```

## Performance Tips

1. **GPU Training**: System automatically detects and uses GPU if available
2. **Mixed Precision**: Enabled by default for faster training (~2x speedup)
3. **Optimal Models**: 
   - **Face**: MobileNetV2 (default) - 92% accuracy, 2.26M params, <100ms inference
   - **Text**: BiLSTM + GloVe (default) - >94% accuracy
4. **Face Detection**: MTCNN (default) for 100% accuracy vs Haar's 92.5%
5. **Data Loading**: Adjust `NUM_WORKERS` in config based on your CPU
6. **Batch Sizes**: 32 (face), 64 (text) - optimal for most hardware

**📊 See [OPTIMAL_CONFIG.md](OPTIMAL_CONFIG.md) for detailed benchmarks and optimization strategies!**

## Troubleshooting

**No faces detected:**

- Ensure you're using MTCNN (default: `FACE_DETECTION_METHOD = 'mtcnn'`)
- Try adjusting `FACE_PADDING` in config
- Use Haar Cascade for ultra-fast detection (trade-off: lower accuracy)
- Ensure images contain visible faces

**Low accuracy:**

- Increase epochs (try 50+ for training from scratch)
- Use MobileNetV2 with finetune mode (recommended)
- Enable data augmentation (remove `--no-augmentation`)
- Ensure GloVe embeddings are downloaded for text model
- See [OPTIMAL_CONFIG.md](OPTIMAL_CONFIG.md) for detailed troubleshooting

**Out of memory:**

- Reduce batch size (try 16 or 8)
- Use smaller image size or max sequence length
- MobileNetV2 (default) is already optimized for low memory
- Enable gradient accumulation

## Documentation

- **[OPTIMAL_CONFIG.md](OPTIMAL_CONFIG.md)** - Research-backed optimal configurations, benchmarks, and training commands
- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup instructions

## Model Comparison

| Model | Accuracy | Parameters | Inference Speed | Best For |
|-------|----------|------------|-----------------|----------|
| **MobileNetV2** ⭐ | 92% | 2.26M | <100ms (GPU) | Real-time, webcam, web apps |
| EfficientNet-B0 | 83% | ~5M | ~150ms | Balanced |
| ResNet18 | Competitive | 11M | ~120ms | Baseline |
| ResNet34 | Higher | 21M | ~180ms | Offline training |

**⭐ = Default configuration**

## License

This project is for educational purposes.

## Acknowledgments

- PyTorch and torchvision for deep learning frameworks
- OpenCV for face detection
- The emotion dataset providers
