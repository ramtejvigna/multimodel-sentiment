# Face Detection + Emotion Recognition System

A complete facial emotion recognition system that combines face detection and emotion classification using deep learning. The system can classify 7 emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Features

- **Face Detection**: Haar Cascade (fast) and MTCNN (accurate) support
- **Multiple Model Architectures**: ResNet18, ResNet34, MobileNetV2, EfficientNet, Custom CNN
- **Two Training Modes**:
  - Train from scratch on your dataset
  - Fine-tune pretrained models
- **Data Augmentation**: Rotation, flips, brightness adjustment, etc.
- **Real-time Webcam Demo**: Live emotion detection with temporal smoothing
- **Batch Processing**: Process folders of images
- **Comprehensive Metrics**: Confusion matrix, F1-score, classification report

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

### 1. Test Preprocessing

Test face detection and data augmentation:
```bash
python test_preprocessing.py
```

### 2. Train Model

**Train from scratch:**
```bash
python train.py --model resnet18 --mode scratch --epochs 50
```

**Fine-tune pretrained model:**
```bash
python train.py --model resnet18 --mode finetune --epochs 30
```

**Available models:** `resnet18`, `resnet34`, `mobilenet_v2`, `efficientnet_b0`, `custom_cnn`

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
├── config.py                 # Configuration settings
├── image_preprocessing.py    # Face detection and preprocessing
├── data_loader.py           # PyTorch dataset and dataloader
├── model.py                 # Model architectures
├── train.py                 # Training script
├── predict.py               # Batch inference
├── demo.py                  # Real-time webcam demo
├── test_preprocessing.py    # Testing utilities
├── face_detection.py        # Standalone face detection
├── requirements.txt         # Dependencies
└── output/                  # Generated files
    ├── models/              # Saved models
    ├── logs/                # TensorBoard logs
    └── results/             # Metrics and visualizations
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
2. **Mixed Precision**: Enabled by default for faster training (set `USE_AMP = False` in config to disable)
3. **Data Loading**: Adjust `NUM_WORKERS` in config based on your CPU
4. **Face Detection**: Use Haar Cascade for speed, MTCNN for accuracy

## Troubleshooting

**No faces detected:**
- Try adjusting `FACE_PADDING` in config
- Use MTCNN instead of Haar Cascade
- Ensure images contain visible faces

**Low accuracy:**
- Train for more epochs
- Try different model architectures
- Enable data augmentation
- Use fine-tuning mode

**Out of memory:**
- Reduce batch size
- Use smaller image size
- Use lighter model (MobileNetV2 or Custom CNN)

## License

This project is for educational purposes.

## Acknowledgments

- PyTorch and torchvision for deep learning frameworks
- OpenCV for face detection
- The emotion dataset providers
