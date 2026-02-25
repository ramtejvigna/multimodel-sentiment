# Optimal Configurations for Multimodal Sentiment Analysis

This document provides the optimal configurations and training commands based on benchmarks for facial emotion recognition and sentiment analysis tasks.

## Summary of Optimizations

Your system has been configured with the following optimal settings:

### 1. Face Detection: MTCNN ✓
- **Accuracy**: 100% vs Haar Cascade's 92.5% on challenging poses/occlusions
- **Configured in**: `config.py` → `FACE_DETECTION_METHOD = 'mtcnn'`
- **Use Case**: Production-ready for robust preprocessing
- **Fallback**: Haar Cascade available for ultra-fast prototyping

### 2. Facial Emotion Model: MobileNetV2 ✓
- **Test Accuracy**: 92% on emotion benchmarks
- **Parameters**: 2.26M (lightweight)
- **Speed**: Real-time inference (<100ms with GPU)
- **Configured in**: `config.py` → `DEFAULT_MODEL = 'mobilenet_v2'`
- **Training Mode**: `finetune` for transfer learning (30 epochs recommended)

### 3. Text Sentiment Model: BiLSTM + GloVe + Attention ✓
- **Accuracy**: >94% on IMDb
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Embeddings**: GloVe 300d (pretrained)
- **Max Sequence**: 500 tokens
- **Configured in**: `text_config.py`

### 4. Fusion Strategy: Adaptive Late Fusion with Confidence Scores ✓
- **Method**: Learnable weights + confidence-based attention
- **Benefits**: Handles modality conflicts, avoids overfitting
- **Configured in**: `fusion_model.py` → `AdaptiveFusionModel`

---

## Model Comparison Table

| Model | Test Accuracy | Params (M) | Best For |
|-------|---------------|------------|----------|
| **MobileNetV2** ⭐ | 92% | 2.26 | Real-time webcams, web apps |
| EfficientNet-B0 | 83% | ~5 | Balanced accuracy/speed |
| ResNet18 | Competitive | 11 | Finetuning baseline |
| ResNet34 | Higher but slower | 21 | Offline training |

**⭐ = Currently configured default**

---

## Optimal Training Commands

### 1. Train Facial Emotion Recognition (MobileNetV2)

**Initial Fine-tuning (Recommended Start):**
```bash
# Fine-tune pretrained MobileNetV2 without augmentation first
python train.py \
  --model mobilenet_v2 \
  --mode finetune \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.001 \
  --no-augmentation
```

**Advanced Fine-tuning (If accuracy needs improvement):**
```bash
# Add augmentation and increase epochs
python train.py \
  --model mobilenet_v2 \
  --mode finetune \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --augmentation
```

**Train from Scratch (Only if dataset is large >100k samples):**
```bash
python train.py \
  --model mobilenet_v2 \
  --mode scratch \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --augmentation
```

**Alternative Models (for comparison):**
```bash
# EfficientNet-B0 (if you want to compare)
python train.py --model efficientnet_b0 --mode finetune --epochs 30

# ResNet18 (baseline comparison)
python train.py --model resnet18 --mode finetune --epochs 30
```

### 2. Train Text Sentiment Analysis (BiLSTM + GloVe)

**Using final_sentiment_data.csv:**
```bash
# Train on your sentiment dataset with GloVe embeddings
python train_text.py \
  --data final_sentiment_data.csv \
  --data-format sentiment_csv \
  --embedding-type glove \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001 \
  --max-seq-length 500
```

**Download GloVe embeddings first (if not already present):**
```bash
# Create embeddings directory
mkdir -p embeddings

# Download GloVe 6B 300d (840MB)
wget http://nlp.stanford.edu/data/glove.6B.zip -P embeddings/
unzip embeddings/glove.6B.zip -d embeddings/
```

**Using IMDb dataset (for comparison):**
```bash
python train_text.py \
  --data-format imdb \
  --embedding-type glove \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001
```

### 3. Run Multimodal Inference

**Single Image + Text:**
```bash
python multimodal_inference.py \
  --image path/to/image.jpg \
  --text "This is a great product!" \
  --face-model output/models/best_mobilenetv2.pth \
  --text-model output/text_models/best_text_model.pth \
  --fusion adaptive
```

### 4. Run Webcam Demo

**Optimal Configuration for Real-Time Performance:**
```bash
python demo.py \
  --model output/models/best_mobilenetv2.pth \
  --face-detection mtcnn \
  --confidence-threshold 0.5 \
  --temporal-smoothing
```

**For Web Application:**
```bash
python app.py \
  --model output/models/best_mobilenetv2.pth \
  --text-model output/text_models/best_text_model.pth \
  --port 5000
```

---

## Performance Optimization Tips

### 1. Enable Mixed Precision Training (AMP)
Already configured in both `config.py` and `text_config.py`:
- `USE_AMP = True` (Face model)
- `TEXT_USE_AMP = True` (Text model)
- Speeds up training by ~2x on modern GPUs

### 2. GPU Acceleration
```bash
# Check GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# For multi-GPU training (if available)
CUDA_VISIBLE_DEVICES=0,1 python train.py --model mobilenet_v2 --mode finetune
```

### 3. Batch Size Optimization
- **GPU (8GB VRAM)**: batch_size=32 (face), batch_size=64 (text)
- **GPU (4GB VRAM)**: batch_size=16 (face), batch_size=32 (text)
- **CPU**: batch_size=8 (both), expect slower training

### 4. Temporal Smoothing for Video
- **webcam demo**: Use `--temporal-smoothing` for stable predictions
- **smoothing window**: 5 frames (configurable in `config.py`)

---

## Troubleshooting

### Low Accuracy on Face Emotion Model
1. **Increase epochs**: 50-100 instead of 30
2. **Enable augmentation**: Remove `--no-augmentation` flag
3. **Try different model**: ResNet34 for better accuracy (slower)
4. **Check dataset balance**: Ensure all 7 emotion classes are well-represented

### Low Accuracy on Text Sentiment Model
1. **Verify GloVe embeddings**: Ensure embeddings are downloaded
2. **Increase max sequence length**: Try 500-1000 for longer texts
3. **Check data quality**: Clean text preprocessing is crucial
4. **Experiment with attention**: Already enabled in `text_config.py`

### Slow Inference Speed
1. **Use MobileNetV2**: Already configured (fastest model)
2. **Enable CUDA**: Ensure GPU is being used
3. **Reduce batch size**: For real-time applications
4. **Use Haar Cascade**: Switch to `FACE_DETECTION_METHOD = 'haar'` for 2x faster face detection (trade-off: accuracy)

### Out of Memory (OOM) Errors
1. **Reduce batch size**: Lower `BATCH_SIZE` in configs
2. **Reduce sequence length**: Lower `MAX_SEQ_LENGTH` for text
3. **Use gradient accumulation**: Split batches across multiple steps
4. **Close other applications**: Free up GPU memory

---

## Expected Performance Benchmarks

### Training Time (on typical GPU like RTX 3060)
- **MobileNetV2 (30 epochs)**: ~2-3 hours on FER-2013 (35k images)
- **BiLSTM Text (30 epochs)**: ~1-2 hours on IMDb (25k reviews)
- **ResNet18 (30 epochs)**: ~3-4 hours

### Inference Speed (single sample)
- **MobileNetV2 + MTCNN (GPU)**: <100ms
- **MobileNetV2 + MTCNN (CPU)**: ~300-500ms
- **BiLSTM Text (GPU)**: <50ms
- **BiLSTM Text (CPU)**: ~100ms
- **Multimodal Fusion**: +10-20ms

### Expected Accuracy
- **Face Emotion (MobileNetV2)**: 85-92% on test set
- **Text Sentiment (BiLSTM + GloVe)**: 92-94% on IMDb
- **Multimodal Fusion**: Depends on task, typically 88-95%

---

## Configuration Files Reference

All optimal settings have been applied to:

1. **`config.py`**:
   - `FACE_DETECTION_METHOD = 'mtcnn'`
   - `DEFAULT_MODEL = 'mobilenet_v2'`
   - `TRAINING_MODE = 'finetune'`
   - `BATCH_SIZE = 32`
   - `NUM_EPOCHS = 30`
   - `LEARNING_RATE = 0.001`
   - `USE_AMP = True`

2. **`text_config.py`**:
   - `EMBEDDING_TYPE = 'glove'`
   - `MAX_SEQ_LENGTH = 500`
   - `LSTM_BIDIRECTIONAL = True`
   - `USE_ATTENTION = True`
   - `TEXT_BATCH_SIZE = 64`
   - `TEXT_NUM_EPOCHS = 30`
   - `TEXT_USE_AMP = True`

3. **`fusion_model.py`**:
   - `AdaptiveFusionModel` with confidence-based weighting
   - Learnable fusion weights
   - Handles modality conflicts

4. **`text_data_loader.py`**:
   - Added `load_sentiment_csv_data()` for `final_sentiment_data.csv`
   - Supports `data_format='sentiment_csv'`

---

## Quick Start Workflow

### Step 1: Train Face Emotion Model
```bash
python train.py --model mobilenet_v2 --mode finetune --epochs 30 --no-augmentation
```

### Step 2: Download GloVe Embeddings
```bash
mkdir -p embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip -P embeddings/
unzip embeddings/glove.6B.zip -d embeddings/
```

### Step 3: Train Text Sentiment Model
```bash
python train_text.py \
  --data final_sentiment_data.csv \
  --data-format sentiment_csv \
  --embedding-type glove \
  --epochs 30
```

### Step 4: Test Webcam Demo
```bash
python demo.py \
  --model output/models/best_model.pth \
  --face-detection mtcnn
```

### Step 5: Run Web Application
```bash
python app.py --port 5000
```

---

## References

1. **MTCNN vs Haar Cascade**: [hrcak.srce](https://hrcak.srce.hr/en/330887)
2. **MobileNetV2 for FER**: [jurnal.polgan.ac](https://jurnal.polgan.ac.id/index.php/sinkron/article/view/15370)
3. **Lightweight CNN Benchmarks**: [pure.ulster.ac](https://pure.ulster.ac.uk/en/publications/lightweight-cnn-benchmarks-for-facial-emotion-recognition-in-pepp/)
4. **BiLSTM Sentiment Analysis**: [github.com/pranavphoenix](https://github.com/pranavphoenix/LSTM-sentiment-analysis)
5. **Multimodal Fusion Strategies**: Research on adaptive late fusion with confidence scores

---

## Notes

- All configurations are optimized for **real-time performance** on typical hardware (Fedora setups)
- **MobileNetV2** balances accuracy (92%) with speed (2.26M params)
- **MTCNN** provides robust face detection despite slight speed trade-offs
- **GloVe embeddings** with **BiLSTM + Attention** achieve >94% on sentiment tasks
- **Adaptive late fusion** handles modality conflicts better than early fusion
- Use **final_sentiment_data.csv** for textual analysis as requested

For additional help, consult the main [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).
