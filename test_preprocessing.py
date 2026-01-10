"""
Testing and Visualization Script for Preprocessing Pipeline
Tests face detection, augmentation, and data loading
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import torch

import config
from image_preprocessing import FaceDetector, ImagePreprocessor
from data_loader import create_data_loaders


def test_face_detection():
    """Test face detection on sample images"""
    print("\n" + "="*60)
    print("TESTING FACE DETECTION")
    print("="*60)
    
    detector = FaceDetector(method='haar')
    
    # Get sample images from each emotion
    sample_images = []
    for emotion in config.EMOTION_CLASSES:
        emotion_dir = config.TRAIN_DIR / emotion
        images = list(emotion_dir.glob('*'))[:2]  # 2 per emotion
        sample_images.extend(images)
    
    # Test detection
    results = []
    for img_path in sample_images:
        image = Image.open(img_path).convert('RGB')
        faces = detector.detect_faces(image)
        results.append((img_path, image, faces))
        print(f"{img_path.parent.name}/{img_path.name}: {len(faces)} face(s)")
    
    # Visualize
    n_samples = min(12, len(results))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (img_path, image, faces) in enumerate(results[:n_samples]):
        ax = axes[idx]
        
        # Draw image
        img_array = np.array(image)
        
        # Draw bounding boxes
        for (x, y, w, h) in faces:
            # Draw rectangle on image
            import cv2
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        ax.imshow(img_array)
        ax.set_title(f"{img_path.parent.name}\n{len(faces)} face(s)", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = config.RESULTS_DIR / 'face_detection_test.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def test_augmentation():
    """Test data augmentation"""
    print("\n" + "="*60)
    print("TESTING DATA AUGMENTATION")
    print("="*60)
    
    # Get a sample image
    sample_image = list((config.TRAIN_DIR / 'happy').glob('*'))[0]
    image = Image.open(sample_image).convert('RGB')
    
    # Create preprocessor with augmentation
    detector = FaceDetector(method='haar')
    preprocessor = ImagePreprocessor(
        face_detector=detector,
        img_size=config.IMG_SIZE,
        augment=True
    )
    
    # Generate augmented versions
    augmented_images = []
    
    # Original
    tensor, _ = preprocessor.preprocess(image, detect_face=True)
    augmented_images.append(tensor)
    
    # Multiple augmented versions
    for _ in range(11):
        tensor, _ = preprocessor.preprocess(image, detect_face=True)
        augmented_images.append(tensor)
    
    # Denormalize for visualization
    def denormalize(tensor):
        mean = torch.tensor(config.NORMALIZATION_MEAN).view(3, 1, 1)
        std = torch.tensor(config.NORMALIZATION_STD).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy()
    
    # Visualize
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, tensor in enumerate(augmented_images):
        ax = axes[idx]
        img_array = denormalize(tensor)
        ax.imshow(img_array)
        if idx == 0:
            ax.set_title('Original (Preprocessed)', fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'Augmented {idx}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f'Data Augmentation Test - {sample_image.parent.name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / 'augmentation_test.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def test_data_loader():
    """Test data loader"""
    print("\n" + "="*60)
    print("TESTING DATA LOADER")
    print("="*60)
    
    # Create data loaders
    dataloaders, class_weights = create_data_loaders(
        train_dir=config.TRAIN_DIR,
        test_dir=config.TEST_DIR,
        batch_size=16,
        num_workers=0,
        use_face_detection=True,
        augment_train=True
    )
    
    # Get a batch
    train_loader = dataloaders['train']
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Emotions in batch: {[config.IDX_TO_CLASS[l.item()] for l in labels]}")
    
    # Visualize batch
    def denormalize(tensor):
        mean = torch.tensor(config.NORMALIZATION_MEAN).view(3, 1, 1)
        std = torch.tensor(config.NORMALIZATION_STD).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx in range(min(16, len(images))):
        ax = axes[idx]
        img = denormalize(images[idx])
        emotion = config.IDX_TO_CLASS[labels[idx].item()]
        
        ax.imshow(img)
        ax.set_title(f'{emotion}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Training Batch', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / 'dataloader_test.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def visualize_preprocessing():
    """Comprehensive preprocessing visualization"""
    print("\n" + "="*60)
    print("COMPREHENSIVE PREPROCESSING VISUALIZATION")
    print("="*60)
    
    # Run all tests
    test_face_detection()
    test_augmentation()
    test_data_loader()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("="*60)


if __name__ == '__main__':
    # Create results directory
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run visualization
    visualize_preprocessing()
