"""
PyTorch Dataset and DataLoader for Facial Emotion Recognition
Handles data loading with integrated preprocessing
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import config
from image_preprocessing import FaceDetector, ImagePreprocessor


class EmotionDataset(Dataset):
    """Custom dataset for facial emotion recognition"""
    
    def __init__(self, root_dir, transform=None, face_detector=None, augment=False):
        """
        Args:
            root_dir: Root directory with emotion folders
            transform: Optional transform to be applied
            face_detector: FaceDetector instance for face detection
            augment: Whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.face_detector = face_detector
        self.augment = augment
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            face_detector=face_detector,
            img_size=config.IMG_SIZE,
            augment=augment
        )
        
        # Load dataset
        self.samples = []
        self.labels = []
        
        for emotion_idx, emotion in enumerate(config.EMOTION_CLASSES):
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            # Get all image files
            image_files = list(emotion_dir.glob('*.jpg')) + \
                         list(emotion_dir.glob('*.png')) + \
                         list(emotion_dir.glob('*.jpeg'))
            
            for img_path in image_files:
                self.samples.append(str(img_path))
                self.labels.append(emotion_idx)
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution"""
        from collections import Counter
        counts = Counter(self.labels)
        print("\nClass distribution:")
        for emotion_idx, count in sorted(counts.items()):
            emotion = config.IDX_TO_CLASS[emotion_idx]
            print(f"  {emotion}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Returns:
            tuple: (tensor, label)
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            # Preprocess image
            tensor, _ = self.preprocessor.preprocess(
                img_path,
                detect_face=self.face_detector is not None
            )
            
            return tensor, label
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid sample instead
            random_idx = np.random.randint(0, len(self))
            return self.__getitem__(random_idx)
    
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced dataset
        
        Returns:
            torch.Tensor: Class weights
        """
        from collections import Counter
        counts = Counter(self.labels)
        total = len(self.labels)
        
        weights = []
        for i in range(config.NUM_CLASSES):
            if i in counts:
                weights.append(total / (config.NUM_CLASSES * counts[i]))
            else:
                weights.append(0.0)
        
        return torch.FloatTensor(weights)


def create_data_loaders(train_dir, test_dir=None, val_split=None, 
                        batch_size=None, num_workers=None, 
                        use_face_detection=True, augment_train=True):
    """
    Create train, validation, and test data loaders
    
    Args:
        train_dir: Training data directory
        test_dir: Test data directory (optional)
        val_split: Validation split ratio (if no test_dir provided)
        batch_size: Batch size (default from config)
        num_workers: Number of workers (default from config)
        use_face_detection: Whether to use face detection
        augment_train: Whether to augment training data
        
    Returns:
        dict: Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    val_split = val_split or config.VALIDATION_SPLIT
    
    # Initialize face detector if needed
    face_detector = None
    if use_face_detection:
        face_detector = FaceDetector(method=config.FACE_DETECTION_METHOD)
    
    # Create training dataset
    train_dataset = EmotionDataset(
        root_dir=train_dir,
        face_detector=face_detector,
        augment=augment_train
    )
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Create test loader if test directory is provided
    if test_dir:
        test_dataset = EmotionDataset(
            root_dir=test_dir,
            face_detector=face_detector,
            augment=False  # No augmentation for test
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY
        )
        
        dataloaders['test'] = test_loader
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_subset)}")
    print(f"  Validation: {len(val_subset)}")
    if test_dir:
        print(f"  Test: {len(test_dataset)}")
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    print(f"\nClass weights: {class_weights}")
    
    return dataloaders, class_weights


if __name__ == '__main__':
    # Test data loader
    print("Testing data loader...")
    
    dataloaders, class_weights = create_data_loaders(
        train_dir=config.TRAIN_DIR,
        test_dir=config.TEST_DIR,
        batch_size=4,
        num_workers=0  # Use 0 for testing
    )
    
    # Test train loader
    print("\nTesting train loader...")
    train_loader = dataloaders['train']
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels}")
        print(f"  Emotions: {[config.IDX_TO_CLASS[l.item()] for l in labels]}")
        
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("\nData loader test completed!")
