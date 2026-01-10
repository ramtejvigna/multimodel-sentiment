"""
Image Preprocessing Module for Facial Emotion Recognition
Handles face detection, alignment, normalization, and augmentation
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
import config


class FaceDetector:
    """Face detection using Haar Cascade or MTCNN"""
    
    def __init__(self, method='haar'):
        """
        Initialize face detector
        
        Args:
            method: 'haar' for Haar Cascade or 'mtcnn' for MTCNN
        """
        self.method = method
        
        if method == 'haar':
            # Load Haar Cascade
            cascade_path = cv2.data.haarcascades + config.HAAR_CASCADE_PATH
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise ValueError(f"Failed to load Haar Cascade from {cascade_path}")
        elif method == 'mtcnn':
            try:
                from facenet_pytorch import MTCNN
                self.detector = MTCNN(keep_all=True, device=config.DEVICE)
            except ImportError:
                raise ImportError("facenet-pytorch is required for MTCNN. Install with: pip install facenet-pytorch")
        else:
            raise ValueError(f"Unknown face detection method: {method}")
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: PIL Image or numpy array (RGB or BGR)
            
        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.method == 'haar':
            # Convert to grayscale for Haar Cascade
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=config.MIN_FACE_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces if len(faces) > 0 else []
        
        elif self.method == 'mtcnn':
            # MTCNN expects RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            boxes, _ = self.detector.detect(Image.fromarray(image))
            
            if boxes is not None:
                # Convert to (x, y, w, h) format
                faces = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
                return faces
            return []
    
    def crop_face(self, image, face_box, padding=None):
        """
        Crop face from image with padding
        
        Args:
            image: PIL Image or numpy array
            face_box: (x, y, w, h) tuple
            padding: Padding ratio (default from config)
            
        Returns:
            Cropped face as PIL Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        x, y, w, h = face_box
        padding = padding or config.FACE_PADDING
        
        # Calculate padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Apply padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.width, x + w + pad_w)
        y2 = min(image.height, y + h + pad_h)
        
        # Crop
        face_crop = image.crop((x1, y1, x2, y2))
        
        return face_crop


class ImagePreprocessor:
    """Image preprocessing pipeline"""
    
    def __init__(self, face_detector=None, img_size=None, augment=False):
        """
        Initialize preprocessor
        
        Args:
            face_detector: FaceDetector instance (optional)
            img_size: Target image size (default from config)
            augment: Whether to apply augmentation
        """
        self.face_detector = face_detector
        self.img_size = img_size or config.IMG_SIZE
        self.augment = augment
        
        # Build transformation pipeline
        self.transform = self._build_transform(augment)
    
    def _build_transform(self, augment=False):
        """Build torchvision transforms pipeline"""
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize((self.img_size, self.img_size)))
        
        if augment:
            # Data augmentation
            if config.AUGMENTATION.get('horizontal_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            if config.AUGMENTATION.get('rotation_range', 0) > 0:
                rotation = config.AUGMENTATION['rotation_range']
                transform_list.append(transforms.RandomRotation(degrees=rotation))
            
            if config.AUGMENTATION.get('brightness_range') or config.AUGMENTATION.get('zoom_range'):
                # Color jitter for brightness
                brightness = config.AUGMENTATION.get('brightness_range', (1.0, 1.0))
                if isinstance(brightness, tuple):
                    brightness_factor = brightness[1] - brightness[0]
                    transform_list.append(
                        transforms.ColorJitter(
                            brightness=brightness_factor,
                            contrast=0.2,
                            saturation=0.2
                        )
                    )
            
            # Random affine for zoom and translation
            if config.AUGMENTATION.get('zoom_range') or config.AUGMENTATION.get('width_shift_range'):
                scale = (1 - config.AUGMENTATION.get('zoom_range', 0), 
                        1 + config.AUGMENTATION.get('zoom_range', 0))
                translate = (config.AUGMENTATION.get('width_shift_range', 0),
                            config.AUGMENTATION.get('height_shift_range', 0))
                transform_list.append(
                    transforms.RandomAffine(
                        degrees=0,
                        translate=translate,
                        scale=scale
                    )
                )
        
        # Convert to tensor and normalize
        transform_list.append(transforms.ToTensor())
        transform_list.append(
            transforms.Normalize(
                mean=config.NORMALIZATION_MEAN,
                std=config.NORMALIZATION_STD
            )
        )
        
        return transforms.Compose(transform_list)
    
    def preprocess(self, image, detect_face=True):
        """
        Preprocess a single image
        
        Args:
            image: PIL Image, numpy array, or path to image
            detect_face: Whether to detect and crop face
            
        Returns:
            Preprocessed tensor and face box (if detected)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        face_box = None
        
        # Detect and crop face if requested
        if detect_face and self.face_detector is not None:
            faces = self.face_detector.detect_faces(image)
            
            if len(faces) > 0:
                # Take the first (largest) face
                face_box = faces[0]
                image = self.face_detector.crop_face(image, face_box)
        
        # Apply transformations
        tensor = self.transform(image)
        
        return tensor, face_box
    
    def preprocess_batch(self, images, detect_faces=True):
        """
        Preprocess a batch of images
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            detect_faces: Whether to detect and crop faces
            
        Returns:
            Batch tensor and list of face boxes
        """
        tensors = []
        face_boxes = []
        
        for image in images:
            try:
                tensor, face_box = self.preprocess(image, detect_face=detect_faces)
                tensors.append(tensor)
                face_boxes.append(face_box)
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if len(tensors) == 0:
            return None, []
        
        # Stack tensors
        batch_tensor = torch.stack(tensors)
        
        return batch_tensor, face_boxes


def apply_histogram_equalization(image):
    """
    Apply histogram equalization to improve contrast
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Image with equalized histogram
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(result)


def visualize_face_detection(image, faces, save_path=None):
    """
    Visualize detected faces with bounding boxes
    
    Args:
        image: PIL Image or numpy array
        faces: List of face boxes (x, y, w, h)
        save_path: Optional path to save visualization
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Make a copy
    vis_image = image.copy()
    
    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save or display
    if save_path:
        Image.fromarray(vis_image).save(save_path)
    
    return vis_image


if __name__ == '__main__':
    # Test face detection
    print("Testing face detection...")
    detector = FaceDetector(method='haar')
    
    # Test on a sample image
    sample_dir = config.TRAIN_DIR / 'happy'
    sample_images = list(sample_dir.glob('*'))[:5]
    
    if sample_images:
        print(f"Testing on {len(sample_images)} sample images")
        
        for img_path in sample_images:
            image = Image.open(img_path)
            faces = detector.detect_faces(image)
            print(f"{img_path.name}: {len(faces)} face(s) detected")
            
            if len(faces) > 0:
                # Visualize
                vis = visualize_face_detection(image, faces)
                print(f"  Face box: {faces[0]}")
    else:
        print("No sample images found")
