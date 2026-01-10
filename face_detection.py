"""
Face Detection Module
Standalone face detection utilities using OpenCV and MTCNN
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import after checking availability
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not installed. MTCNN will not be available.")


class HaarCascadeDetector:
    """Face detection using Haar Cascade"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        if self.detector.empty():
            raise ValueError(f"Failed to load Haar Cascade from {cascade_path}")
    
    def detect(self, image):
        """
        Detect faces in image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            List of (x, y, w, h) tuples
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces if len(faces) > 0 else []


class MTCNNDetector:
    """Face detection using MTCNN"""
    
    def __init__(self, device='cpu'):
        if not MTCNN_AVAILABLE:
            raise ImportError("facenet-pytorch is required for MTCNN")
        
        self.detector = MTCNN(keep_all=True, device=device)
    
    def detect(self, image):
        """
        Detect faces in image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            List of (x, y, w, h) tuples
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        boxes, _ = self.detector.detect(image)
        
        if boxes is not None:
            # Convert to (x, y, w, h) format
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                faces.append((x1, y1, x2 - x1, y2 - y1))
            return faces
        
        return []


def visualize_comparison(image_path, save_path=None):
    """
    Compare Haar Cascade and MTCNN detection
    
    Args:
        image_path: Path to image
        save_path: Optional path to save comparison
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    
    # Detect with both methods
    haar_detector = HaarCascadeDetector()
    haar_faces = haar_detector.detect(image)
    
    # Create visualization
    if MTCNN_AVAILABLE:
        mtcnn_detector = MTCNNDetector()
        mtcnn_faces = mtcnn_detector.detect(image)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Haar Cascade
        haar_img = img_array.copy()
        for (x, y, w, h) in haar_faces:
            cv2.rectangle(haar_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        axes[1].imshow(haar_img)
        axes[1].set_title(f'Haar Cascade ({len(haar_faces)} faces)')
        axes[1].axis('off')
        
        # MTCNN
        mtcnn_img = img_array.copy()
        for (x, y, w, h) in mtcnn_faces:
            cv2.rectangle(mtcnn_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        axes[2].imshow(mtcnn_img)
        axes[2].set_title(f'MTCNN ({len(mtcnn_faces)} faces)')
        axes[2].axis('off')
    
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Haar Cascade
        haar_img = img_array.copy()
        for (x, y, w, h) in haar_faces:
            cv2.rectangle(haar_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        axes[1].imshow(haar_img)
        axes[1].set_title(f'Haar Cascade ({len(haar_faces)} faces)')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Test on provided image
        image_path = sys.argv[1]
        visualize_comparison(image_path)
    else:
        # Test on sample from dataset
        sample_dir = Path('dataset/train/happy')
        if sample_dir.exists():
            sample_images = list(sample_dir.glob('*'))[:1]
            if sample_images:
                print(f"Testing on: {sample_images[0]}")
                visualize_comparison(sample_images[0])
            else:
                print("No sample images found")
        else:
            print(f"Dataset not found at {sample_dir}")
            print("Usage: python face_detection.py <image_path>")
