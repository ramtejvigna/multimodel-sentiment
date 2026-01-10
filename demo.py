"""
Real-time Webcam Emotion Detection Demo
Detects faces and predicts emotions in real-time
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import deque
import argparse
import time

import config
from model import create_model
from image_preprocessing import FaceDetector, ImagePreprocessor


class WebcamEmotionDetector:
    """Real-time emotion detection from webcam"""
    
    def __init__(self, model_path, device=None, temporal_smoothing=True):
        """
        Initialize webcam detector
        
        Args:
            model_path: Path to trained model
            device: Device to run on
            temporal_smoothing: Whether to smooth predictions over time
        """
        self.device = device or torch.device(config.DEVICE)
        self.temporal_smoothing = temporal_smoothing
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = create_model(
            model_name='resnet18',
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            training_mode='scratch'
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Initialize detectors
        self.face_detector = FaceDetector(method='haar')  # Haar is faster for real-time
        self.preprocessor = ImagePreprocessor(
            face_detector=None,  # We'll detect faces manually
            img_size=config.IMG_SIZE,
            augment=False
        )
        
        # Temporal smoothing buffer
        self.prediction_buffer = deque(maxlen=config.SMOOTHING_WINDOW)
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=30)
        
        # Emotion colors (BGR format for OpenCV)
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (0, 255, 255), # Yellow
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 128),  # Olive
            'neutral': (255, 255, 255) # White
        }
    
    def predict_emotion(self, face_crop):
        """
        Predict emotion for a face crop
        
        Args:
            face_crop: PIL Image of face
            
        Returns:
            Tuple of (emotion, confidence, probabilities)
        """
        # Preprocess
        tensor, _ = self.preprocessor.preprocess(face_crop, detect_face=False)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get emotion
        emotion_idx = predicted.item()
        emotion = config.IDX_TO_CLASS[emotion_idx]
        confidence_score = confidence.item()
        
        # Get all probabilities
        probs = {
            config.IDX_TO_CLASS[i]: probabilities[0][i].item()
            for i in range(config.NUM_CLASSES)
        }
        
        return emotion, confidence_score, probs
    
    def smooth_predictions(self, probs):
        """
        Smooth predictions over time using temporal averaging
        
        Args:
            probs: Current probability dict
            
        Returns:
            Smoothed emotion and confidence
        """
        # Add to buffer
        self.prediction_buffer.append(probs)
        
        # Average probabilities
        if len(self.prediction_buffer) == 0:
            return None, 0.0
        
        # Calculate average
        avg_probs = {}
        for emotion in config.EMOTION_CLASSES:
            avg_probs[emotion] = np.mean([p[emotion] for p in self.prediction_buffer])
        
        # Get max
        emotion = max(avg_probs, key=avg_probs.get)
        confidence = avg_probs[emotion]
        
        return emotion, confidence, avg_probs
    
    def draw_emotion_bars(self, frame, probs, x_offset=10, y_offset=50):
        """
        Draw probability bars on frame
        
        Args:
            frame: OpenCV frame
            probs: Probability dictionary
            x_offset: X position
            y_offset: Y position
        """
        bar_width = 200
        bar_height = 20
        
        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, prob) in enumerate(sorted_probs):
            y = y_offset + i * (bar_height + 5)
            
            # Background bar
            cv2.rectangle(frame, (x_offset, y), 
                         (x_offset + bar_width, y + bar_height),
                         (50, 50, 50), -1)
            
            # Probability bar
            prob_width = int(bar_width * prob)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x_offset, y),
                         (x_offset + prob_width, y + bar_height),
                         color, -1)
            
            # Text
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(frame, text, (x_offset + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self, camera_id=0, display_width=640, display_height=480):
        """
        Run real-time emotion detection
        
        Args:
            camera_id: Camera device ID
            display_width: Display width
            display_height: Display height
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("\n" + "="*60)
        print("REAL-TIME EMOTION DETECTION")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Take screenshot")
        print("  'r' - Reset temporal buffer")
        print("="*60 + "\n")
        
        screenshot_count = 0
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for face detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame_rgb)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Draw face bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Crop face with padding
                padding = int(w * config.FACE_PADDING)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_crop = Image.fromarray(frame_rgb[y1:y2, x1:x2])
                
                # Predict emotion
                emotion, confidence, probs = self.predict_emotion(face_crop)
                
                # Smooth predictions if enabled
                if self.temporal_smoothing:
                    emotion, confidence, probs = self.smooth_predictions(probs)
                
                # Draw emotion label
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                label = f"{emotion}: {confidence:.2f}"
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(frame, (x, max(0, y - 35)), 
                            (x + text_w + 10, y), color, -1)
                
                # Text
                cv2.putText(frame, label, (x + 5, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Draw probability bars (only for first face)
                if (x, y, w, h) == faces[0]:
                    self.draw_emotion_bars(frame, probs)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            self.fps_buffer.append(fps)
            avg_fps = np.mean(self.fps_buffer)
            
            # Draw info
            info_text = f"FPS: {avg_fps:.1f} | Faces: {len(faces)}"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' for screenshot", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display
            cv2.imshow('Emotion Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = config.RESULTS_DIR / f'screenshot_{screenshot_count:04d}.jpg'
                cv2.imwrite(str(screenshot_path), frame)
                print(f"Screenshot saved: {screenshot_path}")
                screenshot_count += 1
            elif key == ord('r'):
                # Reset buffer
                self.prediction_buffer.clear()
                print("Temporal buffer reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Real-time Emotion Detection Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--width', type=int, default=640,
                       help='Display width')
    parser.add_argument('--height', type=int, default=480,
                       help='Display height')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable temporal smoothing')
    
    args = parser.parse_args()
    
    # Create detector
    detector = WebcamEmotionDetector(
        model_path=args.model,
        temporal_smoothing=not args.no_smoothing
    )
    
    # Run demo
    detector.run(
        camera_id=args.camera,
        display_width=args.width,
        display_height=args.height
    )


if __name__ == '__main__':
    main()
