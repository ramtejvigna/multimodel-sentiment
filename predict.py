"""
Inference Script for Emotion Recognition
Supports single image, batch, and folder processing
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import argparse
import json
import cv2

import config
from model import create_model
from image_preprocessing import FaceDetector, ImagePreprocessor


class EmotionPredictor:
    """Emotion prediction engine"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device or torch.device(config.DEVICE)
        self.model_path = model_path
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model (assuming ResNet18 by default, can be modified)
        self.model = create_model(
            model_name='resnet18',
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            training_mode='scratch'
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
        
        # Initialize face detector and preprocessor
        self.face_detector = FaceDetector(method=config.FACE_DETECTION_METHOD)
        self.preprocessor = ImagePreprocessor(
            face_detector=self.face_detector,
            img_size=config.IMG_SIZE,
            augment=False
        )
    
    def predict_image(self, image_path, return_probs=False):
        """
        Predict emotion for a single image
        
        Args:
            image_path: Path to image or PIL Image
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Detect face
        faces = self.face_detector.detect_faces(image)
        
        if len(faces) == 0:
            return {
                'emotion': 'no_face_detected',
                'confidence': 0.0,
                'face_box': None,
                'probabilities': None
            }
        
        # Preprocess
        tensor, face_box = self.preprocessor.preprocess(image, detect_face=True)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get emotion label
        emotion_idx = predicted.item()
        emotion = config.IDX_TO_CLASS[emotion_idx]
        confidence_score = confidence.item()
        
        result = {
            'emotion': emotion,
            'confidence': confidence_score,
            'face_box': face_box,
        }
        
        if return_probs:
            probs_dict = {
                config.IDX_TO_CLASS[i]: probabilities[0][i].item()
                for i in range(config.NUM_CLASSES)
            }
            result['probabilities'] = probs_dict
        
        return result
    
    def predict_batch(self, image_paths, return_probs=False):
        """
        Predict emotions for multiple images
        
        Args:
            image_paths: List of image paths
            return_probs: Whether to return probabilities
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for img_path in image_paths:
            result = self.predict_image(img_path, return_probs=return_probs)
            result['image_path'] = str(img_path)
            results.append(result)
        
        return results
    
    def annotate_image(self, image_path, output_path=None):
        """
        Annotate image with predicted emotion
        
        Args:
            image_path: Path to image
            output_path: Path to save annotated image
            
        Returns:
            Annotated PIL Image
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Predict
        result = self.predict_image(image, return_probs=True)
        
        if result['emotion'] == 'no_face_detected':
            # Draw "No face detected" message
            text = "No face detected"
            draw.text((10, 10), text, fill=(255, 0, 0))
        else:
            # Draw bounding box
            if result['face_box'] is not None:
                x, y, w, h = result['face_box']
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
            
            # Draw emotion label
            emotion = result['emotion']
            confidence = result['confidence']
            text = f"{emotion}: {confidence:.2f}"
            
            # Position text above bounding box
            if result['face_box'] is not None:
                text_x, text_y = x, max(0, y - 30)
            else:
                text_x, text_y = 10, 10
            
            # Draw background for text
            bbox = draw.textbbox((text_x, text_y), text)
            draw.rectangle(bbox, fill=(0, 255, 0))
            draw.text((text_x, text_y), text, fill=(0, 0, 0))
            
            # Draw probability bars
            if result['probabilities']:
                y_offset = 10
                for emotion_name, prob in sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    bar_length = int(prob * 200)
                    bar_text = f"{emotion_name}: {prob:.2f}"
                    
                    draw.rectangle(
                        [10, y_offset, 10 + bar_length, y_offset + 15],
                        fill=(100, 200, 100)
                    )
                    draw.text((10, y_offset), bar_text, fill=(255, 255, 255))
                    y_offset += 20
        
        # Save if output path provided
        if output_path:
            image.save(output_path)
            print(f"Annotated image saved to {output_path}")
        
        return image


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Emotion Recognition Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--folder', type=str,
                       help='Path to folder of images')
    parser.add_argument('--output-dir', type=str, default='output/predictions',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations with bounding boxes')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results as JSON')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create predictor
    predictor = EmotionPredictor(model_path=args.model)
    
    # Process images
    if args.image:
        # Single image
        print(f"\nProcessing image: {args.image}")
        result = predictor.predict_image(args.image, return_probs=True)
        
        print(f"\nPrediction:")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        
        if result['probabilities']:
            print(f"\n  Probabilities:")
            for emotion, prob in sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"    {emotion}: {prob:.4f}")
        
        # Visualize
        if args.visualize:
            output_path = output_dir / f"annotated_{Path(args.image).name}"
            predictor.annotate_image(args.image, output_path)
        
        # Save JSON
        if args.save_json:
            json_path = output_dir / f"result_{Path(args.image).stem}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to {json_path}")
    
    elif args.folder:
        # Batch processing
        folder_path = Path(args.folder)
        image_files = list(folder_path.glob('*.jpg')) + \
                     list(folder_path.glob('*.png')) + \
                     list(folder_path.glob('*.jpeg'))
        
        print(f"\nProcessing {len(image_files)} images from {args.folder}")
        
        results = predictor.predict_batch(image_files, return_probs=True)
        
        # Print summary
        emotion_counts = {}
        for result in results:
            emotion = result['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\nPrediction summary:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {count}")
        
        # Visualize all
        if args.visualize:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            for img_path in image_files:
                output_path = vis_dir / f"annotated_{img_path.name}"
                predictor.annotate_image(img_path, output_path)
            
            print(f"\nVisualizations saved to {vis_dir}")
        
        # Save JSON
        if args.save_json:
            json_path = output_dir / 'batch_results.json'
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nBatch results saved to {json_path}")
    
    else:
        print("Error: Please provide --image or --folder")


if __name__ == '__main__':
    main()
