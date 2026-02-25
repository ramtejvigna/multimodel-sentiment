"""
Unified Multimodal Inference Pipeline
Combines text sentiment and facial emotion models for prediction
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
from PIL import Image

# Import existing modules
import config
import text_config
from model import create_model as create_face_model
from text_model import create_text_model
from text_preprocessing import TextPreprocessor, Vocabulary
from image_preprocessing import preprocess_face, detect_face
from fusion_model import create_fusion_model


class MultimodalSentimentAnalyzer:
    """Unified multimodal sentiment analyzer"""
    
    def __init__(
        self,
        face_model_path,
        text_model_path,
        vocab_path,
        fusion_type='late',
        fusion_weights=None,
        device=None
    ):
        """
        Initialize multimodal analyzer
        
        Args:
            face_model_path: Path to trained face emotion model
            text_model_path: Path to trained text sentiment model
            vocab_path: Path to vocabulary file
            fusion_type: 'late', 'adaptive', or 'feature'
            fusion_weights: Dict with 'text_weight' and 'face_weight' for late fusion
            device: Device to run models on
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"Initializing Multimodal Sentiment Analyzer...")
        print(f"Device: {self.device}")
        
        # Load vocabulary
        print(f"Loading vocabulary from {vocab_path}...")
        self.vocab = Vocabulary(text_config)
        self.vocab.load(vocab_path)
        
        # Load face emotion model
        print(f"Loading face emotion model from {face_model_path}...")
        self.face_model = self._load_face_model(face_model_path)
        self.face_model.eval()
        
        # Load text sentiment model
        print(f"Loading text sentiment model from {text_model_path}...")
        self.text_model = self._load_text_model(text_model_path)
        self.text_model.eval()
        
        # Text preprocessor
        self.text_preprocessor = TextPreprocessor(text_config)
        
        # Fusion model
        fusion_weights = fusion_weights or {'text_weight': 0.4, 'face_weight': 0.6}
        self.fusion_model = create_fusion_model(fusion_type, **fusion_weights)
        
        # Class names
        self.emotion_classes = config.EMOTION_CLASSES
        self.sentiment_classes = text_config.SENTIMENT_CLASSES
        
        print("Multimodal analyzer initialized successfully!\n")
    
    def _load_face_model(self, model_path):
        """Load face emotion model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model architecture (try to infer from checkpoint)
        # For simplicity, default to resnet18
        model = create_face_model(
            model_name='resnet18',
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            training_mode='scratch'
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def _load_text_model(self, model_path):
        """Load text sentiment model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        vocab_size = checkpoint.get('vocab_size', len(self.vocab))
        
        model = create_text_model(
            vocab_size=vocab_size,
            num_classes=text_config.NUM_SENTIMENT_CLASSES,
            model_type='lstm_attention',
            pretrained_embeddings=None,
            config=text_config
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def analyze_text(self, text):
        """
        Analyze text sentiment
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess
        tokens = self.text_preprocessor.preprocess(text)
        encoded = self.vocab.encode(tokens)
        
        # Pad/truncate
        from text_preprocessing import pad_sequence
        encoded = pad_sequence(
            encoded,
            text_config.MAX_SEQ_LENGTH,
            self.vocab.pad_idx,
            text_config.PADDING_METHOD
        )
        
        # Convert to tensor
        text_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.text_model(text_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Get results
        probs_np = probs.cpu().numpy()[0]
        pred_idx = np.argmax(probs_np)
        pred_label = self.sentiment_classes[pred_idx]
        confidence = probs_np[pred_idx]
        
        return {
            'prediction': pred_label,
            'confidence': float(confidence),
            'probabilities': {
                self.sentiment_classes[i]: float(probs_np[i])
                for i in range(len(self.sentiment_classes))
            },
            'raw_logits': logits.cpu().numpy()[0].tolist(),
            'raw_probs': probs.cpu().numpy()[0]
        }
    
    def analyze_face(self, image):
        """
        Analyze facial emotion
        
        Args:
            image: Image as numpy array (BGR) or PIL Image or path
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect and preprocess face
        face_img = preprocess_face(image)
        
        if face_img is None:
            return {
                'prediction': 'neutral',
                'confidence': 0.0,
                'probabilities': {cls: 0.0 for cls in self.emotion_classes},
                'error': 'No face detected'
            }
        
        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZATION_MEAN,
                std=config.NORMALIZATION_STD
            )
        ])
        
        face_tensor = transform(face_img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.face_model(face_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Get results
        probs_np = probs.cpu().numpy()[0]
        pred_idx = np.argmax(probs_np)
        pred_label = self.emotion_classes[pred_idx]
        confidence = probs_np[pred_idx]
        
        return {
            'prediction': pred_label,
            'confidence': float(confidence),
            'probabilities': {
                self.emotion_classes[i]: float(probs_np[i])
                for i in range(len(self.emotion_classes))
            },
            'raw_logits': logits.cpu().numpy()[0].tolist(),
            'raw_probs': probs.cpu().numpy()[0]
        }
    
    def analyze_multimodal(self, text, image, return_individual=True):
        """
        Analyze both text and face for comprehensive sentiment
        
        Args:
            text: Input text string
            image: Image (numpy array, PIL, or path)
            return_individual: Whether to return individual predictions
            
        Returns:
            Dictionary with multimodal prediction
        """
        # Get individual predictions
        text_result = self.analyze_text(text)
        face_result = self.analyze_face(image)
        
        # Fuse predictions
        text_probs = torch.tensor([text_result['raw_probs']]).to(self.device)
        face_probs = torch.tensor([face_result['raw_probs']]).to(self.device)
        
        fused_result = self.fusion_model(text_probs, face_probs)
        
        # Interpret fusion result
        if isinstance(fused_result, dict):
            # Late fusion with different class spaces
            # Use weighted decision based on confidences
            text_weight = fused_result['text_weight']
            face_weight = fused_result['face_weight']
            
            # Map sentiments to dominant emotion
            sentiment_emotion_map = {
                'positive': ['happy', 'surprise'],
                'negative': ['angry', 'sad', 'fear', 'disgust'],
                'neutral': ['neutral']
            }
            
            text_pred = text_result['prediction']
            face_pred = face_result['prediction']
            
            # Determine final prediction
            if text_pred == 'positive' and face_pred in sentiment_emotion_map['positive']:
                final_pred = 'positive'
                confidence = (text_result['confidence'] * text_weight + 
                            face_result['confidence'] * face_weight)
            elif text_pred == 'negative' and face_pred in sentiment_emotion_map['negative']:
                final_pred = 'negative'
                confidence = (text_result['confidence'] * text_weight + 
                            face_result['confidence'] * face_weight)
            else:
                # Conflicting signals - use higher confidence
                if face_result['confidence'] > text_result['confidence']:
                    final_pred = face_pred
                    confidence = face_result['confidence'] * 0.7  # Reduce due to conflict
                else:
                    final_pred = text_pred
                    confidence = text_result['confidence'] * 0.7
        else:
            # Unified class space - use fused probabilities
            fused_probs = fused_result.cpu().numpy()[0]
            pred_idx = np.argmax(fused_probs)
            final_pred = self.emotion_classes[pred_idx]
            confidence = float(fused_probs[pred_idx])
        
        result = {
            'multimodal_prediction': final_pred,
            'multimodal_confidence': float(confidence),
            'agreement': text_result['prediction'] == face_result['prediction']
        }
        
        if return_individual:
            result['text_analysis'] = text_result
            result['face_analysis'] = face_result
        
        return result


# Convenience function
def create_multimodal_analyzer(
    face_model_path=None,
    text_model_path=None,
    vocab_path=None,
    **kwargs
):
    """
    Create multimodal analyzer with default paths
    
    Args:
        face_model_path: Path to face model (default: output/models/best_model.pth)
        text_model_path: Path to text model (default: output/text_models/best_text_model.pth)
        vocab_path: Path to vocabulary (default: output/text_models/vocabulary.pkl)
        **kwargs: Additional arguments for MultimodalSentimentAnalyzer
        
    Returns:
        MultimodalSentimentAnalyzer instance
    """
    face_model_path = face_model_path or config.MODELS_DIR / 'best_model.pth'
    text_model_path = text_model_path or text_config.TEXT_MODELS_DIR / 'best_text_model.pth'
    vocab_path = vocab_path or text_config.TEXT_MODELS_DIR / 'vocabulary.pkl'
    
    return MultimodalSentimentAnalyzer(
        face_model_path=face_model_path,
        text_model_path=text_model_path,
        vocab_path=vocab_path,
        **kwargs
    )


if __name__ == '__main__':
    print("Multimodal Inference Pipeline Test\n")
    print("="*60)
    print("This script requires trained models to run.")
    print("Please train both face and text models first:")
    print("  1. Train face model: python train.py --model resnet18 --epochs 50")
    print("  2. Train text model: python train_text.py --train-data <path> --epochs 30")
    print("="*60)
