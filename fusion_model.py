"""
Multimodal Fusion Architectures
Combines text sentiment and facial emotion predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionModel(nn.Module):
    """Late fusion: Weighted combination of predictions"""
    
    def __init__(self, text_weight=0.4, face_weight=0.6):
        """
        Initialize late fusion model
        
        Args:
            text_weight: Weight for text sentiment prediction
            face_weight: Weight for facial emotion prediction
        """
        super(LateFusionModel, self).__init__()
        
        # Ensure weights sum to 1
        total = text_weight + face_weight
        self.text_weight = text_weight / total
        self.face_weight = face_weight / total
        
        print(f"Late Fusion initialized:")
        print(f"  Text weight: {self.text_weight:.2f}")
        print(f"  Face weight: {self.face_weight:.2f}")
    
    def forward(self, text_probs, face_probs, sentiment_to_emotion_map=None):
        """
        Fuse text and face predictions
        
        Args:
            text_probs: Text sentiment probabilities (batch_size, num_text_classes)
            face_probs: Face emotion probabilities (batch_size, num_face_classes)
            sentiment_to_emotion_map: Optional mapping from sentiment to emotion indices
            
        Returns:
            Combined prediction or dictionary with both predictions
        """
        # If classes match, simple weighted combination
        if text_probs.shape[1] == face_probs.shape[1]:
            combined = self.text_weight * text_probs + self.face_weight * face_probs
            return combined
        
        # Otherwise, return both (let downstream decide based on task)
        return {
            'text': text_probs,
            'face': face_probs,
            'text_weight': self.text_weight,
            'face_weight': self.face_weight
        }


class AdaptiveFusionModel(nn.Module):
    """
    Adaptive late fusion: Learn fusion weights with confidence scores
    
    This model uses learnable weights to handle modality conflicts,
    using confidence scores for final output as recommended for 
    optimal multimodal sentiment analysis.
    """
    
    def __init__(self, num_text_classes, num_face_classes, use_confidence=True):
        """
        Initialize adaptive fusion model
        
        Args:
            num_text_classes: Number of text sentiment classes
            num_face_classes: Number of facial emotion classes
            use_confidence: Whether to use confidence-based weighting
        """
        super(AdaptiveFusionModel, self).__init__()
        
        self.num_text_classes = num_text_classes
        self.num_face_classes = num_face_classes
        self.use_confidence = use_confidence
        
        # Learnable fusion weights (adaptive late fusion)
        self.fusion_layer = nn.Linear(num_text_classes + num_face_classes, 
                                     max(num_text_classes, num_face_classes))
        
        # Confidence-based attention weights
        if use_confidence:
            self.confidence_network = nn.Sequential(
                nn.Linear(num_text_classes + num_face_classes, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # 2 weights: one for text, one for face
                nn.Softmax(dim=1)  # Normalize weights to sum to 1
            )
            print("AdaptiveFusionModel: Using confidence-based weighting")
        else:
            print("AdaptiveFusionModel: Using standard learnable fusion")
    
    def forward(self, text_probs, face_probs):
        """
        Fuse predictions with learned weights and confidence scores
        
        Args:
            text_probs: Text sentiment probabilities (batch_size, num_text_classes)
            face_probs: Face emotion probabilities (batch_size, num_face_classes)
            
        Returns:
            Fused predictions
        """
        # Concatenate probabilities
        combined = torch.cat([text_probs, face_probs], dim=1)
        
        if self.use_confidence:
            # Calculate confidence-based weights
            confidence_weights = self.confidence_network(combined)  # (batch_size, 2)
            
            # Apply confidence weighting before fusion
            weighted_text = text_probs * confidence_weights[:, 0:1]
            weighted_face = face_probs * confidence_weights[:, 1:2]
            
            # Concatenate weighted probabilities
            weighted_combined = torch.cat([weighted_text, weighted_face], dim=1)
            
            # Learn fusion with confidence-weighted inputs
            output = self.fusion_layer(weighted_combined)
        else:
            # Standard learnable fusion
            output = self.fusion_layer(combined)
        
        output = torch.softmax(output, dim=1)
        
        return output


class FeatureFusionModel(nn.Module):
    """Feature-level fusion: Combine embeddings before classification"""
    
    def __init__(self, text_feature_dim, face_feature_dim, 
                 num_classes, hidden_dims=[256, 128]):
        """
        Initialize feature fusion model
        
        Args:
            text_feature_dim: Dimension of text features
            face_feature_dim: Dimension of face features
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions for classifier
        """
        super(FeatureFusionModel, self).__init__()
        
        self.text_feature_dim = text_feature_dim
        self.face_feature_dim = face_feature_dim
        
        # Feature projection layers (optional, to match dimensions)
        self.text_projection = nn.Linear(text_feature_dim, 256)
        self.face_projection = nn.Linear(face_feature_dim, 256)
        
        # Fusion classifier
        fusion_input_dim = 256 + 256  # Concatenated projected features
        
        layers = []
        in_dim = fusion_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        print(f"Feature Fusion initialized:")
        print(f"  Text features: {text_feature_dim} -> 256")
        print(f"  Face features: {face_feature_dim} -> 256")
        print(f"  Fusion classifier: {fusion_input_dim} -> ... -> {num_classes}")
    
    def forward(self, text_features, face_features):
        """
        Fuse features and classify
        
        Args:
            text_features: Text embeddings (batch_size, text_feature_dim)
            face_features: Face embeddings (batch_size, face_feature_dim)
            
        Returns:
            Classification logits
        """
        # Project features
        text_proj = F.relu(self.text_projection(text_features))
        face_proj = F.relu(self.face_projection(face_features))
        
        # Concatenate
        fused = torch.cat([text_proj, face_proj], dim=1)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits


def create_fusion_model(fusion_type='late', **kwargs):
    """
    Create fusion model
    
    Args:
        fusion_type: 'late', 'adaptive', or 'feature'
        **kwargs: Additional arguments for specific fusion types
        
    Returns:
        Fusion model
    """
    if fusion_type == 'late':
        return LateFusionModel(
            text_weight=kwargs.get('text_weight', 0.4),
            face_weight=kwargs.get('face_weight', 0.6)
        )
    
    elif fusion_type == 'adaptive':
        return AdaptiveFusionModel(
            num_text_classes=kwargs['num_text_classes'],
            num_face_classes=kwargs['num_face_classes'],
            use_confidence=kwargs.get('use_confidence', True)  # Default: use confidence scores
        )
    
    elif fusion_type == 'feature':
        return FeatureFusionModel(
            text_feature_dim=kwargs['text_feature_dim'],
            face_feature_dim=kwargs['face_feature_dim'],
            num_classes=kwargs['num_classes'],
            hidden_dims=kwargs.get('hidden_dims', [256, 128])
        )
    
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


if __name__ == '__main__':
    # Test fusion models
    print("Testing fusion models...\n")
    
    batch_size = 4
    num_text_classes = 2  # positive, negative
    num_face_classes = 7  # 7 emotions
    text_feature_dim = 256
    face_feature_dim = 512
    
    # Test Late Fusion
    print("="*60)
    print("Testing Late Fusion")
    print("="*60)
    late_fusion = create_fusion_model('late', text_weight=0.4, face_weight=0.6)
    
    text_probs = torch.softmax(torch.randn(batch_size, num_text_classes), dim=1)
    face_probs = torch.softmax(torch.randn(batch_size, num_face_classes), dim=1)
    
    result = late_fusion(text_probs, face_probs)
    print(f"\nInput shapes:")
    print(f"  Text probs: {text_probs.shape}")
    print(f"  Face probs: {face_probs.shape}")
    print(f"\nOutput type: {type(result)}")
    if isinstance(result, dict):
        print(f"  Contains: {result.keys()}")
    
    # Test Adaptive Fusion
    print("\n" + "="*60)
    print("Testing Adaptive Fusion")
    print("="*60)
    adaptive_fusion = create_fusion_model(
        'adaptive',
        num_text_classes=num_text_classes,
        num_face_classes=num_face_classes
    )
    
    result = adaptive_fusion(text_probs, face_probs)
    print(f"\nOutput shape: {result.shape}")
    print(f"Sample output: {result[0]}")
    
    # Test Feature Fusion
    print("\n" + "="*60)
    print("Testing Feature Fusion")
    print("="*60)
    feature_fusion = create_fusion_model(
        'feature',
        text_feature_dim=text_feature_dim,
        face_feature_dim=face_feature_dim,
        num_classes=7
    )
    
    text_features = torch.randn(batch_size, text_feature_dim)
    face_features = torch.randn(batch_size, face_feature_dim)
    
    result = feature_fusion(text_features, face_features)
    print(f"\nInput shapes:")
    print(f"  Text features: {text_features.shape}")
    print(f"  Face features: {face_features.shape}")
    print(f"Output logits: {result.shape}")
    
    probs = torch.softmax(result, dim=1)
    print(f"Sample probabilities: {probs[0]}")
    print(f"Sum: {probs[0].sum().item():.4f}")
    
    print("\n" + "="*60)
    print("Fusion models test completed!")
