"""
Emotion Recognition Model Architectures
Supports transfer learning with ResNet, MobileNet, and EfficientNet
"""

import torch
import torch.nn as nn
import torchvision.models as models
import config


class EmotionCNN(nn.Module):
    """Custom CNN for emotion recognition (lightweight option)"""
    
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Convolutional blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNetEmotionClassifier(nn.Module):
    """ResNet-based emotion classifier"""
    
    def __init__(self, model_name='resnet18', num_classes=7, pretrained=True):
        super(ResNetEmotionClassifier, self).__init__()
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
        
        # Replace final fc layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class MobileNetEmotionClassifier(nn.Module):
    """MobileNetV2-based emotion classifier"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super(MobileNetEmotionClassifier, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace classifier
        feature_dim = 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class EfficientNetEmotionClassifier(nn.Module):
    """EfficientNet-based emotion classifier"""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=7, pretrained=True):
        super(EfficientNetEmotionClassifier, self).__init__()
        
        try:
            # Load pretrained EfficientNet
            if model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                feature_dim = 1280
            elif model_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
                feature_dim = 1280
            else:
                raise ValueError(f"Unknown EfficientNet model: {model_name}")
            
            # Replace classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        except AttributeError:
            raise ImportError("EfficientNet requires torchvision >= 0.11.0")
    
    def forward(self, x):
        return self.backbone(x)


def create_model(model_name='resnet18', num_classes=7, pretrained=True, training_mode='scratch'):
    """
    Create emotion recognition model
    
    Args:
        model_name: Model architecture name
        num_classes: Number of emotion classes
        pretrained: Whether to use pretrained weights
        training_mode: 'scratch' or 'finetune'
        
    Returns:
        PyTorch model
    """
    # Create model based on architecture
    if model_name == 'custom_cnn':
        model = EmotionCNN(num_classes=num_classes)
        print(f"Created custom CNN model")
    
    elif 'resnet' in model_name:
        model = ResNetEmotionClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
        print(f"Created {model_name} model (pretrained={pretrained})")
    
    elif 'mobilenet' in model_name:
        model = MobileNetEmotionClassifier(
            num_classes=num_classes,
            pretrained=pretrained
        )
        print(f"Created MobileNetV2 model (pretrained={pretrained})")
    
    elif 'efficientnet' in model_name:
        model = EfficientNetEmotionClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
        print(f"Created {model_name} model (pretrained={pretrained})")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze backbone if finetuning
    if training_mode == 'finetune' and pretrained:
        print("Freezing backbone layers for fine-tuning...")
        # Freeze all layers except the final classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
        
        print("Only classifier layers will be trained")
    
    return model


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {total_params - trainable_params:,}")
    
    return total_params, trainable_params


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...\n")
    
    # Test different architectures
    models_to_test = ['resnet18', 'mobilenet_v2', 'custom_cnn']
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print('='*60)
        
        try:
            # Create model
            model = create_model(
                model_name=model_name,
                num_classes=config.NUM_CLASSES,
                pretrained=True,
                training_mode='scratch'
            )
            
            # Count parameters
            count_parameters(model)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
            output = model(dummy_input)
            print(f"\nOutput shape: {output.shape}")
            print(f"Expected shape: torch.Size([1, {config.NUM_CLASSES}])")
            
            # Test probabilities
            probs = torch.softmax(output, dim=1)
            print(f"Probabilities sum: {probs.sum().item():.4f}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    print("\n" + "="*60)
    print("Model test completed!")
