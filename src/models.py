"""
models.py - Deep Learning Architectures for Kidney CT Abnormality Classification.
Supports:
1. Baseline Lightweight CNN (trained from scratch)
2. Fine-tuned ResNet-50 (Pretrained on ImageNet, custom classification head)
3. Fine-tuned EfficientNet-B0 (Compound scaling, lightweight and highly accurate)
4. MobileNetV3-Large (High inference efficiency for clinical edge deployment)
"""

import torch
import torch.nn as nn
from torchvision import models

class CustomKidneyCNN(nn.Module):
    """
    Baseline 4-stage Convolutional Neural Network trained from scratch.
    Serves as an empirical baseline against pretrained transfer learning architectures.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super(CustomKidneyCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(
    model_name: str = 'resnet50',
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.4
) -> nn.Module:
    """
    Factory function to instantiate models with custom medical classification heads.
    """
    model_name = model_name.lower()

    if model_name == 'custom_cnn':
        return CustomKidneyCNN(num_classes=num_classes, dropout=dropout)

    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        return model

    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        return model

    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        return model

    elif model_name == 'mobilenet_v3':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        return model

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
