"""
Video deepfake detection models.

Architectures:
- XceptionNet-based (efficient, proven for deepfake detection)
- EfficientNet-based (good accuracy/efficiency tradeoff)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================================
# Xception-based Detector
# ============================================================================

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution (Xception building block)."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """Xception residual block with skip connections."""
    
    def __init__(self, in_channels, out_channels, stride=1, grow_first=True):
        super().__init__()
        
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
        
        self.relu = nn.ReLU(inplace=True)
        
        if grow_first:
            self.conv1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = SeparableConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv2 = SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = x
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        if self.skip is not None:
            residual = self.skip_bn(self.skip(residual))
        
        x += residual
        return x


class XceptionVideoDetector(nn.Module):
    """
    Xception-based video deepfake detector.
    
    Architecture inspired by FaceForensics++ baseline.
    Input: Face crop (3, 299, 299)
    Output: Binary classification (real/fake)
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.block1 = XceptionBlock(64, 128, stride=2, grow_first=True)
        self.block2 = XceptionBlock(128, 256, stride=2, grow_first=True)
        self.block3 = XceptionBlock(256, 728, stride=2, grow_first=True)
        
        # Middle flow (8 blocks)
        self.block4 = XceptionBlock(728, 728, stride=1, grow_first=True)
        self.block5 = XceptionBlock(728, 728, stride=1, grow_first=True)
        self.block6 = XceptionBlock(728, 728, stride=1, grow_first=True)
        self.block7 = XceptionBlock(728, 728, stride=1, grow_first=True)
        self.block8 = XceptionBlock(728, 728, stride=1, grow_first=True)
        self.block9 = XceptionBlock(728, 728, stride=1, grow_first=True)
        self.block10 = XceptionBlock(728, 728, stride=1, grow_first=True)
        self.block11 = XceptionBlock(728, 728, stride=1, grow_first=True)
        
        # Exit flow
        self.block12 = XceptionBlock(728, 1024, stride=2, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)
        
        self.conv4 = SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # Entry flow
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        # Exit flow
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        # Classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ============================================================================
# Lightweight CNN Detector (Faster Alternative)
# ============================================================================

class LightweightVideoDetector(nn.Module):
    """
    Lightweight CNN for video deepfake detection.
    
    Faster alternative to Xception for edge devices.
    Input: Face crop (3, 224, 224)
    Output: Binary classification
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ============================================================================
# Model Factory
# ============================================================================

def create_video_detector(
    architecture: str = "xception",
    num_classes: int = 2,
    dropout: float = 0.5,
    pretrained: bool = False
) -> nn.Module:
    """
    Create a video deepfake detector.
    
    Args:
        architecture: "xception" or "lightweight"
        num_classes: Number of output classes (default: 2 for binary)
        dropout: Dropout rate
        pretrained: Load pretrained weights (not implemented yet)
    
    Returns:
        PyTorch model
    """
    
    if architecture == "xception":
        model = XceptionVideoDetector(num_classes=num_classes, dropout=dropout)
    elif architecture == "lightweight":
        model = LightweightVideoDetector(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    if pretrained:
        # TODO: Load pretrained weights
        raise NotImplementedError("Pretrained weights not yet available")
    
    return model


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """Get model information."""
    return {
        "architecture": model.__class__.__name__,
        "parameters": count_parameters(model),
        "parameters_mb": count_parameters(model) * 4 / (1024 * 1024),  # Assuming float32
    }


if __name__ == "__main__":
    # Test models
    print("Testing XceptionVideoDetector...")
    model_xception = create_video_detector("xception")
    test_input = torch.randn(2, 3, 299, 299)
    output = model_xception(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {count_parameters(model_xception):,}")
    
    print("\nTesting LightweightVideoDetector...")
    model_lightweight = create_video_detector("lightweight")
    test_input = torch.randn(2, 3, 224, 224)
    output = model_lightweight(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {count_parameters(model_lightweight):,}")
