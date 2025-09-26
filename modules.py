import torch
import torch.nn as nn
from torchvision import models as modules
import logging
import warnings

# Use logger initialized in main module
logger = logging.getLogger("modules")


class FCN_ResNet50(nn.Module):
    """Fully Convolutional Network with ResNet50 backbone for semantic segmentation.

    Adapts pre-trained ResNet50 for dense pixel-wise prediction by replacing
    fully connected layers with convolutional layers. Utilizes skip connections
    to preserve spatial details while maintaining semantic understanding.

    Args:
        classes (int): Number of segmentation classes in the dataset
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights (bool): Whether to initialize with pre-trained ImageNet weights

    Input Shape:
        (batch_size, channels, height, width)

    Output Shape:
        (batch_size, classes, height, width) - per-pixel class predictions
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing FCN ResNet50: {classes} classes, {channels} input channels"
        )

        # Load torchvision FCN with ResNet50 backbone
        self.module = modules.segmentation.fcn_resnet50(
            weights=(
                modules.segmentation.FCN_ResNet50_Weights.DEFAULT if weights else None
            ),
            num_classes=classes,
        )

        # Modify first convolution layer for non-RGB inputs
        if channels != 3:
            self.module.backbone.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FCN ResNet50 architecture.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation logits of shape (batch, classes, height, width)
        """
        # Extract main output from torchvision model dictionary
        output = self.module(x)
        return output["out"]


class FCN_ResNet101(nn.Module):
    """FCN with deeper ResNet101 backbone for improved segmentation accuracy.

    Similar architecture to FCN_ResNet50 but uses ResNet101 for enhanced
    feature extraction capability. Provides better performance on complex
    scenes at the cost of increased computational requirements.

    Args:
        classes (int): Number of segmentation classes
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights (bool): Whether to use pre-trained ImageNet weights

    Input Shape:
        (batch_size, channels, height, width)

    Output Shape:
        (batch_size, classes, height, width)
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing FCN ResNet101: {classes} classes, {channels} input channels"
        )

        # Initialize with deeper ResNet101 backbone
        self.module = modules.segmentation.fcn_resnet101(
            weights=(
                modules.segmentation.FCN_ResNet101_Weights.DEFAULT if weights else None
            ),
            num_classes=classes,
        )

        # Adjust input layer for non-standard channel counts
        if channels != 3:
            self.module.backbone.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation through FCN ResNet101.

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Class prediction logits of shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]


class DeepLabV3_ResNet50(nn.Module):
    """DeepLabV3 with ResNet50 backbone using Atrous Spatial Pyramid Pooling.

    Implements dilated convolutions through ASPP module to capture multi-scale
    contextual information. Particularly effective for scenes containing objects
    at varying scales and complex spatial relationships.

    Args:
        classes (int): Number of segmentation classes
        channels (int): Input image channels (typically 3 for RGB)
        weights (bool): Whether to load pre-trained ImageNet weights

    Input Shape:
        (batch_size, channels, height, width)

    Output Shape:
        (batch_size, classes, height, width)
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing DeepLabV3 ResNet50: {classes} classes, {channels} input channels"
        )

        # Load DeepLabV3 with ASPP module
        self.module = modules.segmentation.deeplabv3_resnet50(
            weights=(
                modules.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                if weights
                else None
            ),
            num_classes=classes,
        )

        # Modify input layer for non-RGB image formats
        if channels != 3:
            self.module.backbone.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepLabV3 network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions of shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]


class DeepLabV3_ResNet101(nn.Module):
    """DeepLabV3 with ResNet101 backbone for maximum segmentation performance.

    Combines the multi-scale processing capabilities of DeepLabV3's ASPP
    with the enhanced feature extraction of ResNet101. Represents the
    highest accuracy option with correspondingly higher computational cost.

    Args:
        classes (int): Number of segmentation classes
        channels (int): Input image channels (3 for RGB)
        weights (bool): Whether to use pre-trained weights

    Input Shape:
        (batch_size, channels, height, width)

    Output Shape:
        (batch_size, classes, height, width)
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing DeepLabV3 ResNet101: {classes} classes, {channels} input channels"
        )

        # Initialize most powerful DeepLabV3 variant
        self.module = modules.segmentation.deeplabv3_resnet101(
            weights=(
                modules.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
                if weights
                else None
            ),
            num_classes=classes,
        )

        # Handle non-RGB input configurations
        if channels != 3:
            self.module.backbone.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through DeepLabV3 ResNet101.

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Pixel-wise class predictions of shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]


class DeepLabV3_MobileNetV3Large(nn.Module):
    """DeepLabV3 with MobileNetV3-Large backbone optimized for efficiency.

    Employs depthwise separable convolutions and inverted residual blocks
    to minimize computational overhead while maintaining reasonable accuracy.
    Designed for mobile deployment and resource-constrained environments.

    Args:
        classes (int): Number of segmentation classes
        channels (int): Input image channels (3 for RGB)
        weights (bool): Whether to use pre-trained weights

    Input Shape:
        (batch_size, channels, height, width)

    Output Shape:
        (batch_size, classes, height, width)
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing DeepLabV3 MobileNetV3-Large: {classes} classes, {channels} input channels"
        )

        # Load mobile-optimized DeepLabV3 variant
        self.module = modules.segmentation.deeplabv3_mobilenet_v3_large(
            weights=(
                modules.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                if weights
                else None
            ),
            num_classes=classes,
        )

        # MobileNet uses different initial layer architecture
        if channels != 3:
            self.module.backbone.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation through MobileNet-based DeepLabV3.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation logits of shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]


class LRASPP_MobileNetV3Large(nn.Module):
    """Lite R-ASPP with MobileNetV3-Large for real-time segmentation.

    Implements a lightweight version of Atrous Spatial Pyramid Pooling
    optimized for speed over accuracy. Ideal for applications requiring
    real-time inference or high-throughput batch processing.

    Args:
        classes (int): Number of segmentation classes
        channels (int): Input image channels (3 for RGB)
        weights (bool): Whether to use pre-trained weights

    Input Shape:
        (batch_size, channels, height, width)

    Output Shape:
        (batch_size, classes, height, width)
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing LRASPP MobileNetV3-Large: {classes} classes, {channels} input channels"
        )

        # Load fastest available segmentation model
        self.module = modules.segmentation.lraspp_mobilenet_v3_large(
            weights=(
                modules.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                if weights
                else None
            ),
            num_classes=classes,
        )

        # Modify for non-RGB inputs
        if channels != 3:
            self.module.backbone.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Lite R-ASPP architecture.

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Class prediction scores of shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]

class UNet(nn.Module):
    """U-Net architecture for precise semantic segmentation.

    Classic encoder-decoder network with symmetric skip connections
    that preserve spatial information across resolution levels.
    Particularly effective for medical imaging and applications
    requiring precise boundary delineation.

    Args:
        classes (int): Number of segmentation classes
        channels (int): Input image channels (3 for RGB)
        base (int): Base number of feature channels (doubled at each level)
        weights (bool): Ignored - no pre-trained weights available

    Input Shape:
        (batch_size, channels, height, width) - dimensions should be divisible by 16

    Output Shape:
        (batch_size, classes, height, width)
    """

    def __init__(self, classes: int, channels: int = 3, base: int = 64, weights: bool = None) -> None:
        super().__init__()

        if weights is not None:
            warnings.warn(
                "Custom U-Net implementation has no pre-trained weights available. "
                "Parameter 'weights' will be ignored."
            )
            logger.warning(
                "U-Net implementation lacks pre-trained weights - ignoring weights parameter"
            )

        logger.info(
            f"Initializing U-Net: {classes} classes, {channels} input channels, {base} base filters"
        )

        # Encoder pathway with progressive feature extraction
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )

        # Bottleneck layer for deepest feature representation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 16, 3, padding=1),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 16, base * 16, 3, padding=1),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
        )

        # Decoder pathway with upsampling and feature fusion
        self.upconv4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base * 16, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )

        self.upconv3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Output layer for class prediction
        self.final = nn.Conv2d(base, classes, 1)

        # Downsampling operation
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute U-Net forward pass with skip connections.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions of shape (batch, classes, height, width)
        """
        # Contracting path with feature preservation
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bridge between encoder and decoder
        b = self.bottleneck(self.pool(e4))

        # Expanding path with skip connection fusion
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenate skip connection
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Concatenate skip connection
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Concatenate skip connection
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Concatenate skip connection
        d1 = self.dec1(d1)

        return self.final(d1)


class SegNet(nn.Module):
    """SegNet architecture with memory-efficient pooling indices upsampling.

    Uses max pooling indices for precise upsampling without requiring
    skip connections. Provides memory-efficient segmentation with
    excellent boundary preservation capabilities.

    Args:
        classes (int): Number of segmentation classes
        channels (int): Input image channels (3 for RGB)
        weights: Ignored - no pre-trained weights available

    Input Shape:
        (batch_size, channels, height, width) - dimensions should be divisible by 8

    Output Shape:
        (batch_size, classes, height, width)
    """

    def __init__(self, classes: int, channels: int = 3, weights = None) -> None:
        super().__init__()

        if weights is not None:
            warnings.warn(
                "Custom SegNet implementation has no pre-trained weights available. "
                "Parameter 'weights' will be ignored."
            )
            logger.warning(
                "SegNet implementation lacks pre-trained weights - ignoring weights parameter"
            )

        logger.info(
            f"Initializing SegNet: {classes} classes, {channels} input channels"
        )

        # Encoder blocks following VGG-style architecture
        self.encconv1 = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encconv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.encconv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks mirroring encoder structure
        self.decconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decconv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decconv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classes, 3, padding=1),
        )

        # Pooling operations with index preservation
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using pooling indices for precise upsampling.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions of shape (batch, classes, height, width)
        """
        # Encoder path with index preservation
        x1 = self.encconv1(x)
        x1pooled, idx1 = self.pool(x1)

        x2 = self.encconv2(x1pooled)
        x2pooled, idx2 = self.pool(x2)

        x3 = self.encconv3(x2pooled)
        x3pooled, idx3 = self.pool(x3)

        # Decoder path using preserved indices for upsampling
        x3up = self.unpool(x3pooled, idx3)
        x3dec = self.decconv3(x3up)

        x2up = self.unpool(x3dec, idx2)
        x2dec = self.decconv2(x2up)

        x1up = self.unpool(x2dec, idx1)
        output = self.decconv1(x1up)

        return output