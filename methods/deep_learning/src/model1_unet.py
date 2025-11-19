"""
U-Net based segmentation model for copy-move forgery detection.
Uses pre-trained ResNet encoder with U-Net decoder architecture.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class UNetDecoderBlock(nn.Module):
    """
    Decoder block for U-Net architecture.
    Upsamples and concatenates with encoder features.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize decoder block.
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(UNetDecoderBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Parameters:
            x: Input tensor
            skip: Skip connection from encoder (optional)
            
        Returns:
            Output tensor
        """
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        if skip is not None:
            # Crop skip connection to match size if needed
            if x.shape != skip.shape:
                h, w = x.shape[2:]
                skip = nn.functional.adaptive_avg_pool2d(skip, (h, w))
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class UNetModel(nn.Module):
    """
    U-Net model with ResNet encoder for copy-move forgery detection.
    Uses pre-trained ResNet34 as encoder backbone.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        num_classes: int = 1,
        activation: Optional[str] = 'sigmoid'
    ):
        """
        Initialize U-Net model.
        
        Parameters:
            encoder_name: Name of encoder backbone (resnet34, resnet50, etc.)
            encoder_weights: Pre-trained weights ('imagenet' or None)
            num_classes: Number of output classes (1 for binary segmentation)
            activation: Activation function for output ('sigmoid' or None)
        """
        super(UNetModel, self).__init__()
        
        # Load pre-trained encoder
        if encoder_name == 'resnet34':
            encoder = models.resnet34(weights='IMAGENET1K_V1' if encoder_weights == 'imagenet' else None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            encoder = models.resnet50(weights='IMAGENET1K_V2' if encoder_weights == 'imagenet' else None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        # Extract encoder layers
        self.encoder1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.encoder2 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.encoder3 = encoder.layer2
        self.encoder4 = encoder.layer3
        self.encoder5 = encoder.layer4
        
        # Decoder blocks
        self.decoder4 = UNetDecoderBlock(encoder_channels[4] + encoder_channels[3], encoder_channels[3])
        self.decoder3 = UNetDecoderBlock(encoder_channels[3] + encoder_channels[2], encoder_channels[2])
        self.decoder2 = UNetDecoderBlock(encoder_channels[2] + encoder_channels[1], encoder_channels[1])
        self.decoder1 = UNetDecoderBlock(encoder_channels[1] + encoder_channels[0], encoder_channels[0])
        
        # Final upsampling to restore original spatial size (256x256 -> 512x512)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[0] // 2, encoder_channels[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[0] // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final_conv = nn.Conv2d(encoder_channels[0] // 2, num_classes, kernel_size=1)
        
        # Activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net model.
        
        Parameters:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Segmentation mask [B, 1, H, W]
        """
        # Encoder path
        e1 = self.encoder1(x)  # 64 channels
        e2 = self.encoder2(e1)  # 64 channels
        e3 = self.encoder3(e2)  # 128 channels
        e4 = self.encoder4(e3)  # 256 channels
        e5 = self.encoder5(e4)  # 512 channels
        
        # Decoder path with skip connections
        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        
        # Final upsampling to restore original spatial size (256x256 -> 512x512)
        d1_upsampled = nn.functional.interpolate(
            d1, scale_factor=2, mode='bilinear', align_corners=False
        )
        d1_upsampled = self.final_upsample(d1_upsampled)
        
        # Final output
        output = self.final_conv(d1_upsampled)
        
        if self.activation:
            output = self.activation(output)
        
        return output


def create_unet_model(config: dict) -> UNetModel:
    """
    Create U-Net model from configuration.
    
    Parameters:
        config: Model configuration dictionary
        
    Returns:
        Initialized U-Net model
    """
    model_config = config['model1_unet']
    model_settings = config['model_settings']
    
    # For binary segmentation, always use num_classes=1 regardless of config
    # The config might have num_classes=2 for other models, but U-Net needs 1 for binary segmentation
    model = UNetModel(
        encoder_name=model_config['encoder'],
        encoder_weights=model_config['encoder_weights'],
        num_classes=1,  # Binary segmentation always uses 1 channel output
        activation=model_config['activation']
    )
    
    return model

