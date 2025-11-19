"""
ResNet-based encoder-decoder model for copy-move forgery detection.
Uses pre-trained ResNet50 as encoder with custom decoder.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional


class ResNetDecoderBlock(nn.Module):
    """
    Decoder block for ResNet encoder-decoder architecture.
    Upsamples features using transposed convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize decoder block.
        
        Parameters:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(ResNetDecoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Parameters:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.block(x)


class ResNetEncoderDecoder(nn.Module):
    """
    ResNet-based encoder-decoder model for copy-move forgery detection.
    Uses pre-trained ResNet50 as encoder with custom decoder architecture.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        backbone_weights: str = 'imagenet',
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        num_classes: int = 1,
        activation: Optional[str] = 'sigmoid'
    ):
        """
        Initialize ResNet encoder-decoder model.
        
        Parameters:
            backbone: ResNet backbone name ('resnet50', 'resnet101', etc.)
            backbone_weights: Pre-trained weights ('imagenet' or None)
            decoder_channels: List of channel sizes for decoder blocks
            num_classes: Number of output classes (1 for binary segmentation)
            activation: Activation function for output ('sigmoid' or None)
        """
        super(ResNetEncoderDecoder, self).__init__()
        
        # Load pre-trained encoder
        if backbone == 'resnet50':
            encoder = models.resnet50(weights='IMAGENET1K_V2' if backbone_weights == 'imagenet' else None)
            encoder_out_channels = 2048
        elif backbone == 'resnet101':
            encoder = models.resnet101(weights='IMAGENET1K_V2' if backbone_weights == 'imagenet' else None)
            encoder_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract encoder layers
        self.encoder = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4
        )
        
        # Decoder blocks
        decoder_layers = []
        in_channels = encoder_out_channels
        
        for out_channels in decoder_channels:
            decoder_layers.append(ResNetDecoderBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Final output layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        
        # Activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder-decoder model.
        
        Parameters:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Segmentation mask [B, 1, H, W]
        """
        # Encoder path
        features = self.encoder(x)
        
        # Decoder path
        decoded = self.decoder(features)
        
        # Upsample to original input size if needed
        if decoded.shape[2:] != x.shape[2:]:
            decoded = nn.functional.interpolate(
                decoded,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Final output
        output = self.final_conv(decoded)
        
        if self.activation:
            output = self.activation(output)
        
        return output


def create_resnet_model(config: dict) -> ResNetEncoderDecoder:
    """
    Create ResNet encoder-decoder model from configuration.
    
    Parameters:
        config: Model configuration dictionary
        
    Returns:
        Initialized ResNet encoder-decoder model
    """
    model_config = config['model2_resnet']
    model_settings = config['model_settings']
    
    model = ResNetEncoderDecoder(
        backbone=model_config['backbone'],
        backbone_weights=model_config['backbone_weights'],
        decoder_channels=model_config['decoder_channels'],
        num_classes=model_settings['num_classes'],
        activation=model_config['activation']
    )
    
    return model

