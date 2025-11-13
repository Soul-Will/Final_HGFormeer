"""
Models Package Initialization

This file makes the 'models' directory a Python package
and exposes the key components for training and inference.
"""

# Main Encoder
from .hgformer import HGFormer3D

# Main Segmentation Model and Decoder Components
from .decoder import (
    HGFormer3D_ForSegmentation,
    UNetDecoder3D,
    DecoderBlock,
    AttentionGate3D
)

# Main Loss Function "Entry Points"
from .losses import (
    segmentation_loss,
    ssl_total_loss
)

# Individual losses (for logging or custom use)
from .losses import (
    dice_loss,
    focal_loss,
    boundary_weighted_dice_loss,
    boundary_loss_morphological
)

# Public API: what `from models import *` will import
__all__ = [
    # Models
    'HGFormer3D',
    'HGFormer3D_ForSegmentation',
    'UNetDecoder3D',
    'DecoderBlock',
    'AttentionGate3D',
    
    # Loss Entry Points
    'segmentation_loss',
    'ssl_total_loss',
    
    # Individual Losses
    'dice_loss',
    'focal_loss',
    'boundary_weighted_dice_loss',
    'boundary_loss_morphological'
]