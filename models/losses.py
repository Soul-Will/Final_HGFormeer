"""
All Loss Functions for SSL and Segmentation
✅ FIXED: GPU-only boundary loss (no CPU bottleneck)
✅ FIXED: Relative alpha decay
✅ FIXED: Configurable loss selection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

# Corrected logger initialization
logger = logging.getLogger(__name__)

# ============================================================================
# SEGMENTATION LOSSES
# ============================================================================

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice Loss for segmentation

    Args:
        pred: (B, C, D, H, W) - predicted logits
        target: (B, D, H, W) - ground truth labels [0, C-1]
        smooth: smoothing factor
    
    Returns:
        loss: scalar tensor
    """
    pred = F.softmax(pred, dim=1)
    B, C, D, H, W = pred.shape
    
    # One-hot encode target
    target_one_hot = F.one_hot(target.long(), num_classes=C)
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
    
    # Flatten spatial dimensions
    pred_flat = pred.reshape(B, C, -1)
    target_flat = target_one_hot.reshape(B, C, -1)
    
    # Compute Dice
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    
    return dice_loss

def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss to handle class imbalance

    Args:
        pred: (B, C, D, H, W)
        target: (B, D, H, W)
        alpha: weighting factor
        gamma: focusing parameter
    
    Returns:
        loss: scalar tensor
    """
    B, C, D, H, W = pred.shape
    
    # Reshape
    pred_reshaped = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
    target_reshaped = target.reshape(-1)
    
    # Get probabilities
    pred_prob = F.softmax(pred_reshaped, dim=1)
    target_one_hot = F.one_hot(target_reshaped.long(), num_classes=C).float()
    
    pt = (pred_prob * target_one_hot).sum(dim=1)
    
    # Focal term
    focal_term = (1 - pt) ** gamma
    
    # Cross-entropy
    ce_loss = F.cross_entropy(pred_reshaped, target_reshaped.long(), reduction='none')
    
    # Focal loss
    loss = alpha * focal_term * ce_loss
    
    return loss.mean()

def boundary_weighted_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 2.0,
    kernel_size: int = 3,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    ✅ FIXED: Fast GPU-only boundary-weighted Dice loss

    Uses morphological operations (max pooling) to extract boundary,
    then computes Dice loss with emphasis on boundary pixels.
    
    Args:
        pred: (B, C, D, H, W)
        target: (B, D, H, W)
        boundary_weight: Emphasis factor for boundary (2.0 = 2x weight)
        kernel_size: Size of morphological kernel (3 or 5)
        smooth: Smoothing term
    
    Returns:
        loss: scalar tensor
    """
    pred_prob = F.softmax(pred, dim=1)[:, 1]  # (B, D, H, W)
    target_binary = (target > 0).float()
    
    # 1. Standard Dice (full volume)
    intersection = (pred_prob * target_binary).sum()
    union = pred_prob.sum() + target_binary.sum()
    dice_full = (2. * intersection + smooth) / (union + smooth)
    
    # 2. Extract boundary using morphological ops
    target_float = target_binary.unsqueeze(1)  # (B, 1, D, H, W)
    
    # Dilate
    dilated = F.max_pool3d(
        target_float,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )
    # Erode
    eroded = 1 - F.max_pool3d(
        1 - target_float,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )
    
    # Boundary = dilated - eroded
    boundary_mask = (dilated - eroded).squeeze(1)  # (B, D, H, W)
    
    # 3. Dice on boundary only
    boundary_pred = pred_prob * boundary_mask
    boundary_target = target_binary * boundary_mask
    
    intersection_boundary = (boundary_pred * boundary_target).sum()
    union_boundary = boundary_pred.sum() + boundary_target.sum()
    
    # Avoid division by zero if no boundary
    if union_boundary > 0:
        dice_boundary = (2. * intersection_boundary + smooth) / (union_boundary + smooth)
    else:
        dice_boundary = dice_full  # Fallback to full dice
    
    # 4. Combine
    total_dice = (dice_full + boundary_weight * dice_boundary) / (1 + boundary_weight)
    
    return 1 - total_dice

def boundary_loss_morphological(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 3
) -> torch.Tensor:
    """
    ✅ ALTERNATIVE: Boundary-weighted BCE loss

    Faster than weighted Dice, good for quick prototyping.
    
    Args:
        pred: (B, C, D, H, W)
        target: (B, D, H, W)
        kernel_size: Morphological kernel size
    
    Returns:
        loss: scalar tensor
    """
    pred_prob = F.softmax(pred, dim=1)[:, 1]
    target_float = (target > 0).float().unsqueeze(1)
    
    # Extract boundary
    dilated = F.max_pool3d(
        target_float,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )
    eroded = 1 - F.max_pool3d(
        1 - target_float,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )
    boundary = (dilated - eroded).squeeze(1)
    
    # Weight BCE by boundary
    boundary_weight = boundary + 0.01
    target_binary = (target > 0).float()
    
    loss = F.binary_cross_entropy(
        pred_prob,
        target_binary,
        weight=boundary_weight,
        reduction='mean'
    )
    return loss

def segmentation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha_start: float = 1.0,
    epoch: int = 0,
    max_epochs: int = 500,
    loss_config: dict = None
) -> tuple:
    """
    ✅ FIXED: Combined segmentation loss with adaptive weighting

    Changes:
    - ✅ Replaced CPU-bound boundary_loss with GPU-only version
    - ✅ Fixed alpha decay to be relative to max_epochs
    - ✅ Made loss selection configurable
    
    Args:
        pred: (B, C, D, H, W)
        target: (B, D, H, W)
        alpha_start: Initial weight for Dice+Focal
        epoch: Current epoch
        max_epochs: Total training epochs
        loss_config: Optional dict with:
            - 'boundary_type': 'weighted_dice' or 'morphological' or None
            - 'boundary_weight': emphasis factor (2.0 default)
            - 'alpha_min': minimum alpha value (0.01 default)
            - 'decay_type': 'linear' or 'cosine' (linear default)
    
    Returns:
        total_loss: scalar tensor
        loss_dict: dict with individual losses
    """
    loss_config = loss_config or {}
    
    # Compute individual losses
    L_dice = dice_loss(pred, target)
    L_focal = focal_loss(pred, target)
    
    # ✅ FIXED: Use fast boundary loss
    boundary_type = loss_config.get('boundary_type', 'weighted_dice')
    if boundary_type == 'weighted_dice':
        L_boundary = boundary_weighted_dice_loss(
            pred, target,
            boundary_weight=loss_config.get('boundary_weight', 2.0)
        )
    elif boundary_type == 'morphological':
        L_boundary = boundary_loss_morphological(pred, target)
    elif boundary_type is None or boundary_type == 'none':
        L_boundary = torch.tensor(0.0, device=pred.device)
    else:
        raise ValueError(f"Unknown boundary_type: {boundary_type}")
    
    # Dice-Focal combination
    L_dice_focal = L_dice + L_focal
    
    # ✅ FIXED: Adaptive weighting relative to max_epochs
    alpha_min = loss_config.get('alpha_min', 0.01)
    decay_type = loss_config.get('decay_type', 'linear')
    
    if decay_type == 'linear':
        alpha = alpha_start * (1.0 - epoch / max_epochs)
    
    elif decay_type == 'cosine':
        alpha = alpha_min + 0.5 * (alpha_start - alpha_min) * (
            1 + math.cos(math.pi * epoch / max_epochs)
        )
    
    elif decay_type == 'exponential':
        alpha = alpha_start * math.exp(-5.0 * epoch / max_epochs)
    
    else:
        # Fallback to linear
        logger.warning(f"Unknown decay_type '{decay_type}', using linear")
        alpha = alpha_start * (1.0 - epoch / max_epochs)
    
    # Clamp to minimum
    alpha = max(alpha_min, alpha)
    
    # Total loss
    total_loss = alpha * L_dice_focal + (1 - alpha) * L_boundary
    
    loss_dict = {
        'total': total_loss.item(),
        'dice': L_dice.item(),
        'focal': L_focal.item(),
        'boundary': L_boundary.item(),
        'alpha': alpha
    }
    
    return total_loss, loss_dict


# ============================================================================
# SSL LOSSES (unchanged, but included for completeness)
# ============================================================================

def masked_inpainting_loss(model: nn.Module,
                            volume: torch.Tensor,
                            mask_ratio: float = 0.15) -> torch.Tensor:
    """
    Masked Volume Inpainting Loss
    
    Args:
        model: HGFormer3D model with reconstruct() method
        volume: (B, 1, D, H, W)
        mask_ratio: fraction to mask
    
    Returns:
        loss: scalar tensor
    """
    B, C, D, H, W = volume.shape
    device = volume.device
    
    # Create random mask
    mask = (torch.rand(B, D, H, W, device=device) < mask_ratio).float()
    mask = mask.unsqueeze(1)  # (B, 1, D, H, W)
    
    # Apply mask
    masked_volume = volume * (1 - mask)
    
    # Forward pass
    reconstructed = model.reconstruct(masked_volume)
    
    # Loss only on masked regions
    loss = F.l1_loss(reconstructed * mask, volume * mask)
    
    return loss


def rotation_loss(model: nn.Module, volume: torch.Tensor) -> torch.Tensor:
    """
    Rotation Prediction Loss
    
    Args:
        model: HGFormer3D model with predict_rotation() method
        volume: (B, 1, D, H, W)
    
    Returns:
        loss: scalar tensor
    """
    B = volume.shape[0]
    device = volume.device
    
    # Random rotations (0°, 90°, 180°, 270°)
    rotation_angles = torch.randint(0, 4, (B,), device=device)
    
    # Apply rotations
    rotated_volumes = []
    for i in range(B):
        angle = rotation_angles[i]
        rotated = rotate_volume_3d(volume[i], angle * 90)
        rotated_volumes.append(rotated)
    
    rotated_volumes = torch.stack(rotated_volumes)
    
    # Predict rotation
    rotation_logits = model.predict_rotation(rotated_volumes)
    
    # Cross-entropy loss
    loss = F.cross_entropy(rotation_logits, rotation_angles)
    
    return loss


def rotate_volume_3d(volume: torch.Tensor, angle_degrees: int) -> torch.Tensor:
    """
    Rotate volume around z-axis
    
    Args:
        volume: (C, D, H, W)
        angle_degrees: 0, 90, 180, or 270
    
    Returns:
        rotated: (C, D, H, W)
    """
    if angle_degrees == 90:
        return volume.rot90(k=1, dims=(2, 3))
    elif angle_degrees == 180:
        return volume.rot90(k=2, dims=(2, 3))
    elif angle_degrees == 270:
        return volume.rot90(k=3, dims=(2, 3))
    else:
        return volume


def contrastive_loss(model: nn.Module,
                     volume: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
    """
    SimCLR-style Contrastive Learning Loss
    
    Args:
        model: HGFormer3D model with encode() method
        volume: (B, 1, D, H, W)
        temperature: temperature parameter
    
    Returns:
        loss: scalar tensor
    """
    B = volume.shape[0]
    device = volume.device
    
    # Create two augmented views
    view1 = augment_volume(volume)
    view2 = augment_volume(volume)
    
    # Encode both views
    z1 = model.encode(view1)
    z2 = model.encode(view2)
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2B, C)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.T) / temperature  # (2B, 2B)
    
    # Create labels
    labels = torch.cat([
        torch.arange(B, 2*B, device=device),
        torch.arange(0, B, device=device)
    ])
    
    # Mask self-similarities
    mask = torch.eye(2*B, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    
    # NT-Xent loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


def augment_volume(volume: torch.Tensor) -> torch.Tensor:
    """
    Apply random augmentations
    
    Args:
        volume: (B, 1, D, H, W)
    
    Returns:
        augmented: (B, 1, D, H, W)
    """
    # Random noise
    noise = torch.randn_like(volume) * 0.1
    volume = volume + noise
    
    # Random contrast
    contrast_factor = 0.8 + 0.4 * torch.rand(1, device=volume.device)
    volume = volume * contrast_factor
    
    # Random brightness
    brightness = 0.1 * torch.randn(1, device=volume.device)
    volume = volume + brightness
    
    return volume


def weak_label_loss(model: nn.Module,
                    volume: torch.Tensor,
                    marker_type: torch.Tensor) -> torch.Tensor:
    """
    Weak Label Classification Loss
    
    Args:
        model: HGFormer3D model with classify_marker() method
        volume: (B, 1, D, H, W)
        marker_type: (B,) integer labels [0,1,2,3]
    
    Returns:
        loss: scalar tensor
    """
    marker_logits = model.classify_marker(volume)
    loss = F.cross_entropy(marker_logits, marker_type)
    return loss


def ssl_total_loss(model: nn.Module,
                   volume: torch.Tensor,
                   marker_type: torch.Tensor,
                   lambda_inpaint: float = 1.0,
                   lambda_rotation: float = 1.0,
                   lambda_contrastive: float = 1.0,
                   lambda_label: float = 1.0,
                   mask_ratio: float = 0.15,
                   temperature: float = 0.07) -> tuple:
    """
    Combined SSL loss with 4 components
    
    Args:
        model: HGFormer3D model
        volume: (B, 1, D, H, W)
        marker_type: (B,)
        lambda_*: weights for each loss
        mask_ratio: masking ratio
        temperature: temperature for contrastive
    
    Returns:
        total_loss: scalar tensor
        loss_dict: dict with individual losses
    """
    # Compute all losses
    L_inpaint = masked_inpainting_loss(model, volume, mask_ratio)
    L_rot = rotation_loss(model, volume)
    L_contrast = contrastive_loss(model, volume, temperature)
    L_label = weak_label_loss(model, volume, marker_type)
    
    # Weighted combination
    total_loss = (lambda_inpaint * L_inpaint +
                  lambda_rotation * L_rot +
                  lambda_contrastive * L_contrast +
                  lambda_label * L_label)
    
    loss_dict = {
        'total': total_loss.item(),
        'inpaint': L_inpaint.item(),
        'rotation': L_rot.item(),
        'contrastive': L_contrast.item(),
        'label': L_label.item()
    }
    
    return total_loss, loss_dict