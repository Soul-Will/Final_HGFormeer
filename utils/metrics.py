"""
Evaluation metrics for 3D segmentation
✅ FIXED: Uses skeletonize_3d for proper 3D centerline Dice
✅ FIXED: Added anisotropic spacing support
✅ FIXED: Robust error handling
"""

import numpy as np
import torch
from skimage.morphology import skeletonize  # Requires scikit-image
import logging

logger = logging.getLogger(__name__)


def dice_coefficient(pred, gt, smooth=1e-6):
    """
    Compute Dice score
    
    Args:
        pred: (D, H, W) binary prediction (numpy or torch)
        gt: (D, H, W) binary ground truth
        smooth: smoothing factor
    
    Returns:
        dice: scalar in [0, 1]
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def iou_score(pred, gt, smooth=1e-6):
    """
    Compute IoU (Jaccard Index)
    
    Args:
        pred: (D, H, W) binary prediction
        gt: (D, H, W) binary ground truth
        smooth: smoothing factor
    
    Returns:
        iou: scalar in [0, 1]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def centerline_dice(pred, gt, smooth=1e-6):
    """
    ✅ FIXED: 3D centerline Dice coefficient
    
    Uses skeletonize_3d for proper 3D skeleton extraction.
    Critical for evaluating vessel/neuron segmentation quality.
    
    Args:
        pred: (D, H, W) binary prediction
        gt: (D, H, W) binary ground truth
        smooth: smoothing factor
    
    Returns:
        cl_dice: scalar centerline Dice score
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    # Convert to binary
    pred_binary = (pred > 0).astype(bool)
    gt_binary = (gt > 0).astype(bool)
    
    # Check if volumes are empty
    if not pred_binary.any() and not gt_binary.any():
        return 1.0  # Both empty = perfect match
    if not pred_binary.any() or not gt_binary.any():
        return 0.0  # One empty = no match
    
    try:
        # ✅ CRITICAL FIX: Use 3D skeletonization
        pred_skel = skeletonize(pred_binary)
        gt_skel = skeletonize(gt_binary)
        
        # Compute Dice on skeletons
        intersection = (pred_skel & gt_skel).sum()
        union = pred_skel.sum() + gt_skel.sum()
        
        cl_dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return float(cl_dice)
    
    except Exception as e:
        logger.warning(f"Centerline Dice computation failed: {e}")
        return 0.0


def hausdorff_distance(pred, gt, spacing=(2.0, 1.0, 1.0)):
    """
    Compute Hausdorff distance with anisotropic spacing
    
    Args:
        pred: (D, H, W) binary prediction
        gt: (D, H, W) binary ground truth
        spacing: (z, y, x) physical spacing in microns
    
    Returns:
        hausdorff: scalar distance in microns
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    try:
        from scipy.spatial.distance import directed_hausdorff
        
        # Get boundary points with physical spacing
        pred_points = np.argwhere(pred > 0) * np.array(spacing)
        gt_points = np.argwhere(gt > 0) * np.array(spacing)
        
        # Check if volumes are empty
        if len(pred_points) == 0 or len(gt_points) == 0:
            return float('inf')
        
        # Compute directed Hausdorff in both directions
        hd_forward = directed_hausdorff(pred_points, gt_points)[0]
        hd_backward = directed_hausdorff(gt_points, pred_points)[0]
        
        # Return maximum (symmetric Hausdorff)
        hausdorff = max(hd_forward, hd_backward)
        
        return float(hausdorff)
    
    except Exception as e:
        logger.warning(f"Hausdorff distance computation failed: {e}")
        return float('inf')


def precision_recall_f1(pred, gt):
    """
    Compute precision, recall, and F1 score
    
    Args:
        pred: (D, H, W) binary prediction
        gt: (D, H, W) binary ground truth
    
    Returns:
        metrics: dict with precision, recall, f1
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    # Compute confusion matrix elements
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    tn = ((1 - pred) * (1 - gt)).sum()
    
    # Compute metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * tp / (2*tp + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity)
    }


def volume_similarity(pred, gt):
    """
    Compute volume similarity metrics
    
    Args:
        pred: (D, H, W) binary prediction
        gt: (D, H, W) binary ground truth
    
    Returns:
        metrics: dict with volume-based metrics
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    pred_volume = (pred > 0).sum()
    gt_volume = (gt > 0).sum()
    
    # Volume difference (relative)
    if gt_volume > 0:
        volume_diff = abs(pred_volume - gt_volume) / gt_volume
    else:
        volume_diff = float('inf') if pred_volume > 0 else 0.0
    
    # Volume overlap coefficient
    if pred_volume + gt_volume > 0:
        overlap_coef = 2 * (pred * gt).sum() / (pred_volume + gt_volume)
    else:
        overlap_coef = 0.0
    
    return {
        'pred_volume': int(pred_volume),
        'gt_volume': int(gt_volume),
        'volume_diff': float(volume_diff),
        'overlap_coefficient': float(overlap_coef)
    }


def compute_all_metrics(pred, gt, spacing=(2.0, 1.0, 1.0)):
    """
    ✅ FIXED: Compute all segmentation metrics
    
    Changes:
    - ✅ Uses skeletonize_3d for cl_dice
    - ✅ Added anisotropic spacing support for Hausdorff
    - ✅ Added comprehensive error handling
    - ✅ Returns all metrics in a single dict
    
    Args:
        pred: (D, H, W) binary prediction
        gt: (D, H, W) binary ground truth
        spacing: (z, y, x) physical spacing in microns
    
    Returns:
        metrics: dict with all metrics
    """
    metrics = {}
    
    # Basic overlap metrics
    try:
        metrics['dice'] = dice_coefficient(pred, gt)
    except Exception as e:
        logger.error(f"Dice computation failed: {e}")
        metrics['dice'] = 0.0
    
    try:
        metrics['iou'] = iou_score(pred, gt)
    except Exception as e:
        logger.error(f"IoU computation failed: {e}")
        metrics['iou'] = 0.0
    
    # ✅ FIXED: 3D centerline Dice
    try:
        metrics['cl_dice'] = centerline_dice(pred, gt)
    except Exception as e:
        logger.error(f"Centerline Dice computation failed: {e}")
        metrics['cl_dice'] = 0.0
    
    # Distance metrics
    try:
        metrics['hausdorff'] = hausdorff_distance(pred, gt, spacing)
    except Exception as e:
        logger.error(f"Hausdorff distance computation failed: {e}")
        metrics['hausdorff'] = float('inf')
    
    # Classification metrics
    try:
        pr_metrics = precision_recall_f1(pred, gt)
        metrics.update(pr_metrics)
    except Exception as e:
        logger.error(f"Precision/Recall computation failed: {e}")
        metrics.update({
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'specificity': 0.0
        })
    
    # Volume metrics
    try:
        vol_metrics = volume_similarity(pred, gt)
        metrics.update(vol_metrics)
    except Exception as e:
        logger.error(f"Volume metrics computation failed: {e}")
        metrics.update({
            'pred_volume': 0,
            'gt_volume': 0,
            'volume_diff': float('inf'),
            'overlap_coefficient': 0.0
        })
    
    return metrics


def compute_dice_score(pred, target, smooth=1e-6):
    """
    Simple Dice computation (for use in training loop)
    
    Args:
        pred: (B, D, H, W) or (D, H, W) - predictions
        target: same shape as pred - ground truth
        smooth: smoothing factor
    
    Returns:
        dice: scalar
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)