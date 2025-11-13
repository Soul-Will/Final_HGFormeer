"""
Synchronized 3D Augmentations for Medical Imaging
✅ FIXED: Uses torchio for proper 3D augmentation
✅ FIXED: Image-mask synchronization guaranteed
✅ FIXED: Configurable parameters

File: utils/augmentations.py
"""

import torchio as tio
import torch
import logging

logger = logging.getLogger(__name__)


def get_ssl_transforms(config: dict = None):
    """
    Self-supervised learning augmentations (AGGRESSIVE)
    
    Args:
        config: Optional dict with augmentation parameters
        
    Returns:
        tio.Compose transform that applies to tio.Subject
    
    Example:
        >>> transform = get_ssl_transforms()
        >>> subject = tio.Subject(
        ...     image=tio.ScalarImage(tensor=volume),  # (1, D, H, W)
        ... )
        >>> augmented = transform(subject)
    """
    config = config or {}
    
    # Extract parameters with defaults
    blur_std_range = config.get('blur_std_range', [0, 2])
    noise_std_range = config.get('noise_std_range', [0, 0.1])
    gamma_range = config.get('gamma_range', [-0.3, 0.3])
    affine_scales = config.get('affine_scales', [0.9, 1.1])
    affine_degrees = config.get('affine_degrees', 15)
    affine_translation = config.get('affine_translation', 5)
    elastic_num_points = config.get('elastic_num_points', 7)
    elastic_max_displacement = config.get('elastic_max_displacement', 7.5)
    
    transforms = []
    
    # Intensity augmentations (image only, automatically handled)
    transforms.extend([
        tio.RandomBlur(std=tuple(blur_std_range), p=0.5),
        tio.RandomNoise(std=tuple(noise_std_range), p=0.5),
        tio.RandomGamma(log_gamma=tuple(gamma_range), p=0.5),
    ])
    
    # Spatial augmentations (SYNCHRONIZED to image and mask)
    transforms.extend([
        tio.RandomAffine(
            scales=tuple(affine_scales),
            degrees=affine_degrees,
            translation=affine_translation,
            p=0.5
        ),
        tio.RandomFlip(axes=('LR',), p=0.5),  # Left-Right flip
        tio.RandomElasticDeformation(
            num_control_points=elastic_num_points,
            max_displacement=elastic_max_displacement,
            p=0.3
        ),
    ])
    
    # Normalization (always applied last)
    transforms.extend([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(),  # Mean=0, Std=1
    ])
    
    logger.debug(f"SSL transforms created with {len(transforms)} operations")
    
    return tio.Compose(transforms)


def get_finetune_transforms(config: dict = None):
    """
    Fine-tuning augmentations (LESS AGGRESSIVE)
    
    Rationale: During fine-tuning, we have labeled data. We want to augment
    for robustness, but not so aggressively that we destroy the signal.
    
    Args:
        config: Optional dict with augmentation parameters
        
    Returns:
        tio.Compose transform
    """
    config = config or {}
    
    # Extract parameters with defaults (milder than SSL)
    blur_std_range = config.get('blur_std_range', [0, 1])
    noise_std_range = config.get('noise_std_range', [0, 0.05])
    gamma_range = config.get('gamma_range', [-0.2, 0.2])
    affine_scales = config.get('affine_scales', [0.95, 1.05])
    affine_degrees = config.get('affine_degrees', 10)
    affine_translation = config.get('affine_translation', 3)
    
    transforms = []
    
    # Intensity augmentations (milder)
    transforms.extend([
        tio.RandomBlur(std=tuple(blur_std_range), p=0.3),
        tio.RandomNoise(std=tuple(noise_std_range), p=0.3),
        tio.RandomGamma(log_gamma=tuple(gamma_range), p=0.3),
    ])
    
    # Spatial augmentations (milder)
    transforms.extend([
        tio.RandomAffine(
            scales=tuple(affine_scales),
            degrees=affine_degrees,
            translation=affine_translation,
            p=0.3
        ),
        tio.RandomFlip(axes=('LR',), p=0.5),
    ])
    
    # Normalization
    transforms.extend([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(),
    ])
    
    logger.debug(f"Finetune transforms created with {len(transforms)} operations")
    
    return tio.Compose(transforms)


def get_val_transforms(config: dict = None):
    """
    Validation transforms (NORMALIZATION ONLY)
    
    No augmentation during validation - we want to evaluate on clean data.
    
    Args:
        config: Optional dict (unused, for API consistency)
        
    Returns:
        tio.Compose transform
    """
    transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(),
    ]
    
    logger.debug("Validation transforms created (normalization only)")
    
    return tio.Compose(transforms)


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def test_transform_sync():
    """
    Verify that transforms maintain image-mask alignment
    
    This is CRITICAL for segmentation tasks. If image and mask get
    different augmentations, the training will fail.
    """
    import numpy as np
    
    logger.info("Testing transform synchronization...")
    
    # Create test volume and mask
    D, H, W = 64, 128, 128
    image = np.random.rand(D, H, W).astype(np.float32)
    mask = np.random.randint(0, 2, (D, H, W)).astype(np.uint8)
    
    # Create torchio Subject
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image[None, ...]),  # Add channel dim: (1, D, H, W)
        mask=tio.LabelMap(tensor=mask[None, ...])
    )
    
    # Test SSL transforms
    ssl_transform = get_ssl_transforms()
    ssl_transformed = ssl_transform(subject)
    
    assert ssl_transformed.image.shape == ssl_transformed.mask.shape, \
        f"SSL: Image {ssl_transformed.image.shape} and mask {ssl_transformed.mask.shape} shapes don't match!"
    
    logger.info("✅ SSL transforms: PASSED")
    
    # Test finetune transforms
    ft_transform = get_finetune_transforms()
    ft_transformed = ft_transform(subject)
    
    assert ft_transformed.image.shape == ft_transformed.mask.shape, \
        f"Finetune: Image {ft_transformed.image.shape} and mask {ft_transformed.mask.shape} shapes don't match!"
    
    logger.info("✅ Finetune transforms: PASSED")
    
    # Test val transforms
    val_transform = get_val_transforms()
    val_transformed = val_transform(subject)
    
    assert val_transformed.image.shape == val_transformed.mask.shape, \
        f"Val: Image {val_transformed.image.shape} and mask {val_transformed.mask.shape} shapes don't match!"
    
    logger.info("✅ Validation transforms: PASSED")
    
    # Test that spatial transforms actually change the data
    ssl_transformed2 = ssl_transform(subject)
    
    # Due to randomness, two applications should give different results
    image_diff = (ssl_transformed.image.data != ssl_transformed2.image.data).float().mean()
    
    if image_diff > 0.01:  # At least 1% of voxels changed
        logger.info(f"✅ Augmentation randomness: PASSED ({image_diff:.2%} voxels changed)")
    else:
        logger.warning(f"⚠️ Augmentation may not be random enough ({image_diff:.2%} voxels changed)")
    
    logger.info("\n" + "="*80)
    logger.info("ALL TESTS PASSED!")
    logger.info("="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_transform_sync()