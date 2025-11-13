"""
Utils Package Initialization
✅ FIXED: Aligned with the new metadata-driven data pipeline.
✅ FIXED: Exports all current augmentation, data_loader, metrics, and visualization functions.
❌ REMOVED: Obsolete imports from `preprocessing.py`. That logic is now inside `scripts/prepare_data.py`.
"""

# Data loading classes
from .data_loader import (
    VolumeDataset3D,
    SELMA3DDataset,
    validate_metadata_format  # Useful helper for scripts
)

# Augmentation functions (torchio-based)
from .augmentations import (
    get_ssl_transforms,
    get_finetune_transforms,
    get_val_transforms
)

# Metrics functions
from .metrics import (
    compute_all_metrics,  # The main evaluation entry point
    dice_coefficient,
    iou_score,
    centerline_dice,
    hausdorff_distance,
    precision_recall_f1,
    volume_similarity
)

# Visualization functions
# (Assuming visualization.py is in this directory)
try:
    from .visualization import (
        visualize_volume,
        plot_segmentation_results,
        save_prediction_as_tif
    )
except ImportError:
    pass


# Public API: what `from utils import *` will import
__all__ = [
    # data_loader.py
    'VolumeDataset3D',
    'SELMA3DDataset',
    'validate_metadata_format',
    
    # augmentations.py
    'get_ssl_transforms',
    'get_finetune_transforms',
    'get_val_transforms',
    
    # metrics.py
    'compute_all_metrics',
    'dice_coefficient',
    'iou_score',
    'centerline_dice',
    'hausdorff_distance',
    'precision_recall_f1',
    'volume_similarity',
    
    # visualization.py
    'visualize_volume',
    'plot_segmentation_results',
    'save_prediction_as_tif'
]