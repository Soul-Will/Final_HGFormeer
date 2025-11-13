"""
Data preprocessing utilities
"""
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import zoom

def load_brain_volume(brain_folder):
    """
    Load all slices from a brain folder and stack into 3D volume
    
    Args:
        brain_folder: path to folder containing TIF slices
    
    Returns:
        volume: (D, H, W) numpy array
        metadata: dict with spacing info
    """
    # Get all TIF files
    slice_files = sorted(Path(brain_folder).glob("*.tif"))
    
    # Sort by slice number (important!)
    slice_files = sorted(slice_files, 
                        key=lambda x: int(x.stem.split('_')[-1]))
    
    # Load first slice to get dimensions
    first_slice = np.array(Image.open(slice_files[0]))
    H, W = first_slice.shape
    D = len(slice_files)
    
    # Preallocate volume
    volume = np.zeros((D, H, W), dtype=np.float32)
    
    # Load all slices
    print(f"Loading {D} slices...")
    for i, slice_file in enumerate(slice_files):
        slice_img = np.array(Image.open(slice_file))
        volume[i] = slice_img
        
        if (i+1) % 100 == 0:
            print(f"Loaded {i+1}/{D} slices")
    
    # Metadata
    metadata = {
        'shape': (D, H, W),
        'z_spacing': 2.0,  # Typical LSM z-spacing (micrometers)
        'xy_spacing': 1.0,
        'dtype': volume.dtype
    }
    
    return volume, metadata

def normalize_spacing(volume, z_spacing=2.0, xy_spacing=1.0, 
                     target_spacing=1.0):
    """
    Resample volume to isotropic spacing
    
    Args:
        volume: (D, H, W) array
        z_spacing: spacing between slices (micrometers)
        xy_spacing: in-plane pixel spacing
        target_spacing: desired isotropic spacing
    
    Returns:
        resampled_volume: (D', H', W') array
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors
    zoom_z = z_spacing / target_spacing
    zoom_xy = xy_spacing / target_spacing
    
    # Resample
    resampled = zoom(volume, 
                    (zoom_z, zoom_xy, zoom_xy), 
                    order=1)  # Linear interpolation
    
    print(f"Original shape: {volume.shape}")
    print(f"Resampled shape: {resampled.shape}")
    
    return resampled

def create_data_splits(labeled_patches, val_ratio=0.1, test_ratio=0.1, stratify_by='marker_type'):
    """
    Split labeled data with stratification
    
    Args:
        labeled_patches: list of patch paths
        stratify_by: 'marker_type' to ensure balanced splits
    
    Returns:
        train_set, val_set, test_set
    """
    from sklearn.model_selection import train_test_split
    
    # Extract marker types
    marker_types = [get_marker_type(p) for p in labeled_patches]
    
    # First split: separate test set
    train_val, test, train_val_markers, test_markers = train_test_split(
        labeled_patches, marker_types,
        test_size=test_ratio,
        stratify=marker_types,
        random_state=42
    )
    
    # Second split: separate validation from training
    train, val, _, _ = train_test_split(
        train_val, train_val_markers,
        test_size=val_ratio / (1 - test_ratio),
        stratify=train_val_markers,
        random_state=42
    )
    
    print(f"Train: {len(train)} patches")
    print(f"Val: {len(val)} patches")
    print(f"Test: {len(test)} patches")
    
    return train, val, test

def get_marker_type(patch_path):
    """Extract marker type from path"""
    path_str = str(patch_path).lower()
    if 'cfos' in path_str:
        return 0
    elif 'vessel' in path_str:
        return 1
    elif 'nucleus' in path_str:
        return 2
    elif 'ad_plaque' in path_str:
        return 3
    return 0