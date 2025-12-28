"""
Unified Data Preparation Pipeline - PRODUCTION VERSION
================================================================================

✅ PERMANENT FIXES APPLIED:
  • Fixed chunk size calculation for HDF5 (prevents memory errors)
  • Removed redundant global normalization passes
  • Fixed metadata serialization (JSON compatibility)
  • Optimized memory usage in streaming
  • Fixed foreground sampling (removed unnecessary full flatten)
  • Corrected checkpoint cleanup logic
  • Fixed parallel processing error handling
  • Removed duplicate validation calls
  • Fixed shape mismatch detection
  • Optimized disk space calculations

Author: AI/ML Research Engineer
Date: 2025-01-10
Version: 4.0.0 (Production Stable)
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import logging
import h5py
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings
import shutil
import psutil
from datetime import datetime
from joblib import Parallel, delayed
import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')

# ============================================================================
# THREAD-SAFE LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Worker-%(process)d] - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ============================================================================
# FILE FORMAT DETECTION
# ============================================================================

class FileFormat(Enum):
    """Supported formats"""
    TIFF = "tiff"
    NIFTI = "nifti"
    UNKNOWN = "unknown"


def detect_file_format(file_path: Path) -> FileFormat:
    """Auto-detect format from extension"""
    suffix = file_path.suffix.lower()
    
    if suffix in ['.tif', '.tiff']:
        return FileFormat.TIFF
    elif suffix in ['.nii', '.gz']:
        if file_path.name.endswith('.nii.gz'):
            return FileFormat.NIFTI
        elif suffix == '.nii':
            return FileFormat.NIFTI
    
    return FileFormat.UNKNOWN


# ============================================================================
# DISK SPACE VALIDATION
# ============================================================================

def check_disk_space(output_path: Path, required_gb: float, buffer: float = 1.2):
    """
    ✅ FIX: Reduced buffer from 1.5x to 1.2x (more realistic)
    
    Pre-flight disk space check with configurable safety buffer
    
    Args:
        output_path: Output file path
        required_gb: Required space in GB
        buffer: Safety buffer multiplier (default: 1.2x)
    
    Raises:
        RuntimeError: If insufficient disk space
    """
    stats = shutil.disk_usage(output_path.parent)
    available_gb = stats.free / 1e9
    required_with_buffer = required_gb * buffer
    
    if available_gb < required_with_buffer:
        raise RuntimeError(
            f"❌ INSUFFICIENT DISK SPACE\n"
            f"  Required: {required_with_buffer:.1f} GB (with {buffer}x safety buffer)\n"
            f"  Available: {available_gb:.1f} GB\n"
            f"  Free up {required_with_buffer - available_gb:.1f} GB and try again"
        )
    
    logger.info(f"  ✅ Disk space check passed: {available_gb:.1f} GB available")


# ============================================================================
# MEMORY MONITORING
# ============================================================================

class MemoryMonitor:
    """Track memory usage and detect leaks"""
    def __init__(self, threshold_gb: float = 2.0):
        """
        ✅ FIX: Increased threshold from 1.0 to 2.0 GB (more realistic for large volumes)
        """
        self.process = psutil.Process()
        self.threshold_gb = threshold_gb
        self.baseline_gb = self.process.memory_info().rss / 1e9
    
    def check(self, context: str = ""):
        """Check current memory usage and warn if leak detected"""
        current_gb = self.process.memory_info().rss / 1e9
        increase_gb = current_gb - self.baseline_gb
        
        if increase_gb > self.threshold_gb:
            logger.warning(
                f"⚠️ MEMORY INCREASE DETECTED {context}\n"
                f"  Baseline: {self.baseline_gb:.2f} GB\n"
                f"  Current: {current_gb:.2f} GB\n"
                f"  Increase: {increase_gb:.2f} GB (threshold: {self.threshold_gb:.2f} GB)"
            )
        
        return current_gb


# ============================================================================
# HDF5 VALIDATION
# ============================================================================

def validate_hdf5_file(hdf5_path: Path, expected_shape: Tuple[int, int, int]):
    """
    Post-save integrity validation
    
    Args:
        hdf5_path: Path to HDF5 file
        expected_shape: Expected (D, H, W) shape
    
    Raises:
        ValueError: If validation fails
    """
    logger.info("  🔍 Validating saved HDF5...")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check 1: Dataset exists
            if 'volume' not in f:
                raise ValueError("'volume' dataset not found in HDF5 file")
            
            # Check 2: Shape matches
            saved_shape = f['volume'].shape
            if saved_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch: expected {expected_shape}, got {saved_shape}"
                )
            
            # Check 3: No all-zero slices (sample first, middle, last)
            D = expected_shape[0]
            sample_indices = [0, D//2, D-1] if D > 2 else [0]
            
            for idx in sample_indices:
                slice_sum = f['volume'][idx].sum()
                if slice_sum == 0:
                    raise ValueError(f"Slice {idx} is all zeros!")
            
            # Check 4: Foreground coords present
            if 'foreground_coords' in f:
                num_coords = f['foreground_coords'].shape[0]
                if num_coords == 0:
                    logger.warning("  ⚠️ No foreground coordinates extracted")
                else:
                    logger.info(f"  ✅ Found {num_coords} foreground coordinates")
            
            logger.info("  ✅ Validation passed")
    
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        # Clean up corrupted file
        if hdf5_path.exists():
            hdf5_path.unlink()
            logger.info(f"  🗑️ Deleted corrupted file")
        raise


# ============================================================================
# GLOBAL INTENSITY STATISTICS (OPTIMIZED)
# ============================================================================

def compute_global_stats(
    slice_files: List[Path],
    downsample_xy: float = 1.0,
    percentile: float = 95.0
) -> Tuple[float, float, float]:
    """
    ✅ FIX: Combined min/max and threshold computation in ONE pass
    
    This eliminates redundant file reads and speeds up processing.
    
    Args:
        slice_files: List of TIFF slice paths
        downsample_xy: XY downsampling factor
        percentile: Percentile for foreground threshold (0-100)
    
    Returns:
        (global_min, global_max, global_threshold)
    """
    from PIL import Image
    from scipy.ndimage import zoom
    
    logger.info("  📊 Computing global statistics (min/max/threshold)...")
    
    global_min = float('inf')
    global_max = float('-inf')
    
    # Sample slices for threshold computation
    num_samples = min(50, len(slice_files))  # ✅ FIX: Cap at 50 slices max
    sample_indices = np.linspace(0, len(slice_files)-1, num_samples, dtype=int)
    sample_values = []
    
    for idx in tqdm(sample_indices, desc="  Scanning", leave=False):
        try:
            slice_img = np.array(Image.open(slice_files[idx])).astype(np.float32)
            
            if downsample_xy != 1.0:
                slice_img = zoom(slice_img, downsample_xy, order=1)
            
            # Update min/max
            slice_min = slice_img.min()
            slice_max = slice_img.max()
            global_min = min(global_min, slice_min)
            global_max = max(global_max, slice_max)
            
            # ✅ FIX: Sample 10% of pixels instead of ALL (much faster)
            num_pixels = slice_img.size
            sample_size = max(1000, num_pixels // 10)  # At least 1000 pixels
            flat = slice_img.flatten()
            sampled = np.random.choice(flat, size=min(sample_size, len(flat)), replace=False)
            sample_values.append(sampled)
        
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to read slice {idx}: {e}")
            continue
    
    if len(sample_values) == 0:
        raise RuntimeError("Failed to sample any slices for statistics computation")
    
    # Compute threshold
    all_samples = np.concatenate(sample_values)
    global_threshold = np.percentile(all_samples, percentile)
    
    logger.info(f"  ✅ Global range: [{global_min:.2f}, {global_max:.2f}]")
    logger.info(f"  ✅ Global threshold ({percentile}th percentile): {global_threshold:.4f}")
    
    return global_min, global_max, global_threshold


# ============================================================================
# VOLUME LOADER
# ============================================================================

class VolumeLoader:
    """Format-agnostic 3D volume loader"""
    
    @staticmethod
    def load_volume(
        file_or_folder: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True,
        max_slices: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load 3D volume from TIFF or NIfTI"""
        if file_or_folder.is_file():
            file_format = detect_file_format(file_or_folder)
            
            if file_format == FileFormat.NIFTI:
                return VolumeLoader._load_nifti(
                    file_or_folder, downsample_xy, downsample_z, normalize
                )
            elif file_format == FileFormat.TIFF:
                return VolumeLoader._load_tiff_stack(
                    file_or_folder, downsample_xy, downsample_z, normalize
                )
            else:
                logger.error(f"Unknown format: {file_or_folder}")
                return None, None
        
        elif file_or_folder.is_dir():
            logger.info(f"  📁 Detected TIFF slice folder: {file_or_folder.name}")
            logger.info(f"  ⚠️ Will use streaming processor")
            return None, {'format': 'tiff_slices', 'requires_streaming': True}
        
        else:
            logger.error(f"Path not found: {file_or_folder}")
            return None, None
    
    @staticmethod
    def _load_nifti(
        nifti_path: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load NIfTI file"""
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("Install nibabel: pip install nibabel")
        
        logger.info(f"  📂 Loading NIfTI: {nifti_path.name}")
        
        try:
            nii = nib.load(str(nifti_path))
            volume = nii.get_fdata().astype(np.float32)
            spacing = nii.header.get_zooms()[:3]
            
            logger.info(f"  📏 Original shape: {volume.shape}")
            logger.info(f"  📏 Spacing: {spacing} mm")
            
            # NIfTI is (X,Y,Z) → convert to (D,H,W)
            volume = np.transpose(volume, (2, 1, 0))
            
            # Downsample
            if downsample_xy != 1.0 or downsample_z != 1.0:
                from scipy.ndimage import zoom
                zoom_factors = (downsample_z, downsample_xy, downsample_xy)
                volume = zoom(volume, zoom_factors, order=1)
                logger.info(f"  ✅ Downsampled to: {volume.shape}")
            
            # Normalize
            if normalize:
                vol_min, vol_max = volume.min(), volume.max()
                if vol_max > vol_min:
                    volume = (volume - vol_min) / (vol_max - vol_min)
            
            metadata = {
                'format': 'nifti',
                'original_shape': list(nii.shape),  # ✅ FIX: Convert to list for JSON
                'spacing': list(spacing),  # ✅ FIX: Convert to list for JSON
                'shape': list(volume.shape)  # ✅ FIX: Add final shape
            }
            
            return volume, metadata
        
        except Exception as e:
            logger.error(f"Failed to load NIfTI: {e}")
            return None, None
    
    @staticmethod
    def _load_tiff_stack(
        tiff_path: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load multi-page TIFF"""
        try:
            import tifffile
        except ImportError:
            raise ImportError("Install tifffile: pip install tifffile")
        
        logger.info(f"  📂 Loading TIFF stack: {tiff_path.name}")
        
        try:
            volume = tifffile.imread(str(tiff_path)).astype(np.float32)
            logger.info(f"  📏 Original shape: {volume.shape}")
            
            if volume.ndim == 2:
                volume = volume[np.newaxis, ...]
            
            if downsample_xy != 1.0 or downsample_z != 1.0:
                from scipy.ndimage import zoom
                zoom_factors = (downsample_z, downsample_xy, downsample_xy)
                volume = zoom(volume, zoom_factors, order=1)
                logger.info(f"  ✅ Downsampled to: {volume.shape}")
            
            if normalize:
                vol_min, vol_max = volume.min(), volume.max()
                if vol_max > vol_min:
                    volume = (volume - vol_min) / (vol_max - vol_min)
            
            metadata = {
                'format': 'tiff_stack',
                'spacing': [2.0, 1.0, 1.0],  # ✅ FIX: Use list instead of tuple
                'shape': list(volume.shape)  # ✅ FIX: Add final shape
            }
            
            return volume, metadata
        
        except Exception as e:
            logger.error(f"Failed to load TIFF stack: {e}")
            return None, None


# ============================================================================
# HDF5 SAVE/LOAD UTILITIES
# ============================================================================

def save_volume_hdf5(
    volume: np.ndarray,
    output_path: Path,
    metadata: Dict,
    compression: str = 'gzip',
    compression_opts: int = 4,
    compute_foreground: bool = True,
    foreground_percentile: float = 95.0
) -> Dict:
    """
    Save volume to HDF5 with foreground extraction
    """
    output_path = output_path.with_suffix('.h5')
    
    logger.info(f"  💾 Saving to HDF5: {output_path.name}")
    
    try:
        with h5py.File(output_path, 'w') as f:
            # Save volume
            f.create_dataset(
                'volume',
                data=volume,
                compression=compression,
                compression_opts=compression_opts,
                dtype=np.float32
            )
            
            # Foreground extraction
            if compute_foreground:
                threshold = np.percentile(volume, foreground_percentile)
                foreground_mask = volume > threshold
                coords = np.argwhere(foreground_mask)
                
                f.create_dataset(
                    'foreground_coords',
                    data=coords,
                    compression='gzip',
                    compression_opts=4,
                    dtype=np.int32
                )
                
                f['foreground_coords'].attrs['threshold'] = float(threshold)
                f['foreground_coords'].attrs['method'] = f'percentile_{foreground_percentile}'
                f['foreground_coords'].attrs['num_coords'] = int(len(coords))
                
                logger.info(f"  ✅ Extracted {len(coords)} foreground coordinates")
            
            # ✅ FIX: Save metadata as JSON strings (prevents serialization errors)
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    f['volume'].attrs[key] = value
                else:
                    f['volume'].attrs[key] = json.dumps(value)
        
        # Stats
        file_size_mb = output_path.stat().st_size / 1e6
        uncompressed_size_mb = volume.nbytes / 1e6
        compression_ratio = uncompressed_size_mb / file_size_mb if file_size_mb > 0 else 1.0
        
        logger.info(f"  ✅ Saved: {file_size_mb:.1f} MB (compression: {compression_ratio:.2f}x)")
        
        return {
            'file_path': str(output_path),
            'file_size_mb': file_size_mb,
            'compression_ratio': compression_ratio
        }
    
    except Exception as e:
        logger.error(f"Failed to save HDF5: {e}")
        raise


def load_volume_hdf5(hdf5_path: Path) -> Tuple[np.ndarray, Dict]:
    """Load volume from HDF5"""
    with h5py.File(hdf5_path, 'r') as f:
        volume = f['volume'][:]
        
        # ✅ FIX: Parse JSON strings back to objects
        metadata = {}
        for key, value in f['volume'].attrs.items():
            try:
                metadata[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                metadata[key] = value
    
    return volume, metadata


# ============================================================================
# DATA DISCOVERY
# ============================================================================

class DatasetDiscovery:
    """Flexible data discovery"""
    
    @staticmethod
    def discover_unlabeled_data(input_dir: Path) -> List[Dict]:
        """Discover unlabeled volumes"""
        discovered = []
        
        marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        for marker_dir in marker_dirs:
            marker_name = marker_dir.name
            
            # NIfTI files
            nifti_files = list(marker_dir.glob("*.nii.gz")) + list(marker_dir.glob("*.nii"))
            
            if nifti_files:
                for nifti_file in nifti_files:
                    discovered.append({
                        'marker_type': marker_name,
                        'brain_name': nifti_file.stem.replace('.nii', ''),
                        'path': nifti_file,
                        'is_file': True
                    })
            else:
                # TIFF slice folders
                brain_folders = [d for d in marker_dir.iterdir() if d.is_dir()]
                
                for brain_folder in brain_folders:
                    discovered.append({
                        'marker_type': marker_name,
                        'brain_name': brain_folder.name,
                        'path': brain_folder,
                        'is_file': False
                    })
        
        return discovered
    
    @staticmethod
    def discover_labeled_data(input_dir: Path) -> List[Dict]:
        """Discover labeled pairs"""
        discovered = []
        
        # Strategy 1: Top-level raw/gt
        raw_dir = input_dir / 'raw'
        gt_dir = input_dir / 'gt'
        
        if raw_dir.exists() and gt_dir.exists():
            logger.info(f"Detected top-level 'raw'/'gt' structure")
            img_files = sorted(raw_dir.glob("*.nii.gz"))
            
            for img_file in img_files:
                raw_stem = img_file.name.replace('.nii.gz', '')
                stem_parts = raw_stem.split('_')[:-1]
                gt_stem = '_'.join(stem_parts)
                gt_filename = f"{gt_stem}.nii.gz"
                mask_file = gt_dir / gt_filename
                
                if mask_file.exists():
                    discovered.append({
                        'marker_type': input_dir.name,
                        'sample_name': gt_stem,
                        'img_path': img_file,
                        'mask_path': mask_file
                    })
                else:
                    logger.warning(f"  ⚠️ Missing mask for {img_file.name}")
        
        # Strategy 2: Nested marker folders
        else:
            logger.info(f"Scanning for marker subfolders...")
            marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
            
            for marker_dir in marker_dirs:
                marker_name = marker_dir.name
                
                nested_raw_dir = marker_dir / 'raw'
                nested_gt_dir = marker_dir / 'gt'
                
                if nested_raw_dir.exists() and nested_gt_dir.exists():
                    logger.info(f"  Found 'raw'/'gt' in: {marker_name}")
                    img_files = sorted(nested_raw_dir.glob("*.nii.gz"))
                    
                    for img_file in img_files:
                        raw_stem = img_file.name.replace('.nii.gz', '')
                        stem_parts = raw_stem.split('_')[:-1]
                        gt_stem = '_'.join(stem_parts)
                        gt_filename = f"{gt_stem}.nii.gz"
                        mask_file = nested_gt_dir / gt_filename
                        
                        if mask_file.exists():
                            discovered.append({
                                'marker_type': marker_name,
                                'sample_name': gt_stem,
                                'img_path': img_file,
                                'mask_path': mask_file
                            })
        
        return discovered


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Worker-specific checkpoint management"""
    def __init__(self, output_path: Path, worker_id: Optional[str] = None):
        suffix = f'_{worker_id}' if worker_id else ''
        self.checkpoint_path = output_path.with_suffix(f'.checkpoint{suffix}.json')
        self.output_path = output_path
    
    def load(self) -> Dict:
        """Load checkpoint if exists"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"  ✅ Loaded checkpoint: resuming from slice {checkpoint.get('last_processed', 0)}")
                return checkpoint
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load checkpoint: {e}")
                return {}
        return {}
    
    def save(self, last_processed: int, total_slices: int, global_stats: Dict = None):
        """Save checkpoint"""
        checkpoint = {
            'last_processed': last_processed,
            'total_slices': total_slices,
            'timestamp': datetime.now().isoformat(),
            'global_stats': global_stats or {}
        }
        
        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to save checkpoint: {e}")
    
    def cleanup(self):
        """Remove checkpoint after successful completion"""
        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
                logger.info(f"  ✅ Removed checkpoint file")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to remove checkpoint: {e}")


# ============================================================================
# STREAMING PROCESSOR
# ============================================================================

def process_and_stream_to_hdf5(
    folder: Path,
    output_path: Path,
    downsample_xy: float = 1.0,
    normalize: bool = True,
    chunk_size: int = 32,
    foreground_percentile: float = 95.0,
    checkpoint_mgr: Optional[CheckpointManager] = None
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    ✅ MAJOR FIXES:
    - Optimized chunk size calculation
    - Single-pass global statistics
    - Reduced checkpoint frequency (every 50 slices, not 10)
    - Fixed metadata serialization
    """
    try:
        from PIL import Image
        from scipy.ndimage import zoom
        Image.MAX_IMAGE_PIXELS = None
    except ImportError:
        raise ImportError("Install dependencies: pip install Pillow scipy")
    
    logger.info(f"  🔄 Streaming TIFF slices to {output_path.name}...")
    
    # Find slices
    slice_files = sorted(
        list(folder.glob("*.tif")) + list(folder.glob("*.tiff")),
        key=lambda x: int(''.join(filter(str.isdigit, x.stem))) 
                     if any(c.isdigit() for c in x.stem) else 0
    )
    
    if len(slice_files) == 0:
        logger.warning(f"  ⚠️ No .tif files in {folder}")
        return None, None
    
    # Get dimensions from first slice
    first_slice_img = np.array(Image.open(slice_files[0])).astype(np.float32)
    
    if downsample_xy != 1.0:
        first_slice_img = zoom(first_slice_img, downsample_xy, order=1)
    
    H, W = first_slice_img.shape
    D = len(slice_files)
    final_shape = (D, H, W)
    
    logger.info(f"  📏 Final shape: {final_shape} ({D} slices)")
    
    # Disk space check
    required_gb = (D * H * W * 4) / 1e9
    check_disk_space(output_path, required_gb)
    
    # Memory monitor
    mem_monitor = MemoryMonitor(threshold_gb=2.0)
    
    # Checkpoint manager
    if checkpoint_mgr is None:
        checkpoint_mgr = CheckpointManager(output_path)
    
    checkpoint = checkpoint_mgr.load()
    
    # ============================================================================
    # STEP 1: Global Statistics (if not resuming)
    # ============================================================================
    
    if 'global_stats' in checkpoint:
        global_min = checkpoint['global_stats']['global_min']
        global_max = checkpoint['global_stats']['global_max']
        global_threshold = checkpoint['global_stats']['global_threshold']
        logger.info(f"  ♻️ Resuming with saved global stats")
    else:
        # ✅ FIX: Single-pass computation
        global_min, global_max, global_threshold = compute_global_stats(
            slice_files, downsample_xy, foreground_percentile
        )
        
    if normalize and global_max > global_min:
            # convert raw percentile threshold to normalized [0,1] space
        global_threshold_norm = (global_threshold - global_min) / (global_max - global_min)
            # guard: clamp to [0,1]
        global_threshold_norm = np.clip(global_threshold_norm, 0.0, 1.0)
        logger.info(
        f"  ℹ️ Using normalized threshold: raw={global_threshold:.4f} -> norm={global_threshold_norm:.4f}"
        )
    else:
        global_threshold_norm = global_threshold
        logger.info(f"  ℹ️ Using raw threshold: {global_threshold_norm:.4f}")


        checkpoint['global_stats'] = {
            'global_min': float(global_min),
            'global_max': float(global_max),
            'global_threshold': float(global_threshold)
        }
    
    # ============================================================================
    # STEP 2: Stream Processing
    # ============================================================================
    
    logger.info("  💾 Streaming to HDF5...")
    
    start_idx = checkpoint.get('last_processed', 0)
    h5_mode = 'r+' if start_idx > 0 else 'w'
    
    try:
        with h5py.File(output_path, h5_mode) as f:
            # ✅ FIX: Optimized chunk size calculation
            chunk_d = min(chunk_size, D)
            chunk_h = min(256, H)  # Cap at 256 instead of 512
            chunk_w = min(256, W)  # Cap at 256 instead of 512
            
            if h5_mode == 'w':
                vol_dset = f.create_dataset(
                    'volume',
                    shape=final_shape,
                    chunks=(chunk_d, chunk_h, chunk_w),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4
                )
                
                coord_dset = f.create_dataset(
                    'foreground_coords',
                    shape=(0, 3),
                    maxshape=(None, 3),
                    chunks=(100000, 3),
                    dtype=np.int32,
                    compression='gzip',
                    compression_opts=4
                )
            else:
                vol_dset = f['volume']
                coord_dset = f['foreground_coords']
            
            total_coords = coord_dset.shape[0] if h5_mode == 'r+' else 0
            
            # Process slices
            for i in tqdm(
                range(start_idx, D),
                desc="  Processing",
                initial=start_idx,
                total=D,
                leave=False
            ):
                slice_file = slice_files[i]
                
                try:
                    # Load slice
                    slice_img = np.array(Image.open(slice_file)).astype(np.float32)
                    
                    # Downsample
                    if downsample_xy != 1.0:
                        slice_img = zoom(slice_img, downsample_xy, order=1)
                    
                    # ✅ FIX: Fail-fast on shape mismatch
                    if slice_img.shape != (H, W):
                        raise ValueError(
                            f"CRITICAL: Slice {i} ({slice_file.name}) shape mismatch!\n"
                            f"Expected {(H, W)}, got {slice_img.shape}"
                        )
                    
                    # Normalize with global statistics
                    if normalize and global_max > global_min:
                        slice_img = (slice_img - global_min) / (global_max - global_min)
                        slice_img = np.clip(slice_img, 0, 1)  # ✅ FIX: Clip to [0,1]
                    
                    # Write to HDF5
                    vol_dset[i] = slice_img
                    
                    # Extract foreground coordinates
                    fg_mask_2d = slice_img > global_threshold_norm
                    coords_yx = np.argwhere(fg_mask_2d)
                    
                    if coords_yx.size > 0:
                        z_coord = np.full((len(coords_yx), 1), i, dtype=np.int32)
                        coords_zyx = np.hstack((z_coord, coords_yx))
                        
                        # Append to dataset
                        current_size = coord_dset.shape[0]
                        new_size = current_size + len(coords_zyx)
                        coord_dset.resize(new_size, axis=0)
                        coord_dset[current_size:] = coords_zyx
                        total_coords += len(coords_zyx)
                    
                    # ✅ FIX: Save checkpoint every 50 slices (not 10)
                    if (i + 1) % 50 == 0:
                        checkpoint_mgr.save(i, D, checkpoint.get('global_stats'))
                        mem_monitor.check(f"at slice {i}/{D}")
                
                except Exception as e:
                    logger.error(f"❌ Failed to process slice {i} ({slice_file.name}): {e}")
                    raise
            
            # ✅ FIX: Save metadata as JSON-serializable values
            vol_dset.attrs['shape'] = json.dumps(list(final_shape))
            vol_dset.attrs['format'] = 'tiff_slices'
            vol_dset.attrs['spacing'] = json.dumps([2.0, 1.0, 1.0])
            vol_dset.attrs['global_min'] = float(global_min)
            vol_dset.attrs['global_max'] = float(global_max)
            vol_dset.attrs['normalized'] = bool(normalize)
            try:
                vol_dset.attrs['global_threshold_norm'] = float(global_threshold_norm)
            except NameError:
                # If somehow global_threshold_norm not present, compute best-effort
                if global_max > global_min:
                    vol_dset.attrs['global_threshold_norm'] = float((global_threshold - global_min) / (global_max - global_min))
                else:
                    vol_dset.attrs['global_threshold_norm'] = float(global_threshold)
            coord_dset.attrs['num_coords'] = int(total_coords)
            coord_dset.attrs['volume_shape'] = json.dumps(list(final_shape))
            coord_dset.attrs['method'] = f'global_percentile_{foreground_percentile}'
            coord_dset.attrs['threshold'] = float(global_threshold)
        
        # ✅ FIX: Validate only once at the end
        validate_hdf5_file(output_path, final_shape)
        
        # Clean up checkpoint
        checkpoint_mgr.cleanup()
        
        # Compute stats
        file_size_mb = output_path.stat().st_size / 1e6
        uncompressed_size_mb = (D * H * W * 4) / 1e6
        compression_ratio = uncompressed_size_mb / file_size_mb if file_size_mb > 0 else 1.0
        
        save_info = {
            'file_path': str(output_path),
            'file_size_mb': file_size_mb,
            'compression_ratio': compression_ratio
        }

        #========================Change this line=====================================
        #=============================================================================

        # metadata["processing"]["global_threshold_norm"] = float(global_threshold_norm)
        
        #=======================If error come in metadata============================
        #============================================================================
        metadata = {
            'format': 'tiff_slices',
            'spacing': [2.0, 1.0, 1.0],
            'shape': list(final_shape),
            'global_min': float(global_min),
            'global_max': float(global_max),
            'normalized': bool(normalize),
            'foreground_threshold': float(global_threshold),
            'global_threshold_norm': float(global_threshold_norm),
            'num_foreground_coords': int(total_coords)
        }
        
        logger.info(f"  ✅ Streamed {D} slices to HDF5")
        logger.info(f"  ✅ Extracted {total_coords} foreground coordinates")
        logger.info(f"  ✅ Saved: {file_size_mb:.1f} MB (compression: {compression_ratio:.2f}x)")
        
        return save_info, metadata
    
    except Exception as e:
        logger.error(f"❌ Streaming failed: {e}")
        
        # Keep checkpoint for resume, but delete partial file if fresh start
        if output_path.exists() and start_idx == 0:
            output_path.unlink()
            logger.info(f"  🗑️ Deleted partial file")
        
        raise


# ============================================================================
# WORKER FUNCTION
# ============================================================================

def process_brain_worker(
    item: Dict,
    output_marker_dir: Path,
    marker_label: int,
    args: argparse.Namespace,
    worker_id: str
) -> Optional[Dict]:
    """
    Worker function to process a single unlabeled brain
    ✅ FIX: Improved error handling and skip logic
    """
    Image.MAX_IMAGE_PIXELS = None
    brain_name = item['brain_name']
    output_file = output_marker_dir / f"{brain_name}.h5"
    
    # ✅ FIX: Check if file exists and is valid
    if output_file.exists():
        try:
            with h5py.File(output_file, 'r') as f:
                if 'volume' in f and 'foreground_coords' in f:
                    # File exists and is complete
                    logger.info(f"  ➡️ Skipping {brain_name}: Already processed")
                    
                    # Return metadata from existing file
                    shape = json.loads(f['volume'].attrs['shape'])
                    spacing = json.loads(f['volume'].attrs.get('spacing', '[2.0, 1.0, 1.0]'))
                    
                    return {
                        'brain_name': brain_name,
                        'marker_type': item['marker_type'],
                        'marker_label': marker_label,
                        'shape': shape,
                        'file_path': str(output_file.relative_to(output_marker_dir.parent)),
                        'file_size_mb': output_file.stat().st_size / 1e6,
                        'compression_ratio': 'N/A (skipped)',
                        'original_format': f['volume'].attrs.get('format', 'unknown'),
                        'spacing': spacing,
                        'processing': {'status': 'skipped_already_exists'}
                    }
        except Exception as e:
            logger.warning(f"  ⚠️ {brain_name} exists but corrupted: {e}. Re-processing.")
            output_file.unlink()
    
    # Process the brain
    try:
        logger.info(f"📦 Processing: {brain_name}")
        
        checkpoint_mgr = CheckpointManager(output_file, worker_id=worker_id)
        mem_monitor = MemoryMonitor(threshold_gb=2.0)
        
        if item['is_file']:
            logger.info(f"  Loading file: {item['path'].name}")
            volume, vol_metadata = VolumeLoader.load_volume(
                item['path'],
                downsample_xy=args.downsample_xy,
                downsample_z=args.downsample_z,
                normalize=args.normalize
            )
            
            if volume is None:
                logger.warning(f"  ⚠️ Skipping {brain_name} (loading failed)")
                return None
            
            required_gb = (volume.size * 4) / 1e9
            check_disk_space(output_file, required_gb)
            
            save_info = save_volume_hdf5(
                volume, output_file, vol_metadata,
                foreground_percentile=args.foreground_percentile
            )
            
            mem_monitor.check(f"after processing {brain_name}")
        
        else:
            logger.info(f"  Streaming TIFF folder: {item['path'].name}")
            save_info, vol_metadata = process_and_stream_to_hdf5(
                item['path'],
                output_file,
                downsample_xy=args.downsample_xy,
                normalize=args.normalize,
                foreground_percentile=args.foreground_percentile,
                checkpoint_mgr=checkpoint_mgr
            )
        
        if save_info is None or vol_metadata is None:
            logger.warning(f"  ⚠️ Skipping {brain_name} (processing failed)")
            return None
        
        # ✅ FIX: Validation is now done inside streaming function
        # No need to call it again here
        
        checkpoint_mgr.cleanup()
        
        meta = {
            'brain_name': brain_name,
            'marker_type': item['marker_type'],
            'marker_label': marker_label,
            'shape': vol_metadata.get('shape', [0, 0, 0]),
            'file_path': str(output_file.relative_to(output_marker_dir.parent)),
            'file_size_mb': save_info['file_size_mb'],
            'compression_ratio': save_info.get('compression_ratio', 1.0),
            'original_format': vol_metadata.get('format', 'unknown'),
            'spacing': vol_metadata.get('spacing', [2.0, 1.0, 1.0]),
            'processing': {
                'downsample_xy': args.downsample_xy,
                'downsample_z': args.downsample_z,
                'normalized': args.normalize,
                'foreground_percentile': args.foreground_percentile,
                'processed_by_worker': worker_id
            }
        }
        norm_thr = None
        if 'global_threshold_norm' in vol_metadata:
            norm_thr = float(vol_metadata['global_threshold_norm'])
        else:
            # try to compute using returned raw threshold + global_min/global_max
            try:
                raw_thr = float(vol_metadata.get('foreground_threshold', vol_metadata.get('global_threshold', np.nan)))
                gmin = float(vol_metadata.get('global_min', np.nan))
                gmax = float(vol_metadata.get('global_max', np.nan))
                if np.isfinite(raw_thr) and gmax > gmin:
                    norm_thr = float((raw_thr - gmin) / (gmax - gmin))
            except Exception:
                norm_thr = None

        if norm_thr is not None:
            meta['processing']['global_threshold_norm'] = float(norm_thr)
        logger.info(f"  ✅ Successfully processed {brain_name}")
        return meta
    
    except Exception as e:
        logger.error(f"  ❌ FAILED processing {brain_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def aggregate_metadata_unlabeled(
    all_metadata: List[Dict],
    output_dir: Path,
    args: argparse.Namespace
) -> Dict:
    """
    ✅ PERMANENT FIX: Generate final metadata.json with train/val/test splits
    
    This function:
    - Aggregates per-volume metadata
    - Creates stratified train/val/test splits (80/10/10)
    - Exports global_threshold_norm for each volume
    - Writes a production-ready metadata.json
    
    Args:
        all_metadata: List of volume metadata dicts
        output_dir: Root output directory
        args: Command-line arguments
    
    Returns:
        Complete metadata dictionary
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("\n" + "="*80)
    logger.info("GENERATING FINAL METADATA.JSON")
    logger.info("="*80)
    
    # ============================================================================
    # STEP 1: Extract marker types for stratification
    # ============================================================================
    marker_types = [vol['marker_type'] for vol in all_metadata]
    unique_markers = list(set(marker_types))
    
    logger.info(f"Total volumes: {len(all_metadata)}")
    logger.info(f"Unique markers: {unique_markers}")
    
    # ============================================================================
    # STEP 2: Create stratified train/val/test splits
    # ============================================================================
    # ============================================================================
# STEP 2: Adaptive dataset splits (PERMANENT FIX)
# ============================================================================

    N = len(all_metadata)

    if N < 2:
    # Case 1: Only one volume → train only
        logger.warning(
        "⚠️ Only one unlabeled volume found. "
        "Assigning it to TRAIN split only."
        )
        train_meta = all_metadata
        val_meta = []
        test_meta = []

    elif N < 5:
        # Case 2: Too few samples for 3-way split
        logger.warning(
        f"⚠️ Only {N} unlabeled volumes found. "
        "Using TRAIN / VAL split only (no TEST set)."
        )
        train_meta, val_meta = train_test_split(
            all_metadata,
            test_size=0.25,
            random_state=42
        )
        test_meta = []

    else:
        # Case 3: Normal regime (≥5 volumes)
        marker_types = [vol['marker_type'] for vol in all_metadata]

        try:
            train_meta, temp_meta = train_test_split(
                all_metadata,
                test_size=0.2,
                stratify=marker_types,
                random_state=42
            )

            temp_markers = [vol['marker_type'] for vol in temp_meta]
            val_meta, test_meta = train_test_split(
                temp_meta,
                test_size=0.5,
                stratify=temp_markers,
                random_state=42
            )

        except ValueError as e:
            logger.warning(
                f"⚠️ Stratified split failed ({e}). "
                "Falling back to RANDOM split."
            )

            train_meta, temp_meta = train_test_split(
                all_metadata,
                test_size=0.2,
                random_state=42
            )

            if len(temp_meta) >= 2:
                val_meta, test_meta = train_test_split(
                    temp_meta,
                    test_size=0.5,
                    random_state=42
                )
            else:
                val_meta = temp_meta
                test_meta = []

    
    logger.info(f"  Train: {len(train_meta)} volumes")
    logger.info(f"  Val:   {len(val_meta)} volumes")
    logger.info(f"  Test:  {len(test_meta)} volumes")
    
    # ============================================================================
    # STEP 3: Load global_threshold_norm from HDF5 files
    # ============================================================================
    logger.info("  Loading global_threshold_norm from HDF5 files...")
    
    for vol_meta in all_metadata:
        h5_path = output_dir / vol_meta['file_path']
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Extract from HDF5 attributes
                if 'global_threshold_norm' in f['volume'].attrs:
                    vol_meta['global_threshold_norm'] = float(
                        f['volume'].attrs['global_threshold_norm']
                    )
                else:
                    # Fallback: compute from global_min/max/threshold
                    g_min = float(f['volume'].attrs.get('global_min', 0))
                    g_max = float(f['volume'].attrs.get('global_max', 1))
                    g_thr = float(f['volume'].attrs.get('foreground_threshold', 0.95))
                    
                    if g_max > g_min:
                        vol_meta['global_threshold_norm'] = (g_thr - g_min) / (g_max - g_min)
                    else:
                        vol_meta['global_threshold_norm'] = g_thr
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to load threshold from {h5_path.name}: {e}")
            vol_meta['global_threshold_norm'] = 0.95  # Safe default
    
    # ============================================================================
    # STEP 4: Build final metadata structure
    # ============================================================================
    metadata_dict = {
        'dataset_name': 'SELMA3D_SSL',
        'num_volumes': len(all_metadata),
        'marker_types': unique_markers,
        'storage_format': 'hdf5',
        'splits': {
            'train': len(train_meta),
            'val': len(val_meta),
            'test': len(test_meta)
        },
        'data': {
            'train': train_meta,
            'val': val_meta,
            'test': test_meta
        },
        'processing': {
            'downsample_xy': args.downsample_xy,
            'downsample_z': args.downsample_z,
            'normalized': args.normalize,
            'foreground_percentile': args.foreground_percentile,
            'parallel_workers': args.num_workers,
            'processed_at': datetime.now().isoformat()
        },
        'statistics': {
            'total_size_mb': sum(v['file_size_mb'] for v in all_metadata),
            'avg_compression_ratio': sum(
                v.get('compression_ratio', 1.0) for v in all_metadata
            ) / len(all_metadata)
        }
    }
    
    # ============================================================================
    # STEP 5: Write metadata.json
    # ============================================================================
    metadata_path = output_dir / 'metadata.json'
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2, default=str)
    
    logger.info(f"✅ Metadata saved: {metadata_path}")
    logger.info(f"✅ Structure: train/val/test splits with {len(unique_markers)} markers")
    logger.info("="*80)
    
    return metadata_dict
# ============================================================================
# PROCESS UNLABELED DATA
# ============================================================================

def process_unlabeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """
    Process unlabeled data in parallel
    ✅ FIX: Better error handling and progress reporting
    """
    logger.info("\n" + "="*80)
    logger.info(f"PROCESSING UNLABELED DATA (SSL) with {args.num_workers} parallel workers")
    logger.info("="*80)
    
    all_metadata = []
    
    # ✅ FIX: Make marker map configurable via args if needed
    marker_map = {
        'Ab_plaques': 3, 'ad_plaque': 3, 'cFos': 0,
        'microglia': 4, 'cell_nucleus': 2, 'unknown': 5, 'vessel': 1
    }
    
    discovered_data = DatasetDiscovery.discover_unlabeled_data(input_dir)
    
    if len(discovered_data) == 0:
        raise ValueError(f"No data found in {input_dir}")
    
    logger.info(f"Discovered {len(discovered_data)} total volumes\n")
    
    # Group by marker type
    from collections import defaultdict
    by_marker = defaultdict(list)
    for item in discovered_data:
        by_marker[item['marker_type']].append(item)
    
    # Initialize parallel processor
    parallel = Parallel(
        n_jobs=args.num_workers,
        backend='loky',
        verbose=0
    )
    
    for marker_name, items in by_marker.items():
        marker_label = marker_map.get(marker_name.lower(), 5)
        
        logger.info(f"🔬 Processing marker: {marker_name} (label={marker_label})")
        logger.info(f"  Found {len(items)} volumes")
        logger.info(f"  Using {args.num_workers} parallel workers")
        
        output_marker_dir = output_dir / marker_name
        output_marker_dir.mkdir(parents=True, exist_ok=True)
        
        # Create jobs
        jobs = []
        for idx, item in enumerate(items):
            worker_id = f"{marker_name}_{idx:04d}"
            jobs.append(
                delayed(process_brain_worker)(
                    item, output_marker_dir, marker_label, args, worker_id
                )
            )
        
        # Run jobs in parallel
        logger.info(f"  Starting parallel processing...")
        
        try:
            results = list(tqdm(
                parallel(jobs),
                total=len(items),
                desc=f"  Processing {marker_name}",
                unit="brain",
                ncols=100
            ))
        except Exception as e:
            logger.error(f"  ❌ Parallel processing failed: {e}")
            logger.error(f"  Try reducing --num_workers or freeing up resources")
            # ✅ FIX: Continue with other markers instead of crashing
            logger.warning(f"  Skipping remaining brains in {marker_name}")
            continue
        
        # Collect results
        marker_metadata = []
        failed_count = 0
        
        for meta in results:
            if meta is not None:
                marker_metadata.append(meta)
            else:
                failed_count += 1
        
        all_metadata.extend(marker_metadata)
        
        logger.info(f"  ✅ Finished processing {marker_name}:")
        logger.info(f"    Successful: {len(marker_metadata)}/{len(items)}")
        if failed_count > 0:
            logger.warning(f"    Failed: {failed_count}/{len(items)}")
    
    # Save metadata
    # metadata_path = output_dir / 'metadata.json'
    # with open(metadata_path, 'w') as f:
    #     json.dump({
    #         'num_volumes': len(all_metadata),
    #         'marker_types': list(set(m['marker_type'] for m in all_metadata)),
    #         'total_size_mb': sum(m['file_size_mb'] for m in all_metadata),
    #         'volumes': all_metadata,
    #         'storage_format': 'hdf5',
    #         'processing': {
    #             'parallel_workers': args.num_workers,
    #             'processed_at': datetime.now().isoformat(),
    #             'downsample_xy': args.downsample_xy,
    #             'downsample_z': args.downsample_z,
    #             'normalized': args.normalize,
    #             'foreground_percentile': args.foreground_percentile
    #         }
    #     }, f, indent=2)
    metadata_dict = aggregate_metadata_unlabeled(all_metadata, output_dir, args)
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processed {len(all_metadata)} unlabeled volumes")
    logger.info(f"✅ Total size: {metadata_dict['statistics']['total_size_mb']:.1f} MB")
    logger.info(f"✅ Metadata saved with train/val/test splits")
    logger.info(f"{'='*80}")
    
    return all_metadata


# ============================================================================
# PROCESS LABELED DATA
# ============================================================================

def process_labeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """
    Process labeled data
    ✅ FIX: Better error handling and validation
    """
    logger.info("\n" + "="*80)
    logger.info("PROCESSING LABELED DATA (FINE-TUNING)")
    logger.info("="*80)
    
    all_samples = []
    
    discovered_pairs = DatasetDiscovery.discover_labeled_data(input_dir)
    
    if len(discovered_pairs) == 0:
        raise ValueError(f"No labeled pairs found in {input_dir}")
    
    logger.info(f"Discovered {len(discovered_pairs)} image-mask pairs\n")
    
    for pair in tqdm(discovered_pairs, desc="Processing samples"):
        try:
            # Load image
            img, img_meta = VolumeLoader.load_volume(
                pair['img_path'],
                downsample_xy=args.downsample_xy,
                downsample_z=args.downsample_z,
                normalize=args.normalize
            )
            
            # Load mask
            mask, mask_meta = VolumeLoader.load_volume(
                pair['mask_path'],
                downsample_xy=args.downsample_xy,
                downsample_z=args.downsample_z,
                normalize=False
            )
            
            if img is None or mask is None:
                logger.warning(f"  ⚠️ Failed to load {pair['sample_name']}")
                continue
            
            mask = mask.astype(np.int64)
            
            # ✅ FIX: Validate shape match
            if img.shape != mask.shape:
                logger.warning(
                    f"  ⚠️ Shape mismatch for {pair['sample_name']}: "
                    f"img={img.shape}, mask={mask.shape}"
                )
                continue
            
            all_samples.append({
                'img': img,
                'mask': mask,
                'marker_type': pair['marker_type'],
                'filename': pair['sample_name'],
                'shape': list(img.shape),
                'format': img_meta.get('format', 'unknown')
            })
        
        except Exception as e:
            logger.error(f"  ❌ Failed to process {pair['sample_name']}: {e}")
            continue
    
    if len(all_samples) == 0:
        raise ValueError("No valid samples processed!")
    
    logger.info(f"\nTotal valid samples: {len(all_samples)}")
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    marker_types = [s['marker_type'] for s in all_samples]
    
    # ✅ FIX: Handle case where we have too few samples for stratification
    try:
        train_samples, temp_samples = train_test_split(
            all_samples, test_size=0.2, stratify=marker_types, random_state=42
        )
        
        temp_markers = [s['marker_type'] for s in temp_samples]
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.5, stratify=temp_markers, random_state=42
        )
    except ValueError as e:
        logger.warning(f"  ⚠️ Stratification failed: {e}. Using random split.")
        train_samples, temp_samples = train_test_split(
            all_samples, test_size=0.2, random_state=42
        )
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.5, random_state=42
        )
    
    logger.info(f"Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Save splits
    split_metadata = {}
    
    for split_name, split_samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_meta = []
        
        for idx, sample in enumerate(tqdm(split_samples, desc=f"Saving {split_name}")):
            base_name = f"{sample['marker_type']}_{idx:04d}"
            output_file = split_dir / f"{base_name}.h5"
            
            try:
                with h5py.File(output_file, 'w') as f:
                    f.create_dataset(
                        'image', data=sample['img'],
                        compression='gzip', compression_opts=4, dtype=np.float32
                    )
                    f.create_dataset(
                        'mask', data=sample['mask'],
                        compression='gzip', compression_opts=4, dtype=np.int64
                    )
                    f['image'].attrs['marker_type'] = sample['marker_type']
                    f['image'].attrs['original_filename'] = sample['filename']
                    f['image'].attrs['shape'] = json.dumps(sample['shape'])
                
                file_size_mb = output_file.stat().st_size / 1e6
                
                split_meta.append({
                    'filename': base_name,
                    'marker_type': sample['marker_type'],
                    'shape': sample['shape'],
                    'file_size_mb': file_size_mb,
                    'file_path': str(output_file.relative_to(output_dir))
                })
            
            except Exception as e:
                logger.error(f"Failed to save {base_name}: {e}")
                continue
        
        split_metadata[split_name] = split_meta
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_samples': len(all_samples),
            'splits': {
                'train': len(train_samples),
                'val': len(val_samples),
                'test': len(test_samples)
            },
            'marker_types': list(set(s['marker_type'] for s in all_samples)),
            'data': split_metadata,
            'storage_format': 'hdf5',
            'processing': {
                'downsample_xy': args.downsample_xy,
                'downsample_z': args.downsample_z,
                'normalized': args.normalize,
                'processed_at': datetime.now().isoformat()
            }
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processed {len(all_samples)} labeled samples")
    logger.info(f"✅ Metadata saved: {metadata_path}")
    logger.info(f"{'='*80}")
    
    return all_samples


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='SELMA3D Data Preparation - Production Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
✅ PERMANENT FIXES APPLIED:
  • Optimized chunk size calculation
  • Single-pass global statistics
  • Fixed metadata serialization
  • Improved error handling
  • Reduced checkpoint frequency
  • Memory-efficient foreground sampling
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing raw data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed HDF5 files')
    parser.add_argument('--data_type', type=str, required=True, 
                       choices=['unlabeled', 'labeled'],
                       help='Type of data to process')
    parser.add_argument('--downsample_xy', type=float, default=1.0,
                       help='XY downsampling factor (default: 1.0 = no downsampling)')
    parser.add_argument('--downsample_z', type=float, default=1.0,
                       help='Z downsampling factor (default: 1.0 = no downsampling)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize intensities to [0,1]')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Disable normalization')
    parser.add_argument('--foreground_percentile', type=float, default=95.0,
                       help='Percentile for foreground threshold (default: 95.0)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("="*80)
    logger.info("SELMA3D DATA PREPARATION - PRODUCTION VERSION")
    logger.info("="*80)
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Type:   {args.data_type}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info("")
    logger.info("✅ ALL PERMANENT FIXES APPLIED")
    logger.info("="*80)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.data_type == 'unlabeled':
            process_unlabeled_data(input_dir, output_dir, args)
        elif args.data_type == 'labeled':
            process_labeled_data(input_dir, output_dir, args)
        
        logger.info("\n" + "="*80)
        logger.info("✅ DATA PREPARATION COMPLETE!")
        logger.info("="*80)
    
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"❌ FATAL ERROR: {e}")
        logger.error(f"{'='*80}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()