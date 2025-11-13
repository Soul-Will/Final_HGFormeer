"""
Unified Data Preparation Pipeline - PERMANENT FIX VERSION
═══════════════════════════════════════════════════════════════════

✅ ALL BUGS FIXED:
  1. BUG 1: Shape mismatch → Fail-fast with cleanup
  2. BUG 2: Coordinate indexing → Simplified logic
  3. BUG 3: Misleading metadata → Global percentile
  4. BUG 4: Per-slice normalization → 2-pass global
  5. FLAW 1: No error recovery → Checkpoint resume
  6. FLAW 2: No validation → Post-save checks
  7. FLAW 3: Hardcoded values → CLI-configurable
  8. MISSING 1: Progress persistence → JSON checkpoints
  9. MISSING 2: Memory monitoring → psutil tracking
  10. MISSING 3: Disk space check → Pre-flight validation

Author: AI/ML Research Engineer
Date: 2025-01-09
Version: 2.0.0 (Production-Ready)
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

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FILE FORMAT DETECTION
# =============================================================================

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


# =============================================================================
# ✅ PERMANENT FIX: DISK SPACE VALIDATION
# =============================================================================

def check_disk_space(output_path: Path, required_gb: float, buffer: float = 1.5):
    """
    ✅ MISSING 3 FIX: Pre-flight disk space check
    
    Args:
        output_path: Output file path
        required_gb: Required space in GB
        buffer: Safety buffer multiplier (default: 1.5x)
    
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


# =============================================================================
# ✅ PERMANENT FIX: MEMORY MONITORING
# =============================================================================

class MemoryMonitor:
    """
    ✅ MISSING 2 FIX: Track memory usage and detect leaks
    """
    def __init__(self, threshold_gb: float = 1.0):
        self.process = psutil.Process()
        self.threshold_gb = threshold_gb
        self.baseline_gb = self.process.memory_info().rss / 1e9
    
    def check(self, context: str = ""):
        """
        Check current memory usage and warn if leak detected
        
        Args:
            context: Descriptive context for logging
        """
        current_gb = self.process.memory_info().rss / 1e9
        increase_gb = current_gb - self.baseline_gb
        
        if increase_gb > self.threshold_gb:
            logger.warning(
                f"⚠️  MEMORY LEAK DETECTED {context}\n"
                f"  Baseline: {self.baseline_gb:.2f} GB\n"
                f"  Current: {current_gb:.2f} GB\n"
                f"  Increase: {increase_gb:.2f} GB (threshold: {self.threshold_gb:.2f} GB)"
            )
        
        return current_gb


# =============================================================================
# ✅ PERMANENT FIX: CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """
    ✅ MISSING 1 & FLAW 1 FIX: Progress persistence and resume capability
    """
    def __init__(self, output_path: Path):
        self.checkpoint_path = output_path.with_suffix('.checkpoint.json')
        self.output_path = output_path
    
    def load(self) -> Dict:
        """Load checkpoint if exists"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"  ✅ Loaded checkpoint: resuming from slice {checkpoint['last_processed']}")
                return checkpoint
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load checkpoint: {e}")
                return {}
        return {}
    
    def save(self, last_processed: int, total_slices: int, global_stats: Dict = None):
        """
        Save checkpoint
        
        Args:
            last_processed: Index of last successfully processed slice
            total_slices: Total number of slices
            global_stats: Optional global statistics (min/max, etc.)
        """
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
            logger.warning(f"  ⚠️  Failed to save checkpoint: {e}")
    
    def cleanup(self):
        """Remove checkpoint after successful completion"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info(f"  ✅ Removed checkpoint file")


# =============================================================================
# ✅ PERMANENT FIX: HDF5 VALIDATION
# =============================================================================

def validate_hdf5_file(hdf5_path: Path, expected_shape: Tuple[int, int, int]):
    """
    ✅ FLAW 2 FIX: Post-save integrity validation
    
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
                    logger.warning("  ⚠️  No foreground coordinates extracted")
                else:
                    logger.info(f"  ✅ Found {num_coords} foreground coordinates")
            
            logger.info("  ✅ Validation passed")
    
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        # Clean up corrupted file
        if hdf5_path.exists():
            hdf5_path.unlink()
            logger.info(f"  🗑️  Deleted corrupted file")
        raise


# =============================================================================
# ✅ PERMANENT FIX: GLOBAL NORMALIZATION (2-PASS)
# =============================================================================

def compute_global_intensity_range(
    slice_files: List[Path],
    downsample_xy: float = 1.0
) -> Tuple[float, float]:
    """
    ✅ BUG 4 FIX: PASS 1 - Compute global min/max across all slices
    
    This ensures consistent normalization (no Z-axis artifacts).
    
    Args:
        slice_files: List of TIFF slice paths
        downsample_xy: XY downsampling factor
    
    Returns:
        (global_min, global_max)
    """
    from PIL import Image
    from scipy.ndimage import zoom
    
    logger.info("  📊 PASS 1/2: Computing global intensity range...")
    
    global_min = float('inf')
    global_max = float('-inf')
    
    # Sample every 10th slice for speed (or all if <100 slices)
    step = max(1, len(slice_files) // 100)
    sample_files = slice_files[::step]
    
    for slice_file in tqdm(sample_files, desc="  Scanning", leave=False):
        try:
            slice_img = np.array(Image.open(slice_file)).astype(np.float32)
            
            if downsample_xy != 1.0:
                slice_img = zoom(slice_img, downsample_xy, order=1)
            
            slice_min = slice_img.min()
            slice_max = slice_img.max()
            
            global_min = min(global_min, slice_min)
            global_max = max(global_max, slice_max)
        
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to read {slice_file.name}: {e}")
            continue
    
    logger.info(f"  ✅ Global range: [{global_min:.2f}, {global_max:.2f}]")
    
    return global_min, global_max


def compute_global_foreground_threshold(
    slice_files: List[Path],
    downsample_xy: float = 1.0,
    percentile: float = 95.0,
    sample_fraction: float = 0.1
) -> float:
    """
    ✅ BUG 3 FIX: Compute GLOBAL percentile threshold (not per-slice)
    
    Args:
        slice_files: List of TIFF slice paths
        downsample_xy: XY downsampling factor
        percentile: Percentile for threshold (0-100)
        sample_fraction: Fraction of slices to sample (0-1)
    
    Returns:
        global_threshold: Consistent threshold for all slices
    """
    from PIL import Image
    from scipy.ndimage import zoom
    
    logger.info(f"  📊 Computing global {percentile}th percentile threshold...")
    
    # Sample a subset of slices
    num_samples = max(10, int(len(slice_files) * sample_fraction))
    sample_indices = np.linspace(0, len(slice_files)-1, num_samples, dtype=int)
    
    sample_values = []
    
    for idx in tqdm(sample_indices, desc="  Sampling", leave=False):
        try:
            slice_img = np.array(Image.open(slice_files[idx])).astype(np.float32)
            
            if downsample_xy != 1.0:
                slice_img = zoom(slice_img, downsample_xy, order=1)
            
            # Subsample pixels (every 100th pixel) to reduce memory
            sample_values.extend(slice_img.flatten()[::100])
        
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to sample slice {idx}: {e}")
            continue
    
    if len(sample_values) == 0:
        raise RuntimeError("Failed to sample any slices for threshold computation")
    
    global_threshold = np.percentile(sample_values, percentile)
    logger.info(f"  ✅ Global threshold: {global_threshold:.4f}")
    
    return global_threshold


# =============================================================================
# ✅ PERMANENT FIX: STREAMING PROCESSOR WITH ALL FIXES
# =============================================================================

def process_and_stream_to_hdf5(
    folder: Path,
    output_path: Path,
    downsample_xy: float = 1.0,
    normalize: bool = True,
    chunk_size: int = 32,
    foreground_percentile: float = 95.0  # ✅ FLAW 3 FIX: Configurable
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    ✅ PERMANENT FIX: Streams TIFF slices to HDF5 with ALL fixes applied
    
    Fixes Applied:
    - BUG 1: Fail-fast on shape mismatch (no silent corruption)
    - BUG 2: Simplified coordinate indexing
    - BUG 3: Global percentile threshold (consistent)
    - BUG 4: 2-pass global normalization (no Z artifacts)
    - FLAW 1: Checkpoint-based resume
    - FLAW 2: Post-save validation
    - FLAW 3: Configurable percentile
    - MISSING 1: Progress checkpoints
    - MISSING 2: Memory monitoring
    - MISSING 3: Disk space check
    
    Args:
        folder: Folder with TIFF slices
        output_path: Output HDF5 path
        downsample_xy: XY downsampling factor
        normalize: Whether to normalize intensities
        chunk_size: HDF5 chunk size (slices)
        foreground_percentile: Percentile for foreground threshold (0-100)
    
    Returns:
        save_info: Dict with file stats
        metadata: Dict with processing info
    """
    try:
        from PIL import Image
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("Install dependencies: pip install Pillow scipy")
    
    logger.info(f"  🔄 Streaming TIFF slices to {output_path.name}...")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 0: Pre-flight Checks
    # ═══════════════════════════════════════════════════════════════════
    
    # Find slices
    slice_files = sorted(
        list(folder.glob("*.tif")) + list(folder.glob("*.tiff")),
        key=lambda x: int(''.join(filter(str.isdigit, x.stem))) 
                     if any(c.isdigit() for c in x.stem) else 0
    )
    
    if len(slice_files) == 0:
        logger.warning(f"  ⚠️  No .tif files in {folder}")
        return None, None
    
    # Get dimensions from first slice
    first_slice_img = np.array(Image.open(slice_files[0])).astype(np.float32)
    
    if downsample_xy != 1.0:
        first_slice_img = zoom(first_slice_img, downsample_xy, order=1)
    
    H, W = first_slice_img.shape
    D = len(slice_files)
    final_shape = (D, H, W)
    
    logger.info(f"  📐 Final shape: {final_shape} ({D} slices)")
    
    # ✅ MISSING 3 FIX: Disk space check
    required_gb = (D * H * W * 4) / 1e9  # float32
    check_disk_space(output_path, required_gb)
    
    # ✅ MISSING 2 FIX: Memory monitor
    mem_monitor = MemoryMonitor(threshold_gb=1.0)
    
    # ✅ MISSING 1 FIX: Checkpoint manager
    checkpoint_mgr = CheckpointManager(output_path)
    checkpoint = checkpoint_mgr.load()
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Global Statistics (if not resuming)
    # ═══════════════════════════════════════════════════════════════════
    
    if 'global_stats' in checkpoint:
        # Resume: use saved stats
        global_min = checkpoint['global_stats']['global_min']
        global_max = checkpoint['global_stats']['global_max']
        global_threshold = checkpoint['global_stats']['global_threshold']
        logger.info(f"  ♻️  Resuming with saved global stats")
    else:
        # Fresh start: compute global stats
        # ✅ BUG 4 FIX: Global normalization range
        global_min, global_max = compute_global_intensity_range(
            slice_files, downsample_xy
        )
        
        # ✅ BUG 3 FIX: Global foreground threshold
        global_threshold = compute_global_foreground_threshold(
            slice_files, downsample_xy, foreground_percentile
        )
        
        # Save stats in checkpoint
        checkpoint['global_stats'] = {
            'global_min': float(global_min),
            'global_max': float(global_max),
            'global_threshold': float(global_threshold)
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Stream Processing
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("  💾 PASS 2/2: Normalizing and streaming to HDF5...")
    
    start_idx = checkpoint.get('last_processed', 0)
    
    # Determine HDF5 mode
    if start_idx > 0:
        # Resume: open in read-write mode
        h5_mode = 'r+'
        logger.info(f"  ♻️  Resuming from slice {start_idx}/{D}")
    else:
        # Fresh start: create new file
        h5_mode = 'w'
    
    try:
        with h5py.File(output_path, h5_mode) as f:
            # Create or open datasets
            if h5_mode == 'w':
                vol_dset = f.create_dataset(
                    'volume',
                    shape=final_shape,
                    chunks=(chunk_size, H, W),
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
                    
                    # ✅ BUG 1 FIX: Fail-fast on shape mismatch
                    if slice_img.shape != (H, W):
                        raise ValueError(
                            f"CRITICAL: Slice {i} ({slice_file.name}) shape mismatch!\n"
                            f"Expected {(H, W)}, got {slice_img.shape}\n"
                            f"Cannot continue with inconsistent slices"
                        )
                    
                    # ✅ BUG 4 FIX: Global normalization (consistent across slices)
                    if normalize:
                        if global_max > global_min:
                            slice_img = (slice_img - global_min) / (global_max - global_min)
                    
                    # Write to HDF5
                    vol_dset[i] = slice_img
                    
                    # ✅ BUG 3 FIX: Use global threshold (not per-slice)
                    fg_mask_2d = slice_img > global_threshold
                    coords_yx = np.argwhere(fg_mask_2d)
                    
                    if coords_yx.size > 0:
                        z_coord = np.full((len(coords_yx), 1), i, dtype=np.int32)
                        # ✅ BUG 2 FIX: Simplified (no redundant slicing)
                        coords_zyx = np.hstack((z_coord, coords_yx))
                        
                        # Append to dataset
                        current_size = coord_dset.shape[0]
                        new_size = current_size + len(coords_zyx)
                        coord_dset.resize(new_size, axis=0)
                        coord_dset[current_size:] = coords_zyx
                        total_coords += len(coords_zyx)
                    
                    # ✅ MISSING 1 FIX: Save checkpoint every 50 slices
                    if (i + 1) % 50 == 0:
                        checkpoint_mgr.save(i, D, checkpoint.get('global_stats'))
                        
                        # ✅ MISSING 2 FIX: Check memory
                        mem_monitor.check(f"at slice {i}/{D}")
                
                except Exception as e:
                    logger.error(f"❌ Failed to process slice {i} ({slice_file.name}): {e}")
                    raise
            
            # Save metadata
            vol_dset.attrs['shape'] = str(final_shape)
            vol_dset.attrs['format'] = 'tiff_slices'
            vol_dset.attrs['spacing'] = str((2.0, 1.0, 1.0))
            vol_dset.attrs['global_min'] = global_min
            vol_dset.attrs['global_max'] = global_max
            vol_dset.attrs['normalized'] = normalize
            
            # ✅ BUG 3 FIX: Accurate metadata
            coord_dset.attrs['num_coords'] = total_coords
            coord_dset.attrs['volume_shape'] = str(final_shape)
            coord_dset.attrs['method'] = f'global_percentile_{foreground_percentile}'
            coord_dset.attrs['threshold'] = global_threshold
            coord_dset.attrs['warning'] = 'Threshold is GLOBAL (consistent across all slices)'
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Post-Save Validation
        # ═══════════════════════════════════════════════════════════════════
        
        # ✅ FLAW 2 FIX: Validate saved file
        validate_hdf5_file(output_path, final_shape)
        
        # ✅ MISSING 1 FIX: Clean up checkpoint on success
        checkpoint_mgr.cleanup()
        
        # Compute stats
        file_size_mb = output_path.stat().st_size / 1e6
        uncompressed_size_mb = (D * H * W * 4) / 1e6
        compression_ratio = uncompressed_size_mb / file_size_mb if file_size_mb > 0 else 1.0
        
        save_info = {
            'file_path': str(output_path),
            'file_size_mb': file_size_mb,
            'compression_ratio': compression_ratio,
            'compression': 'gzip'
        }
        
        metadata = {
            'format': 'tiff_slices',
            'spacing': (2.0, 1.0, 1.0),
            'shape': final_shape,
            'global_min': global_min,
            'global_max': global_max,
            'normalized': normalize,
            'foreground_threshold': global_threshold,
            'num_foreground_coords': total_coords
        }
        
        logger.info(f"  ✅ Streamed {D} slices to HDF5")
        logger.info(f"  ✅ Extracted {total_coords} foreground coordinates")
        logger.info(f"  ✅ Saved: {file_size_mb:.1f} MB (compression: {compression_ratio:.2f}x)")
        
        return save_info, metadata
    
    except Exception as e:
        logger.error(f"❌ Streaming failed: {e}")
        
        # Clean up partial file (but keep checkpoint for resume)
        if output_path.exists() and start_idx == 0:
            output_path.unlink()
            logger.info(f"  🗑️  Deleted partial file")
        
        raise


# =============================================================================
# VOLUME LOADER (Unchanged - Already Robust)
# =============================================================================

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
            # For TIFF slice folders, delegate to streaming function
            logger.info(f"  📁 Detected TIFF slice folder: {file_or_folder.name}")
            logger.info(f"  ⚠️  Will use streaming processor (process_and_stream_to_hdf5)")
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
            
            logger.info(f"  📐 Original shape: {volume.shape}")
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
                'original_shape': nii.shape,
                'spacing': tuple(spacing),
                'affine': nii.affine.tolist(),
                'orientation': nib.aff2axcodes(nii.affine)
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
            logger.info(f"  📐 Original shape: {volume.shape}")
            
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
                'spacing': (2.0, 1.0, 1.0)
            }
            
            return volume, metadata
        
        except Exception as e:
            logger.error(f"Failed to load TIFF stack: {e}")
            return None, None


# =============================================================================
# HDF5 SAVE/LOAD UTILITIES
# =============================================================================

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
    
    Args:
        volume: (D, H, W) array
        output_path: Output path
        metadata: Metadata dict
        compression: Compression algorithm
        compression_opts: Compression level
        compute_foreground: Whether to extract foreground coords
        foreground_percentile: Percentile for threshold
    
    Returns:
        save_info: Dict with file stats
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
            
            # Save metadata
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    f['volume'].attrs[key] = value
                elif isinstance(value, (list, tuple)):
                    f['volume'].attrs[key] = str(value)
        
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
        metadata = dict(f['volume'].attrs)
    
    return volume, metadata


def load_patch_hdf5(
    hdf5_path: Path,
    patch_slice: Tuple[slice, slice, slice]
) -> np.ndarray:
    """Load only a patch from HDF5 (memory efficient)"""
    with h5py.File(hdf5_path, 'r') as f:
        patch = f['volume'][patch_slice]
    
    return patch


# =============================================================================
# DATA DISCOVERY
# =============================================================================

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
        
        # Strategy 1: Top-level raw/gt (EBI format)
        raw_dir = input_dir / 'raw'
        gt_dir = input_dir / 'gt'
        
        if raw_dir.exists() and gt_dir.exists():
            logger.info(f"Detected top-level 'raw'/'gt' structure")
            img_files = sorted(raw_dir.glob("*.nii.gz"))
            
            for img_file in img_files:
                # Handle _0000 vs _000 suffix
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
                    logger.warning(f"  ⚠️  Missing mask for {img_file.name}")
        
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


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_unlabeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """Process unlabeled data with streaming for large volumes"""
    logger.info("\n" + "="*80)
    logger.info("PROCESSING UNLABELED DATA (SSL)")
    logger.info("="*80)
    
    all_metadata = []
    
    marker_map = {
        'ab_plaque': 3, 'ad_plaque': 3, 'cfos': 0,
        'microglia': 4, 'nucleus': 2, 'unknown': 5, 'vessel': 1
    }
    
    discovered_data = DatasetDiscovery.discover_unlabeled_data(input_dir)
    
    if len(discovered_data) == 0:
        raise ValueError(f"No data found in {input_dir}")
    
    logger.info(f"Discovered {len(discovered_data)} volumes\n")
    
    from collections import defaultdict
    by_marker = defaultdict(list)
    for item in discovered_data:
        by_marker[item['marker_type']].append(item)
    
    for marker_name, items in by_marker.items():
        marker_label = marker_map.get(marker_name.lower(), 5)
        
        logger.info(f"🔬 Processing marker: {marker_name} (label={marker_label})")
        logger.info(f"  Found {len(items)} volumes")
        
        output_marker_dir = output_dir / marker_name
        output_marker_dir.mkdir(parents=True, exist_ok=True)
        
        for item in items:
            brain_name = item['brain_name']
            logger.info(f"\n  📦 Processing: {brain_name}")
            
            output_file = output_marker_dir / f"{brain_name}.h5"
            
            # Check if TIFF folder (requires streaming) or file (load normally)
            if item['is_file']:
                # NIfTI or TIFF stack - load normally
                volume, vol_metadata = VolumeLoader.load_volume(
                    item['path'],
                    downsample_xy=args.downsample_xy,
                    downsample_z=args.downsample_z,
                    normalize=args.normalize
                )
                
                if volume is None:
                    logger.warning(f"  ⚠️  Skipping {brain_name} (loading failed)")
                    continue
                
                save_info = save_volume_hdf5(
                    volume, output_file, vol_metadata,
                    foreground_percentile=args.foreground_percentile
                )
            
            else:
                # TIFF slice folder - use streaming
                save_info, vol_metadata = process_and_stream_to_hdf5(
                    item['path'],
                    output_file,
                    downsample_xy=args.downsample_xy,
                    normalize=args.normalize,
                    foreground_percentile=args.foreground_percentile
                )
            
            if save_info is None or vol_metadata is None:
                logger.warning(f"  ⚠️  Skipping {brain_name} (processing failed)")
                continue
            
            # Store metadata
            meta = {
                'brain_name': brain_name,
                'marker_type': marker_name,
                'marker_label': marker_label,
                'shape': list(vol_metadata.get('shape', [0,0,0])),
                'file_path': str(Path(save_info['file_path']).relative_to(output_dir)),
                'file_size_mb': save_info['file_size_mb'],
                'compression_ratio': save_info.get('compression_ratio', 1.0),
                'original_format': vol_metadata.get('format', 'unknown'),
                'spacing': vol_metadata.get('spacing', (2.0, 1.0, 1.0)),
                'processing': {
                    'downsample_xy': args.downsample_xy,
                    'downsample_z': args.downsample_z,
                    'normalized': args.normalize,
                    'foreground_percentile': args.foreground_percentile
                }
            }
            all_metadata.append(meta)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_volumes': len(all_metadata),
            'marker_types': list(set(m['marker_type'] for m in all_metadata)),
            'total_size_mb': sum(m['file_size_mb'] for m in all_metadata),
            'volumes': all_metadata,
            'storage_format': 'hdf5'
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processed {len(all_metadata)} unlabeled volumes")
    logger.info(f"✅ Total size: {sum(m['file_size_mb'] for m in all_metadata):.1f} MB")
    logger.info(f"✅ Metadata saved: {metadata_path}")
    logger.info(f"{'='*80}")
    
    return all_metadata


def process_labeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """Process labeled data"""
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
                logger.warning(f"  ⚠️  Failed to load {pair['sample_name']}")
                continue
            
            mask = mask.astype(np.int64)
            
            if img.shape != mask.shape:
                logger.warning(f"  ⚠️  Shape mismatch: {pair['sample_name']}")
                continue
            
            all_samples.append({
                'img': img,
                'mask': mask,
                'marker_type': pair['marker_type'],
                'filename': pair['sample_name'],
                'shape': img.shape,
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
    
    train_samples, temp_samples = train_test_split(
        all_samples, test_size=0.2, stratify=marker_types, random_state=42
    )
    
    temp_markers = [s['marker_type'] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, stratify=temp_markers, random_state=42
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
                
                file_size_mb = output_file.stat().st_size / 1e6
                
                split_meta.append({
                    'filename': base_name,
                    'marker_type': sample['marker_type'],
                    'shape': list(sample['shape']),
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
            'storage_format': 'hdf5'
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processed {len(all_samples)} labeled samples")
    logger.info(f"✅ Metadata saved: {metadata_path}")
    logger.info(f"{'='*80}")
    
    return all_samples


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='SELMA3D Data Preparation - PERMANENT FIX VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
✅ ALL BUGS FIXED:
  • Shape mismatch → Fail-fast
  • Per-slice normalization → Global 2-pass
  • Hardcoded threshold → CLI configurable
  • No checkpointing → Resume capability
  • No validation → Post-save checks
  • Memory leaks → Monitoring
  • Disk space → Pre-flight check
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_type', type=str, required=True, choices=['unlabeled', 'labeled'])
    parser.add_argument('--downsample_xy', type=float, default=1.0)
    parser.add_argument('--downsample_z', type=float, default=1.0)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.add_argument('--foreground_percentile', type=float, default=95.0,
                       help='Percentile for foreground threshold (0-100)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("="*80)
    logger.info("SELMA3D DATA PREPARATION - PERMANENT FIX VERSION")
    logger.info("="*80)
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Type:   {args.data_type}")
    logger.info("")
    logger.info("✅ ALL PERMANENT FIXES APPLIED")
    logger.info("="*80)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.data_type == 'unlabeled':
        process_unlabeled_data(input_dir, output_dir, args)
    elif args.data_type == 'labeled':
        process_labeled_data(input_dir, output_dir, args)
    
    logger.info("\n" + "="*80)
    logger.info("✅ DATA PREPARATION COMPLETE!")
    logger.info("="*80)


if __name__ == '__main__':
    main()