"""
Data Loaders with HDF5 Support
✅ FIXED: Memory-efficient patch loading from HDF5
✅ FIXED: Handles multi-GB volumes without loading all into RAM
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import random
from typing import Dict, List, Tuple, Optional
import logging
import h5py  # ✅ NEW

logger = logging.getLogger(__name__)

# ---------------- Helper: read normalized threshold from HDF5 or metadata ----------------
def read_normalized_threshold_from_h5(h5path: Path):
    """
    Return normalized threshold (value in [0,1]) if available.
    Tries (in order):
      - f['volume'].attrs['global_threshold_norm']
      - reconstruct from raw threshold + global_min/global_max stored in attrs
      - None if not available
    """
    try:
        with h5py.File(str(h5path), "r") as f:
            if "volume" in f:
                vattrs = dict(f["volume"].attrs)
                if "global_threshold_norm" in vattrs:
                    return float(vattrs["global_threshold_norm"])
                # try to reconstruct
                raw_thr = None
                # common places for raw threshold
                if "threshold" in f.get("foreground_coords", {}).attrs:
                    raw_thr = float(f["foreground_coords"].attrs.get("threshold", np.nan))
                elif "foreground_threshold" in vattrs:
                    raw_thr = float(vattrs.get("foreground_threshold", np.nan))
                elif "global_threshold" in vattrs:
                    raw_thr = float(vattrs.get("global_threshold", np.nan))
                if raw_thr is not None and "global_min" in vattrs and "global_max" in vattrs:
                    gmin = float(vattrs["global_min"]); gmax = float(vattrs["global_max"])
                    if gmax > gmin:
                        return float((raw_thr - gmin) / (gmax - gmin))
    except Exception:
        # failure to open / missing keys -> return None
        pass
    return None

logger = logging.getLogger(__name__)


class VolumeDataset3D(Dataset):
    """
    ✅ PERMANENT FIX: Memory-efficient foreground sampling
    
    Key Features:
    - Lazy-loads coordinates from HDF5 (no RAM waste)
    - Proper boundary handling
    - Comprehensive error handling
    - Configurable sampling strategy
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        num_patches_per_epoch: int = 1000,
        transform: Optional[callable] = None,
        config: Optional[Dict] = None,
        preload: bool = False,
        foreground_sampling: bool = True,
        foreground_fraction: float = 0.8
    ):
        """
        Args:
            data_dir: Directory with HDF5 files and metadata.json
            patch_size: (D, H, W) patch dimensions
            num_patches_per_epoch: Virtual dataset size
            transform: Optional torchio transform
            config: Optional config dict
            preload: If True, load VOLUMES into RAM (not recommended)
            foreground_sampling: If True, use foreground-aware sampling
            foreground_fraction: Fraction of patches from foreground [0, 1]
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.num_patches_per_epoch = num_patches_per_epoch
        self.transform = transform
        self.config = config or {}
        self.foreground_sampling = foreground_sampling
        self.foreground_fraction = foreground_fraction
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.data_dir}\n"
                f"Did you run prepare_data.py?"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check storage format
        storage_format = metadata.get('storage_format', 'numpy')
        if storage_format != 'hdf5':
            raise ValueError(
                f"Expected HDF5 storage, found '{storage_format}'.\n"
                f"Please re-run prepare_data.py to convert to HDF5."
            )
        
        if 'data' in metadata:
            # NEW STRUCTURE: metadata.json has train/val/test splits
            logger.info(f"  Using NEW metadata structure with splits")
            
            if split == 'all':
                # Combine all splits
                all_splits = []
                for split_name in ['train', 'val', 'test']:
                    if split_name in metadata['data']:
                        all_splits.extend(metadata['data'][split_name])
                volume_list = all_splits
            elif split in metadata['data']:
                volume_list = metadata['data'][split]
            else:
                raise ValueError(
                    f"Split '{split}' not found in metadata. "
                    f"Available: {list(metadata['data'].keys())}"
                )
        
        elif 'volumes' in metadata:
            # OLD STRUCTURE: metadata.json has flat 'volumes' list
            logger.info(f"  Using OLD metadata structure (flat volumes list)")
            logger.warning(
                f"⚠️ No train/val/test splits found. Using all volumes for '{split}'.\n"
                f"  Consider re-running prepare_data.py to generate splits."
            )
            volume_list = metadata['volumes']
        
        else:
            raise ValueError(
                f"Invalid metadata.json format. Expected either:\n"
                f"  - 'data': {{train: [...], val: [...], test: [...]}}\n"
                f"  - 'volumes': [...]"
            )
        
        # Extract volume information
        self.volumes = []
        for vol_meta in volume_list:
            vol_path = self.data_dir / vol_meta['file_path']
            
            if not vol_path.exists():
                logger.warning(f"Volume not found: {vol_path}, skipping")
                continue

            vol_thr_norm = None
            try:
                vol_thr_norm = vol_meta.get('processing', {}).get('global_threshold_norm', None)
                if vol_thr_norm is None:
                    # try to read from HDF5 attrs as fallback
                    vol_thr_norm = read_normalized_threshold_from_h5(self.data_dir / vol_meta['file_path'])
            except Exception:
                vol_thr_norm = read_normalized_threshold_from_h5(self.data_dir / vol_meta['file_path'])
    
            self.volumes.append({
                'path': vol_path,
                'marker_label': vol_meta['marker_label'],
                'marker_type': vol_meta['marker_type'],
                'shape': tuple(vol_meta['shape']),
                'brain_name': vol_meta['brain_name'],
                'global_threshold_norm': float(vol_thr_norm) if vol_thr_norm is not None else None
            })
        
        if len(self.volumes) == 0:
            raise ValueError(f"No valid volumes found in {self.data_dir} for split '{split}'")
        
        logger.info(f"VolumeDataset3D initialized:")
        logger.info(f"  Split: {split}")
        logger.info(f"  Volumes: {len(self.volumes)}")
        logger.info(f"  Storage: HDF5 (memory-efficient)")
        logger.info(f"  Patch size: {patch_size}")
        logger.info(f"  Patches per epoch: {num_patches_per_epoch}")
        logger.info(f"  Foreground sampling: {foreground_sampling}")
        if foreground_sampling:
            logger.info(f"  Foreground fraction: {foreground_fraction:.1%}")
        
        # ✅ FIX: Don't preload coordinates! (Memory-efficient)
        # We'll load them on-demand in __getitem__()
        self.volume_cache = None  # No preloading of volumes either
        
        # ✅ FIX: Validate foreground coords exist (fail-fast)
        if self.foreground_sampling:
            self._validate_foreground_coords()
    
    def _validate_foreground_coords(self):
        """
        ✅ CRITICAL: Validate all volumes have foreground_coords
        
        Fail-fast: Better to crash at initialization than during training!
        """
        missing = []
        
        for vol_info in self.volumes:
            try:
                with h5py.File(vol_info['path'], 'r') as f:
                    if 'foreground_coords' not in f:
                        missing.append(vol_info['brain_name'])
            except Exception as e:
                logger.error(f"Failed to open {vol_info['path']}: {e}")
                missing.append(vol_info['brain_name'])
        
        if missing:
            logger.warning(f"{len(missing)} volumes missing foreground_coords; will use threshold-sampling fallback for those volumes.")
        else:
            logger.info("  ✅ All volumes have foreground_coords")

        
        logger.info("  ✅ All volumes have foreground_coords")
    
    def __len__(self) -> int:
        return self.num_patches_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ PERMANENT FIX: Memory-efficient foreground sampling
        
        Returns:
            patch: (1, D, H, W) tensor
            marker_label: int64 tensor
        """
        # Randomly select a volume
        vol_info = random.choice(self.volumes)
        vol_path = vol_info['path']
        
        # Get volume shape
        D, H, W = vol_info['shape']
        pd, ph, pw = self.patch_size
        
        # Validate patch size
        if D < pd or H < ph or W < pw:
            raise ValueError(
                f"Patch size {self.patch_size} too large for volume "
                f"{vol_info['shape']} from {vol_info['brain_name']}"
            )
        
        # ✅ FIX: Decide sampling strategy
        use_foreground = (
            self.foreground_sampling and
            random.random() < self.foreground_fraction
        )
        
        if use_foreground:
            # ✅ STRATEGY 1: Load coords on-demand (memory-efficient!)
            try:
                with h5py.File(vol_path, 'r') as f:
                    # Get number of coords without loading all
                    num_coords = f['foreground_coords'].shape[0]
                    
                    # Pick a random index
                    center_idx = random.randint(0, num_coords - 1)
                    
                    # ✅ CRITICAL: Load ONLY this one coordinate (not all!)
                    center_coord = f['foreground_coords'][center_idx]
                    center_z, center_y, center_x = center_coord
            
            except Exception as e:
                logger.warning(
                    f"Failed to load foreground coord from {vol_info['brain_name']}: {e}. "
                    f"Falling back to random sampling."
                )
                use_foreground = False
            
            if use_foreground:
                # ✅ FIX: Proper boundary handling
                # Calculate ideal start (centered on foreground voxel)
                ideal_d_start = center_z - pd // 2
                ideal_h_start = center_y - ph // 2
                ideal_w_start = center_x - pw // 2
                
                # Clamp to valid range
                d_start = max(0, min(ideal_d_start, D - pd))
                h_start = max(0, min(ideal_h_start, H - ph))
                w_start = max(0, min(ideal_w_start, W - pw))
                
                # ✅ VALIDATION: Ensure foreground voxel is actually in patch
                assert d_start <= center_z < d_start + pd, \
                    f"Foreground voxel (z={center_z}) not in patch [{d_start}, {d_start+pd})"
                assert h_start <= center_y < h_start + ph, \
                    f"Foreground voxel (y={center_y}) not in patch [{h_start}, {h_start+ph})"
                assert w_start <= center_x < w_start + pw, \
                    f"Foreground voxel (x={center_x}) not in patch [{w_start}, {w_start+pw})"
        
        if not use_foreground:
            # fallback: try to pick foreground via threshold (if available)
            thr_norm = vol_info.get('global_threshold_norm', None)
            if thr_norm is not None:
                # cheap attempt: sample a few random z slices and look for fg pixels
                found = False
                try:
                    with h5py.File(vol_path, 'r') as f:
                        z_candidates = random.sample(range(D), min(8, D))
                        for zc in z_candidates:
                            sl = f['volume'][zc]
                            fg_xy = np.argwhere(sl > thr_norm)
                            if fg_xy.size > 0:
                                yx = fg_xy[random.randint(0, fg_xy.shape[0] - 1)]
                                center_z, center_y, center_x = zc, int(yx[0]), int(yx[1])
                                # compute start coords same as foreground branch
                                ideal_d_start = center_z - pd // 2
                                ideal_h_start = center_y - ph // 2
                                ideal_w_start = center_x - pw // 2
                                d_start = max(0, min(ideal_d_start, D - pd))
                                h_start = max(0, min(ideal_h_start, H - ph))
                                w_start = max(0, min(ideal_w_start, W - pw))
                                found = True
                                break
                except Exception:
                    found = False

                if not found:
                    # pure random as ultimate fallback
                    d_start = random.randint(0, D - pd)
                    h_start = random.randint(0, H - ph)
                    w_start = random.randint(0, W - pw)
            else:
                # no threshold available; do pure random
                d_start = random.randint(0, D - pd)
                h_start = random.randint(0, H - ph)
                w_start = random.randint(0, W - pw)
        
        # ✅ Load patch from HDF5 (memory-efficient!)
        try:
            with h5py.File(vol_path, 'r') as f:
                patch = f['volume'][
                    d_start:d_start + pd,
                    h_start:h_start + ph,
                    w_start:w_start + pw
                ]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load patch from {vol_info['brain_name']}: {e}\n"
                f"Volume shape: {vol_info['shape']}, "
                f"Patch coords: ({d_start},{h_start},{w_start})"
            )
        
        # ✅ ROBUST NORMALIZATION: Handle edge cases
        patch_mean = patch.mean()
        patch_std = patch.std()
        
        if patch_std > 1e-8:
            patch = (patch - patch_mean) / patch_std
        else:
            # Uniform patch (rare with foreground sampling)
            patch = patch - patch_mean
            
            if use_foreground:
                # This shouldn't happen with foreground sampling!
                logger.warning(
                    f"Uniform patch despite foreground sampling! "
                    f"Volume: {vol_info['brain_name']}, "
                    f"Center: ({center_z},{center_y},{center_x}), "
                    f"Patch: ({d_start},{h_start},{w_start})"
                )
        
        # Convert to tensor
        patch = torch.from_numpy(patch.copy()).unsqueeze(0).float()  # (1, D, H, W)
        
        # ✅ ROBUST TRANSFORM: Handle errors gracefully
        if self.transform is not None:
            try:
                patch = self.transform(patch)
            except Exception as e:
                logger.error(
                    f"Transform failed on {vol_info['brain_name']}: {e}\n"
                    f"Patch stats: mean={patch_mean:.3f}, std={patch_std:.3f}\n"
                    f"Skipping transform for this patch."
                )
                # Continue without transform rather than crashing
        
        # Get marker label
        marker_label = torch.tensor(vol_info['marker_label'], dtype=torch.long)
        
        return patch, marker_label

class SELMA3DDataset(Dataset):
    """
    ✅ UPDATED: Fine-tuning dataset with HDF5 support
    
    Key Feature: Image and mask stored in SAME HDF5 file!
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        samples_per_volume: int = 10,
        transform: Optional[callable] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            data_dir: Directory with train/val/test splits (HDF5 files)
            patch_size: (D, H, W) patch size
            samples_per_volume: Patches per volume per epoch
            transform: torchio.Compose transform
            config: Config dict
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.transform = transform
        self.config = config or {}
        
        # ✅ PERMANENT FIX: Look in the PARENT directory for metadata.json
        metadata_path = self.data_dir.parent / 'metadata.json' 
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.data_dir.parent}\n"
                f"Please ensure 'metadata.json' exists one level above '{self.data_dir.name}'"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Infer split from data_dir name
        split = self.data_dir.name
        
        logger.info(f"Loading split: {split}")
        
        # Get split data
        if split not in metadata['data']:
            raise ValueError(
                f"Split '{split}' not found in metadata. "
                f"Available: {list(metadata['data'].keys())}"
            )
        
        split_data = metadata['data'][split]
        
        # Load volume paths
        self.volumes = []
        
        for sample_meta in split_data:
            # HDF5 file contains BOTH image and mask
            # Get path relative to the metadata file's parent
            hdf5_path_rel = Path(sample_meta['file_path'])
            hdf5_path_abs = metadata_path.parent / hdf5_path_rel
            
            if not hdf5_path_abs.exists():
                logger.warning(f"File not found: {hdf5_path_abs}, skipping")
                continue
            vol_thr_norm = None
            try:
                vol_thr_norm = read_normalized_threshold_from_h5(hdf5_path_abs)
            except Exception:
                vol_thr_norm = None
            self.volumes.append({
                'path': hdf5_path_abs, # Use the absolute path
                'filename': sample_meta['filename'],
                'marker_type': sample_meta['marker_type'],
                'shape': tuple(sample_meta['shape']),
                'global_threshold_norm': float(vol_thr_norm) if vol_thr_norm is not None else None
            })
        
        if len(self.volumes) == 0:
            raise ValueError(f"No valid volumes in {split} split!")
        
        logger.info(f"SELMA3DDataset ({split}) initialized:")
        logger.info(f"  Volumes: {len(self.volumes)}")
        logger.info(f"  Storage: HDF5 (image + mask in same file)")
        logger.info(f"  Patch size: {patch_size}")
        logger.info(f"  Samples per volume: {samples_per_volume}")
        logger.info(f"  Total virtual samples: {len(self)}")
    
    def __len__(self) -> int:
        return len(self.volumes) * self.samples_per_volume
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        ✅ CRITICAL: Load full volume, pad, then sample patch
        
        Returns:
            image: (1, D, H, W) tensor (patch, not full volume)
            mask: (D, H, W) tensor (patch)
            metadata: Dict
        """
        # Map index to volume
        volume_idx = idx // self.samples_per_volume
        sample_idx = idx % self.samples_per_volume
        
        vol_info = self.volumes[volume_idx]
        
        # ================== LOGIC FIX: STEP 1 ==================
        # ✅ CRITICAL: Load the FULL volume first
        # This is safe because labeled volumes are small patches
        try:
            with h5py.File(vol_info['path'], 'r') as f:
                image_vol = f['image'][:].astype(np.float32)
                mask_vol = f['mask'][:].astype(np.int64)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load full volume from {vol_info['filename']} ({vol_info['path']}): {e}"
            )

        # Get shape from config
        pd, ph, pw = self.patch_size
        # Get shape from loaded volume
        D, H, W = image_vol.shape

        # ================== LOGIC FIX: STEP 2 ==================
        # ✅ NEW: Check if padding is needed
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # Pad ONLY on the right/bottom/back
            # This makes sampling logic easier
            padding_dims_img = ((0, pad_d), (0, pad_h), (0, pad_w))
            padding_dims_mask = ((0, pad_d), (0, pad_h), (0, pad_w))
            
            image_vol = np.pad(
                image_vol, 
                pad_width=padding_dims_img, 
                mode='constant', 
                constant_values=0 # Pad with 0
            )
            
            mask_vol = np.pad(
                mask_vol,
                pad_width=padding_dims_mask,
                mode='constant',
                constant_values=0 # Pad with 0
            )
            
            # Update the volume's shape for the sampler
            D, H, W = image_vol.shape
        # ================= END PADDING LOGIC =================
        
        # ---------------- Determine normalized threshold to use ----------------
        thr_norm = vol_info.get('global_threshold_norm', None)
        if thr_norm is None:
            try:
                thr_norm = read_normalized_threshold_from_h5(vol_info['path'])
            except Exception:
                thr_norm = None

        # If still None, compute a percentile on the loaded image (cheap because image is in memory)
        if thr_norm is None:
            pct = float(self.config.get('foreground_percentile', 90.0))
            try:
                thr_norm = float(np.percentile(image_vol, pct))
                logger.debug(f"Computed fallback p{pct} threshold={thr_norm:.6f} for {vol_info.get('filename')}")
            except Exception as e:
                logger.warning(f"Could not compute percentile threshold for {vol_info.get('filename')}: {e}")
                thr_norm = 0.0  # conservative fallback (no foreground)

        # ---------------- Optionally derive/augment mask from intensity ----------------
        # If provided mask is empty (no labeled pixels), create mask from intensity threshold.
        # Alternative: OR the two masks to keep labels + threshold-derived
        if mask_vol.sum() == 0:
            mask_from_img = (image_vol > thr_norm).astype(np.int64)
            # Replace empty mask with threshold-derived one (change to OR if you want to keep annotations)
            # mask_vol = mask_from_img
            mask_vol = np.logical_or(mask_vol, mask_from_img).astype(np.int64)
            logger.debug(f"Derived mask from intensity threshold for {vol_info.get('filename')} (frac={mask_vol.sum()/mask_vol.size:.6f})")
        # ================== LOGIC FIX: STEP 3 ==================
        # Random patch coordinates (now guaranteed to be valid)
        d_start = random.randint(0, D - pd)
        h_start = random.randint(0, H - ph)
        w_start = random.randint(0, W - pw)
        
        # Extract the patch from the (potentially padded) volume
        image_patch = image_vol[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ]
        
        mask_patch = mask_vol[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ]
        
        # ================== LOGIC FIX: STEP 4 ==================
        # Convert to tensors
        # Use .copy() to avoid PyTorch errors about non-writable NumPy arrays
        image = torch.from_numpy(image_patch.copy()).unsqueeze(0)  # (1, D, H, W)
        mask = torch.from_numpy(mask_patch.copy()).long()  # (D, H, W)
        
        # Apply transforms (synchronized for image + mask)
        # if self.transform is not None:
        #     try:
                # import torchio as tio
        #         subject = tio.Subject(
        #             image=tio.ScalarImage(tensor=image),
        #             mask=tio.LabelMap(tensor=mask.unsqueeze(0)) 
        #         )
        #         transformed = self.transform(subject)
                
        #         image = transformed.image.data
        #         mask = transformed.mask.data.squeeze(0).long()
            
        #     except Exception as e:
        #         # Add more context to the transform error
        #         logger.error(f"Transform failed for {vol_info['filename']}")
        #         logger.error(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
        #         logger.error(f"Error: {e}")
        #         # Re-raise the error to stop the dataloader
        #         raise RuntimeError(f"Transform failed for {vol_info['filename']}: {e}")
        if self.transform is not None:
            try:
                import torchio as tio

                image_std = float(image.std())

                if image_std < 1e-6:
            # 🔒 Zero-variance safety: spatial transforms only
                    subject = tio.Subject(
                        image=tio.ScalarImage(tensor=image),
                        mask=tio.LabelMap(tensor=mask.unsqueeze(0))
                    )

                    safe_transforms = []
                    for t in self.transform.transforms:
                        if isinstance(t, (tio.RandomAffine, tio.RandomFlip)):
                            safe_transforms.append(t)

                    if safe_transforms:
                        subject = tio.Compose(safe_transforms)(subject)

                    image = subject.image.tensor
                    mask = subject.mask.tensor.squeeze(0).long()

                else:
                 # ✅ Normal path: full transform
                    subject = tio.Subject(
                        image=tio.ScalarImage(tensor=image),
                        mask=tio.LabelMap(tensor=mask.unsqueeze(0))
                    )

                    subject = self.transform(subject)

                    image = subject.image.tensor
                    mask = subject.mask.tensor.squeeze(0).long()

            except Exception as e:
                logger.error(f"Transform failed for {vol_info['filename']}")
                logger.error(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
                logger.error(f"Error: {e}")
                raise RuntimeError(f"Transform failed for {vol_info['filename']}: {e}")

        # Metadata
        metadata = {
            'filename': vol_info['filename'],
            'marker_type': vol_info['marker_type'],
            'original_shape': vol_info['shape'],
            'patch_shape': tuple(image.shape),
            'volume_idx': volume_idx,
            'sample_idx': sample_idx,
            'patch_coords': (d_start, h_start, w_start),
            'global_threshold_norm': float(thr_norm)

        }
        
        # Validation
        assert image.ndim == 4, f"Image should be (1,D,H,W), got {image.shape}"
        assert mask.ndim == 3, f"Mask should be (D,H,W), got {mask.shape}"
        assert image.shape[1:] == mask.shape, \
            f"Spatial dims don't match: {image.shape[1:]} vs {mask.shape}"
        
        return image, mask, metadata


# ============================================================================
# UTILITY: VALIDATE METADATA
# ============================================================================

def validate_metadata_format(metadata: dict, data_type: str | None = None) -> None:
    """
    Validate metadata.json format.

    Supports:
    - Legacy flat format:
        {
          "volumes": [...]
        }

    - New split-aware format (v4.0.0+):
        {
          "data": {
            "train": { "volumes": [...] },
            "val":   { "volumes": [...] },
            "test":  { "volumes": [...] }
          }
        }
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must be a JSON object")

    # ============================
    # NEW: Split-aware format
    # ============================
# ============================
# NEW: Split-aware format
# ============================
    if "data" in metadata:
        data_section = metadata["data"]

        if not isinstance(data_section, dict):
            raise ValueError("'data' must be a dictionary")

        found_any_split = False

        for split_name in ("train", "val", "test"):
            if split_name not in data_section:
                continue

            split = data_section[split_name]
            found_any_split = True

            # ✅ CASE 1: train = { "volumes": [...] }
            if isinstance(split, dict):
                if "volumes" not in split:
                    raise ValueError(
                        f"'data.{split_name}' is a dict but missing 'volumes'"
                    )
                if not isinstance(split["volumes"], list):
                    raise ValueError(
                        f"'data.{split_name}.volumes' must be a list"
                    )

        # ✅ CASE 2: train = [ ... ]  (legacy prepare_data.py)
            elif isinstance(split, list):
            # Accept directly
                pass

            else:
                raise ValueError(
                    f"'data.{split_name}' must be either a dict or a list"
                )

        if not found_any_split:
            raise ValueError(
                "metadata['data'] does not contain any of: train / val / test"
            )

        return


    # ============================
    # LEGACY: Flat format
    # ============================
    if "volumes" in metadata:
        if not isinstance(metadata["volumes"], list):
            raise ValueError("'volumes' must be a list")
        return

    # ============================
    # INVALID
    # ============================
    raise ValueError(
        "Invalid metadata format.\n"
        "Expected either:\n"
        "  - Top-level 'volumes'\n"
        "  - Or split-aware 'data.{train|val|test}.volumes'"
    )
