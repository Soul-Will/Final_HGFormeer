"""
Self-Supervised Pretraining Script
✅ PERMANENT FIX: Robust foreground sampling integration
✅ FIXED: Dynamic config-driven dataset initialization
✅ FIXED: Fail-fast validation (catches errors early)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
import sys

sys.path.append('.')

from models import HGFormer3D
from models.losses import ssl_total_loss
from utils.data_loader import VolumeDataset3D, validate_metadata_format
from utils.augmentations import get_ssl_transforms
from utils.scheduler_builder import build_scheduler
from utils.debug_tools import (
    verify_rotation_task,
    visualize_contrastive_embeddings,
    GradientMonitor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from utils.training_logger import UnifiedTrainingLogger



def parse_args():
    parser = argparse.ArgumentParser(
        description='SSL Pretraining for HGFormer3D',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with processed HDF5 volumes and metadata.json')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save checkpoints')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    return parser.parse_args()

def validate_config(config: dict) -> None:
    """
    ✅ PERMANENT FIX: Validate config has all required fields
    
    Fail-fast: Better to crash at startup than during training!
    """
    required_sections = ['model', 'training', 'ssl_losses', 'data', 'augmentation', 'paths']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required section '{section}' in config file.\n"
                f"Please ensure your config YAML has all required sections."
            )
    
    # Validate data config has patch_size
    if 'patch_size' not in config['data']:
        raise ValueError(
            "Missing 'patch_size' in data config.\n"
            "Example: patch_size: [64, 128, 128]"
        )
    
    # Validate foreground sampling config (with defaults)
    if 'foreground_sampling' not in config['data']:
        logger.warning(
            "⚠️  'foreground_sampling' not specified in config. Defaulting to True.\n"
            "To disable, add: data.foreground_sampling: false"
        )
        config['data']['foreground_sampling'] = True
    
    if 'foreground_fraction' not in config['data']:
        logger.warning(
            "⚠️  'foreground_fraction' not specified. Defaulting to 0.8 (80% foreground).\n"
            "To change, add: data.foreground_fraction: 0.8"
        )
        config['data']['foreground_fraction'] = 0.8
    
    logger.info("✅ Config validation passed")


def create_dataset(config: dict, data_dir: Path) -> VolumeDataset3D:
    """
    ✅ PERMANENT FIX: Create dataset with ALL config parameters
    
    Key Changes:
    - Passes foreground sampling config from YAML
    - Validates foreground coords exist (fail-fast)
    - Logs sampling strategy for transparency
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING DATASET")
    logger.info("="*80)
    
    # Extract config
    data_config = config['data']
    aug_config = config.get('augmentation', {})
    
    # Get transforms
    use_augmentation = aug_config.get('use_augmentation', True)
    
    if use_augmentation:
        train_transform = get_ssl_transforms(aug_config)
        logger.info("  Augmentation: ENABLED")
    else:
        train_transform = None
        logger.info("  Augmentation: DISABLED")
    
    # ✅ CRITICAL FIX: Extract foreground sampling config from YAML
    foreground_sampling = data_config.get('foreground_sampling', True)
    foreground_fraction = data_config.get('foreground_fraction', 0.8)
    
    logger.info(f"  Foreground sampling: {foreground_sampling}")
    if foreground_sampling:
        logger.info(f"  Foreground fraction: {foreground_fraction:.1%}")
        logger.info(f"  Random fraction: {1 - foreground_fraction:.1%}")
    
    # ✅ PERMANENT FIX: Create dataset with ALL parameters from config
    try:
        dataset = VolumeDataset3D(
            data_dir=str(data_dir),
            patch_size=tuple(data_config['patch_size']),
            num_patches_per_epoch=data_config.get('num_patches_per_epoch', 1000),
            transform=train_transform,
            config=config,
            preload=False,  # NEVER preload HDF5 files!
            foreground_sampling=foreground_sampling,  # ✅ NEW
            foreground_fraction=foreground_fraction   # ✅ NEW
        )
    except Exception as e:
        logger.error(f"\n❌ FATAL: Dataset creation failed: {e}")
        logger.error(f"\nThis usually means:")
        logger.error(f"  1. Metadata is missing or corrupt")
        logger.error(f"  2. HDF5 files are missing")
        logger.error(f"  3. Foreground coords are missing (re-run prepare_data.py)")
        logger.error(f"\nTo fix:")
        logger.error(f"  python scripts/prepare_data.py \\")
        logger.error(f"      --input_dir data/raw/train_unlabeled \\")
        logger.error(f"      --output_dir {data_dir} \\")
        logger.error(f"      --data_type unlabeled")
        raise
    
    logger.info(f"\n✅ Dataset created successfully")
    logger.info(f"  Total virtual patches per epoch: {len(dataset)}")
    logger.info("="*80 + "\n")
    
    return dataset


def test_dataset(dataset: VolumeDataset3D, device: torch.device) -> None:
    """
    ✅ PERMANENT FIX: Test dataset BEFORE training starts
    
    Validates:
    - Data loader works
    - Patches are non-empty
    - Transforms work
    - No NaN/Inf values
    
    Fail-fast: Better to crash here than 30 minutes into training!
    """
    logger.info("\n" + "="*80)
    logger.info("TESTING DATASET (Fail-Fast Validation)")
    logger.info("="*80)
    
    try:
        # Test 5 random samples
        for i in range(min(5, len(dataset))):
            logger.info(f"  Testing sample {i+1}/5...")
            
            # Load sample
            patch, marker_label = dataset[i]
            
            # Validate shape
            assert patch.ndim == 4, f"Patch should be (1,D,H,W), got {patch.shape}"
            assert patch.shape[0] == 1, f"Batch dim should be 1, got {patch.shape[0]}"
            
            # Validate non-empty
            patch_std = patch.std().item()
            if patch_std < 1e-6:
                logger.warning(
                    f"⚠️  Sample {i} has very low std: {patch_std:.8f}. "
                    f"This might be a background patch."
                )
            
            # Validate no NaN/Inf
            assert not torch.isnan(patch).any(), f"Sample {i} contains NaN!"
            assert not torch.isinf(patch).any(), f"Sample {i} contains Inf!"
            
            # Validate marker label
            assert marker_label.dim() == 0, f"Label should be scalar, got {marker_label.shape}"
            assert 0 <= marker_label < 10, f"Label {marker_label} out of range [0, 10)"

            # Test device transfer
            patch = patch.to(device)
            marker_label = marker_label.to(device)
            
            logger.info(
                f"    ✅ Sample {i}: shape={tuple(patch.shape)}, "
                f"std={patch_std:.4f}, label={marker_label.item()}"
            )
        
        logger.info("\n✅ Dataset validation PASSED!")
        logger.info("="*80 + "\n")
    
    except Exception as e:
        logger.error(f"\n❌ FATAL: Dataset validation FAILED: {e}")
        logger.error(f"\nThis means your dataset is corrupted or misconfigured.")
        logger.error(f"Please re-run prepare_data.py to regenerate it.")
        raise


def main():
    args = parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger_csv = UnifiedTrainingLogger(
        Path(args.output_dir) / "training_log.csv"
    )

    # ✅ FIX 1: Load and validate config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    validate_config(config)  # ✅ NEW: Fail-fast if config is invalid
    
    # Save config to output dir (for reproducibility)
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to: {config_save_path}")
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(args.gpu)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.2f} GB")
    
    # ✅ FIX 2: Validate metadata
    data_dir = Path(args.data_dir)
    logger.info(f"\nValidating metadata in {data_dir}...")
    
    try:
        import json

        metadata_path = data_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found at {metadata_path}\n"
                f"Did you run prepare_data.py?"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"❌ Invalid JSON in metadata.json")
        raise

    logger.info(f"Metadata type: {type(metadata)}")
    try:
        validate_metadata_format(metadata, data_type='unlabeled')
        logger.info("✅ Metadata validation passed")
    except Exception as e:
        logger.error(f"❌ Metadata validation failed: {e}")
        logger.error(f"\n⚠️  Did you run prepare_data.py first?")
        logger.error(f"    python scripts/prepare_data.py \\")
        logger.error(f"        --input_dir data/raw/train_unlabeled \\")
        logger.error(f"        --output_dir {data_dir} \\")
        logger.error(f"        --data_type unlabeled")
        raise
    
    # ✅ FIX 3: Create model
    logger.info("\nCreating model...")
    model = HGFormer3D(**config['model']).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model created with {num_params:.2f}M parameters")
    
    # ✅ FIX 4: Create dataset with ALL config parameters
    train_dataset = create_dataset(config, data_dir)
    
    # ✅ FIX 5: Test dataset BEFORE training
    test_dataset(train_dataset, device)
    
    # Create dataloader
    logger.info("Creating dataloader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True) and device.type == 'cuda',
        drop_last=True
    )
    
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    
    # Optimizer
    logger.info("\nSetting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Weight decay: {config['training']['weight_decay']}")
    
    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=config['training']['epochs'],
    #     eta_min=config['training'].get('min_lr', 1e-6)
    # )
    scheduler = build_scheduler(optimizer, config)

    logger.info(f"  Scheduler: CosineAnnealingLR")
    logger.info(f"  T_max: {config['training']['epochs']}")
    logger.info(f"  Min LR: {config['training'].get('min_lr', 1e-6)}")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_loss = float('inf')
    
    grad_monitor = GradientMonitor(model)
    debug_dir = output_dir / 'debug'
    debug_dir.mkdir(exist_ok=True)
    
    logger.info("\n✅ Debug tools initialized:")
    logger.info(f"  Gradient monitor: ACTIVE")
    logger.info(f"  Debug output: {debug_dir}")

    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"  Resuming from epoch {start_epoch}")
        logger.info(f"  Best loss so far: {best_loss:.4f}")
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("STARTING SSL PRETRAINING")
    logger.info("="*80)
    logger.info(f"Total epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Patches per epoch: {len(train_dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    logger.info(f"Checkpoint frequency: every {config['training'].get('save_freq', 10)} epochs")
    logger.info("="*80 + "\n")
    
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            model.train()
            epoch_losses = {
                'total': [],
                'inpaint': [],
                'rotation': [],
                'contrastive': [],
                'label': []
            }
            
            pbar = tqdm(
                train_loader, 
                desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}',
                ncols=100
            )
            
            for batch_idx, (volumes, marker_types) in enumerate(pbar):
                volumes = volumes.to(device)
                marker_types = marker_types.to(device)
                
                # ✅ VALIDATION: Check for NaN/Inf in batch
                if torch.isnan(volumes).any() or torch.isinf(volumes).any():
                    logger.error(
                        f"❌ FATAL: NaN/Inf detected in batch {batch_idx}!\n"
                        f"Volume stats: min={volumes.min():.4f}, max={volumes.max():.4f}, "
                        f"mean={volumes.mean():.4f}, std={volumes.std():.4f}"
                    )
                    raise ValueError("NaN/Inf in input data!")
                
                # Forward pass
                loss, loss_dict = ssl_total_loss(
                    model, volumes, marker_types,
                    **config['ssl_losses']
                )
                
                # ✅ VALIDATION: Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(
                        f"❌ FATAL: NaN/Inf loss at epoch {epoch}, batch {batch_idx}!\n"
                        f"Loss dict: {loss_dict}"
                    )
                    raise ValueError("NaN/Inf loss detected!")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training'].get('gradient_clip', 1.0)
                )

                grad_stats = grad_monitor.log_gradients()
                
                optimizer.step()
                
                # Log losses
                for key in epoch_losses.keys():
                    if key in loss_dict:
                        epoch_losses[key].append(loss_dict[key])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                    'grad': f"{grad_stats['avg_grad']:.2e}"
                })
            
            # Step scheduler
            scheduler.step()
            
            # Calculate average losses
            avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
            
            # Print epoch summary
            logger.info(f"\n📊 Epoch {epoch+1} Summary:")
            logger.info(f"  Total Loss: {avg_losses['total']:.4f}")
            logger.info(f"  Inpaint: {avg_losses['inpaint']:.4f}")
            logger.info(f"  Rotation: {avg_losses['rotation']:.4f}")
            logger.info(f"  Contrastive: {avg_losses['contrastive']:.4f}")
            logger.info(f"  Label: {avg_losses['label']:.4f}")
            logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # ✅ Periodic debug checks (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                logger.info("\n🔍 Running debug diagnostics...")
                
                # 1. Verify rotation task
                try:
                    rot_acc = verify_rotation_task(
                        model, 
                        train_dataset, 
                        device,
                        save_path=debug_dir / f'rotation_epoch_{epoch+1:04d}.png'
                    )
                    logger.info(f"  ✅ Rotation accuracy: {rot_acc:.2%}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Rotation verification failed: {e}")
                
                # 2. Visualize embeddings
                try:
                    visualize_contrastive_embeddings(
                        model,
                        train_dataset,
                        device,
                        save_path=debug_dir / f'umap_epoch_{epoch+1:04d}.png'
                    )
                    logger.info(f"  ✅ UMAP visualization saved")
                except Exception as e:
                    logger.warning(f"  ⚠️ UMAP visualization failed: {e}")
                
                # 3. Plot gradient history
                try:
                    grad_monitor.plot_gradient_history(
                        save_path=debug_dir / f'gradients_epoch_{epoch+1:04d}.png'
                    )
                    logger.info(f"  ✅ Gradient history plotted")
                except Exception as e:
                    logger.warning(f"  ⚠️ Gradient plotting failed: {e}")
                
                logger.info("")



            logger_csv.log(
                phase="ssl",
                epoch=epoch + 1,
                step=-1,

                ssl_total_loss=avg_losses["total"],
                ssl_inpaint_loss=avg_losses["inpaint"],
                ssl_rotation_loss=avg_losses["rotation"],
                ssl_contrastive_loss=avg_losses["contrastive"],
                ssl_marker_loss=avg_losses["label"],

                learning_rate=optimizer.param_groups[0]["lr"]
            )

            # Save checkpoint every N epochs
            if (epoch + 1) % config['training'].get('save_freq', 10) == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1:04d}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_losses['total'],
                    'config': config
                }, checkpoint_path)
                logger.info(f"  💾 Checkpoint saved: {checkpoint_path.name}")
            
            # Save best model
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                best_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config
                }, best_path)
                logger.info(f"  ✅ NEW BEST MODEL! Loss: {best_loss:.4f}")
            
            logger.info("")
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
        logger.info("Saving checkpoint before exit...")
        
        interrupt_path = output_dir / 'interrupted_checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'config': config
        }, interrupt_path)
        logger.info(f"Checkpoint saved: {interrupt_path}")
    
    except Exception as e:
        logger.error(f"\n\n❌ Training failed with error: {e}")
        logger.exception("Full traceback:")
        raise
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("SSL PRETRAINING COMPLETED!")
    logger.info("="*80)
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Total epochs: {epoch + 1}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()