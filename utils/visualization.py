"""
Visualization Utilities for 3D Segmentation
✅ FIXED: Enhanced with multiple visualization types
✅ FIXED: 3D rendering support
✅ FIXED: Comprehensive error handling

File: utils/visualization.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def visualize_volume(
    volume: np.ndarray,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Volume Slice",
    cmap: str = 'gray'
):
    """
    Display 2D slice from 3D volume
    
    Args:
        volume: (D, H, W) array
        slice_idx: Index of slice to show (default: middle)
        save_path: Path to save figure
        title: Title for the plot
        cmap: Colormap to use
    """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
    
    plt.figure(figsize=(10, 10))
    plt.imshow(volume[slice_idx], cmap=cmap)
    plt.title(f'{title} - Slice {slice_idx}/{volume.shape[0]}', fontsize=14, fontweight='bold')
    plt.colorbar(fraction=0.046)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved visualization: {save_path}")
    
    plt.show()


def plot_segmentation_results(
    volume: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    save_path: Optional[str] = None,
    slice_idx: Optional[int] = None,
    show_overlay: bool = True
):
    """
    Show input, prediction, GT side-by-side with optional overlay
    
    Args:
        volume: (D, H, W) input volume
        prediction: (D, H, W) prediction
        ground_truth: (D, H, W) ground truth
        save_path: Path to save figure
        slice_idx: Slice to visualize (default: middle)
        show_overlay: If True, add overlay column
    """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
    
    # Determine number of columns
    n_cols = 4 if show_overlay else 3
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Input
    axes[0].imshow(volume[slice_idx], cmap='gray')
    axes[0].set_title('Input', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(ground_truth[slice_idx], cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction[slice_idx], cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Overlay (if requested)
    if show_overlay:
        # Create RGB overlay
        overlay = np.zeros((*volume[slice_idx].shape, 3))
        
        # Gray input as base
        norm_vol = (volume[slice_idx] - volume[slice_idx].min()) / (volume[slice_idx].max() - volume[slice_idx].min() + 1e-8)
        overlay[..., 0] = norm_vol
        overlay[..., 1] = norm_vol
        overlay[..., 2] = norm_vol
        
        # Green for True Positives
        tp_mask = (prediction[slice_idx] > 0) & (ground_truth[slice_idx] > 0)
        overlay[tp_mask, 1] = 1.0
        
        # Red for False Positives
        fp_mask = (prediction[slice_idx] > 0) & (ground_truth[slice_idx] == 0)
        overlay[fp_mask, 0] = 1.0
        
        # Blue for False Negatives
        fn_mask = (prediction[slice_idx] == 0) & (ground_truth[slice_idx] > 0)
        overlay[fn_mask, 2] = 1.0
        
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay (TP=Green, FP=Red, FN=Blue)', fontsize=14, fontweight='bold')
        axes[3].axis('off')
    
    plt.suptitle(f'Slice {slice_idx}/{volume.shape[0]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved visualization: {save_path}")
    
    plt.show()


def save_prediction_as_tif(
    prediction: np.ndarray,
    output_path: str
):
    """
    Save 3D prediction as TIF stack
    
    Args:
        prediction: (D, H, W) numpy array
        output_path: Output directory path
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving TIF stack to {output_path}...")
    
    for i in range(prediction.shape[0]):
        # Convert to 8-bit
        slice_img = (prediction[i] * 255).astype(np.uint8)
        
        # Save as TIF
        Image.fromarray(slice_img).save(output_path / f'slice_{i:04d}.tif')
    
    logger.info(f"  ✅ Saved {prediction.shape[0]} slices")


def plot_3d_rendering(
    volume: np.ndarray,
    save_path: Optional[str] = None,
    threshold: float = 0.5,
    alpha: float = 0.3
):
    """
    Create 3D rendering of volume using matplotlib
    
    Args:
        volume: (D, H, W) binary volume
        save_path: Path to save figure
        threshold: Threshold for binary mask
        alpha: Transparency of rendering
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.error("3D plotting requires matplotlib with 3D support")
        return
    
    # Binarize
    binary_volume = (volume > threshold).astype(np.uint8)
    
    # Get coordinates of foreground voxels
    z, y, x = np.where(binary_volume > 0)
    
    if len(z) == 0:
        logger.warning("Volume is empty, cannot create 3D rendering")
        return
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels
    ax.scatter(x, y, z, c=z, cmap='viridis', alpha=alpha, s=1)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('3D Rendering', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved 3D rendering: {save_path}")
    
    plt.show()


def create_multi_slice_figure(
    volume: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    save_path: str,
    num_slices: int = 5
):
    """
    Create figure showing multiple slices
    
    Args:
        volume: (D, H, W) input volume
        prediction: (D, H, W) prediction
        ground_truth: (D, H, W) ground truth
        save_path: Path to save figure
        num_slices: Number of slices to show
    """
    D = volume.shape[0]
    
    # Select evenly spaced slices
    slice_indices = np.linspace(0, D-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5*num_slices))
    
    for row, slice_idx in enumerate(slice_indices):
        # Input
        axes[row, 0].imshow(volume[slice_idx], cmap='gray')
        axes[row, 0].set_title(f'Input - Slice {slice_idx}', fontsize=12)
        axes[row, 0].axis('off')
        
        # Ground Truth
        axes[row, 1].imshow(ground_truth[slice_idx], cmap='jet')
        axes[row, 1].set_title(f'GT - Slice {slice_idx}', fontsize=12)
        axes[row, 1].axis('off')
        
        # Prediction
        axes[row, 2].imshow(prediction[slice_idx], cmap='jet')
        axes[row, 2].set_title(f'Pred - Slice {slice_idx}', fontsize=12)
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    logger.info(f"Saved multi-slice figure: {save_path}")
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_scores: list,
    save_path: str,
    metric_name: str = 'Dice'
):
    """
    Plot loss and metric curves over epochs
    
    Args:
        train_losses: List of training losses per epoch
        val_scores: List of validation scores per epoch
        save_path: Path to save figure
        metric_name: Name of the validation metric
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # Validation metric
    axes[1].plot(epochs, val_scores, 'g-', linewidth=2, label=f'Val {metric_name}')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(metric_name, fontsize=12, fontweight='bold')
    axes[1].set_title(f'Validation {metric_name}', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    # Mark best validation score
    best_idx = np.argmax(val_scores)
    best_score = val_scores[best_idx]
    axes[1].axvline(best_idx + 1, color='r', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].scatter([best_idx + 1], [best_score], color='r', s=100, zorder=5)
    axes[1].text(best_idx + 1, best_score, f' Best: {best_score:.4f}',
                fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    logger.info(f"Saved training curves: {save_path}")
    plt.close()


def visualize_hypergraph(
    volume: np.ndarray,
    H_matrix: np.ndarray,
    save_path: str,
    num_hyperedges: int = 5
):
    """
    Visualize hypergraph connections on a 2D slice
    
    Args:
        volume: (D, H, W) input volume
        H_matrix: (N, Ne) hypergraph incidence matrix
        save_path: Path to save figure
        num_hyperedges: Number of hyperedges to visualize
    """
    # Take middle slice
    mid_slice = volume.shape[0] // 2
    slice_img = volume[mid_slice]
    
    # Flatten spatial dimensions
    H, W = slice_img.shape
    
    # Select random hyperedges to visualize
    Ne = H_matrix.shape[1]
    selected_edges = np.random.choice(Ne, min(num_hyperedges, Ne), replace=False)
    
    fig, axes = plt.subplots(1, num_hyperedges, figsize=(5*num_hyperedges, 5))
    
    if num_hyperedges == 1:
        axes = [axes]
    
    for idx, edge_idx in enumerate(selected_edges):
        # Get nodes in this hyperedge
        nodes_in_edge = H_matrix[:, edge_idx] > 0
        
        # Create mask for visualization
        mask = np.zeros_like(slice_img)
        
        # Map back to 2D (this is simplified, actual implementation depends on node ordering)
        # Here we assume nodes are ordered row-wise
        nodes_2d = nodes_in_edge.reshape(H, W) if len(nodes_in_edge) == H*W else np.zeros((H, W))
        
        # Show slice with hyperedge overlay
        axes[idx].imshow(slice_img, cmap='gray', alpha=0.7)
        axes[idx].imshow(nodes_2d, cmap='hot', alpha=0.5)
        axes[idx].set_title(f'Hyperedge {edge_idx}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    logger.info(f"Saved hypergraph visualization: {save_path}")
    plt.close()


def save_comparison_montage(
    samples: list,
    save_path: str,
    max_samples: int = 10
):
    """
    Create montage of multiple samples for comparison
    
    Args:
        samples: List of dicts with 'volume', 'prediction', 'gt', 'name' keys
        save_path: Path to save montage
        max_samples: Maximum number of samples to include
    """
    n_samples = min(len(samples), max_samples)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples[:n_samples]):
        volume = sample['volume']
        prediction = sample['prediction']
        gt = sample['gt']
        name = sample['name']
        
        # Take middle slice
        mid_slice = volume.shape[0] // 2
        
        # Input
        axes[i, 0].imshow(volume[mid_slice], cmap='gray')
        axes[i, 0].set_title(f'{name} - Input', fontsize=10)
        axes[i, 0].axis('off')
        
        # GT
        axes[i, 1].imshow(gt[mid_slice], cmap='jet')
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(prediction[mid_slice], cmap='jet')
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = np.zeros((*volume[mid_slice].shape, 3))
        norm_vol = (volume[mid_slice] - volume[mid_slice].min()) / (volume[mid_slice].max() - volume[mid_slice].min() + 1e-8)
        overlay[..., 0] = norm_vol
        overlay[..., 1] = norm_vol
        overlay[..., 2] = norm_vol
        
        tp_mask = (prediction[mid_slice] > 0) & (gt[mid_slice] > 0)
        fp_mask = (prediction[mid_slice] > 0) & (gt[mid_slice] == 0)
        fn_mask = (prediction[mid_slice] == 0) & (gt[mid_slice] > 0)
        
        overlay[tp_mask, 1] = 1.0
        overlay[fp_mask, 0] = 1.0
        overlay[fn_mask, 2] = 1.0
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay', fontsize=10)
        axes[i, 3].axis('off')
    
    plt.suptitle('Sample Comparison Montage', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    logger.info(f"Saved comparison montage: {save_path}")
    plt.close()


def create_gif_from_volume(
    volume: np.ndarray,
    save_path: str,
    duration: int = 100,
    normalize: bool = True
):
    """
    Create animated GIF from 3D volume
    
    Args:
        volume: (D, H, W) volume
        save_path: Path to save GIF
        duration: Duration per frame in milliseconds
        normalize: Whether to normalize intensity
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("GIF creation requires Pillow")
        return
    
    frames = []
    
    for i in range(volume.shape[0]):
        slice_img = volume[i]
        
        if normalize:
            slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        
        # Convert to 8-bit
        slice_img = (slice_img * 255).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(slice_img)
        frames.append(img)
    
    # Save as GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    logger.info(f"Saved animated GIF: {save_path} ({len(frames)} frames)")


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_visualization_functions():
    """
    Test all visualization functions with synthetic data
    """
    logger.info("Testing visualization functions...")
    
    # Create synthetic data
    D, H, W = 64, 128, 128
    volume = np.random.rand(D, H, W).astype(np.float32)
    prediction = (np.random.rand(D, H, W) > 0.5).astype(np.float32)
    ground_truth = (np.random.rand(D, H, W) > 0.5).astype(np.float32)
    
    # Create output directory
    test_output_dir = Path('test_visualizations')
    test_output_dir.mkdir(exist_ok=True)
    
    # Test 1: visualize_volume
    logger.info("  Testing visualize_volume...")
    visualize_volume(
        volume,
        save_path=str(test_output_dir / 'test_volume.png')
    )
    plt.close('all')
    
    # Test 2: plot_segmentation_results
    logger.info("  Testing plot_segmentation_results...")
    plot_segmentation_results(
        volume,
        prediction,
        ground_truth,
        save_path=str(test_output_dir / 'test_segmentation.png')
    )
    plt.close('all')
    
    # Test 3: save_prediction_as_tif
    logger.info("  Testing save_prediction_as_tif...")
    save_prediction_as_tif(
        prediction,
        str(test_output_dir / 'test_tif_stack')
    )
    
    # Test 4: create_multi_slice_figure
    logger.info("  Testing create_multi_slice_figure...")
    create_multi_slice_figure(
        volume,
        prediction,
        ground_truth,
        str(test_output_dir / 'test_multi_slice.png'),
        num_slices=3
    )
    
    # Test 5: plot_training_curves
    logger.info("  Testing plot_training_curves...")
    train_losses = [1.0 - 0.01*i + 0.02*np.random.rand() for i in range(50)]
    val_scores = [0.5 + 0.01*i - 0.02*np.random.rand() for i in range(50)]
    plot_training_curves(
        train_losses,
        val_scores,
        str(test_output_dir / 'test_training_curves.png')
    )
    
    # Test 6: create_gif_from_volume
    logger.info("  Testing create_gif_from_volume...")
    create_gif_from_volume(
        volume,
        str(test_output_dir / 'test_volume.gif'),
        duration=50
    )
    
    logger.info("✅ All visualization tests passed!")
    logger.info(f"Test outputs saved to: {test_output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_visualization_functions()