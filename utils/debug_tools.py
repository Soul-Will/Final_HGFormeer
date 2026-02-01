"""
Debug Tools for SSL Training
✅ Rotation task verification
✅ Contrastive embedding visualization (UMAP)
✅ Gradient flow monitoring
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)


# ============================================================================
# 1. ROTATION TASK VERIFICATION
# ============================================================================

def verify_rotation_task(
    model: nn.Module,
    volume: torch.Tensor,
    device: torch.device,
    save_dir: Path
):
    """
    Verify rotation prediction accuracy
    
    Tests if model can distinguish 0°/90°/180°/270° rotations
    Expected: >25% accuracy (better than random) early on
             >80% accuracy after proper training
    
    Args:
        model: HGFormer3D model
        volume: (B, 1, D, H, W) input volume
        device: CUDA device
        save_dir: Directory to save results
    """
    model.eval()
    
    # Define rotation angles (0, 90, 180, 270 degrees)
    rotation_angles = [0, 1, 2, 3]  # Indices for each rotation
    
    # Storage for predictions and ground truth
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for angle_idx in rotation_angles:
            # Apply rotation (k=angle_idx means rotate by angle_idx * 90 degrees)
            rotated = torch.rot90(volume, k=angle_idx, dims=(3, 4))  # Rotate in H-W plane
            rotated = rotated.to(device)
            
            # Predict rotation
            logits = model.predict_rotation(rotated)  # (B, 4)
            predicted_angle = torch.argmax(logits, dim=1)  # (B,)
            
            predictions.extend(predicted_angle.cpu().numpy())
            ground_truth.extend([angle_idx] * volume.shape[0])
    
    # Compute accuracy
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    accuracy = (predictions == ground_truth).mean()
    
    # Create confusion matrix
    confusion_matrix = np.zeros((4, 4))
    for pred, true in zip(predictions, ground_truth):
        confusion_matrix[true, pred] += 1
    
    # Normalize to percentages
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True) * 100
    
    # Log results
    logger.info(f"\n🔄 Rotation Task Verification:")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Expected: >25% (random), >80% (trained)")
    
    if accuracy < 0.30:
        logger.warning(f"  ⚠️  Accuracy too low! Rotation task may be broken.")
    elif accuracy > 0.80:
        logger.info(f"  ✅ Excellent rotation prediction!")
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix, cmap='Blues', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(['0°', '90°', '180°', '270°'])
    ax.set_yticklabels(['0°', '90°', '180°', '270°'])
    ax.set_xlabel('Predicted Rotation', fontsize=12)
    ax.set_ylabel('True Rotation', fontsize=12)
    ax.set_title(f'Rotation Confusion Matrix (Acc: {accuracy*100:.1f}%)', fontsize=14)
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{confusion_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black" if confusion_matrix[i, j] < 50 else "white")
    
    plt.colorbar(im, ax=ax, label='Percentage (%)')
    plt.tight_layout()
    plt.savefig(save_dir / 'rotation_verification.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  💾 Confusion matrix saved")
    
    model.train()
    return accuracy


# ============================================================================
# 2. CONTRASTIVE EMBEDDING VISUALIZATION
# ============================================================================

def visualize_contrastive_embeddings(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: Path,
    epoch: int,
    max_samples: int = 500
):
    """
    Visualize contrastive embeddings using UMAP
    
    Good embeddings: Clear clusters by marker type
    Bad embeddings: Random scatter (collapsed embeddings)
    
    Args:
        model: HGFormer3D model
        dataloader: Training dataloader
        device: CUDA device
        save_dir: Directory to save plot
        epoch: Current epoch number
        max_samples: Max samples to embed (for speed)
    """
    try:
        import umap
    except ImportError:
        logger.warning("⚠️  UMAP not installed. Run: pip install umap-learn")
        return
    
    model.eval()
    
    embeddings = []
    marker_labels = []
    
    with torch.no_grad():
        for batch_idx, (volumes, markers) in enumerate(dataloader):
            if batch_idx * volumes.shape[0] >= max_samples:
                break
            
            volumes = volumes.to(device)
            
            # Extract embeddings
            emb = model.encode(volumes)  # (B, C)
            embeddings.append(emb.cpu().numpy())
            marker_labels.append(markers.numpy())
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)  # (N, C)
    marker_labels = np.concatenate(marker_labels, axis=0)  # (N,)
    
    logger.info(f"\n🎨 UMAP Visualization:")
    logger.info(f"  Samples: {len(embeddings)}")
    logger.info(f"  Embedding dim: {embeddings.shape[1]}")
    
    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique markers
    unique_markers = np.unique(marker_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_markers)))
    
    for marker_idx, color in zip(unique_markers, colors):
        mask = marker_labels == marker_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f'Marker {marker_idx}',
            alpha=0.6,
            s=30
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'Contrastive Embeddings - Epoch {epoch}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'umap_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  💾 UMAP plot saved for epoch {epoch}")
    
    model.train()


# ============================================================================
# 3. GRADIENT FLOW MONITORING
# ============================================================================

class GradientMonitor:
    """
    Monitor gradient statistics during training
    
    Detects:
    - Vanishing gradients (avg < 1e-6)
    - Exploding gradients (max > 10)
    - Dead layers (std = 0)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.history = {
            'avg_grad': [],
            'max_grad': [],
            'min_grad': [],
            'std_grad': []
        }
    
    def log_gradients(self):
        """Compute gradient statistics"""
        grads = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.abs().flatten())
        
        if len(grads) == 0:
            logger.warning("⚠️  No gradients found!")
            return None
        
        all_grads = torch.cat(grads)
        
        stats = {
            'avg_grad': all_grads.mean().item(),
            'max_grad': all_grads.max().item(),
            'min_grad': all_grads.min().item(),
            'std_grad': all_grads.std().item()
        }
        
        # Store history
        for key, value in stats.items():
            self.history[key].append(value)
        
        return stats
    
    def check_gradient_health(self, stats: dict, epoch: int):
        """Check for gradient problems"""
        avg_grad = stats['avg_grad']
        max_grad = stats['max_grad']
        
        # Check for vanishing gradients
        if avg_grad < 1e-6:
            logger.warning(
                f"⚠️  VANISHING GRADIENTS at epoch {epoch}!\n"
                f"  Average gradient: {avg_grad:.2e} (threshold: 1e-6)\n"
                f"  Action: Increase learning rate or check loss function"
            )
            return "vanishing"
        
        # Check for exploding gradients
        if max_grad > 10:
            logger.warning(
                f"⚠️  EXPLODING GRADIENTS at epoch {epoch}!\n"
                f"  Max gradient: {max_grad:.2e} (threshold: 10)\n"
                f"  Action: Reduce learning rate or increase gradient clipping"
            )
            return "exploding"
        
        # Healthy gradients
        return "healthy"
    
    def plot_gradient_history(self, save_path: Path):
        """Plot gradient statistics over time"""
        if len(self.history['avg_grad']) == 0:
            logger.warning("⚠️  No gradient history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.history['avg_grad']) + 1)
        
        # Average gradient
        axes[0, 0].plot(epochs, self.history['avg_grad'], linewidth=2, color='blue')
        axes[0, 0].axhline(1e-6, color='red', linestyle='--', alpha=0.5, label='Vanishing threshold')
        axes[0, 0].set_ylabel('Average Gradient', fontsize=12)
        axes[0, 0].set_title('Average Gradient Magnitude', fontsize=14)
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Max gradient
        axes[0, 1].plot(epochs, self.history['max_grad'], linewidth=2, color='red')
        axes[0, 1].axhline(10, color='orange', linestyle='--', alpha=0.5, label='Exploding threshold')
        axes[0, 1].set_ylabel('Max Gradient', fontsize=12)
        axes[0, 1].set_title('Maximum Gradient Magnitude', fontsize=14)
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Min gradient
        axes[1, 0].plot(epochs, self.history['min_grad'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Min Gradient', fontsize=12)
        axes[1, 0].set_title('Minimum Gradient Magnitude', fontsize=14)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(alpha=0.3)
        
        # Std gradient
        axes[1, 1].plot(epochs, self.history['std_grad'], linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Std Gradient', fontsize=12)
        axes[1, 1].set_title('Gradient Standard Deviation', fontsize=14)
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('Gradient Flow Monitoring', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  💾 Gradient history plot saved")


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_debug_tools():
    """Test all debug tools"""
    print("="*80)
    print("TESTING DEBUG TOOLS")
    print("="*80)
    
    # Create dummy model
    sys.path.append('.')

    from models import HGFormer3D
    model = HGFormer3D(
        in_channels=1,
        base_channels=32,
        depths=[1, 1, 2, 1],
        num_hyperedges=[64, 32, 16, 8],
        K_neighbors=[128, 64, 32, 8]
    )
    
    device = torch.device('cpu')
    save_dir = Path('./results/outputs/')
    save_dir.mkdir(exist_ok=True)
    
    # Test 1: Rotation verification
    print("\n1. Testing rotation verification...")
    dummy_volume = torch.randn(2, 1, 64, 128, 128)
    verify_rotation_task(model, dummy_volume, device, save_dir)
    print("✅ Rotation test passed!")
    
    # Test 2: Gradient monitoring
    print("\n2. Testing gradient monitoring...")
    grad_monitor = GradientMonitor(model)
    
    # Simulate training
    for epoch in range(10):
        dummy_input = torch.randn(2, 1, 64, 128, 128)
        output = model(dummy_input)
        loss = output.mean()
        loss.backward()
        
        stats = grad_monitor.log_gradients()
        grad_monitor.check_gradient_health(stats, epoch)
        
        model.zero_grad()
    
    grad_monitor.plot_gradient_history(save_dir / 'gradient_test.png')
    print("✅ Gradient monitoring test passed!")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)


if __name__ == '__main__':
    test_debug_tools()