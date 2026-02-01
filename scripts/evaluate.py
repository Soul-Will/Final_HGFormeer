"""
Evaluation Script for 3D Segmentation
✅ FIXED: Supports both .npy and .h5 ground truth files
✅ FIXED: No redundant metrics (Dice = F1)
✅ FIXED: Comprehensive metrics from utils.metrics
✅ FIXED: Beautiful visualizations
✅ FIXED: Per-marker analysis

File: scripts/evaluate.py
"""

import numpy as np
from pathlib import Path
import argparse
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List
import h5py  # ✅ NEW: For loading .h5 ground truth

import sys
sys.path.append('.')

from utils.metrics import compute_all_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate segmentation predictions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory containing predictions (.npy files)')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth (.npy or .h5)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save evaluation results')
    parser.add_argument('--pred_format', type=str, default='npy',
                       choices=['npy', 'tif'],
                       help='Format of prediction files')
    parser.add_argument('--gt_format', type=str, default='auto',
                       choices=['auto', 'npy', 'h5'],
                       help='Format of ground truth files (auto-detect by default)')
    parser.add_argument('--save_per_sample', action='store_true', default=True,
                       help='Save per-sample metrics CSV')
    parser.add_argument('--spacing', type=float, nargs=3, default=[2.0, 1.0, 1.0],
                       help='Physical spacing (z y x) in microns')
    return parser.parse_args()


def load_prediction(pred_path: Path, format: str = 'npy') -> np.ndarray:
    """Load prediction from file"""
    if format == 'npy':
        return np.load(pred_path)
    
    elif format == 'tif':
        # Load TIF stack from directory
        if not pred_path.is_dir():
            raise ValueError(f"For TIF format, pred_path must be a directory: {pred_path}")
        
        from PIL import Image
        tif_files = sorted(pred_path.glob('*.tif'))
        
        if len(tif_files) == 0:
            raise FileNotFoundError(f"No TIF files found in {pred_path}")
        
        slices = [np.array(Image.open(f)) for f in tif_files]
        return np.stack(slices, axis=0)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_ground_truth(gt_path: Path, format: str = 'auto') -> np.ndarray:
    """
    ✅ FIXED: Load ground truth mask from .npy or .h5
    
    Args:
        gt_path: Path to ground truth file
        format: 'auto', 'npy', or 'h5'
    
    Returns:
        mask: (D, H, W) numpy array
    """
    # Auto-detect format from extension
    if format == 'auto':
        if gt_path.suffix == '.npy':
            format = 'npy'
        elif gt_path.suffix == '.h5':
            format = 'h5'
        elif gt_path.suffix in ['.tif', '.tiff']:
            format = 'tif'
        else:
            raise ValueError(f"Unknown ground truth format: {gt_path.suffix}")
    
    # Load based on format
    if format == 'npy':
        return np.load(gt_path)
    
    elif format == 'h5':
        # ✅ NEW: Load from HDF5
        with h5py.File(gt_path, 'r') as f:
            # Check common dataset names
            if 'mask' in f:
                mask = f['mask'][:]
            elif 'label' in f:
                mask = f['label'][:]
            elif 'segmentation' in f:
                mask = f['segmentation'][:]
            else:
                # Use first dataset
                dataset_name = list(f.keys())[0]
                logger.debug(f"Using dataset '{dataset_name}' from {gt_path.name}")
                mask = f[dataset_name][:]
        
        return mask
    
    elif format == 'tif':
        from PIL import Image
        return np.array(Image.open(gt_path))
    
    else:
        raise ValueError(f"Unknown ground truth format: {format}")


def find_matching_pairs(
    pred_dir: Path,
    gt_dir: Path,
    pred_format: str,
    gt_format: str = 'auto'
) -> List[Dict]:
    """
    ✅ FIXED: Find matching prediction-ground truth pairs (handles .npy and .h5)
    
    Returns:
        List of dicts with 'pred', 'gt', 'name' keys
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    pairs = []
    
    if pred_format == 'npy':
        # Look for .npy prediction files
        pred_files = sorted(pred_dir.glob('*_pred.npy'))
        
        for pred_file in pred_files:
            # Extract sample name
            sample_name = pred_file.stem.replace('_pred', '')
            
            # ✅ FIXED: Try multiple naming conventions AND file extensions
            possible_gt_names = [
                # .h5 variants
                f'{sample_name}_mask.h5',
                f'{sample_name}_label.h5',
                f'{sample_name}.h5',
                # .npy variants (backward compatibility)
                f'{sample_name}_mask.npy',
                f'{sample_name}.npy',
                f'{sample_name}_gt.npy'
            ]
            
            gt_file = None
            for gt_name in possible_gt_names:
                candidate = gt_dir / gt_name
                if candidate.exists():
                    gt_file = candidate
                    break
            
            if gt_file is None:
                logger.warning(f"No ground truth found for {sample_name}")
                logger.debug(f"  Tried: {possible_gt_names}")
                continue
            
            pairs.append({
                'pred': pred_file,
                'gt': gt_file,
                'name': sample_name
            })
    
    elif pred_format == 'tif':
        # Look for TIF directories
        pred_dirs = [d for d in pred_dir.iterdir() if d.is_dir()]
        
        for pred_subdir in pred_dirs:
            sample_name = pred_subdir.name
            
            # ✅ FIXED: Try both .h5 and .npy
            possible_gt_names = [
                f'{sample_name}_mask.h5',
                f'{sample_name}.h5',
                f'{sample_name}_mask.npy',
                f'{sample_name}.npy'
            ]
            
            gt_file = None
            for gt_name in possible_gt_names:
                candidate = gt_dir / gt_name
                if candidate.exists():
                    gt_file = candidate
                    break
            
            if gt_file is None:
                logger.warning(f"No ground truth found for {sample_name}")
                continue
            
            pairs.append({
                'pred': pred_subdir,
                'gt': gt_file,
                'name': sample_name
            })
    
    return pairs


def evaluate_sample(
    pred: np.ndarray,
    gt: np.ndarray,
    sample_name: str,
    spacing: tuple
) -> Dict:
    """
    ✅ FIXED: Uses comprehensive metrics from utils.metrics (no redundancy)
    
    Args:
        pred: (D, H, W) prediction
        gt: (D, H, W) ground truth
        sample_name: Sample identifier
        spacing: (z, y, x) physical spacing
        
    Returns:
        Dict with all metrics
    """
    # Use the comprehensive metrics function
    metrics = compute_all_metrics(pred, gt, spacing=spacing)
    
    # Add sample name
    metrics['sample_name'] = sample_name
    
    return metrics


# [... REST OF THE FILE REMAINS UNCHANGED ...]
# (The plotting and main functions stay exactly the same)


def create_evaluation_plots(results_df: pd.DataFrame, output_dir: Path):
    """
    Create comprehensive visualization plots
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    sns.set_palette('husl')
    
    # ========================================
    # Plot 1: Distribution of main metrics
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Dice distribution
    axes[0, 0].hist(results_df['dice'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(results_df['dice'].mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {results_df["dice"].mean():.3f}')
    axes[0, 0].set_xlabel('Dice Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Dice Scores', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # IoU distribution
    axes[0, 1].hist(results_df['iou'], bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].axvline(results_df['iou'].mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {results_df["iou"].mean():.3f}')
    axes[0, 1].set_xlabel('IoU Score', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Distribution of IoU Scores', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Centerline Dice distribution
    axes[1, 0].hist(results_df['cl_dice'], bins=20, edgecolor='black', alpha=0.7, color='mediumseagreen')
    axes[1, 0].axvline(results_df['cl_dice'].mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {results_df["cl_dice"].mean():.3f}')
    axes[1, 0].set_xlabel('Centerline Dice', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Centerline Dice', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Hausdorff Distance distribution
    # Filter out inf values for visualization
    hd_finite = results_df['hausdorff'][results_df['hausdorff'] != float('inf')]
    if len(hd_finite) > 0:
        axes[1, 1].hist(hd_finite, bins=20, edgecolor='black', alpha=0.7, color='mediumpurple')
        axes[1, 1].axvline(hd_finite.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {hd_finite.mean():.2f}')
        axes[1, 1].set_xlabel('Hausdorff Distance (μm)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Distribution of Hausdorff Distance', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'metrics_distributions.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Plot 2: Dice vs IoU scatter
    # ========================================
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['dice'], results_df['iou'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.xlabel('Dice Score', fontsize=14, fontweight='bold')
    plt.ylabel('IoU Score', fontsize=14, fontweight='bold')
    plt.title('Dice vs IoU Correlation', fontsize=16, fontweight='bold')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='Perfect correlation')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / 'dice_vs_iou.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Plot 3: Precision-Recall
    # ========================================
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(results_df['recall'], results_df['precision'], 
                         c=results_df['dice'], cmap='viridis', 
                         alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Dice Score')
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall (colored by Dice)', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'precision_recall.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Plot 4: Box plots of all metrics
    # ========================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    metrics_to_plot = ['dice', 'iou', 'cl_dice', 'precision', 'recall', 'f1', 'specificity', 'overlap_coefficient']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 4, idx % 4]
        
        data_to_plot = results_df[metric]
        
        bp = ax.boxplot([data_to_plot], 
                        labels=[metric.replace('_', ' ').title()],
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the box
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.suptitle('Box Plots of All Metrics', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'metrics_boxplots.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Plot 5: Top and bottom performers
    # ========================================
    results_sorted = results_df.sort_values('dice', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 10
    top_n = min(10, len(results_sorted))
    top_samples = results_sorted.head(top_n)
    
    y_pos = np.arange(len(top_samples))
    axes[0].barh(y_pos, top_samples['dice'], color='green', alpha=0.7, edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_samples['sample_name'], fontsize=9)
    axes[0].set_xlabel('Dice Score', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Top {top_n} Samples', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Bottom 10
    bottom_samples = results_sorted.tail(top_n)
    
    y_pos = np.arange(len(bottom_samples))
    axes[1].barh(y_pos, bottom_samples['dice'], color='orange', alpha=0.7, edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(bottom_samples['sample_name'], fontsize=9)
    axes[1].set_xlabel('Dice Score', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Bottom {top_n} Samples', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'top_bottom_performers.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Plot 6: Correlation heatmap
    # ========================================
    correlation_metrics = ['dice', 'iou', 'cl_dice', 'precision', 'recall', 'f1', 'hausdorff']
    
    # Filter out inf values for correlation
    corr_df = results_df[correlation_metrics].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(corr_df) > 0:
        plt.figure(figsize=(10, 8))
        correlation_matrix = corr_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Metric Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # ========================================
    # Plot 7: Volume comparison
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(results_df['gt_volume'], results_df['pred_volume'], 
                   alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (perfect prediction)
    max_vol = max(results_df['gt_volume'].max(), results_df['pred_volume'].max())
    axes[0].plot([0, max_vol], [0, max_vol], 'r--', linewidth=2, alpha=0.5, label='Perfect prediction')
    
    axes[0].set_xlabel('Ground Truth Volume (voxels)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Volume (voxels)', fontsize=12, fontweight='bold')
    axes[0].set_title('Volume Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Volume difference histogram
    axes[1].hist(results_df['volume_diff'], bins=20, edgecolor='black', alpha=0.7, color='salmon')
    axes[1].axvline(results_df['volume_diff'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {results_df["volume_diff"].mean():.3f}')
    axes[1].set_xlabel('Relative Volume Difference', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Volume Differences', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'volume_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✅ Plots saved to: {plots_dir}")


def create_summary_table(results_df: pd.DataFrame, output_dir: Path):
    """
    Create a beautiful summary table
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save table
    """
    # Compute statistics for each metric
    metrics = ['dice', 'iou', 'cl_dice', 'precision', 'recall', 'f1', 'specificity', 'hausdorff']
    
    summary_data = []
    
    for metric in metrics:
        data = results_df[metric]
        
        # Filter out inf values for statistics
        if metric == 'hausdorff':
            data = data[data != float('inf')]
        
        if len(data) > 0:
            summary_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Mean': f"{data.mean():.4f}",
                'Std': f"{data.std():.4f}",
                'Median': f"{data.median():.4f}",
                'Min': f"{data.min():.4f}",
                'Max': f"{data.max():.4f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_csv = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(summary_csv, index=False)
    
    # Create a pretty plot of the table
    fig, ax = plt.subplots(figsize=(12, len(summary_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✅ Summary table saved: {summary_csv}")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("EVALUATION")
    logger.info("="*80)
    logger.info(f"Predictions: {args.pred_dir}")
    logger.info(f"Ground Truth: {args.gt_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Physical spacing (z,y,x): {args.spacing}")
    logger.info(f"GT format: {args.gt_format}")
    
    # Find matching pairs
    logger.info("\nFinding prediction-ground truth pairs...")
    pairs = find_matching_pairs(
        Path(args.pred_dir),
        Path(args.gt_dir),
        args.pred_format,
        args.gt_format  # ✅ NEW: Pass gt_format
    )
    
    logger.info(f"Found {len(pairs)} pairs")
    
    if len(pairs) == 0:
        logger.error("❌ No matching pairs found!")
        logger.error("Check that prediction and ground truth filenames match.")
        logger.error("Expected patterns:")
        logger.error("  Predictions: <name>_pred.npy")
        logger.error("  Ground Truth: <name>_mask.h5 or <name>_mask.npy")
        return
    
    # Evaluate each sample
    logger.info("\nEvaluating samples...")
    results = []
    
    spacing_tuple = tuple(args.spacing)
    
    for pair in tqdm(pairs, desc="Processing", ncols=100):
        try:
            # Load prediction and ground truth
            pred = load_prediction(pair['pred'], format=args.pred_format)
            gt = load_ground_truth(pair['gt'], format=args.gt_format)  # ✅ NEW: Pass gt_format
            
            # Ensure same shape
            if pred.shape != gt.shape:
                logger.warning(f"Shape mismatch for {pair['name']}")
                logger.warning(f"  Pred: {pred.shape}, GT: {gt.shape}")
                logger.warning("  Skipping this sample")
                continue
            
            # Evaluate
            metrics = evaluate_sample(pred, gt, pair['name'], spacing_tuple)
            results.append(metrics)
        
        except Exception as e:
            logger.error(f"Failed to evaluate {pair['name']}: {e}")
            continue
    
    if len(results) == 0:
        logger.error("❌ No samples were successfully evaluated!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    summary = {
        'num_samples': len(results_df),
        'metrics': {}
    }
    
    metric_cols = ['dice', 'iou', 'cl_dice', 'precision', 'recall', 'f1', 'specificity', 'hausdorff']
    
    for metric in metric_cols:
        if metric in results_df.columns:
            data = results_df[metric]
            
            # Handle inf values in hausdorff
            if metric == 'hausdorff':
                data_finite = data[data != float('inf')]
                if len(data_finite) > 0:
                    stats = {
                        'mean': float(data_finite.mean()),
                        'std': float(data_finite.std()),
                        'median': float(data_finite.median()),
                        'min': float(data_finite.min()),
                        'max': float(data_finite.max()),
                        'num_inf': int((data == float('inf')).sum())
                    }
                else:
                    stats = {'all_inf': True}
            else:
                stats = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'median': float(data.median()),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
            
            summary['metrics'][metric] = stats
            
            # Print summary
            logger.info(f"\n{metric.upper().replace('_', ' ')}:")
            if metric == 'hausdorff' and stats.get('all_inf'):
                logger.info("  All values are infinity (empty predictions)")
            elif metric == 'hausdorff':
                logger.info(f"  Mean:   {stats['mean']:.4f} ± {stats['std']:.4f} μm")
                logger.info(f"  Median: {stats['median']:.4f} μm")
                logger.info(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}] μm")
                if stats.get('num_inf', 0) > 0:
                    logger.info(f"  Note: {stats['num_inf']} samples with inf values (excluded)")
            else:
                logger.info(f"  Mean:   {stats['mean']:.4f} ± {stats['std']:.4f}")
                logger.info(f"  Median: {stats['median']:.4f}")
                logger.info(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Save results
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    
    # Save per-sample results
    if args.save_per_sample:
        results_csv = output_dir / 'per_sample_results.csv'
        results_df.to_csv(results_csv, index=False)
        logger.info(f"  ✅ Per-sample results: {results_csv}")
    
    # Save summary
    summary_json = output_dir / 'summary_statistics.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✅ Summary statistics: {summary_json}")
    
    # Create summary table
    create_summary_table(results_df, output_dir)
    
    # Create plots
    logger.info("\nCreating visualization plots...")
    create_evaluation_plots(results_df, output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETED!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}")
    
    # Print best and worst samples
    logger.info("\n📊 Best samples (by Dice):")
    for idx, row in results_df.nlargest(3, 'dice').iterrows():
        logger.info(f"  {row['sample_name']}: {row['dice']:.4f}")
    
    logger.info("\n📉 Worst samples (by Dice):")
    for idx, row in results_df.nsmallest(3, 'dice').iterrows():
        logger.info(f"  {row['sample_name']}: {row['dice']:.4f}")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()