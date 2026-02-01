"""
Advanced Learning Rate Scheduler Builder
✅ Supports warmup + cosine annealing with restarts
✅ Config-driven implementation
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with:
    - Linear warmup for first N epochs
    - Cosine annealing after warmup
    - Optional restarts at specified epoch
    
    Example config:
        warmup_epochs: 10
        total_epochs: 100
        min_lr: 5e-5
        restart_epoch: 50 (optional)
        restart_multiplier: 0.5 (optional)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        restart_epoch: int = None,
        restart_multiplier: float = 1.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.restart_epoch = restart_epoch
        self.restart_multiplier = restart_multiplier
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current epoch"""
        epoch = self.last_epoch
        
        # Phase 1: Linear warmup
        if epoch < self.warmup_epochs:
            # Linear increase from min_lr to base_lr
            warmup_factor = (epoch + 1) / self.warmup_epochs
            return [
                self.min_lr + (base_lr - self.min_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        
        # Phase 2: Cosine annealing (possibly with restart)
        else:
            # Check for restart
            if self.restart_epoch and epoch >= self.restart_epoch:
                # Restart from reduced base LR
                effective_base_lrs = [
                    base_lr * self.restart_multiplier 
                    for base_lr in self.base_lrs
                ]
                # Recalculate cosine from restart point
                epochs_since_restart = epoch - self.restart_epoch
                remaining_epochs = self.total_epochs - self.restart_epoch
            else:
                effective_base_lrs = self.base_lrs
                epochs_since_restart = epoch - self.warmup_epochs
                remaining_epochs = self.total_epochs - self.warmup_epochs
            
            # Cosine annealing formula
            cosine_factor = 0.5 * (
                1 + math.cos(math.pi * epochs_since_restart / remaining_epochs)
            )
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in effective_base_lrs
            ]


def build_scheduler(optimizer: Optimizer, config: dict):
    """
    Factory function to build scheduler from config
    
    Args:
        optimizer: PyTorch optimizer
        config: Training config dict with keys:
            - learning_rate: base LR
            - min_lr: minimum LR
            - epochs: total epochs
            - warmup_epochs: warmup duration (optional)
            - scheduler_type: "cosine" or "cosine_with_restarts"
            - restart_epoch: when to restart (optional)
            - restart_multiplier: LR multiplier at restart (optional)
    
    Returns:
        PyTorch LR scheduler
    """
    training_config = config['training']
    
    # Extract parameters with defaults
    total_epochs = training_config['epochs']
    min_lr = training_config.get('min_lr', 1e-6)
    warmup_epochs = training_config.get('warmup_epochs', 0)
    scheduler_type = training_config.get('scheduler_type', 'cosine')
    restart_epoch = training_config.get('restart_epoch', None)
    restart_multiplier = training_config.get('restart_multiplier', 1.0)
    
    # Build scheduler based on type
    if warmup_epochs > 0 or scheduler_type == 'cosine_with_restarts':
        # Use advanced scheduler
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
            restart_epoch=restart_epoch,
            restart_multiplier=restart_multiplier
        )
        
        print(f"✅ Advanced Scheduler Created:")
        print(f"  Type: Warmup + Cosine")
        print(f"  Warmup epochs: {warmup_epochs}")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Min LR: {min_lr:.2e}")
        if restart_epoch:
            print(f"  Restart at epoch: {restart_epoch}")
            print(f"  Restart multiplier: {restart_multiplier}")
    
    else:
        # Fallback to basic cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr
        )
        print(f"✅ Basic Cosine Scheduler Created")
        print(f"  T_max: {total_epochs}")
        print(f"  Min LR: {min_lr:.2e}")
    
    return scheduler


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_scheduler():
    """Test the scheduler behavior"""
    import matplotlib.pyplot as plt
    
    # Dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    
    # Test config
    config = {
        'training': {
            'epochs': 100,
            'learning_rate': 8e-4,
            'min_lr': 5e-5,
            'warmup_epochs': 10,
            'scheduler_type': 'cosine_with_restarts',
            'restart_epoch': 50,
            'restart_multiplier': 0.5
        }
    }
    
    scheduler = build_scheduler(optimizer, config)
    
    # Simulate training
    lrs = []
    for epoch in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(lrs, linewidth=2)
    plt.axvline(10, color='green', linestyle='--', alpha=0.5, label='Warmup End')
    plt.axvline(50, color='red', linestyle='--', alpha=0.5, label='Restart')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Advanced Scheduler: Warmup + Cosine with Restart', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('./results/outputs/scheduler_test.png', dpi=150, bbox_inches='tight')
    print("✅ Scheduler test plot saved!")


if __name__ == '__main__':
    test_scheduler()