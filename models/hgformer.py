"""
HGFormer3D Encoder with Auto-Tracking of Downsampling
✅ FIXED: Stores stem_stride and stage_strides for decoder alignment
✅ FIXED: Computes total_downsample_factor automatically
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from .attention import CS_KNN_3D
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.attention import  CS_KNN_3D


class HGFormer3D(nn.Module):
    """
    ✅ FIXED: Self-Documenting Encoder
    
    Key Features:
    - Automatically tracks downsampling at each stage
    - Exposes downsampling factors for decoder
    - No hard-coded values that can cause misalignment
    
    Attributes:
        stem_stride: (D, H, W) downsampling in stem
        stage_strides: List of (D, H, W) downsampling per stage
        total_downsample: Total accumulated downsampling
        channels: List of output channels per stage
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depths: List[int] = [1, 2, 4, 2],
        num_hyperedges: List[int] = [64, 32, 16, 8],
        K_neighbors: List[int] = [128, 64, 32, 8],
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        stem_stride: Tuple[int, int, int] = (2, 4, 4),  # ✅ Now configurable!
        stage_strides: Optional[List[Tuple[int, int, int]]] = None
    ):
        """
        Args:
            stem_stride: (D, H, W) downsampling factor for initial stem
            stage_strides: List of (D, H, W) downsampling per stage
                          If None, defaults to [(2,2,2), (2,2,2), (2,2,2), (1,1,1)]
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depths = depths
        self.num_stages = len(depths)
        
        # ✅ CRITICAL FIX: Store stride information
        self.stem_stride = stem_stride
        
        if stage_strides is None:
            # Default: downsample by 2x at first 3 stages, no downsample at last
            stage_strides = [(2, 2, 2)] * 3 + [(1, 1, 1)]
        
        self.stage_strides = stage_strides
        
        # Validate configuration
        assert len(stage_strides) == self.num_stages, \
            f"stage_strides length {len(stage_strides)} != num_stages {self.num_stages}"
        
        # Calculate channels for each stage
        self.channels = [
            base_channels,      # Stage 0: 32
            base_channels * 2,  # Stage 1: 64
            base_channels * 5,  # Stage 2: 160
            base_channels * 8   # Stage 3: 256
        ]
        
        # ✅ CRITICAL: Compute total downsampling factor
        self.total_downsample = self._compute_total_downsample()
        
        logger.info(f"HGFormer3D Configuration:")
        logger.info(f"  Stem stride: {self.stem_stride}")
        logger.info(f"  Stage strides: {self.stage_strides}")
        logger.info(f"  Total downsample: {self.total_downsample}")
        logger.info(f"  Channels: {self.channels}")
        
        # Initial stem: Conv3D for patch embedding
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                base_channels, 
                kernel_size=stem_stride,  # ✅ Use stored value
                stride=stem_stride,       # ✅ Use stored value
                padding=(0, 0, 0)
            ),
            nn.BatchNorm3d(base_channels),
            nn.GELU()
        )
        
        # Build 4 stages
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            # Determine input channels
            if stage_idx == 0:
                stage_in_channels = base_channels
            else:
                stage_in_channels = self.channels[stage_idx - 1]
            
            stage = HGFormerStage(
                stage_idx=stage_idx,
                in_channels=stage_in_channels,
                out_channels=self.channels[stage_idx],
                depth=depths[stage_idx],
                num_hyperedges=num_hyperedges[stage_idx],
                K=K_neighbors[stage_idx],
                stride=stage_strides[stage_idx],  # ✅ Pass stride
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            self.stages.append(stage)
        
        # SSL heads (for pretraining)
        self.ssl_heads = nn.ModuleDict({
            'reconstruction': nn.Conv3d(self.channels[-1], in_channels, 1),
            'rotation': nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(self.channels[-1], 4)
            ),
            'marker': nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(self.channels[-1], 4)
            )
        })
    
    def _compute_total_downsample(self) -> Tuple[int, int, int]:
        """
        ✅ NEW: Compute total downsampling factor
        
        Returns:
            (D_factor, H_factor, W_factor)
        """
        total_d = self.stem_stride[0]
        total_h = self.stem_stride[1]
        total_w = self.stem_stride[2]
        
        for stride in self.stage_strides:
            total_d *= stride[0]
            total_h *= stride[1]
            total_w *= stride[2]
        
        return (total_d, total_h, total_w)
    
    def get_downsample_factor(self) -> Tuple[int, int, int]:
        """
        ✅ NEW: Public API for decoder to query downsampling
        
        Returns:
            (D_factor, H_factor, W_factor) - total downsampling from input
        """
        return self.total_downsample
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass
        
        Args:
            x: (B, 1, D, H, W) - input volume
            return_features: If True, return features from all stages
        
        Returns:
            If return_features=False: final features (B, C, D', H', W')
            If return_features=True: list of features from each stage
        """
        # Stem
        x = self.stem(x)  # (B, C, D/stem_d, H/stem_h, W/stem_w)
        
        # Store features from each stage
        features = []
        
        # Process through stages
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
        
        if return_features:
            return features
        else:
            return x
    
    # SSL-specific methods (unchanged)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode volume to embedding (for contrastive learning)"""
        features = self.forward(x, return_features=False)
        embedding = features.mean(dim=[2, 3, 4])
        return embedding
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct volume (for inpainting)"""
        features = self.forward(x, return_features=False)
        reconstruction = self.ssl_heads['reconstruction'](features)
        # Upsample to original size
        reconstruction = nn.functional.interpolate(
            reconstruction, 
            size=x.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        return reconstruction
    
    def predict_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Predict rotation angle"""
        features = self.forward(x, return_features=False)
        logits = self.ssl_heads['rotation'](features)
        return logits
    
    def classify_marker(self, x: torch.Tensor) -> torch.Tensor:
        """Classify marker type"""
        features = self.forward(x, return_features=False)
        logits = self.ssl_heads['marker'](features)
        return logits


class HGFormerStage(nn.Module):
    """
    ✅ FIXED: Stage with shared hypergraph construction
    
    Key Changes:
    - Constructs hypergraph ONCE per stage (not per block!)
    - Passes same H matrix to all blocks
    - Stores stride information
    """
    
    def __init__(
        self,
        stage_idx: int,
        in_channels: int,
        out_channels: int,
        depth: int,
        num_hyperedges: int,
        K: int,
        stride: Tuple[int, int, int],
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.stage_idx = stage_idx
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.stride = stride
        
        # Downsampling (if stride != (1,1,1))
        if stride != (1, 1, 1):
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, 
                    out_channels,
                    kernel_size=stride,
                    stride=stride
                ),
                nn.BatchNorm3d(out_channels),
                nn.GELU()
            )
        else:
            # No spatial downsampling, just channel projection
            if in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm3d(out_channels),
                    nn.GELU()
                )
            else:
                self.downsample = nn.Identity()
        
        # ✅ CRITICAL FIX: Single hypergraph constructor for the stage
        self.hypergraph_constructor = CS_KNN_3D(
            embed_dim=out_channels,
            num_hyperedges=num_hyperedges,
            K=K,
            spatial_shape=None,  # Will be set dynamically
            spacing=(2.0, 1.0, 1.0)
        )
        
        # ✅ FIXED: Multiple blocks share the same hypergraph
        self.blocks = nn.ModuleList([
            HGFormerBlockShared(
                embed_dim=out_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stage
        
        Args:
            x: (B, C_in, D, H, W)
        
        Returns:
            x: (B, C_out, D', H', W')
        """
        # Downsample
        x = self.downsample(x)  # (B, C_out, D', H', W')
        
        B, C, D, H, W = x.shape
        
        # Update hypergraph constructor's spatial shape
        self.hypergraph_constructor.spatial_shape = (D, H, W)
        
        # Reshape to sequence: (B, C, D, H, W) → (B, N, C)
        x_flat = rearrange(x, 'b c d h w -> b (d h w) c')
        
        # ✅ CRITICAL FIX: Construct hypergraph ONCE for all blocks in this stage
        H_matrix = self.hypergraph_constructor(x_flat)  # (B, N, Ne)
        
        # Pass through all blocks with the SAME hypergraph
        for block in self.blocks:
            x_flat = block(x_flat, H_matrix)  # Blocks share H!
        
        # Reshape back: (B, N, C) → (B, C, D, H, W)
        x = rearrange(x_flat, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
        return x


class HGFormerBlockShared(nn.Module):
    """
    ✅ NEW: HGFormer Block that receives pre-computed hypergraph
    
    Key Change:
    - Does NOT construct hypergraph (receives it from stage)
    - Only performs node-hyperedge-node messaging
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Import HGA modules
        from .attention import HGA_NodeToHyperedge, HGA_HyperedgeToNode
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # HyperGraph Attention modules
        self.hga_n2e = HGA_NodeToHyperedge(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.hga_e2n = HGA_HyperedgeToNode(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-computed hypergraph
        
        Args:
            x: (B, N, C) - node features
            H: (B, N, Ne) - pre-computed hypergraph matrix
        
        Returns:
            x: (B, N, C) - updated node features
        """
        # Node → Hyperedge messaging
        hyperedges = self.hga_n2e(self.norm1(x), H)  # (B, Ne, C)
        
        # Hyperedge → Node messaging
        x = x + self.hga_e2n(hyperedges, H, self.norm2(x))
        
        # Feed-forward with residual
        x = x + self.mlp(self.norm3(x))
        
        return x