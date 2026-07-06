"""
Node Classifier: Classifies voxels into Free/Occupied/Frontier/Unknown.

三类节点 + 一类特殊节点:
  OCCUPIED:  ρ > τ_high (高密度, 不可通行)
  FREE:      ρ = 0 (真正空白, 可通行)
  FRONTIER:  ρ > 0 且 ρ ≤ τ_high (有高斯证据, 但不算障碍)
  UNKNOWN:   非有限值/无效值 (不参与规划)
"""

import torch
import torch.nn.functional as F
import numpy as np
from enum import IntEnum
from typing import Tuple


class NodeType(IntEnum):
    """节点类型枚举"""
    OCCUPIED = -1    # 障碍物
    FREE = 0         # 自由空间
    FRONTIER = 1     # 边界/不确定区域
    UNKNOWN = 2      # 无效/非有限值 (不参与规划)


class NodeClassifier:
    """
    基于密度场的体素分类器。
    使用自适应阈值策略进行节点分类。
    """
    
    def __init__(self, density_high: float = 0.5, density_low: float = 0.1,
                 obs_expansion: int = 1):
        """
        Args:
            density_high: 高密度阈值 (occupied)
            density_low: 低密度阈值 (free)
            obs_expansion: 障碍物膨胀层数 (安全裕度)
        """
        self.density_high = density_high
        self.density_low = density_low
        self.obs_expansion = obs_expansion
        
        # 统计信息
        self.class_counts = {}
    
    def classify(self, density_grid: torch.Tensor) -> torch.Tensor:
        """
        执行节点分类。

        Uses adaptive thresholding when the raw density values are all above
        the configured thresholds (common in 3DGS scenes where large-covariance
        Gaussians produce uniformly high density).

        Args:
            density_grid: (Rx, Ry, Rz) 连续密度场

        Returns:
            node_types: (Rx, Ry, Rz) 节点类型 (NodeType枚举值)
        """
        node_types = torch.full_like(density_grid, NodeType.UNKNOWN, dtype=torch.long)

        finite_mask = torch.isfinite(density_grid)
        d_flat = density_grid.flatten()
        d_nonzero = d_flat[(d_flat > 0) & torch.isfinite(d_flat)]
        d_finite = d_nonzero

        tau_low = float(self.density_low)
        tau_high = float(self.density_high)

        # Prefer configured thresholds. Only fall back to adaptive percentiles
        # when the configured thresholds produce no usable planning set.
        use_adaptive = False
        if d_finite.numel() > 0:
            free_count = int(((density_grid > 0) & (density_grid <= tau_low)).sum().item())
            occupied_count = int((density_grid > tau_high).sum().item())
            planning_count = free_count + int(((density_grid > tau_low) & (density_grid <= tau_high)).sum().item())
            if planning_count == 0 or occupied_count == 0:
                use_adaptive = True

        if use_adaptive and d_finite.numel() > 0:
            d_clamped = d_finite.float().clamp(max=1e6)
            tau_low = torch.quantile(d_clamped, 0.30).item()
            tau_high = torch.quantile(d_clamped, 0.70).item()
            if tau_high <= tau_low:
                tau_high = tau_low * 2.0

        # Step 1: 基础分类
        node_types[finite_mask & (density_grid > tau_high)] = NodeType.OCCUPIED
        node_types[finite_mask & (density_grid == 0)] = NodeType.FREE
        node_types[finite_mask & (density_grid > 0) & (density_grid <= tau_high)] = NodeType.FRONTIER
        
        # Step 2: 障碍物膨胀 (安全裕度)
        if self.obs_expansion > 0:
            node_types = self._expand_occupied(node_types, self.obs_expansion)
        
        # Step 3: 统计
        self.class_counts = {
            'occupied': (node_types == NodeType.OCCUPIED).sum().item(),
            'free': (node_types == NodeType.FREE).sum().item(),
            'frontier': (node_types == NodeType.FRONTIER).sum().item(),
            'unknown': (node_types == NodeType.UNKNOWN).sum().item(),
            'total': node_types.numel(),
        }
        
        return node_types
    
    def _expand_occupied(self, node_types: torch.Tensor, 
                         expansion: int) -> torch.Tensor:
        """
        对占据体素进行膨胀，将周围自由/边界体素转为占据。
        
        Args:
            node_types: (Rx, Ry, Rz) 原始分类
            expansion: 膨胀层数
        
        Returns:
            expanded: (Rx, Ry, Rz) 膨胀后分类
        """
        expanded = node_types.clone()
        occupied = (node_types == NodeType.OCCUPIED).to(dtype=torch.float32)

        for _ in range(expansion):
            occupied = F.max_pool3d(
                occupied.unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            ).squeeze(0).squeeze(0)
            occupied = (occupied > 0).to(dtype=torch.float32)

        occupied_mask = occupied > 0
        expanded[occupied_mask & (expanded == NodeType.FREE)] = NodeType.OCCUPIED
        expanded[occupied_mask & (expanded == NodeType.FRONTIER)] = NodeType.OCCUPIED

        return expanded
    
    def compute_uncertainty(self, density_grid: torch.Tensor, 
                           node_types: torch.Tensor) -> torch.Tensor:
        """
        计算不确定性指标 (几何代理)。
        
        U(v) = Var_{i∈N(v)}[ρ_i] / max(ρ(v), ε)
        高方差/低密度 = 高不确定性
        
        Args:
            density_grid: (Rx, Ry, Rz) 密度场
            node_types: (Rx, Ry, Rz) 节点类型
        
        Returns:
            uncertainty: (Rx, Ry, Rz) 不确定性值 [0, 1]
        """
        grid = density_grid.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, 3, 3, 3, device=density_grid.device, dtype=density_grid.dtype)
        ones_grid = torch.ones_like(grid)
        count = F.conv3d(ones_grid, kernel, padding=1).squeeze(0).squeeze(0).clamp(min=1)
        local_sum = F.conv3d(grid, kernel, padding=1).squeeze(0).squeeze(0)
        local_sq_sum = F.conv3d(grid ** 2, kernel, padding=1).squeeze(0).squeeze(0)
        local_mean = local_sum / count
        local_sq_mean = local_sq_sum / count
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)

        rho = density_grid.clamp(min=1e-6)
        uncertainty = (local_var / rho).clamp(max=1.0)
        uncertainty = torch.where(density_grid > 0, uncertainty, torch.zeros_like(uncertainty))
        return uncertainty
    
    def get_planning_mask(self, node_types: torch.Tensor) -> torch.Tensor:
        """
        获取参与规划的节点mask (free + frontier)。
        
        Returns:
            mask: (Rx, Ry, Rz) bool, True表示参与规划
        """
        return (node_types == NodeType.FREE) | (node_types == NodeType.FRONTIER)
