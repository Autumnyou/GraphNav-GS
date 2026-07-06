"""
Node Feature Encoder: Encodes rich GS information into graph node features.

16维特征向量设计:
  Geometric  (6): pos(3) + local_curvature(3)
  Density    (3): density(1) + density_sigma(1) + mean_alpha(1)
  Uncertainty(3): confidence(1) + info_gain(1) + frontier_score(1)
  Semantics  (1): avg_color(1)
  Structural (3): free_ratio(1) + neighbor_occupied(1) + dist_to_obs(1)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from .node_classifier import NodeType


class NodeFeatureEncoder:
    """
    节点特征编码器。
    将丰富的GS信息编码为图节点的多维特征向量。

    All _compute_* methods are fully vectorized — no per-node Python loops.
    """

    FEATURE_DIM = 16

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def encode(self, gsplat, density_grid: torch.Tensor,
               voxel_centers: torch.Tensor, node_types: torch.Tensor,
               metadata: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码所有规划节点的特征。

        Args:
            gsplat: GSplatLoader实例
            density_grid: (Rx, Ry, Rz) 密度场
            voxel_centers: (Rx, Ry, Rz, 3) 体素中心
            node_types: (Rx, Ry, Rz) 节点类型
            metadata: 元信息 (bounds等)

        Returns:
            node_features: (N_plan, 16) 特征矩阵
            planning_indices: (N_plan, 3) 规划节点在网格中的索引
        """
        Rx, Ry, Rz = density_grid.shape

        # 找到参与规划的节点
        planning_mask = (node_types == NodeType.FREE) | (node_types == NodeType.FRONTIER)
        indices = torch.nonzero(planning_mask)  # (N_plan, 3)

        N_plan = indices.shape[0]
        if N_plan == 0:
            return torch.zeros(0, self.FEATURE_DIM, device=self.device), indices

        print(f'[FeatureEncoder] Encoding {N_plan} planning nodes...')

        # 初始化特征矩阵
        features = torch.zeros(N_plan, self.FEATURE_DIM, dtype=torch.float32, device=self.device)

        # 获取体素中心坐标
        positions = voxel_centers[indices[:, 0], indices[:, 1], indices[:, 2]]  # (N_plan, 3)

        # --- 特征1-3: 归一化位置 ---
        bounds = metadata['bounds']  # (2, 3)
        bounds_min = torch.as_tensor(bounds[0], dtype=positions.dtype, device=positions.device)
        bounds_max = torch.as_tensor(bounds[1], dtype=positions.dtype, device=positions.device)
        pos_normalized = (positions - bounds_min) / (bounds_max - bounds_min).clamp(min=1e-6)
        features[:, 0:3] = pos_normalized

        # --- 特征4-6: 局部几何曲率 (密度梯度) ---
        local_curvature = self._compute_local_curvature(density_grid, indices)
        features[:, 3:6] = local_curvature

        # --- 特征7: 密度值 ---
        density_values = density_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
        # 归一化到[0,1]
        max_density = max(density_grid.max().item(), 1e-6)
        features[:, 6] = density_values / max_density

        # --- 特征8: 密度一致性 (邻域密度标准差) ---
        density_sigma = self._compute_density_sigma(density_grid, indices)
        features[:, 7] = density_sigma

        # --- 特征9: 平均opacity ---
        mean_alpha = self._compute_mean_alpha(gsplat, positions, metadata)
        features[:, 8] = mean_alpha

        # --- 特征10: 置信度 ---
        uncertainty = self._compute_confidence(density_grid, indices)
        features[:, 9] = uncertainty

        # --- 特征11: 信息增益 (Shannon熵代理) ---
        info_gain = self._compute_info_gain(density_values)
        features[:, 10] = info_gain

        # --- 特征12: frontier标记 ---
        frontier_mask = node_types[indices[:, 0], indices[:, 1], indices[:, 2]] == NodeType.FRONTIER
        features[:, 11] = frontier_mask.float()

        # --- 特征13: 平均颜色 ---
        avg_color = self._compute_avg_color(gsplat, positions, metadata)
        features[:, 12] = avg_color

        # --- 特征14: 自由比例 (邻域中free节点占比) ---
        free_ratio = self._compute_free_ratio(node_types, indices)
        features[:, 13] = free_ratio

        # --- 特征15: 邻域障碍密度 ---
        neighbor_occ = self._compute_neighbor_occupied(node_types, indices)
        features[:, 14] = neighbor_occ

        # --- 特征16: 到最近障碍物距离 ---
        dist_to_obs = self._compute_dist_to_obs(node_types, indices, voxel_centers)
        features[:, 15] = dist_to_obs

        print(f'[FeatureEncoder] Feature shape: {features.shape}')

        return features, indices

    # ------------------------------------------------------------------
    # Vectorized helpers
    # ------------------------------------------------------------------

    def _compute_local_curvature(self, density_grid: torch.Tensor,
                                indices: torch.Tensor) -> torch.Tensor:
        """计算局部几何曲率 (密度Hessian对角元素的近似)

        Uses F.pad + shifted indexing — O(grid) work, no per-node loop.
        """
        # Pad with replicate so boundary nodes get curvature = 0 naturally
        # density_grid: (Rx, Ry, Rz)
        padded = F.pad(
            density_grid.unsqueeze(0).unsqueeze(0),  # (1,1,Rx,Ry,Rz)
            (1, 1, 1, 1, 1, 1),
            mode='replicate'
        ).squeeze(0).squeeze(0)  # (Rx+2, Ry+2, Rz+2)

        # Shifted indices (accounting for +1 padding offset)
        ix = indices[:, 0] + 1
        iy = indices[:, 1] + 1
        iz = indices[:, 2] + 1

        center = padded[ix, iy, iz]  # (N,)

        curvature = torch.stack([
            padded[ix + 1, iy, iz] - 2 * center + padded[ix - 1, iy, iz],  # dim 0
            padded[ix, iy + 1, iz] - 2 * center + padded[ix, iy - 1, iz],  # dim 1
            padded[ix, iy, iz + 1] - 2 * center + padded[ix, iy, iz - 1],  # dim 2
        ], dim=1)  # (N, 3)

        # 归一化
        max_curv = curvature.abs().max()
        if max_curv > 0:
            curvature = curvature / max_curv

        return curvature

    def _compute_density_sigma(self, density_grid: torch.Tensor,
                              indices: torch.Tensor) -> torch.Tensor:
        """计算邻域密度标准差

        Uses 3×3×3 uniform convolution to compute local mean and mean-of-squares,
        then sigma = sqrt(E[x²] - E[x]²).  O(grid) work, no per-node loop.
        """
        grid = density_grid.unsqueeze(0).unsqueeze(0)  # (1,1,Rx,Ry,Rz)

        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device, dtype=grid.dtype)

        # Count valid neighbors per cell (handles boundaries correctly)
        ones_grid = torch.ones_like(grid)
        count = F.conv3d(ones_grid, kernel, padding=1).squeeze(0).squeeze(0)  # (Rx,Ry,Rz)
        count = count.clamp(min=1)

        local_sum = F.conv3d(grid, kernel, padding=1).squeeze(0).squeeze(0)
        local_sq_sum = F.conv3d(grid ** 2, kernel, padding=1).squeeze(0).squeeze(0)

        local_mean = local_sum / count
        local_sq_mean = local_sq_sum / count
        variance = (local_sq_mean - local_mean ** 2).clamp(min=0)
        sigma_grid = variance.sqrt()  # (Rx, Ry, Rz)

        # Index planning nodes
        sigma = sigma_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

        # 归一化
        max_sigma = max(sigma.max().item(), 1e-6)
        return sigma / max_sigma

    def _compute_mean_alpha(self, gsplat, positions: torch.Tensor,
                            metadata: dict) -> torch.Tensor:
        """计算每个体素内高斯的平均opacity

        Spatial-binning approach: O(M + N) instead of O(M·N).
        Assigns each gaussian to its grid cell, then uses scatter_mean.
        """
        N = positions.shape[0]
        mean_alpha = torch.zeros(N, device=self.device)

        if not hasattr(gsplat, 'means') or gsplat.means is None:
            return mean_alpha

        means = torch.as_tensor(gsplat.means, dtype=torch.float32, device=self.device)  # (M, 3)
        raw_opacities = getattr(gsplat, 'opacities', None)
        if raw_opacities is None:
            raw_opacities = np.ones(len(gsplat.means), dtype=np.float32)
        opacities = torch.as_tensor(raw_opacities, dtype=torch.float32, device=self.device).reshape(-1)  # (M,)
        cell_size = torch.as_tensor(metadata['cell_sizes'], dtype=torch.float32, device=self.device)  # (3,)
        bounds = metadata['bounds']  # (2, 3) tensor or array
        bounds_min = torch.as_tensor(bounds[0], dtype=torch.float32, device=self.device)  # (3,)
        bounds_max = torch.as_tensor(bounds[1], dtype=torch.float32, device=self.device)

        # Compute grid dimensions from metadata
        grid_shape = torch.as_tensor(
            [int(((bounds_max[d] - bounds_min[d]) / cell_size[d]).round().item()) for d in range(3)],
            dtype=torch.long, device=self.device
        )
        # Use actual density grid shape if available (more reliable)
        # We'll infer it from positions shape — positions has N entries from
        # an (Rx, Ry, Rz) grid, but we received the grid dims via metadata
        # at encode() level.  Instead, just use the grid_shape we can compute.
        Rx, Ry, Rz = grid_shape[0].item(), grid_shape[1].item(), grid_shape[2].item()

        # --- Assign each gaussian to a cell ---
        # Cell index for each gaussian: floor((mean - bounds_min) / cell_size)
        cell_idx = ((means - bounds_min.unsqueeze(0)) / cell_size.unsqueeze(0)).long()  # (M, 3)

        # Clamp to valid range
        cell_idx[:, 0].clamp_(0, max(Rx - 1, 0))
        cell_idx[:, 1].clamp_(0, max(Ry - 1, 0))
        cell_idx[:, 2].clamp_(0, max(Rz - 1, 0))

        # Flatten cell index
        flat_cell = cell_idx[:, 0] * (Ry * Rz) + cell_idx[:, 1] * Rz + cell_idx[:, 2]  # (M,)

        total_cells = Rx * Ry * Rz

        # Scatter: sum opacities per cell and count per cell
        opacity_sum = torch.zeros(total_cells, device=self.device, dtype=torch.float32)
        cell_count = torch.zeros(total_cells, device=self.device, dtype=torch.float32)
        opacity_sum.scatter_add_(0, flat_cell, opacities)
        cell_count.scatter_add_(0, flat_cell, torch.ones_like(opacities))

        # Mean opacity per cell (cells with 0 gaussians get 0)
        cell_mean_opacity = torch.where(
            cell_count > 0, opacity_sum / cell_count, torch.zeros_like(opacity_sum)
        )

        # Look up planning nodes
        plan_flat = (positions[:, 0] - bounds_min[0]) / cell_size[0]
        plan_flat_y = (positions[:, 1] - bounds_min[1]) / cell_size[1]
        plan_flat_z = (positions[:, 2] - bounds_min[2]) / cell_size[2]
        plan_ix = plan_flat.long().clamp(0, max(Rx - 1, 0))
        plan_iy = plan_flat_y.long().clamp(0, max(Ry - 1, 0))
        plan_iz = plan_flat_z.long().clamp(0, max(Rz - 1, 0))
        plan_flat_idx = plan_ix * (Ry * Rz) + plan_iy * Rz + plan_iz

        mean_alpha = cell_mean_opacity[plan_flat_idx]

        # For cells with no gaussians, use 0.5× nearest gaussian opacity
        # Find cells that had no gaussian assigned
        empty_mask = cell_count[plan_flat_idx] == 0
        if empty_mask.any():
            # Fallback: compute global mean opacity as a reasonable default
            # (avoids O(N_empty * M) nearest-neighbor search)
            global_mean = opacities.mean() * 0.5
            mean_alpha[empty_mask] = global_mean

        return mean_alpha

    def _compute_confidence(self, density_grid: torch.Tensor,
                           indices: torch.Tensor) -> torch.Tensor:
        """计算观测置信度: 1 - U(v)

        Uses 3×3×3 convolution (same as density_sigma) to compute local
        variance, then confidence = max(0, 1 - var/density).
        """
        grid = density_grid.unsqueeze(0).unsqueeze(0)  # (1,1,Rx,Ry,Rz)

        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device, dtype=grid.dtype)

        ones_grid = torch.ones_like(grid)
        count = F.conv3d(ones_grid, kernel, padding=1).squeeze(0).squeeze(0)
        count = count.clamp(min=1)

        local_sum = F.conv3d(grid, kernel, padding=1).squeeze(0).squeeze(0)
        local_sq_sum = F.conv3d(grid ** 2, kernel, padding=1).squeeze(0).squeeze(0)

        local_mean = local_sum / count
        local_sq_mean = local_sq_sum / count
        variance_grid = (local_sq_mean - local_mean ** 2).clamp(min=0)

        # confidence = max(0, 1 - var / density)  where density > 1e-6
        density = density_grid
        ratio = torch.where(
            density > 1e-6,
            (variance_grid / density).clamp(max=1.0),
            torch.zeros_like(density)
        )
        confidence_grid = (1.0 - ratio).clamp(min=0)

        # For density <= 1e-6, original code sets confidence = 1.0
        confidence_grid = torch.where(density > 1e-6, confidence_grid, torch.ones_like(confidence_grid))

        return confidence_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

    def _compute_info_gain(self, density_values: torch.Tensor) -> torch.Tensor:
        """Shannon熵代理: H = -ρ·log(ρ) - (1-ρ)·log(1-ρ)"""
        rho = density_values.clamp(1e-6, 1 - 1e-6)
        entropy = -rho * torch.log(rho) - (1 - rho) * torch.log(1 - rho)
        # 归一化到[0,1]
        return entropy / (np.log(2))  # max entropy = ln(2) at ρ=0.5

    def _compute_avg_color(self, gsplat, positions: torch.Tensor, metadata: dict) -> torch.Tensor:
        """计算体素内高斯的平均RGB亮度

        Spatial-binning approach: O(M + N) instead of O(M·N).
        """
        N = positions.shape[0]
        avg_color = torch.zeros(N, device=self.device)

        if not hasattr(gsplat, 'means') or gsplat.means is None:
            return avg_color

        means = torch.as_tensor(gsplat.means, dtype=torch.float32, device=self.device)
        colors_raw = getattr(gsplat, 'colors', None)
        if colors_raw is None:
            return avg_color + 0.5  # default luminance

        colors = torch.as_tensor(colors_raw, dtype=torch.float32, device=self.device)
        if colors.ndim == 1:
            colors = colors.unsqueeze(-1).repeat(1, 3)

        # Compute per-gaussian luminance (mean of RGB channels)
        luminance = colors.mean(dim=-1)  # (M,)

        cell_size = torch.as_tensor(
            metadata.get('cell_sizes', np.array([0.1, 0.1, 0.1])),
            dtype=torch.float32, device=self.device
        )
        bounds = metadata['bounds']
        bounds_min = torch.as_tensor(bounds[0], dtype=torch.float32, device=self.device)
        bounds_max = torch.as_tensor(bounds[1], dtype=torch.float32, device=self.device)

        grid_shape = torch.as_tensor(
            [int(((bounds_max[d] - bounds_min[d]) / cell_size[d]).round().item()) for d in range(3)],
            dtype=torch.long, device=self.device
        )
        Rx, Ry, Rz = grid_shape[0].item(), grid_shape[1].item(), grid_shape[2].item()

        # Assign each gaussian to a cell
        cell_idx = ((means - bounds_min.unsqueeze(0)) / cell_size.unsqueeze(0)).long()
        cell_idx[:, 0].clamp_(0, max(Rx - 1, 0))
        cell_idx[:, 1].clamp_(0, max(Ry - 1, 0))
        cell_idx[:, 2].clamp_(0, max(Rz - 1, 0))

        flat_cell = cell_idx[:, 0] * (Ry * Rz) + cell_idx[:, 1] * Rz + cell_idx[:, 2]
        total_cells = Rx * Ry * Rz

        lum_sum = torch.zeros(total_cells, device=self.device, dtype=torch.float32)
        cell_count = torch.zeros(total_cells, device=self.device, dtype=torch.float32)
        lum_sum.scatter_add_(0, flat_cell, luminance)
        cell_count.scatter_add_(0, flat_cell, torch.ones_like(luminance))

        cell_mean_lum = torch.where(
            cell_count > 0, lum_sum / cell_count, torch.full_like(lum_sum, 0.5)
        )

        # Look up planning nodes
        plan_ix = ((positions[:, 0] - bounds_min[0]) / cell_size[0]).long().clamp(0, max(Rx - 1, 0))
        plan_iy = ((positions[:, 1] - bounds_min[1]) / cell_size[1]).long().clamp(0, max(Ry - 1, 0))
        plan_iz = ((positions[:, 2] - bounds_min[2]) / cell_size[2]).long().clamp(0, max(Rz - 1, 0))
        plan_flat_idx = plan_ix * (Ry * Rz) + plan_iy * Rz + plan_iz

        avg_color = cell_mean_lum[plan_flat_idx]

        return avg_color

    def _compute_free_ratio(self, node_types: torch.Tensor,
                            indices: torch.Tensor) -> torch.Tensor:
        """26邻域中free节点占比

        Uses 3×3×3 sum convolution on a binary FREE grid.
        """
        free_grid = (node_types == NodeType.FREE).float()  # (Rx,Ry,Rz)
        valid_grid = torch.ones_like(free_grid)

        grid_5d = free_grid.unsqueeze(0).unsqueeze(0)   # (1,1,Rx,Ry,Rz)
        valid_5d = valid_grid.unsqueeze(0).unsqueeze(0)

        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device, dtype=free_grid.dtype)

        free_count = F.conv3d(grid_5d, kernel, padding=1).squeeze(0).squeeze(0)
        total_count = F.conv3d(valid_5d, kernel, padding=1).squeeze(0).squeeze(0)
        total_count = total_count.clamp(min=1)

        ratio_grid = free_count / total_count  # (Rx, Ry, Rz)

        return ratio_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

    def _compute_neighbor_occupied(self, node_types: torch.Tensor,
                                  indices: torch.Tensor) -> torch.Tensor:
        """26邻域中occupied节点数/26

        Uses 3×3×3 sum convolution on a binary OCCUPIED grid.
        """
        occ_grid = (node_types == NodeType.OCCUPIED).float()
        grid_5d = occ_grid.unsqueeze(0).unsqueeze(0)

        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device, dtype=occ_grid.dtype)

        occ_count = F.conv3d(grid_5d, kernel, padding=1).squeeze(0).squeeze(0)

        return occ_count[indices[:, 0], indices[:, 1], indices[:, 2]] / 26.0

    def _compute_dist_to_obs(self, node_types: torch.Tensor,
                             indices: torch.Tensor,
                             voxel_centers: torch.Tensor) -> torch.Tensor:
        """到最近occupied体素的距离/voxel_size (使用k-NN避免OOM)"""
        N = indices.shape[0]
        dist = torch.ones(N, device=self.device)  # 默认1.0 (最远)

        # 找到所有occupied体素位置
        occ_mask = (node_types == NodeType.OCCUPIED)
        if not occ_mask.any():
            return dist

        occ_positions = voxel_centers[occ_mask].reshape(-1, 3)  # (N_occ, 3)
        plan_positions = voxel_centers[indices[:, 0], indices[:, 1], indices[:, 2]]

        # 使用k-NN方式替代全量计算
        k = min(8, occ_positions.shape[0])  # 只找最近8个邻居
        n_plan = plan_positions.shape[0]

        # 分块计算，但限制chunk大小以避免OOM
        chunk_size = 500  # 减小chunk避免中间tensor过大
        for start in range(0, n_plan, chunk_size):
            end = min(start + chunk_size, n_plan)
            chunk_pos = plan_positions[start:end].unsqueeze(1)  # (chunk, 1, 3)

            # 用更小的occupied子集来计算（如果occupied太多，随机采样）
            occ_use = occ_positions
            if occ_use.shape[0] > 50000:
                # 随机采样50000个occupied voxels
                idx = torch.randperm(occ_use.shape[0], device=self.device)[:50000]
                occ_use = occ_use[idx]

            # (chunk, 1, 3) - (1, N_occ, 3) -> (chunk, N_occ) -> min -> (chunk,)
            # 用chunk方式避免一次性分配过大tensor
            chunk_dists = []
            sub_chunk = 50  # 再分小避免OOM
            for c_start in range(0, occ_use.shape[0], sub_chunk):
                c_end = min(c_start + sub_chunk, occ_use.shape[0])
                c_occ = occ_use[c_start:c_end].unsqueeze(0)  # (1, sub_n, 3)
                c_diff = chunk_pos - c_occ  # (chunk, sub_n, 3)
                c_d = c_diff.norm(dim=-1)  # (chunk, sub_n)
                chunk_dists.append(c_d)
                del c_diff, c_d

            # 合并所有sub-chunk结果并取min
            all_dists = torch.cat(chunk_dists, dim=1)  # (chunk, total_occ_sampled)
            min_dists = all_dists.min(dim=1).values
            dist[start:end] = min_dists
            del chunk_pos, all_dists, chunk_dists
            torch.cuda.empty_cache()

        # 归一化
        dist = (dist * 10).clamp(0, 1)  # 简单归一化

        return dist
