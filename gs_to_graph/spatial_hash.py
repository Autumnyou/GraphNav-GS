"""
Spatial Hash Grid for accelerating density field queries.
Provides O(1) average-time neighbor lookups for Gaussian splats.

Optimized version:
  - _build_hash() uses numpy-accelerated loop with pre-allocated structures
  - query() returns a list of 1D LongTensors (sparse) instead of a dense (M, N) bool mask,
    avoiding catastrophic memory usage when M and N are both large
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, List


class SpatialHashGrid:
    """空间哈希网格，用于加速体素密度场估计。"""

    def __init__(self, means: torch.Tensor, covs: torch.Tensor,
                 cell_size: float = 0.3, device: str = 'cuda'):
        """
        Args:
            means: (N, 3) 高斯中心点坐标
            covs: (N, 3, 3) 高斯协方差矩阵
            cell_size: 哈希网格单元大小
            device: 计算设备
        """
        self.device = device
        self.cell_size = cell_size
        means = means.to(device)
        covs = covs.to(device)
        self.N = means.shape[0]
        self._cached_means = means

        # 计算每个高斯的截断半径: 3σ
        # 截断半径 = 3 * sqrt(max(diag(Σ)))
        diag_covs = torch.diagonal(covs, dim1=1, dim2=2)  # (N, 3)
        trunc_radius = 3.0 * torch.sqrt(torch.max(diag_covs, dim=1).values)  # (N,)

        # 计算哈希键
        mins = means - trunc_radius.unsqueeze(1)
        maxs = means + trunc_radius.unsqueeze(1)

        # 空间范围
        self.global_min = mins.min(dim=0).values
        self.global_max = maxs.max(dim=0).values

        # 计算网格维度
        self.grid_dims = torch.ceil(
            (self.global_max - self.global_min) / cell_size
        ).long()

        # 构建空间哈希
        self._build_hash(means, mins, maxs)

        # 保存截断半径供查询时使用
        self.trunc_radii = trunc_radius

        print(f'[SpatialHash] Built grid: {self.grid_dims.tolist()} cells, '
              f'{self.N} Gaussians, cell_size={cell_size:.3f}m')

    def _build_hash(self, means: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor):
        """
        构建空间哈希表。

        Optimized: cell index bounds are computed on GPU, then transferred to CPU
        numpy arrays. The Python loop uses only numpy indexing and dict operations
        (no torch tensor creation inside the loop).  Dict values are converted to
        device LongTensors once at the end.
        """
        # Vectorised cell-range computation (GPU)
        cell_min_idx = torch.floor((mins - self.global_min) / self.cell_size).long()
        cell_max_idx = torch.ceil((maxs - self.global_min) / self.cell_size).long()

        # Move to CPU numpy *once* – avoids per-element .tolist() / tensor indexing
        cell_min_np = cell_min_idx.cpu().numpy()  # (N, 3) int64
        cell_max_np = cell_max_idx.cpu().numpy()  # (N, 3) int64

        # Build dict of lists using plain Python – dict ops are fast
        cell_to_gaussians: dict = defaultdict(list)

        for i in range(self.N):
            x0, y0, z0 = cell_min_np[i]
            x1, y1, z1 = cell_max_np[i]
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        cell_to_gaussians[(x, y, z)].append(i)

        # Pre-convert every value list to a device LongTensor so that query()
        # can use them directly without any per-lookup conversion.
        self.cell_to_gaussians = {
            k: torch.tensor(v, dtype=torch.long, device=self.device)
            for k, v in cell_to_gaussians.items()
        }

    def query(self, points: torch.Tensor,
              query_radius: Optional[float] = None) -> List[torch.Tensor]:
        """
        查询每个点附近的候选高斯索引。

        Args:
            points: (M, 3) 查询点坐标
            query_radius: 查询半径 (默认使用 cell_size)

        Returns:
            candidates: length-M list of 1-D LongTensors on *self.device*.
                        candidates[i] contains the (deduplicated) indices of
                        gaussians whose cells overlap with query point i.
                        Empty tensor for points with no nearby gaussians.
        """
        M = points.shape[0]
        if query_radius is None:
            query_radius = self.cell_size

        # Vectorised cell-range computation (GPU), then move to CPU numpy
        q_mins = points - query_radius
        q_maxs = points + query_radius

        cell_min_idx = torch.floor((q_mins - self.global_min) / self.cell_size).long()
        cell_max_idx = torch.ceil((q_maxs - self.global_min) / self.cell_size).long()

        cmin = cell_min_idx.cpu().numpy()  # (M, 3)
        cmax = cell_max_idx.cpu().numpy()  # (M, 3)

        cell_map = self.cell_to_gaussians  # local ref – avoids repeated attr lookup
        _empty = torch.empty(0, dtype=torch.long, device=self.device)

        candidates: List[torch.Tensor] = [_empty] * M

        for i in range(M):
            x0, y0, z0 = cmin[i]
            x1, y1, z1 = cmax[i]

            # Fast path: single cell (very common when query_radius <= cell_size)
            if x0 == x1 and y0 == y1 and z0 == z1:
                t = cell_map.get((x0, y0, z0))
                if t is not None:
                    candidates[i] = t
                continue

            # Multi-cell: gather and concatenate
            parts = []
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        t = cell_map.get((x, y, z))
                        if t is not None:
                            parts.append(t)

            if len(parts) == 0:
                continue
            elif len(parts) == 1:
                candidates[i] = parts[0]
            else:
                # torch.cat + unique to deduplicate across cell boundaries
                candidates[i] = torch.cat(parts).unique()

        return candidates

    def query_neighbors(self, points: torch.Tensor,
                        k: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查询每个点的 k 个最近高斯邻居。

        Args:
            points: (M, 3) 查询点
            k: 邻居数量

        Returns:
            distances: (M, k) 距离
            indices: (M, k) 高斯索引
        """
        candidates = self.query(points)

        M = points.shape[0]
        distances = torch.full((M, k), float('inf'), dtype=torch.float32, device=self.device)
        indices = torch.full((M, k), -1, dtype=torch.long, device=self.device)

        for i in range(M):
            candidate_idx = candidates[i]
            if candidate_idx.shape[0] == 0:
                continue

            # 计算欧氏距离
            dist = torch.norm(points[i] - self._all_means[candidate_idx], dim=1)

            # 取前 k 个最近邻
            k_actual = min(k, candidate_idx.shape[0])
            topk_dist, topk_idx = torch.topk(dist, k_actual, largest=False)

            distances[i, :k_actual] = topk_dist
            indices[i, :k_actual] = candidate_idx[topk_idx]

        return distances, indices

    @property
    def _all_means(self):
        # 缓存 means (延迟加载)
        if not hasattr(self, '_cached_means'):
            raise RuntimeError("Means not cached. Use query() instead.")
        return self._cached_means

    def get_stats(self) -> dict:
        """返回哈希网格统计信息"""
        cell_sizes = [v.shape[0] for v in self.cell_to_gaussians.values()]
        return {
            'num_cells': len(self.cell_to_gaussians),
            'num_gaussians': self.N,
            'grid_dims': self.grid_dims.tolist(),
            'cell_size': self.cell_size,
            'avg_gaussians_per_cell': float(np.mean(cell_sizes)) if cell_sizes else 0.0,
        }
