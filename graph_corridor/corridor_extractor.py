"""
Graph-Guided Corridor Extractor: Extracts safe flight corridors using graph topology.

Innovation over SplatNav:
  - Uses graph structure to guide corridor width (wider in high-confidence regions)
  - Frontier nodes narrow corridors (uncertainty-aware)
  - Preserves graph topology (no SFC ellipsoid fitting needed)
  - Output format compatible with SplatNav's B-Spline optimizer

Output: List of polytopes [(A, b), ...] where Ax <= b defines each corridor.
"""

import torch
import numpy as np
import os
from typing import List, Tuple, Dict, Optional

if os.environ.get('GRAPHNAV_DISABLE_PYG') == '1':
    HAS_PYG = False
    HeteroData = None
else:
    try:
        import torch_geometric as pyg
        from torch_geometric.data import HeteroData
        HAS_PYG = True
    except ImportError:
        HAS_PYG = False
        HeteroData = None


class GraphCorridorExtractor:
    """
    图结构引导的走廊提取器。

    沿GAT-A*输出的路径节点序列，利用图拓扑和节点属性
    生成安全飞行走廊（Polytope格式，与SplatNav兼容）。
    """

    def __init__(self, uncertainty_width_factor: float = 0.8,
                 min_corridor_width: float = 0.05,
                 corridor_margin: float = 0.02,
                 use_uncertainty: bool = True,
                 max_points_per_corridor: int = 6,
                 max_corridors: int = 6,
                 device: str = 'cuda'):
        """
        Args:
            uncertainty_width_factor: 不确定性→走廊宽度缩放因子
            min_corridor_width: 最小走廊宽度 (m)
            corridor_margin: 走廊安全裕度 (m)
            device: 计算设备
        """
        self.uncertainty_width_factor = uncertainty_width_factor
        self.min_corridor_width = min_corridor_width
        self.corridor_margin = corridor_margin
        self.use_uncertainty = use_uncertainty
        self.max_points_per_corridor = max(2, int(max_points_per_corridor))
        self.max_corridors = max(1, int(max_corridors))
        self.device = device

    def extract(self, graph: HeteroData, path_nodes: List,
                path_coords: torch.Tensor, path_types: List[str],
                metadata: dict, voxel_size: float = 0.1,
                use_uncertainty: Optional[bool] = None,
                gsplat_collision_set=None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        沿路径提取安全走廊。

        Args:
            graph: PyG HeteroData
            path_nodes: 路径节点序列
            path_coords: (K, 3) 路径世界坐标
            path_types: 节点类型列表
            metadata: 图元信息
            voxel_size: 体素大小

        Returns:
            corridors: List[(A, b)] polytope列表
                A: (M_i, 3) 半空间法向量
                b: (M_i,) 半空间偏置
        """
        K = path_coords.shape[0]
        if K < 2:
            return []

        use_unc = self.use_uncertainty if use_uncertainty is None else use_uncertainty

        # Split the path into a small number of overlapping chunks so corridor
        # constraints remain expressive without over-constraining the spline QP.
        step = self.max_points_per_corridor
        sections = []
        start_idx = 0
        while start_idx < K - 1:
            end_idx = min(K, start_idx + step)
            if end_idx - start_idx < 2:
                end_idx = K
            sections.append((start_idx, end_idx))
            if end_idx >= K:
                break
            if len(sections) >= self.max_corridors:
                sections[-1] = (sections[-1][0], K)
                break
            start_idx = end_idx - 1  # one-point overlap for continuity

        corridors = []
        for start_idx, end_idx in sections:
            chunk_coords = path_coords[start_idx:end_idx]
            if chunk_coords.shape[0] < 2:
                # Skip single-point sections (degenerate corridor)
                continue
            chunk_nodes = path_nodes[start_idx:end_idx] if path_nodes is not None else []
            chunk_types = path_types[start_idx:end_idx] if path_types is not None else []

            if use_unc:
                width_samples = []
                for j in range(chunk_coords.shape[0]):
                    node_type = chunk_types[j] if j < len(chunk_types) else 'free'
                    confidence = self._get_node_confidence(
                        graph, chunk_nodes, chunk_types, j
                    )
                    width_samples.append(self._confidence_to_width(confidence, node_type))
                width_factor = float(np.clip(np.mean(width_samples), 0.0, 1.0)) if width_samples else 1.0
            else:
                width_factor = 1.0
            corridors.append(self._compute_chunk_tube(
                chunk_coords, voxel_size, width_factor
            ))

        return corridors

    def extract_corridors(self, graph: HeteroData, path_nodes: List,
                          path_coords: Optional[torch.Tensor] = None,
                          path_types: Optional[List[str]] = None,
                          metadata: Optional[dict] = None,
                          voxel_size: float = 0.1,
                          gsplat_collision_set=None):
        """Backward-compatible wrapper for older call sites."""
        if metadata is None:
            metadata = getattr(graph, 'metadata', {})
        if path_coords is None or path_types is None:
            raise ValueError('extract_corridors now requires path_coords and path_types')
        return self.extract(
            graph, path_nodes, path_coords, path_types, metadata, voxel_size,
            gsplat_collision_set=gsplat_collision_set,
        )

    def _compute_chunk_tube(self, points: torch.Tensor, voxel_size: float,
                            width_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a local tube corridor around all points in a chunk."""
        pts = points.reshape(-1, 3)
        if pts.shape[0] < 2:
            raise ValueError('chunk must contain at least 2 points')

        p0 = pts[0]
        p1 = pts[-1]
        direction = p1 - p0
        length = torch.norm(direction)
        direction = direction / (length + 1e-8)

        helper = torch.tensor([1.0, 0.0, 0.0], device=pts.device, dtype=pts.dtype)
        if float(torch.abs(direction[0])) > 0.9:
            helper = torch.tensor([0.0, 1.0, 0.0], device=pts.device, dtype=pts.dtype)
        local_y = torch.cross(direction, helper, dim=0)
        if float(torch.norm(local_y)) < 1e-8:
            helper = torch.tensor([0.0, 0.0, 1.0], device=pts.device, dtype=pts.dtype)
            local_y = torch.cross(direction, helper, dim=0)
        local_y = local_y / (torch.norm(local_y) + 1e-8)
        local_z = torch.cross(direction, local_y, dim=0)
        local_z = local_z / (torch.norm(local_z) + 1e-8)

        basis = torch.stack([direction, local_y, local_z], dim=0)
        local_pts = torch.matmul(pts, basis.T)

        width_factor = float(np.clip(width_factor, 0.0, 1.0))
        pad_scale = 1.0 + 0.8 * (1.0 - width_factor)
        axis_pad = self.corridor_margin + voxel_size * (1.5 * pad_scale)
        spread = torch.max(torch.abs(local_pts[:, 1:]), dim=0).values
        lateral = torch.max(spread + axis_pad, torch.full_like(spread, float(self.min_corridor_width)))

        mins = torch.min(local_pts, dim=0).values - axis_pad
        maxs = torch.max(local_pts, dim=0).values + axis_pad
        mins[1:] = torch.minimum(mins[1:], -lateral)
        maxs[1:] = torch.maximum(maxs[1:], lateral)

        A = torch.cat([basis, -basis], dim=0)
        b = torch.cat([maxs, -mins], dim=0)
        return A, b

    def _shrink_corridor(self, A: torch.Tensor, b: torch.Tensor,
                         width_factor: float, midpoint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shrink corridor by robot radius and uncertainty-based width factor.
        """
        shrink_amount = self.corridor_margin + (1.0 - width_factor) * self.min_corridor_width

        # Normalize normals
        norms = torch.norm(A, dim=1, keepdim=True) + 1e-8
        A_norm = A / norms

        # Shrink
        b_norm = b / norms.squeeze() - shrink_amount

        return A_norm, b_norm

    def _get_node_confidence(self, graph: HeteroData, path_nodes: List,
                             path_types: List[str], idx: int) -> float:
        """Get confidence value for a path node."""
        if idx >= len(path_nodes):
            return 1.0

        node_type = path_types[idx]
        local_idx = path_nodes[idx]

        if node_type == 'free' and local_idx < graph['free'].x.shape[0]:
            return graph['free'].x[local_idx, 9].item()  # confidence feature
        elif node_type == 'frontier' and local_idx < graph['frontier'].x.shape[0]:
            return graph['frontier'].x[local_idx, 9].item()
        return 0.5  # default

    def _confidence_to_width(self, confidence: float, node_type: str) -> float:
        """Convert confidence to width factor [0, 1]."""
        if node_type == 'frontier':
            # Frontier nodes: narrower corridors
            return 0.3 + 0.4 * confidence
        else:
            # Free nodes: wider corridors proportional to confidence
            return 0.5 + 0.5 * confidence

    def _try_merge_corridors(self, A_prev: List, b_prev: List,
                              A_new: torch.Tensor, b_new: torch.Tensor,
                              midpoint: torch.Tensor) -> Tuple[bool, List, List]:
        """Try to merge new corridor segment with current corridor."""
        # Disabled in the current ablation path: keep one polytope per segment.
        return False, A_prev, b_prev
