"""
Edge Builder: Constructs multi-relation edges for heterogeneous graph.

Three edge relations:
  spatial_free_free:      free <-> free, 26-neighbor
  spatial_free_frontier: free <-> frontier, 26-neighbor
  frontier_frontier:      frontier <-> frontier, 26-neighbor

Edge weight:
  w_ij = lambda_1 * d_ij + lambda_2 * (1 - mean_conf_ij) + lambda_3 * obs_density_ij

Vectorized implementation — avoids Python for-loops over nodes/neighbors.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from .node_classifier import NodeType


def _make_neighbor_offsets() -> torch.Tensor:
    """Generate 26-neighbor offsets as (26, 3) int tensor."""
    offsets = []
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                offsets.append((di, dj, dk))
    return torch.tensor(offsets, dtype=torch.long)


class EdgeBuilder:
    """
    边构建器：为规划图构建多关系加权边。

    All hot loops are vectorized with PyTorch tensor ops and 3-D convolutions.
    """

    # Pre-computed once (CPU); moved to device in __init__
    _NEIGHBOR_OFFSETS_CPU = _make_neighbor_offsets()  # (26, 3)

    def __init__(self, lambda_dist: float = 0.5, lambda_conf: float = 0.3,
                 lambda_obs: float = 0.2, k_neighbors: int = 6,
                 device: str = 'cuda'):
        self.lambda_dist = lambda_dist
        self.lambda_conf = lambda_conf
        self.lambda_obs = lambda_obs
        self.k_neighbors = k_neighbors
        self.device = device
        # Keep on CPU at init time; move lazily in build() to avoid early CUDA
        # context creation issues on some driver/runtime combinations.
        self.NEIGHBOR_OFFSETS = self._NEIGHBOR_OFFSETS_CPU

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, node_types: torch.Tensor, voxel_centers: torch.Tensor,
              node_features: torch.Tensor, planning_indices: torch.Tensor,
              density_grid: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        构建多关系异构图边。

        Args:
            node_types: (Rx, Ry, Rz) 节点类型
            voxel_centers: (Rx, Ry, Rz, 3) 体素中心
            node_features: (N_plan, 16) 节点特征
            planning_indices: (N_plan, 3) 规划节点网格索引
            density_grid: (Rx, Ry, Rz) 密度场

        Returns:
            edges: dict of edge_type -> (edge_index_src, edge_index_dst, edge_weight)
        """
        N_plan = planning_indices.shape[0]
        print(f'[EdgeBuilder] Building edges for {N_plan} planning nodes...')

        # --- grid_to_plan mapping (vectorized scatter) ---
        Rx, Ry, Rz = node_types.shape
        grid_to_plan = torch.full((Rx, Ry, Rz), -1, dtype=torch.long, device=self.device)
        grid_to_plan[
            planning_indices[:, 0],
            planning_indices[:, 1],
            planning_indices[:, 2]
        ] = torch.arange(N_plan, device=self.device)

        # --- node type masks (in plan-index space) ---
        plan_node_types = node_types[
            planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
        ]
        free_plan_mask = (plan_node_types == NodeType.FREE)
        frontier_plan_mask = (plan_node_types == NodeType.FRONTIER)

        # --- confidence & obs_ratio ---
        confidences = node_features[:, 9]  # (N_plan,)
        obs_ratios = self._precompute_obs_ratio(node_types, planning_indices)

        # --- precompute per-voxel distance normalization ---
        max_dist = torch.norm(
            voxel_centers[0, 0, 0] - voxel_centers[1, 1, 1]
        ).item()

        # --- 26-neighbor edges per relation ---
        edges: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # 1. free <-> free
        ff_src, ff_dst, ff_w = self._build_relation(
            free_plan_mask, free_plan_mask, planning_indices,
            grid_to_plan, voxel_centers, confidences, obs_ratios, max_dist
        )
        edges['spatial_free_free'] = (ff_src, ff_dst, ff_w)

        # 2. free <-> frontier
        fr_src, fr_dst, fr_w = self._build_relation(
            free_plan_mask, frontier_plan_mask, planning_indices,
            grid_to_plan, voxel_centers, confidences, obs_ratios, max_dist
        )
        edges['spatial_free_frontier'] = (fr_src, fr_dst, fr_w)

        # 3. frontier <-> frontier
        ffr_src, ffr_dst, ffr_w = self._build_relation(
            frontier_plan_mask, frontier_plan_mask, planning_indices,
            grid_to_plan, voxel_centers, confidences, obs_ratios, max_dist
        )
        edges['frontier_frontier'] = (ffr_src, ffr_dst, ffr_w)

        # 4. k-nearest neighbor extra edges
        knn_src, knn_dst, knn_w = self._build_knn_edges(
            planning_indices, grid_to_plan, voxel_centers,
            confidences, obs_ratios, max_dist
        )

        total_edges = sum(e[0].shape[0] for e in edges.values())
        print(f'[EdgeBuilder] Total edges: {total_edges} '
              f'(ff={ff_src.shape[0]}, fr={fr_src.shape[0]}, '
              f'ffr={ffr_src.shape[0]}, knn={knn_src.shape[0]})')

        return edges

    # ------------------------------------------------------------------
    # Vectorized 26-neighbor relation builder
    # ------------------------------------------------------------------
    def _build_relation(self, src_mask: torch.Tensor, dst_mask: torch.Tensor,
                        planning_indices: torch.Tensor, grid_to_plan: torch.Tensor,
                        voxel_centers: torch.Tensor, confidences: torch.Tensor,
                        obs_ratios: torch.Tensor, max_dist: float):
        """Build 26-neighbor edges between two node type groups — fully vectorized."""
        src_indices = torch.where(src_mask)[0]  # plan-space indices of src nodes
        N_src = src_indices.shape[0]

        if N_src == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.zeros(0, device=self.device)

        Rx, Ry, Rz = grid_to_plan.shape
        if self.NEIGHBOR_OFFSETS.device != planning_indices.device:
            self.NEIGHBOR_OFFSETS = self.NEIGHBOR_OFFSETS.to(planning_indices.device)
        offsets = self.NEIGHBOR_OFFSETS  # (26, 3)

        # Grid positions of every src node: (N_src, 3)
        src_grid = planning_indices[src_indices]

        # All 26 neighbor grid positions: (N_src, 26, 3)
        all_nbr = src_grid.unsqueeze(1) + offsets.unsqueeze(0)

        # Bounds check — each component must be in [0, R*)
        in_bounds = (
            (all_nbr[..., 0] >= 0) & (all_nbr[..., 0] < Rx) &
            (all_nbr[..., 1] >= 0) & (all_nbr[..., 1] < Ry) &
            (all_nbr[..., 2] >= 0) & (all_nbr[..., 2] < Rz)
        )  # (N_src, 26)

        # Clamp so out-of-bounds coords don't cause indexing errors
        nbr_clamped = all_nbr.clone()
        nbr_clamped[..., 0].clamp_(0, Rx - 1)
        nbr_clamped[..., 1].clamp_(0, Ry - 1)
        nbr_clamped[..., 2].clamp_(0, Rz - 1)

        # Look up plan index for each neighbor: (N_src, 26)
        dst_plan = grid_to_plan[
            nbr_clamped[..., 0], nbr_clamped[..., 1], nbr_clamped[..., 2]
        ]

        # Valid edge: in bounds AND maps to a planning node AND dst_mask matches
        is_plan_node = (dst_plan >= 0)
        # dst_mask is (N_plan,) bool; gather the mask values for candidate dst nodes
        # (use 0 as safe index for invalid entries — masked away below)
        dst_plan_safe = dst_plan.clamp(min=0)
        is_dst_type = dst_mask[dst_plan_safe]  # (N_src, 26)

        valid = in_bounds & is_plan_node & is_dst_type  # (N_src, 26)

        # Flat indices of valid edges
        valid_src_local, valid_nbr_idx = torch.where(valid)  # both (E,)
        if valid_src_local.shape[0] == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.zeros(0, device=self.device)

        # Map back to plan-space indices
        edge_src = src_indices[valid_src_local]                # (E,) plan index of src
        edge_dst = dst_plan[valid_src_local, valid_nbr_idx]    # (E,) plan index of dst

        # ---------- compute weights vectorized ----------
        # Positions
        src_pos = voxel_centers[
            planning_indices[edge_src, 0],
            planning_indices[edge_src, 1],
            planning_indices[edge_src, 2]
        ]  # (E, 3)
        dst_pos = voxel_centers[
            planning_indices[edge_dst, 0],
            planning_indices[edge_dst, 1],
            planning_indices[edge_dst, 2]
        ]  # (E, 3)

        dist = torch.norm(src_pos - dst_pos, dim=-1) / max_dist  # (E,)
        mean_conf = (confidences[edge_src] + confidences[edge_dst]) / 2.0
        obs_density = obs_ratios[edge_src]

        w = (self.lambda_dist * dist +
             self.lambda_conf * (1.0 - mean_conf) +
             self.lambda_obs * obs_density)

        return edge_src, edge_dst, w

    # ------------------------------------------------------------------
    # Vectorized k-nearest-neighbor edges
    # ------------------------------------------------------------------
    def _build_knn_edges(self, planning_indices: torch.Tensor,
                         grid_to_plan: torch.Tensor,
                         voxel_centers: torch.Tensor,
                         confidences: torch.Tensor,
                         obs_ratios: torch.Tensor,
                         max_dist: float):
        """Build additional k-nearest neighbor edges — vectorized per chunk."""
        N = planning_indices.shape[0]

        if self.k_neighbors <= 0 or N == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.zeros(0, device=self.device)

        positions = voxel_centers[
            planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
        ]  # (N, 3)

        k_actual = min(self.k_neighbors + 1, N)
        all_src = []
        all_dst = []
        all_w = []

        # Chunked to avoid OOM on large point clouds
        chunk_size = 2048
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_pos = positions[start:end]  # (C, 3)

            # Pairwise distances: (C, N)
            dists = torch.cdist(chunk_pos, positions)

            # k+1 smallest (includes self)
            topk_dists, topk_idx = torch.topk(dists, k_actual, dim=1, largest=False)
            # topk_dists: (C, k_actual), topk_idx: (C, k_actual)

            C = end - start
            src_repeated = torch.arange(start, end, device=self.device).unsqueeze(1).expand(C, k_actual)

            # Mask out self-loops and too-far edges
            not_self = (topk_idx != src_repeated)
            not_far = (topk_dists < 3.0 * max_dist)
            valid = not_self & not_far  # (C, k_actual)

            if not valid.any():
                continue

            v_src_local, v_k = torch.where(valid)
            e_src = v_src_local + start                 # (E_chunk,)
            e_dst = topk_idx[v_src_local, v_k]          # (E_chunk,)
            e_dist = topk_dists[v_src_local, v_k]       # (E_chunk,)

            mean_conf = (confidences[e_src] + confidences[e_dst]) / 2.0
            obs_density = (obs_ratios[e_src] + obs_ratios[e_dst]) / 2.0

            w = (self.lambda_dist * (e_dist / max_dist) +
                 self.lambda_conf * (1.0 - mean_conf) +
                 self.lambda_obs * obs_density)

            all_src.append(e_src)
            all_dst.append(e_dst)
            all_w.append(w)

        if len(all_src) == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.zeros(0, device=self.device)

        return (
            torch.cat(all_src),
            torch.cat(all_dst),
            torch.cat(all_w),
        )

    # ------------------------------------------------------------------
    # Vectorized obstacle ratio via 3-D convolution
    # ------------------------------------------------------------------
    def _precompute_obs_ratio(self, node_types: torch.Tensor,
                              planning_indices: torch.Tensor) -> torch.Tensor:
        """
        Pre-compute 26-neighbor occupied ratio for each planning node.

        Uses a 3×3×3 convolution on the occupancy grid to sum occupied
        neighbors in a single pass instead of an N × 26 Python loop.
        """
        # Binary occupancy grid: 1 where OCCUPIED, 0 elsewhere
        occ_grid = (node_types == NodeType.OCCUPIED).float()  # (Rx, Ry, Rz)

        # 3x3x3 uniform kernel — counts occupied voxels in the 27-neighborhood
        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device)

        # Conv3d expects (B, C, D, H, W)
        occ_5d = occ_grid[None, None]  # (1, 1, Rx, Ry, Rz)
        ones_5d = torch.ones_like(occ_5d)

        occ_count = F.conv3d(occ_5d, kernel, padding=1).squeeze()    # (Rx, Ry, Rz)
        total_count = F.conv3d(ones_5d, kernel, padding=1).squeeze()  # (Rx, Ry, Rz)

        # Subtract the center voxel (we only want the 26 neighbors)
        occ_count = occ_count - occ_grid
        total_count = total_count - 1.0

        # Ratio (guard against division by zero at grid corners with <1 neighbor,
        # though 3D grids always have ≥7 neighbors even at corners)
        obs_ratio_grid = occ_count / total_count.clamp(min=1.0)

        # Index into the grid at planning node positions
        obs_ratios = obs_ratio_grid[
            planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
        ]

        return obs_ratios
