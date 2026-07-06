"""
GAT-A*: Graph Attention Network enhanced A* search.

Key innovation over standard A*:
  - Learned heuristic from GAT (vs hand-crafted Euclidean distance)
  - Uncertainty-aware: frontier nodes penalized
  - Operates on heterogeneous graph (vs binary grid)
  - Interpretable: A* backbone with learned heuristic enhancement

Algorithm:
  1. Map start/goal to nearest graph nodes
  2. Run A* with f(n) = g(n) + h_gat(n, goal)
  3. Extract path as sequence of node indices

Performance notes:
  - All graph data is moved to CPU numpy ONCE before A* search
  - No GPU↔CPU sync inside search loops
  - A* itself is inherently serial (heapq), but prep is vectorized
"""

import heapq
import time
import os
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict

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

from .gat_heuristic import GATHeuristicNet


class GATAStar:
    """
    GAT-enhanced A* search on heterogeneous planning graphs.
    """

    @staticmethod
    def infer_cost_scale(graph=None, metadata: Optional[dict] = None, default: float = 1.0) -> float:
        """Infer a heuristic-to-edge-cost scale from the graph edge statistics."""
        if graph is not None:
            try:
                ratios = []
                edge_types = list(getattr(graph, 'edge_types', []))
                for etype in edge_types:
                    rel = graph[etype]
                    ei = getattr(rel, 'edge_index', None)
                    ew = getattr(rel, 'edge_weight', None)
                    if ei is None or ew is None:
                        continue
                    src_type, _, dst_type = etype
                    src_store = graph[src_type] if hasattr(graph, '__getitem__') else None
                    dst_store = graph[dst_type] if hasattr(graph, '__getitem__') else None
                    src_pos = getattr(src_store, 'pos', None) if src_store is not None else None
                    dst_pos = getattr(dst_store, 'pos', None) if dst_store is not None else None
                    if src_pos is None or dst_pos is None:
                        continue
                    src_pos = src_pos.detach().cpu().numpy() if hasattr(src_pos, 'detach') else np.asarray(src_pos)
                    dst_pos = dst_pos.detach().cpu().numpy() if hasattr(dst_pos, 'detach') else np.asarray(dst_pos)
                    ei_np = ei.detach().cpu().numpy() if hasattr(ei, 'detach') else np.asarray(ei)
                    ew_np = ew.detach().cpu().numpy() if hasattr(ew, 'detach') else np.asarray(ew)
                    if ei_np.ndim != 2 or ei_np.shape[0] != 2 or ew_np.shape[0] != ei_np.shape[1]:
                        continue
                    valid = (
                        np.isfinite(ew_np) &
                        (ei_np[0] >= 0) & (ei_np[0] < len(src_pos)) &
                        (ei_np[1] >= 0) & (ei_np[1] < len(dst_pos))
                    )
                    if not np.any(valid):
                        continue
                    valid_idx = np.flatnonzero(valid)
                    src_xyz = src_pos[ei_np[0, valid_idx], :3]
                    dst_xyz = dst_pos[ei_np[1, valid_idx], :3]
                    dist = np.linalg.norm(src_xyz - dst_xyz, axis=1)
                    valid2 = np.isfinite(dist) & (dist > 1e-6)
                    if np.any(valid2):
                        ratio = ew_np[valid_idx[valid2]] / np.maximum(dist[valid2], 1e-6)
                        ratio = ratio[np.isfinite(ratio)]
                        if ratio.size > 0:
                            ratios.append(float(np.median(ratio)))
                if ratios:
                    scale = float(np.median(ratios))
                    if np.isfinite(scale) and scale > 0.0:
                        return scale
            except Exception:
                pass

        if not isinstance(metadata, dict):
            return float(default)
        voxel_size = metadata.get('voxel_size', None)
        try:
            voxel_size = float(voxel_size)
        except Exception:
            return float(default)
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            return float(default)
        # Edge weights are normalized by the voxel-diagonal scale in the graph
        # builder, so convert world meters to roughly the same cost units.
        return float(1.0 / max(np.sqrt(3.0) * voxel_size, 1e-6))

    def __init__(self, gat_net: Optional[GATHeuristicNet] = None,
                 use_uncertainty: bool = True, alpha: float = 1.0,
                 device: str = 'cuda', verbose: bool = False,
                 z_penalty_weight: float = 10.0,
                 z_reference_weight: float = 10.0,
                 line_bias_weight: float = 0.0,
                 length_penalty_weight: float = 0.0,
                 obstacle_soft_weight: float = 3.0,
                 obstacle_soft_sigma_scale: float = 2.0,
                 z_slope_penalty: float = 1.5,
                 z_curvature_penalty: float = 1.8,
                 z_goal_soft_weight: float = 1.4,
                 warn_snap_distance: float = 0.15,
                 max_snap_distance: float = 0.35,
                 cache_edge_obstacle_penalties: bool = True,
                 cache_occupied_tree: bool = True):
        self.gat_net = gat_net
        self.use_uncertainty = use_uncertainty
        self.alpha = alpha
        self.device = device
        self.verbose = verbose
        self.z_penalty_weight = float(z_penalty_weight)
        self.z_reference_weight = float(z_reference_weight)
        self.line_bias_weight = float(line_bias_weight)
        self.length_penalty_weight = float(length_penalty_weight)
        self.obstacle_soft_weight = float(obstacle_soft_weight)
        self.obstacle_soft_sigma_scale = float(obstacle_soft_sigma_scale)
        self.z_slope_penalty = float(z_slope_penalty)
        self.z_curvature_penalty = float(z_curvature_penalty)
        self.z_goal_soft_weight = float(z_goal_soft_weight)
        self.warn_snap_distance = float(warn_snap_distance)
        self.max_snap_distance = float(max_snap_distance)
        self.cache_edge_obstacle_penalties = bool(cache_edge_obstacle_penalties)
        self.cache_occupied_tree = bool(cache_occupied_tree)
        self.last_search_nodes = 0
        self.last_search_time = 0.0
        self._z_reference = None
        self._line_start = None
        self._line_goal = None
        self._parent_cache = {}
        self._adj_cache = {}
        self._tensor_cache = {}
        self._occupied_tree_cache = {}
        self._edge_penalty_cache = {}

    def _timed_out(self, t_start: float, max_wall_time_s: float) -> bool:
        return (time.time() - t_start) >= float(max_wall_time_s)

    # ------------------------------------------------------------------
    # Index maps (GPU vectorized, called once per graph)
    # ------------------------------------------------------------------
    def _build_index_maps(self, metadata: dict) -> Dict[str, torch.Tensor]:
        voxel_centers = metadata['voxel_centers']
        planning_indices = metadata['planning_indices']
        node_types = metadata['node_types']

        plan_positions = voxel_centers[
            planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
        ]
        plan_types = node_types[
            planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
        ]

        z_floor = metadata.get('z_floor', None)
        plan_types = plan_types.clone()
        if z_floor is not None:
            plan_types[plan_positions[:, 2] <= float(z_floor)] = -1

        free_plan_indices = torch.where(plan_types == 0)[0]
        frontier_plan_indices = torch.where(plan_types == 1)[0]

        plan_to_free_local = torch.full(
            (planning_indices.shape[0],), -1, dtype=torch.long, device=planning_indices.device
        )
        plan_to_frontier_local = torch.full_like(plan_to_free_local, -1)
        if free_plan_indices.numel() > 0:
            plan_to_free_local[free_plan_indices] = torch.arange(
                free_plan_indices.numel(), device=planning_indices.device, dtype=torch.long
            )
        if frontier_plan_indices.numel() > 0:
            plan_to_frontier_local[frontier_plan_indices] = torch.arange(
                frontier_plan_indices.numel(), device=planning_indices.device, dtype=torch.long
            )

        return {
            'plan_positions': plan_positions,
            'plan_types': plan_types,
            'planning_indices': planning_indices,
            'free_plan_indices': free_plan_indices,
            'frontier_plan_indices': frontier_plan_indices,
            'plan_to_free_local': plan_to_free_local,
            'plan_to_frontier_local': plan_to_frontier_local,
        }

    # ------------------------------------------------------------------
    # Main search entry
    # ------------------------------------------------------------------
    def search(self, graph: HeteroData, start_pos, goal_pos,
               metadata: Optional[dict] = None,
               max_expand: int = 10000,
               max_wall_time_s: float = 15.0) -> Dict:
        t_start = time.time()
        if metadata is None:
            metadata = getattr(graph, 'graph_meta', None)
        if metadata is None:
            return self._failed_result('graph metadata not provided', t_start)

        if self._timed_out(t_start, max_wall_time_s):
            return self._failed_result('GAT-A* wall time budget exceeded before setup', t_start)
        maps = self._build_index_maps(metadata)

        # Step 1: Map start/goal to nearest graph node (GPU vectorized)
        start_info = self._pos_to_node(start_pos, graph, metadata, maps)
        goal_info = self._pos_to_node(goal_pos, graph, metadata, maps)

        if start_info is None or goal_info is None:
            return self._failed_result('Start or goal not in free space', t_start)

        start_type, start_local, start_snap_dist = start_info
        goal_type, goal_local, goal_snap_dist = goal_info
        if start_snap_dist > self.max_snap_distance or goal_snap_dist > self.max_snap_distance:
            return self._failed_result(
                f'start/goal too far from free graph nodes: '
                f'd_start={start_snap_dist:.3f}m, d_goal={goal_snap_dist:.3f}m '
                f'(max={self.max_snap_distance:.3f}m)',
                t_start
            )
        if start_snap_dist > self.warn_snap_distance or goal_snap_dist > self.warn_snap_distance:
            print(
                f'[GAT-A*] Warning: large start/goal snapping distance: '
                f'd_start={start_snap_dist:.3f}m, d_goal={goal_snap_dist:.3f}m '
                f'(warn>{self.warn_snap_distance:.3f}m)'
            )

        if self._timed_out(t_start, max_wall_time_s):
            return self._failed_result('GAT-A* wall time budget exceeded after snapping', t_start)
        if self.verbose:
            print(f'[GAT-A*] Start: {start_type}[{start_local}] Goal: {goal_type}[{goal_local}]')

        self._z_reference = 0.5 * (float(start_pos[2]) + float(goal_pos[2]))
        self._line_start = np.asarray(start_pos, dtype=np.float32).reshape(3)
        self._line_goal = np.asarray(goal_pos, dtype=np.float32).reshape(3)

        # Step 2: Build adjacency — GPU batch → CPU once
        adj = self._build_unified_adj(graph)
        if self._timed_out(t_start, max_wall_time_s):
            return self._failed_result('GAT-A* wall time budget exceeded while building adjacency', t_start)

        raw_occupied_positions = getattr(graph, 'occupied_positions', None)
        occupied_positions = raw_occupied_positions
        robot_radius = 0.02
        if isinstance(metadata, dict):
            robot_radius = float(metadata.get('robot_radius', robot_radius))
        occupied_tree = None
        if occupied_positions is not None:
            try:
                occ_np = np.asarray(occupied_positions, dtype=np.float32)
                if occ_np.ndim == 2 and occ_np.shape[0] > 0:
                    occupied_tree = self._get_occupied_tree(raw_occupied_positions)
                    occupied_positions = occ_np
            except Exception:
                occupied_tree = None
        edge_cache_key = (
            id(graph),
            id(raw_occupied_positions) if raw_occupied_positions is not None else 0,
            float(robot_radius),
            float(self.obstacle_soft_weight),
            float(self.obstacle_soft_sigma_scale),
        )

        # Step 3: Compute heuristics — GPU batch → CPU once
        h_values = self._compute_heuristics(
            graph, metadata, goal_type, goal_local, maps,
            t_start=t_start,
            max_wall_time_s=max_wall_time_s,
            heuristic_cost_scale=None,
        )
        if self._timed_out(t_start, max_wall_time_s):
            return self._failed_result('GAT-A* wall time budget exceeded while computing heuristics', t_start)

        # Step 4: A* search (pure CPU Python, no GPU sync)
        free_plan_idx = maps['free_plan_indices']
        front_plan_idx = maps['frontier_plan_indices']
        plan_positions_np = maps['plan_positions'].detach().cpu().numpy()
        free_positions_np = plan_positions_np[free_plan_idx.detach().cpu().numpy()] \
            if free_plan_idx.numel() > 0 else np.zeros((0, 3), dtype=np.float32)
        frontier_positions_np = plan_positions_np[front_plan_idx.detach().cpu().numpy()] \
            if front_plan_idx.numel() > 0 else np.zeros((0, 3), dtype=np.float32)
        node_positions = {
            'free': free_positions_np,
            'frontier': frontier_positions_np,
        }

        # Debug: check start/goal keys exist in adj
        start_key = (start_type, start_local)
        goal_key = (goal_type, goal_local)
        if self.verbose:
            print(f'  [_astar] start_key={start_key}, in adj: {start_key in adj}')
            print(f'  [_astar] goal_key={goal_key}, in adj: {goal_key in adj}')
            print(f'  [_astar] h(start)={h_values.get(start_key, 0):.3f}, h(goal)={h_values.get(goal_key, 0):.3f}')
            occ_np_debug = None
            if occupied_positions is not None:
                occ_np_debug = np.asarray(occupied_positions, dtype=np.float32)
                if occ_np_debug.ndim == 2 and occ_np_debug.shape[0] > 0:
                    if occ_np_debug.shape[1] > 3:
                        occ_np_debug = occ_np_debug[:, :3]
                    print(f'  [_astar] occupied_positions: {occ_np_debug.shape}, robot_radius={robot_radius}')
            if start_key in adj:
                print(f'  [_astar] start neighbors: {adj[start_key][:3]}')
            if goal_key in adj:
                print(f'  [_astar] goal neighbors: {adj[goal_key][:3]}')

        path, nodes_expanded = self._astar_search(
            start_type, start_local, goal_type, goal_local,
            adj, h_values, node_positions,
            occupied_positions=occupied_positions,
            occupied_tree=occupied_tree,
            robot_radius=robot_radius,
            edge_cache_key=edge_cache_key,
            max_expand=max_expand,
            max_wall_time_s=max_wall_time_s,
            t_start=t_start,
        )

        t_end = time.time()
        self.last_search_time = t_end - t_start

        if nodes_expanded < 0:
            return self._failed_result('A* expanded too many nodes (timeout)', t_start)

        if path is None:
            return self._failed_result('No path found', t_start)

        # Step 5: Convert to world coordinates (batch GPU → CPU)
        path_nodes, path_coords, path_types = self._path_to_world(path, graph, metadata, maps)
        self.last_search_nodes = nodes_expanded

        if self.verbose:
            print(f'[GAT-A*] Found path: {len(path_nodes)} nodes, '
                  f'{path_coords.shape[0]} waypoints, '
                  f'searched {self.last_search_nodes} nodes in {self.last_search_time:.3f}s')

        return {
            'path_nodes': path_nodes,
            'path_coords': path_coords,
            'path_types': path_types,
            'search_nodes': self.last_search_nodes,
            'nodes_expanded': self.last_search_nodes,
            'search_time': self.last_search_time,
            'success': True,
            'found': True,
            'start_snap_distance': float(start_snap_dist),
            'goal_snap_distance': float(goal_snap_dist),
        }

    # ------------------------------------------------------------------
    # Nearest node lookup (GPU vectorized)
    # ------------------------------------------------------------------
    def _pos_to_node(self, pos, graph, metadata, maps):
        pos = torch.as_tensor(pos, dtype=torch.float32, device=maps['plan_positions'].device)
        plan_positions = maps['plan_positions']
        plan_types = maps['plan_types']

        dists = torch.norm(plan_positions - pos.unsqueeze(0), dim=1)
        valid = plan_types != -1
        if not valid.any():
            return None

        dists = dists.clone()
        dists[~valid] = float('inf')
        nearest_idx = int(torch.argmin(dists).item())
        nearest_dist = float(dists[nearest_idx].item())
        ntype = int(plan_types[nearest_idx].item())

        if ntype == 0:
            local_idx = int(maps['plan_to_free_local'][nearest_idx].item())
            return 'free', local_idx, nearest_dist
        if ntype == 1:
            local_idx = int(maps['plan_to_frontier_local'][nearest_idx].item())
            return 'frontier', local_idx, nearest_dist
        return None

    # ------------------------------------------------------------------
    # Adjacency: batch GPU→CPU, zero per-edge .item() calls
    # ------------------------------------------------------------------
    def _build_unified_adj(self, graph):
        """Build adjacency dict from graph. All GPU data pulled to CPU in bulk."""
        cache_key = id(graph)
        cached = self._adj_cache.get(cache_key)
        if cached is not None:
            return cached

        n_free = graph['free'].x.shape[0]
        n_frontier = graph['frontier'].x.shape[0]

        # Pre-allocate adjacency lists
        adj = {}
        for i in range(n_free):
            adj[('free', i)] = []
        for i in range(n_frontier):
            adj[('frontier', i)] = []

        free_pos = graph['free'].pos.detach().cpu().numpy() if n_free > 0 else np.zeros((0, 3), dtype=np.float32)
        frontier_pos = graph['frontier'].pos.detach().cpu().numpy() if n_frontier > 0 else np.zeros((0, 3), dtype=np.float32)
        pos_by_type = {
            'free': free_pos,
            'frontier': frontier_pos,
        }

        # Batch-transfer each edge type to CPU numpy (ONE sync per edge type)
        for etype in graph.edge_types:
            src_type, rel, dst_type = etype
            ei = graph[etype].edge_index
            ew = graph[etype].edge_weight

            # ONE GPU→CPU transfer per edge type
            ei_np = ei.detach().cpu().numpy()  # (2, E)
            ew_np = ew.detach().cpu().numpy()  # (E,)

            # Pure Python/numpy loop — no GPU sync
            for e in range(ei_np.shape[1]):
                s, d, w = int(ei_np[0, e]), int(ei_np[1, e]), float(ew_np[e])
                src_pos = pos_by_type[src_type][s, :3]
                dst_pos = pos_by_type[dst_type][d, :3]
                edge_dist = float(np.linalg.norm(dst_pos - src_pos))
                z_delta = float(abs(dst_pos[2] - src_pos[2]))
                adj.setdefault((src_type, s), []).append((dst_type, d, w, edge_dist, z_delta))

        self._adj_cache[cache_key] = adj
        return adj

    def _get_occupied_tree(self, occupied_positions: np.ndarray):
        """Cache a cKDTree for repeated collision queries on a static scene."""
        if occupied_positions is None:
            return None

        occ_np = np.asarray(occupied_positions, dtype=np.float32)
        if occ_np.ndim != 2 or occ_np.shape[0] == 0:
            return None

        cache_key = id(occupied_positions)
        if self.cache_occupied_tree:
            cached = self._occupied_tree_cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(occ_np[:, :3])
        except Exception:
            return None

        if self.cache_occupied_tree:
            self._occupied_tree_cache[cache_key] = tree
        return tree

    def _get_static_edge_penalty(self, edge_cache_key, edge_key, p0, p1,
                                 occupied_positions, robot_radius, occupied_tree):
        """Cache the obstacle penalty for a fixed edge across repeated trials."""
        if not self.cache_edge_obstacle_penalties:
            return self._segment_soft_obstacle_penalty(
                p0, p1, occupied_positions, robot_radius,
                occupied_tree=occupied_tree,
            )

        cache = self._edge_penalty_cache.setdefault(edge_cache_key, {})
        cached = cache.get(edge_key)
        if cached is not None:
            return cached

        penalty = self._segment_soft_obstacle_penalty(
            p0, p1, occupied_positions, robot_radius,
            occupied_tree=occupied_tree,
        )
        cache[edge_key] = float(penalty)
        return float(penalty)

    # ------------------------------------------------------------------
    # Heuristics: GPU vectorized, then bulk CPU transfer
    # ------------------------------------------------------------------
    def _compute_heuristics(self, graph, metadata, goal_type, goal_local, maps,
                            t_start: Optional[float] = None,
                            max_wall_time_s: float = 15.0,
                            heuristic_cost_scale: Optional[float] = None):
        """Compute h-values for all nodes. GPU vectorized, one CPU transfer.

        If self.gat_net is available, uses the learned GAT heuristic.
        Otherwise falls back to hand-crafted Euclidean + uncertainty formula.
        """
        h_values = {}

        if t_start is not None and self._timed_out(t_start, max_wall_time_s):
            return h_values

        n_free = graph['free'].x.shape[0]
        n_frontier = graph['frontier'].x.shape[0]
        n_total = n_free + n_frontier
        cache_key = (id(graph), self.use_uncertainty)
        cached = self._tensor_cache.get(cache_key)

        # Goal position (single GPU→CPU)
        goal_plan_idx = (
            maps['free_plan_indices'][goal_local]
            if goal_type == 'free'
            else maps['frontier_plan_indices'][goal_local]
        )
        goal_pos = maps['plan_positions'][goal_plan_idx]  # (3,) GPU tensor
        # Keep goal coordinates in world space so decode_goal matches the
        # training script and the node features stored in graph.x.
        goal_pos_feat = goal_pos
        if heuristic_cost_scale is None:
            cost_scale = float(self.infer_cost_scale(graph=graph, metadata=metadata))
        else:
            cost_scale = float(heuristic_cost_scale)

        if cached is None:
            # Cache graph tensors for repeated queries on the same scene.
            if n_free > 0 and n_frontier > 0:
                x_all = torch.cat([graph['free'].x, graph['frontier'].x], dim=0)
                pos_all = torch.cat([graph['free'].pos, graph['frontier'].pos], dim=0)
            elif n_free > 0:
                x_all = graph['free'].x
                pos_all = graph['free'].pos
            else:
                x_all = graph['frontier'].x
                pos_all = graph['frontier'].pos

            edge_list = []
            weight_list = []
            for etype in graph.edge_types:
                src_type, rel, dst_type = etype
                ei = graph[etype].edge_index
                ew = graph[etype].edge_weight

                offset_src = 0 if src_type == 'free' else n_free
                offset_dst = 0 if dst_type == 'free' else n_free

                ei_remapped = ei.clone()
                ei_remapped[0] = ei[0] + offset_src
                ei_remapped[1] = ei[1] + offset_dst

                edge_list.append(ei_remapped)
                weight_list.append(ew)

            if edge_list:
                edge_index = torch.cat(edge_list, dim=1)
                edge_weight = torch.cat(weight_list, dim=0)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=x_all.device)
                edge_weight = torch.zeros((0,), device=x_all.device)

            if not self.use_uncertainty:
                x_all = self._neutralize_uncertainty_features(x_all)

            cached = {
                'x_all': x_all,
                'pos_all': pos_all,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
            }
            self._tensor_cache[cache_key] = cached

        # === Try GNN-based heuristic ===
        if self.gat_net is not None and n_total > 0:
            try:
                if t_start is not None and self._timed_out(t_start, max_wall_time_s):
                    return h_values
                x_all = cached['x_all']
                pos_all = cached.get('pos_all', x_all[:, :3])
                edge_index = cached['edge_index']
                edge_weight = cached['edge_weight']

                # Move to same device as gat_net
                net_device = next(self.gat_net.parameters()).device
                x_all = x_all.to(net_device)
                pos_all = pos_all.to(net_device)
                edge_index = edge_index.to(net_device)
                edge_weight = edge_weight.to(net_device)
                goal_pos_net = goal_pos_feat.to(net_device)

                # Extract uncertainty from features (index 9 = confidence)
                if self.use_uncertainty and x_all.shape[1] > 9:
                    uncertainty = x_all[:, 9].clone()
                else:
                    uncertainty = None

                # Cache goal-independent graph embedding once per scene.
                emb_cache = cached.setdefault('graph_emb_by_device', {})
                device_key = str(net_device)
                h_graph = emb_cache.get(device_key)
                self.gat_net.eval()
                with torch.no_grad():
                    if h_graph is None:
                        h_graph = self.gat_net.encode_graph(
                            x=x_all,
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                        )
                        emb_cache[device_key] = h_graph.detach()
                    h_all = self.gat_net.decode_goal(
                        h_graph,
                        x=x_all,
                        goal_pos=goal_pos_net,
                        uncertainty=uncertainty,
                        pos_coords=pos_all,
                        cost_scale=cost_scale,
                    )
                    h_all = h_all * float(self.alpha)

                # Move back to CPU for A* (ensure non-negative)
                h_all = h_all.detach().cpu().numpy()
                h_all = np.maximum(h_all, 0.0)

                for i in range(n_free):
                    h_values[('free', i)] = float(h_all[i])
                for i in range(n_frontier):
                    h_values[('frontier', i)] = float(h_all[n_free + i])

                if self.verbose:
                    print(f'[GAT-A*] GNN heuristic enabled, h range: [{h_all.min():.4f}, {h_all.max():.4f}]')

                return h_values

            except Exception as e:
                if self.verbose:
                    print(f'[GAT-A*] GNN forward failed: {e}, falling back to hand-crafted')

        # === Fallback: hand-crafted Euclidean + uncertainty formula ===
        if self.verbose and self.gat_net is None:
            print('[GAT-A*] No GNN available, using hand-crafted heuristic')

        # --- Free nodes ---
        line_scale = 1.0
        if self._line_start is not None and self._line_goal is not None:
            line_scale = max(1e-3, float(np.linalg.norm(self._line_goal - self._line_start)))

        if n_free > 0:
            free_plan_idx = maps['free_plan_indices']
            free_positions = maps['plan_positions'][free_plan_idx]
            free_h = torch.norm(free_positions - goal_pos.unsqueeze(0), dim=1) * cost_scale
            if self.line_bias_weight > 0.0 and self._line_start is not None and self._line_goal is not None:
                ab = self._line_goal - self._line_start
                denom = float(np.dot(ab, ab))
                if denom > 1e-12:
                    proj_t = torch.clamp(
                        torch.sum((free_positions - torch.as_tensor(self._line_start, device=free_positions.device)) *
                                  torch.as_tensor(ab, device=free_positions.device), dim=1) / denom,
                        0.0, 1.0,
                    )
                    proj = torch.as_tensor(self._line_start, device=free_positions.device) + proj_t.unsqueeze(1) * torch.as_tensor(ab, device=free_positions.device)
                    free_h = free_h + self.line_bias_weight * torch.norm(free_positions - proj, dim=1) / line_scale

            if self.use_uncertainty:
                conf = graph['free'].x[:, 9]
                frontier_flag = graph['free'].x[:, 11]
                free_h = free_h + self.alpha * (1.0 - conf) + 0.5 * frontier_flag

            free_h_np = free_h.detach().cpu().numpy()
            for i in range(n_free):
                h_values[('free', i)] = max(0.0, float(free_h_np[i]))

        # --- Frontier nodes ---
        if n_frontier > 0:
            front_plan_idx = maps['frontier_plan_indices']
            front_positions = maps['plan_positions'][front_plan_idx]
            front_h = torch.norm(front_positions - goal_pos.unsqueeze(0), dim=1) * cost_scale
            if self.line_bias_weight > 0.0 and self._line_start is not None and self._line_goal is not None:
                ab = self._line_goal - self._line_start
                denom = float(np.dot(ab, ab))
                if denom > 1e-12:
                    proj_t = torch.clamp(
                        torch.sum((front_positions - torch.as_tensor(self._line_start, device=front_positions.device)) *
                                  torch.as_tensor(ab, device=front_positions.device), dim=1) / denom,
                        0.0, 1.0,
                    )
                    proj = torch.as_tensor(self._line_start, device=front_positions.device) + proj_t.unsqueeze(1) * torch.as_tensor(ab, device=front_positions.device)
                    front_h = front_h + self.line_bias_weight * torch.norm(front_positions - proj, dim=1) / line_scale

            if self.use_uncertainty:
                conf = graph['frontier'].x[:, 9]
                front_h = front_h + self.alpha * (1.0 - conf) + 0.5 * 1.0

            front_h_np = front_h.detach().cpu().numpy()
            for i in range(n_frontier):
                h_values[('frontier', i)] = max(0.0, float(front_h_np[i]))

        return h_values

    def _neutralize_uncertainty_features(self, x_all: torch.Tensor) -> torch.Tensor:
        """Replace uncertainty-related channels with neutral values."""
        x = x_all.clone()
        if x.ndim == 2 and x.shape[1] > 9:
            x[:, 9] = 1.0
        if x.ndim == 2 and x.shape[1] > 10:
            x[:, 10] = 0.0
        if x.ndim == 2 and x.shape[1] > 11:
            x[:, 11] = 0.0
        return x

    # ------------------------------------------------------------------
    # A* core (pure CPU Python — inherently serial)
    # ------------------------------------------------------------------
    def _astar_search(self, start_type, start_local, goal_type, goal_local,
                      adj, h_values, node_positions,
                      occupied_positions=None,
                      occupied_tree=None,
                      robot_radius: float = 0.02,
                      edge_cache_key=None,
                      collision_sample_step: float = 0.02,
                      collision_safety_margin: float = 1.05,
                      max_expand=10000, max_wall_time_s: float = 15.0,
                      t_start: Optional[float] = None):
        start_key = (start_type, start_local)
        goal_key = (goal_type, goal_local)

        open_set = [(max(0.0, h_values.get(start_key, 0.0)), 0.0, start_key, None, [start_key])]
        if not open_set:
            return None, 0
        closed_set = set()
        g_values = {start_key: 0.0}
        nodes_expanded = 0

        while open_set:
            if nodes_expanded > max_expand:
                return None, -nodes_expanded  # Negative indicates timeout
            if t_start is not None and (time.time() - t_start) >= max_wall_time_s:
                return None, -nodes_expanded

            f, g, current, parent, path = heapq.heappop(open_set)

            if current == goal_key:
                return path, nodes_expanded

            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            if self.verbose and nodes_expanded <= 5:
                print(f'  [_astar] expanded={nodes_expanded}, current={current}, g={g:.3f}, f={f:.3f}, open_set={len(open_set)}, adj[current]={len(adj.get(current, []))} neighbors')
                if nodes_expanded == 1 and self.verbose:
                    sample_neighbors = list(adj.get(current, []))[:3]
                    print(f'    -> sample neighbors: {sample_neighbors}')

            collision_rejects = 0
            for neighbor in adj.get(current, []):
                neigh_key = (neighbor[0], neighbor[1])
                if neigh_key in closed_set:
                    continue

                edge_cost = float(neighbor[2])
                curr_pos = node_positions[current[0]][current[1]]
                neigh_pos = node_positions[neigh_key[0]][neigh_key[1]]
                if len(neighbor) >= 5:
                    edge_dist = float(neighbor[3])
                    z_delta = float(neighbor[4])
                else:
                    edge_dist = float(np.linalg.norm(neigh_pos - curr_pos))
                    z_delta = float(abs(neigh_pos[2] - curr_pos[2]))
                z_ref_pen = 0.0
                if self._z_reference is not None:
                    z_ref_pen = abs(float(neigh_pos[2]) - float(self._z_reference))

                obstacle_pen = 0.0
                if occupied_positions is not None and self.obstacle_soft_weight > 0.0:
                    edge_key = (current[0], int(current[1]), neigh_key[0], int(neigh_key[1]))
                    obstacle_pen = self._get_static_edge_penalty(
                        edge_cache_key=edge_cache_key,
                        edge_key=edge_key,
                        p0=curr_pos,
                        p1=neigh_pos,
                        occupied_positions=occupied_positions,
                        robot_radius=robot_radius,
                        occupied_tree=occupied_tree,
                    )

                slope_pen = 0.0
                curvature_pen = 0.0
                if parent is not None and (self.z_slope_penalty > 0.0 or self.z_curvature_penalty > 0.0):
                    prev_pos = node_positions[parent[0]][parent[1]]
                    slope_pen, curvature_pen = self._z_soft_penalty(prev_pos, curr_pos, neigh_pos)

                z_goal_pen = 0.0
                if self.z_goal_soft_weight > 0.0 and self._z_reference is not None:
                    z_goal_pen = self.z_goal_soft_weight * abs(float(neigh_pos[2]) - float(self._z_reference))

                line_pen = 0.0
                if self._line_start is not None and self._line_goal is not None:
                    ab = self._line_goal - self._line_start
                    denom = float(np.dot(ab, ab))
                    if denom > 1e-12:
                        t = float(np.dot(neigh_pos - self._line_start, ab) / denom)
                        t = max(0.0, min(1.0, t))
                        proj = self._line_start + t * ab
                        line_pen = float(np.linalg.norm(neigh_pos - proj))
                tentative_g = (
                    g + edge_cost
                    + self.length_penalty_weight * edge_dist
                    + self.z_penalty_weight * z_delta
                    + self.z_reference_weight * z_ref_pen
                    + obstacle_pen
                    + slope_pen
                    + curvature_pen
                    + z_goal_pen
                    + self.line_bias_weight * line_pen
                )

                if tentative_g < g_values.get(neigh_key, float('inf')):
                    g_values[neigh_key] = tentative_g
                    h = max(0.0, h_values.get(neigh_key, 0.0))
                    f_new = tentative_g + h
                    heapq.heappush(open_set, (f_new, tentative_g, neigh_key, current, path + [neigh_key]))

            if self.verbose and nodes_expanded <= 5:
                total_neighbors = len(adj.get(current, []))
                print(f'  [_astar] expanded node had {total_neighbors} neighbors, {collision_rejects} collision rejected')

        self.last_search_nodes = nodes_expanded
        if self.verbose:
            print(f'  [_astar] Search ended: expanded={nodes_expanded}, open_set={len(open_set)}, goal never reached')
        return None, nodes_expanded

    # ------------------------------------------------------------------
    # GPU-accelerated A* search (batch neighbor expansion on GPU)
    # ------------------------------------------------------------------
    def _astar_search_gpu(self, start_type, start_local, goal_type, goal_local,
                          adj, h_values, node_positions,
                          occupied_positions=None,
                          occupied_tree=None,
                          robot_radius: float = 0.02,
                          edge_cache_key=None,
                          collision_sample_step: float = 0.02,
                          collision_safety_margin: float = 1.05,
                          batch_size=64):
        """GPU-accelerated A* with batch neighbor expansion.

        Instead of expanding one node at a time (serial), we:
        1. Pop top-K nodes from open set
        2. On GPU: parallel relax all their outgoing edges
        3. Update open set with improved nodes

        This significantly speeds up training where we call A* many times.
        """
        import torch
        import numpy as np

        start_key = (start_type, start_local)
        goal_key = (goal_type, goal_local)

        # Build edge index on GPU for vectorized operations
        # adj is dict: node_key -> [(ntype, local_idx, weight, edge_dist, z_delta), ...]
        all_src, all_dst, all_weight = [], [], []
        for node_key, neighbors in adj.items():
            for neighbor in neighbors:
                ntype, local_idx = neighbor[0], neighbor[1]
                weight = float(neighbor[2])
                all_src.append(node_key)
                all_dst.append((ntype, local_idx))
                all_weight.append(weight)

        if not all_src:
            return None, 0

        # Convert to GPU tensors
        src_keys = all_src
        dst_keys = all_dst
        weights = torch.tensor(all_weight, dtype=torch.float32, device=self.device)

        # Create node key to index mapping
        unique_keys = list(set(src_keys + dst_keys))
        key_to_idx = {k: i for i, k in enumerate(unique_keys)}
        n_nodes = len(unique_keys)

        src_idx = torch.tensor([key_to_idx[k] for k in src_keys], dtype=torch.long, device=self.device)
        dst_idx = torch.tensor([key_to_idx[k] for k in dst_keys], dtype=torch.long, device=self.device)

        # Initialize g values
        g_gpu = torch.full((n_nodes,), float('inf'), device=self.device)
        f_gpu = torch.full((n_nodes,), float('inf'), device=self.device)

        start_idx = key_to_idx.get(start_key)
        goal_idx = key_to_idx.get(goal_key)
        if start_idx is None or goal_idx is None:
            return None, 0

        g_gpu[start_idx] = 0.0

        # Heuristics on GPU
        h_gpu = torch.zeros((n_nodes,), device=self.device)
        for i, k in enumerate(unique_keys):
            h_gpu[i] = h_values.get(k, 0.0)
        f_gpu = g_gpu + h_gpu

        # Build reverse adjacency for fast lookup
        rev_adj = [[] for _ in range(n_nodes)]
        for s, d, w in zip(src_idx.tolist(), dst_idx.tolist(), weights.tolist()):
            rev_adj[d].append((s, w))

        # Open set on CPU (heapq)
        open_set = [(float(f_gpu[start_idx].item()), float(g_gpu[start_idx].item()), start_key, None, [start_key])]
        closed_set = set()
        nodes_explored = 0
        max_iterations = 5000

        # Node positions for edge distance computation (CPU)
        pos_dict = {}
        for ntype in ['free', 'frontier']:
            for i, pos in enumerate(node_positions.get(ntype, np.zeros((0, 3)))):
                pos_dict[(ntype, i)] = pos

        for _ in range(max_iterations):
            if not open_set:
                break

            # Batch pop top-K nodes
            batch = []
            for _ in range(min(batch_size, len(open_set))):
                if open_set:
                    batch.append(heapq.heappop(open_set))

            if not batch:
                break

            # Check if goal in batch
            for f_val, g_val, current, parent, path in batch:
                if current == goal_key:
                    self.last_search_nodes = nodes_explored
                    return path, nodes_explored

            # Process batch
            batch_indices = []
            for f_val, g_val, current, parent, path in batch:
                if current in closed_set:
                    continue
                closed_set.add(current)
                nodes_explored += 1
                batch_indices.append(key_to_idx[current])

                # Expand neighbors (vectorized where possible)
                for neighbor in adj.get(current, []):
                    ntype, local_idx = neighbor[0], neighbor[1]
                    edge_w = float(neighbor[2])
                    neigh_key = (ntype, local_idx)
                    if neigh_key in closed_set:
                        continue

                    edge_dist = float(neighbor[3]) if len(neighbor) >= 4 else float(np.linalg.norm(
                        pos_dict.get(neigh_key, np.zeros(3)) - pos_dict.get(current, np.zeros(3))
                    ))
                    z_delta = float(neighbor[4]) if len(neighbor) >= 5 else float(abs(
                        pos_dict.get(neigh_key, np.zeros(3))[2] - pos_dict.get(current, np.zeros(3))[2]
                    ))

                    # Compute edge distance penalty
                    curr_pos = pos_dict.get(current, np.zeros(3))
                    neigh_pos = pos_dict.get(neigh_key, np.zeros(3))
                    obstacle_pen = 0.0
                    if occupied_positions is not None and self.obstacle_soft_weight > 0.0:
                        edge_key = (current[0], int(current[1]), ntype, int(local_idx))
                        obstacle_pen = self._get_static_edge_penalty(
                            edge_cache_key=edge_cache_key,
                            edge_key=edge_key,
                            p0=curr_pos,
                            p1=neigh_pos,
                            occupied_positions=occupied_positions,
                            robot_radius=robot_radius,
                            occupied_tree=occupied_tree,
                        )

                    slope_pen = 0.0
                    curvature_pen = 0.0
                    if parent is not None and (self.z_slope_penalty > 0.0 or self.z_curvature_penalty > 0.0):
                        prev_pos = pos_dict.get(parent, np.zeros(3))
                        slope_pen, curvature_pen = self._z_soft_penalty(prev_pos, curr_pos, neigh_pos)

                    z_goal_pen = 0.0
                    if self.z_goal_soft_weight > 0.0 and self._z_reference is not None:
                        z_goal_pen = self.z_goal_soft_weight * abs(float(neigh_pos[2]) - float(self._z_reference))

                    tentative_g = g_val + edge_w + self.length_penalty_weight * edge_dist + obstacle_pen + slope_pen + curvature_pen + z_goal_pen

                    if tentative_g < g_gpu[key_to_idx[neigh_key]].item():
                        g_gpu[key_to_idx[neigh_key]] = tentative_g
                        h = h_values.get(neigh_key, 0.0)
                        f_new = tentative_g + h
                        f_gpu[key_to_idx[neigh_key]] = f_new
                        new_path = path + [neigh_key]
                        heapq.heappush(open_set, (f_new, tentative_g, neigh_key, current, new_path))

            # Check goal
            if goal_idx in batch_indices and goal_idx not in closed_set:
                # Reconstruct path
                for f_val, g_val, current, parent, path in batch:
                    if current == goal_key:
                        self.last_search_nodes = nodes_explored
                        return path, nodes_explored

        self.last_search_nodes = nodes_explored
        return None, nodes_explored

    # ------------------------------------------------------------------
    # Path → world coordinates (batch GPU→CPU)
    # ------------------------------------------------------------------
    def _path_to_world(self, path, graph, metadata, maps):
        if not path:
            return [], torch.zeros(0, 3), []

        # Gather plan indices for all path nodes at once
        plan_indices = []
        types_list = []
        path_nodes = []

        for node_key in path:
            node_type, local_idx = node_key
            if node_type == 'free':
                plan_idx = maps['free_plan_indices'][local_idx]
            else:
                plan_idx = maps['frontier_plan_indices'][local_idx]
            plan_indices.append(plan_idx)
            types_list.append(node_type)
            path_nodes.append(local_idx)

        # Batch index → batch GPU→CPU transfer (ONE sync)
        plan_indices_t = torch.stack(plan_indices)  # (K,)
        path_coords = maps['plan_positions'][plan_indices_t].detach().cpu()  # (K, 3)

        # Remove consecutive duplicates
        path_coords = self._smooth_path(path_coords)

        return path_nodes, path_coords, types_list

    def _smooth_path(self, coords):
        if coords.shape[0] <= 1:
            return coords
        keep = [0]
        for i in range(1, coords.shape[0]):
            if torch.norm(coords[i] - coords[keep[-1]]) > 1e-6:
                keep.append(i)
        return coords[keep]

    def _segment_is_collision_free(self, p0, p1, occupied_positions, robot_radius,
                                   sample_step: float = 0.02,
                                   safety_margin: float = 1.05,
                                   gsplat_collision_set=None) -> bool:
        if occupied_positions is not None:
            try:
                occ_np = np.asarray(occupied_positions, dtype=np.float32)
                if occ_np.ndim == 2 and occ_np.shape[0] > 0:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(occ_np[:, :3])
                    seg_len = float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32)))
                    n_check = max(8, int(np.ceil(seg_len / max(robot_radius * 0.5, 1e-4))))
                    n_check = min(n_check, 64)
                    t_vals = np.linspace(0.0, 1.0, n_check, dtype=np.float32)
                    pts = np.asarray(p0, dtype=np.float32)[None, :] * (1.0 - t_vals[:, None]) + np.asarray(p1, dtype=np.float32)[None, :] * t_vals[:, None]
                    dists, _ = tree.query(pts, k=1)
                    return bool(np.all(dists >= robot_radius * safety_margin))
            except Exception:
                return False
        return True

    def _segment_soft_obstacle_penalty(self, p0, p1, occupied_positions, robot_radius,
                                       sample_step: float = 0.02,
                                       safety_margin: float = 1.05,
                                       occupied_tree=None) -> float:
        """Soft penalty for segments that pass close to occupied points."""
        if occupied_positions is None:
            return 0.0
        occ_np = np.asarray(occupied_positions, dtype=np.float32)
        if occ_np.ndim != 2 or occ_np.shape[0] == 0:
            return 0.0

        p0 = np.asarray(p0, dtype=np.float32).reshape(3)
        p1 = np.asarray(p1, dtype=np.float32).reshape(3)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            return 0.0

        effective_step = max(min(float(sample_step), max(float(robot_radius) * 0.5, 1e-4)), 1e-4)
        n_samples = int(max(4, np.ceil(seg_len / effective_step) + 1))
        n_samples = min(n_samples, 32)
        ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
        pts = p0[None, :] + ts[:, None] * seg[None, :]

        dists = None
        if occupied_tree is not None:
            try:
                dists, _ = occupied_tree.query(pts, k=1)
                dists = np.asarray(dists, dtype=np.float32).reshape(-1)
            except Exception:
                dists = None

        if dists is None:
            chunk = 256
            min_dists = []
            for i in range(0, len(pts), chunk):
                sub = pts[i:i + chunk]
                d = np.linalg.norm(sub[:, None, :] - occ_np[None, :, :], axis=-1)
                min_dists.append(np.min(d, axis=1))
            dists = np.concatenate(min_dists, axis=0).astype(np.float32)

        sigma = max(float(robot_radius) * float(self.obstacle_soft_sigma_scale), 1e-4)
        penalty = np.exp(-np.square(dists / sigma))
        return float(self.obstacle_soft_weight * np.mean(penalty) * (0.5 + 0.5 * min(1.0, seg_len / sigma)))

    def _z_soft_penalty(self, p_prev, p_curr, p_next):
        """Return slope and curvature penalties for z motion."""
        z0 = float(np.asarray(p_prev, dtype=np.float32).reshape(3)[2])
        z1 = float(np.asarray(p_curr, dtype=np.float32).reshape(3)[2])
        z2 = float(np.asarray(p_next, dtype=np.float32).reshape(3)[2])
        dz1 = abs(z1 - z0)
        dz2 = abs(z2 - z1)
        z_curv = abs(z2 - 2.0 * z1 + z0)
        slope_pen = self.z_slope_penalty * (dz1 + dz2)
        curvature_pen = self.z_curvature_penalty * z_curv
        return float(slope_pen), float(curvature_pen)

    def _failed_result(self, reason, t_start):
        return {
            'path_nodes': [],
            'path_coords': torch.zeros(0, 3),
            'path_types': [],
            'search_nodes': self.last_search_nodes,
            'nodes_expanded': self.last_search_nodes,
            'search_time': time.time() - t_start,
            'success': False,
            'found': False,
            'reason': reason,
        }
