"""
Graph Builder: Orchestrates the full GS-to-Graph conversion pipeline.

Pipeline:
  GS Model -> [Step 1-2: Density + Voxelization] -> [Step 3: Classification]
           -> [Step 4: Feature Encoding] -> [Step 5-6: Edge Building]
           -> [Step 7: PyG HeteroData Output]
"""

import time
import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional

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


class _SimpleNodeStore:
    def __init__(self):
        pass


class _SimpleEdgeStore:
    def __init__(self):
        pass


class _SimpleHeteroData:
    """Minimal HeteroData replacement for environments without torch_geometric."""

    def __init__(self):
        self._node_store = {}
        self._edge_store = {}
        self.graph_meta = {}
        self.free_positions = None
        self.frontier_positions = None
        self.node_features = None
        self.edge_index = None
        self.edge_weight = None
        self.occupied_positions = None
        self.bounds = None
        self.free_types = None
        self.node_type_counts = {}

    def _ensure_node(self, key):
        if key not in self._node_store:
            self._node_store[key] = _SimpleNodeStore()
        return self._node_store[key]

    def _ensure_edge(self, key):
        if key not in self._edge_store:
            self._edge_store[key] = _SimpleEdgeStore()
        return self._edge_store[key]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._ensure_edge(tuple(key))
        return self._ensure_node(key)

    @property
    def edge_types(self):
        return list(self._edge_store.keys())

    @property
    def num_nodes(self):
        total = 0
        for store in self._node_store.values():
            x = getattr(store, 'x', None)
            if x is not None:
                total += int(x.shape[0])
        return total

    @property
    def num_edges(self):
        total = 0
        for store in self._edge_store.values():
            edge_index = getattr(store, 'edge_index', None)
            if edge_index is not None:
                total += int(edge_index.shape[1])
        return total

    def validate(self, raise_on_error: bool = True):
        return True

from .density_field import DensityFieldEstimator
from .voxelizer import GraphVoxelizer
from .node_classifier import NodeClassifier, NodeType
from .feature_encoder import NodeFeatureEncoder
from .edge_builder import EdgeBuilder


class GSToGraphConverter:
    """
    GS → 异构图转换器 (编排所有Step)。
    
    从3D高斯溅射场景自动构建多模态异构图，
    供GAT-A*图搜索和图引导走廊提取使用。
    """

    def __init__(self, voxel_size: float = 0.1, robot_radius: float = 0.02,
                 density_high: float = 0.5, density_low: float = 0.1,
                 truncation_sigma: float = 3.0, chunk_size: int = 50000,
                 hash_cell_size: float = 0.3,
                 edge_lambda_dist: float = 0.5, edge_lambda_conf: float = 0.3,
                 edge_lambda_obs: float = 0.2, edge_k_neighbors: int = 6,
                 obs_expansion: int = 1,
                 device: str = 'cuda',
                 lower_bound: tuple = None,
                 upper_bound: tuple = None):
        """
        Args:
            voxel_size: 体素大小 (m)
            robot_radius: 机器人半径 (m)
            density_high: 高密度阈值 (occupied)
            density_low: 低密度阈值 (free)
            truncation_sigma: 密度场截断标准差
            chunk_size: 分块计算大小
            hash_cell_size: 空间哈希网格大小
            edge_lambda_*: 边权重参数
            edge_k_neighbors: 额外k近邻边数
            obs_expansion: 占据体素膨胀层数
            device: 计算设备
        """
        self.device = device
        self.robot_radius = robot_radius

        # Step 1-2: Density + Voxelization
        self.voxelizer = GraphVoxelizer(
            voxel_size=voxel_size,
            robot_radius=robot_radius,
            device=device,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        self.voxelizer.density_estimator = DensityFieldEstimator(
            truncation_sigma=truncation_sigma,
            chunk_size=chunk_size,
            hash_cell_size=hash_cell_size,
            device=device
        )

        # Step 3: Classification
        self.classifier = NodeClassifier(
            density_high=density_high,
            density_low=density_low,
            obs_expansion=int(obs_expansion)
        )

        # Step 4: Feature Encoding
        self.encoder = NodeFeatureEncoder(device=device)

        # Step 5-6: Edge Building
        self.edge_builder = EdgeBuilder(
            lambda_dist=edge_lambda_dist,
            lambda_conf=edge_lambda_conf,
            lambda_obs=edge_lambda_obs,
            k_neighbors=edge_k_neighbors,
            device=device
        )

    def convert(self, gsplat) -> 'HeteroData':
        """
        完整转换Pipeline: GS Model -> HeteroData。

        Args:
            gsplat: GSplatLoader实例，需包含:
                means (N,3), covs_inv (N,3,3), covs (N,3,3),
                opacities (N,), colors (N,3), scales (N,3)

        Returns:
            graph: PyG HeteroData 异构图
                - graph['free'].x: (N_free, 16)
                - graph['frontier'].x: (N_frontier, 16)
                - graph['free', 'spatial', 'free'].edge_index: (2, E)
                - graph.metadata: dict with bounds, resolution, etc.
        """
        if not HAS_PYG:
            print("[GraphNav-GS] torch_geometric unavailable, using lightweight fallback graph container.")

        total_start = time.time()

        # ===== Step 1-2: Density Field Estimation + Voxelization =====
        print('\n' + '=' * 60)
        print('[GraphNav-GS] Step 1-2: Density Field + Voxelization')
        print('=' * 60)
        t0 = time.time()

        density_grid, voxel_centers, metadata = self.voxelizer.voxelize(gsplat)

        print(f'[GraphNav-GS] Voxelization done in {time.time() - t0:.3f}s')

        # ===== Step 3: Node Classification =====
        print('\n[GraphNav-GS] Step 3: Node Classification')
        t0 = time.time()

        node_types = self.classifier.classify(density_grid)
        node_types = self._apply_hard_filters(node_types, voxel_centers, metadata)

        print(f'[GraphNav-GS] Classification done in {time.time() - t0:.3f}s')

        # ===== Step 4: Feature Encoding =====
        print('\n[GraphNav-GS] Step 4: Feature Encoding')
        t0 = time.time()

        node_features, planning_indices = self.encoder.encode(
            gsplat, density_grid, voxel_centers, node_types, metadata
        )

        print(f'[GraphNav-GS] Feature encoding done in {time.time() - t0:.3f}s')

        # ===== Step 5-6: Edge Building =====
        print('\n[GraphNav-GS] Step 5-6: Edge Building')
        t0 = time.time()

        edges = self.edge_builder.build(
            node_types, voxel_centers, node_features,
            planning_indices, density_grid
        )

        print(f'[GraphNav-GS] Edge building done in {time.time() - t0:.3f}s')

        # ===== Step 7: Assemble PyG HeteroData =====
        print('\n[GraphNav-GS] Step 7: Assembling HeteroData')
        t0 = time.time()

        graph = self._assemble_hetero_data(
            node_features, node_types, planning_indices, edges, metadata, voxel_centers, density_grid
        )

        print(f'[GraphNav-GS] Assembly done in {time.time() - t0:.3f}s')
        print(f'\n[GraphNav-GS] Total conversion time: {time.time() - total_start:.3f}s')

        # Print summary
        self._print_summary(graph, metadata)

        return graph

    def _apply_hard_filters(self, node_types: torch.Tensor,
                            voxel_centers: torch.Tensor,
                            metadata: dict) -> torch.Tensor:
        """Remove low-floor and low-clearance planning nodes before graph assembly."""
        filtered = node_types.clone()
        bounds = metadata.get('bounds', None)
        if bounds is None:
            return filtered

        bounds_t = torch.as_tensor(bounds, dtype=torch.float32, device=filtered.device)
        cell_sizes = metadata.get('cell_sizes', None)
        cell_sizes_t = None
        if cell_sizes is not None:
            cell_sizes_t = torch.as_tensor(cell_sizes, dtype=torch.float32, device=filtered.device)

        # Conservative floor: remove nodes at or below the lower scene interface.
        floor_margin = float(self.robot_radius)
        if cell_sizes_t is not None and cell_sizes_t.numel() >= 3:
            floor_margin = max(floor_margin, float(cell_sizes_t[2].item()))
        z_floor = float(bounds_t[0, 2].item() + floor_margin)

        flat_centers = voxel_centers.reshape(-1, 3)
        flat_types = filtered.reshape(-1)
        flat_types[flat_centers[:, 2] <= z_floor] = NodeType.OCCUPIED

        metadata['z_floor'] = z_floor
        metadata['occupied_positions'] = flat_centers[flat_types == NodeType.OCCUPIED].detach().cpu().numpy()
        self.classifier.class_counts = {
            'occupied': int((flat_types == NodeType.OCCUPIED).sum().item()),
            'free': int((flat_types == NodeType.FREE).sum().item()),
            'frontier': int((flat_types == NodeType.FRONTIER).sum().item()),
            'unknown': int((flat_types == NodeType.UNKNOWN).sum().item()),
            'total': int(flat_types.numel()),
        }
        return flat_types.reshape_as(filtered)

    def _assemble_hetero_data(self, node_features: torch.Tensor,
                                node_types: torch.Tensor,
                                planning_indices: torch.Tensor,
                                edges: Dict, metadata: dict,
                                voxel_centers: torch.Tensor,
                                density_grid: torch.Tensor) -> HeteroData:
        """
        组装 PyG HeteroData 对象。

        Args:
            node_features: (N_plan, 16) 所有规划节点特征
            node_types: (Rx, Ry, Rz) 节点类型网格
            planning_indices: (N_plan, 3) 规划节点索引
            edges: 边字典
            metadata: 元信息

        Returns:
            HeteroData
        """
        data = HeteroData() if HAS_PYG else _SimpleHeteroData()

        # Get per-type masks
        plan_types = node_types[
            planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
        ]
        plan_positions = voxel_centers[
            planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
        ]
        free_mask = (plan_types == NodeType.FREE)
        frontier_mask = (plan_types == NodeType.FRONTIER)

        # Map local indices to global plan indices
        free_global = torch.where(free_mask)[0]
        frontier_global = torch.where(frontier_mask)[0]

        # Node features
        data['free'].x = node_features[free_global]
        data['frontier'].x = node_features[frontier_global]

        # Node positions (store for planning queries)
        data['free'].pos = plan_positions[free_mask]
        data['frontier'].pos = plan_positions[frontier_mask]

        # Edges - need to remap from global plan indices to per-type local indices
        # free nodes: global i -> local index = position in free_global
        n_plan = node_features.shape[0]
        free_global_to_local = torch.full((n_plan,), -1, dtype=torch.long,
                                         device=self.device)
        if free_global.numel() > 0:
            free_global_to_local[free_global] = torch.arange(free_global.numel(),
                                                             device=self.device)

        frontier_global_to_local = torch.full((n_plan,), -1, dtype=torch.long,
                                              device=self.device)
        if frontier_global.numel() > 0:
            frontier_global_to_local[frontier_global] = torch.arange(frontier_global.numel(),
                                                                     device=self.device)

        # spatial_free_free
        if edges['spatial_free_free'][0].shape[0] > 0:
            src_g = edges['spatial_free_free'][0]
            dst_g = edges['spatial_free_free'][1]
            w = edges['spatial_free_free'][2]
            src_l = free_global_to_local[src_g]
            dst_l = free_global_to_local[dst_g]
            valid = (src_l >= 0) & (dst_l >= 0)
            data['free', 'spatial', 'free'].edge_index = torch.stack(
                [src_l[valid], dst_l[valid]], dim=0
            )
            data['free', 'spatial', 'free'].edge_weight = w[valid]
        else:
            data['free', 'spatial', 'free'].edge_index = torch.zeros(
                2, 0, dtype=torch.long, device=self.device
            )
            data['free', 'spatial', 'free'].edge_weight = torch.zeros(
                0, device=self.device
            )

        # spatial_free_frontier
        if edges['spatial_free_frontier'][0].shape[0] > 0:
            src_g = edges['spatial_free_frontier'][0]
            dst_g = edges['spatial_free_frontier'][1]
            w = edges['spatial_free_frontier'][2]
            src_l = free_global_to_local[src_g]
            dst_l = frontier_global_to_local[dst_g]
            valid = (src_l >= 0) & (dst_l >= 0)
            data['free', 'spatial', 'frontier'].edge_index = torch.stack(
                [src_l[valid], dst_l[valid]], dim=0
            )
            data['free', 'spatial', 'frontier'].edge_weight = w[valid]
            # Reverse direction
            data['frontier', 'spatial', 'free'].edge_index = torch.stack(
                [dst_l[valid], src_l[valid]], dim=0
            )
            data['frontier', 'spatial', 'free'].edge_weight = w[valid]
        else:
            for etype in [('free', 'spatial', 'frontier'), ('frontier', 'spatial', 'free')]:
                data[etype[0], etype[1], etype[2]].edge_index = torch.zeros(
                    2, 0, dtype=torch.long, device=self.device
                )
                data[etype[0], etype[1], etype[2]].edge_weight = torch.zeros(
                    0, device=self.device
                )

        # frontier_frontier
        if edges['frontier_frontier'][0].shape[0] > 0:
            src_g = edges['frontier_frontier'][0]
            dst_g = edges['frontier_frontier'][1]
            w = edges['frontier_frontier'][2]
            src_l = frontier_global_to_local[src_g]
            dst_l = frontier_global_to_local[dst_g]
            valid = (src_l >= 0) & (dst_l >= 0)
            data['frontier', 'connects', 'frontier'].edge_index = torch.stack(
                [src_l[valid], dst_l[valid]], dim=0
            )
            data['frontier', 'connects', 'frontier'].edge_weight = w[valid]
        else:
            data['frontier', 'connects', 'frontier'].edge_index = torch.zeros(
                2, 0, dtype=torch.long, device=self.device
            )
            data['frontier', 'connects', 'frontier'].edge_weight = torch.zeros(
                0, device=self.device
            )

        # Store metadata (use graph_meta to avoid shadowing PyG's metadata() method)
        data.graph_meta = {
            'voxel_size': metadata['voxel_size'],
            'robot_radius': self.robot_radius,
            'bounds': metadata['bounds'],
            'resolution': metadata['resolution'],
            'cell_sizes': metadata.get('cell_sizes', None),
            'z_floor': metadata.get('z_floor', None),
            'num_free': free_mask.sum().item(),
            'num_frontier': frontier_mask.sum().item(),
            'num_occupied': self.classifier.class_counts.get('occupied', 0),
            'num_total': self.classifier.class_counts.get('total', 0),
            'planning_indices': planning_indices,
            'node_types': node_types,
            'voxel_centers': voxel_centers,
            'occupied_positions': metadata.get('occupied_positions', None),
            'density_grid': density_grid.detach().cpu().numpy(),
            'uncertainty_grid': self.classifier.compute_uncertainty(
                density_grid, node_types
            ).detach().cpu().numpy(),
        }

        # Compatibility fields for legacy planners / evaluators.
        free_positions = plan_positions[free_mask]
        frontier_positions = plan_positions[frontier_mask]
        free_features = node_features[free_global].clone()
        if free_features.shape[0] > 0:
            free_features[:, :3] = free_positions

        data.free_positions = free_positions.detach().cpu().numpy()
        data.frontier_positions = frontier_positions.detach().cpu().numpy()
        data.node_features = free_features.detach().cpu().numpy()
        data.edge_index = data['free', 'spatial', 'free'].edge_index.detach().cpu().numpy()
        data.edge_weight = data['free', 'spatial', 'free'].edge_weight.detach().cpu().numpy()
        data.occupied_positions = voxel_centers[node_types == NodeType.OCCUPIED].reshape(-1, 3).detach().cpu().numpy()
        data.bounds = metadata['bounds'].detach().cpu().numpy() if hasattr(metadata['bounds'], 'detach') else np.asarray(metadata['bounds'])
        data.free_types = plan_types[free_mask].detach().cpu().numpy()
        data.node_type_counts = {
            'free': free_mask.sum().item(),
            'frontier': frontier_mask.sum().item(),
            'occupied': (plan_types == NodeType.OCCUPIED).sum().item(),
            'unknown': (plan_types == NodeType.UNKNOWN).sum().item(),
        }

        data.validate(raise_on_error=True)

        return data

    def _print_summary(self, graph: HeteroData, metadata: dict):
        """Print graph summary."""
        gm = graph.graph_meta if hasattr(graph, 'graph_meta') and isinstance(graph.graph_meta, dict) else {}
        print('\n' + '=' * 60)
        print('[GraphNav-GS] Graph Summary')
        print('=' * 60)
        print(f'  Voxel size: {metadata["voxel_size"]:.3f}m')
        print(f'  Resolution: {metadata["resolution"].tolist()}')
        print(f'  Free nodes: {gm.get("num_free", "?")}')
        print(f'  Frontier nodes: {gm.get("num_frontier", "?")}')
        print(f'  Occupied nodes: {gm.get("num_occupied", "?")}')
        print(f'  Free features: {graph["free"].x.shape}')
        print(f'  Frontier features: {graph["frontier"].x.shape}')
        for edge_type in graph.edge_types:
            print(f'  Edges {edge_type}: {graph[edge_type].edge_index.shape[1]}')
        print('=' * 60)

    def convert_to_binary_grid(self, gsplat) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        轻量转换：仅生成二值占据网格 (兼容SplatNav的A*)。
        用于消融实验中的 baseline 对比。

        Args:
            gsplat: GSplatLoader实例

        Returns:
            non_navigable: (Rx, Ry, Rz) bool, True=occupied
            grid_centers: (Rx, Ry, Rz, 3) 体素中心
            metadata: dict
        """
        density_grid, voxel_centers, metadata = self.voxelizer.voxelize(gsplat)
        node_types = self.classifier.classify(density_grid)
        non_navigable = (node_types == NodeType.OCCUPIED) | (node_types == NodeType.UNKNOWN)
        return non_navigable, voxel_centers, metadata
