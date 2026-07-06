"""
Train GAT Heuristic Network.

支持两种加载方式:
1. PLY 文件: --scene data/stonehenge.ply
2. .ckpt 模型 (splatnav 方式): --scene_name stonehenge

Usage:
  # 从 .ckpt 加载 (splatnav 方式, 推荐)
  python train_gat.py --scene_name stonehenge --output checkpoints/gat_stonehenge.pth --epochs 100

  # 从 PLY 加载 (原有方式)
  python train_gat.py --scene data/stonehenge.ply --output checkpoints/gat_stonehenge.pth --epochs 100

  # 继续训练
  python train_gat.py --scene_name stonehenge --output checkpoints/gat_stonehenge.pth \
       --checkpoint checkpoints/gat_stonehenge.pth --epochs 100
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from utils.scene_config import get_scene_config, get_scene_names, load_repo_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# splatnav 路径
SPLATNAV_ROOT = os.path.join(PROJECT_ROOT, 'baseyuan', 'splatnav')
_additional_paths = [SPLATNAV_ROOT]
for p in _additional_paths:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')
SCENE_CONFIGS = get_scene_names(CONFIG_PATH)
REPO_CFG = load_repo_config(CONFIG_PATH)
GAT_CFG = dict(REPO_CFG.get('gat_astar', {}))
TRAIN_OBJ_CFG = {
    'length_weight': 1.0,
    'clearance_weight': float(GAT_CFG.get('obstacle_soft_weight', 3.0)),
    'smoothness_weight': float(GAT_CFG.get('z_slope_penalty', 1.5)),
    'curvature_weight': float(GAT_CFG.get('z_curvature_penalty', 1.6)),
    'clearance_feature_idx': 15,
    'smoothness_feature_idx': 14,
    'curvature_feature_slice': [3, 6],
}
TRAIN_OBJ_CFG.update(dict(REPO_CFG.get('train_objective', {})))


class GSplatModelWrapper:
    """将 splatnav 的 GSplatLoader 结果包装成 GraphNav 需要的格式"""

    def __init__(self, gsplat):
        self.gsplat = gsplat
        self.device = gsplat.device

        self.positions = gsplat.means
        self.means = gsplat.means
        self.scales = gsplat.scales
        self.rotations = gsplat.rots
        self.quats = gsplat.rots
        self.colors = gsplat.colors
        self.opacities = gsplat.opacities
        self.covs = gsplat.covs
        self.covs_inv = getattr(gsplat, 'covs_inv', None)
        self.n_gaussians = self.means.shape[0]

        print(f"Wrapped GSplat model: {self.n_gaussians} Gaussians")


def load_gsplat_model(scene_name, device='cuda', config_path=CONFIG_PATH):
    """按照 splatnav 的方式加载 GS 模型"""
    from splat.splat_utils import GSplatLoader

    scene_cfg = get_scene_config(scene_name, config_path)

    cfg_path = os.path.join(SPLATNAV_ROOT, scene_cfg['path'])
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print(f"Loading {scene_name} from {cfg_path}")
    gsplat = GSplatLoader(cfg_path, device)
    model = GSplatModelWrapper(gsplat)
    return model


def filter_gaussians_by_bounds(model, lower_bound, upper_bound):
    """根据边界过滤高斯"""
    pos = model.positions
    if hasattr(pos, 'cpu'):
        pos = pos.detach().cpu().numpy()
    else:
        pos = np.asarray(pos)

    mask = np.all((pos >= lower_bound) & (pos <= upper_bound), axis=1)

    class FilteredModel:
        pass

    filtered = FilteredModel()
    for attr in ['positions', 'means', 'scales', 'rotations', 'quats', 'colors', 'opacities', 'covs', 'covs_inv', 'n_gaussians']:
        val = getattr(model, attr, None)
        if val is not None:
            if hasattr(val, 'cpu'):
                val = val.detach().cpu().numpy()
            if isinstance(val, np.ndarray) and len(val) == model.n_gaussians:
                val = val[mask]
                if hasattr(val, 'to'):
                    val = torch.from_numpy(val)
            setattr(filtered, attr, val)

    filtered.n_gaussians = mask.sum()
    filtered.bounds = np.array([lower_bound, upper_bound])
    print(f"Filtered to {filtered.n_gaussians} Gaussians within bounds")
    return filtered


def build_graph_from_model(model, scene_cfg, device='cuda'):
    """从 GS 模型构建图"""
    from gs_to_graph.graph_builder import GSToGraphConverter
    edge_cfg = scene_cfg.get('edge', {}) if isinstance(scene_cfg, dict) else {}
    lower_bound = np.array(scene_cfg['lower_bound'], dtype=np.float32) if 'lower_bound' in scene_cfg else None
    upper_bound = np.array(scene_cfg['upper_bound'], dtype=np.float32) if 'upper_bound' in scene_cfg else None

    # 训练空间与实际规划空间保持一致：只要场景定义了规划边界，就裁剪到该切片。
    if lower_bound is not None and upper_bound is not None:
        model = filter_gaussians_by_bounds(model, lower_bound, upper_bound)

    converter = GSToGraphConverter(
        voxel_size=scene_cfg.get('voxel_size', 0.05),
        robot_radius=scene_cfg.get('robot_radius', 0.02),
        density_high=scene_cfg.get('density_high', 2.0),
        density_low=scene_cfg.get('density_low', 0.3),
        truncation_sigma=3.0,
        chunk_size=50000,
        hash_cell_size=0.3,
        obs_expansion=0,
        edge_lambda_dist=edge_cfg.get('lambda_dist', 0.5),
        edge_lambda_conf=edge_cfg.get('lambda_conf', 0.3),
        edge_lambda_obs=edge_cfg.get('lambda_obs', 0.2),
        edge_k_neighbors=edge_cfg.get('k_neighbors', 6),
        device=device,
        lower_bound=tuple(lower_bound) if lower_bound is not None else None,
        upper_bound=tuple(upper_bound) if upper_bound is not None else None,
    )

    print("Building graph...")
    graph = converter.convert(model)
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    return graph, converter


# --------------------------------------------------------------------------- #
# 以下为原有训练代码 (最小改动)
# --------------------------------------------------------------------------- #

from utils.gs_loader import load_ply
from gs_to_graph.graph_builder import GSToGraphConverter
from gat_astar.gat_heuristic import GATHeuristicNet
from gat_astar.gat_astar import GATAStar
from gat_astar.gpu_dijkstra import get_cost_cache, build_unified_edge_index


def dijkstra_true_costs(graph, goal_global_idx: int, n_free: int, device: str = 'cuda') -> torch.Tensor:
    """Compatibility wrapper around the GPU goal-cost cache."""
    cost_cache = get_cost_cache()
    return cost_cache.get_cost_for_goal(graph, n_free, goal_global_idx, torch.device(device))


def load_pairs_from_json(path: str) -> list:
    """Load (start, goal) pairs from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    pairs = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'start' in item and 'goal' in item:
                pairs.append((item['start'], item['goal']))
            elif isinstance(item, list) and len(item) == 2:
                pairs.append((item[0], item[1]))
    return pairs


def generate_random_pairs(graph, num_pairs: int, seed: int = 42) -> list:
    """Generate random (start, goal) pairs in free space."""
    free_positions = getattr(graph, 'free_positions', None)
    if free_positions is None:
        metadata = getattr(graph, 'graph_meta', None)
        if metadata is not None:
            planning_indices = metadata['planning_indices'].cpu().numpy()
            voxel_centers = metadata['voxel_centers'].cpu().numpy()
            node_types_grid = metadata['node_types'].cpu().numpy()

            plan_types = node_types_grid[
                planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
            ]
            plan_positions = voxel_centers[
                planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
            ]
            free_mask = plan_types == 0
            free_positions = plan_positions[free_mask]
        else:
            free_positions = np.zeros((0, 3))

    if len(free_positions) < 2:
        print(f"Warning: only {len(free_positions)} free positions, cannot generate pairs")
        return []

    free_positions = np.asarray(free_positions)

    np.random.seed(seed)
    pairs = []
    for _ in range(num_pairs):
        si = np.random.randint(0, len(free_positions))
        gi = np.random.randint(0, len(free_positions))
        while si == gi:
            gi = np.random.randint(0, len(free_positions))
        pairs.append((free_positions[si].tolist(), free_positions[gi].tolist()))
    return pairs


def _prepare_pairs(graph, pairs, metadata, device, n_free):
    """Snap query endpoints once and reuse unified graph indices across epochs."""
    planning_indices = metadata['planning_indices']
    voxel_centers = metadata['voxel_centers']
    plan_positions = voxel_centers[
        planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
    ]
    plan_types = metadata['node_types']
    plan_types_flat = plan_types[
        planning_indices[:, 0], planning_indices[:, 1], planning_indices[:, 2]
    ]

    valid = plan_types_flat != -1
    free_plan_indices = torch.where(plan_types_flat == 0)[0]
    frontier_plan_indices = torch.where(plan_types_flat == 1)[0]

    plan_to_unified = torch.full((planning_indices.shape[0],), -1, dtype=torch.long, device=device)
    if free_plan_indices.numel() > 0:
        plan_to_unified[free_plan_indices] = torch.arange(free_plan_indices.numel(), dtype=torch.long, device=device)
    if frontier_plan_indices.numel() > 0:
        plan_to_unified[frontier_plan_indices] = n_free + torch.arange(frontier_plan_indices.numel(), dtype=torch.long, device=device)

    start_pos_tensor = torch.as_tensor(np.asarray([p[0] for p in pairs], dtype=np.float32), device=device)
    goal_pos_tensor = torch.as_tensor(np.asarray([p[1] for p in pairs], dtype=np.float32), device=device)

    dists = torch.cdist(start_pos_tensor, plan_positions)
    dists[:, ~valid] = float('inf')
    start_plan_idx = torch.argmin(dists, dim=1)
    start_global = plan_to_unified[start_plan_idx]
    snapped_start_pos = plan_positions[start_plan_idx]

    dists = torch.cdist(goal_pos_tensor, plan_positions)
    dists[:, ~valid] = float('inf')
    goal_plan_idx = torch.argmin(dists, dim=1)
    goal_global = plan_to_unified[goal_plan_idx]
    snapped_goal_pos = plan_positions[goal_plan_idx]

    records = []
    for idx, (start, goal) in enumerate(pairs):
        records.append({
            'start_pos': snapped_start_pos[idx].detach().cpu().numpy().astype(np.float32),
            'start_query_pos': np.asarray(start, dtype=np.float32),
            'goal_pos': snapped_goal_pos[idx].detach().cpu().numpy().astype(np.float32),
            'goal_query_pos': np.asarray(goal, dtype=np.float32),
            'start_global': int(start_global[idx].item()),
            'goal_global': int(goal_global[idx].item()),
        })
    anchor_indices = torch.cat([start_global, goal_global], dim=0).unique(sorted=True)
    return records, anchor_indices


def _build_training_tensors(graph, n_free, device):
    x_all = torch.cat([graph['free'].x, graph['frontier'].x], dim=0).to(device)
    pos_parts = []
    if hasattr(graph['free'], 'pos') and graph['free'].pos is not None:
        pos_parts.append(graph['free'].pos)
    if hasattr(graph['frontier'], 'pos') and graph['frontier'].pos is not None:
        pos_parts.append(graph['frontier'].pos)
    if pos_parts:
        pos_all = torch.cat(pos_parts, dim=0).to(device)
    else:
        pos_all = x_all[:, :3]
    edge_index, edge_weight = build_unified_edge_index(graph, n_free, device)
    return x_all, pos_all, edge_index, edge_weight


def train_one_pair(gat_net, x_all, pos_all, edge_index, edge_weight, graph,
                   start_pos, goal_pos, start_global_idx, goal_global_idx, n_free,
                   optimizer, device, cost_cache, max_train_nodes_per_query,
                   objective_cfg: dict) -> float:
    """Train on one (start, goal) query."""
    goal_costs_t = dijkstra_true_costs(graph, goal_global_idx, n_free, device)
    best_cost = goal_costs_t[int(start_global_idx)].item()
    if not np.isfinite(best_cost) or best_cost < 0:
        return None

    reachable_mask = torch.isfinite(goal_costs_t) & (goal_costs_t >= 0)
    reachable_indices = torch.where(reachable_mask)[0]
    if reachable_indices.numel() < 2:
        return None

    train_budget = min(max_train_nodes_per_query, reachable_indices.numel()) if max_train_nodes_per_query > 0 else reachable_indices.numel()

    goal_costs_sorted = goal_costs_t[reachable_indices].sort()
    sorted_indices = reachable_indices[goal_costs_sorted.indices]
    sorted_values = goal_costs_sorted.values

    max_cost = sorted_values[-1].item()
    if max_cost <= 0:
        return None

    cost_range = max_cost - sorted_values[0].item()
    band_lo = goal_costs_t[sorted_indices[0]].item()
    band_hi = goal_costs_t[sorted_indices[-1]].item()
    cost_scale_hint = max(band_hi - band_lo, 1.0)

    hard_threshold = 0.01 * cost_scale_hint
    band_indices = sorted_indices[(sorted_values - sorted_values[0]) <= 0.20 * cost_range]
    hard_indices = sorted_indices[(sorted_values - sorted_values[0]) <= hard_threshold]

    best_idx_in_reachable = 0
    special_idx = sorted_indices[best_idx_in_reachable:best_idx_in_reachable + 1]

    used_mask = torch.zeros(reachable_indices.max().item() + 1, dtype=torch.bool, device=device)
    used_mask[band_indices] = True
    used_mask[hard_indices] = True
    used_mask[special_idx] = True
    background_pool = reachable_indices[~used_mask[reachable_indices]]
    n_background = max(16, train_budget - int(band_indices.numel()) - int(hard_indices.numel()) - int(special_idx.numel()))
    if background_pool.numel() > 0:
        if background_pool.numel() > n_background:
            perm = torch.randperm(background_pool.numel(), device=device)[:n_background]
            background_idx = background_pool[perm]
        else:
            background_idx = background_pool
    else:
        background_idx = torch.zeros((0,), dtype=torch.long, device=device)

    train_idx_t = torch.unique(torch.cat([band_indices, hard_indices, background_idx, special_idx], dim=0), sorted=False)
    train_costs_t = goal_costs_t[train_idx_t]

    if len(train_idx_t) < 2:
        return None

    goal_pos_t = torch.tensor(goal_pos, dtype=torch.float32, device=device)
    cost_scale = GATAStar.infer_cost_scale(graph=graph, metadata=getattr(graph, 'graph_meta', None))
    node_pos = pos_all
    geo_h_all = torch.norm(node_pos - goal_pos_t.unsqueeze(0), dim=-1) * float(cost_scale)

    gat_net.train()
    uncertainty = None
    if gat_net.use_uncertainty and x_all.ndim == 2 and x_all.shape[1] > 9:
        uncertainty = x_all[:, 9]
    h_all = gat_net.forward(
        x=x_all, edge_index=edge_index, edge_weight=edge_weight,
        goal_pos=goal_pos_t, uncertainty=uncertainty, pos_coords=pos_all, cost_scale=cost_scale,
    )

    pred = h_all[train_idx_t]
    geo_h = geo_h_all[train_idx_t]

    smoothness_weight = float(objective_cfg.get('smoothness_weight', 0.0))
    curvature_weight = float(objective_cfg.get('curvature_weight', 0.0))
    clearance_weight = float(objective_cfg.get('clearance_weight', 0.0))
    length_weight = float(objective_cfg.get('length_weight', 1.0))
    clearance_feature_idx = int(objective_cfg.get('clearance_feature_idx', 15))
    smoothness_feature_idx = int(objective_cfg.get('smoothness_feature_idx', 14))
    curvature_slice = objective_cfg.get('curvature_feature_slice', [3, 6])
    if not isinstance(curvature_slice, (list, tuple)) or len(curvature_slice) != 2:
        curvature_slice = [3, 6]
    curv_start = int(curvature_slice[0])
    curv_end = int(curvature_slice[1])

    path_feat = x_all[train_idx_t]
    if clearance_feature_idx < path_feat.shape[1]:
        clearance_proxy = 1.0 - path_feat[:, clearance_feature_idx].clamp(0.0, 1.0)
    else:
        clearance_proxy = torch.zeros_like(train_costs_t)

    if smoothness_feature_idx < path_feat.shape[1]:
        smoothness_proxy = path_feat[:, smoothness_feature_idx].clamp(min=0.0)
    else:
        smoothness_proxy = torch.zeros_like(train_costs_t)

    if curv_end <= path_feat.shape[1] and curv_end > curv_start:
        curvature_proxy = torch.norm(path_feat[:, curv_start:curv_end], dim=1)
    else:
        curvature_proxy = torch.zeros_like(train_costs_t)

    composite_true = (
        length_weight * train_costs_t
        + clearance_weight * clearance_proxy
        + smoothness_weight * smoothness_proxy
        + curvature_weight * curvature_proxy
    )

    residual_pred = pred - geo_h
    residual_true = composite_true - geo_h
    residual_scale = torch.clamp(composite_true.abs().mean(), min=1.0)
    mse = F.smooth_l1_loss(residual_pred / residual_scale, residual_true / residual_scale)

    overestimate = F.relu(pred - composite_true)
    over_penalty = 0.1 * overestimate.mean()

    ranking_penalty = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if pred.numel() >= 8:
        pair_count = min(2048, pred.numel() * 4)
        idx_i = torch.randint(0, pred.numel(), (pair_count,), device=pred.device)
        idx_j = torch.randint(0, pred.numel(), (pair_count,), device=pred.device)
        pair_mask = idx_i != idx_j
        idx_i = idx_i[pair_mask]
        idx_j = idx_j[pair_mask]
        if idx_i.numel() > 0:
            true_delta = composite_true[idx_j] - composite_true[idx_i]
            pred_delta = pred[idx_j] - pred[idx_i]
            informative = true_delta.abs() > 0.05 * residual_scale * 0.1
            if informative.any():
                true_sign = torch.sign(true_delta[informative])
                pred_norm = pred_delta[informative] / residual_scale
                ranking_penalty = F.relu(0.05 - pred_norm * true_sign).mean()

    monotonic_penalty = torch.zeros((), dtype=pred.dtype, device=pred.device)
    train_mask = torch.zeros((x_all.shape[0],), dtype=torch.bool, device=device)
    train_mask[train_idx_t] = True
    edge_mask = train_mask[edge_index[0]] & train_mask[edge_index[1]]
    if edge_mask.any():
        edge_src = edge_index[0][edge_mask]
        edge_dst = edge_index[1][edge_mask]
        train_pos_map = torch.full((x_all.shape[0],), -1, dtype=torch.long, device=device)
        train_pos_map[train_idx_t] = torch.arange(train_idx_t.numel(), dtype=torch.long, device=device)
        edge_src_pos = train_pos_map[edge_src]
        edge_dst_pos = train_pos_map[edge_dst]
        valid_edges = (edge_src_pos >= 0) & (edge_dst_pos >= 0)
        true_edge_delta = composite_true[edge_dst_pos[valid_edges]] - composite_true[edge_src_pos[valid_edges]]
        informative_edges = true_edge_delta.abs() > 1e-4
        if informative_edges.any():
            valid_src = edge_src[valid_edges][informative_edges]
            valid_dst = edge_dst[valid_edges][informative_edges]
            pred_edge_delta = h_all[valid_dst] - h_all[valid_src]
            true_sign = torch.sign(true_edge_delta[informative_edges])
            monotonic_penalty = F.relu(0.02 * residual_scale - pred_edge_delta * true_sign).mean() / residual_scale

    start_goal_bias = torch.zeros((), dtype=pred.dtype, device=pred.device)
    start_h = h_all[int(start_global_idx)]
    goal_h = h_all[int(goal_global_idx)]
    start_goal_bias = F.relu(goal_h - 0.05 * residual_scale) + F.relu((composite_true.max() - start_h).abs() - 0.1 * max(float(composite_true.max().item()), 1.0))

    loss = mse + over_penalty + 0.20 * ranking_penalty + 0.10 * monotonic_penalty + 0.05 * start_goal_bias

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gat_net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def train(gat_net, graph, pairs, hard_pairs=None, num_epochs=100, lr=0.001,
          device='cuda', verbose=True, max_train_nodes_per_query=0, hard_pair_repeat=0,
          objective_cfg: dict | None = None) -> list:
    """Train on multiple (start, goal) pairs for num_epochs."""
    if objective_cfg is None:
        objective_cfg = {}
    n_free = graph['free'].x.shape[0]
    metadata = getattr(graph, 'graph_meta', None)

    records, anchor_indices = _prepare_pairs(graph, pairs, metadata, torch.device(device), n_free)
    x_all, pos_all, edge_index, edge_weight = _build_training_tensors(graph, n_free, device)

    cost_cache = get_cost_cache()
    optimizer = torch.optim.Adam(gat_net.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        n_processed = 0

        for record in records:
            loss = train_one_pair(
                gat_net, x_all, pos_all, edge_index, edge_weight, graph,
                record['start_pos'], record['goal_pos'],
                record['start_global'], record['goal_global'],
                n_free, optimizer, device, cost_cache, max_train_nodes_per_query,
                objective_cfg
            )
            if loss is not None:
                epoch_losses.append(loss)
                n_processed += 1

        # Process hard pairs
        if hard_pairs and (epoch % (hard_pair_repeat + 1) == 0):
            for record in _prepare_pairs(graph, hard_pairs, metadata, torch.device(device), n_free)[0]:
                loss = train_one_pair(
                    gat_net, x_all, pos_all, edge_index, edge_weight, graph,
                    record['start_pos'], record['goal_pos'],
                    record['start_global'], record['goal_global'],
                    n_free, optimizer, device, cost_cache, max_train_nodes_per_query,
                    objective_cfg
                )
                if loss is not None:
                    epoch_losses.append(loss)
                    n_processed += 1

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch}/{num_epochs}: avg loss = {avg_loss:.4f} ({n_processed} queries)")
        elif verbose:
            print(f"Epoch {epoch}/{num_epochs}: no valid queries")

    return losses


def main():
    parser = argparse.ArgumentParser(description='Train GAT Heuristic Network')
    parser.add_argument('--scene', type=str, help='Path to .ply file (legacy)')
    parser.add_argument('--scene_name', type=str, choices=SCENE_CONFIGS,
                        help='Scene name for .ckpt loading (splatnav style)')
    parser.add_argument('--output', type=str, required=True, help='Output checkpoint path')
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--queries', type=int, default=100, help='Random query count')
    parser.add_argument('--pairs', type=str, help='JSON file with (start, goal) pairs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_train_nodes', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型和构建图
    if args.scene_name:
        # splatnav 方式加载 .ckpt
        model = load_gsplat_model(args.scene_name, device)
        scene_cfg = get_scene_config(args.scene_name, CONFIG_PATH)

        if scene_cfg.get('apply_bounds_filter', True) and 'lower_bound' in scene_cfg and 'upper_bound' in scene_cfg:
            lower = np.array(scene_cfg['lower_bound'], dtype=np.float32)
            upper = np.array(scene_cfg['upper_bound'], dtype=np.float32)
            model = filter_gaussians_by_bounds(model, lower, upper)

        graph, converter = build_graph_from_model(model, scene_cfg, device)
    elif args.scene:
        # 原有 PLY 方式加载
        print(f"Loading PLY from {args.scene}")
        model = load_ply(args.scene)
        scene_cfg = {
            'voxel_size': 0.05,
            'robot_radius': 0.02,
            'density_high': 0.5,
            'density_low': 0.1,
        }
        graph, converter = build_graph_from_model(model, scene_cfg, device)
    else:
        print("Error: must provide either --scene or --scene_name")
        sys.exit(1)

    # 创建 GAT 网络
    gat_net = GATHeuristicNet(
        in_dim=16,
        hidden_dim=GAT_CFG.get('hidden_dim', 64),
        num_heads=GAT_CFG.get('num_heads', 4),
        num_layers=GAT_CFG.get('num_layers', 2),
        dropout=GAT_CFG.get('dropout', 0.1),
    ).to(device)

    # 加载 checkpoint 如果有
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        gat_net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")

    # 生成或加载训练对
    if args.pairs:
        pairs = load_pairs_from_json(args.pairs)
        print(f"Loaded {len(pairs)} training pairs from {args.pairs}")
    else:
        pairs = generate_random_pairs(graph, args.queries)
        print(f"Generated {len(pairs)} random training pairs")

    # 训练
    print(f"\nStarting training for {args.epochs} epochs...")
    losses = train(
        gat_net, graph, pairs,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        verbose=True,
        max_train_nodes_per_query=args.max_train_nodes,
        objective_cfg=TRAIN_OBJ_CFG,
    )

    # 保存 checkpoint
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': gat_net.state_dict(),
        'losses': losses,
        'model_kwargs': {
            'in_dim': 16,
            'hidden_dim': GAT_CFG.get('hidden_dim', 64),
            'num_heads': GAT_CFG.get('num_heads', 4),
            'num_layers': GAT_CFG.get('num_layers', 2),
            'dropout': GAT_CFG.get('dropout', 0.1),
        },
        'train_objective': TRAIN_OBJ_CFG,
    }, args.output)
    print(f"\nCheckpoint saved to {args.output}")

    # Strict round-trip validation: the saved checkpoint must reload into the same architecture.
    verify_net = GATHeuristicNet(
        in_dim=16,
        hidden_dim=GAT_CFG.get('hidden_dim', 64),
        num_heads=GAT_CFG.get('num_heads', 4),
        num_layers=GAT_CFG.get('num_layers', 2),
        dropout=GAT_CFG.get('dropout', 0.1),
    ).to(device)
    reloaded = torch.load(args.output, map_location=device, weights_only=False)
    verify_net.load_state_dict(reloaded['model_state_dict'], strict=True)
    print("[CHECK] Checkpoint reload validation passed")


if __name__ == '__main__':
    main()
