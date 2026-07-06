"""
GraphNav-GS: Graph-Augmented Navigation in Gaussian Splatting Maps

Main entry point for the complete planning pipeline:

  Pipeline:
    1. Load 3DGS scene (.ply)
    2. GS-to-Graph: Extract heterogeneous graph
    3. GAT-A*: Search for global path
    4. Corridor Extraction: Build safe corridors along path
    5. B-Spline Optimization: Generate smooth trajectory
    6. Evaluate: Compute metrics and visualize

Usage:
  python run_graphnav.py --config config.yaml --scene <ply_path> --start x y z --goal x y z
  python run_graphnav.py --mode eval --config config.yaml --results_dir ./results
  python run_graphnav.py --mode ablation --config config.yaml --scene <ply_path>
"""

import os
import sys
import argparse
import time
import yaml

# Avoid OpenMP duplicate-runtime crashes on Windows when multiple numeric libs load.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/graphnav_mplconfig')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/graphnav_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = os.path.dirname(PROJECT_ROOT)
if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graphnav_gs.utils.runtime_deps import bootstrap_runtime_deps

# Make repo-local wheels and HSL available before baseline imports.
_runtime_deps = bootstrap_runtime_deps(PROJECT_ROOT)

# Prefer a repo-local torch build when present AND the system torch is missing
# or lacks support for the current GPU.  When a conda env already provides a
# working torch (e.g. graphnav env with torch 2.0.1+cu117), skip the vendor
# override — the vendor build is older (1.13.1) and disables torch_geometric.
_local_torch_dir = os.path.join(PROJECT_ROOT, '.vendor', 'torch-cu117')
_local_numpy_dir = os.path.join(PROJECT_ROOT, '.vendor', 'numpy-1.26.4')

def _system_torch_ok():
    try:
        import torch as _t
        return True
    except ImportError:
        return False

if not _system_torch_ok():
    if os.path.isdir(_local_torch_dir):
        sys.path.insert(0, _local_torch_dir)
        os.environ.setdefault('GRAPHNAV_DISABLE_PYG', '1')
    if os.path.isdir(_local_numpy_dir):
        sys.path.insert(0, _local_numpy_dir)

# NOTE: Do NOT inject conda site-packages here. When CONDA_PREFIX points to the
# base environment (e.g. /home/user/anaconda3) while the active Python is from a
# child env, this would prepend the base env's site-packages and shadow the child
# env's packages with incompatible versions (e.g. numpy 1.26 with a broken
# pathlib.py shim on Python 3.11). The conda env's interpreter already has the
# correct sys.path; no manual injection is needed.

import numpy as np
import torch
from pathlib import Path

from graphnav_gs.gs_to_graph.graph_builder import GSToGraphConverter
from graphnav_gs.gat_astar.gat_astar import GATAStar
from graphnav_gs.gat_astar.gat_heuristic import GATHeuristicNet
from graphnav_gs.graph_corridor.corridor_extractor import GraphCorridorExtractor
from graphnav_gs.graph_corridor.bspline_optimizer import GraphSplineOptimizer
from graphnav_gs.utils.gs_loader import load_ply, GaussianModel
from graphnav_gs.utils.metrics import (
    evaluate_trajectory, TrajectoryResult, MetricsAggregator
)
from graphnav_gs.utils.visualization import (
    plot_trajectory_3d, plot_comparison_trajectories,
    plot_graph_structure, plot_metrics_comparison, set_publication_style
)
from graphnav_gs.utils.data_io import (
    load_config, save_results_json, load_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='GraphNav-GS: Graph-Augmented Navigation in GS Maps'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml')
    parser.add_argument('--scene', type=str, default=None,
                        help='Path to .ply scene file')
    parser.add_argument('--start', type=float, nargs=3, default=None,
                        help='Start position (x y z)')
    parser.add_argument('--goal', type=float, nargs=3, default=None,
                        help='Goal position (x y z)')
    parser.add_argument('--mode', type=str, default='plan',
                        choices=['plan', 'eval', 'ablation', 'graph_only', 'baseline'],
                        help='Run mode')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Compute device selection')
    parser.add_argument('--num_trials', type=int, default=50,
                        help='Number of random trials for eval mode')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to GAT checkpoint for loading')
    parser.add_argument('--neural_astar_checkpoint', type=str, default=None,
                        help='Path to NeuralAStar checkpoint')
    parser.add_argument('--scene_name', type=str, default=None,
                        help='Scene name (for config scene presets)')
    return parser.parse_args()


def _resolve_gs_cfg(cfg: dict, scene_name: str = None) -> dict:
    """Resolve GS-to-Graph config, optionally overridden by scene preset."""
    gs_cfg = dict(cfg.get('gs_to_graph', {}))
    if scene_name and scene_name in cfg.get('scenes', {}):
        sc = cfg['scenes'][scene_name]
        for k in ('voxel_size', 'robot_radius', 'density_high', 'density_low',
                  'truncation_sigma', 'chunk_size', 'hash_cell_size', 'obs_expansion'):
            if k in sc:
                gs_cfg[k] = sc[k]
        if isinstance(sc.get('edge', None), dict):
            base_edge = dict(gs_cfg.get('edge', {}))
            base_edge.update(sc.get('edge', {}))
            gs_cfg['edge'] = base_edge
    return gs_cfg


def _resample_polyline(points: np.ndarray, step: float = 0.02) -> np.ndarray:
    """Resample a polyline with uniform arc-length spacing."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim == 2 and pts.shape[1] > 3:
        pts = pts[:, :3]
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts[:1, :3], 2, axis=0).astype(np.float32)

    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(arc[-1])
    if total <= 1e-8:
        return np.repeat(pts[:1, :3], 2, axis=0).astype(np.float32)

    step = max(float(step), 1e-3)
    n_samples = int(max(2, np.ceil(total / step) + 1))
    targets = np.linspace(0.0, total, n_samples, dtype=np.float32)
    sampled = np.zeros((n_samples, 3), dtype=np.float32)
    for d in range(3):
        sampled[:, d] = np.interp(targets, arc, pts[:, d]).astype(np.float32)
    return sampled


def _smooth_graphnav_path(trajectory: np.ndarray,
                          start: np.ndarray,
                          goal: np.ndarray,
                          occupied_positions: np.ndarray,
                          robot_radius: float,
                          use_los: bool = True,
                          use_spline: bool = True,
                          los_step: float = 0.02,
                          interp_step: float = 0.02,
                          spline_samples: int = 120,
                          safety_margin: float = 1.05):
    """Reuse the baseline smoothing helper when available, with a safe fallback."""
    try:
        from graphnav_gs.baselines import _smooth_graph_trajectory

        return _smooth_graph_trajectory(
            trajectory=trajectory,
            start=start,
            goal=goal,
            occupied_positions=occupied_positions,
            robot_radius=robot_radius,
            use_los=use_los,
            use_spline=use_spline,
            los_step=los_step,
            interp_step=interp_step,
            spline_samples=spline_samples,
            safety_margin=safety_margin,
        )
    except Exception as exc:
        traj = np.asarray(trajectory, dtype=np.float32)
        start = np.asarray(start, dtype=np.float32).reshape(3)
        goal = np.asarray(goal, dtype=np.float32).reshape(3)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        if traj.ndim == 2 and traj.shape[1] > 3:
            traj = traj[:, :3]
        if traj.shape[0] == 0:
            traj = np.vstack([start, goal]).astype(np.float32)

        if np.linalg.norm(traj[0] - start) > 1e-9:
            traj = np.vstack([start, traj]).astype(np.float32)
        else:
            traj[0] = start
        if np.linalg.norm(traj[-1] - goal) > 1e-9:
            traj = np.vstack([traj, goal]).astype(np.float32)
        else:
            traj[-1] = goal

        resampled = _resample_polyline(traj, step=interp_step)
        return resampled, {
            'raw_points': int(traj.shape[0]),
            'simplified_points': int(traj.shape[0]),
            'final_points': int(resampled.shape[0]),
            'spline_used': False,
            'variant': 'linear_resample',
            'fallback_reason': str(exc),
        }


def setup_pipeline(cfg: dict, device: str = 'cuda', scene_name: str = None,
                   include_converter: bool = True,
                   include_gat_net: bool = True):
    """Initialize all pipeline modules from config."""
    print("=" * 60)
    print("GraphNav-GS Pipeline Initialization")
    print("=" * 60)

    converter = None
    if include_converter:
        # M1: GS-to-Graph converter
        gs_cfg = _resolve_gs_cfg(cfg, scene_name=scene_name)
        edge_cfg = gs_cfg.get('edge', {})
        converter = GSToGraphConverter(
            voxel_size=gs_cfg.get('voxel_size', 0.1),
            robot_radius=gs_cfg.get('robot_radius', 0.02),
            density_high=gs_cfg.get('density_high', 0.5),
            density_low=gs_cfg.get('density_low', 0.1),
            truncation_sigma=gs_cfg.get('truncation_sigma', 3.0),
            chunk_size=gs_cfg.get('chunk_size', 50000),
            hash_cell_size=gs_cfg.get('hash_cell_size', 0.3),
            obs_expansion=gs_cfg.get('obs_expansion', 1),
            edge_lambda_dist=edge_cfg.get('lambda_dist', 0.5),
            edge_lambda_conf=edge_cfg.get('lambda_conf', 0.3),
            edge_lambda_obs=edge_cfg.get('lambda_obs', 0.2),
            edge_k_neighbors=edge_cfg.get('k_neighbors', 6),
            device=device,
        )

    # M2: GAT-A* searcher
    gat_cfg = cfg.get('gat_astar', {})
    gat_net = None
    if include_gat_net:
        print(f"[Pipeline] Initializing GAT heuristic network on {device}...", flush=True)
        gat_net = GATHeuristicNet(
            in_dim=16,
            hidden_dim=gat_cfg.get('hidden_dim', 64),
            num_heads=gat_cfg.get('num_heads', 4),
            num_layers=gat_cfg.get('num_layers', 2),
            dropout=gat_cfg.get('dropout', 0.1),
        ).to(device)
    else:
        print("[Pipeline] Skipping GAT heuristic network init (baseline-safe mode).", flush=True)

    searcher = GATAStar(
        gat_net=gat_net,
        use_uncertainty=gat_cfg.get('use_uncertainty', True),
        alpha=gat_cfg.get('alpha', 1.0),
        device=device,
        z_penalty_weight=gat_cfg.get('z_penalty_weight', 1.2),
        z_reference_weight=gat_cfg.get('z_reference_weight', 0.0),
        line_bias_weight=gat_cfg.get('line_bias_weight', 0.0),
        length_penalty_weight=gat_cfg.get('length_penalty_weight', 0.5),
        warn_snap_distance=gat_cfg.get('warn_snap_distance', 0.15),
        max_snap_distance=gat_cfg.get('max_snap_distance', 0.35),
    )

    # M3: Corridor extractor + B-Spline optimizer
    corr_cfg = cfg.get('graph_corridor', {})
    corridor_extractor = GraphCorridorExtractor(
        uncertainty_width_factor=corr_cfg.get('uncertainty_width_factor', 0.8),
        min_corridor_width=corr_cfg.get('min_corridor_width', 0.05),
        corridor_margin=corr_cfg.get('corridor_margin', 0.02),
        device=device,
    )

    spline_optimizer = GraphSplineOptimizer(
        spline_deg=corr_cfg.get('spline_deg', 6),
        n_sec=corr_cfg.get('n_sec', 10),
        device=device,
    )

    print("[Pipeline] All modules initialized successfully")
    return converter, searcher, corridor_extractor, spline_optimizer


def _cuda_arch_supported() -> tuple[bool, str]:
    """Return whether the active CUDA device is supported by this PyTorch build.

    CUDA hardware forward-compatibility: a PyTorch build compiled for sm_X0
    can execute on sm_X1/sm_X2/... of the same generation.  We therefore
    accept the GPU as long as any sm_{major}* entry is present in the arch
    list, or a real CUDA kernel can be dispatched successfully.
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())

        if supported and arch not in supported:
            # Accept same-generation compatibility (e.g. sm_61 with sm_60 build)
            same_gen = any(s.startswith(f"sm_{major}") for s in supported)
            if not same_gen:
                # Last resort: try dispatching an actual CUDA op
                try:
                    torch.zeros(1, device='cuda')
                    same_gen = True
                except Exception:
                    pass
            if not same_gen:
                supported_list = ", ".join(sorted(supported))
                return False, (
                    f"GPU arch {arch} is not included in this PyTorch build "
                    f"(supported: {supported_list})"
                )
    except Exception as exc:
        return True, f"could not verify CUDA arch compatibility: {exc}"

    return True, ""


def resolve_device(requested: str) -> str:
    """Resolve the actual compute device to use."""
    if requested == 'cpu':
        return 'cpu'

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this runtime')

    supported, reason = _cuda_arch_supported()
    if not supported:
        raise RuntimeError(reason)

    return 'cuda'


def run_single_plan(converter, searcher, corridor_extractor, spline_optimizer,
                     model: GaussianModel, start: np.ndarray, goal: np.ndarray,
                     scene_name: str = 'scene', visualize: bool = False,
                     output_dir: str = './output') -> TrajectoryResult:
    """
    Execute full pipeline for a single start-goal pair.

    Returns:
        TrajectoryResult with all metrics
    """
    start_time = time.time()
    compute_device = torch.device(getattr(spline_optimizer, 'device', getattr(converter, 'device', 'cpu')))

    # Step 1: GS -> Graph
    print(f"\n[Step 1] GS-to-Graph conversion...")
    t1 = time.time()
    graph = converter.convert(model)
    print(f"  -> Graph built in {(time.time()-t1)*1000:.1f} ms")
    print(f"     Nodes: {graph.num_nodes} | Edges: {graph.num_edges}")

    # Step 2: GAT-A* Search
    print(f"[Step 2] GAT-A* path search...")
    t2 = time.time()
    search_result = searcher.search(graph, start, goal, metadata=getattr(graph, 'graph_meta', None))

    if not search_result['found']:
        print("  -> No path found!")
        return TrajectoryResult(
            success=False,
            planning_time=(time.time() - start_time) * 1000,
            search_nodes=search_result.get('nodes_expanded', 0),
        )

    path_nodes = search_result['path_nodes']
    path_coords = search_result['path_coords']
    search_nodes = search_result['nodes_expanded']
    print(f"  -> Path found in {(time.time()-t2)*1000:.1f} ms, "
          f"{len(path_nodes)} nodes, {search_nodes} expanded")

    # Step 3: Corridor Extraction
    print(f"[Step 3] Graph corridor extraction...")
    t3 = time.time()
    corridors = corridor_extractor.extract(
        graph, path_nodes, torch.as_tensor(path_coords, dtype=torch.float32, device=compute_device),
        search_result['path_types'], getattr(graph, 'graph_meta', {}),
        voxel_size=float(getattr(graph, 'graph_meta', {}).get('voxel_size', 0.1))
    )
    print(f"  -> {len(corridors)} corridors in {(time.time()-t3)*1000:.1f} ms")

    # Step 4: B-Spline Optimization
    print(f"[Step 4] B-Spline trajectory optimization...")
    t4 = time.time()
    trajectory = None
    opt_success = False
    try:
        x0 = torch.as_tensor(start, dtype=torch.float32, device=compute_device)
        xf = torch.as_tensor(goal, dtype=torch.float32, device=compute_device)
        trajectory, opt_success = spline_optimizer.optimize(
            corridors, x0, xf,
            path_coords=torch.as_tensor(path_coords, dtype=torch.float32, device=compute_device)
        )
        m3_timing_ms = dict(getattr(spline_optimizer, 'last_timing_ms', {}) or {})
        if m3_timing_ms:
            print(f"  -> M3 timing ms: {m3_timing_ms}")
    except Exception as e:
        print(f"  -> B-Spline optimization error: {e}")

    if trajectory is None or not opt_success:
        print("  -> B-Spline optimization failed, using smoothed fallback")
        trajectory, _ = _smooth_graphnav_path(
            trajectory=path_coords,
            start=start,
            goal=goal,
            occupied_positions=getattr(graph, 'occupied_positions', None),
            robot_radius=float(converter.robot_radius),
            use_los=True,
            use_spline=True,
            los_step=0.02,
            interp_step=0.02,
            spline_samples=120,
            safety_margin=1.05,
        )

    print(f"  -> Trajectory ({len(trajectory)} points) in {(time.time()-t4)*1000:.1f} ms")

    # B-spline may return 12-dim (pos,vel,acc,jerk); extract positions only
    traj_pos = trajectory
    if hasattr(trajectory, 'shape') and len(trajectory.shape) == 2 and trajectory.shape[1] > 3:
        traj_pos = trajectory[:, :3]
    if hasattr(traj_pos, 'detach'):
        traj_pos = traj_pos.detach().cpu().numpy()
    else:
        traj_pos = np.asarray(traj_pos, dtype=np.float32)

    # Step 5: Evaluate
    total_time = (time.time() - start_time) * 1000
    result = evaluate_trajectory(
        trajectory=traj_pos,
        start_time=start_time,
        goal=goal,
        occupied_positions=graph.occupied_positions if hasattr(graph, 'occupied_positions') else None,
        robot_radius=converter.robot_radius,
        search_nodes=search_nodes,
    )
    result.m3_timing_ms = dict(getattr(spline_optimizer, 'last_timing_ms', {}) or {})

    print(f"\n[Result] Success: {result.success} | "
          f"Path length: {result.path_length:.3f}m | "
          f"Time: {total_time:.1f}ms")

    # Visualization
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        plot_trajectory_3d(
            traj_pos, start, goal,
            title=f'GraphNav-GS: {scene_name}',
            save_path=os.path.join(output_dir, f'{scene_name}_trajectory.png')
        )

    return result


def run_eval_mode(cfg: dict, scene_path: str, num_trials: int,
                   scene_name: str = 'scene', device: str = 'cuda',
                   visualize: bool = False, output_dir: str = './output'):
    """Run multiple random start-goal trials and aggregate results."""
    print(f"\n{'='*60}")
    print(f"Evaluation Mode: {num_trials} trials on {scene_name}")
    print(f"{'='*60}")

    # Load scene from .ply only
    model = load_ply(scene_path)

    # Apply scene-specific config if available
    if scene_name in cfg.get('scenes', {}):
        scene_cfg = cfg['scenes'][scene_name]
        # Filter by bounds only when explicitly enabled.
        if scene_cfg.get('apply_bounds_filter', True) and 'lower_bound' in scene_cfg and 'upper_bound' in scene_cfg:
            from graphnav_gs.utils.gs_loader import filter_gaussians_by_bounds
            lower = np.array(scene_cfg['lower_bound'])
            upper = np.array(scene_cfg['upper_bound'])
            model = filter_gaussians_by_bounds(model, lower, upper)

    # Setup pipeline
    converter, searcher, corr_ext, spline_opt = setup_pipeline(cfg, device, scene_name=scene_name)

    # Load GAT checkpoint if provided
    checkpoint_path = os.path.join(output_dir, f'{scene_name}_gat_best.pth')
    if os.path.exists(checkpoint_path):
        print(f"[Eval] Loading GAT checkpoint: {checkpoint_path}")
        epoch, loss = load_checkpoint(checkpoint_path, searcher.gat_net)

    # Generate random start-goal pairs
    bounds = model.bounds
    aggregator = MetricsAggregator()

    for trial in range(num_trials):
        # Random within bounds with margin
        margin = 0.1
        start = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
        goal = np.random.uniform(bounds[0] + margin, bounds[1] - margin)

        # Ensure start != goal
        while np.linalg.norm(start - goal) < 0.3:
            goal = np.random.uniform(bounds[0] + margin, bounds[1] - margin)

        print(f"\n--- Trial {trial+1}/{num_trials} ---")
        result = run_single_plan(
            converter, searcher, corr_ext, spline_opt,
            model, start, goal,
            scene_name=f'{scene_name}_t{trial}',
            visualize=(trial < 3 and visualize),  # Only visualize first 3
            output_dir=os.path.join(output_dir, scene_name),
        )
        aggregator.add(result)

    # Summary
    summary = aggregator.summarize()
    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {scene_name}")
    print(f"{'='*60}")
    print(aggregator.format_summary(summary))

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_results_json(
        {'scene': scene_name, 'num_trials': num_trials, 'summary': summary},
        os.path.join(output_dir, f'{scene_name}_results.json')
    )
    aggregator.to_csv(os.path.join(output_dir, f'{scene_name}_trials.csv'))

    return summary



def run_baseline_comparison(cfg: dict, scene_path: str,
                            scene_name: str = 'scene',
                            start=None, goal=None,
                            device: str = 'cuda',
                            output_dir: str = './output',
                            checkpoint: str = None,
                            neural_astar_checkpoint: str = None):
    """Run all baselines plus GraphNav-GS for comparison."""
    from graphnav_gs.baselines import (
        get_available_baselines, run_all_baselines
    )

    baselines_cfg = cfg.get('baselines', {})

    graph_device_env = os.environ.get('GRAPHNAV_BASELINE_GRAPH_DEVICE', '').strip().lower()
    if graph_device_env:
        graph_device = graph_device_env
    else:
        graph_device = device

    print(f"\n{'='*60}")
    print("GraphNav-GS: Baseline Comparison Mode")
    print(f"{'='*60}")
    print("[Baseline] graph/planner default scene source = scene (--scene ply)")
    print("[Baseline] Built-in scenes prefer baseyuan nerfstudio for SplatNav/SPLANNING when available.")
    if graph_device != device:
        print(f"[Baseline] graph build device override: {graph_device} (planner device: {device})")
    if (not graph_device_env) and str(device).lower() == 'cuda':
        print('[Baseline] Graph voxelization now runs on CUDA by default.')
        print('[Baseline] Set GRAPHNAV_BASELINE_GRAPH_DEVICE=cpu to force CPU graph build.')
    os.makedirs(output_dir, exist_ok=True)

    # Load scene from .ply only
    model = load_ply(scene_path)
    if scene_name in cfg.get('scenes', {}):
        scene_cfg = cfg['scenes'][scene_name]
        if scene_cfg.get('apply_bounds_filter', True) and 'lower_bound' in scene_cfg and 'upper_bound' in scene_cfg:
            from graphnav_gs.utils.gs_loader import filter_gaussians_by_bounds
            lower = np.array(scene_cfg['lower_bound'], dtype=np.float32)
            upper = np.array(scene_cfg['upper_bound'], dtype=np.float32)
            model = filter_gaussians_by_bounds(model, lower, upper)

    # Build graph once using only the converter to keep GPU memory available for baselines.
    gs_cfg = _resolve_gs_cfg(cfg, scene_name=scene_name)
    edge_cfg = gs_cfg.get('edge', {})

    # Get bounds from scene config if available
    scene_bounds_lower = None
    scene_bounds_upper = None
    if scene_name in cfg.get('scenes', {}):
        sc = cfg['scenes'][scene_name]
        if 'lower_bound' in sc and 'upper_bound' in sc:
            scene_bounds_lower = tuple(sc['lower_bound'])
            scene_bounds_upper = tuple(sc['upper_bound'])

    print(f"[Baseline] Initializing graph converter on {graph_device}...", flush=True)
    converter = GSToGraphConverter(
        voxel_size=gs_cfg.get('voxel_size', 0.1),
        robot_radius=gs_cfg.get('robot_radius', 0.02),
        density_high=gs_cfg.get('density_high', 0.5),
        density_low=gs_cfg.get('density_low', 0.1),
        truncation_sigma=gs_cfg.get('truncation_sigma', 3.0),
        chunk_size=gs_cfg.get('chunk_size', 50000),
        hash_cell_size=gs_cfg.get('hash_cell_size', 0.3),
        obs_expansion=gs_cfg.get('obs_expansion', 1),
        edge_lambda_dist=edge_cfg.get('lambda_dist', 0.5),
        edge_lambda_conf=edge_cfg.get('lambda_conf', 0.3),
        edge_lambda_obs=edge_cfg.get('lambda_obs', 0.2),
        edge_k_neighbors=edge_cfg.get('k_neighbors', 6),
        device=graph_device,
        lower_bound=scene_bounds_lower,
        upper_bound=scene_bounds_upper,
    )
    print("[Baseline] Converter initialized. Building graph...", flush=True)
    graph = converter.convert(model)
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    converter_robot_radius = converter.robot_radius
    del converter
    if graph_device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get scene bounds for random query generation
    bounds = graph.bounds if hasattr(graph, 'bounds') else None

    # Get available baselines
    gat_checkpoint = checkpoint if checkpoint else os.path.join(output_dir, f'{scene_name}_gat_best.pth')
    # Use provided checkpoint path, otherwise auto-generated path
    if neural_astar_checkpoint is None:
        neural_astar_checkpoint = os.path.join(output_dir, f'{scene_name}_neural_astar.pth')
    baselines = get_available_baselines(
        splatnav_root=baselines_cfg.get('splatnav_root', ''),
        foci_root=baselines_cfg.get('foci_root', ''),
        splanning_root=baselines_cfg.get('splanning_root', ''),
        gat_checkpoint=gat_checkpoint if os.path.exists(gat_checkpoint) else '',
        neural_astar_checkpoint=neural_astar_checkpoint if os.path.exists(neural_astar_checkpoint) else '',
        rrt_max_iter=int(baselines_cfg.get('rrt', {}).get('max_iter', 5000)),
        rrt_step_size=float(baselines_cfg.get('rrt', {}).get('step_size', 0.05)),
        rrt_goal_tolerance=float(baselines_cfg.get('rrt', {}).get('goal_tolerance', 0.1)),
        rrt_rewiring_radius=float(baselines_cfg.get('rrt', {}).get('rewiring_radius', 0.5)),
        rrt_no_improve_patience=int(baselines_cfg.get('rrt', {}).get('no_improve_patience', 20)),
        rrt_improve_epsilon=float(baselines_cfg.get('rrt', {}).get('improve_epsilon', 1e-4)),
        rrt_goal_bias=float(baselines_cfg.get('rrt', {}).get('goal_bias', 0.15)),
    )
    print(f"\nAvailable baselines: {list(baselines.keys())}")

    # Generate queries
    if start is not None and goal is not None:
        queries = [(np.array(start, dtype=np.float32),
                    np.array(goal, dtype=np.float32))]
    elif bounds is not None:
        margin = 0.1 * (bounds[1] - bounds[0]).max()
        queries = []
        for _ in range(50):
            s = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
            g = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
            while np.linalg.norm(s - g) < 0.3:
                g = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
            queries.append((s, g))
    else:
        print("Error: provide --start/--goal or scene with valid bounds")
        return

    # Train NeuralAStar if no checkpoint exists
    baselines_to_run = baselines  # copy reference
    if 'Neural A*' in baselines and not os.path.exists(neural_astar_checkpoint):
        print(f"\n[NeuralAStar] No checkpoint found, skipping Neural A*...")
        baselines_to_run = {k: v for k, v in baselines.items() if k != 'Neural A*'}
        if not baselines_to_run:
            print("[Baseline] No baselines available to run.")
            return

    # Initialize the heavier GraphNav-GS modules only after baselines finish.
    _, searcher, corr_ext, spline_opt = setup_pipeline(
        cfg,
        device,
        scene_name=scene_name,
        include_converter=False,
        include_gat_net=True,
    )

    checkpoint_path = gat_checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[Baseline] Loading GAT checkpoint: {checkpoint_path}")
        load_checkpoint(checkpoint_path, searcher.gat_net)

    def _run_graphnav_on_graph(start_pos, goal_pos):
        t0 = time.time()
        device_t = torch.device(device)
        search_result = searcher.search(
            graph, start_pos, goal_pos,
            metadata=getattr(graph, 'graph_meta', None),
            max_expand=int(cfg.get('gat_astar', {}).get('max_expand', 50000)),
            max_wall_time_s=float(cfg.get('gat_astar', {}).get('max_time_s', 20.0)),
        )
        if not search_result['found']:
            return {
                'found': False,
                'trajectory': None,
                'planning_time_ms': (time.time() - t0) * 1000,
                'nodes_expanded': search_result.get('nodes_expanded', 0),
                'path_length': 0.0,
                'reason': search_result.get('reason', 'search failed'),
            }

        path_nodes = search_result['path_nodes']
        path_coords = search_result['path_coords']
        path_types = search_result['path_types']
        corridors = corr_ext.extract(
            graph, path_nodes, torch.as_tensor(path_coords, dtype=torch.float32, device=device_t),
            path_types, getattr(graph, 'graph_meta', {}),
            voxel_size=float(getattr(graph, 'graph_meta', {}).get('voxel_size', 0.1))
        )

        opt_success = False
        trajectory = None
        try:
            x0 = torch.as_tensor(start_pos, dtype=torch.float32, device=device_t)
            xf = torch.as_tensor(goal_pos, dtype=torch.float32, device=device_t)
            trajectory, opt_success = spline_opt.optimize(
                corridors, x0, xf,
                path_coords=torch.as_tensor(path_coords, dtype=torch.float32, device=device_t)
            )
            m3_timing_ms = dict(getattr(spline_opt, 'last_timing_ms', {}) or {})
            if m3_timing_ms:
                print(f"    -> M3 timing ms: {m3_timing_ms}")
        except Exception as exc:
            print(f"    -> B-Spline optimization error: {exc}")

        smooth_info = None
        if trajectory is None or not opt_success:
            trajectory, smooth_info = _smooth_graphnav_path(
                trajectory=path_coords,
                start=start_pos,
                goal=goal_pos,
                occupied_positions=getattr(graph, 'occupied_positions', None),
                robot_radius=float(converter_robot_radius),
                use_los=True,
                use_spline=True,
                los_step=0.02,
                interp_step=0.02,
                spline_samples=120,
                safety_margin=1.05,
            )

        if hasattr(trajectory, 'shape') and len(trajectory.shape) == 2 and trajectory.shape[1] > 3:
            trajectory = trajectory[:, :3]
        if hasattr(trajectory, 'detach'):
            traj_np = trajectory.detach().cpu().numpy()
        else:
            traj_np = np.asarray(trajectory)
        traj_np = np.asarray(traj_np, dtype=np.float32)
        if traj_np.ndim == 1:
            traj_np = traj_np.reshape(1, -1)
        if traj_np.ndim == 2 and traj_np.shape[1] > 3:
            traj_np = traj_np[:, :3]
        if traj_np.shape[0] == 0:
            traj_np = np.vstack([
                np.asarray(start_pos, dtype=np.float32).reshape(1, 3),
                np.asarray(goal_pos, dtype=np.float32).reshape(1, 3),
            ])
        else:
            traj_np[0, :3] = np.asarray(start_pos, dtype=np.float32).reshape(3)
            traj_np[-1, :3] = np.asarray(goal_pos, dtype=np.float32).reshape(3)

        raw_path_points = int(np.asarray(path_coords, dtype=np.float32).shape[0])
        if smooth_info is None:
            smooth_info = {
                'raw_points': raw_path_points,
                'simplified_points': raw_path_points,
                'final_points': int(traj_np.shape[0]),
                'spline_used': bool(opt_success),
            }
        else:
            smooth_info['raw_points'] = raw_path_points
            smooth_info['final_points'] = int(traj_np.shape[0])
            smooth_info['spline_used'] = bool(smooth_info.get('spline_used', False) or opt_success)

        eval_result = evaluate_trajectory(
            trajectory=traj_np,
            start_time=t0,
            goal=goal_pos,
            occupied_positions=graph.occupied_positions if hasattr(graph, 'occupied_positions') else None,
            robot_radius=converter_robot_radius,
            search_nodes=search_result['nodes_expanded'],
        )
        return {
            'found': eval_result.success,
            'trajectory': traj_np,
            'planning_time_ms': eval_result.planning_time,
            'nodes_expanded': search_result['nodes_expanded'],
            'm3_timing_ms': dict(getattr(spline_opt, 'last_timing_ms', {}) or {}),
            'path_length': eval_result.path_length,
            'min_collision_dist': eval_result.min_collision_dist,
            'smoothness': eval_result.smoothness,
            'search_nodes': eval_result.search_nodes,
            'path_points': eval_result.path_points,
            'path_nodes': path_nodes,
            'path_types': path_types,
            'reason': 'GraphNav-GS',
            'path_smoothing': smooth_info,
            'path_smoothing_used': True,
            'start_snap_distance': float(search_result.get('start_snap_distance', 0.0)),
            'goal_snap_distance': float(search_result.get('goal_snap_distance', 0.0)),
        }

    # Run comparisons
    all_results = {}
    all_summaries = {}
    for i, (s, g) in enumerate(queries):
        print(f"\n--- Query {i+1}/{len(queries)} ---")
        # Convert model to numpy for occupancy
        occupied_positions = (
            model.positions.cpu().numpy() if hasattr(model.positions, 'cpu')
            else np.array(model.positions if model.positions is not None else model.means)
        )

        results = run_all_baselines(
            baselines_to_run, s, g,
            graph=graph,
            gs_scene=model,
            device=device,
            occupied_positions=occupied_positions,
            gsplat_path=scene_path,
            ply_file=scene_path,
            scene_name=scene_name,
            warn_snap_distance=float(cfg.get('gat_astar', {}).get('warn_snap_distance', 0.15)),
            max_snap_distance=float(cfg.get('gat_astar', {}).get('max_snap_distance', 0.35)),
            robot_radius=float(converter_robot_radius),
            astar_use_los=bool(cfg.get('baselines', {}).get('astar', {}).get('use_los', True)),
            astar_use_spline=bool(cfg.get('baselines', {}).get('astar', {}).get('use_spline', True)),
            astar_los_step=float(cfg.get('baselines', {}).get('astar', {}).get('los_step', 0.02)),
            astar_interp_step=float(cfg.get('baselines', {}).get('astar', {}).get('interp_step', 0.02)),
            astar_spline_samples=int(cfg.get('baselines', {}).get('astar', {}).get('spline_samples', 120)),
            astar_safety_margin=float(cfg.get('baselines', {}).get('astar', {}).get('safety_margin', 1.05)),
            astar_z_ref_weight=float(cfg.get('baselines', {}).get('astar', {}).get('z_ref_weight', 0.0)),
            astar_line_bias_weight=float(cfg.get('baselines', {}).get('astar', {}).get('line_bias_weight', 0.0)),
            astar_outside_bbox_weight=float(cfg.get('baselines', {}).get('astar', {}).get('outside_bbox_weight', 0.0)),
            astar_outside_bbox_margin=float(cfg.get('baselines', {}).get('astar', {}).get('outside_bbox_margin', 0.02)),
            astar_corridor_radius=float(cfg.get('baselines', {}).get('astar', {}).get('corridor_radius', 0.0)),
            astar_corridor_weight=float(cfg.get('baselines', {}).get('astar', {}).get('corridor_weight', 0.0)),
            astar_max_line_deviation=cfg.get('baselines', {}).get('astar', {}).get('max_line_deviation', None),
            astar_min_z=cfg.get('baselines', {}).get('astar', {}).get('min_z', None),
            astar_max_z=cfg.get('baselines', {}).get('astar', {}).get('max_z', None),
            astar_reference_z=float(cfg.get('baselines', {}).get('astar', {}).get('reference_z', 0.5 * (s[2] + g[2]))),
            rrt_goal_tolerance=float(cfg.get('baselines', {}).get('rrt', {}).get('goal_tolerance', 0.06)),
            rrt_step_size=float(cfg.get('baselines', {}).get('rrt', {}).get('step_size', 0.05)),
            rrt_rewiring_radius=float(cfg.get('baselines', {}).get('rrt', {}).get('rewiring_radius', 0.3)),
            rrt_obstacle_clearance=float(cfg.get('baselines', {}).get('rrt', {}).get('obstacle_clearance', max(float(converter_robot_radius) * 2.0, 0.025))),
            goal_tolerance=float(cfg.get('baselines', {}).get('goal_tolerance', 0.1)),
            splatnav_max_gaussians=int(cfg.get('baselines', {}).get('splatnav', {}).get('max_gaussians', 120000)),
            # Keep SplatNav GPU-first by default. Only force CPU when the
            # config explicitly asks for it.
            splatnav_force_cpu=bool(cfg.get('baselines', {}).get('splatnav', {}).get('force_cpu', False)),
            splatnav_allow_cpu_fallback=bool(cfg.get('baselines', {}).get('splatnav', {}).get('allow_cpu_fallback', False)),
            splatnav_device=cfg.get('baselines', {}).get('splatnav', {}).get('device', None),
            splatnav_debug=bool(cfg.get('baselines', {}).get('splatnav', {}).get('debug', False)),
            splanning_robot_radius=float(converter_robot_radius),
            splanning_geometry_collision_weight=float(cfg.get('baselines', {}).get('splanning', {}).get('geometry_collision_weight', 20.0)),
            splanning_geometry_clearance_margin=float(cfg.get('baselines', {}).get('splanning', {}).get('geometry_clearance_margin', 1.1)),
            splanning_geometry_sample_step=float(cfg.get('baselines', {}).get('splanning', {}).get('geometry_sample_step', 0.02)),
            splanning_waypoint_margin=float(cfg.get('baselines', {}).get('splanning', {}).get('waypoint_margin', 0.3)),
            splanning_path_length_weight=float(cfg.get('baselines', {}).get('splanning', {}).get('path_length_weight', 0.0)),
            splanning_curvature_weight=float(cfg.get('baselines', {}).get('splanning', {}).get('curvature_weight', 0.0)),
            splanning_z_ref_weight=float(cfg.get('baselines', {}).get('splanning', {}).get('z_ref_weight', 0.0)),
            splanning_line_bias_weight=float(cfg.get('baselines', {}).get('splanning', {}).get('line_bias_weight', 0.0)),
            splanning_outside_bbox_weight=float(cfg.get('baselines', {}).get('splanning', {}).get('outside_bbox_weight', 0.0)),
            splanning_outside_bbox_margin=float(cfg.get('baselines', {}).get('splanning', {}).get('outside_bbox_margin', 0.02)),
            splanning_max_line_deviation=cfg.get('baselines', {}).get('splanning', {}).get('max_line_deviation', None),
            splanning_interp_step=float(cfg.get('baselines', {}).get('splanning', {}).get('interp_step', cfg.get('baselines', {}).get('splanning', {}).get('geometry_sample_step', 0.02))),
            splanning_no_improve_patience=int(cfg.get('baselines', {}).get('splanning', {}).get('no_improve_patience', 15)),
            splanning_improve_epsilon=float(cfg.get('baselines', {}).get('splanning', {}).get('improve_epsilon', 1e-5)),
            splanning_min_iter_before_stop=int(cfg.get('baselines', {}).get('splanning', {}).get('min_iter_before_stop', 5)),
            splanning_device=cfg.get('baselines', {}).get('splanning', {}).get('device', device),
            isolate_crash_prone_baselines=bool(cfg.get('baselines', {}).get('isolate_crash_prone', True)),
            isolated_baseline_names=cfg.get('baselines', {}).get(
                'isolated_baseline_names',
                ['SplatNav', 'FOCI'],
            ),
            isolated_baseline_timeout_s=float(cfg.get('baselines', {}).get('isolated_baseline_timeout_s', 180.0)),
            isolated_start_method=str(cfg.get('baselines', {}).get('isolated_start_method', 'auto')),
        )
        print('  Running GraphNav-GS...')
        graphnav_result = _run_graphnav_on_graph(s, g)
        graphnav_status = '✓ found' if bool(graphnav_result.get('found', False)) else '✗ not found'
        print(
            f'    {graphnav_status} '
            f'({float(graphnav_result.get("planning_time_ms", 0.0)):.1f} ms, '
            f'path_len={float(graphnav_result.get("path_length", 0.0)):.3f})'
        )
        if not bool(graphnav_result.get('found', False)):
            reason = graphnav_result.get('reason', '')
            if reason:
                print(f'      reason: {reason}')

        query_results = {k: v.to_dict() for k, v in results.items()}
        query_results['GraphNav-GS'] = graphnav_result
        all_results[f'query_{i}'] = {
            'start': s.tolist(), 'goal': g.tolist(),
            'results': query_results,
        }
        all_summaries[f'query_{i}'] = query_results

        trajs = {}
        for name, result in query_results.items():
            traj = result.get('trajectory', None)
            if traj is None:
                continue
            traj = np.asarray(traj)
            if traj.ndim == 2 and traj.shape[1] >= 3 and len(traj) > 1:
                label = name
                if not bool(result.get('found', False)):
                    label = f'{name} (failed)'
                elif bool(result.get('fallback_used', False)):
                    label = f'{name} (fallback)'
                trajs[label] = traj[:, :3]
        if len(trajs) >= 2:
            plot_comparison_trajectories(
                trajs, s, g,
                obstacles=occupied_positions,
                scene_model=model,
                scene_label=f'{scene_name} raw GS model',
                title=f'{scene_name}: Baseline vs GraphNav-GS (query {i})',
                save_path=os.path.join(output_dir, f'{scene_name}_comparison_q{i}.png')
            )

    # Optional summary plot for the last query / single-query case
    if all_summaries:
        last_query = next(reversed(all_summaries.values()))
        metric_summary = {}
        for planner_name, result in last_query.items():
            metric_summary[planner_name] = {
                'success': {'mean': float(bool(result.get('found', False))), 'std': 0.0},
                'path_length': {'mean': float(result.get('path_length', 0.0)), 'std': 0.0},
                'planning_time': {'mean': float(result.get('planning_time_ms', 0.0)), 'std': 0.0},
                'min_collision_dist': {'mean': float(result.get('min_collision_dist', 0.0)), 'std': 0.0},
                'smoothness': {'mean': float(result.get('smoothness', 0.0)), 'std': 0.0},
                'search_nodes': {'mean': float(result.get('search_nodes', 0)), 'std': 0.0},
            }
        plot_metrics_comparison(
            metric_summary,
            metrics_to_plot=['success', 'path_length', 'planning_time', 'min_collision_dist', 'search_nodes'],
            title=f'{scene_name}: Planner Comparison',
            save_path=os.path.join(output_dir, f'{scene_name}_comparison_metrics.png')
        )

    # Save
    save_results_json(
        {'scene': scene_name, 'baselines': list(baselines.keys()) + ['GraphNav-GS'],
         'num_queries': len(queries), 'results': all_results},
        os.path.join(output_dir, f'{scene_name}_baseline_comparison.json')
    )
    print(f"\n[DONE] Baseline comparison saved to {output_dir}")


def run_graph_only(cfg: dict, scene_path: str, scene_name: str = 'scene',
                    device: str = 'cuda', output_dir: str = './output'):
    """Only build graph and save/visualize (no planning)."""
    print(f"\n[Graph-Only Mode] Building graph for {scene_name}")

    model = load_ply(scene_path)

    converter, _, _, _ = setup_pipeline(cfg, device, scene_name=scene_name)
    graph = converter.convert(model)

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    os.makedirs(output_dir, exist_ok=True)

    # Save graph statistics
    save_results_json({
        'scene': scene_name,
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges,
        'node_types': graph.node_type_counts if hasattr(graph, 'node_type_counts') else {},
        'bounds': graph.bounds.tolist() if hasattr(graph, 'bounds') else None,
    }, os.path.join(output_dir, f'{scene_name}_graph_stats.json'))

    # Visualize graph structure
    set_publication_style()
    node_pos = graph.free_positions if hasattr(graph, 'free_positions') else np.zeros((0, 3))
    node_types = graph.free_types if hasattr(graph, 'free_types') else None

    if len(node_pos) > 0 and len(node_pos) < 50000:  # Don't visualize huge graphs
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            edge_idx = graph.edge_index.detach().cpu().numpy() if hasattr(graph.edge_index, 'detach') else np.asarray(graph.edge_index)
        else:
            edge_idx = None
        plot_graph_structure(
            node_pos, node_types, edge_idx,
            title=f'Graph Structure: {scene_name}',
            save_path=os.path.join(output_dir, f'{scene_name}_graph.png')
        )

    return graph


def main():
    args = parse_args()

    # Load config
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    cfg = load_config(cfg_path)

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        args.device = resolve_device(args.device)
    except RuntimeError as exc:
        print(f'[Error] {exc}')
        print('[Error] This shell does not currently expose a usable CUDA runtime.')
        print('[Error] Check `nvidia-smi` and `/dev/nvidia*`, then rerun with the GTX 1080 available.')
        sys.exit(1)

    print(f'[Device] Using {args.device}')
    print(f'[Device] torch={torch.__version__} from {torch.__file__}')

    if args.mode == 'plan':
        """Single planning query."""
        if args.scene is None or args.start is None or args.goal is None:
            print("Error: --scene, --start, and --goal required for plan mode")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("GraphNav-GS: Single Planning Mode")
        print(f"{'='*60}")

        # Load scene from .ply only
        model = load_ply(args.scene)

        start = np.array(args.start, dtype=np.float32)
        goal = np.array(args.goal, dtype=np.float32)

        # Apply scene preset if specified
        if args.scene_name and args.scene_name in cfg.get('scenes', {}):
            from graphnav_gs.utils.gs_loader import filter_gaussians_by_bounds
            sc = cfg['scenes'][args.scene_name]
            if sc.get('apply_bounds_filter', True) and 'lower_bound' in sc:
                model = filter_gaussians_by_bounds(
                    model,
                    np.array(sc['lower_bound']),
                    np.array(sc['upper_bound'])
                )

        converter, searcher, corr_ext, spline_opt = setup_pipeline(
            cfg, args.device, scene_name=args.scene_name
        )

        # Load GAT checkpoint
        if args.checkpoint and os.path.exists(args.checkpoint):
            _, _ = load_checkpoint(args.checkpoint, searcher.gat_net)

        scene_name = args.scene_name or Path(args.scene).stem
        result = run_single_plan(
            converter, searcher, corr_ext, spline_opt,
            model, start, goal,
            scene_name=scene_name,
            visualize=args.visualize,
            output_dir=args.output_dir,
        )

        save_results_json({
            'scene': scene_name,
            'start': start.tolist(),
            'goal': goal.tolist(),
            'success': result.success,
            'path_length': result.path_length,
            'planning_time': result.planning_time,
            'search_nodes': result.search_nodes,
        }, os.path.join(args.output_dir, f'{scene_name}_plan_result.json'))

    elif args.mode == 'eval':
        """Multi-trial evaluation."""
        if args.scene is None:
            print("Error: --scene required for eval mode")
            sys.exit(1)

        run_eval_mode(
            cfg, args.scene,
            num_trials=args.num_trials,
            scene_name=args.scene_name or Path(args.scene).stem,
            device=args.device,
            visualize=args.visualize,
            output_dir=args.output_dir,
        )

    elif args.mode == 'baseline':
        """Run all baselines for comparison."""
        if args.scene is None:
            print("Error: --scene required for baseline mode")
            sys.exit(1)

        run_baseline_comparison(
            cfg, args.scene,
            scene_name=args.scene_name or Path(args.scene).stem,
            start=args.start, goal=args.goal,
            device=args.device,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            neural_astar_checkpoint=args.neural_astar_checkpoint,
        )

    elif args.mode == 'ablation':
        """Ablation study - run all variants."""
        if args.scene is None:
            print("Error: --scene required for ablation mode")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("GraphNav-GS: Ablation Study Mode")
        print(f"{'='*60}")

        from graphnav_gs.ablation import run_ablation_study

        run_ablation_study(
            cfg=cfg,
            scene_path=args.scene,
            scene_name=args.scene_name or Path(args.scene).stem,
            num_trials=args.num_trials,
            device=args.device,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            start=np.array(args.start, dtype=np.float32) if args.start is not None else None,
            goal=np.array(args.goal, dtype=np.float32) if args.goal is not None else None,
        )

    elif args.mode == 'graph_only':
        """Build graph only (for analysis/visualization)."""
        if args.scene is None:
            print("Error: --scene required")
            sys.exit(1)

        run_graph_only(
            cfg, args.scene,
            scene_name=args.scene_name or Path(args.scene).stem,
            device=args.device,
            output_dir=args.output_dir,
        )

    print("\n[DONE] GraphNav-GS pipeline complete.")


if __name__ == '__main__':
    main()
