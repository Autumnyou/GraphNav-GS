# GraphNav-GS

**Heterogeneous Graph Abstraction of Gaussian Splatting Scenes for GNN-Guided Path Search**

GraphNav-GS is a planning framework that enables safe and efficient UAV navigation in 3D Gaussian Splatting (3DGS) scenes. Instead of planning directly on millions of Gaussian primitives, it reorganizes the scene into a structured, uncertainty-aware graph and performs learned heuristic search on that graph.

## What It Does

Given a 3DGS reconstruction (a `.ply` file), GraphNav-GS produces a collision-free, smooth continuous trajectory from any start to any goal. The pipeline has three stages:

1. **GS-to-Graph (M1).** Converts the continuous Gaussian scene into a heterogeneous planning graph. Each node encodes local geometry, density statistics, reconstruction confidence, and obstacle proximity. Free-space nodes and frontier nodes participate in search; occupied and unknown regions act as structural constraints.

2. **GAT-A* Search (M2).** A graph attention network learns a goal-conditioned heuristic from the graph structure, which guides a classical A* search. The heuristic adapts to clutter, uncertainty, and topology without sacrificing A*'s transparency.

3. **Trajectory Repair and Smoothing (M3).** Repairs unsafe straight-line segments with detour anchors from nearby free nodes, then smooths the repaired path into a curvature-continuous trajectory suitable for execution.

## Key Insight

Conventional planners typically reduce a 3DGS scene to a binary occupancy map, discarding the rich information encoded in each Gaussian primitive. GraphNav-GS takes the opposite approach: the per-primitive density, opacity, and covariance *already* encode planning-relevant geometry and reconstruction confidence. By reorganizing, rather than discarding, this information into a heterogeneous graph, the planner can reason about traversability at a finer granularity while maintaining computational tractability.

## Quick Start

```bash
pip install -r requirements.txt

python run_graphnav.py \
  --mode plan \
  --scene data/stonehenge.ply \
  --scene_name stonehenge \
  --start -0.45 0.15 0.0 \
  --goal 0.25 0.15 0.0 \
  --checkpoint checkpoints/gat_stonehenge.pth \
  --device cuda
```

## Reproducing Paper Results

Pre-trained GAT checkpoints for all three scenes are in `checkpoints/`. Deterministic start-goal query lists are in `data/`. Scene `.ply` files are in `data/`.

```bash
# Full baseline comparison (50 queries, matched conditions)
python run_splatnav_style.py --scene stonehenge --mode baseline \
  --checkpoint checkpoints/gat_stonehenge.pth \
  --neural_astar_checkpoint checkpoints/neural_astar_stonehenge.pth \
  --device cuda --num_trials 50
```

## Scenes

| Scene | Voxel size | Robot radius | GAT checkpoint |
|---|---|---|---|
| Stonehenge | 0.02 m | 0.02 m | `gat_stonehenge.pth` |
| Statues | 0.03 m | 0.03 m | `gat_statues.pth` |
| Old_union2 | 0.05 m | 0.01 m | `gat_old_union2.pth` |

## Citation

If you use GraphNav-GS in your research, please cite:

```bibtex
@article{gao2026graphnav,
  title={GraphNav-GS: Heterogeneous Graph Abstraction of Gaussian Splatting
         Scenes for GNN-Guided Path Search},
  author={Gao, Qiushuang and Qu, Zhenshen and Gao, Yubo and Yu, Zhiwei and Bai, Jingyuan},
  journal={The Visual Computer},
  year={2026}
}
```

## License

[To be added]
