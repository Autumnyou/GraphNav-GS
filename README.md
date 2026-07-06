# GraphNav-GS

**Heterogeneous Graph Abstraction of Gaussian Splatting Scenes for GNN-Guided Path Search**

GraphNav-GS is a planning framework that enables safe and efficient UAV navigation in 3D Gaussian Splatting (3DGS) scenes. Instead of planning directly on millions of Gaussian primitives, it reorganizes the scene into a structured, uncertainty-aware graph and performs learned heuristic search on that graph.
This paper is currently being submitted to The Visual Computer. The complete code will be made public upon acceptance.

## What It Does

Given a 3DGS reconstruction (a `.ply` file), GraphNav-GS produces a collision-free, smooth continuous trajectory from any start to any goal. 


## Key Insight

Conventional planners typically reduce a 3DGS scene to a binary occupancy map, discarding the rich information encoded in each Gaussian primitive. GraphNav-GS takes the opposite approach: the per-primitive density, opacity, and covariance *already* encode planning-relevant geometry and reconstruction confidence. By reorganizing, rather than discarding, this information into a heterogeneous graph, the planner can reason about traversability at a finer granularity while maintaining computational tractability.


## Reproducing Paper Results

Pre-trained GAT checkpoints for all three scenes are in `checkpoints/`. Deterministic start-goal query lists are in `data/`. Scene `.ply` files are in `data/`.


## License

[To be added]
