"""
GPU shortest-path utilities for goal-conditioned training.

The training loop only needs cost-to-goal labels. This module computes them
with batched GPU relaxation and caches results per snapped goal node.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def build_unified_edge_index(graph, n_free: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a single edge list for the heterogeneous planning graph."""
    edge_list = []
    weight_list = []

    for etype in graph.edge_types:
        src_type, _, dst_type = etype
        rel = graph[etype]
        ei = rel.edge_index.to(device=device, dtype=torch.long)
        ew = rel.edge_weight.to(device=device, dtype=torch.float32)
        off_src = 0 if src_type == 'free' else n_free
        off_dst = 0 if dst_type == 'free' else n_free
        edge_list.append(torch.stack([ei[0] + off_src, ei[1] + off_dst], dim=0))
        weight_list.append(ew)

    if edge_list:
        edge_index = torch.cat(edge_list, dim=1)
        edge_weight = torch.cat(weight_list, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.zeros((0,), dtype=torch.float32, device=device)

    return edge_index, edge_weight


def gpu_shortest_paths_goal_batch(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
    goal_nodes: torch.Tensor,
    max_iter: int | None = None,
    infinity: float = 1e9,
) -> torch.Tensor:
    """
    Batched shortest-path relaxation on GPU.

    Args:
        edge_index: (2, E) directed edge list.
        edge_weight: (E,) nonnegative weights.
        n_nodes: number of nodes in the unified planning graph.
        goal_nodes: (K,) goal node indices.
        max_iter: optional relaxation cap; default is n_nodes - 1.

    Returns:
        (K, n_nodes) cost-to-goal tensor.
    """
    device = edge_index.device
    goal_nodes = torch.as_tensor(goal_nodes, dtype=torch.long, device=device).view(-1)
    k = int(goal_nodes.numel())

    dist = torch.full((k, n_nodes), float(infinity), dtype=torch.float32, device=device)
    if k == 0:
        return dist
    dist[torch.arange(k, device=device), goal_nodes] = 0.0

    if edge_index.numel() == 0:
        return dist

    src = edge_index[0]
    dst = edge_index[1]
    e = int(src.numel())

    if max_iter is None:
        max_iter = n_nodes - 1
    else:
        max_iter = min(int(max_iter), n_nodes - 1)

    batch_offsets = torch.arange(k, device=device, dtype=torch.long) * n_nodes
    flat_src = (batch_offsets[:, None] + src[None, :]).reshape(-1)
    flat_dst = (batch_offsets[:, None] + dst[None, :]).reshape(-1)
    flat_w = edge_weight.unsqueeze(0).expand(k, e).reshape(-1)
    flat_dist = dist.reshape(-1)

    for _ in range(max_iter):
        cand = flat_dist.index_select(0, flat_src) + flat_w
        new_flat = flat_dist.clone()
        new_flat.scatter_reduce_(0, flat_dst, cand, reduce='amin', include_self=True)
        if torch.equal(new_flat, flat_dist):
            break
        flat_dist = new_flat

    return flat_dist.view(k, n_nodes)


def gpu_shortest_paths_single_goal(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
    goal_node: int,
    max_iter: int | None = None,
    infinity: float = 1e9,
) -> torch.Tensor:
    """Convenience wrapper for a single goal."""
    return gpu_shortest_paths_goal_batch(
        edge_index=edge_index,
        edge_weight=edge_weight,
        n_nodes=n_nodes,
        goal_nodes=torch.as_tensor([goal_node], dtype=torch.long, device=edge_index.device),
        max_iter=max_iter,
        infinity=infinity,
    )[0]


class GoalCostCache:
    """Cache for cost-to-goal tensors computed on GPU."""

    def __init__(self):
        self._graph_cache: Dict[Tuple[int, str], Dict[str, object]] = {}

    def _graph_key(self, graph, device: torch.device) -> Tuple[int, str]:
        return (id(graph), str(device))

    def _get_graph_tensors(self, graph, n_free: int, device: torch.device) -> Dict[str, object]:
        key = self._graph_key(graph, device)
        cached = self._graph_cache.get(key)
        if cached is not None:
            return cached

        edge_index, edge_weight = build_unified_edge_index(graph, n_free, device)
        n_frontier = graph['frontier'].x.shape[0]
        n_total = n_free + n_frontier
        cached = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'n_total': n_total,
            'goal_costs': {},
        }
        self._graph_cache[key] = cached
        return cached

    def prime_goals(
        self,
        graph,
        n_free: int,
        goal_indices: torch.Tensor,
        device: torch.device,
        batch_size: int = 32,
    ) -> None:
        self.get_costs_for_goals(graph, n_free, goal_indices, device, batch_size=batch_size)

    def get_cost_for_goal(
        self,
        graph,
        n_free: int,
        goal_index: int,
        device: torch.device,
        batch_size: int = 32,
    ) -> torch.Tensor:
        costs = self.get_costs_for_goals(
            graph=graph,
            n_free=n_free,
            goal_indices=torch.as_tensor([goal_index], dtype=torch.long, device=device),
            device=device,
            batch_size=batch_size,
        )
        return costs[0]

    def get_costs_for_goals(
        self,
        graph,
        n_free: int,
        goal_indices: torch.Tensor,
        device: torch.device,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Get cost-to-goal vectors for one or more snapped goal nodes.

        Results are cached per goal index and reused across epochs.
        """
        cached = self._get_graph_tensors(graph, n_free, device)
        edge_index = cached['edge_index']
        edge_weight = cached['edge_weight']
        n_total = int(cached['n_total'])
        goal_cache: Dict[int, torch.Tensor] = cached['goal_costs']  # type: ignore[assignment]

        goal_list = [int(g) for g in torch.as_tensor(goal_indices, dtype=torch.long).view(-1).tolist()]
        if len(goal_list) == 0:
            return torch.empty((0, n_total), dtype=torch.float32, device=device)

        missing = list(dict.fromkeys(g for g in goal_list if g not in goal_cache))
        if missing:
            step = max(1, int(batch_size))
            for i in range(0, len(missing), step):
                chunk = missing[i:i + step]
                chunk_tensor = torch.as_tensor(chunk, dtype=torch.long, device=device)
                chunk_costs = gpu_shortest_paths_goal_batch(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    n_nodes=n_total,
                    goal_nodes=chunk_tensor,
                )
                for idx, goal_idx in enumerate(chunk):
                    goal_cache[goal_idx] = chunk_costs[idx].detach()

        stacked = [goal_cache[g] for g in goal_list]
        return torch.stack(stacked, dim=0)


_goal_cost_cache = GoalCostCache()


def get_cost_cache() -> GoalCostCache:
    return _goal_cost_cache
