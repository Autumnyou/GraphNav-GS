"""
GAT Heuristic Network: Graph Attention Network for learned A* heuristic.

Implements iA*-style self-supervised training where the GAT learns
to predict the true shortest path cost, replacing the hand-crafted
Euclidean distance heuristic.

Architecture:
  Input: Node features (16-dim) + goal-relative position (3-dim)
  Layers: 2-layer GAT (4 heads, 64 hidden) + MLP
  Output: Scalar heuristic value h(v, goal)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

try:
    import torch_geometric as pyg
    from torch_geometric.nn import GATConv, HeteroConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    GATConv = None
    HeteroConv = None


def _segment_softmax(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over incoming edges grouped by destination node."""
    if values.numel() == 0:
        return values

    if values.ndim == 1:
        values = values.unsqueeze(-1)

    num_heads = values.shape[1]
    index = index.long()
    index_2d = index.unsqueeze(-1).expand(-1, num_heads)

    max_per_node = torch.full(
        (num_nodes, num_heads),
        float('-inf'),
        device=values.device,
        dtype=values.dtype,
    )
    max_per_node.scatter_reduce_(0, index_2d, values, reduce='amax', include_self=True)

    shifted = values - max_per_node[index]
    shifted = torch.clamp(shifted, min=-60.0, max=60.0)
    exp = torch.exp(shifted)

    denom = torch.zeros((num_nodes, num_heads), device=values.device, dtype=values.dtype)
    denom.index_add_(0, index, exp)

    return exp / (denom[index] + 1e-9)


class _TorchGATLayer(nn.Module):
    """Pure PyTorch multi-head graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.internal_dim = self.out_dim * self.heads

        self.lin = nn.Linear(self.in_dim, self.internal_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, self.heads, self.out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, self.heads, self.out_dim))
        self.att_edge = nn.Parameter(torch.empty(1, self.heads, self.out_dim))
        self.lin_edge = nn.Linear(1, self.internal_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.internal_dim))
        self.out_proj = nn.Identity() if self.internal_dim == self.in_dim else nn.Linear(self.internal_dim, self.in_dim)
        self.norm = nn.LayerNorm(self.internal_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.zeros_(self.bias)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        num_nodes = x.shape[0]
        if num_nodes == 0:
            return (x, None) if return_attention else x

        h = self.lin(x).view(num_nodes, self.heads, self.out_dim)

        if edge_index is None or edge_index.numel() == 0:
            out = h.reshape(num_nodes, self.internal_dim)
            out = self.norm(out + self.bias)
            out = self.out_proj(out)
            return (out, None) if return_attention else out

        edge_index = edge_index.long()
        src = edge_index[0]
        dst = edge_index[1]

        self_loops = torch.arange(num_nodes, device=x.device, dtype=torch.long)
        loop_index = torch.stack([self_loops, self_loops], dim=0)
        src = torch.cat([src, loop_index[0]], dim=0)
        dst = torch.cat([dst, loop_index[1]], dim=0)

        if edge_attr is not None:
            edge_attr = edge_attr.reshape(-1, 1).to(dtype=x.dtype, device=x.device)
            loop_attr = torch.ones((num_nodes, 1), device=x.device, dtype=x.dtype)
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        else:
            edge_attr = torch.ones((src.shape[0], 1), device=x.device, dtype=x.dtype)

        h_src = h[src]
        h_dst = h[dst]
        edge_h = self.lin_edge(torch.log1p(edge_attr.clamp(min=0.0))).view(-1, self.heads, self.out_dim)
        attn = (h_src * self.att_src).sum(dim=-1)
        attn = attn + (h_dst * self.att_dst).sum(dim=-1)
        attn = attn + (edge_h * self.att_edge).sum(dim=-1)
        attn = self.leaky_relu(attn)
        attn = _segment_softmax(attn, dst, num_nodes)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        msg = h_src * attn.unsqueeze(-1)
        out = torch.zeros((num_nodes, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        out_flat = out.view(num_nodes, self.internal_dim)
        msg_flat = msg.reshape(msg.shape[0], self.internal_dim)
        out_flat.index_add_(0, dst, msg_flat)
        out = out_flat.view(num_nodes, self.heads, self.out_dim)
        out = out.reshape(num_nodes, self.internal_dim)
        out = self.norm(out + self.bias)
        out = self.out_proj(out)
        if return_attention:
            return out, {
                'edge_index': torch.stack([src, dst], dim=0),
                'alpha': attn,
            }
        return out


class GATHeuristicNet(nn.Module):
    """
    图注意力网络启发式函数。

    输入节点的特征和目标位置，输出估计的到目标成本。
    支持 iA* 风格的自监督训练。
    """

    def __init__(self, in_dim: int = 16, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, use_uncertainty: bool = True,
                 w_geo: float = 1.0, w_unc: float = 0.5, w_sem: float = 0.3,
                 node_in_dim: Optional[int] = None, edge_in_dim: Optional[int] = None):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: GAT隐藏层维度
            num_heads: 多头注意力头数
            num_layers: GAT层数
            dropout: Dropout率
            use_uncertainty: 是否使用不确定性感知
            w_geo: 几何启发式权重
            w_unc: 不确定性启发式权重
            w_sem: 语义优先性权重
        """
        super().__init__()

        if node_in_dim is not None:
            in_dim = node_in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_uncertainty = use_uncertainty
        self.w_geo = w_geo
        self.w_unc = w_unc
        self.w_sem = w_sem
        self.edge_in_dim = edge_in_dim

        # Goal-relative position encoding used only in the lightweight decode head.
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.type_embed = nn.Embedding(2, hidden_dim)

        self.device = torch.device('cpu')

        # GAT layers (fallback to a pure PyTorch implementation if torch_geometric is unavailable)
        self.gat_layers = nn.ModuleList()
        if HAS_PYG and GATConv is not None:
            for i in range(num_layers):
                in_channels = hidden_dim if i > 0 else hidden_dim
                self.gat_layers.append(
                    GATConv(
                        in_channels=in_channels,
                        out_channels=hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                        edge_dim=1,  # edge_weight
                    )
                )
        else:
            print('[GATHeuristic] torch_geometric unavailable, using pure PyTorch GAT fallback.')
            manual_out_dim = max(1, math.ceil(hidden_dim / num_heads))
            for _ in range(num_layers):
                self.gat_layers.append(
                    _TorchGATLayer(
                        in_dim=hidden_dim,
                        out_dim=manual_out_dim,
                        heads=num_heads,
                        dropout=dropout,
                    )
                )

        # Output MLP: hidden_dim -> scalar
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Uncertainty-aware extra branch
        if use_uncertainty:
            self.unc_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_weight: Optional[torch.Tensor] = None,
                     return_attention: bool = False):
        """Encode scene graph into goal-independent node embeddings."""
        N = x.shape[0]

        # Project input features
        h = self.input_proj(x)  # (N, hidden_dim)

        # Explicit node-type signal: frontier vs free.
        # HomoGraph zeroes the frontier flag, so this becomes a real ablation.
        if x.ndim == 2 and x.shape[1] > 11:
            node_type = (x[:, 11] > 0.5).long().clamp_(0, 1)
        else:
            node_type = torch.zeros((N,), dtype=torch.long, device=x.device)
        h = h + self.type_embed(node_type)

        # GAT message passing
        attention_records = []
        for gat in self.gat_layers:
            edge_attr = None
            if edge_weight is not None:
                edge_attr = edge_weight.view(-1, 1)
            if return_attention:
                if HAS_PYG and GATConv is not None and isinstance(gat, GATConv):
                    h_res, attn_info = gat(
                        h, edge_index, edge_attr=edge_attr,
                        return_attention_weights=True,
                    )
                    if attn_info is not None:
                        attn_edge_index, attn_alpha = attn_info
                        attention_records.append({
                            'edge_index': attn_edge_index,
                            'alpha': attn_alpha,
                        })
                else:
                    h_res, attn_info = gat(h, edge_index, edge_attr=edge_attr, return_attention=True)
                    if attn_info is not None:
                        attention_records.append(attn_info)
            else:
                h_res = gat(h, edge_index, edge_attr=edge_attr)
            h = h_res + h  # residual connection
            h = F.elu(h)
            h = F.dropout(h, p=0.1, training=self.training)

        if return_attention:
            return h, attention_records
        return h

    def decode_goal(self, h: torch.Tensor, x: torch.Tensor,
                    goal_pos: Optional[torch.Tensor] = None,
                    uncertainty: Optional[torch.Tensor] = None,
                    pos_coords: Optional[torch.Tensor] = None,
                    cost_scale: float = 1.0) -> torch.Tensor:
        """Decode a goal-conditioned heuristic from cached graph embeddings."""
        if goal_pos is not None and goal_pos.shape[0] == 3:
            goal_feat = self.goal_encoder(goal_pos.unsqueeze(0))  # (1, hidden_dim)
            h = h + goal_feat

        # Predict an unconstrained residual around the geometric heuristic.
        # The previous tanh clamp was too tight for Stonehenge-scale costs and
        # could saturate the gradient before the model had room to correct.
        learned_residual = self.output_mlp(h).squeeze(-1) * float(cost_scale)

        if goal_pos is not None and goal_pos.shape[0] == 3:
            if pos_coords is not None:
                node_pos = pos_coords
            else:
                node_pos = x[:, :3]
            geo_h = torch.norm(node_pos - goal_pos.unsqueeze(0), dim=-1) * float(cost_scale)
            h_out = geo_h + learned_residual
        else:
            h_out = learned_residual

        # Add uncertainty penalty
        if self.use_uncertainty and uncertainty is not None:
            unc_out = F.softplus(self.unc_mlp(uncertainty.unsqueeze(-1)).squeeze(-1))
            h_out = h_out + 0.15 * self.w_unc * unc_out

        return h_out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                goal_pos: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None,
                pos_coords: Optional[torch.Tensor] = None,
                cost_scale: float = 1.0,
                return_attention: bool = False):
        """Convenience wrapper for encode_graph + decode_goal."""
        if return_attention:
            h, attention_records = self.encode_graph(
                x, edge_index, edge_weight=edge_weight, return_attention=True
            )
            out = self.decode_goal(
                h, x,
                goal_pos=goal_pos,
                uncertainty=uncertainty,
                pos_coords=pos_coords,
                cost_scale=cost_scale,
            )
            return out, attention_records

        h = self.encode_graph(x, edge_index, edge_weight=edge_weight)
        return self.decode_goal(
            h, x,
            goal_pos=goal_pos,
            uncertainty=uncertainty,
            pos_coords=pos_coords,
            cost_scale=cost_scale,
        )

    def _geometric_heuristic(self, x: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
        """
        可微分的几何启发式 (基于节点位置特征)。

        Args:
            x: (N, in_dim) 节点特征 (前3维是归一化位置)
            goal_pos: (3,) 目标位置

        Returns:
            geo_h: (N,) 欧氏距离估计
        """
        # Extract normalized position from features (first 3 dims)
        pos = x[:, :3]  # (N, 3)
        # goal_pos is in world coordinates; we approximate with normalized distance
        # This is a rough approximation; for exact distance, use actual positions
        geo_dist = torch.norm(pos - goal_pos.unsqueeze(0), dim=-1)
        return geo_dist

    def compute_loss(self, pred_costs: torch.Tensor, true_costs: torch.Tensor,
                      node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        iA* 自监督损失。

        L = Σ_{v in V} (h(v, goal) - c*(v, goal))^2
        where c*(v, goal) is the true shortest path cost.

        Args:
            pred_costs: (N,) GAT预测的启发式值
            true_costs: (N,) 真实最短路径成本
            node_mask: (N,) 可选，只计算mask=True的节点

        Returns:
            loss: scalar
        """
        if node_mask is not None:
            diff = (pred_costs[node_mask] - true_costs[node_mask]) ** 2
        else:
            diff = (pred_costs - true_costs) ** 2

        # MSE loss
        mse_loss = diff.mean()

        # Light overestimation penalty (heuristic should be admissible: h <= true_cost)
        # Reduced from 10.0 to 0.5 - was too aggressive
        overestimate = F.relu(pred_costs - true_costs)
        over_penalty = 0.5 * overestimate.mean()

        return mse_loss + over_penalty

    def train_self_supervised(self, graph, paths: list, num_epochs: int = 100,
                              lr: float = 0.001, batch_size: int = 16,
                              device: str = 'cuda'):
        """
        iA* 自监督训练。

        使用已知的最优路径反向传播真值成本。

        Args:
            graph: PyG HeteroData
            paths: list of (node_indices, costs) tuples from A* planning
            num_epochs: 训练轮数
            lr: 学习率
            batch_size: 批大小
            device: 计算设备
        """
        self.to(device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        # Collect training data from paths
        train_nodes = []
        train_costs = []
        for path_indices, path_costs in paths:
            for i, (node_id, cost) in enumerate(zip(path_indices, path_costs)):
                train_nodes.append(node_id)
                train_costs.append(cost)

        if len(train_nodes) == 0:
            print('[GATHeuristic] No training data, skipping training.')
            return

        train_nodes = torch.tensor(train_nodes, dtype=torch.long, device=device)
        train_costs = torch.tensor(train_costs, dtype=torch.float32, device=device)

        # Normalize costs
        cost_mean = train_costs.mean()
        cost_std = train_costs.std() + 1e-6
        train_costs_norm = (train_costs - cost_mean) / cost_std

        print(f'[GATHeuristic] Training on {len(train_nodes)} node samples...')
        print(f'  Cost range: [{cost_min := train_costs.min().item():.4f}, {train_costs.max().item():.4f}]')

        # Build unified graph for message passing
        x_all = torch.cat([graph['free'].x, graph['frontier'].x], dim=0)
        edge_indices = []
        edge_weights = []
        for etype in graph.edge_types:
            ei = graph[etype].edge_index
            ew = graph[etype].edge_weight
            edge_indices.append(ei)
            edge_weights.append(ew)

        # Remap frontier indices (offset by num_free)
        n_free = graph['free'].x.shape[0]
        all_edge_index = []
        all_edge_weight = []
        for i, etype in enumerate(graph.edge_types):
            ei = graph[etype].edge_index
            ew = graph[etype].edge_weight
            src_type, _, dst_type = etype
            offset_src = 0 if src_type == 'free' else n_free
            offset_dst = 0 if dst_type == 'free' else n_free
            remapped = ei.clone()
            remapped[0] += offset_src
            remapped[1] += offset_dst
            all_edge_index.append(remapped)
            all_edge_weight.append(ew)

        if all_edge_index:
            unified_edge_index = torch.cat(all_edge_index, dim=1)
            unified_edge_weight = torch.cat(all_edge_weight, dim=0)
        else:
            unified_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            unified_edge_weight = torch.zeros(0, device=device)

        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass on full graph
            h_all = self.forward(x_all, unified_edge_index, unified_edge_weight)

            # Compute loss on training nodes
            pred = h_all[train_nodes]
            true = train_costs_norm
            loss = self.compute_loss(pred, true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}')

        self.eval()
        print('[GATHeuristic] Training complete.')

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)
        print(f'[GATHeuristic] Saved to {path}')

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        current_state = self.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(key)
        missing = [k for k in current_state.keys() if k not in compatible]
        self.load_state_dict(compatible, strict=False)
        if missing or skipped:
            print(f'[GATHeuristic] Loaded from {path} with missing={len(missing)} skipped={len(skipped)} keys')
        else:
            print(f'[GATHeuristic] Loaded from {path}')
