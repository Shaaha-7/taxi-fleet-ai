"""Zone-level rebalancing policies (parameter-shared GNNs, so size scales with zones, not taxis).

BilevelPolicy  - Gammelli et al. style: Dirichlet over zones = desired distribution of next-slot
                 supply; the min-cost-flow LP turns it into feasible moves.
MeanFieldPolicy - each zone's idle taxis split over {stay, k nearest neighbours} via a per-zone
                 Dirichlet; flows applied directly with no optimisation layer. This is the
                 "remove the LP layer" ablation and a decentralised mean-field style controller.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.nn import GCNConv


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:              # [S, N, F] -> [S, N, hidden]
        h = F.relu(self.inp(x))
        for conv in self.convs:
            h = h + F.relu(conv(h, self.edge_index, self.edge_weight))
        return h


class _ActorCritic(nn.Module):
    def __init__(self, in_dim, hidden, edge_index, edge_weight):
        super().__init__()
        self.enc = GNNEncoder(in_dim, hidden, edge_index, edge_weight)
        # Separate critic encoder: critic updates never move the actor.
        self.critic_enc = GNNEncoder(in_dim, hidden, edge_index, edge_weight)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_enc(x).mean(1)).squeeze(-1)


class BilevelPolicy(_ActorCritic):
    def __init__(self, in_dim, hidden, edge_index, edge_weight, concentration_scale: float):
        super().__init__(in_dim, hidden, edge_index, edge_weight)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.scale = concentration_scale

    def forward(self, x):
        h = self.enc(x)
        conc = F.softplus(self.actor(h).squeeze(-1)) * self.scale + 1e-3   # [S, N]
        return Dirichlet(conc), self.value(x)


class MeanFieldPolicy(_ActorCritic):
    def __init__(self, in_dim, hidden, edge_index, edge_weight, neighbor_table: torch.Tensor,
                 dist_feat: torch.Tensor, concentration_scale: float):
        super().__init__(in_dim, hidden, edge_index, edge_weight)
        self.register_buffer("nb", neighbor_table)                         # [N, K+1], col 0 = self
        self.register_buffer("dist_feat", dist_feat)                       # [N, K+1]
        self.actor = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.scale = concentration_scale
        stay_bias = torch.zeros(neighbor_table.shape[1])
        stay_bias[0] = 3.0                                                 # start close to "stay"
        self.register_buffer("stay_bias", stay_bias)

    def forward(self, x):
        h = self.enc(x)                                                    # [S, N, H]
        s, n, d = h.shape
        k = self.nb.shape[1]
        h_src = h[:, :, None].expand(s, n, k, d)
        h_dst = h[:, self.nb]                                              # [S, N, K+1, H]
        z = torch.cat([h_src, h_dst, self.dist_feat[None, :, :, None].expand(s, n, k, 1)], dim=-1)
        logits = self.actor(z).squeeze(-1) + self.stay_bias
        conc = F.softplus(logits) * self.scale + 1e-3                      # [S, N, K+1]
        return Dirichlet(conc), self.value(x)
