"""Spatio-temporal demand forecasters sharing one interface.

All models map a DemandDataset batch to predicted counts [B, H, N, 2] (pickups, destination
demand). `gcngru` is the blueprint's GCN + GRU ST-GNN (T-GCN style); `gcngru_ae` adds
STAEformer-style adaptive node embeddings plus time-of-day / day-of-week embeddings,
the cheapest upgrade with consistent gains in the 2023-2026 literature.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from ..ingestion.dataset import CityData, Scaler


class Forecaster(nn.Module):
    trainable = True

    def __init__(self, scaler: Scaler):
        super().__init__()
        self.scaler = scaler

    def forward_scaled(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: dict) -> torch.Tensor:
        """Predicted counts, clamped to be non-negative."""
        return self.scaler.inverse(self.forward_scaled(batch)).clamp_min(0.0)


class LastValue(Forecaster):
    trainable = False

    def __init__(self, scaler: Scaler, horizon: int):
        super().__init__(scaler)
        self.horizon = horizon

    def forward_scaled(self, batch):
        return batch["x"][:, -1:].expand(-1, self.horizon, -1, -1)


class HistoricalAverage(Forecaster):
    """Mean count per (weekday/weekend, slot-of-day, zone) over the training period."""
    trainable = False

    def __init__(self, scaler: Scaler, city: CityData):
        super().__init__(scaler)
        series = city.series
        slots = list(city.split_slots("train"))
        wkd = (city.dow[slots] >= 5).astype(int)
        tod = city.tod[slots]
        table = torch.zeros(2, city.slots_per_day, city.graph.num_zones, 2)
        for w in (0, 1):
            for s in range(city.slots_per_day):
                m = (wkd == w) & (tod == s)
                if m.any():
                    table[w, s] = torch.as_tensor(series[slots][m].mean(0))
        self.register_buffer("table", table)

    def forward(self, batch):
        return self.table[batch["y_wkd"], batch["y_tod"]]


class GCNGRU(Forecaster):
    def __init__(self, scaler: Scaler, num_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 hidden: int, horizon: int, emb_dim: int = 0, slots_per_day: int = 96, in_dim: int = 2):
        super().__init__(scaler)
        self.horizon, self.emb_dim = horizon, emb_dim
        feat = in_dim
        if emb_dim:
            self.node_emb = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
            self.tod_emb = nn.Embedding(slots_per_day, emb_dim)
            self.dow_emb = nn.Embedding(7, emb_dim)
            feat += 3 * emb_dim
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.inp = nn.Linear(feat, hidden)
        self.gc1 = GCNConv(hidden, hidden)
        self.gc2 = GCNConv(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + emb_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, horizon * in_dim))

    def forward_scaled(self, batch):
        x = batch["x"]                                            # [B, P, N, F]
        b, p, n, c = x.shape
        if self.emb_dim:
            e = self.emb_dim
            x = torch.cat([x,
                           self.node_emb.expand(b, p, n, e),
                           self.tod_emb(batch["x_tod"])[:, :, None].expand(b, p, n, e),
                           self.dow_emb(batch["x_dow"])[:, :, None].expand(b, p, n, e)], dim=-1)
        h = F.relu(self.inp(x)).reshape(b * p, n, -1)             # static graph, batched nodes
        h = h + F.relu(self.gc1(h, self.edge_index, self.edge_weight))
        h = h + F.relu(self.gc2(h, self.edge_index, self.edge_weight))
        h = h.reshape(b, p, n, -1).permute(0, 2, 1, 3).reshape(b * n, p, -1)
        _, last = self.gru(h)
        z = last[-1].reshape(b, n, -1)
        if self.emb_dim:
            z = torch.cat([z, self.node_emb.expand(b, n, -1)], dim=-1)
        return self.head(z).reshape(b, n, self.horizon, c).permute(0, 2, 1, 3)


def build_forecaster(name: str, city: CityData, scaler: Scaler, fcfg: dict) -> Forecaster:
    g = city.graph
    if name == "last":
        return LastValue(scaler, fcfg["horizon"])
    if name == "ha":
        return HistoricalAverage(scaler, city)
    if name in ("gcngru", "gcngru_ae"):
        return GCNGRU(scaler, g.num_zones, g.edge_index, g.edge_weight, fcfg["hidden"], fcfg["horizon"],
                      emb_dim=fcfg["emb_dim"] if name == "gcngru_ae" else 0,
                      slots_per_day=city.slots_per_day)
    from . import sota_forecasters as sota
    if name == "agcrn":
        return sota.AGCRN(scaler, g.num_zones, fcfg["horizon"])
    if name == "gwnet":
        src, dst = g.edge_index
        adj = torch.zeros(g.num_zones, g.num_zones)
        adj[src, dst] = g.edge_weight
        return sota.GraphWaveNet(scaler, g.num_zones, fcfg["horizon"], adj)
    if name == "staeformer":
        return sota.STAEformer(scaler, g.num_zones, fcfg["horizon"], fcfg["history"] + fcfg["periodic_days"],
                               city.slots_per_day)
    raise ValueError(f"unknown forecaster {name}")
