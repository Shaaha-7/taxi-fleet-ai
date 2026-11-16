"""Cross-city transfer: pretrain a zone-count-agnostic ST-GNN on NYC TLC, apply it to the Dubai-like city.

Node-agnostic design (no per-zone ID embeddings), following OpenCity / TPB / STGP practice:
  * node identity  = k smallest non-trivial eigenvectors of the normalised graph Laplacian
                     (random sign flips during training, since eigenvector signs are arbitrary);
  * scale          = per-sample, per-zone instance normalisation of the input window (RevIN-style),
                     so a model trained on NYC volumes transfers to a city with different volumes;
  * time           = fraction-of-day sin/cos + weekend flag (both cities have a Sat-Sun weekend).
Protocol: zero-shot, then few-shot with k target days (head-only and full fine-tuning), against the
same architecture trained from scratch on those k days.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv

from ..evaluation.metrics import forecast_metrics
from ..ingestion.dataset import CityData, DemandDataset, Scaler, valid_targets


class IdentityScaler(Scaler):
    def __init__(self):
        self.mean, self.std = np.zeros(2), np.ones(2)


def laplacian_eigvecs(city: CityData, k: int) -> torch.Tensor:
    g = city.graph
    a = np.zeros((g.num_zones, g.num_zones))
    src, dst = np.nonzero(g.adjacency)
    a[src, dst] = np.exp(-g.dist_km[src, dst] / g.dist_km[src, dst].mean())
    d = a.sum(1)
    dinv = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-9)), 0)
    lap = np.eye(len(a)) - dinv[:, None] * a * dinv[None]
    _, vecs = np.linalg.eigh(lap)
    return torch.as_tensor(vecs[:, 1:k + 1] * np.sqrt(len(a)), dtype=torch.float32)


class CityGraphTensors:
    def __init__(self, city: CityData, k: int, device):
        self.eig = laplacian_eigvecs(city, k).to(device)
        self.edge_index = city.graph.edge_index.to(device)
        self.edge_weight = city.graph.edge_weight.to(device)
        self.spd = city.slots_per_day


class TransferGNN(nn.Module):
    def __init__(self, hidden: int, horizon: int, k_eig: int, in_dim: int = 2):
        super().__init__()
        self.horizon, self.in_dim = horizon, in_dim
        feat = in_dim + in_dim + k_eig + 3                       # normalised x, log level, eig, time
        self.inp = nn.Linear(feat, hidden)
        self.gc1 = GCNConv(hidden, hidden)
        self.gc2 = GCNConv(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + k_eig, hidden), nn.ReLU(), nn.Linear(hidden, horizon * in_dim))

    def forward(self, batch: dict, g: CityGraphTensors, flip_signs: bool = False) -> torch.Tensor:
        x = batch["x"]                                           # raw counts [B, P, N, C]
        b, p, n, c = x.shape
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1.0
        eig = g.eig
        if flip_signs:
            eig = eig * (torch.randint(0, 2, (1, eig.shape[1]), device=eig.device) * 2 - 1)
        ang = 2 * np.pi * batch["x_tod"].float() / g.spd
        time = torch.stack([torch.sin(ang), torch.cos(ang), (batch["x_dow"] >= 5).float()], -1)
        h = torch.cat([(x - mu) / sd, torch.log1p(mu).expand(b, p, n, c), eig.expand(b, p, n, -1),
                       time[:, :, None].expand(b, p, n, 3)], -1)
        h = F.relu(self.inp(h)).reshape(b * p, n, -1)
        h = h + F.relu(self.gc1(h, g.edge_index, g.edge_weight))
        h = h + F.relu(self.gc2(h, g.edge_index, g.edge_weight))
        h = h.reshape(b, p, n, -1).permute(0, 2, 1, 3).reshape(b * n, p, -1)
        _, last = self.gru(h)
        z = torch.cat([last[-1].reshape(b, n, -1), eig.expand(b, n, -1)], -1)
        out = self.head(z).reshape(b, n, self.horizon, c).permute(0, 2, 1, 3)
        return (out * sd + mu).clamp_min(0.0)


def city_datasets(city: CityData, fcfg: dict, train_days: list[int] | None = None) -> dict:
    """Raw-count datasets. `train_days` restricts training to specific day indices (few-shot)."""
    sc = IdentityScaler()
    spd = city.slots_per_day
    out = {}
    for split in ("train", "val", "test"):
        slots = city.split_slots(split)
        t = valid_targets(city, slots, fcfg["history"], fcfg["periodic_days"], fcfg["horizon"])
        if split == "train" and train_days is not None:
            t = t[np.isin(t // spd, train_days)]
        out[split] = DemandDataset(city, t, sc, fcfg)
    return out


def _to(b, device):
    return {k: v.to(device) for k, v in b.items()}


@torch.no_grad()
def predict(model, ds, g, device):
    model.eval()
    return torch.cat([model(_to(b, device), g).cpu() for b in DataLoader(ds, batch_size=128)])


def fit(model, ds: dict, g: CityGraphTensors, device, epochs: int, lr: float, params=None, log=None,
        patience: int = 6) -> nn.Module:
    model.to(device)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=1e-4)
    scale = float(ds["train"].y.float().mean()) + 1e-6
    best, best_mae, bad = copy.deepcopy(model.state_dict()), float("inf"), 0
    for ep in range(epochs):
        model.train()
        for b in DataLoader(ds["train"], batch_size=32, shuffle=True):
            b = _to(b, device)
            loss = (model(b, g, flip_signs=True) - b["y"]).abs().mean() / scale
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        mae = forecast_metrics(predict(model, ds["val"], g, device), ds["val"].y)["all"]["MAE"]
        if log:
            log(f"    epoch {ep + 1:3d}  val MAE {mae:.3f}")
        if mae < best_mae - 1e-4:
            best, best_mae, bad = copy.deepcopy(model.state_dict()), mae, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    return model


def run_transfer(source: CityData, target: CityData, cfg: dict, device, log=print) -> dict:
    t = cfg["transfer"]
    fcfg = cfg["forecast"]
    torch.manual_seed(cfg["seed"])
    gs, gt = CityGraphTensors(source, t["k_eig"], device), CityGraphTensors(target, t["k_eig"], device)

    log(f"[transfer] pretraining on source city ({source.graph.num_zones} zones, {len(source.slot_index) // source.slots_per_day} days)")
    pre = fit(TransferGNN(t["hidden"], fcfg["horizon"], t["k_eig"]), city_datasets(source, fcfg), gs, device,
              t["pretrain_epochs"], t["lr"], log=log)
    torch.save(pre.state_dict(), cfg["paths"]["output_dir"] / "transfer_pretrained.pt")

    tgt_full = city_datasets(target, fcfg)
    res = {}

    def score(name, model):
        m = forecast_metrics(predict(model, tgt_full["test"], gt, device), tgt_full["test"].y)
        res[name] = m
        log(f"  {name:28s} test MAE {m['all']['MAE']:.3f}  RMSE {m['all']['RMSE']:.3f}")

    score("zero-shot (NYC pretrained)", pre)
    train_days = sorted({int(x) for x in np.asarray(tgt_full["train"].t) // target.slots_per_day})
    for k in t["few_shot_days"]:
        days = train_days[-k:]
        ds = city_datasets(target, fcfg, train_days=days)
        scratch = fit(TransferGNN(t["hidden"], fcfg["horizon"], t["k_eig"]), ds, gt, device, t["finetune_epochs"], t["lr"])
        score(f"scratch, {k} days", scratch)
        head = copy.deepcopy(pre)
        fit(head, ds, gt, device, t["finetune_epochs"], t["lr"], params=head.head.parameters())
        score(f"pretrained + head-only, {k} days", head)
        full = copy.deepcopy(pre)
        fit(full, ds, gt, device, t["finetune_epochs"], t["lr"] * 0.3)
        score(f"pretrained + full FT, {k} days", full)
    scratch_all = fit(TransferGNN(t["hidden"], fcfg["horizon"], t["k_eig"]), tgt_full, gt, device, t["pretrain_epochs"], t["lr"])
    score("scratch, all training days", scratch_all)
    return res
