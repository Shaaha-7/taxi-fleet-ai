"""Zero-shot baseline from the published OpenCity foundation model (HKUDS, arXiv 2408.10269).

Uses the authors' model code (external/OpenCity, MIT/Apache) and HuggingFace weights
(external/opencity_weights/OpenCity-{base,plus}.pth) without any training on our data.
Input conventions follow the repo: one day of history, values in raw units (the model applies
instance normalisation internally), minute-of-day (end of interval, 15..1440) and weekday (Mon=1..7)
features for history and prediction windows, 8 Laplacian eigenvectors, and I + D^-1/2 A D^-1/2.
For 15-min data the history is 96 steps; the model's patching handles the 3x coarser interval.
NYC taxi data was in OpenCity's pretraining corpus, so only the Dubai-like city is a true zero-shot.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..evaluation.metrics import forecast_metrics
from ..ingestion.dataset import CityData, valid_targets
from ..utils import PROJECT_ROOT

REPO = PROJECT_ROOT / "external" / "OpenCity"
WEIGHTS = PROJECT_ROOT / "external" / "opencity_weights"


def _load_class():
    spec = importlib.util.spec_from_file_location("opencity_model", REPO / "model" / "OpenCity" / "OpenCity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OpenCity


def _graph_inputs(city: CityData):
    a = city.graph.adjacency.astype(float)
    d = a.sum(1)
    dinv = np.where(d > 0, d ** -0.5, 0.0)
    lap = np.eye(len(a)) - dinv[:, None] * a * dinv[None]
    vals, vecs = np.linalg.eigh(lap)
    isolated = int((d == 0).sum())
    lape = vecs[:, isolated + 1: isolated + 9]
    adj = np.eye(len(a)) + dinv[:, None] * a * dinv[None]
    sh = np.where(a > 0, 1.0, 511.0)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    return t(adj), t(sh), t(lape)


def build_opencity(city: CityData, variant: str, device):
    sd = torch.load(WEIGHTS / f"OpenCity-{variant}.pth", map_location="cpu", weights_only=True)
    sd = {k.replace("module.predictor.", ""): v for k, v in sd.items()}
    embed = sd["spatial_embedding.embedding_lap_pos_enc.weight"].shape[0]
    depth = len({k.split(".")[1] for k in sd if k.startswith("encoder_blocks.")})
    out_window = sd["linear.weight"].shape[0]
    adj, sh, lape = _graph_inputs(city)
    args = SimpleNamespace(
        adj_mx_dict={"city": adj}, sh_mx_dict={"city": sh}, lap_mx_dict={"city": lape},
        embed_dim=embed, skip_dim=embed, lape_dim=8, geo_num_heads=0, sem_num_heads=0, tc_num_heads=16,
        t_num_heads=16, mlp_ratio=2, qkv_bias=True, drop=0.1, attn_drop=0.3, drop_path=0.0, s_attn_size=3,
        t_attn_size=1, enc_depth=depth, type_ln="pre", input_window=288, output_window=out_window, far_mask_delta=5)
    model = _load_class()(args, ["city"], device, dim_in=1)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def opencity_zero_shot(city: CityData, cfg: dict, variant: str, device, split: str = "test", batch: int = 8,
                       native_30min: bool = True) -> dict:
    """Pickup forecast metrics [S, H, N, 1] on `split` at 15-min resolution.

    native_30min: OpenCity's taxi pretraining used 30-min bins (patch gap 6); 15-min input uses an
    unseen gap of 3. With this flag, input is aggregated to 30-min, the model forecasts 30-min counts,
    and each is split evenly into its two 15-min slots for scoring against the same 15-min targets.
    """
    model = build_opencity(city, variant, device)
    spd, horizon = city.slots_per_day, cfg["forecast"]["horizon"]
    agg = 2 if native_30min else 1
    series15 = city.pickups
    series = series15.reshape(-1, agg, series15.shape[1]).sum(1)
    sl = spd // agg
    idx = np.arange(len(series))
    minute = (idx % sl + 1) * (24 * 60 // sl)
    week = city.dow[::agg] + 1
    hist = sl
    t15 = valid_targets(city, city.split_slots(split), spd, 0, spd)
    t15 = t15[t15 % agg == 0]                                     # 30-min aligned forecast origins
    preds, ys = [], []
    n = series.shape[1]
    steps = -(-horizon // agg)
    for i in range(0, len(t15), batch):
        tt = t15[i:i + batch] // agg
        hidx = tt[:, None] + np.arange(-hist, 0)[None]
        pidx = tt[:, None] + np.arange(0, hist)[None]
        rep = lambda a: a[..., None].repeat(n, -1)
        x = np.stack([series[hidx], rep(minute[hidx]), rep(week[hidx])], -1)
        lbl = np.stack([np.zeros_like(series[pidx]), rep(minute[pidx]), rep(week[pidx])], -1)
        out = model(torch.as_tensor(x, dtype=torch.float32, device=device),
                    torch.as_tensor(lbl, dtype=torch.float32, device=device), "city")[:, :steps].clamp_min(0).cpu()
        out = (out / agg).repeat_interleave(agg, dim=1)[:, :horizon]
        preds.append(out)
        t_15 = tt * agg
        ys.append(torch.as_tensor(series15[t_15[:, None] + np.arange(horizon)[None]])[..., None])
    return forecast_metrics(torch.cat(preds), torch.cat(ys))
