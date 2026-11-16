"""Forecasting and fleet-control metrics."""
from __future__ import annotations

import torch


def forecast_metrics(pred: torch.Tensor, y: torch.Tensor, lo=None, hi=None, channel: int = 0,
                     mape_threshold: float = 1.0) -> dict:
    """pred, y: [S, H, N, C]. Reports pickup-channel MAE/RMSE/masked MAPE per horizon and overall."""
    p, t = pred[..., channel], y[..., channel]
    out = {}
    horizons = list(range(p.shape[1]))
    for h in horizons + [None]:
        ph, th = (p, t) if h is None else (p[:, h], t[:, h])
        key = "all" if h is None else f"h{h + 1}"
        err = ph - th
        mask = th >= mape_threshold
        out[key] = dict(MAE=err.abs().mean().item(), RMSE=err.pow(2).mean().sqrt().item(),
                        MAPE=(err.abs()[mask] / th[mask]).mean().item() * 100)
        if lo is not None:
            l, u = (lo[..., channel], hi[..., channel]) if h is None else (lo[:, h, ..., channel], hi[:, h, ..., channel])
            out[key]["coverage"] = ((th >= l) & (th <= u)).float().mean().item() * 100
            out[key]["width"] = (u - l).mean().item()
    return out


def episode_metrics(totals: dict) -> dict:
    """Aggregate simulator counters from one episode into the reporting metrics."""
    req = max(totals["requests"], 1.0)
    served = max(totals["served"], 1.0)
    empty = totals["empty_km_rebalance"] + totals["empty_km_pickup"] + totals["empty_km_cruise"]
    total_km = empty + totals["occupied_km"]
    return {
        "served_rate_%": 100 * totals["served"] / req,
        "mean_wait_min": totals["wait_min_sum"] / served,
        "p90_wait_min": totals.get("p90_wait_min", float("nan")),
        "empty_km": empty,
        "empty_km_per_trip": empty / served,
        "empty_km_share_%": 100 * empty / max(total_km, 1e-9),
        "rebalance_km": totals["empty_km_rebalance"],
        "profit_AED": totals["profit"],
        "utilization_%": 100 * totals["busy_vehicle_slots"] / max(totals["vehicle_slots"], 1.0),
        "service_gini": totals.get("service_gini", float("nan")),
        "decision_ms": 1000 * totals.get("decision_s", 0.0) / max(totals.get("decisions", 0), 1),
    }
