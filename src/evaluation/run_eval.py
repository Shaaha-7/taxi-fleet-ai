"""Run every control method on the same held-out demand days and aggregate metrics."""
from __future__ import annotations

import numpy as np
import torch

from ..control.baselines import ControlContext, GreedyNeighbor, LPRebalance, NoRebalance
from ..ingestion.dataset import CityData
from ..simulation.fleet_sim import FleetSimTensor
from ..utils import mean_ci95
from .metrics import episode_metrics


def load_forecasts(cfg: dict) -> dict:
    path = cfg["paths"]["output_dir"] / "forecasts.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python main.py forecast` first.")
    return dict(np.load(path))


def _forecasts_for(name: str, cfg: dict, default: dict) -> dict:
    """Methods suffixed `_dfl` use the decision-focused forecaster's export instead."""
    if not name.endswith("_dfl"):
        return default
    path = cfg["paths"]["output_dir"] / "forecasts_dfl.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python main.py dfl` first.")
    return dict(np.load(path))


def make_controller(name: str, ctx: ControlContext, cfg: dict):
    name = name.removesuffix("_dfl")
    if name == "none":
        return NoRebalance(ctx)
    if name == "greedy":
        return GreedyNeighbor(ctx)
    if name in ("lp", "lp_robust", "oracle_lp"):
        return LPRebalance(ctx, {"lp": "mean", "lp_robust": "hi", "oracle_lp": "oracle"}[name])
    if name in ("bilevel_rl", "meanfield_rl", "residual_rl"):
        from ..models.train_rl import load_policy_controller
        return load_policy_controller(name, ctx, cfg)
    raise ValueError(f"unknown control method {name}")


@torch.no_grad()
def run_episode(sim: FleetSimTensor, controller) -> list[dict]:
    obs, done = sim.reset(), False
    while not done:
        obs, _, done = sim.step(controller.act(obs))
    totals = sim.episode_totals()
    for d in totals:
        d["decision_s"], d["decisions"] = controller.decision_s / sim.S, controller.decisions
    return totals


def evaluate_methods(city: CityData, cfg: dict, methods: list[str], split: str = "test", log=print,
                     forecasts: dict | None = None, days: list[int] | None = None, sim_kwargs: dict | None = None) -> dict:
    forecasts = forecasts if forecasts is not None else load_forecasts(cfg)
    days = days if days is not None else city.split_day_starts(split)
    fleet = cfg["sim"]["fleet_size"]
    results = {}
    for name in methods:
        ctx = ControlContext(city, cfg, _forecasts_for(name, cfg, forecasts), days, fleet)
        controller = make_controller(name, ctx, cfg)
        sim = FleetSimTensor(city, cfg, days, fleet, **(sim_kwargs or {}))
        per_day = [episode_metrics(t) for t in run_episode(sim, controller)]
        results[name] = {k: mean_ci95([d[k] for d in per_day]) for k in per_day[0]}
        r = results[name]
        log(f"[eval] {name:13s} served {r['served_rate_%'][0]:5.1f}%  wait {r['mean_wait_min'][0]:5.2f} min  "
            f"empty share {r['empty_km_share_%'][0]:5.1f}%  profit {r['profit_AED'][0]:,.0f} AED")
    return results
