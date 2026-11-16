"""Decision-focused fine-tuning of the forecaster through the rebalancing LP.

The forecast enters the LP's right-hand side (the coverage constraint), so SPO+/Fenchel-Young losses,
which need the prediction in the objective, do not apply (Mandi et al., JAIR 2024 note RHS forecasts
are under-studied). Instead, the decision loss is Gaussian-smoothed and differentiated with the
score-function identity (zeroth-order gradient through the exact LP, in the spirit of perturbed
optimizers, Berthet et al. 2020), avoiding solver-layer dependencies:

    L(theta) = E_Z[ cost(LP(theta + sigma*Z), true demand) ]
    dL/dtheta = E_Z[ cost(LP(theta + sigma*Z)) * Z ] / sigma        (antithetic pairs, sigma ~ sqrt(theta))

cost = -(riders served next slot) + empty rebalancing km * cost/km / mean fare  (units: trips).
Training loss = MAE (keeps forecasts calibrated) + beta * decision surrogate, starting from the
MAE-trained weights. States (idle taxis, arrivals, queues) are logged from the fluid simulator
running the robust-LP controller on training days, so the LP sees realistic supply.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..control.baselines import ControlContext, LPRebalance
from ..ingestion.dataset import CityData, DemandDataset
from ..simulation.fleet_sim import FleetSimTensor
from .train_forecast import _to, export_forecasts, make_datasets, predict


@torch.no_grad()
def collect_decision_states(city: CityData, cfg: dict, forecasts: dict, days: list[int]) -> dict:
    ctx = ControlContext(city, cfg, forecasts, days, cfg["sim"]["fleet_size"])
    teacher = LPRebalance(ctx, "hi")
    sim = FleetSimTensor(city, cfg, days)
    obs, done = sim.reset(), False
    rec = {k: [] for k in ("target", "idle", "arr", "queue", "true")}
    while not done:
        target = np.asarray(days) + obs["t"] + 1
        rec["target"].append(target)
        rec["idle"].append(obs["idle"].numpy())
        rec["arr"].append(obs["incoming"][:, 0].numpy())
        rec["queue"].append(obs["queue"].numpy())
        rec["true"].append(obs["next_true_pickups"].numpy())
        obs, _, done = sim.step(teacher.act(obs))
    return {k: np.concatenate(v) for k, v in rec.items()}


class DecisionDataset(DemandDataset):
    def __init__(self, city, states: dict, scaler, fcfg):
        super().__init__(city, states["target"], scaler, fcfg)
        self.idle = torch.as_tensor(states["idle"])
        self.arr = torch.as_tensor(states["arr"])
        self.queue = torch.as_tensor(states["queue"])
        self.true = torch.as_tensor(states["true"])

    def __getitem__(self, i):
        d = super().__getitem__(i)
        d.update(idle=self.idle[i], arr=self.arr[i], queue=self.queue[i], true=self.true[i])
        return d


def decision_cost(solver, dist, idle, arr, need, true, budget, km_cost_trips) -> float:
    flows = solver.solve(idle, arr, need, budget)
    supply = idle - flows.sum(1) + flows.sum(0) + arr
    return -np.minimum(supply, true).sum() + km_cost_trips * (flows * dist).sum()


def perturbed_gradient(theta: np.ndarray, b: dict, solver, dist, budget_frac, km_cost_trips, sigma, samples, rng):
    """Antithetic score-function estimate of d E[cost] / d theta for a batch [B, N]."""
    grad = np.zeros_like(theta)
    costs = []
    for i in range(len(theta)):
        idle, arr, q, true = (b[k][i].numpy() for k in ("idle", "arr", "queue", "true"))
        true = true + q
        scale = sigma * (np.sqrt(theta[i]) + 0.5)
        budget = budget_frac * idle.sum()
        for _ in range(samples):
            z = rng.standard_normal(theta.shape[1])
            c_pos = decision_cost(solver, dist, idle, arr, np.clip(theta[i] + scale * z, 0, None) + q, true, budget, km_cost_trips)
            c_neg = decision_cost(solver, dist, idle, arr, np.clip(theta[i] - scale * z, 0, None) + q, true, budget, km_cost_trips)
            grad[i] += (c_pos - c_neg) / 2 * z / scale
            costs.append((c_pos + c_neg) / 2)
        grad[i] /= samples
    return grad, float(np.mean(costs))


def train_decision_focused(model, city: CityData, cfg: dict, forecasts: dict, device, log=print):
    d = cfg["dfl"]
    fcfg = cfg["forecast"]
    rng = np.random.default_rng(cfg["seed"])
    ds, scaler = make_datasets(city, fcfg)
    model.scaler = scaler
    skip = fcfg["periodic_days"] + 1
    log("[dfl] logging decision states from the robust-LP controller")
    train_states = collect_decision_states(city, cfg, forecasts, city.split_day_starts("train")[skip:])
    val_states = collect_decision_states(city, cfg, forecasts, city.split_day_starts("val"))
    train_ds = DecisionDataset(city, train_states, scaler, fcfg)
    val_ds = DecisionDataset(city, val_states, scaler, fcfg)

    ctx = ControlContext(city, cfg, forecasts, [0], cfg["sim"]["fleet_size"])
    solver, dist = ctx.solver, city.graph.dist_km
    s = cfg["sim"]
    mean_fare = s["fare_flag"] + s["fare_per_km"] * float((city.od.sum(0) * dist).sum() / city.od.sum())
    km_cost_trips = s["empty_cost_per_km"] / mean_fare

    def val_decision_cost():
        pred = predict(model, val_ds, device)[:, 0, :, 0].numpy()
        return float(np.mean([decision_cost(solver, dist, val_ds.idle[i].numpy(), val_ds.arr[i].numpy(),
                                            pred[i] + val_ds.queue[i].numpy(), val_ds.true[i].numpy() + val_ds.queue[i].numpy(),
                                            s["rebalance_budget"] * val_ds.idle[i].numpy().sum(), km_cost_trips)
                              for i in range(len(val_ds))]))

    model.to(device)
    std = torch.as_tensor(scaler.std, dtype=torch.float32, device=device)
    mean = torch.as_tensor(scaler.mean, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=d["lr"])
    best_cost, best_state = val_decision_cost(), copy.deepcopy(model.state_dict())
    log(f"  MAE-trained model: val decision cost {best_cost:.2f} trips/decision")
    loader = DataLoader(train_ds, batch_size=d["batch_size"], shuffle=True)
    for epoch in range(d["epochs"]):
        model.train()
        steps, total_cost = 0, 0.0
        for b in loader:
            if steps >= d["steps_per_epoch"]:
                break
            bd = _to({k: v for k, v in b.items() if k in ("t", "x", "x_tod", "x_dow", "y", "y_tod", "y_wkd")}, device)
            out = model.forward_scaled(bd)
            mae = (out - (bd["y"] - mean) / std).abs().mean()
            theta = scaler.inverse(out)[:, 0, :, 0].clamp_min(0)                 # next-slot pickups [B, N]
            grad, cost = perturbed_gradient(theta.detach().cpu().numpy(), b, solver, dist, s["rebalance_budget"],
                                            km_cost_trips, d["sigma"], d["samples"], rng)
            surrogate = (theta * torch.as_tensor(grad, dtype=torch.float32, device=device)).sum(1).mean()
            loss = mae + d["beta"] * surrogate
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            steps += 1
            total_cost += cost
        vc = val_decision_cost()
        tag = ""
        if vc < best_cost:
            best_cost, best_state, tag = vc, copy.deepcopy(model.state_dict()), "  *best*"
        log(f"  epoch {epoch + 1}  train decision cost {total_cost / max(steps, 1):.2f}  val decision cost {vc:.2f}{tag}")
    model.load_state_dict(best_state)
    return model, ds, scaler
