"""Rebalancing controllers. Each maps a simulator observation to flows [S, N, N].

Decision timing: after serving slot t, leftover idle taxis are repositioned for slot t+1.
"need" for slot t+1 = forecast pickups (from forecast made with data up to t) + riders still queued.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..ingestion.dataset import CityData
from .flow_solver import FlowSolver


class ControlContext:
    """Static data shared by controllers: forecasts, distances, normalisation scales."""

    def __init__(self, city: CityData, cfg: dict, forecasts: dict, day_starts: list[int], fleet: float):
        self.city, self.cfg = city, cfg
        self.fc_mean, self.fc_hi = forecasts["mean"], forecasts["hi"]
        self.day_starts = np.asarray(day_starts)
        self.fleet = fleet
        self.dist = city.graph.dist_km
        self.solver = FlowSolver(self.dist)
        self.supply_scale = fleet / city.graph.num_zones
        self.demand_scale = float(city.pickups[list(city.split_slots("train"))].mean()) + 1e-6
        self.spd = city.slots_per_day

    def forecast(self, t: int, which: str = "mean") -> np.ndarray:
        """[S, H, N] forecast for slots t+1 .. t+H, issued after slot t."""
        arr = self.fc_mean if which == "mean" else self.fc_hi
        return np.nan_to_num(arr[self.day_starts + t + 1])

    def budget(self, idle: np.ndarray) -> float:
        return self.cfg["sim"]["rebalance_budget"] * idle.sum()

    def features(self, obs: dict) -> torch.Tensor:
        """[S, N, 12] node features for learned policies (scaled so they are fleet-size invariant)."""
        ss, ds = self.supply_scale, self.demand_scale
        fc = torch.as_tensor(self.forecast(obs["t"]), dtype=torch.float32)            # [S,H,N]
        hi = torch.as_tensor(self.forecast(obs["t"], "hi")[:, 0], dtype=torch.float32)
        inc = obs["incoming"]
        ang = 2 * np.pi * obs["tod"] / self.spd
        s, n = obs["idle"].shape
        cols = [obs["idle"] / ss, inc[:, 0] / ss, inc[:, 1] / ss, inc[:, 2:].sum(1) / ss, obs["queue"] / ds]
        cols += [fc[:, h] / ds for h in range(min(4, fc.shape[1]))]
        cols += [hi / ds, torch.full((s, n), np.sin(ang)), torch.full((s, n), np.cos(ang))]
        return torch.stack(cols, dim=-1)


class Controller:
    name = "base"

    def __init__(self, ctx: ControlContext):
        self.ctx = ctx
        self.decision_s, self.decisions = 0.0, 0

    def act(self, obs: dict) -> torch.Tensor:
        t0 = time.perf_counter()
        flows = self._act(obs)
        self.decision_s += time.perf_counter() - t0
        self.decisions += 1
        return flows

    def _act(self, obs: dict) -> torch.Tensor:
        raise NotImplementedError

    def need(self, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def _solve_all(self, obs: dict, need: np.ndarray) -> torch.Tensor:
        idle, arr = obs["idle"].numpy(), obs["incoming"][:, 0].numpy()
        flows = [self.ctx.solver.solve(idle[i], arr[i], need[i], self.ctx.budget(idle[i])) for i in range(len(idle))]
        return torch.as_tensor(np.stack(flows), dtype=torch.float32)


class NoRebalance(Controller):
    name = "none"

    def _act(self, obs):
        s, n = obs["idle"].shape
        return torch.zeros(s, n, n)


class GreedyNeighbor(Controller):
    """Each surplus zone sends taxis to its adjacent zone with the largest forecast deficit."""
    name = "greedy"

    def _act(self, obs):
        adj = self.ctx.city.graph.adjacency
        need = self.ctx.forecast(obs["t"])[:, 0] + obs["queue"].numpy()
        idle, arr = obs["idle"].numpy(), obs["incoming"][:, 0].numpy()
        s, n = idle.shape
        flows = np.zeros((s, n, n))
        for k in range(s):
            supply = idle[k] + arr[k]
            surplus = np.minimum(np.clip(supply - need[k], 0, None), idle[k])
            deficit = np.clip(need[k] - supply, 0, None)
            budget = self.ctx.budget(idle[k])
            for i in np.argsort(-surplus):
                if surplus[i] <= 0 or budget <= 0:
                    break
                nb = np.nonzero(adj[i])[0]
                j = nb[np.argmax(deficit[nb])]
                move = min(surplus[i], deficit[j], budget)
                if move > 0:
                    flows[k, i, j] += move
                    deficit[j] -= move
                    budget -= move
        return torch.as_tensor(flows, dtype=torch.float32)


class LPRebalance(Controller):
    """Min-cost flow toward forecast need (mean, conformal upper bound, or perfect information)."""

    def __init__(self, ctx, mode: str):
        super().__init__(ctx)
        self.mode = mode
        self.name = {"mean": "lp", "hi": "lp_robust", "oracle": "oracle_lp"}[mode]

    def _act(self, obs):
        if self.mode == "oracle":
            demand = obs["next_true_pickups"].numpy()
        else:
            demand = self.ctx.forecast(obs["t"], self.mode)[:, 0]
        return self._solve_all(obs, demand + obs["queue"].numpy())
