"""Batched rider-taxi matching for the vehicle-level simulator.

greedy     first-come-first-served: each rider (oldest first) takes the nearest free taxi
hungarian  batch assignment minimising total pickup time (scipy linear_sum_assignment)
value      batch assignment maximising  fare - pickup cost + gamma^dt V(dropoff zone, t') - V(taxi zone, t),
           the learning-and-planning edge weight of Xu et al. (KDD 2018) used in DiDi-style dispatch
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

INFEASIBLE = -1e9


def road_km(a: np.ndarray, b: np.ndarray, detour: float) -> np.ndarray:
    """Pairwise road distance [len(a), len(b)] between points (km)."""
    d = a[:, None, :] - b[None, :, :]
    return detour * 0.5 * (np.sqrt((d ** 2).sum(-1)) + np.abs(d).sum(-1))


def match(method: str, veh_pos, veh_zone, req_pos, req_dest_zone, req_fare, req_trip_min, req_age,
          slot: int, detour: float, max_pickup_km: float, empty_cost: float, value=None, gamma: float = 0.95,
          kmh: float = 30.0, slot_min: int = 15, pickup_penalty: float = 0.0):
    """Return list of (vehicle index, request index) pairs."""
    if len(veh_pos) == 0 or len(req_pos) == 0:
        return []
    pk = road_km(veh_pos, req_pos, detour)                       # [V, R]
    feasible = pk <= max_pickup_km
    if method == "greedy":
        pairs, taken = [], np.zeros(len(veh_pos), dtype=bool)
        for r in np.argsort(-req_age):
            cand = np.where(feasible[:, r] & ~taken)[0]
            if len(cand):
                v = cand[np.argmin(pk[cand, r])]
                taken[v] = True
                pairs.append((v, r))
        return pairs
    if method == "hungarian":
        score = -pk
    elif method == "value":
        # Semi-MDP step k counts pickup ETA plus trip time (DiDi CVNet / V1D3 convention).
        k = np.maximum(np.ceil((pk / kmh * 60 + req_trip_min[None, :]) / slot_min), 1).astype(int)   # [V, R]
        v_now = value.lookup(veh_zone, slot)                     # [V]
        v_next = value.lookup(np.broadcast_to(req_dest_zone[None, :], k.shape), slot + k)
        # pickup_penalty (AED/km) is the Omega * U_ij service-quality term that keeps pickups short.
        score = req_fare[None, :] + gamma ** k * v_next - (empty_cost + pickup_penalty) * pk - v_now[:, None]
    else:
        raise ValueError(method)
    score = np.where(feasible, score, INFEASIBLE)
    rows, cols = linear_sum_assignment(score, maximize=True)
    return [(r, c) for r, c in zip(rows, cols) if feasible[r, c]]
