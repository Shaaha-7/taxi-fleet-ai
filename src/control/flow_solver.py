"""Min-cost rebalancing LP: the optimisation layer under every LP-based controller.

    min  sum_ij d_ij x_ij + lam * sum_i s_i
    s.t. idle_i - out_i + in_i + arrivals_i + s_i >= need_i     (cover next-slot need; s = shortfall)
         out_i <= idle_i                                        (move only idle taxis)
         sum_ij x_ij <= budget                                  (operational cap per slot)
         x, s >= 0,  x_ij = 0 if d_ij > max_km

Moving one taxi is worth it when it closes a shortfall and d_ij < lam, so `lam` is the
largest detour (km) the operator accepts to cover one unit of unmet demand.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


class FlowSolver:
    def __init__(self, dist_km: np.ndarray, lam_km: float = 12.0, max_km: float = 15.0):
        self.dist = dist_km
        n = len(dist_km)
        mask = (dist_km <= max_km) & ~np.eye(n, dtype=bool)
        self.src, self.dst = np.nonzero(mask)
        self.cost = dist_km[self.src, self.dst]
        self.n, self.lam = n, lam_km
        p = len(self.src)
        rows = np.concatenate([self.src, self.dst])
        cols = np.concatenate([np.arange(p), np.arange(p)])
        vals = np.concatenate([np.ones(p), -np.ones(p)])
        net_out = sp.csr_matrix((vals, (rows, cols)), shape=(n, p))           # out_i - in_i
        out_only = sp.csr_matrix((np.ones(p), (self.src, np.arange(p))), shape=(n, p))
        eye = sp.identity(n, format="csr")
        zeros = sp.csr_matrix((n, n))
        self.A = sp.vstack([sp.hstack([net_out, -eye]),
                            sp.hstack([out_only, zeros]),
                            sp.hstack([sp.csr_matrix(np.ones((1, p))), sp.csr_matrix((1, n))])]).tocsr()
        self.c = np.concatenate([self.cost, np.full(n, lam_km)])

    def solve(self, idle: np.ndarray, arrivals: np.ndarray, need: np.ndarray, budget: float) -> np.ndarray:
        if idle.sum() < 1e-6 or budget <= 0:
            return np.zeros((self.n, self.n))
        b = np.concatenate([idle + arrivals - need, idle, [budget]])
        res = linprog(self.c, A_ub=self.A, b_ub=b, bounds=(0, None), method="highs")
        flows = np.zeros((self.n, self.n))
        if res.status == 0:
            x = res.x[:len(self.src)]
            flows[self.src, self.dst] = np.where(x > 1e-6, x, 0.0)
        return flows
