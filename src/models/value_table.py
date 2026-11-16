"""Spatio-temporal state values V(zone, slot) learned from simulated driver trajectories.

Policy evaluation by backward induction over the operating day (Xu et al., KDD 2018):
    V(z, s) = mean over transitions starting at (z, s) of  [ r + gamma^(s' - s) * V(z', s') ]
Transitions are completed trips (reward = fare - pickup cost) and idle slots (reward = -cruise cost).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ValueTable:
    def __init__(self, num_zones: int, slots_per_day: int, gamma: float):
        self.v = np.zeros((num_zones, slots_per_day + 8))      # padded: values past the day end are 0
        self.gamma, self.spd = gamma, slots_per_day

    def lookup(self, zone, slot):
        slot = np.clip(np.asarray(slot), 0, self.v.shape[1] - 1)
        return self.v[zone, slot]

    def fit(self, transitions: np.ndarray, smoothing: float = 1.0) -> "ValueTable":
        """transitions: [M, 5] = zone, slot, next_zone, next_slot, reward (next_slot > slot)."""
        z, s, z2, s2, r = (transitions[:, i] for i in range(5))
        z, s, z2, s2 = (a.astype(int) for a in (z, s, z2, s2))
        s2 = np.clip(s2, s + 1, self.v.shape[1] - 1)
        n, spd = self.v.shape[0], self.spd
        for slot in range(spd - 1, -1, -1):
            m = s == slot
            if not m.any():
                continue
            target = r[m] + self.gamma ** (s2[m] - slot) * self.v[z2[m], s2[m]]
            total = np.bincount(z[m], weights=target, minlength=n)
            count = np.bincount(z[m], minlength=n)
            prior = target.mean()                               # shrink sparse zones toward the slot mean
            self.v[:, slot] = (total + smoothing * prior) / (count + smoothing)
            self.v[count == 0, slot] = prior if count.sum() else 0.0
        return self

    def save(self, path: Path) -> None:
        np.savez(path, v=self.v, gamma=self.gamma, spd=self.spd)

    @staticmethod
    def load(path: Path) -> "ValueTable":
        d = np.load(path)
        t = ValueTable(d["v"].shape[0], int(d["spd"]), float(d["gamma"]))
        t.v = d["v"]
        return t
