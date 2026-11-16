"""Zone graph G=(V,E): zone geometry, adjacency and time-dependent travel times."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def speed_profile_kmh(slot_tod: np.ndarray, slots_per_day: int, base: float, peak: float) -> np.ndarray:
    """Average road speed by time of day, with AM (07-09) and PM (17-20) congestion peaks."""
    hour = slot_tod * 24.0 / slots_per_day
    congestion = np.exp(-((hour - 8.0) ** 2) / 2.0) + np.exp(-((hour - 18.5) ** 2) / 3.0)
    congestion = np.clip(congestion, 0.0, 1.0)
    return base - (base - peak) * congestion


@dataclass
class ZoneGraph:
    coords_km: np.ndarray        # [N, 2]
    dist_km: np.ndarray          # [N, N] road distance; diagonal = mean intra-zone trip
    adjacency: np.ndarray        # [N, N] bool, no self loops
    names: list[str]
    types: list[str]

    @property
    def num_zones(self) -> int:
        return len(self.names)

    @property
    def edge_index(self) -> torch.Tensor:
        src, dst = np.nonzero(self.adjacency)
        return torch.as_tensor(np.stack([src, dst]), dtype=torch.long)

    @property
    def edge_weight(self) -> torch.Tensor:
        src, dst = np.nonzero(self.adjacency)
        d = self.dist_km[src, dst]
        return torch.as_tensor(np.exp(-d / d.mean()), dtype=torch.float32)

    def neighbor_table(self, k: int) -> np.ndarray:
        """[N, k+1] table: column 0 is the zone itself, then up to k nearest neighbours.

        Zones with fewer than k neighbours are padded with themselves (a move to self = stay).
        """
        n = self.num_zones
        table = np.repeat(np.arange(n)[:, None], k + 1, axis=1)
        for i in range(n):
            nb = np.nonzero(self.adjacency[i])[0]
            nb = nb[np.argsort(self.dist_km[i, nb])][:k]
            table[i, 1:1 + len(nb)] = nb
        return table

    def travel_minutes(self, speed_kmh: float) -> np.ndarray:
        return self.dist_km / speed_kmh * 60.0

    @staticmethod
    def from_coords(coords_km: np.ndarray, names, types, detour: float, link_radius_km: float,
                    intra_zone_km: float) -> "ZoneGraph":
        diff = coords_km[:, None, :] - coords_km[None, :, :]
        euclid = np.sqrt((diff ** 2).sum(-1))
        manhattan = np.abs(diff).sum(-1)
        dist = detour * 0.5 * (euclid + manhattan)  # road distance between crow-fly and grid
        np.fill_diagonal(dist, intra_zone_km)
        adj = (euclid <= link_radius_km) & (euclid > 0)
        return ZoneGraph(coords_km, dist, adj, list(names), list(types))

    def to_npz_dict(self) -> dict:
        return dict(coords_km=self.coords_km, dist_km=self.dist_km, adjacency=self.adjacency,
                    zone_names=np.array(self.names), zone_types=np.array(self.types))

    @staticmethod
    def from_npz(d) -> "ZoneGraph":
        return ZoneGraph(d["coords_km"], d["dist_km"], d["adjacency"].astype(bool),
                         [str(s) for s in d["zone_names"]], [str(s) for s in d["zone_types"]])
