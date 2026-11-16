"""City data container, chronological splits and the forecasting dataset."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import nyc_tlc, synthetic
from .graph import ZoneGraph


@dataclass
class CityData:
    graph: ZoneGraph
    od: np.ndarray | None     # [T, N, N] trip counts by pickup slot, origin, destination (None: series only)
    slot_index: pd.DatetimeIndex
    slots_per_day: int
    split_days: tuple[int, int, int]
    pickup_series: np.ndarray | None = None    # [T, N], used when OD is too large to keep (NYC)
    dropoff_series: np.ndarray | None = None

    @property
    def pickups(self) -> np.ndarray:
        if self.od is None:
            return self.pickup_series.astype(np.float32)
        return self.od.sum(2).astype(np.float32)

    @property
    def dropoffs(self) -> np.ndarray:
        if self.od is None:
            return self.dropoff_series.astype(np.float32)
        return self.od.sum(1).astype(np.float32)

    @property
    def series(self) -> np.ndarray:
        """[T, N, 2] pickups and destination demand."""
        return np.stack([self.pickups, self.dropoffs], axis=-1)

    @property
    def tod(self) -> np.ndarray:
        return np.arange(len(self.slot_index)) % self.slots_per_day

    @property
    def dow(self) -> np.ndarray:
        return self.slot_index.dayofweek.to_numpy()

    def split_slots(self, name: str) -> range:
        """Chronological splits; 'stress' covers any scenario days appended after the test period."""
        tr, va, te = self.split_days
        spd = self.slots_per_day
        total = len(self.slot_index) // spd
        bounds = {"train": (0, tr), "val": (tr, tr + va), "test": (tr + va, tr + va + te), "stress": (tr + va + te, total)}
        a, b = bounds[name]
        return range(a * spd, b * spd)

    def split_day_starts(self, name: str) -> list[int]:
        r = self.split_slots(name)
        return list(range(r.start, r.stop, self.slots_per_day))


def prepare_data(cfg: dict, rng: np.random.Generator) -> CityData:
    src = cfg["data"]["source"]
    raw = synthetic.generate(cfg, rng) if src == "synthetic" else nyc_tlc.generate(cfg)
    path = cfg["paths"]["data_dir"] / "processed" / f"city_{src}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"od": raw["od"]} if raw.get("od") is not None else {"pickups": raw["pickups"], "dropoffs": raw["dropoffs"]}
    np.savez_compressed(path, slot_start=str(raw["slot_index"][0]), n_slots=len(raw["slot_index"]),
                        **arrays, **raw["graph"].to_npz_dict())
    return load_data(cfg)


def with_scenario_days(cfg: dict, base: CityData, scenario_days: list[str]) -> CityData:
    """The synthetic city plus appended stress-test days (same city, same normal days, same splits)."""
    raw = synthetic.generate(cfg, np.random.default_rng(cfg["seed"]), scenario_days)
    n_base = len(base.slot_index)
    assert np.array_equal(raw["od"][:n_base], base.od), "regenerated normal days differ from stored data"
    return CityData(base.graph, raw["od"], raw["slot_index"], base.slots_per_day, base.split_days)


def load_data(cfg: dict) -> CityData:
    src = cfg["data"]["source"]
    path = cfg["paths"]["data_dir"] / "processed" / f"city_{src}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python main.py data` first.")
    d = np.load(path)
    slot_min = cfg["data"]["slot_minutes"]
    idx = pd.date_range(str(d["slot_start"]), periods=int(d["n_slots"]), freq=f"{slot_min}min")
    spd = 24 * 60 // slot_min
    days = len(idx) // spd
    f_tr, f_va, _ = cfg["data"]["split"]
    tr, va = int(round(days * f_tr)), int(round(days * f_va))
    if "od" in d.files:
        return CityData(ZoneGraph.from_npz(d), d["od"], idx, spd, (tr, va, days - tr - va))
    return CityData(ZoneGraph.from_npz(d), None, idx, spd, (tr, va, days - tr - va), d["pickups"], d["dropoffs"])


class Scaler:
    def __init__(self, series: np.ndarray):
        self.mean = series.reshape(-1, series.shape[-1]).mean(0)
        self.std = series.reshape(-1, series.shape[-1]).std(0) + 1e-6

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * s + m


def valid_targets(city: CityData, slots: range, history: int, periodic_days: int, horizon: int) -> np.ndarray:
    first = max(history, periodic_days * city.slots_per_day)
    t = np.arange(max(slots.start, first), slots.stop)
    return t[t + horizon <= len(city.slot_index)]


class DemandDataset(Dataset):
    """Each sample predicts slots [t, t+H) from 12 recent slots plus the same slot on previous days."""

    def __init__(self, city: CityData, targets: np.ndarray, scaler: Scaler, fcfg: dict):
        h, p, horizon = fcfg["history"], fcfg["periodic_days"], fcfg["horizon"]
        spd = city.slots_per_day
        series = city.series
        scaled = scaler.transform(series).astype(np.float32)
        recent = targets[:, None] + np.arange(-h, 0)[None]                    # [S, h]
        periodic = targets[:, None] - spd * np.arange(p, 0, -1)[None]         # [S, p]
        x_idx = np.concatenate([periodic, recent], axis=1)                    # oldest first
        y_idx = targets[:, None] + np.arange(horizon)[None]
        tod, dow = city.tod, city.dow
        self.t = torch.as_tensor(targets)
        self.x = torch.as_tensor(scaled[x_idx])                               # [S, P, N, 2]
        self.x_tod = torch.as_tensor(tod[x_idx])
        self.x_dow = torch.as_tensor(dow[x_idx])
        self.y = torch.as_tensor(series[y_idx])                               # raw counts [S, H, N, 2]
        self.y_tod = torch.as_tensor(tod[y_idx])
        self.y_wkd = torch.as_tensor((dow[y_idx] >= 5).astype(np.int64))

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return dict(t=self.t[i], x=self.x[i], x_tod=self.x_tod[i], x_dow=self.x_dow[i],
                    y=self.y[i], y_tod=self.y_tod[i], y_wkd=self.y_wkd[i])
