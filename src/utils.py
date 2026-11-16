"""Shared helpers: config loading, seeding, device selection, paths."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "output_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = p if p.is_absolute() else PROJECT_ROOT / p
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def mean_ci95(values) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation 95% confidence interval."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
