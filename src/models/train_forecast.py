"""Train/evaluate forecasters, calibrate conformal intervals, export forecasts for control."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..evaluation.metrics import forecast_metrics
from ..ingestion.dataset import CityData, DemandDataset, Scaler, valid_targets
from .conformal import SplitConformal
from .forecasting import Forecaster, build_forecaster


def make_datasets(city: CityData, fcfg: dict):
    scaler = Scaler(city.series[list(city.split_slots("train"))])
    ds = {}
    for split in ("train", "val", "test"):
        t = valid_targets(city, city.split_slots(split), fcfg["history"], fcfg["periodic_days"], fcfg["horizon"])
        ds[split] = DemandDataset(city, t, scaler, fcfg)
    return ds, scaler


def _to(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model: Forecaster, ds: DemandDataset, device, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    out = [model(_to(b, device)).cpu() for b in DataLoader(ds, batch_size=batch_size)]
    return torch.cat(out)


def train_forecaster(model: Forecaster, ds: dict, fcfg: dict, device, log=print) -> Forecaster:
    model.to(device)
    if not model.trainable:
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=fcfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fcfg["epochs"])
    std = torch.as_tensor(model.scaler.std, dtype=torch.float32, device=device)
    loader = DataLoader(ds["train"], batch_size=fcfg["batch_size"], shuffle=True)
    best, best_mae, patience = None, float("inf"), 0
    for epoch in range(fcfg["epochs"]):
        model.train()
        t0, total = time.time(), 0.0
        for b in loader:
            b = _to(b, device)
            loss = ((model.forward_scaled(b) - (b["y"] - torch.as_tensor(model.scaler.mean, device=device)) / std)
                    .abs().mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b["t"])
        sched.step()
        val_mae = forecast_metrics(predict(model, ds["val"], device), ds["val"].y)["all"]["MAE"]
        log(f"  epoch {epoch + 1:3d}  train_loss {total / len(ds['train']):.4f}  val_MAE {val_mae:.3f}  ({time.time() - t0:.1f}s)")
        if val_mae < best_mae - 1e-4:
            best, best_mae, patience = copy.deepcopy(model.state_dict()), val_mae, 0
        else:
            patience += 1
            if patience >= 8:
                log("  early stop")
                break
    model.load_state_dict(best)
    return model


def run_forecasting(city: CityData, cfg: dict, device, names: list[str], log=print) -> dict:
    """Train each model, report test metrics, export the configured model's forecasts."""
    fcfg = cfg["forecast"]
    ds, scaler = make_datasets(city, fcfg)
    log(f"samples: train {len(ds['train'])}  val {len(ds['val'])}  test {len(ds['test'])}")
    results, export = {}, None
    for name in names:
        log(f"[forecast] {name}")
        model = train_forecaster(build_forecaster(name, city, scaler, fcfg), ds, fcfg, device, log)
        conf = SplitConformal(fcfg["conformal_alpha"]).calibrate(predict(model, ds["val"], device), ds["val"].y)
        pred = predict(model, ds["test"], device)
        lo, hi = conf.interval(pred)
        results[name] = forecast_metrics(pred, ds["test"].y, lo, hi)
        log(f"  test MAE {results[name]['all']['MAE']:.3f}  RMSE {results[name]['all']['RMSE']:.3f}  "
            f"MAPE {results[name]['all']['MAPE']:.1f}%  coverage {results[name]['all']['coverage']:.1f}%")
        if model.trainable:
            torch.save(model.state_dict(), cfg["paths"]["output_dir"] / f"forecaster_{name}.pt")
        if name == fcfg["model"]:
            export = (model, conf)
    if export is not None:
        export_forecasts(city, cfg, export[0], export[1], scaler, device)
    return results


def forecast_arrays(city: CityData, cfg: dict, model: Forecaster, conf: SplitConformal, scaler: Scaler, device) -> dict:
    """Pickup forecasts for every slot: mean/lo/hi [T, H, N] (NaN where history is insufficient)."""
    fcfg = cfg["forecast"]
    t_all = valid_targets(city, range(0, len(city.slot_index)), fcfg["history"], fcfg["periodic_days"], fcfg["horizon"])
    ds = DemandDataset(city, t_all, scaler, fcfg)
    pred = predict(model, ds, device)
    lo, hi = conf.interval(pred)
    shape = (len(city.slot_index), fcfg["horizon"], city.graph.num_zones)
    out = {k: np.full(shape, np.nan, dtype=np.float32) for k in ("mean", "lo", "hi")}
    for k, v in (("mean", pred), ("lo", lo), ("hi", hi)):
        out[k][t_all] = v[..., 0].numpy()
    return out


def export_forecasts(city: CityData, cfg: dict, model: Forecaster, conf: SplitConformal, scaler: Scaler, device,
                     filename: str = "forecasts.npz"):
    out = forecast_arrays(city, cfg, model, conf, scaler, device)
    np.savez_compressed(cfg["paths"]["output_dir"] / filename, model=cfg["forecast"]["model"], **out)


def load_trained_forecaster(name: str, city: CityData, cfg: dict, device, weights: str | None = None):
    """Rebuild a forecaster with its training-period scaler, load weights, and recalibrate conformal on val."""
    fcfg = cfg["forecast"]
    ds, scaler = make_datasets(city, fcfg)
    model = build_forecaster(name, city, scaler, fcfg).to(device)
    if model.trainable:
        path = cfg["paths"]["output_dir"] / (weights or f"forecaster_{name}.pt")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    conf = SplitConformal(fcfg["conformal_alpha"]).calibrate(predict(model, ds["val"], device), ds["val"].y)
    return model, conf, scaler, ds
