"""Ramadan and special-event stress tests.

Stress days are appended after the test period (same city, same trained models, no retraining), so
they measure robustness to regimes the models never saw: Ramadan's shifted day with an iftar trough
and driver breaks, a concert egress spike, and New Year's Eve closures followed by a post-midnight
surge in the zones around Downtown. Magnitudes are assumptions (see the research notes).
"""
from __future__ import annotations

import numpy as np

from ..ingestion.dataset import CityData, DemandDataset, valid_targets, with_scenario_days
from ..models.train_forecast import forecast_arrays, load_trained_forecaster, predict
from ..utils import get_device
from .metrics import forecast_metrics
from .run_eval import evaluate_methods


def _scenario_days(city: CityData, schedule: list[str]) -> dict[str, list[int]]:
    first = city.split_slots("stress").start // city.slots_per_day
    out: dict[str, list[int]] = {}
    for i, kind in enumerate(schedule):
        out.setdefault(kind, []).append(first + i)
    return out


def run_stress(base: CityData, cfg: dict, methods: list[str], forecasters: list[str], log=print) -> dict:
    st = cfg["stress"]
    device = get_device(cfg)
    city = with_scenario_days(cfg, base, st["schedule"])
    kinds = _scenario_days(city, st["schedule"])
    spd, fcfg = city.slots_per_day, cfg["forecast"]
    results = {"forecast": {}, "control": {}}

    # ---------------- forecasting under regime shift
    log("[stress] forecast accuracy by scenario (whole operating days)")
    export = None
    for name in forecasters:
        model, conf, scaler, ds = load_trained_forecaster(name, city, cfg, device)
        res = {"normal test": forecast_metrics(predict(model, ds["test"], device), ds["test"].y)["all"]}
        for kind in ("ramadan", "concert", "nye", "nye_after"):
            slots = np.concatenate([np.arange(d * spd, (d + 1) * spd) for d in kinds[kind]])
            t = valid_targets(city, range(slots.min(), slots.max() + 1), fcfg["history"], fcfg["periodic_days"], fcfg["horizon"])
            sds = DemandDataset(city, t, scaler, fcfg)
            pred = predict(model, sds, device)
            lo, hi = conf.interval(pred)
            res[kind] = forecast_metrics(pred, sds.y, lo, hi)["all"]
        results["forecast"][name] = res
        log(f"  {name:10s} " + "  ".join(f"{k} MAE {v['MAE']:.2f}" for k, v in res.items()))
        if name == fcfg["model"]:
            export = forecast_arrays(city, cfg, model, conf, scaler, device)

    # ---------------- control under regime shift
    iftar = np.zeros(spd)
    iftar[int(18.5 * spd / 24)] = st["iftar_break_frac"]
    runs = {
        "ramadan": dict(days=kinds["ramadan"], sim_kwargs=dict(offline_frac=iftar)),
        "concert": dict(days=kinds["concert"], sim_kwargs={}),
        "nye": dict(days=kinds["nye"], sim_kwargs=dict(start_slot=st["nye_start_slot"])),
    }
    for scen, r in runs.items():
        log(f"[stress] control: {scen} ({len(r['days'])} day(s))")
        days = [d * spd for d in r["days"]]
        results["control"][scen] = evaluate_methods(city, cfg, methods, forecasts=export, days=days,
                                                    sim_kwargs=r["sim_kwargs"], log=log)
    return results


def write_stress_tables(res: dict, path) -> str:
    lines = ["## Stress tests (models trained on normal days only)", "",
             "### Pickup forecast MAE by regime", ""]
    regimes = list(next(iter(res["forecast"].values())))
    lines += ["| Model | " + " | ".join(regimes) + " |", "|" + "---|" * (len(regimes) + 1)]
    for name, r in res["forecast"].items():
        lines.append(f"| {name} | " + " | ".join(f"{r[k]['MAE']:.2f}" for k in regimes) + " |")
    cov = {n: {k: r[k].get("coverage") for k in regimes if "coverage" in r[k]} for n, r in res["forecast"].items()}
    lines += ["", "90% conformal interval coverage (%): " +
              "; ".join(f"{n}: " + ", ".join(f"{k} {v:.0f}" for k, v in c.items()) for n, c in cov.items()), ""]
    for scen, table in res["control"].items():
        lines += [f"### Control under {scen}", "",
                  "| Method | served % | mean wait | p90 wait | empty km/trip | profit AED |", "|---|---|---|---|---|---|"]
        for m, r in table.items():
            lines.append(f"| {m} | {r['served_rate_%'][0]:.1f} | {r['mean_wait_min'][0]:.2f} | {r['p90_wait_min'][0]:.1f} | "
                         f"{r['empty_km_per_trip'][0]:.2f} | {r['profit_AED'][0]:,.0f} |")
        lines.append("")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text
