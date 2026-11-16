"""Vehicle-level validation: learn V(zone, slot), then compare rebalancers x matchers taxi by taxi."""
from __future__ import annotations

import numpy as np

from ..control.baselines import ControlContext, LPRebalance
from ..ingestion.dataset import CityData
from ..models.value_table import ValueTable
from ..simulation.event_sim import FleetSimEvent
from ..simulation.fleet_sim import FleetSimTensor
from ..utils import mean_ci95
from .metrics import episode_metrics
from .run_eval import load_forecasts, make_controller


def init_share(city: CityData, cfg: dict) -> np.ndarray:
    share = FleetSimTensor(city, cfg, [0]).init_share.numpy().astype(float)
    return share / share.sum()


def learn_value_table(city: CityData, cfg: dict, n_days: int, log=print) -> ValueTable:
    """Policy evaluation on trajectories logged under robust-LP rebalancing + Hungarian matching."""
    forecasts, share = load_forecasts(cfg), init_share(city, cfg)
    skip = cfg["forecast"]["periodic_days"] + 1
    days = city.split_day_starts("train")[skip:][-n_days:]
    trans = []
    for d in days:
        ctx = ControlContext(city, cfg, forecasts, [d], cfg["sim"]["fleet_size"])
        tot = FleetSimEvent(city, cfg, d, matcher="hungarian").run(LPRebalance(ctx, "hi"), share, log_transitions=True)
        trans.append(tot["transitions"])
    trans = np.concatenate(trans)
    table = ValueTable(city.graph.num_zones, city.slots_per_day, cfg["event_sim"]["value_gamma"]).fit(trans)
    table.save(cfg["paths"]["output_dir"] / "value_table.npz")
    s = cfg["sim"]["start_slot"]
    log(f"[event] value table from {len(trans):,} transitions on {len(days)} days; "
        f"V range at 08:00 {table.v[:, 32].min():.0f}-{table.v[:, 32].max():.0f} AED (start slot {s})")
    return table


def evaluate_event(city: CityData, cfg: dict, controllers: list[str], matchers: list[str], log=print) -> dict:
    ev = cfg["event_sim"]
    forecasts, share = load_forecasts(cfg), init_share(city, cfg)
    days = city.split_day_starts("test")[:ev["eval_days"]]
    value = ValueTable.load(cfg["paths"]["output_dir"] / "value_table.npz") if "value" in matchers else None
    results = {}
    for ctrl_name in controllers:
        for m in matchers:
            per_day = []
            for d in days:
                ctx = ControlContext(city, cfg, forecasts, [d], cfg["sim"]["fleet_size"])
                ctrl = make_controller(ctrl_name, ctx, cfg) if ctrl_name != "none" else None
                tot = FleetSimEvent(city, cfg, d, matcher=m, value=value).run(ctrl, share)
                if ctrl is not None:
                    tot["decision_s"], tot["decisions"] = ctrl.decision_s, ctrl.decisions
                per_day.append(episode_metrics(tot))
            key = f"{ctrl_name} + {m}"
            results[key] = {k: mean_ci95([p[k] for p in per_day]) for k in per_day[0]}
            r = results[key]
            log(f"[event] {key:26s} served {r['served_rate_%'][0]:5.1f}%  wait {r['mean_wait_min'][0]:5.2f}  "
                f"p90 {r['p90_wait_min'][0]:5.1f}  empty km/trip {r['empty_km_per_trip'][0]:.2f}  profit {r['profit_AED'][0]:,.0f}")
    return results
