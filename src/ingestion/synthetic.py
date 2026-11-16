"""Synthetic Dubai-like demand generator.

No public trip-level Dubai data exists (RTA open data is aggregate only), so the default
pipeline generates a calibrated synthetic city: a coastal corridor running from Jebel Ali
(west) to the airport (east), with zone types, time-of-day profiles, a Sat-Sun weekend
(UAE since 2022), day-level noise and evening event spikes. Output OD counts per 15-min
slot have the same layout as the NYC TLC aggregator, so models train on either source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import ZoneGraph

# Hourly pickup profiles (relative, 24 values) per zone type.
PICKUP_PROFILES = {
    "residential": [.3, .2, .1, .1, .2, .6, 1.4, 2.2, 2.0, 1.2, .9, .9, 1.0, .9, .9, 1.0, 1.2, 1.3, 1.3, 1.2, 1.1, 1.0, .8, .5],
    "cbd":         [.3, .2, .1, .1, .1, .2, .5, .9, 1.0, 1.0, 1.1, 1.3, 1.5, 1.3, 1.2, 1.4, 1.9, 2.4, 2.3, 1.8, 1.4, 1.1, .8, .5],
    "leisure":     [.8, .6, .3, .1, .1, .1, .2, .3, .5, .7, .9, 1.0, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 2.1, 2.3, 2.4, 2.3, 1.9, 1.3],
    "airport":     [1.4, 1.2, 1.0, .9, 1.1, 1.4, 1.6, 1.5, 1.3, 1.2, 1.1, 1.1, 1.2, 1.3, 1.4, 1.4, 1.3, 1.3, 1.3, 1.3, 1.4, 1.5, 1.6, 1.5],
    "old_town":    [.6, .4, .2, .1, .1, .3, .6, .9, 1.1, 1.2, 1.3, 1.3, 1.3, 1.2, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.7, 1.5, 1.2, .9],
    "industrial":  [.2, .1, .1, .1, .3, 1.2, 1.6, 1.0, .5, .4, .4, .5, .5, .4, .4, .6, 1.4, 1.5, .8, .5, .4, .3, .2, .2],
}
# Relative size of each zone type (share of citywide pickups before profiles).
TYPE_WEIGHT = {"residential": 1.0, "cbd": 3.0, "leisure": 2.2, "airport": 4.0, "old_town": 2.2, "industrial": 0.4}
# How strongly a zone type attracts trips (destinations), by period of day.
DEST_WEIGHT_AM = {"residential": 0.6, "cbd": 3.5, "leisure": 0.8, "airport": 2.0, "old_town": 1.8, "industrial": 1.5}
DEST_WEIGHT_PM = {"residential": 2.2, "cbd": 1.0, "leisure": 2.5, "airport": 2.0, "old_town": 1.8, "industrial": 0.3}
# Weekend multipliers (Sat/Sun) and Friday multipliers per zone type.
WEEKEND = {"residential": 0.85, "cbd": 0.55, "leisure": 1.35, "airport": 1.10, "old_town": 1.10, "industrial": 0.45}
FRIDAY = {"residential": 0.95, "cbd": 0.85, "leisure": 1.15, "airport": 1.05, "old_town": 1.05, "industrial": 0.8}


def _zone_layout(rows: int, cols: int) -> tuple[list[str], list[str]]:
    """Assign Dubai-like zone types along a coastal corridor (row 0 = coast, col 0 = Jebel Ali)."""
    names, types = [], []
    for rr in range(rows):
        r = rr * 4 // rows                  # the layout below is written for 4 rows; finer grids map onto it
        for c in range(cols):
            frac = c / max(cols - 1, 1)
            if frac < 0.15:
                t, n = "industrial", "JebelAli"
            elif frac < 0.35:
                t, n = ("leisure", "Marina") if r <= 1 else ("residential", "JVC")
            elif frac < 0.55:
                t, n = ("leisure", "MallOfEmirates") if r == 1 else ("residential", "AlBarsha")
            elif frac < 0.68:
                t, n = ("industrial", "AlQuoz") if r >= 2 else ("residential", "Jumeirah")
            elif frac < 0.82:
                t, n = ("cbd", "Downtown") if r <= 1 else ("leisure", "DubaiMall") if r == 2 else ("residential", "AlKhail")
            elif frac < 0.95:
                t, n = ("old_town", "BurDubai") if r <= 1 else ("residential", "Mirdif")
            else:
                t, n = ("airport", "DXB") if r == 1 else ("old_town", "Deira") if r == 0 else ("residential", "Rashidiya")
            names.append(f"{n}_{rr}{c}")
            types.append(t)
    return names, types


def _hourly_to_slots(profile: list[float], slots_per_day: int) -> np.ndarray:
    hours = np.arange(slots_per_day) * 24.0 / slots_per_day
    return np.interp(hours, np.arange(25), profile + profile[:1])


def build_city(cfg: dict) -> ZoneGraph:
    s = cfg["data"]["synthetic"]
    rows, cols, cell = s["grid_rows"], s["grid_cols"], s["cell_km"]
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    coords = np.stack([cc.ravel() * cell, rr.ravel() * cell], axis=1).astype(float)
    names, types = _zone_layout(rows, cols)
    return ZoneGraph.from_coords(coords, names, types, detour=cfg["sim"]["detour_factor"],
                                 link_radius_km=1.5 * cell, intra_zone_km=0.5 * cell)


# Ramadan hourly multipliers on the normal profile (assumptions shaped by Gulf News / Careem timings,
# see research_notes/Upgrade implementation details/transfer_ramadan_events.md; no published
# Ramadan taxi volumes exist). Iftar is anchored at 18:30.
RAMADAN_WINDOWS = [((0.0, 3.0), 2.0), ((3.0, 4.5), 1.35), ((6.0, 8.5), 0.65), ((8.5, 10.5), 0.9),
                   ((13.5, 16.0), 1.3), ((17.0, 18.5), 1.2), ((18.5, 19.5), 0.4), ((20.5, 23.99), 1.45)]
RAMADAN_DAILY = 0.95
IFTAR_HOURS = (18.5, 19.5)


def ramadan_multiplier(hour: np.ndarray) -> np.ndarray:
    m = np.ones_like(hour)
    for (a, b), f in RAMADAN_WINDOWS:
        m[(hour >= a) & (hour < b)] = f
    return m


def _apply_scenario(kind, lam, att, hour, graph, types, rng, log):
    """Modify one day's pickup intensity lam [spd, N] and attractiveness att [spd, N] in place."""
    n = len(types)
    adj = graph.adjacency
    if kind == "ramadan":
        total = lam.sum()
        lam *= ramadan_multiplier(hour)[:, None]
        evening = ((hour >= 20.5) | (hour < 3.0))[:, None]
        hubs = np.isin(types, ["leisure", "cbd", "old_town"])[None]
        lam *= np.where(evening & hubs, 1.15, 1.0)                       # malls / Downtown late-night uplift
        lam *= RAMADAN_DAILY * total / lam.sum()                         # reshape, don't inflate, the day
    elif kind == "concert":
        venues = [i for i, t in enumerate(types) if t == "leisure"]
        z = int(rng.choice(venues))
        att[(hour >= 17) & (hour < 20), z] *= 2.5                         # arrival ramp (drop-offs)
        t_end = 22.0
        decay = np.exp(-np.clip(hour - t_end, 0, None) * 60 / 40.0)        # tau = 40 min
        spike = np.where((hour >= t_end) & (hour < t_end + 1.5), decay, 0.0)
        lam[:, z] *= 1 + 4.0 * spike                                      # 5x egress peak
        lam[:, adj[z]] *= (1 + 0.6 * spike)[:, None]                      # neighbours 1.6x
        log.append(("concert", graph.names[z]))
    elif kind == "nye":
        cbd = np.isin(types, ["cbd"])
        lam[np.ix_(hour >= 16, cbd)] *= 0.35                              # road closures around Downtown
        att[np.ix_((hour >= 14) & (hour < 21), cbd)] *= 3.0               # crowds arriving for fireworks
        log.append(("nye", "Downtown"))
    elif kind == "nye_after":
        cbd = np.isin(types, ["cbd"])
        ring = adj[cbd].any(0) & ~cbd
        early = (hour >= 0.5) & (hour < 3.0)
        lam[np.ix_(early, ring)] *= 6.0                                   # egress via adjacent zones
        lam[np.ix_(early, cbd)] *= 2.0
    elif kind != "normal":
        raise ValueError(kind)


def generate(cfg: dict, rng: np.random.Generator, scenario_days: list[str] | None = None) -> dict:
    """Generate `days` normal days, then optionally extra scenario days continuing the same city.

    The random stream for the normal days is identical with or without scenario days, so stress
    days can be appended to an existing dataset without changing it.
    """
    s = cfg["data"]["synthetic"]
    spd = 24 * 60 // cfg["data"]["slot_minutes"]
    graph = build_city(cfg)
    n, types = graph.num_zones, graph.types
    kinds = ["normal"] * s["days"] + list(scenario_days or [])
    days = len(kinds)
    dates = pd.date_range(s["start_date"], periods=days, freq="D")

    # Base pickup intensity per zone and slot-of-day: [spd, N]
    prof = np.stack([_hourly_to_slots(PICKUP_PROFILES[t], spd) for t in types], axis=1)
    weight = np.array([TYPE_WEIGHT[t] for t in types]) * rng.lognormal(0, 0.25, n)
    base = prof * weight
    base *= s["daily_trips"] / base.sum()

    # Destination choice: gravity model with time-varying attractiveness: [spd, N, N]
    hour = np.arange(spd) * 24.0 / spd
    pm_share = 1 / (1 + np.exp(-(hour - 13.0)))                      # AM -> PM blend
    am = np.array([DEST_WEIGHT_AM[t] for t in types])
    pm = np.array([DEST_WEIGHT_PM[t] for t in types])
    attract = (1 - pm_share)[:, None] * am + pm_share[:, None] * pm   # [spd, N]
    deterrence = np.exp(-graph.dist_km / 8.0)
    np.fill_diagonal(deterrence, 0.15)                                 # few intra-zone taxi trips
    leisure_zones = [i for i, t in enumerate(types) if t == "leisure"]

    od = np.zeros((days * spd, n, n), dtype=np.int16)
    event_log = []
    for d, (date, kind) in enumerate(zip(dates, kinds)):
        dow = date.dayofweek
        mult = np.ones(n)
        if dow >= 5:
            mult = np.array([WEEKEND[t] for t in types])
        elif dow == 4:
            mult = np.array([FRIDAY[t] for t in types])
        lam = base * mult * rng.lognormal(0, s["day_noise_sigma"])    # [spd, N]
        att = attract.copy()
        if kind == "normal":
            if leisure_zones and rng.random() < s["event_prob"]:
                z = int(rng.choice(leisure_zones))
                arrive = (hour >= 17) & (hour < 20)
                leave = (hour >= 21.5) & (hour < 23.75)
                att[arrive, z] *= 4.0
                lam[leave, z] *= 3.5
                event_log.append((str(date.date()), graph.names[z]))
        else:
            _apply_scenario(kind, lam, att, hour, graph, np.array(types), rng, event_log)
        p = att[:, None, :] * deterrence[None]                          # [spd, N, N]
        p /= p.sum(-1, keepdims=True)
        od[d * spd:(d + 1) * spd] = rng.poisson(lam[:, :, None] * p).astype(np.int16)

    slot_index = pd.date_range(s["start_date"], periods=days * spd, freq=f"{cfg['data']['slot_minutes']}min")
    return dict(graph=graph, od=od, slot_index=slot_index, events=event_log, kinds=kinds)
