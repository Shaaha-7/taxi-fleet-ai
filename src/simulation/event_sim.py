"""Vehicle-level, minute-step fleet simulator used to validate controllers trained on the fluid model.

Every taxi and rider is an individual with a point location:
  * riders appear at a random minute of their 15-min slot, at a random point in the origin zone,
    and abandon after `patience_min` minutes without an assigned taxi;
  * every `match_interval_min` minutes pending riders are matched to free taxis
    (greedy / Hungarian / value-augmented, see control/matching.py) within `max_pickup_km`;
  * at every slot boundary a zone-level controller (the same objects as the fluid simulator uses)
    returns zone-to-zone flows, which are executed by sending the idle taxis closest to the
    destination zone; repositioning taxis can be matched en route.
Accounting matches FleetSimTensor so `episode_metrics` applies unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from ..control.matching import match, road_km
from ..ingestion.dataset import CityData
from ..ingestion.graph import speed_profile_kmh

IDLE, BUSY, REPOSITION = 0, 1, 2


class FleetSimEvent:
    def __init__(self, city: CityData, cfg: dict, day_start: int, fleet_size: int | None = None,
                 matcher: str = "hungarian", value=None, seed: int = 0, offline_frac=None, start_slot=None):
        self.city, self.cfg, self.s = city, cfg, dict(cfg["sim"])
        if start_slot is not None:
            self.s["start_slot"] = start_slot
        self.offline_frac = offline_frac
        self.ev = cfg["event_sim"]
        self.day_start, self.matcher, self.value = day_start, matcher, value
        self.rng = np.random.default_rng(seed + day_start)
        self.fleet = int(fleet_size or self.s["fleet_size"])
        g = city.graph
        self.centers, self.n = g.coords_km, g.num_zones
        self.half = np.diagonal(g.dist_km)                      # half-side of each zone (km)
        self.spd = city.slots_per_day
        self.slot_min = cfg["data"]["slot_minutes"]
        self.speed = speed_profile_kmh(np.arange(self.spd), self.spd, self.s["base_speed_kmh"], self.s["peak_speed_kmh"])

    # ------------------------------------------------------------------ geometry
    def _points_in(self, zones: np.ndarray) -> np.ndarray:
        h = self.half[zones][:, None]
        return self.centers[zones] + self.rng.uniform(-h, h, size=(len(zones), 2))

    def _zone_of(self, pts: np.ndarray) -> np.ndarray:
        return np.argmin(((pts[:, None] - self.centers[None]) ** 2).sum(-1), axis=1)

    def _km(self, a, b) -> np.ndarray:
        d = a - b
        return self.s["detour_factor"] * 0.5 * (np.sqrt((d ** 2).sum(-1)) + np.abs(d).sum(-1))

    def _kmh(self, minute: float) -> float:
        return float(self.speed[int(minute // self.slot_min) % self.spd])

    # ------------------------------------------------------------------ state
    def _reset(self, init_share: np.ndarray):
        z = self.rng.choice(self.n, size=self.fleet, p=init_share)
        self.pos = self._points_in(z)
        self.state = np.full(self.fleet, IDLE)
        self.free_at = np.zeros(self.fleet)
        self.dest = self.pos.copy()
        self.reb_from, self.reb_t0 = self.pos.copy(), np.zeros(self.fleet)
        self.req = dict(t=np.zeros(0), o=np.zeros(0, int), d=np.zeros(0, int),
                        opos=np.zeros((0, 2)), dpos=np.zeros((0, 2)))
        self.tot = {k: 0.0 for k in ("requests", "served", "wait_min_sum", "occupied_km", "empty_km_rebalance",
                                      "empty_km_pickup", "empty_km_cruise", "profit", "busy_vehicle_slots",
                                      "vehicle_slots")}
        self.waits, self.zone_req, self.zone_served = [], np.zeros(self.n), np.zeros(self.n)
        self.transitions = []

    def _current_pos(self, idx: np.ndarray, minute: float) -> np.ndarray:
        """Repositioning taxis are interpolated along their route."""
        p = self.pos[idx].copy()
        r = self.state[idx] == REPOSITION
        if r.any():
            i = idx[r]
            frac = np.clip((minute - self.reb_t0[i]) / np.maximum(self.free_at[i] - self.reb_t0[i], 1e-6), 0, 1)
            p[r] = self.reb_from[i] + frac[:, None] * (self.dest[i] - self.reb_from[i])
        return p

    def _spawn_requests(self, slot: int):
        od = self.city.od[self.day_start + slot]
        o, d = np.nonzero(od)
        cnt = od[o, d].astype(int)
        o, d = np.repeat(o, cnt), np.repeat(d, cnt)
        t = slot * self.slot_min + self.rng.uniform(0, self.slot_min, len(o))
        new = dict(t=t, o=o, d=d, opos=self._points_in(o), dpos=self._points_in(d))
        self.req = {k: np.concatenate([self.req[k], new[k]]) for k in self.req}
        self.tot["requests"] += len(o)
        np.add.at(self.zone_req, o, 1)

    # ------------------------------------------------------------------ controller interface
    def observe(self, slot: int) -> dict:
        """Zone-level observation identical in layout to FleetSimTensor.observe (S = 1)."""
        minute = slot * self.slot_min
        k = self.s["max_travel_slots"]
        idle = np.bincount(self._zone_of(self.pos[self.state == IDLE]), minlength=self.n).astype(float)
        moving = self.state != IDLE
        incoming = np.zeros((k, self.n))
        if moving.any():
            off = np.clip(np.ceil((self.free_at[moving] - minute) / self.slot_min), 1, k).astype(int) - 1
            np.add.at(incoming, (off, self._zone_of(self.dest[moving])), 1)
        pending = self.req["t"] < minute
        queue = np.bincount(self.req["o"][pending], minlength=self.n).astype(float)
        nxt = self.city.od[min(self.day_start + slot, len(self.city.od) - 1)].sum(-1)
        as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)[None]
        return dict(t=slot - 1, tod=(slot - 1) % self.spd, idle=as_t(idle), incoming=as_t(incoming),
                    queue=as_t(queue), next_true_pickups=as_t(nxt), fleet=float(self.fleet))

    def _apply_flows(self, flows: np.ndarray, minute: float):
        idle = np.where(self.state == IDLE)[0]
        if len(idle) == 0:
            return
        zone = self._zone_of(self.pos[idle])
        moves = np.floor(flows + self.rng.random(flows.shape)).astype(int)   # stochastic rounding
        np.fill_diagonal(moves, 0)
        for i, j in zip(*np.nonzero(moves)):
            cand = idle[(zone == i) & (self.state[idle] == IDLE)]
            if len(cand) == 0:
                continue
            near = cand[np.argsort(self._km(self.pos[cand], self.centers[j][None]))][:moves[i, j]]
            target = self._points_in(np.full(len(near), j))
            km = self._km(self.pos[near], target)
            self.reb_from[near], self.reb_t0[near] = self.pos[near], minute
            self.dest[near], self.state[near] = target, REPOSITION
            self.free_at[near] = minute + km / self._kmh(minute) * 60
            self.tot["empty_km_rebalance"] += km.sum()

    # ------------------------------------------------------------------ main loop
    def run(self, controller=None, init_share=None, log_transitions: bool = False) -> dict:
        s, ev = self.s, self.ev
        start, end = s["start_slot"], s["start_slot"] + s["episode_slots"]
        self._reset(init_share)
        cost = s["empty_cost_per_km"]
        for minute in range(start * self.slot_min, end * self.slot_min):
            slot = minute // self.slot_min
            if minute % self.slot_min == 0:
                if controller is not None and slot > start:
                    flows = controller.act(self.observe(slot))[0].numpy()
                    self._apply_flows(flows, minute)
                self._spawn_requests(slot)
                if self.offline_frac is not None and self.offline_frac[slot % self.spd] > 0:
                    idle = np.where(self.state == IDLE)[0]
                    off = idle[self.rng.random(len(idle)) < self.offline_frac[slot % self.spd]]
                    self.state[off], self.free_at[off] = BUSY, minute + 60    # 1-hour break
                n_idle = (self.state == IDLE).sum()
                self.tot["empty_km_cruise"] += n_idle * s["idle_cruise_km"]
                self.tot["profit"] -= cost * n_idle * s["idle_cruise_km"]
                self.tot["vehicle_slots"] += self.fleet
                if log_transitions:
                    z = self._zone_of(self.pos[self.state == IDLE])
                    self.transitions.append(np.stack([z, np.full(len(z), slot), z, np.full(len(z), slot + 1),
                                                      np.full(len(z), -cost * s["idle_cruise_km"])], 1))

            done = (self.state != IDLE) & (self.free_at <= minute)
            self.pos[done] = self.dest[done]
            self.state[done] = IDLE

            # Abandonment
            r = self.req
            keep = (minute - r["t"]) <= ev["patience_min"]
            self.req = {k: v[keep] for k, v in r.items()}

            if minute % ev["match_interval_min"] == 0:
                self._match(minute, slot, cost, log_transitions)
        totals = dict(self.tot)
        w = np.asarray(self.waits)
        totals["p90_wait_min"] = float(np.percentile(w, 90)) if len(w) else float("nan")
        m = self.zone_req > 0
        rate = np.minimum(self.zone_served[m] / self.zone_req[m], 1)
        totals["service_gini"] = float(np.abs(rate[:, None] - rate[None]).mean() / (2 * rate.mean())) if m.any() else float("nan")
        if log_transitions and self.transitions:
            totals["transitions"] = np.concatenate(self.transitions)
        return totals

    def _match(self, minute, slot, cost, log_transitions):
        s = self.s
        r = self.req
        visible = r["t"] <= minute
        if not visible.any():
            return
        ridx = np.where(visible)[0]
        vidx = np.where(self.state != BUSY)[0]
        if len(vidx) == 0:
            return
        vpos = self._current_pos(vidx, minute)
        kmh = self._kmh(minute)
        trip_km = self._km(r["opos"][ridx], r["dpos"][ridx])
        fare = s["fare_flag"] + s["fare_per_km"] * trip_km
        pairs = match(self.matcher, vpos, self._zone_of(vpos), r["opos"][ridx], r["d"][ridx], fare,
                      trip_km / kmh * 60, minute - r["t"][ridx], slot, s["detour_factor"], self.ev["max_pickup_km"],
                      cost, value=self.value, gamma=self.ev["value_gamma"], kmh=kmh, slot_min=self.slot_min,
                      pickup_penalty=self.ev["pickup_penalty_aed_per_km"])
        if not pairs:
            return
        vi = np.array([p[0] for p in pairs])
        ri = np.array([p[1] for p in pairs])
        veh, req = vidx[vi], ridx[ri]
        pk = road_km(vpos[vi], r["opos"][req], s["detour_factor"]).diagonal()
        pick_min = pk / kmh * 60
        trip_min = trip_km[ri] / kmh * 60
        was_reb = self.state[veh] == REPOSITION
        if was_reb.any():                                        # refund the un-driven part of the reposition
            left = self._km(vpos[vi][was_reb], self.dest[veh][was_reb])
            self.tot["empty_km_rebalance"] -= left.sum()
        if log_transitions:
            vz = self._zone_of(vpos[vi])
            drop_slot = (minute + pick_min + trip_min) // self.slot_min
            self.transitions.append(np.stack([vz, np.full(len(vz), slot), r["d"][req], np.maximum(drop_slot, slot + 1),
                                              fare[ri] - cost * pk], 1))
        self.pos[veh] = r["dpos"][req]
        self.dest[veh] = r["dpos"][req]
        self.state[veh] = BUSY
        self.free_at[veh] = minute + pick_min + trip_min
        wait = minute - r["t"][req] + pick_min + 1.0
        self.waits.extend(wait.tolist())
        self.tot["served"] += len(req)
        self.tot["wait_min_sum"] += wait.sum()
        self.tot["occupied_km"] += trip_km[ri].sum()
        self.tot["empty_km_pickup"] += pk.sum()
        self.tot["busy_vehicle_slots"] += (trip_min / self.slot_min).sum()
        self.tot["profit"] += (fare[ri] - cost * pk).sum()
        np.add.at(self.zone_served, r["o"][req], 1)
        keep = np.ones(len(r["t"]), dtype=bool)
        keep[req] = False
        self.req = {k: v[keep] for k, v in r.items()}
