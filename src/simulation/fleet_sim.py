"""Zone-level fleet simulator, batched over S scenarios (one replayed demand day each).

Fluid approximation of a taxi fleet in 15-min slots (vehicle counts are real-valued):

    serve slot t  ->  observe  ->  rebalance leftover idle taxis  ->  idle cruising  ->  t+1

Serving: queued requests (from slot t-1) first, then new ones, using idle taxis in the
origin zone; unmet requests may then be served from adjacent zones (one-hop matching).
Requests unserved after one slot of waiting abandon. Occupied and rebalancing taxis rejoin
the idle pool at their destination after a time-of-day dependent travel time.
Conservation holds every slot: idle + en-route (occupied or rebalancing) = fleet.

Pickup distance in a zone with m idle taxis follows the nearest-vehicle scaling
d ~ d_intra / sqrt(m) (Larson & Odoni), so concentrating supply where demand is shortens waits.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..ingestion.dataset import CityData
from ..ingestion.graph import speed_profile_kmh


class FleetSimTensor:
    def __init__(self, city: CityData, cfg: dict, day_starts: list[int], fleet_size: int | None = None,
                 offline_frac: np.ndarray | None = None, start_slot: int | None = None):
        """offline_frac [slots_per_day]: share of idle taxis taking an (unpaid) 1-hour break in that slot,
        e.g. drivers breaking their fast at iftar. start_slot overrides the configured episode start."""
        s = dict(cfg["sim"])
        if start_slot is not None:
            s["start_slot"] = start_slot
        self.offline_frac = offline_frac
        self.city, self.cfg = city, s
        self.S, self.N = len(day_starts), city.graph.num_zones
        self.K = s["max_travel_slots"]
        self.fleet = float(fleet_size or s["fleet_size"])
        self.slot_min = cfg["data"]["slot_minutes"]
        self.day_starts = torch.as_tensor(day_starts)
        self.dist = torch.as_tensor(city.graph.dist_km, dtype=torch.float32)
        self.adj = torch.as_tensor(city.graph.adjacency, dtype=torch.float32)
        self.intra_km = torch.diagonal(self.dist).clone()
        self.od = city.od
        spd = city.slots_per_day
        self.speed = torch.as_tensor(speed_profile_kmh(np.arange(spd), spd, s["base_speed_kmh"], s["peak_speed_kmh"]),
                                     dtype=torch.float32)
        # Initial idle distribution: training-period pickups around the start hour.
        train = list(city.split_slots("train"))
        tod = city.tod[train]
        near = np.abs(tod - s["start_slot"]) <= 4
        w = city.pickups[train][near].mean(0) + 1e-3
        self.init_share = torch.as_tensor(w / w.sum(), dtype=torch.float32)

    # ---------------------------------------------------------------- helpers
    def _travel_slots(self, slot: int) -> torch.Tensor:
        minutes = self.dist / self.speed[slot % self.city.slots_per_day] * 60.0
        return torch.clamp(torch.ceil(minutes / self.slot_min), 1, self.K).long()

    def _schedule(self, flows: torch.Tensor, slots: torch.Tensor) -> None:
        """Add vehicles moving i->j ([S,N,N]) to the arrival buffer at their arrival slot."""
        for k in range(1, self.K + 1):
            mask = (slots == k).float()
            self.incoming[:, k - 1] += (flows * mask).sum(1)

    def _demand(self, t: int) -> torch.Tensor:
        idx = np.minimum((self.day_starts + t).numpy(), len(self.od) - 1)
        return torch.as_tensor(self.od[idx], dtype=torch.float32)

    def _add_waits(self, minutes: torch.Tensor, weight: torch.Tensor) -> None:
        self.wait_samples.append((minutes.reshape(self.S, -1), weight.reshape(self.S, -1)))

    # ---------------------------------------------------------------- API
    def reset(self) -> dict:
        s = self.cfg
        self.t = s["start_slot"]
        self.end = s["start_slot"] + s["episode_slots"]
        self.idle = (self.init_share * self.fleet).expand(self.S, -1).clone()
        self.incoming = torch.zeros(self.S, self.K, self.N)
        self.queue = torch.zeros(self.S, self.N, self.N)
        zero = lambda: torch.zeros(self.S)
        self.tot = {k: zero() for k in ("requests", "served", "wait_min_sum", "occupied_km", "empty_km_rebalance",
                                         "empty_km_pickup", "empty_km_cruise", "profit", "busy_vehicle_slots",
                                         "vehicle_slots")}
        self.zone_req, self.zone_served = torch.zeros(self.S, self.N), torch.zeros(self.S, self.N)
        self.wait_samples = []
        self.last_reward = self._serve()
        return self.observe()

    def _serve(self) -> torch.Tensor:
        s, t = self.cfg, self.t
        speed = self.speed[t % self.city.slots_per_day]
        new = self._demand(t)                                            # [S,N,N]
        q_out, n_out = self.queue.sum(-1), new.sum(-1)
        req = q_out + n_out
        m = self.idle.clone()

        # Local matching
        local = torch.minimum(m, req)
        pick_km = torch.clamp(self.intra_km / torch.sqrt(torch.clamp(m, min=1.0)), min=0.2)
        local_wait = pick_km / speed * 60.0 + 1.0
        leftover, unmet = m - local, req - local
        pickup_km = local * pick_km

        # One-hop matching from adjacent zones (proportional allocation)
        from_nb = torch.zeros(self.S, self.N, self.N)                    # [S, j(source), i(dest)]
        if s["neighbor_matching"]:
            want = self.adj[None] * unmet[:, None, :]                    # [S,j,i]
            share = want / want.sum(-1, keepdim=True).clamp_min(1e-9)
            offer = leftover[:, :, None] * share
            offered = offer.sum(1)
            ratio = torch.where(offered > 0, torch.minimum(unmet, offered) / offered.clamp_min(1e-9),
                                torch.zeros_like(offered))
            from_nb = offer * ratio[:, None, :]
            leftover = leftover - from_nb.sum(-1)
            pickup_km = pickup_km + (from_nb * self.dist[None]).sum(1)
            self._add_waits((self.dist[None] / speed * 60.0 + 1.0).expand_as(from_nb), from_nb)
        nb_in = from_nb.sum(1)
        served = local + nb_in                                           # by origin zone

        # Serve the queue first, then new requests, proportionally across destinations
        q_frac = torch.where(q_out > 0, torch.minimum(served, q_out) / q_out.clamp_min(1e-9), torch.zeros_like(q_out))
        n_left = served - q_frac * q_out
        n_frac = torch.where(n_out > 0, n_left / n_out.clamp_min(1e-9), torch.zeros_like(n_out)).clamp(0, 1)
        trips = self.queue * q_frac[..., None] + new * n_frac[..., None]   # [S,N,N]
        q_served = q_frac * q_out

        # Queued riders (served locally first) already waited about one slot.
        q_local = torch.minimum(q_served, local)
        self._add_waits(local_wait.expand(self.S, -1), local - q_local)
        wait_sum = (local * local_wait).sum(-1) + (from_nb * (self.dist[None] / speed * 60.0 + 1.0)).sum((1, 2)) \
            + q_served.sum(-1) * self.slot_min
        self._add_waits(local_wait + float(self.slot_min), q_local)

        slots = self._travel_slots(t)
        self.idle = leftover
        self._schedule(trips, slots[None].expand(self.S, -1, -1))
        self.queue = new * (1 - n_frac[..., None])                      # unmet new requests wait one slot

        trip_km = (trips * self.dist[None]).sum((1, 2))
        revenue = trips.sum((1, 2)) * s["fare_flag"] + trip_km * s["fare_per_km"]
        self.tot["requests"] += n_out.sum(-1)
        self.tot["served"] += served.sum(-1)
        self.tot["wait_min_sum"] += wait_sum
        self.tot["occupied_km"] += trip_km
        self.tot["empty_km_pickup"] += pickup_km.sum(-1)
        self.tot["busy_vehicle_slots"] += (trips * slots[None]).sum((1, 2))
        self.zone_req += n_out
        self.zone_served += served
        return revenue - s["empty_cost_per_km"] * pickup_km.sum(-1)

    def observe(self) -> dict:
        t = self.t
        nxt = min(t + 1, self.end)
        return dict(t=t, tod=t % self.city.slots_per_day, idle=self.idle.clone(),
                    incoming=self.incoming.clone(), queue=self.queue.sum(-1),
                    next_true_pickups=self._demand(nxt).sum(-1),   # used only by the oracle baseline
                    fleet=self.fleet)

    def step(self, flows: torch.Tensor) -> tuple[dict, torch.Tensor, bool]:
        """flows [S,N,N]: idle taxis to move from zone i to j (diagonal ignored)."""
        s = self.cfg
        flows = flows.clone().float().clamp_min(0)
        flows[:, torch.arange(self.N), torch.arange(self.N)] = 0
        out = flows.sum(-1)
        scale = torch.where(out > self.idle, self.idle / out.clamp_min(1e-9), torch.ones_like(out))
        flows = flows * scale[..., None]                                 # never move more than available

        reb_km = (flows * self.dist[None]).sum((1, 2))
        self.idle = self.idle - flows.sum(-1)
        self._schedule(flows, self._travel_slots(self.t)[None].expand(self.S, -1, -1))
        if self.offline_frac is not None:
            off = self.idle * float(self.offline_frac[self.t % self.city.slots_per_day])
            self.idle = self.idle - off
            self.incoming[:, min(4, self.K) - 1] += off              # back on shift in ~1 hour
        cruise_km = self.idle.sum(-1) * s["idle_cruise_km"]
        self.tot["empty_km_rebalance"] += reb_km
        self.tot["empty_km_cruise"] += cruise_km
        self.tot["vehicle_slots"] += self.fleet
        cost = s["empty_cost_per_km"] * (reb_km + cruise_km)
        self.tot["profit"] += self.last_reward - cost
        step_reward = self.last_reward - cost

        # Advance time: arrivals join the idle pool, then serve the next slot.
        self.t += 1
        self.idle = self.idle + self.incoming[:, 0]
        self.incoming = torch.cat([self.incoming[:, 1:], torch.zeros(self.S, 1, self.N)], dim=1)
        done = self.t >= self.end
        if not done:
            self.last_reward = self._serve()
        return self.observe(), step_reward, done

    def episode_totals(self) -> list[dict]:
        """Per-scenario totals, incl. p90 wait and zone-level service Gini."""
        minutes = torch.cat([m for m, _ in self.wait_samples], dim=1)
        weights = torch.cat([w for _, w in self.wait_samples], dim=1)
        results = []
        for i in range(self.S):
            order = torch.argsort(minutes[i])
            cw = torch.cumsum(weights[i][order], 0)
            p90 = minutes[i][order][torch.searchsorted(cw, 0.9 * cw[-1]).clamp(max=len(cw) - 1)].item() \
                if cw[-1] > 0 else math.nan
            mask = self.zone_req[i] > 0
            rate = (self.zone_served[i][mask] / self.zone_req[i][mask]).clamp(max=1).numpy()
            gini = float(np.abs(rate[:, None] - rate[None]).mean() / (2 * rate.mean())) if len(rate) else math.nan
            d = {k: float(v[i]) for k, v in self.tot.items()}
            d.update(p90_wait_min=p90, service_gini=gini)
            results.append(d)
        return results
