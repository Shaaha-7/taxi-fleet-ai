"""Record a SUMO episode on the Dubai network for the satellite-map viewer (outputs/sumo_viewer/).

Every `dt` seconds: each taxi's position, heading and state, and each waiting rider. Events: what each
taxi is heading for (the rider it was matched to, the rider's drop-off, or a rebalancing target).
Every 15-min decision: the forecast (mean and conformal upper bound), the supply the controller saw,
and the individual taxis it moved. Saved as one JS file per method (`window.REPLAY[method] = {...}`)
so the viewer opens straight from disk.

Positions are stored as integers: 1e-5 degree steps from (lat0, lon0), about 1 m.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PARKED, PICKUP, OCCUPIED, REPOSITION = 0, 1, 2, 3
GO_PICKUP, GO_DROPOFF, GO_REBALANCE, GO_NONE = 1, 2, 3, 0


class Recorder:
    def __init__(self, net, frame, method: str, clock0_min: int, edge_xy: dict, fleet: int, dt: float = 3.0):
        self.net, self.frame, self.method, self.clock0, self.dt = net, frame, method, clock0_min, dt
        self.edge_xy = edge_xy
        self.lat0, self.lon0 = frame.lat0, frame.lon0
        self.next_t = 0.0
        self.times, self.pos, self.state, self.riders, self.stats = [], [], [], [], []
        self.events, self.decisions, self.moved = [], [], []
        self.taxi_ids = [f"taxi{i}" for i in range(fleet)]   # all of them: some enter a few seconds late
        self.index = {v: i for i, v in enumerate(self.taxi_ids)}
        self.target: dict[int, tuple] = {}
        self.matches_since = 0

    # ------------------------------------------------------------------ helpers
    def _q(self, x, y):
        lon, lat = self.net.convertXY2LonLat(x, y)
        return [round((lat - self.lat0) * 1e5), round((lon - self.lon0) * 1e5)]

    def _edge_q(self, edge):
        xy = self.edge_xy.get(edge)
        return self._q(*xy) if xy else None

    def _set_target(self, t, i, kind, q):
        key = (kind, tuple(q) if q else None)
        if self.target.get(i) != key:
            self.target[i] = key
            self.events.append([round(t), i, kind] + (q or []))

    # ------------------------------------------------------------------ hooks from the bridge
    def matched(self, n: int):
        self.matches_since += n

    def reposition(self, t, taxi: str, src: int, dst: int, edge: str):
        i = self.index.get(taxi)
        if i is None:
            return
        self.moved.append([i, int(src), int(dst)])
        self._set_target(t, i, GO_REBALANCE, self._edge_q(edge))

    def decision(self, t: float, mean, hi, idle, arriving, queue, sent: np.ndarray):
        moves = [(int(i), int(j), int(sent[i, j])) for i, j in zip(*np.nonzero(sent))]
        self.decisions.append(dict(t=t, mean=np.round(mean, 2).tolist(), hi=np.round(hi, 2).tolist(),
                                   idle=idle.astype(int).tolist(), arriving=arriving.astype(int).tolist(),
                                   queue=queue.astype(int).tolist(), moves=moves, taxis=self.moved))
        self.moved = []

    def maybe_frame(self, t: float, traci, served: int, abandoned: int, requested: int):
        if t + 1e-6 < self.next_t:
            return
        self.next_t = t + self.dt
        tv = traci.vehicle
        state = {}
        for v in tv.getTaxiFleet(0):
            state[v] = PARKED if tv.isStopped(v) else REPOSITION
        for v in tv.getTaxiFleet(1):
            state[v] = PICKUP
        for v in tv.getTaxiFleet(2):
            state[v] = OCCUPIED
        res = {r.persons[0]: r for r in traci.person.getTaxiReservations(0)}
        pos, st = [], []
        for i, v in enumerate(self.taxi_ids):
            try:
                x, y = tv.getPosition(v)
                q = self._q(x, y)
                pos.append(q + [round(tv.getAngle(v)) % 360])
            except Exception:                       # not inserted yet
                pos.append(None)
                st.append(PARKED)
                continue
            s = state.get(v, PARKED)
            st.append(s)
            if s in (PICKUP, OCCUPIED):
                people = [p for p in tv.getParameter(v, "device.taxi.currentCustomers").split() if p]
                r = res.get(people[0]) if people else None
                if r is not None and s == PICKUP and not r.state & 8:
                    try:
                        self._set_target(t, i, GO_PICKUP, self._q(*traci.person.getPosition(r.persons[0])))
                    except Exception:
                        pass
                elif r is not None:
                    self._set_target(t, i, GO_DROPOFF, self._edge_q(r.toEdge))
            elif s == PARKED:
                self._set_target(t, i, GO_NONE, None)
        waiting = []
        for r in res.values():
            if r.state & 8:                         # already picked up
                continue
            try:
                waiting.append(self._q(*traci.person.getPosition(r.persons[0])) + [1 if r.state & 4 else 0])
            except Exception:
                pass
        self.times.append(round(t))
        self.pos.append(pos)
        self.state.append(st)
        self.riders.append(waiting)
        self.stats.append((served, len(waiting), abandoned, requested, self.matches_since))
        self.matches_since = 0

    # ------------------------------------------------------------------ output
    def zone_names(self, places_json: Path | None):
        f, n = self.frame, self.frame.rows * self.frame.cols
        places = []
        if places_json and places_json.exists():
            rank = {"suburb": 0, "quarter": 1, "neighbourhood": 2}
            for e in json.loads(places_json.read_text(encoding="utf-8"))["elements"]:
                tg = e.get("tags", {})
                name = tg.get("name:en") or tg.get("name")
                if name and name.isascii() and "cluster" not in name.lower():
                    places.append((name, e["lon"], e["lat"], rank.get(tg.get("place"), 3)))
        names = []
        for z in range(n):
            r, c = divmod(z, f.cols)
            cx, cy = (c + 0.5) * f.cell, (r + 0.5) * f.cell - f.rows * f.cell / 2
            best = None
            for name, lon, lat, rk in places:
                a, b = f.to_frame(lon, lat)
                d = math.hypot(a - cx, b - cy)
                inside = abs(a - cx) <= f.cell / 2 and abs(b - cy) <= f.cell / 2
                score = (0 if inside else 1, rk if inside else 0, d)
                if best is None or score < best[0]:
                    best = (score, name, d)
            if best is None:
                names.append(f"Zone {z}")
            elif best[2] > 3.5:                     # open sea: its riders and moves use the nearest coastal roads
                near = min(places, key=lambda p: math.hypot(*(np.subtract(f.to_frame(p[1], p[2]), (cx, cy)))))
                names.append(f"Coast near {near[0]}")
            else:
                names.append(best[1] if best[0][0] == 0 else f"near {best[1]}")
        return names

    def save(self, out_dir: Path, summary: dict, places_json: Path | None = None) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        f = self.frame
        q = lambda lon, lat: [round((lat - self.lat0) * 1e5), round((lon - self.lon0) * 1e5)]
        zones = [[q(lon, lat) for lon, lat in f.zone_corners(z)] for z in range(f.rows * f.cols)]
        centres = []
        for z in range(f.rows * f.cols):
            r, c = divmod(z, f.cols)
            centres.append(q(*f.to_lonlat((c + 0.5) * f.cell, (r + 0.5) * f.cell - f.rows * f.cell / 2)))
        data = dict(method=self.method, clock0=self.clock0, dt=self.dt, lat0=self.lat0, lon0=self.lon0,
                    times=self.times, taxis=self.taxi_ids, pos=self.pos, state=self.state, riders=self.riders,
                    stats=self.stats, events=self.events, decisions=self.decisions, zones=zones, centres=centres,
                    names=self.zone_names(places_json),
                    summary={k: v for k, v in summary.items() if isinstance(v, (int, float, str))})
        path = out_dir / f"replay_{self.method}.js"
        path.write_text(f"window.REPLAY = window.REPLAY || {{}};\nwindow.REPLAY[{json.dumps(self.method)}] = "
                        + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
        return path
