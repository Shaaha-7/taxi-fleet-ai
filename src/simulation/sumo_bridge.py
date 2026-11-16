"""SUMO bridge: replay a scaled demand window on a microscopic road network with TraCI-dispatched taxis.

* Network: `netgenerate --grid` over the corridor city (1 km blocks, 50 km/h), edges mapped to zones
  by midpoint.
* Taxis: vType with `has.taxi.device`, `--device.taxi.dispatch-algorithm traci`, idle algorithm `stop`.
* Riders: `traci.person.add` + `appendDrivingStage(..., "taxi")` at their request time; riders
  without an assigned taxi after `patience_min` are removed.
* Every `dispatch_period_s`: unassigned reservations (state 1|2) are matched to empty taxis with
  Hungarian assignment on pickup distance, then `traci.vehicle.dispatchTaxi`.
* Every 15 min: the zone-level controller (same objects as the other simulators, forecasts scaled
  to the window's demand share) returns flows; parked idle taxis are sent with changeTarget + resume
  and the taxi device re-parks them at the destination edge. The observation's `incoming` holds
  every non-parked taxi (picking up, occupied, or repositioning) at the zone of its route's last
  edge, in the slot its remaining free-flow route time ends in.
* Demand scale (`sumo.scale`) and fleet scale (`sumo.fleet_scale`) are separate; `sumo.fleet_sweep`
  runs every method at several absolute fleet sizes on the same rider stream.
Results validate the controller ranking under microscopic traffic, not absolute KPIs.
"""
from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..control.baselines import ControlContext
from ..evaluation.run_eval import load_forecasts, make_controller
from ..ingestion.dataset import load_data
from ..simulation.fleet_sim import FleetSimTensor
from ..utils import save_json


def _sumo_home() -> Path:
    try:
        import sumo
        return Path(sumo.SUMO_HOME)
    except ImportError as e:
        if "SUMO_HOME" in os.environ:
            return Path(os.environ["SUMO_HOME"])
        raise RuntimeError("SUMO not installed: pip install eclipse-sumo traci sumolib") from e


def _traci(gui: bool = False):
    if gui:                              # libsumo has no GUI; the socket client drives sumo-gui
        import traci
        return traci, False
    try:
        import libsumo as traci          # in-process, much faster than the socket API
        return traci, True
    except ImportError:
        import traci
        return traci, False


def build_network(cfg: dict, work: Path) -> tuple[Path, list[Path] | None]:
    """(net.xml, optional background polygon files for the GUI)."""
    s = cfg["data"]["synthetic"]
    sc = cfg["sumo"]
    if sc.get("network", "grid") == "osm":
        from .osm_map import build
        return build(cfg, work, _sumo_home())
    block = sc["block_m"]
    nx = int(s["grid_cols"] * s["cell_km"] * 1000 / block) + 1
    ny = int(s["grid_rows"] * s["cell_km"] * 1000 / block) + 1
    net = work / "corridor.net.xml"
    if not net.exists():
        exe = _sumo_home() / "bin" / ("netgenerate.exe" if sys.platform == "win32" else "netgenerate")
        subprocess.run([str(exe), "--grid", "--grid.x-number", str(nx), "--grid.y-number", str(ny),
                        "--grid.x-length", str(block), "--grid.y-length", str(block), "-L", "2",
                        "-S", str(sc["speed_ms"]), "--no-turnarounds", "true", "-o", str(net)], check=True)
    return net, None


def zone_geometry(cfg: dict, net):
    """(zone_of_xy, to_xy, view_angle). to_xy maps corridor (along_m, across_m from centre line) to net XY."""
    s = cfg["data"]["synthetic"]
    cell_m, rows, cols = s["cell_km"] * 1000, s["grid_rows"], s["grid_cols"]
    if cfg["sumo"].get("network", "grid") == "osm":
        from .osm_map import CorridorFrame
        frame = CorridorFrame(cfg)
        zone_of_xy = lambda x, y: frame.zone_of(*net.convertXY2LonLat(x, y))
        to_xy = lambda a, c: net.convertLonLat2XY(*frame.to_lonlat(a / 1000, c / 1000))
        # Rotate the view so the corridor runs left to right (bearing 90 deg = east).
        return zone_of_xy, to_xy, frame.bearing - 90

    def zone_of_xy(x, y):
        col, row = min(int(x // cell_m), cols - 1), min(int(y // cell_m), rows - 1)
        return row * cols + col if x >= 0 and y >= 0 else -1
    return zone_of_xy, (lambda a, c: (a, c + rows * cell_m / 2)), 0.0


def write_taxis(cfg: dict, work: Path, edges_by_zone: dict, share: np.ndarray, fleet: int, rng,
                edge_len: dict) -> Path:
    path = work / "taxis.rou.xml"
    lines = ['<routes>', '  <vType id="taxi" vClass="taxi" personCapacity="4">',
             '    <param key="has.taxi.device" value="true"/>', '  </vType>']
    zones = rng.choice(len(share), size=fleet, p=share)
    for i, z in enumerate(zones):
        e = rng.choice(edges_by_zone[int(z)])
        # Start mid-edge: the 'stop' idle algorithm parks an unassigned taxi where it is, and an idle stop
        # within a car length of the edge start cannot be re-placed when the taxi is later dispatched.
        lines.append(f'  <trip id="taxi{i}" type="taxi" depart="0" departPos="{edge_len[e] / 2:.1f}" from="{e}" '
                     f'to="{e}" line="taxi"/>')
    lines.append('</routes>')
    path.write_text("\n".join(lines))
    return path


VIEW_ID = "View #0"


def zone_travel_minutes(net, edges_by_zone: dict, to_xy, cfg: dict, cache: Path) -> np.ndarray:
    """Fastest free-flow road time (min) between zone representative edges, cached per network.

    One Dijkstra per zone over the edge graph; each edge costs length / speed plus a 5 s junction allowance.
    Used as LP move costs instead of straight-line centroid distance (`sumo.lp_cost: traveltime`)."""
    import heapq
    s = cfg["data"]["synthetic"]
    n, cols, cell = s["grid_rows"] * s["grid_cols"], s["grid_cols"], s["cell_km"] * 1000
    if cache.exists():
        tt = np.load(cache)
        if tt.shape == (n, n):
            return tt
    rep = []
    for z in range(n):
        r, c = divmod(z, cols)
        cx, cy = to_xy((c + 0.5) * cell, (r + 0.5) * cell - s["grid_rows"] * cell / 2)
        rep.append(min(edges_by_zone[z], key=lambda e: (lambda sh: (sh[0][0] - cx) ** 2 + (sh[0][1] - cy) ** 2)(
            net.getEdge(e).getShape())))
    cost = {e.getID(): e.getLength() / max(e.getSpeed(), 1.0) + 5.0 for e in net.getEdges()}
    out = {e.getID(): [o.getID() for o in e.getOutgoing() if o.allows("taxi")] for e in net.getEdges()}
    tt = np.zeros((n, n))
    for i, src in enumerate(rep):
        dist, heap = {src: 0.0}, [(0.0, src)]
        while heap:
            d, e = heapq.heappop(heap)
            if d > dist.get(e, 1e18):
                continue
            for o in out[e]:
                nd = d + cost[o]
                if nd < dist.get(o, 1e18):
                    dist[o] = nd
                    heapq.heappush(heap, (nd, o))
        tt[i] = [dist.get(dst, np.nan) / 60 for dst in rep]
    tt = np.where(np.isnan(tt), np.nanmax(tt), tt)
    np.fill_diagonal(tt, 0.0)
    np.save(cache, tt)
    return tt


def run_sumo_episode(cfg: dict, method: str, log=print, fleet: int | None = None,
                     start_slot: int | None = None, slots: int | None = None, gui: bool = False,
                     snapshots: dict | None = None, record_dir: Path | None = None) -> dict:
    """One episode. gui=True opens sumo-gui with the decision overlay; snapshots maps sim seconds to a
    PNG path or a (view id, PNG path) pair. record_dir saves a replay for the satellite viewer (OSM only)."""
    import sumolib
    sc, s = cfg["sumo"], cfg["data"]["synthetic"]
    work = cfg["paths"]["output_dir"] / "sumo"
    work.mkdir(parents=True, exist_ok=True)
    # Separate streams so riders are identical across methods and fleet sizes (paired comparison).
    rng_taxi, rng_riders, rng = (np.random.default_rng([cfg["seed"], k]) for k in range(3))
    city = load_data(cfg)
    spd = city.slots_per_day
    day = city.split_day_starts("test")[sc["test_day_index"]]
    t0_slot = sc["start_slot"] if start_slot is None else start_slot
    n_slots = sc["slots"] if slots is None else slots
    scale = sc["scale"]
    if fleet is None:
        fleet = int(cfg["sim"]["fleet_size"] * sc.get("fleet_scale", scale))

    net_path, poly_path = build_network(cfg, work)
    net = sumolib.net.readNet(str(net_path))
    zone_of_xy, to_xy, view_angle = zone_geometry(cfg, net)
    n_zones = s["grid_rows"] * s["grid_cols"]
    edges = net.getEdges()
    if sc.get("network", "grid") == "osm":                 # every taxi route must exist on a real network
        from .osm_map import NO_KERBSIDE, strongly_connected
        keep = strongly_connected(net, [e for e in edges if e.allows("taxi")])
        edges = [e for e in edges if e.getID() in keep]
    else:
        NO_KERBSIDE = ()
    edges_by_zone: dict[int, list[str]] = {}               # kerbside edges where riders and taxis are placed
    zone_of_edge: dict[str, int] = {}                      # every usable edge, for the controller's observation
    edge_xy = {}
    edge_len = {e.getID(): e.getLength() for e in net.getEdges()}
    edge_tt = {e.getID(): e.getLength() / e.getSpeed() for e in net.getEdges()}   # free-flow seconds
    taxi_lane = {e.getID(): next((ln.getIndex() for ln in e.getLanes() if ln.allows("taxi")), 0) for e in net.getEdges()}
    for e in edges:
        (x1, y1), (x2, y2) = e.getShape()[0], e.getShape()[-1]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        z = zone_of_xy(x, y)
        edge_xy[e.getID()] = (x, y)
        if z < 0:
            continue
        zone_of_edge[e.getID()] = z
        if e.getLength() >= 50 and e.getType() not in NO_KERBSIDE:
            edges_by_zone.setdefault(z, []).append(e.getID())
    # Zones with no road (open sea on the Gulf side) borrow the nearest zone that has one.
    has = sorted(edges_by_zone)
    for z in range(n_zones):
        if z not in edges_by_zone:
            r, c = divmod(z, s["grid_cols"])
            near = min(has, key=lambda q: (q // s["grid_cols"] - r) ** 2 + (q % s["grid_cols"] - c) ** 2)
            edges_by_zone[z] = edges_by_zone[near]

    share = FleetSimTensor(city, cfg, [day]).init_share.numpy().astype(float)
    share /= share.sum()
    rou = write_taxis(cfg, work, edges_by_zone, share, fleet, rng_taxi, edge_len)

    # Riders: subsample the test day's OD counts in the window.
    riders = []
    for k in range(n_slots):
        od = city.od[day + t0_slot + k]
        o, d = np.nonzero(od)
        cnt = rng_riders.binomial(od[o, d].astype(int), scale)
        for oo, dd, c in zip(o, d, cnt):
            for _ in range(c):
                riders.append((k * 900 + rng_riders.uniform(0, 900), int(oo), int(dd),
                               rng_riders.choice(edges_by_zone[int(oo)]), rng_riders.choice(edges_by_zone[int(dd)])))
    riders.sort()

    # Controller with forecasts scaled to the window's demand share.
    ccfg = copy.deepcopy(cfg)
    ccfg["sim"]["fleet_size"] = fleet
    fc = {k: v * scale for k, v in load_forecasts(cfg).items() if k in ("mean", "lo", "hi")}
    ctx = ControlContext(city, ccfg, fc, [day], fleet)
    if sc.get("lp_cost", "distance") == "traveltime":
        from ..control.flow_solver import FlowSolver
        tt = zone_travel_minutes(net, edges_by_zone, to_xy, cfg, work / f"zone_tt_{net_path.stem}.npy")
        # Minutes -> km-equivalent at a 40 km/h reference, so lam (12 km) and the 15 km cap keep their meaning.
        ctx.solver = FlowSolver(tt * 40 / 60)
    controller = make_controller(method, ctx, ccfg) if method != "none" else None
    value_matcher = sc.get("matcher", "hungarian") == "value"
    if value_matcher:
        from ..control.matching import match, road_km
        from ..models.value_table import ValueTable
        vtable = ValueTable.load(cfg["paths"]["output_dir"] / "value_table.npz")
    decision_s = sc.get("decision_period_s", 900)

    traci, is_lib = _traci(gui)
    err_log = work / f"errors_{method}_{fleet}_{t0_slot}.log"     # SUMO-side errors TraCI does not raise
    exe = "sumo-gui" if gui else "sumo"
    sumo_bin = str(_sumo_home() / "bin" / (f"{exe}.exe" if sys.platform == "win32" else exe))
    args = [sumo_bin, "-n", str(net_path), "-r", str(rou), "--step-length", str(sc["step_s"]),
            "--device.taxi.dispatch-algorithm", "traci", "--device.taxi.idle-algorithm", "stop",
            "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "300",
            "--end", str(n_slots * 900 + 1800), "--error-log", str(err_log)]
    if gui:
        from .sumo_gui import SumoOverlay, write_view_settings
        args += ["--start", "true", "--quit-on-end", "true",
                 "--gui-settings-file",
                 str(write_view_settings(work, sc.get("gui_delay_ms", 20), sc.get("network", "grid") == "osm"))]
        if poly_path:                                  # sea, buildings, water, parks (OSM network only)
            args += ["-a", ",".join(str(p) for p in poly_path if Path(p).exists())]
    if gui and sc.get("gui_3d"):
        # Whole window in 3D (OpenSceneGraph); adding a 3D view mid-run crashes sumo-gui 1.28 on Windows.
        os.environ.setdefault("OSG_FILE_PATH", str(_sumo_home() / "data" / "3D"))
        args += ["--osg-view", "true"]
    traci.start(args)
    overlay = SumoOverlay(traci, cfg, method, to_xy, view_angle) if gui else None
    clock = lambda sec: f"{int(t0_slot * 15 + sec // 60) // 60:02d}:{int(t0_slot * 15 + sec // 60) % 60:02d}"
    snapshots = dict(snapshots or {})
    recorder = None
    if record_dir is not None:
        from .osm_map import CorridorFrame
        from .replay import Recorder
        recorder = Recorder(net, CorridorFrame(cfg), method, t0_slot * 15, edge_xy, fleet)
    n_inc = cfg["sim"]["max_travel_slots"]
    pending, ri = {}, 0
    obs_log = []                                   # (parked idle, en-route incoming) per controller decision
    served, waits, abandoned, rebalance_cmds, errors, unparked = 0, [], 0, 0, 0, 0
    error_types: dict[str, int] = {}
    picked = set()
    # Requests stop at the window end; the drain lets every rider be picked up or abandon.
    t, end = 0.0, n_slots * 900 + (cfg["event_sim"]["patience_min"] + 5) * 60

    held: set[str] = set()                        # taxis driving to, or parked at, a reposition stop

    def send_to(v, edge):
        """Move an empty taxi to `edge` and hold it there. The 'stop' idle algorithm re-parks any empty
        taxi without a stop where it stands, so a bare changeTarget + resume never moves it: give it a
        stop at the target, which it keeps until the dispatcher assigns it (see release())."""
        if traci.vehicle.isStopped(v):
            traci.vehicle.resume(v)
        elif traci.vehicle.getStops(v):
            traci.vehicle.replaceStop(v, 0, "")
        traci.vehicle.changeTarget(v, edge)
        traci.vehicle.setStop(v, edge, pos=edge_len[edge] / 2, laneIndex=taxi_lane[edge], duration=1e6)
        held.add(v)

    def release(v):
        """Drop a reposition stop before the taxi is dispatched to a rider."""
        if v in held:
            held.discard(v)
            if traci.vehicle.isStopped(v):
                traci.vehicle.resume(v)
            elif traci.vehicle.getStops(v):
                traci.vehicle.replaceStop(v, 0, "")

    def fail(kind, exc):
        nonlocal errors
        errors += 1
        key = f"{kind}: {str(exc)[:160]}"
        error_types[key] = error_types.get(key, 0) + 1
    try:
        while t < end:
            # new riders
            while ri < len(riders) and riders[ri][0] <= t:  # riders all depart inside the window
                _, o, d, fe, te = riders[ri]
                pid = f"p{ri}"
                # Mid-edge: a pickup within a car length of the edge start leaves an idle stop SUMO cannot
                # re-place on the next dispatch ("could not assign stop ... after rerouting").
                traci.person.add(pid, fe, edge_len[fe] / 2, depart=t)
                traci.person.appendDrivingStage(pid, te, "taxi")
                pending[pid] = t
                ri += 1
            if recorder is None:
                traci.simulationStep(t + sc["dispatch_period_s"])
                t = traci.simulation.getTime()
            else:                                      # finer steps so the replay animates smoothly
                t_next = t + sc["dispatch_period_s"]
                while t < t_next - 1e-6:
                    traci.simulationStep(min(t + recorder.dt, t_next))
                    t = traci.simulation.getTime()
                    recorder.maybe_frame(t, traci, served, abandoned, ri)

            res = traci.person.getTaxiReservations(0)
            open_res = [r for r in res if r.state in (1, 2)]
            for r in res:                              # picked-up riders: record wait once
                if r.state & 8 and r.persons[0] not in picked:
                    pid = r.persons[0]
                    picked.add(pid)
                    served += 1
                    waits.append((t - pending.get(pid, t)) / 60)
            for r in open_res:                         # abandonment
                pid = r.persons[0]
                if t - pending.get(pid, t) > cfg["event_sim"]["patience_min"] * 60:
                    try:
                        traci.person.remove(pid)
                        abandoned += 1
                    except Exception as e:
                        fail("remove", e)
            open_res = [r for r in open_res if r.persons[0] in traci.person.getIDList()]
            # A taxi idling within a car length of its edge start (after a teleport, say) breaks dispatchTaxi
            # ("could not assign stop ... after rerouting"); move it to another edge of its zone first.
            empty = []
            for v in traci.vehicle.getTaxiFleet(0):
                if traci.vehicle.isStopped(v) and traci.vehicle.getLanePosition(v) < 10:
                    road = traci.vehicle.getRoadID(v)
                    cand = [e for e in edges_by_zone.get(zone_of_edge.get(road, -1), []) if e != road]
                    try:
                        send_to(v, rng.choice(cand or list(edges_by_zone[0])))
                        unparked += 1
                    except Exception as e:
                        fail("unpark", e)
                    continue
                empty.append(v)
            assigned = set()
            if open_res and empty and value_matcher:
                # Value-aware batch matching (Xu et al. 2018): fare - pickup cost + gamma^k V(drop zone) - V(taxi zone).
                ev, sm = cfg["event_sim"], cfg["sim"]
                vpos = np.array([traci.vehicle.getPosition(v) for v in empty]) / 1000
                vz = np.array([max(zone_of_edge.get(traci.vehicle.getRoadID(v), 0), 0) for v in empty])
                rpos = np.array([edge_xy.get(r.fromEdge, (0, 0)) for r in open_res]) / 1000
                dpos = np.array([edge_xy.get(r.toEdge, (0, 0)) for r in open_res]) / 1000
                dz = np.array([max(zone_of_edge.get(r.toEdge, 0), 0) for r in open_res])
                trip_km = np.diag(road_km(rpos, dpos, sm["detour_factor"]))
                kmh = 30.0
                pairs = match("value", vpos, vz, rpos, dz, sm["fare_flag"] + sm["fare_per_km"] * trip_km,
                              trip_km / kmh * 60, np.array([t - pending.get(r.persons[0], t) for r in open_res]),
                              t0_slot + int(t // 900), sm["detour_factor"], ev["max_pickup_km"], sm["empty_cost_per_km"],
                              value=vtable, gamma=ev["value_gamma"], kmh=kmh, slot_min=15,
                              pickup_penalty=ev["pickup_penalty_aed_per_km"])
                for v_i, r_i in pairs:
                    try:
                        release(empty[v_i])
                        traci.vehicle.dispatchTaxi(empty[v_i], [open_res[r_i].id])
                        assigned.add(open_res[r_i].id)
                        if recorder is not None:
                            recorder.matched(1)
                    except Exception as e:
                        fail("dispatch", e)
            elif open_res and empty:
                vpos = np.array([traci.vehicle.getPosition(v) for v in empty])
                rpos = np.array([edge_xy.get(r.fromEdge, (0, 0)) for r in open_res])
                dist = np.abs(vpos[:, None] - rpos[None]).sum(-1) / 1000
                cost = np.where(dist <= cfg["event_sim"]["max_pickup_km"], dist, 1e6)
                rows, cols = linear_sum_assignment(cost)
                for v_i, r_i in zip(rows, cols):
                    if cost[v_i, r_i] < 1e6:
                        try:
                            release(empty[v_i])
                            traci.vehicle.dispatchTaxi(empty[v_i], [open_res[r_i].id])
                            assigned.add(open_res[r_i].id)
                            if recorder is not None:
                                recorder.matched(1)
                        except Exception as e:
                            fail("dispatch", e)

            if overlay is not None:
                overlay.colour_taxis()
                overlay.update_hud(clock(t), served, len(open_res) - len(assigned), abandoned, ri)
                for ts in [ts for ts in snapshots if ts <= t]:
                    view, path = snapshots.pop(ts) if isinstance(snapshots[ts], tuple) else (VIEW_ID, snapshots[ts])
                    traci.gui.screenshot(view, str(path), 1600, 900)

            # zone-level rebalancing at slot boundaries
            if controller is not None and int(t) % decision_s < sc["dispatch_period_s"] and 0 < t < n_slots * 900:
                slot = t0_slot + int(t // 900)
                # Only parked empty taxis: ones already driving to a reposition target cannot be resumed.
                empty = [v for v in traci.vehicle.getTaxiFleet(0) if traci.vehicle.isStopped(v)]
                ez = np.array([zone_of_edge.get(traci.vehicle.getRoadID(v), -1) for v in empty], dtype=int)
                idle = np.bincount(ez[ez >= 0], minlength=city.graph.num_zones).astype(float)
                queue = np.zeros(city.graph.num_zones)
                for r in open_res:                     # riders still waiting for a taxi
                    if r.id in assigned:
                        continue
                    z = zone_of_edge.get(r.fromEdge, -1)
                    if z >= 0:
                        queue[z] += 1
                incoming = np.zeros((n_inc, city.graph.num_zones))
                parked = set(empty)
                busy = set(traci.vehicle.getTaxiFleet(1)) | set(traci.vehicle.getTaxiFleet(2))
                moving = busy | (set(traci.vehicle.getTaxiFleet(0)) - parked)   # + empty repositioning taxis
                for v in moving:
                    try:
                        route, idx = traci.vehicle.getRoute(v), traci.vehicle.getRouteIndex(v)
                        z = zone_of_edge.get(route[-1], -1)
                        if z < 0 or idx < 0:
                            continue
                        cur = route[idx]
                        frac = 1 - traci.vehicle.getLanePosition(v) / max(edge_len[cur], 1.0)
                        rem = edge_tt[cur] * max(frac, 0.0) + sum(edge_tt[e] for e in route[idx + 1:])
                        off = int(np.clip(np.ceil(rem / 900), 1, n_inc)) - 1
                        incoming[off, z] += 1
                    except Exception as e:
                        fail("incoming", e)
                obs_log.append((idle.sum(), incoming.sum()))
                import torch
                obs = dict(t=slot - 1, tod=(slot - 1) % spd, idle=torch.as_tensor(idle, dtype=torch.float32)[None],
                           incoming=torch.as_tensor(incoming, dtype=torch.float32)[None],
                           queue=torch.as_tensor(queue, dtype=torch.float32)[None],
                           next_true_pickups=torch.zeros(1, city.graph.num_zones), fleet=float(fleet))
                flows = controller.act(obs)[0].numpy()
                moves = np.floor(flows + rng.random(flows.shape)).astype(int)
                np.fill_diagonal(moves, 0)
                sent = np.zeros_like(moves)
                for i, j in zip(*np.nonzero(moves)):
                    cand = [v for v, z in zip(empty, ez) if z == i][:moves[i, j]]
                    sent[i, j] = len(cand)
                    for v in cand:
                        try:
                            target = rng.choice(edges_by_zone[j])
                            send_to(v, target)
                            rebalance_cmds += 1
                            if recorder is not None:
                                recorder.reposition(t, v, i, j, target)
                        except Exception as e:
                            fail("reposition", e)
                if overlay is not None:
                    overlay.show_decision(clock(t), ctx.forecast(slot - 1)[0, 0], idle, incoming[0], sent)
                if recorder is not None:
                    recorder.decision(t, ctx.forecast(slot - 1)[0, 0], ctx.forecast(slot - 1, "hi")[0, 0],
                                      idle, incoming[0], queue, sent)
        fleet_km = sum(traci.vehicle.getDistance(v) for v in traci.vehicle.getTaxiFleet(-1)) / 1000
    finally:
        traci.close()
    for line in err_log.read_text(errors="replace").splitlines() if err_log.exists() else []:
        if line.startswith("Error:"):
            fail("sumo", re.sub(r"'[^']*'", "'*'", line[len("Error: "):].split(" at time=")[0]))
    n = len(riders)
    out = {"riders": n, "served_rate_%": 100 * served / max(n, 1),
           "mean_wait_min": float(np.mean(waits)) if waits else float("nan"),
           "p90_wait_min": float(np.percentile(waits, 90)) if waits else float("nan"),
           "abandoned": abandoned, "fleet_km_per_served": fleet_km / max(served, 1),
           "rebalance_commands": rebalance_cmds, "traci_errors": errors, "error_types": error_types, "fleet": fleet,
           "unparked": unparked,
           "start_slot": t0_slot, "slots": n_slots,
           "obs_idle_mean": float(np.mean([a for a, _ in obs_log])) if obs_log else float("nan"),
           "obs_incoming_mean": float(np.mean([b for _, b in obs_log])) if obs_log else float("nan"),
           "engine": "libsumo" if is_lib else "traci"}
    if recorder is not None:
        places = cfg["paths"]["data_dir"] / "raw" / "osm" / "dubai_places.json"
        log(f"[sumo] replay saved to {recorder.save(record_dir, out, places)}")
    log(f"[sumo] {method:10s} " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                          for k, v in out.items() if k != "error_types"))
    if error_types:
        log("       TraCI errors: " + "; ".join(f"{k} x{v}" for k, v in error_types.items()))
    return out


def _window_label(start_slot: int, slots: int) -> str:
    hm = lambda q: f"{q * 15 // 60:02d}:{q * 15 % 60:02d}"
    return f"{hm(start_slot)}-{hm(start_slot + slots)}"


def run_sumo_comparison(cfg: dict, methods: list[str], log=print) -> dict:
    """Every method x every fleet in `sumo.fleet_sweep` x every window in `sumo.windows`, same riders."""
    sc = cfg["sumo"]
    windows = sc.get("windows") or [[sc["start_slot"], sc["slots"]]]
    fleets = sc.get("fleet_sweep") or [int(cfg["sim"]["fleet_size"] * sc.get("fleet_scale", sc["scale"]))]
    res: dict = {}
    for w0, wn in windows:
        wl = _window_label(w0, wn)
        for n in fleets:
            for m in methods:
                log(f"[sumo] window {wl}  fleet {n}  method {m}")
                res.setdefault(wl, {}).setdefault(str(n), {})[m] = run_sumo_episode(cfg, m, log, fleet=n,
                                                                                      start_slot=w0, slots=wn)
            save_json(res, cfg["paths"]["output_dir"] / "sumo_metrics.json")
    text = write_sumo_table(res, cfg)
    (cfg["paths"]["output_dir"] / "sumo_metrics.md").write_text(text, encoding="utf-8")
    log(text)
    return res


def write_sumo_table(res: dict, cfg: dict) -> str:
    sc = cfg["sumo"]
    lines = [f"## SUMO validation: fleet sweep (demand x{sc['scale']}, test day {sc['test_day_index']}, "
             f"{cfg['event_sim']['patience_min']}-min patience)", "",
             "Riders are identical across methods and fleet sizes within a window. "
             "Δ columns are relative to `none` at the same fleet. "
             "'idle / en-route seen' is the mean parked-idle and en-route (incoming) taxi count in the controller's "
             "observation at each 15-min decision.", ""]
    for wl, by_fleet in res.items():
        lines += [f"### Window {wl}", "",
                  "| Fleet | Method | riders | served % | Δ served (pts) | mean wait | Δ wait (min) | p90 wait | abandoned "
                  "| fleet km / served | rebalance cmds | idle / en-route seen | TraCI errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, by_m in by_fleet.items():
            base = by_m.get("none")
            for m, r in by_m.items():
                ds = f"{r['served_rate_%'] - base['served_rate_%']:+.1f}" if base and m != "none" else "-"
                dw = f"{r['mean_wait_min'] - base['mean_wait_min']:+.2f}" if base and m != "none" else "-"
                seen = f"{r['obs_idle_mean']:.0f} / {r['obs_incoming_mean']:.0f}" if m != "none" else "-"
                lines.append(f"| {n} | {m} | {r['riders']} | {r['served_rate_%']:.1f} | {ds} | {r['mean_wait_min']:.2f} | {dw} "
                             f"| {r['p90_wait_min']:.1f} | {r['abandoned']} | {r['fleet_km_per_served']:.2f} "
                             f"| {r['rebalance_commands']} | {seen} | {r['traci_errors']} |")
        lines.append("")
    return "\n".join(lines)


_BASE = {"decision_period_s": 900, "lp_cost": "distance", "matcher": "hungarian"}
ABLATION = [("baseline: 15-min decisions, distance LP, Hungarian", _BASE),
            ("+ 5-min decisions", {**_BASE, "decision_period_s": 300}),
            ("+ travel-time LP costs", {**_BASE, "decision_period_s": 300, "lp_cost": "traveltime"}),
            ("+ value-aware dispatch", {"decision_period_s": 300, "lp_cost": "traveltime", "matcher": "value"})]


def run_sumo_ablation(cfg: dict, method: str = "lp_robust", log=print) -> dict:
    """Add the three control changes one at a time; each row is compared with `none` under the same matcher."""
    sc = cfg["sumo"]
    windows = sc.get("windows") or [[sc["start_slot"], sc["slots"]]]
    fleets = sc.get("fleet_sweep") or [200]
    res: dict = {}
    base_none: dict = {}
    for matcher in ("hungarian", "value"):
        c = copy.deepcopy(cfg)
        c["sumo"]["matcher"] = matcher
        for w0, wn in windows:
            for n in fleets:
                log(f"[ablation] none ({matcher})  window {_window_label(w0, wn)}  fleet {n}")
                base_none[(matcher, _window_label(w0, wn), n)] = run_sumo_episode(c, "none", log, fleet=n, start_slot=w0, slots=wn)
    for name, over in ABLATION:
        c = copy.deepcopy(cfg)
        c["sumo"].update(over)
        for w0, wn in windows:
            for n in fleets:
                log(f"[ablation] {name}  window {_window_label(w0, wn)}  fleet {n}")
                res.setdefault(name, {}).setdefault(_window_label(w0, wn), {})[str(n)] = run_sumo_episode(
                    c, method, log, fleet=n, start_slot=w0, slots=wn)
    out = cfg["paths"]["output_dir"]
    save_json({"variants": res, "none": {f"{m}|{w}|{n}": v for (m, w, n), v in base_none.items()}}, out / "sumo_ablation.json")
    lines = [f"## SUMO ablation: {method}, one change at a time (demand x{sc['scale']}, network {sc.get('network', 'grid')})", "",
             "Δ vs none uses `none` with the same dispatcher (Hungarian, or value-aware for the last row); "
             "Δ vs baseline none always uses Hungarian `none`. Riders are identical in every run of a window.", ""]
    for w0, wn in windows:
        wl = _window_label(w0, wn)
        lines += [f"### Window {wl}", "", "| Variant | Fleet | served % | mean wait | p90 wait | fleet km / served | rebalance cmds "
                  "| Δ served vs none | Δ wait vs none | Δ wait vs baseline none | TraCI errors |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, over in ABLATION:
            for n in fleets:
                r = res[name][wl][str(n)]
                b = base_none[(over.get("matcher", "hungarian"), wl, n)]
                b0 = base_none[("hungarian", wl, n)]
                lines.append(f"| {name} | {n} | {r['served_rate_%']:.1f} | {r['mean_wait_min']:.2f} | {r['p90_wait_min']:.1f} "
                             f"| {r['fleet_km_per_served']:.2f} | {r['rebalance_commands']} | {r['served_rate_%'] - b['served_rate_%']:+.1f} "
                             f"| {r['mean_wait_min'] - b['mean_wait_min']:+.2f} | {r['mean_wait_min'] - b0['mean_wait_min']:+.2f} "
                             f"| {r['traci_errors']} |")
        lines += ["", "| none (value-aware dispatch) | Fleet | served % | mean wait | Δ wait vs Hungarian none |", "|---|---|---|---|---|"]
        for n in fleets:
            v, h = base_none[("value", wl, n)], base_none[("hungarian", wl, n)]
            lines.append(f"| none + value dispatch | {n} | {v['served_rate_%']:.1f} | {v['mean_wait_min']:.2f} | "
                         f"{v['mean_wait_min'] - h['mean_wait_min']:+.2f} |")
        lines.append("")
    text = "\n".join(lines)
    (out / "sumo_ablation.md").write_text(text, encoding="utf-8")
    log(text)
    return res
