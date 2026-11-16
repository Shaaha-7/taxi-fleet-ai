"""Live sumo-gui overlay for the SUMO bridge (`python main.py sumo --gui`).

Shows the zone-level decision loop on top of the microscopic traffic:
* Taxis coloured by state: green parked-empty, orange to pickup, red with rider, blue repositioning.
* Zones shaded by the 15-min gap between forecast demand and supply (idle + arriving):
  red = shortage, blue = surplus. Labels give the zone id, forecast demand, parked and arriving taxis.
* Purple (RL) or teal (LP) lines from each source zone to each destination zone, width ~ taxis moved;
  they stay until the next decision.
* A status line above the map: clock, method, served / waiting / abandoned, last decision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TAXI_COLOURS = {"parked": (40, 180, 70, 255), "pickup": (240, 150, 20, 255),
                "occupied": (220, 40, 40, 255), "reposition": (40, 110, 240, 255)}
VIEW = "View #0"

VIEW_SETTINGS = """<viewsettings>
    <scheme name="taxi-overlay">
        <background backgroundColor="{bg}" showGrid="0"/>
        <vehicles vehicleMode="0" vehicleQuality="1" vehicle_minSize="1" vehicle_exaggeration="{veh}" vehicle_constantSize="0"/>
        <persons personMode="0" personQuality="1" person_minSize="1" person_exaggeration="{per}" person_constantSize="0"/>
        <pois poi_minSize="0" poi_exaggeration="1" poi_constantSize="1" poiName_show="0"
              poiType_show="1" poiType_size="{label}" poiType_color="black" poiType_constantSize="1"/>
    </scheme>
    <delay value="{delay}"/>
</viewsettings>
"""


def write_view_settings(work: Path, delay_ms: int, real_map: bool = False) -> Path:
    """Real map: sand-coloured land under the sea polygons, smaller labels, larger taxis."""
    path = work / "gui_settings.xml"
    path.write_text(VIEW_SETTINGS.format(delay=delay_ms, label=34 if real_map else 48,
                                         bg="242,236,222" if real_map else "white",
                                         veh=45 if real_map else 30, per=35 if real_map else 25))
    return path


class SumoOverlay:
    """`to_xy(along_m, across_m)` maps corridor coordinates (across measured from the centre line) to
    network XY, so the same overlay works on the synthetic grid and on the rotated Dubai corridor."""

    def __init__(self, traci, cfg: dict, method: str, to_xy, view_angle: float = 0.0):
        s = cfg["data"]["synthetic"]
        self.traci, self.method, self.to_xy = traci, method, to_xy
        self.rows, self.cols, self.cell = s["grid_rows"], s["grid_cols"], s["cell_km"] * 1000
        self.n = self.rows * self.cols
        self.half = self.rows * self.cell / 2
        self.is_rl = method.endswith("_rl")
        self.arrow_rgb = (130, 80, 220) if self.is_rl else (20, 150, 120)
        self.arrows: list[str] = []
        self.last_decision = "no decision yet"
        L = self.cols * self.cell
        traci.gui.setSchema(VIEW, "taxi-overlay")
        corners = [to_xy(a, c) for a, c in ((-300, -self.half - 900), (L + 300, -self.half - 900),
                                            (L + 300, self.half + 1500), (-300, self.half + 1500))]
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        if view_angle:
            traci.gui.setAngle(VIEW, view_angle)
            cx, cy = to_xy(L / 2, 300)
            traci.gui.setOffset(VIEW, cx, cy)
            traci.gui.setZoom(VIEW, 100 * 1600 / (L + 1200) * 0.9 * 33)    # ~fit the 30 km corridor
        else:
            traci.gui.setBoundary(VIEW, min(xs), min(ys), max(xs), max(ys))
        for z in range(self.n):
            r, c = divmod(z, self.cols)
            a0, c0 = c * self.cell, r * self.cell - self.half
            shape = [to_xy(a, b) for a, b in ((a0, c0), (a0 + self.cell, c0), (a0 + self.cell, c0 + self.cell),
                                              (a0, c0 + self.cell))]
            traci.polygon.add(f"zone{z}", shape, (200, 200, 200, 40), fill=True, polygonType="zone", layer=-1)
            traci.poi.add(f"zlabel{z}", *to_xy(a0 + self.cell / 2, c0 + self.cell * 0.86), (0, 0, 0, 0),
                          poiType=f"Z{z}", layer=5)
            traci.poi.add(f"zstat{z}", *to_xy(a0 + self.cell / 2, c0 + self.cell * 0.74), (0, 0, 0, 0),
                          poiType="", layer=5)
        traci.poi.add("hud", *to_xy(L / 2, self.half + 1100), (0, 0, 0, 0), poiType="starting", layer=5)
        traci.poi.add("hud2", *to_xy(L / 2, self.half + 500), (0, 0, 0, 0), poiType="", layer=5)
        traci.poi.add("legend", *to_xy(L / 2, -self.half - 700), (0, 0, 0, 0), layer=5, poiType=(
            "Taxis: green parked | orange to pickup | red with rider | blue repositioning.   "
            "Zones: red shortage, blue surplus.   Lines: " + ("RL" if self.is_rl else "LP") + " rebalancing moves"))

    def centre(self, z: int):
        r, c = divmod(z, self.cols)
        return self.to_xy((c + 0.5) * self.cell, (r + 0.5) * self.cell - self.half)

    def colour_taxis(self):
        tv = self.traci.vehicle
        for v in tv.getTaxiFleet(0):
            tv.setColor(v, TAXI_COLOURS["parked" if tv.isStopped(v) else "reposition"])
        for v in tv.getTaxiFleet(1):
            tv.setColor(v, TAXI_COLOURS["pickup"])
        for v in tv.getTaxiFleet(2):
            tv.setColor(v, TAXI_COLOURS["occupied"])

    def show_decision(self, clock: str, demand: np.ndarray, idle: np.ndarray, arriving: np.ndarray, moves: np.ndarray):
        tr = self.traci
        gap = demand - idle - arriving                        # >0: shortage next 15 min
        for z in range(self.n):
            g = float(np.clip(gap[z] / 5.0, -1, 1))
            colour = (220, 60, 60, int(40 + 110 * g)) if g > 0 else (60, 110, 230, int(40 + 110 * -g))
            tr.polygon.setColor(f"zone{z}", colour)
            tr.poi.setType(f"zlabel{z}", f"Z{z}  {demand[z]:.1f} req")
            tr.poi.setType(f"zstat{z}", f"{idle[z]:.0f} idle  {arriving[z]:.0f} in")
        for a in self.arrows:
            tr.polygon.remove(a)
        self.arrows = []
        for i, j in zip(*np.nonzero(moves)):
            (xa, ya), (xb, yb) = self.centre(i), self.centre(j)
            k = int(moves[i, j])
            # Shorten so the line ends inside the target zone; small head drawn as a short spur pair.
            d = np.array([xb - xa, yb - ya]); L = np.linalg.norm(d); u = d / max(L, 1)
            tip = np.array([xb, yb]) - u * self.cell * 0.15
            side = np.array([-u[1], u[0]])
            head = [tuple(tip - u * 350 + side * 220), tuple(tip), tuple(tip - u * 350 - side * 220)]
            pid = f"mv{i}_{j}"
            tr.polygon.add(pid, [(xa, ya), tuple(tip)], (*self.arrow_rgb, 230), fill=False,
                           polygonType="move", layer=4, lineWidth=40 + 25 * min(k, 8))
            tr.polygon.add(pid + "h", head, (*self.arrow_rgb, 230), fill=False, polygonType="move", layer=4,
                           lineWidth=60 + 25 * min(k, 8))
            self.arrows += [pid, pid + "h"]
        n_moved = int(moves.sum())
        self.last_decision = (f"{clock} moved {n_moved} taxis on {len(self.arrows) // 2} routes  |  "
                              f"{int(idle.sum())} parked, {int(arriving.sum())} arriving in 15 min, "
                              f"{demand.sum():.0f} pickups forecast")

    def update_hud(self, clock: str, served: int, waiting: int, abandoned: int, requested: int):
        self.traci.poi.setType("hud", f"{clock}   {self.method}   requests {requested}   served {served}   "
                                      f"waiting {waiting}   abandoned {abandoned}")
        self.traci.poi.setType("hud2", "Last decision: " + self.last_decision)
