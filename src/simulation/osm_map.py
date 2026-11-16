"""Real Dubai road network for the SUMO bridge (`sumo.network: osm`).

The synthetic city is a 10 x 4 grid of 3 km zones along a 30 km corridor. Here that grid is laid on
the real city: the long axis runs from Dubai Marina towards Deira along Sheikh Zayed Road
(`sumo.osm.origin` -> `sumo.osm.toward`), the short axis spans 12 km across it. Zone row 0 is the
inland side, row 3 the Gulf side. Roads, buildings and water come from OpenStreetMap through the
Overpass API (data (c) OpenStreetMap contributors, ODbL); netconvert/polyconvert turn them into a
SUMO network with real junctions, ramps and guessed traffic lights.
"""
from __future__ import annotations

import math
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

KM_PER_DEG_LAT = 111.2
MIRRORS = ["https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
           "https://z.overpass-api.de/api/interpreter"]
ROAD_TYPES = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|"
              "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
NO_KERBSIDE = ("highway.motorway", "highway.motorway_link", "highway.trunk", "highway.trunk_link")


class CorridorFrame:
    """Maps lon/lat <-> (along, across) km in the corridor and to synthetic zone ids."""

    def __init__(self, cfg: dict):
        s, o = cfg["data"]["synthetic"], cfg["sumo"]["osm"]
        self.rows, self.cols, self.cell = s["grid_rows"], s["grid_cols"], s["cell_km"]
        self.lat0, self.lon0 = o["origin"]
        lat1, lon1 = o["toward"]
        self.kx = KM_PER_DEG_LAT * math.cos(math.radians(self.lat0))
        dx, dy = (lon1 - self.lon0) * self.kx, (lat1 - self.lat0) * KM_PER_DEG_LAT
        n = math.hypot(dx, dy)
        self.u = (dx / n, dy / n)                    # along the corridor
        self.v = (-self.u[1], self.u[0])             # across, pointing left of travel (Gulf side)
        self.bearing = math.degrees(math.atan2(dx, dy))

    def to_frame(self, lon, lat):
        x, y = (lon - self.lon0) * self.kx, (lat - self.lat0) * KM_PER_DEG_LAT
        return x * self.u[0] + y * self.u[1], x * self.v[0] + y * self.v[1]

    def to_lonlat(self, along, across):
        x = along * self.u[0] + across * self.v[0]
        y = along * self.u[1] + across * self.v[1]
        return self.lon0 + x / self.kx, self.lat0 + y / KM_PER_DEG_LAT

    def zone_of(self, lon, lat) -> int:
        a, c = self.to_frame(lon, lat)
        col, row = math.floor(a / self.cell), math.floor((c + self.rows * self.cell / 2) / self.cell)
        return row * self.cols + col if 0 <= col < self.cols and 0 <= row < self.rows else -1

    def zone_corners(self, z: int):
        row, col = divmod(z, self.cols)
        a0, c0 = col * self.cell, row * self.cell - self.rows * self.cell / 2
        return [self.to_lonlat(a, c) for a, c in ((a0, c0), (a0 + self.cell, c0),
                                                   (a0 + self.cell, c0 + self.cell), (a0, c0 + self.cell))]

    def poly(self, margin_km: float = 0.5, a0: float | None = None, a1: float | None = None) -> str:
        """Overpass `poly:` filter covering the corridor (or its [a0, a1] km slice) plus a margin."""
        a0 = -margin_km if a0 is None else a0
        L = self.cols * self.cell + margin_km if a1 is None else a1
        W = self.rows * self.cell / 2 + margin_km
        pts = [self.to_lonlat(a, c) for a, c in ((a0, -W), (L, -W), (L, W), (a0, W))]
        return " ".join(f"{lat:.5f} {lon:.5f}" for lon, lat in pts)


def _overpass(query: str, out: Path, log=print):
    data = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(6):
        url = MIRRORS[attempt % len(MIRRORS)]
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "taxi-dispatch-research/1.0 (SUMO)"})
            with urllib.request.urlopen(req, timeout=900) as r, open(out.with_suffix(".part"), "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            out.with_suffix(".part").replace(out)
            log(f"[osm] {out.name}: {out.stat().st_size / 1e6:.1f} MB from {url}")
            return
        except Exception as e:                                  # busy mirrors return 429/502/504
            log(f"[osm] {url} failed ({e}); retrying")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Overpass download failed for {out.name}")


def download(cfg: dict, raw: Path, log=print) -> tuple[Path, list[Path], Path]:
    """Roads, [water/parks + building tiles], coastline. Buildings come in 6 km tiles: one query for
    the whole corridor is too heavy for the public Overpass servers."""
    raw.mkdir(parents=True, exist_ok=True)
    frame = CorridorFrame(cfg)
    poly = frame.poly()
    roads, extra, coast = raw / "dubai_roads.osm", raw / "dubai_water_parks.osm", raw / "dubai_coastline.osm"
    if not roads.exists():
        _overpass(f'[timeout:900][maxsize:1073741824];(way["highway"~"^({ROAD_TYPES})$"](poly:"{poly}");>;);out body;',
                  roads, log)
    if not extra.exists():
        _overpass(f'[timeout:900];(way["natural"~"^(water|beach|sand)$"](poly:"{poly}");'
                  f'way["leisure"~"^(park|golf_course)$"](poly:"{poly}");'
                  f'way["landuse"~"^(grass|forest|recreation_ground)$"](poly:"{poly}"););out body;>;out skel qt;', extra, log)
    if not coast.exists():
        _overpass(f'[timeout:300];way["natural"="coastline"](poly:"{frame.poly(margin_km=3)}");out body;>;out skel qt;',
                  coast, log)
    tiles = []
    L = frame.cols * frame.cell
    for k, a in enumerate(range(0, int(L), 6)):
        tile = raw / f"dubai_buildings_{k}.osm"
        if not tile.exists():
            _overpass(f'[timeout:900];way["building"](poly:"{frame.poly(a0=a - 0.5, a1=min(a + 6, L) + 0.5)}");'
                      f'out body;>;out skel qt;', tile, log)
        tiles.append(tile)
    return roads, [extra] + tiles, coast


def build(cfg: dict, work: Path, sumo_home: Path, log=print) -> tuple[Path, Path]:
    """Download (once) and convert. Returns (net.xml, poly.xml)."""
    net, polys, sea = work / "dubai.net.xml", work / "dubai.poly.xml", work / "dubai_sea.poly.xml"
    roads, extra, coast = download(cfg, cfg["paths"]["data_dir"] / "raw" / "osm", log)
    exe = lambda name: str(sumo_home / "bin" / (f"{name}.exe" if sys.platform == "win32" else name))
    if not net.exists():
        convert_roads(roads, net, sumo_home, log)
    if not polys.exists():
        log("[osm] polyconvert buildings, water and parks")
        subprocess.run([exe("polyconvert"), "--net-file", str(net), "--osm-files", ",".join(map(str, extra)),
                        "-o", str(polys), "--type-file", str(sumo_home / "data" / "typemap" / "osmPolyconvert.typ.xml"),
                        "--osm.keep-full-type", "false", "--no-warnings"], check=True)
        # Drop point-of-interest markers (shops, amenities): their labels clutter the overlay.
        text = polys.read_text(encoding="utf-8")
        polys.write_text("\n".join(l for l in text.splitlines() if not l.lstrip().startswith("<poi ")), encoding="utf-8")
    if not sea.exists():
        import sumolib
        write_background(sumolib.net.readNet(str(net)), cfg, coast, sea)
    return net, [sea, polys]


def convert_roads(roads: Path, net: Path, sumo_home: Path, log=print):
    exe = lambda name: str(sumo_home / "bin" / (f"{name}.exe" if sys.platform == "win32" else name))
    log("[osm] netconvert (a few minutes the first time)")
    subprocess.run([exe("netconvert"), "--osm-files", str(roads), "-o", str(net),
                    "--type-files", str(sumo_home / "data" / "typemap" / "osmNetconvert.typ.xml"),
                    "--geometry.remove", "--ramps.guess", "--roundabouts.guess", "--junctions.join",
                    "--tls.guess-signals", "--tls.discard-simple", "--tls.join", "--tls.default-type", "actuated",
                    "--keep-edges.by-vclass", "passenger", "--remove-edges.isolated",
                    "--keep-edges.components", "1", "--output.street-names", "--no-warnings"], check=True)


def strongly_connected(net, edges: list) -> set[str]:
    """Edges of the largest strongly connected passenger component, so every taxi route exists."""
    best: set[str] = set()
    remaining = set(e.getID() for e in edges)
    for e in sorted(edges, key=lambda e: -e.getLength()):
        if e.getID() not in remaining:
            continue
        fwd = {x.getID() for x in net.getReachable(e, "passenger")}
        bwd = {x.getID() for x in net.getReachable(e, "passenger", useIncoming=True)}
        comp = (fwd & bwd) | {e.getID()}
        remaining -= comp
        if len(comp) > len(best):
            best = comp
        if len(best) > len(remaining):
            break
    return best


def coastline_shapes(coast_osm: Path, frame: CorridorFrame, reach_km: float = 25.0):
    """Stitch OSM coastline ways (land on the left) into (sea polygons, island polygons) in lon/lat.

    Open chains (the mainland shore leaving the query area) are closed far out on the Gulf side of
    the corridor; closed chains are islands such as the Palm and World islands."""
    import xml.etree.ElementTree as ET
    root = ET.parse(coast_osm).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    ways = [[nd.get("ref") for nd in w.iter("nd")] for w in root.iter("way")]
    ways = [w for w in ways if len(w) > 1]
    by_start = {w[0]: i for i, w in enumerate(ways)}
    by_end = {w[-1]: i for i, w in enumerate(ways)}
    used, chains = set(), []
    for i in range(len(ways)):
        if i in used:
            continue
        start, seen = i, {i}
        while (p := by_end.get(ways[start][0])) is not None and p not in used and p not in seen:
            start = p                                 # walk back to the chain start
            seen.add(p)
        chain, cur = list(ways[start]), start
        used.add(start)
        while (n := by_start.get(chain[-1])) is not None and n not in used:
            used.add(n)
            chain += ways[n][1:]
        chains.append(chain)
    sea, islands = [], []
    for ch in chains:
        pts = [nodes[r] for r in ch if r in nodes]
        if len(pts) < 3:
            continue
        if ch[0] == ch[-1]:
            islands.append(pts)
        else:                                         # close far out on the Gulf (+across) side
            (a0, _), (a1, _) = frame.to_frame(*pts[0]), frame.to_frame(*pts[-1])
            sea.append(pts + [frame.to_lonlat(a1, reach_km), frame.to_lonlat(a0, reach_km)])
    return sea, islands


def write_background(net, cfg: dict, coast_osm: Path, out: Path):
    """Sea and island polygons as a SUMO additional file (drawn under roads and zones)."""
    sea, islands = coastline_shapes(coast_osm, CorridorFrame(cfg))
    xy = lambda pts: " ".join(f"{x:.1f},{y:.1f}" for x, y in (net.convertLonLat2XY(lon, lat) for lon, lat in pts))
    lines = ["<additional>"]
    lines += [f'  <poly id="sea{i}" type="sea" color="170,208,232" fill="1" layer="-4" shape="{xy(p)}"/>'
              for i, p in enumerate(sea)]
    lines += [f'  <poly id="island{i}" type="island" color="242,236,222" fill="1" layer="-3" shape="{xy(p)}"/>'
              for i, p in enumerate(islands)]
    out.write_text("\n".join(lines + ["</additional>"]))
    return len(sea), len(islands)
