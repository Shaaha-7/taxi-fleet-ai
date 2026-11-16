"""NYC TLC trip records -> 15-min zone-level OD counts (same layout as the synthetic generator).

Download monthly Yellow/Green/HVFHV Parquet files from the NYC TLC trip record page into
`data/raw/nyc/`, plus a zone centroid CSV (LocationID,lat,lon) derived from the TLC taxi
zone shapefile. Pickup time columns differ by service, so the first one found is used.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .graph import ZoneGraph

PICKUP_TIME_COLS = ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"]


def _project_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(lat.mean())
    x = np.deg2rad(lon - lon.mean()) * 6371.0 * np.cos(lat0)
    y = np.deg2rad(lat - lat.mean()) * 6371.0
    return np.stack([x, y], axis=1)


def centroids_from_shapefile(zip_path, out_csv) -> pd.DataFrame:
    """Area-weighted polygon centroids of the TLC taxi-zone shapefile (EPSG:2263, US feet) in km."""
    import io
    import zipfile

    import shapefile                                        # pyshp
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        get = lambda ext: io.BytesIO(z.read(next(n for n in names if n.lower().endswith(ext))))
        reader = shapefile.Reader(shp=get(".shp"), dbf=get(".dbf"), shx=get(".shx"))
        fields = [f[0] for f in reader.fields[1:]]
        rows = []
        for rec, shp in zip(reader.records(), reader.shapes()):
            pts = np.asarray(shp.points)
            parts = list(shp.parts) + [len(pts)]
            area_sum, cx, cy = 0.0, 0.0, 0.0
            for a, b in zip(parts[:-1], parts[1:]):
                x, y = pts[a:b, 0], pts[a:b, 1]
                cross = x * np.roll(y, -1) - np.roll(x, -1) * y
                area = cross.sum() / 2
                if abs(area) < 1e-9:
                    continue
                cx += ((x + np.roll(x, -1)) * cross).sum() / 6
                cy += ((y + np.roll(y, -1)) * cross).sum() / 6
                area_sum += area
            loc = int(dict(zip(fields, rec))["LocationID"])
            rows.append((loc, cx / area_sum * 0.0003048, cy / area_sum * 0.0003048))
    df = pd.DataFrame(rows, columns=["LocationID", "x_km", "y_km"]).drop_duplicates("LocationID").sort_values("LocationID")
    df.to_csv(out_csv, index=False)
    return df


def build_graph(centroids: pd.DataFrame, knn: int, detour: float) -> ZoneGraph:
    if {"x_km", "y_km"} <= set(centroids.columns):
        coords = centroids[["x_km", "y_km"]].to_numpy(dtype=float)
        coords = coords - coords.mean(0)
    else:
        coords = _project_km(centroids["lat"].to_numpy(), centroids["lon"].to_numpy())
    diff = coords[:, None] - coords[None]
    euclid = np.sqrt((diff ** 2).sum(-1))
    order = np.argsort(euclid, axis=1)[:, 1:knn + 1]
    adj = np.zeros_like(euclid, dtype=bool)
    adj[np.repeat(np.arange(len(coords)), knn), order.ravel()] = True
    adj |= adj.T
    nn_dist = np.take_along_axis(euclid, order[:, :1], axis=1).ravel()
    dist = detour * euclid
    np.fill_diagonal(dist, 0.5 * nn_dist * detour)
    ids = centroids["LocationID"].astype(int).tolist()
    return ZoneGraph(coords, dist, adj, [f"TLC_{i}" for i in ids], ["nyc"] * len(ids))


def generate(cfg: dict) -> dict:
    c = cfg["data"]["nyc_tlc"]
    files = sorted(glob.glob(str(cfg["paths"]["data_dir"].parent / c["parquet_glob"])))
    if not files:
        raise FileNotFoundError(f"No Parquet files match {c['parquet_glob']}; see README for download steps.")
    csv = cfg["paths"]["data_dir"].parent / c["zone_centroids_csv"]
    shp_zip = csv.parent / "taxi_zones.zip"
    if not csv.exists() and shp_zip.exists():
        centroids_from_shapefile(shp_zip, csv)
    centroids = pd.read_csv(csv).sort_values("LocationID")
    zone_pos = {int(z): i for i, z in enumerate(centroids["LocationID"])}
    graph = build_graph(centroids, c["knn"], cfg["sim"]["detour_factor"])
    freq = f"{cfg['data']['slot_minutes']}min"

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        tcol = next(col for col in PICKUP_TIME_COLS if col in df.columns)
        df = df[[tcol, "PULocationID", "DOLocationID"]].rename(columns={tcol: "t"})
        df = df[df["PULocationID"].isin(zone_pos) & df["DOLocationID"].isin(zone_pos)]
        frames.append(df)
    trips = pd.concat(frames, ignore_index=True)
    trips["t"] = pd.to_datetime(trips["t"]).dt.floor(freq)

    # Keep whole days only, dropping stray out-of-month timestamps common in TLC files.
    day_counts = trips["t"].dt.normalize().value_counts()
    valid_days = day_counts[day_counts > day_counts.median() * 0.2].index.sort_values()
    start, end = valid_days.min(), valid_days.max() + pd.Timedelta(days=1)
    slot_index = pd.date_range(start, end, freq=freq, inclusive="left")
    trips = trips[(trips["t"] >= start) & (trips["t"] < end)]

    slot = ((trips["t"] - start) / pd.Timedelta(freq)).astype(int).to_numpy()
    o = trips["PULocationID"].map(zone_pos).to_numpy()
    d = trips["DOLocationID"].map(zone_pos).to_numpy()
    n, t = graph.num_zones, len(slot_index)
    # A full [T, N, N] OD array would need ~2.4 GB for a quarter of NYC; keep zone series only.
    pickups = np.bincount(slot * n + o, minlength=t * n).reshape(t, n).astype(np.int32)
    dropoffs = np.bincount(slot * n + d, minlength=t * n).reshape(t, n).astype(np.int32)
    return dict(graph=graph, od=None, pickups=pickups, dropoffs=dropoffs, slot_index=slot_index, events=[])
