"""Loading raw PathSnap trips and projecting them to a metric plane."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EARTH_R = 6371008.8  # mean Earth radius, metres


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (M, 2) lon, lat
    arcs: np.ndarray             # (A, 2) from_node, to_node
    route: np.ndarray | None     # (L,) reference arc ids in travel order, if supplied
    lat0: float = 0.0
    lon0: float = 0.0
    xy: np.ndarray = field(default=None, repr=False)        # (N, 2) GPS in metres
    node_xy: np.ndarray = field(default=None, repr=False)   # (M, 2) nodes in metres

    @property
    def t(self):
        return self.track[:, 2]


def _read(path, dtype, ncols):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.size == 0:
        return np.empty((0, ncols), dtype=dtype)
    return arr


def project(lon, lat, lon0, lat0):
    """Local equirectangular projection around (lon0, lat0), in metres.

    Each trip spans a few kilometres, where this projection's distance error
    is far below GPS noise, and it avoids a pyproj dependency.
    """
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.column_stack([x, y])


def unproject(xy, lon0, lat0):
    lon = lon0 + np.degrees(xy[:, 0] / (EARTH_R * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(xy[:, 1] / EARTH_R)
    return np.column_stack([lon, lat])


def load_trip(trip_dir, with_route=True):
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", float, 3)
    nodes = _read(trip_dir / f"{tid}.nodes", float, 2)
    arcs = _read(trip_dir / f"{tid}.arcs", np.int64, 2)
    route = None
    route_path = trip_dir / f"{tid}.route"
    if with_route and route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int64, ndmin=1)
    # Drop invalid GPS rows and enforce time order
    ok = np.isfinite(track).all(axis=1)
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]
    trip = Trip(tid, track, nodes, arcs, route)
    trip.lon0, trip.lat0 = float(track[:, 0].mean()), float(track[:, 1].mean())
    trip.xy = project(track[:, 0], track[:, 1], trip.lon0, trip.lat0)
    trip.node_xy = project(nodes[:, 0], nodes[:, 1], trip.lon0, trip.lat0)
    return trip


def list_trips(split_dir):
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir())
