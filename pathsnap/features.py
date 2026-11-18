"""Candidate generation and per-candidate feature engineering.

For every GPS observation we gather up to K nearby road arcs (the candidates).
Each (observation, candidate) pair becomes one row; the binary target is
whether that candidate is the observation's reference arc. A classifier then
scores candidates, and the HMM uses those scores as emission probabilities.
"""
import numpy as np
import pandas as pd

from .labels import align_route
from .network import RoadNetwork

DEFAULT_RADIUS = 60.0      # metres; chosen from the candidate-miss analysis on training trips
DEFAULT_K = 10
FALLBACK_RADII = (120.0, 250.0, 500.0)
WINDOW = 5                 # observations either side used for context features

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "heading_diff", "heading_diff_win", "speed", "moving",
    "arc_len", "proj_t", "at_endpoint", "n_cand",
    "dist_prev", "dist_next", "dist_win_mean", "dist_win_max",
    "tail_out_deg", "head_in_deg", "time_gap",
]


def _heading(xy, min_move=2.0, span=2):
    """Travel bearing (deg, 0 = north) from points `span` steps either side."""
    n = len(xy)
    lo = np.clip(np.arange(n) - span, 0, n - 1)
    hi = np.clip(np.arange(n) + span, 0, n - 1)
    d = xy[hi] - xy[lo]
    move = np.hypot(d[:, 0], d[:, 1])
    hdg = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0
    valid = move >= min_move
    # Carry the last valid heading through stops so a stationary car keeps its direction
    out = np.full(n, np.nan)
    last = np.nan
    for i in range(n):
        if valid[i]:
            last = hdg[i]
        out[i] = last
    # Back-fill the start of the trip if it began stationary
    first = np.argmax(~np.isnan(out)) if (~np.isnan(out)).any() else None
    if first is not None:
        out[:first] = out[first]
    return np.nan_to_num(out, nan=0.0), valid


def _angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def generate_candidates(trip, net, radius=DEFAULT_RADIUS, k=DEFAULT_K):
    """Return (N, k) arc ids (-1 padded) sorted by distance, plus the distances."""
    n = len(trip.xy)
    cand = np.full((n, k), -1, dtype=np.int64)
    dist = np.full((n, k), np.inf)
    for i, p in enumerate(trip.xy):
        ids = net.arcs_near(p, radius)
        r_used = radius
        for r in FALLBACK_RADII:                     # never leave a point without candidates
            if len(ids):
                break
            ids, r_used = net.arcs_near(p, r), r
        if not len(ids):
            continue
        d, _ = net.project_points(np.repeat(p[None], len(ids), 0), ids)
        keep = d <= r_used
        ids, d = ids[keep], d[keep]
        order = np.argsort(d)[:k]
        cand[i, :len(order)] = ids[order]
        dist[i, :len(order)] = d[order]
    return cand, dist


def build_trip_frame(trip, radius=DEFAULT_RADIUS, k=DEFAULT_K, with_labels=True, net=None):
    """One row per (observation, candidate) with features and, if available, the label."""
    net = net or RoadNetwork(trip.node_xy, trip.arcs)
    xy, t = trip.xy, trip.t
    n = len(xy)
    cand, cdist = generate_candidates(trip, net, radius, k)
    valid = cand >= 0

    hdg, moving = _heading(xy)
    step = np.zeros(n)
    step[1:] = np.hypot(*(xy[1:] - xy[:-1]).T)
    dt = np.ones(n)
    dt[1:] = np.maximum(np.diff(t), 1e-6)
    speed = step / dt
    gap = np.zeros(n)
    gap[1:] = np.diff(t)

    obs = np.repeat(np.arange(n), k).reshape(n, k)
    arc = np.where(valid, cand, 0)

    proj_d, proj_t = net.project_points(np.repeat(xy, k, 0), arc.ravel())
    proj_d, proj_t = proj_d.reshape(n, k), proj_t.reshape(n, k)
    min_d = np.where(valid, proj_d, np.inf).min(axis=1, keepdims=True)

    # Context: distance from neighbouring observations to the same candidate arc
    win = []
    for o in range(-WINDOW, WINDOW + 1):
        j = np.clip(np.arange(n) + o, 0, n - 1)
        dj, _ = net.project_points(np.repeat(xy[j], k, 0), arc.ravel())
        win.append(dj.reshape(n, k))
    win = np.stack(win)                               # (2W+1, n, k)
    hd = _angle_diff(hdg[:, None], net.bearing[arc])
    hd_win = np.mean([_angle_diff(hdg[np.clip(np.arange(n) + o, 0, n - 1)][:, None],
                                  net.bearing[arc]) for o in range(-WINDOW, WINDOW + 1, 2)], axis=0)

    feats = {
        "dist": proj_d,
        "dist_rank": np.tile(np.arange(k), (n, 1)).astype(float),
        "dist_minus_min": proj_d - min_d,
        "dist_ratio": proj_d / (min_d + 1.0),
        "heading_diff": hd,
        "heading_diff_win": hd_win,
        "speed": np.repeat(speed[:, None], k, 1),
        "moving": np.repeat(moving[:, None].astype(float), k, 1),
        "arc_len": net.length[arc],
        "proj_t": proj_t,
        "at_endpoint": ((proj_t <= 1e-6) | (proj_t >= 1 - 1e-6)).astype(float),
        "n_cand": np.repeat(valid.sum(axis=1, keepdims=True), k, 1).astype(float),
        "dist_prev": win[WINDOW - 1],
        "dist_next": win[WINDOW + 1],
        "dist_win_mean": win.mean(axis=0),
        "dist_win_max": win.max(axis=0),
        "tail_out_deg": net.out_deg[trip.arcs[arc, 0]].astype(float),
        "head_in_deg": net.in_deg[trip.arcs[arc, 1]].astype(float),
        "time_gap": np.repeat(gap[:, None], k, 1),
    }
    m = valid.ravel()
    df = pd.DataFrame({name: v.ravel()[m] for name, v in feats.items()})
    df.insert(0, "trip", trip.trip_id)
    df.insert(1, "obs", obs.ravel()[m])
    df.insert(2, "arc", cand.ravel()[m])

    labels = None
    if with_labels and trip.route is not None:
        labels, align_d = align_route(trip, net)
        df["label_arc"] = labels[df["obs"].to_numpy()]
        df["y"] = (df["arc"] == df["label_arc"]).astype(np.int8)
        df["align_dist"] = align_d[df["obs"].to_numpy()]
    return df, cand, labels, net
