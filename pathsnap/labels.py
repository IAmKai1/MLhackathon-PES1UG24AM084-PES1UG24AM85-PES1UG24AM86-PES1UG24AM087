"""Derive a reference road segment for every GPS observation.

The .route file is a chronological list of arcs, not one label per GPS point.
Each observation is assigned to a route arc by a monotone alignment: a dynamic
programme that walks forward through the route while minimising the total
point-to-arc distance, advancing at most `max_skip` arcs per second.
"""
import numpy as np


def align_route(trip, net, max_skip=12):
    route = trip.route
    n, L = len(trip.xy), len(route)
    if L == 0 or n == 0:
        return np.full(n, -1), np.full(n, np.inf)
    a, b = net.a[route], net.b[route]                 # (L, 2)
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)

    def dist_row(p):
        t = np.clip(((p - a) * ab).sum(axis=1) / l2, 0, 1)
        q = a + t[:, None] * ab
        return np.hypot(q[:, 0] - p[0], q[:, 1] - p[1])

    INF = 1e18
    back = np.zeros((n, L), dtype=np.int16)           # how many arcs we advanced
    cost = dist_row(trip.xy[0])
    cost[min(L, max_skip + 1):] = INF                  # trip starts near the route start
    for i in range(1, n):
        best = cost.copy()
        arg = np.zeros(L, dtype=np.int16)
        for s in range(1, max_skip + 1):              # advance s arcs from j-s to j
            shifted = np.full(L, INF)
            shifted[s:] = cost[:-s]
            better = shifted < best
            best[better] = shifted[better]
            arg[better] = s
        cost = best + dist_row(trip.xy[i])
        back[i] = arg
    j = int(np.argmin(cost))
    idx = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        idx[i] = j
        j -= int(back[i, j])
    labels = route[idx]
    d, _ = net.project_points(trip.xy, labels)
    return labels, d
