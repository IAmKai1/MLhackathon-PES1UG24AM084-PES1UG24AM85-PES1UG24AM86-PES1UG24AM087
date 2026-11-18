"""Road-network geometry, spatial index and bounded shortest paths."""
import heapq
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree


class RoadNetwork:
    def __init__(self, node_xy, arcs, index_step=20.0):
        self.node_xy = node_xy
        self.arcs = arcs
        self.a = node_xy[arcs[:, 0]]                 # segment start (A, 2)
        self.b = node_xy[arcs[:, 1]]                 # segment end   (A, 2)
        d = self.b - self.a
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0  # 0 = north
        self.n_nodes = len(node_xy)
        self.out_deg = np.bincount(arcs[:, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(arcs[:, 1], minlength=self.n_nodes)
        # Adjacency: node -> list of (arc_id, to_node, length)
        self.out_arcs = [[] for _ in range(self.n_nodes)]
        for aid, (u, v) in enumerate(arcs):
            self.out_arcs[u].append((aid, int(v), float(self.length[aid])))
        self._build_index(index_step)
        self._sp_cache = {}

    def _build_index(self, step):
        """Index sample points every `step` m along each arc, so long arcs are found."""
        pts, owner = [], []
        for aid in range(len(self.arcs)):
            n = max(1, int(np.ceil(self.length[aid] / step)))
            f = (np.arange(n + 1) / n)[:, None]
            pts.append(self.a[aid] + f * (self.b[aid] - self.a[aid]))
            owner.append(np.full(n + 1, aid))
        self.index_pts = np.vstack(pts)
        self.index_owner = np.concatenate(owner)
        self.index_step = step
        self.tree = cKDTree(self.index_pts)

    def project_points(self, p, arc_ids):
        """Distance and fractional position of points p (n,2) onto arcs (n,)."""
        a, b = self.a[arc_ids], self.b[arc_ids]
        ab = b - a
        l2 = (ab ** 2).sum(axis=1)
        t = np.where(l2 > 0, ((p - a) * ab).sum(axis=1) / np.where(l2 > 0, l2, 1), 0.0)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * ab
        return np.hypot(*(p - proj).T), t

    def arcs_near(self, p, radius):
        """Arc ids whose geometry is within roughly `radius` of point p."""
        idx = self.tree.query_ball_point(p, radius + self.index_step / 2)
        return np.unique(self.index_owner[idx]) if idx else np.empty(0, dtype=np.int64)

    def shortest_from(self, src, cutoff):
        """Bounded Dijkstra from node `src`; returns {node: distance} within cutoff."""
        key = (src, cutoff)
        hit = self._sp_cache.get(key)
        if hit is not None:
            return hit
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for _, v, w in self.out_arcs[u]:
                nd = d + w
                if nd < dist.get(v, np.inf) and nd <= cutoff:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._sp_cache) > 20000:
            self._sp_cache.clear()
        self._sp_cache[key] = dist
        return dist

    def route_distance(self, arc_a, t_a, arc_b, t_b, cutoff):
        """Driving distance from position t_a on arc_a to t_b on arc_b (inf if unreachable).

        Small backward moves on the same arc are allowed (GPS jitter) but cost
        their absolute length.
        """
        if arc_a == arc_b:
            return abs(t_b - t_a) * self.length[arc_a]
        rest_a = (1.0 - t_a) * self.length[arc_a]
        head = int(self.arcs[arc_a, 1])
        tail = int(self.arcs[arc_b, 0])
        dist = self.shortest_from(head, cutoff)
        mid = dist.get(tail)
        if mid is None:
            return np.inf
        return rest_a + mid + t_b * self.length[arc_b]

    def connected(self, arc_a, arc_b):
        return arc_a == arc_b or self.arcs[arc_a, 1] == self.arcs[arc_b, 0]
