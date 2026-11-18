"""Derive a per-observation reference arc from the trip's .route file.

The .route file is only a chronological list of traversed arc ids; it does NOT say
which GPS fix belongs to which arc. We therefore align the GPS sequence to the route
sequence with a monotonic dynamic programme:

    cost[t, r] = dist(p_t, arc_r) + min_{r' <= r} cost[t-1, r']

i.e. every fix is assigned to one route position, positions never go backwards in
time, and the total point-to-segment distance is minimised. This respects the order
of the route (important when a route passes the same street twice, or where an
arc and its reverse twin overlap) instead of greedily snapping to the nearest arc.
The labels are a derived reference, not independently verified ground truth.
"""
from __future__ import annotations

import numpy as np


def _route_dists(pxy, nodes_xy, arcs, route, chunk=2000):
    a = nodes_xy[arcs[route, 0]]; b = nodes_xy[arcs[route, 1]]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    D = np.empty((len(pxy), len(route)), dtype=np.float32)
    for s in range(0, len(pxy), chunk):
        p = pxy[s:s + chunk, None, :]
        f = np.clip(((p - a) * ab).sum(2) / L2, 0, 1)
        q = a + f[..., None] * ab
        D[s:s + chunk] = np.hypot(*(p - q).transpose(2, 0, 1))
    return D


def align_route(pxy, nodes_xy, arcs, route):
    """Returns (route_pos per point, arc id per point, distance to that arc)."""
    T, R = len(pxy), len(route)
    D = _route_dists(pxy, nodes_xy, arcs, route)
    back = np.empty((T, R), dtype=np.int32)
    prev = D[0].astype(np.float64)
    back[0] = np.arange(R)
    for t in range(1, T):
        # prefix minimum (and argmin) of prev over r' <= r
        pm = np.minimum.accumulate(prev)
        is_new = np.r_[True, prev[1:] < pm[:-1]]
        idx = np.where(is_new, np.arange(R), 0)
        arg = np.maximum.accumulate(idx)
        back[t] = arg
        prev = pm + D[t]
    pos = np.empty(T, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for t in range(T - 1, 0, -1):
        pos[t - 1] = back[t, pos[t]]
    return pos, route[pos], D[np.arange(T), pos]
