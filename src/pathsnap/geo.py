"""Geometry helpers: projection to a local metric plane, network cropping,
segment distances and a KD-tree based candidate index.

Projection choice: each trip is small (a few km to a few tens of km), so we use a
local equirectangular projection centred on the trip's GPS centroid:
    x = R * cos(lat0) * (lon - lon0),   y = R * (lat - lat0)   (radians, R = 6371008.8 m)
Within ~20 km of the centre the scale error is well below 0.1 %, i.e. far smaller than
GPS noise, and it keeps every distance computation a cheap Euclidean operation.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6371008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = R_EARTH * np.cos(np.radians(lat0)) * np.pi / 180.0
        self.ky = R_EARTH * np.pi / 180.0

    def fwd(self, lon, lat):
        return np.column_stack([(np.asarray(lon) - self.lon0) * self.kx,
                                (np.asarray(lat) - self.lat0) * self.ky])

    def inv(self, xy):
        xy = np.asarray(xy)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def clean_track(track: np.ndarray):
    """Basic GPS cleaning. Returns (clean_track, report dict)."""
    rep = {"n_raw": int(len(track))}
    ok = np.all(np.isfinite(track), axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    rep["n_invalid"] = int((~ok).sum())
    t = track[ok]
    order = np.argsort(t[:, 2], kind="stable")
    rep["n_out_of_order"] = int((order != np.arange(len(order))).sum())
    t = t[order]
    keep = np.r_[True, np.diff(t[:, 2]) > 0]
    rep["n_dup_time"] = int((~keep).sum())
    t = t[keep]
    rep["n_clean"] = int(len(t))
    return t, rep


class Network:
    """Road network cropped to the trip's area, in projected metres.

    Arc ids are kept as the ORIGINAL ids from the .arcs file.
    """

    def __init__(self, nodes_ll, arcs, proj: LocalProjection, bbox_xy=None, buffer_m=1000.0):
        nxy = proj.fwd(nodes_ll[:, 0], nodes_ll[:, 1])
        arc_ids = np.arange(len(arcs))
        if bbox_xy is not None and len(arcs):
            (x0, y0), (x1, y1) = bbox_xy
            x0, y0, x1, y1 = x0 - buffer_m, y0 - buffer_m, x1 + buffer_m, y1 + buffer_m
            inside = (nxy[:, 0] >= x0) & (nxy[:, 0] <= x1) & (nxy[:, 1] >= y0) & (nxy[:, 1] <= y1)
            keep = inside[arcs[:, 0]] | inside[arcs[:, 1]]
            arc_ids = arc_ids[keep]
        self.arc_ids = arc_ids                       # original ids of kept arcs
        self.n_total_arcs = len(arcs)
        self.u = arcs[arc_ids, 0]
        self.v = arcs[arc_ids, 1]
        self.nodes_xy = nxy
        self.nodes_ll = nodes_ll
        self.p0 = nxy[self.u]
        self.p1 = nxy[self.v]
        d = self.p1 - self.p0
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0   # 0 = north
        self.local = {a: i for i, a in enumerate(arc_ids)}               # orig id -> local idx
        n = len(nodes_ll)
        self.out_deg = np.bincount(self.u, minlength=n)
        self.in_deg = np.bincount(self.v, minlength=n)
        pairs = set(zip(self.u.tolist(), self.v.tolist()))
        self.has_twin = np.array([(b, a) in pairs for a, b in zip(self.u, self.v)], dtype=np.int8)
        self._build_index()

    def _build_index(self, step=10.0):
        pts, owner = [], []
        for i in range(len(self.u)):
            k = max(1, int(np.ceil(self.length[i] / step)))
            f = np.linspace(0, 1, k + 1)
            pts.append(self.p0[i] + f[:, None] * (self.p1[i] - self.p0[i]))
            owner.append(np.full(k + 1, i))
        if pts:
            self.sample_xy = np.vstack(pts)
            self.sample_owner = np.concatenate(owner)
            self.tree = cKDTree(self.sample_xy)
        else:
            self.sample_xy = np.zeros((0, 2)); self.sample_owner = np.zeros(0, int); self.tree = None
        self._step = step

    def seg_project(self, pxy, li):
        """Distance from points pxy (M,2) to local arcs li (M,). Returns dist, frac(clipped), raw frac."""
        a = self.p0[li]; b = self.p1[li]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        raw = ((pxy - a) * ab).sum(1) / L2
        f = np.clip(raw, 0, 1)
        q = a + f[:, None] * ab
        return np.hypot(*(pxy - q).T), f, raw

    def candidates(self, pxy, radius, k_max):
        """For each point return list of (local arc idx array, dist array), sorted by distance."""
        out = []
        if self.tree is None:
            return [(np.zeros(0, int), np.zeros(0))] * len(pxy)
        hits = self.tree.query_ball_point(pxy, r=radius + self._step)
        for i, h in enumerate(hits):
            if not h:
                out.append((np.zeros(0, int), np.zeros(0))); continue
            li = np.unique(self.sample_owner[h])
            d, _, _ = self.seg_project(np.repeat(pxy[i:i + 1], len(li), 0), li)
            m = d <= radius
            li, d = li[m], d[m]
            o = np.argsort(d)[:k_max]
            out.append((li[o], d[o]))
        return out


def heading_from_track(xy, half_window=1):
    """Heading (deg, 0=north) estimated from the displacement between t-w and t+w."""
    n = len(xy)
    i0 = np.clip(np.arange(n) - half_window, 0, n - 1)
    i1 = np.clip(np.arange(n) + half_window, 0, n - 1)
    d = xy[i1] - xy[i0]
    return np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0, np.hypot(d[:, 0], d[:, 1])


def ang_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
