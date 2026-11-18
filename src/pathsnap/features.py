"""Candidate generation + feature engineering.

Every GPS observation t gets a candidate set C_t = up to K arcs whose segment lies within
RADIUS metres of the fix. Each (t, arc) pair becomes one row; the supervised target is
1 if the arc equals the reference arc for t (from labels.align_route) else 0.
The classifier therefore learns P(arc is correct | features), which is then normalised
within C_t and used as the HMM emission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import LocalProjection, Network, clean_track, heading_from_track, ang_diff
from .labels import align_route

DEFAULT_CFG = {"radius": 50.0, "k_max": 8, "buffer_m": 800.0, "window": 3}

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_softmax",
    "head_diff", "head_diff_long", "head_diff_undirected",
    "speed", "step_len", "dt",
    "arc_len", "proj_frac", "proj_overshoot",
    "n_cands", "win_dist_mean", "win_dist_max", "dist_prev", "dist_next",
    "out_deg_to", "in_deg_from", "has_twin",
]


class PreparedTrip:
    pass


def prepare_trip(trip, cfg=None, with_labels=True):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    P = PreparedTrip()
    P.trip_id = trip.trip_id
    track, P.clean_report = clean_track(trip.track)
    P.track = track
    lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])
    P.proj = LocalProjection(lon0, lat0)
    pxy = P.proj.fwd(track[:, 0], track[:, 1])
    P.pxy = pxy
    bbox = (pxy.min(0), pxy.max(0))
    P.net = Network(trip.nodes, trip.arcs, P.proj, bbox, cfg["buffer_m"])
    net = P.net
    T = len(pxy)

    # --- kinematics -------------------------------------------------------------
    head, disp = heading_from_track(pxy, 1)
    head_long, _ = heading_from_track(pxy, 5)
    dt = np.r_[1.0, np.diff(track[:, 2])]
    step = np.r_[0.0, np.hypot(*np.diff(pxy, axis=0).T)]
    i0 = np.clip(np.arange(T) - 1, 0, T - 1); i1 = np.clip(np.arange(T) + 1, 0, T - 1)
    speed = disp / np.maximum(track[i1, 2] - track[i0, 2], 1e-6)
    P.dt = dt

    # --- candidates -------------------------------------------------------------
    cands = net.candidates(pxy, cfg["radius"], cfg["k_max"])
    obs = np.concatenate([np.full(len(c[0]), t) for t, c in enumerate(cands)]) if T else np.zeros(0, int)
    li = np.concatenate([c[0] for c in cands]).astype(int) if T else np.zeros(0, int)
    dist = np.concatenate([c[1] for c in cands]) if T else np.zeros(0)
    ncand = np.array([len(c[0]) for c in cands])
    rank = np.concatenate([np.arange(len(c[0])) for c in cands]) if T else np.zeros(0, int)
    mind = np.array([c[1][0] if len(c[1]) else np.nan for c in cands])

    df = pd.DataFrame({"obs": obs, "li": li})
    df["arc"] = net.arc_ids[li]
    df["dist"] = dist
    df["dist_rank"] = rank
    df["dist_minus_min"] = dist - mind[obs]
    s = np.exp(-0.5 * (dist / 10.0) ** 2)
    ssum = np.bincount(obs, weights=s, minlength=T)
    df["dist_softmax"] = s / np.maximum(ssum[obs], 1e-12)
    df["head_diff"] = ang_diff(head[obs], net.bearing[li])
    df["head_diff_long"] = ang_diff(head_long[obs], net.bearing[li])
    df["head_diff_undirected"] = np.minimum(df["head_diff"], 180 - df["head_diff"])
    df["speed"] = speed[obs]
    df["step_len"] = step[obs]
    df["dt"] = dt[obs]
    df["arc_len"] = net.length[li]
    _, frac, raw = net.seg_project(pxy[obs], li)
    df["proj_frac"] = frac
    df["proj_overshoot"] = np.maximum(raw - 1, 0) + np.maximum(-raw, 0)
    df["n_cands"] = ncand[obs]
    w = cfg["window"]
    wd = []
    for k in range(-w, w + 1):
        o2 = np.clip(obs + k, 0, T - 1)
        d2, _, _ = net.seg_project(pxy[o2], li)
        wd.append(d2)
        if k == -1: df["dist_prev"] = d2
        if k == 1: df["dist_next"] = d2
    wd = np.vstack(wd)
    df["win_dist_mean"] = wd.mean(0)
    df["win_dist_max"] = wd.max(0)
    df["out_deg_to"] = net.out_deg[net.v[li]]
    df["in_deg_from"] = net.in_deg[net.u[li]]
    df["has_twin"] = net.has_twin[li]
    df["trip"] = trip.trip_id

    P.n_obs = T
    P.ncand = ncand
    if with_labels and trip.route is not None and len(trip.route):
        pos, lab, ldist = align_route(pxy, net.nodes_xy, trip.arcs, trip.route)
        P.label_arc = lab
        P.label_dist = ldist
        df["y"] = (df["arc"].values == lab[obs]).astype(np.int8)
        hit = np.zeros(T, bool)
        hit[obs[df["y"].values == 1]] = True
        P.cand_hit = hit
    P.df = df
    return P


FINAL_CFG = {"radius": 50.0, "k_max": 8}   # chosen on TRAIN (results/eda/candidate_radius_grid_train.csv)


def filter_candidates(df: pd.DataFrame, radius: float, k_max: int) -> pd.DataFrame:
    """Restrict a wide candidate table to (radius, K) and recompute set-dependent features
    so the result is identical to running prepare_trip with that configuration."""
    d = df[(df.dist <= radius) & (df.dist_rank < k_max)].copy()
    g = d.groupby(["trip", "obs"])
    d["n_cands"] = g.dist.transform("size").astype(int)
    s = np.exp(-0.5 * (d.dist.values / 10.0) ** 2)
    d["_s"] = s
    d["dist_softmax"] = s / d.groupby(["trip", "obs"])._s.transform("sum").values
    return d.drop(columns="_s").reset_index(drop=True)


def attach_nodes(df: pd.DataFrame, net) -> pd.DataFrame:
    df["u"] = net.u[df.li.values]
    df["v"] = net.v[df.li.values]
    return df
