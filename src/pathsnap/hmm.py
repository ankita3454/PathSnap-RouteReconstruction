"""HMM + Viterbi route reconstruction.

Hidden state at observation t : one candidate (arc a, offset s along a) from C_t.
Emission   log b_t(j)  = w_e * log( p_clf(j) / sum_{k in C_t} p_clf(k) + eps )
           (or a Gaussian distance emission when mode="gaussian", used as a baseline)
Transition log a(i->j) = -| d_route(i,j) - d_gc(t-1,t) | / beta_t
           d_route = network driving distance from (a_i, s_i) to (a_j, s_j)
           d_gc    = straight-line distance between the two GPS fixes
           beta_t  = beta * sqrt(max(dt,1))   (looser after GPS gaps)
Infeasible transitions (no path within the search limit, implausible speed, or large
backward move on the same arc) receive a finite penalty instead of -inf, so the
Viterbi chain never breaks; observations with an empty candidate set are skipped and
the next transition spans the gap.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULT_HMM = {"beta": 8.0, "w_e": 1.0, "penalty": 30.0, "limit_m": 3000.0, "vmax": 60.0,
               "back_tol": 8.0, "sigma": 8.0, "mode": "classifier"}


class RouteGraph:
    def __init__(self, net):
        self.net = net
        n = len(net.nodes_xy)
        # keep shortest parallel arc
        self.G = csr_matrix((net.length + 1e-3, (net.u, net.v)), shape=(n, n))
        self._cache = {}
        self.pair_arc = {}
        for i, (a, b) in enumerate(zip(net.u, net.v)):
            j = self.pair_arc.get((a, b))
            if j is None or net.length[i] < net.length[j]:
                self.pair_arc[(a, b)] = i

    def sp_from(self, src, limit):
        r = self._cache.get(src)
        if r is None:
            d = dijkstra(self.G, indices=src, limit=limit)
            idx = np.nonzero(np.isfinite(d))[0]
            r = (idx, d[idx]); self._cache[src] = r
        return r

    def sp_lookup(self, src, targets, limit):
        idx, d = self.sp_from(src, limit)
        pos = np.searchsorted(idx, targets)
        pos = np.clip(pos, 0, max(len(idx) - 1, 0))
        out = np.full(len(targets), np.inf)
        if len(idx):
            ok = idx[pos] == targets
            out[ok] = d[pos[ok]]
        return out

    def path_arcs(self, src_node, dst_node, limit=5000.0):
        """Local arc indices of the shortest path src_node -> dst_node ([] if none)."""
        if src_node == dst_node:
            return []
        d, pred = dijkstra(self.G, indices=src_node, limit=limit, return_predecessors=True)
        if not np.isfinite(d[dst_node]):
            return None
        nodes = [dst_node]
        while nodes[-1] != src_node:
            nodes.append(pred[nodes[-1]])
            if nodes[-1] < 0:
                return None
        nodes = nodes[::-1]
        return [self.pair_arc[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]


def viterbi(P, cand, params=None, graph=None):
    """cand: DataFrame for ONE trip with columns obs, li, arc, p, dist.
    Returns array (T,) of predicted ORIGINAL arc ids (-1 where no candidate) and diagnostics."""
    prm = {**DEFAULT_HMM, **(params or {})}
    net = P.net
    G = graph or RouteGraph(net)
    T = P.n_obs
    pxy, tt = P.pxy, P.track[:, 2]
    cand = cand.sort_values(["obs", "dist", "arc"], kind="mergesort")  # deterministic tie-break (twin arcs have equal dist)
    groups = {o: g for o, g in cand.groupby("obs")}
    obs_list = [o for o in range(T) if o in groups]
    pred = np.full(T, -1, dtype=np.int64)
    if not obs_list:
        return pred, {"n_infeasible": 0, "n_steps": 0}

    def emis(g):
        if prm["mode"] == "gaussian":
            return -0.5 * (g.dist.values / prm["sigma"]) ** 2
        p = g.p.values / max(g.p.values.sum(), 1e-12)
        return prm["w_e"] * np.log(p + 1e-6)

    def offs(o, li):
        _, f, _ = net.seg_project(np.repeat(pxy[o:o + 1], len(li), 0), li)
        return f * net.length[li]

    states, backs = [], []
    g0 = groups[obs_list[0]]
    li0 = g0.li.values
    delta = emis(g0)
    states.append((obs_list[0], li0, offs(obs_list[0], li0)))
    n_inf = 0
    for o_prev, o in zip(obs_list[:-1], obs_list[1:]):
        g = groups[o]
        li = g.li.values; s = offs(o, li)
        _, li_p, s_p = states[-1]
        dt = max(tt[o] - tt[o_prev], 1.0)
        gc = float(np.hypot(*(pxy[o] - pxy[o_prev])))
        beta = prm["beta"] * np.sqrt(dt)
        maxd = prm["vmax"] * dt + 50.0
        D = np.empty((len(li_p), len(li)))
        for i, (a, sa) in enumerate(zip(li_p, s_p)):
            sp = G.sp_lookup(net.v[a], net.u[li], prm["limit_m"])
            d = (net.length[a] - sa) + sp + s
            same = li == a
            fwd = same & (s >= sa - prm["back_tol"])
            d[fwd] = np.abs(s[fwd] - sa)
            D[i] = d
        bad = ~np.isfinite(D) | (D > maxd)
        n_inf += int(bad.all())
        logA = np.where(bad, -prm["penalty"], -np.abs(np.where(bad, 0, D) - gc) / beta)
        score = delta[:, None] + logA
        bp = score.argmax(0)
        delta = score[bp, np.arange(len(li))] + emis(g)
        backs.append(bp)
        states.append((o, li, s))
    # backtrack
    j = int(np.argmax(delta))
    for k in range(len(states) - 1, -1, -1):
        o, li, _ = states[k]
        pred[o] = net.arc_ids[li[j]]
        if k > 0:
            j = int(backs[k - 1][j])
    return pred, {"n_infeasible_steps": n_inf, "n_steps": len(obs_list) - 1}


def reconstruct_route(P, pred_arcs, graph=None):
    """Collapse per-observation arcs into a connected arc sequence (original ids),
    filling holes with shortest paths. Returns (route arcs, n_unfilled_breaks)."""
    net = P.net
    G = graph or RouteGraph(net)
    seq = [a for a in pred_arcs if a >= 0]
    dedup = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    route, breaks = [], 0
    for a in dedup:
        la = net.local[int(a)]
        if route:
            lp = net.local[route[-1]]
            if net.v[lp] != net.u[la]:
                path = G.path_arcs(net.v[lp], net.u[la])
                if path is None:
                    breaks += 1
                else:
                    route.extend(int(net.arc_ids[x]) for x in path)
        route.append(int(a))
    return route, breaks
