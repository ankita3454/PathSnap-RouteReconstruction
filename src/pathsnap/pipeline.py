"""End-to-end inference used by the API. The prediction path never reads the .route file;
evaluation against the reference route happens afterwards in `evaluate_prediction`."""
from __future__ import annotations

import json, os, time

import joblib
import numpy as np

from .features import prepare_trip, filter_candidates, FEATURES
from .hmm import viterbi, reconstruct_route, RouteGraph
from .io import load_trip, load_route_only
from .labels import align_route
from .metrics import continuity, route_level_metrics


class PathSnapPipeline:
    def __init__(self, artifacts_dir: str):
        self.model = joblib.load(os.path.join(artifacts_dir, "final_model.joblib"))
        self.model_cfg = json.load(open(os.path.join(artifacts_dir, "model_config.json")))
        self.hmm_cfg = json.load(open(os.path.join(artifacts_dir, "hmm_config.json")))["params"]

    def predict(self, split_dir: str, trip_id: str) -> dict:
        timings = {}
        t0 = time.time()
        trip = load_trip(split_dir, trip_id, with_route=False)          # <- no ground truth
        # same protocol as training: wide candidate pass (80 m, K=12) then the TRAIN-selected filter,
        # so features (incl. tie order between equidistant twin arcs) match the training data exactly
        P = prepare_trip(trip, {"radius": 80.0, "k_max": 12}, with_labels=False)
        c = self.model_cfg["candidates"]
        P.df = filter_candidates(P.df, c["radius"], c["k_max"])
        timings["preprocess_s"] = time.time() - t0
        df = P.df.copy()
        t0 = time.time()
        df["p"] = self.model.predict_proba(df[FEATURES].values.astype(np.float32))[:, 1] if len(df) else []
        timings["classifier_s"] = time.time() - t0
        t0 = time.time()
        G = RouteGraph(P.net)
        hmm_arcs, diag = viterbi(P, df, self.hmm_cfg, G)
        route, breaks = reconstruct_route(P, hmm_arcs, G)
        timings["hmm_s"] = time.time() - t0
        clf = np.full(P.n_obs, -1)
        if len(df):
            top = df.loc[df.groupby("obs").p.idxmax()]
            clf[top.obs.values] = top.arc.values
        return {"P": P, "graph": G, "hmm_arcs": hmm_arcs, "clf_arcs": clf, "route": route,
                "route_breaks": breaks, "diag": diag, "timings": timings, "n_candidates": int(len(df))}

    @staticmethod
    def evaluate_prediction(split_dir: str, trip_id: str, out: dict) -> dict | None:
        """Server-side scoring against the reference route (never sent to the model/UI)."""
        try:
            ref = load_route_only(split_dir, trip_id)
        except Exception:
            return None
        if len(ref) == 0:
            return None
        P = out["P"]; net = P.net
        trip = load_trip(split_dir, trip_id, with_route=False)
        _, lab, _ = align_route(P.pxy, net.nodes_xy, trip.arcs, ref)
        L = {int(a): float(l) for a, l in zip(net.arc_ids, net.length)}
        res = {"obs_top1_hmm": float(np.mean(out["hmm_arcs"] == lab)),
               "obs_top1_classifier_only": float(np.mean(out["clf_arcs"] == lab))}
        res.update(route_level_metrics(out["route"], ref, L))
        return res

    @staticmethod
    def continuity_of(P, arcs):
        a = arcs[arcs >= 0]
        u = np.array([P.net.u[P.net.local[x]] for x in a]); v = np.array([P.net.v[P.net.local[x]] for x in a])
        return continuity(u, v, a)
