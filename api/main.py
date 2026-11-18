"""PathSnap FastAPI backend (Laptop 1).

Run:
    PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            liveness + loaded artifacts
    GET  /trips             trip ids available in PATHSNAP_EVAL_DIR (no labels)
    POST /predict           {"trip_id": "..."} -> predicted route, GPS, metrics, timings
Every request is logged to the console of this machine (required for the demo).
"""
import logging, os, sys, time
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.pipeline import PathSnapPipeline  # noqa: E402

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "student_test"))
ARTIFACTS = os.environ.get("PATHSNAP_ARTIFACTS", os.path.join(ROOT, "artifacts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipe = PathSnapPipeline(ARTIFACTS)
log.info("Loaded model=%s  hmm=%s  eval_dir=%s", pipe.model_cfg["model"], pipe.hmm_cfg, EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": pipe.model_cfg["model"], "eval_dir": EVAL_DIR,
            "n_trips": len(list_trips(EVAL_DIR))}


@app.get("/trips")
def trips(request: Request):
    ids = list_trips(EVAL_DIR)
    log.info("GET /trips from %s -> %d trips", request.client.host if request.client else "?", len(ids))
    return {"trips": ids}


def _line(P, arc_id):
    li = P.net.local[int(arc_id)]
    u, v = P.net.u[li], P.net.v[li]
    return [[float(P.net.nodes_ll[u, 1]), float(P.net.nodes_ll[u, 0])],
            [float(P.net.nodes_ll[v, 1]), float(P.net.nodes_ll[v, 0])]]


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    if req.trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    client = request.client.host if request.client else "?"
    log.info("POST /predict trip=%s from %s - running pipeline", req.trip_id, client)
    t0 = time.time()
    out = pipe.predict(EVAL_DIR, req.trip_id)
    P = out["P"]
    metrics = pipe.evaluate_prediction(EVAL_DIR, req.trip_id, out)  # server-side only
    total = time.time() - t0
    cont_h = pipe.continuity_of(P, out["hmm_arcs"]); cont_c = pipe.continuity_of(P, out["clf_arcs"])
    step = max(1, P.n_obs // 3000)
    resp = {
        "trip_id": req.trip_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_observations": int(P.n_obs), "n_candidates": out["n_candidates"],
        "gps": [[float(a), float(b)] for b, a in P.track[::step, :2]],
        "predicted_route": [_line(P, a) for a in out["route"]],
        "n_route_arcs": len(out["route"]), "route_breaks": out["route_breaks"],
        "route_length_km": float(sum(P.net.length[P.net.local[a]] for a in out["route"]) / 1000),
        "route_continuity_hmm": cont_h, "route_continuity_classifier_only": cont_c,
        "metrics": metrics, "timings_s": {**out["timings"], "total_s": total},
    }
    m = metrics or {}
    log.info("trip=%s obs=%d cand=%d | top1_hmm=%.4f top1_clf=%.4f route_F1=%.4f continuity=%.4f | %.2fs",
             req.trip_id, P.n_obs, out["n_candidates"], m.get("obs_top1_hmm", float("nan")),
             m.get("obs_top1_classifier_only", float("nan")), m.get("route_f1", float("nan")), cont_h, total)
    return resp
