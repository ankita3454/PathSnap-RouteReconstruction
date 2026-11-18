"""Step 6: HMM/Viterbi decoding.

1. Tune decoder parameters (beta, w_e) on OUT-OF-FOLD TRAIN probabilities of the tuned model
   (20 random train trips, grid search) - student/evaluator data not used for this.
2. Evaluate on STUDENT_TEST: classifier-only argmax vs classifier+HMM, plus a pure
   Gaussian-emission HMM baseline, plus HMM on top of every one of the 7 models.
3. Folium maps for every student-test trip.
"""
import os, sys, json, time, pickle, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.hmm import viterbi, reconstruct_route, RouteGraph, DEFAULT_HMM
from pathsnap.metrics import continuity, route_level_metrics
from pathsnap.io import load_route_only
from pathsnap.models import MODEL_INFO, SEED
from pathsnap.viz import trip_map

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))
OUT = f"{ROOT}/results/hmm"; os.makedirs(OUT, exist_ok=True)
MAPS = f"{ROOT}/results/maps"; FIG = f"{ROOT}/results/figures"

_P, _G = {}, {}


def load(split, tid):
    k = (split, tid)
    if k not in _P:
        _P[k] = pickle.load(open(f"{ROOT}/cache/{split}/{tid}.pkl", "rb")); _G[k] = RouteGraph(_P[k].net)
    return _P[k], _G[k]


def evaluate(split, scores, params, trips=None, want_routes=False):
    rows, routes = [], {}
    for tid, g in scores.groupby("trip"):
        if trips is not None and tid not in trips: continue
        P, G = load(split, tid)
        net = P.net
        top = g.loc[g.groupby("obs").p.idxmax()]
        clf = np.full(P.n_obs, -1); clf[top.obs.values] = top.arc.values
        t0 = time.time(); hmm, diag = viterbi(P, g, params, G); el = time.time() - t0
        uv = lambda arcs: (np.array([net.u[net.local[a]] for a in arcs]), np.array([net.v[net.local[a]] for a in arcs]))
        r = {"trip": tid, "n_obs": P.n_obs, "decode_s": el, **diag}
        for name, pred in [("clf", clf), ("hmm", hmm)]:
            a = pred[pred >= 0]; u, v = uv(a)
            r[f"{name}_top1"] = float(np.mean(pred == P.label_arc))
            r[f"{name}_continuity"] = continuity(u, v, a)
        if want_routes:
            ref = load_route_only(os.path.join(DATA, split), tid)
            L = {int(a): float(l) for a, l in zip(net.arc_ids, net.length)}
            for name, pred in [("clf", clf), ("hmm", hmm)]:
                route, br = reconstruct_route(P, pred, G)
                r.update({f"{name}_{k}": v for k, v in route_level_metrics(route, ref, L).items()})
                r[f"{name}_route_breaks"] = br
                routes[(tid, name)] = route
            routes[(tid, "clf_points")] = clf
        rows.append(r)
    df = pd.DataFrame(rows)
    return df, routes


def pooled(df):
    w = df.n_obs / df.n_obs.sum()
    return {c: float((df[c] * w).sum()) if c.endswith(("top1", "continuity")) else float(df[c].mean())
            for c in df.columns if c not in ("trip",) and df[c].dtype != object}


# ---- 1. tune decoder on OOF train --------------------------------------------
oof = pd.read_parquet(f"{ROOT}/results/bayes/oof_train_tuned.parquet")
rng = np.random.RandomState(SEED)
tune_trips = sorted(rng.choice(sorted(oof.trip.unique()), 20, replace=False).tolist())
grid = []
for beta, we in itertools.product([3.0, 8.0, 20.0], [0.5, 1.0, 2.0]):
    prm = {"beta": beta, "w_e": we}
    d, _ = evaluate("train", oof, prm, set(tune_trips))
    s = pooled(d)
    grid.append({"beta": beta, "w_e": we, "hmm_top1": s["hmm_top1"], "clf_top1": s["clf_top1"],
                 "hmm_continuity": s["hmm_continuity"]})
    print(grid[-1], flush=True)
grid = pd.DataFrame(grid); grid.to_csv(f"{OUT}/decoder_grid_train_oof.csv", index=False)
b = grid.sort_values("hmm_top1", ascending=False).iloc[0]
best = {**DEFAULT_HMM, "beta": float(b.beta), "w_e": float(b.w_e)}
json.dump({"params": best, "tuned_on": "20 TRAIN trips, out-of-fold tuned-LightGBM probabilities",
           "trips": tune_trips}, open(f"{ROOT}/artifacts/hmm_config.json", "w"), indent=1)

# ---- 2. student test -----------------------------------------------------------
te = pd.read_parquet(f"{ROOT}/results/model_student_test.parquet")
tuned = pd.read_parquet(f"{ROOT}/results/bayes/test_scores_tuned.parquet")
d, routes = evaluate("student_test", tuned, best, want_routes=True)
d.to_csv(f"{OUT}/student_test_per_trip_tuned.csv", index=False)
summary = {"tuned_LightGBM": pooled(d)}
gauss = tuned.copy()
dg, _ = evaluate("student_test", gauss, {**best, "mode": "gaussian"}, want_routes=True)
dg.to_csv(f"{OUT}/student_test_per_trip_gaussian_hmm.csv", index=False)
summary["Gaussian_HMM_no_classifier"] = pooled(dg)
for name in MODEL_INFO:
    p = f"{ROOT}/results/models/test_scores_{name}.parquet"
    if not os.path.exists(p): continue
    s = pd.read_parquet(p); s["li"] = te.li.values; s["dist"] = te.dist.values
    dm, _ = evaluate("student_test", s, best)
    summary[name] = pooled(dm)
    print(name, summary[name]["clf_top1"], summary[name]["hmm_top1"], flush=True)
pd.DataFrame(summary).T.to_csv(f"{OUT}/student_test_summary.csv")

# ---- 3. figures + maps ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(d)); w = .38
ax.bar(x - w / 2, 100 * d.clf_top1, w, label="classifier only", color="#d69e2e")
ax.bar(x + w / 2, 100 * d.hmm_top1, w, label="classifier + HMM/Viterbi", color="#e53e3e")
ax.set_xticks(x); ax.set_xticklabels(d.trip, rotation=45); ax.set_ylim(min(80, 100 * d.clf_top1.min() - 3), 100)
ax.set_ylabel("observation Top-1 accuracy (%)"); ax.set_title("STUDENT_TEST: classifier-only vs HMM decoding (tuned LightGBM)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/hmm_vs_clf_per_trip.png", dpi=130)
for tid in d.trip:
    P, _ = load("student_test", tid)
    ref = load_route_only(os.path.join(DATA, "student_test"), tid)
    m = trip_map(P, reference_route=ref, predicted_arcs=routes[(tid, "hmm")],
                 clf_arcs=[a for a in routes[(tid, "clf_points")] if a >= 0],
                 title=f"student_test {tid}")
    m.save(f"{MAPS}/student_test_{tid}.html")
print(json.dumps(summary["tuned_LightGBM"], indent=1))
