"""Step 2: exploratory data analysis on the raw files (train + student_test).

Produces tables in results/eda/ and figures in results/figures/, plus Folium maps in
results/maps/. Radius/K selection uses TRAIN only.
"""
import os, sys, glob, pickle, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.io import list_trips, load_trip
from pathsnap.viz import trip_map

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))
E, F, M = f"{ROOT}/results/eda", f"{ROOT}/results/figures", f"{ROOT}/results/maps"
for d in (E, F, M): os.makedirs(d, exist_ok=True)

inv = []
for split in ["train", "student_test"]:
    for tid in list_trips(os.path.join(DATA, split)):
        d = os.path.join(DATA, split, tid)
        tr = load_trip(os.path.join(DATA, split), tid)
        t = tr.track
        dts = np.diff(t[:, 2])
        n = len(tr.nodes)
        A = coo_matrix((np.ones(len(tr.arcs)), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(n, n))
        ncw, labw = connected_components(A, directed=True, connection="weak")
        ncs, labs = connected_components(A, directed=True, connection="strong")
        deg = np.bincount(tr.arcs.ravel(), minlength=n)
        _pairs = set(zip(tr.arcs[:, 0].tolist(), tr.arcs[:, 1].tolist()))
        twin_frac = float(np.mean([(b, a) in _pairs for a, b in tr.arcs.tolist()])) if len(tr.arcs) else 0.0
        route_ok = np.all(tr.arcs[tr.route[:-1], 1] == tr.arcs[tr.route[1:], 0])
        lat = np.radians(t[:, 1]); lon = np.radians(t[:, 0])
        step = 6371008.8 * np.hypot(np.diff(lat), np.diff(lon) * np.cos(lat[:-1]))
        inv.append({
            "split": split, "trip": tid,
            "files": len(os.listdir(d)),
            "track_bytes": os.path.getsize(f"{d}/{tid}.track"),
            "n_obs": len(t), "n_nan": int((~np.isfinite(t)).any(1).sum()),
            "duration_s": float(t[-1, 2] - t[0, 2]),
            "dt_1s_frac": float(np.mean(np.isclose(dts, 1))), "dt_max": float(dts.max()),
            "n_gaps_gt5s": int((dts > 5).sum()), "non_monotonic": int((dts <= 0).sum()),
            "max_step_m": float(step.max()), "path_len_km": float(step.sum() / 1000),
            "lon_min": t[:, 0].min(), "lon_max": t[:, 0].max(), "lat_min": t[:, 1].min(), "lat_max": t[:, 1].max(),
            "n_nodes": n, "n_arcs": len(tr.arcs), "isolated_nodes": int((deg == 0).sum()),
            "weak_components": int(ncw), "largest_weak_frac": float(np.bincount(labw).max() / n),
            "strong_components": int(ncs), "largest_strong_frac": float(np.bincount(labs).max() / n),
            "n_route_arcs": len(tr.route), "route_unique_arcs": len(set(tr.route.tolist())),
            "route_contiguous": bool(route_ok), "twin_arc_frac": twin_frac,
        })
inv = pd.DataFrame(inv)
inv.to_csv(f"{E}/inventory.csv", index=False)
summ = inv.groupby("split").agg(trips=("trip", "count"), obs=("n_obs", "sum"), obs_min=("n_obs", "min"),
                                obs_median=("n_obs", "median"), obs_max=("n_obs", "max"),
                                nan_rows=("n_nan", "sum"), gaps_gt5s=("n_gaps_gt5s", "sum"),
                                dt_1s_frac=("dt_1s_frac", "mean"), route_contiguous=("route_contiguous", "mean"),
                                nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
                                isolated_nodes=("isolated_nodes", "sum"))
summ.to_csv(f"{E}/summary_by_split.csv")
print(summ.T)

# ---- figures -----------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for s, c in [("train", "#2b6cb0"), ("student_test", "#dd6b20")]:
    ax[0].hist(np.log10(inv[inv.split == s].n_obs), bins=20, alpha=.6, label=s, color=c)
    ax[1].hist(inv[inv.split == s].path_len_km, bins=20, alpha=.6, label=s, color=c)
    ax[2].scatter(inv[inv.split == s].n_arcs, inv[inv.split == s].n_obs, label=s, color=c, s=15)
ax[0].set_xlabel("log10(# GPS observations per trip)"); ax[1].set_xlabel("GPS path length (km)")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("# road arcs in trip network"); ax[2].set_ylabel("# GPS obs")
for a in ax: a.legend()
fig.suptitle("Trip size distribution (train vs student_test)"); fig.tight_layout(); fig.savefig(f"{F}/eda_trip_sizes.png", dpi=130)

# sampling intervals
alldt = []
for split in ["train"]:
    for tid in list_trips(os.path.join(DATA, split)):
        t = load_trip(os.path.join(DATA, split), tid, with_route=False).track
        alldt.append(np.diff(t[:, 2]))
alldt = np.concatenate(alldt)
vals, cnt = np.unique(np.round(alldt), return_counts=True)
pd.DataFrame({"dt_s": vals, "count": cnt}).to_csv(f"{E}/sampling_intervals_train.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar([str(int(v)) if v < 10 else ">=10" for v in vals[:10]] if len(vals) <= 10 else
       [str(int(v)) for v in vals[:9]] + [">=10"],
       list(cnt[:9]) + ([cnt[9:].sum()] if len(vals) > 9 else []), color="#2b6cb0")
ax.set_yscale("log"); ax.set_xlabel("time between consecutive fixes (s)"); ax.set_ylabel("count (log)")
ax.set_title("Sampling interval distribution - TRAIN"); fig.tight_layout(); fig.savefig(f"{F}/eda_sampling_intervals.png", dpi=130)

# trajectory shapes (normalised) for 12 train trips
fig, axs = plt.subplots(3, 4, figsize=(14, 10))
for a, tid in zip(axs.ravel(), list_trips(os.path.join(DATA, "train"))[:12]):
    tr = load_trip(os.path.join(DATA, "train"), tid)
    nodes = tr.nodes; arcs = tr.arcs; t = tr.track
    pad = 0.003
    m = ((nodes[:, 0] > t[:, 0].min() - pad) & (nodes[:, 0] < t[:, 0].max() + pad) &
         (nodes[:, 1] > t[:, 1].min() - pad) & (nodes[:, 1] < t[:, 1].max() + pad))
    sel = arcs[m[arcs[:, 0]] & m[arcs[:, 1]]]
    for u, v in sel[:20000]:
        a.plot(nodes[[u, v], 0], nodes[[u, v], 1], color="#cbd5e0", lw=.4)
    r = tr.route
    for u, v in arcs[r]:
        a.plot(nodes[[u, v], 0], nodes[[u, v], 1], color="#38a169", lw=1.6)
    a.scatter(t[:, 0], t[:, 1], s=1.5, color="#2b6cb0")
    a.set_title(f"train {tid} ({len(t)} fixes)", fontsize=9); a.set_xticks([]); a.set_yticks([])
fig.suptitle("TRAIN: GPS fixes (blue), reference route (green), road network (grey)")
fig.tight_layout(); fig.savefig(f"{F}/eda_trajectories_grid.png", dpi=120)

# ---- labelling + candidate-radius analysis (TRAIN only) ---------------------
ld, ranks = [], []
for p in sorted(glob.glob(f"{ROOT}/cache/train/*.pkl")):
    P = pickle.load(open(p, "rb"))
    ld.append(P.label_dist)
ld = np.concatenate(ld)
pd.Series(ld).describe(percentiles=[.5, .9, .95, .99, .999]).to_csv(f"{E}/label_distance_train.csv")
# Anomalous trips: GPS systematically far from its own reference route (median offset > 50 m)
# -> the derived per-fix labels are unreliable, so these trips are excluded from TRAINING.
rep = pd.read_csv(f"{E}/preprocess_report.csv", dtype={"trip": str})
bad = rep[(rep.split == "train") & (rep.label_dist_median > 50)].trip.str.zfill(8).tolist()
json.dump({"rule": "train trips with median(fix -> reference arc distance) > 50 m", "excluded": bad},
          open(f"{E}/excluded_trips.json", "w"), indent=1)
print("excluded anomalous train trips:", bad)
ft = pd.read_parquet(f"{ROOT}/results/features_train.parquet", columns=["obs", "trip", "dist", "dist_rank", "y"])
ft = ft[~ft.trip.isin(bad)]
nobs = inv[(inv.split == "train") & (~inv.trip.isin(bad))].n_obs.sum()
pos = ft[ft.y == 1]
grid = []
for r in [20, 30, 40, 50, 60, 80]:
    for k in [4, 6, 8, 10, 12]:
        hit = ((pos.dist <= r) & (pos.dist_rank < k)).sum()
        rows = ((ft.dist <= r) & (ft.dist_rank < k)).sum()
        grid.append({"radius": r, "k": k, "miss_rate": 1 - hit / nobs, "rows": int(rows)})
grid = pd.DataFrame(grid); grid.to_csv(f"{E}/candidate_radius_grid_train.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(np.clip(ld, 0, 60), bins=60, color="#2b6cb0"); ax[0].set_xlabel("distance GPS fix -> reference arc (m)")
ax[0].set_ylabel("# fixes"); ax[0].set_title("GPS noise w.r.t. reference arc - TRAIN")
for k, g in grid.groupby("k"):
    ax[1].plot(g.radius, 100 * g.miss_rate, marker="o", label=f"K={k}")
ax[1].set_yscale("log"); ax[1].set_xlabel("candidate radius (m)"); ax[1].set_ylabel("candidate miss rate (%)")
ax[1].legend(); ax[1].set_title("Candidate miss rate vs radius/K - TRAIN")
fig.tight_layout(); fig.savefig(f"{F}/eda_label_distance_and_radius.png", dpi=130)

# ---- Folium maps ------------------------------------------------------------
for split, tid in [("train", "00000005"), ("train", "00000021"), ("student_test", "00000052")]:
    P = pickle.load(open(f"{ROOT}/cache/{split}/{tid}.pkl", "rb"))
    tr = load_trip(os.path.join(DATA, split), tid)
    m = trip_map(P, reference_route=tr.route, title=f"{split} {tid}: raw GPS, network, reference route")
    m.save(f"{M}/eda_{split}_{tid}.html")
print("EDA done")
