"""Step 1: clean every trip, crop the road network, derive per-fix reference arcs,
generate a wide candidate set (radius 80 m, K=12) and engineer features.

Outputs
  cache/<split>/<trip>.pkl         prepared trip (network, projected track, labels)
  results/features_<split>.parquet candidate-level feature table
  results/eda/preprocess_report.csv cleaning + labelling summary per trip
The final radius / K are chosen from the TRAIN partition in 02_eda.py and applied
afterwards as a filter, so this expensive step runs once.
"""
import os, sys, time, pickle
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.io import list_trips, load_trip
from pathsnap.features import prepare_trip

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))
WIDE = {"radius": 80.0, "k_max": 12}

rows = []
for split in ["train", "student_test"]:
    os.makedirs(f"{ROOT}/cache/{split}", exist_ok=True)
    dfs = []
    for tid in list_trips(os.path.join(DATA, split)):
        t0 = time.time()
        trip = load_trip(os.path.join(DATA, split), tid)
        P = prepare_trip(trip, WIDE)
        el = time.time() - t0
        with open(f"{ROOT}/cache/{split}/{tid}.pkl", "wb") as f:
            pickle.dump(P, f, protocol=4)
        dfs.append(P.df)
        rows.append({"split": split, "trip": tid, **P.clean_report,
                     "n_route_arcs": len(trip.route), "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs),
                     "n_arcs_cropped": len(P.net.arc_ids), "n_cand_rows": len(P.df),
                     "label_dist_median": float(np.median(P.label_dist)),
                     "label_dist_p95": float(np.percentile(P.label_dist, 95)),
                     "cand_hit_rate_wide": float(P.cand_hit.mean()), "prep_sec": el})
        print(split, tid, P.n_obs, len(P.df), f"{el:.1f}s", flush=True)
    pd.concat(dfs, ignore_index=True).to_parquet(f"{ROOT}/results/features_{split}.parquet")
pd.DataFrame(rows).to_csv(f"{ROOT}/results/eda/preprocess_report.csv", index=False)
