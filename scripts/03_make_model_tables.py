"""Step 3: apply the TRAIN-selected candidate configuration (radius 50 m, K=8),
drop anomalous training trips, attach arc end-nodes and save model tables."""
import os, sys, json, pickle
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.features import filter_candidates, attach_nodes, FINAL_CFG

ROOT = os.path.join(os.path.dirname(__file__), "..")
bad = json.load(open(f"{ROOT}/results/eda/excluded_trips.json"))["excluded"]
for split in ["train", "student_test"]:
    df = pd.read_parquet(f"{ROOT}/results/features_{split}.parquet")
    if split == "train":
        df = df[~df.trip.isin(bad)]
    df = filter_candidates(df, FINAL_CFG["radius"], FINAL_CFG["k_max"])
    parts, nobs = [], {}
    for t, g in df.groupby("trip"):
        P = pickle.load(open(f"{ROOT}/cache/{split}/{t}.pkl", "rb"))
        parts.append(attach_nodes(g.copy(), P.net)); nobs[t] = P.n_obs
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(f"{ROOT}/results/model_{split}.parquet")
    json.dump(nobs, open(f"{ROOT}/results/nobs_{split}.json", "w"))
    print(split, len(out), "rows", out.trip.nunique(), "trips", "pos rate", round(out.y.mean(), 4))
