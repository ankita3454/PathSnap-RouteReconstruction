"""Evaluation metrics shared by model experiments, HMM evaluation and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, confusion_matrix


def normalise_within_obs(df: pd.DataFrame, col="p") -> np.ndarray:
    s = df.groupby(["trip", "obs"])[col].transform("sum").values
    return df[col].values / np.maximum(s, 1e-12)


def argmax_per_obs(df: pd.DataFrame, col="p") -> pd.DataFrame:
    idx = df.groupby(["trip", "obs"])[col].idxmax()
    return df.loc[idx, ["trip", "obs", "arc", "u", "v", "y"]].reset_index(drop=True)


def row_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision_pos": precision_score(y, yhat, zero_division=0),
        "recall_pos": recall_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def continuity(arcs_u, arcs_v, arcs):
    """Share of consecutive predicted-arc pairs that are physically consistent:
    same arc, or prev.to_node == next.from_node."""
    if len(arcs) < 2:
        return 1.0
    same = arcs[1:] == arcs[:-1]
    conn = arcs_v[:-1] == arcs_u[1:]
    return float(np.mean(same | conn))


def obs_level_metrics(df: pd.DataFrame, n_obs_by_trip: dict, col="p"):
    """Top-1 accuracy over ALL observations (obs with no candidate / miss count as wrong)."""
    top = argmax_per_obs(df, col)
    total = sum(n_obs_by_trip.values())
    acc = top.y.sum() / total
    hit_obs = df[df.y == 1].groupby("trip").obs.nunique().sum()
    cont = []
    for t, g in top.groupby("trip"):
        g = g.sort_values("obs")
        cont.append(continuity(g.u.values, g.v.values, g.arc.values))
    return {"top1_acc": acc, "top1_acc_given_hit": top.y.sum() / max(hit_obs, 1),
            "candidate_miss_rate": 1 - hit_obs / total, "route_continuity": float(np.mean(cont))}


def route_level_metrics(pred_route_arcs, ref_route_arcs, arc_len: dict):
    """Length-weighted precision/recall/F1 between predicted and reference arc sets.
    Twin arcs (reverse direction) are counted as different arcs."""
    P, R = set(map(int, pred_route_arcs)), set(map(int, ref_route_arcs))
    L = lambda S: sum(arc_len.get(a, 0.0) for a in S)
    inter = L(P & R)
    prec = inter / max(L(P), 1e-9); rec = inter / max(L(R), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
