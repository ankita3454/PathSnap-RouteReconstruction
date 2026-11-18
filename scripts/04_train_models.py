"""Step 4: train and compare 7 candidate-scoring models.

Protocol
  * 5-fold GroupKFold on the 78 usable TRAIN trips (group = trip, so no trip leaks across folds)
  * refit on all TRAIN, evaluate once on the 10 STUDENT_TEST trips
  * metrics: candidate-row macro-F1 / precision / recall / log-loss (thr 0.5),
             observation-level Top-1 accuracy, candidate miss rate, route continuity,
             train time, inference time
Outputs: results/models/*.csv|json, artifacts/models/<name>.joblib
"""
import os, sys, json, time
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.features import FEATURES
from pathsnap.models import make_model, MODEL_INFO, SEED
from pathsnap.metrics import row_metrics, obs_level_metrics

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = f"{ROOT}/results/models"; os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{ROOT}/artifacts/models", exist_ok=True)
names = sys.argv[1:] or list(MODEL_INFO)

tr = pd.read_parquet(f"{ROOT}/results/model_train.parquet")
te = pd.read_parquet(f"{ROOT}/results/model_student_test.parquet")
nobs_tr = json.load(open(f"{ROOT}/results/nobs_train.json"))
nobs_te = json.load(open(f"{ROOT}/results/nobs_student_test.json"))
X, y, groups = tr[FEATURES].values.astype(np.float32), tr.y.values, tr.trip.values
Xte, yte = te[FEATURES].values.astype(np.float32), te.y.values
rng = np.random.RandomState(SEED)


def fit(name, model, Xa, ya, eval_sets=None):
    if name == "KNN" and len(Xa) > 40000:
        idx = rng.choice(len(Xa), 40000, replace=False); Xa, ya = Xa[idx], ya[idx]
    if name == "LightGBM" and eval_sets:
        model.fit(Xa, ya, eval_set=eval_sets, eval_names=["train", "valid"], eval_metric="binary_logloss")
    elif name == "XGBoost" and eval_sets:
        model.fit(Xa, ya, eval_set=eval_sets, verbose=False)
    else:
        model.fit(Xa, ya)
    return model


def history(name, model):
    if name == "LightGBM":
        r = model.evals_result_; return {"train_logloss": r["train"]["binary_logloss"], "valid_logloss": r["valid"]["binary_logloss"]}
    if name == "XGBoost":
        r = model.evals_result(); return {"train_logloss": r["validation_0"]["logloss"], "valid_logloss": r["validation_1"]["logloss"]}
    if name == "MLP":
        m = model[-1]; return {"train_loss": list(map(float, m.loss_curve_)), "valid_accuracy": list(map(float, m.validation_scores_))}
    return None


cv_rows, test_rows, hist = [], [], {}
gkf = GroupKFold(n_splits=5)
for name in names:
    print(f"=== {name}", flush=True)
    for k, (a, b) in enumerate(gkf.split(X, y, groups)):
        model = make_model(name)
        ev = [(X[a][::4], y[a][::4]), (X[b], y[b])] if (k == 0 and name in ("LightGBM", "XGBoost")) else None
        t0 = time.time(); fit(name, model, X[a], y[a], ev); ttrain = time.time() - t0
        t0 = time.time(); p = model.predict_proba(X[b])[:, 1]; tinf = time.time() - t0
        val = tr.iloc[b][["trip", "obs", "arc", "u", "v", "y"]].copy(); val["p"] = p
        rm = row_metrics(y[b], p); om = obs_level_metrics(val, {t: nobs_tr[t] for t in val.trip.unique()})
        cv_rows.append({"model": name, "fold": k, "train_s": ttrain, "infer_s": tinf,
                        **{kk: v for kk, v in rm.items() if kk != "confusion"}, **om})
        if k == 0:
            h = history(name, model)
            if h: hist[name] = h
        print(f"  fold {k}: macroF1={rm['macro_f1']:.4f} ll={rm['log_loss']:.4f} top1={om['top1_acc']:.4f} ({ttrain:.0f}s)", flush=True)
    # refit on all TRAIN -> STUDENT_TEST
    model = make_model(name)
    t0 = time.time(); fit(name, model, X, y); ttrain = time.time() - t0
    t0 = time.time(); p = model.predict_proba(Xte)[:, 1]; tinf = time.time() - t0
    val = te[["trip", "obs", "arc", "u", "v", "y"]].copy(); val["p"] = p
    val.to_parquet(f"{OUT}/test_scores_{name}.parquet")
    rm = row_metrics(yte, p); om = obs_level_metrics(val, nobs_te)
    test_rows.append({"model": name, "family": MODEL_INFO[name][0], "config": MODEL_INFO[name][1],
                      "train_s": ttrain, "infer_s": tinf, "infer_ms_per_1k_obs": 1000 * tinf / sum(nobs_te.values()) * 1000 / 1000,
                      **{kk: v for kk, v in rm.items() if kk != "confusion"}, **om,
                      "confusion": json.dumps(rm["confusion"])})
    joblib.dump(model, f"{ROOT}/artifacts/models/{name}.joblib", compress=3)
    print(f"  TEST: macroF1={rm['macro_f1']:.4f} ll={rm['log_loss']:.4f} top1={om['top1_acc']:.4f} cont={om['route_continuity']:.4f}", flush=True)

    # append-safe persistence (lets models be run in separate processes)
    def _append(path, rows):
        new = pd.DataFrame(rows)
        if os.path.exists(path):
            old = pd.read_csv(path); old = old[~old.model.isin(new.model.unique())]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)
    _append(f"{OUT}/cv_folds.csv", [r for r in cv_rows if r["model"] == name])
    _append(f"{OUT}/student_test_results.csv", [r for r in test_rows if r["model"] == name])
    hp = f"{OUT}/histories.json"
    H = json.load(open(hp)) if os.path.exists(hp) else {}
    if name in hist: H[name] = hist[name]
    json.dump(H, open(hp, "w"))
print("done")
