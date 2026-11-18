"""Step 5: Bayesian hyper-parameter optimisation (Optuna TPE) for the selected model (LightGBM).

Objective : maximise mean observation-level Top-1 accuracy over 3-fold GroupKFold on TRAIN
            (the quantity that matters for map matching; log-loss & macro-F1 are logged per trial).
Then      : untuned vs tuned compared on identical 5-fold GroupKFold + STUDENT_TEST,
            out-of-fold TRAIN probabilities of the tuned model saved for HMM tuning,
            tuned model refitted on all TRAIN and exported as the final artifact.
"""
import os, sys, json, time
import numpy as np, pandas as pd, joblib, optuna
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.features import FEATURES, FINAL_CFG
from pathsnap.models import make_model, SEED
from pathsnap.metrics import row_metrics, obs_level_metrics

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = f"{ROOT}/results/bayes"; os.makedirs(OUT, exist_ok=True)
N_TRIALS = int(os.environ.get("N_TRIALS", 20))

tr = pd.read_parquet(f"{ROOT}/results/model_train.parquet")
te = pd.read_parquet(f"{ROOT}/results/model_student_test.parquet")
nobs_tr = json.load(open(f"{ROOT}/results/nobs_train.json"))
nobs_te = json.load(open(f"{ROOT}/results/nobs_student_test.json"))
X, y, groups = tr[FEATURES].values.astype(np.float32), tr.y.values, tr.trip.values
Xte = te[FEATURES].values.astype(np.float32)
meta = ["trip", "obs", "arc", "u", "v", "y"]


def cv_eval(params, n_splits, return_oof=False):
    oof = np.zeros(len(y)); res = []
    for a, b in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model("LightGBM", params); m.fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]; oof[b] = p
        v = tr.iloc[b][meta].copy(); v["p"] = p
        rm = row_metrics(y[b], p); om = obs_level_metrics(v, {t: nobs_tr[t] for t in v.trip.unique()})
        res.append({"macro_f1": rm["macro_f1"], "log_loss": rm["log_loss"], "top1_acc": om["top1_acc"]})
    res = pd.DataFrame(res)
    return (res, oof) if return_oof else res


def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    t0 = time.time(); r = cv_eval(params, 3)
    trial.set_user_attr("cv3_macro_f1", float(r.macro_f1.mean()))
    trial.set_user_attr("cv3_log_loss", float(r.log_loss.mean()))
    trial.set_user_attr("seconds", time.time() - t0)
    print(f"trial {trial.number}: top1={r.top1_acc.mean():.4f} f1={r.macro_f1.mean():.4f} ll={r.log_loss.mean():.4f}", flush=True)
    return float(r.top1_acc.mean())


# persistent study: the search can be resumed if the machine restarts
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6),
                            study_name="lightgbm_top1", storage=f"sqlite:///{OUT}/optuna.db", load_if_exists=True)
done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
t0 = time.time()
if done < N_TRIALS:
    study.optimize(objective, n_trials=N_TRIALS - done)
search_s = float(sum(t.user_attrs.get("seconds", 0) for t in study.trials))
study.trials_dataframe().to_csv(f"{OUT}/optuna_trials.csv", index=False)
best = study.best_params
json.dump({"best_params": best, "best_cv3_top1": study.best_value, "n_trials": N_TRIALS,
           "search_seconds": search_s, "sampler": "TPE(seed=42, 6 random startup trials)",
           "objective": "mean 3-fold GroupKFold observation-level Top-1 accuracy (TRAIN)"},
          open(f"{OUT}/best_params.json", "w"), indent=1)

# ---- untuned vs tuned on identical protocol --------------------------------
rows = {}
for label, params in [("untuned", {}), ("tuned", best)]:
    r, oof = cv_eval(params, 5, return_oof=True)
    m = make_model("LightGBM", params)
    t0 = time.time(); m.fit(X, y); ttrain = time.time() - t0
    t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; tinf = time.time() - t0
    v = te[meta].copy(); v["p"] = p
    rm = row_metrics(te.y.values, p); om = obs_level_metrics(v, nobs_te)
    rows[label] = {"cv5_macro_f1": r.macro_f1.mean(), "cv5_macro_f1_std": r.macro_f1.std(),
                   "cv5_top1": r.top1_acc.mean(), "cv5_log_loss": r.log_loss.mean(),
                   "test_top1": om["top1_acc"], "test_macro_f1": rm["macro_f1"], "test_log_loss": rm["log_loss"],
                   "test_precision": rm["precision_pos"], "test_recall": rm["recall_pos"],
                   "test_route_continuity": om["route_continuity"], "train_s": ttrain, "infer_s": tinf}
    if label == "tuned":
        o = tr[meta + ["li", "dist"]].copy(); o["p"] = oof; o.to_parquet(f"{OUT}/oof_train_tuned.parquet")
        v["li"] = te.li.values; v["dist"] = te.dist.values; v.to_parquet(f"{OUT}/test_scores_tuned.parquet")
        os.makedirs(f"{ROOT}/artifacts", exist_ok=True)
        joblib.dump(m, f"{ROOT}/artifacts/final_model.joblib", compress=3)
        json.dump({"model": "LightGBM", "params": best, "features": FEATURES, "candidates": FINAL_CFG},
                  open(f"{ROOT}/artifacts/model_config.json", "w"), indent=1)
pd.DataFrame(rows).T.to_csv(f"{OUT}/tuned_vs_untuned.csv")
print(pd.DataFrame(rows).T)
