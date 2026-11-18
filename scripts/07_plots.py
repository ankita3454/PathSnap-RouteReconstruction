"""Step 7: comparison figures, built only from the CSV/JSON logs written by steps 4-6."""
import os, sys, json
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.features import FEATURES

ROOT = os.path.join(os.path.dirname(__file__), "..")
F = f"{ROOT}/results/figures"; R = f"{ROOT}/results"
cv = pd.read_csv(f"{R}/models/cv_folds.csv")
te = pd.read_csv(f"{R}/models/student_test_results.csv")
order = cv.groupby("model").macro_f1.mean().sort_values().index.tolist()
COL = "#2b6cb0"

# consolidated table
agg = cv.groupby("model").agg(cv_macro_f1=("macro_f1", "mean"), cv_macro_f1_std=("macro_f1", "std"),
                              cv_top1=("top1_acc", "mean"), cv_log_loss=("log_loss", "mean"),
                              cv_train_s=("train_s", "mean"))
tab = te.set_index("model")[["family", "config", "top1_acc", "macro_f1", "precision_pos", "recall_pos", "log_loss",
                             "route_continuity", "candidate_miss_rate", "train_s", "infer_s"]].join(agg)
hp = f"{R}/hmm/student_test_summary.csv"
if os.path.exists(hp):
    h = pd.read_csv(hp, index_col=0)
    tab = tab.join(h[["hmm_top1", "hmm_continuity"]].rename(columns={"hmm_top1": "top1_after_hmm",
                                                                     "hmm_continuity": "continuity_after_hmm"}))
tab.sort_values("cv_macro_f1", ascending=False).to_csv(f"{R}/models/model_comparison_table.csv")
print(tab.sort_values("cv_macro_f1", ascending=False).drop(columns="config").round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
g = cv.groupby("model").macro_f1
ax.barh(order, g.mean()[order], xerr=g.std()[order], color=COL, capsize=4)
for i, m in enumerate(order): ax.text(g.mean()[m] + .004, i, f"{g.mean()[m]:.4f}", va="center", fontsize=9)
ax.set_xlim(.7, 1); ax.set_xlabel("5-fold GroupKFold macro-F1 (TRAIN), mean ± std")
ax.set_title("Cross-validation macro-F1 by model"); fig.tight_layout(); fig.savefig(f"{F}/cv_macro_f1.png", dpi=130)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(order)); t = te.set_index("model").loc[order]
ax.bar(x - .2, t.top1_acc, .4, label="Top-1 accuracy (obs-level)", color=COL)
ax.bar(x + .2, t.macro_f1, .4, label="macro-F1 (candidate rows)", color="#dd6b20")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=20); ax.set_ylim(.7, 1); ax.legend(loc="lower right")
ax.set_title("STUDENT_TEST: Top-1 accuracy and macro-F1 (classifier only)"); fig.tight_layout()
fig.savefig(f"{F}/student_test_top1_f1.png", dpi=130)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(order, t.log_loss, color="#805ad5"); ax.set_yscale("log"); ax.set_ylabel("log loss (log scale)")
for i, v in enumerate(t.log_loss): ax.text(i, v * 1.05, f"{v:.3f}", ha="center", fontsize=9)
ax.set_title("STUDENT_TEST log loss (lower is better)"); plt.xticks(rotation=20); fig.tight_layout()
fig.savefig(f"{F}/student_test_logloss.png", dpi=130)

fig, axs = plt.subplots(1, len(order), figsize=(3 * len(order), 3))
for a, m in zip(axs, order):
    c = np.array(json.loads(te.set_index("model").loc[m, "confusion"]))
    cn = c / c.sum(1, keepdims=True)
    a.imshow(cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2): a.text(j, i, f"{c[i,j]}\n{cn[i,j]:.2f}", ha="center", va="center", fontsize=8,
                                  color="white" if cn[i, j] > .5 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(m, fontsize=10)
fig.suptitle("STUDENT_TEST confusion matrices (candidate rows, threshold 0.5)"); fig.tight_layout()
fig.savefig(f"{F}/confusion_matrices.png", dpi=120)

H = json.load(open(f"{R}/models/histories.json"))
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for a, m in zip(axs, ["LightGBM", "XGBoost", "MLP"]):
    if m not in H: continue
    h = H[m]
    if m == "MLP":
        a.plot(h["train_loss"], label="train loss"); a2 = a.twinx()
        a2.plot(h["valid_accuracy"], color="#dd6b20", label="internal-val accuracy"); a2.set_ylabel("val accuracy")
        a.set_xlabel("epoch"); a.legend(loc="upper left"); a2.legend(loc="center right")
    else:
        a.plot(h["train_logloss"], label="train (25% subsample of fold-0 train)"); a.plot(h["valid_logloss"], label="fold-0 validation")
        a.set_xlabel("boosting round"); a.legend()
    a.set_ylabel("log loss"); a.set_title(f"{m}: CV fold 0 training history")
fig.tight_layout(); fig.savefig(f"{F}/training_histories.png", dpi=120)

if os.path.exists(f"{R}/bayes/optuna_trials.csv"):
    o = pd.read_csv(f"{R}/bayes/optuna_trials.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(o.number, o.value, s=18, color=COL, label="trial")
    ax.plot(o.number, o.value.cummax(), color="#e53e3e", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("3-fold CV Top-1 accuracy"); ax.legend()
    ax.set_title("Optuna TPE search - LightGBM"); fig.tight_layout(); fig.savefig(f"{F}/bayes_history.png", dpi=130)
    tv = pd.read_csv(f"{R}/bayes/tuned_vs_untuned.csv", index_col=0)
    cols = ["cv5_macro_f1", "cv5_top1", "test_top1", "test_macro_f1"]
    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(cols))
    ax.bar(x - .2, tv.loc["untuned", cols], .4, label="untuned", color="#a0aec0")
    ax.bar(x + .2, tv.loc["tuned", cols], .4, label="tuned", color=COL)
    ax.set_xticks(x); ax.set_xticklabels(cols); ax.set_ylim(.85, 1); ax.legend()
    ax.set_title("LightGBM untuned vs Bayesian-tuned"); fig.tight_layout(); fig.savefig(f"{F}/tuned_vs_untuned.png", dpi=130)
    m = joblib.load(f"{ROOT}/artifacts/final_model.joblib")
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES).sort_values()
    imp.to_csv(f"{R}/models/feature_importance_gain.csv")
    fig, ax = plt.subplots(figsize=(7, 6)); ax.barh(imp.index, imp.values / imp.sum(), color=COL)
    ax.set_xlabel("share of total split gain"); ax.set_title("Tuned LightGBM feature importance")
    fig.tight_layout(); fig.savefig(f"{F}/feature_importance.png", dpi=130)
print("plots done")

# ---- error analysis of classifier-only Top-1 mistakes (student test, untuned LightGBM) ----
s = pd.read_parquet(f"{R}/models/test_scores_LightGBM.parquet")
top = s.loc[s.groupby(["trip", "obs"]).p.idxmax()]
pos = s[s.y == 1][["trip", "obs", "u", "v"]].rename(columns={"u": "tu", "v": "tv"})
w = top[top.y == 0].merge(pos, on=["trip", "obs"], how="left")
ea = {"n_errors": int(len(w)),
      "reverse_twin_of_truth": float(((w.u == w.tv) & (w.v == w.tu)).mean()),
      "shares_node_with_truth": float(((w.u == w.tv) | (w.v == w.tu) | (w.u == w.tu) | (w.v == w.tv)).mean()),
      "truth_not_in_candidates": float(w.tu.isna().mean())}
json.dump(ea, open(f"{R}/models/error_analysis_lightgbm_test.json", "w"), indent=1)
print(ea)
