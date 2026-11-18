"""Model zoo for candidate scoring: P(candidate arc is the reference arc | features)."""
from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np

SEED = 42

# heavy-tailed features are log1p-transformed for the scale-sensitive models
LOG_COLS_IDX = None  # set by caller via make_models(feature_names)


def _log1p_nonneg(X):
    return np.sign(X) * np.log1p(np.abs(X))


MODEL_INFO = {
    "LogReg":   ("Linear / probabilistic", "StandardScaler(signed log1p) + L2 LogisticRegression C=1.0, lbfgs"),
    "GaussNB":  ("Linear / probabilistic", "GaussianNB on signed-log1p + scaled features, var_smoothing=1e-9"),
    "KNN":      ("Instance-based", "distance-weighted KNN k=25, scaled; fitted on 40k-row subsample"),
    "RandomForest": ("Ensemble (bagging)", "150 trees, min_samples_leaf=10, max_samples=0.25, max_features=sqrt"),
    "LightGBM": ("Boosting", "GBDT 400 rounds, lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8"),
    "XGBoost":  ("Boosting", "hist, 400 rounds, eta=0.05, max_depth=7, subsample=0.8, colsample=0.8"),
    "MLP":      ("Neural network", "MLP (64,32) ReLU, adam lr=1e-3, batch 1024, early stopping (10% val)"),
}


def make_model(name: str, params: dict | None = None):
    params = params or {}
    pre = lambda: [FunctionTransformer(_log1p_nonneg), StandardScaler()]
    if name == "LogReg":
        return make_pipeline(*pre(), LogisticRegression(C=params.get("C", 1.0), max_iter=500, random_state=SEED))
    if name == "GaussNB":
        return make_pipeline(*pre(), GaussianNB())
    if name == "KNN":
        return make_pipeline(*pre(), KNeighborsClassifier(n_neighbors=params.get("k", 25), weights="distance",
                                                          algorithm="brute", n_jobs=2))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=10, max_samples=0.25, max_features="sqrt",
                                      n_jobs=2, random_state=SEED)
    if name == "LightGBM":
        p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, min_child_samples=20, reg_lambda=0.0, random_state=SEED, n_jobs=2, verbose=-1)
        p.update(params)
        return lgb.LGBMClassifier(**p)
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=SEED,
                                 eval_metric="logloss")
    if name == "MLP":
        return make_pipeline(*pre(), MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3,
                                                   batch_size=1024, max_iter=40, early_stopping=True,
                                                   validation_fraction=0.1, n_iter_no_change=5,
                                                   random_state=SEED))
    raise ValueError(name)
