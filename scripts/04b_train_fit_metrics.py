"""Step 4b: in-sample (training-set) metrics of each final model, used as the
over/under-fitting check: compare with 5-fold CV and student-test metrics."""
import os, sys, json
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.features import FEATURES
from pathsnap.models import MODEL_INFO
from pathsnap.metrics import row_metrics, obs_level_metrics

ROOT = os.path.join(os.path.dirname(__file__), "..")
tr = pd.read_parquet(f"{ROOT}/results/model_train.parquet")
nobs = json.load(open(f"{ROOT}/results/nobs_train.json"))
X = tr[FEATURES].values.astype(np.float32)
rows = []
for name in MODEL_INFO:
    m = joblib.load(f"{ROOT}/artifacts/models/{name}.joblib")
    p = m.predict_proba(X)[:, 1]
    v = tr[["trip", "obs", "arc", "u", "v", "y"]].copy(); v["p"] = p
    rm = row_metrics(tr.y.values, p); om = obs_level_metrics(v, nobs)
    rows.append({"model": name, "train_macro_f1": rm["macro_f1"], "train_log_loss": rm["log_loss"],
                 "train_top1": om["top1_acc"]})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/results/models/train_fit_metrics.csv", index=False)
