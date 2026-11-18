"""Recovery helper: the cloud machine restarted during the Optuna search. The trials that had
already finished are re-inserted into the persistent study from the Optuna log, unchanged
(same parameters, same objective value, same per-trial CV metrics), so the search resumes."""
import os, re, ast, sys, optuna
from optuna.distributions import IntDistribution as I, FloatDistribution as F
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = f"{ROOT}/results/bayes"
log = open(sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/bayes_interrupted.log").read()
D = {"num_leaves": I(15, 127, log=True), "learning_rate": F(0.01, 0.2, log=True), "n_estimators": I(100, 500, step=50),
     "min_child_samples": I(5, 300, log=True), "subsample": F(0.5, 1.0), "colsample_bytree": F(0.4, 1.0),
     "reg_lambda": F(1e-3, 10.0, log=True)}
extra = {int(n): (float(f), float(l)) for n, f, l in re.findall(r"^trial (\d+): top1=\S+ f1=(\S+) ll=(\S+)", log, re.M)}
st = optuna.create_study(direction="maximize", study_name="lightgbm_top1", storage=f"sqlite:///{OUT}/optuna.db",
                         load_if_exists=True)
from datetime import datetime
ts = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S,%f")
prev = ts(re.search(r"\[I (\S+ \S+)\] A new study created", log).group(1))
for stamp, n, v, p in re.findall(r"\[I (\S+ \S+)\] Trial (\d+) finished with value: (\S+) and parameters: (\{.*?\})\.", log):
    f1, ll = extra[int(n)]
    sec = (ts(stamp) - prev).total_seconds(); prev = ts(stamp)
    st.add_trial(optuna.trial.create_trial(params=ast.literal_eval(p), distributions=D, value=float(v),
                 user_attrs={"cv3_macro_f1": f1, "cv3_log_loss": ll, "seconds": sec, "recovered_from_log": True}))
print(len(st.trials), "trials in study")
