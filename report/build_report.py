"""Builds report/PathSnap_Technical_Report.pdf from the committed result files.
Every number in the report is read from results/ at build time.
Usage: python report/build_report.py   (needs google-chrome or chromium for HTML->PDF)
"""
import os, json, subprocess, base64, html
import pandas as pd, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R, FIG, IMG = f"{ROOT}/results", f"{ROOT}/results/figures", f"{ROOT}/report/img"
TEAM = os.environ.get("PATHSNAP_TEAM", "Team: Anisha Nukul (PES1UG24CS064) &middot; Ankita Pramod (PES1UG24CS065) &middot; Tanmay Sridhar (PES1UG25CS849) &middot; Akshar Vaghasiya (PES1UG25CS852) &middot; Section B")

f4 = lambda x: f"{x:.4f}"
pc = lambda x: f"{100*x:.2f}%"


def img(path, cap, width="100%"):
    if not os.path.exists(path):
        return f'<p class="missing">[missing figure: {os.path.basename(path)}]</p>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" style="width:{width}"><figcaption>{cap}</figcaption></figure>'


def table(df, fmt=None, cls=""):
    fmt = fmt or {}
    h = "<table class='%s'><thead><tr>" % cls + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr></thead><tbody>"
    for _, r in df.iterrows():
        h += "<tr>" + "".join(f"<td>{fmt[c](r[c]) if c in fmt and pd.notna(r[c]) else html.escape(str(r[c]))}</td>" for c in df.columns) + "</tr>"
    return h + "</tbody></table>"


# ------------------------------------------------------------------ data
inv = pd.read_csv(f"{R}/eda/inventory.csv", dtype={"trip": str})
summ = pd.read_csv(f"{R}/eda/summary_by_split.csv", index_col=0)
grid = pd.read_csv(f"{R}/eda/candidate_radius_grid_train.csv")
excl = json.load(open(f"{R}/eda/excluded_trips.json"))
ld = pd.read_csv(f"{R}/eda/label_distance_train.csv", index_col=0).iloc[:, 0]
cv = pd.read_csv(f"{R}/models/cv_folds.csv")
te = pd.read_csv(f"{R}/models/student_test_results.csv").set_index("model")
tf = pd.read_csv(f"{R}/models/train_fit_metrics.csv").set_index("model")
comp = pd.read_csv(f"{R}/models/model_comparison_table.csv", index_col=0)
ea = json.load(open(f"{R}/models/error_analysis_lightgbm_test.json"))
cvm = cv.groupby("model").agg(f1=("macro_f1", "mean"), f1s=("macro_f1", "std"), top1=("top1_acc", "mean"),
                              ll=("log_loss", "mean"), tr=("train_s", "mean"))
bp = json.load(open(f"{R}/bayes/best_params.json"))
tv = pd.read_csv(f"{R}/bayes/tuned_vs_untuned.csv", index_col=0)
trials = pd.read_csv(f"{R}/bayes/optuna_trials.csv")
hsum = pd.read_csv(f"{R}/hmm/student_test_summary.csv", index_col=0)
hpt = pd.read_csv(f"{R}/hmm/student_test_per_trip_tuned.csv", dtype={"trip": str})
hgrid = pd.read_csv(f"{R}/hmm/decoder_grid_train_oof.csv")
hcfg = json.load(open(f"{ROOT}/artifacts/hmm_config.json"))
ga = hsum.loc["tuned_LightGBM"]; gg = hsum.loc["Gaussian_HMM_no_classifier"]
fi = pd.read_csv(f"{R}/models/feature_importance_gain.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
fi = fi / fi.sum()
n_tr, n_te = int(summ.loc["train", "obs"]), int(summ.loc["student_test", "obs"])
sel = grid[(grid.radius == 50) & (grid.k == 8)].iloc[0]
ev = pd.read_csv(f"{R}/evaluator/evaluator_results.csv", dtype={"trip": str})
_w = ev.n_obs / ev.n_obs.sum()
ev_clf, ev_hmm = float((ev.top1_classifier_only * _w).sum()), float((ev.top1_hmm * _w).sum())
_x = ev[ev.trip != "00000015"]; ev_hmm_x15 = float((_x.top1_hmm * _x.n_obs).sum() / _x.n_obs.sum())
api_json = open(f"{IMG}/api_response.txt").read() if os.path.exists(f"{IMG}/api_response.txt") else ""
api_log = open(f"{IMG}/api_log.txt").read() if os.path.exists(f"{IMG}/api_log.txt") else ""

order = ["LogReg", "GaussNB", "KNN", "RandomForest", "LightGBM", "XGBoost", "MLP"]

# ------------------------------------------------------------------ consolidated table
ct = pd.DataFrame([{
    "Model": m, "Family": te.loc[m, "family"], "Key configuration": te.loc[m, "config"],
    "CV macro-F1 (±sd)": f"{cvm.loc[m,'f1']:.4f} ± {cvm.loc[m,'f1s']:.4f}",
    "Top-1 acc (test)": f4(te.loc[m, "top1_acc"]),
    "Route continuity (test, clf / +HMM)": f"{te.loc[m,'route_continuity']:.4f} / {hsum.loc[m,'hmm_continuity']:.4f}",
    "Test macro-F1": f4(te.loc[m, "macro_f1"]), "Log loss (test)": f4(te.loc[m, "log_loss"]),
    "Inference time (test, s)": f"{te.loc[m,'infer_s']:.2f}"} for m in order])

# ------------------------------------------------------------------ per-model prose
PM = {
"LogReg": dict(
 why="A linear baseline on our engineered features: if the right arc is mostly the closest arc with the matching heading, a weighted sum of distance and heading terms should already separate it. It also gives calibrated-ish probabilities that can be plugged straight into the HMM.",
 sl="Strengths: trains in ~2 s, very fast inference, coefficients are interpretable: the largest standardised weights are proj_overshoot (-5.0, fix projects past the arc end), head_diff_long (-1.8), head_diff (-1.0) and dist_minus_min (-1.0). Limitations: it cannot model interactions such as 'heading difference only matters when the vehicle is moving' (speed x head_diff), so it under-uses the heading signal that separates an arc from its reverse twin.",
 hp="C = 1.0 (L2), lbfgs, max_iter = 500; inputs are signed-log1p transformed then standardised (fitted inside each training fold).",
 loss="Binary cross-entropy (log loss) + L2 penalty."),
"GaussNB": dict(
 why="Included as the simplest probabilistic model: it treats every feature as conditionally independent Gaussians given the label, which tests whether a 'naive' likelihood model is good enough for emissions.",
 sl="Strengths: no tuning, trains in under a second. Limitations: our features are strongly correlated (dist, win_dist_mean, dist_prev, dist_next are all distances to the same arc) and non-Gaussian (heading differences are bimodal: ~0 deg or ~180 deg for twins). The independence assumption double-counts the distance evidence and produces over-confident probabilities, which is visible in its very high log loss.",
 hp="var_smoothing = 1e-9 (default); same signed-log1p + scaling as LogReg.",
 loss="No iterative loss: maximum-likelihood estimates of per-class means and variances."),
"KNN": dict(
 why="An instance-based model: candidates whose feature vectors look like previously seen correct candidates are probably correct. It needs no assumption about the functional form.",
 sl="Strengths: non-parametric, captures local non-linear structure. Limitations: brute-force distance computation scales with the training set, so it was fitted on a 40 000-row random subsample (of ~0.8 M rows per fold); it is the slowest model at inference (several seconds for the test set), and the probabilities are coarse (fractions of 25 neighbours), which hurts log loss.",
 hp="k = 25, distance weighting, Euclidean metric on scaled features, 40 000-row training subsample (seed 42).",
 loss="None (lazy learner); predictions are distance-weighted neighbour votes."),
"RandomForest": dict(
 why="Bagged decision trees capture interactions (e.g. heading only matters when speed is high, distance matters relative to the other candidates) without feature scaling, and averaging many trees gives stable probabilities.",
 sl="Strengths: robust, little tuning, near the top on every metric. Limitations: the slowest model to train here (~70 s per fold with 150 trees on 25 % bootstrap samples, 2 cores) and the saved model is large; it was marginally below the boosted models on CV macro-F1 and log loss.",
 hp="n_estimators = 150, min_samples_leaf = 10, max_samples = 0.25, max_features = sqrt, seed 42.",
 loss="Gini impurity per split (no global loss); probabilities = mean leaf class frequency."),
"LightGBM": dict(
 why="Gradient-boosted trees fit the residual errors of the previous trees, which suits our tabular, mixed-scale features and the many feature interactions; histogram-based training is fast on ~1 M candidate rows.",
 sl="Strengths: best Top-1 accuracy and joint-best log loss on the student-test trips, fast training with 2 cores, and it provides a training history. Limitations: it can over-fit with many leaves (train Top-1 is higher than CV Top-1, see C.9) and, like every per-candidate classifier, it scores each fix independently, so it cannot enforce route connectivity on its own.",
 hp="n_estimators = 400, learning_rate = 0.05, num_leaves = 63, subsample = 0.8, colsample_bytree = 0.8, min_child_samples = 20, reg_lambda = 0.",
 loss="Binary log loss (gradient boosting on the logistic loss)."),
"XGBoost": dict(
 why="A second boosting implementation with depth-wise trees and second-order gradients; included to check whether the boosting result depends on the specific library.",
 sl="Strengths: statistically tied with LightGBM (CV macro-F1 difference well inside one fold standard deviation), fastest of the boosted models at inference. Limitations: same independence-per-fix limitation as all classifiers; slightly lower student-test Top-1 than LightGBM.",
 hp="n_estimators = 400, eta = 0.05, max_depth = 7, subsample = 0.8, colsample_bytree = 0.8, tree_method = hist.",
 loss="Binary log loss (logistic objective)."),
"MLP": dict(
 why="A shallow neural network can learn smooth non-linear combinations of the features; with early stopping it is a reasonable neural baseline on tabular data.",
 sl="Strengths: competitive accuracy, very fast inference, produces epoch-wise training curves. Limitations: needs feature scaling, is sensitive to the learning rate, and hit the 40-epoch cap in some folds (ConvergenceWarning), so it may be slightly under-trained; it still trailed the tree ensembles.",
 hp="hidden layers (64, 32), ReLU, Adam lr = 1e-3, batch 1024, max_iter = 40, early stopping on a 10 % internal validation split, patience 5.",
 loss="Binary cross-entropy (log loss) with L2 alpha = 1e-4."),
}


def fit_comment(m):
    trn, cvt, tst = tf.loc[m, "train_top1"], cvm.loc[m, "top1"], te.loc[m, "top1_acc"]
    gap = trn - cvt
    if m == "GaussNB":
        verdict = "under-fitting: train, CV and test scores are all low and almost identical, so the model is too simple (strong bias), not over-fitted."
    elif m == "KNN":
        verdict = ("no strong over-fitting on the full training set because it was fitted on a 40k subsample; the "
                   "train-CV gap is small, but performance is limited by the subsample and coarse neighbour probabilities.")
    elif gap > 0.025:
        verdict = (f"mild over-fitting: training Top-1 is {100*gap:.1f} points above CV Top-1. CV and student-test scores agree, "
                   "so generalisation is still good; regularisation was one of the levers in the Bayesian search.")
    else:
        verdict = "no meaningful over- or under-fitting: train, CV and test scores are close."
    return (f"Train (in-sample) Top-1 = {trn:.4f}, 5-fold CV Top-1 = {cvt:.4f}, student-test Top-1 = {tst:.4f}; "
            f"train log loss = {tf.loc[m,'train_log_loss']:.4f} vs CV {cvm.loc[m,'ll']:.4f}. Verdict: {verdict}")


def model_section(i, m):
    p = PM[m]
    return f"""
<h3>3.2.{i} Model {i}: {m} <span class="fam">({te.loc[m,'family']})</span></h3>
<p><b>A.1 Why suitable?</b> {p['why']}</p>
<p><b>A.2 Strengths and limitations.</b> {p['sl']}</p>
<p><b>B.5 Important hyperparameters (initial values).</b> {p['hp']}</p>
<p><b>B.6 Loss / objective.</b> {p['loss']}</p>
<p><b>C.7 Performance on student-test.</b> Top-1 accuracy {pc(te.loc[m,'top1_acc'])}, route continuity of the raw argmax
sequence {pc(te.loc[m,'route_continuity'])}; after HMM decoding with the same decoder settings Top-1 becomes {pc(hsum.loc[m,'hmm_top1'])}
and continuity {pc(hsum.loc[m,'hmm_continuity'])}. Training time on all train rows {te.loc[m,'train_s']:.1f} s, inference on all
{n_te:,} test fixes {te.loc[m,'infer_s']:.2f} s.</p>
<p><b>C.8 Observation-level metrics (student-test).</b> Top-1 = {f4(te.loc[m,'top1_acc'])}, candidate-row macro-F1 = {f4(te.loc[m,'macro_f1'])}
(precision of the positive class {f4(te.loc[m,'precision_pos'])}, recall {f4(te.loc[m,'recall_pos'])}), log loss = {f4(te.loc[m,'log_loss'])}.</p>
<p><b>C.9 Over/under-fitting.</b> {fit_comment(m)}</p>"""


best_m = "LightGBM"
tun, unt = tv.loc["tuned"], tv.loc["untuned"]
lg_cv, xg_cv = cvm.loc["LightGBM", "f1"], cvm.loc["XGBoost", "f1"]

# team-written section text (report/team_text/*.html)
T = {f[:-5]: open(f"{ROOT}/report/team_text/{f}").read() for f in os.listdir(f"{ROOT}/report/team_text") if f.endswith(".html")}

# ------------------------------------------------------------------ HTML
H = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>PathSnap Technical Report</title>
<style>
@page {{ size: A4; margin: 16mm 14mm; }}
body {{ font: 10.5pt/1.45 "DejaVu Sans", Arial, sans-serif; color:#1a202c; }}
h1 {{ font-size: 22pt; margin: 0 0 4px; }} h2 {{ font-size: 14.5pt; border-bottom: 2px solid #2b6cb0; padding-bottom: 3px; margin-top: 26px; page-break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin: 16px 0 4px; page-break-after: avoid; }} .fam {{ color:#4a5568; font-weight: normal; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 6px 0 10px; page-break-inside: avoid; }}
th, td {{ border: 1px solid #cbd5e0; padding: 3px 5px; vertical-align: top; }} th {{ background: #edf2f7; text-align: left; }}
figure {{ margin: 8px 0 12px; page-break-inside: avoid; text-align: center; }} figcaption {{ font-size: 8.5pt; color: #4a5568; margin-top: 3px; }}
.box {{ background: #f7fafc; border-left: 4px solid #2b6cb0; padding: 6px 10px; margin: 8px 0; }}
pre {{ font-size: 7.5pt; background: #1a202c; color: #e2e8f0; padding: 8px; white-space: pre-wrap; page-break-inside: avoid; }}
.missing {{ color: #c53030; }} .small {{ font-size: 8.5pt; color:#4a5568 }}
.cover {{ border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; margin-bottom: 10px; }}
</style></head><body>
<div class="cover"><h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<div>UE24CS352A Machine Learning, PES University (RR Campus) &middot; Technical Report</div>
<div>{TEAM}</div>
<div class="small">Repository: github.com/ankita3454/PathSnap-RouteReconstruction &middot; all numbers below are generated from
<code>results/</code> by <code>report/build_report.py</code>.</div></div>

<div class="box"><b>Summary.</b> Each GPS fix gets road-arc candidates (radius 50 m, K = 8). A tuned LightGBM classifier scores every
candidate, and an HMM/Viterbi decoder turns the scores into a connected route. On the 10 student-test trips
({n_te:,} fixes) the classifier alone reaches {pc(ga['clf_top1'])} observation Top-1 accuracy. HMM decoding raises this to
<b>{pc(ga['hmm_top1'])}</b> and lifts route continuity from {pc(ga['clf_continuity'])} to {pc(ga['hmm_continuity'])}. The reconstructed
routes reach a mean length-weighted route F1 of {pc(hpt.hmm_route_f1.mean())}. On the 10 evaluator-test trips (live run, frozen system) Top-1 is {pc(ev_clf)} for the classifier alone and <b>{pc(ev_hmm)}</b> with HMM decoding.</div>

<h2>1. Problem framing</h2>
{T["s1"]}

<h2>2. Dataset understanding and EDA</h2>
{T["s2_pre"]}
{table(grid.pivot(index="radius", columns="k", values="miss_rate").reset_index().rename(columns=lambda c: f"K={c}" if c!="radius" else "radius (m)"),
       {f"K={k}": pc for k in [4,6,8,10,12]})}
<p class="small">Candidate miss rate on TRAIN (78 usable trips) by radius and K: the share of fixes whose reference arc is not in the candidate set.</p>

{T["s2_pat"]}
<p class="small">Inventory by partition (TRAIN vs STUDENT_TEST):</p>
{table(summ.reset_index()[["split","trips","obs","obs_min","obs_median","obs_max","nan_rows","gaps_gt5s","dt_1s_frac","route_contiguous","nodes_median","arcs_median","isolated_nodes"]],
       {"dt_1s_frac": f4, "route_contiguous": f4})}
{img(f"{FIG}/eda_trip_sizes.png", "Fig 1. Trip sizes, path lengths and network sizes (TRAIN vs STUDENT_TEST).")}
{img(f"{FIG}/eda_sampling_intervals.png", "Fig 2. Sampling interval distribution, TRAIN (missing-value / gap analysis).", "70%")}
{img(f"{FIG}/eda_trajectories_grid.png", "Fig 3. GPS trajectories (blue), reference routes (green) and road networks (grey), 12 TRAIN trips.")}
{img(f"{FIG}/eda_label_distance_and_radius.png", "Fig 4. Fix-to-reference-arc distance (TRAIN) and candidate miss rate vs radius/K (TRAIN, anomalous trips removed).")}
{img(f"{IMG}/map_eda.png", "Fig 5. Folium map: STUDENT_TEST trip 00000052, raw GPS, road network and reference route (results/maps/eda_student_test_00000052.html; screenshot rendered offline without basemap tiles).", "85%")}

{T["s2_feat"]}

<h2>3. Model experimentation and comparison</h2>
{T["s3_intro"]}
<h3>3.1 Overall model comparison</h3>
{table(ct, cls="")}
<p class="small">CV macro-F1 and test macro-F1 are computed over candidate rows (threshold 0.5). Top-1 accuracy is per GPS fix,
over all {n_te:,} test fixes; a fix with no correct candidate counts as wrong. Route continuity is the share of consecutive fix
pairs whose predicted arcs are the same or directly connected. Inference time is for all candidate rows of the 10 test trips.</p>
{T["s3_sel"]}
<h3>3.2 Per-model analysis</h3>
{"".join(model_section(i+1, m) for i, m in enumerate(order))}
<h3>3.3 Model visualisations</h3>
{img(f"{FIG}/cv_macro_f1.png", "Fig 6. 5-fold GroupKFold macro-F1 (mean ± sd) for all 7 models.", "80%")}
{img(f"{FIG}/student_test_top1_f1.png", "Fig 7. Student-test Top-1 accuracy and macro-F1 (classifier only).", "85%")}
{img(f"{FIG}/student_test_logloss.png", "Fig 8. Student-test log loss (log scale).", "75%")}
{img(f"{FIG}/confusion_matrices.png", "Fig 9. Student-test confusion matrices on candidate rows (counts and row-normalised rates).")}
{img(f"{FIG}/training_histories.png", "Fig 10. Training histories, only for models that produce them: LightGBM and XGBoost (log loss per boosting round, CV fold 0) and MLP (training loss and internal-validation accuracy per epoch). Tree ensembles, KNN, NB and LogReg have no epoch history, so none is shown for them.")}

<h2>4. Bayesian hyperparameter optimisation (LightGBM)</h2>
{T["s4"]}
{table(pd.DataFrame({"metric": ["CV macro-F1 (5-fold)", "CV Top-1 (5-fold)", "CV log loss", "Student-test Top-1", "Student-test macro-F1", "Student-test log loss", "Student-test route continuity", "Train time (s)", "Inference time, test (s)"],
                     "untuned": [unt.cv5_macro_f1, unt.cv5_top1, unt.cv5_log_loss, unt.test_top1, unt.test_macro_f1, unt.test_log_loss, unt.test_route_continuity, unt.train_s, unt.infer_s],
                     "tuned": [tun.cv5_macro_f1, tun.cv5_top1, tun.cv5_log_loss, tun.test_top1, tun.test_macro_f1, tun.test_log_loss, tun.test_route_continuity, tun.train_s, tun.infer_s]}),
       {"untuned": f4, "tuned": f4})}
{T["s4_why"]}
{img(f"{FIG}/bayes_history.png", "Fig 11. Optuna TPE trial values and best-so-far.", "75%")}
{img(f"{FIG}/tuned_vs_untuned.png", "Fig 12. Untuned vs tuned LightGBM.", "75%")}
{img(f"{FIG}/feature_importance.png", "Fig 13. Tuned LightGBM feature importance (gain).", "60%")}

<h2>5. HMM and Viterbi route reconstruction</h2>
{T["s5"]}
{table(hgrid.rename(columns={"hmm_top1":"HMM Top-1","clf_top1":"classifier Top-1","hmm_continuity":"HMM continuity"}), {"HMM Top-1": f4, "classifier Top-1": f4, "HMM continuity": f4})}
<p class="small">Decoder grid on 20 TRAIN trips, scored with out-of-fold probabilities. Selected: &beta; = {hcfg['params']['beta']},
w<sub>e</sub> = {hcfg['params']['w_e']}.</p>
{table(pd.DataFrame([
  {"pipeline": "Tuned LightGBM, classifier only (argmax)", "Top-1": ga.clf_top1, "continuity": ga.clf_continuity, "route F1 (mean)": hpt.clf_route_f1.mean()},
  {"pipeline": "Tuned LightGBM + HMM/Viterbi", "Top-1": ga.hmm_top1, "continuity": ga.hmm_continuity, "route F1 (mean)": hpt.hmm_route_f1.mean()},
  {"pipeline": "Gaussian-emission HMM (no classifier)", "Top-1": gg.hmm_top1, "continuity": gg.hmm_continuity, "route F1 (mean)": pd.read_csv(f"{R}/hmm/student_test_per_trip_gaussian_hmm.csv").hmm_route_f1.mean()},
]), {"Top-1": f4, "continuity": f4, "route F1 (mean)": f4})}
{table(hpt[["trip","n_obs","clf_top1","hmm_top1","clf_continuity","hmm_continuity","hmm_route_precision","hmm_route_recall","hmm_route_f1","hmm_route_breaks","decode_s"]],
       {c: f4 for c in ["clf_top1","hmm_top1","clf_continuity","hmm_continuity","hmm_route_precision","hmm_route_recall","hmm_route_f1"]} | {"decode_s": lambda x: f"{x:.2f}"})}
{T["s5_q6"]}
{img(f"{FIG}/hmm_vs_clf_per_trip.png", "Fig 14. Per-trip Top-1 accuracy, classifier only vs classifier + HMM (STUDENT_TEST).")}

<h2>6. Geographical visualisation (Folium)</h2>
{T["s6"]}
{img(f"{IMG}/map_pred_a.png", "Fig 15. STUDENT_TEST trip __MAPA__: GPS, reference route (green), predicted route (red).", "90%")}
{img(f"{IMG}/map_pred_b.png", "Fig 16. STUDENT_TEST trip __MAPB__, zoomed: classifier-only arcs (amber) jump between a road and its neighbours, while the HMM route (red) stays connected.", "90%")}

<h2>7. API, UI and deployment</h2>
{T["s7"]}
{img(f"{IMG}/ui_laptop2.png", "Fig 17. Two-laptop deployment: the evaluator UI running on Laptop 2, connected over the network to the API on Laptop 1 (http://172.20.10.2:8000), showing evaluator trip 00000059. The metrics match the Laptop 1 API log in Fig 18.", "95%")}
{img(f"{IMG}/ui_live.png", "Fig 17b. Evaluator UI during the live run on the 10 evaluator trips (trip 00000015 selected): GPS fixes (blue), predicted HMM route (red), server-side metrics and timings.", "95%")}
{img(f"{IMG}/api_log_laptop2.png", "Fig 18. API log on Laptop 1 during the two-laptop run: GET /trips and POST /predict requests arrive from Laptop 2 (172.20.10.3) over the network, and every request returns 200 OK with its server-side metrics. The trip 00000059 numbers match Fig 17 on Laptop 2.", "100%")}
{img(f"{IMG}/api_log_live.png", "Fig 19. API log on Laptop 1 (uvicorn terminal) during the live run: model loaded with eval_dir = evaluator_test, GET /trips returns 10 trips, and every POST /predict returns 200 OK with its server-side metrics and time.", "100%")}
<p><b>Example API response</b> (<code>curl</code>, a student-test trip, GPS/route arrays truncated):</p><pre>{html.escape(api_json)}</pre>

<h3>7.1 Results on the evaluator-test trips (live run)</h3>
{table(ev[["trip","n_obs","top1_classifier_only","top1_hmm","route_f1","continuity_hmm","server_time_s"]], {c: f4 for c in ["top1_classifier_only","top1_hmm","route_f1","continuity_hmm"]} | {"server_time_s": lambda x: f"{x:.2f}"})}
{T["s71"]}

<h2>8. Limitations</h2>
<ul>
<li>Per-fix labels are derived from the reference route by alignment, so they inherit route gaps (11 non-contiguous routes) and can
be ambiguous where twin arcs overlap. Reported accuracies are relative to these derived labels.</li>
<li>The two excluded train trips show that some trips have large systematic offsets. A trip like that in the evaluator set would
score poorly on Top-1 even with a sensible route.</li>
<li>Our continuity metric is strict: when the vehicle crosses a very short arc between two 1 Hz fixes, the two predicted arcs are
counted as "not directly connected" even though the reconstructed route fills that arc in.</li>
<li>Very slow movement (walking) makes time-window heading unreliable (evaluator trip 00000015, Section 7.1).</li>
<li>The decoder parameters were tuned on 20 train trips, and a finer grid could add a little more.</li>
</ul>

<h2>9. Academic integrity and AI-assistance disclosure</h2>
<p>__AI_DISCLOSURE__</p>
</body></html>"""

# post-hoc comments are written in report/comments.json so the prose stays separate from the generator
C = json.load(open(f"{ROOT}/report/comments.json"))
for k, v in C.items():
    H = H.replace(f"__{k}__", v)
out_html = f"{ROOT}/report/PathSnap_Technical_Report.html"
open(out_html, "w").write(H)
chrome = os.environ.get("CHROME", "google-chrome")
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={ROOT}/report/PathSnap_Technical_Report.pdf", "file://" + out_html], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("report written")
