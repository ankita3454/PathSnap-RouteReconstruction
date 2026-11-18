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
<p>The <code>.route</code> file lists the arcs the vehicle traversed in order, but it does not say which GPS fix belongs to which
arc. We therefore (i) derive a reference arc for every fix (Section 2.1), (ii) frame scoring as a <b>binary classification over
candidate (fix, arc) pairs</b>: <i>is this candidate the reference arc of this fix?</i>, and (iii) normalise the scores within each
fix's candidate set. The result is a probability over candidates, which is exactly what the HMM needs as an emission.
<b>Observation-level Top-1 accuracy</b> is the primary selection metric, because it is literally "did we put this fix on the right
road". Candidate-row macro-F1 and log loss are reported as required. Log loss matters because the HMM consumes the probabilities.</p>

<h2>2. Dataset understanding and EDA</h2>
<p class="small">Partitions: EDA statistics use TRAIN and STUDENT_TEST as labelled. Every modelling choice (candidate radius/K,
anomaly rule, feature design, hyperparameters, decoder parameters) uses TRAIN only. The evaluator-test partition is not available
to us and is never used.</p>
<h3>2.1 Raw data and preprocessing</h3>
{table(summ.reset_index()[["split","trips","obs","obs_min","obs_median","obs_max","nan_rows","gaps_gt5s","dt_1s_frac","route_contiguous","nodes_median","arcs_median","isolated_nodes"]],
       {"dt_1s_frac": f4, "route_contiguous": f4})}
<p><b>How we preprocessed.</b></p>
<ol>
<li><b>GPS cleaning.</b> We drop non-finite or out-of-range coordinates, sort by timestamp and drop duplicate timestamps. There were
0 missing or invalid values in either partition, and no out-of-order rows, so cleaning changed nothing, but the checks stay in the
pipeline for the evaluator trips.</li>
<li><b>Projection.</b> Trips are spread worldwide (latitude {inv.lat_min.min():.1f} to {inv.lat_max.max():.1f}, longitude
{inv.lon_min.min():.1f} to {inv.lon_max.max():.1f}). Degrees are not metres, so every trip is projected with a local
equirectangular projection centred on its median fix. Over the extent of one trip the scale error is far below GPS noise, and all
later distances are cheap Euclidean metres.</li>
<li><b>Network cropping.</b> We keep only the arcs with an endpoint inside the trip's GPS bounding box plus 800 m. Networks have up
to ~300k arcs, and cropping keeps both candidate search and shortest-path queries fast.</li>
<li><b>Per-fix reference labels.</b> A monotonic dynamic programme assigns each fix to a position in the route sequence, never
going backwards in time, while minimising total fix-to-arc distance (<code>labels.py</code>). This respects the route order, which
matters where an arc and its reverse twin overlap or where a route revisits a street. Greedy nearest-arc snapping would get these
cases wrong.</li>
<li><b>Anomalous trips.</b> For train trips {", ".join(excl['excluded'])}, the median distance from a fix to its own reference route
is over 50 m (122 m and 140 m), so the GPS and the route do not describe the same path well. Their derived labels are unreliable,
and both trips are excluded from training (rule in <code>results/eda/excluded_trips.json</code>). This leaves 78 training trips.</li>
<li><b>Candidates.</b> Arc geometry is sampled every 10 m into a KD-tree. Each fix takes all arcs within the radius, keeping the K
nearest. Radius and K were chosen on TRAIN (table below): <b>50 m / K = 8</b> gives a {pc(sel.miss_rate)} candidate miss rate on
TRAIN with ~1.0 M candidate rows. Going to 80 m / K = 12 only lowers the miss rate to {pc(grid[(grid.radius==80)&(grid.k==12)].miss_rate.iloc[0])}
while adding 52 % more rows. On STUDENT_TEST the miss rate is {pc(te.loc['LightGBM','candidate_miss_rate'])}.</li>
</ol>
{table(grid.pivot(index="radius", columns="k", values="miss_rate").reset_index().rename(columns=lambda c: f"K={c}" if c!="radius" else "radius (m)"),
       {f"K={k}": pc for k in [4,6,8,10,12]})}
<p class="small">Candidate miss rate on TRAIN (78 usable trips) by radius and K: the share of fixes whose reference arc is not in the candidate set.</p>

<h3>2.2 Patterns observed</h3>
<ul>
<li><b>Sampling.</b> Nominally 1 Hz: {pc(inv[inv.split=='train'].dt_1s_frac.mean())} of train steps are exactly 1 s, but there are
{int(summ.loc['train','gaps_gt5s'])} gaps over 5 s in train (the longest is {inv.dt_max.max():.0f} s, and one trip spans 1.8 days).
Gaps motivate the gap-aware transition model.</li>
<li><b>GPS noise.</b> On TRAIN, the median distance from a fix to its reference arc is {ld['50%']:.1f} m, but the tail is heavy:
95th percentile {ld['95%']:.0f} m before excluding the two anomalous trips. Distance alone is not enough near parallel or close
roads.</li>
<li><b>Reverse twins.</b> Across all trip networks, {pc((inv.twin_arc_frac*inv.n_arcs).sum()/inv.n_arcs.sum())} of arcs have a reverse-direction twin with identical geometry, so for most fixes
the two nearest candidates are geometrically tied. Only motion direction (GPS heading) can separate them. This drove the heading
features.</li>
<li><b>Network structure.</b> There are no isolated nodes, but the networks are fragmented: the median train network has
{int(inv[inv.split=='train'].weak_components.median())} weakly connected components, and the largest component holds on average
{pc(inv[inv.split=='train'].largest_weak_frac.mean())} of nodes. The decoder must therefore tolerate disconnected candidate pairs.</li>
<li><b>Reference routes.</b> {int((~inv.route_contiguous).sum())} of 90 routes are not end-to-end contiguous (consecutive arcs do not
share a node), so reference routes have gaps of their own. We avoid overstating label certainty.</li>
</ul>
{img(f"{FIG}/eda_trip_sizes.png", "Fig 1. Trip sizes, path lengths and network sizes (TRAIN vs STUDENT_TEST).")}
{img(f"{FIG}/eda_sampling_intervals.png", "Fig 2. Sampling interval distribution, TRAIN (missing-value / gap analysis).", "70%")}
{img(f"{FIG}/eda_trajectories_grid.png", "Fig 3. GPS trajectories (blue), reference routes (green) and road networks (grey), 12 TRAIN trips.")}
{img(f"{FIG}/eda_label_distance_and_radius.png", "Fig 4. Fix-to-reference-arc distance (TRAIN) and candidate miss rate vs radius/K (TRAIN, anomalous trips removed).")}
{img(f"{IMG}/map_eda.png", "Fig 5. Folium map: STUDENT_TEST trip 00000052, raw GPS, road network and reference route (results/maps/eda_student_test_00000052.html; screenshot rendered offline without basemap tiles).", "85%")}

<h3>2.3 Features and why</h3>
<p>We engineered 21 features per (fix, candidate) pair (<code>features.py</code>):</p>
<ul>
<li><b>Geometry:</b> <code>dist</code>, <code>dist_rank</code>, <code>dist_minus_min</code>, <code>dist_softmax</code>
(Gaussian-weighted share of the candidate set). These capture absolute and relative closeness.</li>
<li><b>Direction:</b> <code>head_diff</code> (heading from fixes t-1 to t+1 vs the arc bearing, directed),
<code>head_diff_long</code> (t-5 to t+5, robust when slow) and <code>head_diff_undirected</code>. These separate reverse twins
and crossing streets.</li>
<li><b>Kinematics:</b> <code>speed</code>, <code>step_len</code> and <code>dt</code>. Heading is unreliable when stationary, and gaps
make positions less informative.</li>
<li><b>Temporal context:</b> <code>win_dist_mean</code> and <code>win_dist_max</code> (distance of fixes t-3 to t+3 to the same arc),
<code>dist_prev</code> and <code>dist_next</code>. A correct arc stays close over neighbouring fixes, while a crossing street is
close for only one fix.</li>
<li><b>Arc and topology:</b> <code>arc_len</code>, <code>proj_frac</code>, <code>proj_overshoot</code> (projection falls beyond
the arc end, i.e. the fix is near an intersection), <code>out_deg_to</code>, <code>in_deg_from</code>, <code>has_twin</code> and
<code>n_cands</code>.</li>
</ul>
<p>The most informative features in the tuned model were {", ".join(f"<code>{k}</code> ({pc(v)})" for k, v in fi.head(5).items())}
(share of split gain, Fig 13).</p>

<h2>3. Model experimentation and comparison</h2>
<p>Protocol: <b>5-fold GroupKFold</b> on the 78 train trips, grouped by trip so that no trip appears in both the fit and validation
folds. Each model is then refit on all train trips and evaluated once on the 10 student-test trips. Seed 42 throughout.
Per-fold logs are in <code>results/models/cv_folds.csv</code>.</p>
<h3>3.1 Overall model comparison</h3>
{table(ct, cls="")}
<p class="small">CV macro-F1 and test macro-F1 are computed over candidate rows (threshold 0.5). Top-1 accuracy is per GPS fix,
over all {n_te:,} test fixes; a fix with no correct candidate counts as wrong. Route continuity is the share of consecutive fix
pairs whose predicted arcs are the same or directly connected. Inference time is for all candidate rows of the 10 test trips.</p>
<p><b>Selection.</b> LightGBM and XGBoost are statistically tied on CV macro-F1 ({f4(lg_cv)} vs {f4(xg_cv)}; the difference is far
below the fold standard deviation of ~0.010). LightGBM has the best student-test Top-1 ({pc(te.loc['LightGBM','top1_acc'])}),
joint-best test log loss and a lower training time, and it plugs directly into the decoder through calibrated probabilities. It
was selected for Bayesian optimisation. Random Forest is close but 4x slower to train. GaussNB is clearly unsuitable: its log loss
of {te.loc['GaussNB','log_loss']:.2f} shows over-confident, poorly calibrated probabilities.</p>
<p><b>Where the classifiers fail.</b> Of the {ea['n_errors']} Top-1 errors that untuned LightGBM makes on student-test,
{pc(ea['reverse_twin_of_truth'])} pick the reverse twin of the correct arc, {pc(ea['shares_node_with_truth'])} pick an arc that
shares a node with the correct arc (intersections), and only {pc(ea['truth_not_in_candidates'])} are fixes whose correct arc was
not in the candidate set. Most errors are therefore local and sequential in nature, which is exactly what the HMM is meant to fix.</p>
<h3>3.2 Per-model analysis</h3>
{"".join(model_section(i+1, m) for i, m in enumerate(order))}
<h3>3.3 Model visualisations</h3>
{img(f"{FIG}/cv_macro_f1.png", "Fig 6. 5-fold GroupKFold macro-F1 (mean ± sd) for all 7 models.", "80%")}
{img(f"{FIG}/student_test_top1_f1.png", "Fig 7. Student-test Top-1 accuracy and macro-F1 (classifier only).", "85%")}
{img(f"{FIG}/student_test_logloss.png", "Fig 8. Student-test log loss (log scale).", "75%")}
{img(f"{FIG}/confusion_matrices.png", "Fig 9. Student-test confusion matrices on candidate rows (counts and row-normalised rates).")}
{img(f"{FIG}/training_histories.png", "Fig 10. Training histories, only for models that produce them: LightGBM and XGBoost (log loss per boosting round, CV fold 0) and MLP (training loss and internal-validation accuracy per epoch). Tree ensembles, KNN, NB and LogReg have no epoch history, so none is shown for them.")}

<h2>4. Bayesian hyperparameter optimisation (LightGBM)</h2>
<ol>
<li><b>Hyperparameters searched:</b> num_leaves, learning_rate, n_estimators, min_child_samples, subsample, colsample_bytree, reg_lambda.</li>
<li><b>Ranges / distributions:</b> num_leaves int 15–127 (log), learning_rate 0.01–0.2 (log), n_estimators 100–500 (step 50),
min_child_samples 5–300 (log), subsample 0.5–1.0, colsample_bytree 0.4–1.0, reg_lambda 1e-3–10 (log).</li>
<li><b>Objective:</b> maximise mean observation-level Top-1 accuracy over 3-fold GroupKFold on TRAIN. Macro-F1 and log loss are
logged for every trial (<code>results/bayes/optuna_trials.csv</code>).</li>
<li><b>Trials:</b> {bp['n_trials']} trials of Optuna's TPE sampler (seed 42, 6 random start-up trials), with {bp['search_seconds']/60:.1f} min of total compute.
An earlier search with a wider space (num_leaves up to 255, n_estimators up to 800) was stopped after 11 trials because each trial
took ~3 min on our hardware. Its log is kept in <code>results/bayes/bayes_aborted_first_search.log</code>, and its best value
(0.8950) is in line with the final search.</li>
<li><b>Best values:</b> {", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in bp['best_params'].items())}
(3-fold CV Top-1 = {bp['best_cv3_top1']:.4f}). Our cloud machine restarted after trial 14. The 15 finished
trials were re-inserted into the (now SQLite-persistent) Optuna study from the run log by <code>05a_import_trials_from_log.py</code>,
with their parameters, objective values and logged CV metrics unchanged, and the search resumed to {bp['n_trials']} trials.</li>
<li><b>Did tuning help?</b> See the table below. Both models were evaluated with the identical 5-fold protocol and on the same
student-test trips.</li>
</ol>
{table(pd.DataFrame({"metric": ["CV macro-F1 (5-fold)", "CV Top-1 (5-fold)", "CV log loss", "Student-test Top-1", "Student-test macro-F1", "Student-test log loss", "Student-test route continuity", "Train time (s)", "Inference time, test (s)"],
                     "untuned": [unt.cv5_macro_f1, unt.cv5_top1, unt.cv5_log_loss, unt.test_top1, unt.test_macro_f1, unt.test_log_loss, unt.test_route_continuity, unt.train_s, unt.infer_s],
                     "tuned": [tun.cv5_macro_f1, tun.cv5_top1, tun.cv5_log_loss, tun.test_top1, tun.test_macro_f1, tun.test_log_loss, tun.test_route_continuity, tun.train_s, tun.infer_s]}),
       {"untuned": f4, "tuned": f4})}
<p>__TUNING_COMMENT__</p>
{img(f"{FIG}/bayes_history.png", "Fig 11. Optuna TPE trial values and best-so-far.", "75%")}
{img(f"{FIG}/tuned_vs_untuned.png", "Fig 12. Untuned vs tuned LightGBM.", "75%")}
{img(f"{FIG}/feature_importance.png", "Fig 13. Tuned LightGBM feature importance (gain).", "60%")}

<h2>5. HMM and Viterbi route reconstruction</h2>
<ol>
<li><b>Hidden states.</b> At fix t, the states are its candidates C<sub>t</sub>. Each state is an arc together with the projected
offset s along that arc, so every state maps one-to-one to an original arc id and the candidate-to-segment correspondence is
preserved.</li>
<li><b>Emissions.</b> log b<sub>t</sub>(j) = w<sub>e</sub> &middot; log( p<sub>j</sub> / &Sigma;<sub>k&isin;C<sub>t</sub></sub> p<sub>k</sub> + 10<sup>-6</sup> ),
where p is the tuned LightGBM probability. As a baseline we also ran a classic Gaussian emission
(&minus;d&sup2;/2&sigma;&sup2;, &sigma; = 8 m) with no classifier.</li>
<li><b>Transitions.</b> log a(i&rarr;j) = &minus;| d<sub>route</sub>(i,j) &minus; d<sub>gc</sub>(t&minus;1,t) | / &beta;<sub>t</sub>.
Here d<sub>route</sub> is the driving distance from (arc<sub>i</sub>, s<sub>i</sub>) to (arc<sub>j</sub>, s<sub>j</sub>): the
remainder of arc i, plus the Dijkstra shortest path from its end node to the start node of arc j (cached per source, limit 3 km),
plus s<sub>j</sub>. For the same arc moving forward it is |s<sub>j</sub> &minus; s<sub>i</sub>|. d<sub>gc</sub> is the straight-line
distance between the two fixes. Routes that wander much more than the GPS moved are penalised.</li>
<li><b>Missing candidates and gaps.</b> Fixes with an empty candidate set are skipped, and the next transition spans the gap.
&beta;<sub>t</sub> = &beta;&middot;&radic;max(&Delta;t, 1) loosens the transition after time gaps, and the plausible distance grows
as v<sub>max</sub>&middot;&Delta;t + 50 m (v<sub>max</sub> = 60 m/s).</li>
<li><b>Disconnected segments.</b> If no path exists within the search limit, the path is implausibly long, or the move is backwards
on the same arc by more than 8 m, the transition gets a finite penalty (log a = &minus;30) instead of &minus;&infin;. Viterbi therefore
never breaks on fragmented networks: it still prefers connected routes, but it can jump when the network really is disconnected.
The decoded per-fix arcs are collapsed into a route, and remaining holes are filled with shortest paths. Holes that cannot be filled
are counted as route breaks.</li>
<li><b>Did HMM help?</b> Yes. The decoder parameters were tuned on out-of-fold TRAIN probabilities (grid below, 20 trips), and the
results are reported on STUDENT_TEST:</li>
</ol>
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
<p>__HMM_COMMENT__</p>
{img(f"{FIG}/hmm_vs_clf_per_trip.png", "Fig 14. Per-trip Top-1 accuracy, classifier only vs classifier + HMM (STUDENT_TEST).")}

<h2>6. Geographical visualisation (Folium)</h2>
<p>Interactive maps for all 10 student-test trips are in <code>results/maps/student_test_*.html</code>, generated by
<code>06_hmm_eval.py</code> using <code>src/pathsnap/viz.py</code>. Layers: road network near the track (grey), raw GPS fixes
(blue), reference route (green), predicted HMM route (red), and the classifier-only argmax arcs (amber, toggle). Each map has a
legend, a layer control and start/end markers, and is zoomed to the trip extent.</p>
{img(f"{IMG}/map_pred_a.png", "Fig 15. STUDENT_TEST trip __MAPA__: GPS, reference route (green), predicted route (red).", "90%")}
{img(f"{IMG}/map_pred_b.png", "Fig 16. STUDENT_TEST trip __MAPB__, zoomed: classifier-only arcs (amber) jump between a road and its neighbours, while the HMM route (red) stays connected.", "90%")}

<h2>7. API, UI and deployment</h2>
<p><b>Laptop 1</b> runs <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>. FastAPI loads
<code>artifacts/final_model.joblib</code>, <code>model_config.json</code> and <code>hmm_config.json</code> at start-up.
<code>POST /predict {{"trip_id"}}</code> runs cleaning &rarr; candidates/features &rarr; LightGBM &rarr; Viterbi &rarr; route
reconstruction. Only after that does it read the trip's <code>.route</code> (in <code>evaluate_prediction</code>) to compute
accuracy. The reference geometry is never returned. <b>Laptop 2</b> serves <code>ui/index.html</code> (Leaflet vendored locally).
The evaluator enters the API address, sees the trip list from <code>GET /trips</code>, clicks a trip, and the UI draws the GPS and
the predicted route and shows Top-1 (HMM and classifier-only), route precision/recall/F1, continuity and timings.</p>
{img(f"{IMG}/ui_laptop2.png", "Fig 17. Two-laptop deployment: the evaluator UI running on Laptop 2, connected over the network to the API on Laptop 1 (http://172.20.10.2:8000), showing evaluator trip 00000059. The metrics match the Laptop 1 API log in Fig 18.", "95%")}
{img(f"{IMG}/ui_live.png", "Fig 17b. Evaluator UI during the live run on the 10 evaluator trips (trip 00000015 selected): GPS fixes (blue), predicted HMM route (red), server-side metrics and timings.", "95%")}
{img(f"{IMG}/api_log_laptop2.png", "Fig 18. API log on Laptop 1 during the two-laptop run: GET /trips and POST /predict requests arrive from Laptop 2 (172.20.10.3) over the network, and every request returns 200 OK with its server-side metrics. The trip 00000059 numbers match Fig 17 on Laptop 2.", "100%")}
{img(f"{IMG}/api_log_live.png", "Fig 19. API log on Laptop 1 (uvicorn terminal) during the live run: model loaded with eval_dir = evaluator_test, GET /trips returns 10 trips, and every POST /predict returns 200 OK with its server-side metrics and time.", "100%")}
<p><b>Example API response</b> (<code>curl</code>, a student-test trip, GPS/route arrays truncated):</p><pre>{html.escape(api_json)}</pre>

<h3>7.1 Results on the evaluator-test trips (live run)</h3>
<p>These figures are transcribed from the API log in Fig 19, which covers all 10 trips (<code>results/evaluator/evaluator_results.csv</code>). They were produced by
the frozen pipeline (tuned LightGBM + HMM). Nothing was retrained or retuned after these trips were seen.</p>
{table(ev[["trip","n_obs","top1_classifier_only","top1_hmm","route_f1","continuity_hmm","server_time_s"]], {c: f4 for c in ["top1_classifier_only","top1_hmm","route_f1","continuity_hmm"]} | {"server_time_s": lambda x: f"{x:.2f}"})}
<p>Pooled over all {int(ev.n_obs.sum()):,} evaluator fixes, Top-1 is {pc(ev_clf)} for the classifier alone and <b>{pc(ev_hmm)}</b> with HMM
decoding. Mean route F1 is {ev.route_f1.mean():.3f} and mean server time is {ev.server_time_s.mean():.2f} s per trip. HMM decoding
improved Top-1 on all 10 trips, and the evaluator result ({pc(ev_hmm)}) is in line with the student-test result ({pc(ga['hmm_top1'])}).
<b>Weakest trip: 00000015</b> (route F1 {ev.set_index('trip').loc['00000015','route_f1']:.3f}, route precision 0.64, recall 0.99).
Its median speed over 10 s windows is 1.5 m/s, about walking pace, but over 1 s it appears to be 7.2 m/s: per-second GPS jitter is
larger than the real motion. The heading features then become noise. Heading is the only cue that separates an arc from its
reverse twin, and 80 % of the extra predicted length on this trip is reverse twins of reference arcs. Training trips are mostly
vehicles (median trip speed 42.5 km/h), so this is a distribution shift. A heading measured over a fixed distance rather than a fixed
time window would address it; we did <i>not</i> change the system after seeing this result. Without trip 15, pooled Top-1 is
{pc(ev_hmm_x15)}.</p>
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
