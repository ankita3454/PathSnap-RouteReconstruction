# PathSnap: Route Reconstruction (Map Matching)

UE24CS352A Machine Learning hackathon, PES University.

PathSnap takes a noisy GPS trajectory and recovers the road segments the vehicle actually drove on, in four stages:

1. **Candidate generation**: every GPS fix gets the road arcs within 50 m (at most 8).
2. **Candidate scoring**: a supervised classifier gives each (fix, arc) pair `P(arc is correct)`. Seven models were compared and LightGBM was tuned with Bayesian optimisation.
3. **HMM / Viterbi decoding**: the classifier probabilities are the emissions. Transitions compare network driving distance with straight-line GPS distance, so the most likely route that is also connected gets decoded.
4. **Serving**: a FastAPI backend (Laptop 1) and a Leaflet evaluator UI (Laptop 2).

The detailed results are in `report/` and `results/`.

## Repository layout

```
src/pathsnap/        library code
  io.py              raw .track/.nodes/.arcs/.route readers
  geo.py             projection, GPS cleaning, network cropping, KD-tree candidate index
  labels.py          monotonic DP alignment of GPS fixes to the reference route (per-fix labels)
  features.py        candidate generation + 21 engineered features
  models.py          the 7 model definitions
  metrics.py         Top-1, macro-F1, log loss, continuity, route P/R/F1
  hmm.py             HMM, Viterbi, shortest-path transitions, route reconstruction
  pipeline.py        end-to-end inference used by the API (never reads .route)
  viz.py             Folium maps
scripts/             numbered, reproducible experiment steps (see below)
api/main.py          FastAPI backend
ui/index.html        evaluator UI (Leaflet is vendored in ui/vendor, so no CDN is needed)
artifacts/           final_model.joblib, model_config.json, hmm_config.json, models/*.joblib
results/             every CSV/JSON log, figures/, maps/ (interactive Folium HTML)
report/              technical report (PDF) and its generator
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The raw data is not committed. Point `PATHSNAP_DATA` at the folder that contains `train/` and `student_test/`, or symlink it as `./data`:

```bash
ln -s "/path/to/Student folder" data
```

## Reproduce the experiments

```bash
bash scripts/run_all.sh        # about 45 min on a 2-core machine
```

| step | script | output |
|---|---|---|
| 1 | `01_build_dataset.py` | cleaning, cropped networks, per-fix labels, wide candidate features |
| 2 | `02_eda.py` | `results/eda/*`, EDA figures, Folium maps, radius/K choice, anomalous-trip exclusion |
| 3 | `03_make_model_tables.py` | applies radius 50 m / K = 8 |
| 4 | `04_train_models.py`, `04b_train_fit_metrics.py` | 7 models, 5-fold GroupKFold CV, student-test metrics, train-fit metrics |
| 5 | `05_bayes_opt.py` | Optuna TPE search (`results/bayes/optuna_trials.csv`), tuned vs untuned, final artifact |
| 6 | `06_hmm_eval.py` | decoder tuning on out-of-fold train, HMM vs classifier-only, student-test maps |
| 7 | `07_plots.py` | comparison figures, consolidated table, error analysis |

Random seed: 42 everywhere. The evaluator-test partition is not used anywhere in training, tuning or model selection.

## Deployment (two laptops)

**Laptop 1: model + API**

```bash
export PATHSNAP_EVAL_DIR="/path/to/evaluator_test"     # folder with the 10 reserved trip sub-folders
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Find the laptop's LAN IP with `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows).
- Every request is logged to this terminal: trip, number of fixes and candidates, Top-1, route F1, continuity and time. Show this terminal to the evaluator.
- If `PATHSNAP_EVAL_DIR` is not set, the API serves `data/student_test`.

**Laptop 2: UI**

```bash
cd ui && python3 -m http.server 5500
# open http://localhost:5500, enter http://<laptop1-ip>:8000, press Connect
```

The UI lists the trips in `PATHSNAP_EVAL_DIR`. Click a trip and the UI sends `POST /predict {"trip_id": ...}`, then draws the raw GPS and the predicted route and shows the metrics.

The ground-truth `.route` file is read only in `PathSnapPipeline.evaluate_prediction`, after the prediction is finished. The reference geometry is never sent to the UI.

### API

| method | path | body | returns |
|---|---|---|---|
| GET | `/health` | none | status, model, eval dir, number of trips |
| GET | `/trips` | none | list of trip ids |
| POST | `/predict` | `{"trip_id": "00000052"}` | GPS points, predicted route polylines, Top-1 (HMM and classifier-only), route precision/recall/F1, continuity, timings |

Test it from a terminal:

```bash
curl http://<laptop1-ip>:8000/trips
curl -X POST http://<laptop1-ip>:8000/predict -H 'Content-Type: application/json' -d '{"trip_id":"00000052"}'
```
