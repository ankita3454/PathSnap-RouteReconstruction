#!/usr/bin/env bash
# Reproduce every experiment end to end (≈45 min on a 2-core laptop).
# Usage: PATHSNAP_DATA="/path/to/Student folder" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/01_build_dataset.py      # clean, crop network, labels, wide candidates, features
python3 scripts/02_eda.py                # EDA tables/figures/folium maps, radius/K choice, anomaly exclusion
python3 scripts/03_make_model_tables.py  # apply radius 50 m / K 8, attach arc nodes
python3 scripts/04_train_models.py       # 7 models: 5-fold GroupKFold + student-test
python3 scripts/04b_train_fit_metrics.py  # in-sample metrics for the over/under-fitting check
python3 scripts/05_bayes_opt.py          # Optuna TPE on LightGBM, tuned vs untuned, final artifact
python3 scripts/06_hmm_eval.py           # decoder tuning on OOF train, HMM vs classifier-only, maps
python3 scripts/07_plots.py              # comparison figures
