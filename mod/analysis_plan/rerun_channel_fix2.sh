#!/bin/bash
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3
until grep -q RERUN1_DONE results/rerun_chain.log; do sleep 60; done
mv results/r5_window_split/GBM results/r5_window_split/GBM_oldorder 2>/dev/null
python - <<'PY' > results/rerun_r5_gbm.log 2>&1
import json, numpy as np
from joblib import Parallel, delayed
from r5_window_split import run, OUT
import os
res = Parallel(n_jobs=3)(delayed(run)("GBM", k) for k in range(10))
json.dump(res, open(os.path.join(OUT, "r5_window_split_gbm_fixed.json"), "w"), indent=1); print("GBM window-level mean", np.mean([r["macro_auc"] for r in res]))
PY
mv results/a7_learning_curve.json results/a7_learning_curve_oldorder.json 2>/dev/null
python a7_learning_curve.py > results/rerun_a7.log 2>&1 || echo FAILED_a7
python fig_leakage_new.py > results/rerun_fig_leakage.log 2>&1 || echo FAILED_leak
echo RERUN2_DONE
