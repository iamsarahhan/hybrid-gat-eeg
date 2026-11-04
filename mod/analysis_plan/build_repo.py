"""Assemble a GitHub-ready repository (code + splits + out-of-fold predictions + summary results + figures + manuscript)
from the working folder D:\\eeg_seizure into D:\\eeg_seizure\\hybrid-gat-cceeg. Large raw/derived data are excluded."""
import os, re, shutil, json, glob
SRC = r"D:\eeg_seizure"; DST = os.path.join(SRC, "hybrid-gat-cceeg")
if os.path.exists(DST): shutil.rmtree(DST)
def cp(rel, dst_rel=None):
    s = os.path.join(SRC, rel); d = os.path.join(DST, dst_rel or rel); os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d)
def cpglob(pattern, exclude=()):
    for s in glob.glob(os.path.join(SRC, pattern)):
        rel = os.path.relpath(s, SRC)
        if any(x in rel for x in exclude) or os.path.isdir(s): continue
        cp(rel)
# ---- training code + split + out-of-fold predictions (no checkpoints) ----
for f in ["DataLoader_4class_patient_hybrid.py", "GAT_4class_patient_hybrid_auc.py", "graph_refinement.py", "fold_splits_patient.json"]: cp(f"patient_split/{f}")
for v in ["result_4class_patient_hybrid_auc", "result_4class_patient_softlabel_hybrid_auc", "result_4class_patient_gmmgumbel_hybrid_auc"]:
    for f in ["hyperparameters.txt", "average_results.txt"]:
        if os.path.exists(os.path.join(SRC, "patient_split", v, f)): cp(f"patient_split/{v}/{f}")
    for k in range(1, 11):
        for f in ["y_true.npy", "y_pred.npy", "y_score.npy", "metrics.txt"]:
            p = f"patient_split/{v}/fold_{k}/{f}"
            if os.path.exists(os.path.join(SRC, p)): cp(p)
# ---- handover analyses ----
cpglob("mod/analysis_patient_grouped/*.py"); cpglob("mod/analysis_patient_grouped/*.txt"); cpglob("mod/analysis_patient_grouped/*.json"); cpglob("mod/analysis_patient_grouped/*.png")
for f in ["attention_rollout_class_matrices.npz", "seed_stability_permutation_null_rollouts.npz", "patient_clustered_ci_bootstrap.npz"]: cp(f"mod/analysis_patient_grouped/{f}")
PQ = "['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']"
NOTE = "  # channel order of the stored psd_sb/fc_sb files = HMS parquet column order (corrected 29 Oct 2026; the 10-20 list used before mislabelled electrodes)"
for f, var in [("attention_rollout_and_causal_ablation.py", "CHANNELS"), ("extra_analyses_oof.py", "CH")]:
    p = os.path.join(DST, "mod", "analysis_patient_grouped", f); s = open(p, encoding="utf-8").read()
    s2 = re.sub(rf"^{var} = \[.*?\]", f"{var} = {PQ}{NOTE}", s, count=1, flags=re.S | re.M); assert s2 != s, f; open(p, "w", encoding="utf-8").write(s2)
# ---- analysis plan code + summary results + predictions ----
cpglob("mod/analysis_plan/*.py", exclude=("build_repo.py",)); cpglob("mod/analysis_plan/*.sh")
cpglob("mod/analysis_plan/results/*.json", exclude=("_oldorder",)); cpglob("mod/analysis_plan/results/*.png")
for d in ["r1_baselines", "r2_gat_controls", "r2_gat_window_split", "a6_surrogate", "r5_window_split", "r6_center10s", "figs_v5"]:
    for s in glob.glob(os.path.join(SRC, "mod/analysis_plan/results", d, "**", "*"), recursive=True):
        if os.path.isdir(s) or s.endswith(".pth") or "_oldorder" in s: continue
        cp(os.path.relpath(s, SRC))
# ---- manuscript ----
os.makedirs(os.path.join(DST, "manuscript", "figures"), exist_ok=True)
for f in ["EEG_GAT_manuscript_v20.docx", "EEG_GAT_manuscript_v20_clean.docx"]: shutil.copy2(os.path.join(SRC, "manuscript_v3", f), os.path.join(DST, "manuscript", f))
for s in glob.glob(os.path.join(SRC, "manuscript_v3", "figs_v5", "*.png")): shutil.copy2(s, os.path.join(DST, "manuscript", "figures", os.path.basename(s)))
SCR = r"C:\Users\User\AppData\Local\Temp\claude\D--eeg-seizure\61974db0-f6ca-4539-af08-8bc7b49e42a0\scratchpad"
os.makedirs(os.path.join(DST, "manuscript", "build"), exist_ok=True)
for f in ["build_docx_v9.py", "v20_body.json"]: shutil.copy2(os.path.join(SCR, f), os.path.join(DST, "manuscript", "build", f))
# ---- root files ----
shutil.copy2(os.path.join(SRC, "requirements.txt"), os.path.join(DST, "requirements-train.txt"))
open(os.path.join(DST, "requirements-analysis.txt"), "w").write("""# CPU analysis environment used for mod/analysis_plan (Python 3.12)
numpy>=2.0
pandas>=2.2
scipy>=1.14
scikit-learn>=1.7
statsmodels>=0.14
matplotlib>=3.9
joblib
pyarrow
torch>=2.4      # CPU build is sufficient for r2_gat_controls_cpu.py
torch-geometric>=2.6
python-docx
beautifulsoup4
""")
open(os.path.join(DST, ".gitignore"), "w").write("""# raw and derived data (download/compute locally; see README)
train.csv
train_eegs.zip
train_eegs/
psd_sb/
fc_sb/
psd_sb.zip
fc_sb.zip
*.pth
mod/analysis_plan/results/graph_cache.pt
mod/analysis_plan/results/r1_flat_features.npz
mod/analysis_plan/results/r6_features.npz
mod/analysis_plan/results/r3_features/
mod/analysis_plan/results/a6_features.csv
mod/analysis_plan/results/a4_asymmetry_features.csv
mod/analysis_plan/results/*_oldorder*
mod/analysis_plan/results/*.log
__pycache__/
*.pyc
.ipynb_checkpoints/
venv/
.venv/
""")
open(os.path.join(DST, "LICENSE"), "w").write("""MIT License

Copyright (c) 2026 Hyun-Jee Han, Donghyeok Jo, Minju Kang, SangWook Lee, Hakseung Kim, Hyub Huh, Dong-Joo Kim

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
""")
open(os.path.join(DST, "CITATION.cff"), "w", encoding="utf-8").write("""cff-version: 1.2.0
message: "If you use this code or the patient-grouped fold assignment, please cite the manuscript below."
title: "Hybrid graph attention network over multi-measure functional connectivity for seizures and periodic discharges in critical-care EEG"
authors:
  - family-names: Han
    given-names: Hyun-Jee
  - family-names: Jo
    given-names: Donghyeok
  - family-names: Kang
    given-names: Minju
  - family-names: Lee
    given-names: SangWook
  - family-names: Kim
    given-names: Hakseung
  - family-names: Huh
    given-names: Hyub
  - family-names: Kim
    given-names: Dong-Joo
version: "1.0.0"
date-released: "2026-11-02"
license: MIT
""")
open(os.path.join(DST, "CHANGELOG.md"), "w", encoding="utf-8").write("""# Changelog

## 2026-11-02 — repository assembled for submission
- Code, patient-grouped fold assignment, out-of-fold predictions, summary statistics, figures and manuscript v20.

## 2026-10-29 — electrode-order correction
- The stored `psd_sb/` and `fc_sb/` matrices index channels in the HMS parquet column order
  `Fp1 F3 C3 P3 F7 T3 T5 O1 Fz Cz Pz Fp2 F4 C4 P4 F8 T4 T6 O2`, not the 10–20 order assumed earlier.
  Verified against the raw EEG (raw correlation of the parquet-ordered segment reproduces the stored matrices, r 0.6–1.0).
- Model performance is unaffected (ordering is consistent inside the pipeline). Every electrode-labelled output was regenerated:
  attention hubs (Fp2 seizure, T6 LPD, Fz GPD/Others), topomaps, hemispheric asymmetry, graph-theory features/GBM, learning curve.
- `common.py`, `attention_rollout_and_causal_ablation.py` and `extra_analyses_oof.py` now carry the corrected list.

## 2026-10-21 — corrected pipeline (handover)
- Patient-grouped StratifiedGroupKFold, GATConv edge_dim=4 fix, hybrid graph + flat-connectivity architecture, fixed seeds,
  corrected DeLong test. Earlier window-level results (macro AUROC ≈0.97) were leaky and are not used anywhere.
""")
open(os.path.join(DST, "README.md"), "w", encoding="utf-8").write("""# Hybrid graph attention network for critical-care EEG (HMS)

Code, patient-grouped fold assignment, out-of-fold predictions and analysis outputs for the manuscript
*A hybrid graph attention network over multi-measure functional connectivity improves discrimination of seizures and periodic discharges*
(Han, Jo, Kang, Lee, Kim, Huh, Kim; Korea University, 2026).

The model classifies 50-s, 19-channel windows of the HMS Harmful Brain Activity Classification dataset into seizure, LPD, GPD and Others.
It fuses a **network pathway** (four edge-attributed GAT layers over per-channel Welch spectra, edges carrying Pearson correlation,
coherence, amplitude-envelope correlation and phase-locking value) with a **matrix pathway** (a linear encoder of the flattened
connectivity matrices), and is trained either on the modal expert label or on the normalized expert vote distribution.
All evaluation is patient-grouped 10-fold cross-validation with patient-resampled uncertainty.

## Repository layout

| Path | Contents |
|---|---|
| `patient_split/` | Training code (`GAT_4class_patient_hybrid_auc.py`, `DataLoader_4class_patient_hybrid.py`, `graph_refinement.py`), the single patient-grouped split `fold_splits_patient.json`, and per-fold out-of-fold predictions (`result_*/fold_k/y_true.npy, y_pred.npy, y_score.npy`) for the hard-label, soft-label and GMM/Gumbel variants, plus the tuned hyper-parameters |
| `mod/analysis_patient_grouped/` | Handover analyses on the trained models: corrected DeLong tests, BH correction, patient-clustered CI, input ablations, leave-one-measure-out, TOST, attention rollout + causal ablation + seed stability + label-permutation null, soft-label calibration, Others-subtype recall (with their result files) |
| `mod/analysis_plan/` | Analyses added for the manuscript: `common.py` (loaders, channel order), `a1`–`a8` (vote-fraction relation, simulated annotator, patient-level error, hemispheric asymmetry, hub perturbation, graph-theory GBM baseline, learning curve, periodicity), `r1` (retrained SVM/MLP), `r2` (topology controls and window-level GAT on CPU), `r3`/`r6` (central-10-s and time-axis features), `r4` (patient-paired comparisons), `r5` (window-level GBM/MLP), `v5`/`v7` (statistics, calibration, operating points), `fig_*.py` (all figures), `results/` (summary JSON, per-fold predictions of every comparator and control, figures) |
| `manuscript/` | `EEG_GAT_manuscript_v20.docx` (revision marks), `EEG_GAT_manuscript_v20_clean.docx` (submission copy), `figures/`, and the build script with the manuscript body |

## Data

The EEG data are public: [HMS – Harmful Brain Activity Classification (Kaggle)](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).
Download `train.csv` and `train_eegs/` into the repository root (both are git-ignored). The analysed subset is the 36,000 windows
(9,000 per class; 3,000 each of LRDA, GRDA and Other inside *Others*) listed in `patient_split/fold_splits_patient.json`.

The per-window features used for training (`psd_sb/<file>.npy`, 19×101 log-free Welch PSD; `fc_sb/{corr,coh,aec,plv}/<file>.npy`, 19×19)
are derived data (about 2.4 GB) and are not versioned here. They are produced by the feature-extraction script of the training pipeline
(pre-processing: 1–40 Hz causal 5th-order Butterworth, ±500 µV clipping, 3-SD interpolation, per-channel z-normalization; see Methods 2.2–2.4).
**Channel order inside these files is the parquet column order** `Fp1 F3 C3 P3 F7 T3 T5 O1 Fz Cz Pz Fp2 F4 C4 P4 F8 T4 T6 O2` (see `CHANGELOG.md`).

## Environment

Training (GPU, Python 3.11): see the header of `requirements-train.txt` (PyTorch 2.8 cu128, PyTorch Geometric 2.7).
Analyses (CPU, Python 3.12): `pip install -r requirements-analysis.txt`.

## Reproducing the results

1. **Training the hybrid GAT** (GPU): `python patient_split/GAT_4class_patient_hybrid_auc.py` writes `patient_split/result_*/fold_k/`.
   Hyper-parameters are in `patient_split/result_*/hyperparameters.txt`.
2. **Handover statistics**: run the scripts in `mod/analysis_patient_grouped/` from the repository root; each writes a `*_result.txt` / `*.json`.
3. **Manuscript analyses** (from `mod/analysis_plan/`, in this order):
   `a6_graph_features_surrogate.py` → `a1 … a5, a7` → `r1_baselines_sklearn.py` →
   `r2_gat_controls_cpu.py --cond {hybrid,graph_only,flat_only,edge_const,edge_shuffle} --folds 1 2 3 --epochs 30 --patience 6 --batch 64`
   and `--split window` → `r5_window_split.py` → `r3_center10s_and_dfc_features.py` → `r6_center10s_compare.py` →
   `r4_compare_all.py` → `v5_figures_and_stats.py`, `v7_review_items.py` → `fig_*.py`.
   Summary outputs land in `mod/analysis_plan/results/` (`r4_comparison.json`, `v5_stats.json`, `v7_review_items.json`, `r6_center10s.json`, …).
4. **Manuscript**: `cd manuscript/build && BODY=v20_body.json OUTF=out.docx python build_docx_v9.py` (set `CLEAN=1` for an all-black copy).

## Key results (patient-grouped, 36,000 windows, 1,752 patients)

| Model | Macro AUROC [patient-clustered 95% CI] |
|---|---|
| Hybrid GAT, hard labels | 0.839 [0.817–0.856] |
| Hybrid GAT, soft labels | 0.842 [0.821–0.857] |
| SVM (RBF), same features | 0.819 [0.798–0.837] |
| MLP, same features | 0.782 [0.760–0.805] |
| Graph-theory features + GBM | 0.777 [0.758–0.797] |

Under a window-level split the same pipeline scores 0.93–0.98. Topology controls on folds 1–3 (532 patients): constant edges Δ = 0.000
[−0.002, +0.002], edge-shuffled Δ = +0.001 [−0.003, +0.006]; the fusion gain is complementarity of spectral and connectivity representations,
not topology-specific learning.

## License and citation

MIT License. Please cite the manuscript (see `CITATION.cff`).
""")
# tree summary
n = 0; size = 0
for root, dirs, files in os.walk(DST):
    for f in files: n += 1; size += os.path.getsize(os.path.join(root, f))
print("files", n, "total MB", round(size / 1e6, 1))
big = sorted(((os.path.getsize(os.path.join(r, f)) / 1e6, os.path.relpath(os.path.join(r, f), DST)) for r, _, fs in os.walk(DST) for f in fs), reverse=True)[:8]
print("largest:", [(round(s, 1), p) for s, p in big])
