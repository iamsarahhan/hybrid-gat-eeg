# Hybrid graph attention network for critical-care EEG (HMS)

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

## Figures

Figures are regenerated by the `fig_*.py` scripts in `mod/analysis_plan/` into `mod/analysis_plan/results/figs_v5/`.

## License and citation

MIT License. Please cite the manuscript (see `CITATION.cff`).
