# Changelog

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
