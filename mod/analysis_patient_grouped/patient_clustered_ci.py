"""
Figure 4e reproduction, recomputed for the final patient-grouped GAT (hybrid:
GATConv edge_dim=4 graph branch + flat-FC branch, see patient_split/
GAT_4class_patient_hybrid_auc.py). Supersedes review/analysis/patient_clustered_ci.py,
which used the old leaky (non-patient-grouped) split and pre-edge_dim-fix GAT.

Uses only already-saved artifacts (no retraining):
  - patient_split/fold_splits_patient.json : test-set files + patient_id per fold
  - patient_split/result_4class_patient_hybrid_auc/fold_i/{y_true,y_pred}.npy

Pools all 10 folds' held-out test predictions, then compares:
  (a) segment-level bootstrap: resample individual segments
  (b) patient-clustered bootstrap: resample patients (all of a resampled patient's
      segments move together), respecting within-patient dependence.
"""
import os
import json
import numpy as np

RNG_SEED = 42
N_BOOT = 10000
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_PATH = os.path.join(REPO_ROOT, 'patient_split', 'fold_splits_patient.json')
RESULT_DIR = os.path.join(REPO_ROOT, 'patient_split', 'result_4class_patient_hybrid_auc')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng(RNG_SEED)

with open(SPLIT_PATH) as f:
    splits = json.load(f)

all_true, all_pred, all_patients = [], [], []
for i, fold in enumerate(splits, start=1):
    y_true = np.load(os.path.join(RESULT_DIR, f'fold_{i}', 'y_true.npy')).flatten()
    y_pred = np.load(os.path.join(RESULT_DIR, f'fold_{i}', 'y_pred.npy')).flatten()
    patient_ids = np.array(fold['test']['patient_id'])
    assert len(patient_ids) == len(y_true) == len(y_pred), f"fold {i} length mismatch"
    assert np.array_equal(y_true, np.array(fold['test']['labels_int'])), f"fold {i} order mismatch"
    all_true.append(y_true)
    all_pred.append(y_pred)
    all_patients.append(patient_ids)

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)
patient_ids = np.concatenate(all_patients)
correct = (all_true == all_pred).astype(np.int64)

N = len(correct)
point_estimate = correct.mean()
print(f"Pooled out-of-fold segments: {N}")
print(f"Unique patients: {len(np.unique(patient_ids))}")
print(f"Point estimate accuracy: {point_estimate:.4f}")


def segment_bootstrap(correct, n_boot, rng, chunk=1000):
    N = len(correct)
    accs = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, N, size=(b, N), dtype=np.int32)
        accs[done:done + b] = correct[idx].mean(axis=1)
        done += b
    return accs


def patient_clustered_bootstrap(correct, patient_ids, n_boot, rng):
    patients, inverse = np.unique(patient_ids, return_inverse=True)
    n_patients = len(patients)
    correct_sum = np.bincount(inverse, weights=correct, minlength=n_patients)
    n_seg = np.bincount(inverse, minlength=n_patients)

    sample_idx = rng.integers(0, n_patients, size=(n_boot, n_patients))
    total_correct = correct_sum[sample_idx].sum(axis=1)
    total_n = n_seg[sample_idx].sum(axis=1)
    return total_correct / total_n


print(f"\nRunning segment-level bootstrap ({N_BOOT} resamples)...")
seg_accs = segment_bootstrap(correct, N_BOOT, rng)
seg_ci = np.percentile(seg_accs, [2.5, 97.5])
seg_width = seg_ci[1] - seg_ci[0]

print(f"Running patient-clustered bootstrap ({N_BOOT} resamples)...")
pc_accs = patient_clustered_bootstrap(correct, patient_ids, N_BOOT, rng)
pc_ci = np.percentile(pc_accs, [2.5, 97.5])
pc_width = pc_ci[1] - pc_ci[0]

ratio = pc_width / seg_width

report = f"""
=== Patient-Clustered vs Segment-Level Bootstrap CI (GAT hybrid, pooled 10-fold out-of-fold predictions) ===

Model: GAT hybrid (GATConv edge_dim=4 graph branch + flat-FC branch, patient-grouped, optuna AUC-tuned)
N segments               : {N}
N unique patients        : {len(np.unique(patient_ids))}
Point estimate accuracy  : {point_estimate:.4f}

Segment-level bootstrap:
  95% CI                 : [{seg_ci[0]:.4f}, {seg_ci[1]:.4f}]
  width                  : {seg_width:.4f}

Patient-clustered bootstrap:
  95% CI                 : [{pc_ci[0]:.4f}, {pc_ci[1]:.4f}]
  width                  : {pc_width:.4f}

Width ratio (clustered / segment): {ratio:.2f}x

NOTE: unlike the superseded review/analysis/patient_clustered_ci.py, this split
(patient_split/fold_splits_patient.json, StratifiedGroupKFold on patient_id) IS
patient-grouped -- no patient's segments appear in more than one of
train/val/test within a fold. This analysis therefore reflects both the
corrected statistical unit (patient-clustered CI) AND leakage-free predictions.
"""
print(report)

with open(os.path.join(OUT_DIR, 'patient_clustered_ci_result.txt'), 'w') as f:
    f.write(report)

np.savez(os.path.join(OUT_DIR, 'patient_clustered_ci_bootstrap.npz'),
         seg_accs=seg_accs, pc_accs=pc_accs,
         seg_ci=seg_ci, pc_ci=pc_ci, point_estimate=point_estimate)

print(f"Saved: {os.path.join(OUT_DIR, 'patient_clustered_ci_result.txt')}")
print(f"Saved: {os.path.join(OUT_DIR, 'patient_clustered_ci_bootstrap.npz')}")
