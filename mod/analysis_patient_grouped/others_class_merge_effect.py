"""
R1-M1: OTHERS class merge effect, recomputed for the final patient-grouped
GAT hybrid. Supersedes review/analysis/others_class_merge_effect.py.

Uses only existing saved artifacts (no retraining):
  - patient_split/fold_splits_patient.json : pre-merge original labels_str
  - patient_split/result_4class_patient_hybrid_auc/fold_i/y_pred.npy : post-merge predictions
"""
import os
import json
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_PATH = os.path.join(REPO_ROOT, 'patient_split', 'fold_splits_patient.json')
RESULT_DIR = os.path.join(REPO_ROOT, 'patient_split', 'result_4class_patient_hybrid_auc')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_MAP = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "OTHERS": 3}
CLASS_NAMES = ["SEIZURE", "LPD", "GPD", "OTHERS"]
SUBTYPES = ["LRDA", "GRDA", "OTHER"]

with open(SPLIT_PATH) as f:
    splits = json.load(f)

all_labels_str, all_pred = [], []
for i in range(1, 11):
    y_pred = np.load(os.path.join(RESULT_DIR, f'fold_{i}', 'y_pred.npy')).flatten()
    labels_str = splits[i - 1]["test"]["labels_str"]
    assert len(labels_str) == len(y_pred)
    all_labels_str.extend(labels_str)
    all_pred.extend(y_pred)

all_labels_str = np.array(all_labels_str)
all_pred = np.array(all_pred)

lines = []
lines.append("=== R1-M1: OTHERS class merge effect (GAT hybrid, pooled 10-fold out-of-fold predictions) ===\n")
lines.append(f"{'Subtype':<10}{'n':>8}   Recall (predicted OTHERS)   Predicted SEIZURE / LPD / GPD (misroute)")
lines.append("-" * 90)

per_subtype = {}
for sub in SUBTYPES:
    mask = all_labels_str == sub
    n = int(mask.sum())
    preds = all_pred[mask]
    recall = (preds == LABEL_MAP["OTHERS"]).mean()
    misroute = {c: (preds == LABEL_MAP[c]).mean() for c in ["SEIZURE", "LPD", "GPD"]}
    per_subtype[sub] = {"n": n, "recall": float(recall), "misroute": misroute}
    lines.append(
        f"{sub:<10}{n:>8}   {recall:>8.4f}                   "
        f"SEIZURE={misroute['SEIZURE']:.4f}  LPD={misroute['LPD']:.4f}  GPD={misroute['GPD']:.4f}"
    )

mask_merged = np.isin(all_labels_str, SUBTYPES)
merged_recall = (all_pred[mask_merged] == LABEL_MAP["OTHERS"]).mean()
lines.append("-" * 90)
lines.append(f"{'MERGED':<10}{int(mask_merged.sum()):>8}   {merged_recall:>8.4f}"
             f"                   (should match this run's OTHERS recall in Table 2)")

report = "\n".join(lines)
print(report)

with open(os.path.join(OUT_DIR, 'others_class_merge_effect_result.txt'), 'w') as f:
    f.write(report + "\n")

with open(os.path.join(OUT_DIR, 'others_class_merge_effect_result.json'), 'w') as f:
    json.dump({
        "per_subtype": per_subtype,
        "merged_recall": float(merged_recall),
        "merged_n": int(mask_merged.sum()),
    }, f, indent=2)

print(f"\nSaved: {os.path.join(OUT_DIR, 'others_class_merge_effect_result.txt')}")
print(f"Saved: {os.path.join(OUT_DIR, 'others_class_merge_effect_result.json')}")
