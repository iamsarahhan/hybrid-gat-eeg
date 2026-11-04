"""
R1-M3: rebuild Figure 5 (macro-average ROC, GAT vs SVM vs CNN vs MLP) from the
final patient-grouped, optuna-tuned results. Supersedes
review/analysis/figure5_macro_roc.py (old leaky split, untuned baselines,
pre-edge_dim-fix GAT).

No training -- pure post-hoc computation from existing y_true/y_score.npy per fold.
"""
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("GAT hybrid (ours)", os.path.join(REPO_ROOT, 'patient_split', 'result_4class_patient_hybrid_auc'), "#2a78d6"),
    ("SVM",               os.path.join(REPO_ROOT, 'patient_split', 'result_SVM_patient_auc'),           "#1baf7a"),
    ("CNN",               os.path.join(REPO_ROOT, 'patient_split', 'result_CNN_patient_auc'),           "#eda100"),
    ("MLP",               os.path.join(REPO_ROOT, 'patient_split', 'result_MLP_patient_auc'),           "#e34948"),
]
N_CLASSES = 4
CLASS_NAMES = ["SEIZURE", "LPD", "GPD", "OTHERS"]

fig, ax = plt.subplots(figsize=(6.2, 6), dpi=150)
mean_fpr = np.linspace(0, 1, 200)

results = []
for name, result_dir, color in MODELS:
    all_true, all_score = [], []
    for i in range(1, 11):
        y_true = np.load(os.path.join(result_dir, f'fold_{i}', 'y_true.npy')).flatten()
        y_score = np.load(os.path.join(result_dir, f'fold_{i}', 'y_score.npy'))
        all_true.append(y_true)
        all_score.append(y_score)
    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score, axis=0)

    tprs, aucs = [], []
    for c in range(N_CLASSES):
        y_true_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, c])
        aucs.append(auc(fpr, tpr))
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    macro_tpr = np.mean(tprs, axis=0)
    macro_tpr[0] = 0.0
    macro_auc = np.mean(aucs)

    ax.plot(mean_fpr, macro_tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {macro_auc:.3f})")
    results.append((name, macro_auc, aucs))

ax.plot([0, 1], [0, 1], color="#9a9990", linestyle="--", linewidth=1)
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set_title("Macro-average ROC: GAT hybrid vs. baselines\n(patient-grouped, pooled 10-fold out-of-fold predictions)",
              fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'figure5_macro_roc.png'), facecolor="white")

print("Model              Macro AUC   Per-class AUC (SEIZURE, LPD, GPD, OTHERS)")
for name, macro_auc, aucs in results:
    print(f"{name:18s} {macro_auc:.4f}     " + ", ".join(f"{a:.4f}" for a in aucs))

with open(os.path.join(OUT_DIR, 'figure5_macro_roc_result.txt'), 'w') as f:
    f.write("Model              Macro AUC   Per-class AUC (SEIZURE, LPD, GPD, OTHERS)\n")
    for name, macro_auc, aucs in results:
        f.write(f"{name:18s} {macro_auc:.4f}     " + ", ".join(f"{a:.4f}" for a in aucs) + "\n")

print(f"\nSaved: {os.path.join(OUT_DIR, 'figure5_macro_roc.png')}")
print(f"Saved: {os.path.join(OUT_DIR, 'figure5_macro_roc_result.txt')}")
