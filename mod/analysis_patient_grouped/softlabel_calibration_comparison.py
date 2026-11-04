"""
R2-5 hard-vs-soft label ablation: does training against the vote-distribution
target (patient_split/GAT_4class_patient_softlabel_hybrid_auc.py) improve
calibration over the standard hard-label hybrid, as R2-5's "over-confidence"
concern would predict, and at what AUC/accuracy cost (if any)?

  HARD = patient_split/result_4class_patient_hybrid_auc            (macro AUC 0.8347, acc 0.6150)
  SOFT = patient_split/result_4class_patient_softlabel_hybrid_auc  (macro AUC 0.8341, acc 0.6097)

No retraining -- uses existing saved y_true / y_score arrays only. DeLong
implementation reused verbatim from delong_correct.py / delong_gmmgumbel_ablation.py.

Calibration metrics (standard definitions, computed on pooled test predictions
across all 10 folds, top-1 confidence = max predicted class probability):
  - ECE (Expected Calibration Error): 15 equal-width confidence bins,
    weighted-average |accuracy(bin) - mean_confidence(bin)|.
  - Multiclass Brier score: mean over samples of sum_c (p_c - onehot(y)_c)^2.
Both are proper scoring / calibration measures that directly speak to R2-5's
"over-confidence" claim, independent of accuracy/AUC.
"""
import os
import numpy as np
from scipy import stats


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_two_model_test(y_true_bin, score_a, score_b):
    order = np.argsort(-y_true_bin, kind="stable")
    y_ord = y_true_bin[order]
    m = int(np.sum(y_ord == 1))
    preds = np.vstack([score_a[order], score_b[order]])
    aucs, cov = fast_delong(preds, m)
    auc_a, auc_b = aucs[0], aucs[1]
    var_a, var_b, cov_ab = cov[0, 0], cov[1, 1], cov[0, 1]
    se = np.sqrt(max(var_a + var_b - 2 * cov_ab, 0))
    z = (auc_a - auc_b) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, var_a, var_b, cov_ab, z, p


def ece(y_true, y_score, n_bins=15):
    """Standard top-1 ECE: bin by max predicted probability, compare to
    accuracy within each bin, weight by bin occupancy."""
    confidence = y_score.max(axis=1)
    pred = y_score.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    total = 0.0
    per_bin = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = mask.sum()
        if count == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        total += (count / n) * abs(bin_acc - bin_conf)
        per_bin.append((lo, hi, count, bin_acc, bin_conf))
    return total, per_bin


def multiclass_brier(y_true, y_score, num_classes=4):
    onehot = np.eye(num_classes)[y_true]
    return np.mean(np.sum((y_score - onehot) ** 2, axis=1))


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "HARD (argmax label)": os.path.join(REPO_ROOT, 'patient_split', 'result_4class_patient_hybrid_auc'),
    "SOFT (vote distribution)": os.path.join(REPO_ROOT, 'patient_split', 'result_4class_patient_softlabel_hybrid_auc'),
}
N_CLASSES = 4
CLASS_NAMES = ["SEIZURE", "LPD", "GPD", "OTHERS"]

if __name__ == "__main__":
    pooled = {}
    for name, d in MODELS.items():
        yt, ys = [], []
        for i in range(1, 11):
            yt.append(np.load(os.path.join(d, f'fold_{i}', 'y_true.npy')).flatten())
            ys.append(np.load(os.path.join(d, f'fold_{i}', 'y_score.npy')))
        pooled[name] = (np.concatenate(yt), np.concatenate(ys, axis=0))

    a, b = "HARD (argmax label)", "SOFT (vote distribution)"
    y_true_a, score_a = pooled[a]
    y_true_b, score_b = pooled[b]
    assert np.array_equal(y_true_a, y_true_b), "y_true mismatch between models (fold split should be identical)"
    y_true = y_true_a

    lines = ["=== R2-5 ablation: hard-label vs vote-distribution soft-label training ===",
             f"\n{'='*70}\n  {a}  vs  {b}\n{'='*70}"]

    lines.append("\n--- DeLong AUC comparison ---")
    per_class_diff, per_class_var_a, per_class_var_b, per_class_cov = [], [], [], []
    for c in range(N_CLASSES):
        y_bin = (y_true == c).astype(int)
        auc_a, auc_b, var_a, var_b, cov_ab, z, p = delong_two_model_test(
            y_bin, score_a[:, c], score_b[:, c]
        )
        per_class_diff.append(auc_a - auc_b)
        per_class_var_a.append(var_a)
        per_class_var_b.append(var_b)
        per_class_cov.append(cov_ab)
        sig = "(*)" if p < 0.05 else ""
        lines.append(f"  [{CLASS_NAMES[c]:8s}] AUC HARD={auc_a:.4f} SOFT={auc_b:.4f}  "
                     f"diff(SOFT-HARD)={auc_b-auc_a:+.4f}  z={z:+.3f}  p={p:.4g} {sig}")

    macro_diff = np.mean(per_class_diff)  # HARD - SOFT
    macro_var = np.mean([va + vb - 2 * cv for va, vb, cv in
                          zip(per_class_var_a, per_class_var_b, per_class_cov)])
    macro_se = np.sqrt(max(macro_var, 0) / N_CLASSES)
    macro_z = macro_diff / macro_se if macro_se > 0 else 0.0
    macro_p = 2 * (1 - stats.norm.cdf(abs(macro_z)))
    sig = "(*)" if macro_p < 0.05 else ""
    lines.append("  ---")
    lines.append(f"  [MACRO]    diff(SOFT-HARD)={-macro_diff:+.4f}  z={-macro_z:+.3f}  p={macro_p:.4g} {sig}")

    lines.append("\n--- Calibration (ECE, 15 bins; multiclass Brier score) ---")
    for name in (a, b):
        yt, ys = pooled[name]
        e, bins = ece(yt, ys, n_bins=15)
        brier = multiclass_brier(yt, ys, N_CLASSES)
        acc = (ys.argmax(axis=1) == yt).mean()
        mean_conf = ys.max(axis=1).mean()
        lines.append(f"  [{name:25s}] ECE={e:.4f}  Brier={brier:.4f}  "
                     f"accuracy={acc:.4f}  mean_confidence={mean_conf:.4f}  "
                     f"(confidence - accuracy = {mean_conf-acc:+.4f})")

    lines.append("\nNote: per-class DeLong covariance is exact (paired, same test samples);")
    lines.append("cross-class combination still assumes independence across the 4 one-vs-rest")
    lines.append("AUCs (conservative approximation, same as delong_correct.py). ECE/Brier are")
    lines.append("point estimates on pooled test predictions, not paired-tested (no established")
    lines.append("paired test for ECE across models in this codebase).")

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(OUT_DIR, 'softlabel_calibration_comparison_result.txt'), 'w') as f:
        f.write(report + "\n")
    print(f"\nSaved: {os.path.join(OUT_DIR, 'softlabel_calibration_comparison_result.txt')}")
