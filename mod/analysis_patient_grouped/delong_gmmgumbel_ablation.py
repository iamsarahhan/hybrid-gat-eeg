"""
R1-1 on/off ablation: does the GMM+Gumbel edge-refinement module
(patient_split/graph_refinement.py) actually help over the plain hybrid GAT?

  OFF = patient_split/result_4class_patient_hybrid_auc            (macro AUC 0.8347)
  ON  = patient_split/result_4class_patient_gmmgumbel_hybrid_auc  (macro AUC 0.8326)

Same fast-midrank DeLong implementation as delong_correct.py, reused verbatim
(see that file for the reference/history). No retraining -- uses existing
saved y_true / y_score arrays only.
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


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "OFF (plain hybrid)": os.path.join(REPO_ROOT, 'patient_split', 'result_4class_patient_hybrid_auc'),
    "ON (GMM+Gumbel refined)": os.path.join(REPO_ROOT, 'patient_split', 'result_4class_patient_gmmgumbel_hybrid_auc'),
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

    a, b = "OFF (plain hybrid)", "ON (GMM+Gumbel refined)"
    y_true_a, score_a = pooled[a]
    y_true_b, score_b = pooled[b]
    assert np.array_equal(y_true_a, y_true_b), "y_true mismatch between models"
    y_true = y_true_a

    lines = ["=== R1-1 ablation: GMM+Gumbel edge refinement, DeLong test (fast midrank) ===",
             f"\n{'='*70}\n  {a}  vs  {b}\n{'='*70}"]
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
        lines.append(f"  [{CLASS_NAMES[c]:8s}] AUC OFF={auc_a:.4f} ON={auc_b:.4f}  "
                     f"diff(ON-OFF)={auc_b-auc_a:+.4f}  z={z:+.3f}  p={p:.4g} {sig}")

    macro_diff = np.mean(per_class_diff)  # OFF - ON
    macro_var = np.mean([va + vb - 2 * cv for va, vb, cv in
                          zip(per_class_var_a, per_class_var_b, per_class_cov)])
    macro_se = np.sqrt(max(macro_var, 0) / N_CLASSES)
    macro_z = macro_diff / macro_se if macro_se > 0 else 0.0
    macro_p = 2 * (1 - stats.norm.cdf(abs(macro_z)))
    sig = "(*)" if macro_p < 0.05 else ""
    lines.append("  ---")
    lines.append(f"  [MACRO]    diff(ON-OFF)={-macro_diff:+.4f}  z={-macro_z:+.3f}  p={macro_p:.4g} {sig}")

    lines.append("\nNote: per-class DeLong covariance is exact (paired, same test samples);")
    lines.append("cross-class combination still assumes independence across the 4 one-vs-rest")
    lines.append("AUCs (conservative approximation, same as delong_correct.py).")

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(OUT_DIR, 'delong_gmmgumbel_ablation_result.txt'), 'w') as f:
        f.write(report + "\n")
    print(f"\nSaved: {os.path.join(OUT_DIR, 'delong_gmmgumbel_ablation_result.txt')}")
