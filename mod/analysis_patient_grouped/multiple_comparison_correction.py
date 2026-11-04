"""
Multiple-comparison correction across the full family of pairwise-comparison
tests (paired t-test x3 metrics, DeLong, McNemar, paired bootstrap x3 metrics),
recomputed for the final patient-grouped results. Supersedes
review/analysis/multiple_comparison_correction.py.

Benjamini-Hochberg (FDR) correction across all N pairwise tests treated as
complementary (not independent) views, per reviewer request.
"""
import os
import sys
import itertools
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

from statistical_comparison import (
    load_fold_results, concat_folds, mcnemar_test, paired_bootstrap, paired_ttest_folds,
)
from delong_correct import delong_two_model_test

MODEL_DIRS = {
    "GAT": os.path.join(REPO_ROOT, 'patient_split', 'result_4class_patient_hybrid_auc'),
    "SVM": os.path.join(REPO_ROOT, 'patient_split', 'result_SVM_patient_auc'),
    "CNN": os.path.join(REPO_ROOT, 'patient_split', 'result_CNN_patient_auc'),
    "MLP": os.path.join(REPO_ROOT, 'patient_split', 'result_MLP_patient_auc'),
}
N_FOLDS = 10
N_CLASSES = 4

print("Loading results...")
all_results = {}
for name, d in MODEL_DIRS.items():
    yt, ys, yp = load_fold_results(d, N_FOLDS)
    all_results[name] = {"y_trues": yt, "y_scores": ys, "y_preds": yp}

fold_metrics = {}
for name, res in all_results.items():
    accs, aucs, f1s = [], [], []
    for yt, ys, yp in zip(res["y_trues"], res["y_scores"], res["y_preds"]):
        accs.append(accuracy_score(yt, yp))
        aucs.append(roc_auc_score(yt, ys, multi_class="ovr", average="macro"))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    fold_metrics[name] = {"accuracy": accs, "auc": aucs, "f1": f1s}

concat_results = {}
for name, res in all_results.items():
    yt, ys, yp = concat_folds(res["y_trues"], res["y_scores"], res["y_preds"])
    concat_results[name] = {"y_true": yt, "y_score": ys, "y_pred": yp}

pairs = list(itertools.combinations(MODEL_DIRS.keys(), 2))
tests = []

for ma, mb in pairs:
    for metric in ["accuracy", "auc", "f1"]:
        t, p = paired_ttest_folds(fold_metrics[ma][metric], fold_metrics[mb][metric])
        tests.append((f"{ma} vs {mb} | t-test | {metric}", p))

    yt_a, ys_a, yp_a = concat_results[ma]["y_true"], concat_results[ma]["y_score"], concat_results[ma]["y_pred"]
    yt_b, ys_b, yp_b = concat_results[mb]["y_true"], concat_results[mb]["y_score"], concat_results[mb]["y_pred"]

    diffs, vas, vbs, covs = [], [], [], []
    for c in range(N_CLASSES):
        y_bin = (yt_a == c).astype(int)
        auc_a, auc_b, var_a, var_b, cov_ab, _, _ = delong_two_model_test(y_bin, ys_a[:, c], ys_b[:, c])
        diffs.append(auc_a - auc_b); vas.append(var_a); vbs.append(var_b); covs.append(cov_ab)
    macro_diff = np.mean(diffs)
    macro_se = np.sqrt(max(np.mean([va+vb-2*cv for va, vb, cv in zip(vas, vbs, covs)]), 0) / N_CLASSES)
    from scipy import stats as _stats
    macro_z = macro_diff / macro_se if macro_se > 0 else 0.0
    macro_p = 2 * (1 - _stats.norm.cdf(abs(macro_z)))
    tests.append((f"{ma} vs {mb} | DeLong | macro_auc", macro_p))

    chi2, p = mcnemar_test(yt_a, yp_a, yp_b)
    tests.append((f"{ma} vs {mb} | McNemar | accuracy", p))

    for metric in ["accuracy", "auc", "f1"]:
        _, _, _, _, _, p = paired_bootstrap(yt_a, yp_a, ys_a, yp_b, ys_b, metric=metric)
        tests.append((f"{ma} vs {mb} | bootstrap | {metric}", p))

labels = [t[0] for t in tests]
pvals = np.array([t[1] for t in tests])

order = np.argsort(pvals)
ranked = pvals[order]
m = len(pvals)
bh = ranked * m / (np.arange(m) + 1)
bh = np.minimum.accumulate(bh[::-1])[::-1]
bh = np.clip(bh, 0, 1)
adj_pvals = np.empty_like(bh)
adj_pvals[order] = bh

lines = []
lines.append(f"=== Benjamini-Hochberg correction across {m} pairwise-comparison tests (patient-grouped final results) ===\n")
lines.append(f"{'Test':<42}{'raw p':>14}{'BH-adj p':>14}  sig(raw) sig(BH)")
lines.append("-" * 90)
n_lost = 0
for lab, p, adj_p in zip(labels, pvals, adj_pvals):
    sig_raw = "*" if p < 0.05 else ""
    sig_bh = "*" if adj_p < 0.05 else ""
    if sig_raw and not sig_bh:
        n_lost += 1
    lines.append(f"{lab:<42}{p:>14.4g}{adj_p:>14.4g}    {sig_raw:^7}  {sig_bh:^6}")

lines.append(f"\nTotal tests: {m}")
lines.append(f"Significant at raw p<0.05: {int((pvals < 0.05).sum())}")
lines.append(f"Significant at BH-adjusted p<0.05: {int((adj_pvals < 0.05).sum())}")
lines.append(f"Tests that LOST significance after BH correction: {n_lost}")

report = "\n".join(lines)
print(report)

with open(os.path.join(THIS_DIR, 'multiple_comparison_correction_result.txt'), 'w') as f:
    f.write(report + "\n")
print(f"\nSaved: {os.path.join(THIS_DIR, 'multiple_comparison_correction_result.txt')}")
