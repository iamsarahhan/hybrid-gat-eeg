"""
Figure 8c: TOST (two one-sided tests, Schuirmann 1987) equivalence analysis
across the 4 single-FC-measure hybrid models (patient_split/
single_fc_measure_hybrid.py), patient-grouped, 10-fold each.

Different from a regular significance test: a non-significant paired t-test
only fails to detect a difference (absence of evidence), it does not
establish equivalence. TOST instead tests, for a pre-specified equivalence
margin +/-delta, whether the true mean difference can be excluded from lying
outside [-delta, +delta] -- i.e. actively demonstrates "practically the
same," not just "not proven different."

EQUIVALENCE MARGIN: delta = 0.02 macro AUC (2 points). Chosen as a
practically-negligible-difference threshold for this application, consistent
in magnitude with the leave-one-out FC-ablation drops found in
fc_ablation_result.txt (0.0086-0.0194) -- i.e. roughly the same scale as
"removing one measure from the full model" costs. This is a judgment call
(no established convention for this task), reported explicitly so the
threshold's effect on the conclusion can be assessed independently: the
per-pair mean difference and CI are also reported in full, so a reader can
apply a different margin if desired.

Uses per-fold macro AUC (n=10, paired by fold) from each single-measure
model's saved metrics.txt / y_true+y_score.npy.
"""
import os
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATIENT_SPLIT_DIR = os.path.join(REPO_ROOT, 'patient_split')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FC_TYPES = ['corr', 'coh', 'aec', 'plv']
DELTA = 0.02
ALPHA = 0.05


def fold_macro_aucs(result_dir, n_folds=10):
    aucs = []
    for i in range(1, n_folds + 1):
        yt = np.load(os.path.join(result_dir, f'fold_{i}', 'y_true.npy')).flatten()
        ys = np.load(os.path.join(result_dir, f'fold_{i}', 'y_score.npy'))
        aucs.append(roc_auc_score(yt, ys, multi_class='ovr', average='macro'))
    return np.array(aucs)


def tost_paired(x, y, delta, alpha=0.05):
    d = np.array(x) - np.array(y)
    n = len(d)
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    se = sd_d / np.sqrt(n)

    # H0_lower: mean_d <= -delta  vs  H1: mean_d > -delta
    t1 = (mean_d - (-delta)) / se
    p1 = 1 - stats.t.cdf(t1, df=n - 1)
    # H0_upper: mean_d >= +delta  vs  H1: mean_d < +delta
    t2 = (mean_d - delta) / se
    p2 = stats.t.cdf(t2, df=n - 1)

    p_tost = max(p1, p2)
    equivalent = (p1 < alpha) and (p2 < alpha)

    tcrit = stats.t.ppf(1 - alpha, df=n - 1)  # for the (1-2*alpha) CI used in TOST
    ci_lo = mean_d - tcrit * se
    ci_hi = mean_d + tcrit * se

    return {
        'mean_diff': float(mean_d), 'se': float(se), 'n': n,
        'ci_90_lo': float(ci_lo), 'ci_90_hi': float(ci_hi),
        'p_lower': float(p1), 'p_upper': float(p2), 'p_tost': float(p_tost),
        'equivalent_at_alpha': bool(equivalent),
    }


if __name__ == "__main__":
    aucs = {}
    for t in FC_TYPES:
        result_dir = os.path.join(PATIENT_SPLIT_DIR, f'result_4class_patient_singlefc_{t}')
        aucs[t] = fold_macro_aucs(result_dir)
        print(f"{t:6s}: per-fold macro AUC = " + ", ".join(f"{a:.4f}" for a in aucs[t]) +
              f"   mean={aucs[t].mean():.4f}")

    pairs = [(a, b) for i, a in enumerate(FC_TYPES) for b in FC_TYPES[i + 1:]]

    lines = ["=== Figure 8c: TOST equivalence across single-FC-measure models (10-fold paired) ===\n"]
    lines.append(f"Equivalence margin: delta = +/-{DELTA} macro AUC, alpha = {ALPHA}\n")
    lines.append("Per-condition mean macro AUC:")
    for t in FC_TYPES:
        lines.append(f"  {t:6s}: {aucs[t].mean():.4f} +/- {aucs[t].std():.4f}")

    lines.append(f"\n{'Pair':<14}{'Mean diff':>12}{'90% CI':>22}{'p(TOST)':>10}  Equivalent@delta")
    lines.append("-" * 80)
    tost_results = {}
    for a, b in pairs:
        r = tost_paired(aucs[a], aucs[b], DELTA, ALPHA)
        tost_results[f"{a}_vs_{b}"] = r
        eq = "YES" if r['equivalent_at_alpha'] else "no"
        ci_str = f"[{r['ci_90_lo']:+.4f}, {r['ci_90_hi']:+.4f}]"
        lines.append(f"{a+' vs '+b:<14}{r['mean_diff']:>+12.4f}{ci_str:>22}{r['p_tost']:>10.4g}  {eq}")

    n_equiv = sum(1 for r in tost_results.values() if r['equivalent_at_alpha'])
    lines.append(f"\n{n_equiv}/{len(pairs)} pairs statistically equivalent at delta=+/-{DELTA}, alpha={ALPHA}.")
    lines.append(
        "\nInterpretation: a pair is 'equivalent' only if the 90% CI for the mean AUC\n"
        "difference falls entirely within [-delta, +delta]. This is a positive claim of\n"
        "practical interchangeability between single-measure models, not merely a failure\n"
        "to detect a difference. Combined with fc_ablation_result.txt (removing any one\n"
        "measure from the FULL model costs 0.009-0.019 AUC, all significant), the overall\n"
        "picture is: each measure carries some unique signal relative to the full model,\n"
        "but as standalone single-measure models they may still land within a modest\n"
        "practical-equivalence band of each other -- these are compatible, not\n"
        "contradictory, claims (complementary-but-partially-substitutable)."
    )

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(OUT_DIR, 'tost_equivalence_result.txt'), 'w') as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, 'tost_equivalence_summary.json'), 'w') as f:
        json.dump({
            'delta': DELTA, 'alpha': ALPHA,
            'per_condition_mean_auc': {t: float(aucs[t].mean()) for t in FC_TYPES},
            'pairwise': tost_results,
        }, f, indent=2)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'tost_equivalence_result.txt')}")
