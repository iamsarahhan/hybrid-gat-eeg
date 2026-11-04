"""A5. Which connectivity measures and electrode pairs matter, tested causally with a null.

Uses the A6 surrogate (fast to re-score) as a stand-in for the GAT so that thousands of
perturbations are affordable on CPU; the GAT version of the same protocol runs on the
server with attention_rollout_and_causal_ablation.py.

Protocol (per class, one-vs-rest AUROC on test folds):
  measure level: zero one measure's features -> AUROC drop
  pair level: for each of 171 electrode pairs, permute (across clips) every feature
              derived from that pair's two nodes for a given measure -> AUROC drop
  null: same drops after the class labels are permuted (B = 200), BH-FDR at q = 0.05
  convergence: pairs that are significant here AND in the GAT attention-ablation list
               are the only ones reported as 'load-bearing'.
Outputs: results/a5_causal.json
"""
import numpy as np, pandas as pd, itertools
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from statsmodels.stats.multitest import multipletests
from common import *

MEAS = ["corr", "coh", "aec", "plv"]; splits = load_splits(); F = pd.read_csv(os.path.join(OUT, "a6_features.csv"), index_col=0)
label = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 3, "OTHER": 3}
rng = np.random.default_rng(0); R = {}
k = 0; s = splits[k]  # fold 1 for compute reasons, like the GAT audit
Xtr = F.loc[s["train"]["files"]].values; ytr = np.array([label[l] for l in s["train"]["labels_str"]])
Xte = F.loc[s["test"]["files"]]; yte = np.array(s["test"]["labels_int"])
w = len(ytr) / (4 * np.bincount(ytr, minlength=4))
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42).fit(Xtr, ytr, sample_weight=w[ytr])
def auc_c(X, y, c): return roc_auc_score(y == c, clf.predict_proba(X)[:, c])
base = {c: auc_c(Xte.values, yte, c) for c in range(4)}
cols = list(F.columns)
# measure level
R["measure_drop"] = {}
for m in MEAS:
    X = Xte.copy(); X[[c for c in cols if c.startswith(m + "_")]] = 0
    R["measure_drop"][m] = {CLASSES[c]: float(base[c] - auc_c(X.values, yte, c)) for c in range(4)}
# pair level
pairs = list(itertools.combinations(range(19), 2))
def pair_cols(m, i, j): return [c for c in cols if c.startswith(m + "_") and (c.endswith("_" + CH[i]) or c.endswith("_" + CH[j]))]
def drops(y):
    out = np.zeros((4, len(MEAS), len(pairs)))
    b = {c: auc_c(Xte.values, y, c) for c in range(4)}
    for mi, m in enumerate(MEAS):
        for pi_, (i, j) in enumerate(pairs):
            X = Xte.copy(); cc = pair_cols(m, i, j); X[cc] = X[cc].sample(frac=1, random_state=pi_).values
            P = clf.predict_proba(X.values)
            for c in range(4): out[c, mi, pi_] = b[c] - roc_auc_score(y == c, P[:, c])
    return out
real = drops(yte)
B = 200; null = np.stack([drops(rng.permutation(yte)) for _ in range(B)])
p = (1 + (null >= real[None]).sum(0)) / (B + 1)
R["pairs"] = {}
for c in range(4):
    for mi, m in enumerate(MEAS):
        rej, q, _, _ = multipletests(p[c, mi], alpha=0.05, method="fdr_bh")
        sig = [(f"{CH[i]}-{CH[j]}", float(real[c, mi, n]), float(q[n])) for n, (i, j) in enumerate(pairs) if rej[n]]
        R["pairs"][f"{CLASSES[c]}|{m}"] = sorted(sig, key=lambda x: -x[1])[:15]
R["base_auc"] = {CLASSES[c]: float(base[c]) for c in range(4)}; R["note"] = "fold 1, surrogate model, B=200 label permutations, BH q<0.05"
print({k: len(v) for k, v in R["pairs"].items()}); save("a5_causal.json", R)
