"""R6. Does restricting features to the labelled central 10 s, or adding time-axis (periodicity) features, change discrimination?

Same learner (histogram gradient boosting, class-balanced) on the same patient-grouped folds, five feature sets built from
results/r3_features (raw EEG recomputed): full 50-s PSD+FC, central 10-s PSD+FC, each plus 10 dynamic-connectivity /
envelope-periodicity features, and the dynamic features alone. Patient-clustered CIs and patient-paired bootstrap differences.
Also a sanity check that the recomputed 50-s features match the stored psd_sb / fc_sb features.
Outputs: results/r6_center10s.json, results/r6_center10s/<cond>/y_score.npy (pooled, in the order of pooled_predictions("hard")).
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from common import ROOT, OUT, pooled_predictions, save

H = pooled_predictions("hard"); files = H.file.values; y = H.y.values; folds = H.fold.values; pat = H.patient.values
iu = np.triu_indices(19, 1); FD = os.path.join(OUT, "r3_features")
def load(f):
    z = np.load(os.path.join(FD, f.replace(".npy", ".npz")))
    def flat(s): return np.concatenate([np.log(z[f"psd{s}"] + 1e-12).ravel()] + [z[f"{m}{s}"][iu] for m in ("corr", "coh", "aec", "plv")])
    return flat("50"), flat("10"), z["dyn"]
cache = os.path.join(OUT, "r6_features.npz")
if os.path.exists(cache):
    c = np.load(cache); X50, X10, XD = c["X50"], c["X10"], c["XD"]
else:
    rows = [load(f) for f in files]; X50 = np.stack([r[0] for r in rows]).astype(np.float32); X10 = np.stack([r[1] for r in rows]).astype(np.float32); XD = np.stack([r[2] for r in rows]).astype(np.float32)
    np.savez_compressed(cache, X50=X50, X10=X10, XD=XD)
XD = np.nan_to_num(XD)
print("features", X50.shape, X10.shape, XD.shape, flush=True)

# sanity: recomputed 50-s features vs stored features
rng = np.random.default_rng(0); idx = rng.choice(len(files), 200, replace=False); cors = {"psd": [], "corr": [], "coh": [], "aec": [], "plv": []}
for i in idx:
    z = np.load(os.path.join(FD, files[i].replace(".npy", ".npz"))); ps = np.load(os.path.join(ROOT, "psd_sb", files[i]))
    if ps.shape == z["psd50"].shape: cors["psd"].append(np.corrcoef(np.log(ps.ravel() + 1e-12), np.log(z["psd50"].ravel() + 1e-12))[0, 1])
    for m in ("corr", "coh", "aec", "plv"):
        p = os.path.join(ROOT, "fc_sb", m, files[i])
        if os.path.exists(p): cors[m].append(np.corrcoef(np.abs(np.load(p))[iu], np.abs(z[f"{m}50"])[iu])[0, 1])
sanity = {k: (float(np.nanmedian(v)) if v else None) for k, v in cors.items()}; print("sanity median corr recomputed vs stored:", sanity, flush=True)

def cv(X):
    sc = np.zeros((len(y), 4), dtype=np.float32)
    for k in np.unique(folds):
        tr = folds != k; w = len(y[tr]) / (4 * np.bincount(y[tr], minlength=4))
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)
        clf.fit(X[tr], y[tr], sample_weight=w[y[tr]]); sc[folds == k] = clf.predict_proba(X[folds == k]); print(f"  fold {k} done ({clf.n_iter_} iters)", flush=True)
    return sc
def macro(s, m=None):
    m = np.ones(len(y), bool) if m is None else m; return float(np.mean([roc_auc_score(y[m] == c, s[m, c]) for c in range(4)]))
def macro_w(s, idx):  # macro AUROC on a patient-resampled index set
    yy = y[idx]; return float(np.mean([roc_auc_score(yy == c, s[idx, c]) for c in range(4)]))
P = {p: np.where(pat == p)[0] for p in np.unique(pat)}; pats = list(P)
def boot_idx(r): return np.concatenate([P[p] for p in r.choice(pats, len(pats), replace=True)])
B = 500; rng = np.random.default_rng(1); BOOT = [boot_idx(rng) for _ in range(B)]

conds = {"full50": X50, "center10": X10, "full50_plus_dyn": np.c_[X50, XD], "center10_plus_dyn": np.c_[X10, XD], "dyn_only": XD}
R = {"sanity_recomputed_vs_stored_median_corr": sanity, "n_boot": B}; S = {}
for name, X in conds.items():
    print(name, flush=True); d = os.path.join(OUT, "r6_center10s", name); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "y_score.npy")): sc = np.load(os.path.join(d, "y_score.npy"))
    else: sc = cv(X); np.save(os.path.join(d, "y_score.npy"), sc)
    S[name] = sc; bs = [macro_w(sc, b) for b in BOOT]
    R[name] = {"macro_auc": macro(sc), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "per_class": {c: float(roc_auc_score(y == i, sc[:, i])) for i, c in enumerate(["Seizure", "LPD", "GPD", "Others"])},
               "LPD_vs_GPD_auc": float(roc_auc_score(y[np.isin(y, [1, 2])] == 1, (sc[:, 1] - sc[:, 2])[np.isin(y, [1, 2])]))}
    print(name, R[name], flush=True)
R["paired"] = {}
for a, b in [("center10", "full50"), ("full50_plus_dyn", "full50"), ("center10_plus_dyn", "center10"), ("center10_plus_dyn", "full50")]:
    diffs = np.array([macro_w(S[a], i) - macro_w(S[b], i) for i in BOOT]); d0 = macro(S[a]) - macro(S[b])
    R["paired"][f"{a}_minus_{b}"] = {"diff": d0, "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}
    print(a, "-", b, R["paired"][f"{a}_minus_{b}"], flush=True)
# agreement-stratified: does the 10-s restriction matter more on contested windows?
c6 = np.load(os.path.join(OUT, "cons6.npy"))
R["by_agreement"] = {n: {"n": int(m.sum()), "full50": macro(S["full50"], m), "center10": macro(S["center10"], m)} for n, m in [("<0.5", c6 < 0.5), ("0.5-0.75", (c6 >= .5) & (c6 < .75)), ("0.75-1", (c6 >= .75) & (c6 < .999)), ("=1", c6 >= .999)]}
save("r6_center10s.json", R); print("saved", flush=True)
