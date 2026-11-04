"""A6. Is a graph neural network needed? Classical graph-theory features + gradient boosting
on the identical patient-grouped folds.

Features per clip (~270):
  band powers per channel from the 101-bin PSD: delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-50 (19x5, log)
  per measure (corr, coh, aec, plv): node strength (19), weighted clustering coefficient (19),
  global mean, std, global efficiency, hemispheric asymmetry index (from A4), modularity proxy (ratio of intra- to inter-hemispheric mean)
Model: HistGradientBoostingClassifier, class-balanced, early stopping on the stored validation split.
Outputs: results/a6_surrogate/fold_k/y_{true,pred,score}.npy in the same format as the GAT runs,
         results/a6_surrogate.json (macro AUROC, per-class, DeLong vs GAT, feature importance groups)
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from common import *

MEAS = ["corr", "coh", "aec", "plv"]; ensure_fc(MEAS); PSD = ensure_psd()
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
L = [CH.index(c) for c in LEFT]; Rr = [CH.index(c) for c in RIGHT]

def wcc(W):
    """Onnela weighted clustering coefficient."""
    W = W / (W.max() + 1e-9); k = (W > 0).sum(1); W3 = np.cbrt(W)
    num = np.diag(W3 @ W3 @ W3); return np.where(k > 1, num / np.maximum(k * (k - 1), 1), 0)

def geff(W):
    """Global efficiency with distance = 1/weight (Floyd-Warshall on 19 nodes)."""
    D = 1 / (W + 1e-9); np.fill_diagonal(D, 0); n = len(D)
    for k in range(n): D = np.minimum(D, D[:, [k]] + D[[k], :])
    inv = 1 / D[~np.eye(n, dtype=bool)]; return inv.mean()

def feats(file):
    psd = np.load(os.path.join(PSD, file)); f = {}
    for b, (lo, hi) in BANDS.items():
        bp = np.log(psd[:, lo:hi].mean(1) + 1e-12)
        for i, c in enumerate(CH): f[f"bp_{b}_{c}"] = bp[i]
    fc = load_fc(file, MEAS)
    for m in MEAS:
        W = np.abs(fc[m]).astype(float); np.fill_diagonal(W, 0)
        s = W.sum(1) / 18; cc = wcc(W)
        for i, c in enumerate(CH): f[f"{m}_str_{c}"] = s[i]; f[f"{m}_cc_{c}"] = cc[i]
        iu = W[np.triu_indices(19, 1)]
        f[f"{m}_mean"] = iu.mean(); f[f"{m}_std"] = iu.std(); f[f"{m}_geff"] = geff(W)
        iL = W[np.ix_(L, L)].sum() / (8 * 7); iR = W[np.ix_(Rr, Rr)].sum() / (8 * 7); inter = W[np.ix_(L, Rr)].mean()
        f[f"{m}_AI"] = (iL - iR) / (iL + iR + 1e-9); f[f"{m}_absAI"] = abs(f[f"{m}_AI"]); f[f"{m}_intra_inter"] = (iL + iR) / 2 / (inter + 1e-9)
    return f

if __name__ == "__main__":
    splits = load_splits(); meta = load_meta()
    allfiles = sorted(set(f for s in splits for part in ("train", "val", "test") for f in s[part]["files"]))
    cache = os.path.join(OUT, "a6_features.csv")
    if os.path.exists(cache): F = pd.read_csv(cache, index_col=0)
    else:
        F = pd.DataFrame(Parallel(n_jobs=11, verbose=1)(delayed(feats)(f) for f in allfiles), index=allfiles); F.to_csv(cache)
    label = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 3, "OTHER": 3}
    R = {"folds": []}; pooled = []
    for k, s in enumerate(splits):
        Xtr, ytr = F.loc[s["train"]["files"]].values, np.array([label[l] for l in s["train"]["labels_str"]])
        Xva, yva = F.loc[s["val"]["files"]].values, np.array([label[l] for l in s["val"]["labels_str"]])
        Xte, yte = F.loc[s["test"]["files"]].values, np.array(s["test"]["labels_int"])
        w = len(ytr) / (4 * np.bincount(ytr, minlength=4)); sw = w[ytr]
        clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, validation_fraction=None, random_state=42 + k)
        # sklearn's early stopping uses an internal split; emulate the stored val split by fitting on train and scoring val for n_iter choice
        clf.set_params(early_stopping=False); best, best_auc = None, -1
        for it in [150, 300, 450, 600]:
            c = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False, random_state=42 + k).fit(Xtr, ytr, sample_weight=sw)
            a = roc_auc_score(yva, c.predict_proba(Xva), multi_class="ovr", average="macro")
            if a > best_auc: best, best_auc, best_it = c, a, it
        ys = best.predict_proba(Xte); yp = ys.argmax(1)
        d = os.path.join(OUT, "a6_surrogate", f"fold_{k+1}"); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "y_true.npy"), yte); np.save(os.path.join(d, "y_pred.npy"), yp); np.save(os.path.join(d, "y_score.npy"), ys)
        macro = roc_auc_score(yte, ys, multi_class="ovr", average="macro")
        R["folds"].append({"fold": k + 1, "n_iter": best_it, "val_auc": float(best_auc), "test_macro_auc": float(macro), "test_acc": float((yp == yte).mean())})
        print(R["folds"][-1]); pooled.append((yte, ys))
        if k == 0:
            pi = permutation_importance(best, Xte, yte, n_repeats=3, random_state=0, n_jobs=11, scoring="roc_auc_ovr")
            grp = {}
            for name, imp in zip(F.columns, pi.importances_mean):
                g = "bandpower" if name.startswith("bp_") else name.split("_")[0] + "_" + ("AI" if "AI" in name else name.split("_")[1])
                grp[g] = grp.get(g, 0) + float(imp)
            R["fold1_permutation_importance_by_group"] = dict(sorted(grp.items(), key=lambda x: -x[1]))
    yt = np.concatenate([p[0] for p in pooled]); ys = np.concatenate([p[1] for p in pooled])
    R["pooled"] = {"macro_auc": float(roc_auc_score(yt, ys, multi_class="ovr", average="macro")), "acc": float((ys.argmax(1) == yt).mean()),
                   "per_class_auc": {c: float(roc_auc_score(yt == i, ys[:, i])) for i, c in enumerate(CLASSES)}}
    R["mean_fold_macro_auc"] = float(np.mean([f["test_macro_auc"] for f in R["folds"]])); R["sd_fold_macro_auc"] = float(np.std([f["test_macro_auc"] for f in R["folds"]]))
    print(R["pooled"], R["mean_fold_macro_auc"]); save("a6_surrogate.json", R)
