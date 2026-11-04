"""R1. Recover the SVM and MLP baselines with per-window predictions on the stored patient folds.

Input: flattened connectivity (4 x 19 x 19 = 1,444) + flattened PSD (19 x 101 = 1,919) = 3,363 features,
standardized on each fold's training split. Class-balanced weights. Hyperparameters: SVM RBF C=1, gamma='scale'
(original Optuna values are not in the handover; recorded as such); MLP 256-128 ReLU, Adam, early stopping on validation.
Outputs: results/r1_baselines/{SVM,MLP}/fold_k/{y_true,y_pred,y_score}.npy  (same format as the GAT runs)
"""
import os, json, numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import ROOT, OUT, load_splits, ensure_fc, ensure_psd

MEAS = ["corr", "coh", "aec", "plv"]; ensure_fc(MEAS); PSD = ensure_psd(); splits = load_splits()
raw = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 3, "OTHER": 3}
cache = os.path.join(OUT, "r1_flat_features.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); FILES = list(z["files"]); X = z["X"]
else:
    FILES = sorted(set(f for s in splits for part in ("train", "val", "test") for f in s[part]["files"]))
    def feat(f):
        fc = np.concatenate([np.load(os.path.join(ROOT, "fc_sb", m, f)).ravel() for m in MEAS]); return np.concatenate([fc, np.load(os.path.join(PSD, f)).ravel()]).astype(np.float32)
    X = np.stack(Parallel(n_jobs=10, verbose=1)(delayed(feat)(f) for f in FILES)); np.savez(cache, files=np.array(FILES), X=X)
idx = {f: i for i, f in enumerate(FILES)}

def run(model_name, k):
    s = splits[k - 1]; tr, va, te = ([idx[f] for f in s[p]["files"]] for p in ("train", "val", "test"))
    ytr = np.array([raw[l] for l in s["train"]["labels_str"]]); yva = np.array([raw[l] for l in s["val"]["labels_str"]]); yte = np.array(s["test"]["labels_int"])
    sc = StandardScaler().fit(X[tr]); Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
    if model_name == "SVM":
        clf = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", decision_function_shape="ovr", cache_size=2000).fit(Xtr, ytr)
        d = clf.decision_function(Xte); score = np.exp(d) / np.exp(d).sum(1, keepdims=True)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=False, max_iter=200, random_state=42 + k, alpha=1e-4)
        best, best_auc = None, -1
        for it in [20, 40, 80, 150]:
            c = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=it, random_state=42 + k, alpha=1e-4).fit(Xtr, ytr)
            a = roc_auc_score(yva, c.predict_proba(Xva), multi_class="ovr", average="macro")
            if a > best_auc: best, best_auc = c, a
        score = best.predict_proba(Xte)
    fd = os.path.join(OUT, "r1_baselines", model_name, f"fold_{k}"); os.makedirs(fd, exist_ok=True)
    np.save(os.path.join(fd, "y_true.npy"), yte); np.save(os.path.join(fd, "y_pred.npy"), score.argmax(1)); np.save(os.path.join(fd, "y_score.npy"), score)
    m = {"model": model_name, "fold": k, "macro_auc": float(roc_auc_score(yte, score, multi_class="ovr", average="macro")), "acc": float((score.argmax(1) == yte).mean())}
    json.dump(m, open(os.path.join(fd, "metrics.json"), "w")); print(m, flush=True); return m

res = Parallel(n_jobs=10, verbose=5)(delayed(run)(mn, k) for mn in ["MLP", "SVM"] for k in range(1, 11))
json.dump(res, open(os.path.join(OUT, "r1_baselines_summary.json"), "w"), indent=1)
for mn in ["MLP", "SVM"]: print(mn, "mean macro AUC", np.mean([r["macro_auc"] for r in res if r["model"] == mn]))
