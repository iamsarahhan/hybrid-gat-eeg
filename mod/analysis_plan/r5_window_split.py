"""R5. Window-level (leaky) evaluation of the CORRECTED pipeline, for the leakage comparison.

Builds 10 stratified folds over windows ignoring patient identity (seed 42), on the same 36,000 windows,
and evaluates (a) the graph-theory GBM and (b) the MLP on identical features to their patient-grouped runs.
The GAT under the same window-level folds is run by r2_gat_controls_cpu.py --split window.
Outputs: results/window_folds.json, results/r5_window_split/{GBM,MLP}/fold_k/*.npy, results/r5_window_split.json
"""
import os, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from common import ROOT, OUT, load_splits, load_meta

splits = load_splits(); meta = load_meta()
files = sorted(set(f for s in splits for part in ("train", "val", "test") for f in s[part]["files"]))
raw = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 3, "OTHER": 3}
y = np.array([raw[meta.loc[f].expert_consensus.upper()] for f in files]); pid = meta.loc[files].patient_id.values
wf = os.path.join(OUT, "window_folds.json")
if os.path.exists(wf): W = json.load(open(wf))
else:
    skf = StratifiedKFold(10, shuffle=True, random_state=42); W = []
    for k, (tr, te) in enumerate(skf.split(files, y)):
        rng = np.random.default_rng(k); tr = rng.permutation(tr); nva = len(tr) // 5
        W.append({"fold": k, "train": [files[i] for i in tr[nva:]], "val": [files[i] for i in tr[:nva]], "test": [files[i] for i in te],
                  "test_patients_also_in_train": float(np.isin(pid[te], pid[tr]).mean())})
    json.dump(W, open(wf, "w")); print("window folds built; share of test windows whose patient is in train:", np.mean([w["test_patients_also_in_train"] for w in W]))

F = pd.read_csv(os.path.join(OUT, "a6_features.csv"), index_col=0)
z = np.load(os.path.join(OUT, "r1_flat_features.npz"), allow_pickle=True); FLAT = pd.DataFrame(z["X"], index=list(z["files"]))
lab = dict(zip(files, y))

def run(model, k):
    w = W[k]; ytr = np.array([lab[f] for f in w["train"]]); yva = np.array([lab[f] for f in w["val"]]); yte = np.array([lab[f] for f in w["test"]])
    if model == "GBM":
        Xtr, Xva, Xte = F.loc[w["train"]].values, F.loc[w["val"]].values, F.loc[w["test"]].values; sw = (len(ytr) / (4 * np.bincount(ytr)))[ytr]
        best, ba = None, -1
        for it in [150, 300, 450, 600]:
            c = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False, random_state=42 + k).fit(Xtr, ytr, sample_weight=sw)
            a = roc_auc_score(yva, c.predict_proba(Xva), multi_class="ovr", average="macro")
            if a > ba: best, ba = c, a
    else:
        sc = StandardScaler().fit(FLAT.loc[w["train"]].values); Xtr, Xva, Xte = (sc.transform(FLAT.loc[w[p]].values) for p in ("train", "val", "test")); best, ba = None, -1
        for it in [20, 40, 80, 150]:
            c = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=it, random_state=42 + k, alpha=1e-4).fit(Xtr, ytr)
            a = roc_auc_score(yva, c.predict_proba(Xva), multi_class="ovr", average="macro")
            if a > ba: best, ba = c, a
    sc_ = best.predict_proba(Xte); d = os.path.join(OUT, "r5_window_split", model, f"fold_{k+1}"); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "y_true.npy"), yte); np.save(os.path.join(d, "y_pred.npy"), sc_.argmax(1)); np.save(os.path.join(d, "y_score.npy"), sc_)
    m = {"model": model, "fold": k + 1, "macro_auc": float(roc_auc_score(yte, sc_, multi_class="ovr", average="macro")), "acc": float((sc_.argmax(1) == yte).mean())}; print(m, flush=True); return m

if __name__ == "__main__":
    res = Parallel(n_jobs=3)(delayed(run)(m, k) for m in ["GBM", "MLP"] for k in range(10))
    json.dump(res, open(os.path.join(OUT, "r5_window_split.json"), "w"), indent=1)
    for m in ["GBM", "MLP"]: print(m, "window-level mean macro AUC", np.mean([r["macro_auc"] for r in res if r["model"] == m]))
