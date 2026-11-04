"""A7. Would more patients help, or is label noise the ceiling?

Subsample training PATIENTS (not clips) at 10/25/50/100% within each stored fold, fit the
A6 surrogate, evaluate on the untouched test fold. Repeat 3 draws per fraction.
A curve that is still rising at 100% says more data would help; a flat curve says the
representation or the labels are the limit. Compare the curve on unanimous test clips
vs low-agreement test clips: if only the unanimous curve rises, labels are the limit.
Outputs: results/a7_learning_curve.json, results/a7_learning_curve.png
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

F = pd.read_csv(os.path.join(OUT, "a6_features.csv"), index_col=0); splits = load_splits(); meta = load_meta()
label = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 3, "OTHER": 3}
FR = [0.1, 0.25, 0.5, 1.0]
def run(k, frac, rep):
    s = splits[k]; rng = np.random.default_rng(100 * k + rep)
    pats = np.array(s["train"]["patient_id"]); keep = rng.choice(np.unique(pats), int(len(np.unique(pats)) * frac), replace=False); m = np.isin(pats, keep)
    Xtr = F.loc[np.array(s["train"]["files"])[m]].values; ytr = np.array([label[l] for l in s["train"]["labels_str"]])[m]
    w = len(ytr) / (4 * np.bincount(ytr, minlength=4))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=rep).fit(Xtr, ytr, sample_weight=w[ytr])
    Xte = F.loc[s["test"]["files"]].values; yte = np.array(s["test"]["labels_int"]); P = clf.predict_proba(Xte)
    cons = meta.loc[s["test"]["files"]].consensus_frac.values
    def mauc(mask): return float(roc_auc_score(yte[mask], P[mask], multi_class="ovr", average="macro")) if len(np.unique(yte[mask])) == 4 else np.nan
    return {"fold": k + 1, "frac": frac, "rep": rep, "n_patients": int(len(keep)), "all": mauc(np.ones_like(yte, bool)), "unanimous": mauc(cons >= 0.999), "low_agreement": mauc(cons < 0.75)}
res = Parallel(n_jobs=11, verbose=1)(delayed(run)(k, f, r) for k in range(10) for f in FR for r in range(3))
df = pd.DataFrame(res); g = df.groupby("frac")[["all", "unanimous", "low_agreement"]].agg(["mean", "std"])
R = {"curve": {str(f): {c: [float(g.loc[f, (c, "mean")]), float(g.loc[f, (c, "std")])] for c in ["all", "unanimous", "low_agreement"]} for f in FR}}
print(g); save("a7_learning_curve.json", R)
fig, ax = plt.subplots(figsize=(5, 3))
for c, col in [("all", "#1f4e79"), ("unanimous", "#2a9d8f"), ("low_agreement", "#c0504d")]:
    ax.errorbar(FR, g[(c, "mean")], yerr=g[(c, "std")], fmt="o-", label=c, color=col, capsize=3)
ax.set_xscale("log"); ax.set_xticks(FR); ax.set_xticklabels(["10%", "25%", "50%", "100%"]); ax.set_xlabel("Training patients used"); ax.set_ylabel("Macro AUROC (surrogate)"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "a7_learning_curve.png"), dpi=180)
