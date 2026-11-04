"""A4. Does left-right asymmetry of connectivity carry the lateralized-vs-generalized distinction,
and does the model use it?

Hypothesis-driven replacement for the attention maps. For each clip and each measure:
  intra_L = mean connectivity among the 8 left electrodes, intra_R = same on the right
  AI = (intra_L - intra_R) / (intra_L + intra_R)         signed asymmetry
  |AI|                                                     magnitude of lateralization
  node-strength asymmetry per homologous pair (Fp1-Fp2, F7-F8, ...), 8 values

Tests
 1. AUROC of |AI| (single feature, no training) for LPD vs GPD and LRDA vs GRDA, per measure.
 2. Patient-grouped logistic regression on the 4x(1+8) asymmetry features: LPD vs GPD.
 3. Does the model's P(LPD) - P(GPD) correlate with |AI| within the LPD+GPD clips?
    If yes, the model has learned lateralization; if no, it separates the classes by
    something else (and the asymmetry feature is complementary).
Outputs: results/a4_asymmetry.json, results/a4_asymmetry.png
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

MEAS = ["corr", "coh", "aec", "plv"]; ensure_fc(MEAS)
L = [CH.index(c) for c in LEFT]; Rr = [CH.index(c) for c in RIGHT]
def feats(file):
    fc = load_fc(file, MEAS); out = {}
    for m in MEAS:
        M = np.abs(fc[m]); np.fill_diagonal(M, np.nan)
        iL = np.nanmean(M[np.ix_(L, L)]); iR = np.nanmean(M[np.ix_(Rr, Rr)])
        out[f"{m}_AI"] = (iL - iR) / (iL + iR + 1e-9)
        s = np.nanmean(M, 1)
        for l, r in zip(LEFT, RIGHT): out[f"{m}_ns_{l}{r}"] = (s[CH.index(l)] - s[CH.index(r)]) / (s[CH.index(l)] + s[CH.index(r)] + 1e-9)
    return out

df = pooled_predictions("hard")
F = pd.DataFrame(Parallel(n_jobs=10)(delayed(feats)(f) for f in df.file)); F.index = df.index
for m in MEAS: F[f"{m}_absAI"] = F[f"{m}_AI"].abs()
R = {"single_feature_auroc": {}}
for name, (a, b) in {"LPD_vs_GPD": ("LPD", "GPD"), "LRDA_vs_GRDA": ("LRDA", "GRDA")}.items():
    m = df["sub"].isin([a, b]).values; yb = (df["sub"][m] == a).astype(int).values
    R["single_feature_auroc"][name] = {mm: float(roc_auc_score(yb, F.loc[m, f"{mm}_absAI"])) for mm in MEAS}
# 2. patient-grouped logistic regression, LPD vs GPD, using the stored folds
m = df["sub"].isin(["LPD", "GPD"]).values; X = F.loc[m, [c for c in F.columns if "_AI" in c or "_ns_" in c]].values; X = np.c_[X, np.abs(X)]
yb = (df["sub"][m] == "LPD").astype(int).values; folds = df.fold[m].values; sc = np.zeros(len(yb))
for k in np.unique(folds):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X[folds != k], yb[folds != k]); sc[folds == k] = clf.predict_proba(X[folds == k])[:, 1]
R["logreg_asymmetry_LPD_vs_GPD_auroc"] = float(roc_auc_score(yb, sc))
pm = df.loc[m]; model_lat = (pm.p_LPD - pm.p_GPD).values
R["model_LPDvsGPD_auroc_same_clips"] = float(roc_auc_score(yb, model_lat))
# 3. does the model's lateralization score track measured asymmetry?
R["spearman_model_vs_absAI"] = {mm: float(spearmanr(model_lat, F.loc[m, f"{mm}_absAI"])[0]) for mm in MEAS}
# complementarity: logistic on [model score + asymmetry]
X2 = np.c_[X, model_lat]; sc2 = np.zeros(len(yb))
for k in np.unique(folds):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X2[folds != k], yb[folds != k]); sc2[folds == k] = clf.predict_proba(X2[folds == k])[:, 1]
R["model_plus_asymmetry_auroc"] = float(roc_auc_score(yb, sc2))
print(R); save("a4_asymmetry.json", R); F.to_parquet(os.path.join(OUT, "a4_asymmetry_features.parquet")) if False else F.to_csv(os.path.join(OUT, "a4_asymmetry_features.csv"), index=False)
fig, ax = plt.subplots(figsize=(5, 3))
for i, mm in enumerate(MEAS):
    for j, (lab, cols) in enumerate([("LPD", "#1f4e79"), ("GPD", "#c0504d")]):
        v = F.loc[df["sub"].values == lab, f"{mm}_absAI"]; ax.boxplot(v, positions=[i * 3 + j], widths=.8, patch_artist=True, boxprops=dict(facecolor=cols, alpha=.6), showfliers=False)
ax.set_xticks([i * 3 + .5 for i in range(4)]); ax.set_xticklabels(MEAS); ax.set_ylabel("|asymmetry index|"); ax.set_title("Lateralization of connectivity: LPD (blue) vs GPD (red)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "a4_asymmetry.png"), dpi=180)
