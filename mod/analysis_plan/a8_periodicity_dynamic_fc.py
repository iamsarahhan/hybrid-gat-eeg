"""A8. Does the periodicity that static connectivity discards discriminate LPD and GPD?

Reads raw EEG (train_eegs.zip -> parquet, 200 Hz, 19 EEG channels + EKG) for a stratified
subset (default 1,000 clips per class, patient-grouped by the stored folds), computes:
  dynamic FC: 2-s windows, 0.5-s step, |Pearson correlation| across the 19 channels
              -> global connectivity time series g(t), 97 points per 50-s clip
  periodicity features: Welch spectrum of g(t) and of each channel's Hilbert envelope,
              peak frequency in 0.3-3 Hz, peak prominence, spectral entropy, lag-1 autocorr,
              dwell statistics of g(t) (std, fraction of time above mean+1sd)
Tests
 1. Periodicity features alone (12-30 features) -> macro AUROC with gradient boosting on the
    same folds; LPD+GPD vs Seizure+Others, and LPD vs GPD.
 2. Added to the A6 static features on the same subset -> gain over static alone.
Outputs: results/a8_periodicity.json, results/a8_periodicity_features.csv
Requires: pip install pyarrow
"""
import numpy as np, pandas as pd, zipfile, io
from scipy.signal import welch, hilbert, butter, sosfiltfilt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import Parallel, delayed
from common import *

N_PER_CLASS = 1000; FS = 200; WIN, STEP = 2 * FS, FS // 2
splits = load_splits(); meta = load_meta(); label = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 3, "OTHER": 3}
test_files = [(k + 1, f) for k, s in enumerate(splits) for f in s["test"]["files"]]
df = pd.DataFrame(test_files, columns=["fold", "file"]); df["y"] = [label[meta.loc[f].expert_consensus.upper()] for f in df.file]
sub = df.groupby("y", group_keys=False).apply(lambda d: d.sample(min(N_PER_CLASS, len(d)), random_state=0)).reset_index(drop=True)
sos = butter(4, [0.5, 50], btype="band", fs=FS, output="sos")
ZIP = os.path.join(ROOT, "train_eegs.zip")

def clip(file):
    eeg_id, sub_id = file.replace(".npy", "").split("_"); row = meta.loc[file]
    with zipfile.ZipFile(ZIP) as z: t = pd.read_parquet(io.BytesIO(z.read(f"train_eegs/{eeg_id}.parquet")))
    off = int(float(pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["eeg_id", "eeg_sub_id", "eeg_label_offset_seconds"]).query("eeg_id==@eeg_id and eeg_sub_id==@sub_id").eeg_label_offset_seconds.iloc[0]) * FS) if False else None
    return t

def feats(file, offset):
    eeg_id = file.split("_")[0]
    with zipfile.ZipFile(ZIP) as z: t = pd.read_parquet(io.BytesIO(z.read(f"train_eegs/{eeg_id}.parquet")))
    x = t[CH].values[int(offset * FS): int(offset * FS) + 50 * FS].T.astype(float); x = np.nan_to_num(x)
    x = sosfiltfilt(sos, x, axis=1); x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
    starts = range(0, x.shape[1] - WIN + 1, STEP); g = []
    for s0 in starts:
        C = np.corrcoef(x[:, s0:s0 + WIN]); g.append(np.abs(C[np.triu_indices(19, 1)]).mean())
    g = np.array(g); fsg, Pg = welch(g - g.mean(), fs=FS / STEP, nperseg=min(64, len(g)))
    band = (fsg >= 0.3) & (fsg <= 3); f = {}
    f["g_peak_freq"] = fsg[band][Pg[band].argmax()]; f["g_peak_prom"] = Pg[band].max() / (Pg[band].mean() + 1e-12)
    pn = Pg / Pg.sum(); f["g_spec_entropy"] = -(pn * np.log(pn + 1e-12)).sum(); f["g_std"] = g.std(); f["g_lag1"] = np.corrcoef(g[:-1], g[1:])[0, 1]
    f["g_frac_high"] = (g > g.mean() + g.std()).mean()
    env = np.abs(hilbert(x, axis=1)); fe, Pe = welch(env - env.mean(1, keepdims=True), fs=FS, nperseg=FS * 4, axis=1)
    be = (fe >= 0.3) & (fe <= 3); pk = Pe[:, be]
    f["env_peak_freq_med"] = np.median(fe[be][pk.argmax(1)]); f["env_peak_prom_med"] = np.median(pk.max(1) / (pk.mean(1) + 1e-12)); f["env_peak_prom_max"] = (pk.max(1) / (pk.mean(1) + 1e-12)).max()
    f["env_peak_freq_iqr"] = np.subtract(*np.percentile(fe[be][pk.argmax(1)], [75, 25]))
    return f

offsets = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["eeg_id", "eeg_sub_id", "eeg_label_offset_seconds"])
offsets["file"] = offsets.eeg_id.astype(str) + "_" + offsets.eeg_sub_id.astype(str) + ".npy"; off = offsets.set_index("file").eeg_label_offset_seconds
cache = os.path.join(OUT, "a8_periodicity_features.csv")
if os.path.exists(cache): P = pd.read_csv(cache, index_col=0)
else:
    P = pd.DataFrame(Parallel(n_jobs=6, verbose=1)(delayed(feats)(f, off[f]) for f in sub.file), index=sub.file); P.to_csv(cache)
R = {}
S = pd.read_csv(os.path.join(OUT, "a6_features.csv"), index_col=0).loc[sub.file]
def cv(X, y, folds):
    sc = np.zeros((len(y), 4))
    for k in np.unique(folds):
        w = len(y[folds != k]) / (4 * np.bincount(y[folds != k], minlength=4))
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(X[folds != k], y[folds != k], sample_weight=w[y[folds != k]]); sc[folds == k] = c.predict_proba(X[folds == k])
    return sc
y = sub.y.values; folds = sub.fold.values
for name, X in [("periodicity_only", P.values), ("static_only", S.values), ("static_plus_periodicity", np.c_[S.values, P.values])]:
    sc = cv(X, y, folds); R[name] = {"macro_auc": float(roc_auc_score(y, sc, multi_class="ovr", average="macro")),
        "periodic_vs_nonperiodic_auc": float(roc_auc_score(np.isin(y, [1, 2]), sc[:, 1] + sc[:, 2])),
        "LPD_vs_GPD_auc": float(roc_auc_score(y[np.isin(y, [1, 2])] == 1, (sc[:, 1] - sc[:, 2])[np.isin(y, [1, 2])]))}
    print(name, R[name])
R["n_per_class"] = int(sub.groupby("y").size().min()); R["single_feature_auc_periodic_vs_not"] = {c: float(roc_auc_score(np.isin(y, [1, 2]), P[c])) for c in P.columns}
save("a8_periodicity.json", R)
