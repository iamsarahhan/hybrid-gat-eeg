"""Shared loaders for the analysis plan (items A1-A8).

Everything here reads only files already in the handover folder:
  patient_split/fold_splits_patient.json      the single patient-grouped split
  patient_split/result_*/fold_k/y_*.npy        saved out-of-fold predictions
  train.csv                                     labels, patient_id, expert votes
  fc_sb/<measure>/<file>.npy, psd_sb/<file>.npy per-segment features
"""
import os, json, zipfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
CLASSES = ["Seizure", "LPD", "GPD", "Others"]
VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
# Channel order of the stored psd_sb / fc_sb files = HMS parquet column order (verified 29 Oct 2026 against raw EEG);
# the 10-20 list previously used here mislabelled every electrode-level result.
CH = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
CH_1020_OLD = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
LEFT  = ['Fp1','F7','F3','T3','C3','T5','P3','O1']
RIGHT = ['Fp2','F8','F4','T4','C4','T6','P4','O2']
VARIANTS = {"hard": "result_4class_patient_hybrid_auc",
            "soft": "result_4class_patient_softlabel_hybrid_auc",
            "gmm":  "result_4class_patient_gmmgumbel_hybrid_auc"}

def load_splits():
    return json.load(open(os.path.join(ROOT, "patient_split", "fold_splits_patient.json")))

def load_meta():
    t = pd.read_csv(os.path.join(ROOT, "train.csv"))
    t["file"] = t.eeg_id.astype(str) + "_" + t.eeg_sub_id.astype(str) + ".npy"
    v = t[VOTE_COLS].values.astype(float)
    t["n_votes"] = v.sum(1)
    v4 = np.stack([v[:, 0], v[:, 1], v[:, 2], v[:, 3:].sum(1)], 1)
    for i, c in enumerate(CLASSES): t[f"vote_{c}"] = v4[:, i] / np.maximum(v4.sum(1), 1)
    t["consensus_frac"] = v4.max(1) / np.maximum(v4.sum(1), 1)
    return t.set_index("file")

def pooled_predictions(variant="hard"):
    """Pooled out-of-fold predictions with metadata. Row order == fold test order (verified by label match)."""
    splits, meta = load_splits(), load_meta()
    rows = []
    for k, f in enumerate(splits):
        d = os.path.join(ROOT, "patient_split", VARIANTS[variant], f"fold_{k+1}")
        yt, yp, ys = (np.load(os.path.join(d, n)) for n in ("y_true.npy", "y_pred.npy", "y_score.npy"))
        assert (yt == np.array(f["test"]["labels_int"])).all(), f"fold {k+1}: prediction order mismatch"
        m = meta.loc[f["test"]["files"]]
        df = pd.DataFrame({"file": f["test"]["files"], "fold": k + 1, "y": yt, "pred": yp,
                           "eeg_id": m.eeg_id.values, "patient": m.patient_id.values, "sub": m.expert_consensus.values,
                           "cons": m.consensus_frac.values, "nv": m.n_votes.values})
        for i, c in enumerate(CLASSES):
            df[f"p_{c}"] = ys[:, i]; df[f"v_{c}"] = m[f"vote_{c}"].values
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

def ensure_fc(measures=("corr", "coh", "aec", "plv"), files=None):
    """Extract any fc_sb/<measure>/<file> missing on disk from fc_sb.zip.
    The unzipped coh/aec folders in the handover are one file short, so check per file."""
    if files is None:
        s = load_splits(); files = sorted(set(f for x in s for part in ("train", "val", "test") for f in x[part]["files"]))
    need = [f"fc_sb/{m}/{f}" for m in measures for f in files if not os.path.exists(os.path.join(ROOT, "fc_sb", m, f))]
    if need:
        print(f"extracting {len(need)} connectivity files from fc_sb.zip")
        with zipfile.ZipFile(os.path.join(ROOT, "fc_sb.zip")) as z:
            have = set(z.namelist()); z.extractall(ROOT, [n for n in need if n in have])
    return [os.path.join(ROOT, "fc_sb", m) for m in measures]

def ensure_psd():
    d = os.path.join(ROOT, "psd_sb")
    if not os.path.isdir(d):
        with zipfile.ZipFile(os.path.join(ROOT, "psd_sb.zip")) as z: z.extractall(ROOT)
    return d

def load_fc(file, measures=("corr", "coh", "aec", "plv")):
    return {m: np.load(os.path.join(ROOT, "fc_sb", m, file)) for m in measures}

def clustered_ci(df, stat_fn, n=1000, seed=0):
    rng = np.random.default_rng(seed); pats = df.patient.values; up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}; vals = []
    for _ in range(n):
        ii = np.concatenate([idx[p] for p in rng.choice(up, len(up), replace=True)]); vals.append(stat_fn(df.iloc[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1, default=float); print("saved", name)
