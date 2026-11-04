"""R3. Features from raw EEG for the labelled central 10 s and for the time axis.

For every analysed window (36,000; read each recording's parquet once):
  centre-10 s:  PSD (19x101, Welch 1-s Hann, 50% overlap) and corr/coh/aec/plv (19x19) on samples [20 s, 30 s)
  full 50 s:    the same recomputed here, as a check against the stored features
  dynamic:      2-s windows, 0.5-s step over 50 s -> global |corr| time series g(t); Welch peak frequency and
                prominence in 0.3-3 Hz, spectral entropy, std, lag-1 autocorrelation, fraction above mean+1sd;
                per-channel Hilbert-envelope periodicity (median/max peak prominence, median peak frequency, IQR)
Outputs: results/r3_features/<file>.npz  and results/r3_index.csv
Requires: pyarrow. Runtime: hours (16 GB zip); run in background.
"""
import os, io, zipfile, json, numpy as np, pandas as pd
from scipy.signal import welch, hilbert, butter, sosfiltfilt, coherence
from joblib import Parallel, delayed
from common import ROOT, OUT, CH, load_splits, load_meta

FS = 200; ZIP = os.path.join(ROOT, "train_eegs.zip"); OUTD = os.path.join(OUT, "r3_features"); os.makedirs(OUTD, exist_ok=True)
splits = load_splits(); meta = load_meta()
files = sorted(set(f for s in splits for part in ("train", "val", "test") for f in s[part]["files"]))
off = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["eeg_id", "eeg_sub_id", "eeg_label_offset_seconds"])
off["file"] = off.eeg_id.astype(str) + "_" + off.eeg_sub_id.astype(str) + ".npy"; OFF = off.set_index("file").eeg_label_offset_seconds.to_dict()
by_rec = {}
for f in files: by_rec.setdefault(f.split("_")[0], []).append(f)
sos = butter(5, [0.5, 50], btype="band", fs=FS, output="sos")

def fc_set(x):
    n = x.shape[0]; corr = np.corrcoef(x); an = hilbert(x, axis=1); ph = np.angle(an); env = np.abs(an)
    plv = np.abs(np.exp(1j * ph) @ np.exp(-1j * ph).T) / x.shape[1]; aec = np.corrcoef(env)
    coh = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            f_, c = coherence(x[i], x[j], fs=FS, nperseg=min(256, x.shape[1])); m = (f_ >= 0.5) & (f_ <= 50); coh[i, j] = coh[j, i] = c[m].mean()
    return {"corr": corr.astype(np.float32), "coh": coh.astype(np.float32), "aec": aec.astype(np.float32), "plv": plv.astype(np.float32)}

def psd(x):
    f_, P = welch(x, fs=FS, nperseg=FS, noverlap=FS // 2, axis=1); return P[:, :101].astype(np.float32)

def dyn(x):
    W, S = 2 * FS, FS // 2; g = []
    for s0 in range(0, x.shape[1] - W + 1, S): C = np.corrcoef(x[:, s0:s0 + W]); g.append(np.abs(C[np.triu_indices(19, 1)]).mean())
    g = np.array(g); fg, Pg = welch(g - g.mean(), fs=FS / S, nperseg=min(64, len(g))); b = (fg >= 0.3) & (fg <= 3); pn = Pg / Pg.sum()
    env = np.abs(hilbert(x, axis=1)); fe, Pe = welch(env - env.mean(1, keepdims=True), fs=FS, nperseg=FS * 4, axis=1); be = (fe >= 0.3) & (fe <= 3); pk = Pe[:, be]; prom = pk.max(1) / (pk.mean(1) + 1e-12)
    return np.array([fg[b][Pg[b].argmax()], Pg[b].max() / (Pg[b].mean() + 1e-12), -(pn * np.log(pn + 1e-12)).sum(), g.std(), np.corrcoef(g[:-1], g[1:])[0, 1], (g > g.mean() + g.std()).mean(),
                     np.median(fe[be][pk.argmax(1)]), np.median(prom), prom.max(), np.subtract(*np.percentile(fe[be][pk.argmax(1)], [75, 25]))], dtype=np.float32)

def do_rec(eeg_id, fl):
    done = [f for f in fl if not os.path.exists(os.path.join(OUTD, f.replace(".npy", ".npz")))]
    if not done: return 0
    with zipfile.ZipFile(ZIP) as z: t = pd.read_parquet(io.BytesIO(z.read(f"train_eegs/{eeg_id}.parquet")))
    sig = np.nan_to_num(t[CH].values.T.astype(np.float64))
    for f in done:
        a = int(OFF[f] * FS); x = sig[:, a:a + 50 * FS]
        if x.shape[1] < 50 * FS: continue
        x = sosfiltfilt(sos, x, axis=1); x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
        c10 = x[:, 20 * FS:30 * FS]; f10 = fc_set(c10); f50 = fc_set(x)
        np.savez_compressed(os.path.join(OUTD, f.replace(".npy", ".npz")), psd10=psd(c10), psd50=psd(x), dyn=dyn(x), **{f"{k}10": v for k, v in f10.items()}, **{f"{k}50": v for k, v in f50.items()})
    return len(done)

if __name__ == "__main__":
    recs = list(by_rec.items()); print(len(recs), "recordings", len(files), "windows", flush=True)
    n = Parallel(n_jobs=5, verbose=2, batch_size=4)(delayed(do_rec)(r, fl) for r, fl in recs)
    print("done", sum(n), flush=True)
