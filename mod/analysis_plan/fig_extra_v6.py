"""Two extra figures in the requested journal style.
F9: ROC (per class) / calibration / decision curve / boxplot of predicted probability by true class  (4 panels A-D)
F10: representative raw EEG per class: 10-s trace of one channel, time-frequency, scalp topography of 1-4 Hz power, power spectrum
"""
import os, io, zipfile, json, numpy as np, pandas as pd
from scipy.signal import welch, spectrogram, butter, sosfiltfilt
from scipy.interpolate import griddata
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from common import ROOT, OUT, CH, pooled_predictions, load_meta

FIG = os.path.join(OUT, "figs_v5")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "axes.linewidth": 0.9, "legend.frameon": False})
CLS = ["Seizure", "LPD", "GPD", "Others"]; COLS = ["#1f4e79", "#2a9d8f", "#e9c46a", "#e76f51"]
def label_panel(ax, letter): ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)

S = pooled_predictions("soft"); P = S[[f"p_{c}" for c in CLS]].values; y = S.y.values
V5 = json.load(open(os.path.join(OUT, "v5_stats.json")))

# ---------------- F9
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
a = ax[0, 0]; label_panel(a, "A")
for i, c in enumerate(CLS):
    fpr, tpr, _ = roc_curve(y == i, P[:, i]); a.plot(fpr, tpr, color=COLS[i], lw=1.4, label=f"{c} (AUC = {roc_auc_score(y == i, P[:, i]):.3f})")
a.plot([0, 1], [0, 1], "r--", lw=0.8); a.set_xlabel("1 − Specificity"); a.set_ylabel("Sensitivity"); a.set_title("ROC curves (patient-grouped, pooled test folds)"); a.legend(loc="lower right", fontsize=8); a.set_xlim(-0.01, 1); a.set_ylim(0, 1.01)
a = ax[0, 1]; label_panel(a, "B")
conf = P.max(1); corr = (P.argmax(1) == y); edges = np.linspace(0, 1, 11); xs, ys_ = [], []
for b in range(10):
    m = (conf > edges[b]) & (conf <= edges[b + 1])
    if m.sum() > 50: xs.append(conf[m].mean()); ys_.append(corr[m].mean())
a.plot(xs, ys_, "s-", color="#1f77b4", ms=5, label="Soft-label hybrid GAT")
H = pooled_predictions("hard"); PH = H[[f"p_{c}" for c in CLS]].values; confh = PH.max(1); corrh = (PH.argmax(1) == H.y.values); xh, yh = [], []
for b in range(10):
    m = (confh > edges[b]) & (confh <= edges[b + 1])
    if m.sum() > 50: xh.append(confh[m].mean()); yh.append(corrh[m].mean())
a.plot(xh, yh, "o-", color="#ff7f0e", ms=5, label="Hard-label hybrid GAT"); a.plot([0, 1], [0, 1], "k--", lw=0.9, label="Ideal")
a.set_xlabel("Mean predicted probability"); a.set_ylabel("Fraction correct"); a.set_title("Calibration curve"); a.legend(loc="upper left", fontsize=8); a.grid(axis="y", ls=":", lw=0.5)
bs = V5["balanced"]["soft"]; a.text(0.98, 0.06, f"Brier score: {bs['brier']:.4f}\nExpected Calibration Error (ECE): {bs['ece']:.4f}\nMaximum Calibration Error (MCE): {bs['mce']:.4f}", transform=a.transAxes, ha="right", va="bottom", fontsize=9)
a = ax[1, 0]; label_panel(a, "C"); TH = np.linspace(0.02, 0.9, 60); d = V5["dca"]["Seizure"]
a.plot(TH, d["treat_all"], color="k", label="Treat all"); a.axhline(0, color="#666", ls="--", lw=0.9, label="Treat none"); a.plot(TH, d["soft"], color="#2a2ad0", lw=1.3, label="Hybrid GAT (seizure vs rest)")
a.set_ylim(-0.1, 0.25); a.set_xlabel("Threshold probability"); a.set_ylabel("Net benefit"); a.set_title("Decision curve analysis (native seizure prevalence 0.20)"); a.legend(fontsize=8)
a = ax[1, 1]; label_panel(a, "D")
data = [100 * P[y == i, i] for i in range(4)]
bp = a.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, medianprops=dict(color="k", lw=1.3))
for patch, col in zip(bp["boxes"], COLS): patch.set_facecolor(col); patch.set_alpha(0.75)
rng = np.random.default_rng(0)
for i in range(4):
    sub = rng.choice(data[i], 60, replace=False); a.scatter(np.full(60, i + 1) + rng.normal(0, 0.06, 60), sub, s=8, marker="x", color=COLS[i], alpha=0.8)
a.set_xticks(range(1, 5)); a.set_xticklabels(CLS); a.set_ylabel("Model probability of the expert label (%)"); a.set_title("Predicted probability by true class"); a.set_ylim(0, 105)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F9_roc_cal_dca_box.png"), dpi=int(os.environ.get("FIGDPI","200")), bbox_inches="tight"); plt.close(); print("F9 saved")

# ---------------- F10 representative EEG
meta = load_meta(); off = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["eeg_id", "eeg_sub_id", "eeg_label_offset_seconds"]); off["file"] = off.eeg_id.astype(str) + "_" + off.eeg_sub_id.astype(str) + ".npy"; OFF = off.set_index("file").eeg_label_offset_seconds
FS = 200; sos = butter(4, [0.5, 40], btype="band", fs=FS, output="sos")
XY = {'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95), 'F7': (-0.81, 0.59), 'F3': (-0.41, 0.52), 'Fz': (0, 0.55), 'F4': (0.41, 0.52), 'F8': (0.81, 0.59), 'T3': (-1, 0), 'C3': (-0.5, 0), 'Cz': (0, 0), 'C4': (0.5, 0), 'T4': (1, 0), 'T5': (-0.81, -0.59), 'P3': (-0.41, -0.52), 'Pz': (0, -0.55), 'P4': (0.41, -0.52), 'T6': (0.81, -0.59), 'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)}
def topomap(ax, vals, cmap="RdYlBu_r"):
    xs = np.array([XY[c][0] for c in CH]); ys = np.array([XY[c][1] for c in CH]); gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, 150), np.linspace(-1.1, 1.1, 150))
    z = griddata((xs, ys), vals, (gx, gy), method="cubic"); z = np.where(np.isnan(z), griddata((xs, ys), vals, (gx, gy), method="nearest"), z); z[(gx ** 2 + gy ** 2) > 1.1 ** 2] = np.nan
    im = ax.imshow(z, extent=[-1.1, 1.1, -1.1, 1.1], origin="lower", cmap=cmap); ax.add_patch(Circle((0, 0), 1.1, fill=False, lw=1.2, color="k")); ax.add_patch(Polygon([[-0.1, 1.08], [0, 1.25], [0.1, 1.08]], closed=True, fill=False, lw=1.2, color="k"))
    ax.scatter(xs, ys, s=10, c="k"); ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.15, 1.3); ax.set_aspect("equal"); ax.axis("off"); return im
# pick one unanimous, correctly classified window per class with many votes
S["pred_ok"] = S.pred == S.y
picks = {}
for i, c in enumerate(CLS):
    d = S[(S.y == i) & (S.cons >= 0.999) & (S.nv >= 10) & S.pred_ok].sort_values(f"p_{c}", ascending=False); picks[c] = d.file.iloc[0]
print(picks)
fig = plt.figure(figsize=(13, 11.5)); gs = fig.add_gridspec(4, 4, width_ratios=[2.2, 1.6, 1.45, 1.2], hspace=0.55, wspace=0.3)
chan = {"Seizure": "T4", "LPD": "O1", "GPD": "Cz", "Others": "Fz"}; first_row = []
with zipfile.ZipFile(os.path.join(ROOT, "train_eegs.zip")) as z:
    for r, c in enumerate(CLS):
        f = picks[c]; eeg_id = f.split("_")[0]; t = pd.read_parquet(io.BytesIO(z.read(f"train_eegs/{eeg_id}.parquet")))
        a0 = int(OFF[f] * FS); x = np.nan_to_num(t[CH].values.T.astype(float))[:, a0:a0 + 50 * FS]; x = sosfiltfilt(sos, x, axis=1)
        ci = CH.index(chan[c]); seg = x[ci, 20 * FS:30 * FS]; tt = np.arange(len(seg)) / FS
        ax = fig.add_subplot(gs[r, 0]); ax.plot(tt, seg, color="k", lw=0.6); ax.set_xlim(0, 10); ax.set_yticks([]); ax.set_xlabel("Time (s)" if r == 3 else ""); ax.set_title(f"{c}: 10-s epoch at {chan[c]} (labelled interval)", loc="left", fontsize=10)
        for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
        rng_ = seg.max() - seg.min(); ax.plot([8.5, 9.5], [seg.min() - 0.08 * rng_] * 2, color="k", lw=2); ax.text(9.0, seg.min() - 0.22 * rng_, "1 s", ha="center", va="top", fontsize=8); ax.set_ylim(seg.min() - 0.35 * rng_, seg.max() + 0.05 * rng_)
        if r == 0: first_row.append(ax)
        ax = fig.add_subplot(gs[r, 1]); fq, ts, Sx = spectrogram(x[ci], fs=FS, nperseg=FS, noverlap=int(FS * 0.9)); m = fq <= 30
        ax.pcolormesh(ts, fq[m], 10 * np.log10(Sx[m] + 1e-12), shading="gouraud", cmap="hot"); ax.set_ylabel("Frequency (Hz)"); ax.set_xlabel("Time (s)" if r == 3 else ""); ax.set_title(f"Time–frequency at {chan[c]}, 50 s", loc="left", fontsize=10); ax.axvspan(20, 30, color="w", alpha=0.0); ax.axvline(20, color="w", ls="--", lw=0.7); ax.axvline(30, color="w", ls="--", lw=0.7)
        if r == 0: first_row.append(ax)
        ax = fig.add_subplot(gs[r, 2]); fw, Pw = welch(x[:, 20 * FS:30 * FS], fs=FS, nperseg=FS, axis=1); bp = np.log10(Pw[:, (fw >= 1) & (fw <= 4)].mean(1)); im = topomap(ax, bp); ax.set_title("δ power (1–4 Hz), labelled 10 s", fontsize=10, pad=2)
        if r == 0: first_row.append(ax)
        ax = fig.add_subplot(gs[r, 3]); fw2, Pw2 = welch(x[ci, 20 * FS:30 * FS], fs=FS, nperseg=FS * 2); mm = fw2 <= 10; ax.plot(fw2[mm], Pw2[mm], color="r", lw=1); ax.set_xlabel("Frequency (Hz)" if r == 3 else ""); ax.set_ylabel("Power (a.u.)"); ax.set_title(f"Power spectrum at {chan[c]}", loc="left", fontsize=10)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        if r == 0: first_row.append(ax)
fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
tbs = [a.get_tightbbox(rend).transformed(inv) for a in first_row]; top = max(tb.y1 for tb in tbs)
for tb, L in zip(tbs, "ABCD"): fig.text(tb.x0, top + 0.004, L, fontsize=16, fontweight="bold", ha="left", va="bottom")
plt.savefig(os.path.join(FIG, "F10_representative_eeg.png"), dpi=int(os.environ.get("FIGDPI","200")), bbox_inches="tight"); plt.close(); print("F10 saved")
