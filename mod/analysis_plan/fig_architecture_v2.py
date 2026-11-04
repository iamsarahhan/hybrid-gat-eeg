"""Figure 1 (architecture) drawn with real data flowing through: the seizure window from Figure 2.
Row A: raw 19-channel trace -> per-channel PSD (real) -> four connectivity matrices (real) -> electrode graph with edges weighted by real coherence
Row B: network pathway (plain blocks with tensor shapes), matrix pathway, fusion, and the model's actual output probabilities for this window.
"""
import os, io, zipfile, numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon
from common import ROOT, OUT, CH, pooled_predictions

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8.5})
FILE = "1969473879_1.npy"; FS = 200
S = pooled_predictions("soft"); row = S[S.file == FILE].iloc[0]; probs = [row[f"p_{c}"] for c in ["Seizure", "LPD", "GPD", "Others"]]
off = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["eeg_id", "eeg_sub_id", "eeg_label_offset_seconds"]); off["file"] = off.eeg_id.astype(str) + "_" + off.eeg_sub_id.astype(str) + ".npy"
a0 = int(off.set_index("file").eeg_label_offset_seconds[FILE] * FS)
with zipfile.ZipFile(os.path.join(ROOT, "train_eegs.zip")) as z: t = pd.read_parquet(io.BytesIO(z.read(f"train_eegs/{FILE.split('_')[0]}.parquet")))
x = np.nan_to_num(t[CH].values.T.astype(float))[:, a0:a0 + 50 * FS]; x = sosfiltfilt(butter(5, [0.5, 50], btype="band", fs=FS, output="sos"), x, axis=1)
psd = np.load(os.path.join(ROOT, "psd_sb", FILE)); fc = {m: np.load(os.path.join(ROOT, "fc_sb", m, FILE)) for m in ["corr", "coh", "aec", "plv"]}
XY = {'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95), 'F7': (-0.81, 0.59), 'F3': (-0.41, 0.52), 'Fz': (0, 0.55), 'F4': (0.41, 0.52), 'F8': (0.81, 0.59), 'T3': (-1, 0), 'C3': (-0.5, 0), 'Cz': (0, 0), 'C4': (0.5, 0), 'T4': (1, 0), 'T5': (-0.81, -0.59), 'P3': (-0.41, -0.52), 'Pz': (0, -0.55), 'P4': (0.41, -0.52), 'T6': (0.81, -0.59), 'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)}

fig = plt.figure(figsize=(12.5, 8.6))
def block(ax, x, y, w, h, lines, fc="#f2f2f2", ec="k", lw=0.8, fs=8, bold_first=False):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw));
    for k, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - (k + 0.5) * h / len(lines) if len(lines) > 1 else y + h / 2, ln, ha="center", va="center", fontsize=fs, fontweight="bold" if (bold_first and k == 0) else "normal")
def arr(ax, x0, y0, x1, y1, lw=0.9, color="k", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0))

# ======================= Row A =======================
fig.text(0.012, 0.965, "A", fontsize=15, fontweight="bold")
# A1 raw traces (10 s labelled interval, all 19 channels)
ax = fig.add_axes([0.03, 0.56, 0.20, 0.38]); seg = x[:, 20 * FS:30 * FS]; seg = seg / (np.abs(seg).max() + 1e-9)
for i in range(19): ax.plot(np.arange(seg.shape[1]) / FS, seg[i] * 0.45 - i, color="k", lw=0.4)
ax.set_yticks(-np.arange(19)); ax.set_yticklabels(CH, fontsize=6); ax.set_xlim(0, 10); ax.set_ylim(-18.8, 0.8); ax.set_xlabel("Time (s)", fontsize=7); ax.tick_params(axis="x", labelsize=6)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.set_title("19-channel EEG, 50-s window (10 s shown)\n0.5–50 Hz, clipped, z-scored", fontsize=8, loc="left")
# A2 PSD heatmap
ax = fig.add_axes([0.285, 0.775, 0.14, 0.15]); ax.imshow(np.log10(psd[:, :50] + 1e-12), aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks([0, 9, 18]); ax.set_yticklabels([CH[0], CH[9], CH[18]], fontsize=6); ax.set_xticks([0, 25, 49]); ax.set_xticklabels(["0", "25", "50 Hz"], fontsize=6)
ax.set_title("Node features: Welch PSD\n19 × 101 (log power, 0–50 Hz shown)", fontsize=8, loc="left")
# A3 four connectivity matrices
names = {"corr": "Correlation", "coh": "Coherence", "aec": "AEC", "plv": "PLV"}
for k, m in enumerate(["corr", "coh", "aec", "plv"]):
    ax = fig.add_axes([0.285 + (k % 2) * 0.072, 0.525 + (1 - k // 2) * 0.078, 0.062, 0.062]); ax.imshow(np.abs(fc[m]), cmap="Greys", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(names[m], fontsize=7, pad=2)
fig.text(0.285, 0.70, "Edge attributes: 4 connectivity measures\n171 pairs × 4 (one 19 × 19 matrix each)", fontsize=8, va="bottom")
# A4 electrode graph with real coherence edges
ax = fig.add_axes([0.48, 0.56, 0.22, 0.38]); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((0, 0), 1.12, fill=False, lw=1.2, color="k")); ax.add_patch(Polygon([[-0.1, 1.1], [0, 1.26], [0.1, 1.1]], closed=True, fill=False, lw=1.2, color="k"))
W = np.abs(fc["coh"]); thr = np.percentile(W[np.triu_indices(19, 1)], 70)
for i in range(19):
    for j in range(i + 1, 19):
        w = W[i, j]; (x0, y0), (x1, y1) = XY[CH[i]], XY[CH[j]]
        ax.plot([x0, x1], [y0, y1], color="#1f4e79" if w >= thr else "#c8ccd2", lw=0.3 + 2.2 * w, alpha=0.9 if w >= thr else 0.5, zorder=1)
for c in CH: ax.add_patch(Circle(XY[c], 0.085, fc="white", ec="k", lw=0.8, zorder=3)); ax.text(XY[c][0], XY[c][1], c, ha="center", va="center", fontsize=5.5, zorder=4)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.25, 1.35); ax.set_title("Brain graph for this window\n19 nodes, 171 edges (line width = coherence;\nblue: top 30%)", fontsize=8, loc="left")
# A5 flattened vector
ax = fig.add_axes([0.74, 0.60, 0.235, 0.05]); v = np.concatenate([np.abs(fc[m]).ravel() for m in ["corr", "coh", "aec", "plv"]]); ax.imshow(v[None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1); ax.set_yticks([]); ax.set_xticks([0, 361, 722, 1083, 1443]); ax.set_xticklabels(["corr", "coh", "aec", "plv", ""], fontsize=6)
ax.set_title("Flattened connectivity vector, 1 × 1,444\n(same four matrices, no electrode positions)", fontsize=8, loc="left")
fig.text(0.74, 0.80, "Labelled interval: central 10 s\nModel input: full 50 s\nAnnotators: up to 28 per window\nTarget: normalized vote distribution", fontsize=8, va="top", bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="k", lw=0.6))
# arrows row A (figure coords)
axF = fig.add_axes([0, 0, 1, 1]); axF.axis("off"); axF.set_xlim(0, 1); axF.set_ylim(0, 1); axF.set_zorder(-1)
arr(axF, 0.235, 0.77, 0.28, 0.84); arr(axF, 0.235, 0.77, 0.28, 0.66); arr(axF, 0.43, 0.84, 0.48, 0.80); arr(axF, 0.43, 0.66, 0.48, 0.72); arr(axF, 0.43, 0.63, 0.74, 0.625)

# ======================= Row B =======================
fig.text(0.012, 0.47, "B", fontsize=15, fontweight="bold")
axB = fig.add_axes([0.03, 0.05, 0.95, 0.42]); axB.set_xlim(0, 100); axB.set_ylim(0, 39); axB.axis("off")
# network pathway
block(axB, 1, 26, 10, 9, ["Graph", "x: 19 × 101", "E: 342 × 4"], fc="white", bold_first=True)
dims = [("GATConv 1", "13 heads × 49", "edge_dim = 4", "→ 19 × 637"), ("GATConv 2", "13 heads × 49", "", "→ 19 × 637"), ("GATConv 3", "13 heads × 49", "", "→ 19 × 637"), ("GATConv 4", "1 head × 49", "(heads averaged)", "→ 19 × 49")]
for k, d in enumerate(dims):
    xk = 14 + k * 12.5; block(axB, xk, 26, 10.5, 9, [l for l in d if l], fc="#e8edf3", bold_first=True, fs=7.5)
    arr(axB, xk - 2, 30.5, xk, 30.5)
axB.text(14 + 1.5 * 12.5 + 5.25, 24.4, "ELU and dropout 0.25 after each layer", ha="center", fontsize=7.5, style="italic")
block(axB, 64.5, 26, 9.5, 9, ["Global mean", "pool", "→ 1 × 49"], fc="#e8edf3", bold_first=True, fs=7.5); arr(axB, 62, 30.5, 64.5, 30.5)
axB.text(1, 36.5, "Network pathway", fontsize=9, fontweight="bold", ha="left")
# matrix pathway
block(axB, 1, 8, 10, 7, ["Flattened", "connectivity", "1 × 1,444"], fc="white", bold_first=True)
block(axB, 14, 8, 20, 7, ["Linear 1,444 → 64", "ReLU, dropout 0.25", "→ 1 × 64"], fc="#fbf3dc", bold_first=True, fs=7.5); arr(axB, 11, 11.5, 14, 11.5)
axB.text(1, 16.5, "Matrix pathway", fontsize=9, fontweight="bold", ha="left")
# fusion
block(axB, 78, 15, 10, 12, ["Concatenate", "49 + 64 = 113", "Linear 113 → 49, ELU", "Linear 49 → 4", "softmax"], fc="white", bold_first=True, fs=7.5)
arr(axB, 74, 30.5, 78, 25); arr(axB, 34, 11.5, 78, 18)
# output: real probabilities for this window
axo = fig.add_axes([0.90, 0.19, 0.06, 0.14]); axo.barh(range(4), probs, color=["#1f4e79", "#2a9d8f", "#e9c46a", "#e76f51"], edgecolor="k", lw=0.5)
axo.set_yticks(range(4)); axo.set_yticklabels(["Seizure", "LPD", "GPD", "Others"], fontsize=7); axo.yaxis.tick_right(); axo.invert_yaxis(); axo.set_xlim(0, 1); axo.set_xticks([0, 0.5, 1]); axo.tick_params(axis="x", labelsize=6)
for i, pv in enumerate(probs): axo.text(pv + 0.03 if pv < 0.6 else pv - 0.03, i, f"{pv:.2f}", va="center", ha="left" if pv < 0.6 else "right", fontsize=6.5, color="k" if pv < 0.6 else "white")
axo.set_title("Output for this\nwindow (soft-label\nmodel)", fontsize=7.5, loc="left")
for sp in ["top", "right"]: axo.spines[sp].set_visible(False)
arr(axF, 0.868, 0.25, 0.90, 0.25)
# attention equation, boxed, plain
axB.text(36, 2.6, r"Attention per edge:  $\alpha_{ij}=\mathrm{softmax}_j\!\left(\mathrm{LeakyReLU}\left(a^{\top}[Wh_i \,\|\, Wh_j \,\|\, W_e e_{ij}]\right)\right)$,   node update:  $h_i'=\mathrm{ELU}\!\left(\sum_{j\in N(i)} \alpha_{ij} W h_j\right)$", fontsize=8.5, ha="left", va="center", bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="k", lw=0.6))
plt.savefig(os.path.join(OUT, "figs_v5", "F0_architecture.png"), dpi=int(os.environ.get("FIGDPI","220")), bbox_inches="tight"); print("saved", [round(p, 3) for p in probs])
