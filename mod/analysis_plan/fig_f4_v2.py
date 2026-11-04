"""Redraw the attention-audit figure with legends clear of the bars."""
import os, numpy as np
from scipy.interpolate import griddata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from common import ROOT, OUT, CH
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "legend.frameon": False})
def label_panel(ax, letter): ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
CLS = ["Seizure", "LPD", "GPD", "Others"]
XY = {'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95), 'F7': (-0.81, 0.59), 'F3': (-0.41, 0.52), 'Fz': (0, 0.55), 'F4': (0.41, 0.52), 'F8': (0.81, 0.59), 'T3': (-1, 0), 'C3': (-0.5, 0), 'Cz': (0, 0), 'C4': (0.5, 0), 'T4': (1, 0), 'T5': (-0.81, -0.59), 'P3': (-0.41, -0.52), 'Pz': (0, -0.55), 'P4': (0.41, -0.52), 'T6': (0.81, -0.59), 'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)}
def topomap(ax, vals, title):
    xs = np.array([XY[c][0] for c in CH]); ys = np.array([XY[c][1] for c in CH]); gx, gy = np.meshgrid(np.linspace(-1.15, 1.15, 160), np.linspace(-1.15, 1.15, 160))
    z = griddata((xs, ys), vals, (gx, gy), method="cubic"); z = np.where(np.isnan(z), griddata((xs, ys), vals, (gx, gy), method="nearest"), z); z[(gx ** 2 + gy ** 2) > 1.15 ** 2] = np.nan
    im = ax.imshow(z, extent=[-1.15, 1.15, -1.15, 1.15], origin="lower", cmap="magma", vmin=0, vmax=1)
    ax.add_patch(Circle((0, 0), 1.15, fill=False, lw=1.2, color="k")); ax.add_patch(Polygon([[-0.12, 1.13], [0, 1.3], [0.12, 1.13]], closed=True, fill=False, lw=1.2, color="k"))
    ax.scatter(xs, ys, s=8, c="k"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.3, 1.4); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10, pad=4); return im
att = np.load(os.path.join(ROOT, "mod", "analysis_patient_grouped", "attention_rollout_class_matrices.npz"))
fig = plt.figure(figsize=(11, 6.6)); gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.5, wspace=0.25, left=0.07, right=0.90, top=0.92, bottom=0.12)
for i, (c, k) in enumerate(zip(CLS, ["SEIZURE", "LPD", "GPD", "OTHERS"])):
    ax = fig.add_subplot(gs[0, i])
    if i == 0: label_panel(ax, "A")
    M = att[k]; M = M.mean(0) if M.ndim > 2 else M; v = M.mean(0); v = (v - v.min()) / (v.max() - v.min() + 1e-9); im = topomap(ax, v, f"{c}: attention received")
cax = fig.add_axes([0.915, 0.58, 0.012, 0.28]); cb = fig.colorbar(im, cax=cax); cb.set_label("Rollout attention (max-normalized)", fontsize=9)
ax = fig.add_subplot(gs[1, 0:2]); label_panel(ax, "B")
drops = {"Seizure": (0.0002, -0.0004, 0.0014, 0.258), "LPD": (0.0008, 0.0010, 0.0018, 0.839), "GPD": (0.0002, 0.0009, 0.0014, 0.032), "Others": (-0.0004, -0.0007, 0.0009, 1.0)}; x = np.arange(4)
ax.bar(x - 0.2, [drops[c][0] for c in CLS], 0.4, color="#1f4e79", label="Ablate top-10 attention edges", edgecolor="k", lw=0.5)
ax.bar(x + 0.2, [drops[c][1] for c in CLS], 0.4, yerr=[drops[c][2] for c in CLS], color="#bbbbbb", capsize=3, label="Ablate 10 random edges (30 draws, ± SD)", edgecolor="k", lw=0.5)
for i, c in enumerate(CLS): ax.text(i, 0.0031, f"P = {drops[c][3]:.2f}", ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(CLS); ax.set_ylabel("Accuracy drop"); ax.set_ylim(-0.0027, 0.0038)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); ax.set_title("Causal edge ablation vs random-edge null (fold 1)")
ax = fig.add_subplot(gs[1, 2:4]); label_panel(ax, "C")
seed = {"Seizure": (0.442, 0.912), "LPD": (0.378, 0.550), "GPD": (0.660, 0.875), "Others": (0.294, 0.352)}
ax.bar(x - 0.2, [seed[c][0] for c in CLS], 0.4, color="#1f4e79", label="Top-10 Jaccard across 5 seeds", edgecolor="k", lw=0.5); ax.bar(x + 0.2, [seed[c][1] for c in CLS], 0.4, color="#c0504d", label="r(real-label map, permuted-label map)", edgecolor="k", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(CLS); ax.set_ylim(0, 1.05); ax.set_ylabel("Value"); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1); ax.set_title("Reproducibility and label dependence (fold 1)")
plt.savefig(os.path.join(OUT, "figs_v5", "F4_attention_audit.png"), dpi=int(os.environ.get("FIGDPI","200")), bbox_inches="tight"); print("F4 saved")
