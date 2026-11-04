"""Figure (data and evaluation design), redrawn with non-overlapping labels and legends."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT, OUT, CH, load_splits, pooled_predictions, ensure_fc

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9.5, "legend.frameon": False})
CLS = ["Seizure", "LPD", "GPD", "Others"]; COLS = ["#1f4e79", "#2a9d8f", "#e9c46a", "#e76f51"]
splits = load_splits(); H = pooled_predictions("hard"); y = H.y.values
V = H[[f"v_{c}" for c in CLS]].values; vote = {c: V[y == i].mean(0) for i, c in enumerate(CLS)}
rng = np.random.default_rng(7); files = H.file.values
sample = {c: rng.choice(np.where(y == i)[0], 300, replace=False) for i, c in enumerate(CLS)}
ensure_fc(["coh"], files=[files[j] for s in sample.values() for j in s])
meanFC = {c: np.mean([np.abs(np.load(os.path.join(ROOT, "fc_sb", "coh", files[j]))) for j in sample[c]], 0) for c in CLS}

fig = plt.figure(figsize=(12, 8.2)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.35, 0.95], wspace=0.6, left=0.07, right=0.97, top=0.94, bottom=0.62); gsD = fig.add_gridspec(1, 4, wspace=0.28, left=0.07, right=0.93, top=0.46, bottom=0.07)
def letter(ax, L): ax.annotate(L, xy=(0, 1), xycoords="axes fraction", xytext=(-40, 28), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
# A
ax = fig.add_subplot(gs[0, 0:2]); letter(ax, "A")
names = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]; nat = [20933, 14856, 16702, 16640, 18861, 18806]; used = [9000, 9000, 9000, 3000, 3000, 3000]
x = np.arange(6); ax.bar(x - 0.2, nat, 0.4, color="#bbbbbb", edgecolor="k", lw=0.4, label="Native (106,798 windows)"); ax.bar(x + 0.2, used, 0.4, color="#1f4e79", edgecolor="k", lw=0.4, label="Analysed (36,000 windows)")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Labelled 50-s windows"); ax.set_ylim(0, 26000); ax.legend(loc="upper right", ncol=1, fontsize=8.5); ax.set_title("Dataset composition and class balancing", pad=26)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
# B
ax = fig.add_subplot(gs[0, 2]); letter(ax, "B"); bottom = np.zeros(4)
for i, c in enumerate(CLS):
    vals = [vote[k][i] for k in CLS]; ax.bar(range(4), vals, bottom=bottom, color=COLS[i], edgecolor="k", lw=0.4, label=c); bottom += vals
ax.set_xticks(range(4)); ax.set_xticklabels(CLS, fontsize=8, rotation=20, ha="right"); ax.set_ylabel("Mean annotator vote share"); ax.set_ylim(0, 1); ax.set_title("Votes by modal label", pad=26)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, fontsize=7.5, title="Vote for", title_fontsize=7.5, handlelength=1.2, borderaxespad=0)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
# C
ax = fig.add_subplot(gs[0, 3]); letter(ax, "C"); ax.bar(range(1, 11), [s["n_unique_patients_test"] for s in splits], color="#1f4e79", edgecolor="k", lw=0.4)
ax.set_xlabel("Test fold"); ax.set_ylabel("Held-out patients"); ax.set_title("Patients per test fold", pad=26); ax.set_xticks([1, 5, 10])
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
# D
axes = []
for i, c in enumerate(CLS):
    ax = fig.add_subplot(gsD[0, i]); axes.append(ax)
    if i == 0: letter(ax, "D")
    im = ax.imshow(meanFC[c], cmap="viridis", vmin=0.1, vmax=0.6); ax.set_title(f"Mean coherence, {c}", fontsize=9.5)
    tk = range(0, 19, 2); ax.set_xticks(tk); ax.set_xticklabels([CH[j] for j in tk], fontsize=7, rotation=90); ax.set_yticks(tk); ax.set_yticklabels([CH[j] for j in tk], fontsize=7)
cb = fig.colorbar(im, ax=axes, shrink=0.62, pad=0.012, aspect=22); cb.set_label("|coherence|")
plt.savefig(os.path.join(OUT, "figs_v5", "F1_design.png"), dpi=int(os.environ.get("FIGDPI","200")), bbox_inches="tight"); print("saved")
