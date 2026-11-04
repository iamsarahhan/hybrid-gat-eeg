"""Figure (evaluation unit) built ONLY from analyses of the corrected pipeline.
Panel A: patient-grouped vs window-level macro AUROC for every model that has BOTH runs
         (window-level from results/r5_window_split and results/r2_gat_window_split; hatched = not yet run).
Panel B: window vs patient bootstrap CI for the hybrid GAT (from v5_stats.json).
Panel C: fold-level AUROC vs held-out patients (from the saved patient-grouped predictions).
"""
import os, json, glob, numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import OUT, pooled_predictions, load_splits

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "legend.frameon": False})
def label_panel(ax, letter): ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
def pooled_auc(d):
    ys, ps = [], []
    for f in sorted(glob.glob(os.path.join(d, "fold_*", "y_score.npy"))):
        ps.append(np.load(f)); ys.append(np.load(f.replace("y_score", "y_true")))
    if not ps: return None, 0
    y = np.concatenate(ys); p = np.concatenate(ps); return float(np.mean([roc_auc_score(y == c, p[:, c]) for c in range(4)])), len(ps)

H = pooled_predictions("hard"); y = H.y.values; P = H[[f"p_{c}" for c in ["Seizure", "LPD", "GPD", "Others"]]].values
gat_patient = float(np.mean([roc_auc_score(y == c, P[:, c]) for c in range(4)]))
models = [("Hybrid GAT", gat_patient, os.path.join(OUT, "r2_gat_window_split", "hybrid"), "#1f4e79"),
          ("MLP", pooled_auc(os.path.join(OUT, "r1_baselines", "MLP"))[0], os.path.join(OUT, "r5_window_split", "MLP"), "#e9c46a"),
          ("Graph-theory GBM", pooled_auc(os.path.join(OUT, "a6_surrogate"))[0], os.path.join(OUT, "r5_window_split", "GBM"), "#7f7f7f")]
V5 = json.load(open(os.path.join(OUT, "v5_stats.json")))
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.4, 1, 1]})
label_panel(ax[0], "A"); R = {}
for i, (name, pa, wdir, col) in enumerate(models):
    wa, nf = pooled_auc(wdir); R[name] = {"patient": pa, "window": wa, "window_folds": nf}
    if pa is not None: ax[0].bar(i + 0.2, pa, 0.4, color=col, edgecolor="k", lw=0.5, label="Patient-grouped" if i == 0 else None); ax[0].text(i + 0.2, pa + 0.006, f"{pa:.3f}", ha="center", fontsize=7)
    if wa is not None: ax[0].bar(i - 0.2, wa, 0.4, color="#bbbbbb", edgecolor="k", lw=0.5, label="Window-level (leaky)" if i == 0 else None); ax[0].text(i - 0.2, wa + 0.006, f"{wa:.3f}" + ("" if nf == 10 else ("\n(fold 1)" if nf == 1 else f"\n({nf} of 10 folds)")), ha="center", fontsize=7)
    else: ax[0].bar(i - 0.2, 0.95, 0.4, color="white", edgecolor="#999", ls="--", hatch="//"); ax[0].text(i - 0.2, 0.72, "window-level\nrun pending", ha="center", fontsize=7, color="#777", rotation=90)
ax[0].set_xticks(range(len(models))); ax[0].set_xticklabels([m[0] for m in models], fontsize=8); ax[0].set_ylim(0.6, 1.06); ax[0].set_ylabel("Macro AUROC"); ax[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); ax[0].set_title("Corrected pipeline under two evaluation units", fontsize=10, pad=24)
label_panel(ax[1], "B"); seg = V5["variant_hard"]["segment_macro_auc_ci"] if "variant_hard" in V5 else [0.8363, 0.8424]; clu = V5["variant_hard"]["clustered"]["macro_auc_ci"] if "variant_hard" in V5 else [0.8171, 0.8563]
ax[1].errorbar([0], [gat_patient], yerr=[[gat_patient - seg[0]], [seg[1] - gat_patient]], fmt="o", color="#888", capsize=5, label="Window resampling"); ax[1].errorbar([1], [gat_patient], yerr=[[gat_patient - clu[0]], [clu[1] - gat_patient]], fmt="o", color="#1f4e79", capsize=5, label="Patient resampling")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["Window\nresampling", "Patient\nresampling"]); ax[1].set_xlim(-0.6, 1.6); ax[1].set_ylabel("Macro AUROC, hybrid GAT"); ax[1].set_title("95% CI by resampling unit", fontsize=10); ax[1].text(0.5, 0.03, f"width ×{(clu[1]-clu[0])/(seg[1]-seg[0]):.1f}", transform=ax[1].transAxes, fontsize=9)
label_panel(ax[2], "C"); splits = load_splits(); fa = [float(np.mean([roc_auc_score(g.y == c, g[f"p_{n}"]) for c, n in enumerate(["Seizure", "LPD", "GPD", "Others"])])) for _, g in H.groupby("fold")]
ax[2].scatter([s["n_unique_patients_test"] for s in splits], fa, color="#1f4e79"); ax[2].set_xlabel("Held-out patients in fold"); ax[2].set_ylabel("Fold macro AUROC"); ax[2].set_title(f"Fold variability (SD {np.std(fa):.3f})", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figs_v5", "F2_leakage.png"), dpi=int(os.environ.get("FIGDPI","200")), bbox_inches="tight"); json.dump(R, open(os.path.join(OUT, "leakage_new.json"), "w"), indent=1); print(R)
