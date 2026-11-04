"""R4. Patient-paired comparison of every model/condition for which per-window predictions exist.

Collects pooled out-of-fold predictions from:
  patient_split/result_4class_patient_{hybrid,softlabel_hybrid,gmmgumbel_hybrid}_auc   (GAT variants, GPU, 10 folds)
  results/a6_surrogate                                                                  (graph-theory GBM, 10 folds)
  results/r1_baselines/{SVM,MLP}                                                        (recovered baselines, 10 folds)
  results/r2_gat_controls/<cond>                                                        (CPU controls, whichever folds exist)
For controls run on a subset of folds, every comparison is restricted to the same folds.
Outputs: results/r4_comparison.json, results/figs_v5/F3_graph_contribution.png (replaces the pending-bar version)
"""
import os, json, glob, numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT, OUT, CLASSES, load_splits, pooled_predictions

splits = load_splits(); rng = np.random.default_rng(11)
H = pooled_predictions("hard"); y_all = H.y.values; pats_all = H.patient.values; fold_all = H.fold.values
def load_dir(d):
    out = {}
    for k in range(1, 11):
        f = os.path.join(d, f"fold_{k}", "y_score.npy")
        if os.path.exists(f): out[k] = np.load(f)
    return out
MODELS = {"GAT hybrid": {k: g.values for k, g in H.groupby("fold")[[f"p_{c}" for c in CLASSES]]},
          "GAT soft": {k: g.values for k, g in pooled_predictions("soft").groupby("fold")[[f"p_{c}" for c in CLASSES]]},
          "Graph-theory GBM": load_dir(os.path.join(OUT, "a6_surrogate")),
          "SVM": load_dir(os.path.join(OUT, "r1_baselines", "SVM")), "MLP": load_dir(os.path.join(OUT, "r1_baselines", "MLP"))}
for cond in ["hybrid", "graph_only", "edge_shuffle", "edge_const", "flat_only"]:
    d = load_dir(os.path.join(OUT, "r2_gat_controls", cond))
    if d: MODELS[f"CPU {cond}"] = d
MODELS = {k: v for k, v in MODELS.items() if v}
print({k: sorted(v) for k, v in MODELS.items()})

def pooled(m, folds):
    return np.concatenate([MODELS[m][k] for k in folds]), np.concatenate([y_all[fold_all == k] for k in folds]), np.concatenate([pats_all[fold_all == k] for k in folds])
def macro(y, p): return float(np.mean([roc_auc_score(y == c, p[:, c]) for c in range(4)]))
def paired(ma, mb, n=1000):
    folds = sorted(set(MODELS[ma]) & set(MODELS[mb])); pa, y, pats = pooled(ma, folds); pb, _, _ = pooled(mb, folds)
    up = np.unique(pats); idx = {p: np.where(pats == p)[0] for p in up}; d = []
    for _ in range(n):
        ii = np.concatenate([idx[p] for p in rng.choice(up, len(up), replace=True)]); d.append(macro(y[ii], pa[ii]) - macro(y[ii], pb[ii]))
    d = np.array(d); return {"folds": folds, "n_patients": int(len(up)), "auc_a": macro(y, pa), "auc_b": macro(y, pb), "diff": macro(y, pa) - macro(y, pb),
                             "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}
R = {"models": {}, "pairs": {}}
for m in MODELS:
    folds = sorted(MODELS[m]); p, y, pats = pooled(m, folds)
    up = np.unique(pats); idx = {q: np.where(pats == q)[0] for q in up}; b = []
    for _ in range(500):
        ii = np.concatenate([idx[q] for q in rng.choice(up, len(up), replace=True)]); b.append(macro(y[ii], p[ii]))
    R["models"][m] = {"folds": folds, "macro_auc": macro(y, p), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "acc": float((p.argmax(1) == y).mean()),
                      "per_class": {c: float(roc_auc_score(y == i, p[:, i])) for i, c in enumerate(CLASSES)}}
    print(m, R["models"][m]["macro_auc"], R["models"][m]["ci"], folds)
pairs = [("GAT hybrid", "SVM"), ("GAT hybrid", "MLP"), ("GAT hybrid", "Graph-theory GBM"), ("SVM", "MLP"), ("SVM", "Graph-theory GBM")]
pairs += [(f"CPU hybrid", f"CPU {c}") for c in ["graph_only", "edge_shuffle", "edge_const", "flat_only"] if f"CPU {c}" in MODELS and "CPU hybrid" in MODELS]
pairs += [("GAT hybrid", "CPU hybrid")] if "CPU hybrid" in MODELS else []
for a, b in pairs:
    if a in MODELS and b in MODELS: R["pairs"][f"{a} vs {b}"] = paired(a, b); print(a, "vs", b, R["pairs"][f"{a} vs {b}"])
json.dump(R, open(os.path.join(OUT, "r4_comparison.json"), "w"), indent=1)

# ---- Figure 3 (method-introduction version): all arms with patient CIs
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "legend.frameon": False})
order = [m for m in ["GAT hybrid", "GAT soft", "SVM", "MLP", "Graph-theory GBM", "CPU hybrid", "CPU graph_only", "CPU flat_only", "CPU edge_shuffle", "CPU edge_const"] if m in R["models"]]
labels = {"GAT hybrid": "Hybrid GAT\n(10 folds)", "GAT soft": "Hybrid GAT,\nsoft labels", "SVM": "SVM (RBF)", "MLP": "MLP", "Graph-theory GBM": "Graph-theory\nfeatures + GBM",
          "CPU hybrid": "Hybrid\n(control run)", "CPU graph_only": "Graph branch\nonly", "CPU flat_only": "Flat branch\nonly", "CPU edge_shuffle": "Edge-shuffled\ngraph", "CPU edge_const": "Constant\nedges"}
cols = {"GAT hybrid": "#1f4e79", "GAT soft": "#4f81bd", "SVM": "#2a9d8f", "MLP": "#e9c46a", "Graph-theory GBM": "#7f7f7f", "CPU hybrid": "#1f4e79", "CPU graph_only": "#5b8db8", "CPU flat_only": "#b8860b", "CPU edge_shuffle": "#c0504d", "CPU edge_const": "#e07b6a"}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
ax[0].annotate("A", xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
for i, m in enumerate(order):
    r = R["models"][m]; ax[0].bar(i, r["macro_auc"], color=cols[m], edgecolor="k", lw=0.5); ax[0].errorbar(i, r["macro_auc"], yerr=[[r["macro_auc"] - r["ci"][0]], [r["ci"][1] - r["macro_auc"]]], color="k", capsize=3, lw=1)
    ax[0].text(i, r["ci"][1] + 0.006, f"{r['macro_auc']:.3f}", ha="center", fontsize=7)
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([labels[m] for m in order], fontsize=7.5); ax[0].set_ylim(0.6, 0.92); ax[0].set_ylabel("Macro AUROC (patient-clustered 95% CI)")
ax[0].set_title("Models and topology controls on identical patient-grouped folds", fontsize=10)
if any(m.startswith("CPU") for m in order): ax[0].axvline(4.5, color="#999", ls="--", lw=0.8); ax[0].text(4.6, 0.905, f"controls: folds {R['models']['CPU hybrid']['folds']}", fontsize=7, color="#666")
ax[1].annotate("B", xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
pk = list(R["pairs"])
for i, k in enumerate(pk):
    d = R["pairs"][k]; ax[1].errorbar(d["diff"], i, xerr=[[d["diff"] - d["ci"][0]], [d["ci"][1] - d["diff"]]], fmt="o", color="#1f4e79", capsize=3)
ax[1].axvline(0, color="k", lw=0.8); ax[1].set_yticks(range(len(pk))); ax[1].set_yticklabels([k.replace("CPU ", "").replace("Graph-theory GBM", "GBM") for k in pk], fontsize=7.5); ax[1].invert_yaxis()
ax[1].set_xlabel("Δ macro AUROC (first − second), patient-paired 95% CI"); ax[1].set_title("Paired differences", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figs_v5", "F3b_comparators.png"), dpi=200, bbox_inches="tight"); print("figure saved")
