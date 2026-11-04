"""v5: statistics and figures requested by the Reviewer-2 report.
 - patient-level PAIRED bootstrap for model comparisons (macro AUROC difference, per-class)
 - natural-prevalence re-weighting: accuracy, sensitivity, PPV, ECE, decision curves
 - MCE/ECE/Brier per model; class-wise reliability
 - figures F1-F8 in a classical journal style (serif, framed panels, A-D labels)
Outputs: results/v5_stats.json, results/figs_v5/F*.png
"""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy.interpolate import griddata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from common import *

FIG = os.path.join(OUT, "figs_v5"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.linewidth": 0.9, "axes.titlesize": 11, "legend.frameon": False})
COL = {"GAT": "#1f4e79", "GAT soft": "#4f81bd", "Surrogate": "#7f7f7f", "SVM": "#2a9d8f", "MLP": "#e9c46a", "CNN": "#e76f51"}
CLS = ["Seizure", "LPD", "GPD", "Others"]
NATIVE = {"Seizure": 20933, "LPD": 14856, "GPD": 16702, "Others": 16640 + 18861 + 18806}
PREV = np.array([NATIVE[c] for c in CLS]) / sum(NATIVE.values())
rng = np.random.default_rng(7); R = {}

def label_panel(ax, letter): ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)

# ------------------------------------------------------------------ load
H = pooled_predictions("hard"); S = pooled_predictions("soft"); G = pooled_predictions("gmm")
P = lambda df: df[[f"p_{c}" for c in CLS]].values
splits = load_splits()
sur = []
for k in range(10):
    d = os.path.join(OUT, "a6_surrogate", f"fold_{k+1}"); sur.append(np.load(os.path.join(d, "y_score.npy")))
SUR = np.concatenate(sur); assert len(SUR) == len(H)
y = H.y.values; pats = H.patient.values

def macro_auc(y, p): return np.mean([roc_auc_score(y == c, p[:, c]) for c in range(4)])
def class_aucs(y, p): return [roc_auc_score(y == c, p[:, c]) for c in range(4)]

# ------------------------------------------------------------------ 1. paired patient bootstrap
def paired_boot(pa, pb, n=1000):
    up = np.unique(pats); idx = {p: np.where(pats == p)[0] for p in up}; d = []; dc = []
    for _ in range(n):
        ii = np.concatenate([idx[p] for p in rng.choice(up, len(up), replace=True)])
        a = class_aucs(y[ii], pa[ii]); b = class_aucs(y[ii], pb[ii]); d.append(np.mean(a) - np.mean(b)); dc.append(np.array(a) - np.array(b))
    d = np.array(d); dc = np.array(dc)
    return {"diff": float(macro_auc(y, pa) - macro_auc(y, pb)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "per_class": {c: {"diff": float(np.mean(dc[:, i])), "ci": [float(np.percentile(dc[:, i], 2.5)), float(np.percentile(dc[:, i], 97.5))]} for i, c in enumerate(CLS)}}
R["paired_patient_bootstrap"] = {"GAT_vs_surrogate": paired_boot(P(H), SUR), "soft_vs_hard": paired_boot(P(S), P(H)), "gmm_vs_hard": paired_boot(P(G), P(H))}
print(json.dumps(R["paired_patient_bootstrap"], indent=1)[:800])

# ------------------------------------------------------------------ 2. natural prevalence re-weighting
def weighted_metrics(df, p):
    w = PREV[df.y.values] / 0.25; pred = p.argmax(1); yy = df.y.values
    cm = np.zeros((4, 4))
    for i in range(4):
        for j in range(4): cm[i, j] = w[(yy == i) & (pred == j)].sum()
    sens = np.diag(cm) / cm.sum(1); ppv = np.diag(cm) / np.maximum(cm.sum(0), 1e-9); acc = np.trace(cm) / cm.sum()
    conf = p.max(1); corr = (pred == yy); edges = np.linspace(0, 1, 16); e = 0; mce = 0
    for b in range(15):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum(): gap = abs(np.average(conf[m], weights=w[m]) - np.average(corr[m], weights=w[m])); e += w[m].sum() / w.sum() * gap; mce = max(mce, gap)
    return {"acc": float(acc), "sens": dict(zip(CLS, map(float, sens))), "ppv": dict(zip(CLS, map(float, ppv))), "ece": float(e), "mce": float(mce)}
def unweighted_cal(df, p):
    conf = p.max(1); corr = (p.argmax(1) == df.y.values); edges = np.linspace(0, 1, 16); e = 0; mce = 0; pts = []
    for b in range(15):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum(): gap = abs(conf[m].mean() - corr[m].mean()); e += m.mean() * gap; mce = max(mce, gap); pts.append((conf[m].mean(), corr[m].mean(), m.sum()))
    brier = float(((p - np.eye(4)[df.y.values]) ** 2).sum(1).mean()); return {"ece": float(e), "mce": float(mce), "brier": brier, "pts": pts}
R["balanced"] = {k: unweighted_cal(df, p) | {"acc": float((p.argmax(1) == df.y.values).mean()), "ppv": {c: float(((p.argmax(1) == i) & (df.y.values == i)).sum() / max((p.argmax(1) == i).sum(), 1)) for i, c in enumerate(CLS)}} for k, (df, p) in {"hard": (H, P(H)), "soft": (S, P(S)), "surrogate": (H, SUR)}.items()}
R["natural_prevalence"] = {k: weighted_metrics(df, p) for k, (df, p) in {"hard": (H, P(H)), "soft": (S, P(S)), "surrogate": (H, SUR)}.items()}
for k in R["balanced"]: R["balanced"][k].pop("pts")
print("prevalence", json.dumps(R["natural_prevalence"]["hard"], indent=0)[:400])

# decision curves (one-vs-rest) under natural prevalence
def net_benefit(yb, s, w, pts):
    N = w.sum(); out = []
    for t in pts:
        pred = s >= t; tp = w[pred & yb].sum() / N; fp = w[pred & ~yb].sum() / N; out.append(tp - fp * t / (1 - t))
    return np.array(out)
TH = np.linspace(0.02, 0.9, 60); R["dca"] = {}
for i, c in enumerate(CLS):
    w = PREV[y] / 0.25; yb = y == i; prev = w[yb].sum() / w.sum()
    R["dca"][c] = {"prevalence": float(prev), "hard": net_benefit(yb, P(H)[:, i], w, TH).tolist(), "soft": net_benefit(yb, P(S)[:, i], w, TH).tolist(),
                   "surrogate": net_benefit(yb, SUR[:, i], w, TH).tolist(), "treat_all": (prev - (1 - prev) * TH / (1 - TH)).tolist()}

# operating points at natural prevalence: threshold for 90% sensitivity -> specificity, PPV
R["operating_points"] = {}
for i, c in enumerate(CLS):
    w = PREV[y] / 0.25; yb = y == i; s = P(S)[:, i]
    fpr, tpr, thr = roc_curve(yb, s, sample_weight=w); j = np.argmax(tpr >= 0.90); t = thr[j]
    pred = s >= t; ppv = w[pred & yb].sum() / w[pred].sum(); R["operating_points"][c] = {"threshold": float(t), "sensitivity": float(tpr[j]), "specificity": float(1 - fpr[j]), "ppv_at_prevalence": float(ppv), "alerts_per_100_windows": float(100 * w[pred].sum() / w.sum())}

# ------------------------------------------------------------------ 3. dataset descriptors for F1
meta = load_meta(); files = H.file.values
V = H[[f"v_{c}" for c in CLS]].values; R["vote_dist_by_class"] = {c: V[y == i].mean(0).round(3).tolist() for i, c in enumerate(CLS)}
R["fold_patients"] = [s["n_unique_patients_test"] for s in splits]; R["fold_windows"] = [len(s["test"]["files"]) for s in splits]
ensure_fc(["corr"], files=list(files[:1]))
sample = {c: rng.choice(np.where(y == i)[0], 300, replace=False) for i, c in enumerate(CLS)}
ensure_fc(["corr", "coh", "aec", "plv"], files=[files[j] for s in sample.values() for j in s])
meanFC = {c: np.mean([np.abs(np.load(os.path.join(ROOT, "fc_sb", "coh", files[j]))) for j in sample[c]], 0) for c in CLS}
psd = ensure_psd(); meanPSD = {c: np.mean([np.load(os.path.join(psd, files[j])) for j in sample[c]], 0) for c in CLS}

json.dump(R, open(os.path.join(OUT, "v5_stats.json"), "w"), indent=1)

# ================================================================== FIGURES
XY = {'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95), 'F7': (-0.81, 0.59), 'F3': (-0.41, 0.52), 'Fz': (0, 0.55), 'F4': (0.41, 0.52), 'F8': (0.81, 0.59),
      'T3': (-1, 0), 'C3': (-0.5, 0), 'Cz': (0, 0), 'C4': (0.5, 0), 'T4': (1, 0), 'T5': (-0.81, -0.59), 'P3': (-0.41, -0.52), 'Pz': (0, -0.55),
      'P4': (0.41, -0.52), 'T6': (0.81, -0.59), 'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)}
def topomap(ax, vals, title, cmap="magma", vmin=None, vmax=None):
    xs = np.array([XY[c][0] for c in CH]); ys = np.array([XY[c][1] for c in CH])
    gx, gy = np.meshgrid(np.linspace(-1.15, 1.15, 160), np.linspace(-1.15, 1.15, 160))
    z = griddata((xs, ys), vals, (gx, gy), method="cubic"); z2 = griddata((xs, ys), vals, (gx, gy), method="nearest"); z = np.where(np.isnan(z), z2, z)
    z[(gx ** 2 + gy ** 2) > 1.15 ** 2] = np.nan
    im = ax.imshow(z, extent=[-1.15, 1.15, -1.15, 1.15], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_patch(Circle((0, 0), 1.15, fill=False, lw=1.2, color="k")); ax.add_patch(Polygon([[-0.12, 1.13], [0, 1.3], [0.12, 1.13]], closed=True, fill=False, lw=1.2, color="k"))
    ax.scatter(xs, ys, s=8, c="k"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.3, 1.4); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10); return im

# ---- F1 study design: dataset, votes, folds, mean representation
fig = plt.figure(figsize=(11, 7.2)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
ax = fig.add_subplot(gs[0, 0:2]); label_panel(ax, "A")
names = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]; nat = [20933, 14856, 16702, 16640, 18861, 18806]; used = [9000, 9000, 9000, 3000, 3000, 3000]
x = np.arange(6); ax.bar(x - 0.2, nat, 0.4, color="#bbbbbb", label="Native (106,798 windows)"); ax.bar(x + 0.2, used, 0.4, color=COL["GAT"], label="Analysed (36,000 windows)")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Labelled 50-s windows"); ax.legend(fontsize=8); ax.set_title("Dataset composition and class balancing")
ax = fig.add_subplot(gs[0, 2]); label_panel(ax, "B")
bottom = np.zeros(4); colors = ["#1f4e79", "#2a9d8f", "#e9c46a", "#e76f51"]
for i, c in enumerate(CLS):
    vals = [R["vote_dist_by_class"][k][i] for k in CLS]; ax.bar(range(4), vals, bottom=bottom, color=colors[i], label=c); bottom += vals
ax.set_xticks(range(4)); ax.set_xticklabels(CLS, fontsize=8); ax.set_ylabel("Mean annotator vote share"); ax.set_title("Votes by modal label"); ax.legend(fontsize=7, loc="lower right")
ax = fig.add_subplot(gs[0, 3]); label_panel(ax, "C")
ax.bar(range(1, 11), R["fold_patients"], color=COL["GAT"]); ax.set_xlabel("Test fold"); ax.set_ylabel("Held-out patients"); ax.set_title("Patients per test fold")
for i, (c, M) in enumerate(meanFC.items()):
    ax = fig.add_subplot(gs[1, i]);
    if i == 0: label_panel(ax, "D")
    im = ax.imshow(M, cmap="viridis", vmin=0.1, vmax=0.6); ax.set_title(f"Mean coherence, {c}", fontsize=9); ax.set_xticks(range(0, 19, 3)); ax.set_xticklabels([CH[j] for j in range(0, 19, 3)], fontsize=6, rotation=90); ax.set_yticks(range(0, 19, 3)); ax.set_yticklabels([CH[j] for j in range(0, 19, 3)], fontsize=6)
fig.colorbar(im, ax=fig.axes[-4:], shrink=0.6, label="|coherence|")
plt.savefig(os.path.join(FIG, "F1_design.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- F2 leakage
old = {"GAT": 0.9732, "SVM": 0.9333, "MLP": 0.8993, "CNN": 0.9186}; new = {"GAT": 0.8395, "SVM": 0.7891, "MLP": 0.7738, "CNN": 0.7675}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.4, 1, 1]})
label_panel(ax[0], "A"); order = ["GAT", "SVM", "MLP", "CNN"]; x = np.arange(4)
ax[0].bar(x - 0.2, [old[m] for m in order], 0.4, color="#bbbbbb", label="Segment-level split"); ax[0].bar(x + 0.2, [new[m] for m in order], 0.4, color=[COL[m] for m in order], label="Patient-grouped split")
for i, m in enumerate(order): ax[0].text(i + 0.2, new[m] + 0.008, f"{new[m]:.3f}", ha="center", fontsize=7); ax[0].text(i - 0.2, old[m] + 0.008, f"{old[m]:.3f}", ha="center", fontsize=7, color="#555")
ax[0].set_xticks(x); ax[0].set_xticklabels(order); ax[0].set_ylim(0.5, 1.02); ax[0].set_ylabel("Macro-average AUROC"); ax[0].legend(fontsize=8, loc="lower left"); ax[0].set_title("Same pipeline, two evaluation units")
label_panel(ax[1], "B"); seg = [0.8363, 0.8424]; clu = [0.8171, 0.8563]; pt = 0.8395
ax[1].errorbar([0], [pt], yerr=[[pt - seg[0]], [seg[1] - pt]], fmt="o", color="#888", capsize=5, label="Window bootstrap"); ax[1].errorbar([1], [pt], yerr=[[pt - clu[0]], [clu[1] - pt]], fmt="o", color=COL["GAT"], capsize=5, label="Patient-clustered bootstrap")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["Window\nresampling", "Patient\nresampling"]); ax[1].set_xlim(-0.6, 1.6); ax[1].set_ylabel("Macro AUROC, GAT hybrid"); ax[1].set_title("95% CI by resampling unit"); ax[1].text(0.5, 0.03, "width ×6.4", transform=ax[1].transAxes, fontsize=9)
label_panel(ax[2], "C"); fa = [0.8459, 0.8480, 0.8539, 0.8192, 0.8256, 0.8468, 0.8564, 0.8613, 0.7973, 0.7925]
ax[2].scatter(R["fold_patients"], fa, color=COL["GAT"]); ax[2].set_xlabel("Held-out patients in fold"); ax[2].set_ylabel("Fold macro AUROC"); ax[2].set_title("Fold variability (SD 0.024)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F2_leakage.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- F3 representation / graph contribution (with pending arms)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.3, 1, 1]})
label_panel(ax[0], "A")
arms = [("Spectra only\n(inference)", 0.7451, 0.0250, "#bbbbbb"), ("Connectivity only\n(inference)", 0.7469, 0.0324, "#bbbbbb"), ("Graph-theory\nfeatures + GBM", 0.7667, 0.0332, COL["Surrogate"]), ("Hybrid GAT", 0.8347, 0.0235, COL["GAT"]), ("GAT only\n(pending S1)", np.nan, 0, "white"), ("Edge-shuffled\n(pending S1)", np.nan, 0, "white")]
for i, (n, m, s, c) in enumerate(arms):
    if np.isnan(m): ax[0].bar(i, 0.9, color="white", edgecolor="#999", ls="--", hatch="//"); ax[0].text(i, 0.72, "pending", ha="center", fontsize=8, color="#777")
    else: ax[0].bar(i, m, yerr=s, color=c, capsize=4, edgecolor="k", lw=0.5)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels([a[0] for a in arms], fontsize=7); ax[0].set_ylim(0.65, 0.9); ax[0].set_ylabel("Macro AUROC (mean ± SD, 10 folds)"); ax[0].set_title("Representation arms on identical patient folds")
label_panel(ax[1], "B"); pb = R["paired_patient_bootstrap"]
for i, (k, lab) in enumerate([("GAT_vs_surrogate", "Hybrid GAT − graph-theory GBM"), ("soft_vs_hard", "Soft − hard labels"), ("gmm_vs_hard", "GMM refinement − none")]):
    d = pb[k]; ax[1].errorbar(d["diff"], i, xerr=[[d["diff"] - d["ci"][0]], [d["ci"][1] - d["diff"]]], fmt="o", color=COL["GAT"], capsize=4)
ax[1].axvline(0, color="k", lw=0.8); ax[1].set_yticks(range(3)); ax[1].set_yticklabels(["GAT − GBM", "soft − hard", "GMM − none"], fontsize=8); ax[1].set_xlabel("Δ macro AUROC, patient-paired bootstrap 95% CI"); ax[1].set_title("Paired comparisons (patient unit)")
label_panel(ax[2], "C"); loo = [("Full", 0.8347), ("− corr", 0.8192), ("− coh", 0.8198), ("− PLV", 0.8217), ("− AEC", 0.8266)]
ax[2].bar(range(5), [l[1] for l in loo], color=[COL["GAT"]] + ["#c0504d"] * 4, edgecolor="k", lw=0.5)
for i, l in enumerate(loo[1:], 1): ax[2].text(i, l[1] + 0.002, f"{l[1] - 0.8347:+.3f}", ha="center", fontsize=7)
ax[2].set_xticks(range(5)); ax[2].set_xticklabels([l[0] for l in loo]); ax[2].set_ylim(0.78, 0.85); ax[2].set_ylabel("Macro AUROC"); ax[2].set_title("Leave-one-measure-out (retrained)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F3_graph_contribution.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- F4 attention audit with topomaps
att = np.load(os.path.join(ROOT, "mod", "analysis_patient_grouped", "attention_rollout_class_matrices.npz"))
fig = plt.figure(figsize=(11, 6.3)); gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1])
for i, (c, k) in enumerate(zip(CLS, ["SEIZURE", "LPD", "GPD", "OTHERS"])):
    ax = fig.add_subplot(gs[0, i]);
    if i == 0: label_panel(ax, "A")
    M = att[k]; M = M.mean(0) if M.ndim > 2 else M; v = M.mean(0); v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    im = topomap(ax, v, f"{c}: attention received", vmin=0, vmax=1)
fig.colorbar(im, ax=fig.axes[:4], shrink=0.5, label="Rollout attention (max-normalized)")
ax = fig.add_subplot(gs[1, 0:2]); label_panel(ax, "B")
drops = {"Seizure": (0.0002, -0.0004, 0.0014, 0.258), "LPD": (0.0008, 0.0010, 0.0018, 0.839), "GPD": (0.0002, 0.0009, 0.0014, 0.032), "Others": (-0.0004, -0.0007, 0.0009, 1.0)}; x = np.arange(4)
ax.bar(x - 0.2, [drops[c][0] for c in CLS], 0.4, color=COL["GAT"], label="Ablate top-10 attention edges", edgecolor="k", lw=0.5)
ax.bar(x + 0.2, [drops[c][1] for c in CLS], 0.4, yerr=[drops[c][2] for c in CLS], color="#bbbbbb", capsize=3, label="Ablate 10 random edges (30 draws)", edgecolor="k", lw=0.5)
for i, c in enumerate(CLS): ax.text(i, 0.0031, f"P = {drops[c][3]:.2f}", ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(CLS); ax.set_ylabel("Accuracy drop"); ax.set_ylim(-0.0025, 0.0038); ax.legend(fontsize=8, loc="lower left"); ax.set_title("Causal edge ablation vs random-edge null (fold 1)")
ax = fig.add_subplot(gs[1, 2:4]); label_panel(ax, "C")
seed = {"Seizure": (0.442, 0.912), "LPD": (0.378, 0.550), "GPD": (0.660, 0.875), "Others": (0.294, 0.352)}
ax.bar(x - 0.2, [seed[c][0] for c in CLS], 0.4, color=COL["GAT"], label="Top-10 Jaccard across 5 seeds", edgecolor="k", lw=0.5); ax.bar(x + 0.2, [seed[c][1] for c in CLS], 0.4, color="#c0504d", label="r(real-label map, permuted-label map)", edgecolor="k", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(CLS); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_title("Reproducibility and label dependence (fold 1)")
plt.savefig(os.path.join(FIG, "F4_attention_audit.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- F5 static representation: measure contribution (GAT LOO vs surrogate drop), asymmetry negative, pending temporal
a5 = json.load(open(os.path.join(OUT, "a5_causal.json"))); a4 = json.load(open(os.path.join(OUT, "a4_asymmetry.json")))
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
label_panel(ax[0], "A"); meas = ["corr", "coh", "aec", "plv"]; gat = [0.0194, 0.0170, 0.0086, 0.0138]; surd = [np.mean(list(a5["measure_drop"][m].values())) for m in meas]
ax[0].bar(np.arange(4) - 0.2, gat, 0.4, color=COL["GAT"], label="GAT, retrained leave-one-out", edgecolor="k", lw=0.5); ax[0].bar(np.arange(4) + 0.2, surd, 0.4, color=COL["Surrogate"], label="GBM, measure zeroed at inference", edgecolor="k", lw=0.5)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["Correlation", "Coherence", "AEC", "PLV"]); ax[0].set_ylabel("Macro AUROC loss"); ax[0].legend(fontsize=8); ax[0].set_title("Measure importance is model-dependent")
label_panel(ax[1], "B"); F = pd.read_csv(os.path.join(OUT, "a4_asymmetry_features.csv")); sub = H["sub"].values
for i, m in enumerate(meas):
    for j, (lab, col) in enumerate([("LPD", COL["GAT"]), ("GPD", "#c0504d")]):
        ax[1].boxplot(F.loc[sub == lab, f"{m}_absAI"], positions=[i * 3 + j], widths=0.8, patch_artist=True, boxprops=dict(facecolor=col, alpha=0.6), medianprops=dict(color="k"), showfliers=False)
ax[1].set_xticks([i * 3 + 0.5 for i in range(4)]); ax[1].set_xticklabels(["Corr", "Coh", "AEC", "PLV"]); ax[1].set_ylabel("|Hemispheric asymmetry index|"); ax[1].set_title("LPD (blue) vs GPD (red): AUROC 0.43–0.52")
label_panel(ax[2], "C"); ax[2].axis("off"); ax[2].text(0.5, 0.5, "Dynamic connectivity /\nraw-waveform arms\n\npending (A8, S2)\n\nstatic vs. time-resolved\nrepresentation under a\nfixed architecture", ha="center", va="center", fontsize=10, color="#777", bbox=dict(boxstyle="round", fc="white", ec="#999", ls="--"))
ax[2].set_title("Time-axis comparison")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F5_representation.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- F6 expert uncertainty and calibration
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
a = ax[0, 0]; label_panel(a, "A")
for k, df, p, lab, col in [("hard", H, P(H), "Hard label", COL["GAT"]), ("soft", S, P(S), "Soft label", COL["GAT soft"]), ("sur", H, SUR, "Graph-theory GBM", COL["Surrogate"])]:
    cal = unweighted_cal(df, p); pts = np.array(cal["pts"]); a.plot(pts[:, 0], pts[:, 1], "o-" if k != "sur" else "s--", color=col, ms=4, label=f"{lab} (ECE {cal['ece']:.3f}, MCE {cal['mce']:.3f}, Brier {cal['brier']:.3f})")
a.plot([0.25, 1], [0.25, 1], "k--", lw=0.8, label="Ideal"); a.set_xlabel("Mean predicted probability (top class)"); a.set_ylabel("Fraction correct"); a.legend(fontsize=7, loc="upper left"); a.set_title("Reliability (balanced test set)")
a = ax[0, 1]; label_panel(a, "B"); strata = [(0, .5, "<0.5"), (.5, .75, "0.5–0.75"), (.75, .999, "0.75–1"), (.999, 1.01, "=1")]
for df, p, lab, col, mk in [(H, P(H), "Hard label", COL["GAT"], "o"), (S, P(S), "Soft label", COL["GAT soft"], "s"), (H, SUR, "Graph-theory GBM", COL["Surrogate"], "^")]:
    vals = [macro_auc(df.y.values[m], p[m]) for lo, hi, _ in strata for m in [(df.cons.values >= lo) & (df.cons.values < hi)]]; a.plot(range(4), vals, marker=mk, color=col, label=lab)
a.set_xticks(range(4)); a.set_xticklabels([s[2] for s in strata]); a.set_xlabel("Fraction of annotators agreeing with modal label"); a.set_ylabel("Macro AUROC"); a.legend(fontsize=8); a.set_title("Discrimination vs expert agreement")
a = ax[1, 0]; label_panel(a, "C"); ptrue = P(S)[np.arange(len(y)), y]
data = [ptrue[(H.cons.values >= lo) & (H.cons.values < hi)] for lo, hi, _ in strata]
bp = a.boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color="k"));
for patch, col in zip(bp["boxes"], ["#e76f51", "#e9c46a", "#4f81bd", "#1f4e79"]): patch.set_facecolor(col); patch.set_alpha(0.7)
a.set_xticklabels([s[2] for s in strata]); a.set_xlabel("Annotator agreement"); a.set_ylabel("Model probability of the modal label"); a.set_title("Soft-label model, probability of expert label")
a = ax[1, 1]; label_panel(a, "D"); a2 = json.load(open(os.path.join(OUT, "a2_expert_agreement.json")))["soft"]
x = np.arange(4); a.bar(x - 0.2, [a2["per_class"][c]["single_expert_PA"] for c in CLS], 0.4, color="#bbbbbb", label="Simulated single annotator (upper bound)", edgecolor="k", lw=0.5); a.bar(x + 0.2, [a2["per_class"][c]["model_recall"] for c in CLS], 0.4, color=COL["GAT soft"], label="Soft-label model", edgecolor="k", lw=0.5)
a.set_xticks(x); a.set_xticklabels(CLS); a.set_ylabel("Agreement with modal label (recall)"); a.set_ylim(0, 1.05); a.legend(fontsize=8, loc="lower right"); a.set_title("Model vs one reader, per class")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F6_uncertainty_calibration.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- F7 natural prevalence: decision curves + operating points
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
for i, c in enumerate(CLS):
    a = ax.flat[i]; label_panel(a, "ABCD"[i]); d = R["dca"][c]
    a.plot(TH, d["treat_all"], color="k", label="Treat all"); a.axhline(0, color="#888", ls="--", label="Treat none")
    a.plot(TH, d["hard"], color=COL["GAT"], label="GAT hard"); a.plot(TH, d["soft"], color=COL["GAT soft"], label="GAT soft"); a.plot(TH, d["surrogate"], color=COL["Surrogate"], ls="--", label="Graph-theory GBM")
    a.set_ylim(-0.05, max(d["prevalence"] * 1.15, 0.1)); a.set_xlabel("Threshold probability"); a.set_ylabel("Net benefit"); a.set_title(f"{c} vs rest at native prevalence ({d['prevalence']:.2f})");
    if i == 0: a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F7_prevalence_decision.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---- F8 learning curve + patient-level
a7 = json.load(open(os.path.join(OUT, "a7_learning_curve.json")))["curve"]; a3 = json.load(open(os.path.join(OUT, "a3_patient_failure.json")))
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
a = ax[0]; label_panel(a, "A"); fr = [0.1, 0.25, 0.5, 1.0]
for key, lab, col in [("all", "All test windows", COL["GAT"]), ("unanimous", "Unanimous labels", "#2a9d8f"), ("low_agreement", "Agreement < 0.75", "#c0504d")]:
    m = np.array([a7[str(f)][key][0] for f in fr]); s = np.array([a7[str(f)][key][1] for f in fr]); a.plot(fr, m, "o-", color=col, label=lab); a.fill_between(fr, m - s, m + s, color=col, alpha=0.18)
a.set_xscale("log"); a.set_xticks(fr); a.set_xticklabels(["10%", "25%", "50%", "100%"]); a.set_xlabel("Training patients used"); a.set_ylabel("Macro AUROC (graph-theory GBM)"); a.legend(fontsize=8); a.set_title("Learning curve by label agreement")
a = ax[1]; label_panel(a, "B"); g = H.assign(err=(H.pred != H.y)).groupby("patient"); pa = pd.DataFrame({"acc": 1 - g.err.mean(), "n": g.size(), "cons": g.cons.mean()})
a.hist(pa.acc, bins=20, color=COL["GAT"], edgecolor="k", lw=0.4); a.set_xlabel("Per-patient accuracy (hard-label GAT)"); a.set_ylabel("Patients"); a.set_title("625 all correct, 249 all wrong")
a = ax[2]; label_panel(a, "C"); sc = a.scatter(pa.cons, pa.acc, s=np.sqrt(pa.n) * 2.5, c=np.log(pa.n), cmap="viridis", alpha=0.5, edgecolors="none")
a.set_xlabel("Mean annotator agreement (patient)"); a.set_ylabel("Per-patient accuracy"); a.set_title("Agreement predicts accuracy (β = +0.36)"); fig.colorbar(sc, ax=a, label="log windows per patient")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F8_learning_patient.png"), dpi=200, bbox_inches="tight"); plt.close()
print("figures done")
