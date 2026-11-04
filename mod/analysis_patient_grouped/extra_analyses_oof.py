"""Extra analyses on saved out-of-fold predictions (no retraining).
Outputs: extra_results.json + figures in ./figs
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\eeg_seizure"
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figs"); os.makedirs(FIG, exist_ok=True)
CLASSES = ["Seizure", "LPD", "GPD", "Others"]
VARIANTS = {"hard": "result_4class_patient_hybrid_auc",
            "gmm": "result_4class_patient_gmmgumbel_hybrid_auc",
            "soft": "result_4class_patient_softlabel_hybrid_auc"}
rng = np.random.default_rng(42)
R = {}

splits = json.load(open(os.path.join(ROOT, "patient_split", "fold_splits_patient.json")))
train = pd.read_csv(os.path.join(ROOT, "train.csv"))
train["file"] = train.eeg_id.astype(str) + "_" + train.eeg_sub_id.astype(str) + ".npy"
votes = train[[c for c in train.columns if c.endswith("_vote")]].values
train["n_votes"] = votes.sum(1)
train["consensus_frac"] = votes.max(1) / np.maximum(votes.sum(1), 1)
meta = train.set_index("file")[["eeg_id", "patient_id", "expert_consensus", "consensus_frac", "n_votes"]]

def pooled(variant):
    rows = []
    for k, f in enumerate(splits):
        d = os.path.join(ROOT, "patient_split", VARIANTS[variant], f"fold_{k+1}")
        yt = np.load(os.path.join(d, "y_true.npy")); yp = np.load(os.path.join(d, "y_pred.npy")); ys = np.load(os.path.join(d, "y_score.npy"))
        files = f["test"]["files"]
        assert (yt == np.array(f["test"]["labels_int"])).all()
        m = meta.loc[files]
        df = pd.DataFrame({"file": files, "fold": k+1, "y": yt, "pred": yp,
                           "eeg_id": m.eeg_id.values, "patient": m.patient_id.values,
                           "sub": m.expert_consensus.values, "cons": m.consensus_frac.values, "nv": m.n_votes.values})
        for c in range(4): df[f"p{c}"] = ys[:, c]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

P = {v: pooled(v) for v in VARIANTS}
R["n_segments"] = int(len(P["hard"])); R["n_patients"] = int(P["hard"].patient.nunique()); R["n_recordings"] = int(P["hard"].eeg_id.nunique())

def metrics(df):
    ys = df[[f"p{c}" for c in range(4)]].values; y = df.y.values; pred = df.pred.values
    out = {"acc": float((y == pred).mean()), "macro_f1": float(f1_score(y, pred, average="macro"))}
    aucs = [roc_auc_score(y == c, ys[:, c]) for c in range(4)]
    aps = [average_precision_score(y == c, ys[:, c]) for c in range(4)]
    out["auc"] = dict(zip(CLASSES, map(float, aucs))); out["macro_auc"] = float(np.mean(aucs))
    out["auprc"] = dict(zip(CLASSES, map(float, aps))); out["macro_auprc"] = float(np.mean(aps))
    return out

def ece(df, bins=15):
    ys = df[[f"p{c}" for c in range(4)]].values; conf = ys.max(1); corr = (ys.argmax(1) == df.y.values)
    edges = np.linspace(0, 1, bins + 1); e = 0; curve = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i+1])
        if m.sum():
            e += m.mean() * abs(conf[m].mean() - corr[m].mean()); curve.append((float(conf[m].mean()), float(corr[m].mean()), int(m.sum())))
    onehot = np.eye(4)[df.y.values]; brier = float(((ys - onehot) ** 2).sum(1).mean())
    return float(e), brier, curve

# ---- 1. pooled metrics, calibration, clustered bootstrap per variant
def clustered_boot(df, n=1000):
    pats = df.patient.values; up = np.unique(pats); idx = {p: np.where(pats == p)[0] for p in up}
    accs, maucs, pc = [], [], []
    for _ in range(n):
        samp = rng.choice(up, len(up), replace=True)
        ii = np.concatenate([idx[p] for p in samp]); d = df.iloc[ii]
        ys = d[[f"p{c}" for c in range(4)]].values; y = d.y.values
        a = [roc_auc_score(y == c, ys[:, c]) for c in range(4)]
        accs.append((y == d.pred.values).mean()); maucs.append(np.mean(a)); pc.append(a)
    pc = np.array(pc)
    return {"acc_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "macro_auc_ci": [float(np.percentile(maucs, 2.5)), float(np.percentile(maucs, 97.5))],
            "class_auc_ci": {CLASSES[c]: [float(np.percentile(pc[:, c], 2.5)), float(np.percentile(pc[:, c], 97.5))] for c in range(4)}}

def segment_boot(df, n=1000):
    maucs = []
    for _ in range(n):
        d = df.sample(len(df), replace=True, random_state=int(rng.integers(1e9)))
        ys = d[[f"p{c}" for c in range(4)]].values; y = d.y.values
        maucs.append(np.mean([roc_auc_score(y == c, ys[:, c]) for c in range(4)]))
    return [float(np.percentile(maucs, 2.5)), float(np.percentile(maucs, 97.5))]

for v, df in P.items():
    m = metrics(df); e, b, curve = ece(df)
    m.update({"ece": e, "brier": b, "reliability": curve, "clustered": clustered_boot(df), "segment_macro_auc_ci": segment_boot(df)})
    m["confusion"] = confusion_matrix(df.y, df.pred).tolist()
    R[f"variant_{v}"] = m
    print(v, {k: m[k] for k in ["acc", "macro_auc", "macro_auprc", "ece", "brier"]}, m["clustered"]["macro_auc_ci"], m["segment_macro_auc_ci"])

# ---- 2. recording-level aggregation (eeg_id), hard model
def recording_level(df):
    g = df.groupby("eeg_id")
    mixed = float((g.y.nunique() > 1).mean())
    agg = g.agg(**{f"p{c}": (f"p{c}", "mean") for c in range(4)}, y=("y", lambda s: s.mode().iloc[0]), n=("y", "size"))
    ys = agg[[f"p{c}" for c in range(4)]].values; y = agg.y.values; pred = ys.argmax(1)
    aucs = [roc_auc_score(y == c, ys[:, c]) for c in range(4)]
    return {"n_recordings": int(len(agg)), "frac_mixed_label_recordings": mixed, "median_segments_per_recording": float(agg.n.median()),
            "acc": float((y == pred).mean()), "macro_auc": float(np.mean(aucs)), "auc": dict(zip(CLASSES, map(float, aucs)))}
for v in ["hard", "soft"]:
    R[f"recording_level_{v}"] = recording_level(P[v]); print("recording", v, R[f"recording_level_{v}"])

# ---- 3. consensus-stratified performance
df = P["hard"]
R["consensus_distribution"] = {"median_frac": float(df.cons.median()), "frac_ge_0.75": float((df.cons >= 0.75).mean()), "frac_eq_1": float((df.cons >= 0.999).mean()), "median_n_votes": float(df.nv.median())}
strata = [(0, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.01)]
R["consensus_strata"] = {}
for lo, hi in strata:
    d = df[(df.cons >= lo) & (df.cons < hi)]
    if len(d) > 100 and d.y.nunique() == 4:
        m = metrics(d); R["consensus_strata"][f"[{lo},{hi})"] = {"n": int(len(d)), "acc": m["acc"], "macro_auc": m["macro_auc"]}
        for v in ["soft"]:
            d2 = P[v].loc[d.index]; m2 = metrics(d2); R["consensus_strata"][f"[{lo},{hi})"][f"acc_{v}"] = m2["acc"]; R["consensus_strata"][f"[{lo},{hi})"][f"macro_auc_{v}"] = m2["macro_auc"]
print(R["consensus_strata"])
# agreement with expert when model confident vs expert consensus level
hi = df[df.cons >= 0.75]; R["high_consensus_hard"] = metrics(hi); R["high_consensus_hard"]["n"] = int(len(hi))
R["high_consensus_soft"] = metrics(P["soft"].loc[hi.index])

# ---- 4. Others subtypes & per-fold variance vs patient count
R["others_subtype_recall"] = {s: float((df[df["sub"] == s].pred == 3).mean()) for s in ["LRDA", "GRDA", "Other"]}
fold_acc = df.groupby("fold").apply(lambda d: (d.y == d.pred).mean()); fold_npat = df.groupby("fold").patient.nunique(); fold_n = df.groupby("fold").size()
from scipy.stats import spearmanr
R["fold_level"] = {"acc": fold_acc.round(4).tolist(), "n_patients": fold_npat.tolist(), "n_segments": fold_n.tolist(), "spearman_acc_vs_npat": float(spearmanr(fold_acc, fold_npat)[0])}

# ---- 5. attention audit arrays
att = np.load(os.path.join(ROOT, "mod", "analysis_patient_grouped", "attention_rollout_class_matrices.npz"))
print("npz keys", list(att.keys()))
caus = json.load(open(os.path.join(ROOT, "mod", "analysis_patient_grouped", "causal_ablation_summary.json")))
seed = json.load(open(os.path.join(ROOT, "mod", "analysis_patient_grouped", "seed_stability_permutation_null_summary.json")))
R["causal_ablation"] = caus; R["seed_summary"] = seed

json.dump(R, open(os.path.join(OUT, "extra_results.json"), "w"), indent=1)

# ================= FIGURES =================
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"GAT": "#1f4e79", "SVM": "#2a9d8f", "MLP": "#e9c46a", "CNN": "#e76f51"}

# Fig 2: leakage before/after
old = {"GAT": 0.9732, "SVM": 0.9333, "CNN": 0.9186, "MLP": 0.8993}
new = {"GAT": 0.8395, "SVM": 0.7891, "MLP": 0.7738, "CNN": 0.7675}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3), gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(4); w = 0.38; order = ["GAT", "SVM", "MLP", "CNN"]
ax[0].bar(x - w/2, [old[m] for m in order], w, color="#bbbbbb", label="Segment-level split (leaky)")
ax[0].bar(x + w/2, [new[m] for m in order], w, color=[C[m] for m in order], label="Patient-grouped split")
for i, m in enumerate(order):
    ax[0].text(i + w/2, new[m] + 0.01, f"{new[m]:.3f}", ha="center", fontsize=7); ax[0].text(i - w/2, old[m] + 0.01, f"{old[m]:.3f}", ha="center", fontsize=7, color="#555")
ax[0].set_xticks(x); ax[0].set_xticklabels(order); ax[0].set_ylim(0.5, 1.02); ax[0].set_ylabel("Macro-average AUROC"); ax[0].legend(frameon=False, fontsize=7, loc="lower left"); ax[0].set_title("a  Effect of evaluation unit", loc="left", fontweight="bold")
seg = R["variant_hard"]["segment_macro_auc_ci"]; clu = R["variant_hard"]["clustered"]["macro_auc_ci"]; pt = R["variant_hard"]["macro_auc"]
ax[1].errorbar([0], [pt], yerr=[[pt - seg[0]], [seg[1] - pt]], fmt="o", color="#888", capsize=4, label="Segment bootstrap")
ax[1].errorbar([1], [pt], yerr=[[pt - clu[0]], [clu[1] - pt]], fmt="o", color=C["GAT"], capsize=4, label="Patient-clustered bootstrap")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["Segment\nCI", "Patient-\nclustered CI"]); ax[1].set_xlim(-0.6, 1.6); ax[1].set_ylabel("Macro AUROC (GAT hybrid)"); ax[1].set_title("b  95% CI width", loc="left", fontweight="bold")
ax[1].text(0.5, 0.02, f"width ×{(clu[1]-clu[0])/(seg[1]-seg[0]):.1f}", transform=ax[1].transAxes, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_leakage.png"), dpi=200); plt.close()

# Fig 3: per-class AUC with clustered CI, GAT variants + baselines (point)
base = {"SVM": [0.7927, 0.7725, 0.8434, 0.7476], "CNN": [0.7691, 0.7593, 0.8127, 0.7288], "MLP": [0.7753, 0.7685, 0.8148, 0.7369]}
fig, ax = plt.subplots(figsize=(7.2, 3))
x = np.arange(4)
for j, (v, lab, col) in enumerate([("hard", "GAT hybrid (hard label)", C["GAT"]), ("soft", "GAT hybrid (soft label)", "#4f81bd")]):
    m = R[f"variant_{v}"]; pts = [m["auc"][c] for c in CLASSES]; ci = [m["clustered"]["class_auc_ci"][c] for c in CLASSES]
    ax.errorbar(x + (j - 0.5) * 0.18, pts, yerr=[[p - c[0] for p, c in zip(pts, ci)], [c[1] - p for p, c in zip(pts, ci)]], fmt="o", color=col, capsize=3, label=lab)
for m, col, mk in [("SVM", C["SVM"], "s"), ("MLP", C["MLP"], "^"), ("CNN", C["CNN"], "v")]:
    ax.scatter(x + 0.32, base[m], marker=mk, color=col, label=m, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(CLASSES); ax.set_ylabel("One-vs-rest AUROC"); ax.set_ylim(0.7, 0.92); ax.legend(frameon=False, fontsize=7, ncol=2)
ax.set_title("Per-class discrimination, patient-grouped 10-fold (error bars: patient-clustered 95% CI)", loc="left", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_perclass.png"), dpi=200); plt.close()

# Fig 4: representation decomposition
rep = [("PSD only\n(edges & flat-FC zeroed)", 0.7451, 0.0250), ("FC only\n(PSD zeroed)", 0.7469, 0.0324), ("Full hybrid", 0.8347, 0.0235)]
loo = [("Full", 0.8347), ("− corr", 0.8192), ("− coh", 0.8198), ("− PLV", 0.8217), ("− AEC", 0.8266)]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
ax[0].bar(range(3), [r[1] for r in rep], yerr=[r[2] for r in rep], color=["#999", "#999", C["GAT"]], capsize=4)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([r[0] for r in rep], fontsize=7); ax[0].set_ylim(0.6, 0.9); ax[0].set_ylabel("Macro AUROC (mean ± SD, 10 folds)"); ax[0].set_title("a  Node vs edge information", loc="left", fontweight="bold")
ax[1].bar(range(5), [l[1] for l in loo], color=[C["GAT"]] + ["#c0504d"] * 4)
for i, l in enumerate(loo[1:], 1): ax[1].text(i, l[1] + 0.002, f"{l[1]-0.8347:+.3f}", ha="center", fontsize=7)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([l[0] for l in loo]); ax[1].set_ylim(0.78, 0.85); ax[1].set_ylabel("Macro AUROC"); ax[1].set_title("b  Leave-one-FC-measure-out (all DeLong p<0.001)", loc="left", fontweight="bold", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_representation.png"), dpi=200); plt.close()

# Fig 5: attention audit
CH = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']  # channel order of the stored psd_sb/fc_sb files = HMS parquet column order (corrected 29 Oct 2026; the 10-20 list used before mislabelled electrodes)
keys = [k for k in att.keys()]
fig, axes = plt.subplots(1, 5, figsize=(9.5, 2.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.3]})
for i, c in enumerate(["SEIZURE", "LPD", "GPD", "OTHERS"]):
    k = [kk for kk in keys if c.lower() in kk.lower()]
    M = att[k[0]] if k else np.zeros((19, 19))
    if M.ndim > 2: M = M.mean(0)
    im = axes[i].imshow(M, cmap="viridis"); axes[i].set_title(CLASSES[i], fontsize=8); axes[i].set_xticks([]); axes[i].set_yticks([])
ax = axes[4]
drops = {"Seizure": (0.0002, -0.0004, 0.0014, 0.258), "LPD": (0.0008, 0.0010, 0.0018, 0.839), "GPD": (0.0002, 0.0009, 0.0014, 0.032), "Others": (-0.0004, -0.0007, 0.0009, 1.0)}
x = np.arange(4)
ax.bar(x - 0.2, [drops[c][0] for c in CLASSES], 0.4, color=C["GAT"], label="Top-10 attention edges")
ax.bar(x + 0.2, [drops[c][1] for c in CLASSES], 0.4, yerr=[drops[c][2] for c in CLASSES], color="#aaa", capsize=2, label="10 random edges (30 reps)")
for i, c in enumerate(CLASSES): ax.text(i, 0.0028, f"p={drops[c][3]:.2f}", ha="center", fontsize=6)
ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=7); ax.set_ylabel("Accuracy drop after ablation", fontsize=7); ax.set_ylim(-0.003, 0.0035); ax.legend(frameon=False, fontsize=6, loc="lower left")
ax.set_title("Causal ablation", fontsize=8)
plt.suptitle("Attention rollout (graph branch) per class and its causal (non-)effect", fontsize=9, x=0.02, ha="left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_attention.png"), dpi=200); plt.close()

# Fig 6: calibration + consensus
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
for v, lab, col in [("hard", f"Hard label (ECE {R['variant_hard']['ece']:.3f})", C["GAT"]), ("soft", f"Soft label (ECE {R['variant_soft']['ece']:.3f})", "#4f81bd")]:
    cur = np.array(R[f"variant_{v}"]["reliability"]); ax[0].plot(cur[:, 0], cur[:, 1], "o-", color=col, label=lab, ms=3)
ax[0].plot([0.25, 1], [0.25, 1], "k--", lw=0.7); ax[0].set_xlabel("Mean predicted confidence"); ax[0].set_ylabel("Observed accuracy"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("a  Reliability (pooled OOF, 15 bins)", loc="left", fontweight="bold", fontsize=8)
st = R["consensus_strata"]; labs = list(st.keys())
ax[1].plot(range(len(labs)), [st[k]["macro_auc"] for k in labs], "o-", color=C["GAT"], label="Hard label")
ax[1].plot(range(len(labs)), [st[k]["macro_auc_soft"] for k in labs], "s--", color="#4f81bd", label="Soft label")
ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(["<0.5", "0.5–0.75", "0.75–1", "=1"], fontsize=7); ax[1].set_xlabel("Expert consensus fraction"); ax[1].set_ylabel("Macro AUROC")
for i, k in enumerate(labs): ax[1].text(i, st[k]["macro_auc"] + 0.008, f"n={st[k]['n']}", ha="center", fontsize=6)
ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("b  Performance vs label agreement", loc="left", fontweight="bold", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig6_calibration.png"), dpi=200); plt.close()
print("done")
