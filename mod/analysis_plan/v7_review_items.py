"""v7: review items that need no retraining.
 - per-class sensitivity, specificity, PPV, NPV at argmax (balanced and native-prevalence weighted) + confusion matrix
 - class-wise calibration: per-class ECE and reliability curves (one-vs-rest probability), Brier per class
 - Others subtype recall (LRDA / GRDA / Other)
 - per-patient alert burden at the 90%-sensitivity thresholds
 - minimal detectable paired difference in macro AUROC at this cohort size (from patient-paired bootstrap SD)
 - seed variability of the hybrid GAT on fold 1 (from seed_stability summary)
Outputs: results/v7_review_items.json, results/figs_v5/S1_classwise_calibration.png
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT, OUT, pooled_predictions

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "legend.frameon": False})
def label_panel(ax, letter): ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
CLS = ["Seizure", "LPD", "GPD", "Others"]; COLS = ["#1f4e79", "#2a9d8f", "#e9c46a", "#e76f51"]
NATIVE = np.array([20933, 14856, 16702, 16640 + 18861 + 18806]); PREV = NATIVE / NATIVE.sum()
R = {}
for variant in ["soft", "hard"]:
    df = pooled_predictions(variant); P = df[[f"p_{c}" for c in CLS]].values; y = df.y.values; pred = P.argmax(1)
    out = {}
    for wname, w in [("balanced", np.ones(len(y))), ("native", PREV[y] / 0.25)]:
        cm = np.zeros((4, 4))
        for i in range(4):
            for j in range(4): cm[i, j] = w[(y == i) & (pred == j)].sum()
        tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = cm.sum() - tp - fn - fp
        out[wname] = {"confusion": (cm / w.sum() * len(y)).round(0).tolist(), "sensitivity": dict(zip(CLS, (tp / (tp + fn)).round(3))), "specificity": dict(zip(CLS, (tn / (tn + fp)).round(3))),
                      "ppv": dict(zip(CLS, (tp / (tp + fp)).round(3))), "npv": dict(zip(CLS, (tn / (tn + fn)).round(3))), "accuracy": float(tp.sum() / cm.sum()), "balanced_accuracy": float((tp / (tp + fn)).mean())}
    # class-wise calibration (one-vs-rest probability)
    cw = {}
    for i, c in enumerate(CLS):
        p = P[:, i]; t = (y == i).astype(float); edges = np.linspace(0, 1, 11); e = 0; pts = []
        for b in range(10):
            m = (p > edges[b]) & (p <= edges[b + 1])
            if m.sum() > 30: gap = abs(p[m].mean() - t[m].mean()); e += m.mean() * gap; pts.append((float(p[m].mean()), float(t[m].mean()), int(m.sum())))
        cw[c] = {"ece_ovr": float(e), "brier_ovr": float(((p - t) ** 2).mean()), "reliability": pts}
    out["classwise_calibration"] = cw
    out["others_subtype_recall"] = {s: float((df[df["sub"] == s].pred == 3).mean()) for s in ["LRDA", "GRDA", "Other"]}
    out["others_subtype_misroute"] = {s: {c: float((df[df["sub"] == s].pred == i).mean()) for i, c in enumerate(CLS[:3])} for s in ["LRDA", "GRDA", "Other"]}
    # per-patient alert burden at 90% sensitivity thresholds (native weights)
    w = PREV[y] / 0.25; burden = {}
    for i, c in enumerate(CLS):
        fpr, tpr, thr = roc_curve(y == i, P[:, i], sample_weight=w); t = thr[np.argmax(tpr >= 0.90)]; flag = P[:, i] >= t
        g = pd.DataFrame({"patient": df.patient, "flag": flag, "pos": y == i}).groupby("patient")
        frac = g.flag.mean(); anypos = g.pos.any()
        burden[c] = {"threshold": float(t), "median_fraction_of_patient_windows_flagged": float(frac.median()), "patients_with_any_alert": float((g.flag.any()).mean()),
                     "patients_without_class_but_alerted": float(((~anypos) & g.flag.any()).mean() / max((~anypos).mean(), 1e-9))}
    out["alert_burden_90sens"] = burden
    R[variant] = out
    print(variant, out["balanced"]["sensitivity"], out["native"]["ppv"], {c: round(cw[c]["ece_ovr"], 3) for c in CLS}, out["others_subtype_recall"])
# minimal detectable difference from paired bootstrap of soft vs hard (same cohort)
v5 = json.load(open(os.path.join(OUT, "v5_stats.json"))); ci = v5["paired_patient_bootstrap"]["soft_vs_hard"]["ci"]; sd = (ci[1] - ci[0]) / (2 * 1.96)
R["minimal_detectable_paired_diff_macro_auc"] = {"bootstrap_sd": float(sd), "mdd_80pct_power_alpha05": float(2.8 * sd), "note": "from patient-paired bootstrap SD of the soft-vs-hard difference; applies to models with similar error structure"}
seed = json.load(open(os.path.join(ROOT, "mod", "analysis_patient_grouped", "seed_stability_permutation_null_summary.json")))
acc = [r["test_acc"] for r in seed["real_runs"]]; va = [r["val_auc"] for r in seed["real_runs"]]
R["seed_variability_fold1"] = {"n_seeds": len(acc), "test_acc_mean": float(np.mean(acc)), "test_acc_sd": float(np.std(acc)), "val_auc_mean": float(np.mean(va)), "val_auc_sd": float(np.std(va))}
json.dump(R, open(os.path.join(OUT, "v7_review_items.json"), "w"), indent=1)
# supplementary figure: class-wise reliability, soft vs hard
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
for i, c in enumerate(CLS):
    for variant, mk, col in [("soft", "s-", COLS[i]), ("hard", "o--", "#888")]:
        pts = np.array(R[variant]["classwise_calibration"][c]["reliability"]); ax[i].plot(pts[:, 0], pts[:, 1], mk, color=col, ms=4, label=f"{variant} (ECE {R[variant]['classwise_calibration'][c]['ece_ovr']:.3f})")
    ax[i].plot([0, 1], [0, 1], "k:", lw=0.8); ax[i].set_title(f"{c} vs rest", fontsize=10); ax[i].set_xlabel("Predicted probability"); ax[i].legend(fontsize=7, loc="upper left"); label_panel(ax[i], "ABCD"[i])
ax[0].set_ylabel("Observed frequency")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figs_v5", "S1_classwise_calibration.png"), dpi=200, bbox_inches="tight"); print("done")
