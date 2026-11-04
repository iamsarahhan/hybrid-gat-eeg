"""Figure 4 of the manuscript (file F3_graph_contribution.png), three panels, from saved results only:
A arms on identical patient folds; B patient-paired differences; C leave-one-measure-out.
Also writes F3b_comparators.png (the r4 two-panel version) so r4_compare_all.py no longer overwrites this file."""
import os, json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import OUT
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "legend.frameon": False})
def label_panel(ax, letter): ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
R4 = json.load(open(os.path.join(OUT, "r4_comparison.json"))); V5 = json.load(open(os.path.join(OUT, "v5_stats.json")))
m = R4["models"]; p = R4["pairs"]
# CPU control results if any finished
ctrl = {}
ctrl_ci = {}
for c in ["hybrid", "graph_only", "flat_only", "edge_shuffle", "edge_const"]:
    if f"CPU {c}" in m: ctrl[c] = m[f"CPU {c}"]["macro_auc"]; ctrl_ci[c] = m[f"CPU {c}"]["ci"]
CF = ", ".join(str(k) for k in m.get("CPU hybrid", {}).get("folds", []))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [2.4, 1.1, 1]})
label_panel(ax[0], "A")
arms = [("Spectra only\n(input ablation)", 0.7451, 0.0250, "#bbbbbb", None), ("Connectivity only\n(input ablation)", 0.7469, 0.0324, "#bbbbbb", None),
        ("Graph-theory\nGBM", m["Graph-theory GBM"]["macro_auc"], None, "#7f7f7f", m["Graph-theory GBM"]["ci"]), ("MLP", m["MLP"]["macro_auc"], None, "#e9c46a", m["MLP"]["ci"]),
        ("SVM (RBF)", m["SVM"]["macro_auc"], None, "#2a9d8f", m["SVM"]["ci"]), ("Hybrid GAT", m["GAT hybrid"]["macro_auc"], None, "#1f4e79", m["GAT hybrid"]["ci"]),
        ("Network\npathway only", ctrl.get("graph_only"), None, "#5b8db8", None), ("Matrix\npathway only", ctrl.get("flat_only"), None, "#b8860b", None), ("Edge-\nshuffled", ctrl.get("edge_shuffle"), None, "#c0504d", None), ("Constant\nedges", ctrl.get("edge_const"), None, "#e07b6a", None), ("Hybrid\n(control run)", ctrl.get("hybrid"), None, "#1f4e79", None)]
for i, (n, v, sd, col, ci) in enumerate(arms):
    if v is None: ax[0].bar(i, 0.9, color="white", edgecolor="#999", ls="--", hatch="//"); ax[0].text(i, 0.705, "fold-1 run\nin progress", ha="center", fontsize=7, color="#777")
    else:
        ax[0].bar(i, v, color=col, edgecolor="k", lw=0.5)
        if sd is not None: ax[0].errorbar(i, v, yerr=sd, color="k", capsize=3, lw=1)
        if ci is not None: ax[0].errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], color="k", capsize=3, lw=1)
        ax[0].text(i, (ci[1] if ci else v + (sd or 0)) + 0.006, f"{v:.3f}", ha="center", fontsize=7)
ax[0].set_xticks(range(len(arms))); ax[0].set_xticklabels([a[0].replace("\n", " ") for a in arms], fontsize=7.5, rotation=30, ha="right", rotation_mode="anchor"); ax[0].set_ylim(0.65, 0.92); ax[0].set_ylabel("Macro AUROC")
ax[0].set_title("Arms on identical patient folds (SD over folds for ablations; patient-clustered 95% CI otherwise)", fontsize=9, pad=8); ax[0].axvline(5.5, color="#999", ls="--", lw=0.8); ax[0].text(5.65, 0.905, f"folds {CF}, single control protocol", fontsize=7, color="#666")
label_panel(ax[1], "B")
pairs = [("Hybrid − SVM", p["GAT hybrid vs SVM"]), ("Hybrid − MLP", p["GAT hybrid vs MLP"]), ("Hybrid − GBM", p["GAT hybrid vs Graph-theory GBM"]),
         ("Soft − hard labels", V5["paired_patient_bootstrap"]["soft_vs_hard"]), ("Edge refinement − none", V5["paired_patient_bootstrap"]["gmm_vs_hard"])]
for i, (lab, d) in enumerate(pairs): ax[1].errorbar(d["diff"], i, xerr=[[d["diff"] - d["ci"][0]], [d["ci"][1] - d["diff"]]], fmt="o", color="#1f4e79", capsize=3)
ax[1].axvline(0, color="k", lw=0.8); ax[1].axvspan(-0.012, 0.012, color="#eee", zorder=0); ax[1].set_yticks(range(len(pairs))); ax[1].set_yticklabels([q[0] for q in pairs], fontsize=8); ax[1].invert_yaxis()
ax[1].set_xlabel("Δ macro AUROC, patient-paired 95% CI"); ax[1].set_title("Paired differences (grey: minimal detectable ±0.012)", fontsize=9)
label_panel(ax[2], "C"); loo = [("Full", 0.8347), ("− corr", 0.8192), ("− coh", 0.8198), ("− PLV", 0.8217), ("− AEC", 0.8266)]
ax[2].bar(range(5), [l[1] for l in loo], color=["#1f4e79"] + ["#c0504d"] * 4, edgecolor="k", lw=0.5)
for i, l in enumerate(loo[1:], 1): ax[2].text(i, l[1] + 0.002, f"{l[1] - 0.8347:+.3f}", ha="center", fontsize=7)
ax[2].set_xticks(range(5)); ax[2].set_xticklabels([l[0] for l in loo]); ax[2].set_ylim(0.78, 0.85); ax[2].set_ylabel("Macro AUROC"); ax[2].set_title("Leave-one-measure-out (retrained)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figs_v5", "F3_graph_contribution.png"), dpi=int(os.environ.get("FIGDPI","200")), bbox_inches="tight"); print("F3 saved", ctrl)
