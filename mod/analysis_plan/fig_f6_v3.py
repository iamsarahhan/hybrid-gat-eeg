"""Redraw F6 with annotator agreement defined on the six original patterns (cons6), matching Section 2.1 and 3.6."""
import os, json, numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import OUT, pooled_predictions
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "legend.frameon": False})
def label_panel(ax, letter): ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12), textcoords="offset points", fontsize=15, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
CLS = ["Seizure", "LPD", "GPD", "Others"]; COL = {"GAT": "#1f4e79", "GAT soft": "#4f81bd", "Surrogate": "#7f7f7f"}
H = pooled_predictions("hard"); S = pooled_predictions("soft"); P = lambda df: df[[f"p_{c}" for c in CLS]].values
c6 = np.load(os.path.join(OUT, "cons6.npy"))
SUR = np.concatenate([np.load(os.path.join(OUT, "a6_surrogate", f"fold_{k}", "y_score.npy")) for k in range(1, 11)])
def cal(df, p):
    conf = p.max(1); corr = (p.argmax(1) == df.y.values); edges = np.linspace(0, 1, 16); e = 0; mce = 0; pts = []
    for b in range(15):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum(): gap = abs(conf[m].mean() - corr[m].mean()); e += m.mean() * gap; mce = max(mce, gap); pts.append((conf[m].mean(), corr[m].mean()))
    return e, mce, float(((p - np.eye(4)[df.y.values]) ** 2).sum(1).mean()), np.array(pts)
def macro(y, p): return np.mean([roc_auc_score(y == c, p[:, c]) for c in range(4)])
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
a = ax[0, 0]; label_panel(a, "A")
for df, p, lab, col, mk in [(H, P(H), "Hard label", COL["GAT"], "o-"), (S, P(S), "Soft label", COL["GAT soft"], "o-"), (H, SUR, "Graph-theory GBM", COL["Surrogate"], "s--")]:
    e, mce, br, pts = cal(df, p); a.plot(pts[:, 0], pts[:, 1], mk, color=col, ms=4, label=f"{lab} (ECE {e:.3f}, MCE {mce:.3f}, Brier {br:.3f})")
a.plot([0.25, 1], [0.25, 1], "k--", lw=0.8, label="Ideal"); a.set_xlabel("Mean predicted probability (top class)"); a.set_ylabel("Fraction correct"); a.legend(fontsize=7, loc="upper left"); a.set_title("Reliability (balanced test set)")
a = ax[0, 1]; label_panel(a, "B"); strata = [(0, .5, "<0.5"), (.5, .75, "0.5–0.75"), (.75, .999, "0.75–1"), (.999, 1.01, "=1")]
for df, p, lab, col, mk in [(H, P(H), "Hard label", COL["GAT"], "o"), (S, P(S), "Soft label", COL["GAT soft"], "s"), (H, SUR, "Graph-theory GBM", COL["Surrogate"], "^")]:
    vals = [macro(df.y.values[m], p[m]) for lo, hi, _ in strata for m in [(c6 >= lo) & (c6 < hi)]]; a.plot(range(4), vals, marker=mk, color=col, label=lab)
ns = [int(((c6 >= lo) & (c6 < hi)).sum()) for lo, hi, _ in strata]
a.set_xticks(range(4)); a.set_xticklabels([f"{s[2]}\n(n = {n:,})" for s, n in zip(strata, ns)], fontsize=8.5); a.set_xlabel("Fraction of annotators agreeing with the modal pattern (six patterns)"); a.set_ylabel("Macro AUROC"); a.legend(fontsize=8); a.set_title("Discrimination vs expert agreement")
a = ax[1, 0]; label_panel(a, "C"); y = S.y.values; ptrue = P(S)[np.arange(len(y)), y]
data = [ptrue[(c6 >= lo) & (c6 < hi)] for lo, hi, _ in strata]
bp = a.boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
for patch, col in zip(bp["boxes"], ["#e76f51", "#e9c46a", "#4f81bd", "#1f4e79"]): patch.set_facecolor(col); patch.set_alpha(0.7)
a.set_xticklabels([s[2] for s in strata]); a.set_xlabel("Annotator agreement (six patterns)"); a.set_ylabel("Model probability of the modal label"); a.set_title("Soft-label model, probability of expert label")
a = ax[1, 1]; label_panel(a, "D"); a2 = json.load(open(os.path.join(OUT, "a2_expert_agreement.json")))["soft"]; x = np.arange(4)
a.bar(x - 0.2, [a2["per_class"][c]["single_expert_PA"] for c in CLS], 0.4, color="#bbbbbb", label="Simulated single annotator (upper bound)", edgecolor="k", lw=0.5)
a.bar(x + 0.2, [a2["per_class"][c]["model_recall"] for c in CLS], 0.4, color=COL["GAT soft"], label="Soft-label model", edgecolor="k", lw=0.5)
a.set_xticks(x); a.set_xticklabels(CLS); a.set_ylabel("Agreement with modal label (recall)"); a.set_ylim(0, 1.3); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); a.legend(fontsize=8, loc="upper center", ncol=1); a.set_title("Model vs one reader, per class")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figs_v5", "F6_uncertainty_calibration.png"), dpi=int(os.environ.get("FIGDPI","200")), bbox_inches="tight"); print("F6 saved", ns)
print("medians C:", [round(float(np.median(d)), 3) for d in data])
