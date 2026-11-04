"""A1. Is the model's class probability a graded scale that tracks expert uncertainty?

Research question: when experts are split between two patterns, does the model's
probability mass split the same way -- and only for pairs that are neighbours on
the ictal-interictal continuum (seizure<->LPD, seizure<->GPD, LPD<->LRDA, GPD<->GRDA)?

Tests
 1. Per class: Spearman rho between model P(class) and expert vote fraction, with
    the curve of mean P(class) in 10 bins of vote fraction.
 2. Split-vote clips (top two vote shares both >= 0.25): fraction of model mass on
    the two voted classes vs on the other two. Compared across class pairs.
 3. KL(votes || model) hard vs soft model, patient-clustered CI.
Outputs: results/a1_continuum.json, results/a1_continuum.png
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, entropy
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

R = {}
for variant in ["hard", "soft"]:
    df = pooled_predictions(variant); P = df[[f"p_{c}" for c in CLASSES]].values; V = df[[f"v_{c}" for c in CLASSES]].values
    r = {}
    # 1. monotonic relation per class
    for i, c in enumerate(CLASSES):
        rho = spearmanr(P[:, i], V[:, i])[0]
        bins = np.clip((V[:, i] * 10).astype(int), 0, 9)
        curve = [float(P[bins == b, i].mean()) if (bins == b).sum() > 30 else None for b in range(10)]
        r[c] = {"spearman": float(rho), "binned_mean_prob": curve, "n_per_bin": [int((bins == b).sum()) for b in range(10)]}
    # 2. split-vote clips
    top2 = np.argsort(-V, 1)[:, :2]; s2 = np.sort(V, 1)[:, -2]
    split = s2 >= 0.25
    pairs = {}
    for a in range(4):
        for b in range(a + 1, 4):
            m = split & (((top2[:, 0] == a) & (top2[:, 1] == b)) | ((top2[:, 0] == b) & (top2[:, 1] == a)))
            if m.sum() >= 50:
                mass = P[m][:, [a, b]].sum(1)
                pairs[f"{CLASSES[a]}-{CLASSES[b]}"] = {"n": int(m.sum()), "model_mass_on_pair": float(mass.mean()),
                    "expert_mass_on_pair": float(V[m][:, [a, b]].sum(1).mean()),
                    "frac_argmax_in_pair": float(np.isin(P[m].argmax(1), [a, b]).mean())}
    r["split_vote_pairs"] = pairs; r["n_split_clips"] = int(split.sum())
    # 3. KL and agreement with vote distribution
    kl = entropy(V.T + 1e-9, P.T + 1e-9)
    df["kl"] = kl
    r["kl_votes_vs_model_mean"] = float(kl.mean()); r["kl_ci"] = clustered_ci(df, lambda d: d.kl.mean(), n=300)
    R[variant] = r
    print(variant, {c: round(r[c]["spearman"], 3) for c in CLASSES}, "split clips", r["n_split_clips"], "KL", round(r["kl_votes_vs_model_mean"], 3))
save("a1_continuum.json", R)

fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
x = np.arange(10) / 10 + 0.05
for i, c in enumerate(CLASSES):
    y = [v if v is not None else np.nan for v in R["hard"][c]["binned_mean_prob"]]
    ax[0].plot(x, y, "o-", ms=3, label=f"{c} (rho={R['hard'][c]['spearman']:.2f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=.6); ax[0].set_xlabel("Expert vote fraction for class"); ax[0].set_ylabel("Mean model probability (hard-label model)"); ax[0].legend(fontsize=7, frameon=False)
pairs = R["hard"]["split_vote_pairs"]; names = list(pairs)
ax[1].barh(range(len(names)), [pairs[k]["model_mass_on_pair"] for k in names], color="#1f4e79")
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels([f"{k} (n={pairs[k]['n']})" for k in names], fontsize=7); ax[1].set_xlabel("Model probability mass on the two voted classes"); ax[1].axvline(0.5, color="k", lw=.6, ls="--")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "a1_continuum.png"), dpi=180)
