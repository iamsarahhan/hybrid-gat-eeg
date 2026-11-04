"""A2. Does the model agree with the expert consensus as often as a single expert does?

The vote counts per clip let us simulate a 'random single annotator': draw one vote
from the clip's vote distribution and ask whether it matches the modal label. Repeated
over all clips this is the expected majority-agreement (PA) of an individual expert on
exactly our subset -- the fair yardstick for the model's accuracy (Jing et al. 2023
report PA 65% overall against majority, pairwise 52%, on the full MGH set).

Tests
 1. Model accuracy vs simulated single-expert PA, overall and per class, 200 draws,
    patient-clustered CI for the model.
 2. Same on the >=0.75-consensus subset and on unanimous clips (where PA = 1 by
    construction, so the model's shortfall there is pure model error).
Outputs: results/a2_expert_agreement.json
"""
import numpy as np
from common import *

rng = np.random.default_rng(1)
R = {}
for variant in ["hard", "soft"]:
    df = pooled_predictions(variant); V = df[[f"v_{c}" for c in CLASSES]].values; y = df.y.values
    sim = []
    for _ in range(200):
        draw = np.array([rng.choice(4, p=v / v.sum()) for v in V])
        sim.append([(draw == y).mean()] + [(draw[y == c] == c).mean() for c in range(4)])
    sim = np.array(sim)
    r = {"model_acc": float((df.pred == y).mean()), "model_acc_ci": clustered_ci(df, lambda d: (d.pred == d.y).mean(), n=300),
         "single_expert_PA_mean": float(sim[:, 0].mean()), "single_expert_PA_sd": float(sim[:, 0].std()),
         "per_class": {c: {"model_recall": float((df.pred[y == i] == i).mean()), "single_expert_PA": float(sim[:, i + 1].mean())} for i, c in enumerate(CLASSES)}}
    for name, m in [("consensus_ge_0.75", df.cons >= 0.75), ("consensus_lt_0.5", df.cons < 0.5), ("unanimous", df.cons >= 0.999)]:
        d = df[m]; Vs = V[m.values]; ys = y[m.values]
        pa = np.mean([(np.array([rng.choice(4, p=v / v.sum()) for v in Vs]) == ys).mean() for _ in range(20)])
        r[name] = {"n": int(len(d)), "model_acc": float((d.pred == d.y).mean()), "single_expert_PA": float(pa)}
    R[variant] = r; print(variant, r["model_acc"], r["model_acc_ci"], "expert PA", r["single_expert_PA_mean"], r["unanimous"])
save("a2_expert_agreement.json", R)
