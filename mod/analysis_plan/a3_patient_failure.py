"""A3. Where does the error live: in a few patients, in low-agreement labels, or everywhere?

Tests
 1. Per-patient accuracy distribution; share of all errors contributed by the worst
    10% of patients (concentration) vs the share expected if errors were spread evenly.
 2. Patient-level regression: accuracy ~ log(n_clips) + mean consensus + label entropy
    (patients whose clips carry several different labels) + dominant class.
 3. Recording-level: do patients with >1 recording show consistent errors across recordings?
Outputs: results/a3_patient_failure.json, results/a3_patient_failure.png
"""
import numpy as np, pandas as pd
from scipy.stats import entropy
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

df = pooled_predictions("hard"); df["err"] = (df.pred != df.y).astype(int)
g = df.groupby("patient")
pat = pd.DataFrame({"n": g.size(), "acc": 1 - g.err.mean(), "errors": g.err.sum(), "cons": g.cons.mean(),
                    "label_entropy": g.y.apply(lambda s: entropy(np.bincount(s, minlength=4))),
                    "dominant": g.y.agg(lambda s: s.mode().iloc[0]), "n_rec": g.eeg_id.nunique()})
pat = pat.sort_values("errors", ascending=False)
R = {"n_patients": int(len(pat)), "median_clips_per_patient": float(pat.n.median()),
     "acc_quartiles": pat.acc.quantile([.1, .25, .5, .75, .9]).round(3).to_dict(),
     "share_of_errors_from_worst_10pct_patients": float(pat.errors.head(len(pat) // 10).sum() / pat.errors.sum()),
     "share_of_clips_in_worst_10pct_patients": float(pat.n.head(len(pat) // 10).sum() / pat.n.sum()),
     "patients_with_zero_correct": int((pat.acc == 0).sum()), "patients_all_correct": int((pat.acc == 1).sum())}
X = pd.DataFrame({"log_n": np.log(pat.n), "cons": pat.cons, "label_entropy": pat.label_entropy})
X = pd.concat([X, pd.get_dummies(pat.dominant, prefix="dom", drop_first=True).astype(float)], axis=1)
fit = sm.WLS(pat.acc, sm.add_constant(X), weights=pat.n).fit()
R["patient_regression"] = {"coef": fit.params.round(4).to_dict(), "p": fit.pvalues.round(4).to_dict(), "r2": float(fit.rsquared)}
multi = pat[pat.n_rec > 1]
rec = df[df.patient.isin(multi.index)].groupby(["patient", "eeg_id"]).err.mean().reset_index()
within = rec.groupby("patient").err.var().mean(); between = rec.groupby("patient").err.mean().var()
R["recording_consistency"] = {"n_patients_multi_recording": int(len(multi)), "between_patient_var": float(between), "within_patient_var": float(within)}
print(R)
save("a3_patient_failure.json", R)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].hist(pat.acc, bins=20, color="#1f4e79"); ax[0].set_xlabel("Per-patient accuracy"); ax[0].set_ylabel("Patients")
ax[1].scatter(pat.cons, pat.acc, s=np.sqrt(pat.n) * 2, alpha=.3, color="#1f4e79"); ax[1].set_xlabel("Mean expert consensus (patient)"); ax[1].set_ylabel("Accuracy")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "a3_patient_failure.png"), dpi=180)
