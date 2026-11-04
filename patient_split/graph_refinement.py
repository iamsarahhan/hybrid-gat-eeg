"""
GMM + iterative Gumbel-softmax edge refinement module (R1-1).

Operationalizes the Methods claim in review/draft.docx (paragraph 37): "edge
weights were modelled with a Gaussian mixture and refined by iterative Gumbel
sampling" -- no such module existed anywhere in the codebase before this
(GAT_4class_patient_hybrid_auc.py feeds the raw 4-dim [corr,coh,aec,plv]
edge_attr straight into GATConv). See review/SESSION_LOG_patient_grouped_rework.md
section 12 for the on/off ablation this module was built to support.

Design: a single global K-component Gaussian mixture (diagonal covariance) is
learned over the 4-dim raw edge-feature space, shared across all edges/samples.
For each edge, the mixture posterior responsibility is computed, a component is
drawn via the Gumbel-softmax relaxation (differentiable, reparameterized), and
the edge feature is refined by mixing it with the sampled component's learned
prototype mean. This repeats for T iterations, each iteration's output feeding
the next as the new "raw" input -- the "iterative" part of the refinement.
Output has the same shape as the input edge_attr, so it's a drop-in replacement
feeding GATConv exactly as raw edge_attr did.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMGumbelEdgeRefinement(nn.Module):
    def __init__(self, in_dim=4, num_components=3, num_iters=3, temperature=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_components = num_components
        self.num_iters = num_iters
        self.temperature = temperature  # mutated per-epoch by the training loop for annealing

        self.component_mean = nn.Parameter(torch.randn(num_components, in_dim) * 0.5)
        self.component_logvar = nn.Parameter(torch.zeros(num_components, in_dim))
        self.component_prior_logit = nn.Parameter(torch.zeros(num_components))
        self.refine = nn.Linear(in_dim * 2, in_dim)

    def _log_gaussian(self, x):
        # x: (E, D) -> (E, K) log N(x_e; mean_k, diag(var_k)), summed over D
        var = self.component_logvar.exp()
        diff = x.unsqueeze(1) - self.component_mean.unsqueeze(0)          # (E, K, D)
        log_prob = -0.5 * (diff.pow(2) / var.unsqueeze(0)
                            + self.component_logvar.unsqueeze(0)
                            + math.log(2 * math.pi))
        return log_prob.sum(-1)                                            # (E, K)

    def forward(self, edge_attr):
        x = edge_attr
        z = None
        for _ in range(self.num_iters):
            log_posterior = self._log_gaussian(x) + self.component_prior_logit.unsqueeze(0)
            z = F.gumbel_softmax(log_posterior, tau=self.temperature, hard=False, dim=-1)  # (E, K)
            prototype = z @ self.component_mean                            # (E, D)
            x = self.refine(torch.cat([x, prototype], dim=-1))
        return x, z
