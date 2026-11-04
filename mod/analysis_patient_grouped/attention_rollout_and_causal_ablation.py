"""
Phase 3-1 + 3-2: attention rollout (approximate GAtt) and causal edge-ablation
for the GAT hybrid model's graph branch, cross-checked against each other per
reviewer #2's requirement ("interpret only edges that are convergent and
significant").

IMPORTANT SCOPE NOTE: this interprets only the graph branch (GATConv,
edge_dim=4). The hybrid model's flat-FC branch does not route through
attention at all, and the branch-ablation study
(patient_split/ablation_hybrid_branches.py) already showed both branches
contribute comparably and complementarily (~0.745-0.76 macro AUC each alone
vs 0.8347 combined). So attention here explains how the graph branch
reasons, not the full hybrid model's decision -- this caveat must appear
wherever these figures are used.

Not the exact GAtt algorithm (Abnar & Zuidema 2020 attention flow, adapted
for GNNs) -- this is a simplified rollout: per-layer attention averaged
across heads into a per-sample 19x19 (target x source) matrix (self-loops
included, as GATConv adds them automatically for edge_dim compatibility),
then chained via matrix multiplication across layers. Approximation, not a
computation-tree-corrected attribution.

Causal edge-ablation: for each class, take the top-K channel pairs by mean
rollout attention (computed on that class's true-labelled samples), zero
their edge_attr (both directions) at inference, and measure the accuracy/
macro-AUC drop on that class's samples -- compared against a null
distribution from ablating K random channel pairs (30 repeats). An edge
group is only interpreted as "load-bearing" if the attention-selected
ablation drop exceeds the random-ablation null at p<0.05.

No retraining -- uses existing trained checkpoints (best_model.pth per fold).
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import accuracy_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATIENT_SPLIT_DIR = os.path.join(REPO_ROOT, 'patient_split')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATIENT_SPLIT_DIR)
from DataLoader_4class_patient_hybrid import GraphLoader

N_NODES = 19
CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']  # channel order of the stored psd_sb/fc_sb files = HMS parquet column order (corrected 29 Oct 2026; the 10-20 list used before mislabelled electrodes)
CLASS_NAMES = ['SEIZURE', 'LPD', 'GPD', 'OTHERS']
TOP_K = 10
N_RANDOM_CONTROL = 30
RNG_SEED = 42


class GATHybridModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, dropout,
                 fc_in_dim, fc_hidden):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, edge_dim=4))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads,
                                       dropout=dropout, edge_dim=4))
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False,
                                   dropout=dropout, edge_dim=4))
        self.dropout = dropout
        self.fc_branch = nn.Sequential(
            nn.Linear(fc_in_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lin1 = nn.Linear(hidden_channels + fc_hidden, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch, fc_flat, return_attention=False):
        attn_layers = []
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if return_attention:
                x, (att_edge_index, alpha) = conv(x, edge_index, edge_attr, return_attention_weights=True)
                attn_layers.append((att_edge_index.detach().cpu(), alpha.detach().cpu()))
                x = F.elu(x)
            else:
                x = F.elu(conv(x, edge_index, edge_attr))
        graph_embed = global_mean_pool(x, batch)
        fc_embed = self.fc_branch(fc_flat)
        combined = torch.cat([graph_embed, fc_embed], dim=1)
        combined = F.dropout(combined, p=self.dropout, training=self.training)
        combined = F.elu(self.lin1(combined))
        out = self.lin2(combined)
        out = F.log_softmax(out, dim=1)
        if return_attention:
            return out, attn_layers
        return out

    def forward_ablated(self, x, edge_index, edge_attr, batch, fc_flat, ablate_mask):
        """ablate_mask: (E,) bool, True = zero this edge's edge_attr."""
        edge_attr = edge_attr.clone()
        edge_attr[ablate_mask] = 0.0
        return self.forward(x, edge_index, edge_attr, batch, fc_flat)


def rollout_batch(attn_layers, n_graphs_in_batch):
    """attn_layers: list of (edge_index [2,E], alpha [E,heads]) per conv layer,
    for one forward pass over a batch of n_graphs_in_batch fixed-19-node graphs.
    Returns: (n_graphs_in_batch, 19, 19) rollout matrices (target x source)."""
    n_layers = len(attn_layers)
    per_layer_mats = np.zeros((n_layers, n_graphs_in_batch, N_NODES, N_NODES), dtype=np.float64)

    for l, (edge_index, alpha) in enumerate(attn_layers):
        alpha_mean = alpha.mean(dim=1).numpy()  # (E,) mean over heads
        src = edge_index[0].numpy()
        tgt = edge_index[1].numpy()
        graph_idx = src // N_NODES  # fixed graph size -> same for src and tgt
        local_src = src % N_NODES
        local_tgt = tgt % N_NODES
        for e in range(len(src)):
            per_layer_mats[l, graph_idx[e], local_tgt[e], local_src[e]] = alpha_mean[e]

    rollout = per_layer_mats[0]
    for l in range(1, n_layers):
        rollout = np.einsum('gij,gjk->gik', per_layer_mats[l], rollout)
    return rollout


@torch.no_grad()
def get_predictions_and_rollouts(model, loader, device):
    all_true, all_pred, all_rollout = [], [], []
    for data in loader:
        data = data.to(device)
        out, attn_layers = model(data.x, data.edge_index, data.edge_attr, data.batch, data.fc_flat,
                                  return_attention=True)
        n_graphs = data.num_graphs
        rollout = rollout_batch(attn_layers, n_graphs)
        all_rollout.append(rollout)
        all_true.extend(data.y.cpu().numpy())
        all_pred.extend(out.argmax(dim=1).cpu().numpy())
    return np.array(all_true), np.array(all_pred), np.concatenate(all_rollout, axis=0)


@torch.no_grad()
def evaluate_with_edge_ablation(model, loader, device, ablate_pairs):
    """ablate_pairs: set of frozenset({i,j}) channel-index pairs to zero (both directions)."""
    all_true, all_pred = [], []
    for data in loader:
        data = data.to(device)
        src = data.edge_index[0] % N_NODES
        tgt = data.edge_index[1] % N_NODES
        mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
        for i in range(data.edge_index.size(1)):
            pair = frozenset({int(src[i]), int(tgt[i])})
            if pair in ablate_pairs:
                mask[i] = True
        mask = mask.to(device)
        out = model.forward_ablated(data.x, data.edge_index, data.edge_attr, data.batch, data.fc_flat, mask)
        all_true.extend(data.y.cpu().numpy())
        all_pred.extend(out.argmax(dim=1).cpu().numpy())
    return np.array(all_true), np.array(all_pred)


if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    psd_folders = [os.path.join(REPO_ROOT, 'psd_sb')]
    fc_folders = [os.path.join(REPO_ROOT, 'fc_sb')]
    fc_types = ['corr', 'coh', 'aec', 'plv']
    label_file = os.path.join(REPO_ROOT, 'train.csv')
    fc_in_dim = len(fc_types) * N_NODES * N_NODES
    result_dir = os.path.join(PATIENT_SPLIT_DIR, 'result_4class_patient_hybrid_auc')

    with open(os.path.join(PATIENT_SPLIT_DIR, 'optuna_best_params_hybrid_auc.json')) as f:
        params = json.load(f)['params']

    split_path = os.path.join(PATIENT_SPLIT_DIR, 'fold_splits_patient.json')
    fold_datasets = GraphLoader.create_data_loaders(
        psd_folders, fc_folders, fc_types, batch_size=32,
        label_file=label_file, random_state=42,
        save_split=False, split_save_path=split_path
    )

    # ---- Step 1: pool rollout attention + predictions across all 10 folds' test sets ----
    class_rollouts = {c: [] for c in range(4)}
    all_models = []  # keep for later ablation pass (fold, model, test_loader)

    for fold, (_train_loader, _val_loader, test_loader) in enumerate(fold_datasets, 1):
        print(f"\n[Rollout] Fold {fold}/10")
        fold_dir = os.path.join(result_dir, f'fold_{fold}')
        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pth'), map_location=device)
        model = GATHybridModel(in_channels=101, hidden_channels=params['hidden_channels'], out_channels=4,
                                num_heads=params['num_heads'], num_layers=params['num_layers'],
                                dropout=params['dropout'], fc_in_dim=fc_in_dim,
                                fc_hidden=params['fc_hidden']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        all_models.append((fold, model, test_loader))

        y_true, y_pred, rollout = get_predictions_and_rollouts(model, test_loader, device)
        for c in range(4):
            mask = y_true == c
            if mask.sum() > 0:
                class_rollouts[c].append(rollout[mask])

    class_mean_rollout = {}
    for c in range(4):
        pooled = np.concatenate(class_rollouts[c], axis=0)
        class_mean_rollout[c] = pooled.mean(axis=0)  # (19,19)
        print(f"Class {CLASS_NAMES[c]}: pooled {pooled.shape[0]} samples for rollout averaging")

    np.savez(os.path.join(OUT_DIR, 'attention_rollout_class_matrices.npz'),
             **{CLASS_NAMES[c]: class_mean_rollout[c] for c in range(4)},
             channels=np.array(CHANNELS))

    # ---- Step 2: top-K channel pairs per class by rollout attention (symmetrized) ----
    top_pairs_per_class = {}
    lines = ["=== Phase 3-1/3-2: Attention rollout top channel-pairs + causal edge-ablation ===\n"]
    lines.append("Rollout attention is graph-branch-only (see module docstring for hybrid-model caveat).\n")

    for c in range(4):
        M = class_mean_rollout[c].copy()
        np.fill_diagonal(M, 0)  # exclude self-loops from "channel pair" ranking
        sym = M + M.T
        pairs = []
        for i in range(N_NODES):
            for j in range(i + 1, N_NODES):
                pairs.append((i, j, sym[i, j]))
        pairs.sort(key=lambda t: -t[2])
        top = pairs[:TOP_K]
        top_pairs_per_class[c] = [(i, j) for i, j, _ in top]
        lines.append(f"\n[{CLASS_NAMES[c]}] top-{TOP_K} channel pairs by rollout attention:")
        for i, j, w in top:
            lines.append(f"  {CHANNELS[i]:>4s} -- {CHANNELS[j]:<4s}  weight={w:.5f}")

    # ---- Step 3: causal edge-ablation, cross-checked against random-edge control ----
    print("\n[Ablation] running causal edge-ablation vs random control...")
    lines.append(f"\n\n=== Causal ablation: top-{TOP_K} attention edges vs {N_RANDOM_CONTROL} random-edge controls ===")
    lines.append(f"{'Class':<10}{'BaseAcc':>10}{'TopAblAcc':>12}{'TopDrop':>10}{'RandDropMean':>14}{'RandDropSD':>12}{'p(perm)':>10}")
    lines.append("-" * 90)

    ablation_summary = {}
    for c in range(4):
        top_pairs = set(frozenset(p) for p in top_pairs_per_class[c])

        base_correct, base_total = 0, 0
        top_correct, top_total = 0, 0
        rand_drops = []

        # base + top-ablation accuracy, restricted to this class's true-label samples, pooled over folds
        for fold, model, test_loader in all_models:
            y_true, y_pred = evaluate_with_edge_ablation(model, test_loader, device, set())
            mask = y_true == c
            base_correct += (y_pred[mask] == c).sum()
            base_total += mask.sum()

            y_true2, y_pred2 = evaluate_with_edge_ablation(model, test_loader, device, top_pairs)
            top_correct += (y_pred2[mask] == c).sum()
            top_total += mask.sum()

        base_acc = base_correct / base_total
        top_acc = top_correct / top_total
        top_drop = base_acc - top_acc

        # random-edge control: N_RANDOM_CONTROL repeats of K random pairs, pooled over folds (subsample folds for speed: use fold 1 only for the null, matching the fold-1-only scope used elsewhere for expensive repeated runs)
        fold1_model = all_models[0][1]
        fold1_loader = all_models[0][2]
        y_true_f1, y_pred_f1 = evaluate_with_edge_ablation(fold1_model, fold1_loader, device, set())
        mask_f1 = y_true_f1 == c
        base_acc_f1 = (y_pred_f1[mask_f1] == c).mean()

        all_possible_pairs = [frozenset({i, j}) for i in range(N_NODES) for j in range(i + 1, N_NODES)]
        for _ in range(N_RANDOM_CONTROL):
            rand_pairs = set(rng.choice(len(all_possible_pairs), size=TOP_K, replace=False))
            rand_pairs = set(all_possible_pairs[i] for i in rand_pairs)
            _, y_pred_rand = evaluate_with_edge_ablation(fold1_model, fold1_loader, device, rand_pairs)
            rand_acc = (y_pred_rand[mask_f1] == c).mean()
            rand_drops.append(base_acc_f1 - rand_acc)

        rand_drops = np.array(rand_drops)
        top_drop_f1 = base_acc_f1 - (
            evaluate_with_edge_ablation(fold1_model, fold1_loader, device, top_pairs)[1][mask_f1] == c
        ).mean()
        p_perm = (np.sum(rand_drops >= top_drop_f1) + 1) / (N_RANDOM_CONTROL + 1)

        ablation_summary[CLASS_NAMES[c]] = {
            'base_acc_10fold': float(base_acc), 'top_ablated_acc_10fold': float(top_acc),
            'top_drop_10fold': float(top_drop),
            'fold1_base_acc': float(base_acc_f1), 'fold1_top_drop': float(top_drop_f1),
            'random_drop_mean': float(rand_drops.mean()), 'random_drop_sd': float(rand_drops.std()),
            'p_perm': float(p_perm),
        }
        sig = "*" if p_perm < 0.05 else ""
        lines.append(f"{CLASS_NAMES[c]:<10}{base_acc:>10.4f}{top_acc:>12.4f}{top_drop:>+10.4f}"
                     f"{rand_drops.mean():>+14.4f}{rand_drops.std():>12.4f}{p_perm:>10.4g} {sig}")
        print(f"  {CLASS_NAMES[c]}: 10-fold top-drop={top_drop:+.4f}, fold1 top-drop={top_drop_f1:+.4f}, "
              f"fold1 random-drop={rand_drops.mean():+.4f}+/-{rand_drops.std():.4f}, p={p_perm:.4g}")

    lines.append("\nNote: 10-fold columns (BaseAcc/TopAblAcc/TopDrop) use all 10 folds' test sets for the")
    lines.append("point estimate. The random-edge null (RandDropMean/SD, p) uses fold 1 only (30 repeats)")
    lines.append("for compute reasons -- same fold-1-only scoping as the seed-stability / permutation-null")
    lines.append("checks. 'Convergent and significant' (per reviewer #2) = top-attention-edge ablation drop")
    lines.append("exceeds the random-edge null at p<0.05.")

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(OUT_DIR, 'attention_rollout_causal_ablation_result.txt'), 'w') as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, 'causal_ablation_summary.json'), 'w') as f:
        json.dump(ablation_summary, f, indent=2)
    with open(os.path.join(OUT_DIR, 'top_channel_pairs_per_class.json'), 'w') as f:
        json.dump({CLASS_NAMES[c]: [[CHANNELS[i], CHANNELS[j]] for i, j in top_pairs_per_class[c]]
                   for c in range(4)}, f, indent=2)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'attention_rollout_causal_ablation_result.txt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'attention_rollout_class_matrices.npz')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'causal_ablation_summary.json')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'top_channel_pairs_per_class.json')}")
