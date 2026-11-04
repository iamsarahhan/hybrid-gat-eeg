"""
Phase 3-3 + 3-4: seed-stability and label-permutation null for the GAT hybrid
model's graph-branch attention, per reviewer #2's R2-4.

Scope note (consistent with the causal-ablation script and stated to the
user up front): fold 1's train/val/test split is fixed; only the random
seed (weight init + minibatch order) or the training-label permutation
varies. This is standard practice for these checks -- the goal is not a new
10-fold performance estimate, it's whether attention is (a) reproducible
across random inits with the real task, and (b) actually driven by the
real labels rather than by data/graph structure alone.

  - Seed-stability: train fold 1, optuna-best hyperparameters, 5 different
    seeds, real labels. Compare the 5 runs' rollout attention matrices
    (per class) for pairwise similarity.
  - Label-permutation null: same, but training labels are permuted once
    (fixed for that run's whole training), 5 permutation seeds. Val/test
    labels stay real, so val-based early stopping and test accuracy serve
    as a sanity check that permutation actually destroyed the signal
    (test accuracy should collapse toward chance, ~0.25).

Uses GATHybridModel / rollout_batch / get_predictions_and_rollouts from
attention_rollout_and_causal_ablation.py (imported, not duplicated).
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATIENT_SPLIT_DIR = os.path.join(REPO_ROOT, 'patient_split')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATIENT_SPLIT_DIR)
sys.path.insert(0, OUT_DIR)
from DataLoader_4class_patient_hybrid import GraphLoader
from attention_rollout_and_causal_ablation import (
    GATHybridModel, rollout_batch, get_predictions_and_rollouts, N_NODES, CHANNELS, CLASS_NAMES, TOP_K
)

N_SEEDS = 5
SEED_BASE_REAL = 1000
SEED_BASE_PERM = 2000


class PermutedLabelDataset(torch.utils.data.Dataset):
    """Wraps a fold's train Subset, replacing each sample's label with a
    fixed (per-run) permutation of the true training-label array. Does not
    mutate the underlying shared GraphLoader/Subset."""
    def __init__(self, base_subset, permuted_labels):
        self.base_subset = base_subset
        self.permuted_labels = permuted_labels

    def __len__(self):
        return len(self.base_subset)

    def __getitem__(self, idx):
        data = self.base_subset[idx].clone()
        data.y = torch.LongTensor([self.permuted_labels[idx]])
        return data


def compute_class_weights(labels, num_classes=4):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1
    weights = counts.sum() / (num_classes * counts)
    return torch.FloatTensor(weights)


def train_one_run(seed, permute_labels, train_loader, val_loader, test_loader,
                   fold1_train_labels_int, params, fc_in_dim, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    if permute_labels:
        permuted = rng.permutation(fold1_train_labels_int)
        run_train_dataset = PermutedLabelDataset(train_loader.dataset, permuted)
        run_train_loader = PyGDataLoader(run_train_dataset, batch_size=32, shuffle=True, num_workers=4)
        class_labels_for_weights = permuted
    else:
        run_train_loader = train_loader
        class_labels_for_weights = fold1_train_labels_int

    class_weights = compute_class_weights(np.array(class_labels_for_weights)).to(device)

    model = GATHybridModel(in_channels=101, hidden_channels=params['hidden_channels'], out_channels=4,
                            num_heads=params['num_heads'], num_layers=params['num_layers'],
                            dropout=params['dropout'], fc_in_dim=fc_in_dim,
                            fc_hidden=params['fc_hidden']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_auc = 0
    patience = 15
    counter = 0
    best_state = None

    for epoch in range(1, 121):
        model.train()
        for data in run_train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.fc_flat)
            loss = F.nll_loss(out, data.y, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        y_true_val, y_score_val = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.fc_flat)
                y_true_val.extend(data.y.cpu().numpy())
                y_score_val.extend(F.softmax(out, dim=1).cpu().numpy())
        val_auc = roc_auc_score(y_true_val, y_score_val, multi_class='ovr', average='macro')
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    y_true_test, y_pred_test, rollout_test = get_predictions_and_rollouts(model, test_loader, device)
    test_acc = accuracy_score(y_true_test, y_pred_test)

    class_mean_rollout = {}
    for c in range(4):
        mask = y_true_test == c
        class_mean_rollout[c] = rollout_test[mask].mean(axis=0) if mask.sum() > 0 else np.zeros((N_NODES, N_NODES))

    return {
        'seed': seed, 'permute_labels': permute_labels,
        'best_val_auc': best_val_auc, 'test_acc': test_acc,
        'class_mean_rollout': class_mean_rollout,
    }


def top_k_pairs(matrix, k=TOP_K):
    M = matrix.copy()
    np.fill_diagonal(M, 0)
    sym = M + M.T
    pairs = []
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            pairs.append((i, j, sym[i, j]))
    pairs.sort(key=lambda t: -t[2])
    return set(frozenset({i, j}) for i, j, _ in pairs[:k])


def matrix_correlation(a, b):
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    psd_folders = [os.path.join(REPO_ROOT, 'psd_sb')]
    fc_folders = [os.path.join(REPO_ROOT, 'fc_sb')]
    fc_types = ['corr', 'coh', 'aec', 'plv']
    label_file = os.path.join(REPO_ROOT, 'train.csv')
    fc_in_dim = len(fc_types) * N_NODES * N_NODES
    raw_to_int = {'SEIZURE': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 3, 'OTHER': 3, 'OTHERS': 3}

    with open(os.path.join(PATIENT_SPLIT_DIR, 'optuna_best_params_hybrid_auc.json')) as f:
        params = json.load(f)['params']

    split_path = os.path.join(PATIENT_SPLIT_DIR, 'fold_splits_patient.json')
    fold_datasets = GraphLoader.create_data_loaders(
        psd_folders, fc_folders, fc_types, batch_size=32,
        label_file=label_file, random_state=42,
        save_split=False, split_save_path=split_path
    )
    train_loader, val_loader, test_loader = fold_datasets[0]  # fold 1 only

    with open(split_path) as f:
        fold1_info = json.load(f)[0]
    fold1_train_labels_int = [raw_to_int[l] for l in fold1_info['train']['labels_str']]
    assert len(fold1_train_labels_int) == len(train_loader.dataset)

    runs = []
    for i in range(N_SEEDS):
        seed = SEED_BASE_REAL + i
        print(f"\n[Seed-stability] training run {i+1}/{N_SEEDS}, seed={seed}, real labels...")
        r = train_one_run(seed, False, train_loader, val_loader, test_loader,
                           fold1_train_labels_int, params, fc_in_dim, device)
        print(f"  val_auc={r['best_val_auc']:.4f}  test_acc={r['test_acc']:.4f}")
        runs.append(r)

    perm_runs = []
    for i in range(N_SEEDS):
        seed = SEED_BASE_PERM + i
        print(f"\n[Permutation-null] training run {i+1}/{N_SEEDS}, seed={seed}, permuted train labels...")
        r = train_one_run(seed, True, train_loader, val_loader, test_loader,
                           fold1_train_labels_int, params, fc_in_dim, device)
        print(f"  val_auc={r['best_val_auc']:.4f}  test_acc={r['test_acc']:.4f}  (expect near chance ~0.25)")
        perm_runs.append(r)

    # ---- Seed-stability: pairwise rollout correlation across the 5 real-label runs ----
    lines = ["=== Phase 3-3: Seed-stability (fold 1, 5 seeds, real labels) ===\n"]
    lines.append(f"{'Seed':<8}{'Val AUC':>10}{'Test Acc':>10}")
    for r in runs:
        lines.append(f"{r['seed']:<8}{r['best_val_auc']:>10.4f}{r['test_acc']:>10.4f}")
    accs = [r['test_acc'] for r in runs]
    aucs = [r['best_val_auc'] for r in runs]
    lines.append(f"\nTest acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    lines.append(f"Val AUC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    lines.append("\nPairwise rollout-matrix correlation across seeds (per class, real labels):")
    seed_stability_summary = {}
    for c in range(4):
        corrs = []
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                corrs.append(matrix_correlation(runs[i]['class_mean_rollout'][c], runs[j]['class_mean_rollout'][c]))
        seed_stability_summary[CLASS_NAMES[c]] = {'mean_corr': float(np.mean(corrs)), 'sd_corr': float(np.std(corrs))}
        lines.append(f"  {CLASS_NAMES[c]:<10}: mean r={np.mean(corrs):+.4f}  sd={np.std(corrs):.4f}  (n={len(corrs)} pairs)")

    lines.append("\nTop-K channel-pair Jaccard overlap across seeds (per class, real labels):")
    for c in range(4):
        jaccards = []
        topk_sets = [top_k_pairs(r['class_mean_rollout'][c]) for r in runs]
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                inter = len(topk_sets[i] & topk_sets[j])
                union = len(topk_sets[i] | topk_sets[j])
                jaccards.append(inter / union if union > 0 else 0.0)
        seed_stability_summary[CLASS_NAMES[c]]['mean_jaccard'] = float(np.mean(jaccards))
        lines.append(f"  {CLASS_NAMES[c]:<10}: mean Jaccard={np.mean(jaccards):.4f}  (top-{TOP_K}, n={len(jaccards)} pairs)")

    # ---- Permutation null ----
    lines.append("\n\n=== Phase 3-4: Label-permutation null (fold 1, 5 permutations) ===\n")
    lines.append(f"{'Seed':<8}{'Val AUC':>10}{'Test Acc':>10}")
    for r in perm_runs:
        lines.append(f"{r['seed']:<8}{r['best_val_auc']:>10.4f}{r['test_acc']:>10.4f}")
    p_accs = [r['test_acc'] for r in perm_runs]
    p_aucs = [r['best_val_auc'] for r in perm_runs]
    lines.append(f"\nTest acc: {np.mean(p_accs):.4f} +/- {np.std(p_accs):.4f}  (chance = 0.25 for 4 balanced classes)")
    lines.append(f"Val AUC:  {np.mean(p_aucs):.4f} +/- {np.std(p_aucs):.4f}  (chance = 0.5)")

    real_mean_rollout = {c: np.mean([r['class_mean_rollout'][c] for r in runs], axis=0) for c in range(4)}
    perm_mean_rollout = {c: np.mean([r['class_mean_rollout'][c] for r in perm_runs], axis=0) for c in range(4)}

    lines.append("\nReal-label mean rollout vs permuted-label mean rollout (per class):")
    perm_comparison_summary = {}
    for c in range(4):
        corr = matrix_correlation(real_mean_rollout[c], perm_mean_rollout[c])
        real_topk = top_k_pairs(real_mean_rollout[c])
        perm_topk = top_k_pairs(perm_mean_rollout[c])
        inter = len(real_topk & perm_topk)
        union = len(real_topk | perm_topk)
        jac = inter / union if union > 0 else 0.0
        perm_comparison_summary[CLASS_NAMES[c]] = {'real_vs_perm_corr': float(corr), 'real_vs_perm_jaccard': float(jac)}
        lines.append(f"  {CLASS_NAMES[c]:<10}: r={corr:+.4f}  top-{TOP_K} Jaccard={jac:.4f}")

    lines.append("\nInterpretation guide:")
    lines.append("- If permuted-label test acc collapses to ~0.25, permutation worked as intended (sanity check).")
    lines.append("- High seed-stability correlation/Jaccard (real labels) = attention pattern is reproducible,")
    lines.append("  a necessary (not sufficient) condition for trusting it as interpretation.")
    lines.append("- High real-vs-permuted correlation/Jaccard = attention looks similar regardless of whether")
    lines.append("  labels are real or nonsense, i.e. driven by graph/data structure rather than the learned")
    lines.append("  task -- this would undermine using attention as a circuit-level explanation, independent")
    lines.append("  of the causal-ablation finding (attention_rollout_and_causal_ablation.py) that top-attention")
    lines.append("  edges are not causally load-bearing either.")

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(OUT_DIR, 'seed_stability_permutation_null_result.txt'), 'w') as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, 'seed_stability_permutation_null_summary.json'), 'w') as f:
        json.dump({
            'seed_stability': seed_stability_summary,
            'real_vs_permuted': perm_comparison_summary,
            'real_runs': [{'seed': r['seed'], 'val_auc': r['best_val_auc'], 'test_acc': r['test_acc']} for r in runs],
            'permuted_runs': [{'seed': r['seed'], 'val_auc': r['best_val_auc'], 'test_acc': r['test_acc']} for r in perm_runs],
        }, f, indent=2)

    np.savez(os.path.join(OUT_DIR, 'seed_stability_permutation_null_rollouts.npz'),
              **{f'real_{CLASS_NAMES[c]}': real_mean_rollout[c] for c in range(4)},
              **{f'perm_{CLASS_NAMES[c]}': perm_mean_rollout[c] for c in range(4)},
              channels=np.array(CHANNELS))

    print(f"\nSaved: {os.path.join(OUT_DIR, 'seed_stability_permutation_null_result.txt')}")
