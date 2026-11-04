"""
R2-Minor-3: FC-only vs PSD-only ablation to isolate where the classification
signal lives, reimplemented for the final GAT hybrid architecture (GATConv
edge_dim=4 graph branch + flat-FC branch, patient_split/
GAT_4class_patient_hybrid_auc.py). Supersedes review/analysis/
fc_only_psd_only_ablation.py, which targeted the old pre-edge_dim-fix GAT
(edge_attr was silently ignored there, so that script's "fc_only" condition
was a topology-only manipulation, not a real FC-signal isolation).

In the current hybrid model, FC information enters through two places:
edge_attr (graph branch) and fc_flat (flat-FC branch). PSD enters only as
node features x (graph branch). So, unlike the branch ablation
(review/analysis for ablation_hybrid_branches.py, which zeroes whole
branches), this ablation zeroes by INFORMATION SOURCE across both branches:

  - psd_only : real PSD (x), FC zeroed everywhere (edge_attr=0, fc_flat=0)
  - fc_only  : PSD zeroed (x=0), real FC everywhere (edge_attr, fc_flat unchanged)
  - original : everything real (sanity check -- should match the reported
               10-fold average from result_4class_patient_hybrid_auc)

No retraining -- uses existing trained checkpoints (best_model.pth per fold)
via inference only, on the same patient-grouped test splits.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATIENT_SPLIT_DIR = os.path.join(REPO_ROOT, 'patient_split')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, PATIENT_SPLIT_DIR)
from DataLoader_4class_patient_hybrid import GraphLoader


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

    def forward(self, x, edge_index, edge_attr, batch, fc_flat):
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index, edge_attr))
        graph_embed = global_mean_pool(x, batch)
        fc_embed = self.fc_branch(fc_flat)
        combined = torch.cat([graph_embed, fc_embed], dim=1)
        combined = F.dropout(combined, p=self.dropout, training=self.training)
        combined = F.elu(self.lin1(combined))
        return F.log_softmax(self.lin2(combined), dim=1)


@torch.no_grad()
def evaluate(model, loader, device, condition):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for data in loader:
        data = data.to(device)
        x = data.x
        edge_attr = data.edge_attr
        fc_flat = data.fc_flat
        if condition == 'psd_only':
            edge_attr = torch.zeros_like(edge_attr)
            fc_flat = torch.zeros_like(fc_flat)
        elif condition == 'fc_only':
            x = torch.zeros_like(x)
        output = model(x, data.edge_index, edge_attr, data.batch, fc_flat)
        pred = output.argmax(dim=1)
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_score.extend(F.softmax(output, dim=1).cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def compute_macro_auc(y_true, y_score, n_classes=4):
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    aucs = []
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_score[:, c])
        aucs.append(auc(fpr, tpr))
    return np.mean(aucs)


if __name__ == "__main__":
    result_dir = os.path.join(PATIENT_SPLIT_DIR, 'result_4class_patient_hybrid_auc')
    psd_folders = [os.path.join(REPO_ROOT, 'psd_sb')]
    fc_folders = [os.path.join(REPO_ROOT, 'fc_sb')]
    fc_types = ['corr', 'coh', 'aec', 'plv']
    label_file = os.path.join(REPO_ROOT, 'train.csv')
    fc_in_dim = len(fc_types) * 19 * 19

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PATIENT_SPLIT_DIR, 'optuna_best_params_hybrid_auc.json')) as f:
        params = json.load(f)['params']
    print(f"Model architecture params: {params}")

    split_path = os.path.join(PATIENT_SPLIT_DIR, 'fold_splits_patient.json')
    fold_datasets = GraphLoader.create_data_loaders(
        psd_folders, fc_folders, fc_types, batch_size=32,
        label_file=label_file, random_state=42,
        save_split=False, split_save_path=split_path
    )

    conditions = ['original', 'psd_only', 'fc_only']
    results = {c: {'acc': [], 'macro_auc': []} for c in conditions}

    for fold, (_train_loader, _val_loader, test_loader) in enumerate(fold_datasets, 1):
        fold_dir = os.path.join(result_dir, f'fold_{fold}')
        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pth'), map_location=device)

        model = GATHybridModel(in_channels=101, hidden_channels=params['hidden_channels'], out_channels=4,
                                num_heads=params['num_heads'], num_layers=params['num_layers'],
                                dropout=params['dropout'], fc_in_dim=fc_in_dim,
                                fc_hidden=params['fc_hidden']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        row = []
        for c in conditions:
            y_true, y_pred, y_score = evaluate(model, test_loader, device, c)
            acc = accuracy_score(y_true, y_pred)
            macro_auc = compute_macro_auc(y_true, y_score)
            results[c]['acc'].append(acc)
            results[c]['macro_auc'].append(macro_auc)
            row.append(f"{c}={acc:.4f}/{macro_auc:.4f}")
        print(f"Fold {fold}: " + "  ".join(row))

    lines = ["=== R2-Minor-3: FC-only vs PSD-only ablation, GAT hybrid (10-fold, inference-only) ===\n"]
    lines.append(f"{'Condition':<12}{'Mean acc':>12}{'SD':>8}{'Mean macro AUC':>18}{'SD':>8}   Description")
    lines.append("-" * 100)
    descr = {
        "original": "real PSD + real edge_attr + real flat-FC (sanity check vs Table 2)",
        "psd_only": "real PSD node features; edge_attr and flat-FC both zeroed",
        "fc_only":  "PSD node features zeroed; edge_attr and flat-FC both real",
    }
    for c in conditions:
        accs = np.array(results[c]['acc'])
        aucs = np.array(results[c]['macro_auc'])
        lines.append(f"{c:<12}{accs.mean():>12.4f}{accs.std():>8.4f}{aucs.mean():>18.4f}{aucs.std():>8.4f}   {descr[c]}")

    lines.append("\nPer-fold accuracy / macro AUC:")
    for c in conditions:
        accs = results[c]['acc']
        aucs = results[c]['macro_auc']
        lines.append(f"  {c:<10}: acc=" + ", ".join(f"{a:.4f}" for a in accs))
        lines.append(f"  {'':<10}  auc=" + ", ".join(f"{a:.4f}" for a in aucs))

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(OUT_DIR, 'fc_only_psd_only_ablation_result.txt'), 'w') as f:
        f.write(report + "\n")
    print(f"\nSaved: {os.path.join(OUT_DIR, 'fc_only_psd_only_ablation_result.txt')}")
