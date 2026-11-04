"""
Retrain GAT hybrid (GATConv graph branch + flat-FC MLP branch, edge_dim=4,
class-weighted loss) on all 10 patient-grouped folds using optuna AUC-tuned
hyperparameters (optimization_patient_hybrid_auc.py ->
optuna_best_params_hybrid_auc.json).
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torchmetrics import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, roc_auc_score
from tqdm import tqdm
from DataLoader_4class_patient_hybrid import GraphLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


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
        out = self.lin2(combined)
        return F.log_softmax(out, dim=1)


def compute_class_weights(labels, num_classes=4):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1
    weights = counts.sum() / (num_classes * counts)
    return torch.FloatTensor(weights)


def train_epoch(model, train_loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.fc_flat)
        loss = F.nll_loss(out, data.y, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def compute_val_auc(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch, data.fc_flat)
            y_true.extend(data.y.cpu().numpy())
            y_score.extend(F.softmax(output, dim=1).cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')


def compute_metrics(model, loader, device, num_classes=4):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for data in tqdm(loader, desc="Computing metrics", leave=False):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch, data.fc_flat)
            pred = output.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_score.extend(F.softmax(output, dim=1).cpu().numpy())
            confmat.update(pred, data.y)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    confusion_matrix = confmat.compute().cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    class_names = ['SEIZURE', 'LPD', 'GPD', 'OTHERS']
    auc_scores = {}
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_score[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        auc_scores[class_names[i]] = auc(fpr, tpr)

    class_metrics = {}
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = confusion_matrix.sum() - (tp + fp + fn)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_c = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        class_metrics[class_names[i]] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision_c,
            'f1': f1_c,
            'auc': auc_scores[class_names[i]]
        }

    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    macro_metrics = {
        'sensitivity': np.mean([m['sensitivity'] for m in class_metrics.values()]),
        'specificity': np.mean([m['specificity'] for m in class_metrics.values()]),
        'precision': np.mean([m['precision'] for m in class_metrics.values()]),
        'f1': np.mean([m['f1'] for m in class_metrics.values()]),
        'auc': np.mean([m['auc'] for m in class_metrics.values()])
    }

    return {
        'accuracy': accuracy,
        'macro_metrics': macro_metrics,
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score,
    }


if __name__ == "__main__":
    result_dir = os.path.join(SCRIPT_DIR, 'result_4class_patient_hybrid_auc')
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n결과 저장 경로: {result_dir}")

    psd_folders = [os.path.join(REPO_ROOT, 'psd_sb')]
    fc_folders = [os.path.join(REPO_ROOT, 'fc_sb')]
    fc_types = ['corr', 'coh', 'aec', 'plv']
    label_file = os.path.join(REPO_ROOT, 'train.csv')
    raw_to_int = {'SEIZURE': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 3, 'OTHER': 3, 'OTHERS': 3}
    fc_in_dim = len(fc_types) * 19 * 19

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(os.path.join(SCRIPT_DIR, 'optuna_best_params_hybrid_auc.json')) as f:
        best = json.load(f)
    params = best['params']
    print(f"Using optuna params (val_auc={best['best_val_auc']:.4f}): {params}")

    with open(os.path.join(result_dir, 'hyperparameters.txt'), 'w') as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"source: optimization_patient_hybrid_auc.py / optuna_best_params_hybrid_auc.json (val_auc={best['best_val_auc']:.4f})\n")
        f.write("architecture: GATConv (edge_dim=4) graph branch + flat-FC MLP branch, concat -> classifier\n")
        f.write("loss: class-weighted NLL (balanced, computed per-fold from train split)\n")
        f.write("split: patient-grouped StratifiedGroupKFold (patient_id, no leakage)\n")

    split_path = os.path.join(SCRIPT_DIR, 'fold_splits_patient.json')
    fold_datasets = GraphLoader.create_data_loaders(
        psd_folders, fc_folders, fc_types, batch_size=32,
        label_file=label_file, random_state=42,
        save_split=False, split_save_path=split_path
    )

    with open(split_path) as f:
        fold_split_info = json.load(f)

    fold_results = []
    for fold, (train_loader, val_loader, test_loader) in enumerate(fold_datasets, 1):
        print(f"\nFold {fold}/10")
        torch.manual_seed(42 + fold)
        torch.cuda.manual_seed_all(42 + fold)

        train_labels = np.array([raw_to_int[l] for l in fold_split_info[fold - 1]['train']['labels_str']])
        class_weights = compute_class_weights(train_labels).to(device)

        model = GATHybridModel(in_channels=101, hidden_channels=params['hidden_channels'], out_channels=4,
                                num_heads=params['num_heads'], num_layers=params['num_layers'],
                                dropout=params['dropout'], fc_in_dim=fc_in_dim,
                                fc_hidden=params['fc_hidden']).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_val_auc = 0
        patience = 15
        counter = 0

        fold_dir = os.path.join(result_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in tqdm(range(1, 120), desc="Epochs", leave=False):
            loss = train_epoch(model, train_loader, optimizer, device, class_weights)
            val_auc = compute_val_auc(model, val_loader, device)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                counter = 0
                model_path = os.path.join(fold_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'loss': loss,
                }, model_path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered. Best val AUC: {best_val_auc:.4f}")
                    break

        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        fold_metrics = compute_metrics(model, test_loader, device)
        fold_results.append(fold_metrics)

        np.save(os.path.join(fold_dir, 'y_true.npy'), fold_metrics['y_true'])
        np.save(os.path.join(fold_dir, 'y_pred.npy'), fold_metrics['y_pred'])
        np.save(os.path.join(fold_dir, 'y_score.npy'), fold_metrics['y_score'])

        metrics_path = os.path.join(fold_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"=== Fold {fold} Performance Report (patient-grouped, hybrid GAT+flatFC) ===\n\n")
            f.write(f"Overall Accuracy: {fold_metrics['accuracy']:.4f}\n\n")
            f.write("=== Macro-averaged Metrics ===\n")
            for metric_name, value in fold_metrics['macro_metrics'].items():
                f.write(f"{metric_name.capitalize()}: {value:.4f}\n")
            f.write("\n=== Class-wise Performance ===\n")
            for class_name, metrics in fold_metrics['class_metrics'].items():
                f.write(f"\n{class_name}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name.capitalize()}: {value:.4f}\n")
            f.write("\n=== Confusion Matrix ===\n")
            np.savetxt(f, fold_metrics['confusion_matrix'], fmt='%-8d')

        print(f"\n=== Fold {fold} Performance Report ===")
        print(f"Overall Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"Macro AUC: {fold_metrics['macro_metrics']['auc']:.4f}")

    avg_results_path = os.path.join(result_dir, 'average_results.txt')
    with open(avg_results_path, 'w') as f:
        f.write("=== Average Performance Across All Folds (patient-grouped, hybrid GAT+flatFC) ===\n\n")
        avg_accuracy = np.mean([fold['accuracy'] for fold in fold_results])
        f.write(f"Overall Average Accuracy: {avg_accuracy:.4f}\n\n")
        f.write("=== Macro-averaged Metrics ===\n")
        for metric in ['sensitivity', 'specificity', 'precision', 'f1', 'auc']:
            value = np.mean([fold['macro_metrics'][metric] for fold in fold_results])
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
        f.write("\n=== Class-wise Average Performance ===\n")
        class_names = ['SEIZURE', 'LPD', 'GPD', 'OTHERS']
        for class_name in class_names:
            f.write(f"\n{class_name}:\n")
            for metric in ['sensitivity', 'specificity', 'precision', 'f1', 'auc']:
                avg_value = np.mean([fold['class_metrics'][class_name][metric] for fold in fold_results])
                f.write(f"{metric.capitalize()}: {avg_value:.4f}\n")
        f.write("\n=== Average Confusion Matrix ===\n")
        avg_conf_matrix = np.mean([fold['confusion_matrix'] for fold in fold_results], axis=0)
        np.savetxt(f, avg_conf_matrix, fmt='%.2f')

    print("\n=== Average Performance Across All Folds ===")
    print(f"Overall Accuracy: {avg_accuracy:.4f}")
    print(f"\n모든 결과가 {result_dir} 폴더에 저장되었습니다.")
