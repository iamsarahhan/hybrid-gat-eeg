"""S1 (server, GPU). Isolate the contribution of graph topology in the hybrid GAT.

Three retraining conditions, identical folds and hyperparameters to
patient_split/GAT_4class_patient_hybrid_auc.py:
  --cond graph_only      flat-FC branch removed (fc_hidden=0): message passing alone
  --cond edge_shuffle    every clip's 171 edge-attribute vectors randomly permuted across
                         pairs (topology destroyed, value distribution kept)
  --cond edge_const      all edge attributes set to 1 (attention from node features only)
Run from patient_split/:   python s1_server_graph_only_edge_shuffle.py --cond graph_only
Outputs: result_4class_patient_{cond}_auc/ in the same format; then run
mod/analysis_patient_grouped/delong_correct.py-style comparison against the hybrid.
Interpretation:
  hybrid >> graph_only ~ flat_only  -> the gain is complementarity, not message passing
  hybrid ~ graph_only >> edge_shuffle -> topology carries information (the paper's claim)
  edge_shuffle ~ hybrid               -> edges are used only as a bag of values
"""
import argparse, os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GAT_4class_patient_hybrid_auc import GATHybridModel, compute_class_weights, train_epoch, compute_val_auc, compute_metrics, SCRIPT_DIR, REPO_ROOT
from DataLoader_4class_patient_hybrid import GraphLoader

p = argparse.ArgumentParser(); p.add_argument("--cond", choices=["graph_only", "edge_shuffle", "edge_const"], required=True); args = p.parse_args()

class Wrapped(torch.nn.Module):
    """Applies the condition at the batch level so the stored loader is unchanged."""
    def __init__(self, base, cond): super().__init__(); self.base, self.cond = base, cond
    def forward(self, x, edge_index, edge_attr, batch, fc_flat):
        if self.cond == "edge_shuffle":
            # permute edge rows within each graph: graphs have identical edge counts (342), so a block permutation works
            E = 342; n = edge_attr.shape[0] // E
            perm = torch.cat([torch.randperm(E, device=edge_attr.device) + i * E for i in range(n)]); edge_attr = edge_attr[perm]
        elif self.cond == "edge_const": edge_attr = torch.ones_like(edge_attr)
        elif self.cond == "graph_only": fc_flat = torch.zeros_like(fc_flat)
        return self.base(x, edge_index, edge_attr, batch, fc_flat)

if __name__ == "__main__":
    result_dir = os.path.join(SCRIPT_DIR, f"result_4class_patient_{args.cond}_auc"); os.makedirs(result_dir, exist_ok=True)
    params = json.load(open(os.path.join(SCRIPT_DIR, "optuna_best_params_hybrid_auc.json")))["params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); fc_types = ["corr", "coh", "aec", "plv"]
    folds = GraphLoader.create_data_loaders([os.path.join(REPO_ROOT, "psd_sb")], [os.path.join(REPO_ROOT, "fc_sb")], fc_types, batch_size=32,
                                            label_file=os.path.join(REPO_ROOT, "train.csv"), random_state=42, save_split=False)
    split = json.load(open(os.path.join(SCRIPT_DIR, "fold_splits_patient.json"))); raw = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 3, "OTHER": 3}
    for k, (tr, va, te) in enumerate(folds, 1):
        torch.manual_seed(42 + k); cw = compute_class_weights(np.array([raw[l] for l in split[k - 1]["train"]["labels_str"]])).to(device)
        base = GATHybridModel(101, params["hidden_channels"], 4, params["num_heads"], params["num_layers"], params["dropout"], 4 * 361, params["fc_hidden"])
        model = Wrapped(base, args.cond).to(device); opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        best, wait, fd = 0, 0, os.path.join(result_dir, f"fold_{k}"); os.makedirs(fd, exist_ok=True)
        for ep in range(1, 120):
            train_epoch(model, tr, opt, device, cw); auc = compute_val_auc(model, va, device)
            if auc > best: best, wait = auc, 0; torch.save({"model_state_dict": model.state_dict(), "val_auc": auc, "epoch": ep}, os.path.join(fd, "best_model.pth"))
            else:
                wait += 1
                if wait >= 15: break
        model.load_state_dict(torch.load(os.path.join(fd, "best_model.pth"))["model_state_dict"]); m = compute_metrics(model, te, device)
        for n in ("y_true", "y_pred", "y_score"): np.save(os.path.join(fd, f"{n}.npy"), m[n])
        print(f"fold {k} {args.cond}: acc {m['accuracy']:.4f} macroAUC {m['macro_metrics']['auc']:.4f}")
