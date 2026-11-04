"""R2. Graph-topology controls for the hybrid GAT, runnable on CPU.

Conditions (identical folds, hyperparameters, seeds):
  hybrid        reference, retrained under this CPU protocol so comparisons are like-for-like
  graph_only    flat-FC branch input zeroed (message passing alone)
  edge_shuffle  edge-attribute vectors permuted across the 342 directed edges within each graph,
                at train and test time (topology destroyed, value distribution kept)
  edge_const    all edge attributes = 1 (attention from node features only)
  flat_only     node features zeroed (flat-FC branch alone)
Usage: python r2_gat_controls_cpu.py --cond hybrid --folds 1 2 3 [--epochs 60] [--time]
Outputs: results/r2_gat_controls/<cond>/fold_k/{y_true,y_pred,y_score}.npy, metrics.json
"""
import os, sys, json, time, argparse, numpy as np, torch, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from common import ROOT, OUT, load_splits
sys.path.insert(0, os.path.join(ROOT, "patient_split"))
from GAT_4class_patient_hybrid_auc import GATHybridModel, compute_class_weights
from DataLoader_4class_patient_hybrid import GraphLoader

p = argparse.ArgumentParser(); p.add_argument("--cond", required=True, choices=["hybrid", "graph_only", "edge_shuffle", "edge_const", "flat_only"])
p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3]); p.add_argument("--epochs", type=int, default=60); p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch", type=int, default=32); p.add_argument("--time", action="store_true"); p.add_argument("--threads", type=int, default=6)
p.add_argument("--split", choices=["patient", "window"], default="patient", help="window = leaky stratified folds from results/window_folds.json (built by r5_window_split.py)")
args = p.parse_args(); torch.set_num_threads(args.threads)
CACHE = os.path.join(OUT, "graph_cache.pt"); E = 342

def build_cache():
    ds = GraphLoader(os.path.join(ROOT, "psd_sb"), os.path.join(ROOT, "fc_sb"), ["corr", "coh", "aec", "plv"], os.path.join(ROOT, "train.csv"), 42)
    splits = load_splits(); need = set(f for s in splits for part in ("train", "val", "test") for f in s[part]["files"])
    assert need == set(ds.file_names), "loader file selection differs from stored split"
    items = {}
    for i, f in enumerate(ds.file_names):
        d = ds.get(i); assert d.edge_attr.shape[0] == E; items[f] = d
        if i % 5000 == 0: print("cached", i, flush=True)
    torch.save(items, CACHE); return items

items = torch.load(CACHE, weights_only=False) if os.path.exists(CACHE) else build_cache()
_pj = os.path.join(ROOT, "patient_split", "optuna_best_params_hybrid_auc.json")
if os.path.exists(_pj): params = json.load(open(_pj))["params"]
else:  # handover ships hyperparameters.txt instead of the optuna json
    params = {}
    for line in open(os.path.join(ROOT, "patient_split", "result_4class_patient_hybrid_auc", "hyperparameters.txt")):
        k, _, v = line.partition(":")
        if k.strip() in ("hidden_channels", "num_heads", "num_layers", "fc_hidden"): params[k.strip()] = int(v)
        elif k.strip() in ("dropout", "lr", "weight_decay"): params[k.strip()] = float(v)
print("params", params)
splits = load_splits(); raw = {"SEIZURE": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 3, "OTHER": 3}

def apply_cond(data, cond, gen):
    ea, ff, x = data.edge_attr, data.fc_flat, data.x
    if cond == "edge_shuffle":
        n = ea.shape[0] // E; perm = torch.cat([torch.randperm(E, generator=gen) + i * E for i in range(n)]); ea = ea[perm]
    elif cond == "edge_const": ea = torch.ones_like(ea)
    elif cond == "graph_only": ff = torch.zeros_like(ff)
    elif cond == "flat_only": x = torch.zeros_like(x)
    return x, ea, ff

def run_epoch(model, loader, opt, cw, cond, gen, train=True):
    model.train(train); tot = 0; ys, ps = [], []
    for data in loader:
        x, ea, ff = apply_cond(data, cond, gen)
        with torch.set_grad_enabled(train):
            out = model(x, data.edge_index, ea, data.batch, ff)
            if train:
                loss = F.nll_loss(out, data.y, weight=cw); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += loss.item()
        ys.append(data.y.numpy()); ps.append(out.exp().detach().numpy())
    y = np.concatenate(ys); pr = np.concatenate(ps); return tot / max(len(loader), 1), y, pr

outdir = os.path.join(OUT, "r2_gat_controls" if args.split == "patient" else "r2_gat_window_split", args.cond); os.makedirs(outdir, exist_ok=True)
if args.split == "window":
    W = json.load(open(os.path.join(OUT, "window_folds.json"))); lab = {f: int(items[f].y) for f in items}
    splits = [{"train": {"files": w["train"], "labels_str": None}, "val": {"files": w["val"]}, "test": {"files": w["test"]}} for w in W]
for k in args.folds:
    s = splits[k - 1]; torch.manual_seed(42 + k); gen = torch.Generator().manual_seed(1000 + k)
    tr = [items[f] for f in s["train"]["files"]]; va = [items[f] for f in s["val"]["files"]]; te = [items[f] for f in s["test"]["files"]]
    ytr_int = np.array([raw[l] for l in s["train"]["labels_str"]]) if s["train"]["labels_str"] is not None else np.array([int(d.y) for d in tr])
    cw = compute_class_weights(ytr_int)
    model = GATHybridModel(101, params["hidden_channels"], 4, params["num_heads"], params["num_layers"], params["dropout"], 4 * 361, params["fc_hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]); sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True); val = DataLoader(va, batch_size=256); tel = DataLoader(te, batch_size=256)
    if args.time:
        t0 = time.time(); n = 0
        for data in trl:
            x, ea, ff = apply_cond(data, args.cond, gen); out = model(x, data.edge_index, ea, data.batch, ff); loss = F.nll_loss(out, data.y, weight=cw); opt.zero_grad(); loss.backward(); opt.step(); n += 1
            if n == 60: break
        dt = (time.time() - t0) / n; print(f"{dt:.3f} s/batch -> {dt * len(trl) / 60:.1f} min/epoch ({len(trl)} batches)"); sys.exit()
    fd = os.path.join(outdir, f"fold_{k}"); os.makedirs(fd, exist_ok=True); best, wait, hist = -1, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); loss, yv_, _ = run_epoch(model, trl, opt, cw, args.cond, gen, True)
        _, yv, pv = run_epoch(model, val, opt, cw, args.cond, gen, False); auc = roc_auc_score(yv, pv, multi_class="ovr", average="macro"); sched.step(auc)
        hist.append({"epoch": ep, "loss": loss, "val_auc": float(auc), "sec": time.time() - t0}); print(f"[{args.cond}] fold {k} ep {ep} loss {loss:.4f} val_auc {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best: best, wait = auc, 0; torch.save(model.state_dict(), os.path.join(fd, "best_model.pth"))
        else:
            wait += 1
            if wait >= args.patience: break
    model.load_state_dict(torch.load(os.path.join(fd, "best_model.pth"))); _, yt, pt = run_epoch(model, tel, opt, cw, args.cond, gen, False)
    np.save(os.path.join(fd, "y_true.npy"), yt); np.save(os.path.join(fd, "y_pred.npy"), pt.argmax(1)); np.save(os.path.join(fd, "y_score.npy"), pt)
    m = {"fold": k, "best_val_auc": float(best), "test_macro_auc": float(roc_auc_score(yt, pt, multi_class="ovr", average="macro")), "test_acc": float((pt.argmax(1) == yt).mean()), "epochs": len(hist), "history": hist}
    json.dump(m, open(os.path.join(fd, "metrics.json"), "w"), indent=1); print(f"[{args.cond}] fold {k} TEST macroAUC {m['test_macro_auc']:.4f} acc {m['test_acc']:.4f}", flush=True)
