#!/bin/bash
# Sequential CPU chain: topology controls on folds 2 and 3, then window-level GAT folds 2 and 3.
export PYTHONIOENCODING=utf-8
for f in 2 3; do
  for c in hybrid graph_only flat_only edge_const edge_shuffle; do
    python r2_gat_controls_cpu.py --cond $c --folds $f --epochs 30 --patience 6 --batch 64 --threads 6 >> results/r2_${c}_fold${f}.log 2>&1
  done
done
for f in 2 3; do
  python r2_gat_controls_cpu.py --cond hybrid --split window --folds $f --epochs 30 --patience 6 --batch 64 --threads 6 >> results/r2_window_hybrid_fold${f}.log 2>&1
done
echo CHAIN_DONE >> results/r2_chain.log
