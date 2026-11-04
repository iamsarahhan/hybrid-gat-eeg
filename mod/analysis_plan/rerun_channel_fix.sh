#!/bin/bash
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
for s in a6_*.py a4_*.py a5_*.py r4_compare_all.py v7_review_items.py v5_figures_and_stats.py fig_design_v2.py fig_f4_v2.py fig_f3_full.py fig_f6_v3.py fig_extra_v6.py fig_architecture_v2.py; do
  echo "=== $s $(date)"; python $s > results/rerun_${s%.py}.log 2>&1 || echo "FAILED $s"
done
echo RERUN1_DONE
