#!/bin/bash
export PYTHONIOENCODING=utf-8 FIGDPI=600
for s in fig_architecture_v2.py fig_design_v2.py fig_leakage_new.py fig_f3_full.py fig_f4_v2.py fig_f6_v3.py v5_figures_and_stats.py v7_review_items.py fig_extra_v6.py; do
  echo "=== $s"; python $s > results/hires_${s%.py}.log 2>&1 || echo "FAILED $s"
done
echo HIRES_DONE
