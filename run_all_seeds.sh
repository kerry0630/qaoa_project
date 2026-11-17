#!/usr/bin/env bash
# 重複驗證：3 個獨立主種子（不同的問題實例 + 不同的優化初值）
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  python scripts/benchmark.py --seed $s --sizes 8 10 12 --instances 8 --inits 4 > results/log_seed$s.txt 2>&1
done
echo ALLDONE > results/ALLDONE
