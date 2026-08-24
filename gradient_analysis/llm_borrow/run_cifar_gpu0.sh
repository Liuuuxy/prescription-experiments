#!/bin/bash
# GPU0 chain: cheap (pull-based + jest) allocators first, then learning_progress_ucb.
set -x
cd /data/xinyua11/robocasa/gradient_analysis/llm_borrow
PY=/data/xinyua11/conda/envs/robocasa/bin/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
$PY part1_driver.py --mode cifar --budgets 1.5,3,6,12,24 --reps 5 \
    --allocators oracle,jest,jest_then_confirm,successive_halving,random,regmix \
    --out part1_cifar.json --cache wind_tunnel_cifar_cache.json --time-budget-s 9000
$PY part1_driver.py --mode cifar --budgets 1.5,3,6,12,24 --reps 5 \
    --allocators learning_progress_ucb \
    --out part1_cifar.json --cache wind_tunnel_cifar_cache.json --time-budget-s 9000
echo DONE_GPU0
