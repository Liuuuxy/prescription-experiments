#!/bin/bash
cd /data/xinyua11/robocasa/gradient_analysis/llm_borrow
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYOPENGL_PLATFORM
PY=/data/xinyua11/conda/envs/openpi/bin/python
$PY -u loss_eval.py --jobs jobsA_diag.json --sets setsA.json --out_dir lossmat --k 8 --draws 8
$PY -u loss_eval.py --jobs jobsA_pull.json --sets setsA.json --out_dir lossmat --k 8 --draws 4
echo "RUNA COMPLETE"
