#!/bin/bash
cd /data/xinyua11/robocasa/gradient_analysis/llm_borrow
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYOPENGL_PLATFORM
while pgrep -f "[j]obsB.json" > /dev/null; do sleep 30; done
/data/xinyua11/conda/envs/openpi/bin/python -u loss_eval.py --jobs jobsQ.json --sets setsB.json --out_dir lossmat --k 8 --draws 2 --dtype float16
echo "RUNQ COMPLETE"
