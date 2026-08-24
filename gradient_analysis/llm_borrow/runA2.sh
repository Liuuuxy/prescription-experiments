#!/bin/bash
cd /data/xinyua11/robocasa/gradient_analysis/llm_borrow
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYOPENGL_PLATFORM
# wait for the diag process (started by runA.sh) to exit before touching the GPU again
while pgrep -f "[j]obsA_diag.json" > /dev/null; do sleep 20; done
/data/xinyua11/conda/envs/openpi/bin/python -u loss_eval.py --jobs jobsA_pull.json --sets setsA.json --out_dir lossmat --k 8 --draws 2
echo "RUNA2 COMPLETE"
