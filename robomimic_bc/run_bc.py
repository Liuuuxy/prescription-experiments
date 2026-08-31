"""Wrapper: patch the fork's PrintLogger for newer transformers, then run train.py."""
import sys, runpy
import robomimic.utils.log_utils as LU
LU.PrintLogger.isatty = lambda self: False
LU.PrintLogger.fileno = lambda self: 1
sys.argv = ["train.py", "--config", "/data/xinyua11/robomimic_runs/bc_lift_config.json"]
runpy.run_path("/data/xinyua11/robomimic/robomimic/scripts/train.py", run_name="__main__")
