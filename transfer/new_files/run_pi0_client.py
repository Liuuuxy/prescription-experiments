"""Single-task pi0 client driver for the RoboCasa smoke test.

Connects to the pi0 websocket server (serve_policy.py) and runs a few rollouts
of one task (default PickPlaceCounterToSink), instead of looping a whole task
set. Run in the env that has robocasa + openpi_client, with MUJOCO_GL=egl.
"""
import argparse
import os
from main import eval_env

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--task", default="PickPlaceCounterToSink")
    p.add_argument("-s", "--split", default="pretrain")
    p.add_argument("-n", "--num_trials", type=int, default=2)
    p.add_argument("--resize_size", type=int, default=224)
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-o", "--log_dir",
                   default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/evals/pi0_pretrain_human300_75000")
    args = p.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    print(f">>> pi0 client: task={args.task} split={args.split} n={args.num_trials} "
          f"replan={args.replan_steps} -> server {args.host}:{args.port}")
    eval_env(args.task, args.split, args.log_dir, args.num_trials,
             args.resize_size, args.replan_steps, args.host, args.port, args.seed)
    print(f">>> pi0 client done. results under {args.log_dir}")
