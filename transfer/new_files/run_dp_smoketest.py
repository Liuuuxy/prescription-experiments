"""Single-task smoke test for the RoboCasa Diffusion Policy checkpoint.

Drives eval_robocasa.eval_task for ONE task (PickPlaceCounterToSink) with a
couple of rollouts, instead of the full task-set loop. Verifies a real trained
DP model loads and runs on this machine. Run with MUJOCO_GL=egl.
"""
import argparse
import os
from eval_robocasa import eval_task

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("-t", "--task", default="PickPlaceCounterToSink")
    p.add_argument("-s", "--split", default="pretrain")
    p.add_argument("-n", "--num_rollouts", type=int, default=2)
    p.add_argument("-e", "--num_envs", type=int, default=1)
    p.add_argument("-o", "--output_dir", default=None)
    p.add_argument("-d", "--device", default="cuda:0")
    args = p.parse_args()

    out = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "smoketest_evals"
    )
    print(f">>> DP smoke test: task={args.task} split={args.split} "
          f"n={args.num_rollouts} envs={args.num_envs}")
    eval_task(
        checkpoint=args.checkpoint,
        base_output_dir=out,
        device=args.device,
        task=args.task,
        num_rollouts=args.num_rollouts,
        num_envs=args.num_envs,
        split=args.split,
        overwrite=True,
    )
    print(f">>> done. results under {out}")
