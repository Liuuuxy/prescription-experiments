"""Single-task GR00T smoke test for RoboCasa.

Runs the GR00T server (in a thread) + client for ONE task
(PickPlaceCounterToSink) with a couple of rollouts, instead of looping a whole
task set. Injects a one-task entry into TASK_SET_REGISTRY. Run in groot_env
with MUJOCO_GL=egl.
"""
import argparse
import os
import threading
import time

from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

# inject a one-task "task set" so run_client evaluates only this task
TASK_SET_REGISTRY["_smoke_ppc2sink"] = ["PickPlaceCounterToSink"]

from run_eval import run_server, run_client  # noqa: E402  (run from scripts/ dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_config", default="panda_omron")
    p.add_argument("--embodiment_tag", default="new_embodiment")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--n_episodes", type=int, default=2)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--n_action_steps", type=int, default=16)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--video_dir",
                   default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/evals/groot_pretrain_80000")
    args = p.parse_args()
    os.makedirs(args.video_dir, exist_ok=True)

    print(f">>> GR00T smoke: PickPlaceCounterToSink split={args.split} "
          f"n={args.n_episodes} envs={args.n_envs}")
    server_thread = threading.Thread(
        target=run_server,
        args=(args.data_config, args.model_path, args.embodiment_tag, args.port),
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)  # client blocks/retries until the server's model is loaded
    run_client(
        host="localhost",
        port=args.port,
        task_set_list=["_smoke_ppc2sink"],
        video_dir=args.video_dir,
        split=args.split,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
    )
    print(f">>> GR00T smoke done. results under {args.video_dir}")
