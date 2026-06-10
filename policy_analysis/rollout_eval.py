"""Run a policy on a RoboCasa task and produce quantitative + qualitative analysis.

Quantitative: per-episode success joined with episode metadata (object category,
object init position relative to the robot, layout/style), aggregated into
overall + per-bucket success rates (see analysis.py) and written to metrics.json.

Qualitative: rollout videos saved into videos/success/ and videos/failure/ so you
can eyeball *how* it succeeds/fails -- enough to triage a model as good/bad
without watching every episode.

Self-test (no checkpoint needed):
    MUJOCO_GL=egl python rollout_eval.py --task PickPlaceCounterToSink \
        --n 3 --steps 30 --policy random --out-dir ./results/smoke

Real model (on the H100): implement an adapter in policies.py and pass
--policy checkpoint --checkpoint <path>.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np

# robocasa import registers envs and pulls in robosuite
from robocasa.utils.env_utils import create_env

import analysis
from policies import RandomPolicy

# the object to be manipulated is registered under this name in pick-place tasks
TARGET_OBJ_NAME = "obj"


def _target_object_meta(env):
    """Category + split of the manipulated object from episode metadata."""
    ep = env.get_ep_meta()
    for cfg in ep.get("object_cfgs", []):
        if cfg.get("name") == TARGET_OBJ_NAME:
            info = cfg.get("info", {})
            return info.get("cat", "unknown"), info.get("split", "unknown")
    return "unknown", "unknown"


def _target_object_xy(env):
    """Object init (x, y), absolute and relative to the robot base."""
    abs_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[TARGET_OBJ_NAME]][:2])
    base = np.array(env.get_ep_meta().get("init_robot_base_pos", [0, 0, 0])[:2])
    return abs_pos, abs_pos - base


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    vid_dir = os.path.join(args.out_dir, "videos")
    if args.save_videos != "none":
        os.makedirs(os.path.join(vid_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(vid_dir, "failure"), exist_ok=True)

    camera_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    env = create_env(
        env_name=args.task,
        split=args.split,
        seed=args.seed,
        camera_names=camera_names,
        camera_widths=128,
        camera_heights=128,
    )
    # robots (and thus action limits) are only instantiated after a reset
    env.reset()
    low, high = env.action_spec

    if args.policy == "random":
        policy = RandomPolicy(low, high, seed=args.seed)
        policy_name = "random"
    else:  # pragma: no cover - requires checkpoint + framework on H100
        from policies import CheckpointPolicyStub

        policy = CheckpointPolicyStub(args.checkpoint, camera_names)
        policy_name = f"checkpoint:{os.path.basename(args.checkpoint)}"

    records = []
    try:
        import imageio
    except ImportError:
        imageio = None

    for ep_i in range(args.n):
        obs = env.reset()
        policy.reset()
        cat, split = _target_object_meta(env)
        xy_abs, xy_rel = _target_object_xy(env)
        ep_meta = env.get_ep_meta()

        frames = []
        success = False
        steps_to_success = None
        for step_i in range(args.steps):
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            if args.save_videos != "none" and imageio is not None:
                frames.append(
                    env.sim.render(height=512, width=768,
                                   camera_name="robot0_agentview_left")[::-1]
                )
            if env._check_success():
                success = True
                steps_to_success = step_i + 1
                break

        video_path = None
        if args.save_videos != "none" and imageio is not None and frames:
            if args.save_videos == "all" or (args.save_videos == "failure" and not success):
                bucket = "success" if success else "failure"
                video_path = os.path.join(
                    vid_dir, bucket, f"ep{ep_i:03d}_{cat}.mp4"
                )
                w = imageio.get_writer(video_path, fps=20)
                for f in frames:
                    w.append_data(f)
                w.close()

        rec = {
            "episode": ep_i,
            "seed": args.seed,
            "success": bool(success),
            "steps_to_success": steps_to_success,
            "episode_length": step_i + 1,
            "object_category": cat,
            "object_split": split,
            "obj_xy_abs": [float(xy_abs[0]), float(xy_abs[1])],
            "obj_xy_rel": [float(xy_rel[0]), float(xy_rel[1])],
            "layout_id": ep_meta.get("layout_id"),
            "style_id": ep_meta.get("style_id"),
            "video": video_path,
        }
        records.append(rec)
        print(
            f"[ep {ep_i + 1}/{args.n}] success={success} "
            f"obj={cat} region_xy_rel=({xy_rel[0]:.2f},{xy_rel[1]:.2f}) "
            f"layout={rec['layout_id']} style={rec['style_id']}"
        )

    env.close()

    summary = analysis.summarize(records)
    out = {
        "task": args.task,
        "split": args.split,
        "policy": policy_name,
        "n": args.n,
        "steps": args.steps,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "episodes": records,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    report = analysis.format_report(summary, task=args.task, policy=policy_name)
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nWrote metrics.json + report.txt (+ videos) to {args.out_dir}")
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target", "all"])
    p.add_argument("--n", type=int, default=20, help="number of rollouts")
    p.add_argument("--steps", type=int, default=400, help="max steps per rollout")
    p.add_argument("--policy", default="random", choices=["random", "checkpoint"])
    p.add_argument("--checkpoint", default=None, help="checkpoint path (policy=checkpoint)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--save-videos", default="failure", choices=["all", "failure", "none"])
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_dir = os.path.join("results", f"{args.task}_{args.policy}_{ts}")
    run(args)
