"""Capture full-frame rollouts (agentview) of pi0 for a filmstrip figure.
Keeps the first SUCCESS and the first no-grasp FAILURE (fallback: any failure).
Run in openpi env with a pi0 server up, MUJOCO_GL=egl. Reuses analyze_pi0_weakregions helpers.
"""
import argparse, os, collections, sys
import numpy as np
import gymnasium as gym
import robocasa  # noqa: F401
from robocasa.utils.env_utils import convert_action
from robocasa.utils.dataset_registry_utils import get_task_horizon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pi0_weakregions import pi0_action, target_meta, TARGET_OBJ, _wcp


def run(a):
    horizon = min(int(get_task_horizon(a.task)), a.max_steps)
    client = _wcp.WebsocketClientPolicy(a.host, a.port)
    os.makedirs(a.out, exist_ok=True)
    got_succ = False
    got_fail = False          # a no-grasp failure (preferred)
    fallback_fail = None      # (frames, cat) any failure
    seed = a.seed
    for t_ep in range(a.max_tries):
        env = gym.make(f"robocasa/{a.task}", split=a.split, seed=seed)
        seed += 1
        rs = env.unwrapped
        obs, info = env.reset()
        lang = obs["annotation.human.task_description"]
        cat = (target_meta(rs)[0] or "obj")

        def objz():
            return float(rs.sim.data.body_xpos[rs.obj_body_id[TARGET_OBJ]][2])
        init_z = objz()
        frames, plan, success, maxlift = [], collections.deque(), False, 0.0
        for step in range(horizon):
            frames.append(np.asarray(obs["video.robot0_agentview_left"]).copy())
            act = pi0_action(client, obs, plan, lang, a.resize_size, a.replan_steps)
            obs, r, term, trunc, info = env.step(convert_action(act))
            maxlift = max(maxlift, objz() - init_z)
            if info["success"]:
                frames.append(np.asarray(obs["video.robot0_agentview_left"]).copy())
                success = True
                break
        env.close()
        grasped = maxlift > 0.05
        F = np.stack(frames)
        print(f"try {t_ep} seed {seed-1} cat={cat} success={success} grasped={grasped} "
              f"maxlift={maxlift:.3f} frames={len(frames)} shape={F.shape}", flush=True)

        if success and not got_succ:
            np.save(f"{a.out}/success_{cat}.npy", F); got_succ = True
            print(f"  >>> saved SUCCESS {cat} {F.shape}", flush=True)
        if not success:
            if fallback_fail is None:
                fallback_fail = (F, cat)
            if (not grasped) and not got_fail:
                np.save(f"{a.out}/failure_{cat}.npy", F); got_fail = True
                print(f"  >>> saved FAILURE(no-grasp) {cat} {F.shape}", flush=True)
        if got_succ and got_fail:
            break

    if not got_fail and fallback_fail is not None:
        F, cat = fallback_fail
        np.save(f"{a.out}/failure_{cat}.npy", F)
        print(f"  >>> saved FAILURE(fallback) {cat} {F.shape}", flush=True)
        got_fail = True
    print(f"DONE got_succ={got_succ} got_fail={got_fail}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8040)
    p.add_argument("--resize_size", type=int, default=224)
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--max_tries", type=int, default=12)
    p.add_argument("--max_steps", type=int, default=350)
    p.add_argument("--out", default="/data/xinyua11/robocasa/talk/rollout_frames")
    run(p.parse_args())
