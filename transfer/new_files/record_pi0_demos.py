"""Record pi0-driven SUCCESSFUL demos in robomimic hdf5 format for mimicgen.

mimicgen needs source demos with full mujoco states (v1.0.1's LeRobot data drops
them). This drives the gym env exactly like the pi0 client (so pi0's obs
preprocessing is identical and it actually succeeds at ~58%), records mujoco
states + actions for each episode, keeps only the successful ones, and writes a
single robomimic-format hdf5 (data group + per-demo model_file/ep_meta/states/
actions) finalized with convert_to_robomimic_format.

Run in openpi_env (has robocasa + openpi_client), with the pi0 server up and
MUJOCO_GL=egl.
"""
import argparse
import collections
import datetime
import json
import os

import gymnasium as gym
import h5py
import numpy as np
import robocasa  # noqa: F401  (registers envs)
import robosuite
import mujoco
from robosuite.controllers import load_composite_controller_config
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

from robocasa.utils.env_utils import convert_action
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format


def build_env_info(env_name, robots="PandaOmron"):
    """env_kwargs json (mirrors collect_demos.config) so robomimic/mimicgen can
    recreate the env from the dataset."""
    controller_config = load_composite_controller_config(controller=None, robot=robots)
    config = {
        "env_name": env_name,
        "robots": robots,
        "controller_configs": controller_config,
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_heights": 256,
        "camera_widths": 256,
        "translucent_robot": False,
    }
    return json.dumps(config)


def pi0_action(client, obs, action_plan, task_lang, resize_size, replan_steps):
    """Replicate examples/robocasa/main.py: build element, query server, replan."""
    if not action_plan:
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.ascontiguousarray(obs["video.robot0_agentview_left"]),
                resize_size, resize_size))
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.ascontiguousarray(obs["video.robot0_eye_in_hand"]),
                resize_size, resize_size))
        right = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.ascontiguousarray(obs["video.robot0_agentview_right"]),
                resize_size, resize_size))
        state = np.concatenate((
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ), axis=0)
        element = {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/right_image": right,
            "observation/state": state,
            "prompt": task_lang,
        }
        chunk = client.infer(element)["actions"]
        action_plan.extend(chunk[:replan_steps])
    return action_plan.popleft()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--n_success", type=int, default=10, help="target #successful demos")
    p.add_argument("--max_attempts", type=int, default=40)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resize_size", type=int, default=224)
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out", default=None)
    # targeted-region control via rejection sampling on the init state:
    # only run the policy when the target object's height is inside the band.
    # (targeted arm: --min_obj_height 0.10 ; random arm: leave unset)
    p.add_argument("--min_obj_height", type=float, default=None)
    p.add_argument("--max_obj_height", type=float, default=None)
    args = p.parse_args()

    out = args.out or os.path.join(
        "/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/mimicgen_src",
        f"{args.task}_pi0_src.hdf5")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    horizon = int(get_task_horizon(args.task))
    env_info = build_env_info(args.task)
    client = _wcp.WebsocketClientPolicy(args.host, args.port)

    demos = []  # list of dict(states, actions, model_file, ep_meta)
    attempts = 0
    while len(demos) < args.n_success and attempts < args.max_attempts:
        env = gym.make(f"robocasa/{args.task}", split=args.split, seed=args.seed + attempts)
        rs = env.unwrapped
        obs, info = env.reset()

        # rejection sampling for the targeted region (cheap: no policy queries)
        if args.min_obj_height is not None or args.max_obj_height is not None:
            o = rs.objects["obj"]
            try:
                oh = float(o.top_offset[2] - o.bottom_offset[2])
            except Exception:
                oh = None
            in_band = (oh is not None
                       and (args.min_obj_height is None or oh >= args.min_obj_height)
                       and (args.max_obj_height is None or oh <= args.max_obj_height))
            if not in_band:
                attempts += 1
                print(f"[attempt {attempts}] rejected (obj height={oh}) — outside target band")
                env.close()
                continue

        task_lang = obs["annotation.human.task_description"]
        model_xml = rs.model.get_xml()
        ep_meta = json.dumps(rs.get_ep_meta())
        states = [rs.sim.get_state().flatten()]
        actions = []
        action_plan = collections.deque()
        success = False
        for t in range(horizon):
            a = pi0_action(client, obs, action_plan, task_lang,
                           args.resize_size, args.replan_steps)
            actions.append(np.asarray(a, dtype=np.float64))
            obs, reward, term, trunc, info = env.step(convert_action(a))
            states.append(rs.sim.get_state().flatten())
            if info["success"]:
                success = True
                break
        attempts += 1
        if success:
            del states[-1]  # DataCollector convention: drop trailing state
            assert len(states) == len(actions)
            demos.append(dict(states=np.array(states), actions=np.array(actions),
                              model_file=model_xml, ep_meta=ep_meta))
            print(f"[attempt {attempts}] SUCCESS ({len(actions)} steps) "
                  f"-> {len(demos)}/{args.n_success} demos")
            if len(demos) % 10 == 0:  # crash-safe periodic flush
                write_demos(out, demos, env_info, args.task)
        else:
            print(f"[attempt {attempts}] fail")
        env.close()

    if not demos:
        print("No successful demos collected. Aborting.")
        return

    write_demos(out, demos, env_info, args.task)
    convert_to_robomimic_format(out)
    print("Converted to robomimic format (env_args set).")


def write_demos(out, demos, env_info, task):
    """Write demos in collect_demos gather-format hdf5 (overwrites)."""
    now = datetime.datetime.now()
    f = h5py.File(out, "w")
    grp = f.create_group("data")
    grp.attrs["date"] = now.strftime("%Y-%m-%d")
    grp.attrs["time"] = now.strftime("%H:%M:%S")
    grp.attrs["repository_version"] = robosuite.__version__
    grp.attrs["robocasa_version"] = robocasa.__version__
    grp.attrs["robosuite_version"] = robosuite.__version__
    grp.attrs["mujoco_version"] = mujoco.__version__
    grp.attrs["env"] = task
    grp.attrs["env_info"] = env_info
    for i, d in enumerate(demos):
        g = grp.create_group(f"demo_{i + 1}")
        g.attrs["model_file"] = d["model_file"]
        g.attrs["ep_meta"] = d["ep_meta"]
        g.attrs["num_samples"] = len(d["actions"])
        g.create_dataset("states", data=d["states"])
        g.create_dataset("actions", data=d["actions"])
    f.close()
    print(f"Wrote {len(demos)} demos to {out}")


if __name__ == "__main__":
    main()
