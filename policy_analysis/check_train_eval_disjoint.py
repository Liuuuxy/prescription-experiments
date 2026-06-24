"""Integrity check: does a training-demo hdf5 overlap the frozen eval set?

The eval protocol requires train seeds (>=1000) be disjoint from eval seeds
(0..max, <1000). Seeds were NOT logged when the phase1 pools were generated, so
we verify disjointness *directly on the init states*: for each training demo we
reset the env to its stored initial state and recompute the SAME init-state
descriptor build_eval_set.py records (target-object category + position relative
to the robot base). A train demo that shares an eval seed produces a byte-for-byte
identical object placement, so an exact (category, obj_xy_rel) match == overlap.

Usage:
  MUJOCO_GL=egl python policy_analysis/check_train_eval_disjoint.py \
      --hdf5 phase1_data/ppc2sink_pi0_baseline100.hdf5 \
      --eval_set phase1_data/eval_set.json
"""
import argparse
import json
import h5py
import numpy as np

import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils

TARGET = "obj"
TOL = 1e-4  # metres; identical seeds match far tighter than this


def descriptor_from_env(env):
    """Recompute the build_eval_set descriptor from the *current* env state."""
    rs = env.env  # underlying robosuite env
    ep = rs.get_ep_meta()
    cat = next((c.get("info", {}).get("cat") for c in ep.get("object_cfgs", [])
                if c.get("name") == TARGET), "unknown")
    base = np.array(ep.get("init_robot_base_pos", [0, 0, 0])[:2])
    xy = np.array(rs.sim.data.body_xpos[rs.obj_body_id[TARGET]][:2]) - base
    return cat, np.array([float(xy[0]), float(xy[1])]), ep.get("layout_id"), ep.get("style_id")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--eval_set", default="phase1_data/eval_set.json")
    args = p.parse_args()

    ev = json.load(open(args.eval_set))
    eval_recs = ev["weak_seeds"] + ev["other_seeds"]
    eval_xy = {}  # category -> list of (seed, xy)
    for r in eval_recs:
        eval_xy.setdefault(r["object_category"], []).append(
            (r["seed"], np.array(r["obj_xy_rel"])))
    print(f"eval set: {len(eval_recs)} seeds "
          f"(range {min(r['seed'] for r in eval_recs)}..{max(r['seed'] for r in eval_recs)})")

    f = h5py.File(args.hdf5, "r")
    demos = list(f["data"].keys())
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.hdf5)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=[], camera_height=84, camera_width=84,
        reward_shaping=False)

    overlaps = []
    for demo in demos:
        dd = f["data"][demo]
        initial_state = dict(states=dd["states"][()][0], model=dd.attrs["model_file"])
        initial_state["ep_meta"] = dd.attrs.get("ep_meta", None)
        env.reset()
        env.reset_to(initial_state)
        cat, xy, _, _ = descriptor_from_env(env)
        for seed, exy in eval_xy.get(cat, []):
            if np.linalg.norm(xy - exy) < TOL:
                overlaps.append((demo, cat, seed, float(np.linalg.norm(xy - exy))))
                break

    f.close()
    print(f"\nchecked {len(demos)} train demos against eval set")
    if overlaps:
        print(f"!!! OVERLAP: {len(overlaps)} train demos coincide with eval seeds:")
        for demo, cat, seed, d in overlaps[:20]:
            print(f"  {demo}: {cat} == eval seed {seed} (|dxy|={d:.2e})")
    else:
        print("OK: no train demo matches any eval seed -> train/eval DISJOINT")


if __name__ == "__main__":
    main()
