"""Build the FIXED, region-stratified held-out eval set for Phase 1.

Scans candidate eval seeds (CPU-only, no rendering), records each seed's init
state (target object category, height, width, position), and selects a balanced
eval set: N_weak seeds with TALL objects (the weak region) + N_other seeds with
short objects. Output json is the frozen eval protocol — every Phase 1
checkpoint is evaluated on exactly these seeds, so before/after and
targeted-vs-random comparisons are paired and stratified.

Train/eval disjointness: training data uses seeds >=1000; eval seeds are drawn
from 0..max_seed (<1000).
"""
import argparse
import json
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa  # noqa: F401

TARGET = "obj"


def probe_seed(task, seed):
    """Create env with this seed, reset, record the init-state descriptor."""
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(
        env_name=task, robots="PandaOmron", controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        use_object_obs=True, ignore_done=True, translucent_robot=False,
        obj_instance_split="pretrain", layout_ids=-2, style_ids=-2, seed=seed)
    env.reset()
    ep = env.get_ep_meta()
    cat = next((c.get("info", {}).get("cat") for c in ep.get("object_cfgs", [])
                if c.get("name") == TARGET), "unknown")
    o = env.objects[TARGET]
    try:
        h = float(o.top_offset[2] - o.bottom_offset[2])
    except Exception:
        h = None
    try:
        w = float(o.horizontal_radius) * 2.0
    except Exception:
        w = None
    base = np.array(ep.get("init_robot_base_pos", [0, 0, 0])[:2])
    xy = np.array(env.sim.data.body_xpos[env.obj_body_id[TARGET]][:2]) - base
    rec = dict(seed=seed, object_category=cat, obj_height=h, obj_width=w,
               obj_xy_rel=[float(xy[0]), float(xy[1])],
               layout_id=ep.get("layout_id"), style_id=ep.get("style_id"))
    env.close()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--n_weak", type=int, default=50, help="tall-object eval seeds")
    p.add_argument("--n_other", type=int, default=50, help="short-object eval seeds")
    p.add_argument("--height_thresh", type=float, default=0.10,
                   help="tall = height > this (m); from the n=150 analysis")
    p.add_argument("--max_seed", type=int, default=600)
    p.add_argument("--out", default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/phase1_data/eval_set.json")
    args = p.parse_args()

    weak, other, scanned = [], [], []
    for seed in range(args.max_seed):
        if len(weak) >= args.n_weak and len(other) >= args.n_other:
            break
        try:
            r = probe_seed(args.task, seed)
        except Exception as e:
            print(f"seed {seed}: probe failed ({e}); skipping")
            continue
        scanned.append(r)
        if r["obj_height"] is None:
            continue
        if r["obj_height"] > args.height_thresh and len(weak) < args.n_weak:
            weak.append(r)
        elif r["obj_height"] <= args.height_thresh and len(other) < args.n_other:
            other.append(r)
        if (seed + 1) % 20 == 0:
            print(f"scanned {seed+1} seeds: weak {len(weak)}/{args.n_weak}, "
                  f"other {len(other)}/{args.n_other}")

    out = dict(task=args.task, split="pretrain",
               height_thresh=args.height_thresh,
               weak_region="obj_height > thresh (tall objects)",
               n_scanned=len(scanned),
               weak_seeds=weak, other_seeds=other)
    json.dump(out, open(args.out, "w"), indent=2)
    hw = [r["obj_height"] for r in weak]; ho = [r["obj_height"] for r in other]
    print(f"\nEval set: {len(weak)} weak (tall, h mean {np.mean(hw):.3f}) + "
          f"{len(other)} other (short, h mean {np.mean(ho):.3f}) seeds")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
