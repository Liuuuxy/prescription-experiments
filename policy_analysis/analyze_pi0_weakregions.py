"""Per-bucket weak-region analysis of pi0 on a RoboCasa task (project core method).

Runs pi0 (via the websocket server) for N episodes, logging per-episode success
joined with init-state metadata (target object category, object position
relative to the robot, layout/style), then buckets failures with analysis.py to
answer *where* pi0 fails -- the input to the targeted-data experiment.

Run in openpi_env with the pi0 server up and MUJOCO_GL=egl.
"""
import argparse
import collections
import json
import os
import signal
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp
from robocasa.utils.env_utils import convert_action
from robocasa.utils.dataset_registry_utils import get_task_horizon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis  # noqa: E402

TARGET_OBJ = "obj"


class _RolloutTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _RolloutTimeout()


signal.signal(signal.SIGALRM, _alarm_handler)  # per-episode watchdog (env build/reset can hang)


def target_meta(rs):
    ep = rs.get_ep_meta()
    cat, split = "unknown", "unknown"
    for cfg in ep.get("object_cfgs", []):
        if cfg.get("name") == TARGET_OBJ:
            cat = cfg.get("info", {}).get("cat", "unknown")
            split = cfg.get("info", {}).get("split", "unknown")
    base = np.array(ep.get("init_robot_base_pos", [0, 0, 0])[:2])
    abs_xy = np.array(rs.sim.data.body_xpos[rs.obj_body_id[TARGET_OBJ]][:2])
    return cat, split, abs_xy, abs_xy - base, ep.get("layout_id"), ep.get("style_id")


def build_pi0_element(obs, lang, resize):
    def prep(k):
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(obs[k]), resize, resize))
    return {
        "observation/image": prep("video.robot0_agentview_left"),
        "observation/wrist_image": prep("video.robot0_eye_in_hand"),
        "observation/right_image": prep("video.robot0_agentview_right"),
        "observation/state": np.concatenate((
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"]), axis=0),
        "prompt": lang,
    }


def pi0_action(client, obs, plan, lang, resize, replan):
    if not plan:
        plan.extend(client.infer(build_pi0_element(obs, lang, resize))["actions"][:replan])
    return plan.popleft()


def pi0_uncertainty(client, obs, lang, resize, k):
    """K action-chunk samples at the SAME state -> mean per-coord std (epistemic).
    pi0 is a stochastic flow policy, so repeated infer() calls differ."""
    el = build_pi0_element(obs, lang, resize)
    samples = [np.asarray(client.infer(el)["actions"], dtype=np.float64) for _ in range(k)]
    return float(np.stack(samples, 0).std(axis=0).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resize_size", type=int, default=224)
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-k", "--k_unc", type=int, default=0,
                   help="K action samples at first state -> per-episode pi0 uncertainty (0=off)")
    p.add_argument("--categories", default=None,
                   help="comma-sep object categories to RESTRICT eval to (stratified/powered eval); "
                        "scenes with other objects are skipped (cheap reset, no rollout)")
    p.add_argument("--per_cat", type=int, default=0,
                   help="with --categories: stop once every listed category has this many episodes")
    p.add_argument("--out_dir", default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/weakregion/pi0_PickPlaceCounterToSink")
    p.add_argument("--rollout_timeout", type=int, default=180,
                   help="skip a scene if env build/reset exceeds this many seconds (0=off); "
                        "guards the mujoco placement-sampling hang that can stall the eval")
    p.add_argument("--save_obs_dir", default=None,
                   help="if set, save each episode's initial agentview image (ep{i}.npy) for "
                        "category-free failed-state retrieval; join to records by episode index")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    target_cats = (set(c.strip().lower() for c in args.categories.split(",")) if args.categories else None)
    cat_count = collections.defaultdict(int)

    horizon = int(get_task_horizon(args.task))
    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    records = []
    for i in range(args.n):
        if args.rollout_timeout:
            signal.alarm(args.rollout_timeout)
        try:
            env = gym.make(f"robocasa/{args.task}", split=args.split, seed=args.seed + i)
            rs = env.unwrapped
            obs, info = env.reset()
        except _RolloutTimeout:
            print(f"[timeout] seed {args.seed + i}: env build/reset > {args.rollout_timeout}s -- skip", flush=True)
            signal.alarm(0)
            continue
        signal.alarm(0)  # reset survived; don't guard the bounded rollout (avoids uncaught mid-rollout raise)
        lang = obs["annotation.human.task_description"]
        if args.save_obs_dir:
            os.makedirs(args.save_obs_dir, exist_ok=True)
            _img0 = obs.get("video.robot0_agentview_left")
            if _img0 is not None:
                np.save(os.path.join(args.save_obs_dir, f"ep{i:04d}.npy"), np.asarray(_img0))
        cat, osplit, xy_abs, xy_rel, layout, style = target_meta(rs)
        # stratified eval: only spend a (600-step) rollout on a targeted category that isn't full yet
        if target_cats is not None:
            cn = (cat or "").lower()
            if cn not in target_cats or (args.per_cat and cat_count[cn] >= args.per_cat):
                env.close()
                if args.per_cat and all(cat_count[c] >= args.per_cat for c in target_cats):
                    break
                continue
            cat_count[cn] += 1
        uncertainty = (pi0_uncertainty(client, obs, lang, args.resize_size, args.k_unc)
                       if args.k_unc else None)

        # per-episode object geometry (direct, not category-averaged) for the
        # failure predictor: height + width of the target object.
        _o = rs.objects[TARGET_OBJ]
        try:
            obj_h = float(_o.top_offset[2] - _o.bottom_offset[2])
        except Exception:
            obj_h = None
        try:
            obj_w = float(_o.horizontal_radius) * 2.0
        except Exception:
            obj_w = None

        # failure-phase signals: track object lift (grasp) and proximity to the
        # sink (transport). sink location proxied by the distractor that sits in
        # the sink (distr_sink).
        def obj_xyz():
            return np.array(rs.sim.data.body_xpos[rs.obj_body_id[TARGET_OBJ]])
        sink_xy = (np.array(rs.sim.data.body_xpos[rs.obj_body_id["distr_sink"]][:2])
                   if "distr_sink" in getattr(rs, "obj_body_id", {}) else None)
        init_z = float(obj_xyz()[2])
        max_lift, min_sink_dist = 0.0, float("inf")

        plan = collections.deque()
        success, steps = False, horizon
        for t in range(horizon):
            a = pi0_action(client, obs, plan, lang, args.resize_size, args.replan_steps)
            obs, r, term, trunc, info = env.step(convert_action(a))
            pz = obj_xyz()
            max_lift = max(max_lift, float(pz[2]) - init_z)
            if sink_xy is not None:
                min_sink_dist = min(min_sink_dist, float(np.linalg.norm(pz[:2] - sink_xy)))
            if info["success"]:
                success, steps = True, t + 1
                break

        grasped = max_lift > 0.05  # lifted >5cm
        near_sink = sink_xy is not None and min_sink_dist < 0.15  # within 15cm
        if success:
            phase = "success"
        elif not grasped:
            phase = "fail_no_grasp"
        elif not near_sink:
            phase = "fail_grasped_no_transport"
        else:
            phase = "fail_reached_sink_no_place"

        records.append(dict(
            episode=i, success=bool(success), steps_to_success=steps if success else None,
            failure_phase=phase, max_lift=round(max_lift, 3),
            min_sink_dist=(round(min_sink_dist, 3) if min_sink_dist != float("inf") else None),
            object_category=cat, object_split=osplit,
            obj_height=(round(obj_h, 4) if obj_h is not None else None),
            obj_width=(round(obj_w, 4) if obj_w is not None else None),
            obj_xy_abs=[float(xy_abs[0]), float(xy_abs[1])],
            obj_xy_rel=[float(xy_rel[0]), float(xy_rel[1])],
            layout_id=layout, style_id=style,
            uncertainty=(round(uncertainty, 5) if uncertainty is not None else None)))
        print(f"[ep {i+1}/{args.n}] success={success} phase={phase} obj={cat} "
              f"unc={uncertainty} rel=({xy_rel[0]:.2f},{xy_rel[1]:.2f}) layout={layout} style={style}")
        env.close()
        if (i + 1) % 20 == 0:  # crash-safe partial dump (full summary written at end)
            json.dump({"task": args.task, "split": args.split, "policy": "pi0",
                       "n": i + 1, "episodes": records},
                      open(os.path.join(args.out_dir, "weakregion.json"), "w"), indent=2)

    # failure-mode breakdown (among failed episodes)
    fails = [r for r in records if not r["success"]]
    phase_counts = collections.Counter(r["failure_phase"] for r in fails)
    phase_lines = ["", "Failure modes (of {} failures):".format(len(fails))]
    for ph, c in sorted(phase_counts.items(), key=lambda kv: -kv[1]):
        phase_lines.append("  {:<28} {:>3} ({:.0%})".format(ph, c, c / max(len(fails), 1)))
    phase_report = "\n".join(phase_lines)

    summary = analysis.summarize(records)
    out = {"task": args.task, "split": args.split, "policy": "pi0", "n": args.n,
           "timestamp": datetime.now().isoformat(timespec="seconds"),
           "summary": summary, "episodes": records}
    with open(os.path.join(args.out_dir, "weakregion.json"), "w") as f:
        json.dump(out, f, indent=2)
    report = analysis.format_report(summary, task=args.task, policy="pi0") + "\n" + phase_report
    with open(os.path.join(args.out_dir, "weakregion_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nWrote weakregion.json + report to {args.out_dir}")


if __name__ == "__main__":
    main()
