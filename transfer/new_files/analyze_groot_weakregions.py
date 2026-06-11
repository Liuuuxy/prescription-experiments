"""GR00T weak-region + failure analysis (cross-policy check vs pi0).

Subclasses SimulationInferenceClient to capture per-episode metadata (object
category, geometry height/width, position-rel-to-robot, layout/style) + a
failure-phase tag, in the SAME record format as analyze_pi0_weakregions.py, so
predict_failure.py works unchanged. Runs single-env. Starts the GR00T server in
a thread. Run in groot_env with USE_TF=0 MUJOCO_GL=egl.
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY  # noqa: F401
from robocasa.utils.dataset_registry_utils import get_task_horizon
from gr00t.eval.simulation import (
    SimulationInferenceClient, SimulationConfig, MultiStepConfig, VideoConfig,
)
from run_eval import run_server

TARGET = "obj"


def capture(rs):
    ep = rs.get_ep_meta()
    cat = "unknown"
    for cfg in ep.get("object_cfgs", []):
        if cfg.get("name") == TARGET:
            cat = cfg.get("info", {}).get("cat", "unknown")
    base = np.array(ep.get("init_robot_base_pos", [0, 0, 0])[:2])
    abs_xy = np.array(rs.sim.data.body_xpos[rs.obj_body_id[TARGET]][:2])
    o = rs.objects[TARGET]
    try:
        h = float(o.top_offset[2] - o.bottom_offset[2])
    except Exception:
        h = None
    try:
        w = float(o.horizontal_radius) * 2.0
    except Exception:
        w = None
    sink_xy = (np.array(rs.sim.data.body_xpos[rs.obj_body_id["distr_sink"]][:2])
               if "distr_sink" in getattr(rs, "obj_body_id", {}) else None)
    return {
        "object_category": cat, "obj_height": (round(h, 4) if h else None),
        "obj_width": (round(w, 4) if w else None),
        "obj_xy_abs": [float(abs_xy[0]), float(abs_xy[1])],
        "obj_xy_rel": [float(abs_xy[0] - base[0]), float(abs_xy[1] - base[1])],
        "layout_id": ep.get("layout_id"), "style_id": ep.get("style_id"),
        "_init_z": float(rs.sim.data.body_xpos[rs.obj_body_id[TARGET]][2]),
        "_sink_xy": sink_xy,
    }


class WRClient(SimulationInferenceClient):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.records = []

    def run_simulation(self, config):
        self.env = self.setup_environment(config)
        rs = self.env.envs[0].unwrapped
        obs, _ = self.env.reset()
        meta = capture(rs)
        succ = False
        max_lift, min_sink = 0.0, float("inf")
        while len(self.records) < config.n_episodes:
            actions = self._get_actions_from_server(obs)
            obs, rewards, term, trunc, infos = self.env.step(actions)
            p = rs.sim.data.body_xpos[rs.obj_body_id[TARGET]]
            max_lift = max(max_lift, float(p[2]) - meta["_init_z"])
            if meta["_sink_xy"] is not None:
                min_sink = min(min_sink, float(np.linalg.norm(p[:2] - meta["_sink_xy"])))
            succ |= bool(infos["success"][0][0])
            if term[0] or trunc[0]:
                grasped = max_lift > 0.05
                near = meta["_sink_xy"] is not None and min_sink < 0.15
                phase = ("success" if succ else "fail_no_grasp" if not grasped
                         else "fail_grasped_no_transport" if not near
                         else "fail_reached_sink_no_place")
                rec = {k: v for k, v in meta.items() if not k.startswith("_")}
                rec.update(episode=len(self.records), success=bool(succ),
                           failure_phase=phase, max_lift=round(max_lift, 3))
                self.records.append(rec)
                print(f"[ep {len(self.records)}/{config.n_episodes}] success={succ} "
                      f"phase={phase} obj={rec['object_category']} h={rec['obj_height']}")
                succ, max_lift, min_sink = False, 0.0, float("inf")
                meta = capture(rs)  # env auto-reset -> new episode
        self.env.close()
        self.env = None
        return config.env_name, [r["success"] for r in self.records]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_config", default="panda_omron")
    p.add_argument("--embodiment_tag", default="new_embodiment")
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n_action_steps", type=int, default=16)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--out_dir", default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/weakregion/groot_PickPlaceCounterToSink")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    threading.Thread(target=run_server,
                     args=(args.data_config, args.model_path, args.embodiment_tag, args.port),
                     daemon=True).start()
    time.sleep(1)

    horizon = int(get_task_horizon(args.task))
    cfg = SimulationConfig(
        env_name=f"robocasa/{args.task}", split=args.split, n_episodes=args.n, n_envs=1,
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(n_action_steps=args.n_action_steps, max_episode_steps=horizon),
    )
    client = WRClient(host="localhost", port=args.port)
    client.run_simulation(cfg)

    sys.path.insert(0, "/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/policy_analysis")
    import analysis
    summary = analysis.summarize(client.records)
    out = {"task": args.task, "split": args.split, "policy": "groot", "n": args.n,
           "summary": summary, "episodes": client.records}
    json.dump(out, open(os.path.join(args.out_dir, "weakregion.json"), "w"), indent=2)
    rep = analysis.format_report(summary, task=args.task, policy="groot")
    fails = [r for r in client.records if not r["success"]]
    import collections
    pc = collections.Counter(r["failure_phase"] for r in fails)
    rep += "\n\nFailure modes (of {} failures):\n".format(len(fails))
    rep += "\n".join("  {:<28} {:>3} ({:.0%})".format(k, v, v / max(len(fails), 1))
                     for k, v in pc.most_common())
    open(os.path.join(args.out_dir, "weakregion_report.txt"), "w").write(rep + "\n")
    print("\n" + rep)
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
