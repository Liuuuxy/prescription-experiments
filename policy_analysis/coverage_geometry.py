"""A2: coverage + object-geometry analysis for PickPlaceCounterToSink.

Tests two competing explanations for pi0's grasp failures:
  (coverage)  failures hit objects the env samples *rarely* (data-sparse), or
  (geometry)  failures hit objects that are *tall/wide* (hard to grasp).

CPU-only: builds the env with no rendering, runs N resets to measure the env's
object-category sampling frequency + per-object geometry (height, width), then
joins with pi0's per-episode success from the weak-region run.

Run in openpi_env or robocasa env (no GPU, no server needed).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis  # noqa: E402

TARGET = "obj"


def make_env(task):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return robosuite.make(
        env_name=task, robots="PandaOmron", controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        use_object_obs=True, ignore_done=True, translucent_robot=False,
        obj_instance_split="pretrain", layout_ids=-2, style_ids=-2,
    )


def obj_geom(env):
    o = env.objects[TARGET]
    h = w = None
    try:
        h = float(o.top_offset[2] - o.bottom_offset[2])
    except Exception:
        pass
    try:
        w = float(o.horizontal_radius) * 2.0
    except Exception:
        pass
    # fallback: AABB from sim geoms belonging to the object
    if h is None or w is None:
        try:
            bid = env.obj_body_id[TARGET]
            gids = [i for i in range(env.sim.model.ngeom)
                    if env.sim.model.geom_bodyid[i] == bid]
            if gids:
                pos = env.sim.data.geom_xpos[gids]
                sz = env.sim.model.geom_size[gids]
                lo = (pos - sz).min(0); hi = (pos + sz).max(0)
                h = h or float(hi[2] - lo[2])
                w = w or float(max(hi[0] - lo[0], hi[1] - lo[1]))
        except Exception:
            pass
    return h, w


def cat_of(env):
    for cfg in env.get_ep_meta().get("object_cfgs", []):
        if cfg.get("name") == TARGET:
            return cfg.get("info", {}).get("cat", "unknown")
    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--weakregion_json",
                   default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/weakregion/pi0_PickPlaceCounterToSink/weakregion.json")
    p.add_argument("--out", default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/weakregion/coverage_geometry.json")
    args = p.parse_args()

    env = make_env(args.task)
    cat_count = defaultdict(int)
    cat_h, cat_w = defaultdict(list), defaultdict(list)
    for i in range(args.n):
        env.reset()
        c = cat_of(env)
        h, w = obj_geom(env)
        cat_count[c] += 1
        if h is not None:
            cat_h[c].append(h)
        if w is not None:
            cat_w[c].append(w)
        if (i + 1) % 25 == 0:
            print(f"  reset {i+1}/{args.n}")
    env.close()

    geom = {c: {"freq": cat_count[c],
                "height": round(float(np.mean(cat_h[c])), 3) if cat_h[c] else None,
                "width": round(float(np.mean(cat_w[c])), 3) if cat_w[c] else None}
            for c in cat_count}

    # join with pi0 per-episode success
    wr = json.load(open(args.weakregion_json))
    eps = wr["episodes"]
    # per-episode: attach this category's avg geometry, then bin by height
    rows = []
    for e in eps:
        c = e["object_category"]
        g = geom.get(c, {})
        rows.append({"success": e["success"], "object_category": c,
                     "height": g.get("height"), "width": g.get("width"),
                     "freq": g.get("freq", 0)})

    def rate(sel):
        s = [r for r in rows if sel(r)]
        k = sum(1 for r in s if r["success"])
        return (len(s), k, round(k / len(s), 3) if s else None)

    # geometry bins (height) — tests the graspability hypothesis
    hs = [r["height"] for r in rows if r["height"] is not None]
    out = {"task": args.task, "n_resets": args.n, "n_eval_eps": len(rows),
           "per_category": dict(sorted(geom.items(), key=lambda kv: -kv[1]["freq"]))}
    if hs:
        med = float(np.median(hs))
        out["height_split"] = {
            "median_height": round(med, 3),
            "tall_objects (>median)": rate(lambda r: r["height"] and r["height"] > med),
            "short_objects (<=median)": rate(lambda r: r["height"] and r["height"] <= med),
        }
    ws = [r["width"] for r in rows if r["width"] is not None]
    if ws:
        medw = float(np.median(ws))
        out["width_split"] = {
            "median_width": round(medw, 3),
            "wide_objects (>median)": rate(lambda r: r["width"] and r["width"] > medw),
            "narrow_objects (<=median)": rate(lambda r: r["width"] and r["width"] <= medw),
        }
    # coverage: do rare categories fail more? split by sampling frequency
    out["frequency_split"] = {
        "rare (freq<=median)": rate(lambda r: r["freq"] <= np.median([g["freq"] for g in geom.values()])),
        "common (freq>median)": rate(lambda r: r["freq"] > np.median([g["freq"] for g in geom.values()])),
    }

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== Coverage vs Geometry ===")
    if "height_split" in out:
        hsd = out["height_split"]
        print(f"median height={hsd['median_height']}m")
        print(f"  TALL  (>median): success {hsd['tall_objects (>median)'][2]} ({hsd['tall_objects (>median)'][1]}/{hsd['tall_objects (>median)'][0]})")
        print(f"  SHORT (<=median): success {hsd['short_objects (<=median)'][2]} ({hsd['short_objects (<=median)'][1]}/{hsd['short_objects (<=median)'][0]})")
    if "width_split" in out:
        wsd = out["width_split"]
        print(f"median width={wsd['median_width']}m")
        print(f"  WIDE   (>median): success {wsd['wide_objects (>median)'][2]} ({wsd['wide_objects (>median)'][1]}/{wsd['wide_objects (>median)'][0]})")
        print(f"  NARROW (<=median): success {wsd['narrow_objects (<=median)'][2]} ({wsd['narrow_objects (<=median)'][1]}/{wsd['narrow_objects (<=median)'][0]})")
    fsd = out["frequency_split"]
    print(f"  RARE objects:   success {fsd['rare (freq<=median)'][2]} ({fsd['rare (freq<=median)'][1]}/{fsd['rare (freq<=median)'][0]})")
    print(f"  COMMON objects: success {fsd['common (freq>median)'][2]} ({fsd['common (freq>median)'][1]}/{fsd['common (freq>median)'][0]})")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
