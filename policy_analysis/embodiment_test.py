"""Embodiment-limit test: are pi0's failures data-fixable or a physical limit?

The decisive question (per the active-learning paper's epistemic-vs-aleatoric
lens): do the objects pi0 fails on EXCEED the gripper's max aperture (physically
ungraspable -> no data helps), or are they within reach of the gripper (a
skill/data gap -> targeting CAN help)?

Measures the Panda gripper max aperture from the model, then joins it with the
n=150 weak-region data (which has per-episode obj_width = max horizontal extent
+ obj_height + success). CPU-only, no rendering, no GPU.
"""
import argparse
import json
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa  # noqa: F401


def gripper_aperture(task="PickPlaceCounterToSink"):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(
        env_name=task, robots="PandaOmron", controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        use_object_obs=True, ignore_done=True, obj_instance_split="pretrain",
        layout_ids=-2, style_ids=-2, translucent_robot=False)
    env.reset()
    # sum the max range of the gripper finger joints
    names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    fnames = [n for n in names if n and "gripper" in n and "finger" in n]
    rng = env.sim.model.jnt_range
    total = 0.0
    for n in fnames:
        jid = env.sim.model.joint_name2id(n)
        lo, hi = rng[jid]
        total += float(hi - lo)
    env.close()
    return total, fnames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weakregion_json",
                   default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/weakregion/pi0_PickPlaceCounterToSink_n150/weakregion.json")
    args = p.parse_args()

    aperture, fnames = gripper_aperture()
    print(f"Gripper finger joints: {fnames}")
    print(f"Max aperture (sum of finger joint travel): {aperture:.4f} m  (~{aperture*100:.1f} cm)\n")

    eps = json.load(open(args.weakregion_json))["episodes"]
    rows = [e for e in eps if e.get("obj_width") is not None]
    W = np.array([e["obj_width"] for e in rows])      # max horizontal extent (diameter)
    H = np.array([e["obj_height"] for e in rows])
    S = np.array([1.0 if e["success"] else 0.0 for e in rows])

    def rate(sel):
        n = int(sel.sum()); k = int(S[sel].sum())
        return n, k, (k / n if n else None)

    print(f"n={len(rows)} episodes, overall success={S.mean():.1%}")
    print(f"\n=== object width vs gripper aperture ({aperture:.3f} m) ===")
    print(f"objects WIDER than aperture (max-extent > {aperture:.3f}): {int((W>aperture).sum())}/{len(W)}")
    n, k, r = rate(W <= aperture); print(f"  graspable-width (<=aperture): success {r:.0%} ({k}/{n})" if r is not None else "  none")
    n, k, r = rate(W > aperture);  print(f"  over-aperture  (> aperture): success {r:.0%} ({k}/{n})" if r is not None else "  none")

    # the key cross-tab: among the WEAK region (tall objects), are failures on graspable-width objs?
    med_h = float(np.median(H))
    tall = H > med_h
    print(f"\n=== the weak region: TALL objects (height>{med_h:.3f}m), n={int(tall.sum())} ===")
    n, k, r = rate(tall & (W <= aperture)); print(f"  tall & graspable-width: success {r:.0%} ({k}/{n})" if r is not None else "  none")
    n, k, r = rate(tall & (W > aperture));  print(f"  tall & over-aperture:   success {r:.0%} ({k}/{n})" if r is not None else "  none")

    over_rate = rate(W > aperture)[2]
    print("\n=== VERDICT ===")
    print("NOTE: obj_width here = MAX horizontal extent (bounding-circle diameter),")
    print("which OVER-estimates ungraspability (gripper can grasp the narrow side / a part).")
    print(f"Empirical check: objects with diameter > aperture STILL succeed {over_rate:.0%}.")
    if over_rate and over_rate > 0.30:
        print("=> If 'over-aperture' objects still succeed substantially, the aperture is NOT")
        print("   the binding constraint -> NO aperture-based embodiment limit; the objects are")
        print("   graspable. The HEIGHT effect is therefore most consistent with a SKILL/grasp-")
        print("   strategy gap (epistemic, DATA-ADDRESSABLE), not a physical limit.")
        print("   Caveat: not a 100% ruling-out of a height-specific physical effect (grasp")
        print("   stability for tall objects) — but width/aperture is cleared.")
    else:
        print("=> Over-aperture objects rarely succeed -> a real aperture/embodiment limit.")


if __name__ == "__main__":
    main()
