"""Continuous task-progress signal — sharper failure localization than binary.

Binary success is high-variance at small n. This builds a continuous 0-1
progress score from the per-episode signals already logged (max_lift,
min_sink_dist) and tests whether object height predicts *progress* (and the
grasp proxy max_lift) more powerfully than it predicts binary success.

CPU-only, uses the existing weak-region json.
"""
import argparse
import json
import numpy as np

GRASP_LIFT = 0.05      # object lifted >5cm = grasped
SINK_NEAR, SINK_FAR = 0.15, 0.5


def progress(e):
    if e["success"]:
        return 1.0
    ml = e.get("max_lift") or 0.0
    ms = e.get("min_sink_dist")
    if ml > GRASP_LIFT:  # grasped but didn't place
        transport = 0.0
        if ms is not None:
            transport = np.clip((SINK_FAR - ms) / (SINK_FAR - SINK_NEAR), 0, 1)
        return 0.6 + 0.35 * float(transport)
    # never grasped: partial credit for how close to lifting
    return 0.5 * float(np.clip(ml / GRASP_LIFT, 0, 1))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def linreg_r2(X, y):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ beta
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot if ss_tot > 0 else 0.0), beta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weakregion_json",
                   default="/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/weakregion/pi0_PickPlaceCounterToSink_n150/weakregion.json")
    args = p.parse_args()
    eps = [e for e in json.load(open(args.weakregion_json))["episodes"]
           if e.get("obj_height") is not None]
    prog = np.array([progress(e) for e in eps])
    succ = np.array([1.0 if e["success"] else 0.0 for e in eps])
    H = np.array([e["obj_height"] for e in eps])
    W = np.array([e["obj_width"] for e in eps])
    lat = np.array([e["obj_xy_rel"][0] for e in eps])
    dep = np.array([e["obj_xy_rel"][1] for e in eps])
    lift = np.array([(e.get("max_lift") or 0.0) for e in eps])

    print(f"n={len(eps)}, success={succ.mean():.1%}, mean progress={prog.mean():.2f}")

    # how far did the FAILURES get? (granularity behind '86% no-grasp')
    fails = [e for e in eps if not e["success"]]
    stages = {"never_touched (<1cm lift)": 0, "lift_attempt (1-5cm)": 0,
              "grasped_not_near_sink": 0, "grasped_near_sink": 0}
    for e in fails:
        ml = e.get("max_lift") or 0.0; ms = e.get("min_sink_dist") or 1.0
        if ml <= 0.01:
            stages["never_touched (<1cm lift)"] += 1
        elif ml <= GRASP_LIFT:
            stages["lift_attempt (1-5cm)"] += 1
        elif ms > SINK_NEAR:
            stages["grasped_not_near_sink"] += 1
        else:
            stages["grasped_near_sink"] += 1
    print("\nFailure granularity (of {} failures):".format(len(fails)))
    for k, v in stages.items():
        print(f"  {k:<28} {v:>3} ({v/max(len(fails),1):.0%})")

    print("\nDoes object HEIGHT predict failure? (more negative = taller -> worse)")
    print(f"  corr(height, binary success):     {pearson(H, succ):+.3f}")
    print(f"  corr(height, continuous progress):{pearson(H, prog):+.3f}")
    print(f"  corr(height, max_lift / grasp):   {pearson(H, lift):+.3f}   <- the grasp signal")

    print("\nVariance explained (R^2) by all features [height,width,lateral,depth]:")
    X = np.column_stack([H, W, lat, dep])
    r2s, _ = linreg_r2(X, succ); r2p, _ = linreg_r2(X, prog); r2l, b = linreg_r2(X, lift)
    print(f"  -> binary success:      R^2 = {r2s:.3f}")
    print(f"  -> continuous progress: R^2 = {r2p:.3f}")
    print(f"  -> max_lift (grasp):    R^2 = {r2l:.3f}")
    print(f"     max_lift coeffs [int,height,width,lat,depth]: {[round(float(c),3) for c in b]}")


if __name__ == "__main__":
    main()
