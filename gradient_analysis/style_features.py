"""Execution-style features + style_hi/style_lo arm selection (branch 2, 2026-08-03).

Scores every W-pool demo on HOW it was executed (not where): SREE-style
consistency to the per-category modal trajectory shape, grasp retries
(gripper-action cycles), smoothness (xyz jerk + arm-action jitter), pause
fraction, and duration. All features are z-scored WITHIN CATEGORY (fallback:
within region) so the hi/lo arms match the pool on category/region mix and the
contrast isolates pure execution style. Selection is region-stratified to the
pool's region shares.

Data: per-episode parquets of the pool LeRobot set (observation.state 16d:
s7:10 = EE xyz, s14 = finger width; action 12d: a11 = gripper +-1, a5:11 = arm).
Output: gradient_analysis/style_features.parquet, style_assignment.json,
style_report.json. CPU-only, robocasa env.
"""
import glob
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/xinyua11/robocasa")

GA = "/data/xinyua11/robocasa/gradient_analysis"
DATA_GLOB = ("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
             "PickPlaceCounterToSink/*/mg/demo/*/lerobot/data/chunk-*/episode_*.parquet")
LISTS = f"{GA}/demo_lists.json"
N_ARM = 1500
RESAMPLE_T = 32
SEED = 20260803


def episode_features(path):
    df = pd.read_parquet(path, columns=["observation.state", "action", "episode_index"])
    ep = int(df["episode_index"].iloc[0])
    st = np.stack(df["observation.state"].to_numpy())
    ac = np.stack(df["action"].to_numpy())
    T = len(st)
    xyz = st[:, 7:10]
    width = st[:, 14]
    grip = ac[:, 11]

    # gripper close-command sign: the a11 value while the fingers are closed
    closed = width < 0.008
    close_sign = 1.0 if (closed.any() and grip[closed].mean() > 0) else -1.0
    trans = np.diff((grip == close_sign).astype(int))
    n_close = int((trans == 1).sum()) + int(grip[0] == close_sign)
    retries = max(0, n_close - 1)

    v = np.diff(xyz, axis=0)
    speed = np.linalg.norm(v, axis=1)
    path_len = float(speed.sum())
    jerk = float((np.diff(v, axis=0) ** 2).sum(axis=1).mean())
    arm = ac[:, 5:11]
    act_jitter = float((np.diff(arm, axis=0) ** 2).sum(axis=1).mean())
    core = speed[5:-5] if T > 20 else speed
    pause_frac = float((core < 0.002).mean())

    # shape descriptor: start-centered xyz resampled to RESAMPLE_T
    t_src = np.linspace(0, 1, T)
    t_dst = np.linspace(0, 1, RESAMPLE_T)
    shape = np.stack([np.interp(t_dst, t_src, xyz[:, d] - xyz[0, d]) for d in range(3)], 1)
    return {"episode_index": ep, "T": T, "retries": retries, "path_len": path_len,
            "jerk": jerk, "act_jitter": act_jitter, "pause_frac": pause_frac,
            "shape": shape.astype(np.float32).ravel()}


def main():
    lists = json.load(open(LISTS))
    region = {int(e): r for e, r in lists["region_assignment"].items()}
    files = {}
    for f in glob.glob(DATA_GLOB):
        ep = int(f.rsplit("episode_", 1)[1][:6])
        if ep in region:
            files[ep] = f
    missing = set(region) - set(files)
    assert not missing, f"{len(missing)} pool episodes without parquet"
    print(f"[style] {len(files)} pool episode files", flush=True)

    with ProcessPoolExecutor(max_workers=16) as ex:
        feats = list(ex.map(episode_features, [files[e] for e in sorted(files)],
                            chunksize=64))
    fdf = pd.DataFrame(feats).set_index("episode_index").sort_index()
    fdf["region"] = [region[e] for e in fdf.index]

    from bandit_v1 import pool as bpool
    pdf = bpool.build_pool_table(write=False).set_index("episode_index")
    fdf["category"] = pdf.loc[fdf.index, "category"]

    # consistency: distance to per-category mean shape
    shapes = np.stack(fdf["shape"].to_numpy())
    dist = np.zeros(len(fdf))
    for cat, idxs in fdf.groupby("category").indices.items():
        grp = shapes[idxs]
        ref_idx = idxs if len(idxs) >= 30 else np.arange(len(fdf))
        mu = shapes[ref_idx].mean(0)
        dist[idxs] = np.linalg.norm(grp - mu, axis=1)
    fdf["shape_dist"] = dist
    fdf = fdf.drop(columns=["shape"])

    # within-category z (fallback within-region for small cats)
    BAD = ["shape_dist", "retries", "jerk", "act_jitter", "pause_frac", "T"]
    z = pd.DataFrame(index=fdf.index, dtype=float)
    for c in BAD:
        col = fdf[c].astype(float)
        def grp_z(g):
            s = g.std(ddof=0)
            return (g - g.mean()) / (s if s > 1e-9 else 1.0)
        by_cat = col.groupby(fdf["category"]).transform(grp_z)
        small = fdf.groupby("category")["T"].transform("size") < 30
        by_reg = col.groupby(fdf["region"]).transform(grp_z)
        z[c] = np.where(small, by_reg, by_cat)
    fdf["quality"] = -z[BAD].mean(axis=1)   # high = consistent, clean, smooth, brisk

    # region-stratified top/bottom selection
    shares = fdf["region"].value_counts(normalize=True)
    hi, lo = [], []
    for r, s in shares.items():
        sub = fdf[fdf["region"] == r].sort_values("quality")
        k = int(round(N_ARM * s))
        lo += list(sub.index[:k])
        hi += list(sub.index[-k:])
    print(f"[style] style_hi n={len(hi)} style_lo n={len(lo)}", flush=True)

    rep = {"n_pool": len(fdf), "n_hi": len(hi), "n_lo": len(lo)}
    for name, ids in (("hi", hi), ("lo", lo)):
        sub = fdf.loc[ids]
        rep[name] = {
            "quality_mean": round(float(sub["quality"].mean()), 3),
            "region_share": {k: round(v, 3) for k, v in
                             sub["region"].value_counts(normalize=True).items()},
            "top_categories": Counter(sub["category"]).most_common(5),
            "raw_means": {c: round(float(sub[c].mean()), 4) for c in BAD},
        }
    rep["pool_raw_means"] = {c: round(float(fdf[c].mean()), 4) for c in BAD}
    corr = {c: round(float(np.corrcoef(fdf["quality"], fdf[c].astype(float))[0, 1]), 3)
            for c in BAD}
    rep["quality_corr_raw"] = corr

    fdf.drop(columns=[], errors="ignore").to_parquet(f"{GA}/style_features.parquet")
    json.dump({"style_hi": [int(x) for x in hi], "style_lo": [int(x) for x in lo],
               "n_arm": N_ARM, "seed": SEED, "features": BAD},
              open(f"{GA}/style_assignment.json", "w"))
    json.dump(rep, open(f"{GA}/style_report.json", "w"), default=str)
    print("[style] hi:", rep["hi"], flush=True)
    print("[style] lo:", rep["lo"], flush=True)
    print("[style] wrote style_features.parquet, style_assignment.json, style_report.json", flush=True)


if __name__ == "__main__":
    main()
