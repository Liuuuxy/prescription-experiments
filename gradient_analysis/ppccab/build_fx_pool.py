"""Build an fx_pool.json-equivalent for a second task, from lerobot proprio only.

Task-1's fx_pool.json (weakregion/factor_analysis/) was built with sim access;
for PickPlaceCounterToCabinet we only have the MG lerobot export, so every
per-episode feature is reconstructed from proprio + episode metadata:
  category    : parsed from the lang string ("Pick the X from the counter ...")
  bx/by/byaw  : base_position[0:2] + yaw(base_rotation quat) at t=0
  gx/gy       : GLOBAL end-effector xy at the detected grasp frame (object proxy;
                exactly how task-1's pool_features.json defined gx/gy)
  x/y         : gx-bx, gy-by  -- global-frame deltas, matching fx_pool + states.py
  side        : dominant-axis sign rule (pool.py docstring)
  layout      : base-anchor cluster id (NOT a robocasa layout id -- ground truth
                is unrecoverable from lerobot; see cluster_layouts())
  h/w         : per-category geometry copied from task-1 fx_pool's cats table
                (object-intrinsic, registry-shared); unseen categories get pool means
  cats[].sr   : 0.5 placeholder until task-2 diagnosis writes real ones

VALIDATE mode (--validate) runs the same extractor on task-1 episodes and
compares against fx_pool.json's ground-truth rows -- the conventions
(quat order, eef frame, grasp detection) are only trusted because this passes.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = "/data/xinyua11/robocasa"
sys.path.insert(0, REPO)

T1_LR = ("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
         "PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot")
T2_LR = ("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
         "PickPlaceCounterToCabinet/20250819/mg/demo/2025-08-20-21-56-25/lerobot")
T1_FX = f"{REPO}/weakregion/factor_analysis/fx_pool.json"
OUT = f"{REPO}/gradient_analysis/ppccab/fx_pool_ppccab.json"


def yaw_from_quat(q, order):
    """order='xyzw' or 'wxyz'."""
    if order == "xyzw":
        x, y, z, w = q
    else:
        w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def parse_category(lang):
    """'Pick the salt and pepper shaker from the counter...' -> salt_and_pepper_shaker"""
    s = lang.lower()
    assert s.startswith("pick the "), lang
    obj = s[len("pick the "):].split(" from the ")[0]
    return obj.replace(" ", "_")


def detect_grasp_frame(st, deep=0.75, run_need=12, open_lvl=0.95):
    """Last fully-open frame before the first sustained closing that FOLLOWS the
    initial open plateau. Tuned against task-1 pool_features.json ground truth
    (300-episode sample): 93.7% of derived gx/gy within 5cm, median grasp-frame
    error 1 frame, p90 5. The ~6% tail is regrasp episodes whose first close is
    not the recorded one -- symmetric noise wrt any arm split built from these
    features. Episodes can START gripper-closed, hence the initial-plateau skip."""
    gap = np.abs(st[:, 14]) + np.abs(st[:, 15])
    o, c = np.percentile(gap, 95), gap.min()
    g = (gap - c) / (o - c + 1e-9)
    run, t0 = 0, 0
    for t in range(len(g)):
        run = run + 1 if g[t] >= 0.9 else 0
        if run >= 5:
            t0 = t - 4
            break
    run, t_closed = 0, None
    for t in range(t0, len(g)):
        run = run + 1 if g[t] < deep else 0
        if run >= run_need:
            t_closed = t - run_need + 1
            break
    if t_closed is None:
        return int(t0 + np.argmin(g[t0:]))
    ob = np.where(g[t0:t_closed] >= open_lvl)[0]
    return int(t0 + ob[-1]) if len(ob) else max(t0, t_closed - 10)


def episode_features(pq_path, quat_order, eef_frame):
    df = pd.read_parquet(pq_path, columns=["observation.state"])
    st = np.stack(df["observation.state"].values)          # (T,16)
    bx, by = float(st[0, 0]), float(st[0, 1])
    byaw = yaw_from_quat(st[0, 3:7], quat_order)
    gf = detect_grasp_frame(st)
    b_t, q_t, e_t = st[gf, 0:3], st[gf, 3:7], st[gf, 7:10]
    if eef_frame == "base":
        yw = yaw_from_quat(q_t, quat_order)
        c, s = np.cos(yw), np.sin(yw)
        gx = float(b_t[0] + c * e_t[0] - s * e_t[1])
        gy = float(b_t[1] + s * e_t[0] + c * e_t[1])
    else:                                                   # global delta
        gx, gy = float(b_t[0] + e_t[0]), float(b_t[1] + e_t[1])
    return dict(bx=bx, by=by, byaw=byaw, gx=gx, gy=gy, grasp_frame=gf, len=len(st))


def side_of(x, y):
    v = x if abs(x) >= abs(y) else y
    return 1 if v > 0 else -1


def iter_parquets(lr):
    files = sorted(glob.glob(f"{lr}/data/*/episode_*.parquet"))
    for f in files:
        yield int(os.path.basename(f)[8:-8]), f


def validate(n_sample=300):
    fx = json.load(open(T1_FX))
    F = {r[0]: dict(zip(fx["fields"], r)) for r in fx["rows"]}
    cats = fx["cats"]
    files = dict(iter_parquets(T1_LR))
    rng = np.random.default_rng(0)
    ids = rng.choice(sorted(set(F) & set(files)), n_sample, replace=False)
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in open(f"{T1_LR}/meta/episodes.jsonl")}
    best = None
    for quat_order in ("xyzw", "wxyz"):
        for eef_frame in ("base", "global"):
            dx, dy, catok, sideok = [], [], 0, 0
            for i in ids:
                f = episode_features(files[i], quat_order, eef_frame)
                t = F[i]
                dx.append(abs((f["gx"] - f["bx"]) - t["x"]))
                dy.append(abs((f["gy"] - f["by"]) - t["y"]))
                cat_lang = parse_category(meta[i]["tasks"][0])
                from bandit_v1 import categories
                if categories.canonical_category(cat_lang) == \
                   categories.canonical_category(cats[t["cat"]]["name"]):
                    catok += 1
                if side_of(f["gx"] - f["bx"], f["gy"] - f["by"]) == t["side"]:
                    sideok += 1
            med = float(np.median(dx) + np.median(dy))
            frac5 = float(np.mean([a < 0.05 and b < 0.05 for a, b in zip(dx, dy)]))
            print(f"[validate] quat={quat_order} eef={eef_frame}: "
                  f"median |dx|+|dy|={med:.3f} within5cm={frac5:.2%} "
                  f"cat={catok}/{len(ids)} side={sideok}/{len(ids)}", flush=True)
            if best is None or med < best[0]:
                best = (med, quat_order, eef_frame, frac5)
    print(f"[validate] BEST: quat={best[1]} eef={best[2]} within5cm={best[3]:.2%}", flush=True)
    return best


def cluster_layouts(rows, res=0.25):
    """Deterministic parking-grid cell id from (bx,by) rounded to `res` m.
    NOT a robocasa layout id: on task-1, position clustering recovers true
    layouts at only ~63% purity (several layouts share parking coordinates),
    so we don't pretend to. The cell is a scene-context key for exact-matching
    -- two episodes in the same cell present the policy the same egocentric
    parking geometry, which is what matching needs; both arms of any matched
    pair use the same key, so residual impurity is symmetric."""
    cells = {}
    for r in sorted(rows, key=lambda r: r["i"]):
        key = (round(r["bx"] / res), round(r["by"] / res))
        r["layout"] = cells.setdefault(key, len(cells))
    return len(cells)


def build(quat_order, eef_frame):
    from bandit_v1 import categories
    fx1 = json.load(open(T1_FX))
    hw = {c["name"]: (c["h"], c["w"]) for c in fx1["cats"]}
    h_mean = float(np.mean([c["h"] for c in fx1["cats"]]))
    w_mean = float(np.mean([c["w"] for c in fx1["cats"]]))
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in open(f"{T2_LR}/meta/episodes.jsonl")}
    rows = []
    for i, f in iter_parquets(T2_LR):
        ft = episode_features(f, quat_order, eef_frame)
        cat = categories.canonical_category(parse_category(meta[i]["tasks"][0]))
        x, y = ft["gx"] - ft["bx"], ft["gy"] - ft["by"]
        rows.append(dict(i=i, cat_name=cat, x=round(x, 4), y=round(y, 4),
                         side=side_of(x, y), len=ft["len"], bx=ft["bx"], by=ft["by"],
                         r=round(float(np.hypot(x, y)), 4)))
        if len(rows) % 1000 == 0:
            print(f"[build] {len(rows)} episodes...", flush=True)
    n_anchor = cluster_layouts(rows)
    names = sorted({r["cat_name"] for r in rows})
    cat_idx = {n: k for k, n in enumerate(names)}
    n_unseen = sum(1 for n in names if n not in hw)
    cats = [{"name": n, "sr": 0.5, "n": sum(r["cat_name"] == n for r in rows),
             "h": hw.get(n, (h_mean, w_mean))[0], "w": hw.get(n, (h_mean, w_mean))[1]}
            for n in names]
    fields = ["i", "cat", "h", "w", "layout", "r", "x", "y", "side", "ambig", "len"]
    out_rows = [[r["i"], cat_idx[r["cat_name"]], cats[cat_idx[r["cat_name"]]]["h"],
                 cats[cat_idx[r["cat_name"]]]["w"], r["layout"], r["r"], r["x"],
                 r["y"], r["side"], 0, r["len"]] for r in rows]
    out = {"fields": fields, "rows": out_rows, "cats": cats,
           "provenance": {"builder": "gradient_analysis/ppccab/build_fx_pool.py",
                          "date": "2026-08-17", "quat_order": quat_order,
                          "eef_frame": eef_frame,
                          "layout_semantics": f"base-anchor cluster (res 0.35m), {n_anchor} anchors",
                          "sr_source": "PLACEHOLDER 0.5 until task-2 diagnosis",
                          "hw_source": "task-1 fx_pool cats (object-intrinsic)",
                          "n_unseen_categories": n_unseen}}
    json.dump(out, open(OUT, "w"))
    print(f"[build] wrote {OUT}: {len(out_rows)} rows, {len(cats)} cats "
          f"({n_unseen} unseen h/w->mean), {n_anchor} anchor clusters", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--quat", default=None)
    ap.add_argument("--eef", default=None)
    a = ap.parse_args()
    if a.validate:
        best = validate()
        if a.build:
            build(best[1], best[2])
    elif a.build:
        build(a.quat or "xyzw", a.eef or "base")
