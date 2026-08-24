"""Arm definitions for the robot loss-reward bandit (owner "1", 2026-08-05).

12 arms as episode-id lists: 6 gradient clusters (k=6 k-means on whitened
pool sketches, same SEED as gradarm_cluster -> reproduces the frozen labels),
style_hi/style_lo, the 3 condition regions, and random (empty list = draw's
uniform-pool path). Output: gradient_analysis/ucb_robot/arms.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis")
from gradarm_cluster import load_merged, whiten, KS, SEED  # noqa: E402

OUT = "/data/xinyua11/robocasa/gradient_analysis/ucb_robot"


def main():
    from sklearn.cluster import KMeans
    os.makedirs(OUT, exist_ok=True)
    eps, X, norms, reg = load_merged()
    Xw, _ = whiten(X)
    km = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(Xw)
    arms = {}
    for c in range(6):
        arms[f"gc{c}"] = [int(eps[i]) for i in np.where(km.labels_ == c)[0]]
    sa = json.load(open("/data/xinyua11/robocasa/gradient_analysis/style_assignment.json"))
    arms["style_hi"] = [int(x) for x in sa["style_hi"]]
    arms["style_lo"] = [int(x) for x in sa["style_lo"]]
    lists = json.load(open("/data/xinyua11/robocasa/gradient_analysis/demo_lists.json"))
    for r in ["tall_vessel_grasp_fail", "mid_band", "easy_band"]:
        arms[r] = [int(e) for e, rr in lists["region_assignment"].items() if rr == r]
    arms["random"] = []   # sentinel: draw's uniform-pool path
    json.dump(arms, open(f"{OUT}/arms.json", "w"))
    print("arms:", {a: len(v) for a, v in arms.items()}, flush=True)


if __name__ == "__main__":
    main()
