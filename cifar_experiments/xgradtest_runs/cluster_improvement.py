"""Per-cluster improvement: train base+200 demos drawn from EACH of the 6 gradient
clusters (k=6, same recipe/seed as all cluster analyses), 4 paired-seed rounds,
same fine-tune recipe as xarm_race.py. c1/c5 already exist as gradarm_a/b in the
Q4 race; this runs the missing c0, c2, c3, c4. Output: xgradtest/armrace/cluster_results.json"""
import json, os, sys, zlib
import numpy as np
sys.path.insert(0, "/data/xinyua11/xgradtest")
from xarm_race import (load_sketches, whiten_basis, unit, finetune_eval, ROUNDS, B,
                       RES)
import xarm_race
from sklearn.cluster import KMeans
import torch

OUT = "/data/xinyua11/xgradtest/armrace/cluster_results.json"
results = json.load(open(OUT)) if os.path.exists(OUT) else {}

raw, nrm, pool, labels = load_sketches()
Uu = unit(raw, ax=1)
P10 = whiten_basis(Uu, 10)
Uw = Uu - (Uu @ P10.T) @ P10
km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(unit(Uw, ax=1))
l = km.labels_
sizes = np.bincount(l, minlength=6)
print("cluster sizes:", sizes.tolist(), "(expect [2176,336,1893,1167,1799,629])", flush=True)

tr, te = xarm_race.cifar100(True), xarm_race.cifar100(False)
sp = json.load(open("/data/xinyua11/xgradtest/splits.json"))
base = sp["base"]
base_state = torch.load(f"/data/xinyua11/xgradtest/ckpts/step6000.pt", map_location=xarm_race.DEV)
ref = json.load(open(RES))["base_ref"]

for r in range(ROUNDS):
    for c in (0, 2, 3, 4):
        pid = f"cluster{c}_r{r}"
        if pid in results:
            print(f"{pid}: done, skip", flush=True)
            continue
        rng = np.random.RandomState(zlib.crc32(f"cluster{c}_{r}".encode()) % (2**31))
        pos = np.where(l == c)[0]
        draw = rng.choice(pos, size=min(B, len(pos)), replace=False)
        train_idx = list(base) + [int(pool[p]) for p in draw]
        accs = finetune_eval(base_state, tr, te, train_idx, 1000 + r, sp)
        row = {"arm": f"cluster{c}", "round": r,
               "delta_overall": accs["acc_overall"] - ref["acc_overall"],
               "delta_rare": accs["acc_rare"] - ref["acc_rare"],
               "delta_common": accs["acc_common"] - ref["acc_common"],
               "draw_rare_frac": float((labels[draw] < 20).mean()), **accs}
        results[pid] = row
        json.dump(results, open(OUT, "w"), indent=1)
        print(f"{pid}: d_rare {row['delta_rare']*100:+.2f} d_overall {row['delta_overall']*100:+.2f} "
              f"(draw rare frac {row['draw_rare_frac']:.2f})", flush=True)
print("DONE", flush=True)
