"""Prescription sandbox on robomimic Can-MH: build D0/W split + all run masks.

Can-MH: 300 demos, 6 operators (2 better / 2 okay / 2 worse), ground-truth
execution-quality heterogeneity. Framing: D0 = the operator's existing dataset
(100 random demos); W = the 200 not-yet-collected demos. An arm = a rule for
choosing B new demos from W (better/okay/worse operators, or random). Every
(arm, B, seed) gets a mask written into a WORKING COPY of the hdf5 so training
just selects a filter key. Paired seeds: the same seed s shares D0 draw across
all arms; collection draws are seeded per (arm, B, s).
"""
import json, shutil, zlib
import h5py
import numpy as np

SRC = "/data/xinyua11/robomimic_datasets/can/mh/low_dim_v141.hdf5"
WORK = "/data/xinyua11/robomimic_runs/prescribe/can_mh_work.hdf5"
OUT = "/data/xinyua11/robomimic_runs/prescribe/runs_manifest.json"
BUDGETS = [25, 50, 100]
SEEDS = [0, 1, 2]

shutil.copy(SRC, WORK)
with h5py.File(WORK, "a") as f:
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    masks = {k: [x.decode() for x in f["mask"][k][:]] for k in f["mask"]}
    print("masks available:", {k: len(v) for k, v in masks.items()})
    groups = {g: sorted(masks[g], key=lambda s: int(s.split("_")[1]))
              for g in ("better", "okay", "worse") if g in masks}
    assert all(len(v) == 100 for v in groups.values()), {k: len(v) for k, v in groups.items()}

    rng0 = np.random.RandomState(20260825)
    D0 = sorted(rng0.choice(demos, 100, replace=False), key=lambda s: int(s.split("_")[1]))
    W = [d for d in demos if d not in set(D0)]
    wells = {g: [d for d in groups[g] if d in set(W)] for g in groups}
    wells["random"] = W
    print("D0=100, W=200 | wells:", {k: len(v) for k, v in wells.items()})

    def put(name, lst):
        if f"mask/{name}" in f: del f[f"mask/{name}"]
        f.create_dataset(f"mask/{name}", data=np.array(lst, dtype="S"))

    put("D0", D0)
    runs = [{"name": f"D0only_s{s}", "mask": "D0", "arm": "null", "B": 0, "seed": s} for s in SEEDS]
    for B in BUDGETS:
        for arm, well in wells.items():
            for s in SEEDS:
                if B > len(well): continue
                r = np.random.RandomState(zlib.crc32(f"{arm}_{B}_{s}".encode()) % (2**31))
                pick = list(r.choice(well, B, replace=False))
                mname = f"run_{arm}_B{B}_s{s}"
                put(mname, D0 + pick)
                runs.append({"name": mname, "mask": mname, "arm": arm, "B": B, "seed": s})
json.dump(runs, open(OUT, "w"), indent=1)
print(f"wrote {len(runs)} run specs -> {OUT}")
