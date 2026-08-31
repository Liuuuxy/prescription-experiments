"""(A) trajectory-feature AUCs vs 'worse' labels; (B) create D0_bo mask (D0 minus
worse-operator demos) for a clean reference policy."""
import json
import h5py
import numpy as np

P = "/data/xinyua11/robomimic_runs/prescribe"
with h5py.File(f"{P}/can_mh_work.hdf5", "a") as f:
    D0 = [x.decode() for x in f["mask/D0"][:]]
    groups = {g: set(x.decode() for x in f[f"mask/{g}"][:]) for g in ("better", "okay", "worse")}
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    W = [d for d in demos if d not in set(D0)]
    feats = {}
    for d in W:
        g = f[f"data/{d}"]
        act = g["actions"][:]
        eef = g["obs/robot0_eef_pos"][:]
        T = len(act)
        path = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum())
        jerk = float(np.abs(np.diff(act[:, :6], n=2, axis=0)).mean())
        amag = float(np.abs(act[:, :6]).mean())
        feats[d] = {"T": T, "path": path, "jerk": jerk, "amag": amag,
                    "group": next(k for k, s in groups.items() if d in s)}
    d0_bo = [d for d in D0 if d not in groups["worse"]]
    if "mask/D0_bo" in f: del f["mask/D0_bo"]
    f.create_dataset("mask/D0_bo", data=np.array(d0_bo, dtype="S"))
    print(f"D0_bo mask: {len(d0_bo)} demos (D0 minus {len(D0)-len(d0_bo)} worse-operator demos)")

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
json.dump(feats, open(f"{P}/w_feats.json", "w"))
print("\ntrajectory-feature AUC(worse > rest):")
for k in ("T", "path", "jerk", "amag"):
    w = [v[k] for v in feats.values() if v["group"] == "worse"]
    r = [v[k] for v in feats.values() if v["group"] != "worse"]
    print(f"  {k:5s}: {auc(w, r):.3f}")
