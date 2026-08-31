"""Phase-1 setup for the Can-PH condition-region benchmark.

- Copies can/ph low_dim_v141.hdf5 -> can_ph_work.hdf5 (originals stay read-only).
- Defines the 2x2 region grid from DATASET medians of the can's initial (x, y).
- Writes region reference masks + pilot D0-only masks (balanced 40/60/80).
- All draw seeds via zlib.crc32 (never python hash()).

Regions (grid frozen here, reused everywhere incl. eval stratification):
  xlo_ylo, xlo_yhi, xhi_ylo, xhi_yhi   (x/y vs dataset median)
"""
import h5py, json, shutil, zlib
import numpy as np

SRC = "/data/xinyua11/robomimic_datasets/can/ph/low_dim_v141.hdf5"
WORK = "/data/xinyua11/robomimic_runs/prescribe_ph/can_ph_work.hdf5"
REGIONS = ["xlo_ylo", "xlo_yhi", "xhi_ylo", "xhi_yhi"]

def rng_for(tag):
    return np.random.RandomState(zlib.crc32(tag.encode()) % (2**31))

shutil.copy(SRC, WORK)
f = h5py.File(WORK, "r+")
demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
init = np.array([f["data"][d]["obs"]["object"][0][:2] for d in demos])
qx, qy = float(np.median(init[:, 0])), float(np.median(init[:, 1]))
lab = (init[:, 0] > qx).astype(int) * 2 + (init[:, 1] > qy).astype(int)
region_of = {d: REGIONS[l] for d, l in zip(demos, lab)}

if "mask" in f: del f["mask"]
g = f.create_group("mask")
def write_mask(name, dlist):
    g.create_dataset(name, data=np.array(sorted(dlist, key=lambda d: int(d.split("_")[1])), dtype="S"))

by_region = {r: [d for d in demos if region_of[d] == r] for r in REGIONS}
for r in REGIONS:
    write_mask(f"region_{r}", by_region[r])

# pilot D0-only balanced draws (one draw per size; training seeds vary in the runs)
pilot = {}
for n in (40, 60, 80):
    rng = rng_for(f"ph_pilot_D0_{n}")
    picks = []
    for r in REGIONS:
        picks += list(rng.choice(by_region[r], n // 4, replace=False))
    write_mask(f"pilot_D0_{n}", picks)
    pilot[n] = sorted(picks, key=lambda d: int(d.split("_")[1]))

meta = {
    "grid": {"qx": qx, "qy": qy, "labels": REGIONS,
             "rule": "2*(x>qx)+(y>qy) over obs/object[0][:2]"},
    "region_counts": {r: len(by_region[r]) for r in REGIONS},
    "region_of": region_of,
    "pilot_D0": {str(k): v for k, v in pilot.items()},
    "lengths": {d: int(f["data"][d].attrs["num_samples"]) for d in demos},
}
f.close()
json.dump(meta, open("/data/xinyua11/robomimic_runs/prescribe_ph/regions.json", "w"), indent=1)
print("grid qx=%.4f qy=%.4f  counts=%s" % (qx, qy, meta["region_counts"]))
print("pilot masks written:", [f"pilot_D0_{n}" for n in (40, 60, 80)])
