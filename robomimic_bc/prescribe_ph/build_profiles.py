"""Build screen profiles + arm masks at the frozen D0 size N (PREREG_PH_BENCHMARK.md).

Phase A (build_profiles.py N):
  profiles: balanced + starved_<r> for 2 pre-registered rotations (screen set)
  masks per profile: D0, 4 one-region adds, 2 independent b_div adds, b_cov add,
                     4 removals. Appends run entries (seeds 0,1) to screen manifest.
Phase B (build_profiles.py N pfail):
  reads each profile's D0-baseline PROBE results (both seeds), computes
  b_pfail proportional to q_r*(1-Jhat_r), writes {prof}_add_pfail masks + runs.
All draws crc32-seeded. Wells = region demos not in that profile's D0.
"""
import h5py, json, sys, zlib
import numpy as np

P = "/data/xinyua11/robomimic_runs/prescribe_ph"
N = int(sys.argv[1])
PHASE = sys.argv[2] if len(sys.argv) > 2 else "A"
B = 24
meta = json.load(open(f"{P}/regions.json"))
REGIONS = meta["grid"]["labels"]
by_region = {r: sorted([d for d, rr in meta["region_of"].items() if rr == r],
                       key=lambda d: int(d.split("_")[1])) for r in REGIONS}
q = json.load(open(f"{P}/deploy_shares.json"))["natural_reset_shares"]
# screen rotations (pre-registered): starve one x-lo and one x-hi region
SCREEN_STARVED = ["xlo_yhi", "xhi_ylo"]

def rng_for(tag):
    return np.random.RandomState(zlib.crc32(tag.encode()) % (2**31))

def rounded_alloc(weights):
    """largest-remainder rounding of B*weights to integers summing to B"""
    w = np.array([weights[r] for r in REGIONS], dtype=float)
    w = w / w.sum()
    raw = B * w
    base = np.floor(raw).astype(int)
    rem = B - base.sum()
    order = np.argsort(-(raw - base))
    base[order[:rem]] += 1
    return {r: int(b) for r, b in zip(REGIONS, base)}

def profile_counts(kind):
    if kind == "balanced":
        return {r: N // 4 for r in REGIONS}
    starved = kind.replace("starved_", "")
    s = max(2, round(N / 20))
    rest, k = N - s, 3
    counts = {}
    others = [r for r in REGIONS if r != starved]
    for i, r in enumerate(others):
        counts[r] = rest // k + (1 if i < rest % k else 0)
    counts[starved] = s
    return counts

def draw_D0(prof, counts):
    picks = []
    for r in REGIONS:
        rng = rng_for(f"screen_{prof}_D0_{r}")
        picks += list(rng.choice(by_region[r], counts[r], replace=False))
    return sorted(picks, key=lambda d: int(d.split("_")[1]))

def alloc_draw(prof, tag, alloc, d0):
    picks = []
    for r in REGIONS:
        well = [d for d in by_region[r] if d not in d0]
        assert len(well) >= alloc[r], f"{prof}/{tag}: well {r} has {len(well)} < {alloc[r]}"
        rng = rng_for(f"screen_{prof}_{tag}_{r}")
        picks += list(rng.choice(well, alloc[r], replace=False))
    return picks

f = h5py.File(f"{P}/can_ph_work.hdf5", "r+")
g = f["mask"]
def write_mask(name, dlist):
    if name in g: del g[name]
    g.create_dataset(name, data=np.array(sorted(dlist, key=lambda d: int(d.split("_")[1])), dtype="S"))

manifest_path = f"{P}/screen_manifest.json"
try: manifest = json.load(open(manifest_path))
except FileNotFoundError: manifest = []
have = {m["name"] for m in manifest}
def add_runs(mask, arm, prof):
    for s in (0, 1):
        nm = f"{mask}_s{s}"
        if nm not in have:
            manifest.append({"name": nm, "mask": mask, "seed": s, "arm": arm, "profile": prof, "B": B})

profiles = ["balanced"] + [f"starved_{r}" for r in SCREEN_STARVED]
spec = {}
for prof in profiles:
    counts = profile_counts(prof)
    d0 = draw_D0(prof, counts)
    spec[prof] = {"counts": counts, "D0": d0}
    if PHASE == "A":
        write_mask(f"{prof}_D0", d0)
        add_runs(f"{prof}_D0", "D0", prof)
        for r in REGIONS:                                   # one-region adds
            adds = alloc_draw(prof, f"add_{r}", {rr: (B if rr == r else 0) for rr in REGIONS}, d0)
            write_mask(f"{prof}_add_{r}", d0 + adds)
            add_runs(f"{prof}_add_{r}", f"add_{r}", prof)
        for k in (1, 2):                                    # independent diverse nulls
            adds = alloc_draw(prof, f"add_div{k}", rounded_alloc(q), d0)
            write_mask(f"{prof}_add_div{k}", d0 + adds)
            add_runs(f"{prof}_add_div{k}", f"add_div{k}", prof)
        # coverage water-filling: greedy argmin of sum_r((n_r+b_r)/(N+B)-q_r)^2
        b = {r: 0 for r in REGIONS}
        for _ in range(B):
            best = min(REGIONS, key=lambda r: ((counts[r] + b[r] + 1) / (N + B) - q[r]) ** 2
                                              - ((counts[r] + b[r]) / (N + B) - q[r]) ** 2)
            b[best] += 1
        adds = alloc_draw(prof, "add_cov", b, d0)
        write_mask(f"{prof}_add_cov", d0 + adds)
        add_runs(f"{prof}_add_cov", "add_cov", prof)
        spec[prof]["b_cov"] = b
        for r in REGIONS:                                   # removals
            keep = [d for d in d0 if meta["region_of"][d] != r]
            write_mask(f"{prof}_rm_{r}", keep)
            add_runs(f"{prof}_rm_{r}", f"rm_{r}", prof)
    elif PHASE == "pfail":
        res = json.load(open(f"{P}/results.json"))
        Jr = {r: np.mean([res[f"{prof}_D0_s{s}"]["J_region"][r] for s in (0, 1)]) for r in REGIONS}
        w = {r: q[r] * (1 - Jr[r]) for r in REGIONS}
        if sum(w.values()) == 0: w = dict(q)
        b = rounded_alloc(w)
        adds = alloc_draw(prof, "add_pfail", b, d0)
        write_mask(f"{prof}_add_pfail", d0 + adds)
        add_runs(f"{prof}_add_pfail", "add_pfail", prof)
        spec[prof]["b_pfail"] = b
        spec[prof]["Jhat_probe"] = {r: float(Jr[r]) for r in REGIONS}
f.close()
json.dump(manifest, open(manifest_path, "w"), indent=1)
out = f"{P}/screen_spec_{PHASE}.json"
json.dump(spec, open(out, "w"), indent=1, default=str)
print(f"phase {PHASE}: {len(manifest)} total screen runs; profiles: "
      + "; ".join(f"{p}:{spec[p]['counts']}" for p in profiles))
