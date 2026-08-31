"""Build the Square heterogeneity screen at frozen N (PREREG_SQUARE_HET.md).
Usage: build_screen_sq.py N   -> masks (balanced D0, 4 add, 4 rm, div1, div2) +
screen manifest (11 arms x seeds 0,1,2 = 33 runs)."""
import h5py, json, sys, zlib
import numpy as np
P="/data/xinyua11/robomimic_runs/prescribe_sq"
N=int(sys.argv[1]); B=24
meta=json.load(open(f"{P}/regions.json")); REG=meta["grid"]["labels"]; rof=meta["region_of"]
q=json.load(open(f"{P}/deploy_shares.json"))["natural_reset_shares"]
def rng_for(tag): return np.random.RandomState(zlib.crc32(tag.encode())%(2**31))
f=h5py.File(f"{P}/square_ph_work.hdf5","r+"); g=f["mask"]
by={r:sorted([d for d,rr in rof.items() if rr==r],key=lambda d:int(d.split("_")[1])) for r in REG}
def wm(name,dl):
    if name in g: del g[name]
    g.create_dataset(name,data=np.array(sorted(dl,key=lambda d:int(d.split("_")[1])),dtype="S"))
d0=[]
for r in REG:
    rng=rng_for(f"sq_screen_D0_{r}")
    d0+=list(rng.choice(by[r],N//4,replace=False))
wm("screen_D0",d0)
def draw(tag,alloc):
    picks=[]
    for r in REG:
        well=[d for d in by[r] if d not in d0]
        assert len(well)>=alloc[r],f"{tag}: well {r} {len(well)}<{alloc[r]}"
        rng=rng_for(f"sq_screen_{tag}_{r}")
        picks+=list(rng.choice(well,alloc[r],replace=False))
    return picks
man=[]
def add_runs(mask,arm):
    for s in (0,1,2): man.append({"name":f"{mask}_s{s}","mask":mask,"seed":s,"arm":arm,"B":B})
add_runs("screen_D0","D0")
for r in REG:
    wm(f"screen_add_{r}",d0+draw(f"add_{r}",{rr:(B if rr==r else 0) for rr in REG}))
    add_runs(f"screen_add_{r}",f"add_{r}")
    wm(f"screen_rm_{r}",[d for d in d0 if rof[d]!=r])
    add_runs(f"screen_rm_{r}",f"rm_{r}")
qa=np.array([q[r] for r in REG]); raw=B*qa/qa.sum(); base=np.floor(raw).astype(int)
order=np.argsort(-(raw-base)); base[order[:B-base.sum()]]+=1
div={r:int(b) for r,b in zip(REG,base)}
for k in (1,2):
    wm(f"screen_add_div{k}",d0+draw(f"add_div{k}",div))
    add_runs(f"screen_add_div{k}",f"add_div{k}")
f.close()
json.dump(man,open(f"{P}/screen_manifest.json","w"),indent=1)
print(f"N={N}: {len(man)} screen runs; b_div={div}")
