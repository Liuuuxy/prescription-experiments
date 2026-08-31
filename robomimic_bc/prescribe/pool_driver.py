"""Run manifest entries with bounded concurrency; harvest Success_Rate per run.
Usage: pool_driver.py [filter_substring ...]  (runs entries whose name contains any)"""
import json, os, re, subprocess, sys, time

P = "/data/xinyua11/robomimic_runs/prescribe"
RES = f"{P}/results.json"
PY = "/data/xinyua11/conda/envs/rmimic/bin/python"
CONC = 6
runs = json.load(open(f"{P}/runs_manifest.json"))
pats = sys.argv[1:] or [""]
todo = [r for r in runs if any(p in r["name"] for p in pats)]
results = json.load(open(RES)) if os.path.exists(RES) else {}
todo = [r for r in todo if r["name"] not in results]
print(f"[pool] {len(todo)} runs to do (conc={CONC})", flush=True)

env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
env.pop("PYOPENGL_PLATFORM", None)
active = {}
def harvest(name, logf):
    srs = [float(m) for m in re.findall(r'"Success_Rate": ([0-9.]+)', open(logf).read())]
    spec = next(r for r in runs if r["name"] == name)
    results[name] = {**{k: spec[k] for k in ("arm", "B") if k in spec},
                     "seed": spec.get("seed"), "success_rates": srs}
    json.dump(results, open(RES, "w"), indent=1)
    print(f"[pool] DONE {name}: SR={srs}", flush=True)

while todo or active:
    while todo and len(active) < CONC:
        r = todo.pop(0)
        logf = f"{P}/log_{r['name']}.txt"
        pr = subprocess.Popen([PY, f"{P}/rmimic_run.py", r["name"], r["mask"], str(r.get("seed", 0))],
                              stdout=open(logf, "w"), stderr=subprocess.STDOUT, env=env)
        active[r["name"]] = (pr, logf, time.time())
        print(f"[pool] start {r['name']}", flush=True)
    time.sleep(20)
    for n in list(active):
        pr, logf, t0 = active[n]
        if pr.poll() is not None:
            del active[n]
            if pr.returncode == 0: harvest(n, logf)
            else: print(f"[pool] FAILED {n} rc={pr.returncode} — see {logf}", flush=True)
        elif time.time() - t0 > 7200:
            pr.kill(); del active[n]; print(f"[pool] TIMEOUT {n}", flush=True)
print("[pool] ALL DONE", flush=True)
