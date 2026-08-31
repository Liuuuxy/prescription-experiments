"""ADPIL surface pool driver — adapted from vardecomp2/pool_vd2.py.
Env: GPU (CUDA device), MANIFEST, RES, CONC (default 4). RNN runs are slower than
MLP, so timeout raised to 6h and default concurrency lowered to 4."""
import json,os,subprocess,time
P="/data/xinyua11/robomimic_runs/adpil/surface"; RES=os.environ.get("RES",f"{P}/results.json")
PY="/data/xinyua11/conda/envs/rmimic/bin/python"; CONC=int(os.environ.get("CONC","4"))
runs=json.load(open(os.environ.get("MANIFEST",f"{P}/manifest.json")))
res=json.load(open(RES)) if os.path.exists(RES) else {}
todo=[r for r in runs if r["name"] not in res]
print(f"[pool] {len(todo)} runs (conc={CONC})",flush=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ.get("GPU","0"),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4"); env.pop("PYOPENGL_PLATFORM",None)
act={}
while todo or act:
    while todo and len(act)<CONC:
        r=todo.pop(0); lf=f"{P}/log_{r['name']}.txt"
        pr=subprocess.Popen([PY,f"{P}/run_surface.py",r["name"],r["config"],r["mask"],str(r["seed"])],
                            stdout=open(lf,"w"),stderr=subprocess.STDOUT,env=env)
        act[r["name"]]=(pr,time.time()); print(f"[pool] start {r['name']}",flush=True)
    time.sleep(20)
    for n in list(act):
        pr,t0=act[n]
        if pr.poll() is not None:
            del act[n]
            fe=f"{P}/out/{n}/var_eval.json"
            if pr.returncode==0 and os.path.exists(fe):
                full=json.load(open(fe)); res[n]={k:full[k] for k in ("name","config","mask","seed","best_epoch") if k in full}
                res[n]["test_h500"]=full["test"]["h500"]; res[n]["probe_h500"]=full["probe"]["h500"]["J_deploy"]
                json.dump(res,open(RES,"w"),indent=1)
                print(f"[pool] DONE {n}: test={res[n]['test_h500']['J_deploy']:.3f}",flush=True)
            else: print(f"[pool] FAILED {n} rc={pr.returncode}",flush=True)
        elif time.time()-t0>21600: pr.kill(); del act[n]; print(f"[pool] TIMEOUT {n}",flush=True)
print("[pool] ALL DONE",flush=True)
