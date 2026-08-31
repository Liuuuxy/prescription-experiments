import json,os,subprocess,sys,time
# --- wait for the 32-run variance sweep to finish before touching GPU1 ---
VAR="/data/xinyua11/robomimic_runs/variance"
while True:
    n=len(json.load(open(f"{VAR}/results.json"))) if os.path.exists(f"{VAR}/results.json") else 0
    busy=subprocess.run(["pgrep","-f","run_var.py"],capture_output=True).returncode==0
    if n>=32 and not busy: break
    print(f"[cap] waiting: variance sweep {n}/32 done, busy={busy}",flush=True); time.sleep(120)
print("[cap] variance sweep clear, starting",flush=True)
P="/data/xinyua11/robomimic_runs/capability"; RES=f"{P}/results.json"
PY="/data/xinyua11/conda/envs/rmimic/bin/python"; CONC=int(os.environ.get("CONC","6"))
runs=json.load(open(f"{P}/manifest.json"))
res=json.load(open(RES)) if os.path.exists(RES) else {}
todo=[r for r in runs if r["name"] not in res]
print(f"[pool] {len(todo)} runs (conc={CONC})",flush=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4"); env.pop("PYOPENGL_PLATFORM",None)
act={}
while todo or act:
    while todo and len(act)<CONC:
        r=todo.pop(0); lf=f"{P}/log_{r['name']}.txt"
        pr=subprocess.Popen([PY,f"{P}/run_cap.py",r["name"],r["config"],r["mask"],str(r["seed"])],
                            stdout=open(lf,"w"),stderr=subprocess.STDOUT,env=env)
        act[r["name"]]=(pr,time.time()); print(f"[pool] start {r['name']}",flush=True)
    time.sleep(20)
    for n in list(act):
        pr,t0=act[n]
        if pr.poll() is not None:
            del act[n]
            fe=f"{P}/out/{n}/var_eval.json"
            if pr.returncode==0 and os.path.exists(fe):
                res[n]=json.load(open(fe)); json.dump(res,open(RES,"w"),indent=1)
                print(f"[pool] DONE {n}: test={res[n]['test']['J_deploy']:.3f}",flush=True)
            else: print(f"[pool] FAILED {n} rc={pr.returncode}",flush=True)
        elif time.time()-t0>10800: pr.kill(); del act[n]; print(f"[pool] TIMEOUT {n}",flush=True)
print("[pool] ALL DONE",flush=True)
