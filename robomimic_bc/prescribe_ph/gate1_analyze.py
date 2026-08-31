"""Gate 1 unblinding, per PREREG_PH_BENCHMARK.md. REFUSES to run without MDE.json.
Reads sealed E_test for all screen arms; reports deployment-weighted advantages
over b_div (mean of the two null draws at matched seed), per profile, with
paired-seed replication check and the pre-registered gate verdicts.
All verdict strings are CONDITIONAL on the data (no pre-written conclusions).
"""
import json, os, time
import numpy as np

P = "/data/xinyua11/robomimic_runs/prescribe_ph"
assert os.path.exists(f"{P}/MDE.json"), "run mde_compute.py first (prereg order)"
mde = json.load(open(f"{P}/MDE.json"))
X = mde["X"]
PROFILES = ["balanced", "starved_xlo_yhi", "starved_xhi_ylo"]
REGIONS = ["xlo_ylo", "xlo_yhi", "xhi_ylo", "xhi_yhi"]
ARMS = [f"add_{r}" for r in REGIONS] + ["add_cov", "add_pfail"]

def jt(name, field="J_deploy"):
    fp = f"{P}/out/{name}/fixed_eval_test.json"
    if not os.path.exists(fp): return None
    return json.load(open(fp))[field]

print(f"MDE X = {X*100:+.2f}pp (frozen {mde['timestamp']})\n")
gate1_hits = []
for prof in PROFILES:
    print(f"== {prof} ==")
    for s in (0, 1):
        base = jt(f"{prof}_D0_s{s}")
        div = np.mean([jt(f"{prof}_add_div{k}_s{s}") for k in (1, 2)])
        print(f"  seed {s}: J(D0)={base:.3f}  J(div)={div:.3f}  div-delta={div-base:+.3f}")
    for arm in ARMS:
        advs = []
        for s in (0, 1):
            j = jt(f"{prof}_{arm}_s{s}")
            if j is None: continue
            div = np.mean([jt(f"{prof}_add_div{k}_s{s}") for k in (1, 2)])
            advs.append(j - div)
        if not advs: continue
        m = np.mean(advs)
        rep = (len(advs) == 2 and np.sign(advs[0]) == np.sign(advs[1]))
        flag = ""
        if m > X and rep:
            flag = "  <-- GATE-1 HIT (adv > X, sign-replicated)"
            gate1_hits.append((prof, arm, m))
        print(f"  {arm:12s} adv vs div: " + " ".join(f"{a*100:+.1f}" for a in advs)
              + f"  mean {m*100:+.1f}pp{flag}")
    print()
if gate1_hits:
    print("GATE 1 (exploratory screen): OPPORTUNITY DETECTED —")
    for prof, arm, m in gate1_hits:
        print(f"  {prof}/{arm}: +{m*100:.1f}pp > X={X*100:.1f}pp, both seeds same sign")
    print("Next per prereg: confirmatory seed 2 + remaining rotations; then Gate 2.")
else:
    print("GATE 1 (exploratory screen): NO allocation beats diverse by > X with "
          "sign replication. Per prereg: run confirmatory seeds on the nearest "
          "misses ONLY if any mean adv > X/2; otherwise STOP before Square/pi0.")
json.dump({"X": X, "hits": [{"profile": p, "arm": a, "adv": float(m)} for p, a, m in gate1_hits],
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
          open(f"{P}/gate1_result.json", "w"), indent=1)
