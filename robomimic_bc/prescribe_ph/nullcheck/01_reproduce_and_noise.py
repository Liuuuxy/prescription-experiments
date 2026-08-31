import json, os, itertools
import numpy as np
from scipy import stats

P = "/data/xinyua11/robomimic_runs/prescribe_ph"
PROFILES = ["balanced", "starved_xlo_yhi", "starved_xhi_ylo"]
REGIONS = ["xlo_ylo", "xlo_yhi", "xhi_ylo", "xhi_yhi"]
ARMS = [f"add_{r}" for r in REGIONS] + ["add_cov", "add_pfail"]
ALL8 = ARMS + ["add_div1", "add_div2"]          # exchangeable set: all 104-demo additions
X_FROZEN = json.load(open(f"{P}/MDE.json"))["X"]

def jt(name, field="J_deploy"):
    return json.load(open(f"{P}/out/{name}/fixed_eval_test.json"))[field]
def succ(name):
    return np.array(json.load(open(f"{P}/out/{name}/fixed_eval_test.json"))["successes"])

# ---------- observed table: J[profile][seed][arm] ----------
J = {p: {s: {a: jt(f"{p}_{a}_s{s}") for a in ALL8} for s in (0,1)} for p in PROFILES}
D0 = {p: {s: jt(f"{p}_D0_s{s}") for s in (0,1)} for p in PROFILES}

def hits_from(Jtab, div_labels, X):
    """div_labels: dict profile -> (name1,name2). Returns list of (prof,arm,mean_adv)."""
    out = []
    for p in PROFILES:
        d1, d2 = div_labels[p]
        armset = [a for a in ALL8 if a not in (d1, d2)]
        for a in armset:
            advs = [Jtab[p][s][a] - 0.5*(Jtab[p][s][d1] + Jtab[p][s][d2]) for s in (0,1)]
            m = float(np.mean(advs))
            if m > X and np.sign(advs[0]) == np.sign(advs[1]):
                out.append((p, a, m))
    return out

obs_hits = hits_from(J, {p: ("add_div1","add_div2") for p in PROFILES}, X_FROZEN)
print("OBSERVED (frozen X=%.4f): %d hits" % (X_FROZEN, len(obs_hits)))
for p,a,m in obs_hits: print("   %-16s %-12s %+.1fpp" % (p,a,m*100))
obs_n = len(obs_hits)
obs_max = max(m for _,_,m in obs_hits)
obs_sum = sum(m for _,_,m in obs_hits)
print("   max %.4f  sum %.4f" % (obs_max, obs_sum))

# ---------- noise scales ----------
per_seed_d = [J[p][s]["add_div1"] - J[p][s]["add_div2"] for p in PROFILES for s in (0,1)]
sd_d = np.std(per_seed_d, ddof=1)
print("\nnull-pair d per profile-seed:", np.round(per_seed_d,3), " sd=%.4f" % sd_d)
print("implied sd of ONE run's J_deploy (training noise) = sd_d/sqrt2 = %.4f" % (sd_d/np.sqrt(2)))
# eval-set (binomial) component
pbar = np.mean([J[p][s][a] for p in PROFILES for s in (0,1) for a in ALL8])
print("mean J over 18 addition runs = %.4f ; binomial se on 200 starts = %.4f"
      % (pbar, np.sqrt(pbar*(1-pbar)/200)))
# spread of all 8 within each profile-seed
for p in PROFILES:
    for s in (0,1):
        v = np.array([J[p][s][a] for a in ALL8])
        print("  %-16s s%d  8-arm J: min %.3f max %.3f sd %.3f   D0 %.3f  div-mean %.3f"
              % (p, s, v.min(), v.max(), v.std(ddof=1), D0[p][s],
                 0.5*(J[p][s]['add_div1']+J[p][s]['add_div2'])))
