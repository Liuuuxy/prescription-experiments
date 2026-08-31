import json, itertools
import numpy as np
from scipy import stats

P = "/data/xinyua11/robomimic_runs/prescribe_ph"
PROFILES = ["balanced", "starved_xlo_yhi", "starved_xhi_ylo"]
REGIONS = ["xlo_ylo", "xlo_yhi", "xhi_ylo", "xhi_yhi"]
ALL8 = [f"add_{r}" for r in REGIONS] + ["add_cov", "add_pfail", "add_div1", "add_div2"]
X_FROZEN = json.load(open(f"{P}/MDE.json"))["X"]
def jt(n): return json.load(open(f"{P}/out/{n}/fixed_eval_test.json"))["J_deploy"]

# V[p] = 8x2 array of J_deploy (arm x seed).  Column pairing preserved (joint permutation).
V = {p: np.array([[jt(f"{p}_{a}_s{s}") for s in (0,1)] for a in ALL8]) for p in PROFILES}
OBS_IDX = (ALL8.index("add_div1"), ALL8.index("add_div2"))

def profile_stats(v, i, j):
    """v: 8x2. i,j = ordered indices chosen as div1,div2.
    returns (list of 6 mean-advs, list of 6 sign-replicated bools, profile-mean d)"""
    div = 0.5*(v[i] + v[j])                     # length-2 (per seed)
    keep = [k for k in range(8) if k not in (i, j)]
    adv = v[keep] - div                          # 6x2
    m = adv.mean(axis=1)
    rep = np.sign(adv[:,0]) == np.sign(adv[:,1])
    d = v[i] - v[j]
    return m, rep, float(d.mean())

# precompute for every ordered (i,j) per profile
ORD = [(i,j) for i in range(8) for j in range(8) if i != j]
PS = {p: {ij: profile_stats(V[p], *ij) for ij in ORD} for p in PROFILES}

def X_from(dmeans):
    sd = float(np.std(dmeans, ddof=1))
    tcrit = stats.t.ppf(0.95, df=2) + stats.t.ppf(0.80, df=2)   # 3.98065
    return tcrit * sd / np.sqrt(3), sd

# ---------- enumeration ----------
res = {"frozenX_nhits": [], "reX_nhits": [], "reX": [], "reX_sd": [],
       "frozenX_sum": [], "frozenX_max": [], "reX_sum": [], "reX_max": []}
for a in ORD:
    for b in ORD:
        for c in ORD:
            sel = ((PROFILES[0],a),(PROFILES[1],b),(PROFILES[2],c))
            ms, reps, ds = [], [], []
            for p, ij in sel:
                m, r, d = PS[p][ij]; ms.append(m); reps.append(r); ds.append(d)
            m_all = np.concatenate(ms); r_all = np.concatenate(reps)
            Xr, sdr = X_from(ds)
            hf = (m_all > X_FROZEN) & r_all
            hr = (m_all > Xr) & r_all
            res["frozenX_nhits"].append(int(hf.sum()))
            res["reX_nhits"].append(int(hr.sum()))
            res["reX"].append(Xr); res["reX_sd"].append(sdr)
            res["frozenX_sum"].append(float(m_all[hf].sum()))
            res["frozenX_max"].append(float(m_all[hf].max()) if hf.any() else 0.0)
            res["reX_sum"].append(float(m_all[hr].sum()))
            res["reX_max"].append(float(m_all[hr].max()) if hr.any() else 0.0)
R = {k: np.array(v) for k,v in res.items()}
np.savez("/data/xinyua11/tmp/claude-969153092/-data-xinyua11-robocasa/a6f630e5-2182-431d-8c49-04c4acb3a9d1/scratchpad/perm.npz", **R)
N = len(R["reX_nhits"])
print("full enumeration of ordered div-pair relabelings: N =", N, "(56^3)")
obs_n, obs_sum, obs_max = 7, 1.3366, 0.2486
print("\n--- A. frozen X = %.4f ---" % X_FROZEN)
for k in range(0, 13):
    print("   P(nhits >= %2d) = %.4f" % (k, (R["frozenX_nhits"] >= k).mean()))
print("   mean nhits = %.2f   P(>=7) = %.4f" % (R["frozenX_nhits"].mean(), (R["frozenX_nhits"]>=7).mean()))
print("   P(nhits>=7 AND sum>=%.3f) = %.4f" % (obs_sum, ((R["frozenX_nhits"]>=7)&(R["frozenX_sum"]>=obs_sum)).mean()))

print("\n--- B. X recomputed inside each relabeling (full prereg pipeline) ---")
print("   X distribution: min %.4f  q05 %.4f  q25 %.4f  median %.4f  q75 %.4f  q95 %.4f  max %.4f"
      % tuple(np.percentile(R["reX"], [0,5,25,50,75,95,100])))
print("   observed X = %.4f -> percentile %.3f%%   P(X <= obs) = %.5f"
      % (X_FROZEN, 100*(R["reX"] <= X_FROZEN).mean(), (R["reX"] <= X_FROZEN).mean()))
for k in range(0, 13):
    print("   P(nhits >= %2d) = %.4f" % (k, (R["reX_nhits"] >= k).mean()))
print("   mean nhits = %.2f   P(>=7) = %.5f" % (R["reX_nhits"].mean(), (R["reX_nhits"]>=7).mean()))
print("   P(nhits>=7 AND sum>=%.3f) = %.5f" % (obs_sum, ((R["reX_nhits"]>=7)&(R["reX_sum"]>=obs_sum)).mean()))
print("   P(nhits>=7 AND max>=%.3f) = %.5f" % (obs_max, ((R["reX_nhits"]>=7)&(R["reX_max"]>=obs_max)).mean()))
