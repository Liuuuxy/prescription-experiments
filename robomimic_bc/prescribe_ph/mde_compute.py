"""Compute the Gate-1 MDE X from null pairs, per PREREG_PH_BENCHMARK.md.
Reads ONLY: D0 baselines and the two independent b_div additions (E_test rows
for NULL/BASELINE arms only — targeted arms are never touched here).
Writes MDE.json with a timestamp; must be run BEFORE gate1_analyze.py.

Null statistic, per profile x seed: d = J_deploy(add_div1) - J_deploy(add_div2)
(identical expected effect, independent draws -> d ~ noise of a paired
allocation contrast at matched seed). Profile-level clustering: average the
per-seed d within profile first, then treat profiles as the units.
X = smallest true effect with one-sided alpha=.05, power=.80 given sd(d_profile):
X = (z_.95 + z_.80) * sd_profile = 2.486 * sd_profile (t-adjusted below).
"""
import glob, json, os, time
import numpy as np

P = "/data/xinyua11/robomimic_runs/prescribe_ph"
PROFILES = ["balanced", "starved_xlo_yhi", "starved_xhi_ylo"]
assert not os.path.exists(f"{P}/MDE.json"), "MDE.json already exists — never recompute after unblinding"

def jt(name):
    fp = f"{P}/out/{name}/fixed_eval_test.json"
    return json.load(open(fp))["J_deploy"]

per_seed_d, prof_means = [], []
for prof in PROFILES:
    ds = []
    for s in (0, 1):
        d = jt(f"{prof}_add_div1_s{s}") - jt(f"{prof}_add_div2_s{s}")
        ds.append(d)
        per_seed_d.append({"profile": prof, "seed": s, "d": d})
    prof_means.append(float(np.mean(ds)))
sd_prof = float(np.std(prof_means, ddof=1))
sd_seed = float(np.std([r["d"] for r in per_seed_d], ddof=1))
n = len(prof_means)
from scipy import stats
tcrit = stats.t.ppf(0.95, df=n - 1) + stats.t.ppf(0.80, df=n - 1)
X = float(tcrit * sd_prof / np.sqrt(n))
out = {"X": X, "sd_profile_level": sd_prof, "sd_per_seed": sd_seed,
       "n_profiles": n, "t_multiplier": float(tcrit),
       "null_pairs": per_seed_d, "profile_means": prof_means,
       "rule": "one-sided alpha=.05, power=.80, profile-clustered",
       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "note": "computed from b_div null pairs ONLY; no targeted E_test row read"}
json.dump(out, open(f"{P}/MDE.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("X", "sd_profile_level", "sd_per_seed", "t_multiplier")}, indent=1))
