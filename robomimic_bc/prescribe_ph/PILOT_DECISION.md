# Pilot decision (2026-08-25, per PREREG_PH_BENCHMARK.md pilot rule, probe-only)
J_deploy on E_probe (seeds 100/101):
  N=40: 0.432 / 0.265  mean 0.349  -> BELOW band [0.40, 0.75]
  N=60: 0.403 / 0.360  mean 0.382  -> BELOW band
  N=80: 0.662 / 0.553  mean 0.607  -> IN band, no floor-dead seed
Rule: smallest qualifying N -> FROZEN D0 SIZE N = 80 (balanced 20/region;
starved profile: 4 starved / 26 / 25 / 25).
Notes: paired-seed spread 4-17pp (vs MH's 20-35pp); N=40 shows large regional
headroom (xhi_ylo 0.20-0.24) but fails the band; non-monotonic 40-vs-60 on one
seed is within seed noise. E_test untouched.
