"""Morning read-out for the Can-MH prescription sandbox.
Reads results.json; per (arm, B): mean final SR over seeds, delta vs the
paired-seed D0only baseline, and the collection-value ordering test."""
import json
import numpy as np

R = json.load(open("/data/xinyua11/robomimic_runs/prescribe/results.json"))
d0 = {v["seed"]: v["success_rates"][-1] for k, v in R.items() if k.startswith("D0only") and v["success_rates"]}
print(f"D0-only baseline (100 demos): " +
      ", ".join(f"s{s}={sr:.2f}" for s, sr in sorted(d0.items())) +
      (f"  mean {np.mean(list(d0.values())):.3f}" if d0 else "  (pending)"))
if any(len(v["success_rates"]) > 1 for k, v in R.items() if k.startswith("D0only")):
    for k, v in sorted(R.items()):
        if k.startswith("D0only") and len(v["success_rates"]) > 1:
            print(f"  {k}: SR@150={v['success_rates'][0]:.2f} SR@300={v['success_rates'][-1]:.2f}")
print()
rows = {}
for k, v in R.items():
    if k.startswith("run_") and v["success_rates"]:
        rows.setdefault((v["arm"], v["B"]), {})[v["seed"]] = v["success_rates"][-1]
print(f"{'arm':>8s} {'B':>4s} {'per-seed SR':>22s} {'mean SR':>8s} {'delta vs D0only (paired)':>26s}")
for (arm, B), by in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    srs = [by[s] for s in sorted(by)]
    deltas = [by[s] - d0[s] for s in sorted(by) if s in d0]
    print(f"{arm:>8s} {B:>4d} {' '.join(f'{x:.2f}' for x in srs):>22s} {np.mean(srs):>8.3f} "
          f"{' '.join(f'{x:+.2f}' for x in deltas):>20s}  mean {np.mean(deltas):+.3f}" if deltas else "")
print("\nordering test (pre-registered): better > random > worse at each B, paired per seed")
for B in (25, 50, 100):
    b = rows.get(("better", B)); r = rows.get(("random", B)); w = rows.get(("worse", B))
    if b and r and w:
        common = set(b) & set(r) & set(w)
        bw = [b[s] - w[s] for s in common]; br = [b[s] - r[s] for s in common]; rw = [r[s] - w[s] for s in common]
        print(f"  B={B}: better-worse {np.mean(bw):+.3f} ({['+' if x>0 else '-' for x in bw]})  "
              f"better-random {np.mean(br):+.3f}  random-worse {np.mean(rw):+.3f}")
