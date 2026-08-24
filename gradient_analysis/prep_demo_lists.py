"""Q0 prep: demo lists for the gradient-encoding gate (gradient analysis, 2026-07-30).

Builds gradient_analysis/demo_lists.json from the frozen bandit_v1 ledger:
  - region assignment for every W demo via the race's own wells.assign_regions
    (behavior z-space, frozen arms.yaml + map_models -- never reimplemented here)
  - Q0 gate samples: n in-region (tall_vessel_grasp_fail) vs n out-of-region,
    seeded, disjoint from each other
  - per-pull demo_ids + deltas for all ok pulls (Q1 treatment sets)

READ-ONLY on the ledger (ledger.read + frozen artifact loads). Output goes to
gradient_analysis/ only, per GRADIENT_ANALYSIS_HANDOFF.md convention 2.

Run: /data/xinyua11/conda/envs/robocasa/bin/python gradient_analysis/prep_demo_lists.py
"""
import json
import sys

import numpy as np
import yaml

sys.path.insert(0, "/data/xinyua11/robocasa")
from bandit_v1 import config, ledger, map_fit, pool, wells  # noqa: E402

OUT = "/data/xinyua11/robocasa/gradient_analysis/demo_lists.json"
N_GATE = 120  # per side; prior smoke gates used 100/100
SEED = 20260730
TALL_ARM = "tall_vessel_grasp_fail"


def main():
    pool_df = ledger.read("pool_demos")
    models = map_fit.load()
    arms_spec = yaml.safe_load(open(config.LEDGER_DIR / "arms.yaml"))
    regions = wells.assign_regions(pool_df, models, arms_spec)
    counts = regions.value_counts()
    print(f"[prep] W={len(regions)} region counts: {counts.to_dict()}", flush=True)

    pulls = ledger.read("pulls")
    ok = pulls[pulls.status == "ok"]
    pull_rows = {}
    for _, r in ok.iterrows():
        demo_ids = [int(x) for x in r.demo_ids] if r.demo_ids is not None else []
        pull_rows[str(r.pull_id)] = {
            "arm": str(r.arm), "round_j": int(r.round_j), "seed": int(r.seed),
            "delta": float(r.delta), "n_demos": len(demo_ids), "demo_ids": demo_ids,
        }
        print(f"[prep] pull {r.pull_id}: arm={r.arm} round={r.round_j} "
              f"delta={r.delta:+.4f} n_demos={len(demo_ids)}", flush=True)

    rng = np.random.RandomState(SEED)
    in_region_all = sorted(int(e) for e in regions.index[regions.values == TALL_ARM])
    out_region_all = sorted(int(e) for e in regions.index[regions.values != TALL_ARM])
    gate_in = sorted(int(x) for x in rng.choice(in_region_all, size=N_GATE, replace=False))
    # out-of-region: stratified evenly across the other arms
    other_arms = [a for a in counts.index if a != TALL_ARM]
    per_arm = N_GATE // len(other_arms)
    gate_out = []
    for a in other_arms:
        members = sorted(int(e) for e in regions.index[regions.values == a])
        gate_out += [int(x) for x in rng.choice(members, size=per_arm, replace=False)]
    gate_out = sorted(gate_out)

    # D0 sample for the retention/base-competence direction g_R (D0 = the 400
    # base episodes every pull's fine-tune reuses; excluded from W by design)
    d0_eps = sorted(int(e) for e in pool_df.loc[pool_df.in_d0, "episode_index"])
    d0_sample = sorted(int(x) for x in rng.choice(d0_eps, size=min(120, len(d0_eps)), replace=False))

    out = {
        "seed": SEED,
        "d0_sample": d0_sample,
        "tall_arm": TALL_ARM,
        "region_counts": {str(k): int(v) for k, v in counts.items()},
        "gate_in_region": gate_in,
        "gate_out_region": gate_out,
        "gate_out_by_arm": {a: per_arm for a in other_arms},
        "pulls": pull_rows,
        "region_assignment": {str(int(e)): str(a) for e, a in regions.items()},
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[prep] wrote {OUT}: gate {len(gate_in)} in vs {len(gate_out)} out; "
          f"{len(pull_rows)} ok pulls", flush=True)


if __name__ == "__main__":
    main()
