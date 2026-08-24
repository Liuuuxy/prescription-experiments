"""Loss-probe calibration for the ClusterUCB plan (2026-08-04, owner: "try the
actual UCB on clusters").

Gate question: does probe-set flow-matching loss at a fine-tuned checkpoint
track that checkpoint's KNOWN rollout-SR delta? We have ~20 fully-trained pull
checkpoints with ground-truth deltas -- forward passes only, no training.
If corr(probe-loss, SR-delta) is real -> loss is a valid dense UCB reward.
If ~0 -> loss-reward UCB is dead on arrival; we saved a week.

Probes (fixed, from demo_lists.json): gate_in_region (120 tall/target),
gate_out_region (120 other-region), d0_sample (120 trained-on, retention).
Output: gradient_analysis/loss_probe_calibration.parquet (resume-safe).
Run: openpi env, one GPU, MEM_FRACTION<=0.18.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/policy_analysis")
import influence_score as inf  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

GA = "/data/xinyua11/robocasa/gradient_analysis"
OUT = f"{GA}/loss_probe_calibration.parquet" if os.environ.get("PROBE_STEP","19999")=="19999" else f"{GA}/loss_probe_calibration_{os.environ['PROBE_STEP']}.parquet"
STEP = int(os.environ.get("PROBE_STEP", "19999"))
CKPT_GLOBS = [f"/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_bandit_a/*/{STEP}",
              f"/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_bandit_b/*/{STEP}"]
REF = ("pi0base_ref", "/data/xinyua11/openpi/checkpoints/pi0_ppc2sink_pi0base/pi0_v1")
K_FRAMES = 8
N_DRAWS = 2
REAL_DIM = 12



def log(*a):
    print(f"[lossprobe {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    lists = json.load(open(f"{GA}/demo_lists.json"))
    probes = {"target": [int(e) for e in lists["gate_in_region"]],
              "balanced": [int(e) for e in lists["gate_out_region"]],
              "retention": [int(e) for e in lists["d0_sample"]]}

    ckpts = [REF] if STEP == 19999 else []
    for g in CKPT_GLOBS:
        for d in sorted(glob.glob(g)):
            pid = d.rstrip("/").split("/")[-2]
            ckpts.append((pid, os.path.dirname(d)))
    log(f"checkpoints: {len(ckpts)} ({[c[0] for c in ckpts]})")

    cfg, raw, tds = inf.build_dataset(inf.MG_DEFAULT)
    starts = inf.episode_frame_starts(raw)

    @nnx.jit
    def loss_fn(model, rng, obs, act):
        return inf.masked_flow_loss(model, rng, obs, act, REAL_DIM)

    done = set()
    rows = []
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")
        done = set(prev["pull_id"])
        log(f"resuming: {len(done)} ckpts already probed")

    # keep probes as episode-id lists; build each demo's batch INSIDE the loop
    # (pre-stacking 360 image batches on-device was an 8GB OOM -- see log)
    probes = {p: [e for e in eps if e in starts] for p, eps in probes.items()}
    for pname, eps in probes.items():
        log(f"probe {pname}: {len(eps)} demos")

    import gc
    for pid, root in ckpts:
        if pid in done:
            continue
        inf.CKPT_BASE = root
        t0 = time.time()
        model = inf.load_model(cfg, STEP)
        rec = {"pull_id": pid}
        for pname, eps in probes.items():
            vals = []
            for e in eps:
                s, L = starts[e]
                o, a = inf.batch_to_inputs(inf.stack_batch(tds, inf.frame_indices(s, L, K_FRAMES)))
                for d in range(N_DRAWS):
                    r = jax.random.fold_in(jax.random.key(int(e)), STEP + d)
                    vals.append(float(loss_fn(model, r, o, a)))
            rec[f"loss_{pname}"] = float(np.mean(vals))
        del model
        gc.collect()
        rows.append(rec)
        pd.DataFrame(rows).to_parquet(OUT)
        log(f"{pid}: " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "pull_id")
            + f" ({time.time()-t0:.0f}s)")
    log("LOSSPROBE COMPLETE")


if __name__ == "__main__":
    main()
