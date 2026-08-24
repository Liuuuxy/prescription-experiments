"""Step ablation (owner-approved, 2026-08-04): do 10k/15k checkpoints match the
19999 result on the frozen E? If yes, future pulls can train 10k steps (~4h)
instead of 20k (~8h). No training -- serves already-saved intermediate
checkpoints of two completed pulls and evals them. Pinned to GPU0 (GPU1 is the
labmate's). Sequential: 4 evals x ~1h.

Caveat encoded in the design: a 10k snapshot of a 20k cosine schedule has
~half-decayed LR, so it UNDERESTIMATES a proper 10k-schedule run -- if the
snapshot already matches 19999 within noise, cutting steps is safe a fortiori.
"""
import json
import sys
import time

sys.path.insert(0, "/data/xinyua11/robocasa")
import os
os.chdir("/data/xinyua11/robocasa")

from bandit_v1 import pull, eval_set, config

GPU = 0
PORT = 8130
TARGETS = [  # (pull_id, slot, arm, steps to eval)
    ("mid_band_j5", "a", "mid_band", [10000, 15000]),
    ("random_j5", "b", "random", [10000, 15000]),
]
BASELINE = 0.5133


def log(*a):
    print(f"[ablate {time.strftime('%H:%M:%S')}]", *a, flush=True)


results = {}
for pid, slot, arm, steps in TARGETS:
    cfg_name = pull.train_config_name_for_slot(slot)
    for st in steps:
        ckpt = config.LEDGER_DIR.parent.parent  # placeholder, real path below
        ckpt = f"/data/xinyua11/openpi/checkpoints/{cfg_name}/{pid}/{st}"
        policy_id = f"ablate_{pid}_ck{st}"
        log(f"serving {ckpt} on gpu {GPU} port {PORT}")
        serve_log = config.LEDGER_DIR / "pull_logs" / f"{policy_id}_serve.log"
        _, proc = pull.launch_server(cfg_name, ckpt, PORT, GPU, serve_log)
        try:
            up = pull.wait_for_port("127.0.0.1", PORT, proc=proc, log=log)
            if not up:
                log(f"SERVER FAILED for {policy_id} -- skipping")
                continue
            r = eval_set.eval_checkpoint(PORT, policy_id, arm, policy_id,
                                         workers=4, resume=True, log=log)
            sr = r["mean"]
            results[policy_id] = r
            ps = {k: round(v, 3) for k, v in r["per_stratum_mean"].items()}
            log(f"RESULT {policy_id}: sr={sr:.4f} delta_vs_baseline={(sr-BASELINE)*100:+.2f}pp "
                f"per_stratum={ps}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            time.sleep(10)

json.dump({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
           for k, v in results.items()},
          open("/data/xinyua11/robocasa/gradient_analysis/step_ablation_results.json", "w"),
          default=str)
log("ABLATION COMPLETE")
