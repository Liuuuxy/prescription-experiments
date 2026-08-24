"""Task-2 (PickPlaceCounterToCabinet) diagnosis rollouts — parallel driver.

run_diagnosis.py's chunked-serial loop, replaced by ONE parallel_eval.run_parallel
call (workers env DIAG_WORKERS, default 4) because 2,400 serial episodes at
~45s each would be ~30h; with 4 osmesa workers against one served policy this
is ~8h (task-1 eval harness numbers). Same ledger contract: phase="diag",
policy_id="pi0", per-episode durable appends, done_pairs()-style resume.

MUST run with BANDIT_TASK_PROFILE=ppccab (asserted) so config points at the
ledger_ppccab/states_ppccab tree — the task-1 ledger is never touched.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "/data/xinyua11/robocasa")
os.chdir("/data/xinyua11/robocasa")

assert os.environ.get("BANDIT_TASK_PROFILE") == "ppccab", \
    "refusing to run: BANDIT_TASK_PROFILE=ppccab not set (would write the task-1 ledger)"

import pandas as pd

from bandit_v1 import config, ledger, parallel_eval

PHASE, POLICY_ID = "diag", "pi0"


def log(*a):
    print(f"[diag2 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8134)
    ap.add_argument("--repeats", type=int, default=config.M_DIAG)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DIAG_WORKERS", "4")))
    args = ap.parse_args()

    pq = config.LEDGER_DIR / "diag_conditions.parquet"
    conds = pd.read_parquet(pq)
    assert len(conds) == config.N_DIAG and conds.start_id.nunique() == config.N_DIAG, \
        f"diag_conditions has {len(conds)} rows / {conds.start_id.nunique()} unique -- expected {config.N_DIAG}"
    start_dirs = [config.DIAG_DIR / sid for sid in conds.start_id]

    done = set()
    try:
        df = ledger.read("episodes")
    except FileNotFoundError:
        df = None          # fresh ledger tree: no episodes table until the first append
    if df is not None and len(df):
        d = df[(df.phase == PHASE) & (df.policy_id == POLICY_ID)]
        done = {(r.start_id, int(r.repeat_idx)) for r in d.itertuples()}
    log(f"{len(start_dirs)} starts x {args.repeats} repeats, {len(done)} episodes already done, "
        f"workers={args.workers}, ledger={config.LEDGER_DIR.name}")

    # parallel_eval's default per-worker timeout (3h) is sized for ~113-episode
    # eval shards; a 600-episode diag shard runs ~7h and got two healthy workers
    # killed on 2026-08-18. Scale it to the shard actually assigned.
    per_worker = (len(start_dirs) * args.repeats - len(done)) / max(1, args.workers)
    timeout_s = max(10800, int(per_worker * 90 * 1.5))
    log(f"per-worker timeout {timeout_s}s for ~{per_worker:.0f} episodes/worker")
    parallel_eval.run_parallel(args.host, args.port, start_dirs, args.repeats,
                               phase=PHASE, policy_id=POLICY_ID,
                               workers=args.workers, skip_pairs=done or None,
                               timeout=timeout_s)

    d = ledger.read("episodes")
    d = d[(d.phase == PHASE) & (d.policy_id == POLICY_ID)]
    log(f"DIAG TOTAL {len(d)} episodes, success rate {d.success.mean():.4f}")
    # episodes rows already carry category (rollout.py copies start features);
    # merging conds' category in again created category_x/_y and crashed the
    # 2026-08-18 run's summary (rollouts themselves were complete and safe).
    if "category" not in d.columns:
        d = d.merge(conds[["start_id", "category"]], on="start_id", how="left")
    per_cat = d.groupby("category").success.agg(["mean", "count"]).sort_values("mean")
    log("hardest 10:\n" + per_cat.head(10).to_string())
    log("easiest 10:\n" + per_cat.tail(10).to_string())
    log("PPCCAB DIAG COMPLETE")


if __name__ == "__main__":
    main()
