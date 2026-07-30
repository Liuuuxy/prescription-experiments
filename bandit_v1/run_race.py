"""Race runner: nulls -> successive-elimination race, resumable (bandit_v1
Task 14 step 3).

Design: weakregion/BANDIT_V1_DESIGN.md section 3 ("One pull"), section 4
("Scheduler"), section 1 item 6 (noise floor), section 8 (defaults: T=12-16,
2 null pulls). Brief: .superpowers/sdd/task-14-brief.md's Step 3 ("RUN THE
RACE"). This module is the ONE orchestrator that ties together every prior
task's module (pull.run_pull, scheduler.decide, eval_set.eval_checkpoint,
wells.assign_regions, clustering.load_arms_yaml) into the actual multi-day
race loop -- everything below is pure/monkeypatched-testable; the real
end-to-end run is DEFERRED (this task explicitly does not launch it).

Round numbering: pull_seed(j) = 1000 + j is shared across null pulls AND
real arm pulls (config.py's one seed formula) -- the 2 null pulls consume
j=1 (seed 1001) and j=2 (seed 1002), so the first REAL race round is
j=RACE_FIRST_ROUND=3, matching the design brief's own "loop rounds
j=3,4,...".

Two phases, run in order every time this module's `main()` starts (each
phase is itself a no-op if the ledger already shows it complete -- this is
what makes a `nohup`'d run_race.py process safe to kill and relaunch at any
point):

  1. **Phase NULLS**: recipe R on D0 alone (no extra demos) at seeds
     1001/1002 -- the "honest zero point" the design's noise floor is
     measured against. Once both are `ok`, sigma_e = max(sample std of the
     2 null deltas, the eval-only sigma_e_eval already recorded in
     config.yaml's `baseline:` block by run_baseline.sh) is computed and
     written ONCE to config.yaml's `noise_floor:` block (guarded against a
     double-append on a later restart, same convention as
     eval_set.append_baseline_to_config_yaml / clustering.
     append_arms_freeze_to_config_yaml).

  2. **Phase RACE**: successive-elimination over the frozen arms.yaml's
     cluster names + "random", `scheduler.decide()`-driven. Rounds are
     processed strictly in order; within a round, every arm that doesn't
     yet have an `ok` pull-row at that round number gets pulled (this is
     what makes a crash MID-round resumable -- see `_current_round_and_roster`'s
     docstring for why `scheduler.decide()` itself must never be called on a
     partially-completed round's data).

GPU/slot policy: `pull.run_pull`'s own `wait_for_free_gpu` already knows how
to wait for a free GPU one-at-a-time; this module additionally offers a
"2-wide when both GPU slots are claimable" mode (`run_batch_two_wide`) for
running up to 2 pulls concurrently (one per training slot/GPU), via plain OS
threads (each thread's own `pull.run_pull` call blocks on its OWN training
job -- the actual parallelism is the GPU work happening on 2 different
`CUDA_VISIBLE_DEVICES`, same "2-wide" the design's cost estimate assumes,
~2 pulls/day). `both_slots_claimable`'s own nvidia-smi query tolerates a
broken/unavailable NVML query (treats it as "claimable" rather than
"unknown -> assume busy") -- the exact same stance run_baseline.sh's
`free_mb()`/guard already takes (EGL/driver issues on this box have nothing
to do with whether CUDA/JAX training itself works).

Resumability contract (the one invariant every function below is written
to preserve): a fresh `run_race.py` process, given nothing but the current
`bandit_v1/ledger/pulls.parquet` + `config.yaml`, must reconstruct EXACTLY
the same (round, live-arm-roster) state a never-interrupted run would be in
at that point -- never re-run an already-`ok` (arm, round) pull, never
silently mis-attribute an elimination to an arm that simply hasn't been
pulled yet this round because of a crash.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import re
import yaml

from . import clustering, config, draw, eval_set, ledger, map_fit, pool, pull, scheduler, wells

POLL_SECS = 120                 # preconditions-wait poll interval
NULL_ROUNDS = (1, 2)            # seeds 1001/1002 via config.pull_seed
RACE_FIRST_ROUND = 3            # real race rounds start where the null seeds leave off
NULL_DELTA_LOUD_THRESHOLD = 0.05
# (historical -- superseded, see the "FLIPPED TO 4" block just above
# SLOT_GPU below for the current, deliberate contract) LOUD: SERIAL IS THE
# DEFAULT FOR EVALS, NOT A FALLBACK (emergency null-
# takeover fix, task-nulltakeover-report.md). This box hangs 4-worker
# parallel_eval evals under CPU contention (see task-baselinerecover-
# report.md/task-r3overlap-report.md: repeated 2-4 workers stuck importing
# for 3h, self-killed by parallel_eval's own timeout, no useful speedup
# ever realized in practice) -- `None` routes eval_set.eval_checkpoint
# through the plain serial rollout.run path every time, which is slower per
# pull but has never once hung. DO NOT bump this back up to a parallel
# worker count until the underlying box-contention/driver-hang issue is
# actually fixed and re-verified under a real multi-hour eval -- revisit
# then, not before.
# (historical -- superseded below) EVAL_WORKERS = None  # SERIAL, FINAL: 2026-07-29 re-test on an IDLE box (load 11/128) reproduced the 4-worker import-stage deadlock (workers hang after robosuite import warnings; retry cycles burned a full night) -- the hang is concurrency-count-triggered, NOT contention. Serial per-episode evals are the only proven mode; 2 pulls still eval concurrently (one per GPU).
# import-hang root cause (task-importhang-report.md, 2026-07-29): logging
# blind spot -- rollout.py's per-episode loop prints NOTHING, so a worker's
# log looks identical ("stuck" right after the robosuite/robocasa import
# warnings) whether it is genuinely hung OR healthy and just finished
# (verified directly: a fully-successful 4-worker run's logs end at that
# exact line too), compounded by a REAL, verified thread-oversubscription
# bug (each worker defaulted to ~322 OS threads -- unset OMP_NUM_THREADS/
# OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/NUMEXPR_NUM_THREADS/LP_NUM_THREADS
# let numpy/BLAS/Mesa-llvmpipe each size their pool to nproc=128 -- N
# workers self-inflict N*~300 threads independent of ambient box load,
# matching "reproduces on an idle box"). Fix: parallel_eval.py's
# `_worker_env` (thread caps + PYTHONUNBUFFERED=1) + `run_worker_inline`
# per-episode progress prints, validated in isolation 2026-07-29 (repro.py/
# repro_real.py/validate_run_parallel.py: n=2/4/6/8, osmesa+egl, plain
# imports through full fake-server rollout episodes via the REAL
# `run_parallel`/`_spawn_worker` path, 3x clean each, thread count 322->12
# confirmed on live spawned workers). Caveat: a genuine multi-hour stall
# was never reproduced in CPU-only isolation (no GPU/real-policy load), so
# this is a verified hardening + diagnosability fix, not a proven-sufficient
# root-cause fix -- flip to 4 only after one supervised parallel wave.
#
# FLIPPED TO 4 (task-ledgerlock-report.md, 2026-07-29): both prerequisites
# the fix above called for have now landed. (1) One supervised parallel
# wave, run for real against a live served policy -- easy_band_j3: 4
# workers, 171 rollouts, clean finish, merge into "episodes" verified --
# exactly the "flip to 4 only after one supervised parallel wave" bar set
# above. (2) The OTHER real risk this box has actually hit is now closed
# too -- not the import hang, but a genuine cross-process race in
# ledger.append_rows itself, surfaced by tonight's incident: an external
# bulk append and this runner's own per-episode appends both doing
# read-concat-write-tmp-rename against the SAME "episodes" table at the
# same time (one writer's tmp file vanished out from under the other --
# FileNotFoundError -- and a row landing between another writer's read and
# write could be silently lost; self-healed by eval-resume that time, zero
# dupes, but not something to rely on happening again). append_rows now
# holds a cross-process fcntl.flock across its whole read-concat-write-
# rename (LOCK_TIMEOUT_S=120s, loud TimeoutError naming the table + a
# holder hint on timeout, never a silent indefinite hang) plus a per-pid
# tmp filename as belt-and-braces, so `merge_shards`'s single append_rows
# call (parallel_eval.py, the parent-side merge after every worker exits)
# and this runner's own serial per-episode appends can never interleave
# again. Both prerequisites now hold. Revisit back to None only if a NEW
# multi-hour stall surfaces under real GPU/training load -- per the caveat
# above, that specific failure mode was hardened against, never actually
# reproduced end-to-end.
EVAL_WORKERS = 4
SLOT_GPU = {"a": 0, "b": 1}     # slot <-> physical GPU pinning for 2-wide concurrent pulls


# =============================================================================
# 1. preconditions wait
# =============================================================================

def preconditions_status(cfg_path=None, e_manifest_path=None) -> dict:
    """{baseline_ready, e_manifest_ready, ready} from the CURRENT on-disk
    state -- read fresh every call (config.LEDGER_DIR read at call time, not
    cached at import), so a test's monkeypatched config.LEDGER_DIR or an
    explicit path override is always honored. `baseline_ready` requires
    config.yaml's `baseline:` block to exist AND carry non-None b/
    per_stratum_b/sigma_e_eval (run_baseline.sh's
    append_baseline_to_config_yaml writes all three together, so a partial
    block should never occur in practice, but a partial/malformed block is
    treated as "not ready" rather than crashing this poll loop)."""
    cfg_path = Path(config.LEDGER_DIR) / "config.yaml" if cfg_path is None else Path(cfg_path)
    e_manifest_path = (Path(config.LEDGER_DIR) / "E_manifest.parquet" if e_manifest_path is None
                        else Path(e_manifest_path))

    baseline_ready = False
    if cfg_path.exists():
        doc = yaml.safe_load(cfg_path.read_text()) or {}
        b = doc.get("baseline") if isinstance(doc, dict) else None
        if isinstance(b, dict):
            baseline_ready = all(b.get(k) is not None for k in ("b", "per_stratum_b", "sigma_e_eval"))

    e_manifest_ready = e_manifest_path.exists()
    return {"baseline_ready": baseline_ready, "e_manifest_ready": e_manifest_ready,
            "ready": baseline_ready and e_manifest_ready}


def wait_for_preconditions(cfg_path=None, e_manifest_path=None, poll_secs=POLL_SECS,
                            sleep_fn=time.sleep, log=print, max_polls=None) -> dict:
    """Block (polling every `poll_secs`) until `preconditions_status(...)
    ["ready"]`. `max_polls` is a testability-only escape hatch (production
    callers never pass it): raises TimeoutError after that many polls
    instead of looping forever, so a test can assert the loop actually
    polls the expected number of times against a fake `sleep_fn` rather than
    hanging."""
    polls = 0
    while True:
        status = preconditions_status(cfg_path=cfg_path, e_manifest_path=e_manifest_path)
        log(f"waiting for preconditions: baseline_ready={status['baseline_ready']} "
            f"e_manifest_ready={status['e_manifest_ready']}")
        if status["ready"]:
            return status
        polls += 1
        if max_polls is not None and polls >= max_polls:
            raise TimeoutError(f"wait_for_preconditions: still not ready after {polls} polls")
        sleep_fn(poll_secs)


def load_baseline(cfg_path=None):
    """(b, per_stratum_b, sigma_e_eval) from config.yaml's `baseline:` block
    (written by run_baseline.sh / eval_set.append_baseline_to_config_yaml).
    Callers are expected to only invoke this once `preconditions_status(...)
    ["baseline_ready"]` is True."""
    cfg_path = Path(config.LEDGER_DIR) / "config.yaml" if cfg_path is None else Path(cfg_path)
    doc = yaml.safe_load(cfg_path.read_text())
    b = doc["baseline"]
    return float(b["b"]), {k: float(v) for k, v in b["per_stratum_b"].items()}, float(b["sigma_e_eval"])


def load_frozen_B(cfg_path=None) -> int:
    """B from config.yaml's `arms_freeze:` block (frozen at clustering
    finalize time -- see clustering.append_arms_freeze_to_config_yaml)."""
    cfg_path = Path(config.LEDGER_DIR) / "config.yaml" if cfg_path is None else Path(cfg_path)
    doc = yaml.safe_load(cfg_path.read_text())
    return int(doc["arms_freeze"]["B"])


def frozen_arm_names(arms_spec=None) -> list:
    """Every cluster name (index order) from the frozen arms.yaml, plus
    "random" appended last if `arms_spec["random_arm"]` (always True in
    practice) -- the FULL initial roster round 1 (== RACE_FIRST_ROUND here)
    of the race must pull. `arms_spec` defaults to
    `clustering.load_arms_yaml()`."""
    arms_spec = clustering.load_arms_yaml() if arms_spec is None else arms_spec
    names = [a["name"] for a in sorted(arms_spec["arms"], key=lambda a: a["index"])]
    if arms_spec.get("random_arm", True):
        names = names + [draw.RANDOM_ARM]
    return names


# =============================================================================
# 2. noise-floor note (config.yaml, append-once)
# =============================================================================

def append_noise_floor_to_config_yaml(null_deltas, sigma_e, path=None, log=lambda *a: None) -> Path:
    """Append a one-time `noise_floor:` block ({null_deltas, sigma_e}) to
    config.yaml -- guarded against a double append (same append-once
    convention as pull.append_gradient_analysis_note_to_config_yaml /
    eval_set.append_baseline_to_config_yaml / clustering.
    append_arms_freeze_to_config_yaml), since a resumed run_race.py process
    recomputes sigma_e from the ledger on every restart but must not grow
    config.yaml a fresh copy of this note every time."""
    path = Path(config.LEDGER_DIR) / "config.yaml" if path is None else Path(path)
    if path.exists():
        doc = yaml.safe_load(path.read_text()) or {}
        if isinstance(doc, dict) and "noise_floor" in doc:
            existing = doc["noise_floor"]
            same = (existing.get("sigma_e") == float(sigma_e)
                    and [float(x) for x in existing.get("null_deltas", [])]
                    == [float(x) for x in null_deltas])
            if same:
                log(f"noise_floor block already present in {path} -- not re-appending")
                return path
            # A mismatched pre-existing block is stale/foreign (e.g. test junk
            # written via a defaulted path): replace it LOUDLY with the real
            # measurement rather than silently keeping the wrong numbers.
            log(f"WARNING: replacing mismatched noise_floor block in {path} "
                f"(had {existing}, real = null_deltas={list(null_deltas)}, sigma_e={sigma_e})")
            text = path.read_text()
            text = re.sub(r"\n?#[^\n]*Phase NULLS[^\n]*\n(#[^\n]*\n)*noise_floor:\n(  [^\n]*\n)*", "", text)
            text = re.sub(r"noise_floor:\n(  [^\n]*\n|  - [^\n]*\n)*", "", text)
            path.write_text(text)

    block = {"noise_floor": {"null_deltas": [float(x) for x in null_deltas], "sigma_e": float(sigma_e)}}
    header = (
        "\n# bandit_v1 run_race Phase NULLS: measured noise floor (2 null pulls --\n"
        "# recipe R on D0 alone at seeds 1001/1002 -- vs the frozen baseline b).\n"
        "# sigma_e = max(sample_std(null_deltas, ddof=1), baseline.sigma_e_eval).\n"
    )
    dumped = yaml.safe_dump(block, sort_keys=False, default_flow_style=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(header)
        f.write(dumped)
    log(f"appended noise_floor block to {path}: sigma_e={sigma_e:.4f}")
    return path


# =============================================================================
# 3. GPU-slot claimability (2-wide gate)
# =============================================================================

def gpu_free_mib_or_none(gpu, query=None):
    """`pull.free_mib(gpu, query=query)`, or None if the nvidia-smi query
    itself fails (broken NVML/driver, matching this box's EGL-broken-box-
    wide situation) -- None is NOT "0 MiB free"; it means "unknown, and a
    broken query says nothing about whether CUDA/JAX training actually
    works" (run_baseline.sh's `free_mb()` takes the exact same stance)."""
    try:
        return pull.free_mib(gpu, query=query)
    except Exception:
        return None


def gpu_claimable(gpu, need_mib=pull.TRAIN_GPU_NEED_MIB, query=None) -> bool:
    free = gpu_free_mib_or_none(gpu, query=query)
    return free is None or free >= need_mib


def both_slots_claimable(query=None) -> bool:
    return gpu_claimable(0, query=query) and gpu_claimable(1, query=query)


def _run_one_capturing(run_one, spec, errors, results):
    try:
        results.append(run_one(spec))
    except Exception as e:  # noqa: BLE001 -- deliberately broad, see run_batch_two_wide
        errors.append(e)


def run_batch_two_wide(specs, run_one, claimable_fn=both_slots_claimable, log=print,
                        should_stop_fn=None, sleep_fn=time.sleep) -> list:
    """Run `specs` (each a dict, at minimum carrying "slot") via
    `run_one(spec)`: one at a time, UNLESS `claimable_fn()` says both GPU
    slots are currently claimable, in which case each adjacent PAIR of specs
    runs concurrently (one Python thread per spec -- `pull.run_pull` itself
    blocks on subprocess/socket waits, so 2 threads genuinely overlap 2
    separate training/eval jobs on 2 separate GPUs). A trailing odd spec (or
    every spec, when `claimable_fn()` is False) runs sequentially in the
    caller's own thread.

    `should_stop_fn` (optional, defaults to never-stop): re-checked
    immediately before EVERY spec is about to be dispatched -- once before
    each pair (a concurrent pair is one dispatch decision: either both
    members start together or neither does) and, in the sequential branch,
    additionally before each individual spec within that pair. If it
    returns True, this batch stops immediately: nothing further is started,
    and whatever's already been collected (results/errors) is returned/
    raised as usual. This is the seam `run_race.run_race_phase` uses to
    enforce the T-cap (`t_max`) budget mid-round, not just between whole
    rounds (race-runner review Finding 2) -- `should_stop_fn` re-reads the
    ledger fresh on every call so the check reflects pulls THIS batch has
    already completed, not a stale count taken before the batch started.

    Every spec in `specs` is attempted regardless of an earlier one's
    failure (one arm's pull failing should never prevent an already-decided
    sibling pull from running) -- exceptions are collected and the FIRST one
    is re-raised only after every spec has been attempted, so a caller still
    sees a hard failure (never silently swallowed) but never loses whatever
    other pulls in this same batch would otherwise have completed cleanly.
    Returns the list of `run_one` return values for every spec that did NOT
    raise, in `specs` order within each pair (pair order across the whole
    batch is preserved; the two members of a concurrently-run pair may
    finish either order, but the results themselves are only relied on by
    callers as an unordered "these landed OK" set, not positionally).
    """
    errors, results = [], []
    i = 0
    stopped = False
    while i < len(specs) and not stopped:
        if should_stop_fn is not None and should_stop_fn():
            log(f"run_batch_two_wide: should_stop_fn() signalled stop -- not starting "
                f"the remaining {len(specs) - i} spec(s) in this batch "
                f"({[s.get('arm', s.get('slot')) for s in specs[i:]]})")
            break
        pair = specs[i:i + 2]
        # Claimability can be transiently false right after a kill/restart while
        # GPU memory is still releasing -- one stale glance must not commit the
        # whole pair to a ~9h sequential path. Poll briefly before falling back.
        pair_claimable = False
        if len(pair) == 2:
            for _attempt in range(20):                       # up to ~5 min
                if claimable_fn():
                    pair_claimable = True
                    break
                if _attempt == 0:
                    log("run_batch_two_wide: slots not claimable yet -- polling up to "
                        "5 min before falling back to sequential")
                sleep_fn(15)
        if pair_claimable:
            log(f"run_batch_two_wide: both GPU slots claimable -- running "
                f"{[s.get('arm', s.get('slot')) for s in pair]} concurrently")
            threads = [threading.Thread(target=_run_one_capturing, args=(run_one, spec, errors, results))
                       for spec in pair]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for spec in pair:
                if should_stop_fn is not None and should_stop_fn():
                    log(f"run_batch_two_wide: should_stop_fn() signalled stop mid-pair -- "
                        f"not starting {spec.get('arm', spec.get('slot'))}")
                    stopped = True
                    break
                _run_one_capturing(run_one, spec, errors, results)
        i += 2
    if errors:
        raise errors[0]
    return results


# =============================================================================
# 4. Phase NULLS
# =============================================================================

def missing_null_rounds(pulls_df) -> list:
    """Subset of NULL_ROUNDS (1, 2) that do NOT yet have an `ok` null pull
    in `pulls_df` -- what Phase NULLS still needs to run. Empty once both
    seeds 1001/1002 are done."""
    if pulls_df is None or pulls_df.empty:
        done = set()
    else:
        ok = pulls_df[(pulls_df["arm"] == "null") & (pulls_df["status"] == "ok")]
        done = set(int(j) for j in ok["round_j"].unique())
    return [j for j in NULL_ROUNDS if j not in done]


def run_null_phase(read_pulls_fn, run_one, sigma_e_eval, claimable_fn=both_slots_claimable,
                    cfg_path=None, log=print, sleep_fn=time.sleep) -> float:
    """Run whichever of the 2 null pulls (seeds 1001/1002) aren't `ok` yet
    (no-op if both already are -- the resumable "phase is a no-op if
    already complete" contract), then compute + record sigma_e =
    max(sample_std(null_deltas, ddof=1), sigma_e_eval) and return it.

    Loudly (but non-fatally) flags any null delta whose magnitude exceeds
    NULL_DELTA_LOUD_THRESHOLD (0.05): per the design brief, no two prior
    finetunes ever differed only in seed, so a large null delta is genuine
    evidence that training-seed noise alone can dominate the signal this
    whole race is trying to measure -- worth a human's attention (this is
    the "PushNotification-worthy" line the owner asked for -- run_race.py
    itself has no notification channel, so this is rendered as a maximally
    loud, greppable log line instead), but is NOT a reason to halt: the
    design's own stance is "measure sigma_e, then interpret every ranking
    through it" -- exactly what happens right after this warning.
    """
    pulls_df = read_pulls_fn()
    missing = missing_null_rounds(pulls_df)

    if not missing:
        log("Phase NULLS: both null pulls (j=1,2) already ok -- skipping")
    else:
        log(f"Phase NULLS: running missing null pull(s) j={missing}")
        specs = [{"arm": "null", "j": j, "slot": pull.SLOTS[i % 2]} for i, j in enumerate(missing)]
        specs = pull.resolve_sticky_slots(specs, log=log)
        run_batch_two_wide(specs, run_one, claimable_fn=claimable_fn, log=log, sleep_fn=sleep_fn)
        pulls_df = read_pulls_fn()

    ok_null = pulls_df[(pulls_df["arm"] == "null") & (pulls_df["status"] == "ok")]
    if len(ok_null) < 2:
        raise RuntimeError(
            f"Phase NULLS: only {len(ok_null)}/2 null pulls are ok even after "
            f"attempting the missing round(s) -- run_pull's own retry-once "
            f"logic was exhausted. HUMAN INTERVENTION REQUIRED (see "
            f"ledger/pull_logs/null_j*); re-run once fixed, it will resume here.")

    null_deltas = [float(x) for x in ok_null.sort_values("round_j")["delta"]]
    for j, d in zip(ok_null.sort_values("round_j")["round_j"], null_deltas):
        if abs(d) > NULL_DELTA_LOUD_THRESHOLD:
            log(f"!!!!! LOUD WARNING (PushNotification-worthy): null pull j={int(j)} delta="
                f"{d:+.4f} exceeds +/-{NULL_DELTA_LOUD_THRESHOLD} -- training-seed noise may be "
                f"dominating the signal this race measures. Continuing (sigma_e below already "
                f"accounts for this).")

    sigma_e_null = float(np.std(null_deltas, ddof=1)) if len(null_deltas) > 1 else 0.0
    sigma_e = max(sigma_e_null, sigma_e_eval)
    log(f"Phase NULLS: null_deltas={null_deltas} sigma_e_null={sigma_e_null:.4f} "
        f"sigma_e_eval={sigma_e_eval:.4f} -> sigma_e={sigma_e:.4f}")
    append_noise_floor_to_config_yaml(null_deltas, sigma_e, path=cfg_path, log=log)
    return sigma_e


# =============================================================================
# 5. Phase RACE
# =============================================================================

def _ok_arms_at_round(pulls_df, j) -> set:
    if pulls_df is None or pulls_df.empty:
        return set()
    sub = pulls_df[(pulls_df["round_j"] == j) & (pulls_df["status"] == "ok") & (pulls_df["arm"] != "null")]
    return set(sub["arm"].unique())


def _roster_before_round(pulls_df, m, all_arms, sigma_e, delta, t_max):
    """The arm roster that SHOULD be pulled at round `m`, computed using
    ONLY rows with round_j < m -- i.e. never looking at round `m`'s own
    (possibly partial) rows. This is what makes resume-mid-round safe: if
    round `m` is only partially done (a crash happened after some, not all,
    of that round's arms got an ok row), `scheduler.decide()` must never see
    those partial round-`m` rows, or it would misread "hasn't been pulled at
    round m YET" as "eliminated at its last successful round" (decide()'s
    cohort is defined as "last ok pull is at the overall max round" -- an
    arm that just hasn't had its turn yet this round would incorrectly fall
    outside that cohort). Filtering to round_j < m sidesteps this precisely
    because every round strictly before `m` is -- by this same module's own
    invariant -- ALWAYS already fully complete by the time round `m` starts."""
    if m <= RACE_FIRST_ROUND:
        return list(all_arms)
    prior = pulls_df[pulls_df["round_j"] < m] if pulls_df is not None and not pulls_df.empty else pulls_df
    decision = scheduler.decide(prior, sigma_e, delta=delta, t_max=t_max)
    return decision["survivors"]


def log_ranking(decision, log=print) -> None:
    log("RANKING (arm, mean, lcb, ucb, n):")
    for arm, mean, lcb, ucb, n in decision["ranking"]:
        flag = ""
        if arm in decision["eliminated"]:
            flag = f"  [ELIMINATED @ round {decision['eliminated'][arm]}]"
        elif arm in decision.get("tied_with_leader", []):
            flag = "  [tied with leader]"
        log(f"    {arm:28s} mean={mean:+.4f}  lcb={lcb:+.4f}  ucb={ucb:+.4f}  n={n}{flag}")
    if decision.get("no_data"):
        log(f"    no_data (no ok pulls yet): {decision['no_data']}")
    log(f"  done={decision['done']}  next_round={decision['next_round']}")


def _current_round_and_roster(pulls_df, all_arms, sigma_e, delta, t_max, log=print):
    """(j, alive) to resume Phase RACE at, or (None, decision) if the race is
    already done -- purely from `pulls_df`'s current ok rows for non-null
    arms. See `_roster_before_round`'s docstring for why a partially-
    completed round is handled by recomputing that SAME round's roster from
    strictly-prior data, rather than trusting `scheduler.decide()`'s
    `next_round` blindly on a possibly-partial ledger."""
    ok_real = (pulls_df[(pulls_df["status"] == "ok") & (pulls_df["arm"] != "null")]
               if pulls_df is not None and not pulls_df.empty else pd.DataFrame())
    if ok_real.empty:
        return RACE_FIRST_ROUND, list(all_arms)

    m = int(ok_real["round_j"].max())
    roster_m = _roster_before_round(pulls_df, m, all_arms, sigma_e, delta, t_max)
    satisfied = _ok_arms_at_round(pulls_df, m)

    if all(a in satisfied for a in roster_m):
        # Round m is genuinely complete for its own roster -- safe to hand
        # scheduler.decide() the FULL ledger and advance.
        decision = scheduler.decide(pulls_df, sigma_e, delta=delta, t_max=t_max)
        log_ranking(decision, log=log)
        if decision["done"]:
            return None, decision
        return decision["next_round"], decision["survivors"]

    return m, roster_m


def _ok_non_null_pull_count(pulls_df) -> int:
    """Count of `status == "ok"` non-null rows -- exactly the quantity
    `t_max` budgets (scheduler.decide's own "total_ok_pulls" -- see that
    module's docstring on why null pulls are excluded from T)."""
    if pulls_df is None or pulls_df.empty:
        return 0
    ok = pulls_df[(pulls_df["status"] == "ok") & (pulls_df["arm"] != "null")]
    return len(ok)


def run_race_phase(sigma_e, read_pulls_fn, run_one, all_arms, claimable_fn=both_slots_claimable,
                    t_max=config.T_MAX_PULLS, delta=config.DELTA_CONF, log=print, sleep_fn=time.sleep) -> dict:
    """Successive-elimination race over `all_arms`, starting at
    RACE_FIRST_ROUND (3). Resumable: recomputes (round, live roster) from
    the ledger on every iteration (see `_current_round_and_roster`), pulls
    whichever of the live roster lacks an ok row at the current round
    (never re-running an already-ok (arm, round) pull), then re-derives the
    scheduler's verdict. Halts LOUDLY (raises) if an arm still lacks an ok
    row after its one attempt this round -- never silently proceeds to
    scheduler.decide() on an incomplete round (see _roster_before_round's
    docstring for why that would risk a wrong elimination). Returns the
    final `scheduler.decide()` dict once done.

    T-cap (`t_max`) enforcement (race-runner review Finding 2): the ok-pull
    count is checked BEFORE every individual pull is dispatched within a
    round, not merely between whole rounds -- `_ok_non_null_pull_count` is
    re-read fresh from the ledger and passed as `run_batch_two_wide`'s
    `should_stop_fn`. Previously the cap was only re-checked via
    `scheduler.decide` once an entire round's pulls had already all run, so
    an elimination that shrank the live roster to an "unlucky" size could
    overshoot `t_max` by up to `roster_size - 1` pulls; now the round stops
    cleanly as soon as the cap is hit, however many of that round's planned
    pulls have actually started. Hitting the cap mid-round is NOT treated
    as a failure: whichever arms in `to_pull` never got dispatched are
    simply not attempted (no "HUMAN INTERVENTION REQUIRED" raise -- that
    path is reserved for an arm that WAS attempted and still didn't produce
    an ok row) -- the current ranking is logged and the (now budget-decided)
    `scheduler.decide()` verdict is returned directly.
    """
    def _t_cap_reached() -> bool:
        return _ok_non_null_pull_count(read_pulls_fn()) >= t_max

    while True:
        pulls_df = read_pulls_fn()
        j, alive = _current_round_and_roster(pulls_df, all_arms, sigma_e, delta, t_max, log=log)
        if j is None:
            decision = alive
            log(f"=== DONE: race complete -- survivors={decision['survivors']} "
                f"eliminated={decision['eliminated']} ===")
            return decision

        done_this_round = _ok_arms_at_round(pulls_df, j)
        to_pull = [a for a in alive if a not in done_this_round]

        if to_pull:
            log(f"Round {j}: to_pull={to_pull} (already ok this round: "
                f"{sorted(set(alive) - set(to_pull))})")
            specs = [{"arm": a, "j": j, "slot": pull.SLOTS[i % 2]} for i, a in enumerate(to_pull)]
            specs = pull.resolve_sticky_slots(specs, log=log)
            run_batch_two_wide(specs, run_one, claimable_fn=claimable_fn, log=log, sleep_fn=sleep_fn,
                                should_stop_fn=_t_cap_reached)

            pulls_df = read_pulls_fn()
            if _t_cap_reached():
                decision = scheduler.decide(pulls_df, sigma_e, delta=delta, t_max=t_max)
                log_ranking(decision, log=log)
                log(f"=== DONE: race complete (T-cap t_max={t_max} reached mid-round "
                    f"{j}) -- survivors={decision['survivors']} "
                    f"eliminated={decision['eliminated']} ===")
                return decision

            still_missing = [a for a in to_pull if a not in _ok_arms_at_round(pulls_df, j)]
            if still_missing:
                msg = (
                    f"!!! Round {j}: arm(s) {still_missing} did NOT produce an ok pull even "
                    f"after run_pull's own retry -- refusing to call the scheduler on an "
                    f"incomplete round (would risk mis-eliminating them). HUMAN INTERVENTION "
                    f"REQUIRED -- investigate ledger/pull_logs/ for "
                    f"{[pull.pull_id_for(a, j) for a in still_missing]}, then re-run "
                    f"run_race.py (it resumes here, retrying only the still-missing arm(s))."
                )
                log(msg)
                raise RuntimeError(msg)
        else:
            log(f"Round {j}: already fully complete for {alive} on resume -- re-deciding")

        decision = scheduler.decide(pulls_df, sigma_e, delta=delta, t_max=t_max)
        log_ranking(decision, log=log)
        if decision["done"]:
            log(f"=== DONE: race complete after round {j} -- survivors="
                f"{decision['survivors']} eliminated={decision['eliminated']} ===")
            return decision
        # loop back around: next iteration re-derives (round, roster) fresh.


# =============================================================================
# 6. dry-run reporting (read-only, never launches/writes anything)
# =============================================================================

def _read_pulls_df():
    try:
        return ledger.read("pulls")
    except FileNotFoundError:
        return pd.DataFrame(columns=["pull_id", "arm", "round_j", "delta", "status"])


def dry_run_report(read_pulls_fn=None, cfg_path=None, e_manifest_path=None,
                    arms_spec=None, log=print) -> dict:
    """Print (and return) exactly what `main()` would do next, given the
    CURRENT on-disk ledger/config.yaml state -- read-only, never launches
    training/serving/eval, never writes to the ledger or config.yaml (not
    even the gradient_analysis note or the noise_floor block) -- always safe
    to run against the REAL ledger, at any time, including while another
    real process (the baseline chain, or a live run_race) is running."""
    read_pulls_fn = _read_pulls_df if read_pulls_fn is None else read_pulls_fn
    log("=== run_race --dry-run: planned next action (read-only) ===")

    pre = preconditions_status(cfg_path=cfg_path, e_manifest_path=e_manifest_path)
    log(f"preconditions: baseline_ready={pre['baseline_ready']} e_manifest_ready={pre['e_manifest_ready']}")
    if not pre["ready"]:
        log("NEXT ACTION: would keep polling for preconditions (not yet met)")
        return {"phase": "waiting_preconditions", **pre}

    pulls_df = read_pulls_fn()
    missing = missing_null_rounds(pulls_df)
    if missing:
        log(f"NEXT ACTION: Phase NULLS -- would run null pull(s) j={missing}")
        return {"phase": "nulls", "missing": missing}

    _, _, sigma_e_eval = load_baseline(cfg_path=cfg_path)
    ok_null = pulls_df[(pulls_df["arm"] == "null") & (pulls_df["status"] == "ok")]
    null_deltas = [float(x) for x in ok_null.sort_values("round_j")["delta"]]
    sigma_e_null = float(np.std(null_deltas, ddof=1)) if len(null_deltas) > 1 else 0.0
    sigma_e = max(sigma_e_null, sigma_e_eval)

    arms_spec = clustering.load_arms_yaml() if arms_spec is None else arms_spec
    all_arms = frozen_arm_names(arms_spec)
    j, alive = _current_round_and_roster(pulls_df, all_arms, sigma_e, config.DELTA_CONF,
                                          config.T_MAX_PULLS, log=log)
    if j is None:
        log("NEXT ACTION: race is already DONE -- nothing left to pull")
        return {"phase": "done", "decision": alive}

    done_this_round = _ok_arms_at_round(pulls_df, j)
    to_pull = [a for a in alive if a not in done_this_round]
    log(f"NEXT ACTION: Phase RACE round {j} -- would pull {to_pull} (already ok this round: "
        f"{sorted(set(alive) - set(to_pull))}), sigma_e={sigma_e:.4f}")
    return {"phase": "race", "round": j, "to_pull": to_pull, "sigma_e": sigma_e}


# =============================================================================
# 7. top-level orchestration
# =============================================================================

def _make_eval_fn(workers=EVAL_WORKERS):
    """eval_fn for pull.run_pull: ALWAYS resumable (emergency null-takeover
    fix -- resume=True is unconditional, not an opt-in flag) -- a pull's eval
    step can be killed and rerun (crashed worker, box-wide contention, a
    human takeover) at any point and this eval_fn will pick up only the
    still-missing (start_id, repeat_idx) pairs for that pull's policy_id
    rather than redoing (or double-counting) whatever's already durably in
    the ledger. See eval_set.eval_checkpoint's `resume` docstring for why
    this is a provable no-op the first time any given policy_id is
    evaluated (nothing to resume from yet). `workers` defaults to
    EVAL_WORKERS (4 == parallel, since task-ledgerlock-report.md,
    2026-07-29; None would mean serial -- see that constant's own comment
    for the full history and why this default changed)."""
    def eval_fn(port, policy_id, arm, pull_id):
        return eval_set.eval_checkpoint(port, policy_id, arm, pull_id, workers=workers, resume=True)
    return eval_fn


def main(log=print, sleep_fn=time.sleep, workers=EVAL_WORKERS,
         read_pulls_fn=None, run_pull_fn=pull.run_pull,
         claimable_fn=both_slots_claimable) -> dict:
    """The full, resumable run_race entry point: gradient_analysis note (one
    -time) -> wait for preconditions -> Phase NULLS -> Phase RACE. Every
    injectable seam here defaults to the real implementation; tests pass in
    synthetic `read_pulls_fn`/`run_pull_fn`/`claimable_fn`/`sleep_fn` so this
    function's CONTROL FLOW is fully exercised without a GPU, policy server,
    or real ledger."""
    read_pulls_fn = _read_pulls_df if read_pulls_fn is None else read_pulls_fn

    pull.append_gradient_analysis_note_to_config_yaml(log=log)
    wait_for_preconditions(sleep_fn=sleep_fn, log=log)

    b, per_stratum_b, sigma_e_eval = load_baseline()
    B = load_frozen_B()
    arms_spec = clustering.load_arms_yaml()
    all_arms = frozen_arm_names(arms_spec)

    pool_df = pool.build_pool_table(write=False)
    models = map_fit.load()
    regions = wells.assign_regions(pool_df, models, arms_spec)   # cached once, per module docstring
    e_features = eval_set.load_manifest()
    eval_fn = _make_eval_fn(workers=workers)

    def run_one(spec):
        arm, j, slot = spec["arm"], spec["j"], spec["slot"]
        gpu = SLOT_GPU[slot]
        log(f"pull start: arm={arm!r} round={j} slot={slot} gpu={gpu}")
        if arm == "null":
            row = run_pull_fn("null", j, slot, B=0, eval_fn=eval_fn,
                               baseline=b, baseline_per_stratum=per_stratum_b, gpu=gpu, log=log)
        else:
            row = run_pull_fn(arm, j, slot, B, eval_fn=eval_fn,
                               pool_df=pool_df, regions=regions, e_features=e_features,
                               baseline=b, baseline_per_stratum=per_stratum_b, gpu=gpu, log=log)
        log(f"pull done: pull_id={row['pull_id']} status={row['status']}")
        return row

    sigma_e = run_null_phase(read_pulls_fn, run_one, sigma_e_eval, claimable_fn=claimable_fn, log=log, sleep_fn=sleep_fn)
    decision = run_race_phase(sigma_e, read_pulls_fn, run_one, all_arms,
                               claimable_fn=claimable_fn, log=log, sleep_fn=sleep_fn)
    return decision


def _main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="print the planned next action from the current ledger/config "
                          "state and exit -- never launches training/serving/eval, never "
                          "writes to the ledger or config.yaml.")
    args = ap.parse_args()

    if args.dry_run:
        dry_run_report(log=print)
        return

    main(log=print)


if __name__ == "__main__":
    _main()
