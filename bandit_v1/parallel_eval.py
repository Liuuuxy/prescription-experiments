"""N-parallel-worker rollout engine (bandit_v1 rollout-speedup #1): the same
served-policy rollout work bandit_v1/rollout.py's `run()` does serially (one
env, one websocket client, episodes appended to ledger table "episodes" one at
a time), but split across `workers` OS subprocesses hitting the SAME policy
server, each subprocess driving its own persistent robocasa env over its own
shard of `start_dirs`.

Why sharding is by whole start, not by (start, repeat) pair: 0a2ac49 added
states.restore's warm-model fast path, which is ~10x cheaper for a REPEAT of
the same start on the SAME env (identical model.xml already compiled into
env.sim.model) than for a genuinely different start. Splitting one start's
repeats across two workers would mean two separate envs each compile that
start's model at least once, throwing the whole speedup away for exactly the
episodes it was built for. `shard_round_robin` below hands every start_dir's
FULL run of `repeats` to exactly one worker, so each worker's own
bandit_v1.rollout.run call sees the same "all repeats of a start, back to
back" access pattern the serial engine already had, per-worker.

Why subprocesses (not threads): the served policy is the shared, external
resource here, not anything in-process -- each worker needs its own
bandit_v1.rollout.run call, which owns a persistent mujoco env AND a
websocket_client_policy connection; mujoco/robosuite state is emphatically not
thread-safe (states.py's whole restore() fast-path design leans on exactly one
env's internal C++/mujoco state), so N independent OS processes, each with its
own env and its own websocket connection to the one served policy, is the
correct isolation boundary. See this module's own investigation notes (in the
task report) on openpi's websocket_policy_server.py: the server DOES accept
multiple concurrent client connections (asyncio, one handler coroutine per
connection), but its `_handler` calls `self._policy.infer(obs)` as a plain
synchronous (non-awaited) call inside that single-threaded event loop --
meaning inference itself is serialized across every connected client
regardless of worker count. Multiple workers are still safe (no corruption:
each connection has its own local packer/timing state, and the shared
`_policy` object is only ever touched from the one event-loop thread) and
still speed up the NON-inference latency of a rollout (mujoco stepping,
image prep, warm-restore, wall-clock spent waiting on the network round trip)
running in parallel across workers -- just not the served model's own compute,
which remains a shared, serialized resource.

Concurrency-safety mechanics: ledger.append_rows (ledger.py) is a whole-
parquet read-modify-write against table "episodes" and was never meant to be
called by more than one process at a time. Rather than touch that contract,
each worker's rollout.run call is given a `episodes_sink` (rollout.py's new
seam) that redirects every completed row to that worker's OWN shard file
(bandit_v1/ledger/shards/<run-tag>_<worker>.parquet, via ledger.
append_rows_to_path -- a SEPARATE, per-path read-modify-write with no shared
state between workers). Only after every worker subprocess has exited does the
parent call `merge_shards`, which reads every shard file that exists,
concatenates them, and appends the union to "episodes" in exactly ONE
append_rows call -- so the shared table is still only ever written by a
single process, still only ever via the existing, unmodified append_rows.

Crash isolation: a worker that dies partway (bad action, OOM, policy server
hiccup, ...) has already durably written every episode it completed so far to
its own shard file (each row is its own append_rows_to_path call, same
per-row-durability convention as the serial engine); `run_parallel` still
merges every OTHER (surviving) shard -- including that worker's own partial
one -- and only THEN raises, carrying the dead worker's log tail, so a crash
never silently loses already-completed episodes and never silently hides the
failure either.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

from . import config, ledger, rollout

WORKER_TMP_DIR = Path("/data/xinyua11/tmp")   # per-worker spec + log files
SHARDS_SUBDIR = "shards"                      # bandit_v1/ledger/shards/<run-tag>_<worker>.parquet
LOG_TAIL_LINES = 40
DEFAULT_WORKER_TIMEOUT_S = 10800              # 3h -- a genuinely hung worker (stuck rollout,
                                               # dead policy connection that never errors) must
                                               # not block run_parallel forever; see run_parallel's
                                               # `timeout` kwarg docstring.


# =============================================================================
# Sharding (pure, unit-testable)
# =============================================================================

def shard_round_robin(start_dirs, workers):
    """Split `start_dirs` into exactly `workers` shards by round-robin
    assignment (start i -> worker i % workers), each shard keeping its
    elements in their original relative order (shard j == start_dirs[j::
    workers]). Every input start_dir appears in EXACTLY ONE output shard,
    exactly once -- the union of all shards, in shard order then within-shard
    order, is a permutation of `start_dirs` grouped by worker, never a split
    of any single start_dir's repeats (repeats happen entirely inside one
    rollout.run call per shard -- see module docstring for why keeping a
    start's repeats together matters). Balanced to within 1 element across
    shards (Python's own slice semantics: len(start_dirs) items over `workers`
    slices differ in length by at most 1). Raises ValueError if `workers` < 1.
    """
    if workers < 1:
        raise ValueError(f"shard_round_robin: workers must be >= 1, got {workers}")
    start_dirs = list(start_dirs)
    return [start_dirs[i::workers] for i in range(workers)]


def _make_run_tag(phase, policy_id, pull_id=None) -> str:
    """Unique-enough tag identifying one run_parallel call, used to namespace
    its shard/spec/log files so concurrent run_parallel calls (or repeated
    calls in a long process) never collide."""
    base = f"{phase}_{policy_id}" + (f"_{pull_id}" if pull_id else "")
    return f"{base}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


# =============================================================================
# Worker-side: what one subprocess actually does
# =============================================================================

def run_worker_inline(spec: dict) -> None:
    """The real work item for one worker: reconstruct `skip_pairs` from its
    JSON-safe form ([[start_id, repeat_idx], ...] -> set of tuples) and call
    rollout.run over this worker's shard of start_dirs, with `episodes_sink`
    redirecting every completed row into this worker's OWN shard file
    (ledger.append_rows_to_path) instead of the shared "episodes" table --
    see module docstring's "Concurrency-safety mechanics" section for why.

    Kept separate from the CLI entry point (`_main`) below so it is directly
    unit-testable with a monkeypatched `rollout.run` and a plain in-memory
    spec dict -- no subprocess, no spec file on disk, no live env or policy
    server involved.

    Progress visibility (task-importhang-report.md): `rollout.run`'s own
    per-episode loop prints nothing at all -- a worker's stdout, once
    robosuite/robocasa's import-time warnings are done, is otherwise
    permanently silent for the rest of its run, whether it is healthy and
    still grinding through a long shard or genuinely stuck. This made a
    worker's log tail useless for telling the two apart in practice: a
    fully healthy, successfully-completed 4-worker run's logs were verified
    to end at the EXACT SAME "mimicgen not installed" line a genuinely
    hung run's would, since that line is the last thing printed before
    `rollout.run` is ever called. `sink` below now prints one flushed line
    per completed episode (count + elapsed seconds), plus a marker line
    right before `rollout.run` is called at all, so a worker that is merely
    slow (or has every pair already in skip_pairs -- see the empty-shard
    case) is now visibly distinguishable, in its own log file, from one
    that truly never advances -- see `_worker_env`'s PYTHONUNBUFFERED
    default for why these prints reliably reach the log file promptly.
    """
    skip_pairs = {tuple(p) for p in (spec.get("skip_pairs") or [])}
    start_dirs = [Path(p) for p in spec["start_dirs"]]
    shard_path = Path(spec["shard_path"])

    t_start = time.time()
    n_done = 0

    def sink(row):
        nonlocal n_done
        ledger.append_rows_to_path(shard_path, [row])
        n_done += 1
        print(f"parallel_eval worker: {n_done} episode(s) written to shard "
              f"(elapsed {time.time() - t_start:.1f}s)", flush=True)

    print(f"parallel_eval worker: starting rollout.run over {len(start_dirs)} "
          f"start(s), {spec['repeats']} repeat(s) each ({len(skip_pairs)} pair(s) "
          "already done upstream, will be skipped)", flush=True)
    rollout.run(
        spec["policy_host"], spec["policy_port"], start_dirs, spec["repeats"],
        phase=spec["phase"], policy_id=spec["policy_id"],
        arm=spec.get("arm"), pull_id=spec.get("pull_id"),
        skip_pairs=(skip_pairs if skip_pairs else None),
        episodes_sink=sink,
    )
    print(f"parallel_eval worker: done, {n_done} new episode(s) written in "
          f"{time.time() - t_start:.1f}s", flush=True)


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="path to a JSON worker spec file")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    run_worker_inline(spec)


if __name__ == "__main__":
    _main()


# =============================================================================
# Parent-side: spawn, wait, merge
# =============================================================================

# Per-worker resource caps (task-importhang-report.md): without these, every
# worker's numpy/BLAS AND Mesa's llvmpipe (the osmesa software renderer
# bandit_v1's CPU-only eval envs use) size their own thread pools to
# os.cpu_count() by default -- verified directly on this box: a single
# worker with none of these set spins up ~322 OS threads (32 of them
# llvmpipe render-pool workers) for just ONE persistent rollout env.
# `workers` concurrent worker subprocesses then each independently
# oversubscribe to that SAME core count, which is a property of how many
# workers are spawned, not of what else is running on the box at the time
# (consistent with the "reproduced on an idle box" observation in
# run_race.py's EVAL_WORKERS comment). Capping each pool to a small fixed
# size bounds every worker's thread footprint regardless of `workers`,
# without touching rollout.py/states.py's rendering or numeric code at all.
#
# PYTHONUNBUFFERED=1 additionally guarantees run_worker_inline's own
# progress prints (see its docstring) reach the log file promptly instead
# of sitting in an unflushed block buffer -- `-m bandit_v1.parallel_eval`
# is invoked below without `-u`, so a worker's stdout, redirected to a log
# FILE (not a tty), is block-buffered by default.
#
# Every key here is only a DEFAULT (`dict.setdefault`): a caller that has
# already set any of these in its own environment before calling
# run_parallel keeps that value untouched.
_WORKER_THREAD_CAP = "2"
_WORKER_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": _WORKER_THREAD_CAP,
    "OPENBLAS_NUM_THREADS": _WORKER_THREAD_CAP,
    "MKL_NUM_THREADS": _WORKER_THREAD_CAP,
    "NUMEXPR_NUM_THREADS": _WORKER_THREAD_CAP,
    "LP_NUM_THREADS": _WORKER_THREAD_CAP,   # Mesa llvmpipe (osmesa) render-thread pool
    "PYTHONUNBUFFERED": "1",
}


def _worker_env() -> dict:
    """The full parent environment (`os.environ`), with `_WORKER_ENV_DEFAULTS`
    filled in for any key the caller has not already set (`setdefault`
    semantics -- an explicit, pre-existing value in the parent's own
    environment always wins over these defaults). See
    `_WORKER_ENV_DEFAULTS`'s comment for why these particular defaults
    exist."""
    env = os.environ.copy()
    for k, v in _WORKER_ENV_DEFAULTS.items():
        env.setdefault(k, v)
    return env


def _spawn_worker(spec_path, log_path):
    """Launch one worker subprocess: the SAME python interpreter this parent
    process is already running under (`sys.executable` -- "direct env
    python", i.e. whichever conda env is already active, not a `conda run -n
    ...` wrapper), invoked as `python -m bandit_v1.parallel_eval --spec
    <spec_path>`, with stdout+stderr both redirected to `log_path` and the
    full parent environment forwarded (`_worker_env()`, see its docstring)
    so MUJOCO_GL/CUDA_VISIBLE_DEVICES/etc. still come from whatever the
    caller already has set, never hardcoded here -- only a small, fixed set
    of thread-count/output-buffering env vars get a DEFAULT value layered on
    top when the caller has not already set them. Returns the
    `subprocess.Popen` handle."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        return subprocess.Popen(
            [sys.executable, "-m", "bandit_v1.parallel_eval", "--spec", str(spec_path)],
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(config.REPO), env=_worker_env(),
        )


def _log_tail(log_path, n_lines=LOG_TAIL_LINES) -> str:
    p = Path(log_path)
    if not p.exists():
        return "(no log file)"
    lines = p.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def merge_shards(shard_paths, table="episodes") -> list:
    """Read every path in `shard_paths` that actually exists (a worker whose
    shard was empty -- ran zero episodes, e.g. every one of its pairs was
    already in `skip_pairs` -- or that crashed before completing even one
    episode never created a file at all; both are simply skipped, not an
    error), concatenate their rows in the given order (worker-index order, by
    construction of `shard_paths`), append the UNION to ledger table `table`
    in exactly ONE `ledger.append_rows` call, then delete every shard file
    that was read. Returns the combined rows as a list of dicts -- the same
    shape rollout.run/run_parallel return.

    Zero existing shard files (every worker crashed before its first episode,
    or every shard was empty) returns [] and performs no ledger write at all
    -- consistent with rollout.run's own zero-new-episodes behavior (see
    test_rollout.py's "skips entire start" case: no episodes.parquet file is
    created for a run that appends nothing)."""
    frames, existing = [], []
    for p in shard_paths:
        p = Path(p)
        if p.exists():
            frames.append(ledger.read_path(p))
            existing.append(p)
    if not frames:
        return []

    combined = pd.concat(frames, ignore_index=True)
    rows = combined.to_dict("records")
    ledger.append_rows(table, rows)
    for p in existing:
        p.unlink()
    return rows


def run_parallel(policy_host, policy_port, start_dirs, repeats, phase, policy_id,
                  workers=4, arm=None, pull_id=None, skip_pairs=None,
                  spawn_fn=None, log_dir=None, spec_dir=None,
                  timeout=DEFAULT_WORKER_TIMEOUT_S) -> list:
    """Parallel counterpart to bandit_v1.rollout.run: same return contract (a
    list of the completed-episode row dicts, already durably merged into
    ledger table "episodes"), but shards `start_dirs` round-robin across
    `workers` OS subprocesses (see `shard_round_robin`'s docstring for why
    ALL repeats of one start always land on the same worker/env), each
    running rollout.run against the SAME served policy at (policy_host,
    policy_port) over its own shard, writing to its own ledger shard file
    (never the shared "episodes" table -- see module docstring). The parent
    waits for every worker to exit, then merges whatever shard files exist
    (`merge_shards`) -- including a crashed worker's partial shard -- into
    "episodes" with a single append_rows call, and only THEN raises (if any
    worker exited nonzero), carrying that worker's log tail in the message,
    so a crash never silently discards already-completed episodes and never
    silently hides the failure.

    `skip_pairs` is accepted and forwarded through to every worker exactly
    like rollout.run's own parameter (bandit_v1's resume mechanism,
    run_diagnosis.py-style): passed through UNFILTERED to every worker's spec
    (matching run_diagnosis.py's own "pass the full done_pairs() set, not a
    per-chunk-filtered slice" convention -- see run_diagnosis.py's
    test_run_all_skips_fully_done_chunk_and_resumes_partial_chunk docstring),
    since rollout.run inside each worker only ever matches entries against
    its own shard's start_ids anyway.

    `workers` <= 1 falls back directly to a plain rollout.run call (no
    subprocess, no sharding, no shard files) -- this function is meant to be
    called only when the caller has already decided to go parallel (e.g.
    eval_set.eval_checkpoint's `workers` opt-in), but staying correct (if
    wasteful of the shard machinery) for a degenerate `workers=1` caller is
    free and avoids a footgun.

    `spawn_fn` (default `_spawn_worker`), `log_dir`/`spec_dir` (default
    WORKER_TMP_DIR, i.e. /data/xinyua11/tmp) are injection seams for tests --
    no real subprocess, GPU, env, or policy server is needed to exercise the
    sharding/merge/crash-handling logic this function owns.

    `timeout` (default DEFAULT_WORKER_TIMEOUT_S, 3h): per-worker wall-clock
    cap on `proc.wait()`. A worker that exits normally (0 or nonzero) is
    handled exactly as before, regardless of how long it took; `timeout`
    only bounds a worker that never exits at all (a genuinely hung rollout --
    e.g. a policy connection that blocks forever instead of erroring, which
    a plain unbounded `.wait()` would never surface). On breach, that
    worker's process is killed (`proc.kill()`, then reaped with a short
    follow-up wait so it never becomes a zombie) -- its shard file, if any
    rows were written before it hung, is left exactly where it is (workers
    write per-episode via `ledger.append_rows_to_path`, so whatever
    completed before the hang is already durable) and IS still merged, same
    as any other worker's partial/crashed shard. The timeout is surfaced in
    the raised error exactly like a nonzero-exit failure (log tail included,
    `failures` entry present), just labeled as a timeout rather than an exit
    code.
    """
    if workers is None or workers <= 1:
        return rollout.run(policy_host, policy_port, start_dirs, repeats, phase=phase,
                            policy_id=policy_id, arm=arm, pull_id=pull_id,
                            skip_pairs=skip_pairs)

    spawn_fn = _spawn_worker if spawn_fn is None else spawn_fn
    log_dir = WORKER_TMP_DIR if log_dir is None else Path(log_dir)
    spec_dir = WORKER_TMP_DIR if spec_dir is None else Path(spec_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    run_tag = _make_run_tag(phase, policy_id, pull_id)
    shards = shard_round_robin(list(start_dirs), workers)
    skip_list = [[sid, r] for (sid, r) in (skip_pairs or ())]
    shard_dir = Path(config.LEDGER_DIR) / SHARDS_SUBDIR

    procs = []  # [{"worker", "proc", "log_path", "shard_path"}, ...]
    for i, shard in enumerate(shards):
        if not shard:
            continue  # fewer start_dirs than workers -- nothing for this worker to do
        shard_path = shard_dir / f"{run_tag}_{i}.parquet"
        spec = {
            "policy_host": policy_host, "policy_port": policy_port,
            "start_dirs": [str(Path(p)) for p in shard],
            "repeats": repeats, "phase": phase, "policy_id": policy_id,
            "arm": arm, "pull_id": pull_id, "skip_pairs": skip_list,
            "shard_path": str(shard_path),
        }
        spec_path = spec_dir / f"{run_tag}_worker{i}_spec.json"
        spec_path.write_text(json.dumps(spec))
        log_path = log_dir / f"{run_tag}_worker{i}.log"

        proc = spawn_fn(spec_path, log_path)
        procs.append({"worker": i, "proc": proc, "log_path": log_path, "shard_path": shard_path})

    failures = []
    for p in procs:
        try:
            rc = p["proc"].wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p["proc"].kill()
            try:
                p["proc"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # best-effort reap; a kill -9 that still won't die is not this function's problem
            failures.append({"worker": p["worker"], "returncode": None, "timed_out": True,
                              "log_tail": _log_tail(p["log_path"])})
            continue
        if rc != 0:
            failures.append({"worker": p["worker"], "returncode": rc, "timed_out": False,
                              "log_tail": _log_tail(p["log_path"])})

    combined_rows = merge_shards([p["shard_path"] for p in procs])

    if failures:
        detail = "\n".join(
            (f"--- worker {f['worker']} TIMED OUT after {timeout}s (killed) ---\n{f['log_tail']}"
             if f["timed_out"] else
             f"--- worker {f['worker']} exited {f['returncode']} ---\n{f['log_tail']}")
            for f in failures)
        raise RuntimeError(
            f"run_parallel: {len(failures)}/{len(procs)} worker(s) failed "
            f"(surviving shards already merged into ledger table 'episodes', "
            f"{len(combined_rows)} rows kept)\n{detail}")

    return combined_rows
