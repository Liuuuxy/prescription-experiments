"""Pull orchestrator: draw -> dataset -> slot train -> eval -> ledger
(bandit_v1 Task 12).

Design: weakregion/BANDIT_V1_DESIGN.md section 3 ("One pull") + section 6
(ledger schema). This module is the ONE function that executes a single pull
of arm `arm` at round `j` on training slot `slot`:

  1. draw B demo episode_index values for `arm` (draw.pull_demos), skipped
     entirely for the noise-measurement arm "null" (demo_ids = [] always --
     "dataset = D0 alone", per the brief).
  2. materialize `FT_ARMS_ROOT/ppc2sink_bandit_<arm>_j<j>` = D0 + demo_ids
     (via policy_analysis/build_lerobot_subset.py) and point
     `FT_ARMS_ROOT/ppc2sink_bandit_slot_<slot>` at it (`ln -sfn` semantics).
  3. train the slot's openpi TrainConfig (`pi0_ppc2sink_bandit_a`/`_b`,
     Task 5) at seed `config.pull_seed(j)`, retrying once on failure.
  4. serve the resulting checkpoint on the slot's port and call the
     caller-supplied `eval_fn`.
  5. compute delta vs an (optional, caller-supplied) baseline and append the
     full row to `ledger/pulls.parquet`.

Two interface seams, both because their upstream tasks do not exist yet at
the time this module was written (Task 9's `eval_set.eval_checkpoint`, Task
10's fitted region/cluster model + `arms.yaml`) -- exactly draw.py's own
"zero import-time dependency on Task 10" stance, extended to this module:

  - `eval_fn(policy_port, policy_id, arm, pull_id) -> dict` is injected by
    the caller. When `None` (the default), the eval step raises
    `NotImplementedError("Task 9 eval_set not built yet")` -- loudly, not
    silently skipped, so a real run_race.py invocation can never mistake "no
    evaluator wired up" for "evaluated, scored zero". The expected return
    shape (documented once here, not re-derived per call site):
        {"per_repeat_means": [float, ...],          # len == however many
                                                      # repeats the real
                                                      # eval_checkpoint ran
                                                      # internally (e.g.
                                                      # config.EVAL_REPEATS,
                                                      # or 1 for a smoke)
         "per_stratum_means": {stratum: [float, ...], ...}}   # same
                                                      # per-repeat shape,
                                                      # per E-set stratum
    `eval_fn` is responsible for actually serving-adjacent concerns like
    running the EVAL_REPEATS-many rollout passes over E and writing their
    raw per-episode rows to the shared ledger "episodes" table (same table
    run_diagnosis.py's diag phase already writes to) -- this module only
    serves the checkpoint and hands eval_fn the port to talk to.
  - `regions` (pd.Series, episode_index -> arm name) and `e_features`
    (pd.DataFrame: category, x_rel, y_rel) are Task 10 outputs `draw.
    pull_demos` needs; since Task 10 doesn't exist yet either, they are
    required caller-supplied arguments for any non-null arm (there is no
    sane default to fall back to) -- passing `None` for either raises
    immediately with a message pointing at this seam, rather than pull.py
    silently importing a file that doesn't exist. `pool_df` DOES have a
    real, available-today default: `pool.build_pool_table(write=False)`.

Verified-vs-brief CLI deviation (train.py flags): the original task-5/12
briefs write `--exp_name`/`--num_train_steps` (underscores). Task 5's actual
`conda run -n openpi python scripts/train.py <config> --help` run (recorded
in bandit_v1/ledger/config.yaml's `cli_flags_confirmed` block) showed tyro
renders these fields as `--exp-name`/`--num-train-steps` (dashes) -- the
underscore spelling does not parse. This module uses the VERIFIED dash
flags (`train_cmd` below), not the brief's literal (unverified) text, for
the same reason Task 5 did: a command that doesn't parse is not "close
enough". Same principle applied to policy_analysis/build_lerobot_subset.py:
its real argparse (--src/--arms/--which/--dst) differs from the original
brief's guess (--episodes/--out); this module wires the real one (see
`build_dataset_cmd`).

Every subprocess-launching / network-touching helper below takes an
injectable seam (`popen_fn`, `dataset_runner`, `connect_fn`, `sleep_fn`)
defaulting to the real implementation, so `run_pull`'s full control flow
(retry-on-failure, dry_run early-return, row schema) is unit-testable
without a GPU, a policy server, or real disk I/O against the (huge)
mimicgen lerobot pool. `dry_run=True` performs REAL dataset materialization
+ symlinking (cheap file-copy/dir-listing work, no GPU) and REAL config/
command resolution, then returns before ever touching training or serving
-- "stops after dataset materialization + config resolution", per the
brief; it is "pure-testable" via the same `dataset_runner` injection every
other path uses, not by skipping that step outright.
"""
import json
import os
import socket
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config, draw, ledger, pool

# --- pull.py-local constants (task-specific, same precedent as rollout.py's
# RESIZE_SIZE/REPLAN_STEPS/etc. living in the module that uses them rather
# than in the shared config.py) -------------------------------------------
OPENPI_PY = Path("/data/xinyua11/conda/envs/openpi/bin/python")
ROBOCASA_PY = Path("/data/xinyua11/conda/envs/robocasa/bin/python")

SLOTS = ("a", "b")
SLOT_TRAIN_CONFIG = {"a": "pi0_ppc2sink_bandit_a", "b": "pi0_ppc2sink_bandit_b"}
SLOT_PORT_BASE = 8130                     # slot "a" -> 8130, "b" -> 8131

TRAIN_GPU_NEED_MIB = 70000                # matches launch_pi0.sh's threshold
GPU_POLL_SECS = 300
CKPT_POLL_SECS = 60
PORT_WAIT_TRIES = 90
PORT_WAIT_SLEEP = 5
MAX_TRAIN_ATTEMPTS = 2                    # brief: "re-run once" == 2 total attempts

PULLS_TABLE = "pulls"
ARMS_SUBDIR = "pull_arms"                 # ledger/pull_arms/<pull_id>.json (provenance, kept)
LOGS_SUBDIR = "pull_logs"                 # ledger/pull_logs/<pull_id>_*.log


# --- 0. deterministic per-pull draw seed -----------------------------------

def pull_rng_seed(arm: str, j: int) -> int:
    """Deterministic, cross-process-stable seed for one pull's demo draw.

    The brief's literal formula was `hash((arm, j)) % 2**32`. Python's
    built-in `hash()` of a `str` (and therefore of any tuple containing one)
    is SALTED PER PROCESS -- `PYTHONHASHSEED` randomization has been on by
    default since Python 3.3 specifically to stop str-hash-based attacks,
    which means `hash(("random", 3))` returns a DIFFERENT value every time a
    fresh `python` process starts. bandit_v1's entire reproducibility model
    depends on a pull's demo draw being a pure function of `(arm, j)` alone
    -- a retry after a crash, an audit script run later, or (this task's
    stated deferral) an actual smoke run in a separate process from this
    one's tests must all draw the IDENTICAL demo_ids for the same (arm, j),
    or "redraw every pull" (design section 3) silently stops meaning what it
    says. `zlib.crc32` is an unsalted, fully deterministic 32-bit checksum
    with no process-identity dependency at all -- the same string always
    produces the same integer on any Python process, any machine, forever.
    """
    return zlib.crc32(f"{arm}:{j}".encode()) % (2 ** 32)


# --- 1. pull_id / path helpers (pure) ---------------------------------------

def pull_id_for(arm: str, j: int) -> str:
    return f"{arm}_j{j}"


def dataset_dir_for(arm: str, j: int) -> Path:
    return config.FT_ARMS_ROOT / f"ppc2sink_bandit_{arm}_j{j}"


def slot_symlink_for(slot: str) -> Path:
    if slot not in SLOTS:
        raise ValueError(f"slot_symlink_for: unknown slot {slot!r}, expected one of {SLOTS}")
    return config.FT_ARMS_ROOT / f"ppc2sink_bandit_slot_{slot}"


def slot_port(slot: str) -> int:
    if slot not in SLOTS:
        raise ValueError(f"slot_port: unknown slot {slot!r}, expected one of {SLOTS}")
    return SLOT_PORT_BASE + SLOTS.index(slot)


# --- 2. episode-list assembly (pure) ----------------------------------------

def load_d0_episode_ids(arms_json=None) -> list:
    """D0 = weakregion/arms.json's "base_episodes" list (400 ints, config.
    ARMS_JSON by default). Same source pool.py's `in_d0` flag uses."""
    p = arms_json if arms_json is not None else config.ARMS_JSON
    return list(json.load(open(p))["base_episodes"])


def assemble_episode_ids(d0_ids: list, demo_ids: list) -> list:
    """D0 + demo_ids, order-preserving concat, D0 first.

    demo_ids are always drawn from `pool.well_mask` (== NOT in D0, see
    draw.py's `pull_demos`), so overlap with D0 should be structurally
    impossible; this is a defensive assertion, not a silent dedup -- a
    silent dedup would quietly shrink a pull's EFFECTIVE new-data count
    below B without anyone noticing, exactly the kind of drift the design's
    invariants section (§9) warns corrupts the experiment. A null pull
    (demo_ids == []) degenerates to exactly D0, unchanged -- the brief's
    "null-pull = D0 alone".
    """
    d0_set, demo_set = set(d0_ids), set(demo_ids)
    overlap = d0_set & demo_set
    if overlap:
        raise ValueError(f"assemble_episode_ids: demo_ids overlap D0: {sorted(overlap)}")
    if len(demo_set) != len(demo_ids):
        seen, dups = set(), set()
        for e in demo_ids:
            (dups.add(e) if e in seen else seen.add(e))
        raise ValueError(f"assemble_episode_ids: duplicate demo_ids: {sorted(dups)}")
    return list(d0_ids) + list(demo_ids)


# --- 3. dataset materialization (build_lerobot_subset.py wiring) -----------

def pull_arms_json_path(pull_id: str) -> Path:
    return config.LEDGER_DIR / ARMS_SUBDIR / f"{pull_id}.json"


def write_pull_arms_json(pull_id: str, d0_ids: list, demo_ids: list, path=None):
    """Write a small arms.json-SHAPED file -- the only interface policy_
    analysis/build_lerobot_subset.py's real argparse exposes is `--arms
    <json with "<key>_episodes" lists> --which <'+'-joined keys>`, and our
    per-pull episode list is not one of the named lists that script's
    author anticipated (weakregion/arms.json only has base/core/random).
    Kept under ledger/pull_arms/ as provenance (audit trail of exactly which
    episode_index list produced dataset_path) rather than deleted after use.

    Returns (path, which_str); `which` is "base" alone when demo_ids is
    empty (null pull) so build_lerobot_subset.py never has to special-case
    an empty second key -- it simply is not given one.
    """
    path = Path(path) if path is not None else pull_arms_json_path(pull_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"base_episodes": list(d0_ids), "pull_episodes": list(demo_ids)}
    path.write_text(json.dumps(payload))
    which = "base" if not demo_ids else "base+pull"
    return path, which


def build_dataset_cmd(arms_json_path, which: str, dst, src=None) -> list:
    src = str(src if src is not None else config.POOL_LEROBOT)
    return [str(ROBOCASA_PY), "policy_analysis/build_lerobot_subset.py",
            "--src", src, "--arms", str(arms_json_path), "--which", which,
            "--dst", str(dst)]


def run_dataset_build(arms_json_path, which: str, dst, src=None,
                       runner=subprocess.run, log_path=None) -> list:
    """Actually invoke build_lerobot_subset.py -- ROBOCASA_PY's binary
    directly, NOT `conda run -n robocasa ...` (the pipe-buffering gotcha
    run_diagnosis.sh documents: `conda run` relays a long child's stdout
    through a non-line-flushed pipe, so a log file can sit empty for many
    minutes with real progress happening on disk). `runner` is injectable
    (default subprocess.run) purely so tests never touch config.POOL_LEROBOT
    for real. Raises FileExistsError up front (mirroring the script's own
    `assert not os.path.exists(a.dst)`, so the caller gets a clear error
    before spawning a process that would just immediately assert-fail) and
    RuntimeError on a nonzero exit.
    """
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(f"run_dataset_build: dst already exists: {dst}")
    cmd = build_dataset_cmd(arms_json_path, which, dst, src=src)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            result = runner(cmd, cwd=str(config.REPO), stdout=f, stderr=subprocess.STDOUT)
    else:
        result = runner(cmd, cwd=str(config.REPO))
    rc = getattr(result, "returncode", 0)
    if rc != 0:
        raise RuntimeError(f"build_lerobot_subset.py failed rc={rc}: {' '.join(cmd)}")
    return cmd


def symlink_slot(dataset_path, symlink_path) -> None:
    """`ln -sfn dataset_path symlink_path` semantics in pure Python: replace
    whatever currently sits at symlink_path (a stale symlink from a PRIOR
    pull that used this same slot, or nothing at all) with a fresh symlink
    pointing at `dataset_path`. Never touches or requires `dataset_path`
    itself to exist (a plain `symlink_to` call does not resolve/verify its
    target), which is what lets `dry_run` exercise this against a fake
    `dataset_runner` that never actually creates a directory."""
    symlink_path = Path(symlink_path)
    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(Path(dataset_path))


# --- 4. training config/command resolution (pure) ---------------------------

def train_config_name_for_slot(slot: str) -> str:
    try:
        return SLOT_TRAIN_CONFIG[slot]
    except KeyError:
        raise ValueError(f"train_config_name_for_slot: unknown slot {slot!r}, expected one of {SLOTS}")


def train_exp_name(arm: str, j: int) -> str:
    return f"{arm}_j{j}"


def train_cmd(config_name: str, exp_name: str, seed: int, num_train_steps=None) -> list:
    """argv for scripts/train.py. Flags are the VERIFIED dash form
    (--exp-name/--num-train-steps/--seed/--overwrite) recorded in bandit_v1/
    ledger/config.yaml's `cli_flags_confirmed` block (Task 5's `--help`
    check) -- see this module's docstring for why the brief's underscore
    spelling is not used. `--overwrite` is always passed (matches ft.sh /
    launch_pi0.sh): a retried attempt (this module's "re-run once" logic)
    must be able to overwrite a partial checkpoint dir left by the failed
    first attempt.
    """
    cmd = [str(OPENPI_PY), "scripts/train.py", config_name,
           "--exp-name", exp_name, "--seed", str(seed), "--overwrite"]
    if num_train_steps is not None:
        cmd += ["--num-train-steps", str(num_train_steps)]
    return cmd


def ckpt_root_dir(config_name: str, exp_name: str) -> Path:
    return config.OPENPI / "checkpoints" / config_name / exp_name


def ckpt_final_dir(config_name: str, exp_name: str, num_train_steps=None) -> Path:
    """Final checkpoint step dir name. scripts/train.py saves at `step ==
    config.num_train_steps - 1` unconditionally (in addition to the
    save_interval cadence) -- see scripts/train.py's checkpoint-save
    condition -- so a `num_train_steps` override (the brief's smoke-test
    `--num_train_steps 60`) changes the final step name to
    `num_train_steps - 1` (59, not config.FINAL_CKPT_STEP=19999). Only the
    full, un-overridden 20000-step recipe lands on FINAL_CKPT_STEP."""
    final_step = (num_train_steps - 1) if num_train_steps is not None else config.FINAL_CKPT_STEP
    return ckpt_root_dir(config_name, exp_name) / str(final_step)


# --- 5. GPU selection (mirrors launch_pi0.sh's wait-for-GPU loop) ----------

def free_mib(gpu: int, query=None) -> int:
    """MiB free on `gpu`, via nvidia-smi by default. `query` is injectable
    (real signature: `query(gpu) -> int`) so tests never shell out."""
    if query is not None:
        return query(gpu)
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)])
    return int(out.decode().strip())


def wait_for_free_gpu(gpus=(0, 1), need_mib=TRAIN_GPU_NEED_MIB, poll_secs=GPU_POLL_SECS,
                       query=None, sleep_fn=time.sleep, log=lambda *a: None) -> int:
    """Poll `gpus` in order every `poll_secs` until one has >= need_mib free;
    return its index. Identical policy to launch_pi0.sh's proven loop (same
    threshold/cadence) -- `run_pull` only calls this when its caller doesn't
    pass an explicit `gpu=`, which every test in this module does (to avoid
    faking nvidia-smi/real sleeping just to exercise the rest of the
    pipeline)."""
    while True:
        for g in gpus:
            f = free_mib(g, query=query)
            log(f"gpu{g} free={f}MB (need >= {need_mib})")
            if f >= need_mib:
                return g
        sleep_fn(poll_secs)


# --- 6. training launch + checkpoint wait -----------------------------------

def _train_env(gpu: int) -> dict:
    env = dict(os.environ)
    env.update(TMPDIR="/data/xinyua11/tmp", HF_HOME="/data/xinyua11/.cache/huggingface",
               MUJOCO_GL="egl", WANDB_MODE="offline", WANDB_DIR="/data/xinyua11/wandb",
               CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.9")
    return env


def launch_training(cmd: list, gpu: int, log_path, popen_fn=subprocess.Popen):
    """Start scripts/train.py as a background child (non-blocking Popen);
    the caller polls `wait_for_checkpoint` against the returned process's
    liveness. `popen_fn` is injectable so tests never spawn a real
    process."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    return popen_fn(cmd, cwd=str(config.OPENPI), env=_train_env(gpu), stdout=logf, stderr=subprocess.STDOUT)


def wait_for_checkpoint(proc, ckpt_dir, poll_secs=CKPT_POLL_SECS, sleep_fn=time.sleep,
                         exists_fn=None, poll_fn=None, log=lambda *a: None) -> bool:
    """Poll every `poll_secs` until EITHER `ckpt_dir` exists (True, success)
    OR `proc` has exited without producing it (False, failure) -- the same
    shape as run_diagnosis.sh's server-wait loop (`kill -0` check + break on
    death), applied to a checkpoint directory instead of an open port.
    `exists_fn`/`poll_fn` default to `Path(ckpt_dir).is_dir` / `proc.poll`
    and exist only so tests can override them; production callers should
    never need to."""
    exists_fn = exists_fn if exists_fn is not None else (lambda: Path(ckpt_dir).is_dir())
    poll_fn = poll_fn if poll_fn is not None else proc.poll
    while True:
        if exists_fn():
            return True
        rc = poll_fn()
        if rc is not None:
            log(f"wait_for_checkpoint: training process exited rc={rc} before {ckpt_dir} appeared")
            return False
        sleep_fn(poll_secs)


# --- 7. serving + eval hookup ------------------------------------------------

def serve_cmd(config_name: str, ckpt_dir, port: int) -> list:
    return [str(OPENPI_PY), "scripts/serve_policy.py", "--port", str(port),
            "policy:checkpoint", "--policy.config", config_name, "--policy.dir", str(ckpt_dir)]


def _serve_env(gpu: int) -> dict:
    env = dict(os.environ)
    env.update(TMPDIR="/data/xinyua11/tmp", HF_HOME="/data/xinyua11/.cache/huggingface",
               MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.25")
    return env


def launch_server(config_name: str, ckpt_dir, port: int, gpu: int, log_path, popen_fn=subprocess.Popen):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    cmd = serve_cmd(config_name, ckpt_dir, port)
    proc = popen_fn(cmd, cwd=str(config.OPENPI), env=_serve_env(gpu), stdout=logf, stderr=subprocess.STDOUT)
    return cmd, proc


def _default_connect(host: str, port: int) -> bool:
    try:
        s = socket.socket()
        s.settimeout(1.0)
        s.connect((host, port))
        s.close()
        return True
    except OSError:
        return False


def wait_for_port(host: str, port: int, tries=PORT_WAIT_TRIES, sleep_s=PORT_WAIT_SLEEP,
                   proc=None, sleep_fn=time.sleep, connect_fn=None, log=lambda *a: None) -> bool:
    connect_fn = connect_fn if connect_fn is not None else _default_connect
    for _ in range(tries):
        if connect_fn(host, port):
            return True
        if proc is not None and proc.poll() is not None:
            log(f"wait_for_port: server died while waiting for port {port}")
            return False
        sleep_fn(sleep_s)
    return False


def stop_process(proc, wait_s=2) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        proc.kill()


# --- 8. delta computation (pure) --------------------------------------------

def compute_delta(eval_result: dict, baseline=None, baseline_per_stratum=None) -> dict:
    """From `eval_fn`'s return dict (see module docstring for the expected
    shape) compute the overall mean-of-repeat-means, its delta vs
    `baseline` (design doc's "b"), and the same per E-set stratum vs
    `baseline_per_stratum`. `baseline`/`baseline_per_stratum` are `None`
    until Task 9's baseline-eval pipeline exists to supply them -- deltas
    are then left `None` rather than silently computed against a wrong/zero
    stand-in, the same "raise/None instead of guessing" stance `eval_fn`
    itself uses (module docstring).
    """
    per_repeat = [float(x) for x in eval_result["per_repeat_means"]]
    overall_mean = float(np.mean(per_repeat))
    delta = (overall_mean - baseline) if baseline is not None else None

    per_stratum_means = {k: float(np.mean(v)) for k, v in eval_result.get("per_stratum_means", {}).items()}
    if baseline_per_stratum is not None:
        delta_per_stratum = {k: v - baseline_per_stratum[k]
                              for k, v in per_stratum_means.items() if k in baseline_per_stratum}
    else:
        delta_per_stratum = None

    return {
        "overall_mean": overall_mean,
        "delta": delta,
        "per_repeat_means": per_repeat,
        "per_stratum_means": per_stratum_means,
        "delta_per_stratum": delta_per_stratum,
    }


# --- 9. ledger row schema -----------------------------------------------------

_ROW_KEYS = (
    "pull_id", "arm", "round_j", "slot", "B", "demo_ids",
    "dataset_path", "symlink_path", "arms_json_path",
    "config_name", "exp_name", "seed", "draw_rng_seed",
    "checkpoint_id", "policy_port", "n_train_attempts",
    "per_repeat_means", "per_repeat_vectors_ref", "overall_mean", "delta",
    "per_stratum_means_json", "delta_per_stratum_json",
    "selector_n_demos", "selector_mean_pairwise_dist", "selector_mean_dist_to_nearest_d0",
    "status", "note", "started_at", "finished_at",
)


def _base_row(resolved: dict, seed: int, started_at: str) -> dict:
    """Every field every pulls.parquet row shares, defaulted to None; both
    the failure path and the success path fill in on top of this so every
    row -- failed, smoke, or ok -- has the identical key set (append-only
    tables tolerate ragged schemas fine, but an identical key set makes the
    ledger far easier to query without a bunch of `.get(...)` guards)."""
    row = {k: None for k in _ROW_KEYS}
    row.update({
        "pull_id": resolved["pull_id"], "arm": resolved["arm"], "round_j": resolved["round_j"],
        "slot": resolved["slot"], "B": resolved["B"], "demo_ids": resolved["demo_ids"],
        "dataset_path": resolved["dataset_path"], "symlink_path": resolved["symlink_path"],
        "arms_json_path": resolved["arms_json_path"],
        "config_name": resolved["config_name"], "exp_name": resolved["exp_name"],
        "seed": seed, "draw_rng_seed": resolved["draw_rng_seed"],
        "policy_port": resolved["policy_port"],
        "selector_n_demos": resolved["selector_scores"]["n_demos"],
        "selector_mean_pairwise_dist": resolved["selector_scores"]["mean_pairwise_dist"],
        "selector_mean_dist_to_nearest_d0": resolved["selector_scores"]["mean_dist_to_nearest_d0"],
        "started_at": started_at,
    })
    return row


def _failed_row(resolved: dict, seed: int, started_at: str, attempt: int, note=None) -> dict:
    row = _base_row(resolved, seed, started_at)
    row.update(status="failed", n_train_attempts=attempt, note=note,
               finished_at=datetime.now(timezone.utc).isoformat())
    return row


def _append_pull_row(row: dict) -> None:
    ledger.append_rows(PULLS_TABLE, [row])


# --- 10. the orchestrator -----------------------------------------------------

def run_pull(arm: str, j: int, slot: str, B: int, eval_fn=None, dry_run=False, *,
             pool_df=None, regions=None, e_features=None, rng=None,
             baseline=None, baseline_per_stratum=None,
             num_train_steps=None, smoke=False, gpu=None, gpus=(0, 1),
             max_train_attempts=MAX_TRAIN_ATTEMPTS,
             popen_fn=subprocess.Popen, dataset_runner=subprocess.run,
             connect_fn=None, sleep_fn=time.sleep, log=print) -> dict:
    """One full pull of arm `arm` at round `j` on slot `slot`. See module
    docstring for the 5-step design and the `eval_fn`/`regions`/`e_features`
    interface seams.

    Returns the row dict that was (or, for `dry_run`, would be) appended to
    `ledger/pulls.parquet`. `dry_run=True` returns after dataset
    materialization + config/command resolution, WITHOUT ever calling
    `wait_for_free_gpu`, `launch_training`, `launch_server`, or `eval_fn`,
    and without appending anything to the ledger -- see module docstring.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    pull_id = pull_id_for(arm, j)
    if slot not in SLOTS:
        raise ValueError(f"run_pull: unknown slot {slot!r}, expected one of {SLOTS}")

    # --- step 1: draw demos --------------------------------------------------
    draw_rng_seed = None
    if arm == "null":
        demo_ids = []
    else:
        if regions is None or e_features is None:
            raise ValueError(
                "run_pull: `regions` and `e_features` are required for any non-null "
                "arm. Task 10's fitted region/cluster model (arms.yaml) and the "
                "E-set's start_features table do not exist yet, so pull.py cannot "
                "supply a default the way it does for `pool_df` -- see this "
                "module's docstring, same interface-seam stance as `eval_fn`.")
        if pool_df is None:
            pool_df = pool.build_pool_table(write=False)
        if rng is None:
            draw_rng_seed = pull_rng_seed(arm, j)
            rng = np.random.default_rng(draw_rng_seed)
        demo_ids = draw.pull_demos(arm, B, rng, pool_df=pool_df, regions=regions, e_features=e_features)

    if pool_df is None:
        # Null pull: no draw happened above, but log_selector_scores still
        # needs a pool table (trivially NaN-shaped for an empty demo_ids).
        pool_df = pool.build_pool_table(write=False)
    selector_scores = draw.log_selector_scores(demo_ids, pool_df)

    # --- step 2: materialize dataset + point the slot symlink at it ---------
    d0_ids = load_d0_episode_ids()
    assemble_episode_ids(d0_ids, demo_ids)  # dedup/overlap assertion only; see docstring
    dataset_path = dataset_dir_for(arm, j)
    symlink_path = slot_symlink_for(slot)
    arms_json_path, which = write_pull_arms_json(pull_id, d0_ids, demo_ids)

    if dataset_path.exists():
        log(f"run_pull: {dataset_path} already exists -- reusing, not rebuilding "
            f"(this is expected on a training retry of the same pull)")
    else:
        build_log = config.LEDGER_DIR / LOGS_SUBDIR / f"{pull_id}_build.log"
        run_dataset_build(arms_json_path, which, dataset_path, runner=dataset_runner, log_path=build_log)

    symlink_slot(dataset_path, symlink_path)

    # --- config/command resolution -------------------------------------------
    config_name = train_config_name_for_slot(slot)
    exp_name = train_exp_name(arm, j)
    seed = config.pull_seed(j)
    tcmd = train_cmd(config_name, exp_name, seed, num_train_steps=num_train_steps)
    ckpt_dir = ckpt_final_dir(config_name, exp_name, num_train_steps=num_train_steps)
    port = slot_port(slot)

    resolved = {
        "pull_id": pull_id, "arm": arm, "round_j": j, "slot": slot, "B": B,
        "demo_ids": demo_ids,
        "dataset_path": str(dataset_path), "symlink_path": str(symlink_path),
        "arms_json_path": str(arms_json_path),
        "config_name": config_name, "exp_name": exp_name,
        "train_cmd": tcmd, "ckpt_dir": str(ckpt_dir), "policy_port": port,
        "draw_rng_seed": draw_rng_seed, "selector_scores": selector_scores,
        "status": "dry_run",
    }
    if dry_run:
        return resolved

    # --- step 3: train, retrying once on failure ------------------------------
    ok = False
    attempts = 0
    for attempt in range(1, max_train_attempts + 1):
        attempts = attempt
        train_gpu = gpu if gpu is not None else wait_for_free_gpu(gpus=gpus, sleep_fn=sleep_fn, log=log)
        train_log = config.LEDGER_DIR / LOGS_SUBDIR / f"{pull_id}_train_attempt{attempt}.log"
        proc = launch_training(tcmd, train_gpu, train_log, popen_fn=popen_fn)
        ok = wait_for_checkpoint(proc, ckpt_dir, sleep_fn=sleep_fn, log=log)
        if ok:
            break
        if attempt < max_train_attempts:
            # An attempt that WILL be retried gets its own ledger row here
            # (audit trail of the failed attempt); the row for the FINAL
            # attempt -- whether this loop naturally exhausts on failure, or
            # succeeds -- is appended exactly once, after the loop, by the
            # code below. Appending here unconditionally on every failure
            # (including the last) would double-log the last attempt: once
            # here, once in the post-loop "if not ok" block.
            _append_pull_row(_failed_row(resolved, seed, started_at, attempt,
                                          note="training process exited without producing a checkpoint -- retrying"))
            log(f"run_pull: {pull_id} training attempt {attempt} failed -- retrying once")

    if not ok:
        row = _failed_row(resolved, seed, started_at, attempts,
                           note="training process exited without producing a checkpoint "
                                "(all attempts exhausted)")
        _append_pull_row(row)
        return row

    # --- step 4: serve the checkpoint + call eval_fn --------------------------
    serve_log = config.LEDGER_DIR / LOGS_SUBDIR / f"{pull_id}_serve.log"
    _, server_proc = launch_server(config_name, ckpt_dir, port, train_gpu, serve_log, popen_fn=popen_fn)
    try:
        up = wait_for_port("127.0.0.1", port, proc=server_proc, sleep_fn=sleep_fn,
                            connect_fn=connect_fn, log=log)
        if not up:
            row = _failed_row(resolved, seed, started_at, attempts, note="policy server never came up")
            row["checkpoint_id"] = str(ckpt_dir)
            _append_pull_row(row)
            return row

        if eval_fn is None:
            raise NotImplementedError("Task 9 eval_set not built yet")
        eval_result = eval_fn(port, pull_id, arm, pull_id)
    finally:
        stop_process(server_proc)

    # --- step 5: deltas + final ledger row -------------------------------------
    delta_info = compute_delta(eval_result, baseline=baseline, baseline_per_stratum=baseline_per_stratum)

    row = _base_row(resolved, seed, started_at)
    row.update(
        checkpoint_id=str(ckpt_dir),
        n_train_attempts=attempts,
        per_repeat_means=delta_info["per_repeat_means"],
        per_repeat_vectors_ref=f"ledger/episodes.parquet[pull_id={pull_id}]",
        overall_mean=delta_info["overall_mean"],
        delta=delta_info["delta"],
        per_stratum_means_json=json.dumps(delta_info["per_stratum_means"]),
        delta_per_stratum_json=(json.dumps(delta_info["delta_per_stratum"])
                                 if delta_info["delta_per_stratum"] is not None else None),
        status="smoke" if smoke else "ok",
        note=None,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    _append_pull_row(row)
    return row
