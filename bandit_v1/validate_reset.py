"""Cross-process determinism gate for saved-state capture/restore (bandit_v1 Task 3,
DAY-ONE GATE). Design: weakregion/BANDIT_V1_DESIGN.md section 1 item 4.

Motivation (measured fact from the design doc): same-seed scene replay across
processes only reproduces the exact scene ~2/3 of the time. This gate checks the
thing v1 actually depends on -- `env.reset_to()` restoration of a SAVED state,
not seed replay -- across two independent axes:

  1. Cross-process fingerprint match: capture n starts in this (parent) process,
     then for each start spawn a FRESH subprocess that builds its own env from
     scratch and restores that start; compare fingerprints computed in the parent
     vs. the child. This is the load-bearing check: it is what "E is restorable"
     actually means for the bandit (starts get restored in worker processes that
     never captured them).
  2. In-process double-restore drift: restore the same start twice in THIS
     process, roll 20 zero-actions after each restore, and diff the target
     object's xpos trajectory between the two restores. Confirms reset_to is
     itself deterministic (not just "same as capture") and that the scripted
     success check does not spontaneously read True from a static start.

Usage:
  conda run -n robocasa python -m bandit_v1.validate_reset \\
      --n 10 --seed_base 900000 --out /data/xinyua11/tmp/reset_gate

Internal (re-invoked as a fresh subprocess per start; not meant to be called
directly):
  python -m bandit_v1.validate_reset --restore_only \\
      --start_dir <start_dir> --out_file <start_dir>/fingerprint_restored.json

--warm-check mode (rollout-speedup task): a SECOND, separate gate for
states.restore()'s two speedups (skip the throwaway pre-reset once an env has
been reset at least once; skip both of reset_to's internal model recompiles
entirely when the incoming start's model.xml hash matches the last-restored
one). Captures 3 starts (A, B, C) then, IN THE SAME PROCESS, restores them into
two independent envs -- one always called with warm=True, one always with
warm=False -- following the fixed interleaved sequence A,A,A,B,A,C,C (repeat
-hits for A and C exercise the fast path; the A->B, B->A, A->C transitions
exercise hash-invalidation back onto the full path). At every step, compares
the two envs' fingerprints and full flattened mujoco state vectors
(np.allclose, atol=0 rtol=0 -- exact). Usage:
  conda run -n robocasa python -m bandit_v1.validate_reset --warm-check \\
      --seed_base 920000 --out /data/xinyua11/tmp/reset_gate_warm
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import states

DRIFT_TOL_M = 1e-3
N_ZERO_STEPS = 20


def _restore_only(start_dir, out_file):
    """Child-process entry point: build a brand-new env, restore the given start,
    fingerprint it, write the result. No state is shared with the parent process
    other than the files under start_dir."""
    env = states.make_env()
    try:
        states.restore(env, start_dir)
        fp = states.fingerprint(env)
        Path(out_file).write_text(json.dumps(fp, indent=2))
    finally:
        states.close_env(env)


def _obj_xpos(rs):
    return np.asarray(rs.sim.data.body_xpos[rs.obj_body_id[states.TARGET]], dtype=float).copy()


def _restore_and_roll_zero_actions(env, start_dir, n_steps):
    """Restore `start_dir` into `env`, step `n_steps` zero actions, return
    (final target-object xpos, whether scripted success ever read True)."""
    states.restore(env, start_dir)
    rs = env.env
    action_dim = env.action_dimension
    zero_action = np.zeros(action_dim)
    ever_success = bool(env.is_success().get("task", False))
    for _ in range(n_steps):
        env.step(zero_action)
        ever_success = ever_success or bool(env.is_success().get("task", False))
    return _obj_xpos(rs), ever_success


def _in_process_drift_check(start_dir):
    """Two restores of the SAME start, in the SAME process, each rolled forward
    N_ZERO_STEPS zero-actions. Returns (drift_m, success_flags[2])."""
    env = states.make_env()
    try:
        env.reset()  # env must exist/have been reset once before the first restore
        xpos_0, succ_0 = _restore_and_roll_zero_actions(env, start_dir, N_ZERO_STEPS)
        xpos_1, succ_1 = _restore_and_roll_zero_actions(env, start_dir, N_ZERO_STEPS)
        drift = float(np.linalg.norm(xpos_0 - xpos_1))
        return drift, [succ_0, succ_1]
    finally:
        states.close_env(env)


def _cross_process_check(start_dir, fp_captured, py_exe):
    """Spawn a fresh subprocess that restores `start_dir` and fingerprints it;
    compare against fp_captured (computed in the parent at capture time). Returns
    (ok, detail) where detail is None on success, else a dict describing the
    failure mode (subprocess error, or list of mismatched fingerprint fields)."""
    out_file = start_dir / "fingerprint_restored.json"
    result = subprocess.run(
        [py_exe, "-m", "bandit_v1.validate_reset",
         "--restore_only", "--start_dir", str(start_dir), "--out_file", str(out_file)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode != 0:
        return False, {"kind": "subprocess_error", "returncode": result.returncode,
                        "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}
    fp_restored = json.loads(out_file.read_text())
    diff = states.fingerprint_diff(fp_captured, fp_restored)
    if diff:
        return False, {"kind": "fingerprint_mismatch", "fields": diff,
                        "captured": fp_captured, "restored": fp_restored}
    return True, None


def run_gate(n, seed_base, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py_exe = sys.executable

    print(f"[capture] {n} starts at seed_base={seed_base} -> {out_dir}")
    starts = []  # (start_dir, fp_captured)
    for j in range(n):
        seed = seed_base + j
        start_dir = out_dir / f"start_{j:03d}"
        fp = states.capture_start(seed, start_dir)
        starts.append((start_dir, fp))
        print(f"  start {j}: seed={seed} category={fp['category']} "
              f"layout={fp['layout_id']} style={fp['style_id']} obj_xyz={fp['obj_xyz']}")

    print("\n[cross-process check] restoring each start in a FRESH subprocess...")
    cross_failures = []
    for j, (start_dir, fp) in enumerate(starts):
        ok, detail = _cross_process_check(start_dir, fp, py_exe)
        if ok:
            print(f"  [OK]   start {j}: fingerprints match across processes")
        else:
            cross_failures.append((j, start_dir, detail))
            print(f"  [FAIL] start {j}: {detail.get('kind')} -> {detail}")
    n_cross_ok = n - len(cross_failures)
    print(f"\n{n_cross_ok}/{n} fingerprints match across processes")

    print("\n[in-process double-restore drift check] "
          f"({N_ZERO_STEPS} zero-actions after each of 2 restores per start)...")
    max_drift = 0.0
    drift_failures = []
    for j, (start_dir, _fp) in enumerate(starts):
        drift, succ_flags = _in_process_drift_check(start_dir)
        max_drift = max(max_drift, drift)
        unexpected_success = any(succ_flags)
        status = "OK" if (drift < DRIFT_TOL_M and not unexpected_success) else "FAIL"
        print(f"  [{status}] start {j}: drift={drift:.3e} m, "
              f"scripted_success_after_{N_ZERO_STEPS}_zero_actions={succ_flags}")
        if status == "FAIL":
            drift_failures.append((j, start_dir, drift, succ_flags))

    print(f"\nmax in-process restore drift: {max_drift:.3e} m (tol {DRIFT_TOL_M:.0e} m)")

    gate_pass = (not cross_failures) and (not drift_failures)
    if gate_pass:
        print(f"\n{n_cross_ok}/{n} fingerprints match across processes; "
              f"max in-process restore drift {max_drift:.3e} m < {DRIFT_TOL_M:.0e} m; PASS")
    else:
        print(f"\nGATE FAILED: {len(cross_failures)} cross-process mismatches, "
              f"{len(drift_failures)} in-process drift/success failures")
        for j, start_dir, detail in cross_failures:
            print(f"  cross-process failure at start {j} ({start_dir}): {detail}")
        for j, start_dir, drift, succ_flags in drift_failures:
            print(f"  in-process failure at start {j} ({start_dir}): "
                  f"drift={drift:.3e} m, success_flags={succ_flags}")
    return gate_pass


WARM_CHECK_SEQUENCE = ["A", "A", "A", "B", "A", "C", "C"]
WARM_CHECK_LABELS = ["A", "B", "C"]


def _state_vec(env):
    """Full flattened mujoco state vector for `env` (time, qpos, qvel), the same
    representation env.get_state()["states"] / capture_start's state.npz use."""
    return np.asarray(env.env.sim.get_state().flatten(), dtype=float)


def run_warm_check(seed_base, out_dir):
    """--warm-check gate (rollout-speedup task): captures 3 starts (A, B, C),
    then restores them IN THE SAME PROCESS into two independent envs -- env_warm
    always via states.restore(..., warm=True) (both speedups active: skip the
    pre-reset once initialized, warm-model fast path on a repeated model.xml
    hash), env_cold always via states.restore(..., warm=False) (the original,
    unmodified full path) -- following the fixed interleaved sequence
    A,A,A,B,A,C,C. This sequence deliberately exercises: back-to-back repeats of
    the SAME start (A x3, then C x2 -- the fast path engages on the 2nd+ hit of
    each run), and hash-invalidating transitions onto a genuinely different start
    (A->B, B->A, A->C -- must fall back to the full path). At every step,
    compares fingerprints (states.fingerprint_diff, expect []) and the full
    flattened mujoco state vector (np.allclose, atol=0 rtol=0 -- exact equality)
    between the two envs. Returns True iff every step matches on both axes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dirs = {}
    print(f"[warm-check] capturing {len(WARM_CHECK_LABELS)} starts at "
          f"seed_base={seed_base} -> {out_dir}")
    for i, label in enumerate(WARM_CHECK_LABELS):
        seed = seed_base + i
        start_dir = out_dir / f"start_{label}"
        fp = states.capture_start(seed, start_dir)
        start_dirs[label] = start_dir
        print(f"  start {label}: seed={seed} category={fp['category']} "
              f"layout={fp['layout_id']} style={fp['style_id']} obj_xyz={fp['obj_xyz']}")

    env_warm = states.make_env()
    env_cold = states.make_env()
    all_ok = True
    try:
        print(f"\n[warm-check] sequence: {','.join(WARM_CHECK_SEQUENCE)}  "
              "(env_warm via warm=True, env_cold via warm=False, same process)")
        for step, label in enumerate(WARM_CHECK_SEQUENCE):
            start_dir = start_dirs[label]
            states.restore(env_warm, start_dir, warm=True)
            states.restore(env_cold, start_dir, warm=False)

            fp_warm = states.fingerprint(env_warm)
            fp_cold = states.fingerprint(env_cold)
            fp_diff = states.fingerprint_diff(fp_warm, fp_cold)

            vec_warm = _state_vec(env_warm)
            vec_cold = _state_vec(env_cold)
            vec_match = (vec_warm.shape == vec_cold.shape) and np.allclose(
                vec_warm, vec_cold, atol=0, rtol=0)

            ok = (not fp_diff) and vec_match
            all_ok = all_ok and ok
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] step {step} (start {label}): fingerprint_diff={fp_diff}, "
                  f"state_vec_exact_match={vec_match}")
            if not ok:
                print(f"    fp_warm={fp_warm}")
                print(f"    fp_cold={fp_cold}")
                if vec_warm.shape == vec_cold.shape:
                    max_abs_diff = float(np.max(np.abs(vec_warm - vec_cold)))
                    print(f"    max abs state-vec diff: {max_abs_diff:.3e}")
                else:
                    print(f"    state vec shape mismatch: warm={vec_warm.shape} cold={vec_cold.shape}")
    finally:
        states.close_env(env_warm)
        states.close_env(env_cold)

    n = len(WARM_CHECK_SEQUENCE)
    if all_ok:
        print(f"\n{n}/{n} steps: warm/cold fingerprints + state vectors match exactly; PASS")
    else:
        print("\nWARM-CHECK GATE FAILED: see [FAIL] steps above")
    return all_ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed_base", type=int, default=900000)
    p.add_argument("--out", default="/tmp/reset_gate")
    p.add_argument("--warm-check", action="store_true",
                    help="Run the warm/cold restore-path equivalence gate "
                         "(states.restore()'s speedups) instead of the cross-process gate")
    # internal child-process mode, not part of the public CLI contract
    p.add_argument("--restore_only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--start_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out_file", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.restore_only:
        assert args.start_dir and args.out_file, "--restore_only requires --start_dir and --out_file"
        _restore_only(args.start_dir, args.out_file)
        return

    if args.warm_check:
        gate_pass = run_warm_check(args.seed_base, args.out)
        sys.exit(0 if gate_pass else 1)

    gate_pass = run_gate(args.n, args.seed_base, args.out)
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
