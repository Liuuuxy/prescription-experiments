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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed_base", type=int, default=900000)
    p.add_argument("--out", default="/tmp/reset_gate")
    # internal child-process mode, not part of the public CLI contract
    p.add_argument("--restore_only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--start_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out_file", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.restore_only:
        assert args.start_dir and args.out_file, "--restore_only requires --start_dir and --out_file"
        _restore_only(args.start_dir, args.out_file)
        return

    gate_pass = run_gate(args.n, args.seed_base, args.out)
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
