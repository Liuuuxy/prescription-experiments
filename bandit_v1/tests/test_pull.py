"""Tests for bandit_v1/pull.py (Task 12's pull orchestrator).

Pure-logic + monkeypatched-integration tests only -- no real GPU, policy
server, or subprocess. Every subprocess/network seam (`popen_fn`,
`dataset_runner`, `connect_fn`, `sleep_fn`) is faked; `gpu=` is always passed
explicitly so `wait_for_free_gpu` (which shells out to nvidia-smi) is never
exercised here. The end-to-end 60-step smoke this task's brief also asks for
is explicitly DEFERRED (see .superpowers/sdd/task-12-report.md) -- these
tests cover run_pull's control flow and row schema, not a real training run.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bandit_v1 import config, draw, ledger, pool, pull


# --- shared fixtures/fakes ----------------------------------------------------

class FakeProc:
    """Minimal stand-in for subprocess.Popen: `poll()` always returns
    `always` (None == "still running", an int == "exited with that code");
    `terminate`/`wait` are no-ops that just record they were called."""
    def __init__(self, always=None):
        self._always = always
        self.terminated = False
        self.waited = False

    def poll(self):
        return self._always

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True


def make_fake_popen(poll_value=None):
    def _popen(cmd, **kwargs):
        return FakeProc(always=poll_value)
    return _popen


def fake_dataset_runner(cmd, **kwargs):
    """Stands in for subprocess.run against build_lerobot_subset.py: creates
    the --dst directory (so downstream `.exists()` checks behave like a real
    run) and reports success, without ever touching config.POOL_LEROBOT."""
    dst = Path(cmd[cmd.index("--dst") + 1])
    dst.mkdir(parents=True, exist_ok=True)

    class _Result:
        returncode = 0
    return _Result()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Points every path bandit_v1/pull.py touches at tmp_path: config.
    FT_ARMS_ROOT (dataset dirs + slot symlinks), config.OPENPI (checkpoint
    dirs), config.LEDGER_DIR (pull_arms/pull_logs provenance -- note this is
    a SEPARATE binding from ledger.LEDGER_DIR, see ledger.py: it copies
    config.LEDGER_DIR into its own module attribute at import time, so both
    must be monkeypatched to stay in sync), and config.ARMS_JSON (a small
    synthetic D0 = [1, 2, 3], instead of the real 400-episode D0)."""
    ft_arms_root = tmp_path / "ft_arms"
    openpi_root = tmp_path / "openpi"
    ledger_dir = tmp_path / "ledger"
    arms_json = tmp_path / "arms.json"
    arms_json.write_text(json.dumps({"base_episodes": [1, 2, 3]}))

    monkeypatch.setattr(config, "FT_ARMS_ROOT", ft_arms_root)
    monkeypatch.setattr(config, "OPENPI", openpi_root)
    monkeypatch.setattr(config, "LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(ledger, "LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(config, "ARMS_JSON", arms_json)
    return tmp_path


POOL_COLS = ["episode_index", "category", "h", "w", "layout",
             "x_rel", "y_rel", "side", "traj_len", "in_d0"]


def make_pool_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=POOL_COLS)
    defaults = {"h": 0.1, "w": 0.1, "layout": 0, "side": 1, "traj_len": 50, "in_d0": False}
    out = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        out.append(row)
    return pd.DataFrame(out)[POOL_COLS]


EMPTY_E = pd.DataFrame(columns=["category", "x_rel", "y_rel"])


# --- (a) dataset episode-list assembly ---------------------------------------

def test_assemble_episode_ids_d0_plus_demos_order_preserving():
    assert pull.assemble_episode_ids([1, 2, 3], [10, 20]) == [1, 2, 3, 10, 20]


def test_assemble_episode_ids_null_pull_is_d0_alone():
    assert pull.assemble_episode_ids([1, 2, 3], []) == [1, 2, 3]


def test_assemble_episode_ids_raises_on_d0_overlap():
    with pytest.raises(ValueError, match="overlap"):
        pull.assemble_episode_ids([1, 2, 3], [3, 4])


def test_assemble_episode_ids_raises_on_duplicate_demo_ids():
    with pytest.raises(ValueError, match="duplicate"):
        pull.assemble_episode_ids([1, 2, 3], [10, 10])


def test_write_pull_arms_json_which_is_base_plus_pull_when_demos_present(tmp_path):
    path, which = pull.write_pull_arms_json("random_j1", [1, 2, 3], [10, 20], path=tmp_path / "a.json")
    assert which == "base+pull"
    assert json.loads(path.read_text()) == {"base_episodes": [1, 2, 3], "pull_episodes": [10, 20]}


def test_write_pull_arms_json_which_is_base_alone_for_null_pull(tmp_path):
    _, which = pull.write_pull_arms_json("null_j1", [1, 2, 3], [], path=tmp_path / "a.json")
    assert which == "base"


# --- (b) slot symlink / dataset path computation -----------------------------

def test_dataset_dir_for_and_slot_symlink_for(isolated):
    assert pull.dataset_dir_for("random", 3) == config.FT_ARMS_ROOT / "ppc2sink_bandit_random_j3"
    assert pull.slot_symlink_for("a") == config.FT_ARMS_ROOT / "ppc2sink_bandit_slot_a"
    assert pull.slot_symlink_for("b") == config.FT_ARMS_ROOT / "ppc2sink_bandit_slot_b"


def test_slot_symlink_for_rejects_unknown_slot(isolated):
    with pytest.raises(ValueError):
        pull.slot_symlink_for("c")


def test_slot_port_mapping():
    assert pull.slot_port("a") == 8130
    assert pull.slot_port("b") == 8131
    with pytest.raises(ValueError):
        pull.slot_port("c")


def test_symlink_slot_creates_then_replaces_ln_sfn_semantics(tmp_path):
    target1 = tmp_path / "ds1"; target1.mkdir()
    target2 = tmp_path / "ds2"; target2.mkdir()
    link = tmp_path / "slot_a"

    pull.symlink_slot(target1, link)
    assert link.is_symlink()
    assert link.resolve() == target1.resolve()

    pull.symlink_slot(target2, link)  # must replace, not error or nest
    assert link.is_symlink()
    assert link.resolve() == target2.resolve()


# --- (c) training command construction ---------------------------------------

def test_train_config_name_for_slot():
    assert pull.train_config_name_for_slot("a") == "pi0_ppc2sink_bandit_a"
    assert pull.train_config_name_for_slot("b") == "pi0_ppc2sink_bandit_b"
    with pytest.raises(ValueError):
        pull.train_config_name_for_slot("c")


def test_train_exp_name():
    assert pull.train_exp_name("random", 3) == "random_j3"
    assert pull.train_exp_name("null", 1) == "null_j1"


def test_train_cmd_uses_verified_dash_flags_and_pull_seed_formula():
    j = 3
    seed = config.pull_seed(j)
    assert seed == 1003
    cmd = pull.train_cmd("pi0_ppc2sink_bandit_a", "random_j3", seed)

    assert cmd[0] == str(pull.OPENPI_PY)
    assert cmd[1] == "scripts/train.py"
    assert cmd[2] == "pi0_ppc2sink_bandit_a"
    assert cmd[cmd.index("--exp-name") + 1] == "random_j3"
    assert cmd[cmd.index("--seed") + 1] == "1003"
    assert "--overwrite" in cmd
    # the brief's literal (unverified) underscore spelling must NOT appear --
    # tyro's real --help output (Task 5) confirmed dashes only.
    assert "--exp_name" not in cmd
    assert "--num_train_steps" not in cmd


def test_train_cmd_num_train_steps_override_uses_dash_flag():
    cmd = pull.train_cmd("pi0_ppc2sink_bandit_a", "x_j0", 1000, num_train_steps=60)
    assert cmd[cmd.index("--num-train-steps") + 1] == "60"


def test_ckpt_final_dir_default_step_vs_num_train_steps_override(isolated):
    d = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "random_j3")
    assert d == config.OPENPI / "checkpoints" / "pi0_ppc2sink_bandit_a" / "random_j3" / str(config.FINAL_CKPT_STEP)

    d2 = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "random_j3", num_train_steps=60)
    assert d2.name == "59"  # step == num_train_steps - 1, per scripts/train.py's save condition


# --- (d) crc32 draw-seed determinism (replaces the brief's hash()-based formula) --

def test_pull_rng_seed_deterministic_within_process():
    assert pull.pull_rng_seed("random", 3) == pull.pull_rng_seed("random", 3)


def test_pull_rng_seed_differs_across_arm_or_round():
    assert pull.pull_rng_seed("random", 3) != pull.pull_rng_seed("random", 4)
    assert pull.pull_rng_seed("random", 3) != pull.pull_rng_seed("hard", 3)


def test_pull_rng_seed_stable_across_a_fresh_process():
    """The whole reason NOT to use hash((arm, j)): CPython salts str hashing
    per process (PYTHONHASHSEED), so hash()-based seeds do NOT reproduce
    across process boundaries. zlib.crc32 does. Spawn a genuinely separate
    `python` process (distinct from this pytest process, and therefore with
    a different random hash salt if hash() were used) and confirm it
    recomputes the identical seed."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c", "import zlib; print(zlib.crc32(b'random:3') % (2**32))"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert int(out) == pull.pull_rng_seed("random", 3)


# --- (e) compute_delta (pure) -------------------------------------------------

def test_compute_delta_with_baseline():
    eval_result = {
        "per_repeat_means": [0.5, 0.6, 0.55],
        "per_stratum_means": {"easy": [0.8, 0.8, 0.8], "hard": [0.2, 0.3, 0.25]},
    }
    out = pull.compute_delta(eval_result, baseline=0.5, baseline_per_stratum={"easy": 0.7, "hard": 0.2})
    assert out["overall_mean"] == pytest.approx(0.55)
    assert out["delta"] == pytest.approx(0.05)
    assert out["per_stratum_means"]["easy"] == pytest.approx(0.8)
    assert out["per_stratum_means"]["hard"] == pytest.approx(0.25)
    assert out["delta_per_stratum"]["easy"] == pytest.approx(0.1)
    assert out["delta_per_stratum"]["hard"] == pytest.approx(0.05)


def test_compute_delta_without_baseline_leaves_delta_none():
    eval_result = {"per_repeat_means": [0.4, 0.5], "per_stratum_means": {}}
    out = pull.compute_delta(eval_result)
    assert out["delta"] is None
    assert out["delta_per_stratum"] is None


# --- (f) run_pull: dry_run -----------------------------------------------------

def test_run_pull_dry_run_null_arm_is_d0_alone_and_launches_nothing(isolated):
    popen_calls = []

    def exploding_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        raise AssertionError("dry_run must never launch a process")

    resolved = pull.run_pull(
        "null", 1, "a", B=5, dry_run=True,
        pool_df=make_pool_df([]),
        dataset_runner=fake_dataset_runner,
        popen_fn=exploding_popen,
    )

    assert resolved["demo_ids"] == []
    assert resolved["dataset_path"] == str(config.FT_ARMS_ROOT / "ppc2sink_bandit_null_j1")
    assert Path(resolved["dataset_path"]).exists()  # fake_dataset_runner really created it
    assert Path(resolved["symlink_path"]).resolve() == Path(resolved["dataset_path"]).resolve()
    assert resolved["config_name"] == "pi0_ppc2sink_bandit_a"
    assert resolved["exp_name"] == "null_j1"
    assert "--exp-name" in resolved["train_cmd"]
    assert not popen_calls

    with pytest.raises(FileNotFoundError):
        ledger.read("pulls")  # dry_run must not write to the ledger


def test_run_pull_dry_run_non_null_arm_draws_demos_matching_direct_pull_demos_call(isolated):
    # episode_index starts at 100 -- the `isolated` fixture's D0 is [1, 2, 3],
    # so this avoids any accidental D0 overlap in the synthetic well.
    rows = [{"episode_index": 100 + i, "category": "jar", "x_rel": float(i), "y_rel": 0.0} for i in range(20)]
    pool_df = make_pool_df(rows)
    regions = pd.Series({100 + i: "hard" for i in range(20)})

    resolved = pull.run_pull(
        "hard", 7, "b", B=4, dry_run=True,
        pool_df=pool_df, regions=regions, e_features=EMPTY_E,
        dataset_runner=fake_dataset_runner,
    )

    expected_seed = pull.pull_rng_seed("hard", 7)
    expected_ids = draw.pull_demos("hard", 4, np.random.default_rng(expected_seed), pool_df, regions, EMPTY_E)
    assert resolved["demo_ids"] == expected_ids
    assert resolved["draw_rng_seed"] == expected_seed
    assert resolved["config_name"] == "pi0_ppc2sink_bandit_b"
    assert resolved["selector_scores"]["n_demos"] == 4


def test_run_pull_non_null_arm_requires_regions_and_e_features(isolated):
    with pytest.raises(ValueError, match="regions"):
        pull.run_pull("hard", 1, "a", B=2, dry_run=True, pool_df=make_pool_df([]))


def test_run_pull_dry_run_reuses_existing_dataset_dir_without_rebuilding(isolated):
    dataset_path = pull.dataset_dir_for("null", 2)
    dataset_path.mkdir(parents=True)  # simulate a prior (e.g. retried) pull already materialized it

    calls = []

    def tracking_runner(cmd, **kwargs):
        calls.append(cmd)
        return fake_dataset_runner(cmd, **kwargs)

    pull.run_pull("null", 2, "a", B=5, dry_run=True, pool_df=make_pool_df([]), dataset_runner=tracking_runner)
    assert calls == []  # never invoked -- dataset dir already existed


# --- (g) full run_pull: success path + row completeness ----------------------

def _eval_fn_ok(port, policy_id, arm, pull_id, calls=None):
    if calls is not None:
        calls.append((port, policy_id, arm, pull_id))
    return {
        "per_repeat_means": [0.5, 0.6, 0.55],
        "per_stratum_means": {"easy": [0.8, 0.8, 0.8], "hard": [0.2, 0.3, 0.25]},
    }


def test_run_pull_full_success_row_is_complete_and_deltas_correct(isolated):
    rows = [{"episode_index": i, "category": "jar", "x_rel": float(i), "y_rel": 0.0} for i in range(20)]
    pool_df = make_pool_df(rows)
    regions = pd.Series({i: "hard" for i in range(20)})

    sleep_calls = {"n": 0}
    ckpt_dir_box = {}

    def fake_sleep(secs):
        sleep_calls["n"] += 1
        # Simulate the checkpoint appearing after the first poll -- avoids
        # any real waiting while still exercising the real (non-injected)
        # exists_fn/poll_fn wiring inside wait_for_checkpoint.
        ckpt_dir_box["dir"].mkdir(parents=True, exist_ok=True)

    eval_calls = []

    def eval_fn(port, policy_id, arm, pull_id):
        return _eval_fn_ok(port, policy_id, arm, pull_id, calls=eval_calls)

    # Resolve the would-be checkpoint dir up front (pure computation) so
    # fake_sleep knows what to create.
    ckpt_dir_box["dir"] = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "hard_j2")

    row = pull.run_pull(
        "hard", 2, "a", B=4,
        pool_df=pool_df, regions=regions, e_features=EMPTY_E,
        eval_fn=eval_fn, baseline=0.5, baseline_per_stratum={"easy": 0.7, "hard": 0.2},
        gpu=0, dataset_runner=fake_dataset_runner,
        popen_fn=make_fake_popen(None), connect_fn=lambda host, port: True,
        sleep_fn=fake_sleep,
    )

    assert set(row.keys()) == set(pull._ROW_KEYS)
    assert row["status"] == "ok"
    assert row["pull_id"] == "hard_j2"
    assert row["round_j"] == 2
    assert row["slot"] == "a"
    assert row["B"] == 4
    assert len(row["demo_ids"]) == 4
    assert row["seed"] == config.pull_seed(2) == 1002
    assert row["config_name"] == "pi0_ppc2sink_bandit_a"
    assert row["exp_name"] == "hard_j2"
    assert row["checkpoint_id"] == str(ckpt_dir_box["dir"])
    assert row["n_train_attempts"] == 1
    assert row["overall_mean"] == pytest.approx(0.55)
    assert row["delta"] == pytest.approx(0.05)
    per_stratum = json.loads(row["per_stratum_means_json"])
    assert per_stratum["easy"] == pytest.approx(0.8)
    delta_per_stratum = json.loads(row["delta_per_stratum_json"])
    assert delta_per_stratum["hard"] == pytest.approx(0.05)
    assert row["note"] is None
    assert row["finished_at"] is not None

    assert eval_calls == [(pull.slot_port("a"), "hard_j2", "hard", "hard_j2")]

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 1
    assert ledger_rows.iloc[0]["pull_id"] == "hard_j2"


def test_run_pull_smoke_flag_sets_status_smoke_instead_of_ok(isolated):
    row = pull.run_pull(
        "null", 9, "b", B=0,
        pool_df=make_pool_df([]),
        eval_fn=_eval_fn_ok, gpu=1, smoke=True,
        dataset_runner=fake_dataset_runner, popen_fn=make_fake_popen(None),
        connect_fn=lambda host, port: True,
        sleep_fn=lambda secs: pull.ckpt_final_dir("pi0_ppc2sink_bandit_b", "null_j9").mkdir(
            parents=True, exist_ok=True),
    )
    assert row["status"] == "smoke"


def test_run_pull_raises_not_implemented_when_eval_fn_missing(isolated):
    with pytest.raises(NotImplementedError, match="Task 9"):
        pull.run_pull(
            "null", 4, "a", B=0, pool_df=make_pool_df([]),
            gpu=0, dataset_runner=fake_dataset_runner, popen_fn=make_fake_popen(None),
            connect_fn=lambda host, port: True,
            sleep_fn=lambda secs: pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "null_j4").mkdir(
                parents=True, exist_ok=True),
        )


def test_run_pull_server_never_comes_up_marks_failed_without_calling_eval_fn(isolated):
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "null_j8")
    eval_calls = []

    def eval_fn_should_not_be_called(*a, **kw):
        eval_calls.append((a, kw))
        return _eval_fn_ok(*a, **kw)

    def fake_sleep_creates_ckpt_once(secs):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    row = pull.run_pull(
        "null", 8, "a", B=0, pool_df=make_pool_df([]),
        eval_fn=eval_fn_should_not_be_called, gpu=0,
        dataset_runner=fake_dataset_runner, popen_fn=make_fake_popen(None),
        connect_fn=lambda host, port: False,   # server socket never accepts
        sleep_fn=fake_sleep_creates_ckpt_once,
    )

    assert row["status"] == "failed"
    assert row["note"] == "policy server never came up"
    assert row["checkpoint_id"] == str(ckpt_dir)  # training DID succeed -- only serving failed
    assert not eval_calls

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 1
    assert ledger_rows.iloc[0]["status"] == "failed"


# --- (h) failed training + re-run-once logic ----------------------------------

def test_run_pull_both_training_attempts_fail_logs_two_failed_rows_and_never_evals(isolated):
    eval_calls = []

    def eval_fn_should_not_be_called(*a, **kw):
        eval_calls.append((a, kw))
        return _eval_fn_ok(*a, **kw)

    popen_calls = []

    def failing_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return FakeProc(always=1)  # "exited immediately with rc=1" -- checkpoint never appears

    row = pull.run_pull(
        "null", 5, "a", B=0, pool_df=make_pool_df([]),
        eval_fn=eval_fn_should_not_be_called, gpu=0,
        dataset_runner=fake_dataset_runner, popen_fn=failing_popen,
        sleep_fn=lambda secs: None,
    )

    assert row["status"] == "failed"
    assert row["n_train_attempts"] == 2
    assert not eval_calls

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 2
    assert list(ledger_rows["status"]) == ["failed", "failed"]
    assert list(ledger_rows["n_train_attempts"]) == [1, 2]
    assert len(popen_calls) == 2  # exactly one launch per attempt -- proves the retry happened


def test_run_pull_first_attempt_fails_second_succeeds_then_completes_eval(isolated):
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "null_j6")
    train_popen_calls = []

    def flaky_popen(cmd, **kwargs):
        if "scripts/train.py" in cmd:
            train_popen_calls.append(cmd)
            if len(train_popen_calls) == 1:
                return FakeProc(always=1)  # attempt 1: dies immediately
            return FakeProc(always=None)  # attempt 2: stays alive
        return FakeProc(always=None)  # the (single) serve_policy.py launch

    def fake_sleep(secs):
        # Only reached during attempt 2's wait_for_checkpoint loop (attempt
        # 1 fails on its very first poll, before ever sleeping).
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_calls = []

    row = pull.run_pull(
        "null", 6, "a", B=0, pool_df=make_pool_df([]),
        eval_fn=lambda *a, **kw: _eval_fn_ok(*a, calls=eval_calls, **kw),
        gpu=0, dataset_runner=fake_dataset_runner, popen_fn=flaky_popen,
        connect_fn=lambda host, port: True, sleep_fn=fake_sleep,
    )

    assert row["status"] == "ok"
    assert row["n_train_attempts"] == 2
    assert len(eval_calls) == 1

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 2  # 1 failed-attempt row + 1 final ok row
    assert list(ledger_rows["status"]) == ["failed", "ok"]
    assert len(train_popen_calls) == 2  # exactly one train.py launch per attempt


# --- (i) gradient-analysis training_artifacts (owner request) -----------------

def test_ckpt_steps_present_empty_when_dir_missing(isolated):
    assert pull.ckpt_steps_present(config.OPENPI / "checkpoints" / "cfg" / "exp") == []


def test_ckpt_steps_present_lists_numeric_subdirs_sorted(isolated):
    root = pull.ckpt_root_dir("pi0_ppc2sink_bandit_a", "hard_j3")
    for step in (19999, 5000, 10000, 15000):
        (root / str(step)).mkdir(parents=True)
    (root / "not_a_step").mkdir()  # must never be mistaken for a checkpoint step
    assert pull.ckpt_steps_present(root) == [5000, 10000, 15000, 19999]


def test_wandb_run_id_none_when_file_missing(isolated):
    root = pull.ckpt_root_dir("pi0_ppc2sink_bandit_a", "hard_j3")
    root.mkdir(parents=True)
    assert pull.wandb_run_id(root) is None


def test_wandb_run_id_reads_file_stripped(isolated):
    root = pull.ckpt_root_dir("pi0_ppc2sink_bandit_a", "hard_j3")
    root.mkdir(parents=True)
    (root / "wandb_id.txt").write_text("abc123\n")
    assert pull.wandb_run_id(root) == "abc123"


def test_find_wandb_dir_none_when_run_id_none(tmp_path):
    assert pull.find_wandb_dir(None, wandb_base=tmp_path) is None


def test_find_wandb_dir_none_when_no_match(tmp_path):
    (tmp_path / "wandb").mkdir()
    assert pull.find_wandb_dir("zzz", wandb_base=tmp_path) is None


def test_find_wandb_dir_matches_offline_run_glob_by_id_suffix(tmp_path):
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    (wandb_dir / "offline-run-20260701_095541-rw79hr65").mkdir()
    (wandb_dir / "offline-run-20260702_100000-other123").mkdir()
    found = pull.find_wandb_dir("rw79hr65", wandb_base=tmp_path)
    assert found == str(wandb_dir / "offline-run-20260701_095541-rw79hr65")


def test_collect_training_artifacts_full_shape(isolated, tmp_path):
    root = pull.ckpt_root_dir("pi0_ppc2sink_bandit_a", "hard_j3")
    root.mkdir(parents=True)
    (root / "5000").mkdir()
    (root / "19999").mkdir()
    (root / "wandb_id.txt").write_text("myrunid")
    wandb_base = tmp_path / "wandbroot"
    (wandb_base / "wandb").mkdir(parents=True)
    (wandb_base / "wandb" / "offline-run-x-myrunid").mkdir()

    artifacts = pull.collect_training_artifacts(
        "pi0_ppc2sink_bandit_a", "hard_j3", tmp_path / "train.log", seed=1003,
        wandb_base=wandb_base)

    assert artifacts == {
        "ckpt_root": str(root),
        "ckpt_steps_present": [5000, 19999],
        "train_log_path": str(tmp_path / "train.log"),
        "wandb_dir": str(wandb_base / "wandb" / "offline-run-x-myrunid"),
        "recipe_seed": 1003,
    }


def test_collect_training_artifacts_train_log_path_none_when_never_attempted(isolated):
    artifacts = pull.collect_training_artifacts("pi0_ppc2sink_bandit_a", "never_j1", None, seed=5)
    assert artifacts["train_log_path"] is None
    assert artifacts["ckpt_steps_present"] == []


def test_run_pull_success_row_records_training_artifacts_json(isolated):
    rows = [{"episode_index": i, "category": "jar", "x_rel": float(i), "y_rel": 0.0} for i in range(20)]
    pool_df = make_pool_df(rows)
    regions = pd.Series({i: "hard" for i in range(20)})
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "hard_j2")

    def fake_sleep(secs):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir.parent / "wandb_id.txt").write_text("wid123")

    row = pull.run_pull(
        "hard", 2, "a", B=4,
        pool_df=pool_df, regions=regions, e_features=EMPTY_E,
        eval_fn=_eval_fn_ok, baseline=0.5, baseline_per_stratum={"easy": 0.7, "hard": 0.2},
        gpu=0, dataset_runner=fake_dataset_runner,
        popen_fn=make_fake_popen(None), connect_fn=lambda host, port: True,
        sleep_fn=fake_sleep,
    )

    assert row["status"] == "ok"
    artifacts = json.loads(row["training_artifacts_json"])
    assert artifacts["ckpt_root"] == str(pull.ckpt_root_dir("pi0_ppc2sink_bandit_a", "hard_j2"))
    assert artifacts["ckpt_steps_present"] == [int(ckpt_dir.name)]
    assert artifacts["train_log_path"] == str(
        config.LEDGER_DIR / pull.LOGS_SUBDIR / "hard_j2_train_attempt1.log")
    assert artifacts["recipe_seed"] == config.pull_seed(2)
    # "wid123" is not a real wandb run -- no offline-run dir should match under
    # the real WANDB_BASE_DIR, so this must resolve to None, not raise.
    assert artifacts["wandb_dir"] is None


def test_run_pull_training_exhausted_row_records_training_artifacts_json(isolated):
    def failing_popen(cmd, **kwargs):
        return FakeProc(always=1)

    row = pull.run_pull(
        "null", 5, "a", B=0, pool_df=make_pool_df([]),
        eval_fn=_eval_fn_ok, gpu=0,
        dataset_runner=fake_dataset_runner, popen_fn=failing_popen,
        sleep_fn=lambda secs: None,
    )

    assert row["status"] == "failed"
    artifacts = json.loads(row["training_artifacts_json"])
    assert artifacts["ckpt_steps_present"] == []  # training never produced a checkpoint
    assert artifacts["train_log_path"] == str(
        config.LEDGER_DIR / pull.LOGS_SUBDIR / "null_j5_train_attempt2.log")
    assert artifacts["recipe_seed"] == config.pull_seed(5)


# --- (j) resume-safety: training fast path + eval_failed row (race-runner --
#     review Lead Finding, .superpowers/sdd/task-racerunner-report.md) ------

def _write_complete_checkpoint(ckpt_dir):
    """Mirrors a REAL completed openpi checkpoint's on-disk shape (verified
    against .../pi0_ppc2sink_pi0base/pi0_v1/19999): a top-level
    `_CHECKPOINT_METADATA` file (orbax's per-step commit marker) plus a
    non-empty `params/` subdirectory."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / pull.CKPT_METADATA_FILE).write_text("{}")
    (ckpt_dir / "params").mkdir(exist_ok=True)
    (ckpt_dir / "params" / "manifest.ocdbt").write_text("x")


def test_checkpoint_looks_complete_false_when_dir_missing(isolated):
    assert pull.checkpoint_looks_complete(config.OPENPI / "checkpoints" / "cfg" / "exp" / "19999") is False


def test_checkpoint_looks_complete_false_when_metadata_file_missing(isolated):
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "hard_j2")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "params").mkdir()
    (ckpt_dir / "params" / "x").write_text("x")
    assert pull.checkpoint_looks_complete(ckpt_dir) is False  # no _CHECKPOINT_METADATA


def test_checkpoint_looks_complete_false_when_params_dir_empty(isolated):
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "hard_j2")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / pull.CKPT_METADATA_FILE).write_text("{}")
    (ckpt_dir / "params").mkdir()
    assert pull.checkpoint_looks_complete(ckpt_dir) is False  # params/ present but empty


def test_checkpoint_looks_complete_true_for_real_completed_shape(isolated):
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "hard_j2")
    _write_complete_checkpoint(ckpt_dir)
    assert pull.checkpoint_looks_complete(ckpt_dir) is True


def test_run_pull_skips_training_when_final_checkpoint_already_complete(isolated):
    """The resume-safety fast path: if a PRIOR invocation already produced a
    complete checkpoint (e.g. an eval-stage exception left no ok row, but
    training itself succeeded), a fresh run_pull call must NOT relaunch
    `scripts/train.py --overwrite` -- that would rmtree the very checkpoint
    it's about to serve. popen_fn raises if a train.py launch is ever
    attempted, so this test fails loudly if the fast path regresses."""
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "null_j2")
    _write_complete_checkpoint(ckpt_dir)

    def popen_fn(cmd, **kwargs):
        if "scripts/train.py" in cmd:
            raise AssertionError("training must not be launched -- checkpoint already complete")
        return FakeProc(always=None)  # serve_policy.py

    eval_calls = []
    row = pull.run_pull(
        "null", 2, "a", B=0, pool_df=make_pool_df([]),
        eval_fn=lambda *a, **kw: _eval_fn_ok(*a, calls=eval_calls, **kw),
        gpu=0, dataset_runner=fake_dataset_runner,
        popen_fn=popen_fn, connect_fn=lambda host, port: True,
        sleep_fn=lambda secs: None,
    )

    assert row["status"] == "ok"
    assert row["n_train_attempts"] == 0
    assert row["checkpoint_id"] == str(ckpt_dir)
    assert "fast path" in row["note"]
    assert len(eval_calls) == 1

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 1
    assert ledger_rows.iloc[0]["status"] == "ok"


def test_run_pull_eval_fn_exception_writes_eval_failed_row_and_reraises(isolated):
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "null_j3")

    def fake_sleep(secs):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def exploding_eval_fn(port, policy_id, arm, pull_id):
        raise RuntimeError("worker timed out after 10800s")

    with pytest.raises(RuntimeError, match="worker timed out"):
        pull.run_pull(
            "null", 3, "a", B=0, pool_df=make_pool_df([]),
            eval_fn=exploding_eval_fn, gpu=0,
            dataset_runner=fake_dataset_runner, popen_fn=make_fake_popen(None),
            connect_fn=lambda host, port: True, sleep_fn=fake_sleep,
        )

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 1
    row = ledger_rows.iloc[0]
    assert row["status"] == "eval_failed"
    assert row["checkpoint_id"] == str(ckpt_dir)
    assert list(row["demo_ids"]) == []
    assert "worker timed out after 10800s" in row["note"]
    assert row["n_train_attempts"] == 1
    artifacts = json.loads(row["training_artifacts_json"])
    assert artifacts["ckpt_root"] == str(pull.ckpt_root_dir("pi0_ppc2sink_bandit_a", "null_j3"))


def test_run_pull_eval_failed_note_is_truncated_for_a_long_error():
    long_msg = "x" * (pull.EVAL_ERROR_NOTE_MAX_LEN * 2)
    note = pull._truncate_error(RuntimeError(long_msg))
    assert len(note) < len(long_msg)
    assert note.startswith("eval_failed: RuntimeError:")
    assert "truncated" in note


def test_run_pull_missing_eval_fn_also_writes_eval_failed_row(isolated):
    """`eval_fn is None` raises NotImplementedError from inside the same
    eval-step try/except -- it must get the same ledger-row treatment as
    any other eval-stage exception, not be a second silent gap."""
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "null_j4")

    def fake_sleep(secs):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(NotImplementedError, match="Task 9"):
        pull.run_pull(
            "null", 4, "a", B=0, pool_df=make_pool_df([]),
            gpu=0, dataset_runner=fake_dataset_runner, popen_fn=make_fake_popen(None),
            connect_fn=lambda host, port: True, sleep_fn=fake_sleep,
        )

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 1
    assert ledger_rows.iloc[0]["status"] == "eval_failed"
    assert "Task 9" in ledger_rows.iloc[0]["note"]


def test_run_pull_resume_after_eval_failed_reruns_only_eval_not_training(isolated):
    """The end-to-end resume-safety invariant the race-runner review asked
    for: an (arm, round) whose most recent invocation left an eval_failed
    row (training succeeded, eval blew up) must, on the NEXT run_pull call
    for that same (arm, round, slot), re-run ONLY the eval -- never
    relaunch training against the already-complete checkpoint."""
    ckpt_dir = pull.ckpt_final_dir("pi0_ppc2sink_bandit_a", "null_j5")

    def fake_sleep(secs):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_popen_calls = []

    def popen_fn(cmd, **kwargs):
        if "scripts/train.py" in cmd:
            train_popen_calls.append(cmd)
        return FakeProc(always=None)

    # --- invocation 1: training succeeds, eval blows up ---------------------
    with pytest.raises(RuntimeError, match="eval boom"):
        pull.run_pull(
            "null", 5, "a", B=0, pool_df=make_pool_df([]),
            eval_fn=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("eval boom")),
            gpu=0, dataset_runner=fake_dataset_runner, popen_fn=popen_fn,
            connect_fn=lambda host, port: True, sleep_fn=fake_sleep,
        )
    assert len(train_popen_calls) == 1

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 1
    assert ledger_rows.iloc[0]["status"] == "eval_failed"

    # The training that ran for real above only left a bare directory (this
    # test's fake_sleep just mkdir's it, like every other test in this
    # file) -- overwrite it with the shape a REAL completed checkpoint has,
    # exactly as if the real orbax save had actually finished.
    _write_complete_checkpoint(ckpt_dir)

    # --- invocation 2 (resume): same (arm, j, slot), eval now succeeds -------
    eval_calls = []
    row2 = pull.run_pull(
        "null", 5, "a", B=0, pool_df=make_pool_df([]),
        eval_fn=lambda *a, **kw: _eval_fn_ok(*a, calls=eval_calls, **kw),
        gpu=0, dataset_runner=fake_dataset_runner, popen_fn=popen_fn,
        connect_fn=lambda host, port: True, sleep_fn=fake_sleep,
    )

    assert row2["status"] == "ok"
    assert len(train_popen_calls) == 1   # NOT relaunched on resume
    assert len(eval_calls) == 1

    ledger_rows = ledger.read("pulls")
    assert len(ledger_rows) == 2
    assert list(ledger_rows["status"]) == ["eval_failed", "ok"]


def test_append_gradient_analysis_note_writes_block_and_guards_double_append(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("existing: true\n")

    pull.append_gradient_analysis_note_to_config_yaml(path=cfg, log=lambda *a: None)
    doc = yaml.safe_load(cfg.read_text())
    assert doc["existing"] is True
    assert doc["gradient_analysis"]["rationale"] == "future per-demo gradient/influence analysis"
    assert list(doc["gradient_analysis"]["retained"]) == list(pull.GRADIENT_ANALYSIS_RETAINED)

    before = cfg.read_text()
    pull.append_gradient_analysis_note_to_config_yaml(path=cfg, log=lambda *a: None)
    assert cfg.read_text() == before  # guard fired -- second call is a pure no-op


def test_append_gradient_analysis_note_creates_file_if_absent(tmp_path):
    cfg = tmp_path / "config.yaml"
    path = pull.append_gradient_analysis_note_to_config_yaml(path=cfg, log=lambda *a: None)
    assert path == cfg
    doc = yaml.safe_load(cfg.read_text())
    assert "gradient_analysis" in doc
