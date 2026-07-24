"""Tests for bandit_v1/eval_set.py (Task 9: eval set E + baseline-eval
aggregation).

Synthetic, no env/GPU: `states.capture_start`/`states.start_features` and
`rollout.run` are always monkeypatched (same pattern test_rollout.py/
test_run_diagnosis.py use elsewhere) -- the real 300-candidate capture and the
real served-policy eval are DEFERRED to run_baseline.sh, which is not
exercised here (see its own docstring for why).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bandit_v1 import config, eval_set, ledger, pull


# =============================================================================
# stratify: determinism + 50/50/50 + tercile edges (brief Step 1)
# =============================================================================

def _uniform_cands(n=300, seed=0):
    """n candidates with a p_hat column shuffled into a fixed, arbitrary row
    order (not sorted) -- the exact "given 300 fake candidates with p_hat
    uniform" scenario the brief's Step 1 test describes."""
    rng = np.random.default_rng(seed)
    p_hat = rng.permutation(np.linspace(0.0, 1.0, n))
    return pd.DataFrame({"start_id": [f"start_{i:05d}" for i in range(n)], "p_hat": p_hat})


def test_stratify_50_50_50_and_deterministic_under_fixed_candidate_order():
    cands = _uniform_cands(300)

    out1 = eval_set.stratify(cands, n=150)
    out2 = eval_set.stratify(cands, n=150)  # same input, same order -> identical output

    assert len(out1) == 150
    assert out1["stratum"].value_counts().to_dict() == {"hard": 50, "mid": 50, "easy": 50}
    pd.testing.assert_frame_equal(out1, out2)
    # every selected start_id is unique (no candidate picked into two strata)
    assert out1["start_id"].is_unique


def test_stratify_tercile_edges_do_not_overlap():
    cands = _uniform_cands(300)
    out = eval_set.stratify(cands, n=150)

    hard = out.loc[out["stratum"] == "hard", "p_hat"]
    mid = out.loc[out["stratum"] == "mid", "p_hat"]
    easy = out.loc[out["stratum"] == "easy", "p_hat"]
    assert hard.max() <= mid.min()
    assert mid.max() <= easy.min()
    # hard should span the low end and easy the high end of the full candidate range
    assert hard.min() == pytest.approx(cands["p_hat"].min(), abs=1e-9)
    assert easy.max() == pytest.approx(cands["p_hat"].max(), abs=1e-9)


def test_stratify_preserves_original_columns():
    cands = _uniform_cands(300)
    cands["category"] = "jar"
    out = eval_set.stratify(cands, n=150)
    assert "category" in out.columns
    assert (out["category"] == "jar").all()


def test_stratify_n_not_divisible_by_3_raises():
    cands = _uniform_cands(300)
    with pytest.raises(ValueError, match="divisible"):
        eval_set.stratify(cands, n=100)


def test_stratify_too_few_candidates_raises():
    cands = _uniform_cands(100)
    with pytest.raises(ValueError, match="only 100"):
        eval_set.stratify(cands, n=150)


def test_stratify_missing_p_hat_column_raises():
    cands = pd.DataFrame({"start_id": ["s0", "s1"]})
    with pytest.raises(ValueError, match="p_hat"):
        eval_set.stratify(cands, n=150)


# =============================================================================
# write_manifest: schema + hash
# =============================================================================

def _make_selected_row(start_id="start_00000", seed=500000, stratum="easy", p_hat=0.9):
    return {
        "start_id": start_id, "seed": seed, "category": "jar", "h": 0.194, "w": 0.201,
        "x_rel": 0.1, "y_rel": 0.1, "side": 1, "yaw": 0.0, "layout_id": 1, "style_id": 1,
        "p_hat": p_hat, "stratum": stratum,
    }


def test_write_manifest_schema_and_hash_merge_updates_existing_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "hashes.json").write_text(json.dumps({"fx_pool.json": "deadbeef"}))

    selected = pd.DataFrame([_make_selected_row()])
    path = eval_set.write_manifest(selected)

    assert path == tmp_path / "E_manifest.parquet"
    written = pd.read_parquet(path)
    assert list(written.columns) == eval_set.MANIFEST_COLUMNS
    assert len(written) == 1
    assert written.loc[0, "start_id"] == "start_00000"

    hashes = json.loads((tmp_path / "hashes.json").read_text())
    assert hashes["fx_pool.json"] == "deadbeef"  # pre-existing key untouched
    assert hashes["E_manifest.parquet"] == ledger.file_hash(path)


def test_write_manifest_raises_on_missing_column(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path)
    selected = pd.DataFrame([{"start_id": "s0", "seed": 1}])
    with pytest.raises(ValueError, match="missing columns"):
        eval_set.write_manifest(selected)


# =============================================================================
# build_E: monkeypatched capture, no live env
# =============================================================================

class _FakeModels:
    """predict_p = 1 - h (monotone, invertible) -- lets the test hand-derive
    exactly which scan indices should end up in which stratum."""
    def predict_p(self, df):
        return np.clip(1.0 - df["h"].to_numpy(dtype=float), 0.0, 1.0)


def test_build_E_scores_stratifies_deletes_nonselected_dirs_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    out_dir = tmp_path / "E"

    def fake_capture_start(seed, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fingerprint.json").write_text("{}")  # existence marker only

    def fake_start_features(start_dir):
        idx = int(Path(start_dir).name.split("_")[1])
        return {
            "category": "jar", "h": float(idx) / 100.0, "w": 0.2,
            "x_rel": 0.1, "y_rel": 0.1, "side": 1, "yaw": 0.0,
            "layout_id": 1, "style_id": 1,
        }

    monkeypatch.setattr(eval_set.states, "capture_start", fake_capture_start)
    monkeypatch.setattr(eval_set.states, "start_features", fake_start_features)

    selected = eval_set.build_E(_FakeModels(), n_candidates=30, n=15, out_dir=out_dir,
                                 log=lambda *a: None)

    assert len(selected) == 15
    assert selected["stratum"].value_counts().to_dict() == {"hard": 5, "mid": 5, "easy": 5}
    assert selected["start_id"].is_unique

    kept_dirs = {p.name for p in out_dir.iterdir()}
    assert kept_dirs == set(selected["start_id"])
    assert len(kept_dirs) == 15  # the other 15 of the 30 captured candidates were deleted

    manifest = pd.read_parquet(tmp_path / "ledger" / "E_manifest.parquet")
    assert set(manifest.columns) == set(eval_set.MANIFEST_COLUMNS)
    assert len(manifest) == 15

    hashes = json.loads((tmp_path / "ledger" / "hashes.json").read_text())
    assert "E_manifest.parquet" in hashes


def test_build_E_discards_unbinnable_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "ledger")
    out_dir = tmp_path / "E"

    def fake_capture_start(seed, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fingerprint.json").write_text("{}")

    def fake_start_features(start_dir):
        idx = int(Path(start_dir).name.split("_")[1])
        if idx < 6:
            # first 6 scanned candidates are "unbinnable" (no h/w prior)
            return {"category": "pot", "h": None, "w": None, "x_rel": 0.1, "y_rel": 0.1,
                    "side": 1, "yaw": 0.0, "layout_id": 1, "style_id": 1}
        return {"category": "jar", "h": float(idx) / 100.0, "w": 0.2, "x_rel": 0.1,
                "y_rel": 0.1, "side": 1, "yaw": 0.0, "layout_id": 1, "style_id": 1}

    monkeypatch.setattr(eval_set.states, "capture_start", fake_capture_start)
    monkeypatch.setattr(eval_set.states, "start_features", fake_start_features)

    selected = eval_set.build_E(_FakeModels(), n_candidates=30, n=15, out_dir=out_dir,
                                 log=lambda *a: None)

    assert len(selected) == 15
    assert "start_00000" not in set(selected["start_id"])  # unbinnable, discarded
    # unbinnable capture dirs are deleted same as any other non-selected dir
    kept_dirs = {p.name for p in out_dir.iterdir()}
    assert "start_00000" not in kept_dirs


# =============================================================================
# eval_checkpoint / _aggregate_eval_rows: aggregation math + missing-start gate
# =============================================================================

def _make_manifest(n_per_stratum=2):
    rows = []
    sid = 0
    for stratum in ("hard", "mid", "easy"):
        for _ in range(n_per_stratum):
            rows.append({
                "start_id": f"start_{sid:05d}", "seed": 500000 + sid, "category": "jar",
                "h": 0.1, "w": 0.2, "x_rel": 0.1, "y_rel": 0.1, "side": 1, "yaw": 0.0,
                "layout_id": 1, "style_id": 1, "p_hat": 0.5, "stratum": stratum,
            })
            sid += 1
    return pd.DataFrame(rows)


# Hand-computed success table for a 6-start (2/stratum) x 2-repeat manifest.
# Ordinal 0,1 = hard; 2,3 = mid; 4,5 = easy (manifest row order).
_SUCCESS_TABLE = {
    (0, 0): False, (0, 1): False,  # hard start 0: always fails
    (1, 0): False, (1, 1): True,   # hard start 1: fails r0, succeeds r1 (a "flip")
    (2, 0): True,  (2, 1): True,   # mid start 2: always succeeds
    (3, 0): False, (3, 1): False,  # mid start 3: always fails
    (4, 0): True,  (4, 1): True,   # easy start 4: always succeeds
    (5, 0): True,  (5, 1): False,  # easy start 5: succeeds r0, fails r1
}
# hard = (0.0, 0.5); mid = (0.5, 0.5); easy = (1.0, 0.5) per-repeat means
# per_repeat_means = [3/6, 3/6] = [0.5, 0.5]; per_stratum_mean = hard 0.25, mid 0.5, easy 0.75


def _fake_run_from_table(table, drop=None):
    def fake_run(host, port, start_dirs, reps, phase, policy_id, arm=None, pull_id=None,
                 skip_pairs=None):
        rows = []
        for ordinal, sd in enumerate(start_dirs):
            for r in range(reps):
                if drop is not None and (ordinal, r) in drop:
                    continue
                rows.append({"start_id": Path(sd).name, "repeat_idx": r,
                              "success": table[(ordinal, r)],
                              "phase": phase, "policy_id": policy_id})
        return rows
    return fake_run


def test_eval_checkpoint_aggregates_per_repeat_and_per_stratum_means():
    manifest = _make_manifest(n_per_stratum=2)
    fake_run = _fake_run_from_table(_SUCCESS_TABLE)

    result = eval_set.eval_checkpoint(9999, "pi0_baseline", None, "pi0_baseline",
                                       repeats=2, manifest=manifest, run_fn=fake_run)

    assert result["per_repeat_means"] == pytest.approx([0.5, 0.5])
    assert result["mean"] == pytest.approx(0.5)
    assert result["per_stratum_means"]["hard"] == pytest.approx([0.0, 0.5])
    assert result["per_stratum_means"]["mid"] == pytest.approx([0.5, 0.5])
    assert result["per_stratum_means"]["easy"] == pytest.approx([1.0, 0.5])
    # explicit alias carries identical content
    assert result["per_stratum_per_repeat"] == result["per_stratum_means"]
    assert result["per_stratum_mean"] == pytest.approx({"hard": 0.25, "mid": 0.5, "easy": 0.75})


def test_eval_checkpoint_default_repeats_is_config_eval_repeats():
    manifest = _make_manifest(n_per_stratum=2)
    calls = {}

    def fake_run(host, port, start_dirs, reps, phase, policy_id, arm=None, pull_id=None,
                 skip_pairs=None):
        calls["reps"] = reps
        rows = []
        for sd in start_dirs:
            for r in range(reps):
                rows.append({"start_id": Path(sd).name, "repeat_idx": r, "success": True})
        return rows

    eval_set.eval_checkpoint(9999, "pi0_baseline", None, "pi0_baseline",
                              manifest=manifest, run_fn=fake_run)
    assert calls["reps"] == config.EVAL_REPEATS


def test_eval_checkpoint_default_repeats_reads_config_at_call_time_not_import_time(monkeypatch):
    """`repeats` must resolve config.EVAL_REPEATS inside the function body at
    CALL time, not bind it as a signature default at IMPORT time -- a
    signature default of `config.EVAL_REPEATS` would freeze whatever value
    was live when eval_set.py was first imported, silently ignoring any
    later monkeypatch. Patch config.EVAL_REPEATS to an off-default value
    well after import and confirm an omitted-`repeats` call picks it up."""
    manifest = _make_manifest(n_per_stratum=2)
    monkeypatch.setattr(config, "EVAL_REPEATS", 7)
    calls = {}

    def fake_run(host, port, start_dirs, reps, phase, policy_id, arm=None, pull_id=None,
                 skip_pairs=None):
        calls["reps"] = reps
        rows = []
        for sd in start_dirs:
            for r in range(reps):
                rows.append({"start_id": Path(sd).name, "repeat_idx": r, "success": True})
        return rows

    eval_set.eval_checkpoint(9999, "pi0_baseline", None, "pi0_baseline",
                              manifest=manifest, run_fn=fake_run)
    assert calls["reps"] == 7


def test_eval_checkpoint_missing_start_in_a_repeat_raises_loudly():
    manifest = _make_manifest(n_per_stratum=2)
    # drop the row for ordinal=3 (a "mid" start) at repeat_idx=1
    fake_run = _fake_run_from_table(_SUCCESS_TABLE, drop={(3, 1)})

    with pytest.raises(ValueError, match="missing"):
        eval_set.eval_checkpoint(9999, "pi0_baseline", None, "pi0_baseline",
                                  repeats=2, manifest=manifest, run_fn=fake_run)


def test_eval_checkpoint_duplicate_row_in_a_repeat_raises_loudly():
    manifest = _make_manifest(n_per_stratum=2)

    def fake_run(host, port, start_dirs, reps, phase, policy_id, arm=None, pull_id=None,
                 skip_pairs=None):
        rows = []
        for ordinal, sd in enumerate(start_dirs):
            for r in range(reps):
                rows.append({"start_id": Path(sd).name, "repeat_idx": r,
                              "success": _SUCCESS_TABLE[(ordinal, r)]})
                if ordinal == 2 and r == 0:
                    rows.append({"start_id": Path(sd).name, "repeat_idx": r, "success": True})
        return rows

    with pytest.raises(ValueError, match="duplicate"):
        eval_set.eval_checkpoint(9999, "pi0_baseline", None, "pi0_baseline",
                                  repeats=2, manifest=manifest, run_fn=fake_run)


def test_eval_checkpoint_manifest_missing_a_whole_stratum_raises_loudly():
    """A manifest with only 2 of the 3 STRATA (e.g. a hand-built/corrupted
    manifest missing "easy" entirely) must raise -- not silently mean() an
    empty slice into NaN for the missing stratum."""
    manifest = _make_manifest(n_per_stratum=2)
    manifest = manifest[manifest["stratum"] != "easy"].reset_index(drop=True)

    def fake_run(host, port, start_dirs, reps, phase, policy_id, arm=None, pull_id=None,
                 skip_pairs=None):
        rows = []
        for sd in start_dirs:
            for r in range(reps):
                rows.append({"start_id": Path(sd).name, "repeat_idx": r, "success": True})
        return rows

    with pytest.raises(ValueError, match="easy"):
        eval_set.eval_checkpoint(9999, "pi0_baseline", None, "pi0_baseline",
                                  repeats=2, manifest=manifest, run_fn=fake_run)


def test_eval_checkpoint_zero_rows_raises_loudly():
    manifest = _make_manifest(n_per_stratum=2)

    def fake_run(host, port, start_dirs, reps, phase, policy_id, arm=None, pull_id=None,
                 skip_pairs=None):
        return []

    with pytest.raises(ValueError, match="zero rows"):
        eval_set.eval_checkpoint(9999, "pi0_baseline", None, "pi0_baseline",
                                  repeats=2, manifest=manifest, run_fn=fake_run)


def test_eval_checkpoint_passes_policy_id_and_pull_id_through_separately():
    """The policy_id/pull_id doubling resolution: eval_checkpoint must forward
    each verbatim to rollout.run's like-named parameters -- not silently
    collapse them into one value."""
    manifest = _make_manifest(n_per_stratum=2)
    seen = {}

    def fake_run(host, port, start_dirs, reps, phase, policy_id, arm=None, pull_id=None,
                 skip_pairs=None):
        seen.update(policy_id=policy_id, arm=arm, pull_id=pull_id, phase=phase)
        rows = []
        for sd in start_dirs:
            for r in range(reps):
                rows.append({"start_id": Path(sd).name, "repeat_idx": r, "success": True})
        return rows

    eval_set.eval_checkpoint(9999, "pi0_baseline", "targeted", "pull_x_j1",
                              repeats=1, manifest=manifest, run_fn=fake_run)

    assert seen == {"policy_id": "pi0_baseline", "arm": "targeted",
                     "pull_id": "pull_x_j1", "phase": "eval"}


# =============================================================================
# shape-compatibility with pull.compute_delta
# =============================================================================

def test_eval_checkpoint_result_is_shape_compatible_with_pull_compute_delta():
    manifest = _make_manifest(n_per_stratum=2)
    fake_run = _fake_run_from_table(_SUCCESS_TABLE)

    result = eval_set.eval_checkpoint(9999, "pull_x_j1", "targeted", "pull_x_j1",
                                       repeats=2, manifest=manifest, run_fn=fake_run)

    delta_info = pull.compute_delta(
        result, baseline=0.4, baseline_per_stratum={"hard": 0.1, "mid": 0.4, "easy": 0.6})

    assert delta_info["overall_mean"] == pytest.approx(0.5)
    assert delta_info["delta"] == pytest.approx(0.1)
    assert delta_info["per_stratum_means"] == pytest.approx({"hard": 0.25, "mid": 0.5, "easy": 0.75})
    assert delta_info["delta_per_stratum"] == pytest.approx({"hard": 0.15, "mid": 0.1, "easy": 0.15})
    # compute_delta must consume this dict without ever raising a KeyError/shape error
    assert delta_info["per_repeat_means"] == pytest.approx([0.5, 0.5])


# =============================================================================
# per_start_flip_table
# =============================================================================

def test_per_start_flip_table_flags_disagreeing_repeats():
    df = pd.DataFrame([
        {"start_id": "s0", "repeat_idx": 0, "success": True},
        {"start_id": "s0", "repeat_idx": 1, "success": True},
        {"start_id": "s0", "repeat_idx": 2, "success": True},
        {"start_id": "s1", "repeat_idx": 0, "success": True},
        {"start_id": "s1", "repeat_idx": 1, "success": False},
        {"start_id": "s1", "repeat_idx": 2, "success": True},
        {"start_id": "s2", "repeat_idx": 0, "success": False},
        {"start_id": "s2", "repeat_idx": 1, "success": False},
        {"start_id": "s2", "repeat_idx": 2, "success": False},
    ])
    table = eval_set.per_start_flip_table(df).set_index("start_id")

    assert bool(table.loc["s0", "flip"]) is False
    assert bool(table.loc["s1", "flip"]) is True
    assert bool(table.loc["s2", "flip"]) is False
    assert int(table.loc["s1", "n_success"]) == 2
    assert int(table.loc["s1", "n_repeats"]) == 3


# =============================================================================
# append_baseline_to_config_yaml
# =============================================================================

def test_append_baseline_to_config_yaml_preserves_prior_content_and_is_valid_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("recipe:\n  task: PickPlaceCounterToSink\n")

    result = {
        "per_repeat_means": [0.30, 0.34, 0.32],
        "mean": 0.32,
        "per_stratum_mean": {"hard": 0.10, "mid": 0.32, "easy": 0.54},
    }

    out_path = eval_set.append_baseline_to_config_yaml(result, path=cfg_path, checkpoint_id="/ckpt/19999")
    assert out_path == cfg_path

    text = cfg_path.read_text()
    assert text.startswith("recipe:\n  task: PickPlaceCounterToSink\n")  # untouched prefix

    doc = yaml.safe_load(text)
    assert doc["recipe"]["task"] == "PickPlaceCounterToSink"  # original key survives
    assert doc["baseline"]["b"] == pytest.approx(0.32)
    assert doc["baseline"]["per_stratum_b"] == pytest.approx({"hard": 0.10, "mid": 0.32, "easy": 0.54})
    assert doc["baseline"]["sigma_e_eval"] == pytest.approx(np.std([0.30, 0.34, 0.32], ddof=1))
    assert doc["baseline"]["repeats"] == 3
    assert doc["baseline"]["checkpoint_id"] == "/ckpt/19999"


def test_append_baseline_to_config_yaml_single_repeat_sigma_is_zero(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("x: 1\n")
    result = {"per_repeat_means": [0.5], "mean": 0.5,
              "per_stratum_mean": {"hard": 0.1, "mid": 0.5, "easy": 0.9}}

    eval_set.append_baseline_to_config_yaml(result, path=cfg_path)
    doc = yaml.safe_load(cfg_path.read_text())
    assert doc["baseline"]["sigma_e_eval"] == 0.0
