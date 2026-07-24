"""Frozen constants for bandit v1. Design: weakregion/BANDIT_V1_DESIGN.md. Never edit mid-run."""
import types
from pathlib import Path

REPO = Path("/data/xinyua11/robocasa")
TASK = "PickPlaceCounterToSink"

POOL_LEROBOT = Path("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
                    "PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot")
ARMS_JSON = REPO / "weakregion/arms.json"          # D0 = base_episodes
FX_POOL_JSON = REPO / "weakregion/factor_analysis/fx_pool.json"
D0_DATASET = Path("/data/xinyua11/ft_arms/ppc2sink_base_only")
FT_ARMS_ROOT = Path("/data/xinyua11/ft_arms")

# Prior per-category difficulty source for bandit_v1/diagnosis.py's tercile map
# (task 6). fx_episodes.json's top-level "cats" list ([{"name","sr","n","h","w"}])
# carries a per-category prior success rate ("sr") -- same shape as fx_pool.json's
# "cats" list that states.py's _category_hw already trusts for h/w, so diagnosis.py
# trusts "sr" the same way (no recomputation from fx_episodes.json's own "rows",
# which are a different, feature-joined episode slice -- see task-6-report.md for
# why the two don't match row-for-row). POOLED_EPISODES_CSV is the documented
# fallback if fx_episodes.json is ever absent/lacks "sr".
FX_EPISODES_JSON = REPO / "weakregion/factor_analysis/fx_episodes.json"
POOLED_EPISODES_CSV = REPO / "weakregion/factor_analysis/pooled_episodes.csv"

# hdf5 to source robomimic env_args (env_name/type/env_kwargs: robot, controller,
# camera rig) from -- same PandaOmron + HYBRID_MOBILE_BASE recipe used to generate
# the pool itself. Loaded via robomimic_dataset_utils.get_env_metadata_from_dataset,
# same pattern as policy_analysis/check_train_eval_disjoint.py. No cameras/rendering
# needed for state capture, so only env_kwargs (not the recorded camera sizes) matter.
ENV_ARGS_HDF5 = REPO / "mimicgen_src/PickPlaceCounterToSink_pi0_src.hdf5"

# CATEGORY_ALIASES -- task-3 fix for the cross-process determinism gate's one
# recurring failure mode (see .superpowers/sdd/task-3-report.md's root-cause
# diagnosis). robocasa's object registry (OBJ_CATEGORIES, built by
# robocasa/models/objects/kitchen_object_utils.py from kitchen_objects.py) registers
# some mjcf instances under TWO overlapping category names. Forward sampling
# (sample_kitchen_object_helper, groups=<category name>) can label such an
# instance with either name; env.reset_to()'s reverse mjcf_path->category lookup
# (same function, groups=<xml path>, a first-match loop over
# `for cand_cat in OBJ_CATEGORIES: for reg in obj_registries: ...`) always resolves
# it to whichever of the two names comes first in OBJ_CATEGORIES's dict insertion
# order, regardless of which name the instance was originally forward-sampled
# under. CATEGORY_ALIASES maps every such alias name to that reverse-lookup winner
# (the canonical name), so `category` compares/joins consistently no matter which
# code path (forward sample at capture vs. reverse lookup at restore) produced it.
# Per the owner decision: the mjcf instance path is the identity ground truth;
# category is canonicalized to match it everywhere in bandit_v1.
#
# Generated 2026-07-23 by a one-off script that imported
# robocasa.models.objects.kitchen_object_utils.OBJ_CATEGORIES (robocasa_pkg at
# /data/xinyua11/robocasa_pkg) and replicated sample_kitchen_object_helper's
# reverse-lookup traversal with obj_registries=("objaverse", "lightwheel") --
# environments/kitchen/kitchen.py's own obj_registries default, i.e. the registry
# set bandit_v1's env actually uses -- to find, for every mjcf_path, every
# category name whose registry (in either "objaverse" or "lightwheel") contains
# it, in the exact traversal order the reverse lookup uses; the first name found
# for a path is that path's canonical label.
#
# Registry counts at generation time: 198 total categories, 1,516 distinct
# mjcf_paths, 17 (1.1%) dual-registered under exactly one of 2 overlapping
# category pairs -- and, checked exhaustively, each alias category's mjcf_path set
# is a FULL SUBSET of its canonical counterpart's (no per-instance exceptions, so
# the alias is safe to apply at the category-name level, not just per-instance):
#   - jug_wide_opening (5 paths, OBJ_CATEGORIES index 44) subset of
#     jug (9 paths, index 43) -> jug is earlier -> canonical = "jug".
#   - saucepan_with_lid (12 paths, index 186) subset of
#     saucepan (15 paths, index 185) -> saucepan is earlier -> canonical = "saucepan".
CATEGORY_ALIASES = types.MappingProxyType({
    "jug_wide_opening": "jug",
    "saucepan_with_lid": "saucepan",
})

OPENPI = Path("/data/xinyua11/openpi")
PRETRAIN_CKPT = Path("/data/xinyua11/checkpoints/pi0/pi0_robocasa_pretrain_human300/"
                     "multitask_learning/75000/params")
TRAIN_STEPS = 20000
FINAL_CKPT_STEP = 19999      # checkpoint dir name at end of training (matches prior arms)

BANDIT_DIR = REPO / "bandit_v1"
LEDGER_DIR = BANDIT_DIR / "ledger"
STATES_DIR = BANDIT_DIR / "states"          # saved-state starts: states/<set>/<start_id>/
E_DIR = STATES_DIR / "E"
DIAG_DIR = STATES_DIR / "diag"
DIAG_CHECK_DIR = STATES_DIR / "diag_check"  # scratch out_dir for diagnosis.py's --out_check dry-run
DIAG_TERCILE_MAP_JSON = LEDGER_DIR / "diag_tercile_map.json"  # frozen category->tercile map (task 6)
MAP_MODELS_JOBLIB = LEDGER_DIR / "map_models.joblib"  # fitted p_hat_0 + p_stage (task 8, map_fit.py)

SEED_PI0 = 1000
def pull_seed(j: int) -> int: return 1000 + j       # round j >= 1
NULL_SEEDS = (1001, 1002)
DIAG_ENV_SEED_BASE = 600000
E_ENV_SEED_BASE = 500000

N_DIAG, M_DIAG = 300, 8
N_E, EVAL_REPEATS = 150, 3
EPS_XY = 0.10                                # meters, sink-relative, same-category rule
B_CANDIDATES = (200, 100, 60, 20)
K_RANGE = range(3, 9)
MAX_ARMS = 7                                 # including Random
MIN_CLUSTER_FRAC = 0.05
DELTA_CONF = 0.1
T_MAX_PULLS = 16
EXPLOIT_FACTOR = 2                           # exploit batch = 2*B
