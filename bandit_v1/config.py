"""Frozen constants for bandit v1. Design: weakregion/BANDIT_V1_DESIGN.md. Never edit mid-run."""
from pathlib import Path

REPO = Path("/data/xinyua11/robocasa")
TASK = "PickPlaceCounterToSink"

POOL_LEROBOT = Path("/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/"
                    "PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot")
ARMS_JSON = REPO / "weakregion/arms.json"          # D0 = base_episodes
FX_POOL_JSON = REPO / "weakregion/factor_analysis/fx_pool.json"
D0_DATASET = Path("/data/xinyua11/ft_arms/ppc2sink_base_only")
FT_ARMS_ROOT = Path("/data/xinyua11/ft_arms")

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
