"""A coarse RoboCasa digital twin for PickPlaceCounterToSink + pi0.

Region GROUPS and their parameters are hand-set from the measured study (7 arms,
per-category rates, collection yields, retention regressions) -- NOT least-squares
fit, because 7 aggregate points cannot identify a rich per-region model. The twin is
therefore a plausible, qualitatively-faithful sandbox, not a precise fit; generality
comes from the regime sweep and transfer from the real anchors (see the design spec).

Sources baked in:
  * unfixable_hard: pi0 0/14, GR00T 1/9 on juice/jar/cheese_grater -> collectability ~0.05
  * hard_collectable: colander/pitcher/reamer/... mid yields ~0.65, real headroom
  * retention_toxic: the value arm's grasp cluster (tongs/dish_brush/mug) -> ret_risk>0
  * easy_majority: the non-targeted ~88% that every losing arm regressed
  * feature_fidelity LOW: offline selector quality anti-correlated with deployed success
    (whiten AUC 0.677 -> worst rollout), so cheap features barely track deployed value.
"""
from __future__ import annotations

import numpy as np

from .env import PrescriptionEnv, Region
from .allocators.predictor import PPP
from .allocators.bandit_sh import SuccessiveHalvingBandit

GROUPS = ("unfixable_hard", "hard_collectable", "retention_toxic", "easy_majority")

# (arm, offline selection-quality, deployed stratified-targeted success).
# Offline metric = ranking AUC where the arm optimizes one (whiten 0.677; failretr's
# retrieval similarity ~0.60, approximate -- flagged), else 0.5 (chance = no offline
# ranking signal). Deployed from weakregion/eval_strat_*. The offline-optimized arm
# (whiten) deployed WORST -> a NEGATIVE offline<->online correlation (the paper's
# finding). COARSE (few arms, mixed metrics): the robust, cited fact is the SIGN.
ROBOCASA_OFFLINE_ONLINE = [
    ("whiten",   0.677, 0.268),   # best offline ranking AUC, worst deployed
    ("failretr", 0.600, 0.297),   # retrieval-similarity optimized (approx), low deployed
    ("random",   0.500, 0.351),   # no offline optimization
    ("core",     0.500, 0.371),   # P(fail) heuristic, best deployed
    ("baseline", 0.500, 0.262),   # no fine-tuning (no-selection anchor)
]


def _rankdata(x):
    """Ranks with average-rank tie handling (for a proper Spearman)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def fit_fidelity(offline, deployed, method: str = "spearman") -> float:
    """Fit the twin's feature fidelity = correlation(offline metric, deployed value).

    Negative => offline features anti-track deployed success (the real anti-correlation).
    Spearman (rank) by default so mixed-scale offline metrics only need to be ordinally
    comparable. Returns 0 if either column has no variance.
    """
    a = np.asarray(offline, dtype=float)
    b = np.asarray(deployed, dtype=float)
    if method == "spearman":
        a, b = _rankdata(a), _rankdata(b)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float(np.clip((a * b).sum() / denom, -1.0, 1.0))


def robocasa_env(feature_fidelity=None, n_eval: int = 280) -> PrescriptionEnv:
    """The calibrated twin. feature_fidelity defaults to the value FIT from the real
    offline<->online anti-correlation (negative), not a hand-set guess."""
    if feature_fidelity is None:
        feature_fidelity = fit_fidelity(
            [m for _, m, _ in ROBOCASA_OFFLINE_ONLINE],
            [d for _, _, d in ROBOCASA_OFFLINE_ONLINE],
        )
    regions = [
        # base, headroom, tau, collectability, ret_risk, weight, name
        # uncollectable tail: tiny yield + a small retention cost so wasting budget here loses.
        Region(0.05, 0.05, 20.0, 0.05, 0.0005, 0.10, "unfixable_hard"),
        # the only real headroom: collectable, low retention cost, slow saturation.
        Region(0.30, 0.25, 40.0, 0.65, 0.0002, 0.15, "hard_collectable"),
        # the value arm's grasp cluster: collectable but retention-toxic.
        Region(0.50, 0.05, 20.0, 0.80, 0.0006, 0.05, "retention_toxic"),
        # already solved: ZERO gain headroom, heavy weight -> pure retention liability.
        Region(0.62, 0.00, 20.0, 0.80, 0.0001, 0.70, "easy_majority"),
    ]
    # decoy = P(fail): at low fidelity the predictor's proxy collapses to this (the
    # exact heuristic that mis-ranks deployed value in the study).
    decoy = 1.0 - np.array([r.base for r in regions])
    return PrescriptionEnv(regions, n_eval=n_eval, feature_fidelity=feature_fidelity,
                           decoy=decoy, probe_noise0=0.2)


def anchor_allocations(env, demo_budget, measure_budget, rng):
    """Emit the predictor's and the bandit's committed allocations -- the two real
    pool-selected fine-tunes to run as transfer anchors (which region-group to add
    demos for in each arm)."""
    pred = PPP().allocate(env, demo_budget, measure_budget, rng).allocation
    band = SuccessiveHalvingBandit().allocate(env, demo_budget, measure_budget, rng).allocation
    return {
        "region_names": [r.name for r in env.regions],
        "predictor_pick": pred,
        "bandit_pick": band,
    }
