"""ALLOCATOR WIND TUNNEL — race data-allocation strategies at a COMMON, METERED budget.

THE PROBLEM (unchanged): given a base policy and a budget, decide WHICH DATA TO ADD.
Every borrowed allocator claims to do that cheaply. This harness makes the claims
commensurable by charging all of them in one unit and scoring them by regret against
a known-value arm set.

WHY A WIND TUNNEL. On the robot one pull is 5-9 GPU-h with a +/-3.3pp noise floor, so
we can never race allocators there. The CIFAR sandbox (`/data/xinyua11/xgradtest`) has
the same race structure at ~55 s/pull WITH ground truth (the 'rare' arm is the right
answer, +3.8pp on the target stratum vs +0.9 for random; Q4_CIFAR_MIRROR_REPORT.md).
So: develop and falsify allocators here; port only what survives.

THE ONE UNIT.  1 FTE ("full-tune equivalent") = one full pull = 2000 training steps at
batch 128 + the standard eval (3 passes over the 10k test set). Everything is charged
in TRAINING-STEP EQUIVALENTS: one step at bs=128 = 1 unit, one forward pass over n
examples = n/(128*3) units (fwd ~ 1/3 of fwd+bwd). Measured on the box: eval/train
wall-clock ratio 2.3s/53s = 4.3% vs the model's 78/2000 = 3.9% -- the nominal model is
within 10% of reality and, unlike wall-clock, is deterministic and hardware-independent.

  * cheap proxy runs, learning-progress bursts, held-out loss probes, JEST reference
    models and JEST scoring passes ALL come out of the same budget as real pulls.
    "20 cheap probes" is not free: 20 x 250 steps + evals = 3.2 FTE (unit-tested).
  * cache hits are still charged (a repeated (arm, seed, steps) IS the same fine-tune;
    the cache saves wall-clock, never budget), so no allocator can farm the cache.
  * the meter is inside the environment; allocators cannot spend except through it.

ALLOCATORS (all implement the same interface, see `Allocator`):
  successive_halving      (a) our incumbent: paired-seed rounds + elimination
  learning_progress_exp3  (b) EXP3 over arms, reward = rate of held-out target-loss
  learning_progress_ucb   (b) UCB1 variant of the same reward          [ODM / Graves]
  regmix                  (c) k cheap proxy runs on random mixtures -> ridge on the
                              mixture -> argmax mixture               [RegMix 2407.01492]
  jest                    (d) learnability = learner loss - reference loss, no
                              fine-tuning at all                      [JEST 2406.17711]
  jest_then_confirm       (d+) the practical hybrid: JEST screens, leftover budget
                              confirms the top-2 with real pulls
  random                  (e) uniform-random arm per pull, pick empirical argmax (FLOOR)
  oracle                  (e) knows the true values, spends 0         (CEILING)

CALIBRATION (measured here; wind_tunnel_calibration.json, do not skip):
  * SAME-SEED RUNS DO NOT REPRODUCE. Re-running the IDENTICAL pull (same draw,
    same seed, same steps, same code) three times gave 3.70 / 2.85 / 3.35pp on the
    rare arm (sd 0.43pp) and -0.30 / -0.05 / -2.80pp on the null arm (sd 1.52pp,
    n=3 each, so these sds are themselves rough). Two consequences. (i) The Q4
    CIFAR noise floor of +/-0.4pp -- estimated from four null-arm rounds -- is
    UNDERSTATED; pure GPU nondeterminism at a fixed seed already spans 2.75pp on
    the null arm. Only the rare-vs-random gap (2.9pp) survives that; every
    smaller CIFAR arm difference (gradarms, random-vs-easy) is unresolved.
    (ii) In this sandbox the SEED barely matters -- the opposite of the robot
    stack, where the seed dominates and the data fingerprint is ~1.5%. So
    paired-seed designs, mandatory on the robot, buy almost nothing here: do not
    let the sandbox talk you out of pairing. A cache hit is therefore not truly
    the same number; `MemoBackend(replay_noise_sd=0.0043..0.015)` models a repeat
    as the fresh draw it really is.
  * CHEAP PROXIES DO NOT MIS-RANK HERE -- the opposite of the robot's measured
    constraint 3. Single-arm proxy pulls (2 seeds) rank the arms at Spearman 0.94
    (250 steps) and 1.00 (500 steps) against the full-fidelity truth, and both
    pick the right arm. They do INFLATE the winner (rare reads +7.2pp at 250 steps
    vs +3.8pp at 2000). So the sandbox cannot falsify a proxy-based allocator:
    RegMix/burst methods will always look good here. Their failure mode has to be
    tested in the robot-like regime (--mode synthetic_robot), where the measured
    mis-ranking is an explicit knob (`SyntheticSpec.proxy_bias`).
  * MIXTURE VALUE IS LINEAR to within noise: a real 50/50 rare+easy pull gave
    +1.60pp vs the +1.70pp linear prediction (residual -0.10pp), so scoring
    mixtures as sum_a w_a v_a is safe at this resolution.
  * THE CIFAR MIRROR IS A POSITIVE CONTROL, NOT A DISCRIMINATOR. Its spread/noise
    ratio is so favorable (2.9pp gap vs ~0.4pp noise) that at 6 FTE ALL eight
    allocators -- including the random floor -- pick the right arm and score
    regret 0. Ranking allocators requires either the robot-like noise regime
    (--mode synthetic_robot) or a budget below one paired round.

DECISION SPACE / SCORING. A decision is a weight vector on the simplex over arms
(single-arm allocators return one-hot). Value V(w) = sum_a w_a v_a, regret =
max_a v_a - V(w), normalized regret = regret / regret(random floor). The linear
mixture-value model is an ASSUMPTION; `--check-linearity` tests it with one real
50/50 mixture pull.

BACKENDS.
  SyntheticBackend  zero-GPU arms with known values. Encodes our five measured
                    constraints as tunable knobs: a shared per-seed effect (constraint
                    4: the seed, not the data, steers the run), a fidelity-dependent
                    proxy bias (constraint 3: short-burst proxies MIS-RANK), and a
                    learning-progress signal that can be decoupled from arm value
                    (constraint F4: training-loss effects need not transfer).
                    SPEC_ROBOT_LIKE is parameterized from the MEASURED ledger
                    (spread 0.65pp, noise 1.84pp; decision_accuracy_diagnostic.json).
  CifarBackend      real fine-tunes, reusing `xarm_race.finetune_eval` unchanged
                    (FT_STEPS monkeypatched for cheap proxies) so the 24 completed
                    armrace pulls import straight into the cache as free cache hits,
                    and the armrace arm means are the declared ground truth.

RUN
  tests (no GPU):  python -m pytest test_wind_tunnel.py -q
  synthetic race:  python wind_tunnel.py --mode synthetic --budget 6 --reps 200
  cifar smoke:     CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python wind_tunnel.py \
                       --mode cifar --budget 6 --reps 1 --check-linearity
  (env: /data/xinyua11/conda/envs/robocasa/bin/python)

Writes ONLY into gradient_analysis/llm_borrow/. Never touches the bandit ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XGT = "/data/xinyua11/xgradtest"
ARMRACE_RESULTS = f"{XGT}/armrace/results.json"
CACHE_PATH = f"{HERE}/wind_tunnel_cifar_cache.json"
POSCACHE_PATH = f"{HERE}/wind_tunnel_arm_positions.npz"

# ----------------------------------------------------------------- cost model
FT_BS = 128            # batch size of one training step  (= 1 unit)
FULL_STEPS = 2000      # a full pull's fine-tune length
FWD_PER_BWD = 3.0      # a forward-only pass costs ~1/3 of fwd+bwd per example
TEST_N = 10000
EVAL_PASSES = 3        # xarm_race.finetune_eval calls accuracy() 3x over the test set
B_DRAW = 200           # examples added per pull (mirrors the robot's B=200)


def fwd_units(n_examples: float) -> float:
    """Cost of forward-passing n examples, in training-step equivalents."""
    return n_examples / (FT_BS * FWD_PER_BWD)


EVAL_UNITS = fwd_units(TEST_N * EVAL_PASSES)          # 78.125
FULL_PULL_UNITS = FULL_STEPS + EVAL_UNITS             # 2078.125 units == 1 FTE


def fte(units: float) -> float:
    return units / FULL_PULL_UNITS


def units_of(fte_val: float) -> float:
    return fte_val * FULL_PULL_UNITS


# ---------------------------------------------------------------- the meter
class BudgetExhausted(RuntimeError):
    """Raised INSTEAD of doing the work when a charge would overrun the budget."""


@dataclass
class Charge:
    kind: str
    detail: str
    units: float


class Meter:
    """Single source of truth for what an allocator spent. Charge-before-work:
    a refused charge leaves no trace and no compute happens."""

    def __init__(self, budget_fte: float, name: str = ""):
        self.name = name
        self.budget_units = units_of(budget_fte)
        self.charges: List[Charge] = []

    # --- state
    @property
    def spent_units(self) -> float:
        return sum(c.units for c in self.charges)

    @property
    def spent_fte(self) -> float:
        return fte(self.spent_units)

    @property
    def remaining_units(self) -> float:
        return self.budget_units - self.spent_units

    @property
    def remaining_fte(self) -> float:
        return fte(self.remaining_units)

    @property
    def budget_fte(self) -> float:
        return fte(self.budget_units)

    @property
    def n_ops(self) -> int:
        return len(self.charges)

    # --- accounting
    def can_afford(self, u: float) -> bool:
        return self.spent_units + u <= self.budget_units + 1e-9

    def charge(self, u: float, kind: str, detail: str = "") -> None:
        if u <= 0:
            raise ValueError(f"non-positive charge {u} for {kind}")
        if not self.can_afford(u):
            raise BudgetExhausted(
                f"[{self.name}] {kind} needs {fte(u):.4f} FTE, "
                f"{self.remaining_fte:.4f} left of {self.budget_fte:.2f}")
        self.charges.append(Charge(kind, detail, u))

    def by_kind(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in self.charges:
            out[c.kind] = out.get(c.kind, 0.0) + fte(c.units)
        return out


# ------------------------------------------------------------------ decisions
@dataclass
class Decision:
    weights: Dict[str, float]
    notes: dict = field(default_factory=dict)

    @property
    def top(self) -> str:
        return max(self.weights, key=self.weights.get)


def value_of(d: Decision, truth: Dict[str, float]) -> float:
    return float(sum(w * truth[a] for a, w in d.weights.items()))


def regret_of(d: Decision, truth: Dict[str, float]) -> float:
    return float(max(truth.values()) - value_of(d, truth))


def onehot(arm: str) -> Decision:
    return Decision({arm: 1.0})


def _norm_mix(mix, arms: Sequence[str]) -> Dict[str, float]:
    if isinstance(mix, str):
        mix = {mix: 1.0}
    w = {a: float(v) for a, v in mix.items() if v > 1e-9}
    for a in w:
        if a not in arms:
            raise KeyError(f"unknown arm {a!r}")
    s = sum(w.values())
    if s <= 0:
        raise ValueError("empty mixture")
    return {a: v / s for a, v in sorted(w.items())}


def _mix_key(w: Dict[str, float]) -> str:
    return ",".join(f"{a}:{v:.4f}" for a, v in sorted(w.items()))


# ------------------------------------------------------------- the environment
class OracleToken:
    """Capability object. `env.truth(token)` works only for the exact token the
    runner handed to the oracle allocator, so no other allocator can peek."""


@dataclass
class PullObs:
    mix: Dict[str, float]
    seed: int
    steps: int
    y: float
    cached: bool = False
    extras: dict = field(default_factory=dict)


class WindTunnel:
    """The metered environment. Every op an allocator can perform lives here and
    every op charges. Allocators get nothing else."""

    def __init__(self, backend, budget_fte: float, name: str = "",
                 seed_pool: Optional[Sequence[int]] = None,
                 oracle_token: Optional[OracleToken] = None, seed_offset: int = 0):
        self.backend = backend
        self.meter = Meter(budget_fte, name)
        self.name = name
        self.seed_pool = list(seed_pool) if seed_pool else None
        self.seed_offset = int(seed_offset)
        self._seed_i = 0
        self._oracle_token = oracle_token
        self.wall_s = 0.0

    # --- introspection (free: no compute)
    @property
    def arms(self) -> List[str]:
        return list(self.backend.arms)

    @property
    def spent_fte(self) -> float:
        return self.meter.spent_fte

    @property
    def spent_units(self) -> float:
        return self.meter.spent_units

    @property
    def remaining_fte(self) -> float:
        return self.meter.remaining_fte

    @property
    def budget_fte(self) -> float:
        return self.meter.budget_fte

    def pull_cost_fte(self, steps: int = FULL_STEPS) -> float:
        return fte(steps + EVAL_UNITS)

    def next_seed(self) -> int:
        """Fresh training seed. Seeds are not compute, so they are free; but a
        finite pool (CIFAR: the 4 armrace rounds) means repeats return the same
        fine-tune -- and are still charged."""
        if self.seed_pool:      # finite pool (CIFAR): rotate the START per replicate
            s = self.seed_pool[(self._seed_i + self.seed_offset) % len(self.seed_pool)]
        else:
            s = self._seed_i + 10000 * self.seed_offset
        self._seed_i += 1
        return int(s)

    # --- metered operations
    def pull(self, mix, seed: Optional[int] = None, steps: int = FULL_STEPS) -> PullObs:
        """Fine-tune base + a B=200 draw from `mix` for `steps`, evaluate on the
        target slice. steps<FULL_STEPS = a cheap proxy run (still charged)."""
        w = _norm_mix(mix, self.arms)
        seed = self.next_seed() if seed is None else int(seed)
        self.meter.charge(steps + EVAL_UNITS, "pull", f"{_mix_key(w)}|s{seed}|n{steps}")
        t0 = time.time()
        y, cached, extras = self.backend.run_pull(w, seed, steps)
        self.wall_s += time.time() - t0
        return PullObs(w, seed, steps, float(y), cached, extras)

    def open_session(self, seed: Optional[int] = None) -> "OnlineSession":
        """Start an online (curriculum) run from the base checkpoint. Opening is
        free -- it is a state copy; every burst/probe/eval on it charges."""
        seed = self.next_seed() if seed is None else int(seed)
        return OnlineSession(self, self.backend.session_init(seed), seed)

    def reference(self, steps: int, seed: Optional[int] = None):
        """Train the reference model JEST-style scoring needs. NOT free."""
        seed = self.next_seed() if seed is None else int(seed)
        self.meter.charge(steps, "reference", f"s{seed}|n{steps}")
        t0 = time.time()
        ref = self.backend.make_reference(steps, seed)
        self.wall_s += time.time() - t0
        return ref

    def learnability(self, ref, n_per_arm: int = 512) -> Dict[str, float]:
        """Per-arm JEST score = mean(learner loss - reference loss). Costs two
        forward passes over the scored examples."""
        n_arms = len([a for a in self.arms if self.backend.arm_is_scorable(a)])
        self.meter.charge(2 * fwd_units(n_per_arm * max(n_arms, 1)), "score",
                          f"n{n_per_arm}x{n_arms}")
        t0 = time.time()
        out = self.backend.learnability(ref, n_per_arm)
        self.wall_s += time.time() - t0
        return out

    # --- ceiling only
    def truth(self, token) -> Dict[str, float]:
        if token is None or token is not self._oracle_token:
            raise PermissionError("truth() is oracle-only (capability token)")
        return dict(self.backend.true_values())

    def report(self) -> dict:
        return {"budget_fte": self.budget_fte, "spent_fte": self.spent_fte,
                "unspent_fte": self.remaining_fte, "n_ops": self.meter.n_ops,
                "spend_by_kind": {k: round(v, 4) for k, v in self.meter.by_kind().items()},
                "wall_s": round(self.wall_s, 1)}


class OnlineSession:
    """One long-lived learner that an online allocator steers arm-by-arm
    (Graves TPG / ODM). Its reward is the drop in held-out target-slice loss."""

    def __init__(self, env: WindTunnel, state, seed: int):
        self.env = env
        self.state = state
        self.seed = seed
        self._last_loss: Optional[float] = None
        self.history: List[dict] = []

    def loss(self, n: int = 1024) -> float:
        self.env.meter.charge(fwd_units(n), "loss_probe", f"n{n}")
        t0 = time.time()
        v = float(self.env.backend.session_loss(self.state, n))
        self.env.wall_s += time.time() - t0
        self._last_loss = v
        return v

    def burst(self, arm: str, steps: int, loss_n: int = 1024) -> dict:
        """Train `steps` on this arm's mixture, then re-probe the target loss.
        Returns learning progress = loss drop per 1000 steps."""
        if self._last_loss is None:
            self.loss(loss_n)
        before = self._last_loss
        self.env.meter.charge(steps, "burst", f"{arm}|n{steps}")
        t0 = time.time()
        self.state = self.env.backend.session_burst(self.state, arm, steps)
        self.env.wall_s += time.time() - t0
        after = self.loss(loss_n)
        rec = {"arm": arm, "steps": steps, "loss_before": before, "loss_after": after,
               "progress": (before - after) / steps * 1000.0}
        self.history.append(rec)
        return rec

    def evaluate(self) -> float:
        """Realized target-slice delta of the model this session actually built."""
        self.env.meter.charge(EVAL_UNITS, "eval", f"s{self.seed}")
        t0 = time.time()
        v = float(self.env.backend.session_eval(self.state))
        self.env.wall_s += time.time() - t0
        return v


# -------------------------------------------------------------------- backends
class MemoBackend:
    """Shared pull memo. A repeated (mixture, seed, steps) IS the same fine-tune,
    so it returns the same number and costs no wall-clock -- but `WindTunnel.pull`
    charges before it ever gets here, so the cache can never be farmed for budget.
    Subclasses implement `_compute_pull`."""

    cache: Dict[str, float]

    def __init__(self, replay_noise_sd: float = 0.0):
        self.cache = {}
        self.n_compute = 0
        self.n_cache_hit = 0
        self.compute_wall_s = 0.0
        # MEASURED (wind_tunnel_calibration.json): re-running the IDENTICAL pull
        # (same draw, same seed, same steps) on this box gives 3.70 / 2.85 / 3.35 pp
        # -- sd 0.43pp of pure GPU nondeterminism. So a cache hit is not really the
        # same number; set replay_noise_sd=0.0043 to model a repeat as the fresh
        # draw it actually is. Default 0 keeps pulls a deterministic function.
        self.replay_noise_sd = float(replay_noise_sd)
        self._hits: Dict[str, int] = {}

    @staticmethod
    def _key(w, seed, steps) -> str:
        return f"{_mix_key(w)}|s{int(seed)}|n{int(steps)}"

    def _persist(self) -> None:
        pass

    def _compute_pull(self, w: Dict[str, float], seed: int, steps: int) -> float:
        raise NotImplementedError

    def run_pull(self, w: Dict[str, float], seed: int, steps: int):
        key = self._key(w, seed, steps)
        if key in self.cache:
            self.n_cache_hit += 1
            y = self.cache[key]
            if self.replay_noise_sd > 0:
                i = self._hits.get(key, 0)
                self._hits[key] = i + 1
                y = y + _hrng("replay", key, i).normal(0, self.replay_noise_sd)
            return y, True, {}
        t0 = time.time()
        y = float(self._compute_pull(w, seed, steps))
        self.cache[key] = y
        self._persist()
        self.n_compute += 1
        self.compute_wall_s += time.time() - t0
        return y, False, {}


def _hrng(*parts) -> np.random.Generator:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return np.random.default_rng(int(h.hexdigest(), 16) % (2 ** 63))


@dataclass
class SyntheticSpec:
    values: Dict[str, float]                       # TRUE full-fidelity arm values
    sigma_eps: float = 0.004                       # idiosyncratic per-pull noise
    sigma_seed: float = 0.006                      # SHARED per-seed effect (constraint 4)
    proxy_bias: Optional[Dict[str, float]] = None  # added at zero fidelity (constraint 3)
    lp_reward: Optional[Dict[str, float]] = None   # loss-drop rate per arm (F4 knob)
    lp_noise: float = 0.02
    jest_score: Optional[Dict[str, float]] = None  # learnability signal
    jest_noise: float = 0.05
    seed: int = 0                                  # instance salt


class SyntheticBackend(MemoBackend):
    """Arms with known values, no GPU. The three knobs that make it honest:
    a shared seed effect, a fidelity-dependent proxy bias, and a learning-progress
    signal that need not agree with value."""

    def __init__(self, spec: SyntheticSpec):
        super().__init__()
        self.spec = spec
        self.arms = tuple(spec.values)
        vs = np.array(list(spec.values.values()))
        rank = {a: (v - vs.min()) / (np.ptp(vs) + 1e-12) for a, v in spec.values.items()}
        self._lp = spec.lp_reward or {a: 0.1 + 0.4 * r for a, r in rank.items()}
        self._jest = spec.jest_score or dict(rank)
        self._bias = spec.proxy_bias or {}

    # --- ground truth
    def true_values(self) -> Dict[str, float]:
        return dict(self.spec.values)

    def arm_is_scorable(self, arm: str) -> bool:
        return True

    # --- pulls
    def _v_eff(self, arm: str, steps: int) -> float:
        f = min(steps / FULL_STEPS, 1.0)
        return self.spec.values[arm] + (1.0 - f) * self._bias.get(arm, 0.0)

    def _compute_pull(self, w: Dict[str, float], seed: int, steps: int) -> float:
        sp = self.spec
        mu = sum(wa * self._v_eff(a, steps) for a, wa in w.items())
        b_seed = _hrng("seed", sp.seed, seed).normal(0, sp.sigma_seed)
        eps = _hrng("eps", sp.seed, _mix_key(w), seed, steps).normal(0, sp.sigma_eps)
        return mu + b_seed + eps

    # --- online sessions
    def session_init(self, seed: int):
        return {"seed": seed, "loss": 1.0, "steps": 0, "spent": {}}

    def session_burst(self, st, arm: str, steps: int):
        r = _hrng("lp", self.spec.seed, st["seed"], arm, st["steps"], steps)
        rate = self._lp[arm] + r.normal(0, self.spec.lp_noise)
        decay = 1.0 / (1.0 + st["steps"] / 4000.0)     # progress gets harder over time
        st = dict(st, loss=max(st["loss"] - rate * decay * steps / 1000.0, 0.02),
                  steps=st["steps"] + steps,
                  spent={**st["spent"], arm: st["spent"].get(arm, 0) + steps})
        self.n_compute += 1
        return st

    def session_loss(self, st, n: int) -> float:
        self.n_compute += 1
        return st["loss"] + _hrng("lnoise", self.spec.seed, st["seed"], st["steps"], n).normal(
            0, 0.002 * np.sqrt(1024 / max(n, 1)))

    def session_eval(self, st) -> float:
        self.n_compute += 1
        tot = sum(st["spent"].values())
        if tot == 0:
            return 0.0
        w = {a: s / tot for a, s in st["spent"].items()}
        y, _, _ = self.run_pull(w, st["seed"], FULL_STEPS)
        return y

    # --- JEST
    def make_reference(self, steps: int, seed: int):
        self.n_compute += 1
        return {"steps": steps, "seed": seed}

    def learnability(self, ref, n_per_arm: int) -> Dict[str, float]:
        self.n_compute += 1
        sd = self.spec.jest_noise * np.sqrt(500.0 / max(ref["steps"], 1)) * np.sqrt(
            512.0 / max(n_per_arm, 1))
        return {a: self._jest[a] + _hrng("jest", self.spec.seed, ref["seed"], a).normal(0, sd)
                for a in self.arms}


# ---- specs -------------------------------------------------------------------
# CIFAR-like: the encoding regime (loss sees the target); values ~ armrace means.
SPEC_CIFAR_LIKE = SyntheticSpec(
    values={"null": -0.005, "rare": 0.038, "random": 0.009, "easy": -0.004,
            "gradarm_a": 0.003, "gradarm_b": 0.009},
    sigma_eps=0.004, sigma_seed=0.002,
    proxy_bias={"easy": 0.02, "rare": -0.01},
    lp_reward={"null": 0.10, "rare": 0.45, "random": 0.22, "easy": 0.08,
               "gradarm_a": 0.15, "gradarm_b": 0.20},
    jest_score={"null": 0.0, "rare": 0.9, "random": 0.35, "easy": 0.05,
                "gradarm_a": 0.2, "gradarm_b": 0.3})

# Robot-like: parameterized from the MEASURED ledger (llm_borrow/
# decision_accuracy_diagnostic.json -- arm means in pp, within-arm noise 1.84pp),
# plus the two measured pathologies: proxies mis-rank, loss progress mis-ranks.
SPEC_ROBOT_LIKE = SyntheticSpec(
    values={"null": 0.0, "easy_band": 0.0207, "mid_band": 0.0363,
            "random": 0.0259, "tall_vessel": 0.0296},
    sigma_eps=0.0184, sigma_seed=0.0080,
    proxy_bias={"tall_vessel": 0.030, "mid_band": -0.015, "easy_band": 0.010},
    lp_reward={"null": 0.10, "easy_band": 0.40, "mid_band": 0.18,
               "random": 0.22, "tall_vessel": 0.50},
    jest_score={"null": 0.0, "easy_band": 0.3, "mid_band": 0.25,
                "random": 0.2, "tall_vessel": 0.85})


# --------------------------------------------------------------- CIFAR backend
class CifarBackend(MemoBackend):
    """Real fine-tunes in the xgradtest sandbox.

    Full pulls go through `xarm_race.finetune_eval` UNCHANGED (FT_STEPS monkeypatched
    for proxies), with armrace's exact draw convention
    `RandomState(crc32(f"{arm}_{seed}"))`, so the 24 completed armrace pulls import
    directly into the cache. Sessions / references / scoring use a GPU-resident fast
    path (identical math -- CIFAR has no augmentation -- different batch order).

    Ground truth = per-arm mean delta_rare over the 4 completed armrace rounds.
    """

    ARMS = ("null", "rare", "random", "easy", "gradarm_a", "gradarm_b")

    def __init__(self, cache_path: str = CACHE_PATH, burst_mix_frac: float = 0.5,
                 ref_pool_n: int = 2000, verbose: bool = True,
                 replay_noise_sd: float = 0.0):
        super().__init__(replay_noise_sd=replay_noise_sd)
        self.arms = self.ARMS
        self.cache_path = cache_path
        self.burst_mix_frac = burst_mix_frac
        self.ref_pool_n = ref_pool_n
        self.verbose = verbose
        self.cache = (json.load(open(cache_path)) if os.path.exists(cache_path) else {})
        self._ready = False

    # ---------------- setup
    def _setup(self):
        if self._ready:
            return
        import sys
        import torch
        sys.path.insert(0, XGT)
        import xarm_race as xr                      # noqa: E402
        from xgrad import (CKPTS, DEV, N_RARE, ROOT, cifar100,  # noqa: E402
                           make_model)
        self.torch, self.xr, self.DEV, self.N_RARE = torch, xr, DEV, N_RARE
        self.make_model = make_model
        self.tr, self.te = cifar100(True), cifar100(False)
        self.sp = json.load(open(f"{ROOT}/splits.json"))
        self.base = list(self.sp["base"])
        self.base_state = torch.load(f"{CKPTS}/step6000.pt", map_location=DEV)
        self.armrace = json.load(open(ARMRACE_RESULTS))
        self.base_ref = self.armrace["base_ref"]

        # arm universes (positions into the 8000-example sketch pool)
        self.pool, self.arm_positions = self._arm_positions()

        # held-out target slice: rare-class TRAIN examples beyond the sketch pool,
        # so no arm can ever draw them (leak-free reward probe for the online arm)
        y_tr = np.array(self.tr.targets)
        rest = np.array(self.sp["pool"])[len(self.pool):]
        self.target_slice = rest[y_tr[rest] < N_RARE][:2048]

        # GPU-resident tensors for the fast path
        mean = torch.tensor([0.5071, 0.4865, 0.4409], device=DEV).view(1, 3, 1, 1)
        std = torch.tensor([0.2673, 0.2564, 0.2762], device=DEV).view(1, 3, 1, 1)
        self._mean, self._std = mean, std
        self.Xtr = torch.as_tensor(self.tr.data, device=DEV).permute(0, 3, 1, 2).contiguous()
        self.Ytr = torch.as_tensor(np.array(self.tr.targets), device=DEV).long()
        self.Xte = torch.as_tensor(self.te.data, device=DEV).permute(0, 3, 1, 2).contiguous()
        self.Yte = torch.as_tensor(np.array(self.te.targets), device=DEV).long()
        self.base_idx_t = torch.as_tensor(np.array(self.base), device=DEV).long()
        self._rare_te = (self.Yte < N_RARE).nonzero().squeeze(1)

        self._import_armrace()
        self._ready = True
        if self.verbose:
            print(f"[wt] cifar backend ready | pool {len(self.pool)} | "
                  f"target-loss slice {len(self.target_slice)} (leak-free) | "
                  f"cache {len(self.cache)} entries", flush=True)

    def _arm_positions(self):
        """Reproduce armrace's arm universes (cached: the k-means is the slow part)."""
        from xgrad import LOGS, N_RARE
        meta = json.load(open(f"{LOGS}/meta.json"))
        pool = np.array(meta["pool"])
        labels = np.load(f"{LOGS}/pool_labels.npy")
        is_rare = labels < N_RARE
        if os.path.exists(POSCACHE_PATH):
            z = np.load(POSCACHE_PATH)
            ga, gb = z["ga"], z["gb"]
        else:
            raw = np.asarray(np.load(f"{LOGS}/cand_raw.npy", mmap_mode="r")[
                meta["ckpts"].index(6000)], dtype=np.float64)
            Uu = self.xr.unit(raw, ax=1)
            P10 = self.xr.whiten_basis(Uu, 10)
            Uw = Uu - (Uu @ P10.T) @ P10
            ga, gb, sep = self.xr.gradarm_clusters(self.xr.unit(Uw), seed=0)
            np.savez(POSCACHE_PATH, ga=ga, gb=gb, sep=sep)
        return pool, {
            "rare": np.where(is_rare)[0], "random": np.arange(len(pool)),
            "easy": np.where(~is_rare)[0],
            "gradarm_a": np.where(ga)[0], "gradarm_b": np.where(gb)[0]}

    def _import_armrace(self):
        """The 24 completed armrace pulls become free cache hits (identical
        function, identical draws, identical seeds)."""
        ref = self.base_ref["acc_rare"]
        n = 0
        for k, row in self.armrace.items():
            if k == "base_ref":
                continue
            key = self._key({row["arm"]: 1.0}, row["round"], FULL_STEPS)
            if key not in self.cache:
                self.cache[key] = float(row["acc_rare"] - ref)
                n += 1
        if n:
            self._save_cache()
        if self.verbose:
            print(f"[wt] imported {n} armrace pulls into the cache", flush=True)

    def _persist(self):
        tmp = self.cache_path + ".tmp"
        json.dump(self.cache, open(tmp, "w"))
        os.replace(tmp, self.cache_path)

    _save_cache = _persist

    # ---------------- ground truth
    def true_values(self) -> Dict[str, float]:
        self._setup()
        ref = self.base_ref["acc_rare"]
        out: Dict[str, List[float]] = {a: [] for a in self.arms}
        for k, row in self.armrace.items():
            if k != "base_ref":
                out[row["arm"]].append(row["acc_rare"] - ref)
        return {a: float(np.mean(v)) for a, v in out.items()}

    def true_value_se(self) -> Dict[str, float]:
        self._setup()
        ref = self.base_ref["acc_rare"]
        out: Dict[str, List[float]] = {a: [] for a in self.arms}
        for k, row in self.armrace.items():
            if k != "base_ref":
                out[row["arm"]].append(row["acc_rare"] - ref)
        return {a: float(np.std(v, ddof=1) / np.sqrt(len(v))) for a, v in out.items()}

    def arm_is_scorable(self, arm: str) -> bool:
        return arm != "null"          # null has no data to score (JEST cannot see it)

    # ---------------- draws
    def _draw(self, arm: str, seed: int, n: int) -> np.ndarray:
        """armrace's exact draw rule (so imported pulls are the same fine-tune)."""
        if arm == "null" or n <= 0:
            return np.array([], dtype=int)
        rng = np.random.RandomState(zlib.crc32(f"{arm}_{seed}".encode()) % (2 ** 31))
        pos = rng.choice(self.arm_positions[arm], size=int(n), replace=False)
        return np.array([int(self.pool[p]) for p in pos], dtype=int)

    def _mix_indices(self, w: Dict[str, float], seed: int) -> List[int]:
        sizes = {a: int(round(B_DRAW * wa)) for a, wa in w.items()}
        idx: List[int] = []
        for a, n in sizes.items():
            idx += self._draw(a, seed, n).tolist()
        return idx

    # ---------------- pulls (armrace-exact path)
    def _compute_pull(self, w: Dict[str, float], seed: int, steps: int) -> float:
        self._setup()
        t0 = time.time()
        train_idx = list(self.base) + self._mix_indices(w, seed)
        old = self.xr.FT_STEPS
        self.xr.FT_STEPS = int(steps)
        try:
            accs = self.xr.finetune_eval(self.base_state, self.tr, self.te,
                                         train_idx, 1000 + int(seed), self.sp)
        finally:
            self.xr.FT_STEPS = old
        y = float(accs["acc_rare"] - self.base_ref["acc_rare"])
        if self.verbose:
            print(f"[wt]   pull {self._key(w, seed, steps)}: drare {y:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return y

    def run_pull(self, w: Dict[str, float], seed: int, steps: int):
        self._setup()
        return super().run_pull(w, seed, steps)

    # ---------------- fast GPU-resident path (sessions / reference / scoring)
    def _norm(self, x_uint8):
        return (x_uint8.float() / 255.0 - self._mean) / self._std

    def _fresh_model(self, seed: int):
        torch = self.torch
        torch.manual_seed(seed)
        m = self.make_model()
        m.load_state_dict(self.base_state)
        return m

    def _train_fast(self, model, opt, idx_t, draw_t, steps: int, gen, mix_frac: float):
        torch = self.torch
        F = torch.nn.functional
        model.train()
        n_arm = 0 if draw_t is None or len(draw_t) == 0 else int(FT_BS * mix_frac)
        for _ in range(steps):
            b = torch.randint(0, len(idx_t), (FT_BS - n_arm,), generator=gen,
                              device=idx_t.device)
            ids = idx_t[b]
            if n_arm:
                d = torch.randint(0, len(draw_t), (n_arm,), generator=gen,
                                  device=draw_t.device)
                ids = torch.cat([ids, draw_t[d]])
            x = self._norm(self.Xtr[ids])
            loss = F.cross_entropy(model(x), self.Ytr[ids])
            opt.zero_grad(); loss.backward(); opt.step()
        return model

    def session_init(self, seed: int):
        self._setup()
        torch = self.torch
        m = self._fresh_model(1000 + seed)
        opt = torch.optim.AdamW(m.parameters(), lr=self.xr.FT_LR, weight_decay=5e-4)
        gen = torch.Generator(device=self.DEV); gen.manual_seed(1000 + seed)
        return {"model": m, "opt": opt, "gen": gen, "seed": seed, "spent": {}}

    def session_burst(self, st, arm: str, steps: int):
        t0 = time.time()
        draw = self._draw(arm, st["seed"], B_DRAW)
        draw_t = (self.torch.as_tensor(draw, device=self.DEV).long()
                  if len(draw) else None)
        self._train_fast(st["model"], st["opt"], self.base_idx_t, draw_t, steps,
                         st["gen"], self.burst_mix_frac)
        st["spent"][arm] = st["spent"].get(arm, 0) + steps
        self.n_compute += 1
        self.compute_wall_s += time.time() - t0
        return st

    def session_loss(self, st, n: int) -> float:
        torch = self.torch
        idx = torch.as_tensor(self.target_slice[:n], device=self.DEV).long()
        st["model"].eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(idx), 512):
                ids = idx[s:s + 512]
                out = st["model"](self._norm(self.Xtr[ids]))
                tot += torch.nn.functional.cross_entropy(
                    out, self.Ytr[ids], reduction="sum").item()
                cnt += len(ids)
        return tot / max(cnt, 1)

    def session_eval(self, st) -> float:
        torch = self.torch
        st["model"].eval()
        c = t = 0
        with torch.no_grad():
            for s in range(0, len(self._rare_te), 512):
                ids = self._rare_te[s:s + 512]
                p = st["model"](self._norm(self.Xte[ids])).argmax(1)
                c += (p == self.Yte[ids]).sum().item(); t += len(ids)
        return c / max(t, 1) - self.base_ref["acc_rare"]

    def make_reference(self, steps: int, seed: int):
        """JEST's reference model: base + a uniform pool sample (not free)."""
        self._setup()
        t0 = time.time()
        torch = self.torch
        m = self._fresh_model(7000 + seed)
        opt = torch.optim.AdamW(m.parameters(), lr=self.xr.FT_LR, weight_decay=5e-4)
        gen = torch.Generator(device=self.DEV); gen.manual_seed(7000 + seed)
        rng = np.random.RandomState(zlib.crc32(f"ref_{seed}".encode()) % (2 ** 31))
        pos = rng.choice(len(self.pool), size=self.ref_pool_n, replace=False)
        idx = np.concatenate([np.array(self.base), self.pool[pos]])
        idx_t = torch.as_tensor(idx, device=self.DEV).long()
        self._train_fast(m, opt, idx_t, None, steps, gen, 0.0)
        self.n_compute += 1
        self.compute_wall_s += time.time() - t0
        return {"model": m, "steps": steps, "seed": seed}

    def _mean_loss(self, model, ids_t) -> float:
        torch = self.torch
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for s in range(0, len(ids_t), 512):
                ids = ids_t[s:s + 512]
                out = model(self._norm(self.Xtr[ids]))
                tot += torch.nn.functional.cross_entropy(
                    out, self.Ytr[ids], reduction="sum").item()
        return tot / max(len(ids_t), 1)

    def learnability(self, ref, n_per_arm: int) -> Dict[str, float]:
        self._setup()
        torch = self.torch
        learner = self._fresh_model(0)
        out = {}
        for a in self.arms:
            if not self.arm_is_scorable(a):
                continue
            ids = self._draw(a, 9999, min(n_per_arm, len(self.arm_positions[a])))
            ids_t = torch.as_tensor(ids, device=self.DEV).long()
            out[a] = float(self._mean_loss(learner, ids_t)
                           - self._mean_loss(ref["model"], ids_t))
        return out


# ------------------------------------------------------------------ allocators
class Allocator:
    """THE INTERFACE. One method; the environment is the only way to spend.

        run(env: WindTunnel, rng: np.random.Generator) -> Decision
    """
    name: str = "allocator"

    def run(self, env: WindTunnel, rng: np.random.Generator) -> Decision:
        raise NotImplementedError


class SuccessiveHalving(Allocator):
    """(a) THE INCUMBENT. Paired-seed rounds over the survivors, eliminate the
    bottom 1/eta, repeat while a FULL paired round is affordable (a partial round
    would break the seed pairing that constraint 4 makes mandatory). Leftover
    budget replicates the finalists."""
    name = "successive_halving"

    def __init__(self, eta: float = 2.0, steps: int = FULL_STEPS):
        self.eta, self.steps = eta, steps

    def run(self, env, rng):
        arms = list(env.arms)
        cost = env.pull_cost_fte(self.steps)
        # A budget below one paired round over all arms cannot SEE every arm.
        # Drop the excess at RANDOM (never by list order) and say so, rather than
        # "eliminating" arms on zero observations.
        afford = int(np.floor(env.budget_fte / cost))
        dropped: List[str] = []
        if 0 < afford < len(arms):
            keep = list(rng.choice(arms, size=max(2, afford), replace=False))
            dropped = [a for a in arms if a not in keep]
            arms = [a for a in arms if a in keep]
        obs: Dict[str, List[float]] = {a: [] for a in arms}
        surv, prev = arms, arms
        n_rungs = max(1, int(np.ceil(np.log(len(arms)) / np.log(self.eta))))
        per_rung = env.budget_fte / n_rungs
        rounds = 0

        def paired_round(group):
            s = env.next_seed()
            for a in group:
                obs[a].append(env.pull(a, seed=s, steps=self.steps).y)

        for _ in range(n_rungs):
            reps = max(1, int(np.floor(per_rung / (len(surv) * cost))))
            for _ in range(reps):
                if env.remaining_fte < cost * len(surv) - 1e-9:
                    break
                paired_round(surv); rounds += 1
            if len(surv) <= 1:
                break
            prev = surv
            surv = sorted(surv, key=lambda a: -np.mean(obs[a] or [-1e9]))[
                :max(1, int(np.ceil(len(surv) / self.eta)))]
        # leftover budget goes into the last live COMPARISON (>=2 arms), never
        # into re-pulling an uncontested champion (which buys no information)
        final = surv if len(surv) > 1 else sorted(
            prev, key=lambda a: -np.mean(obs[a] or [-1e9]))[:2]
        while env.remaining_fte >= cost * len(final) - 1e-9:
            paired_round(final); rounds += 1
        seen = {a: v for a, v in obs.items() if v}
        if not seen:
            return Decision({str(rng.choice(arms)): 1.0}, {"reason": "no budget"})
        pool = {a: seen[a] for a in final if a in seen} or seen
        best = max(pool, key=lambda a: float(np.mean(pool[a])))
        return Decision({best: 1.0}, {"rungs": n_rungs, "paired_rounds": rounds,
                                      "finalists": final, "never_seen": dropped,
                                      "pulls_per_arm": {a: len(v) for a, v in seen.items()},
                                      "arm_means_pp": {a: round(float(np.mean(v)) * 100, 3)
                                                       for a, v in seen.items()}})


class LearningProgress(Allocator):
    """(b) ODM / Graves borrow: ONE online learner, arms chosen by EXP3 or UCB1,
    reward = learning progress = rate of decrease of held-out target-slice loss.
    Every burst and every loss probe is charged. NOTE the deliberate asymmetry
    this method buys itself: during its burst an arm's data occupies
    `burst_mix_frac` of each batch, i.e. it is measured under a mixture no full
    pull would ever use -- that mismatch is exactly what F4 warns about."""

    def __init__(self, algo: str = "exp3", chunk_steps: int = 100, loss_n: int = 1024,
                 gamma: float = 0.10, lr: float = 0.4, c_ucb: float = 0.6):
        self.algo, self.chunk, self.loss_n = algo, chunk_steps, loss_n
        self.gamma, self.lr, self.c = gamma, lr, c_ucb
        self.name = f"learning_progress_{algo}"

    def run(self, env, rng):
        arms = env.arms
        K = len(arms)
        w = np.zeros(K)                          # EXP3 log-weights
        n = np.zeros(K)
        mean_r = np.zeros(K)
        raw: List[float] = []
        sess = env.open_session()
        step_cost = fte(self.chunk + fwd_units(self.loss_n))
        t = 0
        while env.remaining_fte >= step_cost:
            if self.algo == "exp3":
                p = np.exp(w - w.max()); p /= p.sum()
                p = (1 - self.gamma) * p + self.gamma / K
                k = int(rng.choice(K, p=p))
            else:
                k = int(np.argmin(n)) if n.min() == 0 else int(np.argmax(
                    mean_r + self.c * np.sqrt(2 * np.log(max(t, 2)) / n)))
                p = None
            try:
                rec = sess.burst(arms[k], self.chunk, self.loss_n)
            except BudgetExhausted:
                break
            g = rec["progress"]
            raw.append(g)
            lo, hi = min(raw), max(raw)
            r01 = 0.5 if hi - lo < 1e-12 else (g - lo) / (hi - lo)
            n[k] += 1
            mean_r[k] += (r01 - mean_r[k]) / n[k]
            if self.algo == "exp3":
                w[k] += self.lr * r01 / max(p[k], 1e-6) / K
            t += 1
        if n.sum() == 0:
            return Decision({str(rng.choice(arms)): 1.0}, {"reason": "no budget"})
        score = w if self.algo == "exp3" else mean_r
        score = np.where(n > 0, score, -np.inf)
        best = arms[int(np.argmax(score))]
        return Decision({best: 1.0}, {"n_bursts": int(n.sum()),
                                      "pulls_per_arm": dict(zip(arms, n.astype(int).tolist())),
                                      "mean_progress_reward": dict(zip(arms, mean_r.round(4).tolist()))})


class RegMix(Allocator):
    """(c) RegMix: k cheap proxy runs on random Dirichlet mixtures -> ridge
    regression of outcome on the mixture -> argmax mixture over the simplex.
    With a linear surrogate the argmax is always a vertex, so RegMix-over-arms
    reduces to regression-based best-arm ID -- but it pools every probe into
    every arm's estimate, which is its real efficiency claim."""
    name = "regmix"

    def __init__(self, proxy_steps: int = 250, n_proxies: Optional[int] = None,
                 max_proxies: int = 24, alpha: float = 1.0, ridge: float = 1e-3,
                 n_simplex: int = 20000, budget_frac: float = 0.95):
        self.proxy_steps, self.n_proxies, self.max_proxies = proxy_steps, n_proxies, max_proxies
        self.alpha, self.ridge, self.n_simplex, self.budget_frac = alpha, ridge, n_simplex, budget_frac

    def run(self, env, rng):
        arms = env.arms
        K = len(arms)
        cost = env.pull_cost_fte(self.proxy_steps)
        k = self.n_proxies or int(min(self.max_proxies,
                                      np.floor(self.budget_frac * env.remaining_fte / cost)))
        k = max(k, 0)
        X, Y = [], []
        for _ in range(k):
            w = rng.dirichlet(np.full(K, self.alpha))
            mix = {a: float(v) for a, v in zip(arms, w) if v > 1e-6}
            try:
                obs = env.pull(mix, steps=self.proxy_steps)
            except BudgetExhausted:
                break
            X.append([mix.get(a, 0.0) for a in arms])
            Y.append(obs.y)
        if len(X) < 2:
            return Decision({str(rng.choice(arms)): 1.0}, {"reason": "no budget"})
        A = np.array(X); b = np.array(Y)
        beta = np.linalg.solve(A.T @ A + self.ridge * np.eye(K), A.T @ b)
        cand = np.vstack([np.eye(K), rng.dirichlet(np.full(K, self.alpha), self.n_simplex)])
        best = cand[int(np.argmax(cand @ beta))]
        wts = {a: float(v) for a, v in zip(arms, best) if v > 1e-6}
        s = sum(wts.values())
        return Decision({a: v / s for a, v in wts.items()},
                        {"n_proxies": len(X), "proxy_steps": self.proxy_steps,
                         "coef": dict(zip(arms, beta.round(5).tolist()))})


class Jest(Allocator):
    """(d) JEST learnability: score = learner loss - reference loss, NO fine-tuning
    of any arm. The reference model is charged (it is a training run), the two
    scoring passes are charged. Cannot score an arm with no data (null)."""
    name = "jest"

    def __init__(self, ref_steps: int = 500, n_per_arm: int = 512, min_ref_steps: int = 100):
        self.ref_steps, self.n_per_arm, self.min_ref_steps = ref_steps, n_per_arm, min_ref_steps

    def _score(self, env, rng):
        steps = int(min(self.ref_steps, np.floor(units_of(env.remaining_fte) * 0.8)))
        if steps < self.min_ref_steps:
            return None, None
        ref = env.reference(steps=steps)
        try:
            return env.learnability(ref, n_per_arm=self.n_per_arm), steps
        except BudgetExhausted:
            return None, steps

    def run(self, env, rng):
        sc, steps = self._score(env, rng)
        if not sc:
            return Decision({str(rng.choice(env.arms)): 1.0}, {"reason": "no budget"})
        best = max(sc, key=sc.get)
        return Decision({best: 1.0}, {"scores": {k: round(v, 4) for k, v in sc.items()},
                                      "ref_steps": steps,
                                      "unspent_fte": round(env.remaining_fte, 3)})


class JestThenConfirm(Jest):
    """(d+) The obvious practical hybrid, and the one the wind tunnel exists to
    test: screen with a near-free learnability score, then spend the (large)
    leftover budget on real paired pulls of the top-2."""
    name = "jest_then_confirm"

    def __init__(self, top_k: int = 2, **kw):
        super().__init__(**kw)
        self.top_k = top_k

    def run(self, env, rng):
        sc, steps = self._score(env, rng)
        if not sc:
            return Decision({str(rng.choice(env.arms)): 1.0}, {"reason": "no budget"})
        short = sorted(sc, key=lambda a: -sc[a])[:self.top_k]
        obs: Dict[str, List[float]] = {a: [] for a in short}
        cost = env.pull_cost_fte()
        while env.remaining_fte >= cost * len(short) - 1e-9:
            s = env.next_seed()
            for a in short:
                obs[a].append(env.pull(a, seed=s).y)
        seen = {a: v for a, v in obs.items() if v}
        best = max(seen, key=lambda a: float(np.mean(seen[a]))) if seen else short[0]
        return Decision({best: 1.0}, {"scores": {k: round(v, 4) for k, v in sc.items()},
                                      "shortlist": short,
                                      "confirm_reps": max((len(v) for v in obs.values()), default=0),
                                      "screened_out": [a for a in sc if a not in short]})


class RandomAlloc(Allocator):
    """(e) FLOOR: uniformly random arm per pull, pick the empirical argmax."""
    name = "random"

    def __init__(self, steps: int = FULL_STEPS):
        self.steps = steps

    def run(self, env, rng):
        obs: Dict[str, List[float]] = {}
        cost = env.pull_cost_fte(self.steps)
        while env.remaining_fte >= cost - 1e-9:
            a = str(rng.choice(env.arms))
            obs.setdefault(a, []).append(env.pull(a, steps=self.steps).y)
        if not obs:
            return Decision({str(rng.choice(env.arms)): 1.0}, {"reason": "no budget"})
        best = max(obs, key=lambda a: float(np.mean(obs[a])))
        return Decision({best: 1.0}, {"n_pulls": sum(len(v) for v in obs.values()),
                                      "arms_seen": sorted(obs)})


class Oracle(Allocator):
    """(e) CEILING: knows the true values, spends nothing. Not implementable --
    it exists to normalize regret."""
    name = "oracle"

    def __init__(self, token: OracleToken):
        self.token = token

    def run(self, env, rng):
        truth = env.truth(self.token)
        return Decision({max(truth, key=truth.get): 1.0}, {"spend": 0.0})


ALLOCATOR_FACTORIES: Dict[str, Callable[[OracleToken], Allocator]] = {
    "successive_halving": lambda tok: SuccessiveHalving(),
    "learning_progress_exp3": lambda tok: LearningProgress(algo="exp3"),
    "learning_progress_ucb": lambda tok: LearningProgress(algo="ucb"),
    "regmix": lambda tok: RegMix(),
    "jest": lambda tok: Jest(),
    "jest_then_confirm": lambda tok: JestThenConfirm(),
    "random": lambda tok: RandomAlloc(),
    "oracle": lambda tok: Oracle(tok),
}


# ---------------------------------------------------------------------- runner
def run_race(backend, budget_fte: float, reps: int = 1, token: Optional[OracleToken] = None,
             seed_pool: Optional[Sequence[int]] = None, names: Optional[Sequence[str]] = None,
             base_seed: int = 0, verbose: bool = True) -> dict:
    """Race every allocator at the SAME budget on the SAME backend (shared cache,
    so repeated (arm, seed, steps) cost budget but not wall-clock)."""
    token = token or OracleToken()
    truth = backend.true_values()
    best_v = max(truth.values())
    out: dict = {}
    for name in (names or list(ALLOCATOR_FACTORIES)):
        alloc = ALLOCATOR_FACTORIES[name](token)
        regs, vals, spends, unspent, walls, kinds, tops, notes = [], [], [], [], [], {}, [], []
        for rep in range(reps):
            env = WindTunnel(backend, budget_fte, name=name, seed_pool=seed_pool,
                             oracle_token=token, seed_offset=rep)
            d = alloc.run(env, np.random.default_rng(
                base_seed + 1000 * rep + zlib.crc32(name.encode()) % 997))
            regs.append(regret_of(d, truth)); vals.append(value_of(d, truth))
            spends.append(env.spent_fte); unspent.append(env.remaining_fte)
            walls.append(env.wall_s); tops.append(d.top); notes.append(d.notes)
            for k, v in env.meter.by_kind().items():
                kinds[k] = kinds.get(k, 0.0) + v / reps
        out[name] = {
            "budget_fte": budget_fte, "reps": reps,
            "mean_regret": float(np.mean(regs)), "se_regret": float(np.std(regs, ddof=1) / np.sqrt(reps)) if reps > 1 else 0.0,
            "mean_value": float(np.mean(vals)),
            "p_correct": float(np.mean([t == max(truth, key=truth.get) for t in tops])),
            "mean_spent_fte": float(np.mean(spends)), "mean_unspent_fte": float(np.mean(unspent)),
            "spend_by_kind": {k: round(v, 4) for k, v in kinds.items()},
            "mean_wall_s": float(np.mean(walls)),
            "picks": tops, "example_notes": notes[0] if notes else {},
        }
    denom = out.get("random", {}).get("mean_regret", 0.0)
    for name, r in out.items():
        r["norm_regret"] = float(r["mean_regret"] / denom) if denom > 1e-12 else 0.0
    if verbose:
        _print_table(out, truth, best_v)
    return out


def _print_table(out: dict, truth: dict, best_v: float) -> None:
    print(f"\ntrue values (pp): " + "  ".join(f"{a}={v*100:+.2f}" for a, v in sorted(truth.items())))
    print(f"best arm: {max(truth, key=truth.get)}  ({best_v*100:+.2f}pp)\n")
    print(f"{'allocator':24s} {'regret_pp':>9s} {'norm':>6s} {'P(correct)':>10s} "
          f"{'spent':>6s} {'unspent':>8s} {'wall_s':>7s}  spend_by_kind")
    for name, r in sorted(out.items(), key=lambda kv: kv[1]["mean_regret"]):
        print(f"{name:24s} {r['mean_regret']*100:9.3f} {r['norm_regret']:6.2f} "
              f"{r['p_correct']:10.2f} {r['mean_spent_fte']:6.2f} {r['mean_unspent_fte']:8.2f} "
              f"{r['mean_wall_s']:7.1f}  {r['spend_by_kind']}")


# ------------------------------------------------------------------- diagnostics
def check_mixture_linearity(backend, seed: int = 7) -> dict:
    """The decision space assumes V(mixture) = sum_a w_a v_a. Test it with ONE
    real 50/50 pull (charged to nobody -- this is harness validation, not an
    allocator's spend) against the two single-arm pulls."""
    truth = backend.true_values()
    y, _, _ = backend.run_pull({"rare": 0.5, "easy": 0.5}, seed, FULL_STEPS)
    pred = 0.5 * truth["rare"] + 0.5 * truth["easy"]
    return {"mix": "rare0.5+easy0.5", "seed": seed, "observed_pp": round(y * 100, 3),
            "linear_prediction_pp": round(pred * 100, 3),
            "residual_pp": round((y - pred) * 100, 3)}


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def check_proxy_fidelity(backend, steps=(250, 500), seeds=(0, 1)) -> dict:
    """Constraint 3, measured HERE: do cheap short-training proxies rank the arms
    the way full pulls do? Unmetered -- this is harness calibration, not a
    method's spend. Every allocator that leans on proxies (RegMix, the burst
    allocators) inherits whatever this says."""
    truth = backend.true_values()
    arms = list(backend.arms)
    tv = [truth[a] for a in arms]
    out = {"truth_pp": {a: round(truth[a] * 100, 3) for a in arms},
           "truth_argmax": max(truth, key=truth.get)}
    for s in steps:
        ys = {a: float(np.mean([backend.run_pull({a: 1.0}, sd, s)[0] for sd in seeds]))
              for a in arms}
        out[f"steps_{s}"] = {
            "values_pp": {a: round(v * 100, 3) for a, v in ys.items()},
            "argmax": max(ys, key=ys.get),
            "spearman_vs_full": round(_spearman([ys[a] for a in arms], tv), 3),
            "picks_the_right_arm": max(ys, key=ys.get) == max(truth, key=truth.get)}
    return out


def verify_cache_import(backend, arm: str = "rare", seed: int = 0) -> dict:
    """Re-run one imported armrace pull from scratch and compare, to prove the
    imported cache entries are the same function we would compute ourselves."""
    key = backend._key({arm: 1.0}, seed, FULL_STEPS)
    imported = backend.cache.pop(key, None)
    y, _, _ = backend.run_pull({arm: 1.0}, seed, FULL_STEPS)
    return {"pull": key, "imported_pp": round((imported or float("nan")) * 100, 3),
            "recomputed_pp": round(y * 100, 3),
            "abs_diff_pp": round(abs((imported or 0) - y) * 100, 3)}


# -------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="synthetic",
                    choices=["synthetic", "synthetic_robot", "cifar"])
    ap.add_argument("--budget", type=float, default=6.0, help="FTE per allocator")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--allocators", default=None, help="comma list (default: all)")
    ap.add_argument("--seed-pool", default=None, help="e.g. 0,1,2,3 (cifar cache hits)")
    ap.add_argument("--sweep", default=None,
                    help="comma list of budgets (FTE) to sweep instead of one race")
    ap.add_argument("--check-linearity", action="store_true")
    ap.add_argument("--check-proxy-fidelity", action="store_true")
    ap.add_argument("--verify-import", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    names = a.allocators.split(",") if a.allocators else None
    pool = [int(x) for x in a.seed_pool.split(",")] if a.seed_pool else None
    if a.mode == "cifar":
        backend = CifarBackend()
        pool = pool or [0, 1, 2, 3]
        se = backend.true_value_se()
        print("[wt] ground truth = armrace arm means (delta rare-class acc), "
              "SE per arm: " + "  ".join(f"{k}={v*100:.2f}pp" for k, v in se.items()))
    else:
        backend = SyntheticBackend(SPEC_CIFAR_LIKE if a.mode == "synthetic"
                                   else SPEC_ROBOT_LIKE)
    t0 = time.time()
    if a.sweep:
        budgets = [float(x) for x in a.sweep.split(",")]
        sweep = {}
        for b in budgets:
            sweep[b] = run_race(backend, b, reps=a.reps, seed_pool=pool, names=names,
                                verbose=False)
        allocs = list(sweep[budgets[0]])
        print(f"\nNORMALIZED REGRET vs budget (1.00 = random floor, 0 = oracle), "
              f"{a.reps} reps, mode={a.mode}")
        print(f"{'allocator':24s} " + " ".join(f"{b:>7.0f}" for b in budgets))
        for al in sorted(allocs, key=lambda x: sweep[budgets[-1]][x]["norm_regret"]):
            print(f"{al:24s} " + " ".join(
                f"{sweep[b][al]['norm_regret']:7.2f}" for b in budgets))
        print(f"\nP(picks the truly best arm) vs budget")
        print(f"{'allocator':24s} " + " ".join(f"{b:>7.0f}" for b in budgets))
        for al in sorted(allocs, key=lambda x: -sweep[budgets[-1]][x]["p_correct"]):
            print(f"{al:24s} " + " ".join(
                f"{sweep[b][al]['p_correct']:7.2f}" for b in budgets))
        out = {"mode": a.mode, "sweep_budgets": budgets, "reps": a.reps,
               "truth": backend.true_values(), "sweep": sweep,
               "elapsed_s": round(time.time() - t0, 1)}
        path = a.out or f"{HERE}/wind_tunnel_{a.mode}_sweep.json"
        json.dump(out, open(path, "w"), indent=1, default=float)
        print(f"\n[wt] wrote {path}  ({out['elapsed_s']}s)")
        return
    res = run_race(backend, a.budget, reps=a.reps, seed_pool=pool, names=names)
    extra = {}
    if a.verify_import and a.mode == "cifar":
        extra["cache_import_check"] = verify_cache_import(backend)
        print("\n[wt] cache-import check:", extra["cache_import_check"])
    if a.check_linearity and a.mode == "cifar":
        extra["mixture_linearity"] = check_mixture_linearity(backend)
        print("[wt] mixture-linearity check:", extra["mixture_linearity"])
    if a.check_proxy_fidelity:
        extra["proxy_fidelity"] = check_proxy_fidelity(backend)
        print("[wt] proxy-fidelity check:", json.dumps(extra["proxy_fidelity"], indent=1))
    if a.mode == "cifar":
        extra["cifar_compute"] = {"n_backend_compute_calls": backend.n_compute,
                                  "n_pull_cache_hits": backend.n_cache_hit,
                                  "compute_wall_s": round(backend.compute_wall_s, 1),
                                  "note": ("compute calls counts every real GPU op "
                                           "(pull, burst, reference), not just pulls")}
        print("[wt] cifar compute:", extra["cifar_compute"])
    out = {"mode": a.mode, "budget_fte": a.budget, "reps": a.reps,
           "truth": backend.true_values(), "results": res, "diagnostics": extra,
           "cost_model": {"unit": "one bs=128 train step", "eval_units": EVAL_UNITS,
                          "full_pull_units": FULL_PULL_UNITS},
           "elapsed_s": round(time.time() - t0, 1)}
    path = a.out or f"{HERE}/wind_tunnel_{a.mode}.json"
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"\n[wt] wrote {path}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
