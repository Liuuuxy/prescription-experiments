"""PART 2 — the F4 wind tunnel: DECOUPLE the cheap signal from the outcome and map
where each borrowed allocator stops beating random.

Our measured constraint 1 (F4): training-loss effects do NOT transfer to closed-loop
success. Measured evidence on the robot (reward_signal_test.json): the held-out-loss
reward would have picked `tall_vessel` in ALL THREE rounds while `mid_band`/`random`
actually won; Spearman(loss reward, realized delta) = -0.06 (non-null pulls) .. +0.25
(all pulls), ns. So on our stack the cheap signal's rank correlation with outcome is
indistinguishable from ZERO.

WHAT IS DECOUPLED. Only the CHEAP-SIGNAL channel (held-out loss progress, JEST
learnability). The OUTCOME channel (a pull's realized target-slice delta) stays
faithful -- that is our real situation: rollout reward is expensive but true, loss
reward is cheap and may not transfer. Allocators that read only outcomes
(successive_halving, random, regmix) must therefore be EXACTLY invariant; that
invariance is checked, not assumed (a leak would void every Part 2 number).

TWO AXES (both required by the task):
  c    correlation between cheap signal and true arm value. The signal is
       s = c*z_truth + sqrt(1-c^2)*z_orth with z_orth a fixed direction orthogonal
       to both z_truth and 1, so Pearson(s, truth) = c EXACTLY, and c=0 is a signal
       that measures something real but orthogonal to what we are scored on.
  SNR  spread(signal across arms) / sd(observation noise on the signal). The measured
       robot operating point is SNR = 0.356 (arm-mean spread 0.654pp vs per-pull noise
       1.837pp, decision_accuracy_diagnostic.json).

REAL-GPU MIRROR (CifarF4Backend): no synthetic knobs.
  A  label noise rate rho on everything the cheap signal looks at (the held-out probe
     slice AND the examples JEST scores). Truth = clean rare-class test accuracy.
  B  off-target probe fraction alpha: alpha of the held-out probe slice is drawn from
     the WELL-LEARNED classes instead of the target classes. The cheap signal then
     measures a real quantity that is not the one we are scored on -- the exact shape
     of F4. Only the loss-progress channel has a probe slice, so B hits
     learning_progress; A hits both.
  Neither mechanism touches the training data or the evaluation: we corrupt the REWARD
  CHANNEL, never the data.

RUN
  python f4_transfer.py synth  --reps 500
  CUDA_VISIBLE_DEVICES=1 python f4_transfer.py cifar --reps 5 --budget 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from race_lib import atomic_json, paired_vs, run_cell, slim  # noqa: E402
from wind_tunnel import (ALLOCATOR_FACTORIES, HERE, SPEC_CIFAR_LIKE,  # noqa: E402
                         SPEC_ROBOT_LIKE, CifarBackend, JestThenConfirm,
                         SyntheticBackend, SyntheticSpec)

# ------------------------------------------------------------------ signal design
LP_MEAN, LP_SPREAD = 0.28, 0.15      # loss-drop rate per 1000 steps
JEST_MEAN, JEST_SPREAD = 0.50, 0.35


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def signal_vector(arms: List[str], values: Dict[str, float], c: float,
                  orth_seed: int = 20260814) -> np.ndarray:
    """s with Pearson(s, truth) == c exactly, unit-variance, mean 0."""
    zt = _z([values[a] for a in arms])
    rng = np.random.default_rng(orth_seed)
    v = rng.normal(size=len(arms))
    v = v - v.mean()
    v = v - (v @ zt) / (zt @ zt) * zt          # orthogonal to truth AND to 1
    zo = _z(v)
    s = c * zt + np.sqrt(max(0.0, 1 - c * c)) * zo
    return _z(s) if s.std() > 1e-9 else s


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def decoupled_spec(base: SyntheticSpec, c: float, snr: float,
                   spec_seed: int = 0) -> SyntheticSpec:
    """Same arms, same true values, same outcome noise; ONLY the cheap-signal
    channel is rewritten to have correlation c with truth and the given SNR."""
    arms = list(base.values)
    s = signal_vector(arms, base.values, c)
    lp = {a: LP_MEAN + LP_SPREAD * si for a, si in zip(arms, s)}
    je = {a: JEST_MEAN + JEST_SPREAD * si for a, si in zip(arms, s)}
    lp_spread = max(lp.values()) - min(lp.values())
    je_spread = max(je.values()) - min(je.values())
    return SyntheticSpec(
        values=dict(base.values), sigma_eps=base.sigma_eps, sigma_seed=base.sigma_seed,
        proxy_bias=dict(base.proxy_bias or {}),
        lp_reward=lp, lp_noise=float(lp_spread / max(snr, 1e-9)),
        jest_score=je, jest_noise=float(je_spread / max(snr, 1e-9)),
        seed=spec_seed)


def signal_diagnostics(base: SyntheticSpec, c: float) -> dict:
    arms = list(base.values)
    s = signal_vector(arms, base.values, c)
    tv = [base.values[a] for a in arms]
    return {"pearson_signal_truth": round(float(np.corrcoef(s, tv)[0, 1]), 4),
            "spearman_signal_truth": round(spearman(s, tv), 4),
            "signal_argmax": arms[int(np.argmax(s))],
            "true_argmax": max(base.values, key=base.values.get),
            "signal": {a: round(float(x), 3) for a, x in zip(arms, s)}}


# --------------------------------------------------- extra allocators (mine, labelled)
def _jtc(k: int):
    def f(tok):
        a = JestThenConfirm(top_k=k)
        a.name = f"jest_then_confirm_k{k}"
        return a
    return f


EXTRA_FACTORIES = {"jest_then_confirm_k3": _jtc(3), "jest_then_confirm_k4": _jtc(4)}
ALL_FACTORIES = {**ALLOCATOR_FACTORIES, **EXTRA_FACTORIES}

LOSS_BASED = ("learning_progress_exp3", "learning_progress_ucb", "jest",
              "jest_then_confirm", "jest_then_confirm_k3", "jest_then_confirm_k4")
OUTCOME_BASED = ("successive_halving", "random", "regmix")


# ---------------------------------------------------------------- synthetic driver
def run_synth(a):
    base = SPEC_ROBOT_LIKE if a.spec == "robot" else SPEC_CIFAR_LIKE
    cs = [float(x) for x in a.cs.split(",")]
    snrs = [float(x) for x in a.snrs.split(",")]
    budgets = [float(x) for x in a.budgets.split(",")]
    allocs = a.allocators.split(",")
    out_path = a.out or f"{HERE}/f4_synth_{a.spec}.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out.setdefault("spec", a.spec)
    out.setdefault("reps", a.reps)
    out.setdefault("truth", dict(base.values))
    out.setdefault("cells", {})
    out["signal_design"] = {str(c): signal_diagnostics(base, c) for c in cs}
    out["measured_robot_anchor"] = {
        "spec_lp_reward_spearman_vs_truth": spearman(
            [SPEC_ROBOT_LIKE.lp_reward[x] for x in SPEC_ROBOT_LIKE.values],
            [SPEC_ROBOT_LIKE.values[x] for x in SPEC_ROBOT_LIKE.values]),
        "spec_jest_spearman_vs_truth": spearman(
            [SPEC_ROBOT_LIKE.jest_score[x] for x in SPEC_ROBOT_LIKE.values],
            [SPEC_ROBOT_LIKE.values[x] for x in SPEC_ROBOT_LIKE.values]),
        "measured_loss_reward_spearman": [-0.0595, 0.2508],
        "measured_outcome_snr": 0.654 / 1.837}
    n_orth, reps_per = a.n_orth, a.reps // a.n_orth
    out["n_orth_directions"] = n_orth
    out["reps_per_direction"] = reps_per
    t0 = time.time()
    for b in budgets:
        for snr in snrs:
            for c in cs:
                # A cell is POOLED over n_orth independent orthogonal nuisance
                # directions, so the map is not an artefact of one arbitrary way of
                # being wrong. Each direction is its own instance (own spec seed,
                # own allocator rng stream).
                pooled: Dict[str, dict] = {}
                for oi in range(n_orth):
                    spec = decoupled_spec(base, c, snr, spec_seed=oi)
                    arms = list(base.values)
                    s = signal_vector(arms, base.values, c, orth_seed=20260814 + 7919 * oi)
                    spec.lp_reward = {x: LP_MEAN + LP_SPREAD * si for x, si in zip(arms, s)}
                    spec.jest_score = {x: JEST_MEAN + JEST_SPREAD * si for x, si in zip(arms, s)}
                    backend = SyntheticBackend(spec)
                    truth = backend.true_values()
                    for name in allocs:
                        cell = run_cell(backend, name, b, reps_per, truth=truth,
                                        base_seed=100000 * oi, factory=ALL_FACTORIES[name])
                        p = pooled.setdefault(name, {"regrets": [], "correct": [],
                                                     "spent": [], "picks": {}})
                        p["regrets"] += cell["regrets"]
                        p["correct"] += [r["correct"] for r in cell["recs"]]
                        p["spent"] += [r["spent_fte"] for r in cell["recs"]]
                        for k2, v2 in cell["pick_counts"].items():
                            p["picks"][k2] = p["picks"].get(k2, 0) + v2
                for name, p in pooled.items():
                    reg = np.array(p["regrets"], float)
                    cor = np.array(p["correct"], float)
                    out["cells"][f"{b}|{snr}|{c}|{name}"] = {
                        "allocator": name, "budget_fte": b, "snr": snr, "c": c,
                        "reps": len(reg), "mean_regret": float(reg.mean()),
                        "sd_regret": float(reg.std(ddof=1)),
                        "se_regret": float(reg.std(ddof=1) / np.sqrt(len(reg))),
                        "p_correct": float(cor.mean()),
                        "mean_spent_fte": float(np.mean(p["spent"])),
                        "pick_counts": p["picks"], "regrets": reg.tolist(),
                        "best_value": max(base.values.values())}
                print(f"[f4] b={b} snr={snr} c={c:+.2f}  " + "  ".join(
                    f"{n[:14]}={out['cells'][f'{b}|{snr}|{c}|{n}']['mean_regret']*100:.2f}"
                    for n in allocs), flush=True)
                atomic_json(out_path, out)
    atomic_json(out_path, out)
    print(f"[f4] wrote {out_path} ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------- real-GPU F4 backend
class CifarF4Backend(CifarBackend):
    """CIFAR backend whose CHEAP-SIGNAL channel is corrupted, outcome channel intact.

    probe_label_noise (rho): fraction of labels randomized in everything the cheap
        signal reads (held-out probe slice + the examples JEST scores).
    probe_offtarget (alpha): fraction of the held-out probe slice drawn from the
        well-learned (non-rare) classes instead of the target (rare) classes.
    Pulls, session_eval and true_values are inherited UNCHANGED, so the pull cache is
    shared with the uncorrupted backend and the ground truth is identical."""

    def __init__(self, probe_label_noise: float = 0.0, probe_offtarget: float = 0.0,
                 probe_n: int = 2048, corrupt_seed: int = 1234, **kw):
        super().__init__(**kw)
        self.rho = float(probe_label_noise)
        self.alpha = float(probe_offtarget)
        self.probe_n = int(probe_n)
        self.corrupt_seed = int(corrupt_seed)
        self._f4_ready = False

    def _setup(self):
        was = self._ready
        super()._setup()
        if self._f4_ready:
            return
        torch, np_ = self.torch, np
        y_tr = np_.array(self.tr.targets)
        rest = np_.array(self.sp["pool"])[len(self.pool):]        # unreachable by arms
        on = rest[y_tr[rest] < self.N_RARE]                       # target (rare) classes
        off = rest[y_tr[rest] >= self.N_RARE]                     # well-learned classes
        n_off = int(round(self.alpha * self.probe_n))
        n_on = self.probe_n - n_off
        probe = np_.concatenate([on[:n_on], off[:n_off]])
        rng = np_.random.default_rng(self.corrupt_seed)
        rng.shuffle(probe)                                        # interleave: probe[:n] is representative
        self.probe_idx = probe
        # global corrupted-label vector: deterministic, applies wherever the cheap
        # signal looks (probe slice and JEST-scored examples). Training is untouched.
        lab = np_.array(self.tr.targets).copy()
        if self.rho > 0:
            m = rng.random(len(lab)) < self.rho
            lab[m] = rng.integers(0, int(lab.max()) + 1, size=int(m.sum()))
        self.Ytr_signal = torch.as_tensor(lab, device=self.DEV).long()
        self.n_corrupted = int((lab != np_.array(self.tr.targets)).sum())
        self._f4_ready = True
        if self.verbose or not was:
            print(f"[f4] cifar backend: rho={self.rho} alpha={self.alpha} "
                  f"probe={len(probe)} ({n_on} target / {n_off} off-target) "
                  f"labels changed={self.n_corrupted}", flush=True)

    # ---- the two corrupted read paths (everything else inherited verbatim)
    def session_loss(self, st, n: int) -> float:
        torch = self.torch
        idx = torch.as_tensor(self.probe_idx[:n], device=self.DEV).long()
        st["model"].eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(idx), 512):
                ids = idx[s:s + 512]
                out = st["model"](self._norm(self.Xtr[ids]))
                tot += torch.nn.functional.cross_entropy(
                    out, self.Ytr_signal[ids], reduction="sum").item()
                cnt += len(ids)
        return tot / max(cnt, 1)

    def _mean_loss(self, model, ids_t) -> float:      # used only by learnability()
        torch = self.torch
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for s in range(0, len(ids_t), 512):
                ids = ids_t[s:s + 512]
                out = model(self._norm(self.Xtr[ids]))
                tot += torch.nn.functional.cross_entropy(
                    out, self.Ytr_signal[ids], reduction="sum").item()
        return tot / max(len(ids_t), 1)


def run_cifar(a):
    configs = [tuple(float(x) for x in c.split(":")) for c in a.configs.split(",")]
    allocs = a.allocators.split(",")
    out_path = a.out or f"{HERE}/f4_cifar.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out.setdefault("budget_fte", a.budget)
    out.setdefault("reps", a.reps)
    out.setdefault("cells", {})
    pool = [0, 1, 2, 3, 4, 5, 6, 7]
    t0 = time.time()
    for (rho, alpha) in configs:
        backend = CifarF4Backend(probe_label_noise=rho, probe_offtarget=alpha,
                                 cache_path=a.cache, verbose=False)
        truth = backend.true_values()
        out["truth"] = truth
        for name in allocs:
            key = f"{rho}|{alpha}|{name}"
            if key in out["cells"] and out["cells"][key].get("reps") == a.reps:
                print(f"[f4] skip {key}", flush=True)
                continue
            cell = run_cell(backend, name, a.budget, a.reps, seed_pool=pool,
                            truth=truth, factory=ALL_FACTORIES[name])
            cell.update({"rho": rho, "alpha": alpha,
                         "n_corrupted_labels": backend.n_corrupted,
                         "probe_composition": {"n": int(len(backend.probe_idx)),
                                               "offtarget_frac": alpha}})
            out["cells"][key] = slim(cell)
            atomic_json(out_path, out)
            print(f"[f4] rho={rho} alpha={alpha} {name:26s} regret "
                  f"{cell['mean_regret']*100:6.2f}+/-{cell['se_regret']*100:4.2f}pp  "
                  f"P(corr) {cell['p_correct']:.2f}  picks {cell['pick_counts']}  "
                  f"{cell['wall_s']:.0f}s", flush=True)
    atomic_json(out_path, out)
    print(f"[f4] wrote {out_path} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth")
    s.add_argument("--spec", default="robot", choices=["robot", "cifar"])
    s.add_argument("--cs", default="1.0,0.75,0.5,0.25,0.0,-0.25,-0.5,-0.75,-1.0")
    s.add_argument("--snrs", default="8,4,2,1,0.356,0.1")
    s.add_argument("--budgets", default="12")
    s.add_argument("--reps", type=int, default=500)
    s.add_argument("--n-orth", type=int, default=4,
                   help="independent nuisance directions pooled per cell")
    s.add_argument("--allocators", default=",".join(ALL_FACTORIES))
    s.add_argument("--out", default=None)
    s.set_defaults(func=run_synth)

    c = sub.add_parser("cifar")
    c.add_argument("--configs", default="0:0,0:0.5,0:1.0,0.5:0,1.0:0",
                   help="comma list of rho:alpha")
    c.add_argument("--budget", type=float, default=6.0)
    c.add_argument("--reps", type=int, default=5)
    c.add_argument("--allocators",
                   default="learning_progress_exp3,jest,jest_then_confirm,"
                           "successive_halving,random,regmix")
    c.add_argument("--cache", default=f"{HERE}/cache_f4.json")
    c.add_argument("--out", default=None)
    c.set_defaults(func=run_cifar)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
