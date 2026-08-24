"""Shared race runner: per-REP records (never just a mean), paired comparisons, CIs.

`wind_tunnel.run_race` reports means only. Everything in this study is within noise
unless proven otherwise, so every cell here keeps the full per-rep vector and every
"X beats Y" claim is a PAIRED test on the same rep index (same env seed sequence,
same instance) with a bootstrap CI.

Writes nothing on import. Used by part1_driver.py and f4_transfer.py.
"""
from __future__ import annotations

import json
import os
import time
import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from wind_tunnel import (ALLOCATOR_FACTORIES, OracleToken, WindTunnel, regret_of,
                         value_of)


def alloc_rng(name: str, rep: int, base_seed: int = 0) -> np.random.Generator:
    """Same convention as wind_tunnel.run_race, so results are comparable."""
    return np.random.default_rng(base_seed + 1000 * rep + zlib.crc32(name.encode()) % 997)


def run_cell(backend, name: str, budget: float, reps: int,
             seed_pool: Optional[Sequence[int]] = None, base_seed: int = 0,
             truth: Optional[Dict[str, float]] = None, factory=None) -> dict:
    """Run ONE (allocator, budget) cell for `reps` independent replicates.

    Replicate r uses seed_offset=r (a different env seed sequence) and a different
    allocator rng, so replicates are the study's 'seeds'. A fresh allocator object
    per rep guarantees no state leaks between replicates."""
    token = OracleToken()
    truth = dict(truth if truth is not None else backend.true_values())
    best_arm = max(truth, key=truth.get)
    best_v = truth[best_arm]
    fac = factory or ALLOCATOR_FACTORIES[name]
    recs: List[dict] = []
    t0 = time.time()
    for rep in range(reps):
        env = WindTunnel(backend, budget, name=name, seed_pool=seed_pool,
                         oracle_token=token, seed_offset=rep)
        d = fac(token).run(env, alloc_rng(name, rep, base_seed))
        recs.append({
            "rep": rep, "regret": regret_of(d, truth), "value": value_of(d, truth),
            "top": d.top, "correct": int(d.top == best_arm),
            "spent_fte": env.spent_fte, "unspent_fte": env.remaining_fte,
            "wall_s": round(env.wall_s, 1),
            "by_kind": {k: round(v, 4) for k, v in env.meter.by_kind().items()},
            "weights": {a: round(w, 4) for a, w in d.weights.items()},
        })
    reg = np.array([r["regret"] for r in recs], float)
    val = np.array([r["value"] for r in recs], float)
    cor = np.array([r["correct"] for r in recs], float)
    return {
        "allocator": name, "budget_fte": budget, "reps": reps,
        "best_arm": best_arm, "best_value": best_v,
        "mean_regret": float(reg.mean()), "sd_regret": float(reg.std(ddof=1)) if reps > 1 else 0.0,
        "se_regret": float(reg.std(ddof=1) / np.sqrt(reps)) if reps > 1 else 0.0,
        "mean_value": float(val.mean()),
        "p_correct": float(cor.mean()), "p_correct_ci": wilson(cor.sum(), reps),
        "frac_of_oracle": float(val.mean() / best_v) if abs(best_v) > 1e-12 else float("nan"),
        "mean_spent_fte": float(np.mean([r["spent_fte"] for r in recs])),
        "mean_unspent_fte": float(np.mean([r["unspent_fte"] for r in recs])),
        "wall_s": round(time.time() - t0, 1),
        "pick_counts": {a: int(sum(1 for r in recs if r["top"] == a))
                        for a in sorted({r["top"] for r in recs})},
        "regrets": reg.tolist(), "values": val.tolist(),
        "recs": recs,
    }


def wilson(k: float, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def paired_vs(cell: dict, ref_cell: dict, n_boot: int = 20000, seed: int = 0) -> dict:
    """PAIRED comparison of two allocators on the same replicates (same env seeds).
    Returns mean regret difference (cell - ref; negative = cell is better), its
    bootstrap 95% CI, and the sign-test p-value. 'Within noise' = CI spans 0."""
    a = np.array(cell["regrets"], float)
    b = np.array(ref_cell["regrets"], float)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = d[np.abs(d) > 1e-12]
    if len(nz):
        k = int((nz < 0).sum())
        from math import comb
        m = len(nz)
        tail = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2 ** m
        p = float(min(1.0, 2 * tail))
    else:
        p = 1.0
    return {"delta_regret": float(d.mean()), "ci95": [float(lo), float(hi)],
            "sign_p": p, "n_pairs": int(n), "n_nonzero_pairs": int(len(nz)),
            "beats_ref": bool(hi < 0), "loses_to_ref": bool(lo > 0),
            "within_noise": bool(lo <= 0 <= hi)}


def b95(cells_by_budget: Dict[float, dict], frac: float = 0.95) -> Optional[float]:
    """Budget-to-95%-of-oracle: smallest budget on the grid whose mean decision
    value >= frac * oracle value, with linear interpolation between grid points.
    None = never reached on this grid."""
    bs = sorted(cells_by_budget)
    if not bs:
        return None
    best_v = cells_by_budget[bs[0]]["best_value"]
    tgt = frac * best_v
    prev_b = prev_v = None
    for b in bs:
        v = cells_by_budget[b]["mean_value"]
        if v >= tgt:
            if prev_b is None or prev_v is None or v == prev_v:
                return float(b)
            return float(prev_b + (b - prev_b) * (tgt - prev_v) / (v - prev_v))
        prev_b, prev_v = b, v
    return None


def atomic_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def slim(cell: dict) -> dict:
    """Cell without the per-rep `recs` blob (keeps the regret vector)."""
    return {k: v for k, v in cell.items() if k != "recs"}
