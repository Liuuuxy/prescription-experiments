"""The arena: a calibrated PrescriptionEnv with hidden ground truth.

Model (per region r), all quantities in success-rate units:
    success_r(m) = base_r + headroom_r * (1 - exp(-m / tau_r))     # concave, saturating
where m is the number of *usable* demos added to region r.

An allocation requests k_r demos for region r. Only a fraction survive:
    usable m_r = k_r * collectability_r        (deterministic / expected)
             or  Binomial(k_r, collectability_r) (stochastic, when an rng is given).

Net deployed success trades region gains against retention (forgetting):
    gross      = sum_r weight_r * success_r(m_r) / sum_r weight_r
    retention  = sum_r ret_risk_r * m_r          # off-manifold added mass degrades the majority
    net        = clip(gross - retention, 0, 1)

With an rng, the final number is a finite-rollout estimate: Binomial(n_eval, net) / n_eval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    """One weak region's hidden ground-truth parameters."""

    base: float           # base success rate in [0, 1]
    headroom: float       # max additional success reachable from data, in [0, 1]
    tau: float            # usable demos to reach ~63% of headroom (> 0)
    collectability: float = 1.0   # P(a requested demo is usable), in (0, 1]
    ret_risk: float = 0.0         # retention cost per usable demo added here (>= 0)
    weight: float = 1.0           # fraction of the eval distribution in this region (>= 0)
    name: str = ""                # human-readable label (optional)


class PrescriptionEnv:
    """A calibrated sandbox with known-but-hidden per-region payoff structure."""

    def __init__(self, regions, n_eval: int = 280, feature_fidelity: float = 1.0,
                 decoy=None, probe_noise0: float = 0.2):
        self.regions = list(regions)
        self.n_eval = int(n_eval)
        self.feature_fidelity = float(feature_fidelity)
        self._decoy = None if decoy is None else np.asarray(decoy, dtype=float)
        self.probe_noise0 = float(probe_noise0)

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def usable(self, allocation, rng=None):
        """Usable demos per region given a requested allocation."""
        k = np.asarray(allocation, dtype=float)
        c = np.array([r.collectability for r in self.regions], dtype=float)
        if rng is None:
            return k * c
        return rng.binomial(k.astype(int), c).astype(float)

    def net_success(self, allocation, rng=None) -> float:
        """Net deployed success of an allocation (requested demos per region)."""
        m = self.usable(allocation, rng=rng)
        base = np.array([r.base for r in self.regions])
        headroom = np.array([r.headroom for r in self.regions])
        tau = np.array([r.tau for r in self.regions])
        ret_risk = np.array([r.ret_risk for r in self.regions])
        weight = np.array([r.weight for r in self.regions])

        success = base + headroom * (1.0 - np.exp(-m / tau))
        gross = float(np.sum(weight * success) / np.sum(weight))
        retention = float(np.sum(ret_risk * m))
        net = float(np.clip(gross - retention, 0.0, 1.0))

        if rng is None:
            return net
        return float(rng.binomial(self.n_eval, net) / self.n_eval)

    # ---- ground-truth marginal value + the predictor's (imperfect, paid) proxy ----

    def marginal_value(self, region_idx: int, chunk: float) -> float:
        """Noise-free Delta net_success from adding `chunk` requested demos to one region."""
        alloc = np.zeros(self.n_regions)
        alloc[region_idx] = chunk
        return self.net_success(alloc) - self.net_success(np.zeros(self.n_regions))

    def _decoy_vector(self) -> np.ndarray:
        """The confound the predictor's features collapse to at zero fidelity.

        Default = P(fail) = 1 - base: exactly the heuristic that mis-ranks value in
        the real study, so a low-fidelity predictor degenerates to the P(fail) arm.
        """
        if self._decoy is not None:
            return self._decoy
        return 1.0 - np.array([r.base for r in self.regions])

    def probe(self, region_idx: int, chunk: float, cost: float, rng,
              fidelity=None) -> float:
        """A paid, noisy proxy of region r's marginal value.

        estimate = fidelity * true_marginal + (1 - |fidelity|) * decoy + noise,
        for fidelity in [-1, 1]: +1 tracks truth, 0 is the pure decoy, -1 tracks the
        NEGATIVE of truth (the actively-misleading offline<->online anti-correlation).
        Noise std shrinks as 1/sqrt(cost); paying more cuts variance but never the bias.
        """
        fid = self.feature_fidelity if fidelity is None else float(fidelity)
        true_mv = self.marginal_value(region_idx, chunk)
        decoy = float(self._decoy_vector()[region_idx])
        signal = fid * true_mv + (1.0 - abs(fid)) * decoy
        noise = rng.normal(0.0, self.probe_noise0 / np.sqrt(max(cost, 1e-9)))
        return float(signal + noise)
