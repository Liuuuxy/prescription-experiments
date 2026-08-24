# Prescription: Predict-vs-Explore Comparison — Design Spec

**Date:** 2026-07-17 · **Owner:** Xinyuan Liu · **Scope:** simulation only (G1 deferred)
**Status:** design approved (calibration + bandit family chosen), pre-implementation.
**Supersedes** the `HEADROOM_PROBE_DESIGN.md` plan (that probe was killed by six verifications:
no collectable failure data, no stable learnable-middle, self-distilling collector, action-space
format landmine, eval blind to middle categories, and the reward is inside the noise at n≈280).

---

## 1. Finalized problem

Given a base policy, a set of candidate weak **regions**, and a fixed data **budget** B, decide how to
spend B across regions to maximize net deployed success. Two philosophies:

- **Predict** the per-region payoff cheaply *before* collecting (a scoring/gate approach), or
- **Explore** it *online* by collecting + retraining + measuring (a bandit).

**Research question (the deliverable): under what conditions does predict beat explore for data
prescription, and which regime is RoboCasa in?** This subsumes the earlier subproblems: SP1 (what
data) = the allocation each method outputs; SP3 (guarantee) = the predictor's calibrated Δ̂ with a CI.

## 2. Why a calibrated sandbox, not a raw RoboCasa horse-race

A real head-to-head on RoboCasa fine-tunes is **unfair and uninformative**: (a) no ground truth —
you don't know each region's true payoff, so you can't score who is closer; (b) no power — the max
measured effect (core−random +2pp) is p=0.62 at n≈280, so both methods collapse to random. Racing on
noise answers nothing. Instead: a controlled **`PrescriptionEnv`** whose hidden ground truth is
**calibrated to real measured statistics** (a digital twin), where "which is better" is measurable
with tight error bars — the same methodology that made the gradient-encoding result clean.

## 3. The arena — `PrescriptionEnv` (hidden ground truth, calibrated)

Per region *r*, hidden parameters:
- **base success** `p_r` ← real per-category baseline rates.
- **marginal-value curve** `h_r(k)` (headroom → saturation, possibly inverted-U) ← real dose-response
  (0→26.2, ~14→35.1, 200→37.1, 610→29.9).
- **collectability** `c_r` ∈ [0,1] (chance a requested demo is usable) ← real yields (pi0 0/14 hard,
  ~65% mid; GR00T 1/9 hard, 76% mid).
- **retention coupling** — adding mass to *r* perturbs the shared-majority success ← real per-arm
  non-targeted regressions (core NT −0.1, value NT −9.4, coverage NT −8.9).
- **noise model** — final success measured as Binomial(n_eval, ·), n_eval a knob (default ≈280).

Net success of an allocation `{k_r}` = `p_base + Σ_r h_r(collected(k_r,c_r)) − retention_penalty(Σ off-manifold mass) + binomial noise`.
**Honesty:** 7 arms is few points, so the twin is intentionally **low-dimensional / coarse** — it is a
plausible twin, not a precise one. Generality comes from the regime sweep (§6); real transfer from the
anchors (§7). Calibration code: `calibrate_from_arms()` reads `weakregion/arms*.json`,
`eval_strat_*`/`eval_balcat_*`, and the yield notes.

## 4. Contenders (pluggable allocators, one interface)

- **Predictor (PPP):** cheap features per region → μ̂_r → greedy/one-shot allocation, **no
  exploration**. Features: **C** collectability probe, **E** encodability (best-single-SVD-mode AUC),
  **H** headroom (dose slope / mini-fit), **R** retention risk (alignment to g_R). Outputs Δ̂_r + CI.
- **Bandit:** **successive-halving best-arm identification** (fixed budget, few expensive noisy
  pulls), reward = **net** success (target gain − retention regression). This is the only bandit
  family per the chosen scope; TSCL/ALP-GMM cited as prior art, not run.
- **Baselines:** `random`, `pfail` (P(fail) top-K heuristic), `oracle` (knows the hidden truth →
  upper bound, used to normalize regret).

## 5. Fair-comparison protocol (the core requirement)

1. **Common currency = one total budget B; everything is charged to it.** The bandit's exploration
   pulls AND the predictor's feature-probes (collectability rollouts, gradient computations) both
   consume B. Nothing is free — if PPP's features cost as much as a few pulls, the comparison must
   show it.
2. **Identical arenas:** same env instances / fixed seeds for every method.
3. **Identical final eval:** resulting allocation's net success, same estimator + noise.
4. **Identical baselines** under the same accounting; **oracle** normalizes across regimes.
5. **Many seeds → mean ± CI, paired tests, per-regime breakdown.** No method sees the hidden truth.
6. **Pre-registered metrics:** (i) simple regret vs oracle at budget B; (ii) budget to reach 95% of
   oracle; (iii) robustness across regimes. Thresholds/metrics fixed before running.

## 6. Regime sweep (what "which is better" yields — a map, not a winner)

Sweep: payoff predictable-from-features vs not; stationary vs non-stationary (saturation during
spend); high vs low collectability; high vs low noise; few vs many regions. Hypothesis (to test, not
assume): **predict** wins when features are predictive AND cheaper than exploration; **explore** wins
when payoff is not feature-predictable or non-stationary; a **hybrid** (predict where confident,
explore where not) upper-bounds both. The calibrated-to-RoboCasa point then says which regime the real
task sits in.

## 7. Real anchors (transfer spot-check, not a powered result)

1–2 **pool-selected** 20k fine-tunes of the predictor's top pick vs the bandit's top pick, evaluated
on the existing stratified protocol (directly comparable to core/random). **Pool-selected, not
generated**, to sidestep the GR00T action-space landmine and stay comparable; this tests the
**selection/headroom+retention** regime (the part doable cleanly on real hardware). Framed as a
sanity spot-check that the twin's recommended region behaves as predicted — not a significance claim.

## 8. Clean code (pure numpy, no GPU, fast, test-first)

```
prescription_compare/
  env.py            # PrescriptionEnv + calibrate_from_arms()
  allocators/
    base.py         # Allocator interface + budget accounting
    random_alloc.py · pfail.py · predictor.py · bandit_sh.py · oracle.py
  evaluate.py       # fair-budget accounting + final-value eval
  experiment.py     # regimes × allocators × seeds → results table
  plots.py          # regret curves, budget-to-target, per-regime bars
  tests/            # env invariants, budget-accounting FAIRNESS, oracle-optimality, allocator units
```
Built **test-first**: the fairness accounting and oracle-optimality are unit-verified, not assumed.

## 9. Metrics summary
- Simple regret = oracle_value − method_value at budget B (per regime, mean ± CI).
- Budget-to-95%-oracle (sample efficiency).
- Win-rate of predict vs explore across seeds, per regime.
- Anchor: did the twin's predicted ordering (predictor-pick vs bandit-pick) match the real fine-tune?

## 10. Limitations (state up front)
- Twin calibrated from only 7 arms → coarse; regimes + anchors carry generality.
- Anchors are under-powered (1–2 runs) → spot-check only.
- Best-arm-ID only (other bandit families cited, not run).
- Sandbox realism is the main external-validity risk; mitigated by regime-conditional claims.
```
