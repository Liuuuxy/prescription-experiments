# The Prescription Criterion (design, 2026-07-21)

**Question:** success is driven by many factors at once (height, distance-from-sink, style, category affordance, + 78% rollout stochasticity). On what criterion can data prescription still operate?

**Provenance:** 4 independent framework proposals (bandit, causal mechanism-matching, optimal-experiment-design, paper-strategy lenses) + 2 adversarial judges, all grounded in the verified findings of `FACTOR_ANALYSIS_REPORT.md`. Full record: `factor_analysis/wf3_prescription_criterion_panel.json`. Judges converged on the merged framework below (working name **GMSR: Gated Mechanism-Source Race**).

---

## 0. Why multifactor is good news, not bad

Because the factors are **additive with zero interactions** (verified: interaction terms add 0.000 held-out R²), you never need to prescribe over the combinatorial cell grid (height × distance × style × category). You prescribe along **axes independently** — linear cost, and information from one probe transfers across the grid. And because the factors act through **dissociable mechanisms** (far→never-touch approach failures; tall→touch-then-fumble grasp failures; style→perceptual, also slows successes), each axis pins its own **demo type**. Demo content is not a free choice — it is determined by the mechanism.

## 1. The criterion

Prescribe units of (mechanism m, data source s, dose k). Score each by four **measured** (never predicted) quantities:

```
Score(m, s) = w_m × a_m(s) × τ̂_m(s) / c_s     subject to structural retention constraints
```

| term | meaning | how measured | cost |
|---|---|---|---|
| **w_m** — deployment mass | fraction of episodes the mechanism affects (far band = 26%, tall ≈ 17%, hard-style tercile ≈ 33%) | free, from pooled_episodes.csv | 0 |
| **a_m(s)** — attainability | can source s even PRODUCE demos in that region? | 30 capped collection attempts per (m,s), Wilson CI | ~half a day, no GPU |
| **τ̂_m(s)** — net teachability | ΔSR of a probe fine-tune **minus a matched-budget generic-new-data control**, on the target slice | probe protocol (§2) | 1 probe ≈ 1 day H100 |
| **c_s** — per-demo cost | expert-loop 1×, MimicGen ~2–3×, human ~20× (but human enters as 1–5 *source* demos that MimicGen multiplies) | known | — |

**Retention is a structural constraint, not a score term:** ≥1/3 coverage floor in every fine-tune mix, plus diversity requirements *inside* targeted generation (interior placement variation, ≥3 styles). Evidence: retention tracked the *composition/diversity* of the selected set (core −0.1pp vs value −9.4pp at the same mixture fraction), not the mixture fraction itself.

### Why each gate exists — each corresponds to a verified failure of a simpler criterion

- **P(fail) alone** (the old criterion) is the degenerate case a=1, τ∝(1−p): all three assumptions are now measured false. The tall cliff has maximal P(fail), near-zero attainability under the expert loop (policy can't collect where it can't succeed), and zero measured teachability from the pool (610-demo saturation arm → tall SR 0.14).
- **Gradient influence** failed the gradient-encoding gate (established earlier).
- **The generic-lift subtraction (τ̂ net of control)** is mandated by random-selection capturing ~80% of the targeted lift (core−random +2.0pp ns). Any un-subtracted probe estimate mostly measures "new data helps", not "prescription works".
- **The attainability gate** operationalizes the generation ceiling *before* GPU is spent: regions unattainable AND unteachable in every sim source route up the escalation ladder (expert-loop → MimicGen → human-source-demos×MimicGen → G1/human bulk) and are logged with their deployment mass as the **quantified human/G1 collection bill** — the generation ceiling becomes a decision output, not a narrative.

## 2. Estimation protocol (the probes) — dictated by the noise floor

The 78% rollout-stochasticity floor sets everything. SE of a paired ΔSR at n paired configs ≈ √(2·p(1−p)·0.78/n) ≈ **4.9pp at n=150, ~3.5pp at n=300**. Consequences (all pre-registered):

1. **Region-level only, never per-category/per-cell** (a cell would need ~300 configs of its own; per-category prescription is forbidden by power — consistent with the observed dose-response r=0.06 being unresolvable noise).
2. **≤3 teachability probe arms** per round (E[max] of 6 noise arms ≈ +6pp ≈ the true effect size — winner's curse eats larger fan-outs). Warm starts are free: expert-loop×tall ≈ 0 and expert-loop×far ≈ 0 are ALREADY measured (saturation arm, rim invariance) — do not re-probe them. Suggested round 1: tall×MimicGen, far×approach-rich-MimicGen, matched-budget generic-new control.
3. **Precision stack:** one frozen base checkpoint, identical LoRA recipe, identical paired seed lists for every arm; config-clustered inference (episode-level tests are anticonservative — verified p=0.003→0.32 example); CUPED variance reduction using the R²≈0.12 factor model; **enrich target evals with mixed configs** (the 55% that both succeed and fail across checkpoints — the only movable ones; ~1.8× effect amplification); duplicate one arm with a second LoRA seed to measure the seed-noise floor — slopes below that floor are declared unrankable.
4. **Positive control:** the probe pipeline must recover the known +9–11pp weak-region lift before any novel τ̂ is believed.
5. **Doses near the measured knee:** the pool dose-curve knee is small (~20–60 demos; at 14 demos ~80% of the 200-demo gain was realized; 610 demos *hurt*). Probe at modest doses + one two-dose satellite on the best arm, because the knee was measured on pool data and may not transfer to new sources.
6. **Elimination discipline:** advance/eliminate on pre-registered confidence bounds only (e.g. eliminate if UCB95(τ) < threshold); round-1 point estimates are never reported or allocated on; survivors replicate on fresh seeds.

## 3. Allocation

Water-fill the confirmed arms in descending w_m·τ̂/c_s at replication-validated doses, subject to the coverage floor. Never fund past ~2× the measured knee (the saturation arm proved overdosing is actively harmful: 610 demos → 0.299 overall vs 200 → 0.371).

## 4. Validation design (confirmatory, pre-registered)

Four arms on identical paired seed lists, config-clustered: **prescribed mixture** vs **matched-budget random-new-data** (the non-negotiable control) vs **P(fail)/FSC-style incumbent** vs **frozen base**. Plus:
- **Mismatched placebo** (the single best causal control from the panel): the same demos with mechanism assignments swapped (approach demos given to tall bins, grasp demos to far bins). If matched beats mismatched, mechanism-matching — not generic new data — does the work. Must be powered at ≥300 configs/band.
- **Never-succeed flips**: successes on the enumerated 28% never-succeed config list (re-verified under the frozen base) — the highest-power binary endpoint the noise floor permits.
- Read all effects against the **pre-computed attainable bound** (Σ w_m·headroom ≈ 10–20pp given the 78% floor), so +6pp reads as success against +15pp-possible, not failure against fantasy.

**Prediction table (every row publishable):** matched > mismatched & > random-new → prescription criterion validated. All τ̂≈0 in sim sources but attainability probes show collectable regions → "can teach but can't collect" — the FSC-style story. Attainable AND unteachable everywhere → the strongest generation-ceiling result: *failure rate marks where you can neither collect nor teach; prescription must be attainability-routed, causally probed, retention-constrained* — and the G1/human bill is the paper's quantified deliverable.

## 5. Known risks (from the judges' flaw-hunting)

- Treatment-effect additivity across (m,s) is a shrinkage prior, NOT verified (only *baseline* factor additivity is) — test with an interaction re-fit, never assume.
- The realistic prior, given core−random = +2pp ns, is a double-null with maybe one positive exception. The design must make that outcome publishable (it does — §4).
- Budgets: stay within ~600 total probe demos; MimicGen tall-replay success rate is itself a free attainability signal — check it before any GPU.
- Mixed-config labels come from past checkpoints (stale-labeling risk); re-verify a sample under the frozen base.
