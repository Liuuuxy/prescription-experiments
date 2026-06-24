# Targeted Data Acquisition for Imitation Learning — Core Algorithm, Critique, Literature, and Plan

*Written 2026-06-23. Task: PickPlaceCounterToSink (RoboCasa). Student: pi0 (~55%). Pool: RoboCasa MimicGen, ~9.9k demos.*

---

## 0. TL;DR

- **Goal:** show that *selecting* extra training demos with our algorithm beats adding *random* extra demos, per demo.
- **Core algorithm (now):** partition the task into **object-category regions**; score each region by **`P(student fails) × uncertainty`** (Wilson lower-bound, robust); allocate a demo budget proportional to score, **capped by pool availability**; subsample those demos (core arm) and compare to a same-size random subsample.
- **Your instinct on uncertainty is correct.** As implemented (variance of K sampled action chunks), it is a **weak failure predictor for pi0 (AUC 0.595)**. The literature explains why: action-sampling spread conflates **aleatoric** uncertainty (many valid ways to act — multimodality) with **epistemic** uncertainty (the model doesn't know). For a competent policy the spread is mostly aleatoric, so it does *not* track failure. **Recommendation: drop the variance-based term** and either (a) keep it as a clean **competence-based** selector (`P(student fails)` only), or (b) swap in a *validated* uncertainty (Diff-DAgger's diffusion loss, FIPER's RND + action-chunk entropy), or (c) move to a more principled criterion entirely (**influence functions**, CUPID).

---

## 1. The problem

Imitation-learning policies are data-hungry, and not all demos are equally useful. Given a fixed budget of *extra* demos to add to a base policy, **which demos should we add?** The hypothesis: demos concentrated where the policy is weak improve it more, per demo, than demos drawn at random. This is an **active learning / data-selection** problem specialized to robot IL.

Two ways to obtain the extra demos:
- **Generate** them with an expert teacher (pi0/GR00T) — limited by the *teacher-bias ceiling*: an expert can only demonstrate what it can already do.
- **Subsample** them from a large existing pool (here, RoboCasa's MimicGen 10k) — no teacher-bias ceiling; the algorithm just *selects*. **This is our current setting.**

---

## 2. The core algorithm (current state)

### 2.1 Derivation
Expected improvement from one demo in region *r* is modeled as proportional to *how badly the student needs it there*:
```
improvement per demo in r   ∝   P(student fails | r) × uncertainty(r)
```
In the **generation** setting, collecting one success in *r* costs ~`1/P(teacher succeeds | r)` attempts, so the *expected improvement per attempt* is
```
score(r) = P(teacher succeeds | r) × P(student fails | r) × uncertainty(r)
```
and you allocate **attempts ∝ score** (cost-aware). In the **subsampling** setting there is no per-attempt cost — a demo is a demo — so the **teacher term drops out**:
```
score(r) = P(student fails | r) × uncertainty(r)          # subsampling
```
Allocate **demos ∝ score**, **capped by how many the pool has for region r** (water-filling), optionally concentrated on the top-K regions.

### 2.2 Robustness
Per-region rates are noisy, so `P(student fails)` and `P(teacher succeeds)` use **Wilson score lower bounds**, not raw fractions — a conservative estimate that won't over-allocate to a region that looks hard from 2 lucky samples.

### 2.3 What "region" is
**Object category** (juice, pitcher, soap_dispenser, …). Chosen because it is (a) estimable, (b) something the pool can be sliced by, and (c) more informative than the older "tall object" heuristic — see §3.

### 2.4 What we actually ran
- Student = pretrained **pi0**, eval'd on 500 scenes → **54.8%** success; per-category failure rates.
- Uncertainty = std of **K=8** sampled pi0 action-chunks at the first state.
- Pool = **9,885** MimicGen demos across 79 objects (median 133/object).
- Allocation (budget 200, top-10): **juice 29, pitcher 24, spray 23, canned_food 23, tupperware 21, jar 17, soap_dispenser 16, ice_cube 16, cheese_grater 16, cream_cheese_stick 15.**
- Random control = 200 demos uniform over the pool (mostly easy objects).

---

## 3. Is the "tall object" hypothesis true? (and what "hard" means)

| | success |
|---|---|
| TALL objects (h > 0.10 m) | **42%** (78/184) |
| SHORT objects (h ≤ 0.10 m) | **62%** (196/316) |
| AUC(height → failure) | **0.60** |

**True but weak.** Tall objects are harder, but height alone barely predicts failure. Critically, **4 of the 10 hardest categories are SHORT** (cream_cheese_stick h=0.02 fails 88%, canned_food, ice_cube, tupperware). So **"hard" = high failure rate, *not* tall.** Targeting by per-category failure is strictly better than "target tall" — it also catches hard short objects the height heuristic misses.

---

## 4. What's wrong with the algorithm (honest critique)

1. **The uncertainty term is the weak link (your point).** Variance of sampled action chunks is a crude *aleatoric*-dominated signal. The literature is explicit that early-rollout action variance "is unclear whether due to aleatoric (many ways to complete the task) or epistemic (model unsure)" uncertainty. Empirically for pi0 it gives **AUC 0.595 ≈ no signal**. Because the subsampling score is `P(fail) × uncertainty` and uncertainty is ~flat (≈0.04–0.06 across categories), **the score essentially reduces to `P(student fails)`** — so right now we are doing **competence-based selection** with an inert uncertainty multiplier. That's not *wrong*, but the third term is currently decorative.
2. **"Improvement ∝ P(fail)" is an assumption, not a guarantee.** Some failures are *not data-fixable* in the targeted region (e.g., perception/grasp-geometry limits). Adding demos there may not help — the algorithm can't tell "needs data" from "fundamentally hard."
3. **No coverage / diversity term.** Data-scaling-law work finds **diversity of objects/scenes matters more than raw count**. Over-concentrating the budget on 10 categories could *hurt* coverage; the right concentration is an empirical question (we made it tunable: proportional ↔ top-K).
4. **Category granularity is coarse.** Failure can be driven by within-category factors (placement, occlusion, layout). Category-level scoring averages over these.
5. **Open-loop proxy.** `P(student fails)` is closed-loop (good), but the demo's *value* to training is not the same as the policy's failure rate there — see influence-function methods (§5) which measure the former directly.

---

## 5. How others do this (literature review, 2024–2026)

### 5.1 Classical active-learning acquisition families
Our score is an **expected-improvement / cost-aware** acquisition function. The canonical families (uncertainty sampling; query-by-committee/disagreement; **expected error reduction**, Roy & McCallum; expected model change; representativeness/diversity & core-sets; **cost-aware**; shift-aware) and their failure modes are surveyed in *Beyond uncertainty in modern active learning* (2026). Key caution from this literature: **pure uncertainty sampling is often *not* the most data-efficient**, and EER/value-of-information methods are stronger but computationally heavy.

### 5.2 Uncertainty & failure prediction for *generative* robot policies (most relevant to your concern)
- **Diff-DAgger** (Lee et al., 2024): instead of action variance, uses the **diffusion training objective (denoising loss)** as the uncertainty signal to decide when to query the expert — improves **failure prediction by 39%** and completion by 20.6%. Directly supports "use a better uncertainty than action std."
- **FIPER** (NeurIPS 2025): combines **random network distillation (RND)** for OOD detection in the observation-embedding space with an **action-chunk entropy (ACE)** score; explicitly separates *benign* OOD (multimodal/aleatoric) from *real* failures — predicts failures more accurately and earlier.
- **TRIAGE** (2026): routes interventions by **gating aleatoric vs epistemic** uncertainty — "don't treat all uncertainty the same." This is exactly the distinction our action-variance term fails to make.

### 5.3 Demonstration curation / selection (the subsampling analog)
- **CUPID / influence-function curation** (2025): estimates each demonstration's **influence on closed-loop performance**, removing harmful demos and keeping beneficial ones — a more principled "which data helps" than failure-rate targeting.
- **DemInf — Robot Data Curation with Mutual Information Estimators** (2025): ranks/selects demos by **mutual information** (quality), filtering low-quality/redundant trajectories.
- **SCIZOR** (2025): self-supervised **task-progress + similarity dedup**, ~**+15%** average by removing suboptimal/redundant state-action pairs.
- **Curating Demonstrations using Online Experience** (2025): uses the robot's **own rollouts** to decide which demos to keep — same spirit as our "run the student to find weak spots."
- **Data Quality in Imitation Learning** (2023): foundational framing of *what makes a demo good*.

### 5.4 Data scaling & active fine-tuning
- **Data Scaling Laws in Imitation Learning** (2024): **diversity ≫ quantity**; beyond a per-object threshold, extra demos add little — argues for spreading across objects, a counter-pressure to sharp targeting.
- **Active Fine-Tuning of Multi-Task Policies** (2024) and **Efficient Evaluation of Multi-Task Robot Policies with Active Experiment Selection** (2025): active selection of *what to fine-tune on* / *what to evaluate* — closest "active" framing to our setting.
- **IntervenGen** (2024): generates *interventional* (corrective) data for data-efficient robust IL — an alternative to subsampling for getting targeted data.

### 5.5 Where our method sits
Our approach is **closed-loop, competence-based active selection at the region level** — a lightweight, interpretable point in this space. It is weaker in principle than **influence-function** (CUPID) or **mutual-information** (DemInf) curation, but far cheaper and directly tied to *where the deployed policy fails*. Its current uncertainty term is the part most clearly superseded by Diff-DAgger / FIPER.

---

## 6. Plan & improvements

### 6.1 Immediate (finish the experiment as designed)
1. Build LeRobot subsets from `arms.json` (core-200 / random-200).
2. Stand up **openpi LoRA fine-tuning** on the H100 (the one infra gap).
3. Fine-tune pi0 on `baseline+core` vs `baseline+random`; eval on a held-out, **region-stratified** set; report per-demo improvement **and** whether core helps the targeted objects *without hurting others* (the coverage check from §5.4).

### 6.2 Fix the uncertainty term (your concern)
- **Default / honest:** **drop it** → score = `P(student fails)` (Wilson-LCB). Report the method as **competence-based selection**; it's defensible and is what the data supports.
- **If keeping an informativeness term:** replace action-std with a **validated** signal — Diff-DAgger-style **denoising/flow loss** or FIPER's **RND + action-chunk entropy** — and re-measure its AUC against failure *before* trusting it.
- **Either way:** ablate **2-term vs 3-term** in the experiment so the uncertainty term has to *earn* its place.

### 6.3 Stronger selection criteria (if we want to go beyond the heuristic)
- **Influence-function selection (CUPID-style):** rank pool demos by estimated effect on closed-loop success — the most principled upgrade.
- **Add a diversity/coverage guard:** cap per-category share or add a coverage term, per the scaling-law evidence.
- **Finer regions:** category × placement, once estimable (needs more eval scenes).

### 6.4 Validation we still owe
- Confirm `P(fail)`-targeted failures are **data-fixable** (not perception ceilings) — e.g., check whether the pool's demos for the hard categories are actually *good*.
- Pre-register the **concentration** (top-K) choice; it trades effect size vs coverage and shouldn't be tuned post hoc.

---

## 7. One-paragraph answer to "what is the core algorithm now?"
*Run the student policy on many scenes; for each object category estimate how often it fails (Wilson lower-bound); score categories by that failure rate (times an uncertainty term that, for pi0, is currently inert); allocate a fixed demo budget to the highest-scoring categories, capped by how many the demo pool contains; subsample those demos as the "targeted" set and an equal number at random as the control; fine-tune the policy on each and compare.* It is **competence-based active data selection** — sound and interpretable, with the uncertainty term being the weakest and most upgradeable component.

---

## Sources
- [Beyond uncertainty in modern active learning (Frontiers, 2026)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1844765/full)
- [Diff-DAgger: Uncertainty Estimation with Diffusion Policy (2024)](https://arxiv.org/abs/2410.14868)
- [FIPER: Failure Prediction at Runtime for Generative Robot Policies (NeurIPS 2025)](https://arxiv.org/abs/2510.09459) · [project page](https://tum-lsy.github.io/fiper_website/)
- [TRIAGE: Aleatoric-Epistemic Gated Interventions (2026)](https://arxiv.org/pdf/2603.08128)
- [Curating Demonstrations using Online Experience (2025)](https://arxiv.org/pdf/2503.03707)
- [Robot Data Curation with Mutual Information Estimators / DemInf (2025)](https://arxiv.org/abs/2502.08623)
- [Data Scaling Laws in Imitation Learning for Robotic Manipulation (2024)](https://arxiv.org/pdf/2410.18647)
- [Active Fine-Tuning of Multi-Task Policies (2024)](https://arxiv.org/pdf/2410.05026)
- [Efficient Evaluation of Multi-Task Robot Policies with Active Experiment Selection (2025)](https://arxiv.org/pdf/2502.09829)
- [IntervenGen: Interventional Data Generation for Data-Efficient Robot IL (2024)](https://arxiv.org/pdf/2405.01472)
- [Data Quality in Imitation Learning (2023)](https://arxiv.org/pdf/2306.02437)
