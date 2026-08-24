# What Transfers from LLM Data-Acquisition to Robot Data Prescription

**Deliverable of the 2026-08-13/14 "LLM borrow" session.**
Artifacts: `/data/xinyua11/robocasa/gradient_analysis/llm_borrow/` (code, JSON results, logs, pre-registrations).
Nothing outside that directory was written. The bandit ledger was read-only. No process this session did not
start was killed; the two concurrent pi0 trainings on GPU0/GPU1 never dropped below 76.8 GB of their own
memory. No new pi0 fine-tune was launched. All GPU work here was CIFAR-scale (~2.6 GB/job) plus forward
passes over already-existing pi0 checkpoints.

---

## 0. Glossary — every term used below, defined once

Read this first if you were not in the session. Nothing later relies on undefined shorthand.

| Term | Definition |
|---|---|
| **π₀ (pi-zero base)** | The deployed base policy under study: a pi0 flow-matching vision-language-action model, LoRA-fine-tuned on the RoboCasa `PickPlaceCounterToSink` task. Concretely the checkpoint `pi0_ppc2sink_pi0base/pi0_v1/19999`. Its measured closed-loop success is **51.33 %**. |
| **Closed-loop success** | Fraction of *rollouts* (full simulated task attempts, robot acting in the loop) that end in task success. This is the deployed metric and the only thing we are ultimately scored on. It is NOT a loss. |
| **Frozen eval set E** | 150 saved scene start-states × 3 repeats = **450 rollouts**, identical for every measurement, so comparisons are paired on scenes. |
| **Pool** | 9,885 existing demonstrations available to draw from. **D0** = 400 fixed demos included in every fine-tune. |
| **Arm** | A rule for choosing *which data to add*. Examples used here: `random` (uniform draw), `mid_band` / `easy_band` / `tall_vessel_grasp_fail` (draw from a predicted-difficulty band or a failure region), `style_hi` / `style_lo` (draw from the top/bottom execution-quality tail), `gradarm_a/b`, `gc0…gc5` (gradient-cluster arms), `survivor_mix`, `null` (add nothing), `planted_bad` (deliberately corrupted control). |
| **Pull** | One execution of an arm: draw 200 demos, LoRA-fine-tune π₀ on D0 + those 200, evaluate on E. **Cost 5–9 GPU-hours.** |
| **Δ (delta)** | A pull's closed-loop success minus the 51.33 % baseline, in **percentage points (pp)**. |
| **σ_e = 3.3 pp** | The measured per-pull noise floor on Δ. Established by null arms (+2.44 and −0.89 pp for two pulls that added *nothing*). Decomposed (`gradient_analysis/NOISE_ANATOMY.md`): rollout/eval stochasticity 0.88 pp < checkpoint-choice-within-a-run ≈ 2.4 pp < training-seed 3.3 pp; **data composition — the quantity we are trying to measure — is the smallest term.** |
| **Target stratum** | A subset of E defined by scene properties (e.g. the "hard"/tall-vessel band). Per-stratum Δ is noisier than the 450-rollout Δ because the stratum is a subset. |
| **Paired seeds** | Arms within a round share the same training seed (seed = 1000 + round). Required by constraint 4 below. |
| **FTE** (wind-tunnel currency) | "Full-training equivalent". **1 FTE = one full pull** = 2000 training steps at batch 128 + one full evaluation. A forward pass on *n* examples costs `n/(128·3)` steps. Calibrated against wall-clock: the cost model says eval is 3.9 % of a pull, measured 2.3 s / 53 s = 4.3 %. |
| **Wind tunnel** | `wind_tunnel.py`: a metered simulator in which allocation algorithms compete under a fixed FTE budget. The environment is the only way to spend; every operation is charged before it does work, including cache hits (otherwise an allocator could farm the cache for free). 25 unit tests, zero GPU, all passing before and after the races. |
| **Regret** | `best_arm_value − value_of_the_arm_the_allocator_chose`, in pp. Lower is better; 0 = picked the best arm. |
| **P(correct)** | Fraction of independent replicates in which the allocator's top-weighted arm is the best arm. |
| **B95** | Smallest budget (FTE) at which the allocator's mean decision value reaches 95 % of the oracle's. |
| **The random floor** | An allocator that spends its whole budget measuring arms uniformly at random and picks the empirical best. This is the honest baseline: it reads the true (expensive) outcome, it just does not prioritize. **Distinct from a random *pick*** (choosing an arm with no measurement at all), which is much worse. |
| **c (decoupling coordinate)** | Correlation between a *cheap* signal and an arm's *true* value. Built as `s = c·z_truth + √(1−c²)·z_orth` so the Pearson correlation is exactly `c`. **c = 1** perfect transfer; **c = 0** the cheap signal measures something real but orthogonal to what we are scored on; **c < 0** anti-correlated. |
| **SNR** | Spread of the cheap signal across arms ÷ standard deviation of the noise on one observation of it. |
| **AUC** | Area under the ROC curve for a score separating two labelled groups; 0.5 = no separation. Always reported here against its own **permutation null floor** (the AUC the same statistic reaches on shuffled labels), because max-over-modes statistics inflate the null above 0.5. |
| **Reliability / attenuation ceiling** | If a measurement has noise, the maximum correlation it can show with anything is `√(reliability)`. Reliability here = `1 − σ_e²/SD(Δ)²`. When arm-to-arm spread is below σ_e, reliability is 0 and **no predictor can correlate with Δ at all** — a fact that decides several results below. |

### The five measured constraints any borrow had to survive

These are prior results of this project, not assumptions.

1. **F4 — training-loss effects do not transfer to closed-loop success.** A one-step loss proxy passed its check while the resulting fine-tune was the worst arm ever run.
2. **A pull costs 5–9 GPU-h against a ±3.3 pp noise floor.** "Many cheap pulls" requires naming what a cheap pull *is* on this stack.
3. **Cheap short training-burst proxies (≈1.5k steps) mis-rank** — measured.
4. **The seed, not the data, steers the fine-tune.** Same-seed weight updates from completely different 200-demo draws are **0.969–0.979** cosine-aligned; same-data/different-seed is 0.505; a bit-identical rerun is 1.0000. The data's weight-space fingerprint is ≈1.5 %. Unpaired-seed single-run comparisons are void.
5. **The flow-matching loss gradient does not encode scene semantics / region** (best-mode AUC 0.577 vs its own shuffle-null floor 0.598) **but does encode action/execution quality** (AUC 0.778). A signal must live where the loss can see it.

---

## 1. Executive summary

**Almost nothing transfers, and tonight we measured why instead of arguing it.** Every LLM data-acquisition
method in the sweep allocates on a *loss* signal — the training loss of the sampled domain (ODM), a
before/after loss difference (Graves' learning progress), per-skill validation loss (Skill-it), a fitted
loss-vs-tokens curve (ADO), a validation-loss response matrix (Aioli, and by its own reduction DoReMi/DoGE/Data
Mixing Laws), a short proxy-run loss (RegMix), or a two-model loss gap (JEST / RHO-LOSS) — and on this stack that
channel is measurably orthogonal to the deployed metric: over the 12 non-null pulls with a loss probe,
Spearman(loss on the target set, realized closed-loop Δ) = **−0.06 (p = 0.85)**, and the loss reward would have
funded the `tall_vessel_grasp_fail` arm in **3 of 3** rounds by a stable margin -- an arm whose realized mean
(+2.96 pp) ranks second of four and is itself statistically indistinguishable from every other arm
(`mid_band` +3.63, `random` +2.59, `easy_band` +2.07; all inside the sigma_e = 3.3 pp per-pull noise floor).
The failure is therefore not "it picks the loser" but the sharper "it commits confidently, every round, among
arms it cannot actually distinguish, with ~zero rank correlation to the outcome"; JEST/RHO-LOSS learnability is likewise null on the failure region (AUC 0.484 vs a
0.598 shuffle-null floor, n = 280 demos) and its within-round top-1 pick is **0.111 correct over 9 rounds against
0.280 chance**, because the single most "learnable" arm is `style_lo`, the one arm measured to *hurt* (−2.41 pp
mean). A metered allocator race (`wind_tunnel.py`, 25 unit tests, 1000 replicates per synthetic cell plus ~250
real CIFAR fine-tunes) then mapped the general condition: a borrowed method needs **c ≳ 0.6–0.9 and per-observation
SNR ≳ 4** to beat even a random-allocation floor, our measured operating point is **c ≈ 0 with a confidence
interval spanning negative values and an outcome-side spread/noise ratio of 0.356**, and at c ≈ 0 a *perfectly
sharp* cheap signal is still **1.8–2.5× worse than the floor** because a confident wrong pick beats nothing and
loses to measurement — while anti-correlation (which our interval does not exclude) costs **6.9×**. What survives
is small and specific: (i) **successive halving on the rollout reward**, the only allocator whose performance is
flat in c because it never reads a cheap channel — adopted, though it is weak in absolute terms here (P(correct)
0.41 → 0.63 from 6 → 24 FTE, never reaching 95 % of oracle by 48 FTE); (ii) the **shape** "cheap signal
*shortlists*, expensive outcome *decides*", which buys 0.2–0.4 of tolerance on the c axis (measured ladder: pure
JEST needs c ≥ 0.80, shortlist-then-confirm with a top-2 list c ≥ 0.60, top-3 c ≥ 0.35, top-4 c ≥ 0.20 at SNR 8) —
worth building only once a channel clears the gate; (iii) three portable pieces of machinery with their loss
rewards removed — Graves' **adaptive quantile rescaling** of rewards, Skill-it's **skills-graph A** re-defined with
Δ*success* entries, and Aioli's **fixed pre-registered δ of budget spent measuring the response matrix** plus its
independently-obtained negative result that no mixing method consistently beats stratified sampling. The single
most actionable output is not an allocator but a **test**: `f4_signal_probe.py` measures a candidate cheap
signal's c and SNR in **385 seconds of GPU time**, and from now on any proposal of the form "use loss progress /
learnability / a 1.5k-step proxy" should be required to report its measured c *before* an allocator is built
around it — on this stack the equivalent robot measurement already exists and reads c ≈ 0.

---

## 2. Verdict table

Verdicts: **ADOPT** = use as-is; **ADAPT** = a named component is portable once its loss reward is replaced;
**REJECT** = the method's core signal is measured non-functional here.

| # | Method (citation) | Reward signal it allocates on | Verdict | Deciding constraint | Evidence measured this session |
|---|---|---|---|---|---|
| 1 | **ODM — Efficient Online Data Mixing** (Albalak, Pan, Raffel, Wang; arXiv:2312.02406) | Raw training loss of the sampled domain's own micro-batch, `R = L(f,x,y)`, as an importance-weighted EMA inside EXP3 with exploration floor `ε_t = min{1/K, √(ln K/(K t))}` | **REJECT** | **1 (F4)** primary; **2** compounding | `reward_signal_test.json`: n = 12 non-null pulls, Spearman(loss_target, Δ) = **−0.060, p = 0.854**; Pearson −0.118, p = 0.714. The reward picks `tall_vessel_grasp_fail` in **3/3** rounds (loss_target 0.440/0.452/0.470 vs 0.487–0.524 for all others — a gap 5–10× the between-arm spread), yet arm means are `mid_band` +3.63 > `tall` +2.96 > `random` +2.59 > `easy` +2.07 pp (n = 14 pulls, σ_e = 3.3 pp, so these means are themselves within noise of one another). Wind tunnel, robot regime, 12 FTE, 1000 reps: EXP3-on-loss regret **0.914 pp vs random floor 0.568 pp**, P(correct) 0.00. Its exploration floor at our scale (K = 4 arms, T ≈ 14 pulls) evaluates to ε = 1/K, i.e. **the algorithm reduces to uniform allocation** — a proof that the allocator cannot matter at this budget. |
| 2 | **Graves et al. — Automated Curriculum Learning** (arXiv:1704.03003), Exp3.S over tasks with 8 "learning-progress" rewards | Prediction Gain `L(x,θ)−L(x,θ′)`, Gradient Prediction Gain `‖∇L‖²`, self/target/mean PG, complexity gains; all per training batch | **ADAPT** — keep **adaptive quantile rescaling** and the *concept* of target progress; reject all 8 reward instantiations | **1** and **3** kill the rewards; **2** kills the cadence (per-batch → per-pull is a 10⁵ change of regime) | **Test B** (below): the closest robot analogue of Target Prediction Gain, measured on 24 pulls with a step-5000 checkpoint — Spearman = **+0.292 (p = 0.170)** over all 24, **+0.457 [+0.057, +0.715], p = 0.057** on the 18 content-only pulls, but the outcome side of that group has SD(Δ) = 2.54 pp **< σ_e = 3.33 pp → reliability 0.000 → maximum attainable |ρ| for *any* predictor is 0.000**. Within-round paired sign agreement 0.557 (chance 0.5); within-round top-1 pick by most progress **0.000 correct over 3 rounds** (chance 0.139), mean regret +3.56 pp. Gradient Prediction Gain analogue (mean gradient norm of a draw at π₀ vs Δ, n = 16): Spearman **+0.069, p = 0.799**. Note the authors' own negative result — GPG was worse than uniform on two of their tasks — which anyone citing gradient-norm-as-progress must cite too. |
| 3 | **Skill-it!** (Chen, Roberts, Bhatia, Wang, Zhang, Sala, Ré; arXiv:2307.14430) — *full-information online mirror descent, not a bandit* | Per-skill **validation loss** observed for all k skills every round, routed through a learned skills graph `A_ij` = loss drop on skill j after training on skill i | **ADAPT** — take the **skills-graph A**, discard the algorithm | **2** is fatal as stated (full information = k evaluations/round = 4 × 5 GPU-h = 20 GPU-h of pure measurement per round, before training); **1** kills the entries | Not exercised tonight — it is build item #2 below. Two reasons it must be rebuilt rather than imported: (a) entries must be Δ*success* per stratum, never Δloss; (b) a loss-estimated A would be **diagonal by construction**, because a pull's loss on region R is dominated by whether it trained on R-like demos (absorption: every pull's own 200 demos collapse from grad-norm 3.2 to 0.4 by step 5000) rather than by whether closed-loop success on R improved — the opposite of the measured behavioural A, in which the tall-vessel arm scored **−2.7 pp on its own target stratum** while helping easy/mid (+13.4 / +4.4 pp in round 4; **n = 1 pull per cell, per-stratum noise exceeds σ_e**, so this is a hypothesis to test, not a finding). |
| 4 | **ADO — Adaptive Data Optimization** (Jiang, Zhou, Feng, Malladi, Kolter; arXiv:2410.11820) | Online per-domain scaling-law fit `L̂(n)=ε+βn^{−α}` to the domain's **training-loss history**; preference ∝ remaining reducible loss `α·(L̂(n)−ε)` | **REJECT** (keep only the *shape* "per-axis dose-response drives allocation", better cited to FSC 2505.07728) | **5 + 1**: its state variable is extinguished here; plus a structural dose-model failure | Prior measurement (Q2): every pull absorbs its own draw by step 5000 (own-draw gradient norm 3.2 → 0.38–0.41 = the always-trained-D0 level) and stays there through step 19999, **for every arm regardless of what it taught**, so "remaining reducible loss" → ~0 identically while closed-loop outcomes differ by ±7 pp. Separately, ADO's power law is **monotone** and therefore structurally cannot represent the measured overdose harm (610 added demos → 0.299 overall success vs 200 demos → 0.371; ~80 % of the 200-demo gain is already realized at 14 demos). A method that cannot express "more of this data hurts" cannot allocate here. |
| 5 | **Aioli** (Chen, Hu, Lourie, Cho, Ré; ICLR 2025, arXiv:2411.05735) — unifying **LMO** framework | Measures the response matrix `A^t` online by spending a fraction **δ ≈ 0.1–0.2 of every round** on one-hot mixture sub-runs and reading validation-loss differences | **ADAPT** — take (a) the **negative result as a citation**, (b) the **LMO reduction**, (c) the **δ budget discipline**; reject the loss instantiation | **2** (δ of a round = 4.5 GPU-h < one pull, and one pull carries ±3.3 pp) and **1** | Not exercised tonight. Its headline — *"no existing method consistently outperforms a simple stratified sampling baseline in terms of average test perplexity"* — is external, independent corroboration of our own central null (`random` tied the best selected arm twice; core − random = +2.0 pp, not significant), obtained in a domain where the loss *does* encode the target and pulls are essentially free. Its LMO reduction (DoReMi, DoGE, Skill-it, Data Mixing Laws are one multiplicative update differing only in how `A` is estimated) is what licenses this report's framing: **the contribution cannot be "a better allocator"; it must be "a better A-estimator, built on the deployed metric."** Adversarial caveats to respect: ODM appears only in Aioli's related work and is never benchmarked (do not write "Aioli refutes ODM"); Aioli's Table 2 reports no error bars or seed variance, so its 0.274-ppl win would not clear our own pre-registered confidence-bound rule. |
| 6 | **AC-ODM** (Ma, Dang, Liao; arXiv:2505.23878, preprint, no venue found) | **Gradient alignment**: `W_i = ⟨∇ℓ_i(θ), Σ_{j≠i} ∇ℓ_j(θ)⟩`, EMA-stabilized, driving a DDPG actor–critic over domain weights | **REJECT** | **5** (encoding gate), then **1** | Constraint 5 is exactly this method's premise and it fails here: region membership is not usably encoded in the π₀ LoRA flow-matching gradient (best-mode AUC 0.577 vs the statistic's own shuffle-null floor 0.598, n = 240; split-half contrast 0.572 vs 0.501 shuffled), and encoding never emerges during training (re-measured at steps 5000/10000/19999 of an arm trained on 200 in-region demos: 0.54–0.59 throughout). No π₀-time gradient statistic predicts Δ (n = 16 pulls: cosine-to-region ρ = +0.038 p = 0.929; mean gradient norm ρ = +0.069 p = 0.799; batch-separation ρ = +0.227 p = 0.399; best |ρ| = 0.23). **Caveat: the sweep entry for this method arrived truncated in my inputs**, so this verdict rests on our own measured gradient nulls, which are decisive for any gradient-alignment reward regardless of the method's remaining details. |
| 7 | **DoReMi / DoGE / Data Mixing Laws** (subsumed) | Group-DRO reweighting on proxy loss / gradient-alignment domain weights / parametric loss-vs-mixture laws | **REJECT as allocators** (subsumed by row 5's reduction) | **1**, and **3** for the proxy-run variants | Not separately tested. They are the same multiplicative update as rows 3 and 5 with different `A`-estimators; every estimator listed is loss-based, so row 1's measurement (Spearman −0.06 on n = 12) applies to all of them. Cite them as the family, not as separate candidates. |
| 8 | **JEST** (Evans, Parthasarathy, Merzic, Hénaff; arXiv:2406.17711 — *citation re-verified by fetching the abstract page this session*) and **RHO-LOSS** (Mindermann et al., **ICML 2022**, arXiv:2206.07137 — *also re-verified*) | **Learnability** `S(z) = L(z; current/π₀) − L(z; reference/stronger model)`: prioritize data the model finds hard but a stronger model finds easy | **REJECT as an arm-selector**; **ADAPT as a data-quality detector** | **1** for selection; **5** explains the direction of the failure | **Test A** (below), the largest single measurement of the session. On the failure region: AUC(in-region > out) = **0.484** (reference 1), **0.470** (reference 2), raw π₀ loss 0.570 — all at or below the shuffle-null floor 0.598 (n = 280 demos × 8 noise/time draws). On the execution-quality axis: **0.752** for the raw π₀ loss, 0.650 / 0.701 for the learnability difference, against a permutation-null 95th percentile of 0.594 — i.e. the signal is real and it is about *quality*, not about *region*, exactly as constraint 5 predicts. Pull-mean learnability vs realized Δ: n = 30 content-only pulls, ρ = **+0.143 [−0.268, +0.509], permutation p = 0.449**. The decisive exhibit is not the correlation: ranked by learnability, the **most learnable arm is `style_lo`** — the bottom execution-quality tail, the one arm measured to reliably hurt (mean −2.41 pp across 5 pulls). Within-round top-1 pick by highest learnability: **0.111 correct over 9 rounds (chance 0.280), mean regret +4.79 pp**; picking the *least* learnable arm scores 0.444. JEST's noise-rejection does fire on the catastrophic control (`planted_bad`, −20 pp, sits at the 7th percentile of learnability) but it funds the mildly harmful arm. Also measured: the RHO-LOSS *subtraction* is unstable — re-measuring the quality AUC with each of 36 retained checkpoints as the reference gives mean 0.761, **sd 0.139, range [0.281, 0.969]**, whereas the reference-free raw π₀ loss is the stable detector. |
| 9 | **RegMix** (arXiv:2407.01492, ICLR 2025) and **Data Mixing Laws** (arXiv:2403.16952, ICLR 2025) | k small **proxy fine-tunes** on random Dirichlet mixtures → regression (ridge / parametric law) → argmax mixture at full scale | **REJECT on this stack**; the *pipeline shape* (probe → predict → allocate) is retained and is what PRESCRIPTION_CRITERION §2 already specifies | **3 (proxy mis-ranking)** — a *different* disease from F4 | Wind tunnel, robot regime, 1000 reps: regret **0.848 pp / P(correct) 0.10** at every budget from 4 to 48 FTE, versus a random floor of 0.568 (12 FTE) / 0.437 (24 FTE). Crucially, RegMix's regret is **invariant across all 72 (c, SNR) decoupling cells (1 distinct value)** — it reads pull outcomes, not loss, so its failure here is proxy bias, not F4. Where proxies *do* rank correctly it is excellent: on real CIFAR fine-tunes it reaches regret 0 from 3 FTE with B95 = **2.51 FTE** versus a random floor that needs > 24 FTE. |
| 10 | **Successive halving / Hyperband** (the non-LLM incumbent; rollout reward, paired seeds, doubling replicates) | The **deployed metric itself** — measured closed-loop Δ | **ADOPT** (it is the incumbent and nothing beat it at our operating point) | Survives all five by never reading a cheap channel | Wind tunnel, robot regime, 1000 reps: regret 0.633 / 0.420 / 0.318 pp and P(correct) 0.41 / 0.55 / 0.63 at 6 / 12 / 24 FTE, versus the random floor's 0.680 / 0.568 / 0.437 and 0.37 / 0.42 / 0.53 — a paired win from 8 FTE upward and the **only** method that converts budget into accuracy (P(correct) 0.21 at 1 FTE → 0.76 at 48 FTE). It is **exactly invariant** across all 72 decoupling cells. Two honest costs: it never reaches 95 % of oracle within 48 FTE (0.947 at 48), and it **cannot spend a budget below one full paired round** — on real CIFAR at 1.5 FTE it spends 0.00 FTE and guesses (P(correct) 0.40). |

---

## 3. The transfer-condition map

### 3.1 What was varied, and the leak check that licenses the map

The wind tunnel was modified along **two axes, applied only to the cheap-signal channel** (held-out loss
progress and JEST learnability). The outcome channel — a pull's realized Δ — was left faithful, because that is
our actual situation: **rollouts are expensive but true; the loss is cheap and may not transfer.**

- **c** as defined in §0, with each cell pooled over **8 independent nuisance directions** (32 in one
  robustness run) so no cell is an artefact of one particular way of being wrong.
- **SNR** as defined in §0.

Allocators that read only outcomes must be bit-identical across the whole grid. They are:

| Allocator | Distinct mean-regret values across all 72 (SNR, c) cells |
|---|---|
| successive_halving | **1** (0.4062 pp) |
| random floor | **1** (0.5309 pp) |
| regmix | **1** (0.8308 pp) |
| oracle | **1** (0) |

`regmix` moving would have voided everything in this section. It does not.

### 3.2 Where our robot setting sits on the c axis (measured, not assumed)

`reward_signal_test.json`, the direct robot measurement of the LLM reward: Spearman(loss reward, realized Δ) =
**−0.06 over the 12 non-null pulls** and **+0.25 over all 14**, neither significant (p = 0.85 and p = 0.39); the
within-round rank correlation of (−loss) against Δ **flips sign across rounds** (Kendall τ = −0.18, +0.67, −0.33
for rounds 3, 4, 5, each n = 4 arms). **Our point estimate is c ≈ 0 and the interval spans negative values.**

On the SNR axis the anchor is **0.356** — the ratio of the arm-mean spread (0.654 pp across the four race arms)
to the per-pull observation noise (1.837 pp), from `decision_accuracy_diagnostic.json`, n = 14 pulls.
*(Honest caveat, expanded in §6: 0.356 is the **outcome-side** ratio used as a stand-in for the cheap channel's
own SNR, which was not separately measured on the robot.)*

### 3.3 Map A — lowest c at which each method still beats the **random floor** (robot regime, 12 FTE, 1000 reps/cell, paired bootstrap CI95 < 0)

"never" = does not beat the floor at any c ∈ [−1, 1] tested.

| Allocator | SNR 8 | SNR 4 | SNR 2 | SNR 1 | **SNR 0.356 (ours)** | SNR 0.1 |
|---|---|---|---|---|---|---|
| **successive_halving** (rollout reward) | **≤ −1.00** | **≤ −1.00** | **≤ −1.00** | **≤ −1.00** | **≤ −1.00** | **≤ −1.00** |
| jest_then_confirm, shortlist k=2 | 0.20 | 0.35 | 0.50 | never | **never** | never |
| jest_then_confirm, k=3 | 0.20 | 0.20 | 0.35 | 0.60 | **never** | never |
| jest_then_confirm, k=4 | 0.20 | 0.20 | 0.70 | never | **never** | never |
| jest (learnability, no confirm) | 0.80 | 0.80 | never | never | **never** | never |
| learning_progress_exp3 | 0.80 | 0.90 | never | never | **never** | never |
| learning_progress_ucb | 0.80 | 0.80 | 0.80 | 0.90 | **never** | never |
| regmix | never | never | never | never | **never** | never |

### 3.4 Map B — same, but measured against our **incumbent** (successive halving), which is the decision-relevant comparison

| Allocator | SNR 8 | SNR 4 | SNR 2 | SNR ≤ 1 |
|---|---|---|---|---|
| jest | 0.90 | 1.00 | never | never |
| jest_then_confirm k=2 | 0.60 | 0.80 | never | never |
| jest_then_confirm k=3 | 0.60 | 1.00 | never | never |
| learning_progress_ucb | 0.90 | 0.90 | 1.00 | never |
| learning_progress_exp3 / k=4 / regmix | never | never | never | never |

**Where we sit: c ≈ 0, SNR = 0.356. Every borrowed method is in the "never" column against both references, at
every budget tested (6, 12 and 24 FTE).** Successive halving beats the floor everywhere on the map by
construction — it never reads the corrupted channel.

To make the shape concrete, the raw robot-regime mean regrets at 12 FTE (pp; random floor = 0.531, successive
halving = 0.406 at every c):

| c → | −1.0 | −0.6 | −0.3 | 0.0 | 0.35 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| jest @ SNR 8 | 3.630 | 2.762 | 1.811 | 0.963 | 0.619 | 0.558 | 0.454 | 0.118 |
| jest @ SNR 0.356 | 1.583 | 1.479 | 1.396 | 1.353 | 1.291 | 1.217 | 1.154 | 1.114 |
| learning_progress_ucb @ SNR 8 | 3.630 | 2.606 | 1.680 | 0.703 | 0.607 | 0.593 | 0.460 | 0.003 |

Read the middle row: **at our SNR, even a perfectly transferring cheap signal (c = 1.0) is worse than the random
floor** (1.114 vs 0.531 pp), because a method that spends its budget on cheap looks instead of measurements cannot
resolve a 0.65 pp arm spread against 1.84 pp per-pull noise.

### 3.5 Noise is not wrongness — and the asymmetry runs the other way from the intuition

Normalized regret, where 1.00 = the random-*allocator* floor (0.53 pp). For scale: a uniform random *pick* with
no measurement at all scores 2.6; always choosing the worst arm scores 6.9. Robot regime, 12 FTE, 1000 reps.

| Method | c = 1, SNR 8 | c = 1, SNR 0.1 (pure noise) | c = 0 (orthogonal) | c = −1 (anti-correlated) |
|---|---|---|---|---|
| jest | 0.22 | 2.41 | 1.81 → 2.55 | **6.85** |
| learning_progress_ucb | 0.01 | 1.82 | 1.32 → 2.49 | **6.85** |
| learning_progress_exp3 | 0.74 | 2.43 | 2.04 → 2.56 | 6.4 |
| successive_halving | 0.77 | 0.77 | 0.77 | 0.77 |

**Corrected statement of the asymmetry** (my pre-registered version of it was half wrong, see §3.6 item 15):
noise and orthogonality are the *same* failure — both collapse a cheap method to a random *pick*, which is
2.4–2.6× worse than the random *allocator* floor, because the floor at least measures. Only anti-correlation is
categorically worse: it drives the method to the *worst* arm at 6.9× the floor. **Since our measured Spearman
interval spans negative values, we cannot currently rule out being in the anti-correlated half.**

### 3.6 The one shape that extends the tolerance: shortlist cheap, decide expensive

Measured c\* ladder, CIFAR-like regime at SNR 8 (lower is better — it survives more decoupling):

`jest` (cheap signal decides) **0.80** → `jest_then_confirm` with a top-2 shortlist **0.60** → top-3 **0.35** →
top-4 of 5 arms **0.20**. Robot regime at SNR 4: `jest` 0.80 → `jest_then_confirm` **0.35**.

I pre-registered that the confirm stage would be useless ("the confirm stage cannot repair a wrong shortlist").
**That was wrong, and it is the most useful miss of the study**: the confirm stage buys 0.2–0.4 of c. But note
what the ladder means at the bottom: at c ≈ 0, a screen is simply "discard arms at random", and it is harmful in
proportion to how many arms it discards — which is why the barely-screening k=4-of-5 variant is the only one still
tied with the floor at low SNR.

### 3.7 Real-GPU confirmation of the whole chain (no synthetic knobs)

`f4_signal_probe.py` and `CifarF4Backend`, on real CIFAR-100 fine-tunes where "truth" = clean rare-class test
accuracy and the arms have the same names as the robot race. **385 s of GPU time** for the channel measurement
(3 seeds × 6 arms × 500-step bursts).

**(i) The reward channel itself:**

| Corruption of the cheap channel | Loss-progress: Spearman / Pearson vs truth | argmax (truth = `rare`) | JEST channel: Spearman | JEST argmax |
|---|---|---|---|---|
| clean | **+1.000 / +0.964** | `rare` (3/3 seeds) | +0.90 | `rare` |
| **off-target probe (α = 1: probe slice drawn from well-learned classes)** | **+0.029 / −0.259** | **`easy`** (3/3 seeds) | n/a | – |
| label noise ρ = 0.25 | +1.000 / +0.945 | `rare` | +0.90 | `rare` |
| label noise ρ = 0.50 | **+1.000** / +0.878 | `rare` | +0.60 | `rare` |
| label noise ρ = 1.00 | +0.486 / +0.409 | `random` | **−0.70** | `gradarm_a` |

Two mechanisms, cleanly separated: **label noise is a far weaker attack on loss progress than intuition
suggests** (at ρ = 0.5 the ranking is still perfect — the loss drop still tracks class coverage), while the
**off-target probe — a signal that is real, low-variance, and simply about the wrong thing — destroys it at
α = 1.** JEST behaves oppositely: label noise *inverts* it at ρ = 1, because noisy data looks maximally
"learnable". Off-target-ness, not noise, is the shape of F4.

**(ii) The allocator race under that corruption** (6 FTE, 5 replicates, real fine-tunes):

| Allocator | aligned (ρ=0, α=0) | **off-target (α = 1)** |
|---|---|---|
| learning_progress_exp3 | 1.42 ± 0.89 pp, P(correct) 0.60, picks `rare` 3/5 | **3.68 ± 0.31 pp, P(correct) 0.00, picks `easy` 3/5** |
| jest | 0.00, P 1.00 | 0.00, P 1.00 (this mechanism has no JEST channel — an invariance check, not a win) |
| jest_then_confirm / k=3 | 0.00, P 1.00 | – |
| successive_halving | 0.00, P 1.00 | invariant by construction |
| regmix | 0.00, P 1.00 | invariant |
| **random floor** | 1.88 ± 0.77 pp, P 0.40 | 1.88 |

The chain closes end-to-end on real hardware: corrupt the reward channel → measured c falls 1.00 → 0.03 → the
allocator flips from *better than the floor* (1.42 pp) to *twice as bad as the floor* (3.68 pp), choosing the
second-worst arm in truth. **n = 5 replicates, so the 1.80 pp gap to the floor is ≈ 2 standard errors** — the
pick distribution (0/5 vs 2/5 correct) is unambiguous but this is confirmation of a 1000-replicate synthetic
prediction, not independent evidence on its own.

### 3.8 Pre-registration scorecard for the wind tunnel (`PREREGISTRATION.json`, written before the first race)

18 point predictions, scored verbatim. **11 HIT, 5 MISS, 2 PARTIAL.**

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Real CIFAR at 6/12/24 FTE: all allocators regret 0, table uninformative | False for the floor (random P = 0.40 @ 6 FTE, 0.60 @ 12) and for learning-progress (1.42 pp @ 6) | **MISS** |
| 2 | CIFAR @ 1.5 FTE: jest / jest-then-confirm win, P ≈ 0.9–1.0 | 1.00 [0.57, 1.00] both | HIT |
| 3 | CIFAR @ 1.5: successive halving ≈ 0.33 | 0.40 [0.12, 0.77] | HIT |
| 4 | CIFAR @ 1.5: random ≈ 0.17 | 0.20 | HIT |
| 5 | CIFAR @ 1.5: learning-progress 0.6–0.8 | 0.40 | MISS (low) |
| 6 | Robot: SH P(correct) 0.30 / 0.40 / 0.50 at 6 / 12 / 24 FTE | 0.41 / 0.55 / 0.63 | HIT (I was pessimistic) |
| 7 | Robot: LP and jest P(correct) = 0, converging confidently on `tall_vessel` | 0.00, `tall_vessel` 100 % of reps | HIT |
| 8 | Robot: P(correct) and regret disagree — a confidently wrong method can still beat the floor on regret | True at ≤ 2 FTE, false from 12 FTE on | **PARTIAL** (right mechanism, wrong budget range) |
| 9 | Robot: jest_then_confirm ≡ jest; a confirm stage cannot repair a wrong shortlist | jtc P = 0.15–0.20 vs jest 0.00; confirm buys 0.2–0.4 of c | **MISS — the most useful miss** |
| 10 | Robot: regmix P(correct) ≈ 0.10 | 0.10 exactly | HIT |
| 11 | Robot: nobody reaches B95 anywhere on the grid | Correct, up to 48 FTE | HIT |
| 12 | Decoupling: SH / random / regmix exactly invariant | 1 distinct value across 72 cells | HIT |
| 13 | LP degrades as a step; c\* between 0.6 and 0.3 | c\* = 0.80 at SNR 8 — degrades **earlier** than predicted, and smoothly | MISS |
| 14 | jest_then_confirm survives ≈ 0.2–0.3 lower c than jest | Ladder 0.80 → 0.60 → 0.35 → 0.20 | **HIT** |
| 15 | Noise pushes methods toward the floor; decoupling pushes them below it | Wrong on the first half (pure noise → random *pick* = 2.4–2.6× the floor, not to the floor); right on the second (c < 0 → 6.9×) | **PARTIAL** |
| 16 | Real CIFAR α=1: LP picks `easy`, regret ≈ 4.2 pp, P = 0 | Picks `easy` 3/5, regret **3.68 ± 0.31 pp**, P = **0.00** | HIT |
| 17 | Real CIFAR ρ=1: LP degrades to a random picker, not to the worst arm | Channel Spearman +0.486 (not 0), argmax → `random` | HIT in direction |
| 18 | ρ = 0.5: LP partial, P(correct) 0.4–0.7 | Channel Spearman still **+1.000** — 50 % label noise does not break loss progress | **MISS** |

---

## 4. Robot-signal tests, with pre-registered predictions beside outcomes

Two tests were pre-registered in `prereg_AB.json` **before** any measurement and are scored verbatim here. Both
ran as forward passes over already-existing checkpoints (no new fine-tunes). Common random numbers were used —
the (preprocess, noise, timestep) random draw depends only on (episode, draw index) and never on the checkpoint —
so every score is a *paired* difference: raw single-draw loss SD 0.1018 → paired-difference SD 0.0584, a 1.7×
variance reduction. Parameter-dtype control: checkpoints load as bfloat16 (the model's own compute dtype) because
fp32 does not fit the 0.12 JAX memory cap; reloading in float16 gives corr(S) = **0.9998**, mean |ΔS| = 0.0012
against SD(S) = 0.092, i.e. quantization noise is 1.3 % of the signal.

### 4.0 The prior test that framed both (`test_llm_rewards_on_robot.py`, zero GPU)

| Quantity | Result |
|---|---|
| Spearman(loss on target set, realized Δ), 12 non-null pulls | **−0.060, p = 0.854** (Pearson −0.118, p = 0.714) |
| Same, all 14 pulls including nulls | +0.251, p = 0.387 |
| Which arm an ODM/Graves-style loss reward funds | `tall_vessel_grasp_fail` in **3/3** rounds |
| That arm's realized mean Δ vs `random`'s | +2.96 vs +2.59 pp — a **+0.37 pp edge, ≈ 1/9 of σ_e = 3.3 pp**, i.e. **not a finding, a tie** |
| Graves' Gradient Prediction Gain analogue (mean gradient norm), n = 16 | Spearman +0.069, p = 0.799 |

### 4.1 TEST A — JEST / RHO-LOSS learnability `S(z) = L(z; π₀) − L(z; reference)`

**Setup.** π₀ as defined in §0. Reference 1 = `gradarm_b_j3` at step 19999 — chosen because "a stronger model"
must mean *deployed success*, and this is the best pull ever run (+7.78 pp). Reference 2 = `mid_band_j4` at 19999
(+3.78 pp, **different training seed 1004**), included specifically as the constraint-4 control. All 794 demos
that appear in either reference's training draw or in D0 were excluded from every scored set, so **absorption
cannot manufacture the score**. 280 diagnostic demos × 8 noise/time draws; 933 pull-sample demos × 2 draws.

**Estimator stability (asked for explicitly, answered before any hypothesis test).** At 8 draws:
SE/between-demo-SD = 0.105 for the raw loss and 0.160 for `S`; split-half r = 0.963 / 0.929 → Spearman–Brown
reliability **0.981 / 0.963.** Eight draws is more than enough; four would do.

| | Pre-registered | Measured | Verdict |
|---|---|---|---|
| **(i) Execution-quality axis** — does S separate the bottom quality tail (`style_lo`) from the top (`style_hi`)? | raw-loss AUC ≥ 0.70 (falsifier < 0.60); S weaker but ≥ 0.60 | raw L(π₀) AUC = **0.752**; S(ref1) 0.650; S(ref2) 0.701; raw L(ref1) 0.760. Permutation-null 95th pct ≤ 0.594 | **Both predictions met** |
| **(ii) Failure region** — does S separate in-region from out-of-region demos? | **NULL**, AUC ∈ [0.45, 0.60] | raw **0.570**, S(ref1) **0.484**, S(ref2) **0.470** — at or below the shuffle-null floor 0.598 | **Confirmed null** |
| **(iii) Does a pull's mean learnability predict its realized Δ?** | **NULL** on content-only arms, \|ρ\| < 0.30 with CI covering 0; any correlation appearing only when quality arms are pooled in is a *quality confound*, to be reported as confounded | content-only **n = 30: ρ = +0.143 [−0.268, +0.509], perm p = 0.449**; all 41 pulls ρ = +0.017, p = 0.917; quality-arms-only n = 11 ρ = +0.218, p = 0.527 | **Confirmed** — and the predicted quality *confound* did not appear either |

**Power accounting for (iii), stated because it changes how the null should be read.** Pull-level learnability
score reliability = 0.521 (mean within-pull SE 0.0217 at 24-of-200 demos sampled, against a between-pull SD of
0.0295). On the outcome side, for the content-only arms **SD(Δ) = 2.68 pp < σ_e = 3.33 pp → Δ-reliability 0.000,
so the maximum |ρ| attainable by *any* predictor is 0.000.** The only group with real outcome signal is the
execution-quality arms (SD(Δ) = 8.87 pp, reliability 0.859, ceiling 0.669); there the disattenuated ρ is +0.326,
not significant. **This test therefore establishes "no measurable signal", not "provably zero signal."**

**The result that matters more than the correlation.** Ranking all 41 pull draws by mean learnability, the top
five are `style_lo` (Δ −2.0), `style_lo` (−2.7), `survivor_mix` (−2.2), `style_lo` (+1.3), `tall_vessel` (+5.1).
**The single most learnable arm is `style_lo` — the bottom execution-quality tail, the only arm in the whole
program with a consistently negative mean (−2.41 pp).** Within-round top-1 selection by highest learnability:
**0.111 correct over 9 rounds against 0.280 chance, mean regret +4.79 pp**; by *lowest* learnability, 0.444.
Same-seed pair sign agreement 0.462 (chance 0.5). Mechanistically this is constraint 5 obeyed exactly: high
learnability = high loss that a stronger model reduces = **inconsistent execution** — a real property, pointing
the wrong way for this decision.

**Reference choice is not innocuous.** Re-measuring the (i) AUC with each of 36 retained checkpoints as the
reference: mean 0.761, **sd 0.139, range [0.281, 0.969]**. The reference-free raw π₀ loss (AUC 0.844 on that
slice) is the stable detector; the RHO-LOSS subtraction adds mostly reference-choice variance.

### 4.2 TEST B — learning-progress reward validity (Graves / ODM)

**Setup.** A fixed 64-demo slice (32 pool demos never trained by *any* pull, 16 D0 demos, 8 + 8 quality-tail
demos), evaluated by forward pass at π₀ and at step 5000 and the final step of every pull with retained
checkpoints: **24 pulls with both step 5000 and final (the 20k-step recipe) plus 12 final-only (the 10k recipe)**,
60 checkpoints, 2 draws (reduced from 4 mid-run because of GPU contention; CRN pairing makes 2 sufficient, and I
state that here rather than running overnight for a result whose ceiling is 0).

| | Pre-registered | Measured |
|---|---|---|
| Spearman(early loss progress, Δ) | **INVALID** if \|ρ\| < 0.30 **and** sign agreement ≈ 0.5; **VALID** requires ρ ≥ 0.50 **and** agreement ≥ 0.70 | all 24: **+0.292, p = 0.170**. Content-only n = 18: **+0.457 [+0.057, +0.715], p = 0.057** |
| Within-round paired sign agreement | ≈ 0.5 | held-out pool **0.557**; D0 slice **0.671** |
| Within-round top-1 pick | — | **0.000 correct over 3 rounds** (chance 0.139), mean regret **+3.56 pp** (held-out) / +2.81 pp (D0) |

**The outcome falls between my two pre-registered brackets and is reported as such.** ρ = +0.457 on the primary
content-only group exceeds my INVALID bound (0.30) but misses the VALID bound (0.50), and the sign-agreement half
of the VALID criterion (0.557 vs 0.70) fails outright. It is also **one of 16 correlations computed** in this
test (0.8 false positives expected at α = 0.05), and its p = 0.057 is exactly what that multiplicity predicts.
Splitting the pairs finds no stable structure: agreement is 0.514 on quality-involving pairs and 0.600 on
content-vs-content for held-out progress, and it inverts for D0 progress (0.771 vs 0.571).

**The binding number.** Across the 24 pulls with a step-5000 checkpoint, **SD(Δ) = 2.75 pp < σ_e = 3.33 pp →
Δ-reliability 0.000 → maximum attainable |ρ| for any predictor = 0.000.** There is no between-arm outcome signal
in this set to predict; every correlation above is a correlation with evaluation noise. **This is the single most
important sentence in Test B, and it applies symmetrically to any future proxy tested on these pulls.**

**The measurement side is fine; the outcome side is not.** Step-5000 held-out loss spans only 0.30946–0.31988
across the 22 non-null pulls (3.4 % relative). Paired on the same 32 demos and the same CRN draws, only **24 % of
the 276 arm pairs are resolvable at |t| > 2**; resolving the median gap would need ≈ 116 held-out demos — 3.6× this
probe, still only ≈ 2 GPU-minutes of forward passes. **So the cheap reward is cheap to sharpen; sharpening it
merely resolves differences that do not map to outcomes.**

**Incidental exhibit, strong and new.** Held-out pool loss goes π₀ **0.588** → step 5000 **0.31–0.33** → final
**0.443–0.598**, and both null arms end *worse than π₀* (0.583, 0.598) while realizing +2.44 / −0.89 pp on the
deployed metric. The 20k-step recipe spends its last 15k steps making held-out loss worse with no consistent
effect on success — an independent, loss-side view of the same "second half of training buys nothing" result the
checkpoint ablation found on the outcome side (10k vs 20k paired difference +0.67 pp [−4.89, +6.44]).

---

## 5. What to build next — ranked, at most three

Ordered cheap-to-expensive, and each unblocks the next. GPU-hour figures assume this stack (one pull = 5–9 GPU-h).

### #1 — The signal-validity gate ("c-meter"), made a standing pre-condition, and pointed at the one channel with a prior

**What it is.** Promote `f4_signal_probe.py` from a one-off into a required gate: **before** any allocator,
selector or prescription rule is built on a cheap signal, measure that signal's **c** (rank and linear
correlation with realized closed-loop Δ) and its **SNR** against a small set of arms whose true values are already
known, and report both with their permutation nulls and their attenuation ceiling. The robot instantiation reuses
the Test-B machinery (`loss_eval.py`, 64-demo slice, CRN draws) over already-retained checkpoints. **The first
real application is the execution-quality channel** — the only channel with a prior reason to clear the bar,
since constraint 5 says the flow-matching gradient does not encode region (AUC 0.577, below its 0.598 null floor)
but does encode action quality (AUC 0.778), and Test A independently found the raw π₀ loss separates the quality
tails at AUC 0.752 (null ≤ 0.594) while being flat on region.

**Cost: ≈ 0.5 GPU-hour, zero new pulls.** (Measured comparables: the CIFAR channel measurement took 385 s; the
Test-B robot probe over 60 checkpoints was ≈ 2 GPU-minutes of forward passes; sharpening it 3.6× to resolve the
median arm pair is still ≈ 2 GPU-minutes.)

**Decision it unblocks.** It converts every future "let's use loss progress / learnability / a 1.5k-step proxy /
gradient alignment" proposal from a 30-GPU-hour round into a 10-minute measurement, and it gives the prescription
paper a methodological contribution that is independent of whether our arms win: *a pre-registered validity
threshold that a cheap reward must clear before it is allowed to allocate.* Concretely it decides whether the
quality channel is allowed to select data for the G1/new-source pipeline.

**The specific risk that would kill it.** *You cannot measure c against arms whose true values are inside the
noise floor.* For the content arms, SD(Δ) = 2.68 pp < σ_e = 3.33 pp, so the attenuation ceiling is exactly 0.000
and the c-meter returns "unmeasurable", not "zero". The gate is only meaningful on an arm set with genuine value
spread — today that means the execution-quality arms (SD(Δ) = 8.87 pp, reliability 0.859, ceiling 0.669) plus the
known-value anchors `planted_bad` (−19.6 / −21.8 pp on two pulls) and `style_lo` (−2.41 pp mean). **Any deployment
of the gate must include those anchors as calibration arms, and must state that a c measured on the quality axis
does not license a claim about the content axis.** Related live work: a causal test of exactly this quality
filter (`gradqual_hi` vs `gradqual_lo` at paired seed 1003) was already running at the time of writing; those two
pulls are the natural first outcome-side anchors for the gate and should be wired in when they land.

### #2 — The transfer matrix **A** with Δ**success** entries, estimated from the pulls we already have

**What it is.** Skill-it's skills graph, rebuilt to survive our constraints: rows = arms already run, columns =
evaluation strata (tall / mid / easy bands, style terciles), entries = **paired Δ closed-loop success versus the
round-matched baseline**, never Δloss. Estimated from `episodes.parquet` (phase == eval) joined to
`E_manifest.parquet` — each existing pull is already one row of A. No multiplicative online update (with ~14–41
pulls at σ_e = 3.3 pp, `exp(η·A·L)` is unidentifiable); A is used **only** to test diagonal versus off-diagonal
structure against a permutation null, with the null arms (+2.44 / −0.89 pp) providing the per-cell noise floor,
and to report how many pulls per cell would be needed to resolve A at σ_e = 3.3 pp.

**Cost: 0 GPU-hours** (ledger read plus analysis, ≈ 1 hour of compute-free work). Write to
`llm_borrow/skill_graph_A.json`.

**Decision it unblocks.** It tests the one assumption `PRESCRIPTION_CRITERION` §5 explicitly flags as an
*unverified shrinkage prior* — treatment-effect additivity across (mechanism, source). And it decides the paper's
central mechanism claim: if A is off-diagonal-dominant, **region-matched prescription is falsified as a
mechanism** and the claim changes from "prescribe to the weak region" to "the weak region is taught from
elsewhere" — a publishable prediction-table row either way. The existing hint points that way (the tall arm:
−2.7 pp on its own target stratum, +13.4 / +4.4 pp on easy / mid in round 4) but is **n = 1 pull per cell**.

**The specific risk that would kill it.** Per-cell noise. Strata are subsets of the 450-rollout evaluation, so
per-cell standard errors exceed the 3.3 pp whole-eval floor, and most cells have one observation. **Pre-committed
kill criterion: if the permutation test on diagonal dominance is null, report the per-cell noise floor and the
required pulls-per-cell, and make no structural claim.** Reporting an off-diagonal pattern that is within
per-cell noise would be exactly the error this whole report exists to prevent.

### #3 — Stop tuning the allocator; buy spread and kill noise (the arm-design + replication round)

**What it is.** The wind tunnel's decision-relevant output is that **at spread/noise = 0.356 no allocator
matters**: successive halving, the best method available, reaches P(correct) 0.41 / 0.55 / 0.63 at 6 / 12 / 24 FTE
and never reaches 95 % of oracle by 48 FTE, while the entire borrowed family is in the "never" column. The
leverage is therefore in the two terms of the ratio, not in the algorithm. So: (a) **define arms with genuine
value spread** — attainability-gated *new* sources rather than re-cuts of a saturated fixed pool, plus one
known-value anchor arm per round — targeting an arm-mean spread of ≥ 3 pp instead of the measured 0.65 pp; and
(b) **engineer the noise floor from its measured decomposition** — buy more *seeds* (3.3 pp, the dominant term),
not more rollouts (0.88 pp, the smallest), fix the checkpoint by convention (choice within a run moves the
reported effect by ≈ 2.4 pp), and keep paired seeds plus the pre-registered elimination bound, now with Graves'
adaptive quantile rescaling of the reward to [−1, 1] against the 20th/80th history percentiles so the bound is
scale-free, and Aioli's discipline of a fixed, pre-registered δ of budget spent measuring rather than assuming.

**Cost: ≈ 40–72 GPU-hours** for a 4-arm × 2-seed round (8 pulls at 5–9 GPU-h), preceded by the free attainability
pre-check (MimicGen replay success rate in the target region costs no GPU). For calibration of what buying
accuracy costs at the *current* spread: 1 pull/arm → decision accuracy 0.62 (28 GPU-h); 5 → 0.75 (140 GPU-h);
20 → 0.89 (560 GPU-h); 40 → 0.94 (1120 GPU-h) — and that table is optimistic, since it treats the observed arm
means as ground truth.

**Decision it unblocks.** Everything downstream in the prescription paper: the §2 probe protocol, the
matched-versus-mismatched-placebo validation, and the head-to-head against FSC (arXiv 2505.07728, the designated
competitor). Without spread above the floor there is no decision for any allocator to make and no effect for any
proxy to predict — which is precisely why Tests A and B both hit an attenuation ceiling of exactly 0.000.

**The specific risk that would kill it.** The **generation ceiling**: if no available source can actually
*produce* demonstrations in the target region (attainability a_m(s) ≈ 0 — already measured as ≈ 0 for the
expert-loop × tall and expert-loop × far cells), the promised spread never materializes and the round returns to
0.356 having spent 40–72 GPU-h. Run the free attainability check first and refuse to fund an arm that fails it.
Secondary risk: **retention** — narrow fine-tuning can lose on both axes at once, so the ≥ 1/3 coverage floor and
in-arm diversity requirements are non-negotiable, and overdosing is measured to be actively harmful (610 added
demos → 0.299 overall success vs 200 → 0.371).

---

## 6. Limitations and everything still unverified

**On the literature side.**
- Only the **"bandit-sources" family** of the sweep reached me in full text (ODM, Graves, Skill-it, ADO, Aioli,
  and a **truncated** AC-ODM entry). Verdicts for JEST / RHO-LOSS and RegMix / Data Mixing Laws rest on tonight's
  own measurements plus citations verified elsewhere: I re-fetched the arXiv abstract pages for **JEST
  (2406.17711)** and **RHO-LOSS (2206.07137, ICML 2022)** in this session and confirmed titles, authors and venue;
  **RegMix (2407.01492, ICLR 2025)** and **Data Mixing Laws (2403.16952, ICLR 2025)** come from the project's
  previously verified reading list. The AC-ODM verdict rests on our own gradient nulls, which are decisive for
  any gradient-alignment reward, but I did not see the full method description.
- **DoReMi, DoGE, Skill-it, ADO and Aioli were never run** — their verdicts are analytic (constraint mapping plus
  the shared loss-reward measurement), not empirical. Only ODM's/Graves' reward, JEST/RHO-LOSS's learnability, and
  RegMix's proxy pipeline were measured on our own artifacts.
- Aioli's own numbers carry no error bars or seed variance in the cited table; its 0.274-perplexity win would not
  clear our pre-registered confidence-bound rule. It is cited for its *negative* result and its reduction, not as
  a validated allocator.

**On the wind tunnel.**
- **Part 1's robot-regime numbers for the cheap methods depend on the harness's encoding of the pathologies**
  (`SPEC_ROBOT_LIKE.lp_reward`, `.jest_score`, `.proxy_bias` are one author's parameterization of measured
  constraints 1 and 3, not per-arm measurements). Part 2 does not — it replaces them with the controlled c axis
  and pools each cell over 8 (or 32) independent nuisance directions. **Treat the transfer map, not Part 1's
  robot table, as the deliverable.**
- **The synthetic robot spec's "correct answer" is the empirical argmax of arm means that are themselves within
  noise of each other** (`mid_band` +3.63 vs `tall` +2.96 vs `random` +2.59 vs `easy` +2.07 pp, σ_e = 3.3 pp).
  So "P(correct)" means "recovers the empirical best arm of the 14-pull race", not "recovers the truly best data
  rule". Every P(correct) figure in §3 must be read that way.
- **The SNR anchor of 0.356 is the outcome-side ratio** (arm-mean spread 0.654 pp ÷ per-pull noise 1.837 pp),
  used as a stand-in for the cheap channel's own SNR, which was never separately measured on the robot. What
  *was* measured on the robot cheap channel is its **resolvability**: only 24 % of 276 arm pairs are separable at
  |t| > 2 with a 32-demo × 2-draw probe, and ~116 demos would fix that for ≈ 2 GPU-minutes. Our failure is on the
  **c** axis, not the SNR axis — which matters, because the map shows that at c ≈ 0 no amount of SNR helps
  (at SNR 8, c = 0: jest 0.963 pp vs the 0.531 pp floor).
- **c is a rank/linear correlation over only 4–6 arms**, a coarse coordinate. That is why cells pool over many
  nuisance directions and why the realized Spearman is reported alongside the design Pearson.
- `jest_then_confirm_k4`'s c\* is non-monotone in SNR (0.20 / 0.20 / 0.70 / never); its margin over the floor is
  small and that row wobbles within noise. **Do not read a mechanism into it.**
- **The real-GPU F4 race is n = 5 replicates per cell.** The direction is unambiguous (0/5 vs 3/5 correct picks;
  3.68 vs 1.42 pp) but the 1.80 pp gap to the floor is ≈ 2 standard errors. It confirms a 1000-replicate
  synthetic prediction; it is not independent evidence.
- **Incomplete cells at the time of writing** (all logs and partial JSON are on disk; nothing below changes a
  verdict): the real-CIFAR F4 configurations ρ = 1.0, ρ = 0.5 and α = 0.5 were still running (each
  learning-progress cell is ≈ 35 min under GPU contention) — the **channel-level** measurements for all of them
  are complete and reported in §3.7(i); real-CIFAR Part 1 is complete for jest, jest-then-confirm, successive
  halving, random, regmix and oracle through 24 FTE, for `learning_progress_ucb` through 3 FTE (regret 0.000,
  P = 1.00 at both 1.5 and 3 FTE) and for `learning_progress_exp3` through 6 FTE.
- CIFAR replicates draw training seeds from a pool; it was extended from 4 to 8 seeds (≈ 50 additional real
  fine-tunes) so that 5 replicates are genuinely distinct. That is why the CIFAR floor numbers here are worse
  than an earlier 1-replicate smoke run's, and why the earlier claim "at 6 FTE all eight allocators score regret
  0 on CIFAR" is retracted.

**On the robot signal tests.**
- **Tests A(iii) and B are underpowered by construction, and this is measured rather than suspected.** For the
  content-only arms the outcome-side reliability is exactly 0.000 (SD(Δ) 2.68 and 2.54 pp, both below
  σ_e = 3.33 pp), so the maximum correlation any predictor could show is 0.000. These tests establish "no
  measurable signal", not "provably zero signal". The one group with genuine outcome signal — the
  execution-quality arms (SD(Δ) = 8.87 pp) — shows a disattenuated ρ of +0.326 that is not significant at n = 11.
- Test B's headline ρ = +0.457 (n = 18, p = 0.057) **falls between the two pre-registered brackets** and is
  reported as such, not rounded into either. It is one of 16 correlations computed in that test.
- Test B used 2 CRN draws rather than the planned 4 (GPU contention); CRN pairing makes 2 adequate for the
  paired difference, but that is a judgement, not a measurement.
- Test A's quality labels come from another session's execution-consistency score tails; the gate shows the loss
  and gradient recover that score, and the outcome-side link is the style race (`style_lo` the only reliably
  negative arm), but the label definition itself is inherited.
- Checkpoints load in bfloat16 (fp32 does not fit the 0.12 JAX memory cap); the float16 control puts quantization
  noise at 1.3 % of the signal SD.
- **The tall-arm off-diagonal pattern (−2.7 pp on its own stratum, +13.4 / +4.4 pp on easy / mid) is n = 1 pull
  per cell** with per-stratum noise larger than σ_e. It motivates build item #2; it is not itself a finding.

**On scope.** Everything here concerns *which existing arm to fund*, on one task
(`PickPlaceCounterToSink`), one base policy (pi0 LoRA), one pool (9,885 demos), one evaluation set (450
rollouts). The prescription problem's harder half — **which new data to collect, from which source, at what
dose** — is not tested by any measurement in this report; the wind tunnel only tells us that the *allocator*
layer of that problem is not where the leverage is at our current spread and noise.

---

## Appendix — artifact index (all paths absolute)

| Path | What it is |
|---|---|
| `/data/xinyua11/robocasa/gradient_analysis/llm_borrow/wind_tunnel.py` | The metered allocator simulator (backends, allocators, FTE meter). |
| `…/test_wind_tunnel.py` | 25 unit tests (fairness, charging, oracle dominance, determinism); zero GPU; passing. |
| `…/wind_tunnel_calibration.{py,json,log}` | Wall-clock calibration of the FTE cost model (eval = 3.9 % predicted vs 4.3 % measured). |
| `…/PREREGISTRATION.json`, `…/prereg_AB.json` | The two pre-registrations, written before the corresponding runs. |
| `…/part1_synthetic_robot.json`, `…/part1_synthetic.json`, `…/part1_cifar.json`, `…/part1_cifar_lp_exp3.json` | Part 1 races (robot-like synthetic, CIFAR-like synthetic, real CIFAR). |
| `…/f4_synth_robot.json` (+ `_pri`, `_d32`, `_budget`), `…/f4_synth_cifar.json` | The (c, SNR) transfer maps, 1000 reps/cell, 8 or 32 nuisance directions. |
| `…/f4_signal_probe.{py,json,log}` | The 385-second channel-validity measurement (the "c-meter"). |
| `…/f4_cifar.json`, `…/f4_cifar_fast.json` | Real-CIFAR allocator races under channel corruption. |
| `…/f4_transfer.py`, `…/race_lib.py`, `…/part1_driver.py`, `…/analyze.py`, `…/analyze_AB.py` | Drivers and analysis. |
| `…/test_llm_rewards_on_robot.py`, `…/reward_signal_test.json` | The direct robot measurement of the ODM/Graves loss reward (zero GPU). |
| `…/loss_eval.py`, `…/build_sets.py`, `…/results_AB.json`, `…/analysis_output.txt` | Tests A and B: set construction, forward-pass harness, full numeric output. |
| `…/decision_accuracy_diagnostic.json` | Arm means, spread/noise = 0.356, and the decision-accuracy-versus-GPU-hours table. |
| `/data/xinyua11/robocasa/gradient_analysis/Q0_REPORT.md`, `Q2_REPORT.md`, `Q4_CIFAR_MIRROR_REPORT.md`, `NOISE_ANATOMY.md` | The prior gradient program these verdicts lean on (encoding gate, absorption, CIFAR positive control, noise decomposition). |
| `/data/xinyua11/robocasa/weakregion/PRESCRIPTION_CRITERION.md`, `READING_LIST.md` | The criterion any borrow must serve, and the prior verified literature sweep. |
