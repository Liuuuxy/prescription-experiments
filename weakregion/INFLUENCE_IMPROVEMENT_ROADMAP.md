# Influence Selector — Theoretical Improvement Roadmap

## 1. The core theoretical diagnosis (one unified statement)

The current selector computes, per pool demo $z$,
$$
\text{score}(z) \;=\; \cos\!\big(g(z),\,g_{\text{val}}\big),\qquad
g(z)=\frac{\nabla_\phi L_{\text{fm}}(z;\theta_0)}{\lVert\nabla_\phi L_{\text{fm}}(z;\theta_0)\rVert},\quad
g_{\text{val}}=\tfrac{1}{n}\textstyle\sum_{i}\nabla_\phi L_{\text{fm}}(w_i;\theta_0),
$$
and takes the top-200 independently. Four defects compound, and they are *not* independent — they share one root.

**(a) Common-mode collapse (F1).** Decompose each unit gradient as $u_i = c_i\,s + r_i$, where $s$ is the shared "generic pick-place" unit direction, $c_i=\langle u_i,s\rangle$, and $r_i\perp s$ is the pose/object-specific residual. Across heterogeneous start/target poses the residuals are approximately orthogonal, so the **residual mean cancels at rate $\lVert\overline{r}\rVert = O(\sigma_r/\sqrt n)$** while the common-mode term keeps $O(1)$ norm:
$$
g_{\text{val}} = \bar c\, s + \overline{r}\;\xrightarrow{\,n=313\,}\;\bar c\,s,\qquad
\text{so}\quad \text{score}(z)\to \bar c\,c_z .
$$
The score therefore ranks demos by their **genericness coefficient $c_z$**, not by help to any target. This is confirmed quantitatively: 97% of candidates have $+\cos$ to $g_{\text{val}}$ (a discriminative axis would split ~50/50), hard-vs-easy AUC $=0.35$ (below chance — generic aligns *more*), and the selected set sits at $+2.54\sigma$ of the $c$-distribution.

**(b) Magnitude-blindness (F2).** Unit-normalization discards $\lVert\nabla L(z)\rVert$, which is monotone in per-demo loss. The Adam update the model actually experiences is $\Delta\theta\approx -\eta\,\nabla L(z)/(\sqrt{v}+\epsilon)$; cosine measures none of its size. Empirically the top-cosine demos carry the **smallest** raw gradients (low-loss, easy, redundant) — the demos that move $\theta$ least.

**(c) Loss ≠ success (F4).** The score attributes flow-matching MSE through the lossy chain $L_{\text{fm}}\!\to\!A\!\to\!J$: MSE upper-bounds per-frame action error only in expectation, and task success $J(\theta)=\mathbb E_{\xi\sim p_\theta}[\mathbb 1\{\text{lift}>\tau_l\wedge \text{sink}<\tau_s\}]$ is a sparse, non-differentiable threshold under the *policy-induced* (closed-loop, covariate-shifting) distribution $p_\theta$. $\partial J/\partial\theta$ does not exist, and the open-loop per-frame $g(z)$ ignores $p_\theta$'s dependence on $\theta$ entirely.

**(d) No retention / no interaction term (F3, F5).** The first-order objective is **modular**: $f(S)=\sum_{z\in S}\text{score}(z)$, maximized by top-k. This drops (i) the second-order $-\tfrac{\eta}{2}\lVert\sum_{z\in S}g(z)\rVert^2_{H}$ term that is **submodular** and penalizes redundant collinear gradients (→ it picked 14 tongs), and (ii) the **retention** coupling: the same $\Delta\theta$ perturbs the shared-grasp subspace, and overall success is $\approx 0.88\cdot\text{NonTarg}+0.12\cdot\text{Targ}$, so a non-surgical set regresses the 88% majority (value NonTarg $0.523$ vs base $0.617$, $+25$ fail_no_grasp).

**Unified root.** (a) and (d) are the same object: the dominant direction $s$ that collapses $g_{\text{val}}$ **is** the retention/forgetting direction $g_R$. The selector simultaneously (i) ranks by alignment-to-$s$ when it averages, yet (ii) the contrast/normalize variants tilt toward whatever *single over-represented* sub-mode of $s$ has the largest residual (handle/elongated cluster $0.36$ vs random $0.21$), which **displaces** the base's broad grasp. So the estimator is a magnitude-blind, curvature-blind, **1-step** estimate of the **wrong objective** (MSE not success), aggregated **modularly** (no diversity, no retention) — and every one of these is on the wrong half of the bilevel problem.

---

## 2. The principled "ideal" objective

The true problem is **bilevel** and on **success**, with a **retention constraint**:
$$
S^\star=\arg\max_{|S|=200}\; J\big(\theta^\star(S)\big)
\quad\text{s.t.}\quad J_{\text{NonTarg}}\big(\theta^\star(S)\big)\ \ge\ J_{\text{NonTarg}}(\theta_0)-\varepsilon,
$$
$$
\theta^\star(S)=\arg\min_\theta \sum_{z\in \text{base}\cup S}\ell_{\text{fm}}(z;\theta),\qquad J=\mathbb E_{\xi\sim p_{\theta}}[\mathbb 1\{\text{solved}\}].
$$
The exact attribution of a train point is the **trajectory inner product** (TracIn / unrolled-Adam):
$$
\text{Infl}(z\!\to\!\text{val})\;\approx\;\int_0^\tau \big\langle\, P_t\,\nabla L(z;\theta_t),\ \nabla J_{\text{surr}}(\theta_t)\,\big\rangle\,dt,
$$
with $P_t=\text{diag}(1/(\sqrt{v_t}+\epsilon))$ the Adam preconditioner. **Every method is a bias/variance approximation of this**, and the current selector is its maximally-degraded special case (drop $P_t$, drop both magnitudes, single off-trajectory neutral reference, target = MSE-mean not $J$, modular top-k, no retention). The methods sort cleanly onto *which* defect they repair:

- **target side** ($g_{\text{val}}$): contrast / cluster / coverage (aggregation-geometry), success-reaim (target-objective).
- **metric** ($P_t$): Adam/Fisher preconditioning (metric-preconditioning, LESS).
- **fidelity** (the integral / magnitudes / $\eta_t$): trajectory TracIn-LESS (estimator-fidelity).
- **set function** (modular → submodular + constraint): gradient-matching coreset, retention hinge / A-GEM (subset-selection, forgetting-aware), mixture weights (mixture-reweighting).
- **the constraint itself** (retention): forgetting-aware, mixture-reweighting.

The honest tension: the *only* arm that satisfied the retention constraint **and** beat baseline is `core` (depth on failing categories), which approximates $S^\star$ by **never leaving the data manifold** and adding density where $J$ is low — it pays nothing for the constraint because it is structurally inside it.

---

## 3. Ranked improvement roadmap

Ordered by expected value-per-cost **here**. "Cheap" = rescore over already-streamed gradients, no new fine-tunes; "machinery" = new features/curvature/optimizer-state reads or extra runs.

| # | Axis | Fixes | Principled form | Cheap pi0-tractable version | Expected effect | Risk / why it may NOT help |
|---|------|-------|-----------------|------------------------------|-----------------|----------------------------|
| 1 | **Retention-constrained group mixture** (mixture, Tier 0) | F3, F5 | $\max_w \sum_g \pi_g s_g(\theta(w))$ s.t. $s_{\text{maj}}\!\ge\! s_{\text{maj}}^0-\varepsilon$ | 3 LeRobot sub-dirs (base / targeted-46 / generic); set `dataset_weights` with base mass pinned; grid $m\in\{0.05,0.1,0.2\}$ | Decouples *which* demos from *how much* mass leaves the base manifold — the only lever that directly buys retention | **Cheap** for Tier 0 but each grid point is a full 20k FT (~8h); at $n{=}300$, $\Delta\!\sim\!0.01$ is within noise → can't statistically separate from core; core already sits at the floor ($0.616$ vs $0.617$) so headroom over core is ~0 |
| 2 | **Trajectory TracIn-LESS rescore** (estimator-fidelity) | F2, F5, (F3) | $\sum_t \eta_t\langle P_t\nabla L(z;\theta_t),\,P_t g_{\text{val},t}\rangle$ | reuse `influence_score.py`; ckpts on a *real* FT trajectory {2k,6k,12k,19999}; weight by $\eta_t$; drop unit-norm (`--score dot`); Adam precond from logged `nu` | Restores magnitude (F2) + schedule + on-path eval (F5); cheapest legitimate fidelity gain | **Cheap rescore.** But Adam precond $1/\sqrt v$ can *up-weight* the low-variance shared-grasp direction → worsen F3; magnitude-aware dot selects high-loss OOD demos; still attributes MSE (F4 untouched) |
| 3 | **Aggregation geometry: clustered-max + facility-location** | F1, F5, (F3) | $\max_S \sum_i \max_{z\in S}\text{sim}(u_i,u_z)$ over $m$ centered $D_{\text{val}}$ centroids; $(1\!-\!1/e)$ greedy | keep per-$D_{\text{val}}$ tugs, JL→2048, k-means $m{=}12{-}16$, center by $s$, top-3 mean, greedy submodular select (numpy) | Structurally kills the $\sqrt n$ collapse + redundancy; +per-cat density floor recovers core's mechanism | **Cheap rescore.** But contrast $g_{\text{val}}$ already did the de-collapse (AUC $0.63$, grasp-cluster frac $0.155$) and **still lost** → F1 is *not* the binding constraint; diversity = *more* perturbation → the spread arm (coverage) forgot worst |
| 4 | **Metric preconditioning (Fisher/Adam whitening)** | F1, F2 | $\langle \Gamma(z),\Gamma_{\text{val}}\rangle,\ \Gamma=\nabla L/(\sqrt v+\epsilon)$ | restore `nu` from opt_state **or** estimate $v$ from $D_{\text{val}}$+ref $g^2$; score with precond-magnitude (no per-demo cosine) | Soft-whitening turns amplitude → z-score; complements contrast | **Cheap rescore.** Sign risk: warmup `nu` is *base-loss* curvature; $1/\sqrt v$ up-weights base-**under**fit coords = the handle-grasp cluster that *caused* the regression → can amplify F3. Use $D_{\text{val}}$-estimated $v$, add $\epsilon$ floor |
| 5 | **Retention hinge / A-GEM projection** (forgetting-aware) | F3, F1, F2 | $\langle u(z),\hat g_{\text{target}}\rangle-\lambda\max(0,-\langle u(z),\hat g_R\rangle)$; or project off $g_R$ | one extra resident $g_R$ vector + 1 vdot/candidate; $\lambda$-sweep in post-proc | Direct algebraic "do-no-harm on retention" | **Cheap but near-no-op:** measured, selected sets sit at $+0.25\sigma$ (failure) / $+2.54\sigma$ (value) above $g_R$ with $\le1\%$ negative cosine → hinge fires on $\le2/200$ picks. Forgetting comes from being *parallel* to $g_R$ (one dominant sub-mode), which a do-no-harm-on-loss term does not penalize |
| 6 | **Gradient-matching coreset (RAGGM)** (subset-selection) | F5, F3, F2, F1 | $\min_{w\ge0,\lVert w\rVert_0\le k}\lVert\sum w_z g(z)-(\alpha g^\*-\beta g_R)\rVert^2$ | JL→2048, greedy OMP over stored sketches; cluster-stacked target for value arm | Submodular: zeroes 2nd tongs (F5), uses magnitude (F2), injects retention via $-\beta g_R$ | **Machinery** (per-demo sketch storage, OMP). $-\beta g_R$ points along high-norm generic grasp → can *re-couple* to shared subspace; JL at $d{=}2048$ is $\epsilon\!\approx\!0.19$ worst-case (not $0.03$), and OMP's argmax is exactly the tail JL doesn't protect |
| 7 | **Target-objective re-aim (success surrogate)** | F4, F3, F1 | $g_{\text{val}}^\*=\nabla[\sum_z w(z)L_{\text{fm}}^\Phi(z)]$, $\Phi$=grasp window, $w$=failure-credit | phase-restrict frames to grasp-approach; weight $D_{\text{val}}$ by rollout failure_phase | Only axis aimed at F4 (success not MSE) | **Provenance bug:** $D_{\text{val}}$ = expert *successes* (no failure_phase); the only legal bridge is category mapping = `core` again. Phase-restriction *concentrates* gradient on grasp motion = the common mode → can amplify F1/F3 |
| 8 | **DoReMi group-DRO** (mixture Tier 2) | F3 | $\min_\theta\max_q \sum_g q_g(L_g-L_g^{\text{ref}})$ | per-group loss logging + 2k proxy FT, export $\bar q$ | Worst-case retention robustness | **Machinery + F4:** protects per-group *MSE*, which the data shows is decoupled from success (vsmoke passed while eval failed). Re-imports the same contrast cancellation that already underperformed |

**Read of the table:** rows 2–5 are genuinely cheap rescores and worth running *as a bundle* (they fix orthogonal halves of the inner product); rows 1, 6, 8 need machinery and have the highest backfire risk; row 5 is provably near-inert; row 7 has a data-provenance blocker. **Crucially, every row except 1 still optimizes a 1-step MSE proxy (F4/F5 survive), and row 1's headroom over core is ~0.**

---

## 4. The single best next method

**RC-LESS: Retention-Constrained, Coverage-aware, Adam-preconditioned influence — a pure rescore + one mini-fine-tune gate.** This is the maximal stack of the cheap, orthogonal fixes (rows 2–5), engineered to attack the *binding* constraints (F3 retention, F1 collapse) without the backfires, and gated against F4.

**Features.** For each candidate $z$ and the $m$ clustered targets, use the **Adam-preconditioned, magnitude-preserving** gradient at trajectory checkpoints:
$$
\Gamma_t(z)=\nabla_\phi L_{\text{fm}}(z;\theta_t)\odot \frac{1}{\sqrt{\hat v_t}+\epsilon},\qquad t\in\{2k,6k,12k,19999\}\ \text{on a real FT run.}
$$
Estimate $\hat v_t$ from the **$D_{\text{val}}$+ref** per-demo $g^2$ mean (not the base-only warmup `nu` — this avoids the verified sign-flip that up-weights the base-underfit grasp cluster). Do **not** unit-normalize $\Gamma_t(z)$.

**Targets (don't average — cover, contrast-centered).** Retain the 313 $D_{\text{val}}$ tugs, project (JL→2048), k-means into $m{=}12$–$16$ inverse-size-balanced centroids $\{s_j\}$; **center** each by the contrast direction $s=$ (mean_hard − mean_ref) so the common mode is subtracted (the form that already smoke-passed at AUC 0.63). Build the retention vector $\hat g_R = \text{normalize}\big(\text{mean over a class-balanced non-targeted hold-out}\big)$ at the same $\Gamma$ metric.

**Score (retention-constrained coverage).**
$$
\text{cov}(z)=\frac{1}{|\Phi|}\!\!\sum_{t\in\Phi}\eta_t\;\underbrace{\text{top-3}_j\big\langle \tilde\Gamma_t(z),\,s_j\big\rangle}_{\text{coverage of target modes}} ,\qquad
\boxed{\,\text{score}(z)=\text{cov}(z)\;-\;\lambda\,\max\!\big(0,\ \langle \Gamma(z),\hat g_R\rangle - \rho\big)\,}
$$
where $\eta_t$ is the cosine-LR weight, and the retention term — **note the corrected sign** — penalizes demos that are *too parallel* to $\hat g_R$ (i.e. exceed a parallelism budget $\rho$). This fixes the verified flaw in the plain do-no-harm hinge (which fired on $\le2/200$ picks because the offending demos are maximally *parallel*, not antagonistic, to $g_R$). The penalty here directly demotes the over-represented single grasp sub-mode that displaced the base.

**Select (submodular, not top-k).** Greedy facility-location over the stored $9885\times m$ coverage matrix with a hard floor $\ge d$ demos per failing category (the explicit reimplementation of core's winning mechanism) and a per-category cap (anti-redundancy, no 14 tongs). $\sim$10 lines of numpy.

**Validation gate (against F4).** Before any 20k arm, run a **2k-step LoRA mini-fit** on the selected 200 and require non-targeted held-out **success** $\ge 0.60$ (the retention guardrail that decided every prior eval) *and* hard-vs-easy AUC $>0.63$ on the score. Reject otherwise. This is the affordable shadow of Datamodels and is the only place success (not MSE) enters.

**Cost.** One rescore pass ($\approx$ the existing 3.6 GPU-hr/ckpt; $\sim$7–12 h for 4 trajectory ckpts) + clustering/greedy in CPU-seconds + one 2k-step mini-fit. No extra full fine-tunes to *rank*.

**Mechanism for beating core (the honest theory).** Core wins by (i) staying on-manifold (retention) and (ii) adding density on failing categories — but it is **category-coarse** and **cannot exploit within-category structure**. RC-LESS could beat it *if and only if* there exist demos that (a) cover a failing target mode core's category-depth misses (coverage term), (b) move $\theta$ substantially in the Adam metric (magnitude/precond), and (c) are *less* parallel to the shared-grasp $g_R$ than core's average pick (retention penalty) — i.e. demos that improve the failure region **without** riding the forgetting eigendirection. The per-category floor guarantees RC-LESS never does *worse* than a core-like set; the coverage + retention terms are the only way it can do *better*. **If no such (a,b,c) demo exists in the 200+400 pool, RC-LESS provably degenerates to core** (the floor binds), at the cost of one extra rescore.

---

## 5. What theory says the ceiling is

**Can a smarter influence selector beat the trivial P(fail) heuristic here? The honest answer is: probably not by much, and quite possibly not at all, within *this* pool — and the burden of proof is on the influence method.**

**The pessimistic case (stronger, and supported by the data).**
1. **The binding constraints are F3 and F4, and influence touches neither well.** The eval is rank-ordered by the non-targeted retention column, and that column is a function of *sparse success*, not flow-MSE. Every gradient method optimizes a faithful estimator of the *wrong* objective; F4 is a wall no rescore crosses.
2. **F1 is empirically *not* load-bearing.** The contrast arm already executed the de-collapse (AUC $0.35\to0.63$, grasp-cluster frac $0.36\to0.155$, targeted_frac $\to0.23$) and **still lost** ($0.553 < 0.580 < 0.593$). The most sophisticated geometry the roadmap offers sharpens a discriminator on an axis *already shown not to decide the eval*.
3. **Core already saturates the retention constraint.** Non-targeted $0.616$ vs baseline $0.617$ — the entire retention apparatus the proposals engineer is something core has *for free*. The retention floor therefore buys **~0 headroom over core**; the only way past core is to raise the *targeted* region itself beyond core's $0.432$, and reweighting/reselecting the **same** demos cannot manufacture targeted success that those demos don't already deliver.
4. **Coverage = more perturbation.** The hardest fact in the postmortem is that the *spread* arms (coverage, value, failure-infl — all "diverse") forgot, and the *least*-diverse depth-on-holes core won. Facility-location/DPP/gradient-matching all *maximize* spread; the diversity objective is mechanically aligned with the failure mode.
5. **Statistical power.** At $n{=}300$ paired, the live $\Delta$s are $0.01$–$0.04$; only value$<$core is significant ($p\!\approx\!0.02$). A 3-point grid of 20k fine-tunes **cannot separate a winner from core** — you would be selecting a hyperparameter on noise. Even a genuinely better method may be *undetectable* here.

**The optimistic case (real, but narrow).**
1. **Core is provably suboptimal at the boundary.** Core is category-coarse: it cannot exploit *within*-category structure (which specific configs/poses fail), cannot use magnitude (it's a count heuristic), and has no notion of redundancy. A method that adds the *right* within-category depth — covering a failing mode core's uniform depth dilutes — has a real, if small, edge. The per-category floor makes this strictly dominant-or-equal to core by construction.
2. **F4 is not unbreakable.** The 2k-step success gate routes *actual rollout success* (not MSE) into the loop. It cannot make selection differentiable, but it can **reject** the MSE-good/success-bad sets that sank value/coverage — turning "optimize the wrong objective" into "filter on the right one." This is the one place the surrogate chain is cut.
3. **The win, if any, is the *combination*, not any single axis.** Every solo axis was critiqued as "fixes one defect, leaves the binding one." The only theoretically coherent shot is the *stack* (don't-average + magnitude/Adam metric + retention penalty with the corrected sign + per-category floor + success gate), because the defects are coupled — fixing F1 alone re-exposes F3, fixing F2 alone re-exposes F1.

**Verdict.** For *this* regime — single task, $n{=}300$ noisy eval, a 200+400 pool where value-data and failure-data are nearly disjoint — **surgical depth-on-failures (core) is at or very near the theoretical optimum, and the realistic best-case for any influence selector is to *match* it, with a small, possibly statistically-undetectable, chance of beating it via within-category coverage.** The deeper finding is that this is a **data-coverage ceiling**, not a selection-algorithm ceiling: when the most-valuable data lies *outside* the failure region and the pool cannot manufacture targeted success, no reweighting/reselection of the existing 9885 demos clears core — only **adding different demos** would. The right next experiment is therefore RC-LESS *gated*, run not with the expectation of beating core, but to (i) cheaply confirm the ceiling is data not algorithm, and (ii) catch the narrow within-category-coverage win if it exists — while treating any sub-$0.04$ improvement over core as **unproven at $n{=}300$** until a larger paired eval (or a multi-seed design) can resolve it.
