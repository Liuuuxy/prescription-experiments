# Influence-Based Data Selection for pi0 (LESS / TracIn)

**Task:** RoboCasa `PickPlaceCounterToSink` · **Student:** pi0 LoRA (openpi, robocasa-pretrained
`pi0_robocasa_pretrain_human300` @ 75000, ~55% baseline) · **Pool:** 9,885 MimicGen demos / 79
object categories · **Budget:** 200 demos selected on top of a shared 400-demo base.

Two gradient-influence selectors were built as **principled alternatives to the P(fail) category
heuristic** (the `core` arm). Both score every pool demo by a gradient-alignment criterion and take
the top-200; both feed the identical fine-tune recipe (only the 200 demos differ).

| arm | question it answers | D_val (target) | g_val direction | result |
|---|---|---|---|---|
| **failure-influence** | which demos most reduce loss *on the failure region*? | targeted-10 failure cats | **contrast** `mean_hard − mean_ref` | `selected_targeted_frac` **0.23** |
| **value-influence** | which demos most improve *overall* held-out success? | class-balanced over all cats | **plain** `mean(D_val)` | `selected_targeted_frac` **0.06** |

---

## 1. Method (LESS adapted to openpi pi0)

For a candidate demo `z`, the influence feature is the gradient of pi0's flow-matching loss w.r.t.
the **LoRA adapters**, taken at a set of reference checkpoints and averaged:

```
g_t(z) = normalize( ∇_LoRA  mean_{K frames of z}  flow_matching_loss(z) )   at checkpoint t
g(z)   = mean_t g_t(z)
score(z) = < g(z), g_val >          # cosine of normalized gradients
select   = top-200 by score(z)
```

- **Gradient target = LoRA adapters (~50M params).** With both gemma variants LoRA, the trainable
  set is the LoRA adapters + the *entire* SigLIP vision tower + the action-expert heads
  (`freeze_filter = All(.*llm.*, Not(.*lora.*))`) — hundreds of millions of params, so a faithful
  fixed Gaussian projection matrix cannot be materialized. The scorer therefore **streams**: it
  builds `g_val` first, then dots each candidate's gradient against it on-device and keeps only the
  scalar — no projection, bounded memory.
- **K = 8 frames** evenly spaced across each demo; loss **masked to the 12 real action dims**
  (`--real_dim 12`) because pi0 pads actions 12→32 and the 20 zero-padded columns have pure-noise
  flow targets that otherwise swamp the gradient.
- **Reference checkpoints** — see §3, this is the single most consequential choice.
- Implementation: `policy_analysis/influence_score.py` (`probe` / `smoke` / `vsmoke` / `full`).

---

## 2. What failed before it worked (the gradient/direction search)

The smoke gate (does the score separate held-out *targeted* "hard" demos from *non-targeted* "easy"
demos?) drove the design, debugged one variable at a time:

| gradient target | g_val | smoke AUC(hard>easy) | verdict |
|---|---|---|---|
| `action_out_proj` last-layer, all 32 dims | plain | 0.44 | ✗ no signal |
| `action_out_proj`, masked to 12 real dims | plain | 0.44 | ✗ padding wasn't the issue |
| `lora` adapters (50M-dim) | plain `mean(hard)` | 0.35 | ✗ wrong direction |
| **`lora` adapters** | **contrast `mean_hard − mean_ref`** | **0.627** | ✓ separates |

**Insight 1 — last-layer TracIn carries no object identity here.** The action readout
(`action_out_proj`) gradient is dominated by the shared grasp/reach *motion*; it is essentially the
same for every pick-and-place demo regardless of the object, so it cannot tell failure-objects from
easy ones. The literature's "last layer is usually ~as good" did **not** hold for this task. Object
identity lives in the vision→LLM pathway, so the **LoRA adapters** (which route through
object-conditioned attention, and are the params actually updated in fine-tuning) are the right
target.

**Insight 2 — a common-mode "generic pick-place" gradient dominates.** With plain `g_val =
mean(hard)`, *every* demo aligns with it about equally (typical/easy demos slightly *more*, hence
AUC 0.35 — the wrong way). The object-specific signal is buried under the shared component.
Subtracting a random-pool reference — `g_val = mean_hard − mean_ref` (**contrast**) — cancels the
common mode and exposes the failure-*specific* direction (AUC 0.63). This is also the better-posed
target for *failure* selection ("help the weak region beyond generic help"). For the *value*
selector, where the goal **is** overall (common) improvement, **plain** `g_val` is the correct
choice — contrast would re-introduce the failure target we are deliberately dropping.

---

## 3. The reference-checkpoint correction (most important methodological point)

The score differentiates the loss **at a reference model**. The first failure-influence run used the
trained `pi0_ppc2sink_core` LoRA checkpoints (5k/10k/15k/19999) as that reference. That is **biased**
for two reasons:

1. **It bakes in one of the selections we compare against.** `core` is already fine-tuned on the
   failure-targeted data, so the reference is not neutral.
2. **The failure-region demos are already absorbed there** → low loss → tiny, noisy normalized
   gradients → the score *systematically under-values the failure region for a spurious reason*,
   contaminating the very "does it pick inside or outside the failure categories?" test.

**Fix — a neutral reference that matches the common fine-tune starting point.** Every arm =
`base(400) + 200`, all starting from pretrained pi0, so the shared prefix is a **base-only LoRA
warmup**: fine-tune pretrained pi0 on just the 400 base demos for 6k steps, saving 2k/4k/6k
(`pi0_ppc2sink_basewarmup`, dataset `ppc2sink_base_only`). The value scorer differentiates there.

**Insight 3 — why a warmup and not pretrained pi0 directly.** Pretrained pi0 is unbiased but:
(a) it has *no* LoRA adapters — they are added fresh, initialized so the adapter is a no-op (one of
the two LoRA matrices is 0), and at that point the gradient w.r.t. that half of the LoRA params is
*identically zero* for every demo → the influence feature collapses onto half its dimensions at a
degenerate fixed point; (b) the base-only checkpoint is not a full LoRA train-state (`model.load`
wants one). A few thousand warmup steps move the adapters off that point, giving **non-degenerate,
loadable, in-regime** LoRA gradients while staying selection-neutral.

---

## 4. D_val (the target distribution)

- **Failure arm:** D_val = held-out demos from the 10 failure categories (77 demos).
- **Value arm:** D_val = **class-balanced** across all 79 categories — up to 4 per category, capped
  by availability → **313 demos** (targeted-10 ≈ 13%, their *balanced* share). Deliberately **not**
  a frequency-uniform sample, which would over-weight common objects and make "value" mean "help the
  common categories"; the balanced D_val mirrors the per-category (stratified) eval.

**Insight 4 — direction vs magnitude in the 1-step sanity check.** The value smoke (`vsmoke`) asks:
do the top-scored demos, applied as a 1-step update, reduce D_val loss more than random? Two
variants were measured at the neutral reference:

| 1-step update (balanced D_val=157, warmup-2000) | TOP-20 | random | BOTTOM-20 | |
|---|---|---|---|---|
| **unit-norm step (direction)** | **−0.00037** | −0.00027 | +0.00011 | **PASS** |
| raw step (sign + scale) | −0.00002 | −0.00004 | +0.00002 | weak |

The **direction** test passes cleanly (top < random < bottom); the **raw** test is weak because
high-cosine (well-aligned) demos tend to have *smaller* gradient norms — they are lower-loss/"easier"
— so their raw 1-step step is tiny and moves the loss less than random's larger steps. Since
selection is on the **normalized** gradient cosine, the unit-norm test is the faithful check, and it
holds at the unbiased reference.

---

## 5. Results

### 5a. What each method selects (`selected_targeted_frac` = fraction of top-200 in the 10 failure cats)

| selection | targeted-frac | overlap w/ core | notes |
|---|---|---|---|
| `core` (P(fail) heuristic) | 1.00 | — | top-10 failure categories by construction |
| **failure-influence** | **0.23** | 30/200 (15%) | 3.4× the pool's ~7% prevalence; enriches failure cats *and* adds non-failure demos that gradient-align with the weak region (egg, whisk, cheese…) |
| **value-influence** | **0.06** | 4/200 | **below** even random/pool (~7%); loads common value-dense objects (tongs 14, dish_brush 12, mug 11, rolling_pin 8, cup, strainer, straw…). score range [−0.07, 0.28] = real separation |
| random / pool prevalence | ~0.07 | — | control |

The value and failure selections are **near-disjoint**: value ∩ failure-influence = 5/200, value ∩
core = 4/200. "Value" data and "failure" data are essentially different sets of demos.

> ⚠️ Caveat: the **failure-influence** selection (0.23) was scored at the *core* reference (later
> identified as biased, §3); the **value-influence** selection (0.06) uses the corrected *neutral*
> reference. The value number is the unbiased one. (Re-scoring the failure arm at the neutral
> reference is a clean follow-up but not yet done.)

### 5b. Eval — COMPLETE (overall n=300, scene-paired seed 100000)

| arm | overall | targeted-10 (n≈37) | non-targeted (n≈263) |
|---|---|---|---|
| **core** (P(fail) heuristic) | **0.593** | 0.432 | 0.616 |
| baseline (no FT) | 0.580 | 0.333 | 0.617 |
| random | 0.563 | 0.282 | 0.605 |
| **failure-influence** | 0.553 | 0.421 | 0.573 |
| coverage | 0.517 | 0.429 | 0.528 |
| **value-influence** | **0.503** | 0.361 | 0.523 |

**Neither gradient-influence method beat the trivial category heuristic (`core`) or even `baseline`
on overall success.**
- **`core` is the only arm that beats baseline overall**, and it wins — the sophisticated LESS/TracIn
  machinery lost to "just add demos from the 10 categories you fail on."
- **failure-influence worked *where aimed* but didn't translate.** On the failure region it ≈ matched
  core (0.421 vs 0.432, both ≫ baseline 0.333), so gradient-targeting genuinely helped the weak spots
  — but it also regressed the non-targeted majority (0.573 vs ~0.616), and that **ate the targeted
  gains** → overall 0.553, *below* baseline.
- **value-influence is worst** (0.503) via broad non-targeted forgetting (see `VALUE_INFLUENCE_POSTMORTEM.md`).

**The non-targeted column predicts the whole ranking:** every arm that disturbs the non-targeted
majority loses overall; **core is the only one that doesn't** (0.616, untouched). core adds *depth on
the failing categories* surgically; both influence methods perturb the *shared grasp behavior*
(value: +25 `fail_no_grasp` vs baseline) and pay for it broadly.

**Statistical honesty (n=300, paired McNemar):** `value < core` is solid (p≈0.02). Everything else is
within noise — `failure-influence vs core` Δ−0.040 (p≈0.30, NS); failure-influence ≈ random ≈ baseline
overall; targeted-10 cells (n≈37) cannot separate core/failure-influence/coverage. Defensible claims:
**core is (weakly) best and the only arm beating baseline; value is significantly worse than core; the
influence methods did not beat the heuristic.**

> Note: an earlier *partial* (hung) stratified eval reported failure-influence at 0.406 on the
> targeted-10 from a ~70-episode snapshot; the **complete** overall eval supersedes it (targeted-10 =
> 0.421, overall = 0.553).

---

## 6. Insights / takeaways

1. **Last-layer TracIn is insufficient for object-conditioned selection here** — the action readout
   is motion-dominated and object-agnostic. Use the LoRA-adapter gradient (full forward, cheap
   backward; ~as fast as last-layer because the forward dominates).
2. **A common-mode gradient dominates per-demo gradients.** Plain cosine to the mean is nearly flat
   across demos. Subtract a reference (contrast) to expose *specific* directions; use plain only when
   the goal genuinely is the common/average improvement (the value selector).
3. **The reference checkpoint is the crux.** It must be (a) selection-neutral and (b) in the actual
   fine-tuning regime with non-degenerate LoRA gradients → a short base-only warmup, not the trained
   arms and not the cold pretrained model.
4. **Selection is directional; validate it directionally.** The natural "1-step update reduces
   val-loss" check conflates direction with gradient magnitude; a unit-norm step isolates what the
   cosine score actually selects on.
5. **The selection criterion (`selected_targeted_frac`) is near-disjoint across methods** — core 1.00,
   failure-influence 0.23, value 0.06 (value ∩ failure = 5/200). The value selector, optimizing
   *balanced overall* loss, picks almost entirely *outside* the failure region. BUT (see #6) this did
   **not** make it a better selector — the demos it called "valuable" hurt — so it is **not** clean
   evidence about "failure ⟹ missing data"; it shows the proxy mis-identified value.
6. **Headline empirical finding — the gradient-influence methods LOSE to the trivial heuristic.**
   Both failure- and value-influence finished *below* `baseline` overall; `core` (top-10 failure
   categories) won. The unifying mechanism is **catastrophic forgetting of the shared grasp behavior**:
   any selector that broadly nudges the policy (spread = coverage, or gradient-alignment = both
   influence arms) regresses the non-targeted ~88% majority and sinks overall; only `core`'s surgical
   *depth-on-failing-categories* leaves the majority intact. **Surgical, distribution-preserving
   targeting beats sophisticated influence selection here.**
7. **Why value-influence specifically failed** (full analysis in `VALUE_INFLUENCE_POSTMORTEM.md`):
   averaging 313 heterogeneous unit gradients into one *plain* `g_val` collapses onto the common-mode
   "generic pick-place" direction (97% of candidates have positive cosine to it; hard/easy AUC 0.35),
   so ranking by cosine pushed the budget to the +2.54σ tail onto one over-represented handle/grasp
   cluster (tongs/dish_brush/mug/rolling_pin, 0.36 vs random 0.21) — overwriting the base's good grasp.
   Magnitude-blind normalized cosine compounds it by favoring low-gradient "easy" demos.

**Caveats:** single task, single seed batch, one fine-tune run per arm; at n=300 most overall gaps are
within noise (only `value < core` is significant, p≈0.02); the failure-arm *selection* used the biased
core reference (re-scoring at the neutral reference is a clean follow-up). Directional, not definitive.

---

## 7. Artifacts & reproduction

- **Scorer:** `policy_analysis/influence_score.py` — modes `probe|smoke|vsmoke|full`; knobs
  `--mode_grad {lastlayer,lora,...}`, `--gval {plain,contrast}`, `--dval {targeted,balanced,uniform}`,
  `--ckpt_base`, `--steps`, `--real_dim`, `--k`, `--n_select`.
- **Selections:** `weakregion/influence_core.json` (failure), `weakregion/value_core.json` (value);
  arms `weakregion/arms_{influence,value}.json`; datasets `/data/xinyua11/ft_arms/ppc2sink_base_{influence,value}`.
- **Reference:** `pi0_ppc2sink_basewarmup` → `openpi/checkpoints/pi0_ppc2sink_basewarmup/warmup_v1/{2000,4000,6000}`.
- **Configs (openpi):** `pi0_ppc2sink_{influence,value,basewarmup}` — each dataclasses-verified
  identical to `pi0_ppc2sink_core` except `data_dirs` (invariant: only the data differs).

```bash
# value selection (neutral reference, balanced D_val, plain g_val)
python policy_analysis/influence_score.py full --mode_grad lora --gval plain --dval balanced \
    --n_val_per_cat 4 --real_dim 12 --k 8 --n_select 200 --steps 2000,4000,6000 \
    --out weakregion/value_core.json
# then: build_arms.py --core_file value_core.json --base_file arms.json  ->  arms_value.json
#       build_lerobot_subset.py --which base+core  ->  ppc2sink_base_value
#       scripts/train.py pi0_ppc2sink_value --exp_name value_v1   ->  eval vs core/random
```
