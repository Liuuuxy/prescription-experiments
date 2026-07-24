# Bandit v1 — Concrete Walkthrough (2026-07-24)

How the pipeline actually works, shown on the run's real artifacts. Provenance labels used throughout:
**REAL** = computed from on-disk run data. **PREVIEW** = the shipped machinery executed for real, but on
the partial-data map (diagnosis is 31% complete at snapshot time); the frozen run recomputes these on the
full batch. **ILLUSTRATIVE** = invented plausible inputs fed to real code to demonstrate a mechanism.

Status at snapshot (2026-07-24 13:12 PDT): `pi_0` trained (20k steps, seed 1000, ckpt `pi0_ppc2sink_pi0base/pi0_v1/19999`);
300/300 conditions captured; diagnosis batch running — 750/2,400 rollouts done (94/300 conditions);
E not yet built (it needs the full map); no pulls yet.

---

## 0. Setup recap (design §1)

- **D0** = 400 fixed MimicGen episodes (`weakregion/arms.json:base_episodes`); **pi_0** = pi0 LoRA
  finetuned on D0 alone with the standard recipe (20k steps, batch 32, seed 1000).
- **Well W** = the other 9,485 pool episodes — the retrieval source for pulls.
- **E** = 150 saved-state eval starts ×3 repeats per eval (built after the map: stratified 50 easy /
  50 mid / 50 hard by p̂ terciles, seeds from 500000, frozen + hashed; demos within 0.10 m same-category
  of any E start are barred from every pull). §1 below shows exactly what one saved start looks like —
  E starts use the identical format.

---

# A. The 300 diagnosis conditions (REAL)

## The 300 diagnosis conditions

**Design in one line.** Each of the 300 saved starts is binned into a 27-cell grid: category-difficulty tercile (t0 = hardest third by prior per-category success rate, t2 = easiest; frozen map in `bandit_v1/ledger/diag_tercile_map.json`) × object-height bin (h0: h < 0.08 m, h1: 0.08–0.212 m, h2: > 0.212 m) × along-counter offset bin on |x_rel| (x0: < 0.325 m, x1: 0.325–0.65 m, x2: > 0.65 m), with a per-cell target of ceil(300/27) = 12.

The scan walked env seeds 600000–602053 (2,054 candidate captures) and kept 300 — a ~14.6% acceptance rate. After scan index 2000 the overflow rule kicked in: 21 rows (7%) whose natural t0 cells were already full were redirected into the nearest undersubscribed cells (t1h2x0 ×9, t1h2x1 ×8, t1h2x2 ×4) rather than discarded.

**Realized per-cell histogram.** 24 of 27 cells hit the full target of 12; the three sparse cells are all tall-object (h2) corners of the easier terciles — tall + easy-category starts are genuinely rare in the live distribution (h lands in h0 ~66% of the time):

| tercile \ (hbin, xbin) | h0x0 | h0x1 | h0x2 | h1x0 | h1x1 | h1x2 | h2x0 | h2x1 | h2x2 |
|---|---|---|---|---|---|---|---|---|---|
| **t0 (hard)** | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| **t1 (mid)**  | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | **7** |
| **t2 (easy)** | 12 | 12 | 12 | 12 | 12 | 12 | **4** | 12 | **1** |

Marginals: tercile 108/103/89, height bin 108/108/84, |x_rel| bin 100/108/92.

**Category coverage.** 73 distinct categories (of 81 in the tercile map). Top 8 by count: dish_brush 18, can 14, wine 13, pitcher 9, ladle 8, shrimp 8, juice 8, jug 7 — the head is dominated by hard-tercile and tall categories, exactly the ones the quota forces the scan to over-collect relative to their natural frequency.

**Knob ranges.**

| knob | min | q25 | median | q75 | max |
|---|---|---|---|---|---|
| h (m) | 0.007 | 0.056 | 0.100 | 0.159 | 0.260 |
| w (m) | 0.057 | 0.090 | 0.120 | 0.201 | 0.367 |
| x_rel (m) | -0.840 | -0.440 | 0.261 | 0.485 | 0.830 |
| y_rel (m) | -0.886 | -0.482 | 0.283 | 0.458 | 0.792 |

side exactly balanced (150 / 150); yaw effectively 5 discrete values; 59 distinct layouts and 60 styles — essentially every kitchen is unique to a handful of starts.

**Six concrete example conditions:**

| start_id | seed | category | h (m) | x_rel | y_rel | layout | style | cell |
|---|---|---|---|---|---|---|---|---|
| start_00075 | 600075 | saucepan_lid | 0.062 | 0.321 | -0.573 | 8 | 8 | t0h0x0 (hard, short, near) |
| start_00116 | 600116 | wine | 0.260 | -0.742 | -0.394 | 24 | 27 | t0h2x2 (hard, tall, far) |
| start_00003 | 600003 | bowl | 0.103 | 0.406 | 0.674 | 48 | 12 | t1h1x1 (all-middle) |
| start_00044 | 600044 | orange | 0.061 | -0.698 | -0.537 | 59 | 14 | t1h0x2 (mid, short, far) |
| start_00000 | 600000 | squash | 0.057 | 0.325 | -0.492 | 40 | 46 | t2h0x0 (easy, short, near) |
| start_02024 | 602024 | kiwi | 0.047 | 0.661 | 0.397 | 21 | 4 | t2h2x2 (rarest cell, n=1) |

## Anatomy of a saved start (`bandit_v1/states/diag/start_00000/`)

Each kept condition is a self-contained, exactly-restorable episode start produced by one `env.reset()` at its seed (600000 here: a squash on the counter right of the sink, layout 40 / style 46). Restoration replays `env.reset(); env.reset_to({states, model, ep_meta})`. The MuJoCo state vector is (166,) float64 for this scene; length varies 166–272 across layouts.

| file | size | contents |
|---|---|---|
| `state.npz` | 1,298 B | flattened MuJoCo sim state, shape (166,), float64 |
| `model.xml` | 353 KiB | fully-compiled scene MJCF (kitchen + fixtures + objects); dominates dir size |
| `ep_meta.json` | 8.8 KiB | layout/style ids, object_cfgs, fixtures, cam configs, robot base pose; `lang` = "Pick the squash from the counter and place it in the sink." |
| `fingerprint.json` | 274 B | category, exact mjcf instance path (`.../squash/squash_7/model.xml`), layout/style, obj_xyz, base_xy — obj_xyz − base_xy reproduces the ledger's (x_rel, y_rel) exactly |

The target entry in `object_cfgs` pins the exact instance and its placement sampler (a 0.3 × 0.4 m region on the counter beside the sink, referenced to the sink fixture).

**Scale:** the 300 kept dirs total 158 MiB (mean 539 KiB, dominated by model.xml); the full 2,054-dir scan cache kept for free resume is 1.07 GiB.

---

# B. Diagnosis results so far (REAL, 31% snapshot)

## B.1 Batch progress and headline numbers

Single snapshot read of the live ledger at 2026-07-24 13:12:43 PDT: **750 of 2,400 planned rollouts** (94/300 conditions; 93 complete at 8/8 repeats).

| Snapshot fact | Value |
|---|---|
| Overall success rate | **393/750 = 52.4%** |
| Grasp-phase failures (never_reached + reached_no_grasp) | **78.4% of all failures** |
| Categories seen | 55 (median 2 conditions per category) |

**5-stage outcome distribution:**

| Stage | Count | % |
|---|---:|---:|
| never_reached | 80 | 10.7% |
| reached_no_grasp | 200 | 26.7% |
| fail_grasped_no_transport | 60 | 8.0% |
| fail_reached_sink_no_place | 17 | 2.3% |
| success | 393 | 52.4% |

**All three grid axes move in the expected direction** — the prior tercile map and the height physics are validated on live data:

| Axis | Bins (easy→hard direction) | Success rates |
|---|---|---|
| Category tercile | t2 → t1 → t0 | **72.4%** → 46.3% → 40.2% |
| Height | h0 → h1 → h2 | **61.4%** → 45.5% → **17.2%** |
| Offset \|x_rel\| | x0 → x1 → x2 | **71.7%** → 47.3% → 48.3% |

**Per-condition repeatability** (93 complete conditions): success-count histogram 0/8→8/8 = [14, 7, 9, 9, 11, 6, 11, 11, 15]. **31.2% of conditions are all-or-nothing** (0/8 or 8/8) vs 0.8% expected under a shared-rate binomial — per-condition difficulty is real and large. At the same time 64/93 conditions are mixed (1–7 successes) — rollout stochasticity is also substantial, which is exactly why m=8 repeats.

## B.2 Three concrete conditions (8 repeats each)

| | Easy | Mixed | Hard |
|---|---|---|---|
| start_id | start_00000 | start_00002 | start_00057 |
| category | squash | sweet_potato | wine |
| h (m) | 0.057 | 0.053 | **0.260** |
| cell | t2h0x0 | t2h0x1 | t0h2x1 |
| outcome | **8/8** | **4/8** | **0/8** |

- **Easy**: `success` ×8 — solved every time.
- **Mixed**: success ×4, reached_no_grasp ×2, fail_reached_sink_no_place ×2 — the *identical* saved state produces three distinct failure stages across repeats: pure policy stochasticity, the noise the repeat design averages over.
- **Hard**: never_reached ×1, reached_no_grasp ×7 — the arm reaches the wine bottle in 7 of 8 attempts and never once grasps it: the canonical tall-object grasp failure, reproduced under controlled repeats.

## B.3 Fitting p̂(x) — PREVIEW on partial data

> **PREVIEW.** Fit on the 750-row snapshot to exercise the pipeline; the frozen fit happens on the full
> 2,400 rows, and the production gate would currently refuse to write this model.

Model: L2 logistic (p_success) + multinomial logistic (p_stage) on `[h, w, x_rel, y_rel, side, h², cat_te]`
(cat_te = k=8 shrinkage target encoding of category; GroupKFold by condition, cat_te refit per fold — no leakage).

| Metric (held-out) | Value |
|---|---:|
| AUC (p_success) | **0.542** — below the 0.55 production gate |
| Log-loss | 0.832 (stage log-loss 1.454) |
| In-sample AUC, for contrast | 0.821 |

The gap is the classic overfit-encoding signature: 55 categories over 94 conditions = median **2 conditions
per category**, so fold-local cat_te barely transfers. The full batch roughly triples conditions-per-category —
whether the frozen fit clears the gate is the thing to watch. Mid-range calibration is decent; both extremes
are inverted (over-confident fold-local cat_te).

**Category encoding extremes** (global mean 0.524): hardest = tupperware 0.175, pitcher 0.175, wine 0.230,
egg 0.258, pear 0.262; easiest = corn 0.841, bell_pepper 0.800, peach 0.800, kiwi 0.762, mushroom 0.762.
Tall/awkward vessels + the fragile egg vs compact convex produce — coherent with the known grasp weakness.

**Physics-check probe (honest negative, PREVIEW):** holding a mid category fixed and moving only coordinates,
tall-far (h=0.25, x=0.7) scores p̂ 0.559 vs short-near 0.468 — inverted vs the known physics. Diagnosed, not a bug:
(1) the height signal is currently carried by category identity (corr(h, cat_te) = −0.455; only juice/pitcher/wine
occupy h2 so far) — the model DOES reproduce the height physics on real rows (predicted 0.614/0.458/0.159 by h-bin
vs empirical 0.614/0.455/0.172), just through the category channel; (2) x_rel enters signed-linear while the
empirical difficulty is a V in |x_rel| (71.7% near vs ~47–48% both far sides). Both should be re-checked on the
frozen fit; (2) suggests adding an |x_rel| feature (commensurable with pool demos) **before** the frozen fit.

---

# C. Clusters → arms + wells + B (PREVIEW)

> Machinery run for real on the real 300 conditions + real 9,485-demo well, using the PREVIEW map.
> The frozen clustering reruns on the full-data map; k=3-vs-4 is within refit noise, regions will be
> recomputed, and naming is a human hard-stop.

**choose_k silhouette sweep** (KMeans n_init=50, rs=0; cap ≤6 clusters + Random):
k=3: **0.2777** ← chosen; k=4: 0.2756; k=5: 0.2515; k=6: 0.2411 (k=7/8 computed, ineligible).
All clusters clear the 5% merge floor — merge was a no-op.

**Cluster cards** (the human-naming input):

| | UNNAMED_0 | UNNAMED_1 | UNNAMED_2 |
|---|---|---|---|
| size (share) | 100 (33.3%) | 126 (42.0%) | 74 (24.7%) |
| mean p̂ | 0.634 | 0.602 | **0.193** |
| dominant stage | success | success (0.96) | **reached_no_grasp (0.77), never_reached (0.20)** |
| top-5 categories | dish_brush, sweet_potato, saucepan_lid, straw, glass_cup | can, dish_brush, ladle, shrimp, jam | **wine, pitcher, juice, jug, bowl** |
| centroid h / w | 0.088 / 0.145 | 0.087 / 0.137 | **0.176** / 0.173 |
| centroid side | −1.00 (pure) | +1.00 (pure) | −0.35 (mixed) |
| candidate name | easy_side_neg | easy_side_pos | **tall_vessel_no_grasp** |

The pipeline rediscovers the project's weak region unsupervised: one cluster of tall, wide vessels at ~2×
the easy clusters' height that the policy reaches but cannot grasp. The two easy clusters split almost purely
on counter side, not difficulty.

**Wells + B rule:**

| arm | well count | share of W | well mean p̂ |
|---|---:|---:|---:|
| UNNAMED_0 | 3,878 | 40.9% | 0.649 |
| UNNAMED_1 | 4,299 | 45.3% | 0.611 |
| UNNAMED_2 | **1,308** | 13.8% | **0.188** |
| random | 9,485 | all of W | — |

B rule: max b in (200,100,60,20) with min cluster well ≥ 3b. Limiting arm UNNAMED_2: 1,308 ≥ 600 → **B = 200**,
with 2.2× headroom. The weak arm's well mean p̂ (0.188) matches its cluster mean (0.193) — assignment is coherent.

---

# D. One pull, end to end — worked example

Draw is REAL (shipped `draw.pull_demos` on the real pool, preview regions); commands are the REAL strings
`pull.py` executes (constructed, not run); the decision ledger is ILLUSTRATIVE, the decision code is real.

## D.1 The draw (REAL)

Arm UNNAMED_2, round j=1, deterministic seed `zlib.crc32(b"UNNAMED_2:1") = 112,618,713`:

| Stage | Count | Rule |
|---|---:|---|
| W ∩ region | **1,308** | nearest-centroid membership, D0 excluded |
| − ε-gate vs E | −0 | E not built yet in this preview; the real pull also filters same-category demos within 0.10 m of any of the 150 E starts |
| − D0-novelty gate | **−235** | must not near-duplicate the 400 base demos |
| = candidates | 1,073 | |
| uniform pre-draw | 600 | 3B, the pull's only randomness (makes redraws differ across pulls) |
| FPS keep | **200** | farthest-point sampling in standardized (h,w,x_rel,y_rel) |

Same seed → byte-identical redraw (verified). First 8 picks alternate short/tall, left/right, near/far —
FPS spending the diversity budget: tupperware h=0.058 | wine h=0.260 | saucepan h=0.138 | egg h=0.051 |
pitcher h=0.236 | kettle h=0.200 | yogurt h=0.045 | saucepan h=0.138. Full draw: 18/19 region categories,
max per-category 17 (vs 175 in the raw region — much flatter), h 0.045–0.260 m, side 109/91.
Selector diagnostics (logged, never used for selection): mean pairwise standardized-knob distance 3.243,
mean distance-to-nearest-D0 0.545.

## D.2 The commands (REAL strings, not executed here)

```bash
# 1. provenance: ledger/pull_arms/UNNAMED_2_j1.json = {base_episodes: [400 ids], pull_episodes: [200 ids]}
# 2. dataset (600 episodes)
python policy_analysis/build_lerobot_subset.py \
  --src .../PickPlaceCounterToSink/.../mg/demo/2025-08-20-22-32-27/lerobot \
  --arms bandit_v1/ledger/pull_arms/UNNAMED_2_j1.json --which base+pull \
  --dst /data/xinyua11/ft_arms/ppc2sink_bandit_UNNAMED_2_j1
# 3. slot swap
ln -sfn /data/xinyua11/ft_arms/ppc2sink_bandit_UNNAMED_2_j1 /data/xinyua11/ft_arms/ppc2sink_bandit_slot_a
# 4. train (~8h; paired round seed 1001; one auto-retry on failure)
python scripts/train.py pi0_ppc2sink_bandit_a --exp-name UNNAMED_2_j1 --seed 1001 --overwrite
# 5. serve slot a on port 8130, then eval_checkpoint: 150 frozen E starts x 3 repeats -> ledger
```

delta = mean over 3 repeats of (success rate on E) − b, plus per-stratum deltas, all in `pulls.parquet`.

## D.3 The decision (REAL `scheduler.decide`, ILLUSTRATIVE ledger)

Round 1 (one pull/arm; σ_e = 0.03 → half-width 1.6449·0.03/√1 = **0.0493**): with deltas
+0.050/+0.020/+0.015/+0.010 nothing eliminates — every rival's UCB clears the leader's LCB (+0.0007).
n=1 cannot kill anything; that is by design.

Round 3 (three pulls/arm; half-width **0.0285**):

| arm | mean Δ | lcb | ucb | fate |
|---|---:|---:|---:|---|
| UNNAMED_2 | +0.0550 | **+0.0265** | +0.0835 | leader |
| UNNAMED_0 | +0.0300 | +0.0015 | +0.0585 | survives; `tied_with_leader` (Δmean 0.025 ≤ σ_e) |
| random | +0.0117 | −0.0168 | +0.0402 | survives |
| UNNAMED_1 | −0.0067 | −0.0352 | **+0.0218** | **eliminated**: ucb < max lcb |

The mechanism in one line: UNNAMED_1's best plausible world (+0.0218) is now worse than the leader's worst
plausible world (+0.0265), so its remaining pull budget is reallocated. Real decide() output:
survivors=[UNNAMED_0, UNNAMED_2, random], eliminated={UNNAMED_1: 3}, next_round=4, tied_with_leader=[UNNAMED_0].

## D.4 Cost

One pull ≈ 8h train + 5h eval (450 rollouts × ~40 s) ≈ **13h GPU wall-clock**; two slots → ~2 pulls/day.
T=12–16 pulls + baseline + 2 null pulls ≈ a week and a half of machine time; every elimination buys back half a day.

---

## Reproducibility

Everything REAL above is recomputable from `bandit_v1/ledger/` + `bandit_v1/states/`; preview artifacts and the
analysis scripts live in the session scratchpad (`walkthrough/`: diag_snapshot.parquet, preview_map.joblib,
preview_arms.json, preview_regions.parquet, section_*_analysis scripts) and were deliberately NOT written into
the repo or ledger — the frozen map/arms/E do not exist yet and nothing here pre-commits them.
