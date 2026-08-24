# Defining bandit "arms" for data prescription + the eval-cost protocol (2026-07-22)

Design panel (4 lenses: BAI theory, real-robot experimentalist, mechanism/prescription, contrarian) + 2 adversarial judges. Winning design: **Gate-Race-Confirm** (attainability-gated, sim-warm-started fixed-budget BAI). Panel record: `factor_analysis/wf6_mab.json`. Builds on `PRESCRIPTION_CRITERION.md` and `FACTOR_ANALYSIS_REPORT.md`.

## 1. Define arms as the 3 dissociable failure MECHANISMS — not the factor groups directly

| arm | data recipe | why it's a distinct arm |
|---|---|---|
| **A1 far-approach** | new demos with placement biased far along the counter (\|offset from sink\|>0.65 m), approach/reach-rich; keep interior variation | approach/never-touch mechanism → reach demos |
| **A2 tall-grasp** (fold category + handled/lidded in here) | grasp-securing demos on tall objects (h>0.21 m) | grasp/touch-then-fumble mechanism → grasp demos |
| **A3 hard-style** | demos in the hard visual tercile (real robot: harder physical scenes; ≥3 styles) | perceptual mechanism → visual diversity |
| **A0 generic-new** (mandatory control) | matched-budget *random* new demos, identical LoRA recipe/seeds | subtracts "new data just helps" |
| + frozen base | no new data | retention denominator |

**Why mechanisms, not the groups you listed:**
- **NOT 79 category arms** — dead on power: per-category SR is ±12pp at n≈58, the measured per-category dose-response is r=0.06 (noise), and 53% of the category signal *is* height, so categories are collinear with the tall axis, not independent. Separating 79 arms would need ~79×300 ≈ 24k real rollouts.
- **NOT per-cell arms** (height×dist×style×category) — interaction terms add **exactly 0.000** held-out R², so a cell carries no information beyond the additive axes (redundant *by measurement*), and the informative cells are tiny (tall&far = 0.5% of episodes — can't even fill 300 configs).
- **Mechanisms are the coarsest partition that is still (a) estimable at real-robot budgets, (b) actionable — each maps to a genuinely different collection recipe, and (c) non-interfering** — additivity means "add far data" and "add tall data" have independent effects, so the arms don't confound each other.

Reality check on which arms are *live*: only **far-approach is a genuinely open sim bet** (the rim penalty was untouched by every prior fixed-pool arm — DD −0.4pp, p=1.0 — because none ever *added* far demos). **Tall almost certainly fails the free attainability gate** (pool 2.7% tall, local MimicGen 0/20) and routes to the human/G1 collection ledger tagged with its 17% deployment mass. **Style is unrecoverable in sim** — a diversity constraint inside every arm, and a physical-scene arm only on the G1.

## 2. Regret-minimizing MAB is the WRONG frame — use cost-aware fixed-budget BAI

The bandit *vocabulary* (arms/pulls/allocation) is fine scaffolding, but UCB1/Thompson/EXP3 are theater here, on four verified counts:
1. **Pulls are few and expensive** (each G1 rollout = minutes of supervised robot time; whole-project budget ~5–15 pulls). Regret bounds are asymptotic in T and vacuous; this is a small *designed experiment*, not online learning.
2. **Rewards saturate and go negative** (knee at 20–60 demos; 610 demos hurt, 0.371→0.299). "Keep pulling the winner" is actively harmful — dose is a within-arm knee, not a repeatable Bernoulli pull.
3. **Rewards are coupled** through the shared retrained policy (adding A2's data changes A1's slice via forgetting). Arms are components of an allocation vector over a mixture simplex, not independent arms. Only *baseline* additivity is verified; treatment-effect additivity is an unverified prior (test it with an interaction re-fit before funding a mixture).
4. **The goal is best-arm identification** (pick which to invest in), i.e. pure exploration, not cumulative-regret minimization.

→ **Cost-aware fixed-budget BAI**, run as probe-then-allocate over 3 phases:
- **Phase 0 (free, no GPU):** attainability gate — 30 capped collection attempts per (mechanism, cheapest source), Wilson CI. Drop unattainable arms and route them to the human/G1 ledger with their deployment mass. *This gate, not the bandit, is the primary decision instrument.*
- **Phase 1 (sim, cheap, legitimately a race):** Sequential-Halving / Successive-Rejects over survivors, config-clustered + CUPED + mixed-config enrichment, run-to-CI, eliminate on UCB95 < threshold, ≤3 arms, mismatched-placebo causal control. ~3,000 sim rollouts ranks the arms.
- **Phase 2 (real robot, fixed budget):** only the ≤2 sim survivors + control + base — a confirmation + retention-safety gate, not a discovery instrument.

## 3. The reward

r(arm) = [ΔSR_target(arm − base) − ΔSR_target(control − base)] − λ·(retention loss on a shared slice), config-clustered on a shared paired seed list, λ set so r equals the **deployment-mass-weighted whole-task SR change**. The control subtraction is forced by core−random = +2pp ns; the retention term by the value-arm −9.4pp forgetting crater. Cap dose at ~2× the 20–60 knee.

## 4. Eval cost — the answer to "same number of evals on every arm?"

**No — and equal allocation is suboptimal.** Details:

**Power (SE of a config-clustered paired ΔSR = √(2·p(1−p)·0.78/n); the 0.78 is the irreducible rollout-stochastic share, config structure cancels under pairing):**

| n / arm | SE | resolves at 80% power |
|---|---|---|
| 150 | 5.0 pp | ~14 pp |
| **230** | 4.0 pp | **~11 pp (the region lift)** |
| 300 | 3.5 pp | ~10 pp net |
| 1,100 | 1.8 pp | ~5 pp |
| ~7,000 | 0.7 pp | ~2 pp |

**The reviewer's correction (proposals got this wrong):** the net contrast τ = SR_arm − SR_control uses the *same base and same paired seeds*, so **the base cancels — SE(τ) is the single-paired SE, NOT √2 larger.** A 10 pp net effect needs **~300 config-clustered rollouts/arm**, not ~600. Several proposals inflated this by ~40%.

**The load-bearing consequence:** the *realistic* prescribed-vs-generic premium is ~2 pp (core−random was +2 pp ns) — resolving it needs ~7,000 rollouts/arm, **impossible**. Do **not** pre-register "arm beats control" at its true margin as the win condition, or you are guaranteed to "fail." Instead power for:
- the **+11 pp region lift** (arm vs base; n≈230), and
- **upstream mechanism endpoints** with bigger effects: tall grasp-rate 0.59→0.20 = 39 pp, far object-touched +7.8 pp, and
- the **never-succeed-config flips** (a high-power binary endpoint),
- and report the premium as "within noise vs the 10–20 pp attainable bound."

**Allocation:** run **base + control + retention slice ONCE** and reuse them (paired) across all arms — this amortization is what makes it affordable. Then:
- *Sim:* **adaptive** — racing puts rollouts on arms whose CI straddles the threshold, eliminates dominated arms early.
- *Robot:* **near-equal** across the ≤2 survivors (their variances match; saturation ⇒ one pull ≈ full info, so adaptivity is phase-level only), with one LUCB top-up to the closest pair and a **futility stop** (kill any arm with retention < −8 pp).

**Budget arithmetic (real robot):** a fully-instrumented policy ≈ 230 (target) + ~300 (retention) ≈ **530 rollouts**. At ~3 min/rollout that's ~26 robot-hours/policy. So **at 300–1000 total real rollouts you can confirm base + control + 1–2 mechanism arms only** — never K≥3 real arms, never per-band slices for multiple arms, never the placebo on the robot. That forces the division of labor: **sim ranks (cheap, ~3k rollouts), the robot confirms sign/magnitude of 1–2 survivors + the retention gate** (~100 robot-hours ≈ 2–3 supervised weeks).

## 5. Honest expected outcome

Given core−random = +2 pp ns and the generation ceiling, the likely result is a **double null on the premium plus one positive exception (far-approach)**. The deliverable is then the **attainability ledger** — "far collectable and marginally teachable; tall neither → quantified human/G1 collection bill" — which *is* the generation-ceiling result, publishable by design. Build the experiment so that outcome is a claim, not a failure.
