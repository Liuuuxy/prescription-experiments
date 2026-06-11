---
name: first-experiment-pickplace-sink
description: First experiment plan — baseline vs targeted-vs-random extra demos on PickPlaceCounterToSink
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

Start with a **single** RoboCasa task, not the full benchmark: **PickPlaceCounterToSink** (simple to debug, but real variation in object position, object type, grasp, transport, placement).

Experiment steps:
1. Train baseline diffusion policy on PickPlaceCounterToSink with a fixed number of demos.
2. Evaluate on held-out variations.
3. Log *where* it fails (object position, object type, scene variation, task phase).
4. Identify high-failure regions.
5. Compare two retraining settings, same demo budget: (a) baseline + random extra demos; (b) baseline + targeted extra demos from high-failure regions.
6. Measure improvement-per-added-demo; check whether targeted data helps the weak region without hurting other regions.

Expected outcome: a small but complete experiment showing whether targeted > random data addition. If it works, scale to more tasks.

**POWER ANALYSIS (`policy_analysis/power_analysis.py`, 2026-06-11):** the naive design (few seeds, 50 rollouts) would be badly UNDERPOWERED. Per-arm seeds needed for 80% power (p=0.45, weak-region fraction 0.5): detect Δ=15% → ~5-9 seeds (feasible); Δ=10% → ~12-24 seeds; Δ=5% → ~50-125 seeds (infeasible). MDE for S=3,R=100 is ~20-28% (unrealistically large). Monte-Carlo confirms low power for small budgets. **sigma_seed (between-seed SD of DP success) is THE dominant unknown — measure it FIRST by training 2-3 baseline seeds.** Combined with the weak targeting signal (geometry R^2~0.08 → small realized Δ), the experiment must either properly power up (>=5 seeds x >=200 rollouts, +variance reduction: PAIRED same-seed design, continuous progress metric, larger weak-region eval fraction) OR be redesigned for a LARGER effect (stronger uncertainty-based acquisition signal per [[pi0-weakregion-finding]], and/or a weaker baseline where added data moves the needle more). See [[robocasa-compute-and-training-stack]].

Verified facts (2026-06): PickPlaceCounterToSink exists in robocasa/utils/dataset_registry.py with both `human_path` and `mg_path` (MimicGen-generated) demo sets under v1.0/pretrain/atomic. Open design question: how to *obtain* targeted demos — generate via MimicGen with controlled initial-state distributions, vs subsample a large existing pool by region. **Decided (2026-06): explore BOTH mechanisms first** (investigate what collect_demos.py / MimicGen actually allow) before committing — this is the queued next task after install verification. See [[robocasa-compute-and-training-stack]].
