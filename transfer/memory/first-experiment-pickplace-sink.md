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

Verified facts (2026-06): PickPlaceCounterToSink exists in robocasa/utils/dataset_registry.py with both `human_path` and `mg_path` (MimicGen-generated) demo sets under v1.0/pretrain/atomic. Open design question: how to *obtain* targeted demos — generate via MimicGen with controlled initial-state distributions, vs subsample a large existing pool by region. **Decided (2026-06): explore BOTH mechanisms first** (investigate what collect_demos.py / MimicGen actually allow) before committing — this is the queued next task after install verification. See [[robocasa-compute-and-training-stack]].
