---
name: pi0-weakregion-finding
description: "pi0's weak-region result on PickPlaceCounterToSink — fails at grasp on tall/awkward objects; the concrete targeting signal for the experiment"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

First weak-region result (2026-06-10), pi0 on PickPlaceCounterToSink, n=50, via `~/robocasa_experiments/policy_analysis/analyze_pi0_weakregions.py`. Output `~/robocasa_experiments/weakregion/pi0_PickPlaceCounterToSink/`.

**pi0 overall ~52-58% (this run 26/50). Dominant failure mode: `fail_no_grasp` = 96% of failures** — pi0 almost never fails at transport/placement; it fails at the GRASP.

Object-type driven: **0%** on tall/cylindrical/awkward-to-grasp objects (juice, kiwi, glass_cup, water_bottle, jug, jug_wide_opening, blender_jug, pitcher, rolling_pin, reamer, cheese_grater, squash, spray, bar_soap, straw); **100%** on compact graspable food (orange, tomato, peach, carrot, eggplant, lemon, steak, fish, milk, jar…). Worst spatial regions: far/lateral — mid-left 29%, far-right 33%, far-center 38%; best near-left 89%.

**Implication for [[first-experiment-pickplace-sink]]:** the targeting signal is concrete — targeted arm = grasp demos for **tall/awkward objects at far/lateral positions**; random arm = full-distribution demos; same budget, compare improvement-per-demo and whether targeted helps the weak region without hurting others. Generate the data with the [[expert-data-generation-loop]] (pi0/GR00T recorder + LeRobot convert), constraining init states for the targeted arm. Next: GR00T cross-policy weak-region (do both strong policies fail on the same objects/regions? = universally-hard targets).
