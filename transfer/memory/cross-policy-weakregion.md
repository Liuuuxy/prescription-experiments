---
name: cross-policy-weakregion
description: "pi0 vs GR00T weak-region comparison — both fail identically on tall objects at the grasp (universal, data-addressable bottleneck)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

Cross-policy weak-region check (2026-06-11), PickPlaceCounterToSink, via `policy_analysis/predict_failure.py` on each policy's weakregion.json. GR00T run: `Isaac-GR00T/scripts/analyze_groot_weakregions.py` (instrumented SimulationInferenceClient, n=100). Output `weakregion/groot_PickPlaceCounterToSink/`.

**GR00T independently replicates pi0's failure structure almost exactly:**
| metric | pi0 (n=150) | GR00T (n=100) |
| overall success | 52.7% | 56.0% |
| dominant failure mode | 86% no-grasp | 80% no-grasp |
| predictor CV AUC (geometry->success) | 0.628 | 0.629 |
| dominant predictor (coeff) | height (-0.57) | height (-0.78) |
| tall objects (>10cm) success | ~36% | ~32% |
| short objects success | ~67% | ~66% |

**Interpretation:** two very different architectures (pi0 flow-matching VLA; GR00T diffusion+Eagle VLM) fail at the same rate, same mode (grasp), same objects (tall), same height effect. => the tall-object grasp weakness is **UNIVERSAL** (a real task property, not a pi0 quirk). Upgrades the height signal from "weak/possibly-noise" to "weak per-episode (instance variance dominates, AUC 0.63) BUT a robust population-level bottleneck." Both policies failing together would usually mean structural/intrinsic, BUT the embodiment-limit test ([[pi0-weakregion-finding]]) cleared the gripper aperture (objects ARE graspable) => it's a **shared training-DATA/skill gap** (both trained on the same RoboCasa data under-serving tall-object grasps), i.e. DATA-ADDRESSABLE. Good for the targeted-data hypothesis.

Caveat: aggregate concordance is high, so there's little policy DISAGREEMENT to exploit as an epistemic acquisition signal here (couldn't do exact same-init-state episode pairing — pi0 used seeds 0..149, GR00T used the client's internal RNG). The robust target is "tall objects" at the population level; the per-episode signal needs instance-level or policy-uncertainty features ([[pi0-weakregion-finding]]). Power: moderate effect => see [[first-experiment-pickplace-sink]] power analysis (need >=5 seeds x >=200 rollouts + variance reduction).
