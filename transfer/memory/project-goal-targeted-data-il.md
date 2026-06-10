---
name: project-goal-targeted-data-il
description: Core research goal — data-efficient imitation learning via targeted demonstrations from policy failure regions
metadata: 
  node_type: memory
  type: project
  originSessionId: 4f31bea1-2bdd-4e89-9f93-63656e3f0e12
---

The project uses RoboCasa as a simulation testbed to study **data-efficient imitation learning**. Central hypothesis: adding targeted demonstrations from a policy's **failure regions** improves a diffusion policy more efficiently (better improvement per added demo) than adding the same number of **random** demonstrations.

Broader goal: a method that evaluates a trained policy, identifies weak regions / failure modes, and recommends *what type* of data to add next — moving beyond "just add more data." Connects back to a real-robot pouring / data-selection problem.

See [[first-experiment-pickplace-sink]] for the concrete first experiment and [[robocasa-compute-and-training-stack]] for the tooling/compute realities.
