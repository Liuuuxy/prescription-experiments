# Memory Index

- [Project goal: targeted-data IL](project-goal-targeted-data-il.md) — hypothesis that targeted failure-region demos beat random demos for data-efficient imitation learning
- [First experiment: PickPlaceCounterToSink](first-experiment-pickplace-sink.md) — concrete baseline vs targeted-vs-random demo experiment on one task
- [Compute & training stack](robocasa-compute-and-training-stack.md) — 12GB 4070 dev box vs H100 for training; RoboCasa diffusion_policy fork; key scripts
- [Policy-analysis harness](policy-analysis-harness.md) — step 5.4 quant+qual triage tool at ~/robocasa_experiments/policy_analysis/; env API facts; weak-region detection
- [DP model smoke-test status](dp-model-smoketest-status.md) — real DP checkpoint runs end-to-end on the 4070 box (official eval_robocasa); install steps + 5 gym/diffusers compat patches + run command; DP 50-rollout = 10% (5/50)
- [pi0 & GR00T local eval setup](pi0-groot-local-eval-setup.md) — openpi_env (pi0) server working + norm-stats patch + jax/chex gotcha; GR00T groot_env (flash-attn wheel); mimicgen pipeline + results; checkpoint paths + run commands. Eval n=50: GR00T 66% > pi0 58% > DP 10%
- [Expert data-generation loop](expert-data-generation-loop.md) — VALIDATED keystone: pi0/GR00T → record states → convert_hdf5_lerobot → trainable LeRobot data, no human demos needed
- [pi0 weak-region finding](pi0-weakregion-finding.md) — pi0 fails at GRASP (76% never-touched); object HEIGHT is a real but WEAK predictor (R²~0.08); NOT an embodiment limit; geometry too weak to target → use uncertainty/instance signals
- [Cross-policy weak-region](cross-policy-weakregion.md) — GR00T replicates pi0 almost exactly (same rate/mode/height effect, AUC 0.63) → tall-object grasp weakness is UNIVERSAL + data-addressable (shared data gap, not embodiment)
- [H100 handoff](../../../robocasa_experiments/H100_HANDOFF.md) — (in robocasa_experiments/) training setup + experiment plan + transfer/ bundle for moving to the H100
- [H100 setup complete](h100-setup-complete.md) — H100 DP train/eval stack BUILT + VERIFIED; exact /data/xinyua11 paths, conda env, the cached-loader + lang-encoder-deadlock fix (~3min/epoch), 3-seed run status, first eval (our seed0 ep400 = 2% n=50, NOT the published 10%)
- [H100 progress report](../../H100_PROGRESS.md) — reverse-handoff doc (H100 → dev box): full setup state, dataloader fix, run status, eval result + caveat, next steps
