# ADPIL-POOL-001 v0.1 — archived reference

The full v0.1 protocol text (owner-authored, 2026-08-30) is the document titled
"Active Data Prescription for Imitation Learning, Protocol ID ADPIL-POOL-001,
Version 0.1 — design-freeze candidate." The owner holds the canonical copy; this
file records its structure and the state it was in when Stage 0 ran, so that
Stage 0 findings are interpretable against it.

## v0.1 structure (section map)

1. Purpose — prospective question: can pre-acquisition information identify which
   demonstration conditions are worth buying? Three publishable outcomes (oracle
   fails / oracle passes but deployable selectors fail / deployable selector wins).
2. Scope and claims; exclusions (no on-demand claim, no optimality claim, no
   hardware confirmation under 12 paired histories).
3. Statistical unit = acquisition history (initial 20 demos + six 10-demo rounds).
4. Stages 0–6 with binding gates (rehearsal → Can oracle gate → tournament →
   Can confirmation → Square transfer → on-demand/SO-101).
5. Hidden-pool query interface (candidate initial scenes visible, trajectories
   sealed until purchase).
6. Six disjoint evaluation partitions incl. 300-scene confirmatory sealed exam.
7. Learner: BC-RNN, 500 epochs, best-checkpoint on probe (v2 recipe),
   equal-per-trajectory weighting, 3 scoring + 3 reporting models per round.
8. Budgets 20→80 in 10-demo rounds, full retrain each round.
9. Frozen K=8 balanced clustering on frozen visual embeddings; prevalence pi_c.
10. Five arms: broad random / geometric coverage / ensemble disagreement /
    persistent-failure x undercoverage water-filling / privileged empirical
    oracles (greedy one-step + beam-4, 10-action set, counterfactual training).
11. Batch allocation with 2-broad reserve + refusal-to-target margin
    (permutation-calibrated delta_action).
12. Harm filtering secondary only (DemInf, frozen).
13. Two-axis cost ledger (demo count + full acquisition-equivalent cost incl.
    query rollouts; GPU time excluded).
14. Primary estimand paired Delta-AULC; oracle gate (floor from historical v2
    variance, 80% power); deployable confirmation gates; tau_t = 0.9 x
    historical broad J(80).
15. Pairing/randomization/demonstrator controls.
16. Leakage and multiplicity controls.
17. Negative-result interpretations per branch.
18. Mandatory task manifest fields.
19. Hashing and append-only amendment policy.
20. Immediate authorized work: pool audit, manifest construction, information-
    barrier tests, 3 disposable rehearsals, runtime pricing, oracle-design review.
21. Existing capability audit (states/model_file replay; MimicGen not
    state-conditioned; run tree unaudited at drafting time).

## Status

Stage 0 executed 2026-08-30 under §20 items 1 and 5. See
stage0/STAGE0_REPORT.md. The Stage-2 oracle design FAILED the §10.5
affordability review and the (review-recommended) design-ceiling check;
AMENDMENT_A1_DRAFT.md is the §10.5-mandated redesign awaiting owner review.
No selector screening, confirmatory, transfer, or hardware work was performed.
