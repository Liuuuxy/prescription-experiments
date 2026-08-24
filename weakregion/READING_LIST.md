# Reading List: Capability-Aware Data Prescription (compiled 2026-07-21)

**Provenance:** deep-research sweep (23 sources fetched, 113 claims extracted, 25 adversarially verified 3-vote, 2 refuted) + two targeted follow-up searches for the empty buckets (failure prediction/eval; bandit/budget allocation). Full verified record: `factor_analysis/wf4_litreview.json`. Every paper below was verified to exist on arXiv.

**Novelty verdict (verified across all sources):** no 2024–mid-2026 paper prescribes NEW demonstration data from a *measured attainability + net-teachability* model. The frontier is (a) failure-anchored generation without learnability modeling (IntervenGen, RaC), (b) predicted-improvement selection from fixed pools (CUPID, DataMIL, ATHENA), (c) a-priori factor/diversity rules (FSC scaling curves, Lin power laws, AgiBot, Gao compositional). Nearest neighbors to the probe-then-allocate criterion: **FSC** (passive scaling-curve extrapolation per factor — no attainability gate, no causal placebo, no mechanism-matched content) and **AMF** (info-gain task selection — no generation, no retention constraint). Consistent with the existing YELLOW novelty verdict: lead with mechanism + diagnostic negatives, benchmark head-to-head vs FSC.

---

## Tier 1 — read this week (each maps to a live design decision)

1. **FSC — Guiding Data Collection via Factored Scaling Curves** (Zha, Badithela, …, Majumdar; arXiv 2505.07728, CoRL 2025). Per-factor success-vs-data scaling curves from pilot data → allocate collection budget to highest-influence factors; +26% real-robot. THE competitor. Confirmed by direct fetch: no failure-driven generation, no attainability gate, no mechanism-matched demo types, no placebo control — exactly the differentiation surface of PRESCRIPTION_CRITERION.md. Mandatory baseline.
2. **CUPID — Curating Data your Robot Loves with Influence Functions** (Agia, Sinha, …, Bohg; arXiv 2506.19121, CoRL 2025). Influence on CLOSED-LOOP return (not training loss) from eval rollouts; SOTA from <33% of data; also subselects newly collected trajectories. The closed-loop signal our RC-LESS/TracIn arms lacked (gradient-encoding principle explains why ours failed). Baseline for any selection comparison; its "new-trajectory subselection" mode is the closest thing to prescription in the selection literature.
3. **RETAIN** (Yadav, Zhou, Wagenmaker, Pertsch, Levine; arXiv 2512.08333, ICLR 2026). Naive narrow fine-tuning of a generalist VLA loses on BOTH axes (forgets generalist skills AND fails to generalize within the new task); weight interpolation between fine-tuned and pretrained beats both endpoints OOD. Directly upgrades our retention machinery: add interpolation to the coverage-floor + diversity constraints.
4. **RaC — Recovery and Correction data** (arXiv 2509.07953, Sept 2025). The IL plateau is a DATA-DISTRIBUTION problem: clean expert demos never teach retry/recovery; recovery+correction segments give ~10× data-efficiency (5 h vs 89 h on shirt-hanging). External validation of "WHAT data is the binding constraint" + concrete demo-content prescription for weak cells (matches our tall = touch-then-fumble → grasp-securing/recovery demos).
5. **IntervenGen** (arXiv 2405.01472, IROS 2024). MimicGen-lineage: amplifies ~10 human corrective interventions into large corrective datasets anchored at policy-mistake states; up to 39× robustness gain. The direct precursor + baseline for our MimicGen-based targeted generation, and the concrete mechanism for "1–5 human source demos × MimicGen" escalation rung.
6. **Lin et al. — Data Scaling Laws in IL** (arXiv 2410.18647, ICLR 2025 Oral). Generalization is a power law in # distinct environments/objects; per-env/object demo count saturates ~50 demos. The amount-axis rule ("buy new diversity, not more repeats") — consistent with our measured dose knee (~20–60) and the 610-demo saturation harm.
7. **AMF — Active Fine-Tuning of Multi-Task Policies** (Bagatella, Hübotter, Martius, Krause; arXiv 2410.05026, ICML 2025). Info-gain selection of which task to demo next under budget, with guarantees. The principled Bayesian alternative to our race/elimination allocator — cite, differentiate (no generation, no attainability, no retention), possibly borrow the info-gain acquisition.
8. **Proxy-run reliability** (Wang et al., arXiv 2512.24503, ICLR 2026). Small proxy runs mis-rank data recipes because optimal hyperparameters are data-dependent; smaller LRs restore rank correlation >0.92. Directly actionable for the headroom probe: pick the probe LoRA recipe so probe rankings transfer — this is a known failure mode of exactly our planned probe design.

## Tier 2 — by bucket

**Failure-driven / active collection (bucket 1)**
- AdaDemo (arXiv 2404.07428): iterative task-level collect-where-failing — the "P(fail) prescription" baseline at task granularity; skim for positioning.
- Screw-geometry bandits (arXiv 2410.18275): literal PAC-bandit over task-space subregions for next kinesthetic demo; niche precedent for region-level bandit prescription; skim.
- (From memory, still relevant: Predictive Red Teaming / RoboART 2502.06575; π*0.6 / RECAP 2511.14759; Mirchandani "So You Think You Can Scale Up" 2411.01813.)

**Data curation/selection for VLAs (bucket 2)**
- DataMIL (arXiv 2505.09603): datamodels for robot data selection, 60+ tasks, rejects similarity heuristics; high-priority read.
- ATHENA (arXiv 2606.16208, June 2026): influence functions scaled to pi0-class VLAs (~313× speedup); half the fine-tuning data is redundant/harmful; skim — fixed-pool only, no novelty threat; margins thin, inverts at small task counts.
- Re-Mix (arXiv 2408.14037): domain reweighting for robot co-training; adjacent anchor.

**Scaling laws & diversity (bucket 3)**
- AgiBot "Is Diversity All You Need?" (arXiv 2507.06219, IROS 2025/TRO 2026): diversity decomposed by axis — TASK diversity beats quantity, but EXPERT/demonstrator diversity actively HURTS (velocity multimodality); debiasing fix ≈ +15%. Must-read: "diversity" is not monolithic; directly informs what to vary inside targeted generation.
- Gao et al. compositional collection (arXiv 2403.05110, RSS 2024): policies compose separately-seen factors (77.5% vs 2.5% on unseen combos at equal effort). Project-critical caution AND support: additive/compositional structure licenses axis-wise prescription — but their result is about visual factors; our additivity finding is the outcome-level analogue.

**Weakness discovery / failure prediction / eval (bucket 5)**
- FIPER (arXiv 2510.09459, NeurIPS 2025): runtime failure prediction for generative policies, no failure data needed (RND-OOD + action-chunk entropy, conformal calibration); must-read — candidate P(fail) signal.
- SAFE (arXiv 2506.09937, NeurIPS 2025): failure detection from VLA internal features, generalizes across tasks incl. pi0-style; must-read — alternative feature space for weak-region modeling beyond scene factors.
- RoboMD (arXiv 2412.02818): RL agent searches embedding space of environment variations to find failure regions, then fine-tunes on them; must-read — nearest neighbor to our weak-region discovery→prescription loop (search vs regression).
- Q-DIG (arXiv 2603.12510, Mar 2026): quality-diversity red-teaming of VLAs over instructions + fine-tune on failures; must-read as 2026 instantiation of the same closed loop (instruction space, not scene space).
- ERT — Embodied Red Teaming (arXiv 2411.18676): canonical red-teaming anchor (language axis); cite/skim.
- Hide-and-Seek trajectories (arXiv 2605.30834, May 2026): contrastive discovery of failure-signal trajectory segments; skim.
- VLAConf (arXiv 2605.29605, May 2026): calibrated task-success confidence for VLAs, single forward pass; must-read for the P(fail) line.
- Vincent et al. statistical BC evaluation (arXiv 2405.05439, RA-L 2024): tight LCBs on policy performance from minimal rollouts; THE citation for "how many trials" + defensible arm comparisons; must-read for the eval design.
- SureSim (arXiv 2510.04354, Princeton): prediction-powered inference combining big sim eval + small real eval; skim now, must-read when the G1 sim2real eval starts.
- Active experiment selection for multi-task eval (arXiv 2502.09829): info-gain allocation of eval episodes across tasks; must-read — same machinery can allocate probe episodes.

**Retention / forgetting (bucket 6)**
- RETAIN (Tier 1 above).
- VLA continual-learning ER study (arXiv 2603.03818, Mar 2026): ~2% experience-replay buffer keeps pi0/GR00T backward transfer at ~0.1–0.2 (vs 0.4–0.5 for from-scratch baselines). Must-read companion; NOTE: the stronger claim "pretraining alone makes forgetting mild" was REFUTED in verification (1-2 vote) — replay is still needed.

**Budget allocation / probe methodology (bucket 7)**
- RegMix (arXiv 2407.01492, ICLR 2025): small proxy runs on random mixtures → regression → pick full-scale mixture at ~10% compute. The canonical probe→predict→allocate pipeline; transfers almost directly to per-region probe fine-tunes. Must-read.
- Data Mixing Laws (arXiv 2403.16952, ICLR 2025): parametric law-fitting alternative to RegMix; skim alongside.
- DBCARE — cost-aware best-arm identification (arXiv 2505.20583, NeurIPS 2025): prices sampling cost vs suboptimality jointly; skim — formalizes the probe→commit stopping rule.
- ADO (arXiv 2410.11820): online scaling-law-driven data allocation during training; adjacent.

## Do NOT cite (refuted in verification)
- "Pretrained VLAs suffer far less forgetting without replay" — refuted 1-2 (replay/interpolation still required).
- "Camera poses / spatial arrangements are THE crucial diversity dimensions" (arXiv 2506.13536's strong reading) — refuted 0-3.

## Open questions the list leaves live
1. Does FSC's curve extrapolation already constitute a "teachability" measurement? (Read §III of FSC with PRESCRIPTION_CRITERION.md open — the differentiators are: probes are causal fine-tunes not curve extrapolations; attainability gate; mechanism-matched content; mismatched placebo; retention-structural.)
2. Do Lin-style diversity thresholds hold for targeted fine-tuning of pretrained VLAs? (Exactly what the headroom probe measures — a publishable sub-result.)
3. No paper yet combines probe fine-tunes with bandit allocation of robot collection budget — the probe-then-allocate slot is still open (AMF and FSC flank it).
