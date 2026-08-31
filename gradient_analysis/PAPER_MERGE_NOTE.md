# Reconciling §4.4 with the new gradient section (editorial note, 2026-08-14)

**The conflict.** `main (1).tex` §4.4 ("Why influence loses: a gradient-encoding ceiling",
lines 432–454) already argues the gradient-encoding principle, using the **object-category**
numbers: CIFAR influence AUC ≈0.96 (best single SVD mode 0.66) vs a RoboCasa object-category
ceiling ≈0.60 (best mode 0.56), plus the nuisance control (same CIFAR gradients at ≈0.53 for
brightness/colour), whitening (0.605→0.677), and the per-instance ceiling (≈0.61).

`gradient_analysis/PAPER_SECTION_DRAFT.tex` argues the *same principle* with the **failure-region**
numbers: best-mode 0.577 against its own shuffle null 0.598; split-half contrast 0.572 vs 0.501
shuffled; CIFAR mirror 0.649 [null 0.591] / 0.870 / 0.961 (now reproducible via
`cifar_experiments/robot_mirrors/cifar_gate.py` → `cifar_experiments/robot_mirrors/cifar_gate_report.json`).

Two number sets for one claim will read as inconsistent — or worse, as the same experiment
reported twice with different values. They are in fact **different experiments**: §4.4 asks
whether the gradient encodes *which object category* a demo contains; the new section asks
whether it encodes *which failure region a demo belongs to*, the quantity the bandit actually
selected on. Both fail; that is the point.

## Recommended structure (one narrative, three claims, no repeated numbers)

**Claim 1 — the principle (keep in §4.4, unchanged).** Influence ranking is bounded by how well
the loss gradient encodes the targeted property; the CIFAR-vs-object-category contrast plus the
brightness/colour nuisance control establishes it. §4.4 already does this well. *Change: none,
except adding one forward reference.*

**Claim 2 — the principle predicts the bandit's failure before it was run (new section, §Y).**
The property the bandit selected on — failure-region membership — is *also* not encoded, and
this time the measurement clears the bar §4.4 never had to: the unsupervised statistic falls
**below its own label-shuffled null floor** (0.577 vs 0.598), and the supervised held-out test
is at chance (0.572 vs 0.501). Present the region numbers *only here*, and say explicitly that
this is a different target property than §4.4's category test. One sentence tying them: "the
same ceiling that bounded category-targeted influence (§4.4) bounds region-targeted selection,
and here we can show it is not merely low but statistically absent."

**Claim 3 — the positive control at the experiment level (new section).** §4.4's CIFAR evidence
is *ranking*-level (AUC). The new section adds the missing rung: the whole six-arm race, same
budget, run in the CIFAR world, where arms *do* separate (+3.8pp rare vs +0.9pp random against a
null-arm floor under ±1pp) and the same gradient statistics predict outcomes (ρ 0.70–0.81 vs
|ρ|≤0.23 on the robot). This is the claim §4.4 cannot make and reviewers will want: the design
and instruments are sound; the domain is what differs.

## Concrete edits

1. §4.4 line ~454, after the takeaway sentence, add: *"The same probe, applied to the property
   our allocation experiment actually selects on—failure-region membership—predicts its outcome
   before the experiment is run (§\ref{sec:gradient})."*
2. New section §Y, opening paragraph: one clause distinguishing the two target properties
   (category vs region) so no reader thinks 0.56 and 0.577 are the same measurement.
3. Do **not** repeat in §Y: the whitening result, the per-instance ceiling, the
   magnitude-vs-direction comparison, or the value-arm common-mode explanation. All are §4.4's,
   and §Y should cite rather than restate them.
4. The abstract currently says "a gradient-encoding ceiling, verified in a controlled sandbox."
   If the new section is included, strengthen to "…verified in a controlled sandbox and shown to
   predict, in advance, the failure of budgeted data allocation" — the sandbox is no longer just
   a mechanism check, it is a positive control for the whole method.
5. One number in §4.4 to double-check against the persisted artifact: the CIFAR best-single-mode
   0.66 vs `cifar_experiments/robot_mirrors/cifar_gate_report.json`'s 0.649 (different sample: §4.4's is the full 8,000-image
   pool; the new one is the 120-vs-120 mirror of the robot gate). If both stay, label them so
   the difference is obviously the sample, not a contradiction.

## What NOT to merge

The noise-floor anatomy (`NOISE_ANATOMY.md`) and the weight-space seed result belong in the
*methodology/limitations* discussion, not in either gradient section — they are about how the
experiment is measured, not about what the gradient encodes. Keeping them separate stops the
gradient sections from becoming a grab-bag.
