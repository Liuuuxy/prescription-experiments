# Reconciling the professor's n=30 report with our weak-region data

Source report: `robocasa_pickplace_full_report.pdf` (n=30: 10 seeds × 3 trials).
Our data: pi0 n=150, GR00T n=100 weak-region runs (`weakregion/*/weakregion.json`).
Reproduce: `python policy_analysis/reconcile_prof_report.py`.

## (b) GR00T grasp-vs-transport split — report claim PARTLY confirmed

The report (§3.3) claims **GR00T = transport stall** (grasps but stalls mid-transport),
vs **pi0 = grasp failure**. Our larger-n phase tagging:

| policy | success | no-grasp | transport/place-stall | lifted >2cm |
|--------|--------:|---------:|----------------------:|------------:|
| pi0 (n=150)  | 53% | 86% | 14% | 18% |
| GR00T (n=100)| 56% | 80% | 20% | 39% |

- **Direction right:** GR00T does stall in transport ~2× more than pi0 (20% vs 14% by
  phase tag; 39% vs 18% by "object lifted >2cm").
- **Magnitude wrong:** it is GR00T's **secondary** mode, not its primary. Both policies
  are **grasp-dominated (~80% no-grasp).** The report generalized a secondary mode from a
  small contact-sheet sample (e.g. the sponge case it highlights).

## (c) Merge: per-object table vs our rates + height

| object | prof Pi0 | prof GR00T | prof comb | our comb | height(cm) |
|--------|--------:|----------:|----------:|---------:|-----------:|
| carrot | 1.00 | 0.67 | 0.83 | 83% | 3.8 |
| bottled drink | 1.00 | 0.67 | 0.83 | 50% | 16.3 |
| sponge | 1.00 | 0.00 | 0.50 | 67% | 4.0 |
| glass cup | 0.33 | 0.33 | 0.33 | 75% | 11.9 |
| cream cheese | 0.33 | 0.33 | 0.33 | 0% | 1.7 |
| salt/pepper | 0.33 | 0.33 | 0.33 | 33% | 10.2 |
| avocado | 0.00 | 0.67 | 0.33 | 100% | 5.4 |
| milk | 0.00 | 0.33 | 0.17 | 50% | 12.1 |
| shrimp | 0.00 | 0.33 | 0.17 | 0% | 2.1 |
| beer | 0.00 | 0.33 | 0.17 | 50% | 19.1 |

- **Prof-combined vs our-combined: r = +0.36** — the two evals only *weakly* agree at the
  single-instance level (e.g. avocado prof 0.33 / ours 100%; cream cheese prof 0.33 / ours
  0%). Object category is a **noisy** label on both sides (different instances + seeds).
- **Prof-combined vs height (his 10 objects): r = -0.04** — height **washes out** on a small
  object subset.
- **Height vs success across ALL 51 of our categories (n≥3): r = -0.40 (R²≈0.16)** — this is
  the robust aggregate signal (taller → harder), consistent with our AUC≈0.63 finding.

**Takeaway:** hand-picked object-identity / geometry is too weak and too sample-hungry to be
a per-episode targeting signal — height only emerges across many categories and vanishes on
small subsets (which is exactly why his n=30 eval couldn't see it, and our early n=50
median-split gave a false null). This **reinforces the plan to target on uncertainty /
disagreement, not geometry** (the DP action-variance probe tests that signal directly).
