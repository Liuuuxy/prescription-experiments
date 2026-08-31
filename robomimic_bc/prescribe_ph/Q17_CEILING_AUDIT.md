# Q17 audit (2026-08-26): the design ceiling, recomputed against b_div(q) = Bq

**Finding: the advisor's suspicion is CONFIRMED.** The workflow-verdict numbers mixed two
quantities. 6.48pp was max-minus-min SPREAD over all allocations (not advantage over any
baseline). The concentrated-q figures (9.98 / 13.74 / 16.56pp) carry no shown baseline
recomputation and are consistent with a fixed near-uniform comparator; they are RETRACTED.
Deployment-proportional diverse b_div(q) = Bq already concentrates where q concentrates, and
charging the ceiling against it changes the answer qualitatively.

## Corrected C(q) = max_b q.S(n0+b) - q.S(n0+b_div(q))
Response: run+region FE, bin effects (pp vs bin 16-23): n=0 -11.98, 1-6 -9.52, 7-15 -2.35,
24-35 +5.73, 36-60 +12.31. Brute force over all 2,925 integer allocations of B=24, cap 50,
b_div(q) rebuilt (largest-remainder, feasibility-clipped) for every q. Run-clustered bootstrap
(800 reps) for uncertainty.

| q                      | D0                      | b_div(q)    | C(q)     | 90% CI        |
|------------------------|-------------------------|-------------|----------|---------------|
| measured (.249/.261/.215/.276) | balanced        | 6/6/5/7     | +0.58pp  |               |
| measured               | starved in high-q region| 6/6/5/7     | +3.35pp  | [+1.8, +6.2]  |
| q_max=0.50             | starved high-q          | 12/4/4/4    | +2.87pp  |               |
| q_max=0.625            | starved high-q          | 15/3/3/3    | +3.58pp  |               |
| q_max=0.70             | starved high-q          | 17/3/2/2    | +4.01pp  | [+1.2, +7.2]  |
| q_max=0.70             | balanced                | 17/3/2/2    | +1.15pp  | [+0.3, +2.0]  |
| q_max=0.85             | starved high-q          | 21/1/1/1    | +0.00pp  |               |

Non-monotone in concentration, peak ~4-5.5pp mid-range, collapsing to ~0 at q_max=0.85 —
exactly because b_div(q) floods the starved region by itself at high concentration.

## The analytic core
Under the confirmed near-linear diagonal response (slope beta=0.45pp/demo), the ceiling has a
closed form: C_lin(q) = beta * B * (q_max - sum q_r^2), maximized at q_max = 5/8 with value
beta*B*0.1875 = **+2.0pp at B=24 for ANY q**. Concentrating q cannot create opportunity
against q-proportional diverse under a linear response; ALL surplus above 2pp in the table
comes from response curvature (the saturation bins). Measured curvature at B=24 buys at most
~3pp more, with UCB 7.2pp.

## Consequence
The advisor's opportunity gate C(q) — abstain when UCB(C(q)) < the detection/practical
threshold — is ADOPTED as the mandatory stage-0 test of the design doc, and applied
retroactively to Can-PH at B=24 it ABSTAINS FOR EVERY q (best UCB 7.2pp, below any honest
threshold this instrument can afford). The concentrated-q redesign is therefore NOT approved.
Levers that could pass a future C(q) gate, in order of promise:
1. **Heterogeneous collection costs c_r != 1** (measured operator time): b_div(q) is then no
   longer automatically near-optimal — cost asymmetry, not weight asymmetry, is what makes
   allocation a real decision. This is native to the planned real-robot phase.
2. **Larger B relative to D0** (traverse more curvature; C_lin scales with B).
3. **A task with genuinely heterogeneous/nonlinear regional response** (the placebo-corrected
   test showed Can-PH regions are homogeneous, Wald p=0.58).
