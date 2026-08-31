# PREREG CAP-4 — Capability re-anchor of the regional-value instrument
Frozen 2026-08-26, before any run. Hashes in PREREG_CAP4_SHA256.txt.

## Why
The 32-run variance sweep (partial, n=8/8/3) shows BC-RNN scores 0.903 on balanced_D0
where BC-MLP scores 0.472, with per-seed sd 1.71pp vs 8.76pp. Every prescription result
and every noise number in this program was measured on BC-MLP. Both the effect sizes and
the noise floor of all four candidate experiments are therefore mis-priced.

## Arms (40 runs, ~2.3 GPU-h, GPU1 only, queued behind the variance sweep)
A1  region_{xlo_ylo,xhi_ylo,xlo_yhi,xhi_yhi} x BC-MLP x seeds 30-33   = 16
A2  same 4 masks              x BC-RNN x seeds 30-33                  = 16
A3  pilot_D0_40 x {BC-MLP, BC-RNN} x seeds 30-33                      =  8
Masks verified single-region (50 demos each, 100% own region). Seeds 30-33 unused anywhere.
Eval: existing frozen E_test (200 states, exactly 50/region, exact reset_to restore),
plus E_probe as a disjoint split-half check. Harness = variance/run_var.py, unmodified
except the output path.

## Primary estimand (WITHIN-RUN, run FE by construction)
c_own = J_own - mean_{r != own} J_r, in pp, for a policy trained on 50 demos of ONE region.

## Tests (exactly two; Holm at alpha=0.05 across the two)
T1  Delta = c_own(MLP) - c_own(RNN). Two-way cell means (4 regions x 2 classes x 4 seeds),
    variance pooled within cell = 24 df. Two-sided t.
T2  c_own(RNN) vs 0, same pooled variance, one-sided (>0).
Per-region breakouts and the N=40 curve anchors are DESCRIPTIVE, not tested.

## Placebo (in-sample null, no import)
For every run, the 3 non-own regions give the identical contrast centred on a region that
received zero demos. 48 placebo values per class. Under "no regional data value" the
observed c_own is exchangeable with them. Detection threshold is read off this null.

## Pre-registered read
GREEN  c_own(RNN) >= 15pp (lower 95% bound > 15) -> regional data value survives at
       competence. Next experiment = Coverage Radius under BC-RNN, re-powered on the RNN
       sigma measured HERE.
AMBER  c_own(RNN) in [5,15) and T1 significant -> regional value shrinks with capability;
       the Q17 ceiling scales with beta so allocation headroom falls below +2pp. Publish
       the structural result; run no further allocation arm in sim; go to SO-101.
KILL   upper 95% bound of c_own(RNN) < 5pp -> one region's 50 demos generalise across the
       whole box for a competent policy. The 4-region partition has no value structure.
       Stop all sim allocation work.
KILL-2 c_own(MLP) < 15pp -> Audit A's +27.7pp dose-span reading was a quantity artifact and
       the within-run instrument is not measuring what we think. Stop and re-audit.
HEADROOM (descriptive) if BC-RNN at N=40 >= 0.80, Can-PH is saturated for RNN and any
       RNN follow-up must move to N <= 20 or to Square.
