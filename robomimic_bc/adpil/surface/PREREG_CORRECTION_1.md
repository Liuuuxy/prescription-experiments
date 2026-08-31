# PREREG correction 1 (append-only) — 2026-08-31

Written after owner review of Amendment A1 and BEFORE any run result of this block
was read by the analyst (the pool results files had not been opened; only run
counts were monitored). Original PREREG_SURFACE.md preserved verbatim.

## What changes

Owner finding (A1 review, item 4): the 16 cells are single dataset constructions;
3 training seeds per cell are nested within ONE draw and do not create independent
causal interventions. The slope contrasts (registered quantities 1–3) therefore
carry unquantified draw noise (n_draw = 1 per cell).

Status changes:

- Quantities 1–3 (beta_own/beta_cross balanced, starved slopes, concavity ratio):
  DOWNGRADED from design inputs to PILOT estimates. They may inform the design of
  the multi-draw surface-v2 block (cell dosing, run pricing) but may NOT feed the
  analytic oracle, the recomputed floor, or any ceiling verdict.
- Quantity 4 (sigma_seed, pooled within-cell, 32 df): UNCHANGED — seeds within a
  fixed dataset is the correct design for this parameter.
- Quantity 5 (sigma_draw from vd_N80_d0..d4, 4 df): UNCHANGED including its
  registered caveat — the five vd draws are independent constructions.

## Why

The binding response surface requires paired add-vs-base contrasts replicated over
independent dataset draws with seeds nested inside draws (surface-v2 design, to be
approved before launch). Running analyze_surface.py on this block remains
authorized, with outputs 1–3 reported under a PILOT label.

## Visibility statement

No J values, slopes, or per-run outcomes from this block were visible when this
correction was written. Pool monitoring exposed only run start/completion events.
