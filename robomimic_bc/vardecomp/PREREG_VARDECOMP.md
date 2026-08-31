# PRE-REGISTRATION: nested variance decomposition (2026-08-27, before any run)

## Question
At fixed data amount N, how much of run-to-run variation comes from WHICH demos were drawn
(draw variance) vs the training seed (seed lottery)? And does the quantity N change either?

## Design (owner-specified: plain BC only; horizon enlarged)
Learner: plain BC-MLP (1024x1024, 300 epochs, batch 100, equal-per-trajectory weighting) —
the same learner as Experiments 1-6, chosen for comparability.
Pool: the 200 proficient-human Can demos. Sizes N in {10, 40, 80, 120} (200 excluded: the
draw is then unique; 120 noted as overlap-compressed, expected pairwise overlap 60%).
Draws: 5 independent uniform draws per N (crc32-seeded; zone composition and demo lengths
recorded per draw for secondary analysis). Seeds: 3 per draw (0,1,2).
Total 4 x 5 x 3 = 60 runs.
Eval: frozen exam, probe 100 + sealed test 200, exact restore; HORIZON = 500 steps
(owner-specified increase from 400). The step of first success is recorded per scene;
success-within-400 is stored alongside success-within-500 so every number remains
comparable with Experiments 1-6. Headline metric: J on the sealed test at horizon 500.

## Primary analysis (frozen)
Nested random-effects decomposition per N and pooled: N fixed; draw random within N; seed
random within draw. Report sigma_draw and sigma_seed with their df (pooled: 16 and 40) and
the F-test of between-draw vs within-draw mean squares. C8 discipline applies: both sd
scales reported; no variance claim below 10 df; lower-tail chi-square of any surprisingly
small estimate.
PLANNING NOTE (honest power): with MLP sigma_seed ~ 9-17pp, this design resolves
sigma_draw reliably only if it is >= roughly 0.6 x sigma_seed; a smaller draw component
will be reported as an upper bound, not a null.

## Secondary (exploratory, labelled as such)
(a) Does seed sensitivity vary across draws? Global variance-heterogeneity test only —
NO per-draw volatility rankings (2 df each). (b) Correlation of draw mean / draw volatility
with draw composition (zone balance, mean demo length) across the 20 draws.
(c) Success-step distributions: does the 400->500 horizon change any conclusion?

## What would be new
Draw variance at fixed N is "does it matter WHICH demos you collect" cleanly separated
from quantity and luck — the purest form of the prescription question. Prior weak estimate:
~8.6pp, not significant. If draw variance ~ 0 while seed variance is ~10-17pp, the paper's
negative result sharpens: composition matters less than an unavoidable random integer.
