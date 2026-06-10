# Model triage (quant + qual)

Success rate with Wilson 95% CI, weakest first. Use the verdict to decide whether a model is worth deeper analysis; open the video_dir to spot-check *how* it succeeds/fails.

| model | task | success | 95% CI | n | verdict | videos |
|---|---|---|---|---|---|---|
| dp_pretrain_human300_ep500 | PickPlaceCounterToSink | 10% (5/50) | 4%–21% | 50 | weak | 4 (0✓/0✗) |
| groot_pretrain_80000 | pretrain | 50% (1/2) | 9%–91% | 2 | partial (LOW confidence, n<10) | 0 (0✓/0✗) |
| pi0_pretrain_human300_75000_n50 | PickPlaceCounterToSink | 58% (29/50) | 44%–71% | 50 | partial | 50 (29✓/21✗) |
| groot_pretrain_80000_n50 | pretrain | 66% (33/50) | 52%–78% | 50 | partial | 0 (0✓/0✗) |
| pi0_pretrain_human300_75000 | PickPlaceCounterToSink | 100% (2/2) | 34%–100% | 2 | very good — near-solved (LOW confidence, n<10) | 2 (2✓/0✗) |

Video dirs:
- **dp_pretrain_human300_ep500**: `/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/evals/dp_pretrain_human300_ep500/PickPlaceCounterToSink`
- **groot_pretrain_80000**: `/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/evals/groot_pretrain_80000/evals/pretrain/PickPlaceCounterToSink`
- **pi0_pretrain_human300_75000_n50**: `/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/evals/pi0_pretrain_human300_75000_n50/evals_1.5/pretrain/PickPlaceCounterToSink/2026-06-09-07-44`
- **groot_pretrain_80000_n50**: `/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/evals/groot_pretrain_80000_n50/evals/pretrain/PickPlaceCounterToSink`
- **pi0_pretrain_human300_75000**: `/home/asurite.ad.asu.edu/xinyua11/robocasa_experiments/evals/pi0_pretrain_human300_75000/evals_1.5/pretrain/PickPlaceCounterToSink/2026-06-09-07-41`
