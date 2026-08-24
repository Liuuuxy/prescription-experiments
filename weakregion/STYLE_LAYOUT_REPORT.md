# Style & Layout effect on success rate — report + catalogs (2026-07-21)

**Data:** all 28 eval runs pooled (7,112 episodes; 27 pi0 checkpoints + 1 GR00T), PickPlaceCounterToSink. Both `style_id` and `layout_id` span ids 11–60 (50 each), ~66–231 episodes/id. Grand-mean SR = 0.465.

**Figures:**
- `plots/sr_by_style.png`, `plots/sr_by_layout.png` — id-vs-SR bar plots (by id + sorted), 95% CI, noise band.
- `plots/montage_styles.png`, `plots/montage_layouts.png` — rendered screenshot of every id, sorted hardest→easiest, labeled `id | SR | n`.
- Per-id numbers: `perid_style_id_stats.csv`, `perid_layout_id_stats.csv`. Render script: `policy_analysis/render_scene.py`.

---

## TL;DR

| | STYLE | LAYOUT |
|---|---|---|
| adj-SR spread (worst→best) | **0.39** (0.27 → 0.66) | 0.22 (0.34 → 0.57) |
| noise band (perm. null 95%) | 0.38–0.55 | 0.39–0.55 |
| verdict | **real, large, checkpoint-stable** | **weak — mostly inside the noise band** |
| split-half r across checkpoints | 0.69 | 0.16 |
| true-effect SD (sampling-corrected) | ~0.084 | ~0.032 (CI touches 0) |

**Visual take:** in `sr_by_style.png` the sorted tails run well outside the gray noise band — some styles are genuinely ~20 points below the grand mean and others ~20 above, and the *same* styles are hard for every checkpoint. In `sr_by_layout.png` almost every bar sits inside or hugs the gray band: layout differences are mostly sampling noise once you control for which checkpoint was evaluated. Both factors act through the grasp phase (established separately). Practical consequence for data prescription: **stratify / pin by style, not by floorplan.**

---

## How to read the plots

- **Run-adjusted SR** = each episode's success minus its checkpoint's mean success, plus the grand mean. Necessary because the 28 runs are different checkpoints (raw mean 0.25–0.59); a naive per-id pool would confound "hard style" with "this style happened to be evaluated more under a weak checkpoint."
- **Grey noise band** = the 2.5–97.5 percentile envelope of per-id adjusted SR when id labels are permuted within run. An id whose bar+CI stays inside the band is statistically indistinguishable from chance at this sample size.
- **Colour** = diverging around the grand mean: red = below average (harder), blue = above average (easier). White ≈ average.
- Winner's-curse caution: with ~140 episodes/id the raw best-minus-worst gap expected under pure noise is ~0.19, so trust the sorted ranks + the band, not any single bar.

## Style — the ids

- **Hardest:** 49 (0.27), 25 (0.27), 42 (0.29), 43 (0.31), 22 (0.33), 21 (0.34).
- **Easiest:** 27 (0.66), 30 (0.64), 26 (0.61), 36 (0.60), 16 (0.59), 31 (0.58).
- Spread 0.39 SR — the largest of any environment factor. Style hurts both grasp and post-grasp phases → consistent with a broad visual/perceptual difficulty (materials, contrast, lighting), not a geometry effect.

**What hard vs easy styles look like** (`montage_styles.png`, hardest top-left → easiest bottom-right): the trend is suggestive, not a crisp rule. Harder styles skew toward (a) busy, high-frequency wall/backsplash textures — red and white brick (21, 43, 22), patterned tile (12) — and (b) extreme object-to-background contrast, either very dark low-contrast surfaces (25) or washed-out bright white marble everything (49). Easier styles (27, 30, 26, 36, 16, 31) tend toward cleaner, more uniform cabinet fronts and moderate contrast. That no single visual attribute cleanly separates the two ends is itself consistent with "style = broad perceptual difficulty" rather than one nameable confound — and with style carrying real signal beyond object category.

## Layout — the ids

- **Hardest:** 28 (0.34), 25 (0.36), 44 (0.36), 19 (0.38), 29 (0.40), 38 (0.40).
- **Easiest:** 18 (0.57), 42 (0.56), 53 (0.55), 20 (0.55), 49 (0.53), 33 (0.52).
- Spread 0.22, and the ranking barely replicates across checkpoints (split-half r 0.16). The marginal layout signal further attenuates to non-significance once object category and style are controlled (see FACTOR_ANALYSIS_REPORT.md §4), and layout hardness is uncorrelated with counter-to-sink distance — i.e. there is no clean geometric driver. Treat per-layout SR as mostly noise.

**What hard vs easy layouts look like** (`montage_layouts.png`): because every cell shares style 11, the montage isolates pure geometry — and there is **no visually obvious distinguisher** between the hard end (28, 25, 44, 19) and the easy end (18, 42, 53, 20). Both ends contain a mix of L-shaped, galley, island, and peninsula floorplans with varied appliance placement. This visual non-separation matches the statistics: layout hardness does not track counter-to-sink distance and barely replicates across checkpoints — i.e. most of the per-layout spread is sampling noise, not a real geometric effect.

## Caveats

- The catalog screenshots fix the *other* factor (style catalog uses layout 11; layout catalog uses style 11) and object=cup, so each montage isolates one axis. The SR numbers, however, are pooled over all layouts/styles/objects in the eval data — a per-id average, not the specific pinned combination shown.
- Effect estimates are effectively pi0-family (27 of 28 runs); the single 100-episode GR00T run is too small to confirm cross-policy per-id rankings.
- style×category (Cramér's V≈0.23) and style×layout (≈0.20) associations exist in the eval sampling; the style verdict survives controlling both, the layout verdict does not.
- Camera: free view aimed at the sink from the room side; among 4 candidate 3/4 azimuths the one with most visual content is kept (robust to per-layout counter orientation). Screenshots are for visual inspection, not the policy's actual observation (the policy sees wrist + agentview cameras).
