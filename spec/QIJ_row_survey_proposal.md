# Row-level survey: restoring QIJ's accuracy and cost on cloudfil (proposal, 28 September 2026)

Coordinator's investigation, cloudfil p2, N = 10 000, draw 1, 14 workers, on branch `clean` (f422fe8). Nothing described here is built into the package yet. The test scripts and arrays are in `Research/QIJ_joint/runs/row_survey_test/`.

## 1. The problem, measured on draw 1

The committed code reaches V_btw/V_ij 0.79–0.92 and spends 3 070 full-data evaluations in refinement (168 s of 220 s). The stencils are exact, so the whole shortfall comes from the partition. The causes, in order:

| output | survey I_j vs true field means, r² | ψ̂₀ (GP) vs true ψ, r² | 17 initial bins' share: on ψ̂₀ as built | on raw I_j | receptive-field ceiling |
|---|---|---|---|---|---|
| x | 0.991 | 0.40 | 0.518 | 0.775 | 0.813 |
| y | 0.982 | 0.49 | 0.557 | 0.788 | 0.868 |
| log_reff | 0.975 | 0.16 | 0.296 | 0.696 | 0.713 |
| log_axis_ratio | 0.969 | 0.21 | 0.295 | 0.737 | 0.799 |
| pa | 0.968 | 0.09 | 0.212 | 0.617 | 0.640 |
| logit_w | 0.982 | 0.40 | 0.506 | 0.809 | 0.826 |
| log_contrast | 0.970 | 0.08 | 0.111 | 0.702 | 0.728 |

1. The field survey is accurate: r² 0.97–0.99 once its weights sum to N (fixed in f422fe8).
2. The GP loses most of it. ψ̂₀ at the points explains 8–49% of ψ, so the initial bins start at 0.11–0.56 and the refinement grinds toward the ceiling. The local width helps (0.47–0.91) but is not enough.
3. The ceiling is 0.64–0.87 for any quantity that is constant within a receptive field. P2's influence lives on its 104 points and is cut off sharply at P2's edge; the fields on that edge mix P2 and background points.
4. Doubling M_X (2 000) moves cost from refinement into the survey and leaves accuracy at 0.76–0.93.

## 2. The proposal

**(a) Survey the rows, not the fields.** Under `survey='moments'` every field is already represented by its own points (n_j ≤ 3) or by 3 simplex vertices that reproduce its mean and covariance. Perturb each **row**'s weight instead of each field's, one quantized evaluation per row. A field's influence is the weighted mean of its rows' influences, so the row survey contains the field survey and replaces it.

**(b) A per-point influence without the GP.** Take a native row's influence as its point's value. In a simplex field, the 3 vertex influences fix an affine function in 2-D; evaluate it at the field's points. That gives ψ̂₀ at every point, including the variation inside each field.

**(c) Expand the few fields where affine is not enough.** From (b), each field's predicted within-field share is max over outputs of Σ_{i∈j} (ψ̂₀_i − mean_j)² / Σ_i ψ̂₀_i². Take the fields with the largest share, represent them by their native points, refit θ_Q on the changed rows (one evaluation), and survey those points one by one. Each expanded field's new point values are recentred on its first-pass field mean.

**Rule for (c), tested:** expand fields in decreasing order of predicted share until the unexpanded fields' total predicted share is ≤ ε/10 for every output (ε = 0.01, QIJ's own tolerance). The affine prediction under-reads fields that are not affine, which is why the threshold is ε/10, not ε: at ε it picked 19 fields and left log_axis_ratio at 0.948.

The initial bins, the full-data stencils and the refinement stay exactly as ported, but run on this ψ̂₀. The GP still supplies σ and v_k for pricing adjacency splits.

## 3. Result on draw 1 (real estimator, real full-data stencils and refinement; ψ̂₀ injected)

| | as built (f422fe8) | (a)+(b) | (c) at ε: 19 fields | (c), 60 fields | **(c) at ε/10: 72 fields** |
|---|---|---|---|---|---|
| point ψ̂₀ vs true ψ, r² | 0.08–0.49 | 0.75–0.94 | 0.94–0.99 | 0.97–0.99 | **0.97–0.99** |
| initial 17-bin share | 0.11–0.56 | 0.83–0.95 | 0.93–0.98 | 0.96–0.98 | **0.96–0.98** |
| V_btw/V_ij after refinement | 0.785–0.918 | 0.877–0.961 | 0.948–0.988 | 0.971–0.993 | **0.972–0.995** |
| refinement evaluations (full data) | 3 070 | 174 | 170 | 165 | **173** |
| refinement wall | 168 s | 47 s | 47 s | 48 s | **48 s** |
| expansion (quantized) | — | — | 190 pts, 7 s | 603 pts, 12 s | **715 pts, 14 s** |
| QIJ total wall | 220 s | ~145 s | ~135 s | ~135 s | **~137 s** |

Per output at ε/10: x 0.991, y 0.990, log_reff 0.983, log_axis_ratio 0.972, pa 0.985, logit_w 0.995, log_contrast 0.980.

The selection in (c) uses the row survey alone. At ε/10 its 72 fields include all 30 of the truth's worst fields and cover 99.7% of the point-influence error. The error is entirely in simplex fields (native fields: r² 0.98–1.00).

Total 135 s is computed from the stage times: the measured run (106 s, with the field survey at 27.7 s) minus the field survey, plus the row survey (38 s), the expansion (12 s) and one θ_Q refit.

**Bootstrap, B = 2 000, same draw and workers: 607 s** (60 degenerate replicates). QIJ at ~137 s is 4.4× faster and uses about 415 full-data evaluations (θ̂, 238 stencils, 173 refinement, 3 for η_full) against 2 000.

**The bootstrap does not agree with V_ij on this draw:** V_boot/V_ij = x 1.39, y 1.37, log_reff 0.59, log_axis_ratio 1.56, pa 0.43, logit_w 0.67, log_contrast 0.66 (IQR-based: 0.48–1.26). The replicate medians sit off θ̂ on P2's shape outputs (log_axis_ratio −0.22, log_contrast +0.13). QIJ targets V_ij, the linearization. With P2 at 104 points, the linearization and the bootstrap differ by up to 2× in both directions. Which one matches the true sampling variance needs the Monte-Carlo truth over many draws (only 20 oracle draws are stored). "As good as the bootstrap" on cloudfil is therefore a coverage question for the study, not settled by V_ij.

## 4. Open before building

1. **The expansion rule** (ε/10) is tested on one draw only.
2. **Consistency across the two passes.** The expanded rows change θ_Q slightly, and each expanded field is recentred on its first-pass mean. That recentring is a choice, not derived.
3. **One draw only.** Draw 0, then the study draws, must confirm this. Pareto, MVT and FP use `survey='points'` and are untouched by (a)–(c).
4. **V̂_win.** The row-based within-bin estimate is near zero by construction (bins are levels of ψ̂₀); the GP's is similar. On draw 1 the remaining true within-bin share is 0.9–2.7%, so V̂_tot ≈ V_btw lies 0.5–2.5% under V_ij.
5. **The linearization against the truth on cloudfil** (section 3, bootstrap): a question for the study, independent of this proposal.
