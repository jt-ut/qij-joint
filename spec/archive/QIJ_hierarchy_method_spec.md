# QIJ on the Voronoi hierarchy: the method as established on 9 October 2026

Status: method specification from emulations and estimator-evaluation checks on two
cases; nothing built in the package. Author's decisions are marked (A). Numbers come
from the checks in the session scratchpad `vor2_check/` (paths at the end).

Cases used: sheared mixture `cloudfil_u`, estimator `p2` (Core 2), draw 45, N = 10000,
d = 2, 7 outputs, 1094 X-VQ prototypes; multivariate t `mvt/all`, draw 0, N = 2000,
d = 10, outputs nu and P_tail, 188 prototypes; Fundamental Plane draw 0 (N = 2000,
d = 3, 4 outputs, 371 prototypes) in the offline checks only. Tolerance eps = 0.01
throughout; "1 eps" means an error of eps times V_ij.

## 0. The method in one paragraph

One X-VQ of the data gives the first-order cells. Each point's m nearest prototypes
define its order-m cell; the order-m cells of a first-order cell are nested, so the
X-VQ carries a tree without any further fit. The representation at any time is the set
of current leaves, each a row at its centroid with weight equal to its point count, so
the weights sum to N and the quantized fit on those rows converges to the full-data fit
as the tree unfolds (A). An estimator evaluation is one forward difference of the
quantized fit on one row's weight; its reading is that row's influence. Every leaf,
the first-order cells included, carries an uncertainty on its within: the Popoviciu
bound from the range of its neighbours' readings, which refinement tests by whether
the range grows. The between term is read from the rows and corrected once per round
by a q x q factor measured with q full-data evaluations (A); the within below a split
is read from the sub-cells (A) through a per-lineage rate on the leaves' geometry, and
an unsplit cell reads its bound. The fit's remaining bias shows as the change of
readings when the fit moves, measured on the rows that carry the between term. There
is one stop: the measured uncertainty of V_hat, within plus fit, is below the
tolerance's share. No region, no forced order, no pilot model, no kappa-hat, no moment
rows (A), no fixed depth (A), no learned multiplier.

## 1. Objects

- Prototypes W: the X-VQ centres (vqlp), as stored in the products.
- First-order cell j: the points whose nearest prototype is j. Order-m cell: the
  points sharing the tuple of their m nearest prototypes. A cell with n >= 2 points and
  a single child at the next order is split by two-means instead.
- Row: a current leaf, with centroid and weight n (its count). Weights sum to N.
- Quantized fit theta_Q: the estimator on the rows, continued from the previous fit
  (the full-data fit theta_hat for the first survey).
- Estimator evaluation: the fit with one row's weight perturbed by the forward step,
  continued from theta_Q; reading I_row = (theta_perturbed - theta_Q) / realized step,
  realized step = t R / N (core.differences.perturbed_weights assumes the weights sum
  to their count R). Readings are mass-centred per survey.
- Conserved reading of a refined cell: the mass-weighted mean of its leaves' readings.
- Lineage: a first-order cell with all its descendants.
- Spread of a node S: s_S = tr Sigma_S, the mean squared distance of its points from
  its centroid (geometry, no evaluation).
- Observed split of a refined node a: b_a = sum_c (n_c/n_a)(I_c - I_a)^2 over its
  children; its geometric factor G_a = s_a - sum_c (n_c/n_a) s_c.
- Lineage rate: kappa_L = sum_a b_a G_a / sum_a G_a^2 over the lineage's own refined
  nodes (least squares through the origin). Realization of the rate at node a:
  b_a / G_a.
- Region R: the smallest set of first-order cells whose contributions
  p_j (I_j - Ibar)^2 cover (1 - eps) of the between term on every output.
- Global factor evaluations: for output c, one full-data forward difference with the
  point weights perturbed in proportion to their row's reading I_row,c (scaled to unit
  maximum); its response for output c, rescaled, is the cross-moment
  X_c = sum_rows p I_c F_c between the readings and the exact cell means.

## 2. The loop

1. Full-data fit theta_hat (one fit; 7 s on draw 45, cold). First-order survey: one
   evaluation per prototype. The cells are the leaves at level 1. No region, no forced
   refinement.
2. Uncertainty of every leaf with n >= 2, at any level: u = (1/4)(max - min of the
   readings of the leaf and its neighbouring leaves)^2 per output, the neighbours being
   the leaves that are the second-nearest leaf centroid of at least one of its points
   and vice versa, recomputed each round. Single points: 0. U_within = sum p u. A
   lineage whose range of readings grew at its last split is flagged: its leaves are
   refined again until a split no longer enlarges the range (a spike goes to the
   point). The range-to-variance step is Popoviciu's inequality; taking the range from
   neighbouring means assumes bounded variation at the leaf scale, which the growth
   test checks.
3. Within read: a leaf below a split, tau = kappa_L s_leaf with the lineage rate from
   its own splits; an unsplit level-1 cell, tau = u (its bound, which the rule that
   left it unsplit keeps below its share); single points 0. V_win_hat = sum p tau.
4. Global factor: q full-data evaluations with point weights 1 + delta_f s_c, s_c the
   row readings for output c scaled to unit maximum; each returns all q responses,
   X[c,c'] = sum_rows p I_c F_c'. With G[c,c'] = sum_rows p I_c I_c', the readings are
   corrected by I_tilde = I (G^-1 X) and the between term is diag(X^T G^-1 X). The
   corrected readings feed tau, u and the re-survey.
5. Re-survey: re-measure the rows whose own contribution p (I - Ibar)^2 exceeds
   eps V_hat; if their readings moved by more than eps V_hat in total, re-measure all
   rows and recompute conserved values. U_fit = sum over the re-measured rows of
   p (I_new - I_old)^2 per output: the fit's remaining movement as the readings see it,
   including an indicator's flips.
6. Refine: (a) the leaves of flagged lineages; (b) leaves ranked by p u / (eps V_hat),
   the smallest prefix leaving the rest under the share on every output; (c) if U_fit
   still puts the total over the share, the multi-point rows ranked by mass times
   spread, in the proportion U_fit / U_total of their total, since those carry the
   coarsening that moves the fit. Refining a leaf = its next-order children (two-means
   where the tuple gives one child) plus the geometric transverse pair (a child whose
   spread orthogonal to its siblings' span is at least its spread per resolved
   direction); the new rows are measured.
7. Stop when, on every output, U_within + U_fit <= eps V_hat and no lineage is
   flagged. V_hat = B_corr + V_win_hat. eps = 0.01.

## 3. Evidence (one case each; stand-in = exact per-point influence in place of the
reading; evaluation = the estimator called)

- Second-order readings vs exact sub-cell means, draw 45, region refined as a block:
  slope 0.94 to 1.03, residual/sd 0.07 to 0.16 (`instrument_y`).
- Measured sub-cell spread vs exact between-sub-cell part: 0.83 to 1.19 (`instrument_y`).
- Staleness of untouched readings across refinement: 0.004 to 0.08 eps on draw 45
  (`instrument_y`, `regional_stop`); order one on mvt, where the base fit moves by
  0.97 to 10 relative between orders and new-only readings fail by 3.5e5 eps
  (`mvt_real`, `mvt_change`).
- Within read, stand-in: lineage rate on the two-order tree 0.96 to 1.26 of truth on
  five of seven outputs (`matrix_rate`); from a lineage with one split, 0.4 to 0.6,
  by every form tried (scalar, matrix, sibling gradient, lineage tails, ANOVA
  residual, facet pair) (`lineage_tail`, `corr_field`, `matrix_rate`, `facet_rate`).
- Depth by spread of own splits, stand-in: draw 45 2600 evaluations, -0.53 to -0.02
  eps; plane 2597, -0.22 to +0.14 eps (`spread_loop`).
- Transverse pair by the geometric rule, stand-in: plane -0.4 to -0.9 eps, filament
  unchanged; pairs everywhere hurt the filament (`transverse_pair`).
- Global factor with estimator evaluations, draw 45: corrected between term 0.97 to
  1.04 of exact at order 2 in R (uncorrected 0.87 to 1.12), 0.98 to 1.01 at order 3;
  mvt nu 0.93 at order 2 (uncorrected 18) (`extrap/global_factor_check.txt`).
- Evaluation costs on draw 45: quantized 16 to 40 ms depending on rows; full-data
  forward difference 71 to 80 ms warm-started; cold full fit 6.4 to 7.3 s.
  The old "7 s per full-data evaluation" was the cold fit.

## 4. What was tried and rejected, with the number that rejected it

- Pilot-cut tree with the GP pilot: the built loop; no longer the design.
- Neighbour-gradient within read for cells next to resolved cells: 60 to 800x over
  (`one_rule_run`); fixed by level-matched reads in the nested tree.
- Patch quadratic read: core over-read up to 10.9x; LOO margin fails on 29 to 98% of
  mass (`patch_quad`).
- Any learned multiplier on the leaf read (per round, pooled, per level, running):
  ratio swings 0.2 to 5 per round, two decades per cell (`calib_ratio`).
- Confirmation stop (realized <= predicted): never fires (`confirm_stop`).
- Cell-by-cell refinement by the within read as the way to converge the fit:
  errors of tens of eps (`real_adaptive`); fixed by the region block.
- Region-mean full-data check as a convergence certificate: blind to a scale error
  (`anchor_region`).
- Moment rows: rejected by the author (A).
- Richardson extrapolation of readings in the quantizer distortion: makes draw 45
  worse, successive changes not geometric (`extrap`).
- The quantized fit's parameter bias as a predictor of the readings' scale error:
  refuted (pa: 0.05 SE bias, 46% scale error) (`extrap/bias_check.txt`).
- Fixed two-order block: works (within 0.5 eps) but is a count (A).

## 5. Worked example: draw 45

Survey 1094 evaluations, 19 s. Region 161 cells, 18.4% of the mass, 99.0 to 99.97%
of the between term. One order in R: 592 new rows, 31 s. Between term at that point
0.87 to 1.12 of exact across outputs; after the 7 global-factor evaluations (0.5 s),
0.97 to 1.04. Within at order 2 in R read by the lineage rate; lineages with one split
carry infinite uncertainty, so step 5 refines them; with the exact-influence stand-in
the loop stops near 2600 evaluations at -0.5 to 0 eps, with 80% of R's mass at order 3
and the rest deeper. Expected wall under 90 s against 240 s for 1000 bootstrap
replicates. With estimator evaluations and the global factor, the between term is
within 3% on every output at order 2 and within 1% at order 3; the full
estimator-driven loop with the global factor has not yet been run end to end.

## 6. Known limits

- The corrected between term errs low by (1 - R^2): 0.3 to 2.8% at order 2 on draw 45,
  6% on mvt nu at order 4. No certificate inside the method for that part.
- A stop on a read fires when the read dips: at the stop the read sits at 0.4 to 0.6
  of truth on the last unrefined level, on every case (`loop_rate`).
- The region R is defined by the between term; the fit's curvature depends on cells
  outside it (mvt nu, 82% of draw 45's mass). The global factor measures that
  dependence; it does not refine it.
- Statistics that live in the within-cell dispersion (nu) need the tree unfolded to
  the points where that dispersion is; on mvt the evaluations are 1 ms, elsewhere
  this is the cost such a statistic carries.
- At N much larger than 10^4 the quantized evaluation's advantage over a full-data
  one grows beyond the factor 5 seen here, and the balance between tree depth and
  full-data evaluations shifts; untested.
- d = 10 (mvt): cells hold ~10 points, the transverse rule fires widely; the
  hierarchy reaches single points by order 3 to 4.

## 7. End-to-end results (estimator evaluations, steps 2 to 8 together; `end_to_end`)

| case | draw | prototypes | quantized evals | full-data evals | loop wall | error at the stop, eps |
|---|---|---|---|---|---|---|
| cloudfil_u p2 | 45 | stored | 2873 | 36 | 78 s | -0.98 to -0.24 |
| cloudfil_u p2 | 1 | fresh X-VQ | 3108 | 36 | 89 s | -0.86 to -0.09 |
| mvt all | 0 | stored | 4466 | 11 | 3 s | -0.07 to 0.00 |
| mvt all | 1 | fresh X-VQ | 3274 | 11 | 2 s | nu -0.82, P_tail -5.31 |
| fp all | 0 | stored | 2545 | 13 | 1 s | -0.21 to +0.10 |
| fp all | 1 | fresh X-VQ | 2531 | 13 | 1 s | -0.26 to +0.04 |
| cloudfil_u p2, FINAL method (`single_stop`) | 45 | stored | 2137 | 50 | 47 s | -0.15 to +0.55 |
| mvt all, FINAL method (`single_stop`) | 1 | fresh X-VQ | 3630 | 9 | 3 s | nu -0.01, P_tail 0.00 |

The first six rows are the superseded region method; the last two are the method of
sections 0 and 2. Intermediate runs: `bound_rule` (the bound as the rule at every
level: draw 45 1926 evaluations, -0.30 to +0.52 eps; mvt draw 1 P_tail -10.6),
`range_growth` (growth flag + matrix factor: draw 45 2122, -0.15 to +0.53; mvt draw 1
still -10.6), `one_order_all` (one order everywhere: twice the evaluations on draw 45
for the same accuracy). Full-data evaluations = one warm baseline fit plus q per round. Draw-1 cases are
scored against V_ij from the IJ product rows (convention checked on draw 0: row file
= P.var(0)/N). The re-survey fired twice on each mvt draw and never elsewhere.
Draw 45 is 78 s against about 240 s for 1000 bootstrap replicates.

mvt draw 1 P_tail, diagnosed (`range_growth/ptail_where.txt`, 2000 full-data forward
differences reproducing the product's V_ij): its influence is an indicator, 19 points
at +0.99. Under the region, bound and growth rules 17 of them were single rows reading
+0.99 and two were single rows reading the ordinary value: not in the tail under the
quantized fit, whose nu was still 8 SE from theta_hat with 1.5% of the mass coarse.
Not within, not scale, not mixing: the fit's bias. The single stop's U_fit term
carries the method to the points there (U_fit 90 eps after round 1, 0 at the stop)
and leaves draw 45 untouched (U_fit under 0.2 eps throughout; final bias at most
0.15 SE).

## 8. Not yet done

- The other four case/draws under the final method (sheared draw 1, mvt draw 0, FP
  both draws); every number for it is from draw 45 and mvt draw 1.
- More draws; N much larger than 10^4; d between 3 and 10 beyond the plane.
- Neither uncertainty term is a guarantee: the within part rests on bounded variation
  at the leaf scale, the fit part on the last movement standing for the remaining one.
  Both are the method's own evidence, with no constant.
- The corrected between term errs low by the part of the exact influence outside the
  span of the q corrected readings: 0.1 to 1.2% at the stop on draw 45.

## Paths

Session scratchpad: `/private/tmp/claude-501/-Users-jtaylor-Dropbox-Software-JT-Py-Pkgs-vqboot-fable/1938d00d-9ab6-436b-9e0e-a9967ccc53f1/scratchpad/vor2_check/<name>/report.txt`
with names as cited; saved surveys with readings: `extrap/survey_{d45,mvt}_L*.npz`.
Products read: `Research/QIJ_joint/runs/scratch/shearmix_diag_d45/...` (draw 45
prototypes), `runs/shearmix4/.../ij/s00045.psi.parquet` (exact influence),
`runs/paper_n2000/{mvt_nu,mvt_tail,fp_all}_N2000/{qij,ij}/s00000.*`.
