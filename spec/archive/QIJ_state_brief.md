# QIJ state brief (planner, 29 September 2026)

Purpose: bring an agent to where the work stands so it can help the
author think, without rediscovering the last two days. Every claim below
is measured unless marked as a projection; sources are named. The
documents this brief sits on are listed at the end. Terms: **cell** =
a region of a data-space quantizer (the glossary's receptive field);
**bin** = a group of points by influence level; **quantized data** =
the weighted point set an estimator is evaluated on; **rows** = that
set's points. Nothing else is coined.

## 1. The setting and the constraint

The talk is at a vector-quantization conference and presents the WSOM+
paper's method, quantized infinitesimal jackknife (QIJ): standard errors
for a black-box estimator T from finite differences of T along cells of
a quantizer of the data, instead of N per-point derivatives (the
influence function) or B refits (the bootstrap). The variance is the
codebook variance of the influence over the cells, measured; the error
is the quantizer's distortion. The demo must be the cloudfil data
(`cloudfil_G_B6_P3_v1`, N = 10 000 points in the plane, a mixture
estimator `P2Mixture`, seven outputs of one subject component: x, y,
log_reff, log_axis_ratio, pa, logit_w, log_contrast). It is the only
case that resembles a real use. The paper's method fails on it. The
rebuild of stage two is the content of the talk, and it must remain
recognizably quantization.

## 2. The paper's method and where it breaks on cloudfil

Two stages. Stage 1: k-means on the data, M_X cells (a cost rule gives
1094 here); T evaluated on the prototypes as weighted data to survey the
influence at each cell (one evaluation per cell). Stage 2: a Gaussian
process interpolates the survey into an influence map; a scalar
quantizer cuts the map into 17 level-set bins; full-data finite
differences along the bins give the between-bin variance, which is the
estimate. Note that the estimate itself is measured on the full data;
the prototypes serve only to make the map.

Measured on cloudfil (draws 0 and 1, exact influence from the stored
`ij` products):

* The influence is a spike: the top 100 of 10 000 points hold 74–94% of
  Σψ², the top 1000 over 99% (scratchpad metric_q, wavelet reports).
* No interpolation from a survey sees a spike. The map's r² against the
  true influence is 0.08–0.49; the paper's stage 2 reaches a share of
  about 0.6 of the true variance on draw 1 (coder's investigation, 28
  Sept; `QIJ_row_survey_proposal.md`).
* No internal estimate of the unmeasured remainder works: depth cascade,
  energy cascade, stationary and non-stationary Gaussian processes all
  failed, mostly by overstating the within-cell variance 4–4000× because
  it is unidentified from cell averages (scratchpad layer1, vwin_*).
* Regressing the influence on the estimator's score recovers it exactly,
  but that is the sandwich estimator and needs the score; the author
  ruled it out as making QIJ pointless (score_basis).

## 3. What replaced stage two, and what it is

A tree-structured vector quantizer of the data grown by measured
influence contrasts: the classic greedy-growth TSVQ (Gersho & Gray ch.
12; Riskin & Gray 1991) with the within-cell influence variance as its
distortion. Each node costs one evaluation of T: the contrast of mean
influence between its two children (a Haar coefficient); the larger
child is measured, the sibling derived from the parent. Growth is by
rounds (level-synchronous, one pool batch per level, bit-identical for
any worker count), opening nodes whose contrast passes a measured noise
floor, until a budget or a tolerance rule. The between-leaf variance is
the estimate. Within each remaining multi-point leaf the affine part of
the influence is measured by one contrast per principal axis (leaf-and-
axis pairs bought in rank order), and a radial quadratic contrast sizes
the non-affine remainder in the leaves with the largest affine term.
What is not bought contributes zero: the estimate is a measured lower
bound at every step, nothing modelled enters the interval. Intervals:
normal on the total, and the ABC interval (acceleration from the
reconstructed influence, curvature from two evaluations per output).

Offline, with exact coefficients, the descent reaches 0.99 of the true
variance on every output at 354 (draw 1) and 547 (draw 0) evaluations,
entering 55–84 cells and resolving 143–181 single points (wavelet_b1).
Built and run on the paper's Pareto case at N = 2000 it converges to
1.005 of the exact variance at 60 tree evaluations, about 20 size-N
units against the paper's 55–70 (qijt_validation/fp_fix/pareto).

## 4. Three defects found by the build, all in the spec, all fixed

1. The sibling derivation is invalid across a change of the evaluation
   set (a cell's prototype replaced by its points): parent and child
   measured on different rows. Fix: an opened cell's mean is re-measured
   on the new rows before anything below it is derived.
2. A second difference at the first-difference step (2√η) divides
   rounding noise by t² and returns noise. Fix: the central stencil at
   the second-difference step.
3. A forward difference from a base fit is valid only if the base is a
   fixed point of the map that produces the perturbed evaluations. The
   quantized base was one continuation short; the residual over t was a
   common offset in every coefficient, invisible to the noise estimate.
   Fix: iterate every base to a fixed point at the fit's own
   reproducibility, a cap of 5 as a failure boundary.

## 5. The measured limit of the quantized data (the open problem)

With the defects fixed, cloudfil still fails when the tree's contrasts
are evaluated on the quantized rows: the coefficients disagree with the
same contrasts on the full data at r² ≈ 0.5, and V_tot/V_ij is 1–19×
(qijt_validation/fp_fix/cloudfil). The cause is the row set, tested on
draws 0 and 1 against the full-data coefficients at the seven coarsest
nodes (scratchpad final_rows, one_point.py):

| row set (rows per cell) | keeps the fit's basin on draw 0 | coarse coefficients agree on both draws |
|---|---|---|
| centroid (1) | no: the subject's covariance collapses | no (log_axis, pa) |
| medoid (1 real point) | no | no |
| seeded real point (1) | no | no |
| matched real pair (2) | no | no |
| seeded real pair (2) | no | no |
| d+1 simplex rows matching mean and covariance (3) | yes (measured earlier: r² 0.95–0.99) | yes | 
| centroids with the subject's cells expanded to native points | draw 1: yes, r² ≥ 0.99 every output | draw 0: untested with the right cells |

Reading: the subject component's orientation and axis ratio are shape
parameters of a component whose width is comparable to the cell size.
Any row set gives its covariance 30–60 rows to be estimated from, and
that is not enough whichever rows they are. Position, size and weight
of the same component are carried at r² 0.93–0.99 by plain centroids.
So the quantized data resolve what is larger than the cells and not
what is at their scale; the paper's cases never had a local scale.

Two things do work. The d+1 simplex rows carry every cell's second
moments exactly and keep the basin, at three times the rows in two
dimensions and d+1 times in general; the author has ruled them out of
the design. Expanding the subject's cells to their native points makes
centroids elsewhere sufficient (draw 1), but which cells to expand is
not known before the influence is, and a first tree pass on centroids
points at the wrong cells when the centroid fit has left the basin
(draw 0: 193 cells opened, 3 in the subject region).

## 6. The surviving design and the decision in front of the author

Run the tree's coarse levels on the full data: the cell-level tree,
contrasts from the full-data fit, exact, tolerance-stopped. It arrives
at the cells that carry influence (62 nodes on draw 1; about 120–200 on
draw 0, a projection from the exact descent). Then, optionally, expand
those cells once and run the within-cell descent, the pairs and the
quadratic contrasts on the quantized rows plus the expanded points,
where evaluations cost about a ninth of N. The quantized second pass is
a cost saving of about two (the fine level is more than half the
evaluations on a spike and grows with N), not a correctness need.

Costs per draw on cloudfil, size-N evaluations (one evaluation on N
points; a quantized evaluation pro-rated by rows): whole tree on the
full data 180–450; coarse on full data and fine on rows 90–245;
bootstrap B = 5000: 5000; B = 2000: 2000. Wall time favours the tree by
a further factor of about seventeen on this estimator because a
contrast is an infinitesimal perturbation of a converged fit (0.25 s)
and a replicate a finite one (4.2 s); that factor is estimator-specific
and the author has ruled that the count is the method's metric.

The author's concern: a method whose coarse level runs on the full
data demotes the X-VQ, and a tree may not read as a quantizer to the
audience. The planner's position: TSVQ is a textbook quantizer; the
paper's own estimate was always measured on the full data along
quantizer cells; the X-VQ keeps two measured roles, the survey where
cells resolve the estimator's scales and the fine level of the tree.
The decision is the author's and is open.

## 7. What the talk can say, as the planner sees it

1. Perturb cells of a quantizer, not points: codebook variance is the
   variance, distortion the error.
2. Stage one on cloudfil: the quantizer carries position, size and
   weight of the subject and cannot carry its shape; cells must be
   finer than the estimator's local scales, and the method measures
   where they are not.
3. The paper's stage two needs a map; a spiked influence has no map;
   measured share 0.6.
4. The same objective grown as a TSVQ from measured contrasts: 0.99 at a
   few hundred evaluations; on the plane, single points on the spike
   and vast cells elsewhere.
5. Cost and coverage from the Stampede3 study (S = 500 draws, B = 5000,
   oracle 10 000; oracle, bootstrap and exact influence are running;
   the tree is held until its design is settled).

## 8. Documents

* Paper: `vqboot/paper/revision/qij_wsom_rev.tex`.
* Paper's method and terms: `qij/spec/QIJ_method_spec.md`,
  `qij/spec/QIJ_glossary.md`.
* The joint package's method as built, every switch, measured defects:
  `qij_joint/spec/QIJ_method_and_code_spec.md`; amendments in
  `QIJ_mods_waves.md`; derivations in `method_notes.md`.
* The estimator: `QIJ_estimator_fit_spec.md`, `QIJ_estimator_search_spec.md`
  (search reaches the true basin on 2 of 8 audited draws; no start ever
  does unaided; the estimator is defined as the best of this search).
* The tree as built and tested: `QIJ_qijt_spec.md` (still describes the
  quantized-row design of section 5, retired by the measurements);
  branch `qijt-int` of qij_joint; reports under
  `~/Dropbox/Research/QIJ_joint/qijt_validation/{v1,v3,fp_fix}` and the
  row-set test in the coder's scratchpad `final_rows/`.
* Offline measurements behind sections 2–3: the planner's scratchpad
  reports wavelet_a1/a3, wavelet_b1, layer1, vwin_gp, vwin_cascade,
  vwin_nsgp, score_basis, metric_q, gmm_diag.

Standing rules that bind any proposal: nothing in the method is
computed by Monte Carlo; every constant is relative to a measured or
declared quantity, none set from a draw; no model-derived variance
enters an interval; measure before proposing; the estimator's internals
(score, responsibilities) are off limits to the method.
