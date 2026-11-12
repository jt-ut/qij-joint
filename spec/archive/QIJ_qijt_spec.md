# QIJT: the tree method (planner, 29 September 2026; one build)

Scope: a new method `qijt` in `qij_joint`, a separate module beside the
existing `qij`. Nothing in `qij.py`, `core/influence_model.py`,
`core/ivq.py`, `core/refine.py`, `core/rounds.py`, `core/joint.py` or
`core/xvq.py`'s survey is touched; the existing method keeps running
alongside for comparison on the same draws. What `qijt` reuses from the
package is named in section 1.3; everything else it needs is specified
here. Code standards: `qij_joint_plan.md` section 5, verbatim in the
build prompt.

Terms. **Cell** = a receptive field RF_j of the X-VQ (glossary), the
data-space quantizer's region only. **Node** = a set of cells (above the
cell level) or a set of native points inside one cell (below it).
**Leaf** = a node of the measured tree that has not itself been
measured. **Coefficient** y_c = the contrast of mean influence between
a node's two children. **Energy** E_c = the between-variance the node's
split adds. **Opening** = replacing a cell's one prototype row by its
native points in the evaluation set. **Rows** = the evaluation set, the
weighted point set T is evaluated on. **Pair** = (leaf, direction) in
the within-term stage. No other new terms.

Notation. N points; x_i the data in T's native coordinates; z_i the same
point in the quantizer's coordinates (Z from `vq_transform` as `qij.py`
applies it; identity when none); d = dim Z. q measured outputs
(`T.measured`, identity when absent, as in `qij.py`); q_full = all of
T's outputs; θ vectors are full width wherever they serve as a `start`.
Mass of a point 1/N, of cell j p_j = n_j/N, of a node the sum of its
members' masses. ψ_i = N ∂T/∂ω_i (method_notes section 1). All per-output
quantities are (q,) vectors; formulas below are written per output.

## 1. Interface, layout, reuse

### 1.1 Class and arguments

`qijt.QIJT(M_X, budget, budget_win, budget_quad, seed=0, vq_transform=None)`
with `fit(X, T, pool=None) -> QIJTResult`. All four integers are
required: M_X the prototype count; `budget` the number of tree
contrast evaluations (section 5); `budget_win` the number of within-term
pairs (section 7.2); `budget_quad` the number of quadratic contrasts
(section 7.3). No cost rule, no tolerance argument, no other options.

### 1.2 Layout

* `src/qij_joint/qijt.py` (≤ 500 lines): `QIJT`, `QIJTResult`, the
  draw's stage sequence (section 9.1), the interval methods.
* `src/qij_joint/core/tree.py` (≤ 900 lines): the hierarchy (section
  3), the weight constructions and task functions (sections 2, 4, 7),
  the round loop (section 5), the anchors (section 6), the within-term
  (section 7), the per-point reconstruction (section 8.1).
* `pipeline.run_qijt` and its row/array writers (≤ 120 lines added),
  `scripts/run.py` gains method `qijt` with `--budget`, `--budget-win`,
  `--budget-quad` (integers, required when the method is `qijt`; `--M-X`
  is required too), `--diag-draws` as for `qij`.
* No other file changes. No new dependency.

### 1.3 Reused as they are

`core.counter.Counter`; `parallel.Pool` (`share`, `map`, `submit`,
`workers`), `parallel.call_T`, `parallel.fit_status`; `qij._wrap`,
`qij._theta_hat_task`; `core.xvq.fit_xvq`, `core.xvq._field_moments`,
`core.xvq._measure_eta_Q`; `core.eta.measure_eta_full`;
`core.differences.forward_step`, `step_parameter`; `core.abc.curvature`,
`core.abc.abc_interval`; `result._normal_interval`; `products.*`.
Where this document gives a formula that one of these already
implements, the implementation calls it rather than restating it.

## 2. The evaluation set

### 2.1 Rows and base weights

The rows are R points in T's native coordinates with base weights ω0
summing to N always:

* initially one row per cell: x_r = the prototype in native coordinates
  (`inverse(xvq.centers)`), ω0_r = n_j, so R = M_used;
* after cell j is opened: its row is removed and its n_j native points
  x_i are appended, each with ω0 = 1. R grows by n_j − 1; Σω0 stays N.

Row r's mass m_r = ω0_r / N. The penalized estimators' penalty is per
unit total weight, so Σω0 = N is what makes the quantized fit comparable
to the full-data fit; the row count is never used as a weight sum.

### 2.2 The two base fits and their reproducibility

1. θ̂ = T(X, 1_N) on the full data (`_theta_hat_task` via the pool when
   given, else the counter), status by `fit_status`. NaN → failed draw.
2. η_full by `measure_eta_full(counter, X, θ̂)`: 3 full evaluations when
   T takes a start, else T's declared η with none. Every quantized
   evaluation below passes `eta=η_full` (the reproducibility a fit
   actually has, not the declared one).
3. θ̂ is brought to a **fixed point** of its own continuation (2.2a):
   θ̂ ← fixed_point(X, 1_N, θ̂). Stage `full_fit`.
4. θ_Q = fixed_point(rows, ω0, θ̂): the quantized base fit as a
   continuation of θ̂ (no multistart on the rows; the search that defines
   the estimator ran once, on the full data), iterated to a fixed point.
   NaN → failed draw.
5. η_Q by `measure_eta_Q_tree` (2.2b): 3 quantized evaluations when T
   takes a start; else η_Q = T's declared η. η_Q is measured once, on the
   initial rows, and is the noise scale of every quantized contrast.

Every later quantized evaluation is `counter(rows, ω, start=θ_Q,
eta=η_full)` with the θ_Q current for those rows (section 5.3). Every
full-data evaluation is `counter(X, ω, start=θ̂, eta=η_full)`. In a pool
task these go through `call_T` with the same arguments.

### 2.2a The fixed-point rule

A forward difference [T(ω(t), start=θ) − θ]/t measures an influence only
if θ = T(ω0, start=θ) to within the fit's reproducibility; otherwise the
residual, divided by t, enters every coefficient as a common offset that
no noise estimate sees. So every base fit (θ̂; θ_Q on the initial rows;
θ_Q after each round's openings, 5.3) is iterated to that fixed point:

    repeat: θ' = T(rows, ω0, start=θ, eta=η_full);
            r = max_o |θ'_o − θ_o| / max(|θ'_o|, |θ_o|)   (0/0 → 0, over every output of T);
            θ ← θ'
    until r ≤ η_full,  at most 5 iterations.

r ≤ η_full is the fit's own reproducibility, so the iteration asks for
nothing the fit cannot deliver; the finish is a trust-region step, so
one or two iterations are expected. Hitting the cap is a fit failure,
not a base to build on: failed draw with `status='base_unconverged'`
(full data) or `'theta_Q_unconverged'` (rows). The iteration count and
the final r are products (`n_fp_full`, `r_fp_full`, `n_fp_Q`, `r_fp_Q`;
for the re-evaluations of 5.3 the sums over rounds). An estimator without
a start has a deterministic T: no iteration, r = 0, no evaluation.

### 2.2b η_Q for the tree

`measure_eta_Q_tree(counter, rows, ω0, m, θ_Q, η_full)` in `core/tree.py`:
the three evaluations of `core.xvq._measure_eta_Q` (two at ω0 continued
from θ_Q, one at the largest-mass row perturbed by a relative 1e-6),
with the same pairwise relative differences, plus the relative
difference of each of the two base evaluations from θ_Q itself (the
fixed-point residual, ≤ η_full by 2.2a); η_Q = the largest of these
five, floored at η_full; `failed` as in `_measure_eta_Q`. The xvq
function is not modified.

### 2.3 The weight construction for a member set

For a member set K of rows with mass p_K = Σ_{r∈K} m_r and weight
parameter t:

    ω_r(t) = ω0_r · [(1 − t) + t · 1_K(r) / p_K]

Σω(t) = N for every t; the member rows' relative change is
t(1/p_K − 1) = δ, i.e. t = `step_parameter(δ, p_K)`; δ = δ_f =
`forward_step(η_f)` = 2√η_f, with η_f = η_full when T takes a start,
else T's declared η. This is method_notes section 1's constructor with
p_K taken from the masses, not from the row count (as
`core.xvq._field_step_weights` does for stacked rows).

The forward difference along K,

    U_K = [T(ω(t)) − θ_base] / t,

is the mass-centred mean influence of K (ψ̄_K − ψ̄), θ_base the base fit
on those rows. Because Σω is fixed, the grand mean of ψ never enters and
the root's U is 0 by construction.

## 3. The hierarchy

### 3.1 Bisection rule (one rule everywhere)

For a node with members at positions z (in Z) and masses m: the weighted
mean μ and weighted scatter S = Σ m (z − μ)(z − μ)ᵀ / Σm; v = the leading
eigenvector of S (ties or S = 0: see 3.4). Seeds: the member with the
smallest projection (z − μ)·v and the member with the largest (ties:
lower member index). Lloyd: assign each member to the nearer seed
(Euclidean in Z; ties to the first seed), recompute the two weighted
means, repeat until the assignment does not change, at most 100
iterations. Child A₀ = the child containing the first seed ("first
child"); the other is the second child. No randomness anywhere.

### 3.2 The cell-level tree

Built once, before any evaluation: root = all cells, members' positions
= `xvq.centers` (already in Z), masses p_j; recursive bisection to
single cells. A node of one cell is a **cell node**. Node ids are
assigned in breadth-first order of construction, root = 0; depth of the
root is 0.

### 3.3 The within-cell tree

Built when a cell is opened: the cell node itself (same id) becomes the
root, its members now the cell's native points (positions z_i, masses
1/N), and the recursive bisection continues to single points. New ids
continue from the current maximum, breadth-first per opening. A node of
one point is a **point leaf**.

### 3.4 Degenerate nodes

A node whose members all share one position (S = 0) cannot split and is
a leaf with C = 0 (section 7.1), whatever its member count. If S ≠ 0 but
the leading eigenvalue is tied, the eigenvector returned by
`numpy.linalg.eigh` for the largest index is used. A non-degenerate node
of two members splits into two leaves.

### 3.5 Children by mass

For every internal node c with children A₀, A₁: the **measured child**
A = the child of larger mass (ties: A₀), the other is B. p_c = p_A + p_B.

## 4. Measuring a node

### 4.1 Coefficient and energy

Measuring node c = one quantized evaluation of U_A along its measured
child's rows (section 2.3) with t_A = `step_parameter(δ_f, p_A)`. The
parent's own mean U_c is known (U_root = 0; otherwise from the parent's
measurement, 4.1 applied one level up), so

    U_B = (p_c · U_c − p_A · U_A) / p_B,
    y_c = U_A − U_B,
    E_c = (1/N) · (p_A · p_B / p_c) · y_c².

E_c is exactly the between-variance (1/N) Σ_leaves p U² gains when c is
split, provided U_c and U_A are measured on the same rows; the
derivation is only valid then. A cell's opening changes its rows, so an
opened cell's U is re-measured on the new rows before anything below it
is derived (5.3); every other node's U is the one its parent's
measurement gave it. The member rows of a node = the rows of its cells
(cell level: each cell's prototype row, or its native points if that cell
has been opened) or its native points (within-cell).

### 4.2 Noise and the pass rule

    s_c = √2 · η_Q · |θ_Q| / t_A · (p_c / p_B)      (per output),

the reproducibility of the two evaluations behind U_A, carried to y_c
through the derivation of U_B. Output o **passes** at node c when
|y_c,o| > z_n · s_c,o, z_n = 5, the filter tested offline (its constant is
declared here and nowhere else). An output's energy is counted only where
it passes: E_c,o ← E_c,o · 1[passes]. Node c is **open** when at least one
output passes; otherwise **closed**, its two children leaves.

A failed evaluation (exception or any NaN, `fit_status` ≠ 'ok' is
recorded but only NaN/exception counts as failure): y_c = NaN on every
output, no output passes, E_c = 0, `n_failed` += 1, the node is closed.

## 5. Growing the tree (level-synchronous rounds)

### 5.1 State

The rows, ω0, θ_Q (the current base fit on those rows), the set of
measured nodes with (U_A, U_B, y, E, s, pass flags, round), the running
V_btw,o = Σ counted E_c,o, and `evals_tree` = quantized evaluations
spent on node measurements.

### 5.2 Rounds

Round 0 measures the root (one evaluation). Round r ≥ 1:

1. **Candidates** = the children of the nodes measured in round r − 1
   that are open, excluding point leaves and degenerate nodes (3.4).
2. **Priority** of a candidate = its parent's priority
   max_o E_parent,o / V_btw,o (counted E, current V_btw; an output with
   V_btw,o = 0 contributes 0). Ties: lower node id.
3. **Selection**: the top min(|candidates|, budget − evals_tree)
   candidates by priority. If the budget is exhausted before any is
   selected, stop with `tree_status = 'budget'`; if there are no
   candidates, stop with `'exhausted'`.
4. **Openings**: every selected candidate that is a cell node is opened
   (section 5.3), all of them together, before the batch.
5. **Batch**: the selected nodes are measured in one batch (section
   9.2), each by 4.1, in selection order.
6. The curve product (section 11.4) gets one line per round.

A selected cell node's measurement is its within-cell root contrast.

### 5.3 Openings

For each cell opened: its prototype row is removed, its native points
appended (2.1), its within-cell tree built (3.3). After all of a round's
openings: the pool re-shares the rows, and θ_Q is re-evaluated to a
fixed point on the new rows, θ_Q ← fixed_point(rows, ω0, θ_Q_previous)
(2.2a), its evaluations counted in stage `opening`. NaN → failed draw
with `status='opening_failed'`; the cap → `'theta_Q_unconverged'`. Then
each opened cell X has its own U
re-measured on the new rows, one evaluation per opened cell in one
batch, member set = X's native points, t = step_parameter(δ_f, p_X),
base = the new θ_Q, counted in stage `opening` (not charged to the
budget). This U_X replaces, for everything below X, the value X carried
from its parent's measurement: the within-cell root contrast derives its
sibling from it. X's ancestors keep their y, E and U as measured. The
shift U_X^new − U_X^old per output is recorded (`open_shift_o` in the
nodes table): it is the within-cell nonlinearity the one prototype row
could not represent. A failed re-measurement keeps U_X^old and counts in
`n_failed`. Coefficients already measured on earlier rows are otherwise
kept as measured (the base drift they carry is a product, 6.3).

## 6. Anchors: scale from the full data

### 6.1 What is measured

After the tree stops, the nodes of the cell-level tree at depth ≤ 2
(root, its 2 children, their up-to-4 children; fewer if the tree is
shallower) are re-measured **on the full data**: rows = X, ω0 = 1_N,
member set = the native points of the measured child's cells,
t = `step_parameter(δ_f, p_A)`, base θ̂, giving U_A^full, then U_B^full and
y_c^full by 4.1 with U_root^full = 0. A second evaluation per node at
δ_f/2 gives y_c^half (t is linear in δ, so this is exactly t_A/2). Any
anchor node the tree itself never measured (its parent did not pass) is
measured on the current rows too, at this point, with the current θ_Q,
so every anchor has a quantized value y_c^Q; those extra quantized
evaluations count in stage `anchors` and are not charged to the budget.
Up to 7 + 7 full evaluations, stage `anchors`; the full-data batch and
the quantized batch are separate `pool.map` calls with their own
`share`.

### 6.2 The scale

Per output, over the anchor nodes with finite y^full and y^Q:

    a_o = Σ_c y_c,o^full · y_c,o^Q / Σ_c (y_c,o^Q)²,
    scatter_o = sqrt( Σ_c (y_c,o^full − a_o y_c,o^Q)² / Σ_c (y_c,o^full)² ),
    step_ratio_c,o = y_c,o^full / y_c,o^half     (per node; a product, expected 1).

Fewer than 2 finite pairs on an output → a_o = NaN and that output's
V_btw, V_win, V_tot, intervals are NaN; the draw is not failed. Every
quantity measured on the rows is scaled by a_o where it is reported
(energies by a_o²): V_btw, V_win, ψ̂. Unscaled values stay in the tables.

### 6.3 Drift

After the within-term and quadratic stages, when the rows are final, one
quantized evaluation re-measures the root (4.1) on them with the final
θ_Q; `root_drift_o = y_root,o^final / y_root,o^initial`. Stage `drift`.
It is a product only.

### 6.4 Bias ingredient

For the root's two children (depth 1) the central second difference
along each child's own member set on the full data, at the second-
difference step, by `core.differences.difference(p_K, central_step(η_f),
evaluate, θ̂)`: t_K = step_parameter((3η_f)^{1/3}, p_K), evaluate(±t) =
T(X, ω(±t)) with 2.3's construction on ω0 = 1_N, start = θ̂, eta = η_full;
D2_K = [T(+t) − 2θ̂ + T(−t)] / t². Four full evaluations, stage `bias`.
The first-difference step δ_f = 2√η_f is not usable here: a second
difference at that step divides rounding noise by t² and returns O(1)
noise for a start-less estimator (η_f = machine ε). The anchors' half-step
evaluations serve the step ratio only. b̂_o = (1/(2N)) · (p_A D2_A,o +
p_B D2_B,o), the plug-in bias of `ivq.bias_and_acceleration` on this
two-set partition.

## 7. The within-term

Runs after the anchors, on the quantized rows. Nothing here is modelled:
every term is a measured contrast, and what is not bought contributes
zero, so V_win is a measured lower bound.

### 7.1 Leaves and their geometry

Leaves: every node in the tree table whose parent is measured and that
is not itself measured (`budget` ≥ 1, so the root is always measured).
Each leaf ℓ has a known U_ℓ (its parent's measurement). Its
**members** are its rows as they stand: a leaf above the cell level
holds its cells' rows (prototype rows, or native points where the tree
opened a cell); a within-cell leaf holds native points; a single-cell
leaf that is unopened holds one row and, for the ranking below, is
described by its native points instead (`_field_moments` applied to Z
gives every cell's n_j, μ_j, C_j at once), since buying any of its
pairs opens it. Every mean over a leaf is mass-weighted: n_ℓ = its
member count, p_ℓ = Σ_members m_r, μ_ℓ = Σ m_r z_r / p_ℓ,
C_ℓ = Σ m_r (z_r − μ_ℓ)(z_r − μ_ℓ)ᵀ / p_ℓ, eigenpairs (λ_j, v_j)
descending, rank r_ℓ = #{j : λ_j > d · λ_1 · machine-ε}. Point leaves and
C = 0 leaves have r_ℓ = 0. Geometry is recomputed on the current rows
before any contrast is built (a bought single-cell leaf's rows are its
native points by then).

Sibling derivative (ranking only, never in any estimate): for leaf ℓ
with sibling s under parent c,

    ĝ_ℓ,o = |U_ℓ,o − U_s,o| / ‖μ_ℓ − μ_s‖      (0 when the norm is 0).

### 7.2 Pairs

A pair is (ℓ, j) with 1 ≤ j ≤ min(r_ℓ, n_ℓ − 1). Score:

    score(ℓ, j) = max_o p_ℓ · λ_j · ĝ_ℓ,o² / (N · V_btw,o)     (unscaled V_btw),

ties: lower leaf node id, then lower j. Buy the top `budget_win` pairs.
Every bought leaf that is an unopened cell is opened first (5.3 in
full: rows appended, one θ_Q re-evaluation for all of them, then each
opened cell's U re-measured on the new rows, stage `opening`); the
re-measured U is the leaf's U from then on (8.1).

Contrast for (ℓ, j): h_r = (z_r − μ_ℓ)·v_j / √λ_j for member rows r, 0
elsewhere (mass-weighted mean 0 and mass-weighted mean of h² = 1 over
ℓ, so Σ ω0_r h_r = 0); ω_r(t) = ω0_r (1 + t h_r), so Σω = N;
t = δ_f / max_r |h_r|. Response and contribution:

    D_ℓj = [T(ω(t)) − θ_Q] / t,        s = √2 · η_Q · |θ_Q| / t,
    W_ℓj,o = (1/N) · D_ℓj,o² / p_ℓ   where |D_ℓj,o| > z_n s_o, else 0.

(For ψ affine in the leaf, D = p_ℓ (g·v_j) √λ_j, so W is exactly
p_ℓ λ_j (g·v_j)²/N, one term of p_ℓ gᵀC_ℓg/N.) One batch for all pairs,
stage `within`. Failure as in 4.2: contribution 0, counted.

### 7.3 Quadratic contrasts

For each leaf with at least one bought pair, its measured affine term
A_ℓ,o = Σ_j W_ℓj,o. Rank leaves by max_o A_ℓ,o / V_btw,o (ties: lower
id), buy the top `budget_quad`. Contrast: m_i = (z_i − μ_ℓ)ᵀ C_ℓ⁺ (z_i − μ_ℓ)
(pseudo-inverse of rank r_ℓ), h̃_i = m_i − mean_ℓ(m) − Σ_{j ≤ r_ℓ} β_j h_ij
with β_j = mean_ℓ(m · h_j) (orthogonal to every principal direction,
bought or not), h = h̃ / sqrt(mean_ℓ(h̃²)); skip the leaf if
mean_ℓ(h̃²) = 0. Weights, step, response Q_ℓ, noise and pass rule as in
7.2; contribution (1/N) Q_ℓ,o²/p_ℓ where it passes. One batch, stage
`quadratic`.

### 7.4 The within-term

    V_win,o = a_o² · [ Σ_bought pairs W_ℓj,o + Σ_bought leaves (1/N) Q_ℓ,o²/p_ℓ ].

The directions bought in a leaf are orthonormal over the leaf, so the
sum never double-counts.

## 8. Variance, reconstruction, intervals

### 8.1 Per-row and per-point influence

    ψ̂_r,o = a_o · [ U_ℓ(r),o + Σ_{bought directions of ℓ(r)} (D_o / p_ℓ) · h_r ]

per row, with the quadratic direction included where bought (Q in place
of D), and only directions that passed (7.2's rule) entering, the same
gate as W and the quadratic term. A native point's ψ̂ is its own row's;
a point in an unopened cell takes its cell's row's value. ψ̂ is
mass-centred because every U is.

### 8.2 Variance and acceleration

    V_btw,o = a_o² · (1/N) Σ_leaves p_ℓ U_ℓ,o²,     V_tot,o = V_btw,o + V_win,o,

the between-leaf variance from the leaf means (7.1's leaves, every leaf
in ascending id order). Within any subtree measured on one row set this
equals the sum of its counted E_c; across an opening it does not, which
is why the leaf sum, not Σ E_c, is the estimate. The `curve` product
reports the same leaf sum for the leaves as they stand after each
round. E_c serves the priority (5.2) and the tables only.
    a_o = (1/N) Σ_i ψ̂_i,o³ / ( 6 √N · ((1/N) Σ_i ψ̂_i,o²)^{3/2} ),

`ivq.bias_and_acceleration`'s acceleration at point masses 1/N; NaN
where the denominator is 0.

### 8.3 Curvature

`core.abc.curvature(counter, sv, θ_Q, I_rows, m, N, pool, outputs=measured)`
on the final rows, with `sv` a `SurveyRows(rows=final rows, row_field=
arange(R), omega0=final ω0, eta_Q=η_Q (2.2 step 4: the declared η when
T lacks a start, never NaN),
step_ratio=full((5, q_full), NaN), quantized_start='full-data',
eta_rows=η_full)` and I_rows (R, q_full) = 8.1's ψ̂ per row (full width;
unmeasured outputs 0). 2q quantized evaluations, stage `curvature`,
giving c_q.

### 8.4 Intervals

`QIJTResult.variance` = V_tot. `interval(level)` =
`_normal_interval(θ̂, V_tot, level)`; `interval_btw(level)` = the same on
V_btw (for the comparison); `abc_interval(level)` =
`core.abc.abc_interval(θ̂, √V_tot, a, b̂, c_q, level)`. All three are
reported by the pipeline; which one the method quotes is decided by
measured coverage, outside this document.

## 9. The draw and its parallelism

### 9.1 Stage order

full_fit → eta_full → xvq (`fit_xvq(Z, M_X, seed, workers)`, no
evaluation; Z and `inverse` from `vq_transform` exactly as `qij.py`
derives them, including the 1-D promotion) → base_Q → eta_Q → tree
(rounds, with `opening` evaluations counted separately) → anchors
(6.1–6.2) → bias (6.4) → within (7.2, with its openings counted in
`opening`) → quadratic (7.3) → drift (6.3) → curvature (8.3) →
assembly. `evals_by_stage` and `rows_by_stage` keyed by these names;
`wall_time_by_stage` the same except that `bias` runs inside the anchor
batch and `quadratic` inside the within batch, so their wall is inside
`anchors` and `within` respectively; `busy_time_total` = the sum of task
wall times. The parent calls `counter.add(1, len(shared array), failed)`
per task.
`budget` ≥ 1; `budget_win`, `budget_quad` ≥ 0.

`QIJTResult` carries every scalar of 11.1 as an attribute (per-output
ones as (q,) arrays), the `nodes`, `leaves`, `curve`, `anchors` tables
as DataFrames, `psi_hat` (N, q) and `leaf_of_point` (N,), and the three
interval methods of 8.4.

### 9.2 Contract

Every batch is one `pool.map` over tasks of the shape of
`core.abc._curvature_task`: `task = (key, ω, start, eta)`, the worker
evaluates `call_T(T, rows_or_X, ω, start, eta)`, returns
(key, value, failed, wall); results are consumed by position. Before a
batch on the rows the parent calls `pool.share(rows)` if the rows
changed since the last share, and `pool.share(X)` before the anchor and
bias batches. With `pool=None` the same task functions run through the
counter in the same order. The parent applies `counter.add` per task.

Bit identity across worker counts is required: every decision (pass,
open, selection, ranking) is a function of returned values only; every
sum over nodes, leaves or points runs in ascending id order; no
floating-point reduction depends on batch composition. Validation V3
checks it.

## 10. Failures

Failed draw (products written with NaN variances and the status): θ̂ NaN
(`'theta_hat_failed'`), θ_Q NaN (`'theta_Q_failed'`), a re-evaluated θ_Q
NaN (`'opening_failed'`), a fixed-point iteration reaching its cap
(`'base_unconverged'`, `'theta_Q_unconverged'`, 2.2a). Not failed: any single contrast, anchor,
within, quadratic or curvature evaluation failing (each handled where
it is defined, counted in `n_failed`), an output with a_o = NaN, the
η measurements' `failed` flags (recorded as `eta_full_failed`,
`eta_Q_failed`). No retries anywhere.

## 11. Products

Method directory `qijt`. Written through `products.write_draw`; the
scalar row is the done marker.

### 11.1 Scalar row

`s, seed, N, M_X, R_final, budget, budget_win, budget_quad, workers,
status, theta_hat_status, theta_Q_status, eta_full, eta_Q,
eta_full_failed, eta_Q_failed, n_fp_full, r_fp_full, n_fp_Q, r_fp_Q,
n_fp_opening, r_fp_opening_max, n_rounds, n_measured, n_opened,
n_leaves, max_depth, n_failed, tree_status, n_pairs_bought,
n_quad_bought, busy_time, wall_time`, per stage `evals_<stage>`,
`rows_<stage>`, `wall_<stage>`, and per output o: `theta_hat_o, V_btw_o,
V_win_o, V_tot_o, a_scale_o, a_scatter_o, root_drift_o, accel_o, b_hat_o,
c_q_o, lo_o, hi_o, lo_btw_o, hi_btw_o, lo_abc_o, hi_abc_o` at level 0.95.

### 11.2 `nodes` (every draw)

One row per node of the measured tree and its leaves: `node, parent,
depth, kind ('cells'|'points'), cell (−1 above the cell level), n_cells,
n_points, p, node_measured (bool), round (−1 for an anchor node measured
after the tree stopped), measured_child, t, evals_rows (R of the node's
round), status` (`fit_status` of the measuring evaluation), and per
output `U_o, y_o, s_o, E_o, pass_o, open_shift_o` (y, s, E unscaled; NaN
where not measured; U known for every node in the table; `open_shift`
NaN except for opened cells, 5.3). The table holds the measured nodes
and their children only, not the unmeasured remainder of the hierarchy.

### 11.2a `pairs` (every draw)

One row per bought within-term contrast: `leaf, j` (the principal-axis
index, 0 for the quadratic contrast), `t`, per output `D_o, s_o, pass_o`
(D unscaled; Q in the D column for j = 0).

### 11.3 `leaves` (every draw)

One row per leaf: `leaf (node id), n_points, p, rank, n_pairs_bought,
quad_bought`, per output `U_o, ghat_o, A_o (affine term, scaled),
Q2_o (quadratic term, scaled)`.

### 11.4 `curve` (every draw)

One row per round: `round, evals_tree, R, n_open`, per output `V_btw_o`
(scaled a posteriori by a_o², so the curve reads in full-data units).

### 11.5 `anchors` (every draw)

One row per anchor node: `node, depth, p_A`, per output `y_full_o,
y_half_o, y_Q_o, step_ratio_o`.

### 11.6 `points` (only `--diag-draws`)

`i, leaf`, per output `psi_hat_o`.

## 12. Validation (the build's acceptance; a report, no pass/fail
    beyond V3)

V1 **Paper cases.** For each of (pareto, shape), (pareto, tail),
(mvt, nu), (mvt, tail), (fp, all), (imf, all) at N = 2000, draws 0–9,
M_X = `cost_rule_M(N, q, 0.01)` (the paper's own prototype count),
budget 400, budget_win 100, budget_quad 50, workers 8: from the `curve`
product, the number of tree evaluations at which V_btw,o first reaches
0.99 of the `ij` product's V_ij,o (per output, per draw; "never" if not
reached; the `ij` products for these draws are produced by
`run.py <case> ij` if absent); V_tot,o / V_ij,o at the end; the three
intervals' coverage of the registry truth over the 10 draws;
evaluations and rows by stage, in size-N units (rows / N summed). Alongside:
the existing method's cost on the same draws from `run.py <case> qij`
at its defaults (run if absent), in the same units. The report states
the numbers; it argues nothing.

V2 **cloudfil, draws 0 and 1**, M_X 1094, budget 600, budget_win 200,
budget_quad 100, workers 14: (a) the curve against the stored `ij`
V_ij,o; (b) V_tot,o / V_ij,o; (c) on every node the tree measured at
depth ≤ 5, the same contrast on the full data (one evaluation each,
outside the method, in the validation script): per output the slope
and r² of y^full against a_o y^Q, and the same for the within-term
pairs bought in the subject component's cells (the `pairs` table's
signed D_o against the same contrast on the full data) — this is where
the position-angle output is judged; also the distribution of
`open_shift_o` over opened cells relative to √V_ij,o; (d) root_drift,
a_scatter, step_ratio;
(e) busy/wall at workers 14 and at workers 1 for draw 0.

V3 **Bit identity.** (pareto, tail) draw 0 and cloudfil draw 0 at
workers 1 and 8: every product file identical (byte-for-byte after
dropping the timing columns). This is the one pass/fail item.

V4 **Smoke.** The existing all-methods cloudfil smoke runs unchanged;
`qijt` is added to it at budget 50, budget_win 10, budget_quad 5.

Reports go to `Research/QIJ_joint/qijt_validation/` with the scripts
that made them.
