# QIJDT: the data tree (planner, 29 September 2026; one build)

Scope: a new method `qijdt` in `qij_joint`, a separate module beside
`qij` and `qijt`. It is the tree of `QIJ_qijt_spec.md` with every
evaluation on the full data: no X-VQ, no quantized rows, no openings, no
anchors, no scale. Sections that are unchanged from the qijt spec are
restated here in full so this document is codeable on its own; where a
formula is already implemented in the package, the implementation is
named and called, not restated in code. Code standards:
`qij_joint_plan.md` section 5, verbatim in the build prompt.

Terms. **Node** = a set of data points; the root is all N. **Leaf** = a
node of the measured tree that has not itself been measured.
**Coefficient** y_c = the contrast of mean influence between a node's
two children. **Energy** E_c = the between-variance the node's split
adds. **Pair** = (leaf, principal axis) in the within-term stage. No
other new terms.

Notation. N points; x_i the data in T's native coordinates; z_i the same
point in the quantizer's coordinates (Z from `vq_transform` exactly as
`qij.py` derives it, with its 1-D promotion; identity when none);
d = dim Z. q measured outputs (`T.measured`, identity when absent, as in
`qij.py`); q_full = all of T's outputs; θ vectors are full width
wherever they serve as a `start`. Mass of a point 1/N, of a node
p = (its point count)/N. ψ_i = N ∂T/∂ω_i (method_notes section 1).
Per-output quantities are (q,) vectors; formulas are written per output.

## 1. Interface, layout, reuse

### 1.1 Class and arguments

`qijdt.QIJDT(eps=0.01, vq_transform=None)` with
`fit(X, T, pool=None) -> QIJDTResult`.

* `eps`: the tolerance, the relative error accepted on the variance;
  it sets the threshold of every purchase (5.2, 7.2, 7.3) and is the
  only thing that stops the tree. The tree is finite (at most N − 1
  splits), so no cap is needed and none exists; the evaluation count is
  a result of the run, reported per stage.

No seed: nothing in the method is random (3.1). The CLI's `--seed`
seeds the draw only, as for every method, and is stored in the scalar
row as the draw's seed. No budget arguments, no cap, no other options.

### 1.2 Layout

* `src/qij_joint/qijdt.py` (≤ 400 lines): `QIJDT`, `QIJDTResult`, the
  stage sequence (9.1), the interval methods (8.4).
* `src/qij_joint/core/tree.py`: the shared primitives, imported, not
  copied: the bisection (3.1), the node measurement (4.1–4.2), the
  round loop (5), the pair and quadratic contrasts (7.2–7.3), the
  reconstruction (8.1). Where the existing function takes a rows
  argument, `qijdt` passes X and unit weights; where it takes a base
  θ_Q, `qijdt` passes θ̂. Rows-only code (openings, anchors, the rows'
  eta, the scale) is not called and not modified. If a primitive cannot
  serve both callers without a flag, the coordinator splits it into two
  functions rather than adding a flag (R2).
* `pipeline.run_qijdt` and its row/array writers (≤ 120 lines added);
  `scripts/run.py` gains method `qijdt` with `--eps` (float, default
  0.01) and `--diag-draws` as for `qij`.
* No other file changes. No new dependency.

### 1.3 Reused as they are

`core.counter.Counter`; `parallel.Pool` (`share`, `map`, `submit`,
`workers`), `parallel.call_T`, `parallel.fit_status`; `qij._wrap`,
`qij._theta_hat_task`; `core.eta.measure_eta_full`;
`core.differences.forward_step`, `central_step`, `step_parameter`,
`difference`; `core.abc.curvature`, `core.abc.abc_interval`;
`result._normal_interval`; `products.*`.

## 2. The base fit

1. θ̂ = T(X, 1_N), as one batch task `('theta_hat', ones(N), None,
   None)` through the same task function every other batch uses (9.2),
   so its `fit_status` comes back from the task the same way at any
   worker count (the pool's `_theta_hat_task` returns no status and
   breaks bit identity of the status column). NaN → failed draw
   (`'theta_hat_failed'`).
2. η_full by `measure_eta_full(counter, X, θ̂)`: 3 evaluations when T
   takes a start, else T's declared η with none.
3. θ̂ ← fixed_point(θ̂) (2.1). Stage `full_fit` holds 1 and 3; stage
   `eta_full` holds 2.

Every later evaluation is `counter(X, ω, start=θ̂, eta=η_full)`; in a
pool task, `call_T(T, X, ω, θ̂, η_full)`. `call_T` drops `start` and
`eta` for an estimator without `takes_start`.

### 2.1 The fixed-point rule

A forward difference [T(ω(t), start=θ) − θ]/t measures an influence only
if θ = T(1_N, start=θ) to within the fit's reproducibility; otherwise
the residual over t is a common offset in every coefficient that no
noise estimate sees. So:

    repeat: θ' = T(X, 1_N, start=θ, eta=η_full);
            r = max over every output of T of |θ'_o − θ_o| / max(|θ'_o|, |θ_o|)   (0/0 → 0);
            θ ← θ'
    until r ≤ η_full, at most 5 iterations.

Reaching the cap is a fit failure: failed draw, `status='base_unconverged'`.
Products `n_fp`, `r_fp`. An estimator without a start is deterministic:
no iteration, r = 0, no evaluation.

The last iteration's two values also give the fit's **absolute
reproducibility per output**, ν_o = max(η_f · |θ̂_o|, |θ'_o − θ_o|) with
θ', θ the final pair (for a start-less T, ν_o = η_f · |θ̂_o|). ν is the
noise scale of every contrast (4.2, 7.2): a relative scale alone
returns no noise for an output near zero, which the build found.
Product `nu_o`.

### 2.2 The weight construction and the step

For a member set K of points with mass p_K = |K|/N and weight parameter
t:

    ω_i(t) = (1 − t) + t · 1_K(i) / p_K,

Σω(t) = N for every t; the member weights' relative change is
t(1/p_K − 1) = δ, so t = `step_parameter(δ, p_K)`. δ = δ_f =
`forward_step(η_f)` = 2√η_f, η_f = η_full when T takes a start, else T's
declared η. The forward difference along K,

    U_K = [T(ω(t)) − θ̂] / t,

is the mass-centred mean influence of K (ψ̄_K − ψ̄). Σω is fixed, so the
grand mean never enters and U_root = 0 by construction.

## 3. The hierarchy

### 3.1 Bisection rule

For a node with member positions z (in Z): μ = the mean, S = the
scatter Σ(z − μ)(z − μ)ᵀ / n; v = the eigenvector of S for its largest
eigenvalue (`numpy.linalg.eigh`; a tie takes the vector it returns for
the largest index). Seeds: the member with the smallest projection
(z − μ)·v and the member with the largest (ties: lower point index).
Lloyd: assign each member to the nearer seed (Euclidean in Z; ties to
the first seed), recompute the two means, repeat until the assignment
does not change, at most 100 iterations. Child A₀ = the child holding
the first seed; the other is A₁. No randomness.

### 3.2 Construction

Root = all N points, id 0, depth 0. A node's children are built when the
node is first selected for measurement (5.2 step 4), never before, so
only the measured part of the hierarchy exists. Ids are assigned in the
order children are created, breadth-first within a round (round's
selected nodes in id order, A₀ before A₁). A node of one point is a
**point leaf**. A node whose members all share one position (S = 0)
cannot split and is a leaf with C = 0 (7.1); a non-degenerate node of
two members splits into two point leaves.

### 3.3 Children by mass

For an internal node c with children A₀, A₁: the **measured child**
A = the child of larger mass (ties: A₀), the other is B. p_c = p_A + p_B.

## 4. Measuring a node

### 4.1 Coefficient and energy

Measuring node c = the step-verified forward difference (4.3) of U_A
along its measured child (2.2), starting at t_A = `step_parameter(δ_f,
p_A)`. U_c is known (U_root = 0; otherwise from the parent's
measurement), so

    U_B = (p_c · U_c − p_A · U_A) / p_B,
    y_c = U_A − U_B,
    E_c = (1/N) · (p_A · p_B / p_c) · y_c².

Every measurement is on the same X from the same θ̂, so the derivation
is valid at every node.

### 4.2 Noise and the pass rule

    s_c = √2 · ν / t · (p_c / p_B)      (per output; ν from 2.1; t the accepted step of 4.3)

Output o **passes** at node c when |y_c,o| > z_n · s_c,o, z_n = 5 (the
filter tested offline; the constant is declared here and nowhere
else). Energy is counted only where it passes: E_c,o ← E_c,o · 1[pass].
A node with no passing output is **closed**: its children are leaves
(not below the floor, 5.2).

### 4.3 Step verification (every contrast: nodes, pairs, quadratic)

A forward difference is accepted only when halving its step does not
change it beyond noise. For a contrast whose first step is t (4.1 for a
node; 7.2, 7.3 for the others), with U(t) its raw difference value and
s(t) = √2 ν / t (times p_c/p_B for a node's y):

    k = 0; measure U(t), U(t/2)
    while some output o has |U(t/2^{k+1}) − U(t/2^k)| > z_n · s(t/2^{k+1})
          and s(t/2^{k+1}) < |U(t/2^{k+1})| on such an o, and k < 8:
        k ← k + 1; measure U(t/2^{k+1})
    accept U(t/2^{k+1}) at step t/2^{k+1}, with its own s.

Two evaluations per contrast in the usual case, one more per halving.
A contrast that leaves the loop by the noise condition or the cap
(k = 8, a failure boundary) is recorded as **non-smooth**: its accepted
value is the smallest-step one, it passes or fails 4.2 as any other,
and `nonsmooth = True` in its table row; `n_nonsmooth` is a scalar
product. The build found nodes whose response changes sign between δ_f
and 2δ_f and nodes exact at δ_f/2 but not at δ_f; both are caught here
and neither is caught by any noise rule. This is the paper's
step-doubling check applied to every contrast.

A failed evaluation (exception, or any NaN in T's output): y_c = NaN on
every output, no output passes, E_c = 0, `n_failed` += 1, the node is
closed; the task's `fit_status` is recorded in the nodes table.

## 5. Growing the tree

### 5.1 State

The measured nodes with (U_A, U_B, y, E, s, pass flags, round, below
flag), the current leaves (7.1's definition, maintained incrementally),
per output V_btw,o = (1/N) Σ_leaves p_ℓ U_ℓ,o² and L = the number of
leaves, and `evals_total` (every stage).

### 5.2 Rounds, the floor, the lineage rule and the stop

M_floor = `cost_rule_M(N, q, eps)` (core.xvq; the paper's prototype
count), the tree's minimum resolution: the survey's resolution the
paper always had under its rule. Round 0 measures the root. Then, per
round r ≥ 1:

1. **Below the floor** (L < M_floor at the round's start): candidates
   = every child of a node measured in round r − 1, excluding point
   leaves and degenerate nodes; nothing closes, the noise rule of 4.2
   only gates energy. Steps 3–6 as below.
2. **At or above the floor**: the threshold per output
   τ_o = eps · V_btw,o / L from the state at the round's start. Each
   node measured in round r − 1 is judged once, now: **below** when
   E_c,o < τ_o on every output (an output that failed the noise rule
   counts as below); nodes measured in the final round carry
   below = False. Candidates = the children of nodes measured in round
   r − 1 that are not closed (4.2), excluding point leaves and
   degenerate nodes, and excluding the children of a node that is below
   AND whose parent is below (a lineage closes after two consecutive
   below-threshold splits).
3. **Priority** of a candidate = its parent's max_o E_o / V_btw,o (an
   output with V_btw,o = 0 contributes 0); ties: lower node id.
4. **Selection**: every candidate, in priority order (the order fixes
   node ids and batch order, nothing else). The selected nodes'
   children are built now (3.2).
5. **Batch**: the selected nodes measured in one `pool.map` per step of
   4.3 (9.2), in selection order.
6. **Level gain and the stop**: G_r,o = Σ over the round's measured
   nodes of counted E_c,o. The descent stops after round r when
   L ≥ M_floor and max_o G_r,o / V_btw,o < eps for this round and the
   previous one (`tree_status='tolerance'`), or when there are no
   candidates (`'exhausted'`). The `curve` product gets one line per
   round with G_r,o.

### 5.3 What the rules certify and what they do not

The floor guarantees the resolution the paper's survey had, so a spike
too deep for coarse contrasts to see (the build found one) is reached
at the cost the paper paid. The level-gain stop has a fixed point on a
spread influence, where a per-leaf threshold does not (the build ran
two paper cases to the full data under the per-leaf rule alone). The
lineage rule prunes below the floor. None of them bounds the remainder
in closed lineages; nothing measured can (QIJ_state_brief.md section
2). The remainder is reported, not estimated (11.1).

## 6. The bias stencil

For the root's two children K ∈ {A, B}, the central second difference
along K at the second-difference step, by
`difference(p_K, central_step(η_f), evaluate, θ̂)` with
evaluate(±t) = T(X, ω(±t)) under 2.2 on member set K, start = θ̂,
eta = η_full: D2_K is its second return value. Four evaluations, stage
`bias`. b̂_o = (1/(2N)) · (p_A D2_A,o + p_B D2_B,o), the plug-in bias of
`ivq.bias_and_acceleration` on this two-set partition. The first return
value (the central U_K) is not used and not stored.

## 7. The within-term

Runs after the tree stops. Every term is a measured contrast; what is
not bought contributes zero, so V_win is a measured lower bound.

### 7.1 Leaves and their geometry

Leaves: every node in the tree table whose parent is measured and that
is not itself measured (the root is always measured). Each leaf ℓ has a
known U_ℓ (its parent's measurement), n_ℓ points, p_ℓ = n_ℓ/N, mean
μ_ℓ and covariance C_ℓ = Σ(z − μ_ℓ)(z − μ_ℓ)ᵀ / n_ℓ over its points in
Z, eigenpairs (λ_j, v_j) descending, rank r_ℓ = #{j : λ_j > d · λ_1 ·
machine-ε}. Point leaves and C = 0 leaves have r_ℓ = 0.

Sibling derivative (ranking only, never in any estimate): for leaf ℓ
with sibling s under parent c,

    ĝ_ℓ,o = |U_ℓ,o − U_s,o| / ‖μ_ℓ − μ_s‖      (0 when the norm is 0).

### 7.2 Pairs

A pair is (ℓ, j) with 1 ≤ j ≤ min(r_ℓ, n_ℓ − 1). Predicted term and
score:

    P_ℓj,o = p_ℓ · λ_j · ĝ_ℓ,o² / N,        score(ℓ, j) = max_o P_ℓj,o / τ_o,

τ_o = eps · V_btw,o / L at the tree's final state (V_btw,o = 0 →
that output contributes 0 to the score). Pairs are bought in descending
score (ties: lower leaf id, then lower j) while score ≥ 1.

Contrast for (ℓ, j): h_i = (z_i − μ_ℓ)·v_j / √λ_j for i ∈ ℓ, 0
elsewhere (mean 0 and mean of h² = 1 over ℓ, so Σ h = 0);
ω_i(t) = 1 + t h_i, Σω = N; t = δ_f / max_i |h_i|. Response and
contribution:

    D_ℓj = [T(ω(t)) − θ̂] / t   step-verified by 4.3,   s = √2 · ν / t at the accepted step,
    W_ℓj,o = (1/N) · D_ℓj,o² / p_ℓ   where |D_ℓj,o| > z_n s_o, else 0.

(For ψ affine in the leaf, D = p_ℓ (g·v_j) √λ_j, so W is exactly one
term p_ℓ λ_j (g·v_j)²/N of the leaf's affine within-variance
p_ℓ gᵀC_ℓg/N.) One batch, stage `within`. Failure as in 4.2.

### 7.3 Quadratic contrasts

For each leaf with at least one bought pair and n_ℓ ≥ r_ℓ + 2, its
measured affine term A_ℓ,o = Σ_j W_ℓj,o. Leaves are bought in descending
max_o A_ℓ,o / τ_o (ties: lower id) while that ratio ≥ 1. Contrast: m_i = (z_i − μ_ℓ)ᵀ C_ℓ⁺ (z_i − μ_ℓ) (pseudo-
inverse of rank r_ℓ), h̃_i = m_i − mean_ℓ(m) − Σ_{j ≤ r_ℓ} β_j h_ij with
β_j = mean_ℓ(m · h_j) (orthogonal to every principal direction, bought
or not), h = h̃ / sqrt(mean_ℓ(h̃²)); skip the leaf if mean_ℓ(h̃²) = 0.
Weights, step, response Q_ℓ, noise and pass rule as in 7.2;
contribution (1/N) Q_ℓ,o²/p_ℓ where it passes. One batch, stage
`quadratic`.

### 7.4 The within-term

    V_win,o = Σ_bought pairs W_ℓj,o + Σ_bought leaves (1/N) Q_ℓ,o²/p_ℓ.

The directions bought in a leaf are orthonormal over the leaf, so the
sum never double-counts.

## 8. Variance, reconstruction, intervals

### 8.1 Per-point influence

    ψ̂_i,o = U_ℓ(i),o + Σ_{bought and passed directions of ℓ(i)} (D_o / p_ℓ) · h_i

(Q in place of D for the quadratic direction). ψ̂ is mass-centred
because every U is.

### 8.2 Variance and acceleration

    V_btw,o = (1/N) Σ_leaves p_ℓ U_ℓ,o²,     V_tot,o = V_btw,o + V_win,o,
    a_o = (1/N) Σ_i ψ̂_i,o³ / ( 6 √N · ((1/N) Σ_i ψ̂_i,o²)^{3/2} ),

`ivq.bias_and_acceleration`'s acceleration at point masses 1/N; NaN
where the denominator is 0.

### 8.3 Curvature

`core.abc.curvature(counter, sv, θ̂, ψ̂_full, p, N, pool, outputs=measured)`
with `sv = SurveyRows(rows=X, row_field=arange(N), omega0=ones(N),
eta_Q=η_full, step_ratio=full((5, q_full), NaN),
quantized_start='full-data', eta_rows=η_full)`, ψ̂_full (N, q_full) =
8.1's ψ̂ in the measured columns and 0 elsewhere, p = full(N, 1/N).
2q evaluations, stage `curvature`, giving c_q. (`curvature` reads
`sv.rows`, `sv.omega0`, `sv.eta_Q`; the array it evaluates on is X,
already shared.)

### 8.4 Intervals

`QIJDTResult.variance` = V_tot. `interval(level)` =
`_normal_interval(θ̂, V_tot, level)`; `interval_btw(level)` = the same on
V_btw; `abc_interval(level)` = `core.abc.abc_interval(θ̂, √V_tot, a, b̂,
c_q, level)`. All three are reported.

## 9. The draw and its parallelism

### 9.1 Stage order and accounting

full_fit → eta_full → tree → bias → within → quadratic → curvature →
assembly. `evals_by_stage`, `rows_by_stage` (= N per evaluation),
`wall_time_by_stage` keyed by these names; `busy_time_total` = the sum
of task wall times. `evals_total` counts every stage; nothing stops a
stage but its own rule.

`QIJDTResult` carries every scalar of 11.1 as an attribute (per-output
ones as (q,) arrays), the `nodes`, `leaves`, `pairs`, `curve` tables as
DataFrames, `psi_hat` (N, q) and `leaf_of_point` (N,), and the three
interval methods.

### 9.2 Contract

`pool.share(X)` once per draw. Every batch is one `pool.map` over tasks
of the shape of `core.abc._curvature_task`: `task = (key, ω, start,
eta)`, the worker evaluates `call_T(T, X, ω, start, eta)`, returns
(key, value, failed, wall); results are consumed by position; the parent
calls `counter.add(1, N, failed)` per task. With `pool=None` the same
task functions run through the counter in the same order.

Bit identity across worker counts is required: every decision (pass,
below, selection, ranking) is a function of returned values only; every
sum over nodes, leaves or points runs in ascending id order; no
floating-point reduction depends on batch composition. Validation V3
checks it.

## 10. Failures

Failed draw (products written with NaN variances and the status): θ̂
NaN (`'theta_hat_failed'`), the fixed-point cap (`'base_unconverged'`).
Not failed: any single node, pair, quadratic, bias or curvature
evaluation failing (handled where defined, counted in `n_failed`), the
eta measurement's `failed` flag (recorded as `eta_full_failed`). No
retries anywhere.

## 11. Products

Method directory `qijdt`, through `products.write_draw`; the scalar row
is the done marker.

### 11.1 Scalar row

`s, seed, N, eps, M_floor, workers, status, theta_hat_status,
eta_full, eta_full_failed, n_fp, r_fp, n_rounds, n_measured, n_leaves,
max_depth, n_failed, n_nonsmooth, n_halvings_total, tree_status,
n_pairs_bought, n_quad_bought, evals_total, busy_time, wall_time`, per
stage `evals_<stage>`, `wall_<stage>`, and per output o: `theta_hat_o,
nu_o, V_btw_o, V_win_o, V_tot_o, accel_o, b_hat_o, c_q_o, lo_o, hi_o,
lo_btw_o, hi_btw_o, lo_abc_o, hi_abc_o` at level 0.95, plus the reported remainder
`P_unbought_o` = the sum of 7.2's predicted terms over pairs NOT bought
(a diagnostic, never in an interval) and `n_below_closed` = the number
of lineages closed by the two-consecutive rule.

### 11.2 `nodes` (every draw)

One row per measured node and per leaf: `node, parent, depth, n_points,
p, node_measured (bool), round, below (bool), measured_child, t
(accepted step), n_halvings, nonsmooth (bool), status`, per output
`U_o, y_o, s_o, E_o, pass_o` (NaN where not measured; U known for every
row).

### 11.3 `leaves` (every draw)

One row per leaf: `leaf, n_points, p, rank, n_pairs_bought, quad_bought`,
per output `U_o, ghat_o, A_o, Q2_o`.

### 11.4 `pairs` (every draw)

One row per bought contrast: `leaf, j` (0 for the quadratic), `t`
(accepted step), `n_halvings, nonsmooth`, per output `D_o, s_o, pass_o`
(Q in the D column for j = 0).

### 11.5 `curve` (every draw)

One row per round: `round, evals_total, L, n_selected, below_floor
(bool)`, per output `V_btw_o, tau_o, G_o` (5.2 step 6).

### 11.6 `points` (only `--diag-draws`)

`i, leaf`, per output `psi_hat_o`.

## 12. Validation

The author's rule (29 Sept): the tree is second-paper material and is
not retested across the paper's cases on every change. Two tiers.

**Per change (the standing check, about ten minutes, no report beyond
the numbers):**
* cloudfil draw 1 (the converged-basin draw) and (pareto, shape) draw 0
  at eps 0.01: V_tot,o / V_ij,o per output, evals_total, tree_status,
  n_nonsmooth, n_halvings_total.
* bit identity on (pareto, shape) draw 0 at workers 1 and 8 (every
  product file byte-identical after dropping the timing columns): the
  one pass/fail item.
* the all-methods cloudfil smoke unchanged, `qijdt` in it at eps 0.05.

**At a milestone the author calls, once (before the tree runs on
Stampede3 for the second paper):** the six paper cases at N = 2000,
draws 0–9, against the pipeline's `ij` and `qij` products (evals to
0.99 of V_ij from the curve, V_tot/V_ij, coverage of the three
intervals); cloudfil draws 0 and 1 with the offline node check against
the exact influence (member sets rebuilt by the same bisection) and
busy/wall at 14 and 1 workers. Not otherwise.

Reports to `Research/QIJ_joint/qijdt_validation/` with the scripts.

## 13. Questions a coder will ask, answered

* **Is there a seed?** Not in the method (1.1); the CLI's `--seed`
  seeds the draw, and the scalar row's `seed` is that draw seed.
* **Where does `below` get set?** At the start of the round after the
  node was measured (5.2 step 1), once; the last round's nodes carry
  False.
* **Does the hierarchy need the whole tree up front?** No (3.2): a
  node's children are built when it is first selected. Memory is the
  measured part only.
* **Where do Z and X differ?** Positions for the bisection and the
  leaf geometry are in Z; T is always evaluated on X. Both are (N, ·)
  arrays over the same points.
* **What is "the parent's measurement gives U_ℓ" for the root's
  children?** U_A is measured, U_B derived with U_root = 0 (4.1).
* **What if V_btw,o = 0 at a round's start?** τ_o = 0: nothing is below
  on that output, and the priority contribution of that output is 0.
  This happens only at round 1 if the root's coefficient failed the
  noise rule on every output, in which case the root is closed and the
  tree stops with `'tolerance'` after round 0.
* **Does the two-consecutive rule look at the noise rule?** Yes: an
  output that fails the noise rule counts as below τ on that output
  (5.2 step 1); a node with no passing output is closed outright (4.2),
  which is stronger.
* **Can a bought pair be in a leaf that is later split?** No: pairs are
  bought after the tree stops (7).
* **What bounds the run?** Only the tolerance rule; the tree has at most
  N − 1 splits. The evaluation count is a product, not an input.
* **Which noise scale for a start-less estimator?** ν_o = η_f · |θ̂_o|
  with the declared η (2.1), the same η_f that sets δ_f; there is no
  fixed-point pair to measure an absolute one from.
* **How many evaluations does a node cost now?** Two in the usual case
  (t and t/2), one more per halving (4.3); the tree's count roughly
  doubles. Batches: one `pool.map` per step of the halving loop, over
  the nodes still unresolved.
* **What happens below the floor?** Every non-leaf child is measured
  (5.2 step 1); no node closes, the noise rule only gates energy; the
  lineage rule and the stop apply only once L ≥ M_floor.
* **What is M_floor on cloudfil?** `cost_rule_M(10000, 7, 0.01)` = 1094,
  the paper's number; on the paper's cases the paper's M_X values.
* **What does `curvature` need that a `SurveyRows` normally carries?**
  Only `rows`, `omega0`, `eta_Q` (8.3); the rest is filled as stated
  and unused.
* **Are the paper's cases run in Z?** Yes where a case has a
  `vq_transform` (mvt); the bisection and leaf geometry use Z, the
  evaluations X, as in `qij.py`.
* **Is anything scaled?** No. Every quantity is measured on the full
  data from θ̂; there is no scale, no anchor, no drift.
