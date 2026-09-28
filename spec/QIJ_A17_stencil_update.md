# A17. The stencils update the influence model (author's instruction, 28 September 2026)

Companion to `QIJ_mods_waves.md` (A17 is listed there) and to the
diagnosis in `QIJ_stage2_groundup.md`. Terms as in `method_notes.md` and
`QIJ_glossary.md`; the ones this document leans on:

- **receptive field**: the data points whose nearest prototype is a given
  prototype of the first-stage quantizer (the 𝒳-VQ).
- **prototype influence** I_jc: the surveyed derivative of output c with
  respect to receptive field j's total weight (`xvq.prototype_influences`).
- **influence model**: the Gaussian-process regression of I_·c on the
  prototype positions (`influence_model.fit_influence_model`), giving
  ψ̂₀_c(x) at every point, its posterior standard deviation σ_c(x), and
  the within-bin posterior variance v_kc (`bin_posterior_variance`).
- **stencil**: a full-data finite-difference evaluation over a set S of
  points: the central stencil of an initial bin (`ivq.bin_differences`)
  or the forward difference of a refinement split's smaller child
  (`refine.apply_split`). Its result U_S is a vector over every output.

## 1. The defect

Every stencil measures, exactly (slope 1.000, r² 1.000 against the
analytic influence on both rehearsal draws), the mean of the influence
over a known set of points, for every output at once. As built, the
refinement uses that measurement for one thing: the between-term
increment of one output. The influence model that steers every later
decision (the split geometry, the kind rule, the expected gain, τ, the
closing rule, V_win_hat) is frozen at the end of stage 1 and is never
told what the stencils found. On the demo estimand the frozen model is
wrong exactly where the variance is (inside P2: v_k at 0.2–0.3 of the
truth even with local width; the point prediction inside P2's receptive
fields explains 0.4–0.7 of the variance on draw 0 and 0.1–0.5 on draw 1),
so the search is steered wrong and the evaluations it pays for cannot
correct it.

## 2. The rule

A stencil over set S is an observation of the linear functional
(1/n_S) Σ_{i∈S} ψ_c(x_i), exact, for every measured output c. A Gaussian
process conditions on such an observation in closed form. Therefore:

**Every stencil enters the influence model of every measured output as
an exact observation, and every open leaf of every output is re-proposed
from the updated model.** Nothing else in the refinement changes: the
kind rule, the gain formulas, τ, the two-strike closing rule, the guard,
mass balance, γ, ρ², V_win_hat are the ported ones, evaluated on the
updated model instead of the frozen one.

## 3. The model with set observations

For output c, with the stage-1 hyperparameters held fixed (trend basis
h(·) of `gptrend`, kernel k(·,·) of `gpwidth` — under `local`,
ℓ_i = c·h_{bmu(i)} for a point and ℓ_j = c·h_j for a prototype, the
nonstationary Matérn-3/2 already in `_matern32_nonstationary` — signal
variance s²_c, survey noise λ_c), the design is

- the prototypes: response I_jc / a_c, noise λ_c (as now);
- the measured sets S_1 … S_n: response U_{S c} (the centred value
  `apply_split`/`bin_differences` already compute), noise 0 (numerically:
  a jitter of 10⁻¹⁰ s²_c on the diagonal).

Covariances follow from the kernel by linearity:

    k(S, w_j)  = (1/n_S) Σ_{i∈S} k(x_i, w_j)
    k(S, S')   = (1/(n_S n_S')) Σ_{i∈S} Σ_{l∈S'} k(x_i, x_l)
    k(x, S)    = (1/n_S) Σ_{i∈S} k(x, x_i)
    h(S)       = (1/n_S) Σ_{i∈S} h(x_i)

The first two sums are the quantities `bin_posterior_variance` already
forms as `s_vec` and `SS_k` (there for one set against itself); the new
code forms them once per set, and once per pair of sets, and caches
them: a set is immutable, so its terms are computed when it is measured
and never again. ψ̂₀_c(x), σ_c(x) and v_kc are then the standard
posterior mean, variance and within-group variance of the augmented
design (the same formulas `_point_terms`/`bin_posterior_variance`
implement, over M_𝒳 + n design rows). The trend coefficients stay
integrated out as now (the G matrix over the augmented design).

**The scale factor a_c.** The survey values are attenuated relative to
the stencils by a factor that is nearly uniform per output (1.3–1.6 on
draw 0 of the demo). Before the first update, after every output's
initial bins are measured, a_c = Σ_k p_k U_kc m_kc / Σ_k p_k m_kc²
(`joint._fit_scale`, m_kc the frozen model's mean over bin k, the
quantity A15 already computes); the survey responses are divided by a_c
and the model refit once with those responses and the initial sets. a_c
is a product (already `a_c` per output) and is never refit. If a_c is
non-finite or ≤ 0 it is 1.

**What is not refit.** The width (c, or ℓ), λ_c and s²_c stay at their
stage-1 REML values. The stencils are exact and enter with zero noise,
so they dominate the survey wherever they disagree with it; the survey
supplies the shape between and beyond the measured sets.

## 4. Schedule

`refine_update` is a new switch: `none` (the ported refinement, default;
bit-identical on every audited path) and `stencils` (this rule, the demo
setting). `stencils` requires `refine_schedule = rounds`; `queue` with
`stencils` raises `ValueError` (a refit per split is not paid for).

Under `stencils`, one round is:

1. **Select** (as `rounds._select_round` under the ported rule, no
   flagged-first ordering): every open leaf at or above its output's own
   τ, sorted by (−g, id), capped at the output's remaining budget.
2. **Evaluate** all selected splits in one pool batch (as now).
3. **Apply** each split (`apply_split` as now: U_small, U_large by mass
   balance, Δ, V_btw, γ, strikes, children created and closed or left
   open). The children's `g`/`v` are NOT priced here.
4. **Update** the model with every new set of this round: both children
   of every applied split (the larger child's U is exact by mass
   balance, so it enters too), for every measured output at once (one
   vector U_S per set). The set terms are formed and cached; the
   augmented factorisation is rebuilt (one per coordinate group).
5. **Re-predict** ψ̂₀_c and σ_c at every point for every output.
6. **Re-propose** every OPEN leaf of every output: `var_k`, `ubar`,
   `v` and the proposal (`propose`: kind rule, split geometry, g) from
   the updated ψ̂₀_c and v_kc. Strike flags and closed leaves are left as
   they are. ρ² is recomputed from the updated ubar's.

Before round 1: the initial bins of every output are measured (as
now), a_c fitted, the model refit with rescaled survey responses and all
initial sets (7 × 17 here), ψ̂₀/σ re-predicted, and every initial leaf
proposed from the updated model. The initial bins themselves are not
rebuilt (their stencils are already paid for; the refinement corrects
the partition).

**Stopping.** An output with nothing selected in a round is NOT retired,
since another output's stencils may change its model: the loop stops
when a round selects nothing for ANY output, or every output is at its
guard. The guard is 1 + M_𝒳_used refinement evaluations per output, as
now.

**Cost accounting.** The refit and re-prediction are numpy work, no
estimator evaluations; their wall time is a product (`update_wall_time`,
summed over rounds), reported beside the refinement's evaluation wall
time. The pairwise set terms are O(n_S · n_S') kernel evaluations per
pair; the initial sets of all outputs together cost (q N)² ≈ 5 × 10⁹
kernel evaluations at N = 10 000, q = 7, once; each child adds
n_child × (q N + M_𝒳). Measure it; do not assert it.

## 5. Removals

With the model corrected by measurement, A15's steering rules have no
role. Remove `refine_trigger` and everything only it used: `flag_leaf`,
`propose_geometric`, `_fit_plane_gradient`, `_ancestor_gradient`,
`_median_projection_split`, `_live_descendants`, `whiten_columns`,
`Z_white`, `parent_id`/`children_of`, `from_flagged_lineage`,
`direction_rule`, `discrepancy`, `flagged`, the products `n_flagged`,
`n_flag_evals`, `n_unflag_evals`, `n_geom_splits`. Keep `a_c` (section
3). The ported `gain` behaviour is what `refine_update = none` is.
`QIJ_mods_waves.md` A15 items 1–5 are superseded by this document; A15's
η_full, local width, and bookkeeping fixes stand.

## 6. Products

Per output: `a_c` (kept), `n_update_rounds` (shared), `update_wall_time`
(shared), `refine_update` in every row. `bin_U` stays. Nothing else new.

## 7. Acceptance, in order

1. **Bit identity.** `refine_update = none`: the Fundamental Plane and
   Pareto audits, and the stored cloudfil rehearsal draw 0 (rounds, gain,
   global width) reproduce V_btw, L and every evaluation count exactly.
2. **The exact simulator, before any estimator is run.** The planner's
   harness `stage2_review/simulator.py` (path in the message to the
   coder) replaces T by θ̂ + (ω − 1)Ψ/N with the stored analytic influence
   Ψ (`ij/s0000{0,1}.psi.parquet`, N × 66) and θ̂ (`oracle/s0000{0,1}
   .parquet`), and runs `run_refinement_rounds` unchanged; it reproduces
   the stored draw-0 and draw-1 results to the evaluation. Run the built
   `stencils` path through it on both draws with `gpwidth = local`,
   `gptrend = quadratic`, the stored survey (`res.prototype_I` in the
   diagnostics pickle). Report per output: V_btw/V_ij, evaluations,
   rounds, update wall time; and, for reference, the same harness under
   `none`. Pass: V_btw/V_ij ≥ 0.97 on every output on both draws, no
   output at its guard.
3. **Then the rehearsal rerun**, exactly as A15 defined the TACC gate,
   with `refine_update = stencils` in place of `refine_trigger =
   measured`.

If step 2 fails, stop and report the table; the next step is the
planner's, not another steering rule.

## 8. Amendments after the coder's review (28 September, evening)

These supersede the corresponding statements above.

1. **No linear dependence in the design.** (a) The design holds only
   each output's CURRENT leaves: when a leaf is split, its row leaves
   the design and its two children's rows enter (a parent's mean is
   exactly the mass-weighted mean of its children's). (b) Set rows are
   not exact. A stencil carries finite-difference error, declared by the
   same rule the survey's noise floor uses: noise sd √2 · η_full · |θ̂_c|
   / t_S for a set measured at step t_S (the initial bins' central
   stencils at their t_k; a split's smaller child at t_small; the larger
   child inherits the smaller's noise scaled by p_small/p_large through
   mass balance). Each set row's noise variance is that sd squared,
   divided by s²_c to enter the model's λ scale. (c) The solve is an
   eigendecomposition of the augmented covariance with a relative cut
   at 10⁻¹⁰ of the largest eigenvalue, never a bare Cholesky: every
   output's partition still sums to the global-mean functional, and
   those near-redundant rows must be handled, not assumed away.
2. **The survey rows are set observations too.** From the first update
   on, prototype j's influence I_jc / a_c enters as an observation of
   the mean of ψ_c over receptive field j (noise λ_c), not as ψ_c at the
   prototype's position. Every set covariance, receptive field or leaf,
   is formed as K · A with A the N × n_sets averaging matrix (column S
   holds 1/n_S on S's points), in row chunks of bounded size; the same
   product gives k(x, S) for re-prediction. The stage-1 fit itself
   (hyperparameters, the first ψ̂₀ for the initial bins) is unchanged.
3. **λ_c is re-estimated once.** After a_c, one-dimensional REML on λ_c
   per output with the set rows in the design at their declared noise,
   the width (ℓ or c) and s²_c held at their stage-1 values; a
   log-spaced grid of five candidates and one bounded refinement, as
   the width search does. Not repeated at later rounds.
4. **Kernel sharing.** Under `gpwidth = local` the factor c is fitted
   per coordinate group (outputs sharing the same finite prototype
   design); the kernel and its set terms are formed once per group, the
   noise and responses per output. One group when every prototype is
   finite for every output; otherwise one per group.
5. **Stopping.** The loop stops when a round selects nothing across all
   outputs, or every output is at its guard.
6. **The simulator's role.** Section 7 step 2 is a PRE-SCREEN, one-sided:
   a fail stops the rehearsal; a pass permits it and proves nothing
   more, since the harness runs on the analytic influence with a linear
   T, which is not the method and not available to a user. Its inputs
   are regenerable from products (the dataset function for X, the `ij`
   product for Ψ, the oracle for θ̂, a diagnostic qij run's
   `prototypes`/`points` products for the survey, the model refit from
   those); the session-scratch pickle is not required. If kept, the
   harness lives in `scripts/` and calls the code as it stands after
   this document's removals.
