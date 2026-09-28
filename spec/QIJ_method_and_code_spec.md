# QIJ: the method and the `qij_joint` codebase, as they stand on 28 September 2026

A self-contained specification for a reviewer or agent who has not seen
this project. It states the method as published, every modification made
since, the code that implements each part, the state of each part
(audited, ruled, built and unaccepted, pending), the measured defects
found on the demo estimand, and the repair that is now being built. Where
it cites earlier documents it also states their content, so nothing here
depends on reading them; they are named so claims can be checked:
`spec/method_notes.md` (the ported method, section by section),
`spec/QIJ_mods_waves.md` (rulings A1–A17), `spec/qij_joint_plan.md`
(package plan, code standards §5), `spec/QIJ_stage2_groundup.md`
(the diagnosis), `spec/QIJ_A17_stencil_update.md` (the repair), and the
paper's own spec `../qij/spec/QIJ_method_spec.md` (revision 10) with
its glossary `../qij/spec/QIJ_glossary.md`.

Author: the project's owner (astronomer, statistics and machine
learning), who rules on every design decision. The planner (this
document's writer) specifies; a coding manager dispatches agents that
build; the author approves every dispatch.

---

## 0. The problem QIJ solves

An estimator T maps a data set X = (x_1, …, x_N), x_i ∈ ℝ^{d_x}, to q
numbers θ̂ = T(X) ∈ ℝ^q. T is a black box: it can be evaluated at any
non-negative weights ω on the points, T(X, ω), with Σ_i ω_i = N, but its
internals (scores, Hessians) are not available. Each evaluation may be
expensive (a mixture fit, a multi-start optimizer). The question is the
sampling variance of θ̂, and a confidence interval, at a fraction of the
bootstrap's cost of B refits.

The infinitesimal jackknife (IJ) writes the variance in terms of the
influence function ψ_i = N · ∂T/∂ω_i at ω = 1 (an N × q array):

    V_ij,c = (1/N²) Σ_i ψ_ic²        (c = 1 … q)

after mass-centring ψ (Σ_i ψ_ic = 0). Computing ψ exactly costs N
evaluations of size N (the method `ijfd` below does exactly this, as the
cost reference). QIJ approximates V_ij with far fewer evaluations by
exploiting one fact: V_ij is a variance, and a variance is recovered by a
partition of the points into groups whose means are measured, up to the
variance left inside the groups. If the influence is nearly constant
within each group, measuring the groups' means suffices.

Define, for a partition of the points into bins k = 1 … L with masses
p_k = n_k/N and bin-mean influences U_kc = (1/n_k) Σ_{i∈k} ψ_ic,

    V_btw,c = (1/N) Σ_k p_k U_kc²  ≤  V_ij,c,

with equality when ψ_c is constant within every bin. V_btw is a LOWER
BOUND on V_ij by the law of total variance, exact as a computation once
the U_k are known. U_k is measured exactly by a finite difference of T
that raises bin k's total weight and lowers everyone else's in
proportion (section 1), at two evaluations per bin. QIJ's job is
therefore to find a partition with few bins that leaves at most a
fraction ε (the tolerance, 0.01 in the study) of V_ij inside the bins,
and to measure it. The paper's estimators are five: the Pareto shape and
tail, the multivariate-t degrees of freedom and tail, the Fundamental
Plane fit, and a Chabrier initial-mass-function fit; on those the method
was validated against the analytic ψ and passed. The demo estimand
(section 8) is a ten-component Gaussian mixture; it is where the method
currently fails, and section 9 states exactly how.

---

## 1. The weight constructor and the finite-difference steps

(`core/differences.py`; method_notes §1; audited bit-exact.)

For a member set K of mass p in base weights ω⁰ (summing to the row
count R), one constructor:

    ω_i(t) = (1 − t) ω⁰_i + t ω⁰_i 1{i ∈ K} / p ,        Σ_i ω_i(t) = R.

A relative step δ on K (the fraction of K's own mass the step moves it
by) is the weight parameter t = δ p/(1 − p). Steps: δ_f = 2√η for a
forward difference (the prototype survey; a refinement split), δ =
(3η)^{1/3} for the central stencil of an initial bin. η is the
estimator's reproducibility (section 7). The central stencil:

    U = [T(+t) − T(−t)] / (2t),     D² = [T(+t) − 2T(0) + T(−t)] / t²,

U the bin-mean influence (all q outputs at once), D² the curvature used
by the ABC interval (section 6). A forward difference for a split's
smaller child: U_small = [T(ω(t_small)) − θ̂]/t_small − (centring
residual); the larger child by mass balance U_large = (p_parent U_parent
− p_small U_small)/p_large.

`Counter` (`core/counter.py`) wraps T, counts every evaluation, its rows
and any NaN result, caches `T.prepare(X)` per array, and passes `start`
and `eta` through to T only when T declares `takes_start`.

MEASURED FACT (28 September, both rehearsal draws, every output): the
final bins' measured U_k regress on the true bin means of the analytic
influence with slope 1.000 and r² 1.000. The stencils are exact. Every
shortfall of V_btw below V_ij is the partition, never the measurement.

---

## 2. Stage 1a: the 𝒳-VQ (data-space quantizer) and the prototype survey

(`core/xvq.py`; method_notes §2; A4, A8, A9.)

**Prototype count** (A4, the ported rule, the author's ruling):

    M_ref = ⌈√(2.7/ε)⌉ = 17 at ε = 0.01,   M_𝒳 = ⌈√((1 + 2 q M_ref) N / 2)⌉,

floored at 20, capped at N/2; q is the number of MEASURED outputs. On the
demo, N = 10 000: 1013 at q = 6; the code counts position as two outputs,
q = 7, giving 1094. `QIJ(M_X=…)` overrides; `M_X_source` records which.

**Quantizer.** k-means on Z (the data, or a per-case `vq_transform` of
it) via `vqlp.VQFitter` (FAISS), two best-matching units per point,
deterministic at a seed and at any thread count; empty prototypes
dropped (M_used ≤ M_𝒳). Products: `centers` (M_used × d_z), `bmu`,
`bmu2` (N,), masses p_j = n_j/N, and CONN, the symmetrized CADJ
adjacency (prototypes j and k are adjacent when some point has them as
its two nearest). A prototype's **receptive field** is the set of points
whose nearest prototype it is.

**Survey.** θ_Q = T on the survey rows at their base weights; then, for
each prototype j, one forward difference at t_j = δ_f p_j/(1 − p_j):
I_j = [T(rows, ω(t_j)) − θ_Q]/t_j ∈ ℝ^q, mass-centred over the finite
prototypes per output. I_j is the **prototype influence**: the derivative
of T with respect to receptive field j's total weight, intended as an
estimate of the mean of ψ over receptive field j. Cost: 1 + M_𝒳
evaluations of survey size.

Two representations of the survey rows (`survey`):

- `points` (ported, audited): one row per prototype, the prototype
  itself, weight M_used p_j. On the mixture estimator this fails (the
  fit collapses on the prototypes: 576 of 601 survey fits NaN, A8).
- `moments` (A8, built, the demo's setting): receptive field j is its
  own n_j native points when n_j ≤ d_x + 1, otherwise the d_x + 1
  vertices of a regular simplex centred at the receptive field's mean,
  mapped through the eigen-factor of its covariance, so its mean and
  covariance are reproduced exactly; row weights sum to the row count R
  (3 269 on the demo); a step at prototype j scales every row of
  receptive field j alike.

**Continuations** (A9, built): T with restarts is a discontinuous
function of the weights wherever the winning start changes, so every
perturbed evaluation continues from the base fit (`start = θ_Q` on the
survey rows, `start = θ̂` on the full data); the base fits keep their
own cold search. `quantized_start = full-data` (the demo) makes θ_Q
itself the continuation of θ̂ onto the survey rows. η_Q, the survey
rows' reproducibility (two fits from θ_Q at the base weights, one at a
10⁻⁶ perturbation; the largest relative difference), sets δ_f = 2√η_Q; a
step-doubling self-check on the five heaviest prototypes is stored
(`step_ratio`, expected ≈ 2; measured 1.98–2.30 on the demo).

MEASURED FACT (28 September, demo, draw 0): the surveyed I_j equals the
analytic influence of the estimator FITTED TO THE SURVEY ROWS (slope
0.995–0.999, r² ≥ 0.993, every output): the survey computes what it is
defined to compute. But that quantized fit's influence is not the mean
of the full-data influence over the receptive field: the true
receptive-field means run 1.29–1.63 × the surveyed values on six
outputs with r² 0.85–0.94, and the position angle has r² 0.20; draw 1:
slopes 0.89–2.13, r² 0.57–0.91. Of the error inside P2's receptive
fields, row placement accounts for 0.1–0.4 % of the variance, the
quantized fit's information matrix for 3–14 %, its parameter shift for
2–5 % (30 % for the angle). Representing P2's receptive fields, or every
receptive field within 0.4 of P2, by native points does not remove it;
only the full data does. The paper's assumption A2 (I_j ≈ receptive-
field mean of ψ) fails for this estimand and cannot be restored by a
local change to the quantizer. A uniform scale error is harmless to the
method (a level-set partition is scale-free and the stencils fix the
scale); the scatter and the angle's collapse are not.

---

## 3. Stage 1b: the influence model

(`core/influence_model.py`; method_notes §3; A1, A2.)

Per measured output c, a Gaussian-process regression of I_·c on the
whitened prototype positions w_j:

    I_jc = h(w_j)ᵀβ_c + f_c(w_j) + e_j,   f_c ~ GP(0, s²_c k),  e_j ~ N(0, s²_c λ_c),

giving ψ̂₀_c(x) (the posterior mean at every data point), σ_c(x) (its
posterior sd), and for any set of points k the within-set posterior
variance v_kc = mean(diag Σ_k) − mean(Σ_k) (`bin_posterior_variance`),
formed from the sums s_vec = Σ_{i∈k} k(x_i, w_·) and
SS_k = Σ_{i,l∈k} k(x_i, x_l) in bounded chunks (no N × M array is ever
formed). Outputs whose survey response is constant to η take a
"constant path" (ψ̂₀ constant, the coordinate failed downstream).

Switches: `gptrend` = `affine` (ported) | `quadratic` (A1; the demo).
`gpwidth` = `global` (ported: one Matérn-3/2 length ℓ per coordinate
group, REML on a five-point log grid plus one bounded refinement) |
`local` (A2: a nonstationary Paciorek–Schervish Matérn-3/2 with
ℓ_j = c · h_j, h_j the median whitened distance to prototype j's CONN
neighbours, one factor c per group by the same search; a data point
takes ℓ of its nearest prototype). λ_c by REML above a declared noise
floor derived from η. `local` was built and audited (identical to
`global` on the Fundamental Plane under the quadratic trend), ruled the
demo's setting on 28 September (A15 item 3, A11), and HAS NEVER BEEN RUN
ON THE DEMO: both stored rehearsal products record `gpwidth = global`.

MEASURED FACT (28 September): the fitted global width is 0.098 (whitened)
against P2's 0.05 × 0.025 core. Initial 17-bin share of V_ij (the
partition's quality before refinement) on draw 0: 0.64–0.90 (angle
0.008) on the built ψ̂₀; 0.78–0.86 (angle 0.32) on the raw survey values
without any GP; 0.87–0.92 (angle 0.30) with local width; 0.92–0.98 with
local width on EXACT receptive-field means; 0.99 on the true ψ. Draw 1:
0.11–0.58 / 0.47–0.74 / 0.49–0.86 / 0.53–0.92 / 0.98. The global-width
model delivers a worse partition than its own inputs. Its posterior v_k
is 0.004–0.5 of the true within-bin variance as built, 0.2–0.3 with
local width, 0.6–1.9 with local width on exact means.

---

## 4. Stage 2: the 𝓘-VQ (influence-space quantizer), stencils and refinement

(`core/ivq.py`, `core/refine.py`, `core/rounds.py`; method_notes §4;
A14, A15, A17.) This is the marginal path, `ivqbins = marginal`, one
partition per measured output, the demo's setting. The joint path
(`ivqbins = joint`, `core/joint.py`, one shared partition grown to the
tolerance and corrected by a measured check, method_notes §6) is built
and measured on the Fundamental Plane and is NOT used on the demo (on
the demo's true influence a joint partition at 1 % needs point-level
bins; the marginal partitions need about 20 groups each).

**Initial bins.** 1-D k-means on ψ̂₀_c at M_init = min(17, distinct
values). Each bin measured by the central stencil on the full data
(`bin_differences`, 2 evaluations per bin, all outputs per evaluation;
`start = θ̂`, step from η_full, section 7); U centred by the mass-weighted
residual; D² kept for ABC. V_btw,c = (1/N) Σ_k p_k U_kc². A NaN in any
initial stencil fails the coordinate (and, in `qij.py`, voids the
draw's variances).

**Refinement (the ported rule).** Every open leaf carries a proposed
split: a LEVEL split (two-means on ψ̂₀ within the leaf) when
Var_k(ψ̂₀) > v_k, priced g = ρ²(p_a ū_a² + p_b ū_b² − p_k ū_k²)/N from the
model's means; otherwise an ADJACENCY split (the leaf's points whose
second-nearest prototype carries a higher prototype influence than the
nearest, against the rest), priced g = ρ² p_k v_k/N. ρ² = V_btw /
((1/N) Σ_k p_k ū_k²) rescales the model to the measured scale. The leaf
with the largest g is split when g ≥ τ = ε V_btw/L: one forward
evaluation on the smaller child, mass balance for the larger, realized
Δ added to V_btw. Closing: two consecutive below-τ splits in a lineage
close both children (one cannot distinguish a constant bin from an
evenly divided one). Guard: at most 1 + M_𝒳 refinement evaluations per
output. γ = clip(Δ/g, 0, 1) discounts the within-term. Final:
V_win_hat = ρ² (1/N) Σ_k p_k γ_k [Var_k(ψ̂₀) + v_k], V_tot_hat = V_btw +
V_win_hat; the refined field ψ̂_i = U_{k(i)} + ρ(ψ̂₀_i − ū_{k(i)}).

`refine_schedule` = `queue` (ported: one split at a time) | `rounds`
(A14, built and accepted: every output's qualifying leaves split
together per round, evaluations pooled; τ frozen within a round; differs
from `queue` by at most ε).

`refine_trigger` = `gain` (ported) | `measured` (A15, built, four
rounds of amendment, never accepted; TO BE REMOVED by A17): initial bins
whose measured U_k disagrees with the model beyond a tolerance are
flagged and their lineages split geometrically (along a plane fitted to
neighbouring cells' measured levels, else the principal axis), priced by
Δ/2.

MEASURED FACT (28 September, exact offline simulator, section 10): the
code as built spends 2 140 (draw 0) and 3 188 (draw 1) refinement
evaluations to reach V_btw/V_ij of 0.76–0.92 and 0.57–0.93. Only 2–13 %
of the evaluations land on bins holding P2's points (the coder's own
diagnostic). With local width and the survey as input: 0.57–0.94 at 598
and 0.62–0.91 at 1 150. With local width and exact receptive-field
means: 0.94–0.99 at 301 and 0.84–0.95 at 663. The measured trigger
changes these by hundredths. The oracle, 17 bins on the true ψ, is 0.99
at 34 evaluations.

**The structural defect.** Every stencil measures, exactly, the mean of
ψ over a known set for every output. As built, that vector is used for
one thing: one output's V_btw increment (the children's vectors are
stored as `bin_U` and read by nothing). The model that steers every
decision (split geometry, kind rule, gains, τ, closing, V_win_hat) is
frozen after stage 1. On the demo it is wrong precisely where the
variance is, and the search cannot correct it however many evaluations
it spends. Eight to fifteen percent (draw 0) and thirteen to thirty-six
percent (draw 1) of V_ij lies INSIDE P2's 15–35 receptive fields, where
no stage-1 quantity can see.

**The repair, A17 (author's instruction, 28 September; spec
`QIJ_A17_stencil_update.md`; dispatched, not yet built).** Every stencil
enters every output's Gaussian process as an exact observation of the
linear functional (1/n_S) Σ_{i∈S} ψ_c(x_i): set covariances by linearity
of the kernel (the same s_vec/SS sums `bin_posterior_variance` forms,
computed once per set and cached), survey responses rescaled once by
a_c = Σ_k p_k U_kc m_kc / Σ_k p_k m_kc² (the least-squares scale of the
measured initial bins on the model's means), hyperparameters held at
their stage-1 values. After each round: update the model with the
round's new sets, re-predict ψ̂₀ and σ, re-propose every open leaf of
every output. Everything else is the ported rule. New switch
`refine_update` = `none` (ported, bit-identical) | `stencils` (the demo;
requires `rounds`). `refine_trigger` and all flagged-lineage code are
removed. Acceptance: bit identity under `none`; then, before any
estimator is run, the exact simulator (section 10) on both rehearsal
draws must give V_btw/V_ij ≥ 0.97 on every output; then the rehearsal
rerun (section 8).

---

## 5. Continuations, η, and the full-data base fit

(`core/eta.py`, `parallel.py`; A9, A15.) θ̂ = T(X, 1) is the full-data
fit, cold (the estimator's own search, section 7), computed once and
shared. η_full, the full data's reproducibility (two continuations from
θ̂ at unit weights, one at a 10⁻⁶ perturbation; largest relative
parameter difference, floored at the polish residual, rounded up to a
power of ten), replaces the declared η in every full-data evaluation of
stage 2: the stencil steps and the polish's acceptance. Every stencil and
every bootstrap replicate passes `start = θ̂`; a continued fit is labelled
against its start's components (minimum Bhattacharyya assignment, A6),
never re-sorted.

---

## 6. Intervals: normal and ABC

(`core/abc.py`, `ivq.bias_and_acceleration`; A10, built and accepted on
the Pareto shape against a B = 20 000 BCa reference.) The normal interval
uses √V_btw. The ABC interval (DiCiccio–Efron; Efron–Tibshirani 1993 ch.
22) uses σ̂ = √V_btw; acceleration a_c = (1/(6√N)) Σ_k p_k U_kc³ /
(Σ_k p_k U_kc²)^{3/2}; bias b̂_c = (1/2N) Σ_k p_k D²_kc from the initial
stencils' curvatures; curvature c_q from two evaluations on the SURVEY
rows along each output's receptive-field-level influence direction,
normalized by N (a build normalized by M_𝒳 was √(N/M_𝒳) too large; the
check that caught it, c_q against the delta-method coefficient on the
Pareto shape, is the acceptance for any change). Endpoints computed on
demand by the comparison layer; λ = w/(1 − a w)². Failed curvature
sides: one-sided second difference, flagged; both sides: ABC NaN, normal
stands.

---

## 7. The estimators

(`estimators.py`: the paper's five, plain functions with analytic
influence, `prepare`, no restarts. `gmm.py`: `GMM2D`. `cloudfil.py`:
`P2Mixture`.)

**`GMM2D(K)`** (method_notes §5; A7, A9, A12, A16): a K-component 2-D
Gaussian mixture by penalized maximum likelihood (Chen–Tan penalty on
the covariances, strength 1/Σω), weighted EM with SQUAREM acceleration,
a Newton gate (the negative Hessian must be positive definite) and a
Newton polish to the declared or per-call η; NaN when the gate or the
polish fails (a failed evaluation, never retried). Outputs: K − 1
weights, K means, K covariances (59 at K = 10). Analytic influence by
Louis's identity plus the penalty's own weight-sensitivity; a fit at
`start` continues from those parameters without any search.
`takes_start = True`. Cold search (`search`): `multistart` (k-means
starts plus greedy residual insertion, screened then polished; the
default, restored on 28 September) | `anneal` (A16, deterministic
annealing EM with perturbations re-injected at every temperature; built,
failed its first form, amended, NOT accepted; build held). Search audit
(A16.7): every cold fit is compared with a continuation from the true
parameters (`search_gap`, `search_failed`).

**`P2Mixture`**: `GMM2D(K = 10)` plus seven derived outputs of the
component identified as P2 (nearest fitted component to P2's true
position): position x, y; ¼ log det Σ; log axis ratio; position angle
(½ atan2(2Σ_xy, Σ_xx − Σ_yy)); logit weight; log peak contrast (P2's
density at its own centre over every other component's density there).
Influence by the chain rule through a finite-difference Jacobian.
`measured` = the seven derived indices; only those get GPs, partitions
and intervals; stencils still return all 66 outputs.

---

## 8. The demo dataset and the study

(`datasets.cloudfil_G_B6_P3_v1`, `data/`; A11.) K = 10: a Gaussian cloud
(weight 0.585, semi-axes 2.0 × 1.2 at 30°); six filament beads along
x = 2.2t, y = 0.55 sin(πt/1.2), t ∈ {±1, ±0.6, ±0.2}, 0.38/6 each, 0.35
along × 0.08 across; three protostars of semi-axes 0.05 × 0.025: P1 at
(0, 0) weight 0.02 at 20°, P2 at (1.32, 0.55) weight 0.01 at −50° (ON the
t = 0.6 bead), P3 at (−1.2, −1.1) weight 0.005 at 70°. N = 10 000 (about
100 points in P2). Draw s uses seed = master + s. The subject is P2; the
talk's point is the influence of a 1 %-mass source embedded in a
nuisance structure.

**Study settings (A11, amended A15):** S = 500 draws; bootstrap B = 5 000
warm-started replicates stored as a stream; the oracle at 10 000 cold
fits with the search audit, run first; `ivqbins = marginal`, `survey =
moments`, `quantized_start = full-data`, `gptrend = quadratic`,
`gpwidth = local`, `refine_schedule = rounds`, `refine_update = stencils`
(A17, replacing `refine_trigger = measured`), ε = 0.01, ABC beside the
normal interval; `ijfd` (section 11) at S = 1 in forward and central
forms for the cost slide. Hardware: TACC, about 100 CPUs per node.

**The rehearsal (local, 14 workers, 27 September) and its gate.** Every
method ran end to end at N = 10 000 on draws 0 and 1. Cold-fit failures 3
of 20 oracle draws; V_btw/V_ij 0.57–0.93; the angle at its guard. The
TACC gate (A15, unchanged by A17 except for the switch): the rehearsal
rerun with the demo settings must give V_btw/V_ij ≥ 0.97 on every
measured output on both draws, no output at its guard, queue/rounds
within ε, degenerate bootstrap replicates explained, the failed oracle
draws attributed. It has not been passed.

---

## 9. The codebase

```
qij_joint/
  pyproject.toml
  scripts/run.py           CLI: python scripts/run.py <dataset> <estimator> <method>
                             --N --draws a:b --out RUNS [--workers] [--seed] [--B]
                             [--eps] [--gptrend] [--gpwidth] [--M-X] [--ivqbins]
                             [--survey] [--quantized-start] [--refine-schedule]
                             [--refine-trigger → refine-update after A17]
                             [--point-curvature] [--diag-draws]
  scripts/make_truth.py    writes data/truth/<dataset>.json
  src/qij_joint/
    registry.py            (dataset, estimator) → Case: draw function, estimator
                           factory, vq_transform. Cases: pareto/shape, pareto/tail,
                           mvt/nu, mvt/tail, fp/all, imf/all, mix11/all, cloudfil/p2.
    datasets.py            the draw functions (seeded)
    estimators.py          the paper's five estimators (analytic influence)
    gmm.py                 GMM2D (section 7)
    cloudfil.py            P2Mixture (section 7)
    qij.py                 QIJ.fit: the whole method on one draw → QIJResult
    result.py              QIJResult dataclass (every product field)
    bootstrap.py           warm-started bootstrap, percentile and BC intervals
    ijfd.py                the finite-difference IJ (section 11)
    pipeline.py            run_oracle / run_ij / run_boot / run_qij / run_ijfd:
                           draws → parquet rows and arrays
    products.py            product layout: <out>/<dataset>_<estimator>_N<N>/<method>/
                           s00000.parquet (scalars), s00000.<kind>.parquet (arrays),
                           runs.log (JSON lines)
    parallel.py            Pool over processes, X shared once, call_T, prepared()
    check.py               the package's ONE check (weighted-mean scale identity)
    core/counter.py        Counter (section 1)
    core/differences.py    weight constructor, steps, central stencil (section 1)
    core/eta.py            η_full (section 5)
    core/xvq.py            𝒳-VQ, survey rows, prototype survey (section 2)
    core/influence_model.py  the GP (section 3)
    core/ivq.py            initial bins, stencils, V_btw, ABC ingredients (§4, §6)
    core/refine.py         proposals, splits, queue, finalize (section 4)
    core/rounds.py         the rounds schedule (section 4)
    core/joint.py          the joint path (not used on the demo)
    core/abc.py            the ABC curvature stage (section 6)
    data/                  cloudfil_G_B6_P3_v1.npz (+ .txt, generating script), truth/
```

`QIJ.fit` order: θ̂ (a pool task started first) → 𝒳-VQ → survey (θ_Q, I_j,
η_Q, step ratios) → influence model → ψ̂₀, σ → ABC curvature (survey
rows) → θ̂ collected, η_full → stage 2 (per output: initial bins,
refinement; or the joint path) → products. Evaluations are counted per
stage (`evals_prototype`, `evals_full_data`, `evals_refinement`,
`evals_curvature`), with rows and wall/busy time.

Git: repository at `qij_joint/`, HEAD 8153393 "A9–A15 as built"; the
working tree holds uncommitted changes to `refine.py`, `rounds.py`,
`gmm.py`, `ijfd.py`, `pipeline.py`, `qij.py`, `registry.py`,
`result.py`, `bootstrap.py`, `cloudfil.py` (A15 rounds 3–4, A16, the
multistart restoration) and deletes `seeding.py`. The audit baseline is
the stored products of `qij_joint@589c803` for draws s ∈ {0, 1, 999} of
every paper case plus MVT ν's failed draws {342, 420, 523}: with every
switch at its first value, every kept field must be equal bit for bit
(NaN == NaN), timing excepted. Approved floating-point reorderings (E8)
are audited at 10⁻¹² relative instead.

**Code standards (the author's, `qij_joint_plan.md` §5, binding on every
build):** a line budget per module; one way to do each thing, no option
without a current use, no compatibility shims (R2); no dead or
commented-out code (R3); spec notation in names (R4); functions fit on a
screen (R5); short docstrings with shapes and the formula or spec
section (R6); try/except only at a declared failure boundary (R7).
Comments describe the code as it stands, never its history (C1–C3). No
Python loop over N (E1); grouped sums by bincount (E2); factorize, never
invert (E3); no N × M or N × N array ever formed (E4); compute once,
pass down, no module-level caches (E5); never evaluate T twice at the
same weights (E6); preallocate (E7); a floating-point reordering needs
the author's approval (E8); a claimed saving is a timed one (E9).
Testing rules: no tests, no assertions, no smoke runs after an edit; the
audit is the correctness check and the measurement the performance
check; besides them only pyflakes and `python -m qij_joint.check`.

**Products that matter for review.** Per qij draw: `V_btw`, `V_win_hat`,
`V_tot_hat`, `L`, `n_refine_evals`, `rho`, `gain_ratio`, `a`, `b_hat`,
`c_q`, `ell`/`c`, `lam`, `eta_Q`, `eta_full`, `a_c` per output; `bin_U`
(final bins' all-output derivatives), `bin_label`, `psi0`, `psi_hat`,
`prototype_I/p/w`, `bmu`, `step_ratio` when `--diag-draws` covers the
draw. The `ij` method stores V_ij per output and the analytic ψ (N × 66)
per draw; the oracle stores θ̂ per draw against θ_true; the bootstrap
stores the replicate stream.

---

## 10. The exact offline simulator (planner, 28 September)

Path: `/private/tmp/claude-501/-Users-jtaylor-Dropbox-Software-JT-Py-Pkgs-vqboot-fable/dc605e35-cf93-477e-a378-464032d7e1c0/scratchpad/stage2_review/simulator.py`
(to be copied into the repository's `scripts/` if kept). It replaces T by
its linearization T(ω) = θ̂ + (ω − 1)Ψ/N, with Ψ the stored analytic
influence (`ij/s0000{s}.psi.parquet`, N × 66) and θ̂ the oracle's fit
(`oracle/s0000{s}.parquet`), rebuilds the 𝒳-VQ at the draw's seed
(bit-identical to the stored prototypes), reads the stored survey and
model from the coder's diagnostic pickle
(`…/c0fd00f7-…/scratchpad/vbtw/draws.pkl`), and runs
`run_refinement_rounds` / `run_refinement` unchanged. It reproduces the
stored rehearsal draws 0 and 1 to three decimals in V_btw and exactly in
every per-output evaluation count. Every stencil is then free and exact,
so stage-2 policies and stage-1 inputs can be compared at zero cost.
Every stage-2 number in this document comes from it. A17's acceptance
runs on it before any estimator is evaluated.

Companion scripts in the same folder (each self-contained, numpy):
`baseline.py` (where the variance lives), `iproto_quality.py` and
`check2.py` (survey against true receptive-field means, both draws),
`quantized_vs_analytic.py` and `decompose.py` (survey error decomposed),
`native_p2.py` (native rows for P2's receptive fields), `gp_variants.py`
(global/local × survey/exact), `partitions.py`, `within_field.py`
(within-receptive-field split efficiency by rule), `adaptive_fields.py`
(unfinished: produced no output).

---

## 11. The other methods in the comparison

`oracle`: cold fit per draw, θ̂ against θ_true over many draws (the
sampling variance). `ij`: the analytic influence and V_ij per draw (the
target QIJ approximates; exists only because these estimators have
closed-form influence). `boot`: B warm-started replicates per draw;
percentile and BC intervals; BCa's jackknife acceleration stated as
arithmetic, not run. `ijfd` (A13): the IJ by finite differences, one
continued fit per observation (N + 1 evaluations forward; 2N + 1 with
`point_curvature`, the central form that also yields exact point-level
b̂ and ABC); the cost reference, S = 1 on TACC.

---

## 12. Open items, in order

1. A17 build and its three-step acceptance (section 4).
2. The rehearsal rerun as the TACC gate (section 8).
3. A16 annealing: rebuild with re-injected perturbations, acceptance on
   draws 0, 1, 3, 6, 7; until then `search = multistart`.
4. The 𝒳-VQ's allocation inside P2 (15–35 receptive fields for 100
   points) bounds what any receptive-field-level information can
   deliver; never specified; measured only as a floor.
5. The prototype survey's bias on this estimand (section 2) is
   documented and, under A17, corrected by measurement rather than
   removed; the paper's assumption A2 should be stated in the talk as
   the method's boundary.
