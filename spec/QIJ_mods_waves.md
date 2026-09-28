# Mods list, waves A and B: specification for the coding manager

Planner, 26 September 2026. Governs the implementation of mods-list items
1a, 1b, 5, 6, 9 in `qij_joint`, after the port of the paper method has
passed its audit (`qij_joint_plan.md` §3). Code standards: `qij_joint_plan.md`
§5 verbatim. Terms: `QIJ_glossary.md`. Where this document is silent, the
ported method's spec (revision 10, `qij/spec/QIJ_method_spec.md`) holds.
Rulings are the author's unless marked (P), a planner default the author
may overrule.

**The mods list** (the author's numbering; this document implements the
items marked ⇒ and refers to the rest by number):

| # | item | status |
|---|---|---|
| 1a | quadratic GP trend | ⇒ wave A |
| 1b | GP correlation length = one factor × local CONN spacing | built in wave A; ruled NOT adopted 26 Sept (FP and MVT tables), `global` is the default, option kept |
| 1c | GP in latent coordinates from the survey | deferred |
| 2 | gradient survey by prototype-position perturbation | deferred |
| 3 | within-field share of ψ̂₀ on second-order CADJ cells | dropped |
| 4 | prototype-count rule: the ported q-dependent rule, unchanged; `M_X` overrides | ruled, A4 |
| 5 | joint 𝓘-VQ grown to the tolerance | ⇒ wave B |
| 6 | refinement replaced by a measured check | ⇒ wave B |
| 7 | store the q × q between-bin covariance | closed: recomputable from `bin_mass` and `bin_U`, not stored |
| 8 | intrinsic dimension of the influence cloud (two-NN over CONN) | deferred |
| 9 | explicit marginal / joint switch | ⇒ wave B |
| 10 | mixture estimator: component labelling by assignment to a reference fit | ⇒ wave A (A6) |
| 11 | the `ij` method stores the q × q oracle covariance | coordinator, outside these waves |
| 12 | parallel structure | coordinator's plan |
| 13 | `survey=moments`: receptive fields as d_x + 1 moment-matching rows | built, A8 |
| 14 | ABC interval: skew and bias correction from derivatives, no resampling | ⇒ A10 |
| 15 | survey reproducibility: stencils as continuations, η measured on the survey rows, step-doubling self-check | ⇒ A9 (prerequisite for everything on the demo) |
| 16 | the demo dataset `cloudfil_G_B6_P3_v1`, its six estimands as wrapper outputs, the TACC study settings | ⇒ A11 |
| 17 | mixture estimator initialization: peak-seeded starts beside the k-means starts | ⇒ A12 |
| 18 | `ijfd`, the finite-difference infinitesimal jackknife as a fifth method (the cost reference, S = 1) | ⇒ A13 |
| 19 | refinement in synchronized rounds across the measured outputs (`refine_schedule`), for 100-CPU nodes | ⇒ A14 |
| 20 | the consolidated repair round: measured trigger in the marginal refinement, zero-level closing, local width for the demo, measured η_full, failed-curvature rule | ⇒ A15 |
| 21 | deterministic annealing as the mixture estimator's search, replacing the start apparatus; the per-draw search audit against the truth | ⇒ A16 |

**Switches.** Three, each a plain argument of the fit with exactly two
values. With every switch at its first value the package reproduces the
ported method bit for bit (timing excepted), and that is the audit for both
waves.

| switch | values | wave |
|---|---|---|
| `gptrend` | `affine` (ported) / `quadratic` | A |
| `gpwidth` | `global` (ported) / `local` | A |
| `ivqbins` | `marginal` (ported) / `joint` | B |

**Testing rules (binding on every agent prompt for these waves).**

1. No tests, no test files, no assertions, no smoke runs after an edit
   (`qij_joint_plan.md` §4). The audit is the only correctness check, and
   the measurement the only performance check. Besides those two, an agent
   may run pyflakes and the package's one check (`python -m qij_joint.check`),
   and nothing else.
2. **The audit** compares the kept fields, bit-exact, against the stored
   products of `qij_joint@589c803` for the audit draws: s ∈ {0, 1, 999} for
   every case, plus the main study's only failing draws, MVT ν at s = 342,
   420, 523. Those stored products were themselves shown equal to the frozen
   `qij@025613e` path, so no old-package run is needed. Their bootstrap ran
   at B = 2000 from one sequential stream, so a B = 20 audit run compares to
   the first 20 replicates. No study loop, no S, no sweep over N, and no
   bootstrap beyond what the audit draws require.
3. **A measurement** is one draw (seed 0) of one estimator at the N values
   named in A4 or B7, with `M_X` given explicitly, no bootstrap, no study
   loop, no repetition over seeds, no N or B beyond those named. Before the
   measurement the agent times ONE estimator call at each N and projects the
   total from the evaluation counts in this document; if the projection
   exceeds 20 minutes of wall time for the whole measurement, the agent
   stops, reports the projection, and waits. Nothing is run "to see what
   happens".
4. The paper's estimands (Pareto, MVT, FP, IMF) are not run in these waves
   except inside the audit draws of rule 2. In particular no IMF or FP
   bootstrap at any B.
5. Nothing under `QIJ_WSOM2026/runs` is read for writing or written to;
   measurement outputs go to the scratch directory the coordinator names.
6. Any evaluation count the agent needs for a comparison and that this
   document gives as arithmetic (for instance the marginal path's initial
   count 1 + 2·q·17) is computed, not run.

**Notation.** N points, d_x data dimensions, q outputs (index c), M_𝒳
prototypes (index j), bins (index k). z: whitened data coordinates (the
GP's input as ported). ψ̂₀: the N × q matrix of predicted influence at the
data points. p_k: bin mass. U_k ∈ ℝ^q: bin k's measured derivative, all
outputs from the same two evaluations. ε: the declared tolerance. CONN:
the symmetrized CADJ graph on live prototypes.

---

## Wave A: the regression (1a, 1b) and the survey's parallelism

### A1. `gptrend=quadratic` (item 1a)

The GP's mean basis h(z) becomes (1, z, {z_a z_b}_{a ≤ b}) in whitened
coordinates: 1 + d_x + d_x(d_x+1)/2 functions (6 in 2-D). Everything else in
the REML fit is as ported: the basis is projected out by the orthonormal
complement Q, one eigendecomposition of QᵀK Q per candidate width shared
across outputs, s_c and λ_c profiled per output. The ported constant-only
fallback (finite design with at most m + 1 points, m the basis size) applies
with the new m. Prediction and posterior variance use the same formulas with
the new basis.

### A2. `gpwidth=local` (item 1b)

**Local spacing.** For live prototype j, h_j = median over its CONN
neighbours k of ‖z_j − z_k‖. Every live prototype has at least one CONN
neighbour (each of its points has a second-best prototype), so h_j is
defined everywhere; with one neighbour the median is that distance. No
floor, no minimum degree, no fallback.

**Correlation length.** ℓ_j = c · h_j with one global factor c. A data
point i takes the length of its best-matching prototype, ℓ(z_i) = ℓ_{b(i)}.

**Kernel** (Paciorek–Schervish non-stationary Matérn-3/2). For two
locations with lengths ℓ, ℓ′ at whitened distance r, with
κ(t) = (1 + √3 t) exp(−√3 t):

    k(r; ℓ, ℓ′) = ( 2ℓℓ′ / (ℓ² + ℓ′²) )^{d_z/2} · κ( r · √2 / √(ℓ² + ℓ′²) ),

with d_z the dimension of the GP's input z (equal to d_x unless a
`vq_transform` changes it). When ℓ = ℓ′ this equals κ(r/ℓ), the ported
kernel, mathematically; `gpwidth=global` evaluates the ported expression
itself, since √2/√(2ℓ²) is not bit-equal to 1/ℓ and the audit is bit-exact.

**Search over c.** The ported search over ℓ becomes the same search over c:
five log-spaced candidates, then one bounded scalar search between the best
candidate's neighbours, skipped when the best candidate is the upper
endpoint. Bounds carried from the ported ones: c_min = ℓ_min / median_j h_j
(ℓ_min the ported lower bound, the median CONN spacing, so c_min ≈ 1);
c_max = ℓ_max / min_j h_j (ℓ_max the ported upper bound, ten times the
largest inter-prototype distance). REML, the λ floor, the group structure
across outputs: unchanged.

**Products added:** `c`, `c_bound` (whether c sits on a bound), and the
per-prototype `h_j` in the prototypes table. With `gpwidth=global` the
ported `ell`, `ell_bound` are written as before.

### A3. Survey parallelism

The prototype survey (1 + M_𝒳 evaluations on M_𝒳 rows: the base, then one
forward step per prototype) runs its M_𝒳 per-prototype evaluations through
the pool of `qij_joint/parallel.py` under the plan's contract. Each task is
(prototype index, its perturbed weights); it returns the q outputs, the
evaluation's own wall time and a failure flag; the parent assembles I_j in
prototype order. `workers == 1` is the ported loop. Results are identical
at any worker count.

### A4. The prototype count (mods item 4, ruled 26 September) and the `M_X` argument

The prototype count rule is the ported one, unchanged, for every estimator
and both values of `ivqbins`:

    M_𝒳 = ⌈ √( (1 + 2·q·M_ref) · N / 2 ) ⌉,   M_ref = 17,
    floored at 20, capped at ⌊N/2⌋,

188 for one output, 321 for the IMF and 371 for the Fundamental Plane at
N = 2000. It balances the survey's rows against a second stage of 17 bins
per output on the full data, and it scales with q for a resolution reason
as well: a q-output estimator has q influence surfaces to survey from the
same prototypes. The paper's runs are validated at these counts.

It promises no resolution of the data's structure below the prototype
spacing: a component too small to receive a prototype is found by the
measured check (B4) at full-data cost, and many flags in one region are the
diagnostic that says to raise the count. Its one misbehaviour is a
large-q estimator at small N, where it reaches the cap (mixture 11 at
q = 53, N = 2000 gives 1343, capped at 1000); that is what the override is
for.

An explicit argument `M_X` overrides the rule when given. Every product
records `M_X` and `M_X_source` (`rule` or `argument`).

The rule is the default for both values of `ivqbins`. Where it reaches the
cap, a large-q estimator at small N, the `M_X` argument is the override,
and that is the whole of the design: item 4 is closed.

### A6. Mixture estimator labelling (mods item 10)

`GMM2D` gains an optional `reference`: the means and covariances of a
reference fit. With a reference, every evaluation labels its fitted
components by the assignment to the reference components that minimizes
the total Bhattacharyya distance between Gaussians,

    D_B(a, b) = ⅛ (μ_a − μ_b)ᵀ Σ̄⁻¹ (μ_a − μ_b) + ½ ln( det Σ̄ / √(det Σ_a det Σ_b) ),
    Σ̄ = (Σ_a + Σ_b)/2,

solved as a K × K assignment (`scipy.optimize.linear_sum_assignment`).
Without a reference the ported ordering by the first mean coordinate
stands. The distance separates a spike from the blob sharing its mean
through the covariance term, and separates equal-covariance components
through the mean term, so neither a same-mean pair nor an equal-scale pair
is labelled by fitting noise.

**Use.** The reference is an option the caller chooses, not a default the
driver imposes; the label-switching of the naive sort under resampling is
part of what the mixture example demonstrates, so both labellings are run.
Without a reference (the naive sort), a bootstrap replicate can swap a
spike with the blob sharing its mean, since a resample moves a fitted mean
by about 1/√N while a QIJ perturbation moves it by about √η; the count of
such swaps is a reported quantity of the comparison, and QIJ's evaluations
within a draw never swap under either labelling. With a reference, the
driver fits once without one on the full data, takes that fit as the
reference, and constructs the estimator used for every other evaluation in
the draw with it, so output c is the same component in stage 1, stage 2 and
every bootstrap replicate. A multi-draw study must use a reference for every
method, and there the reference is the true components, since the naive
sort labels the same-mean pair inconsistently from draw to draw and coverage
for those coordinates is otherwise undefined. The reference is arrays, so
it crosses the pool's boundary under the contract.

### A7. Mixture estimator: the penalized-likelihood estimand (author's ruling, 26 September)

**Why.** The likelihood of a Gaussian mixture with free covariances is
unbounded (a component collapsing onto a few points), so its maximum does
not exist and "the local maximum the best start reaches" is defined by the
optimizer, not the model. `GMM2D`'s estimand is therefore the **penalized**
maximum-likelihood estimate of Chen and Tan (2009; verify the constant
against the paper before writing it down as theirs):

    ℓ_p(θ; X, ω) = ℓ(θ; X, ω) − a · Σ_k [ tr(S Σ_k⁻¹) + log det Σ_k ],
    a = 1 / Σ_i ω_i  (= 1/N at unit weights),
    S = the ω-weighted covariance of the rows passed to T.

The penalty is always on, for every caller and every fit; it is of relative
order 1/N, so it moves a well-behaved fit by less than its sampling error,
and it vanishes in the limit. A version that switched on only near
singularity would make T discontinuous in ω and is excluded. S is defined
from the rows and weights passed, so the quantized-data fit and the
full-data fit use the same definition.

**What changes in the code.**
- The M-step for Σ_k has the closed form of the inverse-Wishart MAP,
  Σ_k = (Σ_i r_ik ω_i (x_i − μ_k)(x_i − μ_k)ᵀ + 2a S) / (Σ_i r_ik ω_i + 2a);
  EM stays monotone in ℓ_p. Weights and means are unchanged.
- The Newton polish, the score and the observed information are those of
  ℓ_p: the penalty's gradient with respect to Σ_k is
  −a (Σ_k⁻¹ − Σ_k⁻¹ S Σ_k⁻¹), its Hessian is closed-form, both are added
  in Louis's identity.
- The analytic influence is that of the penalized M-estimator. Note that
  the penalty depends on ω through S (and through a): the derivative of the
  penalized score with respect to ω_i includes ∂S/∂ω_i and ∂a/∂ω_i, and
  these terms are part of the influence. The detector for a missing term is
  the same as before: Newton's convergence stops being quadratic.
- **EM acceleration (ruled 26 September): SQUAREM**, Varadhan and Roland's
  squared extrapolation, in its standard form: from θ take two EM steps
  giving θ₁, θ₂; r = θ₁ − θ, v = (θ₂ − θ₁) − r; step length
  α = −‖r‖/‖v‖, then **α ← min(α, −1)**: the step is never SHORTER than
  the two EM steps (α = −1 reproduces θ₂ exactly), and it may be much
  longer. (Corrected 26 September: an earlier line here had the bound the
  wrong way round, clipping every long step back to plain EM, which is why
  the first build saved nothing.) Long steps are limited by the paper's
  step-length control: α ← max(α, −m) with m starting at 4 and multiplied
  by 4 each time the limit binds. Candidate θ′ = θ − 2αr + α²v followed by
  one EM step. Safeguard: if θ′ leaves the feasible set (a weight ≤ 0, a
  covariance not positive definite) or ℓ_p(θ′) < ℓ_p(θ₂), discard it and
  take θ₂ (and, per the paper, halve the distance toward α = −1 before
  giving up is NOT done: one candidate, then fall back). So every accepted
  iterate has ℓ_p at least that of plain EM: monotone, deterministic.
  Convergence test as now, on the relative change of ℓ_p; the iteration
  cap counts EM steps.
- **Newton gating (ruled 26 September).** After EM (accelerated) has met
  its tolerance or its cap, form the observed information H of ℓ_p. If −H
  is positive definite (a Cholesky succeeds), run the damped Newton polish
  as built (full step; halve while a weight ≤ 0, a covariance is not PD, or
  ℓ_p falls; at most 20 iterations) until the score norm is ≤ η. If −H is
  not positive definite, the iterate is not near a maximum: continue
  accelerated EM for one more block (the same cap again) and re-test, once.
  If −H is still not positive definite, or Newton does not reach η, the fit
  is NaN under the acceptance rule. No third block, no other retry.
- **Declared η.** The coder measures the score norm the polish reaches on
  the demo mixture over the multi-start (one fit, all starts that pass the
  gate) and declares `eta` at the level reliably reached, rounded up to a
  power of ten; the author is told the number. It is not fixed at 1e-12 in
  advance.
- Acceptance: the norm of the penalized score at the returned point is at
  most η, with η DECLARED as the accuracy the polish reliably reaches
  (recorded in the estimator's `eta`), NaN otherwise. The same rule for
  every caller. Nothing is retried.

**Not part of the estimator.** Warm starts from a reference fit are NOT
part of `GMM2D`'s definition: every call runs the full multi-start fit, so
T is a function of (X, ω) alone and nothing in the package may depend on a
warm start being available (author's ruling). The reference labelling of
A6 stands as the labelling mechanism.

**The demo mixture (author's ruling, 27 September; supersedes the 26
September N = 2000 weights).** The talk's estimand is Chacon mixture 11 at
**N = 5000** with its means and covariances as loaded from `structsynhd`
and these weights ("set B", loaded component order). The three spike
levels put about 25, 75 and 200 points per spike, so the bootstrap's
rare-support behaviour can be shown across its transition while the
penalized estimator still exists at the rarest level (verified: converges,
recovery max |z| = 3.27 over 53 outputs); the original weights, about 14
points per spike, do not fit at all:

| k | component | mean | weight | expected points at N = 5000 |
|---|---|---|---|---|
| 0 | blob | (−1.5, 0) | 0.43 | 2150 |
| 1 | blob | (+1.5, 0) | 0.43 | 2150 |
| 2 | outer spike | (−2.5, −1) | 0.005 | 25 |
| 3 | on-mean spike | (−1.5, 0) | 0.015 | 75 |
| 4 | inner spike | (−0.5, +1) | 0.04 | 200 |
| 5 | middle | (0, 0) | 0.02 | 100 |
| 6 | inner spike | (+0.5, −1) | 0.04 | 200 |
| 7 | on-mean spike | (+1.5, 0) | 0.015 | 75 |
| 8 | outer spike | (+2.5, +1) | 0.005 | 25 |

Sum 1.0000. The coder decides the plumbing (a dataset entry that loads
mixture 11 and replaces the weights; the same generator otherwise, so
`expand_dimension` still applies). Nothing else in this section is
specific to that mixture.

### A8. The survey's representation of a receptive field: `survey` switch (author's ruling, 27 September)

**Why.** The survey fits T to the prototypes with their masses as weights. A
component of the data covered by one or two prototypes has no within-cell
spread in that data set, so the quantized problem is no longer near the
full-data one: on the demo mixture (weight set B, N = 5000, M_𝒳 = 600) the
outer spikes, 27 and 23 points, got one and two prototypes, the quantized
mixture fit split a blob instead and landed on a saddle, and 576 of 601
survey fits were NaN. Nothing in the estimator is wrong; the quantized data
set is not the data where the data are rare.

**The switch.** `survey` with two values: `points` (ported: one row per
receptive field, the prototype, with weight M_𝒳 p_j) and `moments`.

Under `moments`, receptive field j is represented by

    its own n_j points, each with weight proportional to p_j / n_j,  if n_j ≤ d_x + 1;
    otherwise the d_x + 1 vertices of a regular simplex, centred at the
    field's mean, scaled so the vertices' second moment is the identity,
    mapped through the Cholesky factor of the field's covariance, each with
    weight proportional to p_j / (d_x + 1).

Either representation reproduces the field's mean and covariance exactly
(the first also every higher moment); d_x + 1 is the smallest number of
points that can carry a rank-d_x covariance, so no constant and no
regularization enter. **Normalization (corrected 28 September):** the
survey rows' weights sum to N, field j's rows carrying n_j in total, since
the rows stand for the N points. An estimator whose value depends on the
weight total is then the same estimator on the rows as on the data: the
mixture's penalty is a = 1/N on both. The 27 September rule (weights
summing to the row count R ≈ 3 300) made the penalty 1/R on the survey
rows, about nine times stronger relative to the likelihood than on the
data; the quantized fit then pulled the compact component P2's covariance
toward the global one, which was the demo's survey error (P2's angle
r² 0.23 against the true receptive-field means; 0.95 with weights summing
to N). (The first build summed them to
M_𝒳 and the mixture estimator's Newton step, which normalizes its score by
Σω and its information by the row count, came out three times too long.)
**Estimator-side ruling:** `GMM2D` is to normalize its information by Σω
as it does its score, so that T is invariant to a common rescaling of the
weights for any caller, contract or not; this is the weighted-mean scale
identity the package's one check expresses, and it changes no audited
estimand. The simplex's orientation is a rotation that leaves
the moments unchanged; fix it by a deterministic convention (first vertex
along the field's leading eigenvector). The rows of field j are perturbed
together: raising prototype j's mass scales all of its rows' weights by the
same factor, so the prototype influence keeps its definition as the
derivative with respect to the field's mass, θ_Q keeps its definition as the
base fit on the survey rows, and the GP still regresses on the field means
w_j. CADJ, the 𝒳-VQ and everything downstream are untouched. Rows per
survey evaluation become Σ_j min(n_j, d_x + 1) ≤ (d_x + 1) M_𝒳; the
evaluation count is unchanged.

`points` remains the package default; the audit stands. `moments` is the
setting the demo passes.

**Probe before the build is used (testing rule 3 applies).** Mixture 11,
weight set B, N = 5000, seed 0, M_𝒳 = 600, `survey=moments`, `gptrend=
quadratic`, `gpwidth=global`: (i) the base survey fit θ_Q: does it converge
(Newton gate on which test, polished score norm), and its parameters
against the full-data fit θ̂ as (θ_Q − θ̂)/√V_ij; (ii) the survey: the
number of the 601 fits that are finite, and the number of outputs on the
constant path; (iii) timings: one converged fit at 600 rows and at the
`moments` row count, and the survey's wall time on the workers used.
Nothing else. If (ii) is still failing, the fallback is M_𝒳 = 2500 with
`points`, priced beforehand from (iii).

### A9. Survey reproducibility: stencils as continuations (author's ruling, 27 September)

**The defect, measured.** On the demo mixture (set B, N = 5000, M_𝒳 = 600,
`survey=moments`) every surveyed prototype influence was noise: the
reconstructed raw responses T(ω(t_j)) − θ_Q had median magnitude 3 × 10⁻⁵
and reached 1.0, against a step t_j ≈ 3 × 10⁻⁹ and a true response of
about 10⁻¹⁰; 31% of responses were jumps above 10⁻³; corr(ψ̂₀, ψ) ≈ 0 on
all 53 outputs. Cause: each survey evaluation ran a fresh 20-start fit, and
the winner, or its convergence point, differed between the base and the
perturbed weights by far more than the perturbation's effect. A derivative
of "the best of 20 starts" is undefined where the winner changes, and on
the quantized rows it changes everywhere. The declared η = 10⁻¹² described
the full-data polish, not the quantized fit's reproducibility. Assumption
A1 failed for T on the survey rows. The paper's estimators are
deterministic optimizers and did not have this problem.

**The rule.** A finite difference of T is taken between two fits on ONE
branch: the perturbed fit is the continuation of the base fit under the
weight change.

1. `GMM2D` (and any estimator with restarts) accepts an optional `start`:
   initial parameters. With `start` given the fit runs EM (accelerated)
   from those parameters with the given weights, then the polish and the
   acceptance rule as built; no multi-start. Without `start` the behaviour
   is unchanged (bit-identical for every audited path).
   **Labels of a continuation (ruled 28 September).** A continued fit is
   labelled by the minimum-Bhattacharyya assignment to the START's
   components (A6's machinery), never by the canonical sort: a derivative
   is taken along one branch and its labels must follow the branch. As
   first built, `_fit` re-sorted every output by the first mean
   coordinate, and a co-located pair with Δμ_x ≈ 0.01 (P2 and its bead,
   the cloud and its blend) swapped between θ̂ and θ_Q, which made the raw
   parameters of those components anti-correlate with the analytic
   influence (−0.99) while the derived P2 outputs, which re-identify P2
   per call, were unaffected. Verified on the demo: flagged raw outputs
   fell from 24 to 8, every cloud and P2-raw failure fixed; what remains is
   the blended θ̂-component 4 and p2_pa (small-sample noise, see A9's
   acceptance record). Mixture 11's mu6x anomaly (−0.90) is NOT closed by
   this: that estimator already labelled continuations by assignment, and
   the rerun is bit-identical; it stays open and separate. Cold fits keep
   the canonical sort, or the reference when given. Applies to every use of `start`: the survey (to θ_Q), stencils
   and bootstrap replicates (to θ̂), the population continuation (to the
   truth), ijfd's per-point fits.
2. The survey passes `start = θ_Q` to every perturbed prototype
   evaluation; the full-data stencils (bin measurement, check splits) pass
   `start = θ̂`; the curvature evaluations of A10 pass `start = θ_Q`. The
   base fits θ̂ (full data) and θ_Q (quantized rows) keep the multi-start.
   The bootstrap does not use `start`; it is not a derivative.
3. **Declared accuracy on the survey rows.** Before the survey, the
   reproducibility of T on the survey rows is measured: fit twice from
   θ_Q at the base weights, and once from θ_Q at weights perturbed by a
   relative 10⁻⁶ on the largest-mass prototype and back; η_Q = the largest
   relative difference among the returned parameters, floored at the
   polish residual. The survey's step uses η_Q in the ported rule
   δ_f = 2√η_Q. η_Q is a product. (P) If η_Q exceeds 10⁻⁶ the survey is
   still run and the value is reported; no gate.
4. **Self-check, reported not gated.** On the five largest-mass prototypes
   the survey also evaluates at step 2δ_f; the ratio of the two responses,
   per output, is stored as `survey_step_ratio` (expected ≈ 2 for a
   differentiable T; far from 2 means the response is not a derivative).
   Ten extra evaluations of survey size.
5. **The quantized base fit's start (pending the planner's pre-flight).**
   An option `quantized_start` with values `multistart` (default, as built)
   and `full-data` (θ_Q is the continuation of θ̂ onto the quantized rows).
   The pre-flight on the filament mixture decides which the demo uses;
   both stay available.

**Acceptance for the build.** On the Fundamental Plane at N = 2000 (audit
draw s = 0) nothing changes bit for bit (the FP estimator has no
restarts). On the demo mixture, one draw, the A2 comparison: the surveyed
I_j against the analytic influence's mass-centred receptive-field means,
per output; corr above 0.9 on every output whose component holds at least
five prototypes; `survey_step_ratio` within 10% of 2 on the tested
prototypes; and the fraction of NaN survey fits reported. One survey, no
bootstrap, no second stage.

### A10. The ABC interval (mods item 14: skew and bias correction; author's ruling, 27 September)

**What it is.** DiCiccio and Efron's approximate bootstrap confidence
interval (Efron and Tibshirani 1993, chapter 22, the nonparametric ABC and
its quadratic form ABC_q; DiCiccio and Efron 1992): the BCa interval's
second-order accuracy obtained from derivatives of T in the weight space
instead of from resampling. The coder takes the formulas from the book,
not from this document, and names the equations used in the code comments.
QIJ supplies the ingredients:

| ingredient | ABC's definition | QIJ's source |
|---|---|---|
| σ̂_c | root of the influence energy | √V_btw,c |
| a_c (acceleration) | (1/6) Σ_i ψ_i³ / (Σ_i ψ_i²)^{3/2} | with the bin-level influence: (1/(6√N)) Σ_k p_k U_kc³ / (Σ_k p_k U_kc²)^{3/2} |
| b_c (second-order bias) | Σ_i T̈_i / (2N²) | **(ruled 28 September)** b = (1/2N) Σ_k p_k D²_k with D²_k = [T(ω_k⁺) − 2T(ω⁰) + T(ω_k⁻)] / t_k², the second derivative of T in the RELATIVE step along the stencil's own direction. Derivation: along that direction the bin masses move as q_k = p_k + t(1 − p_k), q_l = p_l(1 − t); with the scale identity (Σ_l p_l H_kl = 0, pᵀHp = 0) the second derivative in t equals H_kk exactly, and the plug-in bias ½ tr(H · Cov(p̂)) with the multinomial Cov(p̂) = (diag p − ppᵀ)/N is (1/2N) Σ_k p_k H_kk. At point level (ijfd) this is the book's Σ T̈_i/(2N²). The paper's B̂ used the RAW second difference (≈ t², i.e. ≈ 4η ≈ 10⁻¹⁶) and a (1 − p_k) factor; it was reported and never applied, so no interval depends on it, but the product was wrong. Δ²T_k is stored again. |
| c_c (curvature along the influence) | second difference of T along the direction of ψ_c, scaled by σ̂ | two evaluations on the SURVEY rows (not the full data), along the field-level influence I_j^(c): weights (1 ± ε u_j) on every row of field j; `start = θ_Q`; ε = min(η_Q^{1/4}, ½ / max_j|u_j|). **Normalization (ruled 28 September):** the resampling-vector scale in ABC's curvature is the DATA size N, not M_𝒳; built with M_𝒳 it came out √(N/M_𝒳) too large on the Pareto shape (0.0724 against the delta-method value σ̂/α̂ = 0.0222). The check that fixed it, c_q against the delta-method quadratic coefficient on an estimator where that is closed form, is the acceptance for any future change to c. |
| z₀, endpoints | ABC's closed forms in a, b, c, σ̂ | computed on demand like the normal interval; never stored |

**A failed curvature evaluation (ruled 28 September).** Nothing is
retried. If one side of the ± ε pair returns NaN, c is the one-sided
second difference on the surviving side with ONE additional evaluation at
2ε on that side, [T(1 ∓ 2εu) − 2T(1 ∓ εu) + T(1)]/ε², flagged in the
products as one-sided (O(ε) instead of O(ε²), acceptable for a
second-order term); if both sides fail, c is NaN, the ABC interval for
that output is NaN, the normal interval stands, and the failure is
counted. Non-finite prototype rows are excluded from the direction and
the scale per output, with the mass renormalized over the finite rows,
the convention the influence model already uses.

The quantized rows are used for c because it is a second-order correction:
an error of a few percent in c moves an endpoint by a few percent of a
1/√N term, below what the interval resolves, and the field-level direction
differs from ψ̂ by the within-field variation, a few percent of its norm.
a and b are likewise bin-level approximations of point sums, the same
approximation V_btw makes. Cost: 2q evaluations of survey size per draw.

**Products.** Per output: `a`, `b_hat`, `c_q`, `eta_Q` (shared), and the
endpoints of the ABC interval at the study's levels alongside the normal
interval's, computed by the comparison layer from the stored ingredients.
The bootstrap method gains the bias-corrected percentile interval (BC),
using z₀ from the replicates, at no evaluation cost; BCa with a jackknife
acceleration is NOT run (N refits) and its cost is stated as arithmetic.

**Acceptance for the build (restated 28 September).** One draw of the
Pareto shape at N = 2000 (a cheap estimator with an analytic influence):
the BCa reference computed ONCE at B = 20 000 with the ANALYTIC influence
as the acceleration and each endpoint's Monte Carlo sd estimated by
resampling the replicates; ABC passes when each 0.95 endpoint lies within
2 of those sds of the BCa endpoint. (The first form, 0.6/√B of the width at
B = 2000, held the reference to a tolerance below its own noise.) Also:
c_q equals the delta-method quadratic coefficient σ̂/α̂ to 1% on that draw,
for the qij path AND for ijfd's point-level c; b̂ near the Pareto MLE's
known bias α/n; the step-doubling check on c (ε and 2ε give the same c to
1%). Results of the first rerun, for the record: c_q = 0.02220 (delta
method 0.0222), b̂ = 0.00098 (α/n = 0.0010), ABC [1.9173, 2.0910] against
BCa at B = 2000 [1.9142, 2.0920]; the endpoint formula's λ was corrected
to the book's w/(1 − a·w)². ijfd's c_q was 10³–10⁴ off and must pass the
delta-method check before its demo runs. No other run.

### A11. The demo dataset `cloudfil_G_B6_P3_v1` and its estimands (author's rulings, 27–28 September)

**Supersedes mixture 11 as the talk's estimand.** Naming: `cloudfil` the
family (cloud + filament); `G` a Gaussian cloud (`T` for a multivariate-t
cloud, later); `B6` a filament of six Gaussian beads (`U` for a tube,
later); `P3` three protostars; `v1` this parameter set. Parameters live in
the file, never in the name; a different mass ratio or core weights is
`v2`.

**The file.** `cloudfil_G_B6_P3_v1.npz` (keys `weights`, `means`, `covs`,
`role`, `name`, `N_design`, `subject`) with a readable `.txt` twin and the
generating script `make_cloudfil_G_B6_P3_v1.py`, all in the planner's
scratch at
`/private/tmp/claude-501/-Users-jtaylor-Dropbox-Software-JT-Py-Pkgs-vqboot-fable/dc605e35-cf93-477e-a378-464032d7e1c0/scratchpad/cloudfil_v1/`;
the coder copies all three into the package's `data/` and registers the
dataset. K = 10 components, weights summing to one:

| k | role | weight | mean | semi-axes | angle |
|---|---|---|---|---|---|
| 0 | cloud | 0.585 | (0, 0) | 2.0 × 1.2 | 30° |
| 1–6 | filament beads at t = −1, −0.6, −0.2, 0.2, 0.6, 1 on x = 2.2t, y = 0.55 sin(πt/1.2) | 0.38/6 each | on the curve | 0.35 along the tangent × 0.08 across | local tangent |
| 7 | P1 embedded | 0.02 | (0, 0) | 0.05 × 0.025 | 20° |
| 8 | P2 on-filament | 0.01 | (1.32, 0.55) | 0.05 × 0.025 | −50° |
| 9 | P3 off-filament | 0.005 | (−1.2, −1.1) | 0.05 × 0.025 | 70° |

Design N = 10 000; expected points 200 / 100 / 50 in P1 / P2 / P3. Sampler:
multinomial counts from the weights, then multivariate normals; draw s
uses seed = master + s as for every dataset. The "STARFORGE reading": a
point is a gas particle (or a photon); a source is a compact overdensity of
points; the cloud and the filament are nuisance.

**The estimator.** `GMM2D(K = 10)` as built (penalized likelihood, A9's
`start`, A12's seeding), with the reference labelling of A6: within a draw
every stencil and every bootstrap replicate labels against the full-data
fit; across draws (oracle, coverage) against the true components.

**The measured outputs: P2's six, on the scales below.** The estimator's
raw outputs are the 59 mixture parameters; the demo's estimands are
functions of them, exposed as ADDITIONAL outputs of T (a thin wrapper
estimator around `GMM2D`, `outputs` naming the six), with their analytic
influence by the chain rule, J_g · ψ_θ, where J_g is the 6 × 59 Jacobian of
the functions below at θ̂. J_g may be formed by central finite differences
of the deterministic functions in θ (they are cheap closed forms; the one
integral, if completeness is ever added, by fixed quadrature); the
perturbation check of the influence applies to the six as to any output.
With s the P2 component, λ₁ ≥ λ₂ its covariance's eigenvalues, v₁ the
major-axis eigenvector, b the cloud, F_j the filament beads:

| estimand | definition |
|---|---|
| position x, y | μ_s |
| log effective radius | ¼ log det Σ_s |
| log axis ratio | log(λ₂/λ₁) |
| position angle | ½ atan2(2Σ_s,xy, Σ_s,xx − Σ_s,yy), mod π |
| logit weight | log(π_s / (1 − π_s)) |
| log peak contrast | log π_s − ½ log det Σ_s − log Σ_{k≠s} π_k φ(μ_s; μ_k, Σ_k) + const, the source's density at its own centre over the sum of every other component's density there (cloud and all six beads); the constant −log(2π) cancels in the ratio |

Seven numbers if position counts as two; the slide shows six panels
(position once). The rule's q for the prototype count is the number of
MEASURED outputs, 6, giving M_𝒳 = ⌈√((1 + 2·6·17)·N/2)⌉ = 1013 at N = 10 000;
the fit must therefore take the measured subset as an argument and the
rule must read its length. Completeness, contamination and the
Bhattacharyya overlap are NOT measured in v1.

**Study settings for the TACC run.** N = 10 000; S = 500 draws; bootstrap
B = 5000 per draw with the replicate parameters STORED as one sequential
stream (so every B′ ≤ 5000 is available afterwards by truncation, and the
interval-accuracy-against-B curve is analysis, not a rerun); the bootstrap
replicates warm-started from the full-data fit (A9 rule 2 extended to the
bootstrap: `start = θ̂`; the coder's bootstrap method), with the fraction
of degenerate replicates reported; the bootstrap's percentile and BC
intervals, BCa with a jackknife acceleration stated as arithmetic only;
the oracle at 10 000 draws, cold fits with A12's seeding and the truth
labelling; `ivqbins = marginal` (the joint path is NOT used on this
estimand: on the true influence a joint partition at 1% needs point-level
bins, while the six marginal partitions need about 20 groups each);
`survey = moments`; `quantized_start = full-data` (ruled from the
pre-flight: the cold quantized fit failed in five of six settings, the
warm one sat within 0.1–2.4 standard errors in five of six);
`gptrend = quadratic`, `gpwidth = local` (changed from `global` in the
consolidated repair round, A15: the diagnosis named one global width
against a 0.05-wide core as the cause of the posterior's overconfidence
inside P2), `refine_trigger = measured` (A15), `refine_schedule = rounds`
(A14); ε = 0.01; the ABC interval (A10) beside the normal one.

### A12. Mixture estimator initialization: peak-seeded starts (author's ruling, 28 September)

**The finding.** On `cloudfil` (both overlaps) the 20 k-means starts never
found the truth's basin: the penalized log-likelihood at the true
parameters exceeds the winner's by about 190 nats; the winner collapsed
the cloud's slot to 1.5% weight and put the cloud's mass on a filament slot
and on P1's and P3's slots; only 1 of 20 starts reached even that winner;
96% of initial centres lay inside the cloud's 2σ contour, 0 of 20 starts
had a centre near P3 and 3 of 20 near P2. A search seeded by mass does not
see a fifty-point core among ten thousand points. This is the estimator's
problem, shared by every interval method, and it is fixed in the
estimator.

**The rule.** The multi-start's start set gains peak-seeded starts beside
the k-means starts, and the best final likelihood wins as now:

1. Over-segment: k-means on X (never the weights) with 10·K centres
   (k-means++ seeding, fixed seed as now).
2. Density at each centre: its count divided by the area of its Voronoi
   cell, approximated as π r_j² with r_j the median distance from the
   centre to its members (deterministic, no bandwidth).
3. Adjacency among centres from the second-best assignment of the points
   (the CADJ construction the package already has).
4. Peaks: centres denser than every adjacent centre. Prominence of a peak:
   its density minus the highest saddle density on any adjacency path to a
   denser peak (standard topographic prominence on the graph; the global
   maximum's prominence is its density).
5. The peak-seeded start: the K most prominent peaks as initial means; each
   component's initial covariance from the members of its own cell and its
   adjacent cells; initial weights from the same members' share. If fewer
   than K peaks exist, fill with the highest-count non-peak centres.
   **Amended 28 September (two resolutions).** As built at one resolution
   the seeding missed the basin: only six peaks exist among 100 cells, and
   P1 and P2 sit exactly on the cloud's and a bead's density peaks, one
   peak in position but two components in scale. Steps 1–4 are therefore
   run at TWO resolutions, 10·K cells and 50·K cells (about 20 points per
   cell at N = 10 000), each with its own densities, adjacency, peaks and
   prominence (single-member cells count as density 0). The candidates of
   both levels are POOLED, each carrying its own level's cell mean,
   covariance and mass share, ranked by prominence together, and the K
   most prominent become the seeds. Co-located candidates are NOT
   de-duplicated: a coarse peak and a fine cusp at the same place are the
   bead-plus-core pair, entering with different initial covariances. At
   the fine level a 100-point core in an area of 0.004 has density of
   order 25 000 against about 7 000 for the bead under it, so its
   prominence exceeds every coarse peak's; at the coarse level it is
   diluted into a 100-point cell and vanishes.
   **Selection rule, amended again 28 September:** prominences at the two
   levels are not on one scale (a fine-level noise cell can carry a few
   thousand), and the pooled ranking dropped a coarse peak for a fine noise
   peak, leaving the fit 19.5 nats short on a saddle. Levels seed different
   structures: ALL coarse peaks are seeded first (the top K by coarse
   prominence if there are more than K), and the remaining K − (coarse
   count) seeds are the most prominent FINE peaks, co-located ones kept;
   k-means centres fill only if fine peaks run out.
   **Round 4 (ruled 28 September; supersedes the two selection rules
   above).** Round 2 had the right seeds and lost P2 during EM; round 3
   forced in two noise coarse peaks (an isolated far outlier's cell, a
   straggler) and lost four bead seeds. Three fixes: (i) noise guard at
   both levels, a peak candidate has at least 5 members and at least one
   adjacent cell; (ii) fine peaks are kept only as CUSPS, a fine peak whose
   density exceeds the mean density of the coarse cell containing it by
   more than three standard errors of its own count, density_fine /
   density_coarse > 1 + 3/√n_fine (a bead's tip is at about 1.3 and fails;
   P2 at contrast 3.8 passes near 5, P1 near 9, P3 above 20); (iii) seeds =
   all cusps + the top K − (cusp count) coarse peaks by coarse prominence,
   k-means centres only if short; (iv) a fine seed's initial covariance
   from its OWN cell's members only (round 2's P2 seed started at bead
   scale from its adjacent cells and never shrank back); coarse seeds keep
   own-plus-adjacent.
   **Round 4 also missed** (28 September): 35 of 61 fine candidates passed
   the cusp test, 32 of them noise cells in sparse background where the
   containing coarse cell is nearly empty; the real cores came out at
   ratios 1.44 (P2), 3.08 (P1) and 10.5 (P3) because the coarse cell already
   holds the core's excess and a 20-point fine cell dilutes it. The
   position-density seeding family is closed: four rounds of patches, each
   with its own constant, and the fit still 198 nats short.

   **Round 5, the rule that replaces rounds 1–4: residual-driven insertion
   (greedy EM, Verbeek, Vlassis and Kröse 2003), ruled 28 September.**
   One principle: a component is added where the data exceed what the
   current model explains.

   a. Cells: k-means on X with 50·K centres (seeded as now), counts n_j,
      cell areas A_j = π r_j² (r_j the median member distance; cells with
      fewer than 5 members carry no candidate), cell means c_j and member
      covariances. One resolution; no adjacency is needed.
   b. Start with one component, the sample mean and covariance, weight 1;
      EM (accelerated) to convergence.
   c. Insertion, repeated until K components: with the current model f,
      the residual of cell j is the EXCESS COUNT e_j = n_j − N·A_j·f(c_j).
      The candidate is the cell with the largest e_j. The new component's
      mean is that cell's member mean, its covariance the cell's own
      member covariance, its weight e_j/N (floored at 5/N); the existing
      weights are scaled by 1 − e_j/N. Run a short accelerated EM, 20
      steps, from the enlarged model; recompute the residuals; insert
      again. (A co-located core appears as soon as its bead is in the
      model: P2's excess is about 100·(1 − 1/3.8) ≈ 74 counts against a
      noise cell's ±4.5; P3's about 48; P1's about 178. Before any bead is
      fitted the largest excess lies on the filament, so the beads are
      inserted first.)
   d. The result is ONE start, run through the same full EM, gate and
      polish as any other start, in the multi-start alongside the 20
      k-means starts; the best final likelihood wins. No cusp test, no
      guard beyond the 5-member floor, no second resolution.
      **Start selection (ruled 28 September, after draw 3 converged 99
      nats below the truth's continuation with P2 at six times its weight
      because the 25K family's three variants took all three phase-2
      slots):** every greedy-family start (both cell counts, 10·K and
      25·K, each with its ×½ and ×2 variants; six starts) runs to full
      convergence; the two-phase screen applies to the 20 k-means starts
      only, and no family of variants may occupy more than one finalist
      slot. The winner is the best final penalized likelihood among all
      converged starts. Products per draw: the winning start's family, and
      the spread (best minus second-best) of the converged starts' final
      likelihoods per point, a per-draw measure of search reliability.
   e. Rounds 1–4's seeding code is removed (no dead code, §5 R3); the
      k-means starts stay.
   f. **Weights in the seeding (ruled 28 September).** The cell "counts"
      are weight sums, W_j = Σ_{i∈j} ω_i, and the residual is
      e_j = W_j − (Σω)·A_j·f(c_j); identical to counts at unit weights, so
      nothing changes on samples. Reason: on the population grid used to
      define the coverage target (400 × 300 cells with cell probabilities
      as weights) a count-based seeding is blind, every cell has the same
      count and the density lives in the weights, and the population
      multistart landed in round 1's pathology for that reason. The old
      "X only, never the weights" rule protected the stencils' continuity
      in ω; under A9 no stencil re-seeds, so the seeding may read ω.
      **Coverage target rule:** the population penalized MLE is the higher
      population penalized log-likelihood of (a) the continuation from
      the truth and (b) the weight-aware multistart; if (b) exceeds (a),
      the estimand differs from the truth in the central beads and (b) is
      the study's target for all outputs; otherwise the truth's
      continuation is.

   Acceptance (amended 28 September after the round-5 result): winner ≥
   penalized log-likelihood at the truth on the seed-0 draw, AND recovery
   max |z| < ~4 over the MEASURED outputs; recovery over all raw outputs
   and the insertion order (which true component each inserted candidate
   landed on) are reported, not gated. **Round 5 result:** the greedy start
   wins at 19.1 nats ABOVE the truth's likelihood; P1, P2, P3, the cloud
   and four beads recover within 1–2σ, P2's seven outputs within 1σ; the
   two central beads come out as one long component plus a small one, a
   legitimate alternative optimum of the same objective on this sample,
   in nuisance parameters, which is the weak identifiability of
   overlapping beads. Accepted for the demo. A one-time population fit
   (GMM2D on a fine grid of the true density with cell probabilities as
   weights) decides the coverage target: the generating truth if it
   reproduces the six-bead configuration, that population fit otherwise.
   If greedy insertion had missed, the next step would have been a design
   review of the mixture and estimator together, not a sixth seeding rule.
6. (P) Two further starts from the same peaks with the initial covariances
   scaled by ½ and by 2, so that a compact core is seeded compactly.
   Total: 20 k-means starts + 3 peak-seeded starts, the same two-phase
   screen as now.

**Acceptance for the build**, one draw (seed 0) of `cloudfil_G_B6_P3_v1`
at N = 10 000: the winner's penalized log-likelihood is at least the
penalized log-likelihood at the true parameters; recovery max |z| over the
59 raw outputs below about 4; the peak-seeded start is the winner or ties
it; the number of starts within 10⁻⁶ per point of the winner reported. On
the FP audit draw nothing changes bit for bit (the FP estimator has no
starts). No other run.

### A13. A fifth method, `ijfd`: the infinitesimal jackknife by finite differences (author's ruling, 28 September)

**What it is.** The exact infinitesimal jackknife computed the only way a
black box allows: one perturbed fit per observation. It is the reference
QIJ approximates and the cost it avoids, and its time is to be measured,
not multiplied. A method like the others: (dataset, estimator, method, N,
draws), its own products folder, its own command-line call with any S (the
demo uses S = 1 for the cost slide; nothing forbids more).

**The draw.**
1. θ̂ = T(X, 1), the full-data fit (cold, A12's seeding, the reference
   labelling as for every method).
2. η_full measured as A9 measures η_Q, on the full data: two fits from θ̂ at
   the base weights and one at a 10⁻⁶ perturbation of the heaviest point,
   largest relative parameter difference, floored at the polish residual;
   the step δ_f = 2√η_full.
3. For each observation i: weights ω(i) raising point i's weight by the
   ported forward step (t_i = δ_f p_i/(1 − p_i) with p_i = 1/N, every other
   weight lowered in proportion), the fit started from θ̂ (`start = θ̂`,
   A9), ψ_i = [T(X, ω(i)) − θ̂] / t_i for all outputs; N evaluations of size
   N, independent, through the pool. Mass-centre per output.
4. V_ijfd,c = (1/N²) Σ_i ψ_ic² per output, the normal interval on it at any
   level (computed on demand as for QIJ), and the ABC ingredients from the
   point influences (a_c from Σψ³, b and c as A10 defines them, here with
   point-level quantities; the curvature evaluations along ψ on the full
   data, two per output).
5. Self-check, reported: the step-doubling ratio on the ten heaviest
   points, as A9 rule 4.
6. **`point_curvature` option (ruled 28 September).** The forward draw has
   no per-point second difference, so A10's b cannot be formed from it.
   With `point_curvature` on, every point also gets the backward-step fit
   (2N + 1 evaluations plus check and ABC), ψ_i is the central difference,
   Δ²T_i the three-point second difference, and b_hat and the ABC interval
   are point-level exact. With it off, b_hat and the ABC fields are NaN and
   ψ_i is the forward difference. **Step (added 28 September):** with the
   option on, the ± point stencil uses the CENTRAL step, t ~ η^{1/3}, the
   same rule the bins' central stencil uses, not the forward step
   t ~ 2√η; with the forward step the second difference over t² was
   roundoff (errors up to 10⁵×). On the Pareto shape b̂ then equals the
   closed form Σ(L − L̄)²/(N² L̄³) to 0.02%. Package default OFF; the demo's S = 1 run
   is done BOTH ways, once each, so the cost slide carries the forward and
   the central timings and the central run is the exact reference for
   QIJ's bin-level b_hat. c_q is never broadcast into per-point Δ²T.

**Products.** Per output V_ijfd, a, b_hat, c_q; shared: η_full, evaluations
(N + 1 + the check's and ABC's), rows, wall time, busy time, workers, the
step ratios; per point ψ_i for all outputs (the `ij` method's analytic ψ is
stored the same way, so the comparison layer forms relMSE(ψ_ijfd, ψ_ij) per
output and per point without a rerun); the fraction of perturbed fits that
returned NaN. A NaN at any point fails the output's V (the identity needs
every term); the draw's other outputs stand.

**Cost, stated in advance so the measurement can be judged.** At
N = 10 000 with warm-started fits at about 1.5 s: about 4 h serial per draw
forward, twice that central, under 20 and 40 minutes on 14 workers. It is
NOT run for the S = 500 study. **Where it runs (author's ruling, 28
September):** the two demo runs, forward and central, S = 1, N = 10 000,
go to TACC at the study's worker count (W = 100), in separate product
folders, because they measure cost and must be timed on the same hardware
as QIJ's and the bootstrap's runs. Locally only a one-draw smoke at
N = 2000 with point curvature, to show the method runs on `cloudfil`, plus
the S = 1 run on the paper's FP at N = 2000 (audit draw s = 0) as the cheap
correctness point, where its V must equal the stored `ij` value to the
step's accuracy.

**Acceptance for the build.** On the FP audit draw, relMSE(ψ_ijfd, ψ_ij)
per output below 10⁻⁴ and V_ijfd within 10⁻³ relative of the stored V_ij;
the step ratios within 10% of 2. No other run.

### A14. Refinement in rounds: `refine_schedule` (author's ruling, 28 September)

**Why.** The marginal path's refinement is a queue: the open leaf with the
largest expected gain is split, one evaluation, then τ and the flags are
updated, then the next leaf. Running the measured outputs' queues
concurrently gives at most q-way parallelism, 7 here, on nodes with about
100 CPUs, and refinement was already the largest serial term at N = 2000.

**The switch.** `refine_schedule` with two values: `queue` (ported, the
default; bit-identical, the audit stands) and `rounds`.

Under `rounds`, all measured outputs are refined together in synchronized
rounds. A round: for every output, every open leaf whose stored expected
gain is at or above that output's current τ is split, with the split kind
by the ported rule and the child's evaluation as ported; all those
evaluations, across all outputs, run through the pool at once; then, per
output, V_btw, L and τ are updated from the realized gains, the closing
flags set by the ported two-consecutive rule, and the children given
their proposals. The next round begins. Stop when no output has an open
leaf above its τ, or when an output's evaluation guard 1 + M_𝒳 is
reached; a round is capped at the remaining budget in expected-gain order
so the guard is never overshot. Leaves below τ are not touched and can
qualify in a later round as τ falls, exactly as in the queue.

**What differs from the queue.** τ is recomputed per round rather than per
split, so a leaf late in a round is judged against a slightly larger τ
than the queue would have used, and a round may split a few leaves the
queue would have stopped before. Results are therefore not bit-identical
to the queue (an E8-class difference); the tolerance's meaning and the
closing rule are unchanged. The round count is about the tree's depth,
a handful, and each round is as wide as the sum over outputs of their
qualifying leaves, tens of evaluations, which is what a 100-CPU node can
use. The demo passes `rounds`; the paper's cases keep `queue`.

**Acceptance.** On the FP audit draw, `queue` bit-identical; `rounds`
reported beside it: per output L, evaluations, V_btw (relative difference
to `queue`, expected within the tolerance ε, since both schedules leave at
most ε inside the bins and may differ by up to that; the first draft said
"a few tenths of a percent", which holds at ε = 0.01 but not as a general
statement), and the wall time at 1, 8 and the node's worker count.
**Built and accepted 28 September:** queue bit-identical; on the FP draw
rounds equals queue in L, evaluations and V_btw (the refinement there is
too small to differ); on cloudfil at N = 2000, ε = 0.05, 14 workers,
refinement wall time 10.3 → 3.6 s, evaluations 707 → 677, V_btw identical
on five of seven outputs and different by 0.07% and 3.4% on the two
most-split ones, inside ε. The dress rehearsal at N = 10 000, ε = 0.01
reports the per-output difference and its sign. No other run.

### A15. The consolidated repair round (author's instruction, 28 September): the marginal refinement gets a measured trigger

**Withdrawn (28 September): item 1 (the measured trigger, rounds 1–4), item 2 and item 3's
demo setting are removed from the code; the refinement is the ported rule. The shortfall they
chased was the survey's weight normalization (A8, corrected). The bookkeeping fixes, η_full
and the Newton cap stand.**

**The finding it repairs (rehearsal + diagnostics, draws 0 and 1 of the
demo).** QIJ's between-bin variance reached 57–93% of the analytic
influence variance at ε = 0.01 on every measured output, with the position
angle at its evaluation guard. The stencils are right: the final bins'
true between-bin share equals V_btw/V_ij exactly, so the shortfall is the
partition. The initial 17 bins recover almost nothing on some outputs
(0.8% for the angle), refinement closes most of the gap but spends only
2–13% of its evaluations on bins inside P2, and the reason is the model:
on the bins holding P2's points the posterior's predicted within-bin
variance is 0.2–9% of the true one, so the expected gains inside P2 are
underestimated by one to two orders and the queue spends its budget
outside. The point prediction inside P2 also fails for specific outputs
(angle −0.61, logit weight 0.09 on draw 1). Cause: a global kernel width
tuned to the cloud's and filament's scales, of order 0.3–1, cannot
represent structure inside a 0.05-wide core sampled at nine fields, and
the moments rows make the GP certain of what it cannot see. The refinement
as published trusts the posterior's pricing; on this estimand that trust
is misplaced exactly where it matters.

**The repair, three parts, one round.**

1. **A measured trigger in the marginal refinement** (switch
   `refine_trigger`: `gain`, the ported queue, default and audited;
   `measured`, the demo's setting). After the initial bins of output c
   are measured: a_c = Σ_k p_k U_kc m_kc / Σ_k p_k m_kc² (B4's scale
   factor); bin k is FLAGGED when
   p_k (U_kc − a_c m_kc)² > ε V̂_c / L + p_k a_c² u_kc, with m_kc the bin's
   mean of ψ̂₀ and u_kc the posterior variance of that bin mean (B4's test,
   unchanged). Flagged bins enter the queue ahead of every predicted-gain
   entry, ordered by their measured discrepancy p_k (U_kc − a_c m_kc)². A
   flagged bin is split by the ported kind rule (level if Var_k(ψ̂₀) > v_k,
   else adjacency) with one addition: when the bin lies inside a single
   receptive field, or its level split's children have predicted means
   closer than the bin's allowance, the split is GEOMETRIC, two-means on
   the members' whitened data coordinates with the principal-axis
   initialization (deterministic), because inside a core the prediction
   has nothing left to say and the data's geometry does.
   **Amended 28 September after the first draw-0 check** (ratios
   0.90–0.98 with two outputs at the guard and 95% of evaluations in
   flagged lineages): the flag test decides ENTRY into the queue only,
   once, for the initial bins. Children of a flagged bin are NOT
   re-flagged by the model test; they are governed by the ported
   realized-gain rule like any refined leaf, a child re-enters if its
   realized gain is at least τ and a lineage closes after two consecutive
   below-τ splits. (Re-testing children against the model cannot
   terminate where the model is wrong: the per-bin allowance ε·V̂/L shrinks
   as L grows and demands that a wrong model be matched to ε/L, which is
   the runaway the check measured.) The geometric override's criterion,
   now specified: form the level split's children and their predicted
   means m_a, m_b and bin-mean posterior variances u_a, u_b; the split is
   LEVEL when |m_a − m_b| > √(u_a + u_b), the model separating the children
   by more than its own uncertainty about them, and GEOMETRIC otherwise;
   the adjacency split stays available as ported. Children are measured as
   ported (one forward evaluation, the other by mass balance). Unflagged
   bins keep the ported queue, closing rule and τ. The evaluation guard is
   unchanged.
   **Round 3 (28 September; round 2 regressed to 0.34–0.94 with zero
   geometric splits, because the split-kind rule read the same
   overconfident posterior the repair was meant to bypass, and the
   zero-level closing starved refinement to 250 evaluations).** No
   posterior quantity enters a flagged lineage: (i) entry by the measured
   flag test on the initial bins only, as in round 2; (ii) flagged
   lineages split GEOMETRICALLY, always, two-means on whitened data
   coordinates with the principal-axis initialization, since the flag has
   established by measurement that the model's mean is wrong there and its
   within-bin ordering is not to be trusted; (iii) a flagged lineage's
   children enter the queue with a MEASURED priority, half the parent
   split's realized gain Δ = p_a U_a² + p_b U_b² − p_k U_k², flagged initial
   bins with their measured discrepancy, unflagged bins with the ported
   expected gain; (iv) closing by the ported two-strike rule everywhere;
   (v) unflagged bins refined exactly as ported. Guard unchanged.
   **Round 3 result (draw 0):** 0.93–0.96 on six outputs, the angle at
   0.77, 793 evaluations, lineages closing on their own, geometric splits
   doing all the flagged work. Short because a principal-axis split is
   blind to the influence: its realized gains are erratic and two erratic
   misses close a lineage that still holds variance.
   **Round 4 (28 September), amending (ii) only: the geometric split
   follows the MEASURED local gradient.** In a flagged lineage, take the
   measured cells under the cell's nearest ancestor with at least three
   measured descendant cells (whitened centroids c_j, measured U_j for the
   output being refined, masses p_j), fit the mass-weighted least-squares
   plane U = α + g·c, and split the cell at the median of its members'
   projections onto g; with fewer than three such cells, the
   principal-axis split. The paper's level split with measured local
   levels in place of the GP's; model-free; no extra evaluation. Local to
   the sub-lineage because a quadrupole's gradient vanishes at the core's
   centre but not within each half after the first bisection.
2. ~~Zero-level closing.~~ **Removed in round 3.** One below-τ split
   cannot distinguish a constant bin from an evenly divided one, which is
   why the paper waits for two; round 2 showed what one strike costs.
3. **The demo's GP width becomes local** (`gpwidth = local`, A2, already
   built and audited): the diagnosis names the scale mismatch between one
   global width and a 0.05-wide core as the cause of the overconfidence,
   and the local width gives P2's fields a width of their own spacing.
   On the Fundamental Plane it was identical to `global` under the
   quadratic trend, so nothing already accepted changes. A11's study
   settings are amended accordingly.

Also in the round: the three bookkeeping fixes (curvature term excluding
non-finite prototypes; the oracle's failure count including NaN returns;
ijfd's ABC over the measured subset); and the seeding's second greedy
start at 25·K cells with its scaled variants IF the attribution of draws
3, 6, 7 shows the truth's continuation passing where the multistart
failed (otherwise those are counted estimator failures and nothing is
added).

**Two additions from the attribution and the degenerate-replicate
breakdown (28 September).**

- The cold-fit failures on draws 6 and 7 were saddles that the truth's
  continuation passes at a higher likelihood, so the starts are at fault:
  the second greedy start at 25·K cells with its scaled variants goes in.
  Draw 3 passed on rerun with a gate margin of +1.6 × 10⁻⁵, flipping with
  summation order; the gate stays a sign test, and the rerun reports
  whether the variants move that draw to the higher basin the k-means
  winner missed by 18 nats.
- Of the 11 degenerate bootstrap replicates, 6 were converged fits cut
  off: the score norm at the Newton cap of 20 sat at 2 × 10⁻¹² to
  3 × 10⁻¹¹, the float64 floor for K = 10, 59 parameters and N = 10 000,
  against a declared η of 10⁻¹² carried over from mixture 11 at N = 5000.
  Rule: **η_full is measured per draw on the full data**, as A9 measures
  η_Q and as A13 prescribes for ijfd (two fits from θ̂ at the base weights
  and one at a 10⁻⁶ perturbation of the heaviest point; the largest
  relative parameter difference, floored at the polish residual, rounded
  up to a power of ten), recorded as `eta_full`, and used by every
  full-data evaluation in the draw: the polish's acceptance for stencils
  and bootstrap replicates, and the stencils' step δ_f = 2√η_full. The
  Newton cap becomes 40. The other 5 replicates are genuine stalls on the
  resample's surface and stay counted as degenerate; the warm start stays,
  since a cold multistart fails more of them.

**Products added.** Per output: `n_flagged` (initial bins flagged),
`n_flag_evals` (evaluations spent on flagged lineages), `n_geom_splits`,
`a_c`; shared: `eta_full`; the trigger switch value in every row.

**The gate for TACC: the rehearsal rerun** (same draws, same workers,
demo settings with `refine_trigger = measured`, `gpwidth = local`,
`refine_schedule = rounds`). Pass criteria, all required: V_btw/V_ij ≥ 0.97
on every measured output on both draws; evaluations per output reported
and no output at its guard; the queue/rounds difference within ε on
draw 0; degenerate bootstrap replicates explained by component and cause;
the three failed oracle draws attributed. If any criterion fails, the
next step is the next diagnosis from the rerun's products and the next
repair to the method; the estimand does not change.

### A16. Deterministic annealing as the mixture estimator's search (author's ruling, 28 September)

**Withdrawn (28 September).** As built, the annealing never left its start (all K
components at the sample mean; the per-step perturbation of item 3 was not in the
code), so every cold fit on the demo returned NaN. The multistart search of A12
(`seeding.py`) is restored, with the search audit and its products removed.

**Why.** Every cold fit in the study, and all ten thousand of the oracle's,
must reach the right basin without knowing the truth, or the search's
failure rate becomes coverage loss on P2. The multi-start with k-means and
greedy starts reached the basin on draw 0 and missed it on draw 3 by 99
nats through a screening accident; more starts and selection rules are
patches on a search that descends into the nearest basin. Deterministic
annealing EM (Ueda and Nakano 1998) descends into the deepest: it
maximizes a free energy with a temperature, unimodal at high temperature,
and tracks that maximum as the temperature falls to unity, where the
objective is the penalized likelihood itself. Structures appear as the
temperature falls, the largest first and compact cores last. It needs no
seeds and replaces the whole start apparatus with one deterministic run.

**The procedure** (switch `search`: `multistart`, the current apparatus,
default until A16 is accepted; `anneal`).

1. Inverse temperature β on a geometric schedule from 0.02 to 1 in 25
   steps (factor ≈ 1.17), the standard schedule; the acceptance below
   decides whether it is fine enough, and the only permitted change is a
   finer factor.
2. Start: all K components at the sample mean and covariance with equal
   weights. **Amended 28 September after the first build failed on all
   five draws:** a symmetry break injected ONCE is not deterministic
   annealing. With identical covariances the tempered EM contracts a
   mean-only perturbation by exactly β per step (the builder's derivation,
   checked numerically), so a seed injected at β = 0.02 is below float64
   long before the critical temperatures of the small structures, and all
   K components collapse to one saddle that the gate correctly refuses.
   Rose's procedure re-injects the perturbation at every temperature:
   below a structure's critical temperature it contracts, above it the
   same perturbation grows into the split.
3. At EVERY β step, before that step's EM, every component is perturbed
   deterministically: its mean displaced by 10⁻³ σ_k along its own first
   principal axis, with the sign alternating with the component index so
   that identical copies move apart, and its covariance scaled by
   (1 + 10⁻³) or (1 − 10⁻³) along that axis for alternating copies. Then
   the E-step uses tempered responsibilities,
   r_ik(β) ∝ [π_k φ(x_i; μ_k, Σ_k)]^β normalized over k; the M-step is the
   penalized weighted M-step as built (SQUAREM allowed); iterate to the
   ported convergence tolerance, warm from the previous β. Copies
   re-collapse below their critical temperature and separate above it; the
   perturbation size, 10⁻³ of the local scale, is the one constant.
   Products add, per β, the effective component count by pairwise
   Bhattacharyya distance above 10⁻⁶ (copies below that count as one), so
   the β-trace shows the splits.
4. At β = 1: the Newton gate and polish as built, acceptance at the
   measured η_full.
5. Under `anneal` there are no k-means starts, no greedy insertion, no
   families and no screen. **Amended twice on 28 September:** the first
   build removed the multistart with the anneal and the anneal then failed
   on all five draws, leaving the tree with no cold search, a sequencing
   error of the planner's. The multistart is restored from commit 8153393
   as `search = 'multistart'`, the DEFAULT until the anneal passes its five
   draws; `search = 'anneal'` is the candidate. Once accepted, the
   multistart code comes out under §5 R3 and anneal becomes the only
   search. The multistart's recorded results on draws 0, 1, 3, 6 and 7
   remain the comparison side of the acceptance.
   Continuations with `start` never anneal: they run EM from their start
   on its branch, since annealing would erase the branch a derivative
   depends on; that cold/warm distinction is the design.
6. Products per cold fit: the β at which the effective number of
   components (weights above 5/N) last changed, and the final penalized
   likelihood; plus the search audit of A16.7.

7. **Search audit, for every cold fit in the study (this dataset has a
   known truth).** After the cold fit, one continuation from the TRUE
   parameters on the same draw (start = truth, labels to the truth); the
   product `search_gap` = (cold fit's penalized log-likelihood) − (truth
   continuation's), per point, and `search_failed` = (search_gap < 0 by
   more than the polish residual). The truth never enters any method's
   estimate; it judges the search, as the analytic influence judges the
   variance. The oracle runs FIRST on TACC and its audited failure count
   over ten thousand draws is read before the bootstrap and QIJ runs start.

**Acceptance.** On the demo at N = 10 000, draws 0, 1, 3, 6 and 7 (the
known cases), `anneal` against `multistart` on the same draws: the
annealed fit's penalized log-likelihood at or above the truth's
continuation's on every one of the five (search_gap ≥ 0); P2's seven
outputs recovered within ~4σ on each; wall time per cold fit reported
beside the multistart's. On the FP audit draw nothing changes (the FP
estimator has no search). If a draw fails, the schedule factor is refined
once (to 1.10) and the five are rerun; if it still fails, the failing
draw's β-trace (effective component count against β) is reported and the
planner rules.

### A5. Wave A audit and measurement

Audit: with `gptrend=affine`, `gpwidth=global`, any worker count, every kept
field equals the ported path bit for bit, on the audit draws of testing
rule 2 and nothing else.

Measurement (one agent, after the audit, no package change; testing rule 3
applies). **Author's ruling, 26 September: a measurement of a property of
the code uses a simple estimator already in the package, not the mixture.**
(The first attempt on mixture 11 produced nothing: the K = 9 fit at defaults
did not converge on the seed-0 draw at N = 2000 or 5000, so there was no
reference fit and no oracle influence. Mixture 11 is the talk demo's
estimand, not a test bench.) The measurement is therefore the Fundamental
Plane, `fp all` (q = 4, d_x = 3, analytic influence), N = 2000, one draw at
seed 0, at `M_X = 371` (the rule) and `M_X = 742` (twice it). Per M_X: one
𝒳-VQ and one survey, shared by the four switch combinations (the survey
does not depend on the switches); then four GP fits. **Added 26 September
(author's approval):** the same protocol on the multivariate t, `mvt nu`
and `mvt tail`, with its `vq_transform`, at the rule's count and twice it,
because the Fundamental Plane's influence is near-quadratic and cannot
judge `gpwidth=local`; the author rules on item 1b from the two tables
together. Reported per combination and output: the mass-weighted
mean squared error of ψ̂₀ against the analytic influence; the share of the
output's oracle variance recovered by 17 level-set bins of ψ̂₀, built with
the ported 1-D quantizer on ψ̂₀ and scored with the analytic influence, no
estimator evaluation; and the GP fit time. Evaluations spent: 1 + M_𝒳 per
N, on M_𝒳 rows, and nothing on the full data. One table. Report as found.

---

## Wave B: `ivqbins=joint` (items 5, 6, 9)

`ivqbins=marginal` is the ported second stage in full: per-output 1-D
quantizer at the ported count, per-output refinement queue, per-output
products. Nothing below applies to it.

`ivqbins=joint` replaces the second stage by: one partition grown to the
tolerance on ψ̂₀; measurement of every bin; a measured check that splits
where measurement contradicts the model; products per output plus the
shared bin constituents. Stage 1 is untouched by the switch.

### B1. Standardization

Ψ̃: ψ̂₀ with each column divided by its standard deviation over the N points.
V̂_c = that variance, the predicted total for output c. An output on the
ported constant path (§5.3) is a failed output; under `joint` a failed
output fails the draw (§5.4), since all outputs share the bins.

### B2. Growth (item 5)

Start from one bin holding every point. A round:

1. For every bin k and output c, the bin's contribution to the predicted
   within share, w_kc = p_k · Var_k(ψ̂₀^(c)) / V̂_c (population variance over
   the bin's points; 0 for a bin of one distinct row). S_c = Σ_k w_kc is the
   predicted within share of output c.
2. If S_c ≤ ε for every c, growth is finished.
3. Otherwise split, all in the same round, every bin with max_c w_kc > ε/L,
   L the current bin count. (Whenever some S_c > ε at least one bin
   satisfies this, so every round makes progress.) A bin with fewer than two
   distinct standardized rows is not split (its w_kc is 0).
4. Two-means on a bin's standardized rows: initial centroids are the means
   of the two halves of the bin split at the median of its projection onto
   its first principal axis; then Lloyd iterations until assignments stop
   changing or 100 iterations. Deterministic, no seed.
5. **Cap (P).** Growth stops when L reaches M_𝒳 even if the tolerance is
   not met; `growth_capped` is recorded. (Measurement costs two full-data
   evaluations per bin; this cap makes the initial measurement at most
   2·M_𝒳 evaluations, the same order as the ported refinement guard.)

Rounds are sequential; within a round the two-means of different bins are
independent (in-process numpy in the first build; the pool is not used for
growth).

**Lloyd pass (P, kept only if B7's measurement shows it saves bins).** After
growth, one Lloyd run of all L centroids over all standardized rows until
assignments stop changing or 100 iterations; then S_c is recomputed. Record
S_c before and after. Membership changes here happen before any
measurement.

Growth uses Var_k(ψ̂₀) only; the posterior variance enters in B4.

### B3. Measurement

Every bin is measured by the ported central stencil (`bin_differences`),
two full-data evaluations per bin, returning U_k ∈ ℝ^q; the tasks run
through the pool, one task per bin, results assembled in bin order. The
ported failure rule holds: a NaN in any bin's stencil fails the draw for
every output. Then V_btw,c = Σ_k p_k U_kc² per output (the ported
`between_terms` per output over the shared bins).

### B4. The measured check (item 6)

For every bin k and output c, m_kc = the mean of ψ̂₀^(c) over the bin's
points (the model's prediction of U_kc), and u_kc = the posterior variance
of that bin mean for output c: the average of all entries of the bin's
posterior covariance block, mean(Σ_k). (The ported v_k = mean(diag Σ_k) −
mean(Σ_k) is the expected within-bin variance and is the wrong quantity
here; both come from the same chunked pairwise computation.)

**Scale factor (P, 26 September).** The model's scale for an output can be
off while its ordering is right (the A5 tables show ψ̂₀ for the MVT's ν at
tens to hundreds of times the oracle's magnitude with a recovered share of
0.99). The check must not read a uniform scale error as disagreement in
every bin, so one factor per output is fitted from the bins themselves,

    a_c = Σ_k p_k U_kc m_kc / Σ_k p_k m_kc² ,

the least-squares scale of the measured derivatives on the predicted means
over all bins after the first measurement, and held fixed through the check
rounds. `a_c` is a product; a value far from one is the diagnostic of a
scale error and calls for no action inside the method.

Bin k is **flagged** when for some output c

    p_k · (U_kc − a_c · m_kc)²  >  ε · V̂_c / L  +  p_k · a_c² · u_kc ,

that is, when the measured derivative differs from the (rescaled)
prediction by more than the bin's allotment of the tolerance plus what the
posterior allows for the error of a bin mean. No other constant enters.

A check round: every flagged bin is split by the ported split-kind rule in
its multi-output form (level split when Σ_c Var_k(ψ̂₀^(c))/V_btw,c ≥
Σ_c v_kc/V_btw,c, two-means on the bin's standardized rows as in B2 step 4;
otherwise the ported adjacency split: form each output's ported adjacency
proposal for the bin, take the output whose proposal has the largest
expected gain divided by that output's V_btw,c, and split along that
output's ordering; an infeasible kind falls back to the other; both
infeasible: the bin stays and is closed). One child is measured by the ported forward
evaluation and the other's U follows by mass balance, as the ported
refinement does; all flagged bins' evaluations run through the pool in the
same round. After the round: V_btw,c updated (parent's term replaced by the
children's), L updated, each child checked by the same test with the new L.
Flagged children go into the next round.

Stop when no bin is flagged, or when check evaluations reach the ported
guard 1 + M_𝒳 (`check_capped` recorded). A failed evaluation cancels that
split as ported (parent stays, closed, counted).

There is no τ, no closing flag, no expected-gain queue, no ρ² rescaling
under `joint`. The realized-to-predicted ratio is reported (B6), not used.

### B5. Prototype count under `joint`

As A4: the ported rule when `M_X` is absent, the argument when given.
Nothing new in this wave.

### B6. Products under `joint`

Per output c: `V_btw`; `V_win_hat` = (1/N) Σ_k p_k (Var_k(ψ̂₀^(c)) + v_kc)
over the final bins, with v_kc the ported expected within-bin posterior
variance, no ρ² and no γ_k since the joint path has neither; `V_tot_hat` =
`V_btw` + `V_win_hat`; `S_pred` (the predicted within share at the end of
growth, after the Lloyd pass if kept); `a_c` (the check's scale factor, B4); `gain_ratio` = Σ_splits Δ_c /
Σ_splits g_c over the check splits, with Δ_c = p_a U_ac² + p_b U_bc² −
p_k U_kc² the realized gain and g_c the ported expected gain of the chosen
split kind for output c (NaN when there were no check splits). `B_hat` and
`a_bca` are not products of `qij_joint` (dropped in the port) and do not
return here. Shared: `L0` (bins after growth), `L` (final),
`n_growth_rounds`, `growth_capped`, `S_pred_pre_lloyd` per output (if the
pass is kept), `n_flagged` (bins flagged in the first check),
`n_check_rounds`, `n_check_evals`, `n_level_splits`, `n_adjacency_splits`,
`check_capped`, evaluation and row counts by stage as ported, the three
switch values, `M_X` and `M_X_source`. Bin constituents: `bin_mass`,
`bin_U` (L × q), `bin_m` (L × q, the predicted means), `bin_flagged`. Per
point: `bin_label`. The q × q between-bin matrix Σ_k p_k U_k U_kᵀ is
derivable from `bin_mass` and `bin_U` and is not stored (item 7, closed).

### B7. Wave B audit and measurement

Audit: with `ivqbins=marginal`, every kept field equals the ported path bit
for bit at any worker count, on the audit draws of testing rule 2 and
nothing else.

Measurement (one agent, after the audit, no package change; testing rule 3
applies). By the same ruling as A5, the Fundamental Plane (`fp all`), N = 2000, one
draw at seed 0, `M_X` from the rule (371), `ivqbins=joint`,
`gptrend=quadratic`, `gpwidth=global` (the author's rulings from the A5
tables, 26 September: item 1a adopted; item 1b not adopted, the option kept
with `global` as the default). (At q = 4 the joint path is not expected to save evaluations over the
marginal one, whose initial count is 1 + 2·4·17 = 137; this measurement is
of the mechanism, growth, check and parallel rounds, not of the saving,
which needs a large-q estimand and waits for the demo.) Two runs of the joint path on that one draw, at 1 worker and at 8
workers, identical in every field but timing; plus growth alone, which
spends no evaluations, repeated with and without the Lloyd pass. Reported:
L0 with and without the Lloyd pass and S_pred for each; S_pred against the
oracle within share on the same bins, per output (analytic influence, no
evaluation); n_flagged, n_check_rounds, n_check_evals; V_btw against the
oracle variance per output; wall time by stage at 1 and 8 workers.
Evaluations spent: 1 + M_𝒳 on M_𝒳 rows, then 1 + 2·L0 + n_check_evals on
the full data, twice. The marginal path is NOT run; its initial count
1 + 2·q·17 = 137 for the Fundamental Plane is stated as arithmetic beside
the joint count. No bootstrap. One table. Report as found.

---

## Order and dependencies

Wave A first, audited and measured; then wave B on top of it. B assumes
A's switches exist but does not depend on their values. Item 4 is ruled
(A4) and item 7 is closed (B6); nothing in either wave waits on a ruling.
