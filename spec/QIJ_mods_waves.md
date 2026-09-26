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

**The demo mixture (author's ruling, 26 September).** The talk's estimand is
Chacon mixture 11 with its means and covariances as loaded from
`structsynhd` and these weights (in the loaded component order; the level
sets are nearly unchanged, every component has at least 50 points at
N = 2000, the rarest components stay fifteen times rarer than the blobs):

| k | component | mean | weight |
|---|---|---|---|
| 0 | blob | (−1.5, 0) | 0.3775 |
| 1 | blob | (+1.5, 0) | 0.3775 |
| 2 | outer spike | (−2.5, −1) | 0.025 |
| 3 | on-mean spike | (−1.5, 0) | 0.035 |
| 4 | inner spike | (−0.5, +1) | 0.050 |
| 5 | middle | (0, 0) | 0.025 |
| 6 | inner spike | (+0.5, −1) | 0.050 |
| 7 | on-mean spike | (+1.5, 0) | 0.035 |
| 8 | outer spike | (+2.5, +1) | 0.025 |

Sum 1.0000. The coder decides the plumbing (a dataset entry that loads
mixture 11 and replaces the weights; the same generator otherwise, so
`expand_dimension` still applies). Nothing else in this section is
specific to that mixture.

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
