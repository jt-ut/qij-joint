# Mass-weighted pilot fit: each survey observation's noise scaled by its cell's mass (planner, 30 September 2026; one build)

Scope. One change in `core/influence_model.py fit_influence_model`
behind a new argument `fit_weights`: `'none'` = today's fit,
byte-identical products; `'mass'` = this document. Both width settings
(`gpwidth='global'` and `'local'`). Nothing outside the pilot fit
changes: the survey, growth, the check, the sigma stage and every
product downstream read the pilot as they do today. Code standards:
`qij_joint_plan.md` section 5, verbatim in the build prompt. No check
is added. Terms as the glossary: prototype cells, the pilot, points,
bins; never "rows"/"field".

## 1. Why (measured, offline, no evaluation: scratchpad mass_weighted_fit/REPORT.md, 30 September)

The pilot is fitted by an unweighted profiled marginal likelihood over
the M_X prototypes: every cell counts once whatever its mass. Cell
masses vary by Zador's law (populous cells at density peaks, sparse
cells in the tails) and the influence energy the pilot must serve is
mass-weighted, so at a coarse survey the objective is dominated by the
many light tail cells. Measured on the P1 width-run arrays (draws 1
and 7, both widths), the fit reconstructed exactly and refitted under
the weighted objective below:

* coarse survey (ε 0.141, M_X 596): the fitted width shrinks 18–31%;
  every noise ratio that sat at its ceiling (1e2: log_reff and the
  position angle on draw 1 under both widths; log_axis_ratio near it)
  comes off it by 5–8 orders of magnitude; the pilot's mass-weighted
  error against the exact influence falls 43–58% in 3 of 4 runs
  (unchanged in the 4th, where the width barely moved); the pilot's
  value at the 20 largest exact influences moves toward the truth
  correspondingly;
* fine survey (ε 0.01, M_X 1094): the width moves < 1%, nothing
  changes;
* the weighting applied at the STORED hyperparameters barely moves
  either metric: the gain is the refit of the width, not the
  reweighted mean alone;
* the initial bins' realized shortfall (before the check) is mixed —
  better on draw 1 under the local width, worse under the global,
  unchanged on draw 7. A better mean-squared pilot is not uniformly a
  better worst-bin pilot; the check is what serves the worst bin.

The author's reading, adopted: cheap and a better pilot cannot hurt.
Baseline = the current fit on those draws; oracle inputs: none in the
rule (V_ij enters validation only); the real number before the build
is the table above.

## 2. The rule

For output c with normalized masses m̃_j = p_j / mean_j(p_j) over the
finite prototypes of coordinate c's design,

    A_c = K_c + λ_c · diag(1 / m̃_j)   (+ the stored jitter as today)

in BOTH the profiled marginal likelihood (the width search over ℓ_c or
c_c and the search over λ_c, the declared-noise floor applied to λ_c
as today) AND the posterior: β_c by GLS with this A_c, α_c =
A_c⁻¹(I_c − Hβ_c), s²_c profiled as today, the point mean h(x)ᵀβ_c +
k(x)ᵀα_c unchanged in form, and the posterior variances (`uncertainty`,
`bin_posterior_variance`'s v_kc/u_kc) from the same A_c so every
posterior quantity is the one model. This is the Gaussian likelihood
in which observation j is weighted by its mass, written as a noise
variance s²λ_c/m̃_j; it is one mechanism, in the likelihood only —
no mass weight elsewhere (a weighted log-likelihood on top would count
it twice). Under `'none'` m̃_j ≡ 1 and every expression reduces to
today's, byte for byte. The eigendecomposition-based profiling of the
group's shared width must take the diagonal into account: the
shortcut that diagonalizes K once and profiles λ along its eigenvalues
does not apply to K + λD with D non-scalar; the build states which
route it takes (a per-λ Cholesky of A_c is acceptable at M_X ≤ ~1100 —
cost measured and reported as `ml_wall_time`, the existing product).

Not in this build: any exponent other than 1 on the mass (the
finite-difference noise of a survey value scales as 1/p_j² through
its step t_j ∝ p_j, which argues for a mass-dependent FLOOR; the floor
stays as today, a single n_c², and the note is for a later spec if the
floor ever binds under `'mass'`).

## 3. Interface and products

`QIJ(..., fit_weights='none')`; `scripts/run.py --fit-weights
{none,mass}`; passed to `fit_influence_model` as one argument;
`products.dir_tag` suffix `_massfit` (non-canonical, after the width
suffix: e.g. `qij_gp_eps0.141421_widthlocal_massfit_nosigma_measured`);
`config.json` records `fit_weights`; a keyless folder reads as
`'none'`. Products: `fit_weights` (str); the fitted `ell`/`c`/`lam`/
`s2`/`at_bound`/`ml_wall_time` per output as today (the reader sees
what the weighting changed).

## 4. Validation (author's machine, draws 1 and 7, ε 0.01 and 0.1414213562373095, `--workers 3`, hard time cap; nothing on TACC; both widths, since the sweep's width is being decided in the same round)

V1 (the one pass/fail): `fit_weights='none'` at the new code vs the
current main on draw 1 ε 0.01 (local width), column by column, the new
column excepted; identical.

V2 (pilot metrics, from the diagnostic arrays, offline against the
exact influence as scratchpad width_check/width_metrics.py computes
them): per draw × ε × width × output, `'none'` vs `'mass'`: the fitted
ℓ (or c) and λ and whether λ is at a bound; the pilot's mass-weighted
MSE / mean(ψ²); the median of ψ̂/ψ at the 20 largest |ψ|; the initial
bins' predicted worst share S/ε and realized shortfall r/ε and r/S.
Acceptance (pilot): MSE under `'mass'` ≤ MSE under `'none'` on every
output at ε 0.01 (to within 1%: the fine survey is expected unchanged),
and lower on every output that was at the λ ceiling under `'none'` at
ε 0.141.

V3 (the whole joint stage, `check_rule='measured'`, sigma off): per
draw × ε × width, `'mass'` vs `'none'`: L0, n_check_evals, total
size-N evaluations, the final shortfall 1 − V_btw/V_ij per output.
Acceptance (no harm): shortfall_mass ≤ max(ε, shortfall_none) per
draw × output, AND total size-N evaluations under `'mass'` ≤ under
`'none'` + the check's own noise (state the comparison; equality is
fine; the per-ε difference is reported and the author judges it).

Report: one markdown file with V1's verdict and V2/V3 side by side
per width, the commit, wall time per stage (counts are the record;
Mac walls are throttled).

## 5. Questions a coder will ask

* Where do the masses come from? `xvq`'s prototype masses p_j (the
  survey's `p`), restricted to the finite prototypes of coordinate c's
  design and renormalized to mean 1 there; a coordinate on the constant
  path is untouched.
* Does the declared-noise floor change? No: it bounds λ_c below as
  today; the diagonal multiplies λ_c per observation.
* Does `_point_terms`' cached pass change? Only through A_c; the
  chunked point pass reads α_c, β_c and the kernel as today.
* Does the local width's per-node h_j interact? No: the kernel is
  unchanged; only the noise diagonal is added.
* Line budget: one diagonal in the A construction, one argument
  threaded through; the profiling route (section 2) is the only place
  code may grow, and it is reported.
