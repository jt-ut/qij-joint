# Sigma points: an optional interval stage for QIJ (planner, 29 September 2026; one build)

Scope: an optional stage at the end of the paper's method `qij`, off by
default, that evaluates the estimator at resample-scale perturbations
along the method's own influence directions and returns the outputs'
mean and covariance under the estimator's nonlinearity. It adds an
interval; it changes nothing before it. Under `sigma_points=False` every
product is byte-identical to today's. Code standards:
`qij_joint_plan.md` section 5, verbatim in the build prompt.

What is measured and what is not (the author's three questions). The
mechanism was run on cloudfil draws 0–99 with real evaluations
(scratchpad shape_test/local100/REPORT.md): with directions from the
EXACT influence, the design below scores a mean Winkler of 2.227 against
the bootstrap's plateau 2.248 at B = 200 and the exact-influence normal
interval's 2.59; worst-output coverage 0.87 against the bootstrap's
0.91. What is NOT measured is the design with the directions QIJ
actually has, the refined per-point influence estimate, whose
per-output r² against the exact influence (0.7–0.98 on cloudfil) is the
share of the linear variance those directions keep. The unscented
variance is below the in-basin truth on the shape outputs whatever the
directions (measured saturation); this stage repairs the interval's
shape and bias, not the variance. The judgment is the 500-draw study.

Terms as the glossary: **quantized data** = the weighted point set the
survey evaluates on; **prototype points**; no other new terms.

## 1. Interface

`QIJ(..., sigma_points=False)`; `scripts/run.py --sigma-points` for
`qij`. No other argument. Works with either pilot.

## 2. The stage (new module `core/sigma_points.py`, ≤ 200 lines)

Runs after refinement, on the FULL data, from the full-data base fit.

### 2.1 Inputs

θ̂ (q_full,), the full-data fit; η_full; the refined per-point influence
ψ̂ (N, q) for the measured outputs as the method holds it at the end of
refinement (`psi_hat` in the points product: U_k(i) + rho·(ψ̂₀(x_i) −
ψ̂₀_bar,k), mass-centred as built); N.

### 2.2 Fixed point of the base

The stage's differences are one-sided from θ̂, so θ̂ must be a fixed
point of its own continuation (spec QIJ_qijdt_spec.md 2.1):

    repeat θ' = T(X, 1_N, start=θ, eta=η_full);
           r = max_o |θ'_o − θ_o| / max(|θ'_o|, |θ_o|) (0/0 → 0, every output of T);
           θ ← θ'
    until r ≤ η_full, at most 5 iterations;

reaching the cap sets `sigma_status='base_unconverged'` and the stage's
products NaN (the draw is not failed). An estimator without a start:
no iteration. Products `n_fp_sigma`, `r_fp_sigma`. Typically one
evaluation.

### 2.3 Directions

C = (1/N) · mean_i(ψ̂_i ψ̂_iᵀ) (q × q, the influence covariance of the
measured outputs in the method's own map; V_o = C_oo). Eigen-decompose
C = Σ_k λ_k v_k v_kᵀ, drop λ_k ≤ 1e-12·λ_max (report the count), n = the
number kept. For each kept k the combined influence per point
ψ̂^(k)_i = Σ_o v_k,o ψ̂_i,o (its variance is λ_k) and the two exponents

    d_i^(k±) = ± c · ψ̂^(k)_i / (N · √λ_k),   c = √n.

Weights ω_i = N · exp(d_i) / Σ_j exp(d_j) (positive, Σω = N; to first
order ω_i ≈ 1 + d_i − mean d). Report max_i |d_i| per draw.

### 2.4 Evaluations

2n evaluations T(X, ω^(k±), start=θ̂, eta=η_full) (the estimator
protocol rule of `parallel.call_T`), in one pool batch, the shape of
`core.abc._curvature_task`. Responses R^(k±) = T(ω^(k±))[measured] −
θ̂[measured], (q,). A NaN or exception in an evaluation sets
`n_failed_sigma` += 1 and the stage's products NaN for every output
(the design is joint), `sigma_status='eval_failed'`.

### 2.5 Moments and interval

    m = θ̂ + Σ_{k,±} (1/(2n)) · R^(k±),
    S = Σ_{k,±} (1/(2n)) · (R^(k±) − (m − θ̂)) (R^(k±) − (m − θ̂))ᵀ,

the unscented mean and covariance (κ = 0, equal weights). Interval per
measured output at level ℓ: m_o ± z_{ℓ} · √S_oo, `QIJResult.sigma_interval(level)`;
`interval`, `interval_btw`, `abc_interval` unchanged. `sigma_bias_o =
m_o − θ̂_o`, `sigma_sd_o = √S_oo`.

## 3. Products

Scalar row: `sigma_points` (bool), `sigma_status`, `n_fp_sigma`,
`r_fp_sigma`, `evals_sigma`, `rows_sigma`, `wall_sigma`,
`n_failed_sigma`, `n_dirs_sigma`, `max_abs_d_sigma`, per output
`sigma_mean_o, sigma_sd_o, sigma_bias_o, lo_sigma_o, hi_sigma_o` (level
0.95). Array `sigma_points` (every draw): one row per evaluation
(k, sign, and the response per measured output, raw). All NaN with
`sigma_points=False`.

## 4. Validation (small; real pipeline only)

1. `sigma_points=False`: byte-identical products to the branch's own
   outputs on cloudfil draw 0 (the affine-branch run), timing columns
   dropped. The pass/fail item.
2. `sigma_points=True` with `pilot='gp'` on cloudfil draws 0 and 1 and
   (pareto, shape) draw 0: per output, `sigma_sd_o` against the exact-
   direction unscented sd for the same draws from
   scratchpad/shape_test/local100/draws/ (the 7-direction design's saved
   moments), `sigma_bias_o` likewise, the sigma interval beside the
   normal interval, n_dirs, evaluations, wall. Coverage on three draws
   is not reported. The share of linear variance the method's
   directions keep is Σ_k λ_k / Σ_o V_o^{exact} per output, from the ij
   product: report it.
3. The same with `pilot='affine'` on the two cloudfil draws.

Nothing else. Reports to Research/QIJ_joint/qij_sigma_validation/.

## 5. Questions a coder will ask

* **Which ψ̂?** The refined estimate the method ends with, `psi_hat`,
  measured outputs only, as stored in the points product; not ψ̂₀.
* **Why full data and not the quantized data?** The rows' responses
  track the full data's at r 0.95–0.99 but shrink the variance a further
  10–15%; 2q evaluations at ~1.7 s each is 24 s per draw, not worth the
  loss.
* **q = 1?** n = 1, c = 1, two evaluations at ±1 standard error.
* **Where in `qij.fit`?** After refinement and the ABC curvature stage,
  before assembly; its own stage name `sigma` in the by-stage dicts.
* **Pool?** `pool.share(X)` is already done for the draw; one
  `pool.map` over the 2n tasks; serial through the counter with
  `pool=None`.
